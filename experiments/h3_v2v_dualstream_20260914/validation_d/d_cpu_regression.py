"""Small CPU D regression; no model checkpoint, NPU initialization or training.

Only run with a separately reviewed synthetic-validation policy fixture. This
script does not create/approve policies or accept a dataset/model path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types

from d_cases import CASES, REJECTIONS, MODE, WORLD, fixture, invalid_fixtures, oracle_masks, audit_masks

PREFIX = "vllm_omni.diffusion.models.minimax_h3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_fixture_policy(path, expected):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink() or digest(path) != expected:
        raise RuntimeError("Missing or mismatched synthetic-validation policy fixture")
    policy = json.loads(path.read_text())
    if policy.get("mode") != MODE or policy.get("scope") != "synthetic_validation_only":
        raise RuntimeError("CPU/tiny tests require a validation-only policy, never a formal-run policy")
    os.environ["D_REVIEWED_POLICY_PATH"] = str(path)
    os.environ["D_REVIEWED_POLICY_SHA256"] = expected
    return expected


def load_cpu_modules(candidate_root, golden_path):
    root = Path(candidate_root).resolve()
    # Namespace skeleton avoids importing the huge runtime just to test these
    # pure torch helpers. The real NPU test does NOT use this loader/stub.
    if "vllm_omni" in sys.modules:
        raise RuntimeError("CPU-only regression requires a fresh process")
    parts = PREFIX.split(".")
    for stop in range(1, len(parts) + 1):
        name = ".".join(parts[:stop])
        package = types.ModuleType(name)
        package.__path__ = [str(root if stop == len(parts) else root.parents[len(parts) - stop - 1])]
        sys.modules[name] = package
    loaded = {}
    for name in ("openvdn_npu", "strict_source_layout", "strict_source_attention", "dual_stream_attention", "dual_stream_adapter"):
        loaded[name] = load_file(PREFIX + "." + name, root / (name + ".py"))
    loaded["golden"] = load_file("_d_cpu_golden_original_branch", Path(golden_path))
    return loaded


def tensors(data, torch):
    return dict(img_pos=torch.tensor(data["img_pos"], dtype=torch.int64),
                update_mask=torch.tensor(data["update_mask"], dtype=torch.bool),
                text_pos=torch.tensor(data["text_pos"], dtype=torch.int64),
                audio_pos=torch.tensor(data["audio_pos"], dtype=torch.int64),
                img_position_ids=torch.tensor(data["coords"], dtype=torch.float64).unsqueeze(0),
                cu_seqlens=torch.tensor(data["cu_seqlens"], dtype=torch.int32), metadata=data["metadata"])


def metadata_checks(adapter, torch):
    proof = {}
    for case in CASES:
        data = fixture(case)
        layout = adapter.infer_strict_source_layout(**tensors(data, torch))
        rejected = []
        for name, invalid in invalid_fixtures(case).items():
            try:
                adapter.infer_strict_source_layout(**tensors(invalid, torch))
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError(f"D accepted invalid metadata: {case}/{name}")
        proof[case] = dict(rejected=rejected, layout=vars(layout), oracle=audit_masks(layout, len(data["coords"])))
    return proof


def dense_cpu(q, k, v, layout, torch, *, old_c=False):
    """Independent CPU fp32 scores/probabilities using pairwise oracle mask."""
    masks = oracle_masks(layout, q.shape[0])
    allowed = torch.tensor(masks[0 if old_c else 1][:layout.used_len], dtype=torch.bool)
    qf, kf, vf = (x.detach().float().cpu().permute(1, 0, 2) for x in (q, k, v))
    scores = qf[:, :layout.used_len] @ kf.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    probability = torch.softmax(scores.masked_fill(~allowed.unsqueeze(0), float("-inf")), dim=-1)
    out = torch.zeros_like(qf)
    out[:, :layout.used_len] = probability @ vf
    return out.permute(1, 0, 2).to(device=q.device, dtype=q.dtype)


def candidate_mask_checks(layout, packed, dual):
    dual_layout, routing = dual.kernel_layout_and_routing(layout)
    actual = [[False] * packed for _ in range(packed)]
    assigned = [0] * packed
    for queries, keys in dual.kernel.dual_stream_softmax_plan(dual_layout, routing):
        for lo, hi in queries:
            for q in range(lo, hi):
                assigned[q] += 1
                for start, stop in keys:
                    actual[q][start:stop] = [True] * (stop - start)
    old_c, expected = oracle_masks(layout, packed)
    if actual != expected or assigned != [1] * layout.used_len + [0] * (packed-layout.used_len):
        raise AssertionError("Actual D span plan differs from independent pairwise mask/row ownership")
    if actual[layout.video_start:layout.video_end] != old_c[layout.video_start:layout.video_end]:
        raise AssertionError("D modified C target query keys")
    return True


def compare(name, actual, expected, torch, *, exact=False):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError(f"{name}: nonfinite output")
    atol, rtol = ((0.0, 0.0) if exact else
                  ((1 / 512, 1 / 128) if actual.dtype == torch.bfloat16 else (2e-6, 2e-5)))
    delta = (a - b).abs()
    result = dict(check=name, shape=list(a.shape), atol=atol, rtol=rtol,
                  max_abs=float(delta.max()) if delta.numel() else 0.0,
                  rms_error=float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
                  golden_rms=float(b.square().mean().sqrt()) if b.numel() else 0.0)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return result


def initialize(module, generator, torch):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = torch.randn(parameter.shape, generator=generator)
            value = -1 + .02 * value if name.endswith("A_log") else (1 + .02 * value if name.endswith("norm.weight") else .03 * value)
            parameter.copy_(value.to(parameter.dtype))


def run_cpu(candidate_root, golden_path, policy_path, policy_sha):
    configure_fixture_policy(policy_path, policy_sha)
    import torch
    torch.set_num_threads(1)
    modules = load_cpu_modules(candidate_root, golden_path)
    branch_mod, old, adapter, dual = (modules[n] for n in ("openvdn_npu", "golden", "strict_source_attention", "dual_stream_adapter"))
    dual.reviewed_policy()
    result = dict(status="running", mode=MODE, kind="D_CPU_small", fixture_policy_sha256=policy_sha,
                  precollective_metadata=metadata_checks(adapter, torch), tests=[],
                  checkpoint_loaded=False, npu_initialized=False, autograd_tested=False)
    generator = torch.Generator().manual_seed(9831)
    with torch.inference_mode():
        for case in CASES:
            data = fixture(case)
            layout = adapter.infer_strict_source_layout(**tensors(data, torch))
            source_layout = dual.source_linear_layout(layout)
            n, heads, dim, hidden = len(data["coords"]), 8, 8, 16
            candidate_mask_checks(layout, n, dual)
            for dtype in (torch.float32, torch.bfloat16):
                branch = branch_mod.BidirectionalLinearBranch(hidden, heads, dim).to(dtype=dtype)
                initialize(branch, generator, torch)
                reference = old.BidirectionalLinearBranch(hidden, heads, dim).to(dtype=dtype)
                reference.load_state_dict(branch.state_dict(), strict=True)
                frozen = {name: value.clone() for name, value in branch.state_dict().items()}
                x = (.3 * torch.randn((n, hidden), generator=generator)).to(dtype)
                raw = tuple((.3 * torch.randn((n, heads, dim), generator=generator)).to(dtype) for _ in range(3))
                got_soft = dual.source_hybrid_softmax_attention(*raw, layout, dim ** -.5)
                gold_soft = dense_cpu(*raw, layout, torch)
                legacy_soft = adapter.strict_source_softmax_attention(*raw, layout, dim ** -.5)
                t = slice(layout.video_start, layout.video_end)
                s = slice(layout.source_start, layout.source_end)
                gold = torch.zeros((n, heads * dim), dtype=dtype)
                gold[t] = reference(x, raw, layout)
                gold[s] = reference(x, raw, source_layout)
                beta = branch.beta_proj(x)
                chunks, frame_checks = [], []
                means = []
                for each in (layout, source_layout):
                    sum_parts, count_parts = zip(*(branch_mod.local_frame_sums_counts(x[r*n//WORLD:(r+1)*n//WORLD], each, r*n//WORLD) for r in range(WORLD)))
                    counts = sum(count_parts)
                    expected = torch.full_like(counts, each.tokens_per_frame)
                    frame_checks.append(compare("frame-counts", counts, expected, torch, exact=True))
                    means.append(sum(sum_parts) / counts[:, None])
                for rank in range(WORLD):
                    local_raw = tuple(q[:, rank:rank+1].contiguous() for q in raw)
                    target = branch.forward_head_shard(local_raw, beta[:, rank:rank+1], means[0], layout, head_start=rank)
                    source = branch.forward_head_shard(local_raw, beta[:, rank:rank+1], means[1], source_layout, head_start=rank)
                    chunks.append(target + source)
                got_linear = (torch.cat(chunks, dim=1) * branch.output_gate(x)).flatten(1)
                checks = [compare("cpu-d-mask-softmax", got_soft, gold_soft, torch),
                          compare("cpu-target-C-softmax", got_soft[t], legacy_soft[t], torch),
                          compare("cpu-target-original-linear", got_linear[t], gold[t], torch),
                          compare("cpu-source-original-linear", got_linear[s], gold[s], torch)]
                nonvisual = torch.ones(n, dtype=torch.bool);nonvisual[t] = False;nonvisual[s] = False
                checks.append(compare("cpu-linear-nonvisual-zero", got_linear[nonvisual], torch.zeros_like(got_linear[nonvisual]), torch, exact=True))
                if not bool(gold[s].abs().max() > 0) or not bool(gold[t].abs().max() > 0):
                    raise AssertionError("Linear positive control failed")
                if any(not torch.equal(v, frozen[k]) for k, v in branch.state_dict().items()):
                    raise AssertionError("D mutated shared parameters")
                result["tests"].append(dict(case=case, dtype=str(dtype), checks=checks, frame_checks=frame_checks,
                                            mask_proof=audit_masks(layout,n), candidate_mask_exact=True,
                                            both_linear_nonzero=True, frozen_weights_unchanged=True))
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        raise RuntimeError("CPU validation unexpectedly initialized NPU")
    result.update(status="passed", torch_version=torch.__version__)
    return result


def validate_cpu_report(report, policy_sha):
    if (report.get("status") != "passed" or report.get("mode") != MODE or report.get("kind") != "D_CPU_small"
            or report.get("fixture_policy_sha256") != policy_sha
            or any(report.get(k) is not False for k in ("checkpoint_loaded", "npu_initialized", "autograd_tested"))):
        raise RuntimeError("Invalid D CPU report identity/scope")
    tests = report.get("tests", [])
    if len(tests) != 4 or {(t.get("case"), t.get("dtype")) for t in tests} != {(c,d) for c in CASES for d in ("torch.float32","torch.bfloat16")}:
        raise RuntimeError("Missing D CPU cases/precisions")
    if set(report.get("precollective_metadata", {})) != set(CASES):
        raise RuntimeError("Missing D CPU metadata validation")
    expected_checks = {"cpu-d-mask-softmax", "cpu-target-C-softmax", "cpu-target-original-linear",
                       "cpu-source-original-linear", "cpu-linear-nonvisual-zero"}
    for test in tests:
        if any(test.get(k) is not True for k in ("candidate_mask_exact","both_linear_nonzero","frozen_weights_unchanged")):
            raise RuntimeError("Missing D CPU positive-control/mask proof")
        if set(report["precollective_metadata"][test["case"]].get("rejected",[])) != set(REJECTIONS):
            raise RuntimeError("D CPU did not reject all invalid inputs")
        checks=test.get("checks",[])
        if len(checks)!=len(expected_checks) or {x.get("check") for x in checks}!=expected_checks:
            raise RuntimeError("Incomplete D CPU numeric checks")
        frame_checks=test.get("frame_checks",[])
        if len(frame_checks)!=2 or any(x.get("check")!="frame-counts" for x in frame_checks):
            raise RuntimeError("Both source and target frame count proofs required")
        for check in checks+frame_checks:
            exact=check["check"].endswith("-zero") or check["check"]=="frame-counts"
            expected=(0.,0.) if exact else ((1/512,1/128) if test["dtype"]=="torch.bfloat16" else (2e-6,2e-5))
            if (check.get("atol"),check.get("rtol"))!=expected:
                raise RuntimeError("D CPU tolerance changed")
            if any(type(check.get(k)) not in (int,float) or not math.isfinite(check[k]) or check[k]<0 for k in ("max_abs","rms_error","golden_rms")):
                raise RuntimeError("Invalid D CPU metrics")
            if exact and (check["max_abs"]!=0 or check["rms_error"]!=0):
                raise RuntimeError("Exact D CPU count/zero check failed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_cpu(args.candidate_root, args.golden, args.policy, args.policy_sha256)
    validate_cpu_report(report, args.policy_sha256)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status=report["status"], tests=len(report["tests"]), output=str(args.output))))


if __name__ == "__main__":
    main()
