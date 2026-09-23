"""Reject stale/partial preflights before spending GPUs on a fresh retrain.

The report is produced from actual test receipts, not from this gate. The gate
never fabricates a passing result and never treats pending tests as passed.
"""
import argparse
import hashlib
import json
from pathlib import Path

try:
    from src.training.ref2va_base_contract import RECEIPT_NAME, verify_ref2va_base
except ImportError:  # run beside the module, as the unit tests do
    from ref2va_base_contract import RECEIPT_NAME, verify_ref2va_base

REQUIRED = (
    "dataset_pair_qc_and_coverage", "source_disjoint_split", "shared_clip_encoders",
    "visual_prompt_tags", "dmd8_weights_and_schedule", "train_infer_full_inputs",
    "eager_and_compiled_mask_equivalence", "real_shape_compiled_forward_backward",
    "hsdp_initialization_inputs_and_gradient_sync", "trainable_parameter_scope",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def verify(report, manifest, sample_dir, output, audio_policy, base):
    if report.get("schema") != "savie-retrain-readiness-v1":
        raise ValueError("missing full retrain readiness report; tag-only preflight is insufficient")
    if report.get("audio_input_policy") != audio_policy or not audio_policy:
        raise ValueError("audio policy does not match validated training/inference configuration")
    if report.get("starting_point") != "fresh-dmd8-new-lora":
        raise ValueError("must initialize from correct DMD8, not old SAViE checkpoint")
    # Evidence gathered on the FL2VA h3-base says nothing about the Ref2VA model we serve.
    verify_ref2va_base(base)
    if report.get("base_receipt_sha256") != sha(Path(base) / RECEIPT_NAME):
        raise ValueError("readiness evidence was not produced on this Ref2VA base; rerun the checks")
    if report.get("train_token_skip") is not False or report.get("audio_objective") is not False:
        raise ValueError("unexpected training objective or token-skip setting")
    if Path(report["sample_dir"]).resolve() != Path(sample_dir).resolve():
        raise ValueError("sample directory differs from tested directory")
    if report.get("manifest_sha256") != sha(manifest):
        raise ValueError("manifest changed after validation")
    rows = [json.loads(s) for s in Path(manifest).read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows) != 2000 or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("expected exactly 2K unique selected samples")
    hashes = report.get("code_sha256", {})
    if len(hashes) < 5:
        raise ValueError("missing checked code hashes")
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"code changed since tests: {path}")
    checks = report.get("checks", {})
    for name in REQUIRED:
        check = checks.get(name, {})
        if check.get("passed") is not True:
            raise ValueError(f"required check not passed: {name}")
        path = check.get("receipt_path")
        if not path or sha(path) != check.get("receipt_sha256"):
            raise ValueError(f"missing or changed test evidence: {name}")
    # Streaming training need not wait for all 2K encodes. A validated initial
    # buffer is enough; every later sample is identity/shape-checked on load.
    ready = int(report.get("initial_buffer_size", 0))
    if ready < 4 or ready > len(rows):
        raise ValueError("need a validated initial buffer for both DP replicas")
    for index in range(ready):
        for kind in ("video", "prompt"):
            stem = Path(sample_dir) / f"{kind}_{index:06d}"
            if not stem.with_suffix(".done").is_file() or not stem.with_suffix(".pt").is_file():
                raise ValueError(f"initial streaming buffer incomplete: {stem}")
    if any(Path(output).glob("savie_step*.pt")) or any(Path(output).glob("train_state_step*.pt")):
        raise ValueError("fresh run output contains old training state")
    return dict(passed=True, selected_samples=len(rows), initial_buffer_size=ready)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("report", "manifest", "sample-dir", "output", "base"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--audio-policy", required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(verify(report, args.manifest, args.sample_dir, args.output, args.audio_policy,
                            args.base)))


if __name__ == "__main__":
    main()
