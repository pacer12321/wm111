"""CPU-only regression for the opt-in B loading thread scope.

Executes the actual two pipeline loading methods extracted with AST, replacing
only the large model/checkpoint operations with tiny spies.  Tests the real
thread context and real LoRA merge.  No vLLM imports, checkpoints or NPU calls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Iterable

import torch


class ExpectedLoadFailure(RuntimeError):
    pass


def sha256(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def run_sources(helper_source, pipeline_source, test_source):
    helper = ModuleType("scope_regression_checkpoint")
    sys.modules[helper.__name__] = helper
    exec(compile(helper_source, "openvdn_checkpoint.py", "exec"), helper.__dict__)

    pipeline_ast = ast.parse(pipeline_source)
    pipeline_class = next(
        item for item in pipeline_ast.body
        if isinstance(item, ast.ClassDef) and item.name == "MiniMaxH3Pipeline"
    )
    methods = [
        item for item in pipeline_class.body
        if isinstance(item, ast.FunctionDef) and item.name in ("load_weights", "_load_weights_impl")
    ]
    assert len(methods) == 2
    # Only methods under test are compiled: never import/instantiate the model.
    tiny_class = ast.ClassDef(name="TinyPipeline", bases=[], keywords=[], body=methods, decorator_list=[])
    tiny_module = ast.fix_missing_locations(ast.Module(body=[tiny_class], type_ignores=[]))

    checks = []
    initial_intraop = torch.get_num_threads()
    initial_interop = torch.get_num_interop_threads()
    real_set_intraop = torch.set_num_threads
    real_set_interop = torch.set_num_interop_threads
    changes = []

    def tracked_set_intraop(value):
        changes.append(value)
        return real_set_intraop(value)

    def forbidden_set_interop(value):
        raise AssertionError(f"Interop threads must never be set; requested {value}")

    try:
        real_set_intraop(3)
        torch.set_num_threads = tracked_set_intraop
        torch.set_num_interop_threads = forbidden_set_interop
        for failure in (None, "base", "branch", "lora", "post"):
            events = []
            call_changes = len(changes)

            def observe(phase):
                count = torch.get_num_threads()
                events.append({"phase": phase, "intraop": count})
                assert count == 4
                assert torch.get_num_interop_threads() == initial_interop
                if failure == phase:
                    raise ExpectedLoadFailure(phase)

            def fake_strict(model, weights, *, prefix):
                observe("base")
                for name, tensor in weights:
                    assert name.startswith(prefix)
                    yield name[len(prefix):], tensor

            def fake_complete(model, checkpoint, *, phase_timings):
                for phase in ("branch", "lora"):
                    began = time.monotonic()
                    observe(phase)
                    phase_timings[phase] = time.monotonic() - began
                return {"branch.weight"}, {"phase_seconds": dict(phase_timings)}

            namespace = {
                "torch": torch, "Iterable": Iterable, "time": time,
                "logger": logging.getLogger("pipeline_scope_test"),
                "openvdn_cpu_loading_scope": helper.openvdn_cpu_loading_scope,
                "strict_base_weights": fake_strict,
                "load_complete_openvdn_checkpoint": fake_complete,
            }
            exec(compile(tiny_module, "pipeline_load_methods.py", "exec"), namespace)
            pipeline = namespace["TinyPipeline"]()
            pipeline._openvdn_checkpoint = Path("fake-stage-b")
            pipeline.transformer = SimpleNamespace(
                load_weights=lambda weights: {name for name, tensor in weights},
                post_load_weights=lambda: observe("post"),
            )
            for name in ("text_encoder", "video_vae", "audio_vae"):
                setattr(pipeline, name, SimpleNamespace(named_parameters=lambda: ()))
            try:
                result = pipeline.load_weights(iter((("transformer.base.weight", torch.zeros(2, 2)),)))
            except ExpectedLoadFailure as error:
                assert failure is not None and str(error) == failure
            else:
                assert failure is None
                assert result == {"transformer.base.weight", "transformer.branch.weight"}
            assert changes[call_changes:] == [4, 3]
            assert torch.get_num_threads() == 3
            assert torch.get_num_interop_threads() == initial_interop
            record = pipeline._openvdn_cpu_loading_record
            assert record["intraop_before"] == record["intraop_after"] == 3
            assert record["intraop_loading"] == 4
            assert record["interop_before"] == record["interop_loading"] == record["interop_after"] == initial_interop
            assert record["status"] == ("completed" if failure is None else "failed")
            assert record["scope_seconds"] >= 0
            if failure is None:
                assert set(record["phase_seconds"]) == {"base", "branch", "lora"}
                assert all(value >= 0 for value in record["phase_seconds"].values())
            checks.append({"check": f"actual-pipeline-scope-{failure or 'success'}", "events": events, "record": record})

        # A must neither enter the context nor change the process thread count.
        pipeline._openvdn_checkpoint = None
        before = len(changes)
        pipeline.transformer.post_load_weights = lambda: None
        result = pipeline.load_weights(iter((("transformer.base.weight", torch.zeros(2, 2)),)))
        assert result == {"transformer.base.weight"}
        assert len(changes) == before and torch.get_num_threads() == 3
        checks.append({"check": "A-bypasses-thread-scope", "passed": True})

        # Original function/default chunk and cast-before-add math, unchanged.
        import inspect

        assert inspect.signature(helper.merge_lora_pair_).parameters["chunk_rows"].default == 256
        for dtype in (torch.float32, torch.bfloat16):
            generator = torch.Generator(device="cpu").manual_seed(197)
            a = (torch.randn((64, 96), generator=generator) * 0.03).to(dtype)
            b = (torch.randn((263, 64), generator=generator) * 0.03).to(dtype)
            base = (torch.randn((263, 96), generator=generator) * 0.1).to(dtype)
            expected = base + (b.float() @ a.float()).to(dtype)
            actual = base.clone()
            with helper.openvdn_cpu_loading_scope():
                helper.merge_lora_pair_(actual, a, b, name="tiny_scope_math")
            assert torch.get_num_threads() == 3
            exact = dtype == torch.bfloat16
            torch.testing.assert_close(actual, expected, atol=0 if exact else 1e-7, rtol=0 if exact else 1e-6)
            checks.append({
                "check": "original-FP32-delta-cast-add-with-default-chunk256",
                "dtype": str(dtype), "max_abs": float((actual.float() - expected.float()).abs().max()),
                "bf16_exact_required": exact,
            })
        try:
            with helper.openvdn_cpu_loading_scope():
                helper.merge_lora_pair_(
                    torch.zeros(3, 4), torch.full((64, 4), float("nan")),
                    torch.zeros(3, 64), name="nonfinite_scope_test",
                )
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite LoRA must still fail strict validation")
        assert torch.get_num_threads() == 3
        checks.append({"check": "strict-nonfinite-rejection-restores-threads", "passed": True})
    finally:
        torch.set_num_threads = real_set_intraop
        torch.set_num_interop_threads = real_set_interop
        real_set_intraop(initial_intraop)
    assert torch.get_num_threads() == initial_intraop
    assert torch.get_num_interop_threads() == initial_interop
    return {
        "status": "passed", "device": "cpu", "torch_version": torch.__version__,
        "source_sha256": {
            "patched/openvdn_checkpoint.py": sha256(helper_source),
            "patched/pipeline_minimax_h3.py": sha256(pipeline_source),
            "tests/cpu_loader_thread_scope_regression.py": sha256(test_source),
        },
        "initial_intraop": initial_intraop, "final_intraop": torch.get_num_threads(),
        "initial_interop": initial_interop, "final_interop": torch.get_num_interop_threads(),
        "checks": checks,
        "limitations": "Actual pipeline method orchestration and real CPU scope/LoRA math; large base/branch checkpoint operations are tiny spies. No full-model load, NPU, remote candidate deployment or active-run mutation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    helper = (args.base / "patched/openvdn_checkpoint.py").read_text(encoding="utf-8")
    pipeline = (args.base / "patched/pipeline_minimax_h3.py").read_text(encoding="utf-8")
    test = Path(__file__).read_text(encoding="utf-8")
    report = run_sources(helper, pipeline, test)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"status": "passed", "checks": len(report["checks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
