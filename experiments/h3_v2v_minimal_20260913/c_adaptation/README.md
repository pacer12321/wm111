# C: strict same-latent-frame visual-source Softmax candidate

Status: isolated implementation, **not deployed, not full-model/NPU validated,
and no speed or edit-quality claim**. This folder does not modify B. C reuses
B weights and the same linear branch, QKV, RoPE, gates, denoise schedule, source
noise augmentation and Ulysses communication functions.

## Exact change

Start from B's c5/r1 target-window + first/last target row/column anchors. Delete
only target-query × visual-source-key edges whose latent-frame ordinals differ.
First/last target queries also get this source restriction, while still seeing
all target keys. Text, reference audio, target audio and source-query edges are
unchanged. Trailing packed padding is never a key and has zero attention output.

Source rows are exactly `img_pos[~update_mask]`, not the global prefix. Source
and target frame ordinals come from each clip's own consecutive temporal groups.
Their absolute RoPE time origins may differ and are **not** compared or changed.
Each target frame gets a separate query group so adjacent target frames sharing
a B window cannot accidentally union their source-frame keys.

This restriction is local to each Softmax call. Source queries still attend
globally and therefore hidden states may indirectly carry cross-frame information
across layers. C is not an information-theoretic temporal isolation guarantee.

## Supported / rejected

- Exactly one explicitly declared video reference, with optional soundtrack;
  input preflight runs on each worker before video preparation/encoding.
- Actual encoded source Fs equals target Ft; patch size `(1,2,2)`, >=3 latent
  frames, finite ordered complete spatial grids. Different source/target spatial
  areas are supported; no coordinate rounding or absolute-time equality.
- Complete source/target/audio/text boundaries are cross-checked against the
  actual unsharded packed arrays, not just against requested pixel frame counts.
- T2VA/FL2VA, images, multi-reference, missing or malformed metadata, short
  source Fs mismatch, alternative B interior-group masks and dense fallback are
  rejected. No implicit padding/truncation is added.
- Metadata is forwarded through `DenoiseBranch.static_kwargs`, explicitly
  whitelisted by the transformer, validated before RoPE/refiner/SP attention.
  This relies on the existing pipeline's replicated request/packed metadata;
  it does not introduce a distributed recovery protocol for arbitrarily
  divergent inputs or worker failures.

## Files and reproducible rendering

`strict_source_layout.py`: no-dependency validation and compact span plan.
`strict_source_attention.py`: actual torch adapter using B's unchanged fusion
kernel; its private cache includes the complete source/target layout and mode.
`build_candidate.py`: exact-anchor mechanical patching of local B copies, with
AST guards freezing attention QKV/linear/gate/SP methods and byte-text guards
for `openvdn_npu.py` / checkpoint helper. It never edits B or overwrites output.

```text
python c_adaptation/build_candidate.py --check
python c_adaptation/build_candidate.py --output c_adaptation/candidate
python -m unittest discover -s c_adaptation -p "test_strict_source*.py" -v
```

Paths above are relative to `experiments/h3_v2v_minimal_20260913`. Rendered
`candidate/` contains all six files to overlay into a **separate** C vendor
checkout, plus a SHA manifest. It is not an instruction to overwrite B or to
launch hardware jobs. Upstream snapshots under `upstream/` are read-only
context/test fixtures from the actual 30213 personal vendor (including packer
and denoise loop); they are not intended to replace installed modules.

## Validation scope and remaining gate

The standard-library suite has 26 tests: independent full Boolean B-minus-C
oracle, fixed-QKV numerical grouped/dense agreement, extreme wrong-source value
leakage, anchors/nonvisual/source queries, nonuniform/noninteger time with
different origins, Fs/multi-ref rejection, source-area and layout cache changes,
padding and CPU-emulated row shards, plus integration source/AST guards.

`test_strict_source_torch.py` adds 5 actual torch CPU tests: production adapter
vs independent SDPA mask, actual B→C→B cache isolation, head-shard emulation,
the actual vendor packer, and fail-closed metadata. **On this workstation torch
and numpy are absent, so those tests are explicitly skipped, not counted as
runtime passes.** Run them in a CPU-capable H3 runtime before an NPU tiny gate.

No HCCL/all-to-all test, NPU fusion-kernel test, full-model generation, image
inspection or benchmark was run for C. Current full-coordinate CPU validation
each forward and per-frame query/key gathers can add overhead. First establish
correctness, then measure complete B/C steps; reduced attention edges alone do
not establish faster execution. Fixed same-frame C can fail on temporal edits;
ABC temporal-edit experiments are a test of that boundary, not a promise that
the fixed correspondence is semantically valid.
