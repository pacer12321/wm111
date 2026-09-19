# Track B: isolated shard-at-read prototype and source audit

Status: 2026-09-15. **Not deployed. No server candidate, guard, queue, or GPU job was modified.**

## Verified source identity

Read-only inspection of port 30674, `/cache/zhonghao/h3/a100_v1/candidates/D`.
These remote SHA-256 hashes matched the existing local copies:

- `vllm_omni/diffusion/models/minimax_h3/openvdn_checkpoint.py`: `fbe011bae524bea16f54a14032e61e82f2c68aa4da6d426b98a13e097fb8f19f`
- `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py`: `98da3d83984850cc8df47a6a122b93d662c361baccf90f29f24e120e5c05f4dd`
- `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`: `a3dc6a0189fdd8f3a9e2aaeddf9544df1503a2f74af287b74b7e975dd7ce717c`
- `vllm_omni/diffusion/model_loader/diffusers_loader.py`: `63e794f03d9049b03baba3dc59a955409b1baca0651b61b6c1f1b9b6f38bb28f`
- `vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`: `c716e22f42f9b322a3ff0658d3d17aa9aa809ed96feef218ee2522e8d3755b58`

All following line references are relative to that `vllm_omni/diffusion/` directory.

## Important correction: two different loading paths

1. **Current H3 with only the DLO prohibition removed** follows the regular loader, not automatically mmap. `offloader/offload_plan.py:supports_mmap_loading` requires a module's class to expose `_remap_ckpt_key`. That method is absent from current H3 sources and the checkpoint's Python VAE sources. The shared gate is in `model_loader/diffusers_loader.py:366-400`, matching backend `:1105-1126`.
2. That regular path retains H3/VDN transformations, but fully loads each rank before `DistributedLayerwiseOffloadHook._shard_and_pin` runs. **It does not solve the load-time private-memory peak.**
3. If mmap capability is deliberately added, the loader bypasses `pipeline.load_weights()` and `_process_weights_after_loading()`. That is the path needing the following explicit adaptations. Merely removing the prohibition is neither a complete conversion nor a safe optimization.

## Specific bypass/compatibility checklist

### Base checkpoint and QKV

- Regular path: `models/minimax_h3/pipeline_minimax_h3.py:378-406` calls `strict_base_weights`, original transformer loader, complete Stage-B loader, then `post_load_weights`.
- `openvdn_checkpoint.py:strict_base_weights` checks 535 base tensors, full 50 DiT blocks plus two refiner blocks, exact names/shapes/dtypes, finite values and CPU residency.
- `minimax_h3_transformer.py:419-435` installs the QKV weight loader. It converts per-head `[Q,K,V]` checkpoint row order into `[all Q, all K, all V]`. This applies to 50 DiT + 2 refiner QKV weights.
- The mmap backend `:726-790` assigns checkpoint tensors directly to `nn.Parameter`; it does **not** call these custom weight loaders. A name-only remap cannot implement the row permutation.
- `minimax_h3_transformer.py:688-701` also installs FC1 gate/up loading. At the required DiT TP=1, input order is already gate followed by up, so bypassing it is expected to be numerically identity, but exact tensor shape/order still belongs in the equivalence test. Do not generalize that conclusion to TP>1.
- Fix: compute rank ownership in final model layout, inverse-map QKV rows to source rows, and read only those rows/elements. Preserve complete manifest validation separately; validate values of every loaded fragment, and verify distributed coverage accounts for the entire checkpoint.

### Stage-B 800 branch tensors

- `openvdn_checkpoint.py:branch_name_map` maps 16 suffixes across 50 blocks (`transformer_blocks.i.attn.*` to `blocks.i.attn.*`). These include linear attention alpha/beta/norm/gates/short-convolution tensors, softmax gate and `to_out_linear.weight`.
- The regular path loads `stage-b-step-2000/linear_branch/model.safetensors` explicitly and verifies all 800 entries.
- Generic mmap scans the original H3 `model_path`; the separate Stage-B path is not a weight source there. Its reverse name map does not supply these 800 tensors. No correct VDN model can result from skipping them.
- Fix: a two-source manifest must include all 535 base + 800 branch tensors, each with an exact source file/key. Reject missing/unexpected/duplicate keys and shape/dtype mismatches before payload loading; rank-local reads are then independent of checkpoint directory layout.

### LoRA 208 pairs, not 208 tensors

- Regular path loads 416 adapter tensors (208 A/B pairs): Q/K/V/O for 50 DiT and 2 refiner blocks, rank 64, alpha 64, scale 1.
- It first loads/reorders base, then loads branch, then merges LoRA into the reordered contiguous Q/K/V thirds and output projection. It computes FP32 `B @ A`, casts the delta to the parameter dtype, then performs the parameter-dtype add (`merge_lora_pair_`). Do not add in FP32 and cast only once; that is a different result.
- Generic mmap bypasses the entire pipeline method, so none of these merges run. Calling `post_load_weights` cannot recover them.
- Fix: identify final-layout rows belonging to this rank, read needed B rows and A, compute the same cast-before-add merge, and write only into the local shard. If a flattened shard ends mid-row, compute that row's delta and retain only owned elements. No worker should retain an unmerged shard while another merges it later.

### post-load and buffers

- **Not skipped**: backend `:825-832` explicitly calls each DiT's `post_load_weights`.
- H3 `minimax_h3_transformer.py:1093-1099` only checks designated FP32 parameters/buffers; it does not reorder QKV or load/merge Stage-B.
- Backend saves/restores non-persistent buffers around `to_empty(meta)`, but this is separate from strict base/branch/LoRA validation. Both must be preserved.
- Meta-first construction needs explicit non-persistent buffer reinitialization/saving; a blanket `with torch.device('meta')` around the whole pipeline would also affect encoder/VAE and is not an acceptable shortcut.

### Read-time sharding versus full loading followed by slicing

- Regular path does full per-rank private loading, then `:198-273` flattens metadata and copies only the rank overlap to a shard. The destination is smaller, but the earlier peak remains.
- Native mmap path calls `safe_open(...).get_tensor` to create a full-tensor mapping and `_shard_and_pin` copies only the rank's intersecting flat range. This is not explicit safetensors `get_slice`/byte-range reading, but it avoids a second full **private** base-weight copy when no transformation writes to the mapping.
- Backend comments saying mmap has "no RSS" are too strong: faulted mapped pages count as file-backed RSS/page cache and still interact with the cgroup limit. Read-ahead and file-cache reclamation remain relevant.
- The current generic mmap method converts an already constructed DiT to meta (`:575-646`); it is not a guarantee that construction itself never allocates full-shaped CPU parameters. The isolated design must construct DiT metadata/parameters on meta **before** large allocation and load each final shard directly.
- Avoid full QKV permutation tensors or whole-model LoRA merges on mmap-backed arrays: they can recreate full private copies. Direct rank-owned reads and bounded row temporaries make this boundary explicit.

### SP communication and non-DiT lifecycle

- `distributed_layerwise_backend.py:905-950` uses the existing SP group when DP=1; local pinned CPU shards + H2D + AllGather can feed the same complete-layer computation while preserving Ulysses token parallelism.
- **Additional actual incompatibility**: `_register_on_demand_hook` at `:952-966` currently performs `module.to(self.device)` and keeps the encoder/VAE GPU-resident. The nearby older comment promising on-demand CPU movement is not what the method does.
- H3's normal path at `pipeline_minimax_h3.py:723-730` explicitly loads the encoder for text encoding and offloads it afterward. DLO flags and that branch must be reconciled; a residency change is not a pure CPU storage optimization.
- VAE/encoder were loaded independently during pipeline construction. DLO's DiT shard loader does not fix their initial peaks or automatically shard them. Do not shard text-encoder TP-specific weights over SP again without checking ownership/equality; different TP ranks do not own identical encoder tensors.
- Token-refiner and other non-block DiT modules also need explicit ownership/residency accounting; discovering the 50 main blocks is not full-pipeline coverage.

## Isolated prototype delivered here

`streaming_shards.py` implements a read-only safetensors byte-range reader, rank layout planner, QKV inverse row mapping, BF16/F32 shard storage and a **NumPy reference** of rank-local Stage-B LoRA merging.

It does not instantiate H3. It accepts an explicit ordered tensor plan and returns arrays + `name/offset/numel/shape` metadata corresponding to the existing DLO hook's format. The caller must retain production `named_parameters + named_buffers` order. Current implementation is deliberately small and unoptimized (row-local LoRA reads); it is not a serving implementation or speed result.

9 CPU synthetic tests passed locally using existing `.audio_tools/numpy` (no installs):

- Mixed BF16/F32 layout, per-dtype metadata, padding, rank reconstruction.
- Rank reads only its 50% source range in an identity tensor; bounded individual reads.
- Exact grouped-QKV conversion reconstructed across 1/2/3/7 ranks.
- Cast-before-add QKV LoRA then sharding matches an independent full-reference calculation on binary-exact synthetic inputs, including shard cuts mid-row.
- Output-projection LoRA and branch identity mapping.
- Invalid group/name/layout/LoRA inputs, truncated payload, non-finite values fail closed.

Run from this directory in PowerShell:

```powershell
$env:PYTHONPATH='C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/.audio_tools'
python -m unittest -v test_streaming_shards.py
```

## Follow-up: real header manifest and torch-path implementation

Collected **only 219,360 bytes of safetensors headers** plus small config JSON
from 15 files on port 30674. No tensor payloads were read. Saved locally:

- `real_header_snapshot.json`: SHA-256 `a481e8b225998f2faf57432677e147f03057de1977e5f2feb0d37e21a300e4c2`.
- `real_manifest.json`: SHA-256 `39932a5b3dcd3ff275cf538714b213505d3a522beb7148d81682b53f108fcae1`.
- `manifest_builder.py`: header collection/audit, explicit plan entries, strict real-meta metadata attachment, and guarded one-block conversion into `TensorPlan` objects. Header collection was run once; repeat audits use the local snapshot.

Real counts verified: 535 base tensors + 800 branch tensors + 416 adapter
tensors (208 LoRA pairs). The model manifest has 1,335 target entries and
52 QKV transformations. **It deliberately says `runtime_usable=false`: real
model enumeration and runtime integration are still unverified.** Base key
count alone is not proof of exact model coverage; that last check is gated
on metadata from the real meta model rather than inferred from key order.

An important architecture detail surfaced in this validation:

- `hidden_size=5376`, but attention width is `56 * 128 = 7168`.
- Fused QKV is `[21504, 5376]`, and O projection is `[5376, 7168]`.
- Q/K/V LoRA A is `[64,5376]`, B is `[7168,64]`; O LoRA A is `[64,7168]`, B is `[5376,64]`.
- Do **not** assume attention projection width equals model hidden size or O is square.

Each of the 50 main blocks has 26 BF16 tensors totaling 1,376,730,080 bytes.
The main-block weight payload totals 68,836,504,000 bytes; with two equal
flat shards, each rank would retain 34,418,252,000 bytes for these blocks.
These block sizes divide evenly by two, so no padding is needed for the
current two-rank main-block buffers. Padding logic is nevertheless tested
for arbitrary group sizes. These are **payload estimates, not process RAM
forecasts**: they exclude pinning/transients/encoder/VAE/cache/communication.
Another 35 target entries (1,723,246,144 bytes) sit outside the main blocks;
all 13 F32 entries, including persistent `rope.inv_freq`, are in that group.

### Model order and meta construction

`meta_model_metadata.py` extracts order/shapes/dtypes from an **already
constructed real meta `MiniMaxH3DiTModel`**, validates 50+2 blocks and the
OpenVDN flag, and rejects any real device tensor. It does not substitute
checkpoint order, initialize process groups, load the pipeline or allocate
GPU tensors.

Source inspection shows DiT constructors use ordinary parameter/buffer
factories (including branch `empty/ones` and persistent RoPE `empty`) without
an explicit CPU/GPU device override at those call sites, so constructing
only the DiT under a meta context is a plausible allocation-free metadata
path. However vLLM TP state/current config and its linear constructors must
be initialized/verified by the eventual harness. This path **has not been
executed** here. The entire H3 pipeline must not be instantiated just to get
metadata, since its constructor independently loads encoder and VAEs.

### Torch code added, but torch tests not executed

`torch_streaming_shards.py` implements CPU shard allocation, range reads,
QKV row conversion, and rank-local LoRA merging. Its merge reads only the
intersecting original **256-row aligned chunks**, not arbitrary rank-sized
GEMMs; preserving GEMM shapes reduces an avoidable floating-point-equivalence
hazard. It performs FP32 B@A, BF16 delta cast, BF16 target add. Pinned shard
allocation can be requested directly; it does not call `pin_memory()` on a
second complete buffer.

`test_torch_streaming_shards.py` will compare against the **exact audited
production `merge_lora_pair_` function**, extracted by AST with a SHA guard,
on randomized BF16 inputs, original chunk boundaries and mid-row rank cuts.
It also checks F32 preservation and padded reconstruction. The local Python
does not have torch: **both tests are explicitly skipped**, not passed.
No torch installation or remote execution was attempted while Track A ran.

Latest local suite: **17 tests passed, 2 torch tests skipped** (19 discovered).
This comprises the previous 9 synthetic tests plus 8 real-header/manifest
validation tests. AST syntax checks also passed. Command:

```powershell
$env:PYTHONPATH='C:/Users/DZH/OneDrive/Documents/ChatGPT/wm111/.audio_tools'
python -m unittest -v test_streaming_shards.py test_manifest_builder.py test_torch_streaming_shards.py
```

## Gates still outstanding — do not call Track B complete

1. Real checkpoint-header manifest is generated, but must be attached to the **real meta model** enumeration to verify exact base/branch coverage, parameter/buffer order, shapes and dtypes. Current `model_iteration_index=null` values intentionally prevent runtime use.
2. Execute the new torch CPU tests, then compare representative real tensors and every affected tensor family. Small synthetic tests do not prove production numerical equality; changed GEMM row batching can change FP32 reduction rounding. Torch implementation exists but has not run locally.
3. Integrate a prepared-shard handoff into DLO so it consumes `cpu_shards/metadata` directly without first materializing full CPU parameters or recopying shards to full shapes. Allocate pinned memory directly where supported and monitor its transient peak.
4. Preserve H3 encoder/VAE/refiner/buffer lifecycle explicitly; test memory at constructor, each shard, pinning, first AllGather, text encoding, each denoising step, VAE decode and cleanup.
5. On GPU, compare reconstructed full-layer weights against the original loader (raw-byte hashes and max differences), then block forward outputs for fixed inputs, then B with unchanged seed/source. Verify no NaN/Inf, no double merge and no missing parameters.
6. Only after passing these gates, run **B/C/D all with the same new offload implementation**, same host/settings, to isolate attention changes from memory-transfer changes. Track A exploratory timing is not an official comparison.

## CPU/GPU budget follow-up (separate from weight equivalence)

`memory_budget.py` generated `memory_budget.json` from real H3/VDN headers
and `aux_budget_headers.json`. The latter contains validated TE and audio
VAE headers, but its first collection missed the nested video VAE weight
file. A follow-up read timed out; it was not retried indefinitely. Therefore
the budget currently says **`budget_gate_incomplete`**, not "fits".

Verified quantities, in decimal GB unless bytes are written:

- CPU main-block pinned weight shards: **34,418,252,000 bytes per rank**,
  **68,836,504,000 bytes combined**. Pinned memory IS this storage, not a
  second term to add. A later `unpinned.pin_memory()` copy would cause a
  transient duplicate; the prototype instead requests pinning at allocation.
- Non-main DiT weights: **1,723,246,144 bytes per rank**. The budget's CPU
  envelope conservatively counts both replicas even when a phase moves them
  exclusively to GPU.
- TE retained TP2 payload: **26,348,887,520 bytes per rank**, not simply half
  of the retained checkpoint. Shardable text weights are 50,314,608,640 bytes
  in total; vision **1,190,533,600 bytes** and text norm **1,049,600 bytes**
  are replicated on each rank. Runtime TE dtype is BF16.
- Audio VAE FP32 payload: **605,306,340 bytes per rank** (151,326,585
  elements), from actual header data.
- GPU AllGather buffers: two full layers **2,753,460,160 bytes**, PLUS two
  local input shards **1,376,730,080 bytes** = **4,130,190,240 bytes per GPU**.
  Count both categories; the input shards are not aliases of output buffers.
- Video VAE file was stat'ed at **10,415,548,320 bytes**, but floating dtype
  and element count were not captured. Its exact root config is
  `/cache/zhonghao/h3/models/MiniMax-H3/Ref2VA/video_vae/config.json`;
  `source_safetensors_path` names the nested checkpoint. Read its header,
  then use `sum(numel)*4` for the explicit FP32 runtime contract. File size
  alone is not enough to certify its runtime footprint.

Until that header is supplied, the JSON offers **two conditional scenarios**,
not a rigorous lower/upper proof: if video VAE storage is FP32, its runtime
payload is approximately 10.416 GB; if 16-bit, approximately 20.831 GB. Both
include file header overhead in the approximate number. Other dtype cases
have not been ruled out and would require another bound.

Under those two scenarios, with the root's latest approximately **21 GB CPU
baseline** and a conservative policy retaining CPU copies of all non-main
components on both ranks:

- CPU combined static payload + baseline: **168.02 / 188.85 GB**.
- GPU per-rank static weights/buffers with encoder resident:
  **43.22 / 53.64 GB**.
- GPU per-rank static weights/buffers with encoder returned to CPU before
  denoising: **16.87 / 27.29 GB**.

These terms intentionally provide a conservative simultaneous CPU residency
envelope; actual placement can lower CPU use. **None is a peak-memory
prediction.** The 251-GB CPU quota includes file cache and process overhead.
Each A100 has **80 GiB = 85.899 GB**, not 80 decimal GB. The old GPU0 850-MiB
process consumes another 0.891 GB; GPU1 baseline was reported empty. Any new
external job invalidates admission assumptions, including a CPU recovery
process that later starts GPU work.

### Activations and unmeasured terms

For local token count `L=ceil(total_packed_tokens/2)`, one BF16 hidden tensor
costs `L*5376*2`, fused QKV `L*3*7168*2`, and FFN gate/up
`L*2*14336*2` bytes. At illustrative `L=40,000`, those individual tensors
are 0.430 / 1.720 / 2.294 GB. They are **not** the whole live set or a peak
sum. Ulysses send/receive copies, linear-branch scans, sparse gathers, residuals,
VAEs, text/vision activations, NCCL/FlashAttention/cuBLAS workspace and CUDA
allocator fragmentation remain to be measured. FlashAttention does not
justify ignoring these costs merely because it avoids dense score storage.

CPU also needs room for loader temporaries and file cache. The current TE
loader opens a full safetensors shard before copying its retained TP portion;
the largest observed TE shard file is 4,932,328,944 bytes. File mappings/cache
must not be counted as another fixed private weight replica, but neither may
they be assumed free or immediately reclaimable. Bounded LoRA FP32 chunks
are small relative to model weights; the original production checkpoint load
and allocator-retention paths still require phase measurements.

### Proposed resource acceptance lines — not installed guard changes

1. Fresh read-only preflight: no new foreign GPU job; verify cgroup quota,
   baseline usage, device memory and ownership at the actual launch time.
2. After real pinned shards and component placement, aim for at least
   **32 GiB working-set headroom on CPU**, using `usage - inactive_file` as
   an estimate rather than subtracting all file cache as guaranteed reclaimable.
3. Before first denoising, require **at least 20 GiB device free per GPU** as
   a provisional admission line; then measure the first real block/step.
   Aim for peak total use <=70 GiB/GPU (10 GiB margin) before a long run.
   These are engineering margins, not proven minimum requirements.
4. Monitor loading, pinning, encoder, AllGather, every denoising step and VAE
   decoding through completion. Keep GPU allocated/reserved/device-used,
   CPU rank RSS/PSS and cgroup statistics separate. No phase passing alone
   establishes end-to-end capacity.

The root separately reports that its server CPU suite and full 1,335-tensor,
208-pair/52-QKV real-weight bitwise validation passed (370.478 s,
peak RSS 1,570,712 KiB). Those are the root's reported test results, not this
budget agent's independent execution; they do not close the memory gate.
The meta harness is now with the root for execution. It lives in
`run_meta_harness.py` and was syntax-checked locally but not run here.

The user's latest result ordering is **B/D generated-video timing first,
then quality**, with identical offload/settings and all equivalence/runtime
gates preserved. Earlier B/C/D wording above records the prior plan; it must
not override that latest instruction or mix legacy offload timings.

### Video VAE header resolved by root at 07:43 CST

The root completed the missing read-only check of
`video_vae/source/model.safetensors`: header 64,184 bytes, all F32,
2,603,871,032 elements, payload **10,415,484,128 bytes**. The arithmetic
`8 + 64184 + 10415484128 = 10415548320` matches the previously observed file.
Evidence was recorded in `video_vae_verified_header.json` (root-reported
header evidence, not a payload read by this agent).

`memory_budget_verified_video.json` supersedes the earlier two conditional
video scenarios. Its status is **static payload budget only, not fit proof**:

- Combined CPU conservative model residency + supplied 21-GB baseline:
  **168,022,352,264 bytes**; 82,977,644,152 bytes remain under the cgroup hard
  limit **before** unknown cache/allocator/transient costs.
- GPU static per-rank payload with TE resident: **43,223,114,372 bytes**.
- GPU static per-rank payload with TE returned to CPU for denoising:
  **16,874,226,852 bytes**.
- GPU0 theoretical static remainder with TE resident and old 850-MiB task:
  **41,784,941,948 bytes**, still before activations/CUDA workspace and other
  unknowns. These are not guarantees that runtime will fit.

The initial two-scenario JSON is retained as audit history. Four budget
arithmetic/failure-gate tests passed locally. No running guard was changed.

## Prepared-shard hook seam (isolated, not pipeline integration)

`prepared_shard_hook.py` adds a narrow interface instead of changing the
existing DLO backend or serving candidates:

1. `bundle_from_streaming` translates loader output string dtypes and full
   model names into native hook dtype keys and block-relative names.
2. `prepare_all_meta_blocks` checks **all** bundles, exact parameter/buffer
   order, shapes, offsets, dtype, local rank/world size, contiguity and pinned
   requirement before mutating any block. It then replaces every meta
   parameter/buffer in the main block set with a **zero-element CPU**
   placeholder. Never allocate full shapes via `to_empty('cpu')`.
3. Only after all blocks are prepared, `register_prepared_hook` registers a
   subclass of the real `DistributedLayerwiseOffloadHook`. The subclass's
   `_shard_and_pin` transfers references to prebuilt `cpu_shards/metadata`;
   the superclass's full-parameter sharding implementation is never invoked.
   Native initialization still builds repoint tables and AllGather sizes;
   prefetch, collectives, slot rotation and offload methods are inherited.

Why placeholders must be prepared as a set: assigning real CPU/GPU storage
directly to a meta Parameter is not a safe `.data` transition, and replacing
Parameter objects incrementally while registering a circular hook chain
would leave older hooks holding stale references to their current blocks.
The seam detects changed identities and fails closed. Parameter objects are
replaced with ordinary `nn.Parameter`, like upstream mmap assignment, while
retaining Python attributes and requires-grad flags. This is an explicitly
forward-only, unquantized contract: no checkpoint loader may be reused after
handoff, and tied tensors/previous hooks are rejected.

Ownership constraints: each prepared bundle can be claimed only once,
group/rank must agree, and pinning must already have occurred. The hook does
not call `pin_memory()` to create a hidden second CPU shard. Caller retains
no permission to mutate shard values after handoff; this is not a training
or live-weight-update interface.

**Non-main component lifecycle is intentionally outside this function.**
It does not call generic DLO `enable()` or `_register_on_demand_hook`, and
does not touch encoder/VAE/refiner/top-level buffers. A future isolated H3
integration must load/validate its non-main tensors, preserve ordinary H3
encoder load/encode/finally-offload behavior, then build the main-block ring
and allocate the same two full output + two half input GPU buffers. Keep
VAE and non-main placement consistent with the comparison baseline; never
silently inherit generic DLO's permanent encoder residency.

Local validation:

- `test_prepared_shard_hook.py`: **10 mock CPU protocol tests passed**,
  including superclass sharding never called, zero-copy dictionary/storage
  handoff, wrong rank/size/order/pinning rejection, duplicate claims and stale
  references.
- `test_prepared_hook_torch.py`: **2 actual torch/native-hook tests written
  but skipped locally** (runtime not installed). They use real meta blocks,
  CPU output/input buffers and a fake AllGather to check native two-slot
  repointing, storage aliases, reuse and all-before-any placeholder validation.
  Even when those pass remotely, they are **not CUDA/NCCL async tests**.

Remaining integration gates: actual meta enumeration, real-runtime CPU seam
tests, isolated full-model loader routing, non-main lifecycle preservation,
GPU AllGather with real rank shards, observed stage memory, end-to-end smoke,
then identical-offload B/D timing followed by quality. Do not describe this
seam as DLO connected or the memory budget as runtime accepted.

## Opt-in isolated full-pipeline integration (implementation, not E2E accepted)

`h3_prepared_integration.py` now joins the independently validated storage
pieces. It is intended only for **separate clones of B and D**, never an
in-place edit of the original candidates. Attention code, precision, request,
seed and original H3 text-encoder load/encode/finally-offload method remain
unchanged. Keep ordinary `enable_layerwise_offload=True`; **do not turn on
the generic distributed-layerwise flag**. The integration replaces only the
main DiT blocks' storage implementation while retaining ordinary non-main
component placement/lifetime.

Required runtime kit (same exact bytes for B and D):

- `h3_prepared_integration.py`
- `prepared_shard_hook.py`
- `torch_streaming_shards.py`
- `streaming_shards.py`
- `manifest_builder.py`
- `meta_model_metadata.py`
- `real_manifest.json` (paths must match selected server checkpoints;
  metadata is reattached from the actual meta model on every load)

Also transfer `generate_pipeline_patch.py` if generating the patch on the
server. CPU verification files are `test_prepared_integration.py`,
`test_prepared_shard_hook.py` and `test_prepared_hook_torch.py`. Existing
`B_prepared_bootstrap_lf.patch` and `D_prepared_bootstrap_lf.patch` are
portable-LF review artifacts; the older non-`_lf` files are not preferred.

Generate a patch against each **isolated clone's actual source**, from its
repository root (this command only writes the named patch; it does not
apply it). Replace `/ABS/KIT` and output with approved isolated paths:

```bash
python /ABS/KIT/generate_pipeline_patch.py \
  --source vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py \
  --output /ABS/KIT/isolated_B_bootstrap.patch
git apply --check /ABS/KIT/isolated_B_bootstrap.patch
git apply /ABS/KIT/isolated_B_bootstrap.patch
```

Repeat for isolated D with a different patch output name. Do not run `git
apply` in the original B/D candidate. No application or remote deployment
was performed by the subagent. Add `/ABS/KIT` to the launcher's `PYTHONPATH`
and set `ZHONGHAO_H3_PREPARED_OFFLOAD=1` plus
`ZHONGHAO_H3_PREPARED_MANIFEST=/ABS/KIT/real_manifest.json`. DiT configuration
must be TP1, Ulysses2, ring1, DP1, CFG1, unquantized, pinned CPU storage.

Load sequence and fail-closed checks:

1. Construct the real DiT under `torch.device('meta')`, leaving original
   TE/VAE constructors intact. Enumerate actual order and reject unexpected
   nonpersistent buffers, mismatched checkpoint recipe, changed header
   SHA/file length, or a manifest outside the selected model/Stage-B roots.
2. Ignore the generic full-checkpoint iterator. Read/convert/merge only each
   rank's final portions of 50 main blocks, directly into pinned shards.
   Load the 35 non-main tensors through the same verified conversion/LoRA
   machinery with world size 1, and run H3's F32 `post_load_weights` checks.
3. Generic loader postprocessing normally calls `.to(cpu)` even before
   the unquantized callback, which cannot operate on the remaining meta
   main weights. Skip this only for the audited CUDA unquantized no-op;
   reject any other quantization method/platform. This bypass is logged.
4. Prepare **all** main-block zero-element CPU placeholders before any ring
   hook is attached. Register prepared native hooks on the live SP2 device
   group, allocate **two full output plus two half input buffers**, wire
   native previous-hook links, slots and group-first state. Both input and
   output dictionaries are shared across all 50 hooks; no per-block device
   allocation and no call to the old full-weight `_shard_and_pin` path.
5. Keep the audited native initial prefetch order (it primes the last
   block; group-first entry fetches the first). Preserve original TE/VAE
   lifecycle and keep the existing compiler/cache setup outside this shim.

Required log markers: `H3_PREPARED_STORAGE_CODE_HASHES` (six common source
SHA256s, must agree B/D), `H3_PREPARED_INTEGRATION_INSTALLED`, 50 per-rank
`H3_PREPARED_MAIN` progress records, `H3_PREPARED_LOAD_RECORD`,
`H3_PREPARED_POSTLOAD_NOOP`, and `H3_PREPARED_OFFLOAD_ENABLED`. Missing markers
mean the intended route has not been demonstrated; there is no automatic
fallback to ordinary full-replicated loading.

Validation status:

- Five local no-torch integration tests passed: mode rejection, ring
  topology, no generic iterator consumption, unchanged encoder method,
  ordinary-component moves and exactly one shared input/output allocation.
- Parent reported all 1335 real tensors/208 LoRA pairs/52 QKV conversions
  passed bitwise CPU comparison, and the earlier 12 native-seam tests passed.
- `test_prepared_hook_torch.py` now additionally checks actual native ring
  pre/post-forward and Linear results for 3, 4 and 50 blocks over two turns,
  using CPU buffers/fake synchronous collectives. This **new third test was
  not executed locally**, because torch/vllm-omni are absent. Parent must run
  it in the existing server runtime; no install is needed.
- Neither full pipeline integration, CUDA stream safety, NCCL AllGather,
  encoder/VAE peak memory, cache/compiler behavior, nor E2E equivalence has
  been established by these local tests. Hooks are forward-only, one-shot;
  recovery requires a fresh worker, not reloading into a claimed bundle.

Acceptance order remains real-shard GPU reconstruction/forward validation,
isolated full-model smoke with continuous memory monitoring, then B and D
with the **same storage implementation**. Present generation time first,
quality second, while retaining identical request/seed/precision and timing
boundaries. Do not use old-offload B times as an architectural comparison.

### Small real-CUDA asynchronous ring gate

`validate_gpu_ring.py` is a separate correctness harness, not a deployment or
benchmark launcher. After fresh resource admission, the root agent can run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 /ABS/KIT/validate_gpu_ring.py \
  --output /ABS/NEW_RING_RESULT --rounds 20
```

It constructs tiny BF16 Linear layers with known deterministic weights,
uses actual HookRegistry forward calls, native H2D/AllGather streams and
events, the same initial ring order, and the same two full output + two
half input buffer allocators. Cases have 3, 4 and 50 layers over 20 rounds;
each output must be bitwise identical to a same-device unoffloaded Linear
reference. There is no synchronization/readback inside the layer loop,
and no mock of collectives/prefetch. Only final per-case verification reads
back the device boolean results. The PyTorch allocator is capped at 1 GiB
per rank; CUDA contexts/NCCL allocations outside it are not covered.

Local status: syntax compilation only, no torch/CUDA execution. Successful
execution would validate small-model asynchronous ring ownership/forward
behavior, **not full H3 E2E memory, compiler/cache interaction, attention
semantics or speed**. Each rank writes a separate new JSON result; existing
results are not overwritten. The harness never cleans up external tasks.

Additional options review: actual `configure_openvdn_from_env` returns a
copy of OmniDiffusionConfig with replaced `tf_model_config`, despite the
call site's `transformer_config` variable name; the meta factory's signature
is therefore correct. Ordinary OffloadConfig derives `dp_size` from SP even
without the generic DLO flag. Explicit `dlo_use_allgather=False` is now
rejected rather than silently overridden. Native enable/disable, hook
registration, collective argument and shared-buffer signatures were checked
against the local audited source. Exceptions after bundle handoff still
require whole-worker restart; asynchronous peer failure needs the launcher's
process-group/job timeout, not an attempted in-process fallback.
