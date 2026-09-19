# S reuse / S-skip: two policies, one cache backend

Status: proposed; NOT installed, NOT benchmarked. User confirmation requested
for the new approximation and S refresh schedule. Existing T-skip is unchanged.

## Frozen baseline

SAViE step1000 + DMD8 (8 true forwards), seed4101, SP2 odd/even, current
KV-reuse loader. Same weights, sample, precision, offload, connectivity,
T selector and T refresh forwards 1/5. Do not mix the pending consolidated
KV-pipeline implementation into these experiments until separately validated.

## Policies to compare

- R0: no S reuse; latest T-skip only.
- R1 / whole-S query reuse: S refreshed on proposed forwards 1/2/5;
  on other forwards replace every S block output with its cached value and
  genuinely omit its attention query work, output projection and MLP.
- R2 / selective S-skip: same S refresh forwards, but only low-drift S rows
  reuse cached outputs; active S rows run normally. The proposed decision
  signal is cross-step S hidden-state drift, NOT target x0-versus-source and
  NOT a forced stable percentage. Threshold policy remains to be calibrated.

Both are approximate, even with ST removed: source attention reads text
representations whose AdaLN timestep can change. Fixed source input does not
prove fixed per-layer S states. Skipped outputs must remain available as
inputs to later layers; removing S keys is NOT allowed.

First implementation retains current-step Norm1/AdaLN, QKV and Ulysses K/V
transport. This is NOT full source-branch elimination. Full S-KV reuse that
also skips those operations is a separate implementation stage with extra
memory/invalidation requirements; do not claim its savings for R1/R2.

## Shared implementation prerequisites

1. Source Flash currently asserts all S/text queries active. Support partial
   S queries without removing legal S keys or changing joint softmax.
2. Keep text/global query computation active; keep T masks/caches separate.
3. Cache by request, checkpoint, layer, logical S token, dtype/layout/rank.
   New requests, changed masks, checkpoint/layout changes must invalidate.
4. Separate S and T refresh flags. They need not refresh on the same forward.
5. Use real operation-count/timing checks: computing S then overwriting with a
   cached tensor is a correctness harness, never a speed implementation.
6. Verify request reset, anchor queries, padding, shard balance and unequal
   active counts. Test R0 == disabled backend numerically before R1/R2.

## Resources and measurements

For this sample: 37*1008 source tokens, hidden5376, 50 layers, BF16, SP2.
Full per-layer S output cache is ~9.34 GiB/rank (before allocator overhead).
Caching source K AND V for all 50 layers would additionally cost ~24.90
GiB/rank at 56 heads * 128 channels, SP2. These are sizing calculations,
not measured peaks or a mandate to store both representations.

Report separately: compile/warmup, selector calibration, S calibration,
cache construction, steady 8-forward request, first-use total, peak memory,
actual skipped S rows by layer/step, SS/core/MLP/packing/communication time.
Keep current request timing and a cold/new-sample timing; do not amortize an
extra S calibration forward out of the end-to-end claim.

## Order and combination rule

Measure S drift and cache-memory feasibility first, then common-backend unit
and numerical tests, real-size microbenchmark, then R0/R1/R2 full runs with
no concurrent GPU work. Retain all raw times; no E2E speed claim from FLOPs.
Output difference from R0 is an approximation diagnostic, not a video quality
certificate. If claiming useful speed, also inspect editing/temporal failures.

R1 already skips all S query rows on reuse steps. Applying R2 on those same
rows saves nothing extra. Only a HYBRID schedule could combine them: whole-S
reuse on some forwards, selective reuse on others. This is not orthogonal
or additive; run a hybrid only if measurements justify a concrete schedule
and that changed schedule is approved. Do not automatically launch a redundant
R1+R2 experiment merely because both beat R0.
