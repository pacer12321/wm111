# SVOO × ISA orthogonal acceleration plan

## Frozen baseline

- Base: current fastest Ref2VA + VDN + interleaved sequence parallelism + latent
  target-token skip.
- Baseline runtime: 1410.02002 s for the formal 50-step red-shirt edit.
- Frozen routes during the first study: T-T, T-S and the target-token selector.
- First modified route: `QsKt` (source queries, target keys/values).
- Second, independent route: `QsKs` (source queries, source keys/values).

## Orthogonal roles

### SVOO

- Offline: estimate a conservative exact-attention density per layer and head from
  several calibration videos.
- Online: optionally replace fixed contiguous blocks with Q-K-coupled blocks.
- SVOO answers **how much exact work each layer/head receives and how blocks are
  organized**.

### ISA

- Route high-sharpness query blocks to exact attention.
- For flat query blocks, compute the most important blocks exactly.
- Do not discard the remaining blocks: approximate them with zeroth-order Taylor
  mean-K/V terms and preserve block multiplicity in online softmax.
- ISA answers **which queries require exact work and how omitted exact interactions
  are approximated instead of deleted**.

The first implementation must not stack SVOO selection and ISA pre-selection as
two independent Top-K stages. SVOO supplies the exact-block budget/partition;
ISA supplies query routing and the Taylor fallback.

## Experiment funnel

### Current feasibility evidence (2026-09-17)

- The original per-query/head screening oracle was optimistic because it routed
  only 592 sampled source queries independently.
- A deployability screen now shares one route across contiguous entries of the
  sampled-query tensor. At 50% exact key blocks and 50% dense query blocks:
  - `QsKt`, query block 64: global output errors are 3.31% / 5.16% / 4.85%
    at layers 8 / 24 / 41.
  - `QsKs`, query block 64: 5.03% / 9.32% / 12.42%.
- Important limitation: those 592 queries are ordered as a 4x4 spatial sample per
  frame, so a 16-query block spans an entire frame rather than a true contiguous
  spatial run. `QsKs` receives one corrected contiguous/full-query validation
  before final rejection; the current block result is not treated as conclusive.
- The official LIVEditor Triton kernel runs under the 30674 A100 environment at
  H3 post-Ulysses dimensions (28 heads/rank, D=128, Q=37,312, KV=81,216).
  With half the query blocks dense, measured source-query-row latency reductions
  versus full dense attention are:
  - 25% exact blocks: 26.50%;
  - 37.5% exact blocks: 15.85%;
  - 50% exact blocks: 3.75%;
  - 75% exact blocks: -17.56% (slower).
- This is a kernel feasibility upper bound, not an H3 integration result: it does
  not yet enforce branch-mandatory exact blocks or include Ulysses/model overhead.
  It shows that a uniform conservative budget is unlikely to pay off. The only
  remaining plausible path is an SVOO per-layer/head budget with many tolerant
  layers at 25-37.5% and sensitive layers left dense.

### O0 — Numerical correctness

- 100% exact blocks must reproduce dense attention.
- Passed on a synthetic padded-block test: max absolute error `1.788e-7`, relative
  error `1.136e-7`.

### O1 — Single-input ISA screening oracle

- Capture: red-shirt sample, step 4, layers 8/24/41, 56 heads.
- Branches measured independently: `QsKt` and `QsKs`.
- Block size: 64.
- Exact block ratios: 6.25%, 12.5%, 25%.
- Dense-query fallback fractions: 0%, 50%, selected by coarse-score sharpness.
- All non-tested attention quadrants remain exact.
- Report full-attention-output relative L2, per-query/head error, cosine similarity,
  and ideal branch compute reduction.

Gate: reject configurations with large middle-layer error before any kernel work.
This is a screening oracle, not an end-to-end quality certificate.

### O2 — SVOO cross-input calibration

- Samples: red-shirt couple, desert landscape, dancing people, dog/pond temporal edit.
- Same fastest pipeline, resolution, sequence length and capture implementation.
- Capture early/middle/late denoising steps; start with layers 8/24/41.
- Compute separately for `QsKt` and `QsKs`, per layer and per head:
  - minimum density covering 95% attention mass;
  - mean, standard deviation and conservative upper quantile;
  - support overlap only as a diagnostic, not as an assumption.
- Leave-one-video-out validation: calibrate the density schedule on three videos and
  test attention-output error on the fourth.

Gate: if density is stable but support is not, keep offline budgets and use online
Q-K co-clustering. A static absolute mask is allowed only if held-out mask transfer
also works.

Before cross-input calibration, run an optimistic all-48-layer screen on the
red-shirt step-4 capture. Select the lowest tested exact ratio whose output error
passes the frozen threshold, otherwise mark that layer dense. Combine this layer
schedule with the measured official-kernel latency curve. If even this optimistic
per-query upper bound yields negligible end-to-end potential, stop before
full-query capture, four-video calibration, or model integration.

### O3 — SVOO increment over ISA

Using the best O1 Taylor/query-routing configuration:

1. uniform density + contiguous blocks;
2. SVOO per-layer/head density + contiguous blocks;
3. SVOO per-layer/head density + online Q-K co-clustering.

This ordering isolates the value and overhead of heterogeneous budgets from the
value and overhead of co-clustering.

### K1 — Kernel microbenchmark

- Fused exact-block and Taylor-remainder online softmax; no Python gather/varlen
  implementation is accepted as the final timing path.
- Separately time reductions/coarse scores, sharpness routing, co-clustering,
  block layout, Ulysses communication, exact attention, Taylor attention and output
  merge.
- Compare fixed contiguous blocks with online co-clustered blocks.
- Measure rank load balance under the existing interleaved SP layout.

Gate: total route latency including routing/layout must beat the frozen dense route.

### E1 — End-to-end single-route ablations

1. Frozen fastest baseline.
2. Baseline + `QsKt` hybrid only.
3. Baseline + `QsKs` hybrid only.
4. Baseline + both source-query hybrids.

No other route or selector changes are allowed.

### E2 — Quality and generalization

- Run the four calibration/generalization videos, including temporal and
  non-temporal edits.
- Record end-to-end wall time, attention time, preprocessing/routing time, global
  and edit-region metrics, temporal consistency and synchronized visual review.
- Calibration videos and held-out reporting videos must be identified explicitly.

### E3 — Final stack

Only after the source-query hybrid passes:

- combine it with the existing target-token skip;
- benchmark the combined system rather than adding isolated speedups;
- repeat the frozen baseline and final system under the same kernel/offload setup.

## Interpretation rules

- SVOO input stability means density stability, not necessarily identical block
  support across videos.
- ISA's compressed K/V terms approximate unselected blocks; they are not a license
  to replace every target token with one global summary.
- Offline output error can reject a design but cannot certify final video quality.
- Attention-module speedup and end-to-end speedup must be reported separately.
