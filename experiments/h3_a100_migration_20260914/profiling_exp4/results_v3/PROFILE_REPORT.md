# Experiment 4 DiT block profiling

## Setup

- Server: 30674, 2 × NVIDIA A100-SXM4-80GB, Ulysses degree 2.
- Model path: Ref2VA + OpenVDN B attention + SpotEdit-style partial-query/MLP skip.
- Selector: VAE-perceptual fixed selector; active target ratio 32.3145%.
- Each DiT forward contains 50 blocks.
- Samples: refresh steps 1–3 and partial-skip steps 4–10 on both ranks.
- Refresh step 0 was excluded because it contained TorchDynamo cold-start/recompilation.
- Partial-skip samples captured all 50/50 events with zero event errors. Missing refresh component events were normalized from their observed per-block means to 50 blocks; block totals and attention/communication measurements were complete.

## Main result: steady partial-skip critical rank

Rank 0 is the critical compute rank: 27,976.19 ms per 50-block DiT forward.

- QKV projection: 1,829.07 ms, 6.54%.
- Norm and AdaLN: 778.25 ms, 2.78%.
- RoPE: 678.21 ms, 2.42%.
- Ulysses communication/reductions: 1,307.03 ms, 4.67%.
- Combined retained path (QKV + Norm/AdaLN + RoPE + Ulysses): 4,592.56 ms, 16.42%.
- Attention core (Softmax + linear branch): 17,668.53 ms, 63.16%.
- Attention output projections: 1,345.49 ms, 4.81%.
- MLP: 3,867.43 ms, 13.82%.
- Combined heavy path (Attention core + out projections + MLP): 22,881.45 ms, 81.79%.
- Residual gates, selector/cache scatter, and unattributed remainder: 502.18 ms, 1.80%.

## Rank 1 and synchronization waiting

Rank 1 finishes at a similar 27,909.06 ms, but its distribution is different:

- Combined retained path: 28.50%.
- Combined heavy path: 69.69%.
- Ulysses communication/reductions alone: 16.31%.

This does not mean the network intrinsically costs 16.31%. Rank 1 spends 12.45% of block time in `ulysses_softmax_pre_attention`, versus 1.58% on rank 0. At the same time, rank 0 spends 13.82% in MLP versus 5.17% on rank 1, and 4.81% in output projections versus 2.00% on rank 1. The rank-1 communication timer therefore includes synchronization waiting for the more heavily loaded rank 0.

## Refresh versus partial skip

- Stable refresh critical path: 32,852.90 ms per 50 blocks.
- Partial-skip critical path: 27,976.19 ms per 50 blocks.
- Block-stack time reduction: 14.84%.
- Block-stack speedup: 1.174×.

On the critical rank, almost all savings come from the attention core:

- Attention core: 22,916.52 → 17,668.53 ms, a 22.90% reduction.
- QKV: 1,836.97 → 1,829.07 ms, effectively unchanged.
- Norm/AdaLN: 793.71 → 778.25 ms, effectively unchanged.
- RoPE: 678.60 → 678.21 ms, unchanged.
- Output projections: 1,133.92 → 1,345.49 ms, slower because active-index gather/scatter and shard imbalance offset the reduced rows.
- MLP: 3,776.32 → 3,867.43 ms on rank 0, while rank 1 falls from 3,840.35 to 1,441.61 ms. The MLP skip is real on rank 1 but does not shorten the critical path because rank 0 remains the bottleneck.

## Conclusion

The implementation saves real attention-query and MLP work, but stable tokens do not fully bypass the DiT. On the critical rank, QKV/Norm/RoPE/Ulysses still account for 16.42%, while Attention core/out-proj/MLP account for 81.79%. The immediate engineering bottleneck is not only the retained QKV path: the larger issue is that active rows are unevenly distributed across Ulysses ranks, converting rank-1 compute savings into communication wait.

Recommended next order:

1. Log and rebalance active-row counts per Ulysses shard; otherwise MLP/out-projection skipping cannot translate into end-to-end speedup.
2. After balancing, recompute the same profile.
3. Then evaluate active-only Q projection plus per-layer cached stable K/V. On the current critical rank, QKV + Norm/AdaLN + RoPE is 11.74% of block time, so this is meaningful but not the largest remaining term.
4. Continue reducing partial attention core work, which remains 63.16% of the block stack.
