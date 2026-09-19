# CPU head-shard regression

`cpu_head_shard_regression.py` is a numerical test, not a model benchmark. It
constructs only tiny CPU tensors (8 heads, head dimension 4/8, hidden size 24,
7/17 target frames, spatial grid 2 x 2). It neither loads H3 weights nor creates
NPU tensors. Importing the upstream file can import its optional `torch_npu`
dependency, but the tests never call NPU device APIs.

Run with an existing CPU-capable PyTorch environment:

```sh
python tests/cpu_head_shard_regression.py --mode all --output /path/to/new-results.json
```

The output path must be new. `--mode single` omits distributed validation;
`--mode gloo` runs real four-process Gloo collectives. `--dtype float32` or
`--dtype bfloat16` may select a single precision; the default runs both.

## Assertions

- Original single-sequence sparse Softmax agrees with an independent Boolean
  mask oracle. Non-target prefix/suffix rows (including source and text) remain
  globally visible, and first/last target frames are anchors.
- Four candidate head shards concatenate to the original Softmax and original
  linear branch results using identical deterministic parameters.
- The linear result is exactly zero for source, text, anchors, and padding;
  sparse Softmax is exactly zero for padding.
- Local frame sums/counts reconstruct the original FP32 frame means, including
  shards that cut through frames or contain no interior target rows.
- Four actual Gloo ranks exchange local-sequence/all-head QKV and beta logits,
  all-reduce frame sufficient statistics, calculate each rank's assigned heads,
  perform the inverse exchange, and apply local gates. Results match the full
  original sequence on each rank.
- Cases cover text lengths 0/1/3/5, text before or after target video, padding,
  nonzero source/global prefixes, and 17-frame windows with nontrivial far-field
  linear readout. Seeds vary by case.

FP32 tolerance is `atol=2e-6, rtol=2e-5`. BF16 tolerance is
`atol=1/512, rtol=1/128`, approximately one unit in the last place away from
zero plus a small absolute floor. Exact routing, counts, and zero regions use
zero tolerance. Reports include maximum absolute and RMS errors, not just pass
flags. Small finite parameter initialization avoids uninitialized decay values
or artificial Cholesky conditioning problems; the actual VDN-solve update is
unchanged.

## Not covered

The Gloo exchange adapter is in this test, not the vLLM/HCCL wrapper. Passing
does not prove that the full transformer calls that wrapper correctly. Tests
do not validate checkpoint/LoRA loading, actual model parameters, distributed
output projections, NPU kernels/HCCL, large-shape memory, throughput, video
quality, or numerical drift across 50 layers and many denoising steps.

## Recorded run, 2026-09-13

The CPU test actually passed in the existing remote PyTorch 2.10.0+cpu environment.
Source snapshots were placed under the personal experiment's
`cpu_regression_20260913_1828` directory; SHA256 hashes are embedded in reports.
No NPU tensor or video service was created by these tests.

- `results.json`: five layouts in FP32 and BF16. Maximum error across all
  assertions was 1.9222497940063477e-6 for FP32 and 0.0001220703125 for BF16.
  The latter includes the independent Boolean-mask versus segmented-SDPA oracle.
- `results.gloo.json`: four actual Gloo ranks, five layouts, both precisions.
  Maximum FP32 error was 1.9222497940063477e-6; BF16 was elementwise identical to
  the upstream golden (maximum absolute error zero).
- `metadata_checkpoint_results.json`: 30 additional assertions passed for
  dynamic layout inference/rejections, 208 LoRA target mappings, 800 branch-name
  mappings, official scale-one small-matrix merges, and invalid safetensors
  headers. Synthetic BF16 LoRA merges used zero tolerance.

Run the additional tests with:

```sh
python tests/cpu_metadata_checkpoint_regression.py --output /path/to/new-metadata-results.json
```

These results are a CPU correctness gate only, not evidence that B is running,
that real Stage-B weights have loaded successfully, or that B accelerates V2V.
