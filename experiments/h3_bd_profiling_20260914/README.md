# B/D operator profiling — diagnostic development run

Requested: identify local Softmax, linear branch, packing/copy and communication
costs after the S–S cache audit. No training or cache/attention optimization.

Remote root: `/cache/zhonghao/h3/profiling_bd_v1`, port 31731, cards 0–7.
Latest run: `e6271b1a888b4768bd569d986ef11a35`. Supervisor PID 1211742.
User allocation change: physical cards 4–7 are temporarily reserved for someone
else. A verified SIGTERM was sent to this eight-card supervisor during B model
loading, to stop the entire distributed job and cancel the queued D run. Do not
restart this eight-card workflow until the user restores the allocation. A
four-card adaptation has NOT been authorized or launched in this turn.
Initial run `fd7d6b3dc5cb4506826c740a21af0783` passed the profiler export
smoke, then failed before model inference because its ZMQ temporary pathname
exceeded the Unix socket length limit. Its evidence is preserved, all owned
processes were cleaned and eight cards verified idle. The launcher now uses a
short run-specific temporary directory (96 bytes including ZMQ's UUID suffix).
B then D use these eight cards sequentially, under the existing private device
locks and verified idle checks. Cleanup only targets this run's process groups.

Both use the frozen shirt-red edit, seed 4101, 1344×768, original 50-point
schedule (49 forward calls), USP8 / textTP8 / VAE tile8 / layerwise offload.
No smoke-generation request precedes the measured request. A small NPU profiler
export smoke check precedes B model loading. There is one generation per case.

- Forward 3: rank0 CPU+NPU Level1 operator trace, with nested semantic markers.
- Forward 4: nested current-stream NPU-event intervals on all eight ranks.
- Every forward: CPU wall record; not an independent device-busy metric.
- Original vendor contents are copied to new B/D directories; the sole model
  source edit appends observer installation. Original AST is checked unchanged.
- Hooks return original outputs and call the original function once. They add
  ranges/events and selected step-boundary synchronization, not new mathematics.

Interpretation constraints:

1. Profiling changes overhead. These request timings are not new acceleration
   benchmarks and must not be compared directly with the old A/B/C/D timings.
2. Nested scopes overlap: never add parent and child inclusive times. The
   summarizer provides a separate non-overlapping accounting with signed
   numerical residuals.
3. NPU-event intervals include stream waiting and CPU starvation. Operator trace
   kernel/HCCL/copy records must be inspected before attributing them to compute.
4. Rank0 kernel trace is one rank/one early denoising step, not all49-step total
   or proof of the slowest rank. Event sampling covers all ranks at forward4.
5. D groups multiple query streams into some Softmax launches. A mixed kernel
   cannot be assigned an exact S-only/T-only latency without an additional
   experiment; report mixed groups honestly.
6. Existing duplicate text-state calculations are intentionally left in place
   to profile the actual D implementation.

Reference: official Ascend profiler interfaces were checked against installed
torch_npu 2.10.0.post2. Level1 collects communication diagnostics as described in
https://www.hiascend.com/document/detail/zh/mindstudio/700/T%26ITools/Profiling/atlasprofiling_16_0033.html

The remote supervisor writes `status.json`, logs, trace files and per-rank event
JSON. The local README is a launch record, not a live progress indicator.

Profiler smoke emitted a `RECORD`-state stop warning. The exported kernel and
operator statistics were checked: all three intended MatMulV2 calls and the
custom marker were present. Full-model trace completeness must separately be
checked against the expected 50-layer call counts, not assumed from the smoke.
