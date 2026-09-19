# Targeted T/S text_state timing — one authorized color task

STOPPED BY USER PLATFORM CHANGE: acceleration work now targets free NVIDIA
A100 resources, not Ascend. Run f66b003fd1c64143b2ccba249285c869 was stopped
during server loading; the formal generation request did not start. Supervisor
cleanup completed and the selected Ascend cards were verified idle. Do not
restart this Ascend launcher for acceleration measurements.

Latest user request explicitly authorizes one color inference to measure the
two independent text_state calculations. This is NOT the stopped broad B/D
operator profiling workflow, nor a cache optimization/training run.

Remote: `/cache/zhonghao/h3/text_state_timing_v1`, SSH31731, HTTP19113.
Run: `f66b003fd1c64143b2ccba249285c869`; supervisor1223177.
Five CPU tests and the actual-source AST checks passed. The real NPU event
timer smoke also passed; its small matrix timings are NOT text_state results.
Physical cards0–3 only; cards4–7 remain reserved for others even when idle.
USP4/textTP4/VAE tile4, original stage-B weights and D attention math, same
shirt-red source/prompt/seed4101/resolution and original50-point schedule.

Instrumentation wraps exactly the eight original statements from text_slice
through text_state in BidirectionalLinearBranch.forward_head_shard. The two
call sites are labelled T and S with the actual DiT layer index. Removing the
observer scopes/imports reproduces the original AST. The source-linear scan,
target scan, other statistics, attention rules and all parameters are unchanged.
Only the private diagnostic logging guard additionally admits world4.

Every forward records 50 T and50 S calls per rank. Over49 forwards, expected
2450 calls per side per rank. CPU enqueue intervals and NPU event intervals are
separate. Events are read once per forward after synchronization, not after
each individual block. The original function is called exactly once, and
text_state outputs are neither cached nor recomputed by the observer.

Output: per-rank `profile/text_state.rankN.jsonl`, complete HTTP request elapsed,
and `text_state_summary.json` with both T+S total and the removable second-copy
candidate S interval. Never sum the parallel ranks' elapsed times. Event
intervals include CPU dispatch gaps/device-stream waits and do not measure only
kernel busy time; synchronization/readback adds observer overhead.

Interpretation: compare T+S with this instrumented request's total; assess
deduplication opportunity using ONE copy, not both. Four-card measurements
cannot be equated directly to the earlier eight-card C/D84.01-second gap, and
measured intervals are not guaranteed end-to-end savings.

The supervisor executes exactly one request after a small real NPU event-timer
test. Failures clean up only owned processes; no B task, no auto retries, no
training and no automatic consumption of cards4–7.
