# 31731: fixed eight-card tiny validation

This directory is a self-contained **tiny correctness/interface test**, not a
full-model A/B/C entry point and not an acceleration or video-quality result.
No checkpoint weights are loaded. It retains the existing tested BF16 numerical
fixtures and exercises the real candidate attention forward plus Ulysses/HCCL
against the SHA-pinned original unsharded OpenVDN implementation.

Deploy the contents of this directory (including `golden/`) to exactly
`/cache/zhonghao/h3/validation_code`. Deploy the reviewed candidate vendor to
`/cache/zhonghao/h3/candidates/b_v1/vllm-omni` and the private runtime setup to
`/cache/zhonghao/h3/env_h3_31731.sh`. The four candidate source hashes are pinned
in `profiles.py`; an incomplete/different copy fails before NPU initialization.
The golden file is an exact independent copy of the existing local
`b_adaptation/upstream/openvdn_npu.py`, SHA256
`bb06d4183e79dc5837d19a68ebaf1abdb520098dc50b05b6e22d34e9259cb43c`.

After deployment and explicit scheduling approval, run the **only** allowed group:

```bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/validation_code/run_validation.py --group 01234567 --allow-npu
```

All eight cards form one USP8 group (ring1, DiT TP1, 56/8=7 heads per rank).
Legacy groups 0123 and 4567 are rejected. The rendezvous port is 29673;
run-specific outputs are under `/cache/zhonghao/h3/validation/<group>/runs/`.
All eight card locks are under the same private
`/cache/zhonghao/h3/validation/card_locks` directory. The group has its own
`run.lock`, latest status, temporary/cache directories and ATB shared-memory
suffix. Launches inherit all nine lease descriptors in a new process session.
Do not invoke `launch_validation.sh` or the wrapper directly.

The supervisor reads and binds the real hostname, architecture and kernel boot
ID; neither host identity nor physical group can be supplied as an arbitrary
override. Before each launch it verifies all selected cards explicitly idle,
health OK and at least 8 GiB free HBM each, plus at least 64 GiB host/container
memory headroom and a free group rendezvous port. Personal locks coordinate our
own jobs only: they do not reserve cards against noncooperating users or grant
permission to stop anyone else's work. A tiny pass does not authorize
simultaneous full-model loading or bypass the full-model resource gates.

The wrapper gets ordered physical cards through `mp.spawn` arguments and reads
the real host identity again in every rank. Reports record pinned candidate and
golden hashes plus actual vendor strategy source paths/hashes. The supervisor
requires both source-prefix/source-suffix layouts, all five checks per rank,
unchanged BF16/exact tolerances and all eight matching standalone rank proofs.
Every rank also records actual USP/ring/TP sizes and an NPU-tensor all-gather of
global ranks 0–7 in that same Ulysses communicator. A four-rank/split-group or
old pre-migration proof is rejected; merely starting eight processes is not a pass.
Only `phase=completed`, `validation_passed=true`, `cleanup_completed=true` and
`selected_cards_verified_idle_after_cleanup=true` are a usable completed pass.
SIGTERM/timeout cleanup signals only this run's tracked identity/session/run
marker, then checks selected-card release. No foreign process is stopped.

CPU-only local test (no torch or NPU required):

```bash
python -B test_validation_cpu.py
bash -n launch_validation.sh
```

These tests mock Linux-only locks/process operations on Windows. Production
flock/HCCL and the migrated runtime still require the authorized on-host tiny
run. Never relabel a CPU test as an NPU pass or a tiny pass as a full B result.
The two provenance/fixture-equivalence CPU tests also read the original local
repository files; run this test suite in the local repository before deployment.
