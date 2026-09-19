# Isolated C strict-source eight-card tiny candidate

This directory is a **local, not-yet-NPU-validated candidate**. It changes no
existing A/B/runtime validation files. It does not schedule itself. A full-model
A/B service must finish and release its leases before C tiny can run.

## Scope and correctness contract

- One real HCCL/Ulysses group: 8 physical cards `0,1,2,3,4,5,6,7`, USP8,
  ring1, DiT TP1, all **56 heads**, **7 heads per rank**, head dimension128.
- Instantiate the actual C `MiniMaxH3Attention`, but use the already-reviewed B
  helper's tiny tuple-returning linear projections. Load **no model weights**.
- Compare actual `_run_openvdn_ulysses` and attention `forward` with an independent
  float32 CPU dense Boolean oracle, BF16 output, original fixed tolerances
  `atol=1/512`, `rtol=1/128`. Never enlarge tolerances on failure.
- The oracle explicitly constructs B's pairwise mask and deletes only
  target-query × wrong-frame **visual source** keys. It imports no C plan/cache.
- Compare the linear result with the separately loaded, pinned original B
  linear branch; require QKV and holder weights/buffers unchanged.
- Exercise text, reference audio, target audio and source queries as positive
  controls; only visual source is restricted. Use extreme wrong-source values
  to require exact-zero leakage into first/last target anchors. Target anchors
  retain all target keys. Padding output and noninterior linear output stay zero.
- Two legal source-prefix fixtures have unequal source/target patch areas,
  distinct noninteger time origins, H3's nonuniform temporal increments, source
  and target frames cut across SP ranks, and ranks with no target tokens.

**Do not claim source-suffix support.** Current production C supports only the
actual single-video Ref2VA ordering:
`text | reference audio | visual source | target audio | target | padding`.
The old B tiny source-suffix success case cannot be transplanted as a supported
C case. Here source suffix is one of seven deliberate **fail-closed** cases,
along with multiref, Fs mismatch, prefix-as-source, wrong mode, nonfinite source
coordinates and source frame reordering. Every worker checks CPU metadata before
`set_device`/collectives. This is not recovery for divergent rank inputs.

## Fixed deployment (root must review and deploy; no execution performed here)

- New code: `/cache/zhonghao/h3/c_validation_code`
- C vendor: `/cache/zhonghao/h3/candidates/c_v1/vllm-omni`
- Output: `/cache/zhonghao/h3/c_validation/01234567/runs/<timestamp>_<runid>`
- Shared personal per-card locks: `/cache/zhonghao/h3/validation/card_locks`
- Reused read-only helpers: `/cache/zhonghao/h3/validation_code` (three exact SHA
  pins in `c_profiles.py`; their worker/main are never invoked).
- Fixed rendezvous29675, 180-second collective timeout, 900-second outer timeout.

The supervisor binds the exact31731 host and boot ID, canonical private paths,
six C source hashes, three helper hashes, original golden hash, runtime env and
strategy source manifests. It holds one C-run mutex and the same eight device
mutexes used by A/B. Under the leases it freshly requires all cards idle/healthy,
at least8GiB free HBM per card and64GiB effective host/container memory. Locks do
not authorize displacing another user's processes. No existing job is stopped.

The worker entry requires the exact supervisor parent/start identity, inherited
nine mutex FDs matching inode/path, real held locks, a host/source proof and
nonexistent output files. One run is supervised; cleanup uses the pinned existing
process-owner implementation, terminates only its owned process group, verifies
card release and leaves failure evidence. It never edits A/B latest proof files.

After review/deployment and only when authorized and idle, the only NPU entry is:

```sh
/cache/zhonghao/h3/env/bin/python /cache/zhonghao/h3/c_validation_code/run_c_validation.py --allow-npu
```

No NPU launch command is executed by creating these files or running the tests.
No source30213 access is involved.

## CPU tests

From this directory (or use unittest discovery):

```sh
OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -B -m unittest -v test_c_validation_cpu
```

The tests need the original sibling `strict_source_layout.py`, six `candidate/`
files, and the pinned sibling B validation helper files to check the exact local
sources. For isolated remote CPU testing preserve that relative tree, or use a
separate review workspace; do not overwrite active deployment files. They never
select NPU. On Windows, actual POSIX flock and actual torch tensor tests are
explicit SKIPs, not successes; mock FD tests check only gate logic. With CPU
PyTorch available, the real C adapter is tested against the independent numeric
oracle over all56 heads and emulated8 head shards. **That is not an HCCL test.**

Local Windows result on2026-09-13: **35 tests,32 passed,3 explicitly skipped**
(1 real POSIX lock test,2 real torch CPU tests),4.009seconds, exit0. No remote
runtime or NPU execution was performed for this new candidate.

## Remaining validation

Actual31731 imports, new entry/lease-FD behavior, fused NPU BF16 numerics,
all-eight-rank communication, final projection parity and actual cleanup must
still pass a real authorized run. No full-model checkpoint loading, pipeline
service, video quality, memory scalability, speedup or training result follows
from this tiny test. Existing separate C five-test CPU success does not establish
any of those NPU/full-model claims.
