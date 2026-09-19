# 30213 completed-experiment archive → 31731

Prepared locally and, with explicit follow-up authorization, deployed into new
private v2 tool directories on both nodes. No producer, receiver, real packing,
NPU calls, interruptions or new video requests were performed by this task.

## Last read-only observations

- 2026-09-13 21:04:04 CST: current B is `b_50step_request_running`, formal
  result absent, `error=null`; log progress 35/49 at about 37.51 seconds/iteration,
  displayed remaining 8:45. This is progress, **not** HTTP success or measured
  full-request latency. The 1800-second request timeout from 20:41:21 is near
  21:11:21, earlier than that progress estimate. Do not change the active job
  or repeat its request; archive whichever final outcome is actually recorded.
- Mandatory run: `b_smoke_formal_20260913T122957_063593Z`.
- A results approximately 33.4 MB; B approximately 37.4 MB while active (formal
  output not yet complete); tiny validation approximately 65.6 KB.
- Actual code-scope read-only scan/hash passed: 2,807 files, 54,900,978 bytes,
  no configured credential/weight/path-policy blocker. This is not a sealed
  archive; source-set revalidation still runs at production time.
- Old `b_20260913T105448_452965Z` remains at
  `cleaning_up_own_processes`, without finished/cleanup-complete records. It is
  explicitly listed under `excluded_unsealed_runs`, not silently called complete.
  Source is retained. Separate orphan-evidence sealing would need its own
  explicit verification; this script does not fabricate that missing record.

## Gate and scope

`archive_30213_experiments.py` uses system Python 3.10+ only. Producer is bound
to exact 30213 host; receiver to exact 31731 host. It takes existing A, B and
tiny supervisor `run.lock` files in shared/nonblocking mode; an active job's
exclusive lock blocks sealing. It checks terminal phase, finished timestamp,
cleanup-complete/empty owned groups, and that each recorded supervisor/server/
worker PID with matching start ticks has exited. Mandatory latest B must pass.
Other unsealed runs are listed and excluded.

Included: six root experiment source/config files, complete bounded
`b_adaptation` code/vendor/upstream/tests (excluding git and Python caches), the
existing CPU-regression code, and individually completed A/B/tiny run trees.
Run trees preserve requests, results, videos, runtime/weight-identity manifests,
logs and cleanup evidence. A failed-but-cleaned terminal run remains labelled
failed. This copies evidence; it does not rerun an experiment or normalize its
paths into a new result.

Excluded: mutable live status aliases, locks, active launch logs, deployment
transfer state, scratch directories, environment trees, model/checkpoint payloads,
SSH keys/configs and credential-like content. This archive does not migrate
external source-video inputs; the separate 31731 input preparation owns that.

Per-file bounded streaming SHA256 and stable dev/inode/size/mtime/ctime are
checked during reading/copying; then the entire source inventory and hashes are
checked again before publishing a completed object record. A change leaves an
unpublished failed object and failure record, with source unchanged. It never
resumes, deletes or overwrites an object.

Shared staging is `/temp/zhonghao/h3_30213_archive_20260913_v1`, using unique
UUID objects and one complete JSON record. Consumers never trust a tar without
a fully parsed matching complete record and correct whole-archive hash. This
avoids rename/overwrite requirements on shared NFS. A partial record fails
closed and is not reused. Receiver preflights all member paths/types/duplicates,
allows regular files only, and rejects parent-file collisions, symlinks,
hardlinks and special devices. Linux source/destination walks use `openat` with
`O_NOFOLLOW`. Each extracted file hash must match. A new partial directory is
published locally using Linux `renameat2(RENAME_NOREPLACE)`; no unsafe fallback.

## Future execution by the main agent, after the gate

1. On 30213: `python3 archive_30213_experiments.py inspect` (no file writes;
   holding read-only advisory locks may temporarily prevent new supervised jobs).
2. After inspection passes: `python3 archive_30213_experiments.py produce`.
   Record the returned `archive_id` and exact file/hash/exclusion report.
3. On 31731: `python3 archive_30213_experiments.py receive --archive-id ID`.
   Receives from shared staging into a **new**
   `/cache/zhonghao/h3_30213_archives/30213-ID` directory. Existing or failed
   destinations are never overwritten/resumed/deleted automatically.

These commands are handoff instructions, not evidence that transfer occurred.
Production remains blocked until main-agent review/authorization.
Local toy tests cover terminal/cleanup/PID gates, path and secret screening,
hash mutation failures, round-trip/source preservation, no-overwrite publishing,
host binding, archive-member attacks, and excluding orphan runs while requiring
the latest B to finish. On 31731 at 21:11:15 CST, all **20 Linux CPU toy tests
passed** with no skips, using `/usr/bin/python3 -I -S`, single-thread environment
limits and empty device-visibility variables. This exercised real Linux
nofollow/flock/renameat2 behavior (including existing empty destination refusal),
without importing torch or any NPU modules. The first v1 test attempt exposed
Python 3.10's missing `hashlib.file_digest`; v2 uses bounded chunk hashing. The
v1 files/logs were preserved instead of overwritten.

Deployed v2 roots:

- 30213: `/home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/deploy_31731/archive_review_20260913_2111_v2`
- 31731: `/home/ma-user/workspace/zhonghao/h3_deploy_31731/archive_review_20260913_2111_v2`

Each contains `deployment_manifest.json`; 31731 also contains `cpu_tests.log`.
Script SHA256: `965d52127aea24023cc62f292eb1509bbeeec8d84d31cb852f7ab8afc41b0257`.
Test SHA256: `4247724da7d8af4c081009facc06700eefe39c648620176c12bcbe6be1d26fad`.
