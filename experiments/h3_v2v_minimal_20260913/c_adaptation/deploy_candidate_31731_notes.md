# Independent C vendor deployment candidate

This deployment script and its toy tests have been created **locally only**.
No remote inspect, source copy, C vendor deployment, environment mutation or NPU
execution is authorized by creating these files. Root chooses a suitable time;
do not add this copy to the currently active B loading/formal measurement.

## Fixed paths and smallest isolated copy

Only the small B vendor **source repository** is copied:

- Read: `/cache/zhonghao/h3/candidates/b_v1/vllm-omni`
- Read six pinned overlays: `/cache/zhonghao/h3/c_adaptation/candidate`
- New private reservation: `/cache/zhonghao/h3/candidates/c_v1`
- Staging: `c_v1/vllm-omni.incomplete`
- Verified vendor: `c_v1/vllm-omni`
- New state/evidence: `c_v1/deploy_manifest.json`

No environment, installed runtime, weight directory, service entry, tiny/full
supervisor or A/B proof is copied or changed. No links to B are created: sharing
B through symlinks/hardlinks would weaken isolation, and this script rejects that
shortcut. The whole bounded source tree is necessary for C package-relative and
absolute imports while retaining the rest of the unchanged B vendor context.

## Gates

- Exact31731 hostname, aarch64 and reviewed boot
  `8e904236-bb23-44b5-b7ba-af73ba5c1f77`; host/boot checked again before and after
  publication. A changed boot requires explicit review, not an automatic rebind.
- Canonical disjoint private paths; both B and C overlay directories must already
  exist. **The whole `c_v1` root must not exist.** This is intentionally stricter
  than merely checking `vllm-omni`, so a failed or unknown prior reservation is
  never resumed or overwritten.
- Read-only inventory rejects symlinks (including ancestor links and entries
  named like ignored caches), FIFOs/devices/sockets and other nonregular entries.
  `.git`, `__pycache__` and regular `.pyc` files are excluded. Contents of ignored
  directories are not scanned or copied and are not claimed stable.
- Per-file cap20MiB, included B tree cap128MiB, and resulting B+overlay tree
  cap128MiB. All four reviewed B file hashes and all six C overlay hashes must
  match before creating `c_v1`. Two shared B/C files have identical pins.
- Every copy uses a new file, bounded streaming SHA, size and source inode/time
  checks. Executability is preserved, but setuid/setgid/sticky bits are stripped.
  Overlay replaces only our just-created, reverified staged B file.
- The entire destination is verified for exact paths, directories, size and SHA;
  extra files are rejected, including new runtime caches. Destination regular
  files must each have one link. Included B inventory and six C source records
  must remain identical before/after copying.
- Linux publication uses `renameat2(RENAME_NOREPLACE)`, not check-then-overwriting
  POSIX rename. Even an empty destination appearing concurrently is not replaced.
  Missing kernel/libc support fails closed. No fallback weakens no-overwrite.
- SIGINT/SIGTERM or caught errors retain failed staging and a failure manifest;
  no file deletion, automatic cleanup or retry. Abrupt SIGKILL/power loss can leave
  `staging`, which is also **not usable**. Only `state=completed` plus verification
  is an accepted deployment. A late post-publication error leaves `state=failed`
  and `published=true`; do not run that vendor.

The manifest records host/boot, exact source/destination file inventories and
hashes, before/after identities, ignored paths, caps, counts, bytes and completion
state. The full manifest is intentionally new under C; no active status is edited.

## Commands for root after review and at a suitable time

Read-only default:

```sh
python3 /cache/zhonghao/h3/c_adaptation/deploy_candidate_31731.py
```

Explicit copy (not executed by this subtask):

```sh
python3 /cache/zhonghao/h3/c_adaptation/deploy_candidate_31731.py --apply
```

CPU toy tests, with no deployment or NPU:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m unittest discover -s /cache/zhonghao/h3/c_adaptation \
  -p test_deploy_candidate_31731.py -v
```

Tests inject temporary local paths and toy pins into the engine; CLI permits no
path, host, cap or hash override. Windows cannot prove Linux FIFO, symlink or
renameat2 behavior; the skipped OS-specific checks must be rerun on Linux before
actual deployment. No successful toy result means the real B source inventory
has passed the caps or that C is already deployed.
