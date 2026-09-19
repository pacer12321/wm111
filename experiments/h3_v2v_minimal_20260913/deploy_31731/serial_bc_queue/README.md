# One fixed B6362 → C tiny → C lake continuation

Local implementation only. Nothing has been deployed or submitted by this
subtask. Keep using the existing heartbeat until the owner finishes C deployment,
Linux CPU checks and review. This coordinator contains **no deployment code**.

The only accepted predecessor is:

`/cache/zhonghao/h3/b_model_trial/01234567/lake_snow/B/runs/b_20260913T140300Z_6362ce1811da4db69d335b3b3ca5d1a4/b_status.json`

Its supervisor must be PID4692 / start ticks124486422 on the actual host/boot.
No latest-status or alternative run/source/sample argument is accepted.

## Preparation required before submitting this queue

The owner must separately deploy and review:

- Completed, verified C vendor at `candidates/c_v1/vllm-omni` and its successful
  `deploy_manifest.json`. Incomplete/missing C returns an error **before** ledger
  creation or any child launch. It does not loop, deploy C or repair files.
- The three frozen C model entry files at `c_model_trial_code`.
- The five frozen C tiny verifier/launcher files at `c_validation_code`.
- This coordinator and `bootstrap_bc.sh` at `serial_bc_queue_code`.
- The already frozen AB queue reader and A/B/runtime/validation dependencies.

The C full gate remains the authority for own-C-tiny validity and model startup.
This queue invokes the existing pinned AB `completed_evidence` **reader only**
for B/C formal artifacts. It does not construct an AB scheduler, change its
globals, read its latest state or invoke its old child-environment function.

## Fixed sequence and safety

1. Read the fixed B per-run record until its one formal request, source/prompt/
   sampler/weights/runtime proof, output SHA and decoded metadata are verified,
   cleanup is complete, selected cards are idle and the original supervisor has
   exited. A reused numeric PID is not treated as the old process.
2. Write/fsync `C_tiny_launch_intent.json` exclusively, then launch exactly one
   `/cache/zhonghao/h3/c_validation_code/run_c_validation.py --allow-npu`.
3. Discover only newly created **per-run** directories. A record must match this
   queue's own Popen PID, start ticks, process group and session before binding.
   After binding, only that immutable run path is followed; unknown latest
   pointers are never followed. C's own `tiny_gate` must validate that same run,
   all real rank/source/numeric proofs and cleanup. Require child exit0 and idle.
4. Recheck the complete tiny proof, then exclusively fsync
   `C_full_launch_intent.json` before one full C lake supervisor. That supervisor
   performs its original same-service smoke → strict/load/PID gates → one50-step
   request → cleanup. The queue changes no sampling or attention settings.
5. Check completed C artifacts through the pinned reader. Recheck the smoke log
   prefix length/SHA and its recorded eight-card worker identities against the
   eight load PIDs. Call C's actual strict parser on the saved prefix (one request)
   and complete log (two requests); JSON-normalize and compare respectively with
   the saved smoke strict metadata and `formal_strict_metadata.json`. Malformed,
   changed or mismatched evidence is rejected—not merely recorded by file hash.
   Require exit0 and idle. Finish `completed_review_required`.

The queue holds only its private `serial_bc_queue/queue.lock`, not NPU leases.
Children take their own trial mutex and eight shared per-card flocks and repeat
idle/health/memory gates. C tiny and full C cannot overlap. All8cards, lake source,
seed4101, USP8/textTP8/VAE8, 50steps and the full supervisor's3600-second waiting
ceiling remain fixed. No reversed sample, rerun B, extra dataset, fallback or
automatic retry is included.

The deterministic queue directory
`serial_bc_queue/b_6362ce1811da4db69d335b3b3ca5d1a4_c_tiny_c_lake`
is itself a durable no-restart ledger. Each stage also has a durable exclusive
intent **before Popen**; a crash may omit work, never resend it. Keep failed
ledgers for inspection—do not delete them to retry. Existing directory refusal
does not overwrite its state. Any failure/changed binding stops the chain.

On error/timeout/signal, the coordinator may send SIGTERM only to its current
Popen child after exact identity plus `H3_SERIAL_BC_QUEUE_ID` environment proof;
that supervisor owns cleanup. It never signals B, foreign PIDs/groups or sends
SIGKILL. If its child cannot exit within180seconds, report `needs_attention`;
no next job starts. Waiting is bounded: B4hours, C tiny1500seconds (its actual
validation limit remains900), full C4hours. These limits are not speed claims.

## The previous PATH issue is explicitly handled

`bootstrap_bc.sh` sources the unchanged private `env_h3_31731.sh` before the
coordinator and again before each child supervisor. It requires actual
`command -v npu-smi` → `/usr/local/sbin/npu-smi`, private `python`, and private
`ffmpeg`. Python separately validates real executable resolution, executable
identity and the private CANN9 path before waiting, each check and each spawn.
The verified PATH including `/usr/local/sbin` is preserved in child environments;
arbitrary old H3/attention/Python/device-mask overrides are cleared. No system
shell, CANN installation, shared environment or active B file is modified.

## Commands after review, not authorization to run now

CPU tests (pure fixtures under temporary directories; no real process/NPU):

```bash
python -B -m unittest discover -s serial_bc_queue_code -p test_serial_bc_cpu.py -v
bash -n serial_bc_queue_code/bootstrap_bc.sh
```

The parent may deploy the test file alongside the two production files and run
Linux CPU checks. On Windows, flock is mocked and directory fsync is not claimed
as a Linux-kernel result. The tests explicitly mark all synthetic reports as CPU
fixtures and never publish production pass evidence.

Only when preparation and review are complete, the operator can submit once:

```bash
/bin/bash /cache/zhonghao/h3/serial_bc_queue_code/bootstrap_bc.sh queue
```

The shell accepts only `queue`, `C_tiny`, `C_full`; direct child modes require the
exact queue environment marker. Public Python CLI has only `--allow-c-chain` and
does nothing without that explicit flag. Queue completion proves execution and
cleanup, not edit quality, practical speedup or publishable research conclusions.
