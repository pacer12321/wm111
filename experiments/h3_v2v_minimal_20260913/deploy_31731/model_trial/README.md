# 31731 fixed eight-card A trial

Deploy this directory's three runtime files to exactly
`/cache/zhonghao/h3/model_trial_code`. The already reviewed `validation_code`
directory remains separate and unchanged. This entry point imports its
`profiles`, `supervision_base` and result verifier without modifying globals.
It uses the SAME private per-card mutex files as tiny validation, but different
per-group trial run mutexes and outputs.

After all gates below pass, the priority run repeats original A's lake-to-snow input:

```bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/model_trial_code/run_a_trial.py --group 01234567 --sample lake_snow --allow-npu
```

Omitting `--sample` also selects `lake_snow`. A separately scheduled reversal
run must explicitly select `--sample explicit_reverse_couple_124`. Arbitrary
source/prompt/model/sampler overrides are not accepted. The only allowed group
is `01234567`, HTTP port19098; legacy four-card groups are rejected. Both sample
profiles share all eight per-card locks and cannot run concurrently.
Each launch requires at least 600 GiB host/container headroom and
55 GiB free HBM per selected card. Existing foreign jobs are never stopped.

The launch always does exactly one fresh two-step smoke, validates its decoded
1344x768/24fps/124-frame MP4, verifies the same service's owned eight workers,
then attempts one 50-step formal request. There are no automatic retries. The
service retains all leases between requests and is cleaned afterward using the
independent validation helper's PID/start-time/session/run-marker rules.

Only original `/cache/zhonghao/h3/src/vllm-omni` runs. B activation variables are
cleared and OpenVDN is explicitly disabled. The launcher retains the reviewed
bootstrap's private CANN Python paths while placing original source first.
No candidate code or Stage-B weights
are used for inference. The original pipeline and transformer SHA are pinned;
the CLI retains original dtype defaults, layerwise offload,
tile VAE mode and FLASH_ATTN. The fixed eight-card allocation is
num_gpus8/USP8/ring1/DiTTP1: H3's56 heads split into seven per Ulysses rank.
Text TP8 and VAE patch parallel8 are REQUIRED for this installed pipeline.
The first A8 attempt used4/4 and failed before inference: the text group only
contained ranks0..3 while every rank asserted membership; VAE likewise requires
parallel size1 or the full DiT world. Keep native VAE tile size256/overlap64.
No source model or weights are changed, but different reduction ordering may
produce floating-point differences. Future algorithmic A/B/C comparisons must use the same
A8 hardware profile; A4/B8 ratios are not algorithm-only acceleration.
Startup timeouts are 2400/2400 seconds,
health 2700, and each HTTP request 1800. Startup/smoke are excluded from formal
request end-to-end timing. NFE and pure DiT time remain explicitly unmeasured.

Required preflight evidence:

- Complete current v2 `consume_status.json` plus matching `transfer_verified.json`.
  No startup while weights are still copying. Ref2VA files are rechecked against
  transfer sizes; small configuration SHA and safetensors headers are reread.
  Full tensor payload SHA comes from verified transfer, not a fresh 135 GB hash.
- `/cache/zhonghao/h3/runtime_validation.json` contains the actual CPU checker
  report with `status=passed_cpu_runtime_checks_only`, `host` (real hostname,
  boot_id and machine), `env_script_sha256`, and `source_sha256` mapping the two
  original model filenames to their SHA. It must retain the original
  `device_guard` and `npu_inference_verified=false` fields.
- This group's completed tiny state and eight-rank USP8 collective proof, same host/boot and
  current candidate/runtime hashes, with verified cleanup. The B tiny proof is
  an environment/collective prerequisite, not evidence that A or B full video
  quality has already passed.
- Default `lake_snow` uses unchanged original A source
  `/cache/zhonghao/h3/data/minimax_h3_t2va_50step.mp4`, size12754317 and SHA256
  `e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9`.
  Actual bytes must match both this code pin and the completed migration's
  verified entry. Exact original A prompt is pinned. This source has no derived
  preparation manifest: the run saves a frozen code-and-transfer evidence
  record, without inventing crop provenance or GT. Its original decoded
  dimensions/frame count/FPS are verified before use.
- Explicit `explicit_reverse_couple_124` requires the fixed prepared sample at
  `/cache/zhonghao/h3/data/explicit_reverse_couple_124/manifest.json`, using the
  existing preparation script's schema and exact prompt. Its source SHA and
  decoded frame count/FPS/dimensions/audio status are checked. Source crop is
  original frames104:228, not a pre-reversed input. Generation parameters are
  fixed at1344x768,24fps,5 seconds,seed4101,50 requested steps,flow12,audio-flow3.
  Source is1280x720 with124frames24fps and no audio, SHA256
  `4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2`.
  Reversal prompt is self-authored and text-only classifiable; there is no
  official paired target or generated GT. It does not establish VLM necessity.

Outputs and evidence are isolated under
`/cache/zhonghao/h3/model_trial/01234567/<sample>/A/runs/a_<UTC>_<run-id>/`. The latest
sample state is `model_trial/01234567/<sample>/A/a_status.json`. Source,
prompt, sample identity and same-service smoke evidence cannot be interchanged.
A usable run must have
`phase=formal_completed_review_required`, `formal_50step_completed=true`,
`formal_request_attempts=1`, `cleanup_completed=true` and
`selected_cards_verified_idle_after_cleanup=true`. A successful MP4 does not
establish edit fidelity or an acceleration gain. All nine lease descriptors
(group plus eight cards) remain held until identity-checked cleanup completes.

Local CPU test, no NPU or subprocess launch:

```bash
python -B test_a_trial_cpu.py
bash -n launch_a_trial.sh
```

The CPU provenance tests additionally read the original local repository and
sample-preparation script. Run this suite locally, not as a standalone copied
test directory. Linux lock/process interactions remain an on-host integration
check; do not label CPU toy tests as a launched or passing model run.
