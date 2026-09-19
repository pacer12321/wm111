# Shirt-color ABC v1 — independent 31731 eight-card entry

Only `shirt_red_couple_124` is accepted (also the default). The requested edit changes only the man's pale shirt to solid red while preserving the original actions, order and timing. The official temporal-reordering identifier is source provenance, not this edit instruction. Every sample must first pass actual video-conditioned VLM inference. The independent `vlm/` stage uses the existing complete Qwen3VL conditional-generation checkpoint; this is not the H3 hidden-state encoder being mislabeled as a classifier. There is no training, automatic queue, retry or new DiT change here.

The common A sample gate now calls `a/vlm_gate.py`. It requires a root-reviewed `vlm_admission.json` binding one exact completed VLM run, status/result hashes, source and prompt. The read-only completed-run checker verifies actual raw generation, video inputs, weights/code identity and historical cleanup. A/B/C reuse that one classification; the generation definitions below do not change. An absent/failed/change/uncertain result cannot silently enable strict same-frame C for this color trial. No admission may be created before real inference and independent review. A current execution status must be checked separately; CPU tests are not inference results.

The new source must be an exact byte copy of the existing unreversed 104:228 trim: `/cache/zhonghao/h3/data/shirt_red_couple_124/source.mp4`, SHA256 `4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2`, 1,527,004 bytes, 1280×720, 124 frames at 24 fps, no audio. `manifest.json` in that directory must match the fixed prompt, provenance, generation and all-false router-control flags in `a/trial_gates.py`. Preparation/deployment is performed separately by the root agent; these entries do not create or reencode the source.

## Scope of the changes

The three original supervisors are copied without any function/body changes (only their module descriptions change). New A gates remove both old sample profiles, validate only this fixed color sample and use the new code/output roots. New B/C gate dependencies pin the actual new A/B gate hashes. C requires `source_audio_t=0` for this sole silent sample. Other gate functions retain the original AST, including runtime/transfer/model-header checks, eight-worker loading evidence and C's own actual tiny-proof reader.

A retains original dense Ref2VA, B retains the released complete Stage-B linear branch plus local Softmax/global source, and C retains the already reviewed strict same-latent-frame visual-source rule with otherwise identical B weights/linear behavior. Original vendor trees, weights, environment, runtime/tiny evidence and old production entries are read-only and unchanged. The C tiny proof remains its own `db9ddbbbedfc4aacb13ed6823c69b2fe` run, never B's proof. Missing/stale/mismatched evidence fails closed; do not refresh a proof to bypass a failure.

## Deploy and run

Deploy `a`, `b`, `c` to `/cache/zhonghao/h3/color_trial_v1/{a,b,c}` as new private directories, verify the supplied hashes, and do not overwrite an existing namespace or any old result. All three new code directories must be present before freezing the first run. Deploy this CPU test file beside them if running the same local checks on Linux. It reads the unchanged original trial files for AST comparison and the existing `validation_code` helpers; it does not initialize devices.

Each case writes only its independent result namespace under `/cache/zhonghao/h3/color_trial_v1/results/01234567/shirt_red_couple_124/{A,B,C}`. Each takes its own run lock and the original eight shared per-card locks in `/cache/zhonghao/h3/validation/card_locks`. The ports remain A 19098, B 19099, C 19100. All use USP8/ring1/DiT TP1/text TP8/VAE tile8, original precision/offload, 1344×768, 24 fps, requested duration5, seed4101, 50 steps, flow12/audioflow3. A retains its 1800-second request ceiling; B/C retain 3600 seconds, which is only a waiting limit, not an acceleration setting.

Before each launch, the root agent must verify the preceding run succeeded or is intentionally stopped, all original owned process identities exited, all eight cards are freshly idle/healthy and this case has not already run. These scripts do not coordinate A→B→C automatically. Never launch the three commands together. The supervisors repeat the fresh resource/evidence/port checks under locks: minimum effective host/container RAM600GiB and free HBM55GiB per card. They require a new two-step smoke in the same service before exactly one 50-step request, then clean only their owned process groups.

In a 31731 shell, first source the unchanged environment and check executable resolution (in particular the prior queue PATH issue):

```bash
source /cache/zhonghao/h3/env_h3_31731.sh
command -v npu-smi bash curl ffmpeg
```

`npu-smi` must resolve successfully (the existing bootstrap includes `/usr/local/sbin`), and `ffmpeg` must resolve to `/cache/zhonghao/h3/bin/ffmpeg`. Root-reviewed case commands, executed separately only after the above gates:

```bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/a/run_a_trial.py --group 01234567 --sample shirt_red_couple_124 --allow-npu
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/b/run_b_trial.py --group 01234567 --sample shirt_red_couple_124 --allow-npu
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/c/run_c_trial.py --group 01234567 --sample shirt_red_couple_124 --allow-npu
```

An external supervised detached launch can use the root's existing reviewed procedure; no job has been submitted by this implementation task. Do not restart the old AB/BC queues, run the launcher shell directly, modify live code/evidence, or treat the old lake/reverse completion readers as color-specific evidence.

## CPU validation and final evidence

Local command from the workspace root:

```powershell
& 'C:/Users/DZH/AppData/Local/Programs/Python/Python311/python.exe' -B experiments/h3_v2v_minimal_20260913/deploy_31731/color_trial_v1/test_color_trial_cpu.py
```

Linux command after new-code deployment:

```bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/test_color_trial_cpu.py
```

All tests use synthetic temporary CPU fixtures and mock device/process inspection. They do not establish a real NPU smoke or VLM pass. After VLM admission integration, 28 local ABC tests passed with zero skips, including rejection when real VLM evidence is missing. VLM has its own separate CPU tests. Production shell syntax must also be checked with `bash -n`. `sha256.json` is the historical pre-VLM snapshot; use the separately finalized `release_sha256.json` for deployment, never stale hashes.

Compare new vs old `run_[abc]_trial.py`: only module documentation differs. Compare A gates: fixed single color profile/manifest, code/output roots and required real VLM admission; B gates: new A pin/root and B code/output roots; C gates: new A/B pins/root, C code/output roots and silent-source constraint. Launchers differ only in permitted sample/error text and output namespace; command-line compute flags are unchanged.

The desired empirical latency order is A > B > C, not an assumed finding. Report generation-request times separately from VLM decode/preprocessing/inference and model-loading times. A warm complete-method comparison against original A must include router request cost without artificially adding router overhead to original A. Cold-start loading is a separate comparison with equally specified loading boundaries. Quality and temporal preservation must be assessed; do not proceed to the deferred normal temporal-reordering test merely because C beats B while remaining slower than A.

After each formal result, independently verify the actual per-run status, frozen/current evidence, source/prompt/request fields, HTTP headers/results, output SHA and ffprobe, exact smoke-service worker identities and cleanup/fresh idle. B/C require actual full 535/800/208 loading and restored-thread records. C additionally requires actual strict metadata from all eight worker PIDs for both smoke/formal: verify the smoke log prefix, parse `requests=2` and JSON-normalize against the actual `formal_strict_metadata.json` (not a nonexistent status-embedded SHA). Then inspect whether the man's shirt is red, the woman's clothing stays unchanged, and action ordering/timing/framing are preserved. Successful HTTP or faster latency alone does not establish that the edit works or that quality is equal.
