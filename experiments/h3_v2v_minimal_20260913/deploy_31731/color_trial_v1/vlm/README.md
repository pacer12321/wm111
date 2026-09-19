# Real video-conditioned VLM stage0

This is a new local-only implementation, not a record of successful NPU inference. It uses the complete local Qwen3VL conditional-generation checkpoint in H3's `text_encoder`, including the visual tower and language generation head. It does not substitute H3 text embeddings, a text-only classifier, the source benchmark name or human `router_control` labels for VLM inference.

Four production files deploy together to `/cache/zhonghao/h3/color_trial_v1/vlm`: `vlm_contract.py`, `vlm_worker.py`, `run_vlm.py`, `launch_vlm.sh`. Existing H3/environment/model/tiny/old run files remain unchanged. Root reviews and deploys; this implementation task has not accessed remote machines or started a model.

Fixed input is the exact unreversed `shirt_red_couple_124/source.mp4` plus its fixed shirt-color instruction. PyAV verifies all 124 decoded PTS at 24 fps and samples indices `[0,8,16,25,33,41,49,57,66,74,82,90,98,107,115,123]`, preserving individual RGB hashes and the stacked video hash. Only these 16 frames, not all 124, are shown to the model. The installed processor receives a true video tensor, original frame indices/fps metadata, and `do_sample_frames=False`. Its temporal patching averages timestamps of each pair; both original and pair timestamps, actual token count, nonempty video tensor/grid and CPU tensor SHA are recorded. Pixel budget controls preprocessing, not the H3 generation pipeline.

The API is based on the actual installed transformers5.14.1 sources copied read-only into the sibling `api_reference` directory. Their six source hashes and the checkpoint config hash are pinned. General reference: [Hugging Face Qwen3-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl). The manual two-NPU map is still pending real runtime verification: vision/embedding/rotary/language layers0–31 on physical0, layers32–63/norm/lm_head on physical1. Complete loading key coverage, BF16 parameter byte count and actual map are checked. No fallback downloads, CPU-offload substitution or environment edits are performed.

The supervisor holds the original eight per-card locks and its own stage0 lock, requires fresh eight-card idle/health plus 55GiB free HBM/card and 128GiB effective host/container RAM, but exposes only physical0,1 to one worker. Its unchanged environment bootstrap includes `/usr/local/sbin` for `npu-smi`. The worker verifies all inherited lock descriptors and its direct supervisor/run identity before any NPU path. Greedy generation uses max512 tokens, eager attention and no retry; an 1800-second outer guard cleans only owned process groups. The process must exit and all eight cards must be verified idle before the terminal state becomes `completed`.

After review/deploy, CPU preparation (no torch/torch_npu import or model load):

```bash
source /cache/zhonghao/h3/env_h3_31731.sh
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/vlm/vlm_worker.py --prepare-only
```

One explicitly authorized supervised inference (not executed here):

```bash
source /cache/zhonghao/h3/env_h3_31731.sh
command -v npu-smi bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/color_trial_v1/vlm/run_vlm.py --allow-npu
```

Runs are independent under `/cache/zhonghao/h3/color_trial_v1/vlm_results/shirt_red_couple_124/runs/<UTC>_<runid>`. Keep the actual per-run `vlm_status.json`, frozen evidence, `preprocessing.json`, `model_load.json`, `model_output.json`, `worker_result.json`, log and npu-before/after. Raw generated IDs/text are saved before parsing. Missing/malformed/duplicated/nonfinite JSON, missing visual evidence, schema inconsistency or token-ceiling truncation fails closed; `uncertain` remains uncertain. The expected human class is never an input. VLM `preserve` is not ground truth and does not establish that a later H3 output actually preserves timing.

`run_vlm.verify_completed(run_dir, run_id)` is the reusable read-only admission checker. It returns `status`, `result`, `frozen`, and `records`; `result.classification` is the full parsed object, whose `classification` field is preserve/change/uncertain. It rechecks exact per-run file hashes, current source/model/API/runtime identity, raw-vs-parsed generation, actual video preprocessing, historical npu-after and original process identity exit. It neither imports NPU/torch nor writes/probes current devices. This allows A/B/C's running `assert_unchanged` to reuse the completed evidence; fresh current device admission belongs to the next H3 preflight. Root must bind the exact run/status/result SHA in a separate admission record only after independent validation. No admission has been created here.

Pure CPU protocol/reader tests (synthetic temp fixtures only):

```bash
python -B test_vlm_cpu.py
bash -n launch_vlm.sh
```

Local result:14 tests passed, zero skips. These are not a real processor/model/NPU pass. The production model must still validate actual installed processor loading, NPU eager operations and cross-device generation. A real failure is retained for review, never replaced by a fabricated `preserve` result.
