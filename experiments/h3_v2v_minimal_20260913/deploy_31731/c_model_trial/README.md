# C8 full-model trial: local implementation, not a run result

This is a separate opt-in supervisor. It does not edit or schedule A/B, their
queue, the environment, validation code or the C candidate. It never retries.
Do not start it while A/B or any other owner holds the eight cards.

C removes only target-query to non-corresponding **visual-source latent-frame**
Softmax edges. Text/audio and other non-target queries, B target neighborhoods,
target anchors and the original linear branch stay as implemented in the pinned
C candidate. All released Stage-B weights are retained: 535 Ref2VA base tensors,
800 branch tensors and 208 merged LoRA pairs. This entry implements no VLM router.
The explicit reverse sample is a negative/boundary test of fixed correspondence,
not a claim that strict same-frame masking handles temporal reversal correctly.

## Isolated deployment and unchanged experiment settings

Install these three reviewed production files into
`/cache/zhonghao/h3/c_model_trial_code/`:
`c_trial_gates.py`, `run_c_trial.py`, `launch_c_trial.sh`.
Keep the installed A/B/validation helpers read-only at their existing roots.
The C tiny verifier's five pinned files must be in
`/cache/zhonghao/h3/c_validation_code/`; they are imported only as CPU readers.
The exact six candidate hashes are inherited from the pinned `c_profiles.py`.

Only physical cards `0,1,2,3,4,5,6,7` on the fixed 31731 hostname are allowed.
Actual hostname, boot ID and aarch64 architecture are read, not supplied by CLI.
Compute settings match reviewed A8/B8: USP8, ring1, DiT TP1, text TP8,
VAE patch parallel8/tile, layerwise offload, no precision override.
The two fixed source/prompt profiles and sampling settings are reused verbatim:
1344×768, 24fps, duration5, seed4101, 50 steps, video flow12, audio flow3.

- Vendor: `/cache/zhonghao/h3/candidates/c_v1/vllm-omni`.
- Output: `/cache/zhonghao/h3/c_model_trial/01234567/<sample>/C/runs/c_<UTC>_<UUID>`.
- HTTP: `127.0.0.1:19100`.
- Short TMPDIR: `/cache/zhonghao/h3/tmp/cmodel_<UUID>`.
- ATB namespace: `zhonghao_31731_C_01234567_<UUID>`.
- Shared per-card flock files: `/cache/zhonghao/h3/validation/card_locks/deviceN.lock`.
- Trial/sample mutex: `<output-root>/run.lock`.

The launcher first sources the unchanged private environment, then prepends the
C vendor. Existing automatic internal HCCL ports remain the same framework policy;
the eight shared locks prevent this entry from running beside another A/B/C owner.
Startup requires idle/healthy cards with at least 55GiB HBM free each, available
host and container memory of at least 600GiB, free HTTP port and all nine locks.

## Mandatory evidence sequence

1. Current completed migration, current-host CPU runtime report, unchanged model
   transfer identities/headers/configuration, source SHA and decoded metadata.
2. **C's own** `c_validation/01234567/c_validation_status.json`: completed genuine
   SP8 C run, same host/boot, all six candidate and verifier/strategy/env hashes,
   actual eight rank files, correct numeric/mask checks, cleanup and idle release.
   Missing C proof, a B proof, changed hashes, incomplete ranks or stale latest
   state fail closed. The gate calls the actual C `verify_results` reader.
3. Start one owned service; send exactly one 2-step smoke request.
4. Check successful decoded output, frozen source/prompt/settings, unchanged
   service PID/start ticks, all eight owned live worker PIDs and their complete
   `OPENVDN_B_LOAD_RECORD` / `OPENVDN_B_CPU_LOADING_END` records. The marker names
   are genuinely retained from the unchanged Stage-B helper, not renamed data.
   Loading threads must be four and restored to each worker's original values.
5. Require one real `C strict same-frame source metadata` record from **each**
   of those eight PIDs. These INFO records use `H3C pid=<LogRecord.process>`.
   Parse the actual bounded Python dictionary repr with `ast.literal_eval`,
   never `eval`. Require the strict mode, one video, patch(1,2,2), Fs=Ft37,
   target(37,48,84), target audio207 and consistent metadata across workers.
   Source spatial resolution can differ; text length/source audio are recorded
   from runtime rather than guessed. Reverse source audio must be zero.
6. Only that same service may receive one 50-step request. Afterwards verify the
   frozen evidence, unchanged log prefix and exactly a second matching metadata
   record per worker. Then clean up only this supervisor's owned process groups,
   verify card release and release locks. No automatic fallback/retry.

The strict INFO record is emitted in `pipeline_minimax_h3.py:1010` after actual
VAE/ref-block metadata is made and attached to the denoising branch. It is **not**
a per-layer attention mask/tensor-coordinate trace. A successful request through
the pinned transformer also passes its actual strict packed-layout checks; the
independent C tiny proves the small SP8 mask and preservation behavior. No new
candidate observation patch is required for this evidence scope.

The shape constants are derived from the **actual model code**, not inferred
from an output MP4: `pipeline_minimax_h3.py:440–451` rejects non-24fps and converts
duration5 to `round(5*24)=120`; `time_request.py:5–18` aligns upward to `17n+5=124`
and maps that to `((124-5)//17)*5+2=37`. Its audio planner at lines29–31 computes
`round((124/24)*40)=207`. Pipeline lines1279–1282 use height/16 and width/16, hence
48×84. There is no internal30fps then output24fps path in these functions.
`reference_video.py:116–119` transcodes source at24fps capped at124frames, using
its own `_reference_video_shape` scaling. Both current source aspect ratios yield
1344×768, but the gate records actual source VAE shape and does not assume target
spatial equality. Source soundtracks are encoded from the original file, so lake
audio length is observed rather than forced to target207. Reverse has no audio.
`time_request.py`, `reference_video.py`, `vae.py`, packing and denoising helper
hashes are additionally pinned/frozen by this full-model entry. CPU regressions
execute the real shape planner and AST-extracted real `_resolve_shape` and source
transcode functions with a fake subprocess, never torch/NPU or an actual encode.

Each service/client request timeout is 3600 seconds. That is only a waiting
ceiling, not a speed measurement. Each request result records its full HTTP
end-to-end elapsed time independently. No quality, actual NFE or speedup claim is
made automatically. Final success is `formal_completed_review_required`.

## Commands for the parent after review and resource release

Local CPU tests (no torch/NPU imports or SSH):

```powershell
python -B -m unittest discover -s experiments/h3_v2v_minimal_20260913/deploy_31731/c_model_trial -p test_c_trial_cpu.py -v
```

The test file may be deployed alongside the three files for Linux CPU checks;
it reads installed frozen dependency files. All simulated reports are explicitly
CPU fixtures in temporary directories, never production evidence.

After a real C tiny pass and cleanup, the reviewed operator can run one sample:

```bash
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/c_model_trial_code/run_c_trial.py --group 01234567 --sample lake_snow --allow-npu
```

The only other accepted sample is `explicit_reverse_couple_124`. There are no
arbitrary source/model/step/card flags. Omitting `--allow-npu` exits before host
checks or resource acquisition. Do not invoke the shell launcher directly.
The existing remote A→B queue is intentionally unchanged and cannot launch C.

At implementation delivery: CPU coverage is available; genuine C NPU tiny,
full-model smoke/formal, quality and speed are still unverified. A missing C tiny
proof prevents starting any C model service.
