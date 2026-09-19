# D-only profiling on physical cards 0–3

STOPPED — user clarified that this is cost decomposition, not a full D rerun.
Run `c09f9d07a6c248578ce14f67063f735e`, supervisor1219416, was stopped during
profiler export smoke, before model loading or the full generation request.
Do not dispatch this 50-step workflow again for the current diagnostic task.
Prefer existing evidence and, if needed, a bounded operator/layer/step sample;
do not produce a new full-video D experiment or consume cards4–7.

User asked to continue D cost decomposition after temporarily releasing cards
4–7 to another researcher. This is a separate four-card diagnostic, not a
restart of the old eight-card B→D queue.

Remote root: `/cache/zhonghao/h3/profiling_d4_v1`; host SSH port31731; HTTP19112.
Only device0–3 locks are acquired; only `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` is
exported. No process on cards4–7 is started or stopped by this workflow.

Preserved: D attention math, original stage-B weights/LoRA, source/edit/seed4101,
1344×768, the 50-point schedule (49 forwards), layerwise offload, no training.
Changed: USP4, textTP4, VAE tile4; 14 heads per rank instead of7. A private copy
of the diagnostic logging guard also accepts world4. No source cache fix is
mixed into this measurement.

Forward3 records rank0 CPU/NPU operators. Forward4 records nested current-stream
NPU-event intervals on all four ranks. All forward CPU durations are logged.
The profiler export smoke must pass before model loading. One full request only;
failure triggers owned-process cleanup and no automatic retries or B run.

Interpretation: four-card operator/communication proportions cannot be used to
numerically decompose the previous eight-card 972-second D request. Shared host
load from other researchers can also affect measured performance. Nested
inclusive ranges must not be summed; NPU event intervals include waits/host
starvation and are not kernel busy time. `summarize_events.py` outputs both
inclusive scopes and a separate nonoverlapping self-time accounting.

This README is a configuration record; live status is the remote `status.json`.
