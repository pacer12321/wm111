# Acceleration work moves to NVIDIA A100

## LIVE: 2026-09-15 11:25 CST — corrected admission passed, real D request running

- Controller1730318 reports smoke_running, requested_steps2 (not just loading).
  Serverlog11:25:09: Ref2VA Qwen presentation6159tokens/1reference video.
  This proves the erroneous Ngid gate has been crossed and POST executed.
- GPU0/1 use46568/45714MiB at11:25:12 during source encoding. No D result or
  B/D formal timing yet; do not claim denoising complete or quality verified.
- Continuation1730319 remains gated on successful D output; same-runtime B/D
  comparison only afterward. Auto-cleanup1726455 and full-lifetime guard active.

## LIVE: 2026-09-15 11:21 CST — PID admission fixed and D restarted

- Prepared v2 first smoke1726651 loaded both main/nonmain payloads and returned
  health200, but runner wrongly compared /proc Ngid with NVIDIA host PIDs.
  It exited before POST/denoising, no additional OOM. This was runner admission,
  not an identified foreign job or model/offload failure.
- Fix: gpu_client_admission.py compares containerPID+starttime+exact device
  handles (/dev/nvidia3,/dev/nvidia7 verified by GPU UUID XML). NVIDIA hostPID
  set is tracked separately for insertion detection, never matched to Ngid.
  Unreadable pre-existing container-init descriptors recorded at idle baseline;
  new unreadable processes fail closed. Access permissions not changed.
- Five regression tests passed locally and remotely; real/proc scan passed.
  Old failed attempt archived (no files deleted) under RUN/history/pid_check_v1.
  Model/storage code, checkpoint, seed and memory guard unchanged.
- New D smoke controller1730318, one-shot continuation1730319. At11:22 actual
  encoder loading under SP2. Need verify request admission/denoising, not yet
  proof of E2E success. Guard1726455 still active; B/D formal not started.

## LIVE: 2026-09-15 11:10 CST — user approved upload/execution

- User explicitly replied "上传执行" to destination-specific request. Fixed
  integration, tests, guard and B/D runner/continuation uploaded successfully.
- Remote torch regressions10/10 passed including2 newly added nonmain
  inference-tensor/Parameter alias/flag/module.to checks. No Embedding whitelist
  change. Current integrationSHA1c704588f94f9d18882408ebc4eac59a82617c8c1af9ef3809b05d8521742aa1.
- Isolated RUN=/cache/zhonghao/h3/bd_prepared_20260915_v2. Original B/D and
  failed prepared v1 records preserved. Same frozen source/prompt/seed/config.
- D2step smoke controller1726651 actually loading_model; both workers initialized
  CUDA/SP2, not merely queued. No generated video/timing yet.
- One-shot continuation1726652 waiting_for_D_smoke. It launches B/D comparison
  ONLY after successful D smoke and controller cleanup. Failure => no comparison.
  Same new offload B/D, oldBtime excluded, timing beforequality.
- Auto-cleanup guard1726455 active at bd_gpu_guard_20260915_v3. User explicitly
  authorized stopping new GPU occupiers without asking again; scoped to verified
  same-account GPU device clients on exact30674UUIDs, excludes ownROOT jobs and
  verified old diagnostic. Files preserved. Max12h/STOP/comparison completion.
  It now remains active through recoverable smoke failure (unlike endedv2).
- Full-lifetime CPU non_file>250GB plus OOMcounter monitoring retained equally
  for B/D. Initial GPU preflight showed only old850MiB context1550469.

## CURRENT: 2026-09-15 after 11:05 CST — upload approval required

- User explicitly authorizes automatic termination of newly inserted 30674 GPU
  jobs during this B/D experiment, without repeated per-job questions. No files
  are deleted. Concrete task trees were terminated after identity verification.
- GPU AllGather v2 PASSED both ranks: all50 main blocks/bothslots,2600 tensor
  slot pairs perrank, exact CPU reference hashes, peak allocated4130190336B.
- CUDA/NCCL native asynchronous ring PASSED3/4/50layers each20rounds,1140
  outputs/rank bitwise equal to unoffloaded smallLinear references. Not H3 E2E.
- Native/control-flow CPU suite8/8 passed remotely, including50-layer ring.
- Actual D smoke1721987 in /cache/zhonghao/h3/bd_prepared_20260915_v1 failed
  after main shards built, at first non-main Parameter binding: inference tensor
  requires_grad=True outside InferenceMode. OOM counter stayed3, no guard trip.
  No video/no new B/D timing. Old B1613.776763s remains non-comparable.
- Suspected native Qwen UnquantizedEmbeddingMethod issue was disproven: H3
  uses custom encoder without quant_method. No such whitelist change made.
- Local h3_prepared_integration.py now constructs nonmain Parameter inside
  inference_mode, preserving flags/value/storage alias; logger made visible.
  New2 realtorch regression tests not yet uploaded/executed. No live hot patch.
- Local run_prepared_bd.py now targets isolated bd_prepared_20260915_v2;
  continuation script will launch same-offload B/D only after D smoke succeeds.
  Local guard v3 supports new GPU device clients, excludes ourROOT jobs and
  verified existing diagnostic, continues across recoverable smoke failure,
  max12h/STOP/completion termination. These revisions NOT deployed.
- Remote old guard v2 PID1721732 exited afterv1 failure (ended.json verified).
  Hence there is CURRENTLY NO ACTIVE AUTO-CLEANUP or model generation.
- SCP of fixed integration/tests/newguard/runner/continuation was REJECTED by
  auto-review: explicit destination-specific upload approval requested. Do NOT
  bypass via inline SSH code, remote edits or alternate transfer. Ask permission
  to upload these scripts to user-designated30674/cache/zhonghao/h3 and run.

## CURRENT: Track B verification gates advanced, 2026-09-15 07:49 CST

- User requests B/D generation timing first, then quality; same offload and
  stable memory required for formal comparison. No new B/D video result yet.
- Remote isolated ROOT/track_b_validation_20260915: initial19 CPU tests all pass
  (including2 previously skipped torch tests). Full real-weight gate PASSED:
  1335 target tensors,208 LoRA pairs,52QKV conversions, bitwise equality across
  two tensor-wise shards; elapsed370.478s, peakRSS1570712KiB. This is NOT GPU or
  block-order equivalence. See localtrack_b_offload/real_equivalence_status.json.
- Real meta DiT enumeration passed after metadata-only TP rank/world getter
  fixes:1335entries,0nonpersistentbuffers, CUDA not initialized. Copies saved
  real_meta_v1.json and validated_manifest_v1.json; runtime_usable stillfalse.
- Prepared-shard hook seam passes12 remoteCPU tests incl2 native-hook/fakeAG
  tests: zero-copy handoff, no original full sharding, two-slot repoint/reuse.
  Entirepipeline integration remains unfinished; do not call DLO ready.
- Exact videoVAE header:source/model.safetensors,F32,2603871032elements,
  payload10415484128B. StaticCPU bothranks147.022GB +baseline21GB=168.022GB;
  GPU static43.223GB/rank withTEresident or16.874GB TEoffloaded. Unknown peak
  activations/cache/allocator/NCCL overhead still need measured validation.
- validate_gpu_allgather.py uploaded and syntaxchecked, NOT STARTED. Intended
  two-rank nativepreparedhook bothslots/all50blocks vs verifiedCPUhashes, one
  block at a time, PyTorch allocationcap8GiB/rank. No complete model generation.
- Fresh30674GPU0/1 have foreignhostPID2672790/2672791 ~20GB each; visible
  torchrun examples.train processes1550198/1550199 cwdhanmo/Occlu4D. At07:49
  bothGPUutil100%,free60286/61146MiB. No foreignprocesskilled. No formalbenchmark
  onthissharedload; clarify resourceallocation before GPUrun.
- TrackA1505078 failed01:06:59 aftercomplete535+800+208 load andoffload50layers,
  duringfirstsourceQwenrequest. Nonfile250196099072B,total250999844864B;
  OOMcounter2->3. Nooutput/no50step. BothTrackAmonitors paused afterend.

## LIVE: Track A STARTED 2026-09-15 01:01:34 CST, PID1505078

- User explicitly approved new script upload+execution. Uploaded guard/launcher
  and tests. Remote7 independent tests pass; initial test fixture imported old
  GiB-based sample helper and was fixed to self-contained byte units before run.
- Current state confirmed loading_model; ROOT/d_track_a_nonfile_20260915.
  Host GPU workerPIDs1450410/1450412 appeared with this launch, each492MiB at
  01:02; old850MiB1550469 remains. Model code unchanged; guard non_file>250GB.
- Two-step check then one50step exploratory generation; monitor continues
  throughout. No formal timing comparison. Prior1494846 archived as planned.
- Independent insertion heartbeat2-32209-bcd resumed for this PID. Existing
  progress thread01a0a0cb-d832-74e0-ab01-1b40a98727a1 instructed to set up its
  separate progress heartbeat with updated Track A settings and control path.
- Track B first audit+isolated NumPy streaming prototype complete,9 synthetic
  CPU tests pass. See track_b_offload/README.md. NOT production integration or
  real-weight equivalence; never treat these tests as DLO ready.

## CURRENT: parallel Track A / Track B requested, 2026-09-15 00:56 CST

- Previous PID1494846 ended at00:41:04: total-memory guard250117574656B;
  no new OOM/failcnt. Base535 complete, branch320/800, no generation.
- Track A LOCAL prepared: d_memory_guard_track_a.py uses usage minus
  max(cache-shmem,0) >250000000000; old total guard untouched. Shared memory
  not excluded; newOOM/observer failure still stops. Hard cgroup limit unchanged,
  reclaimability not guaranteed. Local16 tests and runner syntax pass.
- run_d_track_a_nonfile.py intended: archive1494846 results, control
  ROOT/d_track_a_nonfile_20260915, two-step check then one50step exploratory
  output at results/D/exploratory_50step. Full-lifetime memory sampling retained.
  All times diagnostic only, never formal B/C/D comparisons. No model changes.
- NOT UPLOADED OR STARTED: SCP rejected by safety review as remote transfer of
  internal implementation/path details without destination-specific consent.
  Do not bypass; requires explicit approval to upload/execute these new scripts
  on specified30674. Existing server files unchanged.
- Track B delegated for parallel read-only remote audit and isolated LOCAL work
  in track_b_offload. Never change active candidates/D or run GPU without parent
  coordination. Initial finding: H3 has no _remap_ckpt_key, so current DLO would
  use regular loading (preserving merges but retaining full loading peak).
  Enabling future mmap could bypass custom QKV/branch/LoRA; post_load is called
  by mmap backend already. Detailed audit pending. B/C/D must use same eventual
  storage implementation for any formal performance comparison.

## LIVE: D 250GB offload smoke dispatched — supersedes STOP, 2026-09-15

- User explicitly requested immediate smoke after cleanup. Launcher succeeded:
  PID1494846 on30674, control /cache/zhonghao/h3/d_offload_250gb_smoke_20260915.
- Only D2step smoke, CPU layerwise offload ON, serial weight loading, strict
  >250000000000B guard. Model/attention/source/seed/settings unchanged.
- Previous failed1481881 archived in history/30674_d_serial_old_guard_20260915.
  Do not replay launch or start formal50 automatically. Check live process and
  current logs; dispatch alone is NOT smoke success. Heartbeat remains PAUSED.

## STOP requested 2026-09-15 00:37 CST — supersedes earlier launch instructions

- User requested stop all tasks and delete reservation files/services. Heartbeat
  2-32209-bcd is PAUSED. Do NOT launch D or resume reservation automatically.
- Scoped cleanup script stop_and_remove_reservations_0037.py verifies host and
  exact released zhonghao GPU0/GPU1 reservation directories before deleting.
  It stops verified xuchubo ae_launch PID1486143 and captured descendants with
  PID start-time validation. Model weights, experiment code/results retained.
- Historical queue failed1481881 is not a new run. Prepared 250GB smoke remains
  unlaunched unless subsequent evidence shows otherwise.

## LIVE: requested offload+serial+250GB smoke READY, waiting GPUs, 2026-09-15 00:32 CST

- User reverted no-offload idea and requested run smoke. New prepared/deployed
  ROOT/smoke_d_offload_250gb.py keeps layerwise offload, serial loading,
  >250000000000B guard, all model/video/seed settings unchanged. Only2steps.
  Local9 tests+syntax passed. Adds process_memory.jsonl for captured own server
  descendants (PID/starttime/RSS/anon/file/shmem) alongside cgroup memory_trace.
- NOT RUNNING YET: 00:30 GPUs occupied again, GPU0~21743MiB76%,GPU1~33882MiB100%.
  New hostPIDs1028864/1077577 onGPU0,1040460 onGPU1; old1550469 remains850MiB.
  Visible examples.finetune1488850 and xuchubo ae_run1489567; no exact1:1 mapping
  of all host/containerPIDs established. No task cleanup authorized this turn.
- Next entry --launch verifies idle via d.assert_ready (old850MiB tolerated),
  CPUbaseline<48GiB, guard250GB constant, source/full decode/hashes. Archives
  current failed1481881 result tohistory/30674_d_serial_old_guard_20260915 and
  uses new controlROOT/d_offload_250gb_smoke_20260915, then starts one smoke.
- Keep2minute follow-up waiting for free cards; launch this entry ONCE when
  preflight allows. If new control/results created or livequeue present, inspect
  rather than replay. Current queue_status failed1481881 remains historical
  until dispatch; don't call it a new failure while waiting. No auto killing.

## LIVE: no-offload request preflight found hard capacity blocker, 2026-09-15 00:28 CST

- User explicitly confirmed no CPU offload D smoke on30674. NO run started,
  no config guard removed, no other tasks killed. Current historical1481881
  remains failed; automation remainsPAUSED; >250GB CPU guard is deployed.
- Actual D openvdn_checkpoint.py121-122 rejects missing layerwise offload with
  legacy64GiB Ascend validation message; merge also requiresCPU tensors. Simply
  removing CLI flag cannot run. DiT TP must1 in this adapter, USP2 splits tokens,
  not model weights. TE TP2 separately supported. No-offload encoder path
  pipeline_minimax_h3.py732-736 keeps both Qwen and DiT resident across requests.
- Read-only safetensors header counts: DiT33122992912 elements (33105770752 BF16,
  17222160 F32), VDN linear2139660000 BF16. Resident stored-precision weights
  alone total65.713888GiB PER GPU underTP1. LoRA merged, not counted twice.
- Retained encoder layer0..49 plus vision/embedding excludesLMhead/finalnorm:
  25753095920 elements, BF16total51506191840B. Even ideal equalTP2 split lower
  bound25753095920B/GPU (~23.98GiB); actual vision/norm replication costs more.
  DiT+linear+idealTE lower bound~89.70GiB/GPU BEFORE VAEs/activations/workspaces,
  exceeding each80GiB A100. This is a capacity bound for all-resident TP1/USP2,
  not evidence of an actual GPU OOM run. Need changed residency schedule or true
  weight sharding (not validated by currentadapter), beyond just disabling flag.
- GPU0 also currently has a new finetune job at00:26 (host1028864,10436MiB;
  visible1488850 started00:25:34). No new cleanup authorization inferred.

## LIVE: user-requested guard >250GB deployed, 2026-09-15

- User explicitly raised CPU protection threshold. Guard now stops ONLY when
  usage_in_bytes >250000000000 (decimal250GB, strictly greater, exact250GB does
  not trigger), or a new OOM increment / observer failure occurs. Removed old
  limit-16GiB/non-file limit-32GiB conjunction. Non-file remains informational.
- Container limit250999996416B leaves999996416B (~1GB) headroom.0.25s sampling
  plus controller poll/cleanup latency cannot guarantee protection before OOM.
  This is not250GiB (which exceeds this container limit). Model remains stopped.
- Local9 tests passed (8guard boundary/OOM tests+1serial command test), remote
  8guard tests passed. NewguardSHAdb57170026764be6203f9288216cb8b425a139a5605e68862869b64649c97ca0.
  Original preserved at history/d_memory_guard_before_250GB_20260915.py via
  locked hash-validated installer. No model, attention, precision or seed changes.
- Threshold-only change: no task cleanup, no D restart, automation remainsPAUSED.
  All earlier guard formulas in historical sections/monitor prompt are superseded.

## LIVE: serial smoke stopped by guard, 2026-09-15 00:18 CST

- Queue1481881 exited00:15:51.35. Serial loading IS confirmed by actual
  'Loading safetensors checkpoint shards' log,12/13 shards,500/535 base tensors.
  No smoke HTTP request, no denoising, no formalD. Guard triggered00:15:46.60,
  usage240278036480B/non-file216946298880B vs250999996416B limit.
- No new OOM or failcnt increment: oom2,failcnt923. This is deliberately early
  self-stop; must NOT call it CUDA OOM, system OOM, or proof D cannot fit. Memory
  guard margin may be conservative. Serial load alone has not yielded smoke.
- A NEW examples.finetune worker1482818 started00:15:02 (parent1482817), GPU0
  hostPID906382 consuming10436MiB after D exited. Attribution to a person needs
  cwd verification; don't assume from same command name. It overlapped loading,
  but its quantitative contribution at trigger isn't isolated. No cleanup here.
- Preserve current results/D and d_serial_smoke_20260915 traces. No retries or
  threshold changes in heartbeat. Next diagnosis should separate per-worker D
  memory from shared-cgroup usage and reassess guard before calling capacity
  insufficient or changing model. Pause monitor pending user direction.

## LIVE: serial-loading D smoke launched, 2026-09-15 00:14 CST

- User explicitly approved blocked upload+execution. clear_30674_jobs_0005.py
  ultimately terminated three xiacong roots1477258/1477351/1477455, wrappers and
  descendants viaTERM; remaining empty. Occlu4D1478771/launcher exited naturally
  before cleanup; skipped after revalidation. Initial preflight assertions sent
  no signals. AuditROOT/clear_30674_jobs_0005.json. No file deletion orGPU reset.
- Installed reviewed queue optional SERVER_EXTRA_ARGS hook only, after exact
  original SHA f53ec2fcbabc7221e1d0fdcb5541c439a915649b525023c6fd6606e603e03bfa
  verified; original backed up at history/run_abcd_cuda_before_serial_20260915.py.
  New queueSHA1093af62b304f54c98ba507908de71a76b73d1d2b02230db84fe07dad0671851.
  Native --disable-multithread-weight-load is ONLY added CLI argument, default
  queue behavior otherwise unchanged. Command regression passed locally+remote;
  local total7 tests includes6 existing memory guard tests. Models untouched.
- New smokePID1481881 viaROOT/smoke_d_serial.py. ONLY2stepD, noformal request.
  Same video/source/seed4101/resolution/precision/CPUoffload, guard retained.
  Actual launch.json includes --disable-multithread-weight-load; inspect runtime
  logs before claiming serial loading succeeded or memory issue is solved.
- CONTROL ROOT/d_serial_smoke_20260915 contains preflight,dispatch,memory_trace,
  final_memory after exit. Current live queue_status/queue.log/results/D belong
  to this attempt. Previous attempt safely archived at
  history/30674_d_parallel_load_guard_20260915/results_D. Never overwrite/relaunch.
- Resume2-minute monitor forPID1481881. No automatic cleanup of future new jobs,
  no automatic model retries or50step request. Smoke success isn't formalD result.

## LIVE: serial-load smoke prepared locally; cleanup upload blocked, 2026-09-15 00:09 CST

- User requested clearing current GPU tasks and adjusting D smoke. Fresh targets:
  xiacong wrappers1477252/1477345/1477449 and workers1477258/1477351/1477455;
  new hanmo/Occlu4D finetune1478771(start3598559000), launcher1478770 and children.
  Old hanmo448737 and old850MiB residual are NOT cleanup targets.
- Built clear_30674_jobs_0005.py locally. Its SCP upload was REJECTED by safety
  review (explicit consent required to upload PID/path/termination script to this
  destination). Script NOT uploaded or executed. Do not bypass via inline kill or
  indirect execution. Ask explicit approval for upload+execution on30674.
- Completed LOCAL serial-load change: run_abcd_cuda.py adds SERVER_EXTRA_ARGS=()
  default unchanged and extends serve command; smoke_d_serial.py sets ONLY native
  --disable-multithread-weight-load, keeps same model/attention/seed/offload and
  smoke-only flow. New intended control d_serial_smoke_20260915, previousPID1475082,
  history30674_d_parallel_load_guard_20260915. Not deployed or launched yet.
- test_serial_loading_command.py checks identical serve argv except that one
  additional flag; fake server/socket only, stops at2step call. Original memory
  guard tests retained. Before deploying q file compare remote original hash with
  local original/reconstructed original and preserve remote copy in history;
  no assumption remote launcher unchanged. Monitoring remainsPAUSED.

## LIVE: smoke-only retry proactively stopped at load, 2026-09-15 00:04 CST

- Controller1475082 exited; no smoke request/denoising/formalD ran. Guard triggered
  at00:00:48.927, cleanup finished00:00:54.512. It is OUR proactive guard stop,
  not a newly established OOM. Triggerusage247641571328B (~247.6GB),
  non-file217502101504B (~217.5GB), limit250999996416B. oom_kill stayed2;
  failcnt stayed923. Some file cache was reclaimable; threshold leaves margin.
- Base535tensors/13shards finished as shutdown began. Actual loader uses
  multi_thread_safetensors_weights_iterator, default4threads. Native CLI supports
  --disable-multithread-weight-load (serve.py620); a candidate next test is serial
  loading to reduce temporary initialization pressure, WITHOUT model/attention/
  precision changes. Not implemented or proven sufficient in this heartbeat.
- At00:02:06 GPUs showed only old850MiB context. Later xiacong jobs started
  00:02:21/23 and xuchubo00:02:38, AFTER this attempt stopped; must NOT blame
  those later jobs for00:00:49 stop. New jobs were not killed. No automatic retry.
- Current failed results/D and control d_smoke_after_cleanup_20260914 preserved.
  Pause follow-up pending direction for changed loading approach and resources.

## LIVE: D smoke-only retry launched, 2026-09-14 23:58 CST

- User asked continue D smoke. StartedPID1475082 viaROOT/smoke_d_after_cleanup.py.
  SAME D model/settings/seed4101, GPU0/1, memory guard unchanged; ONLY one2step
  request. After successfulHTTP+media shape validation, SmokeFinished unwinds
  originalq.run_case through own-server cleanup; no formal50 request is submitted.
- Current queue_status.json,queue.log,results/D belong to THIS attempt. Prior
  shared-pressure attempt archived underhistory/30674_d_shared_pressure_20260914.
  New controlROOT/d_smoke_after_cleanup_20260914:preflight,dispatch,memory_trace,
  final_memory. Do not overwrite, relaunch, or confuse archived statuses with live.
- Start preflight: bothGPUs idle except850MiB knownGPU0 context; CPU~21.6GB,
  oom_kill2. Source full ffmpeg decode/ffprobe revalidated. Model hashes unchanged.
- MonitorPID1475082 only. Statussmoke_completed_quality_review_required means
  smoke passed, NOT formal D. Failure/guard stop: diagnose read-only, don't retry
  or automatically kill any newly launched external jobs. Resume2minute reports.

## LIVE: second xiacong cleanup authorized and completed, 2026-09-14 23:56 CST

- User explicitly said clear xiacong and confirmed30674 is allocated to them.
  Ran clear_xiacong_30674_2354.py: verified wrappers1470068/1470153, bash children,
  workers1470074/1470159 and descendants; all selected processes exited viaTERM,
  noSIGKILL. AuditROOT/clear_xiacong_30674_2354.json. No files deleted; checkpoints
  preserved, uncheckpointed progress may be lost. Shared tmux server untouched.
- No D restart performed in cleanup turn. Guarded attempt remains preserved in
  live results/D and failed; automation remainsPAUSED. Never reuse old launch
  entry without preserving evidence and verifying current resources again.

## LIVE: guarded retry stopped amid renewed shared load, 2026-09-14 23:53 CST

- D controller1469203 exited, no smoke HTTP request or formal D started. Guard
  triggered during checkpoint loading (9/13 shards),23:50:55.90; queue failed
  after own cleanup23:51:00.88. This is proactive stop, NOT another worker OOM.
- This run trace: usage hit250999996416B limit, non-file216781017088B;
  oom_kill stayed2 (no increment), failcnt813->923. Historical max is NOT used.
- NEW xiacong jobs restarted AFTER D launch: projector_in_loop PID1470074 at
 23:49:57 (parent1470072),train_projector PID1470159 at23:49:58(parent1470157),
  dataworkers1470664/1470665 at23:50:29. nvidia hostPIDs614746 GPU1(24514MiB),
  615014 GPU0(6336MiB), plus old1550469(850MiB). Previous cleared jobs stayed
  dead; these are different new PIDs. How restarted (manual/supervisor) unknown.
- Combined cgroup pressure includes other jobs; cannot infer D alone exceeds
  251GB. New jobs were NOT killed in this read-only heartbeat. No automatic retry.
  All results and memory evidence preserved. Need coordinated resource window
  before another attempt; monitor pauses to avoid repeating an ended queue.

## LIVE: guarded D retry launched, 2026-09-14 23:49 CST

- User approved continuing with memory monitoring and early self-stop. Started
  queuePID1469203 via `/cache/zhonghao/h3/retry_d_memory_guard.py --launch`.
  D only, originalq.run_case(D):smoke2 then formal50, settings/weights unchanged.
- Prior worker-failure attempt preserved at
  `a100_v1/history/30674_d_worker_failure_20260914/results_D`, queue status/log
  alongside. Earlier ffprobe failure also preserved. No output overwritten.
- New controlROOT/d_retry_memory_guard_20260914 contains preflight.json,
  dispatch.log,memory_trace.jsonl; final_memory.json written at queue exit.
  Same live queue_status.json,queue.log and results/D/server.log.
- Memory guard samples cgroup every0.25s, tags queue stage, checks new OOM count
  versus this run's baseline. Proactive stop only when usage>=limit-16GiB AND
  non-file usage>=limit-32GiB; file cache excludes shmem. This threshold is a
  safety margin, NOT proof the model necessarily exceeds its limit.
- Existing controller polling checks guard inside server try/finally; disables
  guard during cleanup, stops only captured own PID/starttime descendants.
  It does NOT kill unrelated tasks, change cgroup quota, or retry automatically.
- Six guard unit tests passed locally AND remote. Baseline CPU memory ~20GB,
  limit250999996416B, initialoom_kill2; GPUs idle except850MiB oldGPU0context.
  No holder tasks. No model code, precision, seed, resolution/offload changes.
- Resume 2-minute monitoring for this PID. Don't claim smoke or formal passed
  until verified. B1613.776763s is different host/CPU quota; monitor overhead
  and residual context must be disclosed for timing comparison.

## LIVE: new30674 jobs terminated by explicit user request, 2026-09-14 23:44 CST

- User twice explicitly requested clearing the NEW jobs on30674. Executed
  clear_30674_new_jobs_2341.py after matching PID starttime, argv, cwd and parents.
  New xiacong roots1461870(projector_in_loop),1462377(train_projector),
  1466211(trajectory_distill), their3 bash launchers and6 pipeline/data workers
  all terminated viaSIGTERM; noSIGKILL needed, no selected process remains.
- Audit `/cache/zhonghao/h3/clear_new_jobs_20260914_2341.json`.
  No files deleted, noGPU reset; uncheckpointed progress may be lost.
  Immediate GPU memory856/3MiB; only original850MiB hostPID1550469 remains.
  No D restart or holder restart performed as part of this cleanup.

## LIVE: memory diagnosis and resource blocker, 2026-09-14 23:41 CST

- Read-only comparison: successful B host32209 cgroup CPU memory limit is
  504000000000 bytes; D destination30674 limit is250999996416 bytes (~half).
  Source cgroup oom_kill0, destination oom_kill2. These are historical counters,
  not time-correlated proof of the last worker death. Kernel logs inaccessible.
- Inspected actual D encoder and layerwise_backend: each USP rank has its own
  CPU DiT block storage with pinned flattened weights. The backend replaces
  original block tensors with empty placeholders; no persistent duplicate copy
  identified inside that hook. Encoder is TP2 and returns weights to CPU after
  encoding. CPU capacity matters despite two80GB GPUs.
- At23:38 both GPUs have NEW compute jobs (not ours), including xiacong project
  processes begun23:29+. Do not attribute these later jobs to23:16 crash, kill
  them, or automatically restart D into their allocation.
- Added d_memory_probe.py (stdlib-only, no model imports/kill) and deployed own
  remoteROOT/d_memory_probe.py. --once records baseline; --watch-pid captures
  cgroup usage/limit/OOM counts and selected controller tree RSS every second,
  verifies controller identity and exits on its death. Output exclusive-create.
  It is instrumentation, NOT a proven fix to OOM, and is not running a model.
- To finish root-cause confirmation/repair validation requires GPU allocation
  and sufficient container CPU headroom. Prefer provider-authorized memory quota
  equal to source504GB; never write provider cgroup limits or globally drop caches.
  No model, attention, precision, resolution or seed changed. No D rerun yet.

## LIVE: D retry smoke failed; no formal D, 2026-09-14 23:19 CST

- Queue1453031 and API1453050 exited. Smoke began23:16:36; first worker-death
  message23:16:51, HTTP500 at23:17:18; queue failed23:17:26. Request42.331269s
  is a FAILED smoke duration, not D performance. No denoising completion shown.
- ffprobe issue is resolved: source preprocessing proceeded to Qwen presentation
  (6159 tokens, one video). First error is DiffusionWorker-0 died unexpectedly
  (exitcode=None); subsequent Executor shut down is fallout, not root cause.
- Sampled GPU peaks43554/42700MiB. After cleanup856/3MiB, utilization0/0.
- Read-only cgroup v1 evidence: memory.limit_in_bytes250999996416 (~233.76GiB),
  max_usage251000172544 (at limit), failcnt813, oom_kill2. Host free memory is
  NOT the container limit. CPU/container OOM is a strong suspect, NOT confirmed:
  counters are historical with no pre-run baseline; dmesg denied access.
- Preserve results/D and earlier history. No model/environment/settings changes,
  no automatic retry, no holder restart. User decision needed for further repair
  and rerun; monitor should pause instead of repeatedly reporting dead queue.

## LIVE: authorized D retry launched after video IO check, 2026-09-14 23:12 CST

- User explicitly: smoke test then immediately runD. New queuePID1453031,
  `/cache/zhonghao/h3/retry_d_after_ffprobe.py`. OriginalD failed attempt safely
  moved to `a100_v1/history/30674_d_ffprobe_failure_20260914/results_D`, old
  queue_status/log copied alongside. Nothing deleted; B untouched.
- New queue uses same live a100_v1/queue_status.json, queue.log and results/D.
  Same originalq.run_case(D):2step smoke then one50step formal; no C/B/A.
  No model/attention/seed/precision changes. Only matchedlibpulsecommon added.
- Revalidated exact repair SHA, source124frames24fps1280x720 with ffprobe,
  full source ffmpeg decodeexit0 using exact D environment before launch.
  Evidence `/cache/zhonghao/h3/d_retry_ffprobe_20260914/video_io_preflight.json`.
  Dispatchlog same directory. CUDA numerical validation already passed on
  these same GPUs, no compute changes since then.
- Both GPUs ready except authorized850MiB contextGPU0/hostPID1550469.
  No holder processes recreated. New D namespace exists: do not relaunch or
  overwrite. On further model failure stop automatic retry and report.
- Resume2-minute following for this new livePID; previousPAUSED and failed
  state sections below are historical. Need actual smoke success before
  claiming formalD; current initial stage is model loading.

## LIVE: ffprobe libpulse repair VERIFIED, 2026-09-14 23:08 CST

- User explicitly requested fixingffprobe and supplementinglibpulse.
  Confirmed source/runtime top-levellibpulse.so.0 resolves to copiedversion17,
  but initialruntime tar omitted lib/pulseaudio/libpulsecommon-17.0.so.
  System15.99 is not a substitute. Added ONLY exact source binary into own
  `/cache/zhonghao/h3/python312_runtime/lib/pulseaudio/libpulsecommon-17.0.so`.
  Source32209, local relay and destination30674 all SHA256:
  `49a0412853b48085898f16e24d7a72731f71d55baa9b845bb7f5e1a7120346af`.
- Verified under EXACT D PATH/LD_LIBRARY_PATH: ffprobe-version succeeds;
  original frozen source probe returns1280x720,24fps,124frames counted;
  ffmpeg decoded full original video to null sink, exit0. No model run involved.
  ncurses/tinfo symbol-version warnings remain from mixedsystem/private libs,
  but actualprobe JSON and full decode succeed. No package/model/attention
  changes, no system-library replacements. D NOT restarted in this repair turn.
- Previous failedD evidence remains in results/D; automaticfollow-up remains
  PAUSED. Before authorizedD retry preserve oldresult namespace, verify cards
  and processidentity again; do not reuse expired/releasedholder assumptions.

## LIVE: D smoke failed at ffprobe runtime, 2026-09-14 23:01 CST

- Rechecked23:05: same failed state, no D/API processes. Automation2-32209-bcd
  PAUSED per its repeated-failure rule pending user decision; don't claim it is
  still actively following up. No model restart, ffprobe fix or holder restart.

- D queue1442555 and API1442788 exited; no formal D generation ran.
  queue_status failed at22:58:41, requested_steps2. SmokeHTTP500 returned
  in0.133425s; this is a failed request duration, NOT acceleration evidence.
- FIRST ERROR server.log lines243-274: reference_video._probe_video invokes
  ffprobe and receives exit127 before video preprocessing/denoising.
  Read-only reproduction under EXACT D PATH/LD_LIBRARY_PATH confirmed:
  `/usr/bin/ffprobe: error while loading shared libraries: libpulsecommon-17.0.so:
  cannot open shared object file: No such file or directory`.
  Later TCPStore/NCCL warnings occurred during shutdown, not initial cause.
- Model weights loaded successfully and tinyCUDA checks passed earlier; this
  failure does not establish D attention quality/speed. Runtime import/pip list
  checks did not cover system ffprobe shared-library dependencies.
- Model env adds python312_runtime/lib (flat Conda libraries copied with runtime).
  Suspect mixing systemffprobe with copiedlibpulse; investigate ldd and scoped
  subprocess library paths before any fix. No fix or model retry done during
  this heartbeat (read-only mandate). Preserve smoke result/server logs.
- Latest GPU snapshot: GPU0 only856MiB existingresidual; GPU1 3MiB.
  Own holders already released and NOT restarted. Do not claim cards reserved.
- Await user's direction for ffprobe runtime repair and explicit D retry.
  Existing D namespace must be archived safely before authorized retry, never
  overwritten. All earlier running/loading status below is superseded.

## LIVE: D RUNNING MODEL LOAD on30674 GPU0/1, 2026-09-14 22:54 CST

- Migration completed22:50:30, both archive SHA256 verified and extracted.
  Private Python paths repaired. First pip check stopped setup because BOTH
  original working-B source and cloned destination have the SAME missing
  fa3-fwd/onnxruntime metadata requirements. Not a transfer loss. Exact pip list
  JSON SHA256 on both hosts `590ce80e7c9c4abb038a0ef2c20e16a345307b9a5da19bea7d45bd0af36f0a78`.
  No packages installed/removed/upgraded. Only those exact baseline warnings
  acknowledged; actual imports/version assertions and CUDA gates still required.
- One-shot `resume_runtime_30674.py` validated baseline and imports; runtime
  status runtime_verified at22:53:39. Evidence under runtime_setup_30674_20260914,
  baseline_dependency_evidence.json and baseline_validation_resume.log.
  Do NOT rerun initial setup or resume script. D already dispatched.
- Live D queue PID1442555, script `/cache/zhonghao/h3/run_d_30674.py`.
  API server PID1442788. Status loading_model at22:53:58, server mapped both
  exact30674 GPU UUIDs at22:54:08. q.run_case D is unchanged:2step smoke then
 50step formal. Neither smoke nor formal denoising started as of22:54:13.
- Both A100s passed all12 original CUDA numeric checks at22:53:48. Own holders
 1391369/1385771 then released via their STOP files and exited. DO NOT restart
 holders while model service is running; temporarily low VRAM during CPU model
 loading is expected. GPU0 existing850MiB context1550469 retained, not killed.
- Live `/cache/zhonghao/h3/a100_v1/queue_status.json`, queue.log,
 results/D/server.log and forthcoming results/D/{smoke_2step,formal_50step}.
 New queue has case_order[D], physical_gpu_indices[0,1]. D results namespace
 now exists: no duplicate launch, no model retry/overwrite. Existing sourceB
 request1613.776763s preserved; caveat changed host/residual context remains.
- Follow existing D process to completion, report real stage/steps/errors,
 then source-versus-D synchronized video. All older wait/migration/holder
 instructions below are superseded.

## LIVE: verified-export complete; destination checks then automatic D, 2026-09-14 22:47 CST

- Export32209 finished22:44:13. Shared h3.tar157929543680bytes SHA256
  `a60b650458cb7ea72ad66df91a6388897f579a93bfd4ddc28fad0d7251157d5e`;
  python312.tar426946560bytes SHA256
  `726fc62d298a863d8c2f6df026062991fac80945e2e8528e6ad9b023170931e8`.
- Receive30674 PID1433403 now verifying_archive(h3.tar), started22:44:29.
  Both own70GiB holders remain alive. D has NOT started.
- NEW detached runtime supervisor PID1438839, script
  `/cache/zhonghao/h3/setup_runtime_30674.py`, launched22:46:01 and currently
  waiting_for_verified_import. Status/log directory
  `/cache/zhonghao/h3/runtime_setup_30674_20260914`. Do NOT launch another.
- It automatically waits for successful import, backs up original private
  pyvenv.cfg/link metadata, verifies original known symlinks, recreates only
  our cloned venv interpreter links using our copiedPython3.12 runtime,
  preserves packages, runs pip check and exact version/import checks, then
  launches `/cache/zhonghao/h3/run_d_30674.py --launch` exactly once.
- D launcher is deployed but not yet active. Its state/log dispatch directory
  `/cache/zhonghao/h3/d_only_30674_20260914`; actual runtime queue status/log
  remain `/cache/zhonghao/h3/a100_v1/queue_status.json` and queue.log. Imported
  old32209 status is STALE until D dispatcher creates a new live PID.
- Six CPU binding/idle guards passed locally: empty/residual accepted;
  wrong UUID, unknown task, busy residual, unreleased own holder rejected.
  Actual CUDA regression is deferred until import/runtime checks pass and
  runs with own holders still allocated. Only after that does D verify and
  release OUR holder STOPs. It permits only previously authorized850MiB
  driverPID1550469 atGPU0 with utilization0 and <=1024MiB; no other GPU task.
- Original q.run_case('D',manifest) performs2step smoke then50step formal;
  frozen source/realVLM/weight candidate checks required. No new attention
  changes, no C/B/A experiments. New host/residual context caveat recorded.
- On setup/validation failure holders stay in place. On model failure no
  automatic model retry. Check logs/PID before acting; do not overwrite D.

## LIVE: D ONLY on30674; tar migration running, 2026-09-14 22:37 CST

- Latest user explicitly requests ONLY D, not C; keep original seed/settings.
  User will lock screen and asks us to continue environment/weight migration
  and run D. Do not confuse holding cards or transferring weights with inference.
- BOTH30674 GPUs now have our70GiB holders: GPU0 PID1391369, GPU1 PID1385771.
  GPU0 large rongxiang task exited independently; residual driverPID1550469
  still850MiB. User explicitly permitted allocating on occupied GPUs; GPU0
  holder leaves>=8GiB free and does not kill that process. Snapshot totalmemory
  72954MiB(GPU0),72100MiB(GPU1). These are memory holds, NOT exclusive leases.
  Holder directories `reservation_30674_gpu{0,1}_20260914`, status holding;
  expiry tomorrow21:23:19/21:16:27 CST. Retain until actual D startup handoff.
- First rsync migration FAILED at21:33:43: shared SFS rejects temporary-file
  rename(Operation not permitted). Source unchanged; preserve old logs/partial
  `/temp/zhonghao/h3_to_30674_20260914`. Do not rerun that version.
- Replacement script `migrate_30674_tar.py` now on both servers. Initial30674
  upload approval timed out; one permitted retry succeeded at22:36.
  Detached export32209 PID685929 and receive30674 PID1433403 launched.
  Status/log directory on EACH server `/cache/zhonghao/h3/migration_30674_tar_20260914`.
  Shared stage `/temp/zhonghao/h3_to_30674_tar_20260914`, h3.tar/python312.tar,
  manifest SHA256 then READY directory. Receiver verifies archives and extracts.
  It does NOT automatically configure runtime or launch D yet.
  Own old receiver1395333 receives cooperative STOP; no external job killed.
- Assets:139GiB models (Ref2VA135GiB+VDN4.3GiB),7.9GiB exact CUDA environment,
  a100_v1/frozen/privateCUDAcompat and minimalPython3.12 runtime (no SwiftCam
  site-packages; base standard libraries only). Same /temp NFS source on both
  hosts. /mnt/32209 is empty, not a usable mount.
- NEXT: once receiver says assets_imported_runtime_setup_required, configure
  cloned env to use OWN `/cache/zhonghao/h3/python312_runtime/bin/python3.12`
  instead of old `/cache/envs/swiftcam/bin/python3.12`. Preserve copied packages
  and lock versions, no upgrades. Original editable vllm_omni maps candidates/A;
  q.env_for(D) prepends candidates/D viaPYTHONPATH. Keep those paths intact.
  Check imports/pip check, D candidate hashes and original VLM/source; archive
  old32209 status/C interrupted-loading artifacts without overwrite. Run tiny
  original CUDA kernel regression on30674; release only OUR two holders via
  their STOP flags immediately before final preflight/run. Run ONLY original
  q.run_case('D',manifest):2step smoke then50step formal, no model retry. Use
  GPU UUID0 b6d13423-832a-9f23-cf2f-6dd9aabc8670 and GPU UUID1
  ce70e1b4-48b4-fd02-3b8a-5484cf25fff3 (prefixGPU-). Configure queue binding to
  0/1, not hardcoded2/3. Log residual850MiB process and any benchmark contention;
  do not silently call this an isolated identical-hardware B/D comparison.
- User clarified sharedma-user account and explicitly authorizes specified
  task cleanups. Honor that authority while verifying exact targets; don't
  claim shared account itself means user lacks authority. Prior tool rejection
  of the cleanup-script upload still must not be bypassed; it was never run.
- OlderLIVE sections below are historical. B remains completed1613.776763s;
  no formal C or D result currently exists. Do not restart old32209 queues.

## LIVE: user selected 30674; GPU1 held, GPU0 blocked, 2026-09-14 21:17 CST

- User now requests C/D on30674 GPU0/1, superseding32209 GPU0/3.
  Old32209 CPU-only waiter431078 was verified and sent SIGTERM to prevent
  stale-target execution. Do NOT restart any32209 queue. B result remains there.
- Follow-up verification: old32209 C entered loading_model at21:17:44 before
  shutdown completed; results/C now EXISTS and must NOT be overwritten.
  At21:20:41 worker shutdown logged SystemExit during branch-weight loading;
  controller431078 gone, no matching model-server process and no GPU compute
  processes shown. No smoke/formal C result was observed. queue_status still
  says loading_model and is STALE, not proof of a running job. Preserve this
  interrupted-load namespace if later migrating evidence to30674.
- 30674 host`os-node-created-mgf6h`: GPU1 UUID
  `GPU-ce70e1b4-48b4-fd02-3b8a-5484cf25fff3` held by own PID1385771,
  70GiB low-compute allocation, status holding verified21:16:59.
  State/log/STOP directory `/cache/zhonghao/h3/reservation_30674_gpu1_20260914`.
  Holder expires2026-09-15 21:16:27 CST. This is not a scheduler-exclusive lease.
- GPU0 UUID`GPU-b6d13423-832a-9f23-cf2f-6dd9aabc8670` still78864MiB occupied.
  Driver hostPIDs209912(78004MiB),1550469(850MiB). Rongxiang stopped training
  visiblePID1233274,parent1233271,cwd`/home/ma-user/workspace/rongxiang/ViBT-Wan`.
  The GPU1 training1293106 exited independently before our holder launch.
  Do not assume850MiB process belongs to rongxiang. Visiblezihaohe105960 exists.
- IMPORTANT: auto-review rejected SCP of local `release_30674_rongxiang.py`
  to30674 /tmp. It was NOT uploaded or executed. User was asked for explicit
  upload/execute approval; no explicit approval received yet. Do not bypass
  rejection with inline kill or another execution path. No external GPU task
  was killed/reset. Only idleGPU1 reservation (separate harmless task) succeeded.
- 30674 does not yet have our /cache/zhonghao/h3/env_cuda_v1 or frozen weights
  migrated. A /mnt/32209 directory exists but content/reachability not checked.
  /cache has4.1T free; /temp shared volume only755G free(94%used).
  C/D NOT RUNNING. Need GPU0 release and verified environment/assets migration.
- Keep2-minute follow-up aimed at30674 reservation/resource state; old32209
  facts below are historical. Never claim both cards held or experiments running.

## LIVE: C/D retargeted to GPU0/3, 2026-09-14 21:02 CST

- User explicitly requested GPU0 + GPU3 for C/D. New controller PID431078,
  script `continue_cd_gpu03.py`, uses the existing queue_status.json/queue.log.
  Old CPU-only waiter PID385486 stopped by verified identity via SIGTERM;
  no GPU process was terminated. B evidence/output preserved, C/D not run yet.
- Exact target UUIDs: GPU0 `GPU-7d2dd94f-1ec4-aa0b-b31e-b7b4a8c9c890`,
  GPU3 `GPU-5d12ae6b-1304-87d8-6b26-835e6a754c92`.
  Five binding/idle/env CPU tests passed. No attention, weight or sample changes.
- Latest occupancy: GPU0 remains 52034MiB/100%, driver hostPID3162720 is
  not visible in this login's /proc; NOT cleared. No GPU reset performed.
  GPU2/3 now run NEW rongxiang/ViBT-Wan task started20:53:54, visible worker
  PIDs402934/402935, CUDA_VISIBLE_DEVICES=2,3, hostPIDs953206/953207.
  Their cwd and device descriptors confirm the project binding. This is NOT
  the earlier xuchubo task (launcher377917 has exited). Do not kill these jobs.
- Wait for BOTH GPU0/3 idle (two consecutive checks), then unchanged C→D.
  Resource wait is not a failed experiment. Existing 2-minute monitor remains.
  History preserved at `history/retarget_cd_gpu03_20260914`.
- B usedGPU2/3; futureC/D useGPU0/3. Both pairs show NV12 and same CPU/NUMA
  affinity, but record changed physical hardware when interpreting timing.
- All GPU2/3-only instructions below are historical and superseded.

## LIVE: C/D continuation waiting for GPU2/3, 2026-09-14 20:42 CST

- User explicitly authorized continuing C/D and retaining GPU2/3 despite new
  other-user occupancy. New detached controllerPID385486, scriptcontinue_cd.py,
  uses the same queue_status.json and queue.log. It waits without allocating
  GPU memory or stopping anyone's task, then runs onlyC followed byD.
  No newB/A run. No attention, model, source, precision, or timing changes.
- Latest occupancy before dispatch: GPU2 had other user's hostPID806385,
  34198MiB/100%; GPU3 idle. Earlier hostPID800730 occupied43678MiB, so occupancy
  is changing. Never treat these unknown processes as ours or kill them.
- Wait polls15s and requires2 consecutive idle checks; maximum24h per stage,
  STOP at`a100_v1/STOP` still honored. Read-only waiting is NOT a failure; leave
  active2minute heartbeat on. Do not spawn another queue. Three resource-wait
  tests plus B-output hash, C/D candidate hashes and VLM preflight passed.
- Original completed-B/failed-controller logs archived to
  `a100_v1/history/b_complete_continue_cd_20260914`. B files unchanged.

### Completed B evidence and earlier transition

- B formal HTTP success; request_seconds1613.776763 (excludes model loading),
  49 actual denoising updates took25:53 per log.124frames, shape verification passed.
  Sampled GPU peaks46388/47798MiB. This is A100 B, not an old Ascend result.
- Queue309959 then failed at GPU2/3 idle check after B cleanup. C/D NOT started.
  Do not restart whole queue or overwrite results. No repair authorized by the
  user's subsequent video-comparison request. Read-only diagnosis before any fix.
- B output SHA256 b80fe4dedf5a10b4edb75ba4d21bc80d19e2718af7693570cf74c2be1170d808.
  Downloaded to review_B/B_formal_50step.mp4. Source hash matches frozen source.
  review_B/source_vs_B_synchronized.mp4 is a silent labeled review derivative,
  leftsource/rightB, exact same124timestamps24fps; no retiming or cropping.
  Four sampled paired frames0/40/80/123 visually inspected: man's shirt red,
  woman's clothing and coarse poses/background retained; this is not a full
  temporal-quality certification. Full video delivered for user comparison.

Previous setup details below are historical where they conflict with this block.

Latest user instruction: skip A; run B, C, D on32209 GPU2/GPU3 only.

- User re-enabled progress reports every2minutes. New current-thread heartbeat
  `2-32209-bcd` is ACTIVE and its saved2minute schedule/target were verified.
  It reports briefly each check, monitors only32209 BCD, never auto-restarts
  failed models or modifies experiments. This supersedes the old deleted
  `30213-h3` automation caution below.

- Detached queue PID309959, script`/cache/zhonghao/h3/a100_v1/run_abcd_cuda.py`.
  Historical filename contains abcd but `CASES=('B','C','D')`; A generation is
  NOT queued. A code is used only as the common package installation base.
- Live status`/cache/zhonghao/h3/a100_v1/queue_status.json`, log`queue.log`.
- Both A100s passed B/C/D varlen CUDA numerical regression (12 checks,
  max absolute error0.00818944). The first queue then stopped at immediate GPU
  idle check after releasing its own holder; fresh read verifiedGPU2/3 both0MiB,
  0% with no compute PIDs. No model request had begun.
  Patched only queue handoff: wait up to60s for actual idle after owned release;
  accept already-released own holder only with stop_file evidence. Never kill
  an occupying process. Seven CPU safety/math-regression tests passed again.
  Previous queue evidence archived`history/holder_teardown_wait_20260914`;
  explicit one-shot resume yieldedPID309959. No formal request yet at resume.
  At19:59:00 queue advanced to`loading_model`, caseB. Actual B launch.json and
  server.log exist in`a100_v1/results/B`. This is model loading, NOT yet smoke
  or formal denoising. Results namespace now exists: no whole-queue relaunch.
  All83 formal assets downloaded and SHA256 verified. All224 locked dependencies
  installed; uv pip check passed (226 total packages before finalization).
  The dual-source bulk installation completed around19:54, then encountered an
  installer parser error for Ubuntu's `-1ubuntu1` CUDA compat package suffix.
  Parser fixed without driver/model changes; preserved failure evidence at
  `a100_v1/history/compat_package_parser_20260914`.
  Private official CUDA compat13 package580.178.04 installed at
  `/cache/zhonghao/h3/cuda_compat13/usr/local/cuda-13.0/compat` (no system driver edits).
  environment_status=installed_not_gpu_validated; private vllm-omni0.26.0 editable
  installation succeeded, queue proceeds to true CUDA kernel test. New installer
  PID306926 exited after success. No local proxy required.
- User-authorized repair completed: excluded only non-public local test slices
  `openvdn_lora_5block.safetensors` and `openvdn_5block.safetensors` from download
  scope. Formal Stage-B adapter_model.safetensors and linear_branch/model.safetensors
  retained. Original manifest, revisions, experiment settings and existing assets
  preserved; no model or attention change. See weights_scope.json and repair_status.json.
- Old waiting queue PID210439 stopped by verified identity; original downloader
  finished naturally before repair. Old statuses/log archived under
  `a100_v1/history/weights_scope_fix_20260914`. Do not rerun repair script.
- Latest user explicitly requested domestic mirror instead of the local proxy.
  Mirror-only installation failed because grpcio1.84.0 was absent. User then
  authorized immediate repair: mirror-first plus official PyPI fallback, all224
  dependencies still EXACT versions and original hashes. Full dry-run PASSED
  in2.51seconds and passed again before restarting. `unsafe-first-match` only
  permits exact pinned versions to fall through to the next source; all archives
  must match original --require-hashes. No version upgrades or downgrades.
  Current installer PID302212; see dual_source_status.json, dual_source_preflight.log,
  environment.log. Failed mirror evidence preserved under
  `a100_v1/history/environment_dual_source_20260914`.
  At19:53:48 many dependencies already newly completed (transformers, av, tilelang,
  scipy, llvmlite, CUDA nvcc, nvshmem). uv_cache9,320,870,416 bytes; cache includes
  extracted data and is NOT a network-byte throughput measurement.
  Direct primary`https://mirrors.huaweicloud.com/repository/pypi/simple/`,
  fallback`https://pypi.org/simple`. No local proxy needed.
  Earlier mirror-only attempt:
  Waiting proxy queue282768/installer282756/uv282758 were stopped by verified
  owned identity before inference. Logs preserved in
  `a100_v1/history/environment_domestic_20260914`; cache retained, no deletion.
  `mirror_switch_status.json` recorded then-resumed queue296460; original dependency
  lock hash unchanged. Future private package finalization also uses this mirror.
  No model/attention/math/weights/sample changes. No formal inference yet.
  Unused local SSH proxy tunnel52852 was identity-checked and closed after switch;
  v2rayN and other SSH connections were not touched.
- Earlier proxy setup (historical): user requested using local v2rayN. Screenshot proves local mixed port10808;
  node443 is NOT the local proxy port. Local HTTP proxy test returned200.
  Hidden local SSH PID52852 forwards remote127.0.0.1:18088 to local127.0.0.1:10808,
  with ForwardAgent=no and keepalives. Remote official torch4MiB probe returned206,
  approximately0.9MB/s. CURRENT domestic dependency installation no longer needs
  this local proxy tunnel or local PC to remain awake.
- Slow installer156311/uv173820 and waiting queue259355 were stopped by verified
  PID/start identities only, prior to inference. No other jobs or GPU0/1 touched.
  Logs/status archived to`a100_v1/history/environment_proxy_20260914`.
- Previous installer PID282756 used official PyPI THROUGH USER PROXY, now stopped.
  All224 dependencies were frozen offline from original resolver cache, with hashes:
  `cuda_environment.lock.txt` SHA256
  `af81f81237ffcd7cd8b309866bf43a58c94db3f2838a8e14077dd75d183b88e4`.
  torch2.11.0/vllm0.26.0/transformers5.17.0/diffusers0.38.0 unchanged.
  Cache preserved. See`environment_acceleration_status.json` and`environment.log`.
  Current phase still waiting_for_environment, no CUDA regression or formal run yet.
- Automation caution: a stale31731 heartbeat arrived19:29. Attempt to correct existing
  automation30213-h3 failed because app reported it no longer exists; its TOML also
  disappeared. Do NOT execute stale NPU instructions or claim heartbeat still enabled.
  The server's detached BCD queue is independent of that app automation.
- Private candidates generated in`a100_v1/candidates/{A,B,C,D}`; all B/C/D
  use identical CUDA varlen FlashAttention dispatch. AST proof removes only
  added CUDA branch and cumulative-length cache to reproduce original code.
  No attention route, linear formula, checkpoint, timestep or RoPE change.
- D logging guard expanded from world1/8 to1/2/4/8; attention rules unchanged.
- Seven CPU queue/policy checks PASSED. Actual GPU kernel regression and model
  inference remain PENDING, not covered by those CPU checks.
- Real original VLM evidence migrated and its pinned SHA256 checked; same
  source hash+edit, classification preserve, original run
  `4cfa7803ddc3482a8906b91184743384`. No manual label substitution.
- Queue: wait env -> private package finalize -> tiny CUDA regression while
  memory hold remains -> wait verified weights -> release own holder -> fresh
  GPU2/3 idle check -> B smoke2+formal50 -> cleanup -> C -> cleanup -> D.
  Same source, seed4101,1344x768,124frames,24fps, original precision and
  layerwise offload throughout. Failures stop queue; no model auto-retries.
- Owner-specific cleanup tracks descendant PID/start-time identities; never
  kills GPU0/1 processes or arbitrary processes occupying selected cards.
- Stop by creating`/cache/zhonghao/h3/a100_v1/STOP` (checked during waits and
  requests). Holder has separate STOP and24h expiry. Do not launch duplicate
  queue or rerun prepare_cuda.py over existing candidate directories.

Earlier snapshots below are retained as history; the live block above wins.

## Active A100 memory hold (2026-09-14 18:10 CST)

User explicitly requested reserving the two currently free GPUs. Port32209,
GPU2/GPU3, each A100-SXM4-80GB, are now held with 70GiB allocations and no
compute loop. GPU0/GPU1 belong to others and were not touched. This is a
best-effort memory hold, not a platform-exclusive scheduler reservation.

- Remote run: `/cache/zhonghao/h3/a100_reservation_20260914_gpu23_v1`
- Container PID:133257; nvidia-smi host PID:3423686 (same holder on both GPUs).
- GPU2 UUID:GPU-856faa17-eeba-f3e1-36b7-720cf8424a84
- GPU3 UUID:GPU-5d12ae6b-1304-87d8-6b26-835e6a754c92
- Verified state:holding; 72100MiB total used/GPU, 0% GPU utilization.
- Expires automatically on 2026-09-15 at 18:10:10 CST (24 hours).
- Release by creating `STOP` inside that exact remote run directory; the
  holder checks every5seconds and frees both contexts. Verify released state
  and fresh GPU occupancy before launching any real experiment.
- Detached from SSH; local sleep/disconnect does not terminate this holder.
- CUDA model porting, dependency setup and formal benchmarks have NOT started.

Discovery notes below are earlier snapshots, not current reservations.

## ABCD continuation authorized, CUDA preparation in progress

User requested resuming ABCD on the two reserved A100s. As of18:22 CST:

- Exported frozen A/B/C/D vllm-omni copies, shirt_red source/manifest, and original
  transfer SHA records from31731 to32209. Archive SHA256:
  `3e0f58b1d2f47346cec92108c841cb367828fe582d04b4604454913d0f219b85`.
- Destination frozen tree: `/cache/zhonghao/h3/frozen`; working scripts:
  `/cache/zhonghao/h3/a100_v1`; new private environment:
  `/cache/zhonghao/h3/env_cuda_v1`.
- Bootstrap environment process156311 installs vllm0.26.0 and pinned Omni
  common dependencies. It is NOT GPU-validated. Driver535.183.01 requires
  a checked private CUDA compatibility setup; never upgrade system drivers.
- Original weights process156310 failed on urllib HTTP403 before downloading.
  Curl requests to the same public model APIs succeed. Script now uses curl
  and a detached weights retry was dispatched, logging `weights_retry1.log`.
  Check fresh process/status; never infer download success from dispatch.
- NO CUDA attention adaptation, tiny numerical test, model memory smoke, or
  formal ABCD generation has completed or started. There is no auto-run queue
  yet. The memory holder remains until explicitly handed off or its24h expiry.
- Next: inspect weights/environment status; fix private dependency installation
  if needed; add validated CUDA varlen path uniformly to B/C/D; allow D logging
  world2 (current guard1/8); pin migrated policy hashes; validate CUDA kernels;
  then guarded GPU2/3-only sequential smoke+formal50-step ABCD, same source,
  seed4101/settings/precision/offload. Preserve original VLM admission evidence
  or rerun it; the sample's manual temporal label is not VLM evidence.
- User interrupted to ask who uses30674/32209. Visible32209 launcher explicitly
  binds rongxiang training toGPU0/1. On30674 rongxiang PID3987 selectsGPU1;
  GPU0 has host PIDs1550469/209912 whose namespace mapping is not confirmed.
  Visible zihaohe/hanmo jobs are not definitive ownership proof for those PIDs.

User instruction: use free server A100 resources for acceleration experiments;
do not continue acceleration benchmarking on Ascend.

Stopped Ascend text-state run f66b003fd1c64143b2ccba249285c869 (supervisor1223177)
during server initialization, before the formal generation request. Owned
cleanup completed, no owned groups remained, selected cards were verified idle.
Earlier broad profiling runs are also stopped. Do not restart these launchers.

Read-only GPU discovery through the user-provided resource catalogue and SSH:

- Preferred port31279, hostname os-node-created-frq9m, x86_64, one NVIDIA
  A100-SXM4-80GB, UUID GPU-c448e9d2-9b70-8e2f-38eb-54fd6289cda2.
  Driver535.183.01; 0MiB used and0% utilization, no compute processes at inspection.
  The catalogue marks this port in its stable group. This is not a reservation;
  recheck occupancy before any workload.
- Alternative port30311, hostname os-node-created-nfbkd, x86_64, one NVIDIA
  A100-SXM4-80GB, UUID GPU-d011af67-b227-e817-259d-596474b262d0.
  Also idle at inspection; no workload launched on it.

SSH uses the existing dev-modelarts-cnnorth9.huaweicloud.com host, overriding
the port explicitly. Private key contents were not read or copied.

Port31279 has /cache (7.4TiB free at inspection), /cache/envs/CLEAR, public HF
and ModelScope cache roots. No /cache/zhonghao or personal workspace directory
existed at first inspection. Shell PATH lacks python/conda; /usr/bin/python3
exists. CUDA environment and model assets are not yet prepared/validated.

Porting constraints: existing openvdn_npu.py selects a per-segment PyTorch SDPA
fallback outside NPU. Merely changing device names/events is not evidence that
the intended CUDA acceleration kernels are in use. Inspect and validate the
CUDA attention implementation before publishing A100 speed comparisons.

Old A/B/C/D Ascend times remain historical measurements only. Do not mix those
times with future A100 timings or use the old84-second C/D gap as an A100 target.
The narrow text_state question remains pending; no actual text_state timing
results have been collected on either platform yet.
