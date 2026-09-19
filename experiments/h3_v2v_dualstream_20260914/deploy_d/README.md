# D inference candidate — not launched by this implementation

This is a new, isolated controller for the explicitly approved **D** experiment.
It does not modify or launch the old A/B/C controllers, does not use S0, and does
not train. The candidate is expected to preserve C's target→target and
target→source computations, and add the original VDN local+linear treatment to
source visual queries. Auxiliary text/audio visibility remains unchanged; an
independent source encoder or reusable source cache is **not** claimed.

## Fixed runtime contract

- Only host 31731, physical cards 0–7, `shirt_red_couple_124`, original raw source,
  unchanged red-shirt instruction, seed 4101 and old 1344×768 / 24 fps settings.
- Released Stage-B checkpoint, USP8, ring1, DiT TP1, text TP8 and VAE patch8.
- New vendor: `/cache/zhonghao/h3/dualstream_v1/candidate/vllm-omni`.
- New code: `/cache/zhonghao/h3/dualstream_v1/deploy_d`.
- New results: `/cache/zhonghao/h3/dualstream_v1/results/01234567/shirt_red_couple_124/D`.
- Public case/mode: `D` / `D_source_hybrid_same_frame_v1`; HTTP port 19101.
- Exactly one two-step smoke, then at most one 50-step formal request in the
  same service. A durable create-only `submission_20260914.json` prevents
  duplicate submissions, including a second UUID after failure. There is no
  automated retry or reset/delete mechanism.

`reviewed_policy.json` is deliberately absent. Invocation requires its explicit
SHA256; final review must bind all eight candidate files, the execution parser,
validation pins and an actual completed D tiny run. Old C/B evidence and
`latest` are not accepted. See `d_contract.POLICY_KEYS` and
`d_execution_verifier.EXPECTED_EXECUTION_CONTRACT` for the exact schema. The
reviewed policy cannot be written until the real D tiny result exists.

## Validation deployment boundary

`d_validation_verifier.py` is authored here but **must be deployed into**
`/cache/zhonghao/h3/dualstream_v1/validation_d/d_validation_verifier.py`, where
the seven-file validation pin set includes it. Do not import the authoring copy
as the production validator. Its fixed per-run input is:

`/cache/zhonghao/h3/dualstream_v1/validation_results/01234567/runs/<UTC>_<run_id>`

The wrapper binds pins and candidate hashes, checks the actual controller
status/source manifest and the synthetic policy SHA, and calls the pinned
`d_npu_regression.verify_results`. That reader validates the actual numeric
report, separate CPU evidence and eight rank files. The wrapper then checks
original supervisor/worker start identities have exited, the marked owned group
is empty, and the historical raw `npu_after.txt` matches recorded release.
It does not query present NPU idleness, so later full-model inference can safely
revalidate historical evidence while using the cards.

The synthetic validation policy has scope `synthetic_validation_only`. It shares
the exact attention semantics and candidate pins with the later inference
policy, **not the final whole-policy hash**: the final policy contains the tiny
result SHA and therefore cannot predate that result.

## Actual execution evidence

`D_EXECUTION_RECORD` is required from every one of 50 DiT layers and every one
of eight actual worker PIDs after source/target Softmax and both original linear
calls return. The first forward of smoke produces 400 records; smoke plus formal
must produce 800. The parser binds the real module path/hash, policy SHA, mode,
ordinal request/forward/layer/rank, seven local heads, BF16/device and exact
37-frame / 1008-token-per-frame geometry. Worker ranks must agree with actual
selected-card process identities. The smoke prefix is length/SHA bound before
the formal request.

These records prove the instrumented first-forward calls ran. They do not alone
prove numerical equivalence on every step, a measured NFE count, edit quality or
speedup. Successful HTTP/media validation and manual quality review remain
separate requirements. No full D model has been started by this implementation.

## Read-only admission before consuming the single submission

With the private environment active and this directory on `sys.path`, call the
following gate functions directly: `policy_gate`, `source_code_manifest`,
`transfer_gate`, `runtime_gate`, `tiny_gate`, `sample_gate`, `weight_manifest`,
and `parallelism(selected_profile(...))`. Do not construct a trial, call
`acquire`, or create a ledger during this read-only check. Policy and tiny
dependencies must already exist. The supervisor still repeats admission and
fresh resource checks while holding all nine actual leases before launch.

## Local verification performed

`python -B -m unittest discover -s experiments/h3_v2v_dualstream_20260914/deploy_d -p 'test*cpu.py' -v`

54 synthetic CPU tests passed: policy/schema and candidate closure, durable
single submission, D log evidence, namespace isolation, and read-only tiny
admission orchestration with explicitly mocked numeric readers. The fixture
files are **not actual VLM/NPU/model evidence**. Full supervisor process behavior,
Linux shell syntax, actual eight-card numeric execution and model-quality/speed
measurements remain outside these local tests. Frozen inherited lifecycle
helpers are retained rather than rewritten.
