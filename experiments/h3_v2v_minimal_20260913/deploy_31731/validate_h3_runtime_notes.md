# Private 31731 CPU runtime validation

Run only after environment extraction, private source transfer and relocation
have completed. This validator does not install or repair anything.

```bash
source /home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/deploy_31731/env_h3_31731.sh
/cache/zhonghao/h3/env/bin/python -B /home/ma-user/workspace/zhonghao/h3_v2v_minimal_20260913/deploy_31731/validate_h3_runtime.py
```

The script writes a JSON report to stdout and returns nonzero on failure. The
host, Python prefix/version, AArch64 ELF ABI and 18 distribution versions are
pinned to the source metadata read on 30213. It checks private module origins,
active Python/library/CANN/ATB paths and seven necessary ELF dependency lists.
The host's system C library and Ascend **driver** libraries are intentionally
allowed; CANN/ATB libraries are required to come from the private environment.

Network connections are blocked, Hugging Face is offline and torch backend
autoload is disabled. After explicitly importing torch_npu, its initialization
flag must still be false; both the Python lazy initializer and C initialization
entry point are blocked before importing the vLLM packages. No availability
query, device count, tensor, model, compile, or inference is requested.

Active `.pth`, editable finders, entry-point scripts and vendor `set_env.sh`
files must not retain old source prefixes. Conda history/package-cache records
and `direct_url.json` provenance are not mistaken for live import bindings.

`passed_cpu_runtime_checks_only` does **not** establish NPU kernel correctness,
card health, full-model loading, performance, or compatibility of all optional
operations. Those need the separately supervised tiny NPU and model gates.
Only toy unit tests and source metadata reads have been performed while writing
this script; the actual 31731 runtime has not been claimed as validated.
