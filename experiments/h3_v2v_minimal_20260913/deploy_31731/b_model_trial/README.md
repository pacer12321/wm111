# 31731 独立 B8 入口

本目录只新增 B 试验入口；不会修改 A、tiny validation、环境脚本、模型或候选源码。还没有通过这个入口运行完整 B8，因此不能把 CPU 测试当作 B8 已跑通、编辑质量合格或获得加速的证据。

## 固定试验

- 单机 31731，物理卡 `0,1,2,3,4,5,6,7`，USP8、ring1、DiT TP1。
- text encoder TP8、VAE tile / patch parallel8。该实现的旧 textTP4 / VAE4 与 8 DiT workers 不兼容；A/B/C 必须采用同一修正配置。
- Ref2VA 原权重 + 完整原版 Stage-B linear branch 和 LoRA。B 的 source 仍全局可见；不加入 C 的对应帧限制，不改线性分支，不训练。
- 固定 `lake_snow`（默认）或 `explicit_reverse_couple_124`。源视频、指令、种子及采样设置复用 A 的冻结配置，不接受任意视频/模型路径。
- 同一服务仅执行一次 2-step smoke；全部门槛通过后仅尝试一次 50-step 正式请求；最后清理并核验 8 卡空闲。失败不重试。

服务端同步请求与客户端 timeout 都为 3600 秒，启动 init/stage-init 为 2400 秒，health 为 2700 秒。3600 只是等待上限：正式计时是 `b_50step/result.json` 中实际完整 HTTP 请求的 `request_end_to_end_seconds`，不包括启动、权重加载和 smoke，也不把超时上限作为运行耗时或加速结果。更长的等待不改变步数、数学或精度。

## 部署与依赖

将本目录三个运行文件部署到 `/cache/zhonghao/h3/b_model_trial_code/`：

- `b_trial_gates.py`
- `run_b_trial.py`
- `launch_b_trial.sh`

只读依赖 `/cache/zhonghao/h3/model_trial_code/trial_gates.py`（源码 SHA 固定）、`/cache/zhonghao/h3/validation_code/`、`/cache/zhonghao/h3/env_h3_31731.sh`，以及当前已通过 tiny 的候选 `/cache/zhonghao/h3/candidates/b_v1/vllm-omni`。如果 A gate 发生变化，必须审查并更新本目录的依赖 pin，不允许跳过门槛。

输出使用 `/cache/zhonghao/h3/b_model_trial/01234567/<sample>/B/runs/b_<UTC>_<runid>/`，最新状态为各 sample/B 目录中的 `b_status.json`。HTTP 为 `127.0.0.1:19099`；TMPDIR 为 `/cache/zhonghao/h3/tmp/b_<runid>`，ATB suffix、cache 与输出均按 run 隔离。每卡 flock 与 A/tiny 共用 `/cache/zhonghao/h3/validation/card_locks/deviceN.lock`，不会因独立输出而重复占卡。

## 运行（仅在审查后，由串行调度器调用）

外部串行调度器负责等待同配置的 A8 成功完成并核验清理，再调用 B。这个入口不启动 A/C、不轮询排队、不终止其他任务：如果卡锁被占或卡不是空闲，立即失败。不要与 A 并行启动。

```bash
source /cache/zhonghao/h3/env_h3_31731.sh
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/b_model_trial_code/run_b_trial.py --group 01234567 --sample lake_snow --allow-npu
```

另一固定样例单独启动，替换 `--sample explicit_reverse_couple_124`。没有 `--allow-npu` 不创建 supervisor、不触卡；不能直接执行 shell launcher 绕过监督器。

## 门槛与结果边界

启动前要求真实 host/boot 绑定、完整迁移、当前 CPU runtime pass、同组 tiny pass 与清理证明、候选与策略源码、源视频及解码帧数、原 Ref2VA 权重身份、Stage-B 800 branch tensors / 416 LoRA tensors、全部 8 个卡锁、8 卡健康和至少 55 GiB 空闲 HBM、宿主/container 至少 600 GiB 可用内存。模型大文件使用已验证的 transfer 全量 SHA，加当前文件身份/大小/header hash；不会冒称每次重新哈希全部 payload。

正式请求前要求同一真实服务 PID/start-ticks/session、同一 source/prompt/config、成功 smoke 的 HTTP/MP4/124帧元数据和输出 SHA，以及8张卡各一个属于本次服务的实际 worker。八个实际 PID 必须各有 535 base tensors、800 branch tensors、208 LoRA pairs 的完整记录，rank64/alpha64/scale1 与官方来源一致；CPU loader 必须完成并恢复 inference intraop/interop 线程。不能复用30213、旧loader、另一次服务或少数rank的 smoke。

`formal_completed_review_required` 只表示正式视频已生成且清理验证成功。编辑质量、实际 NFE、DiT-only latency 和加速解释仍需另行审查。`failed` 与 `needs_attention` 不能自动继续下一个模型。运行期间不要覆写固定代码、候选、权重、transfer/runtime/tiny latest 证明；否则 immutable-evidence gate 会拒绝继续。

## 本地 CPU 测试

```bash
python -B -m unittest discover -s experiments/h3_v2v_minimal_20260913/deploy_31731/b_model_trial -p test_b_trial_cpu.py -v
bash -n experiments/h3_v2v_minimal_20260913/deploy_31731/b_model_trial/launch_b_trial.sh
```

测试仅使用临时 toy 文件和 mock，不加载模型、不初始化 NPU。Windows 仅为 CPU 测试 stub `fcntl`，生产入口仍使用 Linux 真实 flock。
