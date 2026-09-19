# 31731 单次 A lake → B lake 队列

当前实现仅在本地完成，尚未部署、运行。经root明确批准，当前代码绑定2026-09-13 21:35:02.673 CST启动的新A `294205bef2e043dc84b41315e544b36d`，固定per-run目录 `a_20260913T133502Z_294205bef2e043dc84b41315e544b36d`，监督器PID `4089275`、start-ticks `124318608`（真实pgrp/session均为 `4088644`）。新A仍需完成自己的smoke、正式请求与清理；这里不宣称它已成功。之前失败的 `8237f7b4a900494098473e95d6085f48` 不再是合法前序。不能自动跟随latest，未来换绑定仍须明确批准并重新测试。

## 只允许的工作

固定 31731、8张物理卡0–7、lake_snow。同配置 USP8 / ring1 / DiT TP1 / textTP8 / VAE tile8：等待指定 A 的50步正式请求完成与清理，然后最多启动一次已经审查过的 B。无 C、反转样例、新数据、训练、命令模板或任意卡数/模型参数。

等待阶段仅持有个人 `/cache/zhonghao/h3/trial_queue/queue.lock`，不拿8张卡的锁，也不启动模型。仅在 A 成功、正式请求完成、同服务 smoke gate 已通过、全部自有进程清理、8卡idle证明有效、原 supervisor PID/start-ticks 已退出后，重新核查源码/输入/权重/runtime/tiny证明与视频，再做一次只读 `npu-smi` 检查。B 启动时仍须自己拿8张卡锁并重新检查idle：队列的只读快照不是卡资源授权。

等待 A 最多8小时；等待自己启动的 B 最多4小时。每30秒只读轮询；有明确终态。任何失败都不重试、不执行下个工作。B成功终态仍叫 `completed_review_required`，不代表编辑质量、NFE或加速已经得到验证。

## 持久化与中断

每个A只有一个确定性目录：`/cache/zhonghao/h3/trial_queue/a_lake_to_b_lake_<A-run-id>/`。

- 已存在的目录一律拒绝重启，保留旧状态，不能覆盖后再发B。
- B启动前先用 `O_EXCL` 创建并fsync `b_launch_intent.json`，再fsync目录，再 `Popen`。若恰好在这之间崩溃，可能漏跑B，但不会因为重启重复跑B。
- `Popen`失败也消耗唯一尝试；不要删除ledger/tombstone来“重试”。需要人工核查和新的明确批准。
- 队列mutex FD传给B supervisor，队列进程意外退出时不会释放仍由B持有的mutex；B自己的设备锁仍由其监督器维护。
- 正常中断/超时只会在核验 Popen child 的 PID、start-ticks、进程组、session 和 `H3_SERIAL_QUEUE_ID` 后，向**该B监督器**发送SIGTERM，让它执行自己的清理。绝不向A/foreign PID发信号，不kill进程组，不SIGKILL监督器。120秒内无法确认自有B退出则标记 `needs_attention`。
- 若队列被SIGKILL/宿主崩溃，无法保证它执行清理；持久ledger阻止重复提交。仍需检查B自身状态和每卡锁，不得把queue退出当作卡空闲。

输出 `queue_status.json` 保存精确的A/B per-run路径、run ID、host/boot、进程身份、启动次数、完成证明和错误。B的latest只用于发现自己新启动的PID；一旦核验并绑定，后续只读对应的per-run状态，不跟随latest替换。

## 部署与调用

运行文件只部署到 `/cache/zhonghao/h3/trial_queue_code/serial_ab_queue.py`。依赖已经审查的A/B/validation文件及环境脚本；SHA直接固定在 `PINNED_FILES`。使用私人环境Python，不改A/B/validation文件。

下面是当前批准的新A绑定调用方式；替换绑定必须先改代码常量并重新审查测试，不是任意CLI参数：

```bash
source /cache/zhonghao/h3/env_h3_31731.sh
/cache/zhonghao/h3/env/bin/python -B /cache/zhonghao/h3/trial_queue_code/serial_ab_queue.py \
  --predecessor-status /cache/zhonghao/h3/model_trial/01234567/lake_snow/A/runs/a_20260913T133502Z_294205bef2e043dc84b41315e544b36d/a_status.json \
  --predecessor-run-id 294205bef2e043dc84b41315e544b36d \
  --allow-b-launch
```

没有显式 `--allow-b-launch` 不创建队列、不启动子进程。如何在远端托管队列由root另行审查；本脚本不自行nohup或创建调度任务。

## CPU 回归

Windows本地：

```bash
python -B -m unittest discover -s experiments/h3_v2v_minimal_20260913/deploy_31731/trial_queue -p test_serial_ab_queue_cpu.py -v
```

Linux可把测试文件与运行文件一起放在独立测试目录，运行：

```bash
/cache/zhonghao/h3/env/bin/python -B -m unittest discover -s /absolute/reviewed/test-directory -p test_serial_ab_queue_cpu.py -v
```

测试只导入标准库，不导入torch/vllm、不访问NPU或SSH。实际status schema由toy fixtures构造；媒体probe、源/权重/runtime/tiny证明用显式fake返回，测试验证队列逻辑而非假装模型通过。Linux使用真实flock、O_EXCL、fsync及临时文件；Windows测试stub fcntl、O_NOFOLLOW/O_NONBLOCK和目录fsync。固定源SHA测试读取本地仓库对应文件，或Linux上已审查部署的固定路径。
