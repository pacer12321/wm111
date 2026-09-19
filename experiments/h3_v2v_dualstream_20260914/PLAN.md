# H3 V2V D — Source 混合注意力推理实验

## 当前唯一执行版本：D（用户最新明确指令）

保留旧 A/B/C，在其基础上只新增未训练 D：S–S 复用 VDN 原有 c5/r1 local window、本路首尾 anchors 和 linear；T–T 完全保持 C；T–S 完全保持 C 的对应 latent 帧限制。S 的视觉 query 不再直接读取 T；原文本/音频交互和文本 linear state 注入不改，因此不宣称 source 在整个网络中与 T 完全隔离或可跨步缓存。

T 的原始 linear 调用保留，再以 source-layout 单独调用同一套已加载分支参数；不新增随机参数，也不改 RoPE。首尾处理沿用原 VDN：本路 anchor 由 Softmax 全局访问本路，本路 linear 仅处理内部帧。此前 strict-local/no-anchor、source-closed 与 H0/S0/H1/S1 主线均被这个 D 定义覆盖，不据历史段落实施额外改动。

本次唯一正式生成：31731 0–7 八卡，shirt_red_couple_124 原视频、原换红衬衫指令、seed 4101、1344×768、124 帧/24 FPS、请求 50 步、原精度和并行设置。复用带哈希绑定的同一次真实 VLM preserve 判定。新 CPU/NPU 验证通过后先自身 2 步 smoke，再一次正式 D；不自动重复、不训练、不自动改 seed 或追加变体。旧 B/C 只作历史对照，单次 D 不足以证明统计稳健或泛化。

验证进度：4 项 CPU 数值组合及 54 项标准库准入测试通过；真实单卡路径/八卡 SP8 tiny `96d26e0218e846b7b0953eb8a6e21174` 完成并清理，96.631 秒（验证用时，不是模型推理速度）。当前固定策略见 `deploy_d/reviewed_policy.json`；完整模型准入检查已通过。

真实启动：D run `7de1704b57104dc5adc17f78211f6395` 于北京时间 15:21:05 进入模型服务启动阶段，后台监管 PID1098140、server1098165。监管器自动执行自身2步smoke→一次50步正式请求→仅清理自有进程；不重试或训练。已有自动跟进30213-h3已恢复，只监测这个D run，完成后提供原视频/D同步对照，失败时不自动扩展实验。阶段和实际请求是否开始以 `experiment.json` 及远端 per-run 状态为准。

## 历史设计讨论（已被上方 D 决策覆盖，不作为执行依据）

下面保留旧 S0/H0、严格 local、多模态隔离和训练的讨论原文。包括“尚未确认”“未实现”等状态均属于旧时点；当前唯一生效的结构、边界及运行范围以上方 D 定义为准。

更新日期：2026-09-14。最新指令：先做新结构的未训练实验，看效果和速度，再决定训练。状态：S0 推理候选实现与验证中；新模型、新训练均未启动。

当前只推进 S0（同一 W0、无新增训练）。训练数据缺失、反向传播和 optimizer 尚未就绪，不再作为这次前向实验的前置条件。先复用现有 source、换色指令和带摘要绑定的真实 VLM 判定。CPU/单卡/八卡前向测试及新 S0 实际 kernel 激活证据通过后，才能一次新 smoke 加一次正式请求。下文 H1/S1 与持续训练均延期，不能自动开训。

原 OpenVDN anchors 是否保留、文本/音频是否同时隔离已向用户澄清；最终答案必须写进冻结 policy，不能凭未回答的选项默认值当作用户确认。尚未确认前不部署正式模型或生成结果。

本文件取代“下一次先做 C-warmup”的建议，不覆盖旧 ABC 的代码、权重、计时或失败结果。用户已明确允许持续训练，不以人为短时限阻止训练；但数据、训练正确性与保存恢复仍是准入条件。

## 1. 用户指定的新结构

S 表示 source video；T 表示正在去噪的 target video。这里改变的是 DiT 内的注意力，不是修改视频 VAE。

- S 的视觉 Softmax：只访问本路邻域 S。
- S 的视觉 linear：聚合本路远距离 S，不读 T 的 recurrent state。
- T 的视觉 Softmax：访问本路邻域 T，以及对应 latent 帧的 S。
- T 的视觉 linear：聚合本路远距离 T，不直接混入 S 的 recurrent state。
- 首轮对应关系为相同 latent-frame ordinal；不是相同 RoPE 时间值，也不是自动找到的运动对应。
- S/T 可共享同一层的投影和线性分支参数，但卷积边界、帧统计、前后扫描和状态必须分开。首轮拟共享参数以减少新增参数变量，尚未实现。
- 原有位置编码、去噪步数和精度不顺带改变；不同时引入 warmup、自适应 correspondence、DMD 少步蒸馏或简化线性注意力。

这是新结构，不是已跑旧 C。旧 C 中 S-query 仍全局、S 没有 linear 输出，新方案同时改变 S 的计算和跨路信息流。

## 2. 必须显式确定的边界

### 邻域和首尾帧

当前 B 的 VDN 窗口为 c5/r1：每 5 个 latent 帧一组，读取本组与相邻组，最多 15 个 latent 帧；不是简单的前后各 1 帧。首轮建议沿用这个邻域定义，避免额外修改窗口大小。

原 VDN 还有首尾 global anchor 例外，且 linear 只处理内部帧。用户现在明确说“只看相邻帧”，因此新计划按严格 local 解释：不悄悄保留全局 anchor 行/列，首尾帧同样受窗口约束，且要把首尾帧纳入本流 linear 写入和读出。

若改为“局部 + 本流首尾 anchors”，必须单独命名，不能宣称仍是严格 local。这个差异在实现前应向用户说明，不能仅复用旧函数后认为新结构已完成。

### 条件文本、音频和 source 独立性

只禁止 S 直接读 T，并不能阻断 T → 共享 text/audio → S 的跨层回流。当前无音轨输入仍可能存在生成端 target-audio tokens，不能忽略。

必须在模型接入前冻结多模态路由契约：

1. source-side 可读条件集合不能含 T、target audio，或已经读取过这些内容的动态 text/audio；
2. source linear 的条件状态也只能来自 source-side 闭合集合；
3. 可采用不读 T 的条件文本，或分离 source/target 条件状态；不能无说明地改变原 H3 音频能力；
4. 本轮只冻结视觉规则，具体非视觉路由实现尚未选定，属于模型实现准入项。

固定条件、时间和权重后，扰动 T，逐层 S hidden 应保持不变，才可以称 source 独立。这里允许最终 T loss 通过 T→S 路径反传给 S；独立性不等于切断梯度。

不预先承诺 source KV 跨去噪步缓存。当前 source latent 固定，不代表 hidden、时间条件和每层 K/V 跨步固定；训练时更不能跨 optimizer step 复用旧的 detached cache。

## 3. 新实验编号与公平比较

旧 A/B/C 保留原名和全部结果，不能用旧 C 代替下面的 S0。

- A-external：旧原始 dense Ref2VA，系统级参照，不作为纯 mask 消融。
- H0：当前 B 的 hybrid 规则，无新增训练；统一初始化 W0。
- S0：新双序列规则，无新增训练；从同一 W0 初始化。先保存这个结果，再训练，防止只有训练后成片却不知道结构本身造成什么变化。
- H1：H0 规则 + 对齐训练，作为等训练预算对照。
- S1：S0 规则 + 同样训练数据、教师、损失、可训练参数集合和预算。

解释边界：

- H0 对 S0：新路由和执行方式的即时影响。
- S0 对 S1：新结构经过训练后的恢复/改善。
- H0 对 H1：普通训练本身带来的变化。
- H1 对 S1：同等训练条件下的新结构收益。
- A-external 对最终 S1：整个系统的质量/速度对比，不能描述为权重相同的纯消融。

当前优先完成 S0 前向实现和未训练结果；旧 B/H0 仅作历史参照。若 S0 质量可接受，再补同环境的新 H0 配对计时，才讨论可靠的加速收益。训练 H1/S1 暂缓，且必须在用户看过未训练实验后再决定。未来 H1/S1 记录实际更新次数、有效样本/噪声抽样、训练 FLOPs/卡时、初始化和 checkpoint；持续训练也须选相同预算的 checkpoint 比较。

## 4. 对齐训练：拟定方案，不是已完成配置

优先把新注意力当作学生、冻结现有 B 的有效权重作教师。教师选择是初始建议；数据或质量证据若要求换教师，应另立版本，不能中途混用结果。

主损失：教师和学生接收相同 S、编辑条件、target noisy latent、去噪时间与其他 conditioning，匹配 target 区域的速度/去噪预测。必须核对 H3 原始输出参数化和归一化，不能凭通用公式强套。

- 这是注意力结构/函数对齐，不是自动产生 temporal correspondence，也不等于 DMD 减少采样步数。
- 首轮只用一个主要蒸馏目标；不先叠加多个感知损失、完整 attention map 拟合或外部跟踪网络。
- 拟训练注意力 Q/K/V/O 的新增 LoRA、linear 分支与门控；共享投影是否拆分以实际实现为准，H1/S1 的训练参数集合必须一致。
- VAE、外部文本编码器、其余主干初期冻结。source 路径的可训练部分必须确实接到 target loss 的梯度。
- 已合并的公开 Stage-B LoRA 属于 W0，不能重复合并；若新增训练 LoRA，明确它是新零增量适配器，单独存储与命名。
- 真实 paired target 可用于监督 flow matching；缺失时，质量筛选后的教师伪目标必须标记为 pseudo，不得称人工 GT。

只有 source + 编辑指令也不代表监督已齐：需要选定 target latent/轨迹分布或教师生成协议。旧 MP4 没有逐步教师预测，不能冒充现成蒸馏缓存。只做 source 重建不能证明指令编辑能力。

## 5. 数据与拆分

用户要求优先核查 OpenVDN 和 MiniMax H3 官方是否提供可下载的 source + edit + target 三元组。训练代码、latents 格式、权重、prompt demo 与成对编辑训练集分别记录。

已知现有换色样例没有官方 paired target。当前只有少量 source 和 ABC 输出，不能当成可用于泛化验证的训练集。

首轮聚焦保持时序的颜色/材质/局部外观编辑。先按源视频/原始 clip 身份划分 train/validation/test，再生成裁剪和编辑变体；同一 source 的不同裁剪、seed 或编辑不能跨 split 泄漏。

现有 shirt_red_couple_124 已被多次用于开发判断，只能标为 development regression，不再称 untouched test。若拿它做单样本过拟合，只能证明训练管道有信号，不能再用它证明泛化。

数据记录必须含 source 路径与摘要、edit、target 路径与真实/伪标签来源、许可、源 clip 身份、时间对应、裁剪和预处理参数。未下载的数据不得填成已就绪。

VLM 仍作为输入任务判断的一环：首阶段仅接收 preserve 且人工复核的数据，记录路由成本。不把模型输出的动作保持失败改写成指令要求时序变化。动态时序编辑和自动 correspondence 留到该阶段之后。

## 6. 开训顺序与验收门槛

G0 — 契约和数据：冻结视觉/非视觉路由、邻域、端点处理、教师/训练参数；确认官方数据检索结果与可用的训练/验证数据。

G1 — 小尺寸 CPU oracle：检查 S/T 边界、source 无 T 回流、本路近远域覆盖且不重复、首尾正确、同帧映射与 RoPE 解耦。不保存完整真实长度 attention 矩阵。

G2 — 可反传实现：先小形状单卡，再多卡梯度等价；覆盖 QKV、conv、alpha、beta、gate、T loss→S 参数，以及保存/恢复。当前推理前向通过不能当作训练通过。

G3 — H0/S0：在训练框架内确认 H0 与已冻结 B 的输出可比，核对权重和实际模型前向次数；保存 S0 未训练结果。新的内核/音频条件路径如改变了行为，应额外标记，不沿用旧 B 的数值结果代替。

G4 — 持续对齐训练：G0–G3 通过后运行 S1，定期保存 checkpoint 和固定 held-out 验证；同步规划 H1 等预算对照。NaN、梯度缺失、checkpoint 损坏、磁盘不足或验证持续退化时中止并排查，而非因“可以一直训”而无限浪费计算。

G5 — 效果和性能：相同样本/seed 成对评测、固定 VLM 判定协议、保持采样/精度/分辨率等；按 source/output 同步左右播放器展示，区分有限抽帧、完整时序审查与自动指标。

长期训练没有用户指定的截止时长，但不是无限制的数据下载或磁盘占用授权。计划周期性保存最新和最佳 checkpoint，实际保留/清理规则须在实现时限定为新实验目录，不能删除旧实验或他人文件。

## 7. 实际发现的训练工程缺口

- 当前 S linear 不存在：旧 openvdn_npu.py 的 forward_head_shard 只处理 target 内部帧，source/首尾 linear 输出为零。
- 当前 scan 使用 torch.baddbmm(..., out=...)，不能原样沿用为常规 autograd 实现。
- 当前帧统计用原地 dist.all_reduce；现有 USP 前向验证没有证明分布式反传正确。
- 原 serving pipeline/denoise 明确关闭梯度，不能仅加 optimizer 就成为训练入口。
- 31731 当前 src 只列出 vllm、vllm-ascend、vllm-omni；OpenVDN 官方另有训练代码，但尚未证明其 Ref2VA 数据路径、权重迁移和 Ascend backward 可以直接使用。

应在独立新目录准备可训练模块/入口，不能直接改写冻结的旧 ABC 推理依赖。

## 8. 性能与质量口径

原换色单次正式请求：A 855.102 秒、B 956.109 秒、旧 C 888.096 秒。旧 C 对 B 约 7.1% 更快但动作/构图保持失败；这些不是 S0/S1 的结果。

新方案减少 S 的 dense 计算，却增加 S 的 linear 计算、双路状态和训练工作；只凭复杂度不能保证真实 Ascend 加速。单独计 source Softmax/linear、target Softmax/linear、索引/打包/通信、DiT 每步、VAE 与端到端请求耗时。VLM、模型加载、教师训练开销另列，不与推理加速混算。

效果检查至少包括编辑是否完成、非编辑区域与身份保持、动作顺序/时间/视角、时间稳定性。保质和加速都通过，才称目标达成；loss 下降或单帧好看不够。

## 9. 资源和当前状态

- 仅使用用户指定的 31731 八张 Ascend 910B3；30213 不恢复。
- 2026-09-14 14:38:45 +08:00 只读 npu-smi：八卡 OK，AICore 0%，无报告的 NPU 进程。此快照不是未来运行时的资源预约；真正启动前要重新核验。
- 本轮没有启动训练/推理、没有更改远端源码或下载权重/训练数据。
- 旧自动跟进仍为之前确认的 PAUSED；本轮没有通过工具恢复，不能声称已有自动训练监控。
- 机器可用不等于数据/训练入口已就绪。当前状态与未决配置见同目录 experiment.json。

## 10. 官方参考与本地证据

- [OpenVDN 官方训练说明](https://github.com/OpenVDN/vdn-minimax-h3#training-recipe)：训练配方及自备 latent index，不能由此推断已发布编辑训练集。
- [官方 latent dataset reader](https://github.com/OpenVDN/vdn-minimax-h3/blob/main/src/training/dataset_h3_latents.py)：T2VA video/audio/text 读取接口，不是 Ref2VA source/edit/target 三元组接口。
- 旧 source/target 规则：../h3_v2v_minimal_20260913/c_adaptation/candidate/strict_source_layout.py。
- 旧 linear/scan：../h3_v2v_minimal_20260913/b_adaptation/patched/openvdn_npu.py。
- 旧训练/通信风险位置：../h3_v2v_minimal_20260913/b_adaptation/patched/minimax_h3_transformer.py。
- [官方数据检索记录](DATA_AUDIT.md)：已核查，未找到官方规模化 V2V 配对训练集；H3 有编辑 demo，DiffSynth 的 Ref2VA 示例实际是参考图而非 source video。没有下载或启动伪数据生成。
