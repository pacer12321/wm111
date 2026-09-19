# 官方 paired V2V 训练数据核查

日期：2026-09-14。范围：OpenVDN 与 MiniMax H3 官方公开资料；额外核验一个被搜索结果指向的 DiffSynth 示例。未下载视频、权重或大规模数据，未启动训练。

## 结论

在本次核查的官方公开范围内，未找到可直接用于规模训练的 source video + 编辑指令 + edited target 三元组数据集。

不能表述为“完全没有任何样例”：H3 官方确有包含输入、编辑指令和生成输出的演示。但演示输出是模型生成结果，不是人工标注 GT，也没有规模化训练集、分割和质量保证。

## OpenVDN

- [HF 官方组织](https://huggingface.co/OpenVDN)显示 datasets 0 / None public yet。
- [官方数据预处理说明](https://github.com/OpenVDN/vdn-minimax-h3#data-preprocess)要求自行提供 video_index.jsonl 和预编码 video/audio/text latents。
- [Stage-B 配置](https://github.com/OpenVDN/vdn-minimax-h3/blob/main/configs/training/stage_b_c1_vdn_anchor.yaml)的训练 index 需用户填写，未提供配对数据下载地址。
- [H3LatentT2VADataset 实现](https://github.com/OpenVDN/vdn-minimax-h3/blob/main/src/training/dataset_h3_latents.py)读取每条一个 latent_path，以及同名 audio/text；返回 video_latents、audio_latents、prompt_embeds、text_token_tags，没有第二路 source video 或 source/edit/target 数据协议。
- [模型文件树](https://huggingface.co/OpenVDN/vdn-minimax-h3/tree/main)的 h3-base、stage-b-step-2000、stage-dmd-step-250 是权重，不是数据。
- prompts 是少量 T2VA/FL2VA 推理示例。DMD 的 data-free 说明也不是 V2V 数据集发布。

判断：有训练代码和数据格式，不等于公开训练数据；现有官方 loader 不能不改就装载我们需要的双视频编辑三元组。

## MiniMax H3

- [官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)、[开源公告](https://www.minimax.io/news/minimax-h3-open-source)发布 FL2VA/Ref2VA 权重及相关组件，没有发现配对编辑训练集下载入口。
- [MiniMaxAI 数据集列表](https://huggingface.co/MiniMaxAI/datasets)本次可见 7 项，未列 H3 编辑训练集。该观察不排除其他未公开或未来发布的数据。
- [官方研究说明](https://www.minimax.io/blog/minimax-h3)描述多模态训练能力，但不能据此认定训练语料已公开。

### 官方编辑 demo：可用作复现线索，不是规模训练数据

- [原始 source/音频条件/编辑指令请求](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/scripts/readme/full-2k-ref2va-h3-context-ir.sh)。
- [本地 Ref2VA 展开 IR 请求](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/scripts/readme/reproducible-768p-ref2va-request.sh)。
- [生成输出视频](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/assets/ref2va.mp4)。

这是带额外参考音频的说话编辑演示。若以后使用，必须保留全部 conditioning 和输出来源；不把原始简短指令和展开 IR 无说明地互换，不把不同分辨率成片计作大量独立样本。未在本轮下载/播放这些媒体，因此没有声称已亲自验证编辑质量或帧对应。

## DiffSynth Ref2VA 示例的排除核验

[训练脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Pruned-Ref2VA.sh)引用 DiffSynth-Studio/diffsynth_example_dataset，但它不是 MiniMax 官方发布的数据集。

[实际 metadata.json](https://modelscope.cn/api/v1/datasets/DiffSynth-Studio/diffsynth_example_dataset/repo?Revision=master&FilePath=minimax_h3%2FMiniMax-H3-Pruned-Ref2VA%2Fmetadata.json)经只读读取后确认只有 1 条示例，核心字段为：

- video: train_video.mp4。
- input_audio: train_video.mp4。
- references: type=image，image=0.png。
- prompt: 产品网站/UI 滚动动效生成。

它没有 source video 参考条件，不满足本轮 V2V 三元组要求。不能仅凭 Ref2VA 文件夹名或目录中存在另一 mp4，就推定是 source/target 配对数据。

## 对新实验的影响

1. 官方开源资源可作为训练框架和权重起点，但现在不能写成“已有官方 paired 数据可直接开训”。
2. 继续完成双序列可反传实现和小尺寸结构验证，不需要伪造训练数据已齐。
3. 若扩大数据来源，需要另行选择有清晰 source/edit/target 协议和使用许可的第三方数据；本轮没有扩展为大规模第三方检索或下载。
4. 另一条候选是冻结教师生成、质量筛选后的伪配对数据。它需要单独的生成协议/算力与存储安排；本轮没有选择或启动该路线。
5. 现有换色测试不能直接转为训练集后仍声称是未见测试。数据拆分必须按原始 source clip 身份进行。

代码许可、权重许可和视频数据许可分开核验，不把仓库 Apache-2.0 或模型许可证自动当成全部媒体的训练/发布授权。
