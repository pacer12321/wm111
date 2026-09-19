# 倒放 B：正式生成完成，编辑效果另行审查

31731 八卡，样本 `explicit_reverse_couple_124`，原 VDN 局部 Softmax＋linear、source 全局可见。正式 50 requested steps，1344×768、24fps、124帧，seed4101。

正式请求于2026-09-14 00:12:41开始，00:28:04.678091 HTTP200，端到端 **923.0886316811666秒**；00:28:38.185732完成清理。成片4,657,820字节，SHA256 `c15f08b33a5419ee704350133afd7441386738e11836aa7c605e71ae481c75cc`。相比同样例dense A的846.0936839610804秒，本次B延迟增加9.10005%；单次端到端数据，不是纯DiT/NFE或等质比较。

root于00:34:44.388728独立只读核验通过：实际frozen与当前code/runtime/transfer/tiny/sample/model identity/header、B日志配置、两请求及结果/HTTP/实际媒体SHA/ffprobe、实际prefix和完整log的八PID加载及线程恢复、原身份退出与fresh八卡idle/health。详见 `completion_verification.json`。初次核查遇到整数key与JSON字符串key比较不等，归一化后完整复核通过；没有修改任何生产文件。日志存在生成完成之后的关闭阶段Traceback/终止告警，不冒充无告警。

本目录已下载正式视频、request/result/status、smoke gate和server.log；关键SHA与远端核验一致。source复用相邻倒放A目录中的原始source.mp4，不重新处理输入。

画面审查已完成，见 `visual_review/observations.json`（SHA `63f3459a0d9cad62d916513933d9687f192f749746ab4b0f11bf675913f7a1ef`）：九个样本点仍主要按“面对面→靠近/接触→拥抱”正序，未整段倒放；45/60帧有动作阶段偏移，不称逐帧复制。另外男士浅色衬衫变成深蓝色，违背保持外观的指令。独立代理看9点及B原生0/60/123，root看全部三张分段对照图。完整解码、SHA及PTS通过不等于连续画面/音频检查。dense A也未倒放成功，因此不能仅据C同样失败归因稀疏规则，也不能声称等质。
