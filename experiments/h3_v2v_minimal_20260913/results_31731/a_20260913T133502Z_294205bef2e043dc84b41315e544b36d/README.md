# 31731 八卡 A：湖景 → 雪景

运行：294205bef2e043dc84b41315e544b36d，原始 Ref2VA dense，USP8/ring1/textTP8/VAEpatch8，其他固定参数见 experiment.json。

正式请求2026-09-13 21:41:50.442889 CST开始，21:56:21.579085 HTTP200返回，端到端871.1053479220718秒；50 requested steps，不将日志49迭代当作实测NFE。21:56:55.357240自动清理完成，八卡释放、remaining_owned_process_groups为空、needs_attention=false。

输出a_50step.mp4：11,934,240字节，SHA256 1df518c92da95f9f926d1a82284199a6017660a46a502502c08b12219c52c9a8。h264+AAC、1344x768、24fps、124视频帧、容器5.207秒。媒体元数据通过不等于编辑质量验证。纯DiT延迟与NFE未直接计数，尚无B/C正式结果或算法加速比。

source.mp4为原私有湖源的只读副本（SHA e02b3df481f66d0968ab65eac1d56ded8493da952c0bfb21b70ecd5715f87cf9），不是GT target。

本目录只接收这次31731新A产物；不采用30213历史结果。下载后source/output SHA均已核验。5个抽样帧观察见visual_review/observations.json：可见积雪、总体构图及大致镜头变化保留；未连续播放、未听音频，不宣称全程无闪烁或严格对应。

A清理后的旧队列于21:57启动的B在预检因PATH缺npu-smi失败，未加载/占卡；两者退出且历史保留。22:03使用未修改的现有私有env启动唯一新B6362ce1811da4db69d335b3b3ca5d1a4，22:07实测模型加载；不得另起B副本。
