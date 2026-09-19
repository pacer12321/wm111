# 31731 八卡 B：湖景 → 雪景

run6362ce1811da4db69d335b3b3ca5d1a4，原Ref2VA接VDN混合注意力，source在Softmax全局可见，未加C对应帧限制。固定同A的USP8/ring1/DiTTP1/textTP8/VAEpatch8、seed4101和50 requested steps。

正式请求于2026-09-13 22:14:29.025970 CST授权，22:14:29.055894进入request-running，22:30:17.163422 HTTP200返回；端到端948.1026493189856秒。22:30:50.856962清理完成，八卡idle、remaining为空、needs_attention=false。root随后独立复核完整正式请求/产物/冻结证据与supervisor退出，再复核当时八卡实际idle和健康。

输出b_50step.mp4：11,740,798字节，SHA256 ca8aa801cb1c96026b237f4f1dbb93d8e4846071338aae3796ae77c24537d2fa。h264+AAC、1344x768、24fps、124视频帧、容器5.207秒。视频及原始result.json/b_status.json已下载，完整视频SHA验证通过。

同31731八卡A为871.1053479220718秒；本样例本次B/A延迟比1.0883903440，即B慢8.839%，A/B加速倍率0.918788。这是单样例单次系统端到端结果，不是纯DiT/NFE测量，也不是统计稳定或其他编辑任务的结论。未混入旧30213结果。C部署CPU小测试在B烟测期间结束，早于正式请求约10秒；正式期间没有并发C复制/CPU测试/NPU任务。

有限视觉对照见visual_review：source/A/B同0/30/60/90/123帧，不能替代连续视频时序、音频或完整量化评价。Source沿用相邻A结果目录source.mp4，不是GT target。
