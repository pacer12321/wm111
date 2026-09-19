# 显式倒放 A8：生成完成，整段倒放未成功

31731 八卡 run `91c6c0f98ba84c7c828202d5934176cd`，sample `explicit_reverse_couple_124`。原始 dense A，同 USP8/ring1/DiTTP1/textTP8/VAEpatch8；source 是未倒放的104:228截取、124帧24fps无音轨视频，指令要求整段时间倒放。

新2步smoke和同服务gate通过后，只执行一次正式50步。正式HTTP200于2026-09-13 23:55:49.400031 CST完成，完整请求耗时 **846.0936839610804秒**；23:56:22.550435清理终态，8卡释放、remaining为空、无error/needs_attention。输出4,271,598bytes，h264+AAC，1344×768、24fps、124帧、容器5.207秒。

- 正式视频：[a_50step.mp4](a_50step.mp4)，SHA `17b83ee2196302e9523e3e54aa239e3d6e7c8d5070c83d7c066e4f8cf06b38d7`。
- 完成与完整性摘要：[completion_verification.json](completion_verification.json)。root已实际执行reverse-specific只读完成核验；本摘要区分root远端检查和本地下载证据，不冒充原始执行stdout。未重新哈希大模型payload。
- 画面审查：[visual_review/observations.json](visual_review/observations.json)，SHA `c809cc93bbc3a1a8364263ebc3b7069bd5a6242f8c7384ab02de0676535d0eaf`。

九点采样及额外原生帧显示，输出仍是source的正向事件顺序：先面对面靠近、接触/亲吻，再贴胸拥抱，末尾仍在拥抱中望向镜头。要求的倒放应该从最后的拥抱开始、回到开头面对面；因此这次 **dense A不能作为成功倒放基线**。若B/C也失败，不能仅凭此归因C固定同帧限制。指令已经明确要求倒放，这个样例也不证明必须使用VLM。

审查未连续播放、未听音频、未测逐帧运动或精确像素/相机对应；不声称每个微小动作、全视频质量或等质。846.094秒是完整请求时间，不是纯DiT耗时或NFE，也不能和湖景不同样例混算加速。

截至2026-09-14 00:11:17 CST，root实测唯一倒放B `642e2c4f790d4505a5fd0940bcfc43a2` 已于00:10:22进入2步smoke、正式0次，尚无smoke/formal结果，不是烟测已通过；倒放C未启动。湖景ABC结果保留、30213后续取消、旧队列不重启。本目录原始下载视频、source、request/result/status、日志和画面审查文件均未修改。
