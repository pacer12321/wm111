# 显式倒放 C8 正式结果：核验及画面抽查完成，整段倒放失败

固定run `e2a7c9c133ca49a6af00134623e9e719`，31731八卡、sample `explicit_reverse_couple_124`。本run先2步smoke和同服务gate，再只执行一次50步；01:03:12.930187 CST正式HTTP200，完整请求882.104259646032秒。输出4,012,027bytes，h264+AAC、1344×768、24fps、124帧、容器5.207秒。

- [正式视频 c_50step.mp4](c_50step.mp4)：SHA `5dce91778b5908c29e3b61f883b045aad621b5f2109d94faaa31b2c325c513ad`。
- [实际请求 request.json](request.json)：SHA `aaba2c5b197eff00091e0341052f943c6cabe4fd5fecdd1c36a9bf8cd18e308e`。
- [实际结果 result.json](result.json)：SHA `8806ee44a9dc3043c5724862ab588adb936eee18a022a7bf7afc51f44dccc74f`。
- [终态 c_status.json](c_status.json)：SHA `3574508b6b4ce3fac368ea35d4ef1c11c8abee5ce2466fca3cf40bcbe3e408d4`。
- [正式 strict 证据 formal_strict_metadata.json](formal_strict_metadata.json)：SHA `c5cd7f50a9e64de2afec075e0069cf079ecd02ade5c873795708c7f669d31222`。
- [root真实只读完成核验 completion_verification.json](completion_verification.json)：01:09:07.473515 CST pass、exit0/chunk8df829，保存实际工具输出；包括首次独立reader的可选size_bytes字段错误及修正后完整重跑，未改实验本身。

终态01:03:47.663447，cleanup/八卡idle通过、remaining为空、needs_attentionfalse/errornull。root完成全部当前frozen/source/runtime/Ctiny/transfer/weights headers、两request/HTTP/media SHA+ffprobe、完整8PID加载/线程恢复、smoke strict1与gate/正式strict2与实际JSON、十个原身份退出/owned group为空/freshidle核验；7份文件下载SHA通过，未重哈希大权重payload。strict metadata是每请求的实际布局记录，不是逐层mask trace。

关闭阶段有告警：正式编码01:03:12.643后，01:03:15.926开始shutdown；Tracebacks及01:03:46.067 Orchestrator30秒停止超时ERROR均在其后。最终资源清理核验通过，但不能称无告警退出，也不能把关闭告警误作正式生成失败。

**五列9点画面抽查已完成，C没有完成整段倒放。** [observations.json](visual_review/observations.json) SHA `74a818041a6d1bfcae380161e4e34794bc1165a4389461768ac0d86d904bee78`；[extraction_record.json](visual_review/extraction_record.json) SHA `20e7fc536f403b13645bdc1b86c4862bff2c0da1613067d17a01733dafb01885`。45native PNG/4contact，source/A/B/C前后SHA、124帧24fps、所有PTS通过；独立代理看C原生0/30/60/90/123，root看全部3张分段contact。

C与A/B失败方式不同：九点持续以拥抱为主，女子背朝镜头、男子更正面，视向/更近裁切发生改变；虽然开头是拥抱，后半及末尾没回到source起初面对面托头姿态。不能称A/B式正序复制，也不能称静态冻结，存在头部姿态和构图变化。C保持浅衣、没有B的蓝衣变化；女子背部缺source可比视图，不判断服装设计改变，也不推断视向变化究竟来自相机移动还是人物旋转。抽样无黑屏/整体崩坏，未连续播放、未听音频或查全部细瑕，不是等质证明。

本轮六次正式最小验证及抽查完成，工程生成成功不等于编辑成功：ABC均未完成这个整段倒放，不能单独归因C稀疏或证明VLM必要。原始产物/日志/strict/完成核验证据保持原样，不新launch；monitor已由root于01:16:45 CST通过工具确认PAUSED并复读本地配置。六次请求配置对比与耗时汇总见实验根README，后续profiling和成功dense时序编辑基线只是建议，未启动。
