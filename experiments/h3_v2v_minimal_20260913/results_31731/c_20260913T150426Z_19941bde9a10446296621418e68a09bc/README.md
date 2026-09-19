# 湖景 C 正式结果

31731 单机八卡，run `19941bde9a10446296621418e68a09bc`。50 requested steps、seed4101、1344×768、24fps、124帧；49条采样进度不等于实测NFE。23:32:09.816281 HTTP200，request端到端 **904.0943806068972秒**；23:32:44.639042完成清理。旧C只通过smoke后解析失败的记录另存，未冒充此次结果。

正式文件 `c_50step.mp4`：12149617字节，SHA256 `014fc5c0da43984e7a903ee6cba199e5a62939d852d04eda07eda0a80356366f`。本目录result/request/status/smoke gate/strict metadata/server.log均从实际run下载，video/result/status已与远端独立核验SHA匹配。

root23:33复用已审只读完成检查器并显式传入新C日志parser，核验当前host/全部frozen SHA、source/runtime/transfer/tiny/权重身份及header、唯一正式请求/HTTP/媒体SHA和ffprobe、smoke前缀SHA及8worker两次strict记录、实际完整加载记录、cleanup及原supervisor/server/worker身份退出、fresh8idlehealthy全部通过。没有调用旧queue启动流程或改pin，也没有重新hash大模型payload。详见 `completion_verification.json`。

同湖源单次ABC请求端到端：A871.1053479220718秒，B948.1026493189856秒，C904.0943806068972秒。C相对B降低4.6417%，但仍比dense A慢3.7870%。这不是总体加速，也不是统计稳定、纯DiT或NFE结果。

本地 `visual_review` 有四路五帧对照和观察记录：C积雪/结冰更强，但前景树木结构也更偏离source。部分结冰本就在指令内，不能仅此判失败；同时不能宣称等质或严格保持镜头。只抽样看图，未连续播放、未听音频、未做定量质量评估。

抽帧已执行且拒绝覆盖。实际依赖是捆绑Python3.12与工作区`.audio_tools`中的PyAV，不需要安装库。不能因默认Python缺av而重跑已完成的抽帧任务。
