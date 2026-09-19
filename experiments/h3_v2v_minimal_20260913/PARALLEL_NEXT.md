# 最新需求调整（2026-09-14）：仅运行推理期 D，不训练

当前唯一版本 D：S–S 复用现有 VDN local+linear 及本路 anchors，T–T、T–S 保持 C。CPU/实际八卡 tiny 已通过，准备同换色样例与 seed4101 的 2 步 smoke 和一次 50 步正式生成。只用31731物理0–7，不训练、不新增其他变体。状态以 [D 清单](../h3_v2v_dualstream_20260914/experiment.json) 为准。旧 ABC 及以下训练讨论是历史，不构成额外启动授权。

## 历史：先换衣服颜色 ABC，所有样例先 VLM

**2026-09-14 13:57 本轮换色ABC已完成：C比B耗时低7.113%，但动作/视向保留失败，不是保质加速成功。** A855.1022686900105秒、B956.1085797089618秒、C888.0960794389248秒，均同样例单次正式HTTP请求耗时（非pureDiT/NFE，不含VLM）；C对B少68.0125秒/1.07658倍，比A慢3.858%。B/C为对应帧限制的增量对照；A/B初始化同Ref2VA，但B增加VDN分支/gates/mergedLoRA，因此A是系统级基线，不是纯注意力消融。

C `1fa6918c2e4743dc81d256610496f489` 13:50:58正式完成、13:51:32清理终态。完整独立证据审计PASS，十个原身份退出、自有group空、13:54:02八卡健康空闲；当前frozen依赖/完整加载/两请求与媒体/strict1和2实际记录匹配。关闭后Traceback/timeout警告保留，不当作生成失败或称无告警。证据 `deploy_31731/color_C_terminal_audit_20260914.json`，SHA999450c073c060b2f5a3931220c0e714d885da37b58e86278af64e68b007f0b9。

root和独立审查均实际看九点source/C contact及C原生0/60/123：红衬衫成功、女子白衣，但开头已亲吻，后段仍面对面靠近/微笑，没有source原有的后半贴胸拥抱；人物排列、视向、构图和背景明显变动。不是只换色、不是静态冻结；有限抽帧不是全程/音频/等质或算法首因证明。同步视频 `results_31731/c_20260914T052437Z_1fa6918c2e4743dc81d256610496f489/visual_review/source_vs_C_synchronized.mp4` 已下载/解码/PTS/SHA验证并预览，左source右C静音、无改速/倒放/原文件修改。独立QA SHA c78c4a7a08492d5ccf5eb21a90f07d2c64a16cf76c8069bb6d4485e77a58c939。

真实VLM对此指令判preserve，生成耗时之外另记录worker137.7666秒（其中加载37.8592秒、推理81.1199秒、decode/preprocess另计）；共用一次分类不代表完整方案可以漏算router。C的动作变化是保留失败，不是编辑指令要求改时序。当前没有活跃模型或待launch；原自动跟进已实际PAUSED并复读TOML确认（d3b0bd）。不自动进入时序编辑、训练或profiling，保留全部结果，后续需新方向。以下均为历史快照。

2026-09-14 13:41 最新：唯一换色 C `1fa6918c2e4743dc81d256610496f489` 已于13:36:05完成自己的2步烟测（153.01988986390643秒、HTTP200），13:36:09开始一次正式50步请求；13:41:24只读状态仍为 `c_50step_request_running`，formal_attempts1、尚未完成。原supervisor976351/start130013779和server976934/start130016739身份存活；尚无C正式耗时或成片质量结论。实际烟测文件/日志/strict/八worker的轻量独立核验正在进行，终态frozen/产物/退出检查留到完成后。仅跟进当前31731八卡C，不新launch、不修改活跃代码/proof，不碰30213。证据 `deploy_31731/color_C_progress_20260914_1341.json`；以下13:24及更早是历史。

**2026-09-14 13:24 最新绑定：B已完成，C已经唯一提交并真实启动。** 换色B `b52c1cc39548462f9a6b1f6124fef1ea` 正式请求956.1085797089618秒；同样例dense A855.1022686900105秒，B慢约11.8%，不是加速结果。完整终态独立审计PASS；B原十个PID身份退出、八卡释放及C之前的新鲜资源门槛已核验，不再重跑A/B。

B有限画面核验已完成：实际看三张source左/B右contact（0/15/30/45/60/75/90/105/123）和B原生0/60/123。男子浅衬衫均变红、女子上衣保持白色，主要靠近→轻吻→贴胸拥抱顺序、人物与草地远树背景及构图大体保留；衣褶、袖口/躯干轮廓和部分手臂贴合仍有细微重绘。未连续播放、未听音频，不据此证明精确时序、所有细节或等质；独立视觉记录 `results_31731/b_20260914T045104Z_b52c1cc39548462f9a6b1f6124fef1ea/visual_review/independent_observations.json` SHA `3bff87351cc8f1f8ed21cdc80dfceb8915289c5fa3d72b726970a837577edc87`。B左右同步对照：`results_31731/b_20260914T045104Z_b52c1cc39548462f9a6b1f6124fef1ea/visual_review/source_vs_B_synchronized.mp4`。

唯一活跃C run `1fa6918c2e4743dc81d256610496f489`，目录 `/cache/zhonghao/h3/color_trial_v1/results/01234567/shirt_red_couple_124/C/runs/c_20260914T052437Z_1fa6918c2e4743dc81d256610496f489`；13:24:37CST启动，supervisor976351/start130013779、server976934/start130016739，当前server_starting加载中。物理卡仅31731的0–7；同一换红衬衫样例、原source、seed4101和其余固定比较参数不变。监督器先自己的2步smoke，门槛通过才一次50步，尚无C烟测成功或正式结果；不得重复提交、改活跃依赖或碰30213。主任务负责JSON及既有自动跟进；本次记录编辑不产生任何远端/NPU/新launch操作。下方“C仍待启动”和B运行中等均为历史。

2026-09-14 13:11 最新：同一B `b52c1cc39548462f9a6b1f6124fef1ea` 自己的2步smoke已13:01:36完成（HTTP200、149.01836418709718秒、视频SHA9897446c7b0888b8080668287405ce81ac36a563691613780ef48bf37be849f4/5358182bytes）；同服务八worker门槛通过，13:01:40.260803开始唯一正式50步，attempts1、尚未完成。13:08:41日志20/49只是迭代进度非NFE。root13:11:02轻量复核实际smoke/formal请求source/prompt/设置/run/服务绑定、实际HTTP/curl、烟测媒体SHA和保存ffprobe、138441字节smoke log prefix SHA0b9f01b220d8d0973d775389669817b97542c2b81cf2935939801711f0382a92及八PID535/800/208+线程恢复记录全匹配；原十身份仍存活、八卡健康。正式request SHA456cc5ae889866260c49f3e87b5bd335dbf2eb055381b9393421c5b4c19e6964。证据 `deploy_31731/color_B_progress_20260914_1311.json`（76d827/0149ff exit0；一次性只读核验正则转义错误14d297已修，非模型失败）。全部终态current/frozen与正式媒体/退出/质量核验仍待完成；C未启动，不新launch、不改活跃依赖。下一次如B完成及时核验后接C。

2026-09-14 13:00:56 最新只读检查：同一B `b52c1cc39548462f9a6b1f6124fef1ea` 已12:59:07服务healthy并开始自己的2步smoke，当前仍smoke_2step_request_running、formal_attempts0、正式未完成。root实际核实supervisor868715/start129812623、server869394/start129815269以及物理0–7对应worker870084–870091（全部start129820402、PGID/session869394）仍存活、八卡健康；八PID真实535基础/800branch/208LoRA pairs记录及CPU加载线程4恢复192全部匹配frozen权重记录。实际smoke request SHA6f7ba4b40db2b0ca3ebd3e00ccd9a63117c84ad07869142fb4b06872d4c0b72a，source/prompt/参数/run/服务身份绑定通过。日志1/1是2步smoke内部进度，不是正式50步或质量证据。证据 `deploy_31731/color_B_progress_20260914_1300.json`（47301e/d98030均exit0）。未做重型当前依赖重查或新NPU请求，未启动C；保持活跃源码/proof不变，等自己的烟测门槛后监督器自动发唯一正式请求，再由heartbeat核验完成后推进C。

2026-09-14 12:51 最新实际进程：用户要求“现在立马跑bc”。唯一B `b52c1cc39548462f9a6b1f6124fef1ea` 已启动，路径 `/cache/zhonghao/h3/color_trial_v1/results/01234567/shirt_red_couple_124/B/runs/b_20260914T045104Z_b52c1cc39548462f9a6b1f6124fef1ea`；supervisor868715/start129812623、server869394/start129815269，frozen SHAf67ef6473716c853e3e94cc6c694330317ce1b13c5a9853658eae1d33f26954e。12:51:37实际server_starting，八卡锁内预检通过、formal_attempts0，未烟测/正式完成。提交幂等journal `/cache/zhonghao/h3/color_trial_v1/color_B_launch.json`、日志同目录`color_B_launch_20260914T045042Z.log`。禁止重复launch或并发C；B自己的2步smoke+535/800/208八PID加载/恢复线程门槛→唯一50步→退出/资源核验后继续C。真实source/edit/VLM和全部比较参数与A一致。本轮仅外部提交器第一次因日志目录不在H3私有树而在journal/Popen前失败，改为私有ROOT日志后唯一模型提交成功；冻结生产代码未改。证据 `deploy_31731/color_B_submission_20260914.json`。下方B/C未启动与A待审为历史。

A剩余实际请求终态核验已补齐：`deploy_31731/color_A_request_terminal_audit_20260914.json` SHA80ca874389ca06fb51545014fcd4a0c9ea81890ee78bee9d44c504836599f816，chunk9b35dc；smoke/formal实际source/prompt/run/server/参数/HTTP/curl/result/哈希均匹配。B提交前及锁内预检重新确认8卡idle/healthy。A无需再跑，用户认可当前A对照效果；抽帧范围不等于全质量证明。

## 用户确认的成片展示偏好（2026-09-14）

用户看过同步对照版后反馈“挺好的，之后都这样对比”。此后每个成片默认提供可直接播放的左右同步对照：左侧原source、右侧对应生成结果，清楚标注A/B/C或实际方法名和时间。沿用等比例缩放、不裁切、不改原文件；当前静音对照版应明确说明静音，原始输出保留。按照各自原始时间轴从起点同步播放，不为视觉对齐而重排、倒放、变速或截掉差异；未来时序编辑样例也不强求相同事件同时出现。核验由assistant主动执行并说明实际检查范围，用户无需承担必做核验。此次正面反馈仅指当前A效果/对照呈现，不代表B/C加速、整段无缺陷或量化质量已成立；不改变实验门槛与当前进程状态。

用户随后要求左右对照，已生成并解码验证新A `visual_review/source_vs_A_synchronized.mp4`：左原视频、右A，同索引/原24fps/124帧，无倒放、无改时序；等比例缩小加标签、对照版静音，源和A原文件未改。SHA6616740acbbd3f80eb35742631760c9a5f51f872484b3445b48af19c9c0e8be5。root看过对照版frame60检查排版，真实视频已供用户观看；不将生成对照版当成完成连续播放审片。独立审计已落盘 `deploy_31731/color_A_terminal_independent_audit_20260914.json` SHAfb3e0bbae4dd85e17dfc99281614dbac1a814d46fbe674b659e6365f80912e59；该审计明确未重新核对formal request实际内容/哈希且idle为历史，后续启动B前仍应补实际请求终态核对与fresh preflight。本轮没有新模型launch。

2026-09-14 12:43 画面更新：用户要求查看成片，root已完整下载最新换色A（SHA392fd304...9ba，与服务器一致），向用户内嵌真实MP4。source/A均完整解码124帧、每PTS核验；实际看了0/15/30/45/60/75/90/105/123九个同索引对照及A原生0/60/123。男子衬衫各点均红、女子衣服仍白，靠近→轻吻→分开→贴胸拥抱的大顺序保留；抽查支持换色有效，不证明全视频无闪烁/精确时刻相机保持/音频合格/完全等质。记录 `results_31731/a_20260914T035755Z_5bcd246fd46d4965915dbf8d2c990304/visual_review/observations.json`。独立只读依赖审计已报告current==frozen、release19SHA、原十身份退出和清理告警在正式HTTP200后；完整记录与独立视觉意见待落盘。本次只下载/抽帧/查看/只读检查，未启动B/C。自动跟进授权继续，启动B前仍需核验尚未覆盖的请求字段与新鲜资源门槛，不能因本段重复启动A。

2026-09-14 12:30 最新：用户已明确要求恢复自动跟进，取代此前12:19暂停要求。原 `30213-h3` 已实际恢复 ACTIVE、五分钟间隔未变，复读配置确认，不另建任务。后续继续自动衔接 B/C，只有阶段完成/失败/需决策时通知，不重复汇报未变状态。A `5bcd246fd46d4965915dbf8d2c990304` 已12:18:47正式完成，855.1022686900105秒；12:19:20终态清理完成。root12:29:23只读核验实际result/HTTP200/curl0、正式视频SHA392fd304bab4ca92f8a883f6c85c8a27f7adbd61dd5f1b5a6f678e1dee06d9ba、5031092bytes/1344x768/24fps/124frames/5.207s、十个原进程退出/owned group为空/fresh8idle通过。全frozen依赖终态审计、退出告警顺序和画面质量仍待查；不得把生成完成当编辑成功或等质。B/C尚未启动，本次恢复操作未新launch；下一轮先补全A核验、查看换色效果并按原授权顺序接B/C。证据 `deploy_31731/color_A_completion_status_20260914_1229.json`；以下暂停/运行中均为历史快照。

## 12:19 用户已暂停自动跟进

用户明确要求“安排任务停止，不要自动发了”。`30213-h3` 已实际更新为 PAUSED，复读本地automation.toml确认。不要恢复监测、不要因旧heartbeat或下文历史授权自动接B/C；等待用户继续指示。此次未对31731发出任何操作，远端已提交的A没有被停止，仍由自己的监督器完成/清理。最后一次用户主动查询12:17:42 A为46/49、正式未完成；本次暂停不声称新的远端终态。不要把停止安排任务解释为杀掉服务器实验。

## 12:07 当前唯一换色 A 正式生成中

固定A `5bcd246fd46d4965915dbf8d2c990304` 同下方目录及supervisor/server身份不变。12:04:28新2步smoke完成HTTP200、138.0171418611426秒，视频SHAbd1f648647cf2f53252b81ed8ca6001d3007673a18293b8717ba6b0ef823f078，自己的同服务gate通过；12:04:32.195024开始唯一正式50步，attempts1、completedfalse。root12:07:42.897715轻量核验actualsmoke/result/request SHA、媒体SHA、保存ffprobe一致、两次实际请求同source/prompt/设置、十个原身份存活和fresh八卡健康均通过；全frozen/重ffprobe/正式产物与cleanup终态核验待完成，不声称质量或加速。12:06:31日志5/49仅迭代进度。证据 `deploy_31731/color_A_progress_20260914_1207.json`，tool839b67 exit0。八worker760561–760568，前4 start129501518、后4 start129501519，pgrp/session759877。B/C尚未开始，不改活跃code/proof，不新launch，不运行额外NPU或重CPU/IO。

## 11:58 当前唯一换色 A 已启动

A `5bcd246fd46d4965915dbf8d2c990304`，目录 `/cache/zhonghao/h3/color_trial_v1/results/01234567/shirt_red_couple_124/A/runs/a_20260914T035755Z_5bcd246fd46d4965915dbf8d2c990304`，supervisor759621/start129493821，server759877/start129496385。03:58:00UTC实测server_starting、formal_attempts0、未smoke/正式完成；frozen SHA0f5939af72ecaccca74d14cc5c75bc1f04c6e296284da491bbe4cdc267202e7b。启动journal `/cache/zhonghao/h3/color_trial_v1/color_A_launch.json`，已提交不要重复。B/C尚未启动，A完成后独立核验真实请求/产物/完整frozen/进程退出/fresh8idle才顺序推进。现有heartbeat ACTIVE已绑定此A；不并发NPU、重CPU/IO或修改任何活跃code/proof/模型/VLM依赖。

VLM v2 `4cfa7803ddc3482a8906b91184743384` 已03:55:34UTC完成preserve，root03:56:34核验实际raw/真实video tensor/完整加载及原PID退出和fresh8idle。admission `/cache/zhonghao/h3/color_trial_v1/vlm_admission.json` SHA d34fabfc4a6ed038781bd95f3f08148c63f347c18e23557b4b79d4209bf4b956 已创建且A实际sample_gate通过，不重建/不重跑。VLM status SHA89141b67ba58c5b9c7d342de0244002b2c1b605e31583de9317fddc9c10cb1bc、result SHA8ad3b178c916ce20b9dd5157ce9c760caf27b0afd99e6c4d699f88ae36f4d190。真实耗时decode1.1277343991678208、preprocessing15.186808580998331、loading37.85922156693414、inference81.11992237204686、worker_wall137.76659982698038秒；不得漏计router或给原始A人为加router。旧误判与失败完整保留，单个开发样例不是held-out准确率。下述11:53状态为历史准备快照，最新运行以本节和 `experiment.json` 为准。

用户要求先不改时序，仅将同一 couple124 source 中男子浅色衬衫改红；保持女子服装、人物动作及顺序/时刻、场景视点。全部样例先将 source 视频和编辑指令交给真实 VLM，保持/改变/不确定均必须来自真实模型输出，人工 router_control 只是实验意图标签。30213 不使用，旧生产/模型/历史结果不动。

2026-09-14 11:53:54 当前唯一 VLM `4cfa7803ddc3482a8906b91184743384` 正在运行，目录 `/cache/zhonghao/h3/color_trial_v1/vlm_results/shirt_red_couple_124/runs/20260914T035257Z_4cfa7803ddc3482a8906b91184743384`，supervisor755855/start129463979、worker756293/start129466290，frozen SHA45c68caee90b9e8ad0c24082916d36024eef4c80202293a1f4bfa8bc1a35ea26。新版 action_timeline_v2 的 release SHAa70c2cea17e254a7fb2857ea0e5f98816be50fe5e42756eaa60b8df95d659878，实际 Linux CPU21 VLM +28 ABC 全过。持八卡锁、实际使用0/1，不能并行 ABC；本次换色 A/B/C 均未开始。五分钟现有监测 ACTIVE，固定此 run，严禁重复 launch 或修改活跃代码。

旧真实 VLM `7b7b9babe0ed44c68e1cb54c6a54ef9a` 已完成/独立核验清理，却把换色当 events/change（order/speed/duration 均 unchanged）。这是违反原提示定义的语义误判，不可人工改成 preserve。旧代码和完整核验证据存于 `/cache/zhonghao/h3/color_trial_v1/vlm_code_history/before_action_timeline_v2_7b7b9babe0ed44c68e1cb54c6a54ef9a`；当前只做一次提示定义加强的开发验证，输入视频/编辑命令/模型/ABC 不变，不能重试直到 preserve。若本次失败/change/uncertain，保留结果、报告并暂停；若 preserve，须 root 核验实际 raw/视频证据/全模型加载/当前frozen/原PID退出及fresh8idle，再按 `a/vlm_gate.py` 精确 schema 创建唯一 admission，才顺序 A→B→C。结果不能当 held-out router 准确率或视频必要性证明。

用户希望实测耗时 A > B > C，明确“还得看实际实验”。因此不预设结果、不挑选更快样例冒充成功，同时审查编辑成功/时序保持/画质和重复测量稳定性。共用一份VLM判定以控制ABC变量；VLM开销单列，原始A无router与完整新方案的比较计入router，不给原始A人为添加开销来夸大整体收益。先验证换色；只有该阶段整体和增量加速及质量通过，才推进自然动作重排ABC。

## 已暂缓的后续候选：正常动作重排，不做倒放

用户要求“不要倒放，就正常改变时序的edit即可”。后续采用自然向前的动作次序编辑，不再发整段reverse指令。首选沿用未倒放的couple124 source（SHA4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2，124frames/24fps），从轻吻后贴胸拥抱改为先贴胸拥抱、再稍分开轻吻；强调动作内部自然向前及保持人物/服装/视点/构图。详见 `temporal_source_review/normal_temporal_edit_request.json` 的新自写prompt。

ABC定义和八卡匹配设置不变；先验证A是否保留两个事件并按要求重排，再比较B/C。当前只完成样例/指令选择和需求记录，未部署新sample profile、未发送新推理、未恢复暂停的monitor。旧入口仍只接受lake/explicit_reverse，不能修改旧run的prompt或冒用旧profile：后续执行需独立的新样例身份和校验。历史六次结果与旧ledger保留，其中三个倒放结果不计作正常动作重排结果；历史“本轮完成”只针对已结束的上一轮。

## 历史轮次完成（2026-09-14；六次正式生成与画面抽查完成）

**本轮已完成，不再新launch。** 湖景ABC和倒放ABC全部正式生成/完整性核验/清理/下载完成，C最后的五列9点画面抽查也已完成。C `visual_review/observations.json` SHA `74a818041a6d1bfcae380161e4e34794bc1165a4389461768ac0d86d904bee78`，extraction SHA `20e7fc536f403b13645bdc1b86c4862bff2c0da1613067d17a01733dafb01885`；45native PNG/4contact，4视频前后SHA/124帧24fps/全部PTS过。独立代理实际看C原生0/30/60/90/123，root看全部3张分段contact。此前01:09的C画面待审状态已被本段取代。

**C整段倒放仍失败，但与A/B的失败方式不同。** 九点持续以拥抱为主，女子背朝镜头、男子更正面，裁切更近/视向构图改变；虽以拥抱开始，后半和末尾未回到source开头面对面托头姿势。存在头部姿态和构图变化，不能称静态或冻结，也不能称A/B式正序复刻。C男子衣服保持浅色、没有B的深蓝换衣；女子背部缺可比source视图，不判服装设计改变。视向变化不推出相机移动或人物旋转的因果机制。抽样无黑屏/整体崩坏，但未全程播放、未听音频或评估全部细瑕；不称等质。

root01:14:54.378527 CST又逐一只读对比6份远端原始request/result/status（exit0/chunk72b0b4），同样例ABC的source SHA/路径/prompt/全部fields/generation/host/cards一致，parallelism仅listen_port不同、各一次正式。证据 `results_31731/cross_case_comparability_20260914.json` 保留实际tool stdout；不是重复媒体/权重核验，也不补造未下载的湖A/B request文件。

小结：**工程接通已验证；C相对B两例端到端延迟低4.44–4.64%，但仍慢于dense A 3.79–4.26%，不是总体或等质加速。** 三者均未完成本例整段倒放，不能证明VLM必要或把共同失败归因C稀疏。仅建议下一阶段先profiling定位耗时，并建立A能够成功的时序编辑基线；这里不启动新实验。current_case/active_case为空、无活跃模型；现有monitor已由root于01:16:45 CST通过工具确认PAUSED，随后本地automation.toml复读一致。关闭告警与全部历史保留。

## 完成核验证据（2026-09-14 01:09:07 CST；当时C画面审查尚未完成）

**湖景ABC、倒放ABC共六次正式生成均已完成，没有待启动的新模型。** 最后倒放C `e2a7c9c133ca49a6af00134623e9e719` 于01:03:12.930187正式HTTP200，882.104259646032秒，4,012,027bytes，h264+AAC、1344x768/24fps/124frames/5.207秒，video SHA `5dce91778b5908c29e3b61f883b045aad621b5f2109d94faaa31b2c325c513ad`。01:03:47.663447终态 `formal_completed_review_required`、attempts1、started/completedtrue、cleanuptrue、idletrue、remaining{}、needs_attentionfalse/errornull。

root01:09:07.473515完整reverse C只读核验pass（exit0/chunk8df829）：全部source SHA/C3pins、frozen SHA ba8cf0...与当前完整snapshot（C own tiny db9d/47sources/runtime/transfer/sample/weights headers/logging）、两request/实际HTTP/result/两媒体SHA+ffprobe、实际prefix及full8PID535/800/208加载/线程恢复、strict1与gate一致/strict2与实际formal_strict_metadata.json一致、npu_before/after、十个原身份全退出/owned group无成员、fresh八卡idlehealth全部通过，未重hash大权重payload。首次只读reader因C tiny记录没有size_bytes在导入前停止；修正reader后完整重跑通过，未改remote实验。真实tool stdout见 `results_31731/c_20260913T163728Z_e2a7c9c133ca49a6af00134623e9e719/completion_verification.json`，7份文件已下载并由root核SHA。

**保留关闭告警，不称无告警退出。** 正式编码01:03:12.643已结束，shutdown01:03:15.926开始；全部Traceback和唯一ERROR `Orchestrator did not stop within 30.0 seconds; continuing cleanup`（01:03:46.067）都发生在shutdown之后。实际正式媒体已先完成且最终设备/进程清理验证通过，不把这些关闭日志改写成生成失败，也不删除告警。

同样例倒放请求端到端：A846.0936839610804秒、B923.0886316811666秒、C882.104259646032秒；C比B延迟低4.439917319801911%，但比dense A慢4.2560979200747795%。湖景既有A871.1053479220718/B948.1026493189856/C904.0943806068972秒不变。均为单样例单次、不是纯DiT/NFE或等质/总体加速证明。A/B倒放画面负结果保留，**C五列9点实物画面QA正在独立进行，尚不写C倒放成功/失败或完整质量结论**。

剩余仅C画面审查、汇总和由root处理现有monitor暂停；此刻不记录暂停已执行，不新launch、不训练/扩实验、不重启旧队列。30213取消及全部旧历史保留。以下00:58及更早为历史。

## 历史快照（2026-09-14 00:58:10 CST）

**唯一倒放 C `e2a7c9c133ca49a6af00134623e9e719` 正在正式50步，尚未完成。** root00:58:10.174111独立轻量核验exit0/chunk729fc4：phase `c_50step_request_running`、attempts1、startedtrue/completedfalse；status授权时间00:48:30.811521，正式request.json开始00:48:30.819341。自己的2步smoke于00:48:26.232888 HTTP200/curl0，143.01993357599713秒，4,031,027bytes，video SHA `8ad2670465a487eb8e6e4346e5f594c5c37088902b5899a97b43605b9ee7610a`；request SHA `90f1f6cfc48b26e259e3db769519eb0cf094542756db725d22c17cd3c667a13c`，result SHA `eaf1e22501b099b2dc090514b7c9cdcaa66106b2e6e90fe4a59171930cc691c0`。同服务gate通过；实际烟测文件/视频SHA、HTTP、两份request配置、142068-byte日志前缀SHA `ebab3e69d5e769944a65e90f08a5f5cd03c4598a396b53232aa8c426f4559779` 已核实。

十个原进程身份实测仍活：supervisor537245/start125413215、server537293/start125413847；worker537902/start125418920，其余537903..537909/start125418921，均pgrp/session537293。root逐条解析真实strict日志：smoke每PID1条、当前总计每PID2条；source/target `(37,48,84)`、text6138、source_audio_t0、target_audio_t207、patch `(1,2,2)`，八PID及两请求一致。已存加载记录535/800/208和线程恢复已核；这不是per-layer mask trace。00:56:24实际tail为26/49（不是NFE）；00:58读取器正则吞CR导致无有效新进度，不能写成进度回零或实验失败。此次未做完整load-log重解析、ffprobe重核或frozen完整snapshot比较，留待终态；不并发重IO/测试/新NPU任务，不改活跃文件/proof。

**终态核验提醒：** `formal_strict_metadata.json` 没有status内嵌副本或SHA字段，不能要求不存在的status字段。应从烟测gate取得真实八worker PID集合，核前缀bytes/SHA，再调用 `parse_strict_metadata(actual_log, worker_pids, 'explicit_reverse_couple_124', requests=2)`，JSON归一化后与该实际JSON文件比较。其余两request/result/HTTP/媒体SHA+ffprobe、完整加载日志、当前frozen、cleanup/原身份退出/fresh八卡idle仍须核验。C完成后只做核验、下载、同样画面审查与汇总，再暂停现有monitor；不再新launch。倒放A/B画面负结果、湖景ABC及30213取消保留。

## 历史快照（2026-09-14 00:43 CST）

倒放C仍是下述唯一e2a7 run：root00:42:02.662345实测 `server_starting`、formal0，supervisor537245/start125413215和server537293/start125413847原身份仍活，加载日志持续推进；不重复提交。

**倒放B画面审查已完成，结论也是整段倒放失败。** 四列9点显示主要动作仍为面对面→靠近/接触→拥抱→末尾望向镜头；45/60帧动作阶段与source/A有偏移，不能称逐帧复制。男士浅色衬衫变成深蓝色，属于未要求的外观改变。三个视频完整SHA/124frames/24fps/所有PTS检查通过，独立代理额外看B原生0/60/123帧；root也看了全部三张分段contact。`results_31731/b_20260913T160224Z_642e2c4f790d4505a5fd0940bcfc43a2/visual_review/observations.json` SHA `63f3459a0d9cad62d916513933d9687f192f749746ab4b0f11bf675913f7a1ef`。未连续播放或检查音频，不称等质。B923.0886秒比同样例A846.0937秒慢9.10005%；A/B本身均不会此整段倒放，不能仅用C失败归因同帧稀疏或证明VLM必要。完成剩余C并作同样实物核验/画面审查后汇总，不新增训练或大批量试验。

## 00:38启动和生成记录（其中B画面待审状态已被上文更新）

**唯一倒放 C `e2a7c9c133ca49a6af00134623e9e719` 已单次启动，root于00:38:23.302764核验为 `server_starting`、formal_attempts0，尚未smoke。** started_at `2026-09-13T16:37:28.723039+00:00`；supervisor537245/start125413215/pgrp=session536749，server537293/start125413847/pgrp=session537293，于00:37:34.878279启动。固定状态 `/cache/zhonghao/h3/c_model_trial/01234567/explicit_reverse_couple_124/C/runs/c_20260913T163728Z_e2a7c9c133ca49a6af00134623e9e719/c_status.json`；launch `/home/ma-user/workspace/zhonghao/h3_deploy_31731/c8_reverse_launch_20260914_003728.log`；frozen SHA `ba8cf0dc2c43e143828740b46f81b4a19b2faa2fbdab59a65bbdce88d67ba30a`。仅31731物理0..7/HTTP19100，先本run新2步smoke＋同服务8PID加载/线程恢复/strict metadata gate，再一次50步。倒放source无音频，C实际metadata必须source_audio_t0，不能用湖景209。尚无C烟测、正式视频、耗时或质量结论，禁止重复启动。

00:37:28.512106 root启动前核47sources/runtime/transfer/Ctiny db9d/weights/精确reverse样本、B终态SHA `9a3c1c890da391ecaa220f8aa76712a3975c5a9f3daa88fa548b8adc26b39a67`、B旧身份全部不存在及fresh八卡idle/health/memory均通过；C supervisor自身实际preflight再次通过。现有parser a992及candidate/env/runtime/tiny全部冻结，不复跑tiny/测试或并发重IO/NPU；旧lake队列不重启。

**倒放 B642e 已正式完成923.0886316811666秒。** 00:28:04.678091正式HTTP200，视频4,657,820bytes，SHA `c15f08b33a5419ee704350133afd7441386738e11836aa7c605e71ae481c75cc`；00:28:38.185732 cleanup/八卡idle。root00:34:44.388728完成全套reverse-specific只读核验exit0：frozen=current、两份request/result/实际HTTP、两媒体SHA/ffprobe、日志prefix和完整8PID加载535/800/208及线程恢复、npu_before_formal、旧身份退出/npu_after/fresh8idle均通过。Tracebacks仅在00:28:06 shutdown之后，正式HTTP及媒体先完成；不把这些shutdown日志误报为生成失败。6份文件已下载SHA核验，本地实际核验记录 `results_31731/b_20260913T160224Z_642e2c4f790d4505a5fd0940bcfc43a2/completion_verification.json` 保持原样。B画面审查正在独立进行，尚无倒放成功/失败结论。

湖景ABC与倒放A历史全部保留；dense倒放A已观察到正序事件，不能把之后B/C的倒放失败单独归因C同帧规则，也不能称总体加速/等质或VLM必要性。30213后续全部取消。以下00:21及更早为历史，仅监测上面唯一C run。

## 历史快照（2026-09-14 00:21:27 CST）

**唯一倒放 B `642e2c4f790d4505a5fd0940bcfc43a2` 已进入正式50步：00:12:41.576349授权，request.json于00:12:41.583697开始，phase `b_50step_request_running`、attempts1、completedfalse。** 00:20:17日志24/49（不是NFE），00:21:27.374288再次轻量检查仍正式运行。新2步smoke于00:12:38.189001 HTTP200，136.0186711619608秒，6,270,008bytes，SHA `1039903ab3d873fc17e15c2f1c82d354355887e20d1b0ee59d839ccb7eeeb72c`。同服务gate通过；root独立核实际smoke request/result/video SHA、138130-byte日志前缀SHA `4fa1f30953e3c0c86efd2f5ae39a03dcd813ab16f959283c92db7ea46c08fc31`、保存的8PID各535/800/208和线程恢复、正式请求source/prompt/50步配置。此轮未重新解析完整加载日志或重查大模型payload，完整完成核验留待终态。

supervisor428051/start125202744、server428131/start125203148及8worker原身份均实测仍活；workers428757..428762/start125208293，428763..428764/start125208294，均ppid/pgrp/session428131。正式Qwen6138tokens、1refvideo。尚无正式输出、最终耗时或倒放效果，不启动C或并行重IO/CPU/NPU，不改active代码和proof。倒放A的846.094秒与编辑失败结论、湖ABC和30213取消均不变。

完成核验提醒：B的 `evidence_snapshot()` 除host/run_id/sample/selected还需run_dir，并包含本run `formal_logging.json` 的file_record；内容须等于B的logging_configuration（H3B pid前缀）。用object.__new__只调用读取snapshot，JSON归一化后与实际frozen比较；不要调用会写文件的preflight/smoke_gate或旧lake queue。终态必须核实际两份请求/结果/headers/curl/media SHA+ffprobe、prefix及完整日志parse_full_load_evidence与8PID/535/800/208/线程恢复一致、npu_before_formal绑定、旧身份退出、npu_after释放与fresh8idlehealth后才单次C。B不要求C的formal_strict_metadata文件。详细原绑定见下方历史快照，不能混淆末两worker的start_ticks。

## 历史快照（2026-09-14 00:11:17 CST）

**当前唯一运行是倒放 B `642e2c4f790d4505a5fd0940bcfc43a2`；root 于00:11:17.008724实际读取为 `smoke_2step_request_running`，formal_attempts0、formal_startedfalse/completedfalse，尚无smoke或正式结果。** 服务于00:10:22.158801就绪，00:10:22.168157进入2步smoke；这不是smoke已成功。00:02:24.009605启动（原始UTC `2026-09-13T16:02:24.009605+00:00`），固定状态 `/cache/zhonghao/h3/b_model_trial/01234567/explicit_reverse_couple_124/B/runs/b_20260913T160224Z_642e2c4f790d4505a5fd0940bcfc43a2/b_status.json`。supervisor428051/start125202744/pgrp=session427500；server428131/start125203148/pgrp=session428131，于00:02:27.889965启动。frozen SHA `f949d37d90974140850a23c7cb32e3317f676d77df4791b939b9f7949ed54196`。仅31731物理0..7八卡，HTTP19099；B保留原linear分支和local Softmax，source仍全局可见，不是C固定同帧规则。先本run新smoke与8worker加载/线程恢复gate，再最多一次50步；不重复启动、不刷新活跃代码/env/runtime/tiny证明，不并发NPU/重CPU/重IO。倒放C尚未启动。

**倒放 A 已正式完成，但没有完成要求的整段倒放。** A91c6...于23:55:49.400031正式HTTP200，端到端846.0936839610804秒，4,271,598bytes，h264+AAC、1344x768/24fps/124frames/5.207秒，video SHA `17b83ee2196302e9523e3e54aa239e3d6e7c8d5070c83d7c066e4f8cf06b38d7`。23:56:22.550435终态 `formal_completed_review_required`，一次正式请求、cleanuptrue、8idle、remaining{}、needs_attentionfalse/errornull。root已执行reverse-specific只读完成核验：固定run/host/source/prompt、同服务新smoke、正式产物SHA/ffprobe、当前frozen/source/runtime/transfer/tiny/模型身份和headers、原进程身份退出、fresh8idle均过；不使用硬编码lake的AB/BC完成reader，不重hash大权重payload。下载SHA已核实，本地摘要见 `results_31731/a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd/completion_verification.json`；该摘要记录root实际检查与本地证据，不伪作原始工具stdout。

画面代理已看九个采样时刻和额外原生帧：A仍沿source正序，先面对面靠近/接触亲吻，后贴胸拥抱，末尾仍为拥抱中望向镜头；不是从末尾拥抱回到开头面对面的倒放。`visual_review/observations.json` SHA `c809cc93bbc3a1a8364263ebc3b7069bd5a6242f8c7384ab02de0676535d0eaf`。这是整段倒放失败的明确抽样证据，未连续播放、不声称每个微小动作或全程质量均被检查。**dense A自身已失败，不能以后仅凭B/C也失败就归因于C同帧规则；这也不构成VLM必要性证据。** source仍是未倒放的trim104:228无音频片段，原输入与正式文件未改。

湖景ABC历史保留：A871.1053479220718秒、B948.1026493189856秒、C904.0943806068972秒；C比B低4.642%，仍比dense A慢3.787%，不是总体加速/等质结论。30213全部实验/后续归档取消，旧结果不纳入此对照。旧AB/BC队列和失败C ledger保持终态，不能重启。后续仅监测上面唯一倒放B，完成且独立核实/cleanup/退出/8idle后，再决定按既有授权单次倒放C；不要把lake成功冒充reverse完成。

## 历史快照（2026-09-13 23:50 CST）

**唯一倒放A `91c6c0f98ba84c7c828202d5934176cd` 已23:41:43.297391授权正式请求，23:50:03实测 `a_50step_request_running`，29/49、attempts1、completedfalse。** 它的新2步smoke在23:41:40.044135 HTTP200，131.03361859801225秒，5,872,068bytes/h264+AAC/1344x76824fps124frames，SHA13fd0d90d11f92f30b74001353074e0cf86fd8d7c7ae9c2a86ab4db7133e1e34；同服务gate passed。8个worker记录323631..323638/start125045365，均属于server322752/session322752；root检查supervisor322556/start125039821与server322752/start125040225仍活、父子组不变。正式Qwen实际6138tokens、1refvideo，不沿用湖景6167。没有正式视频/最终耗时/倒放效果结论。保持现有A，不启动B/C或远端CPU/重IO。

resource_expansion_review正在本地只读整理reverse-specific完成核验段，不能直接调用硬编码lake的AB/BC completed_evidence，也不能改其globals。下一轮待A成功且cleanup/PID退出/fresh8idle后，按确切reverse样本/请求/结果与当前证据核验，再单次B。下面23:35为历史加载快照。

## 历史快照（23:35 CST）

补充本地画面核验已完成：bundled Python3.12加PYTHONPATH=workspace/.audio_tools复用已有PyAV18.1.0；C/source/A/B全SHA、完整124帧解码/全部PTS通过，20native PNG及4列contact、observations.json已生成，输入未改。root已看contact，独立agent还看5张C原生图和source/A/B末帧。C雪/冰更多，符合指令部分结冰，但frame123前景树枝结构明显比A/B偏离source；不支持等质/严格相机保持。五帧未见黑屏/大范围畸变，不等于全视频无闪烁或音频合格。不要重跑拒绝覆盖的extract脚本；下文“正处理依赖”为更早状态。

**湖景ABC已全部正式完成；现在只运行显式倒放A `91c6c0f98ba84c7c828202d5934176cd`。** 23:35:14.784841开始，23:35:46实测server_starting、formal_attempts0。固定状态 `/cache/zhonghao/h3/model_trial/01234567/explicit_reverse_couple_124/A/runs/a_20260913T153514Z_91c6c0f98ba84c7c828202d5934176cd/a_status.json`；supervisor322556/start125039821/pgrp=session321375，server322752/start125040225/pgrp=session322752、23:35:18.664307启动且父PID322556。launch `/home/ma-user/workspace/zhonghao/h3_deploy_31731/a8_reverse_launch_20260913_233514.log`。冻结证据SHA155236583775cf8822d17e5755693a1efe8f24d32eedec6b541316a1a76aafd9，配置USP8/ring1/DiTTP1/textTP8/VAEpatch8/HTTP19098。真实reverse sourceSHA4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2/1527004bytes，source未预先倒放。prelaunch核C结束/PID全退出、新reverse A namespace不存在、A3生产SHA和样本gate、fresh8idlehealthy/内存合格后只提交一次。A自己的完整preflight已过，先smoke再formal，不改源码或队列、不重复A。

**湖C19941...正式23:32:09.816281 HTTP200，904.0943806068972秒，23:32:44.639042终态formal_completed_review_required/cleanuptrue/8idle/remaining{}/errornull/needs_attentionfalse。** 正式成片12149617bytes，h264+AAC1344x768/24fps/124frames/5.207sec，SHA014fc5c0da43984e7a903ee6cba199e5a62939d852d04eda07eda0a80356366f。root23:33复用已审只读completed_evidence（不调用旧queue加载/调度，显式传新a992 gate）独立核实际完整frozenSHA、当前source/runtime/transfer/tiny/权重header、正式请求/result/媒体SHA+ffprobe、实际log-prefix及8PID两请求strictmetadata、全8PID完整加载记录、旧supervisor/server/workers身份退出和fresh8idlehealthy全部通过。没有重hash大模型payload。旧C失败和两条旧queue保持终态、不能再启。

湖ABC同样例各一次：A871.1053479220718秒，B948.1026493189856秒，C904.0943806068972秒。C相对B请求延迟减少4.641719833153013%，B/C倍率1.0486766311748854；但C比dense A慢3.787031357747628%，A/C倍率0.9635115167260739。**不是总体加速、不是统计结论或纯DiT/NFE测速；质量待目检/正式评测。** C正式video/result/request/status/gate/strict/log已下载至results_31731/c_20260913T150426Z_19941bde9a10446296621418e68a09bc，root本地video/result/status SHA全过。新C抽帧脚本已准备，默认Python缺av，sourceagent正复用现有依赖处理本地画面核验；不影响远端A，无需安装新库或动远端。

后续：只监测上面唯一reverse A；成功清理/退出/freshidle及精确sample/source/request/results核验后，依次B然后C。现有AB/BC的completed_evidence也硬编码lake，**不能用于reverse，也不能改其globals/pins**；可按原函数检查规则核reverse-specific gate与实际manifest。不能用湖源成功冒充倒放能力，先A/B有无倒放效果再判断C。C固定帧规则/原linear/权重不改，不训练、不蒸馏。以下23:21及更早为历史。

## 历史快照（23:21 CST）

23:22:21独立轻量复验通过：新C正式16/49，attempts1、未完成；实际烟测video SHA/16,268,949bytes、142379-byte日志前缀SHA、8PID在smoke/full各1/2条strict metadata、请求/结果/run/source绑定、supervisor/server/8worker身份和父子组均一致，无新ERROR/Traceback。没有模型导入、测试或远端写入/新任务。

**新C `19941bde9a10446296621418e68a09bc` 已23:17:05.685704授权一次正式50步，23:17:05.716179进入 `c_50step_request_running`。** 23:21:33实时状态formal_request_attempts1、formal_50step_startedtrue、completedfalse；23:20日志已11/49，不能将日志49个迭代当NFE。新2步smoke于23:17:00.520183 HTTP200，181.01941359997727秒，SHAebfcd04aee0a55ec425c82b5edf6743dd889070c86c3295022a01c8904e12d1e；同服务8worker加载/线程恢复/strict metadata gate passed，冻结的烟测日志前缀142379bytes/SHA77d4e092ffe3d092c46ba0c248b7d01eb78bba97b162365ef75e6efb346dd6ee。正式请求的8条实际metadata均23:17:35写出，37latent帧/text6167/sourceaudio209/targetaudio207。这是新run自己的smoke与gate，不是复用旧失败run。正式输出/最终耗时尚无；不启动reverse、不运行CPU测试或大IO，不改活跃文件及proof。以下23:14/23:08为更早阶段。

**23:14:20实测：新C `19941bde9a10446296621418e68a09bc` 已进入 `smoke_2step_request_running`，formal_request_attempts仍0。** 23:13:59.463034 server healthy，23:13:59.498365进入烟测。准确started_at23:04:26.971262，固定状态 `/cache/zhonghao/h3/c_model_trial/01234567/lake_snow/C/runs/c_20260913T150426Z_19941bde9a10446296621418e68a09bc/c_status.json`。23:11独立审计确认worker211794..211801全部存活/8卡healthy，supervisor211295/start124855040和server211371/start124855681父子关系正确；6candidate与frozen源码清单一致。继续唯一现有运行，不能重复请求。下面23:08加载阶段为更早快照。

**新 standalone C 已唯一提交，run `19941bde9a10446296621418e68a09bc`，正在模型加载。** 23:04:26启动supervisor211295，23:04:33.220953启动server211371/start124855681；23:08:31实测supervisor仍活。最新状态 `/cache/zhonghao/h3/c_model_trial/01234567/lake_snow/C/c_status.json` 必须先匹配新run_id，再读固定per-run。启动日志 `/home/ma-user/workspace/zhonghao/h3_deploy_31731/c8_lake_parserfixed_launch_20260913_230426.log`。只用31731物理0..7八卡，不再手工提交C。

日志解析修复已部署：c_trial_gates.py SHA `a992a398458a4a9612e737ef5ac1a0713248ea0332fc2fe4a7048bef916820df`，test_c_trial_cpu.py SHA `3488282b4ee120b8abdea9b00e3b061b03f0603bb0d43e7e099cdf42cdbbd316`。作者、独立审计、root各49本地测试通过；远端Linux49项13.190秒全过0skip，并直接重放未修改的旧server.log，8个真实PID均被正确接受。23:04:26.731621 prelaunch全套只读gate/47source records/当前host/权重/runtime/原Ctiny证明通过，fresh8卡idlehealthy，旧C及旧queue所有记录PID退出后才提交。只改有界0..64 ASCII点前缀解析和测试，完整fullmatch/PID/字段/请求数限制保留；candidate、模型、注意力、权重、sampler、supervisor、launcher及tiny证明未改。原38Linux测试证明保留为历史，新修复证明另见 `deploy_31731/c_model_trial/parser_fix_verification_20260913_2304.json`。

新C须重新完成自己的同服务2步smoke、8worker加载/线程恢复/strict metadata gate后，才自动发一次正式50步。**新smoke/正式结果尚未核实，旧C只smoke成功不能冒充正式C。** 从新run开始冻结所有运行代码、candidate、env和proof；不并发NPU/重IO/CPU测试。旧C e13387...与旧BC队列均终态failed/已退出，其ledger和旧gate备份保留，禁止改pin、重启或删除。当前运行不依赖旧BC队列。

A/B湖源正式结果不变：A871.1053479220718秒、B948.1026493189856秒，B本次慢8.839%；尚无C正式测速。湖C完成并核实产物/cleanup/进程退出/8idle后，依次用既有run_a_trial.py、run_b_trial.py、run_c_trial.py的 `--group 01234567 --sample explicit_reverse_couple_124 --allow-npu` 做显式倒放ABC，每次只启动一个。现有AB/BC队列及completed_evidence硬编码lake，不能用于reverse。source仅trim104:228/setpts/-an，未预先倒放，SHA4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2；reverse预期source_audio_t0。先确认A/B也有倒放能力再归因C失败。source-dependent研究继续沿用6578/1374的2x2边界与长度限制，不声称已有VLM结果。

## 历史快照（22:59 CST；其“尚未部署/当前无C”已被上文取代）

**C旧run e13387...在烟测后的日志校验失败，正式0次，已清理退出。** 两步smoke于22:55:32.330304 HTTP200、174.02241408801638秒、124frames1344x76824fps，16,539,087bytes，SHA2c4c358e367a5f3be1fbf1a5fa807048134d2659768f3554321deb775d656f22。随后RuntimeError: Unlabelled, oversized or malformed real C strict metadata record，22:56:09.368073 phasefailed/cleanuptrue/8idle/remaining{} /needs_attentionfalse，formal_attempts0。BC队列也22:56:36.665489failed、own_child_exitedtrue。root确认旧queue116863/C119429/server119694/workers120387..120394全部/proc不存在，fresh8卡idlehealthy；不能重启旧队列/删ledger。

**首因已证实是严格行解析误拒绝。** 实际8条metadata均393或392字符，唯一首行 `.H3C pid=120394 ...` 前导一个ASCII点，其余7行正常。8worker字段完全一致：patch(1,2,2)、source/target(37,48,84)、text6167、sourceaudio209、targetaudio207。旧fullmatch只接受H3C行首，故在字段检查前拒绝。stdout/stderr合并写server.log已证实，但点的具体producer尚未定位，“并发进度点拼接”仅推断，不能称已定位生成代码。

resource_expansion_review仅修改本地c_trial_gates.py+CPUtests，允许有界0..64个ASCII点前缀，保留fullmatch/8PID/字段/4096payload上限/请求数，补真实8line与未知/超限前缀负例；auditagent独立复验待新SHA。模型candidate、run_c_trial.py、launcher、env、权重、sampler、5tiny证明均不改。远端旧gate SHA45e925...已备份/cache/zhonghao/h3/c_model_trial_code_history/c_trial_gates_45e925f83453e225.py，新gate尚未部署。原BCqueue pin旧gate且已经终止，禁止修改/重开；修复经测试/实际log重放后单次新standaloneC（source现有env、先新smoke再formal），可保留原Ctiny证明因tiny相关文件不改。原始server.log/status/smoke result已下载results_31731/c_20260913T144351Z_e13387c0c6184863b9d0705adb3f918e_gatefailed供root回归，不改远端历史。

以下22:53及更早为历史；当前没有正在运行的C或队列。

## 历史快照（22:53 CST）

**C e13387...已22:52:38.305546进入两步烟测，尚未正式50步。** 22:53:06实测相同host/boot，队列C_full_running、Ctiny1/Cfull1，child119429/start124729450身份未变，Cphase smoke_2step_request_running、formal_attempts0、formal_completedfalse、errornull。22:52:45实际Qwen记录6167tokens/1refvideo；没有smoke结果或strict metadata gate结果，不能预称通过。继续唯一队列监测，不改活跃文件、不重复请求。

后续reverse只读审查已完成：三个现有run_a_trial.py/run_b_trial.py/run_c_trial.py均接受--group01234567 --sample explicit_reverse_couple_124 --allow-npu，不需改代码。source只是trim104:228/setpts/-an而未reverse/areverse，固定SHA及124frames24fps无音轨；同target1344x768、24fps、seed4101及USP8/textTP8/VAE8。A请求等待上限1800s、B/C3600s（不是测速差异）。现AB/BC队列和completed_evidence硬编码lake及LAKE_SHA，不能用于reverse，不能改活跃globals；湖C完成/清理/队列退出后可按三个独立supervisor逐个运行并手工核reverse-specific source/prompt/request/result/proofs/cleanup。C需source_audio_t=0。它是显式时序编辑边界对照，无GT/不证明VLM必要性；若C生成但未倒放，必须先对照A/B是否能倒放，不能仅凭C失败归因为同帧规则，运行异常仍要查实现。所有代理均已完成，没有待取的实现任务。

以下22:44及更早为历史。

## 历史快照（22:44 CST）

**C完整模型已真实启动，当前加载，禁止重复启动。** root22:44:11独立调用真实c_trial_gates.tiny_gate，Ctiny db9d...全部数值/mask/8rank/source/proof/cleanup门槛通过，resultSHA70fed339ec809239c4c58f8a0a853f1cd5e4682773b8b90edfbe37f40a89c894。队列已C_full_running、launches tiny1/full1，自动绑定唯一C run e13387c0c6184863b9d0705adb3f918e：/cache/zhonghao/h3/c_model_trial/01234567/lake_snow/C/runs/c_20260913T144351Z_e13387c0c6184863b9d0705adb3f918e/c_status.json。22:43:51.788506开始，supervisor119429/start124729450/pgrp=session119429，server119694/start124732164；当前server_starting、formal_attempts0/errornull。先同服务2步+8worker535/800/208+actualstrictmetadata再50步。所有C/queue/A/B/env/media/runtime/tiny依赖冻结，不运行并发NPU/正式时的CPU或复制。root/代理均已完成本地审计，无待取的实现任务。

以下22:43及更早为历史；当前不再处于“队列未提交/Ctiny未开始”。

## 历史快照（22:43 CST）

**BC队列已真实提交，Ctiny真实八卡已通过。** 固定queue/cache/zhonghao/h3/serial_bc_queue/b_6362ce1811da4db69d335b3b3ca5d1a4_c_tiny_c_lake/queue_status.json，queuePID116863/start124713883，22:41:16.819505创建。launchlog/home/ma-user/workspace/zhonghao/h3_deploy_31731/bc_chain_launch_20260913_224055.log。27真实LinuxCPU全pass0skip0.431s，实际23manifest/env/B证据复验过；22:41:20确认B完成退出清理，随后只启动一次Ctiny。

Ctiny run db9ddbbbedfc4aacb13ed6823c69b2fe，固定/cache/zhonghao/h3/c_validation/01234567/runs/20260913T144145Z_db9ddbbbedfc4aacb13ed6823c69b2fe，supervisor117333/start124716813/pgrp=session117333，worker117689/start124719055。22:41:45.852961开始，22:43:06.238906 completed，validation78.201510102s，resultSHA70fed339ec809239c4c58f8a0a853f1cd5e4682773b8b90edfbe37f40a89c894，validation_passedtrue/cleanuptrue/8idletrue/remaining{} /needs_attentionfalse。两prefix合法布局所有8rank logpassed。这是独立C小张量实际SP8/NPU正确性，不是完整视频/速度。22:43:14队列仍C_tiny_running因30s采样尚未读终态，launches tiny1/full0；它会核验真实Cgate及退出后自动接完整C。root当前只读独立Cgates.tiny_gate复核及查询Cfull新状态，勿手动重复Ctiny/full。活跃queue/C相关文件与proof全部冻结。

以下22:37及更早为历史，以live队列绑定状态为准。自动跟进已更新到唯一BC队列，不再等待B。

## 历史快照（22:37 CST）

**B6362正式完成并已独立复核，不再运行。** 正式22:30:17.163422 HTTP200，948.1026493189856秒，11,740,798bytes，SHA ca8aa801cb1c96026b237f4f1dbb93d8e4846071338aae3796ae77c24537d2fa；1344x768/24fps/124frames/5.207s h264+AAC。22:30:50.856962清理完成，8idle、remaining{}、needs_attentionfalse。root随后用真实AB完成reader复核run/source/prompt/weights/runtime/tiny/产物/监督器退出，且fresh8卡idlehealthy。B输出/result/status已下载results_31731/b_20260913T140300Z_6362ce1811da4db69d335b3b3ca5d1a4，root独立SHA验证。对同八卡A871.105s，本样例本次B慢8.839%，A/B倍率0.918788，不是加速。5帧source/A/B实际目检雪景有效、构图总体接近，无醒目大面积崩坏/异常蓝边，不能说等质或全视频temporal/audio合格；详见该目录visual_review。没有采用30213结果。

**C vendor已经真实部署，不能再次运行部署器覆盖。** /cache/zhonghao/h3/candidates/c_v1/deploy_manifest.json statecompleted，22:32:49.795626完成，2779files/54,252,496bytes，所有SHA/size/nohardlink/source前后unchanged通过。5tiny生产文件已部署/cache/zhonghao/h3/c_validation_code，3full生产+CPUtest已部署/cache/zhonghao/h3/c_model_trial_code；38项真实LinuxCPU全过0skip，13.117s，47source manifest与5tiny文件核验，两launch bash-n通过。详见deploy_31731/c_model_trial/deployment_verification.json。Ctiny NPU仍未开始。

**BC串行队列正在部署前最后核验，尚未提交。** 新serial_bc_queue独立复核指出Cformal strict证明必须重解析真实log而非只hash文件；已修，27本地CPU全pass，root全读修复+独立复现tampered JSON会拒绝。新queueSHA67b5b867711af1a1e235290cd602713e677743ff43db96afcafd3b4c1b054a8f；bootstrapSHA d8ca943a9543f612646ad31ddca230675a4b3e694eb139f15277b196334c53cb。已上传/cache/zhonghao/h3/serial_bc_queue_code，现在执行27项LinuxCPU与真实production/B evidence只读preflight（execsession6493待收结果）。全部通过后可一次nohup bootstrap_bc.sh queue；不要与手工Ctiny重复。固定ledger/cache/zhonghao/h3/serial_bc_queue/b_6362ce1811da4db69d335b3b3ca5d1a4_c_tiny_c_lake，B→Ctiny一次→Cfull一次；无自动部署/重试/未知latest跟随。旧ABfailed ledger保留，不能重启。

以下22:16及更早为历史。

## 历史快照（22:16 CST）

**B6362...正式50步已22:14:29.055894开始，attempts1。** 22:15:31实测phase b_50step_request_running、日志1/49约18.22秒/iter、8卡healthOK，worker5207至5214，尚无正式结果。两步smoke已成功并通过同服务8worker gate才进入正式；不把smoke的两步或日志单iter当完整算法加速结论。现在暂停远端C复制/CPU测试等，正式计时仅轻量只读监测。

**C部署29项真实LinuxCPU全pass0skip，readonly inventory通过。** root22:14:16.098–18.843完成，suite0.566秒，B前后均smoke/formalfalse，未与22:14:29正式重叠，但不能声称smoke无争用。真实B树2777files/54,234,768bytes，叠C后2779files/54,252,496bytes，B4/C6pins及20/128MiB上限/无链接/特殊文件通过，Cvendor仍未创建。三个部署新文件已传到/cache/zhonghao/h3/c_adaptation，生产脚本SHA6a43c17201a0a5a693c9bc8b6286ac009900916fd7f65bea524a2f7d763db7dc；root全读+独立审计通过。详细c_adaptation/deploy_candidate_cpu_31731_status.json。等B完成后可执行脚本--apply一次新c_v1；不要覆盖已有或盲删失败staging。

resource_expansion_review在本地准备新deploy_31731/serial_bc_queue：固定本次B成功+清理→Ctiny一次→Cfull一次，复用已有证明和正确env。尚未完成/部署/提交，不说C已自动排队。Ctiny五文件和Cfull三文件正式目标目录也尚未部署，root须部署核验及审查后再用；无需强等新队列，现有heartbeat可在B后推进。其本地进行中工作不要重复建造。

以下22:08及更早为历史快照。

## 历史快照（22:08 CST）

**A8正式完成，B8新run已启动。** A run294205bef2e043dc84b41315e544b36d 于21:56:21.579085正式HTTP200，request端到端871.1053479220718秒；50 requested steps，纯DiT时间/NFE未测。21:56:55.357240清理完成、八卡idle、无remaining/needs_attention。输出11,934,240字节、1344x768/24fps/124帧/5.207秒、h264+AAC，SHA1df518c92da95f9f926d1a82284199a6017660a46a502502c08b12219c52c9a8。已下载并核验到results_31731/a_20260913T133502Z_294205bef2e043dc84b41315e544b36d/a_50step.mp4，source与原SHA一致。5帧有限目检显示雪景/大致构图及镜头保留；未做连续播放、音频或完整时序评价，观察见同目录visual_review/observations.json。无B/C正式结果，无算法加速比；30213旧结果不采用。

**旧AB队列终态失败，不重启或删除ledger。** 队列只启动一次B run4ab363e41a6b4dc0a3e8bc456f7d4661，于21:57:29预检报Missing required executable:npu-smi，未加载模型/占卡/请求；根因队列PATH漏/usr/local/sbin。B旧supervisor787和queue4094406均已退出，cleanup+八卡idle已核验。队列21:57:59failed，历史保持原样。

**修正启动环境后的唯一新B**：root source现有env_h3_31731.sh（未改env），实测npu-smi=/usr/local/sbin/npu-smi、ffmpeg=privatebin，独立复核A完整成功/清理/产物后于22:03:00.786028启动run6362ce1811da4db69d335b3b3ca5d1a4。固定目录/cache/zhonghao/h3/b_model_trial/01234567/lake_snow/B/runs/b_20260913T140300Z_6362ce1811da4db69d335b3b3ca5d1a4，latest/cache/zhonghao/h3/b_model_trial/01234567/lake_snow/B/b_status.json。supervisor4692/start124486422，server4708/start124486821；22:07:47只读实测server_starting、preflight通过、formal_attempts0。launch log/home/ma-user/workspace/zhonghao/h3_deploy_31731/b8_lake_pathfixed_launch_20260913_220300.log。不要重复启动；模型/权重/attention/配置没变。先本run2步烟测+8PID535/800/208与线程恢复证明，再最多一次50步。B运行期间所有B依赖/env/runtime/tiny证明冻结，不开并发NPU或重CPU/复制大资产。

**C进度**：Ctiny35项真实Linux CPU全过（详见下方历史）；新完整C入口deploy_31731/c_model_trial的38项本地CPU全过，root已全读3运行文件。C独立vendor部署脚本c_adaptation/deploy_candidate_31731.py本地29项=25pass/4OSskip，正在独立复核，尚未真实Linuxtoy或部署。Cvendor/Ctiny NPU/C完整生成均尚未运行。C须先自己的当前host/六candidate/五verifier绑定的真实8卡tiny通过，再其2步→同服务8worker实际strict metadata→50步。不能复用Btiny充当C证明。后续继续显式倒放ABC；source-dependent自然视频6578/1374与2x2自写指令已目检，但事件跨度超5秒，未VLM/未改当前source。

以下21:48及更早均为历史，以顶部和远端实时状态为准。

## 历史快照（21:48 CST）

**A294...正式50步已在运行。** 两步smoke于21:41:46完成HTTP200，request端到端149.026332秒，17,942,745字节，h264+aac，1344x768/24fps/124video帧/5.207秒；outputSHA2b01aeb7a12e41d271bcc01f903b1064784330d63dd43136a4ab4613f55992d7。8个同服务worker PID4089983到4089990、source/proof gate全部通过。21:41:50.442889开始正式50-requested-step，最新日志18/49、约16.86秒/iter；这是日志迭代数不是实测NFE。八卡healthy且实际忙，无formal输出/加速结果。queue4094406仍waiting_for_A、b_launch_attempts0。不要改活文件、开新NPU、迁移或CPU密集验证影响正式计时。

**C35项真实Linux CPU测试全部通过，0skip/0fail/0error。** 21:40:18.233到21:40:47.779、29.545秒外层（suite11.683秒），日志 `/cache/zhonghao/h3/c_adaptation/validation_31731/cpu_review_20260913T134018Z_47d71a3f/cpu_tests.log` SHA0240b85e4a5475889d09712668d2b4c94051d318677bd77b47a4e84c853d130e，root已独立只读核验状态/exit0和logSHA。NPU初始化false，OMP/MKL/OPENBLAS1，测试期间与A烟测重叠但没有正式50计时重叠，不能说smoke无争用。local详细证据c_adaptation/validation_31731/cpu_validation_status.json。Cvendor/C8tiny尚未部署/运行；仅新CPUreview目录写入，原helper只读。

resource_expansion_review正在独立本地新增deploy_31731/c_model_trial（HTTP19100、同8卡/sampler/StageB），必须真实Ctiny通过才可launch。不改A/B/queue或现有Ccandidate；审计是否足够实际strict metadata/layout日志，不凭理论接线声称gate通过。等待报告/rootreview，禁止直接跑未通过C。后续显式倒放ABC仍按既定sample，不替换湖源或在A/B活跃时编辑sampler。

以下21:40及更早为历史快照，以此顶部和远端实时状态优先。

21:40实测：A294...已21:39:17进入smoke_2step_request_running，source转码错误已越过，21:39:23 Qwen日志6167 tokens/1refvideo；截至21:39:59尚无正式50step或视频结果。queue仍waiting_for_A/b_launch_attempts0。root已全读C8tiny五个生产文件，未见新增阻碍；C LinuxCPU复验代理若尚未启动则暂停数值，以免A马上正式计时产生争用。C尚未NPU/完整模型。

21:38补充：**A→B队列已真实提交，勿手工重复启动B。** `/cache/zhonghao/h3/trial_queue/a_lake_to_b_lake_294205bef2e043dc84b41315e544b36d/queue_status.json`，queue PID4094406/start124335724，21:37:53创建、phasewaiting_for_A、b_launch_attempts0。固定绑定本次A294...，只有A正式成功+清理+idle+原supervisor退出及证明复核后最多一次B；失败不重试，不能删除ledger重发。生产serial_ab_queue SHA aca213d2f11c694ac8a6d0b105d38262dd9be1abd572967c6adb96de56727762；22tests已在Windows和31731Linux真实通过。队列等待不占8卡，代码/A/B/媒体/env/proof均冻结。当前A仍server_starting，未正式生成。自动监测已同步新队列及本次A。

新A已21:35:02重新启动：run `294205bef2e043dc84b41315e544b36d`，固定目录 `/cache/zhonghao/h3/model_trial/01234567/lake_snow/A/runs/a_20260913T133502Z_294205bef2e043dc84b41315e544b36d`，supervisor4089275/start124318608/pgrp=session4088644；server4089289/start124319023。当前server_starting。它取代下面21:26的8237...，不得重复提交。

第二次A8237...已正常加载完，但21:30:14首次烟测在source转码处Unknown encoder libx264，1秒HTTP500；未进入正式生成。之后自动清理，8卡idletrue。首因是系统FFmpeg缺编码器，后续HCCL栈不是首因。已新增私有 `/cache/zhonghao/h3/bin/ffmpeg` 包装，执行现有imageio_ffmpeg7.0.2-static（libx264/aac/libopus），不改系统、模型或env脚本。原PATH已经privatebin优先。真实湖源同参数转码成功，`/cache/zhonghao/h3/media_validation/preprocess_0C4V1X7X/prepared.mp4`，h264/1344x768/24fps/124帧/5.167秒。只作CPU媒体验证，不是模型输出，不改原source。

A gate增加wrapper+实际binary可执行和哈希冻结，32CPUtests通过；新SHA b9bfed7e602c9aa030d6838fc8188cfcea2707114a78acc783791978138d029f。wrapperSHA ba503ab283cfd941f93726b7a46c731077f2c70e4bd8908c48dfd0ad5fda71f1，binarySHA6bb182d0d75d23028db82e9e4f723ca69b853d055698486e6984ddb2c06fb8ce。B gate只更新A依赖pin，SHA b0e5b1aafe83b15a9b0fcaa3775f690ec2076c014a9f08842a12f8bb32296507，27CPUtests再次通过，已部署。env/runtime/tiny证明未改，新A启动前实际gates通过。

A→B单次队列22CPUtests已通过，root全读review，正在绑定新294205...并准备Linux测试/部署；当前尚未提交，不能预称已排队。C8tiny本地35CPUtests=32pass/3skip（无torch和Windowsflock），生产不改，实际NPU待做。活跃A相关文件/媒体wrapper/runtime/tiny全部冻结。

以下21:28及更早均为历史快照，不用其中旧run作为当前任务或队列前驱。

## 历史快照（21:28 CST）

用户已要求“继续跑”。31731修正后的A8已真实启动，run `8237f7b4a900494098473e95d6085f48`，21:26:02 CST开始，21:26:05进入server_starting；固定目录 `/cache/zhonghao/h3/model_trial/01234567/lake_snow/A/runs/a_20260913T132602Z_8237f7b4a900494098473e95d6085f48`。supervisor4079992/start_ticks124264548；server4080025/start_ticks124264934。当前仅启动阶段，未声称smoke或正式结果。不得重复启动。

旧A8确定根因：8个DiT rank都创建textTP group，4人组导致rank4到7没有cpu_group；VAE也要求单rank或完整world8。只把textTP和VAEpatch从4改8，USP8/ring1/DiTTP1、原始模型和attention数学不变。30项CPU tests和Bash语法通过；新启动前复核旧PID退出/八卡idle，所有preflight gates通过。A代码、原始source、Bcandidate、env、runtime/tiny latest证据在活跃run期间冻结。

C的5项实际torch CPU tests已在31731通过（9.683s、exit0），不是NPU证明。C独立8卡tiny脚本在本地准备，须等现有组释放再验证。B8入口27项CPU tests通过，textTP8/VAE8，HTTP19099和独立输出；正在root审查部署。A成功并清理后启动一次B的远端串行队列正在准备，尚未提交；不要预称已经排队运行。

30213全部实验和归档已取消，旧结果不采用，文件/共享资产不删除。source-dependent搜索找到Perception Test两段自然实拍6578/1374，事件顺序相反；自写两条相反目标指令可构成2×2最小路由对照，非官方编辑pair或GT。视频长于当前5.167秒配置，暂不强行裁剪，不改当前ABC源。

以下21:16及更早内容只保留历史；其中“当前没有任务”“textTP4/VAEtile4”“归档30213”均不再适用。

## 历史快照（21:16 CST）

**最高优先用户要求：30213实验停下丢弃。** 已21:15:05只读确认旧B六PID全部退出、2367无NPU任务；取消30213全部后续实验、打包和结果迁移，旧结果不纳入新的ABC。实际produce/receive从未执行。只保留已存在历史日志/代码，不删除共享资产。任何下文“归档30213”均被取消。

**A8初次启动失败，当前没有生成任务。** Runa29f72963b7645a7adbcd2179db39e28于21:14:23 phasefailed，startup exit1，formal_startedfalse、cleanuptrue、remaininggroups{}、8卡idletrue。首因Worker4-7在group_coordinator.py121 assert cpu_group is not None，正在审计textTP4/VAEtile4在8rank下的group覆盖（config接受不等于实际支持）。不要盲目重启；只读根因后最小launcher参数修复/tests，再新run。原失败证据保留不伪改。此前“冻结活跃A8代码”已因run退出不再阻碍必要修改；tiny/原始src/env仍保持不变。

用户说要进入Windows电源菜单睡眠。已明确告知：本机Codex、SSH新操作和定时跟进会暂停；远端已提交nohup任务可以独立继续，但目前A8已失败、B/C远端自动队列未建立，不能承诺睡眠期间仍会做ABC和检索。没有修改用户电源设置。机器唤醒后先查真实状态再继续，不伪称持续在线。

B8入口正在resource_expansion_review本地新建b_model_trial，不能直接沿用已失败的textTP4假设；与31731根因审计代理h3_b_smoke_result_audit同步。source_dependent_temporal_edits继续找必须看source的样例，禁止30213新操作。C本地目录已复制31731/cache/zhonghao/h3/c_adaptation，正在真实5项torch CPUtests，不是NPU。

**A8已真实启动，不能重复发起。** 31731 run`a29f72963b7645a7adbcd2179db39e28`，路径`/cache/zhonghao/h3/model_trial/01234567/lake_snow/A/runs/a_20260913T131042Z_a29f72963b7645a7adbcd2179db39e28`，21:10:42开始，21:10:46进入server_starting。监督器4069383/start_ticks124172613；服务4069397/start_ticks124172986。动态状态`/cache/zhonghao/h3/model_trial/01234567/lake_snow/A/a_status.json`，launch log`/home/ma-user/workspace/zhonghao/h3_deploy_31731/a8_lake_launch_20260913_211042.log`。使用全部8卡，USP8/ring1，textTP4/VAEtile4原样保留。先一次2步烟测，再same-service gate，通过才一次50步，无自动重试。当前无正式结果。

以下旧记录中的待启动/待报告已过时。当前model_trial三生产文件、validation_code、Bcandidate、env脚本及runtime/tiny报告均冻结；不能在活跃A8时改写或重跑验证刷新latest。ROOT已30CPUtests、完整静态复核、远端bash-n和实际全部evidence gates通过才启动。

21:08补充：CPU runtime持久化已真实通过，`/cache/zhonghao/h3/runtime_validation.json`，run`20260913T130706Z_5d220bb2fc3649c684d6026760464078`，reporter SHA`0e318b32b934b09a65f2dcf8273c910503d2b6f061497b09026012d6af8d1a53`。八卡tiny root重新verify80项通过，resultSHA`fcc2ba524eafe0f0ec8fddb57eda67233871f3cbd8609fc330e44bb5acbc1090`。fullA8三文件已部署`/cache/zhonghao/h3/model_trial_code`，30本地CPUtests、远端bash-n和所有read-only evidence gate通过（81模型文件）。待最后功能复核后启动一次`run_a_trial.py --group 01234567 --sample lake_snow --allow-npu`；不要重复启动。输出新路径`/cache/zhonghao/h3/model_trial/01234567/lake_snow/A`。代码一旦启动全冻结，且不能再次刷新runtime/tiny报告使same-service evidence失效。

归档脚本已完成本地17pass/1Windows-symlinkskip；正在两端个人deploy目录部署和31731 Linux CPUtoy，尚未执行produce/receive。须等30213Bterminal/cleanup/PIDs exited后再封存。B21:04进度35/49，可能21:11:21触及1800秒timeout；不改活跃代码、不重复请求，按真实终态记录。归档保留source不删除，旧未sealed失败run显式排除并保留在原服务器。

最新实测：私有bootstrap修复后CPU runtime stdout已真实exit0通过；持久化report wrapper尚在解决stdout日志隔离后再部署。tiny8已21:03:18启动且21:04:45完成，run_id`94ab044fd78b4c50884dd2c52dbbc13b`，`/cache/zhonghao/h3/validation/01234567/runs/20260913T130318Z_94ab044fd78b4c50884dd2c52dbbc13b`，8rank同USP8组/真实HCCL全部通过，cleanup及cards idle复核true。tiny生产代码和env此后冻结，避免使proof失效。完整A8仍待model_trial冻结、root审查及report落盘，尚未启动。

用户最新指令：若12卡不可行，全部迁到31731。已确认原A扩散executor的ray/external_launcher未实现，worker写死localhost且rank=local NPU；H3的56heads不支持USP12，USP4×Ring3还有padding/mask和Ascend核未验证，不能用加参数凑12。

因此全部新任务统一31731单机8卡01234567，先重跑同湖景→雪景A8，再同硬件配置B/C及下述明确时序编辑。30213现有正式B只收尾、保存、归档；不再启动30213新任务，不中断别人的任务。旧4卡结果不得与新8卡直接算算法加速比。当前31731尚未通过runtime检查，不声称A8已启动。

- 31731传输已于20:46:04完成，164203385157字节/24019条核验；环境解压和563处路径修正已完成。
- 首轮CPU runtime检查失败于继承`/usr/local/Ascend/cann-8.5.2/lib64`。resource_expansion_review只在本地修私有bootstrap并加测试；root审查部署，不改系统CANN/共享环境。
- source_dependent_temporal_edits在本地将validation/model_trial改成固定8卡，保留host/PID/资源/原始A源码/当前tiny/runtime证据门槛，先lake A8。不在未验证前直接开模型。
- h3_b_smoke_result_audit准备已完成历史run只复制不删除的归档方案，并只读跟进当前B。
- C隔离实现已完成26项纯CPU测试；5项真实torch测试因本地无torch跳过，尚无NPU或完整模型验证。
- 明确倒放样例已20:59准备成功，`/cache/zhonghao/h3/data/explicit_reverse_couple_124/manifest.json`，source SHA `4fb53ba396d3695eb3bb4eadb0b9fe64dc88592e1c8baf3f6cab59a43f66b0a2`，1280×720、124帧、24fps、5.167秒、无音频。用私有imageio_ffmpeg替换失效系统编码器；原source未改，没有生成GT。
- 私有bootstrap路径修正SHA`c02ecc60ad3ff7a8f6730dab97a282c1cce26a0d1890905a6e392c4d04109e9d`，12项本地CPU Bash测试通过，已同步31731两份env_h3_31731.sh；真正CPU runtime checker正在重试，尚未认定通过。

以下20:45记录保留为历史上下文；其四卡分组、双主机继续跑新任务的调度已被上述最新要求取代。

## 历史并行任务（2026-09-13 20:45 CST）

用户最新明确：先用明确写出时序变化的普通source+edit跑ABC，同时继续找必须结合source才知道是否改变时序的样例；不再等待后者才做前者。

## 正在运行

- 30213／2367：湖景→雪景的正式B已于20:41:21开始，动态状态`b_model/b_formal_status.json`为`b_50step_request_running`。此前新2step HTTP200、189.02708秒、124帧；四worker PID与535/800/208及CPU加载线程恢复gate均通过。A早已完成，C尚未进入模型试验。
- 31731：H3资产传输producer140590／consumer4047641，20:42源已复制134GB，目标115GB，继续查live status。环境已安全解压。路径修正进程4055488，log `/home/ma-user/workspace/zhonghao/h3_deploy_31731/relocate_apply_20260913_204331.log`，需检查是否563处全部完成再运行CPU runtime checker。新版relocator27项Linux回归通过，SHA75f6c00770b3a96fb92bfa26c5ed9cc001fac3106a3ff8fd83d93cbac0388414。

## 下一组普通时序编辑（已获推理授权）

选择OmniEdit情侣04最后124帧（原始104到227，24fps）。自写明确“整段倒放”指令，不是官方pair，不生成GT。源先第二次亲吻、后贴胸拥抱。确切schema和prompt见`deploy_31731/prepare_explicit_reverse_couple.py`。

预期manifest `/cache/zhonghao/h3/data/explicit_reverse_couple_124/manifest.json`，source同目录。首次准备失败：系统`/usr/local/ffmpeg/bin/ffmpeg`不支持preset/libx264，未产出source和manifest，但创建了空目录。不能直接重复exist_ok=False。下一步检查此精确目录确实空且为本任务创建，允许复用自己的空目录；改用已迁移env内imageio_ffmpeg的真实二进制（只读定位），保留原裁剪/编码/124帧核验，禁止覆盖任何现有非空产物。不要让数据准备失败挡住当前湖景B。

## 三个并行子任务

- h3_resource_expansion_review：冻结的31731 tiny目录`deploy_31731/validation`已完成，31项CPU测试。root须完整审查、部署`/cache/zhonghao/h3/validation_code`，CPU/runtime依赖检查后，0123与4567分别小张量真机验证。当前agent正新增`deploy_31731/model_trial` A完整模型入口，复用独立profiles/base而不改30213。先A smoke→50step；需当前完成的transfer证据、本组tiny、runtime CPU evidence、sourceSHA、600GiB宿主/容器余量及55GiB/card门槛。
- h3_b_smoke_result_audit：在新`c_adaptation`实现严格C和CPUoracle，不改b_adaptation或远端。C只删除target-query×visual-source-key错latent帧边；source并非全部prefix，绝对RoPE t原点不同，按各自latent帧序号配对；首尾target query也严格同source帧，但target-anchor的target全局连接保留。只单video/Fs==Ft，badmetadata failclosed，其他分支不改。C目前未完成/未验证/未运行。
- source_dependent_temporal_edits：已实际看完7个source，07跳跃最合适source-dependent，04hug定义歧义，08/10需多主体异步对应，其他降级。另查到官方速度01摩托153帧，最后124帧是后续明确变速备选；不替换当前04。继续小范围搜索，未跑VLM/ASR/模型。5个实际pack源已只读SCP到`c_adaptation/upstream`供C开发。

## CPU runtime报告待封装

运行已上传的`validate_h3_runtime.py`（privateenv Python，先source `/cache/zhonghao/h3/env_h3_31731.sh`）。工具stdout报告status成功仅表示CPU imports/path/ABI，不代表NPU通过。为A入口存`/cache/zhonghao/h3/runtime_validation.json`时须增加现场host={hostname,boot_id,machine}、env_script_sha256、原src关键源码SHA；与model_trial最终schema对齐。不伪造通过、不放宽失败门槛。

正式A/B/C速度必须记录同期迁移/其他组负载，不能将跨机器/多卡数/共享I/O争用的时间直接做公平加速比。当前B若与迁移重叠应如实标注。后续VLM意义只在source-dependent平衡对照中验证，普通明确倒放不证明VLM必要。
