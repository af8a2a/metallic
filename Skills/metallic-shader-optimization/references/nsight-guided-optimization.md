# Nsight 驱动的 Shader 优化闭环

用于实际优化和前后比较；只复核历史包时读取已有证据，不自动重启实验。
GPU Trace 确定优化范围，Shader Profiler 和硬件计数器解释原因，普通 A/B 计时
确认收益。不要从代码外观直接宣布带宽、寄存器或 occupancy 瓶颈。

## 1. 确认接口与可比较的基线

优先通过用户连接的 nsight-mcp 读取数据。先发现实际 API manifest、profile、
family/group/tool schema，再调用工具；保留版本、查询参数、原始返回及警告。
Native 与 Workflow 的工具名、session/trace/range/shader ID 不可混用，ID 也不能
直接从 baseline 沿用到 candidate。常见查询线索见
[MCP 诊断与实验衔接](workflows.md#mcp-诊断与实验衔接)，以实际 schema 为准。

登记 baseline 的源码/候选 hash、EXE、GPU/driver、场景、相机/历史、分辨率、质量、
队列、驻留、工作量、预热和缓存条件。关联原始 `.ngfx-gputrace`、必要的
`.ngfx-capture`、普通计时及正确性证据。先核对已有捕获能否代表当前 baseline；
复用合格捕获，不用“最新文件”或旧 marker 名称代替身份验证。

没有 MCP 时先检查连接和安装包的 `docs/ngfx-mcp/`；用 CLI 导出或 NvPerf 补足
已支持的查询，并明确注明来源。缺失的 Profiler/计数器不能补零或伪装为 MCP 实测。
可以继续独立的正确性与计时验证，但报告诊断未完成，不宣称已定位硬件瓶颈；
工具缺失本身不否定已证实的小收益，也不触发自动安装/升级或 UI 操作。

## 2. 先选范围，再读取计数器和热点

通过 MCP 打开 trace，依次读取概览、帧/队列时间线、marker 层级及昂贵的
draw/dispatch 范围。区分关键路径、并行重叠、空闲/同步和 shader 执行，优先研究
能影响目标耗时的范围。保留完整 marker 路径、帧与范围边界、队列、dispatch 参数、
pipeline/module/entry 和实际 shader 身份；范围内若有并行工作，设备计数器不能
独占归因到其中一个 shader，也不能把重叠区间直接相加。

先枚举该范围实际采集的指标及单位，再选能验证假设的一小组，不猜 counter 名称。
下面是查询方向，不是固定 metric ID、自动分类器或必须全过的检查表：

| 假设 | GPU Trace / 计数器证据 | Profiler 或资源证据及交叉检查 |
|---|---|---|
| 内存吞吐受限 | DRAM/L2/L1TEX 吞吐、bytes/sectors、命中率 | 对应 load/texture 热点、访问量及合并情况；低命中率单独不能证明带宽饱和 |
| 访存延迟隐藏不足 | 吞吐、active/eligible warps、实际 occupancy（若有） | long-scoreboard 等采样、依赖链与 live registers；低带宽不排除延迟瓶颈 |
| 指令/执行管线受限 | SM issue、各执行管线吞吐、动态指令计数（若有） | 函数与源码/IL 热点、指令类别；静态 instruction mix 不当动态执行次数 |
| 寄存器/共享内存限制并发 | 实际 occupancy、active warps、资源限制项 | registers、shared memory、spill/local traffic（接口支持时）；源码变量数不代替编译分配 |
| 同步或分歧 | barrier 活动、shared bank conflicts、active threads/warp | barrier/branch 热点及线程组工作分配；stall 占比不能直接换成可回收时间 |
| 队列/提交或其他阶段受限 | GPU 空洞、等待、重叠与关键路径 | 先核实上游依赖/CPU 提交；单个 shader 加速未必改变整图耗时 |

对选定 shader 查询 profile 概览、Top-down/Bottom-up 函数树及 source/intermediate
逐行热点，把采样/stall 与实际源代码、绑定和工作量关联。记录总样本数、范围、
采样配置、缺失映射与无效样本；样本数不是调用数，样本比例不是独立函数耗时。
SM active 不等于 occupancy。具体 counter 的分母是 active cycle 还是 elapsed
cycle、是峰值百分比还是绝对计数，均从工具返回/指标定义确认。

## 3. 声明单一假设和采集计划

用一句可证伪的假设连接证据与改动，例如：“该 dispatch 的相关 load 热点和
DRAM bytes 支持重复读取假设；复用已验证的输入应减少 bytes/dispatch，期望目标
pass 普通耗时下降，同时检查寄存器增加是否抵消收益。”不要只写“减少指令应更快”。

在改动前声明目标 pass、主要指标、预期变化方向、可能代价、正确性要求及普通
计时/诊断预算。一次只验证一个主要原因，保留失败候选与结果。沿用目标 runner
的源码事务、构建和恢复；WorkControl、VSM、CLAS、PathTrace 各自使用其支持的契约，
不要把 `ExperimentRunner` 的单文件限制强加给其他 pass 或绕过其限制。

基线与候选使用相同 Nsight 版本、metric set、时钟策略、触发边界、采样设置、
预热和运行条件。先完成正确性检查，避免给错误输出做性能归因。新捕获写入独立
目录，保存进程结果、采集返回和新鲜的原始文件；使用应用 SDK 触发时保存 SDK
receipt。启动成功不等于采集成功。

需要 Shader Profiler 时确认 Real-Time Shader Profiling 与 metric set/架构兼容，
启用 shader pipeline collection，并通过当前生产编译入口保留优化及源码符号。
`DeepProfile --mode triage` 关闭 pipeline collection，不能作为 Profiler 采集模板。
所需吞吐计数器与 Profiler 不能同次采集时，分成匹配条件的诊断批次，分别标明
模式；不叠加 NvPerf/其他注入，也不把不同模式的时长拼成一个 A/B 结果。
外部 Nsight 注入与应用自注入只用一个入口；应用已有 SDK 目标帧边界时保留它。

诊断符号/配置可能影响二进制身份。分别记录普通与诊断构建的源码、编译选项、
SPIR-V/绑定映射；候选改变 shader 后其 hash 可以不同，比较的是对应工作而非
hash 相等。无法建立映射时说明归因限制，不能把调试构建收益转移给生产构建。

## 4. 对比优化前后捕获

用 MCP 对两侧执行相同语义的查询，分别解析实际范围和 shader ID。若候选拆分/
合并 dispatch，映射完整等价工作集合，比较合计开销，不只挑变快的子 dispatch。
同时观察邻接 pass、整图关键路径及新的热点，检查收益是否被转移的成本抵消。

每项保留 baseline/candidate 原值、单位、范围、聚合/归一化方式、差值和重复性。
百分比指标报告百分点差；绝对量可报告相对变化，但 baseline=0 时相对变化留空。
减少工作量本身是优化时，报告总量及有意义的每单位工作量，不能用归一化掩盖成本。
跨捕获原始 PC sample 数受采样时长影响，需核对覆盖与配置，并同时给出总样本和
占比；不能仅因 stall 百分比下降就断言 stall 绝对成本下降。

在独立报告目录保存查询原文、规范化 JSON/CSV 和 Python 图表，链接原始捕获 hash，
不要修改封存包。报告中的对照表至少包含：

| 范围/指标（含单位） | Baseline | Candidate | 变化/不确定性 | 来源捕获及查询 | 对假设的含义 |
|---|---|---|---|---|---|

普通 pass/graph 计时图与 Nsight 诊断图分开；诊断展示所选 counters、热点及转移
后的成本。沿用 [图表要求](../SKILL.md#python-测量图表审查报告)，单次捕获标 n=1，
不从一条 trace 的采样点制造独立实验置信区间，也不把缺失值画成零。

## 5. 联合判定并推进下一轮

保留 runner 的 `accept/reject/inconclusive` 原始 verdict；人工分析不覆盖机器结果。
报告分别回答：普通计时是否证明收益、Nsight 是否支持原假设、下一步应做什么。

| 观察 | 决策与后续 |
|---|---|
| 正确性通过、普通收益确认、诊断支持 | 接受性能结论；按已有授权处理候选，记录机制与剩余瓶颈 |
| 普通收益确认，计数器未改善/机制不清 | 保留收益结论，标明机制未证实；检查范围/采样、并行工作及瓶颈转移，不新增计数器改善硬门槛 |
| 计数器/热点按预期改善，普通计时跨零 | 保留候选与假设支持证据，性能仍 inconclusive；仅在有新假设或明确测量改进时制定下一份固定预算 |
| 计数器改善但普通耗时退化 | 不接受加速结论；检查寄存器、同步、工作转移及其他 pass，修正候选 |
| 正确性失败或身份不可比较 | 分别处理正确性失败或证据失效，不由好看的计数器抵消 |

维持正收益规则，不恢复 3% 下限。优化迭代以新证据推进：预测未兑现就修正假设，
瓶颈转移就重新选范围；没有新增证据或预算用尽时总结当前结果，不反复重跑到显著。
接口缺失只限制相应诊断结论，不把“所有计数器齐全/全部改善”作为接受的前置条件。

## 接口资料

官方 [MCP 自动化说明](https://docs.nvidia.com/nsight-graphics/UserGuide/automation.html)
说明 Workflow/Native 的发现方式；具体工具参数仍以当前安装的 schema 为准。
采集选项见 [GPU Trace](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html)，
采样和源码归因见 [Shader Profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)。
