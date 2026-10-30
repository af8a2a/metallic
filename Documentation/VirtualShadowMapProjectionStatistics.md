# VSM 投影统计独立实验

本轮只测试一个候选：将 PCF/SMRT 投影中逐像素更新的公共统计合并到已有线程组汇总。`VirtualShadowMapSettings::optimizeProjectionStatistics` 默认 false；两个 Pass 的 `benchmarkProjectionStatistics: true` 显式开启实验。保持生产默认关闭，不因本次测量自动推广。

## 实验边界

不改变阴影算法、采样点、页请求、选级、图集读取、输出层归属或时域历史。既有 Mark、cluster bitset、精确三角 bitset 和点光表共享在两臂保持相同。Hard 继续使用原逐像素统计，不进入组同步；PCF/SMRT 复用原本就执行的两次 `GroupMemoryBarrierWithGroupSync`。

原过滤统计由三个 groupshared uint 汇总，候选扩为八个（12 B → 32 B，共增加 20 B/线程组）。没有新 buffer、纹理、dispatch 或 GPU capability。一个 uint bitmask 记录接收点至多发生一次的六种事件，避免保持六个独立计数器。

| 原公共统计槽 | 含义 | 候选处理 |
| --- | --- | --- |
| 6 | 有效 receiver | mask bit 1 → shared slot 4 |
| 7 | coarse fallback | mask bit 4 → shared slot 6 |
| 8 | background | mask bit 0 → shared slot 3 |
| 9 | coarse outside / filter missing | mask bit 3 与原 filterCounts.z 相加 → shared slot 2 |
| 13 | 更粗 fine level fallback | mask bit 5 → shared slot 7 |
| 29 | local near-plane rejection | mask bit 2 → shared slot 5 |
| 30 / 31 | 原过滤查询计数 | 原 shared slot 0 / 1 不变 |

事件记录保留原位置：receiver/background 在 compact 点光归属判断之前计数，background 每个面计数但仅 face 0 写共享输出；disabled、域拒绝和 coarse outside 仍保留原计数。视口尾部先将 out counters 清零再返回。所有过滤 lane 的提前返回均限制在 helper 内，仍到达两次组 barrier。

统计槽 5 的 build overflow 读取不变；候选只改变上述输出统计的加法位置，过滤与投影不会读取这些正在累计的槽。公共 32-word ABI、每面归属和最终 uint32 模加结果不变；跨面过滤缺页仍归属接收点的 dispatch。诊断 GPUWork 计数独立，不随本开关改变。

## 开关与身份

`VSM_OPTIMIZE_PROJECTION_STATISTICS` 缺省为 0，并与设置一起进入程序缓存身份；切换时重选程序。实际 `[VSMShader]` 日志追加 `projectionStatistics` 和既有 SPIR-V SHA256。宏关闭时保留原事件 atomic 表达式；宏开启时 Hard 仍走 direct atomic 分支。配置值或源码行数不等同于实际 GPU 收益。

候选可能以 shared atomic、额外 shared 清零/汇总或寄存器存活开销抵消全局 atomic 的减少，因此使用独立同源 A/B 普通计时，不用 instrumentation 时间推算性能。仅这一项候选，没有同轮参数扫描或追加第二种实现。

## 2026-10-30 正确性验证

Release 编辑器与 RHI tests 构建通过，Slang ownership audit 通过（387 files / 275 modules / 112 implementation units），23 项本地 A/B runner CPU fixture 测试通过。初次测试编译中的 `float3` 分量访问问题在 GPU 运行前修正，随后构建成功；没有修改实验预算或正确性门槛。

两个针对性 GPU 回归全部通过、无跳过：profiling（103.1 秒）、projection equivalence（238.1 秒）。投影矩阵共 84 状态、四个对照臂，覆盖 Hard/PCF/SMRT、Legacy/compact、背景、视口尾部、禁用、点光面归属、缓存/溢出/恢复、jitter/相机移动、时域历史、源槽重排和双点光。阴影、debug、HDR、深度及完整每面 32 项统计保持精确相等，Vulkan validation 消息为 0。

新增 6 个 NearPlane 状态（3 种过滤 × 2 种布局）均实际记录 `nearPlanePixels=1120`，不是仅靠预期值声明近面路径得到覆盖。双点光及后续近面共 12 状态的 CPU profile 树超过既有每节点 256 section 上限，实际 pointTables 计数明确 unavailable；该限制不取消图像、统计和 GPU 正确性检查，也不能把 expectedPointTables 当成实测。

Sponza 和 Mixed 的独立 diagnostic 均通过：第 16 帧 freeze streaming 并 camera cut，第 16/32/63 帧 PNG、线性 HDR、depth、shadow 精确一致，第 16–63 帧 canonical 输入、完整 stats32、15 项稳定 workload 及全部 45 项 build diagnostics 一致。只有共享 Mark 原子请求次数保留已有的调度豁免。最终两场景截图均已目视检查。实际 Mark 与 Build SPIR-V 集合在 A/B 两臂相同，仅 Projection SPIR-V 按预期改变。

## 固定预算与结果

本轮只运行这一项候选，生产默认保持 false。独立工具为 `.cache/VSMProjectionStatistics/ProjectionStatistics.py`，基于现有 Acceptance receipts 和 ABBA 统计；两臂固定已有 cluster/triangle bitset、局部光 Mark 优化及点光表共享，仅切换 `benchmarkProjectionStatistics`。

RTX 5060、原生 1920×1080、SMRT 2×16、32 帧历史、2048 页、关闭 jitter/debug 输出。每场景先运行两个 diagnostic 进程（64 帧，前 32 帧预热），随后固定两个 discovery ABBA 区组（8 个独立进程，每进程 96 帧，前 32 帧预热）。仅 discovery 有希望时才运行独立三个 confirmation 区组；目标为 projection 至少 3% 收益，graph 回退不超过 2%。预算在首个 GPU 进程前写入 `Budget.json`。

两场景均在 discovery 被判定 `reject / confirmed_regression`，因此均未运行 confirmation。这里的判定名称表示 discovery 区间已经支持回退，不代表完成了独立确认阶段。没有追加样本、扫描参数或改测第二种实现。普通计时关闭 validation、GPUWork 和 capture/readback；保留磁盘缓存，不锁频，不停止用户应用，串行执行不表示系统 GPU 独占。

下表 A/B 为进程中位数再取中位数；收益及 95% CI 为独立 ABBA 区组 log ratio 估计，负值表示变慢，不能直接以展示列相除代替估计。

| 场景 / GPU 范围 | A：原统计（ms） | B：组内汇总（ms） | 收益估计 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Sponza / Projection | 2.377184 | 2.500112 | -5.42% | -8.27%–-2.64% |
| Sponza / Shadows pass | 8.788168 | 8.923336 | -0.69% | -2.28%–0.88% |
| Sponza / RenderGraph | 14.507016 | 14.723720 | -1.07% | -5.76%–3.42% |
| Mixed / Projection | 3.959504 | 4.302712 | -8.37% | -13.06%–-3.87% |
| Mixed / Shadows pass | 8.919392 | 9.256480 | -4.08% | -18.13%–8.29% |
| Mixed / RenderGraph | 11.634120 | 11.933904 | -2.91% | -26.08%–16.00% |

Sponza 的 projection 变慢约 5.42%，Mixed 变慢约 8.37%，两者收益区间均完全低于零。Shadows/graph 的区间跨零，尤其 Mixed graph 很宽，不能把这些阶段的点估计当作确认的整图回退比例或最大回退界限。

结论是保留原生产路径，不推广该候选；实验代码仅由显式开关保留用于复现。减少源码层面的全局 atomic 调用并不自动带来 GPU 加速。shared atomic 竞争、额外 shared 清零/汇总和变量存活开销都是可能原因，但本次没有收集底层寄存器占用、SASS 或硬件计数器，不能将回退确定归因于其中一项。结果仅适用于上述固定快照的 pass/graph 测量，不是呈现 FPS 或所有场景的承诺。

## 证据与复查

最终核对 739 个源码、18 个二进制 hash 与采集时完全一致，记录于 `.cache/VSMProjectionStatistics/FinalIdentityCheck.json`。

- `Regression`、`Regression.log`：两个 GPU 回归、84 状态与完整验证信息。
- `Sponza/{Diagnostic,Discovery}`、`Mixed/{Diagnostic,Discovery}` 及各自 `*Report`：原始封装、输入/输出身份、实际 shader 身份、逐帧数据及全部阶段计时；没有 confirmation 包。
- `Cases`、`Budget.json`、`RunAll.py`、`RunEvidence.py`：固定输入、预算与执行顺序。
- `ProjectionStatistics.py`、`TestProjectionStatistics.py`、`ToolSnapshot`：本次独立工具、23 项 fixture 测试与固定工具快照。生成证据未加入版本控制。

以下复查依赖上述本地缓存与 matplotlib，不重新启动 GPU：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='E:/metallic/.cache/VSMOutputs/Python'
$env:MPLCONFIGDIR='E:/metallic/.cache/VSMOutputs/Matplotlib'
python -B .cache/VSMProjectionStatistics/TestProjectionStatistics.py
python -B .cache/VSMProjectionStatistics/ProjectionStatistics.py report .cache/VSMProjectionStatistics/Sponza/Discovery .cache/vsm-projection-statistics-recheck --diagnostic .cache/VSMProjectionStatistics/Sponza/Diagnostic
```
