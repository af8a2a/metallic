# VSM 精确三角形页枚举

精确三角形 build 复用 cluster dirty-page bitset，按原三角投影包围矩形逐行枚举置位页，替代对矩形内每个页读取 uint4 元数据。`VirtualShadowMapSettings::trianglePageBitset` 默认 true；两个 VSM Pass 读取生产属性 `trianglePageBitset`，显式布尔 `benchmarkTrianglePageBitset` 优先覆盖。有效值还要求 cluster culling 和 cluster-page bitset 均开启；任一依赖关闭时回到原逐页扫描。

## 枚举与生产者契约

- 三角形解码、方向光投影、局部光近面及完整面裁剪、深度域拒绝、矩形端点的 floor/clamp 均保持原表达式。
- 每行的完整页号区间转换为 word 区间，用首尾 mask 限定该行。按递增 word 及 `firstbitlow` 枚举，清除最低置位 bit 后继续，保持单三角形原有 y/x 顺序。
- 缓存未掩码的最后一个 word，支持 31/33 等非 32 对齐页轴、跨行共享 word 和带偏移的局部 mip。每行重新应用 mask，不消费其他行的 bit。
- Reset 清空追加位图。Admit 只为已有 resident、admitted、dirty 的 fine page 置位；Assign 只接收 admitted 缺页，将其变为 resident/dirty 后置位。coarse page 不进入这两种置位路径。
- Physical 只回收未准入 owner，已准入 resident page 被固定，因此 build 时每个置位 bit 与旧 `entry.x != UINT_MAX && (entry.y & 2) && (entry.w & 2)` 条件等价。没有重新读取页元数据的必要。
- owners scratch 为资源 slot 9，Admit 后的 `{1,4,9}` barrier、Physical 后的 `{4,9}` barrier、Assign 后的 `{1,9}` barrier 均覆盖整个 buffer；追加位图区域同样可见。Build 不再修改该区域。

命中页仍使用原来的 `statistics[3]` 原子分配、work capacity 检查和 `statistics[5]` overflow 计数；coarse build/raster、溢出兜底及恢复逻辑保持原实现。跨线程的原子发射顺序仍无序，不能仅因保留单三角形顺序便断言 work buffer 字节排列相同。

本次没有新增 GPU buffer、dispatch 或同步点。已有 cluster 位图的 allocation/reset/admit/assign 完整复用。有效开关同时进入 CPU Programs cache、`VSM_TRIANGLE_PAGE_BITSET` 宏和实际 shader 身份日志。独立 shader 的该宏默认 0，继续兼容旧资源布局的独立探针。

## 计数与接口

公开 statistics32 及原 16 项 workload 布局不变。`buildPageTests` 保持“精确三角形阶段实际 uint4 页表读取次数”的定义：原路径逐候选页递增，位图路径为 0。

Private build diagnostics 从 v2 的 43 项升级到 v3 的 45 项，位于 statistics byte 256，大小从 344 B 增至 360 B。前 43 项 cluster 计数及 histogram 的索引、语义不变，仅追加：

| 索引 | 字段 | 含义 |
| --- | --- | --- |
| 43 | `triangleBitsetWordTests` | 精确阶段实际 uint 位图读取次数；跨行复用不重复计数 |
| 44 | `triangleBitsetPageHits` | 精确阶段枚举出的页命中次数，包括 work capacity 溢出的候选 |

Reset 的线程数、private buffer 分配、单/多灯输出 buffer 与 copy/readback 大小都沿用统一常量。关闭 GPUWork 时没有这两个计数的原子操作或额外诊断输出。新旧元素宽度不同，不能把读取次数变化直接当成 DRAM 流量变化；普通 GPU build 计时才是性能门槛。

A/B 仍严格比较图像、深度、完整页统计、前 43 项 cluster diagnostics 及其余确定性 workload。仅实际页表读取 `buildPageTests` 因算法而变化；共享 Mark 的 `markPageRequests` 依旧受原子调度影响。新的 word/hit 指标单独记录，不伪装成旧页扫描计数。

## 2026-10-30 验证记录

Release 编辑器、VSM Sample、RHI tests 构建通过；Slang ownership audit 通过（387 files / 275 modules / 112 implementation units）。69 项现有 VSM 性能工具测试及 23 项本地 A/B runner 测试通过。GPUProfile 支持 v1/v2/v3，旧证据缺失的新字段保持 unavailable，不补成零；Acceptance 发布 v3 并使用统一的 counter names/size。

6 项 GPU 回归全部通过、无跳过：cache、profiling、cluster page bitset equivalence、Mark samples equivalence、triangle page bitset equivalence、content。新矩阵共 72 状态、10 个生产关闭/依赖关闭控制状态，54 个状态观察到实际页表读取减少；生产默认臂省略实验开关。覆盖 resident/stream、Hard/PCF/SMRT、局部光近面与点光接缝、屏外 caster、31/33 非整 word 行、cold/warm/dirty/overflow/recover。图像、完整 stats32、确定性 work 和前 43 项 cluster diagnostics 严格比较，validation 为 0。旧 cluster 矩阵 62 状态和 Mark samples 矩阵 82 状态同样通过。

完整场景均使用 RTX 5060、原生 1920×1080、SMRT 2×16、32 帧历史、2048 页、关闭 jitter/debug 输出。A/B 显式固定 cluster bitset 与局部光 Mark 优化开启，只切换 `benchmarkTrianglePageBitset`。两种场景独立 diagnostic 均通过：第 16 帧冻结 streaming 并 camera cut，第 16/32/63 帧 PNG、线性 HDR、depth、shadow 精确一致，第 16–63 帧 canonical 输入、完整 stats32、14 项稳定 workload 与前 43 项 cluster diagnostics 一致。实际 Mark SPIR-V 集合在两臂相同，Build SPIR-V 按预期不同。最终两场景截图已目视检查。

### Sponza 工作量与资源

以下是 diagnostic 预热后的帧中位数，仅表示实际工作量，不作为计时样本：

| 项目 | A：原页扫描 | B：置位枚举 |
| --- | ---: | ---: |
| 精确阶段 uint4 页元数据读取 | 1,688,862 | 0 |
| 精确阶段 uint 位图读取 | 0 | 236,345 |
| 精确阶段置位命中 | 0（该计数不适用旧路径） | 5,821.5 |
| `statistics[3]` 发射候选 | 5,821.5 | 5,821.5 |
| 解码三角形 | 68,603 | 68,603 |
| Cluster 位图读取 | 166,530.5 | 166,530.5 |

`statistics[3]` 每帧完整相等，置位命中包含容量溢出的候选，因此不是“成功写入 work buffer”的数量。新旧读取元素宽度不同，这些数值不直接换算 DRAM 带宽。

两臂每帧 resourceMemory 完全相同，Sponza diagnostic 逻辑 buffer 为 11,963,372 B。相对于此前 v2，private 诊断 sidecar 仅增加 16 B/face；普通 GPUWork 关闭时不分配这部分新增计数。生产位图资源复用现有 owners scratch，无额外 buffer、dispatch 或 barrier；普通生产分配与调度不因 A/B 开关变化。

### 独立性能确认

预算在运行前固定并保存在 `.cache/VSMTriangleBitset/Budget.json`：每场景 2 个 diagnostic 进程（64 帧、warmup 32），2 个 discovery ABBA 区组（8 个进程），有希望时才追加独立 3 个 confirmation 区组（12 个进程）。普通计时每进程 96 帧、前 32 帧预热，关闭 validation、GPUWork 和 capture/readback。主目标为 build 至少 3% 收益，graph 最大 2% 回退，95% 区间；不合并 discovery/confirmation，不追加轮次追逐显著性。保留磁盘缓存，不锁频，不停止用户应用，串行执行不代表系统 GPU 独占。

Sponza discovery 后完成独立 confirmation，结果 `valid / confirmed_gain / accept`。A/B 列是进程中位数再取中位数；收益和区间按独立 ABBA 区组 log ratio 计算，不能直接由两列相除代替。

| GPU 范围 | A（ms） | B（ms） | 确认收益 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Build | 2.436880 | 0.239400 | 90.24% | 89.77%–90.69% |
| Shadows pass | 10.939304 | 8.604056 | 21.30% | 17.41%–25.00% |
| RenderGraph | 16.705160 | 14.469528 | 13.53% | 9.54%–17.34% |
| Mark | 4.776736 | 4.800960 | -0.06% | -5.72%–5.31% |
| Allocator | 0.101592 | 0.102184 | -0.56% | -1.87%–0.72% |
| Raster | 0.077768 | 0.078120 | -0.72% | -2.69%–1.22% |
| Projection | 2.358992 | 2.359928 | -0.21% | -1.54%–1.10% |

这是固定 Sponza 快照的 GPU pass/graph 测量，不是端到端呈现 FPS 或动态流送场景承诺。Mark 等未修改阶段的计时仍保留其不确定性，不能将阶段变化自动归因于算法改进。

Mixed 为 Sun/Moon/Point/Spot 共 4 灯 9 面，预热后没有 dirty build 工作；两臂解码、页元数据读取、三角 bitset word/hit 全部为零。2 个 discovery ABBA 区组的 Build 0.225512 → 0.226840 ms，收益 -0.46%，95% CI [-3.09%, +2.10%]，未达主目标，判定 `below_target`，未追加 confirmation。Graph 11.502520 → 11.618168 ms，收益 -0.75%，CI [-4.38%, +2.75%]；点估计回退低于 2%，但区间跨零且下界超过 2% 回退，不能声称严格证明了性能中性或 2% 上限。该空工作场景不作为加速证据。

生产默认的收益依据是 Sponza 独立确认；Mixed 的不确定性完整保留，`trianglePageBitset: false` 可恢复逐页扫描。没有调整实验门槛，也没有隐藏被拒绝或无收益的场景。

## 本地证据与复查

当前源码与二进制最终核对通过：739 个源码、18 个二进制与测量时 hash 全部一致，记录于 `.cache/VSMTriangleBitset/FinalIdentityCheck.json`。

- GPU 回归：`.cache/VSMTriangleBitset/Regression` 与 `Regression.log`。
- 原始采集：`Sponza/{Diagnostic,Discovery,Confirmation}`、`Mixed/{Diagnostic,Discovery}`；各自 `*Report` 包含封装身份、完整帧数据、逐阶段统计和图表。
- `Cases`、`Budget.json`、`RunAll.py`、`RunEvidence.py` 保留输入与执行流程；`TriangleBitset.py`、`TestTriangleBitset.py` 以及 `ToolSnapshot` 保留该任务本地工具和固定 hash。
- 所有生成证据位于 `.cache/VSMTriangleBitset/`，不加入源码控制。工具基于现有 Acceptance receipts 和 ABBA 统计，未新增通用 renderer CLI。

复查依赖上述本地缓存及 matplotlib：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='E:/metallic/.cache/VSMOutputs/Python'
$env:MPLCONFIGDIR='E:/metallic/.cache/VSMOutputs/Matplotlib'
python -B -m unittest discover -s tests/perf -p 'TestVirtualShadowMap*.py'
python -B .cache/VSMTriangleBitset/TestTriangleBitset.py
python -B .cache/VSMTriangleBitset/TriangleBitset.py report .cache/VSMTriangleBitset/Sponza/Confirmation .cache/vsm-triangle-recheck --diagnostic .cache/VSMTriangleBitset/Sponza/Diagnostic --reference .cache/VSMTriangleBitset/Sponza/Discovery
```
