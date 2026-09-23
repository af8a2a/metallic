# ZorahFull 帧起始 Reflex 等待归因

2026-09-23。结论：当前帧起始区间的长耗时来自 `slReflexSleep` 的阻塞墙钟时间；最慢 5% 调用中 Sleep 占 99.967% / 99.971%。不是帧令牌分配、应用侧互斥锁竞争或每 60 帧状态查询。On/Off 对照支持低延迟节奏控制将等待前移的解释，不能将这段时间当作可直接删除的 CPU 计算成本。

## 历史 18.07 ms 的口径

`full-request-selection-pacing-0923/run1` 原始数据为 1295 帧：`Streamline frame begin / Reflex pacing` 均值 3.8560 ms、P95 **18.0661 ms**、最大 21.1955 ms。它包围整个 `StreamlineFrameScope` 构造函数，当时没有内部调用计时。日志确认 mode=On、frame interval=0 us。

此后 CPU 缓存优化与 Streamer 重构已经改变运行特征。本次没有原样复现 18.0661 ms，也不能追溯性地断言旧数据每一毫秒都在 Sleep。本次增加内部计时后，在当前实现上确认同一路径的主要来源。

## 方法与条件

RTX 5070 Ti；Full 编辑器固定漫游；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px；隐藏窗口、VSync=true、2 个 frame slots；就绪后预热 10 秒，采样 30 秒。按 On → Off → On → Off 顺序运行四轮。

四轮校验了相同 executable/shader SHA256、graph 属性、相机路线与显示条件。场景流送正常运行，因此各轮帧数及每帧 cut/驻留不完全相同；这是节奏策略归因对照，不是冻结工作量的 GPU 性能对照。GPU 进程遥测未见其他大型渲染进程，仍有桌面、浏览器等少量后台活动。

可选的 `StreamlineFrameBeginProfile` 仅为 benchmark 记录 mutex acquisition、token、options、sleep、status、marker 和总墙钟时间。普通调用不读取新增时钟；没有改变默认模式、Sleep/marker 顺序、状态刷新频率、帧槽等待或提交逻辑。Off 仍调用 SDK Sleep，符合现有集成方式。

## 结果

单位 ms；“帧槽等待”为 Sleep 前的 `Wait Frame Slot Before Input`。

| 测试 | 帧数 | Sleep 均值 | Sleep P95 | 帧槽等待均值 | 整帧均值 | 整帧 P95 |
|---|---:|---:|---:|---:|---:|---:|
| On 1 | 1434 | 4.633 | 9.530 | 2.142 | 20.928 | 27.761 |
| Off 1 | 1375 | 0.049 | 0.088 | 7.626 | 21.823 | 28.063 |
| On 2 | 1368 | 5.366 | 10.864 | 2.180 | 21.934 | 28.436 |
| Off 2 | 1369 | 0.046 | 0.092 | 6.649 | 21.916 | 28.804 |

On 1 帧起始区间总 P95 为 9.5328 ms，Sleep 为 9.5295 ms。最慢 5% 区间的非 Sleep 部分均值仅 0.00357 ms。互斥锁等待最大 0.0049 ms；状态查询最大 0.0045 ms；SimulationStart marker 最大 0.1372 ms（单个异常），无法解释长尾。四轮采样期间 optionsUpdates=0、suspendedFrames=0，frameLimitUs 始终为 0；所有帧均调用 Sleep。

每帧先相加“帧槽等待 + Sleep”，再计算 P95，四轮分别 **12.276 / 12.694 / 12.718 / 12.614 ms**。此处没有相加独立的 P95。Off 显著减少 Sleep，却增加帧槽背压等待，没有稳定的整帧收益。

交换链 Acquire P95 约 0.007 ms，Present P95 约 0.2–0.3 ms，当前可见的长等待也没有落在这两个 CPU 调用中。这不等同于完全排除呈现队列或 VSync 对驱动节奏决策的影响。

去重后的驱动报告显示，simulation start→GPU render end 均值：On 36.34 / 38.46 ms，Off 43.83 / 43.15 ms。只作为方向性佐证：状态每 60 帧更新，报告对应较早帧，不是当前帧 GPU 耗时，也不包含显示延迟。

## 代码路径与解释边界

1. `EditorApplication::waitForFrameSlotBeforeInput()` 先等待复用帧槽。
2. `StreamlineFrameScope` 获取锁与 token、按需设置选项，再调用 `slReflexSleep`，之后发 SimulationStart，最后才采样 SDL 输入。
3. 本地 Streamline 的 `slReflexSleep → slSetData → compute->sleep()` 进入 Vulkan 低延迟后端。后端选择优先尝试 `VK_NV_low_latency2`，再尝试 NvLowLatencyVk；前者调用 `vkLatencySleepNV`，后者调用 `NvLL_VK_Sleep`，两条路径随后都等待 timeline semaphore。
4. 实际运行使用部署的 NVIDIA 签名 DLL。本次直接计时边界到 `slReflexSleep` 为止，未采集 DLL 内部 API 分段/线程调度栈。不能据应用的通用 “NvLowLatencyVk via Streamline” 日志确认具体选中的底层后端，亦不能精确拆分驱动设定的等待与 OS 唤醒延迟。

应用 frameLimitUs=0 排除了应用主动设置固定限帧间隔；并不排除驱动控制面板限帧、VSync 等外部策略。On/Off 结果表明低延迟模式是当前多毫秒 Sleep 的关键开关。等待减少后，CPU 更早提交并很快在 2-slot 复用点遇到 GPU 背压，因此这不是直接关闭 Reflex 就能回收的帧预算。

## 下一步

保留默认 Reflex On。优化优先回到实际 GPU 关键路径和 CPU Streamer 准备工作，验收同时看整帧分位数、帧槽+Sleep 总等待及延迟报告，避免只把等待移到另一个 scope。

如果后续仍出现整帧长尾而 GPU 工作量平稳，再做可见编辑器的呈现/VSync 对照，并用 CPU 调度与 Vulkan 提交时间线拆分 Sleep 内部等待、唤醒和 GPU 空洞。当前四轮不足以宣称稳定 30 fps：超 33.33 ms 的帧分别为 4 / 2 / 4 / 29。

## 证据与验证

- [结构化结果](ZorahFullReflexFrameBeginResult.json)：历史口径、四轮 Manifest、阶段分布和逐项统计。
- `build-release/full-reflex-begin-{on,off,on-repeat,off-repeat}-0923/run1`：Frames.jsonl、Capture.json、ReflexSummary.json、GPU 遥测及日志。
- `Tools/AnalyzeZorahFullReflex.py`：逐帧归因、最慢 5% 同帧占比、等待合计、去重驱动报告；无额外 GPU 查询。
- Release MetallicGPUDrivenSample / Metallic 构建通过；`MetallicEditorReflex.off/on/boost` 三项 128 帧 smoke 均通过并取得驱动报告，无验证错误。
- Full 四轮捕获完整，采样关闭 Vulkan validation；性能证据与验证 smoke 分开。

复现（PowerShell；Off 对照仅设置当前进程环境，不改变产品默认值）：

```powershell
$env:METALLIC_REFLEX_MODE='on' # 对照轮用 off
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/<new-dir> -Runs 1 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660
python -B Tools/AnalyzeZorahFullReflex.py build-release/<new-dir>/run1
```
