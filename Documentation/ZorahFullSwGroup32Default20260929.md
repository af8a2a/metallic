# Stream SW raster 默认切换到 wave32 / 32 线程（2026-09-29）

在此前 [32 线程正确性验收](ZorahFullSwGroup32Correctness20260929.md) 后，按本次用户要求将候选提升为生产默认。MiniZorah 和 ZorahFull 的流式 SW raster early / late 均自动选择 `streamClusterRasterGroup32Main`；普通启动不需要环境变量或线程组覆盖。此前验收报告中的“生产仍为 128”描述的是提升前状态。

## 选择与兼容

- `VisibilityBufferPass::softwareRasterMode()` 将原共享屏幕顶点 WorkControl 默认模式映射到 32 线程。设备必须报告固定 subgroup size 32，并支持使用到的 subgroup ballot/arithmetic。
- 32 线程 pipeline 正常创建；64／128 线程实验 pipeline 仍需 `METALLIC_SW_GROUP_EXPERIMENT=1`。不满足固定 wave32 的设备保留旧 128 线程默认路径，本机没有验证其他设备。
- 历史 WorkControl / camera-reapply / capture-replay 对照显式选择旧 WorkControl 128，防止提升默认值后改变参考入口。其他旧实验模式仍遵循各自配置。
- 可选 shader warmup 请求补入生产 32 线程入口；没有将 warmup 加入默认构建依赖。本轮未执行完整 warmup。
- 本次没有改变 raster 内核算法或稳定 ID 编码，也没有改变其他计算 pass 的线程组。实际 32 线程 SPIR-V FNV1a64 仍为 `1407237746718151905`。

## 本轮验证

环境为 RTX 5070 Ti；构建 `build-release`、`build-relwithdebinfo` 和 `build-scheduling-release` 的 `MetallicGPUDrivenSample` 均成功。后者的 `MetallicRhiTests` 也已重建。

| 检查 | 结果 |
|---|---|
| GPU 边界簇 fixture | 313 个用例 × 4 个实际 shader 入口通过，启用 RHI validation / bindless |
| Full 默认选择对照 | early 1,348,353 / late 159,696 SW 簇，默认 32 与 WorkControl 128、strided 64／128 的原始深度及可见性输出逐字节一致 |
| Mini 默认选择对照 | early 19,773 / late 2,745 SW 簇，四种入口原始输出逐字节一致 |
| 普通 Release 默认漫游 | 无实验开关、`softwareGroupSize=0`，30 秒 1,274 帧全部绑定 wave32 / 32 线程，流送未冻结 |
| 分析工具回归 | `TestSwGroupCorrectness.py` 5 个、`TestRasterComparisonScopes.py` 3 个通过 |
| 补丁检查 | `git diff --check` 通过 |

两个场景的对照均使用 `swGroupUseDefault32=true`，使 32 线程 case 的 override 为 0。每种入口采样 8 帧，验证 before / after 的 cut、驻留映射、SW 列表及间接调度参数一致，且 early / late 的实际入口正确。启用 validation，未发现 VUID / DeviceLost。此短对照用于默认选择与正确性验证，不作为新的性能验收。

普通漫游使用 1797×660 输出、1198×440 内部渲染，时间抖动开启，10 秒预热后采样，不开启 validation。日志只加载 group32，未加载实验 group64／128。上传 29,673 页、回收 28,765 页，驻留页数 61,311–62,662；load failures、request overflows、BLAS overflow 均为 0。

30 秒漫游帧时间均值 23.56 ms，P95 33.46 ms，70 帧超过 33.33 ms，最长连续 2 帧。因此本次切换不代表已经满足严格持续 30 fps。仍有每帧 1 次页面分配准入失败，共 1,274 次；这与之前验收观察到的预算压力一致，本轮未处理该问题。

## 可复查证据

以下均为本地生成输出，不加入源码管理：

- `build/SwGroupPromotionBuild.log`、`build/SwGroupPromotionBoundary.log`。
- `build-release/sw-group32-default-full-01/run1/`：Capture、Summary、四种入口原始输出。
- `build-release/sw-group32-default-mini-01/run1/`：同上。
- `build-release/sw-group32-default-live-01/run1/`：逐帧遥测、日志、`DefaultPathAudit.json`。
- `build-release/sw-group32-promotion-evidence-20260929/`：本轮源码快照、审计汇总和 SHA-256 清单。

测试入口：

```powershell
.\build-scheduling-release\tests\MetallicRhiTests.exe --rhi-validation --rhi-bindless --filter stream_group_raster_boundaries
python -B -m unittest discover -s tests/perf -p TestSwGroupCorrectness.py
python -B -m unittest discover -s tests/perf -p TestRasterComparisonScopes.py
```

普通默认路径漫游通过 `Tools/RunZorahFullRoam.ps1`，使用 `Tools/Perf/SwGroupCorrectness.Live.json`，不传 `-SoftwareGroupSize` 或 `-SwGroupComparison`。两个默认选择对照使用本地 `build/SwGroupDefaultFull.json` / `build/SwGroupDefaultMini.json`，启用 `-SwGroupComparison -Validation -Rounds 1 -SampleFrames 8 -SettleFrames 8`。
