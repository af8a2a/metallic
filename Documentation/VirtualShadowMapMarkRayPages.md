# VSM 方向光 Mark 射线同页快路径

**状态（2026-10-31）：已按用户决定默认启用；性能确认结论仍为不确定。GPU 等价通过；Sponza 确认 Mark 收益 8.84%、Shadows 收益 4.81%，RenderGraph 区间跨过 −2% 回退门槛，工具判定为 `inconclusive / retain_candidate`。**

`VirtualShadowMapSettings::markRayPages` 默认 `true`。两个 VSM Pass 的 `markRayPages` 生产属性控制该路径，显式布尔 `benchmarkMarkRayPages` 优先。只有方向光编译 `VSM_MARK_RAY_PAGES=1`，局部光及 Hard/PCF 路径不变。有效值进入 `Programs` 缓存键和 `[VSMShader] markRayPages` 日志。

## 实现与数值契约

方向光 SMRT 采样满足 `P(t) = origin + span * t`，其中 `t ∈ [0, 1)`。正交投影下，采样 XY 位于 `t=0..1` 线段上。`vsmMarkRayPages` 对每条射线的每个 receiver 可用级执行：

- 用 `t=0` 采样投影起点，用 `rowX/rowY` 投影 `span` 得到终点。
- 线段包围盒四周各加 1 texel 边距吸收舍入，要求其完全位于同一页，且位于域内缩范围内。
- `t=0` 深度距 `[0, 1]` 边界至少 `1e-4`。其他采样超出深度范围不影响结果，因为它们要么被拒绝，要么请求同一页，而 `t=0` 已请求该页。

满足条件时只标记该页，跳过该级其余采样。不满足的级写入 `sweepLevels`，继续走原逐采样路径；全部级通过时跳过该射线的采样循环。判定只决定是否省略，不用插值结果替代实际保留的采样，因此请求页集合应与原路径一致。

## 计数含义

`markSamplePositions` 与 `markLevelTests` 现在记录实际执行次数，候选臂允许下降。每个保留 sweep 的射线另有一次探测，极端情况下可能略高于基线。`markReceiverDispatches` 及其余非 Mark 计数必须保持一致，`markPageRequests` 保留既有调度例外。

## 验证计划

GPU 测试 `virtual_shadow_map_mark_ray_pages_equivalence` 沿用 82 状态主矩阵和 1–12 层 Cold/Warm SMRT 扫描，比较图像、depth、stats32、build diagnostics，并要求快路径和保留 sweep 的状态都被覆盖。控制臂包括默认关闭、生产属性开启、benchmark 优先级和关闭共享 Mark 缓存。

Sponza 使用 `Tools/Perf/VirtualShadowMapBitset.py --experiment mark-ray-pages`，配置同 Mark clipmap 实验：固定相机、1920×1080、SMRT 2×16、2048 页。诊断要求 16/32/63 帧 PNG/HDR/depth/shadow 和 16–63 帧 stats32、build diagnostics、13 项稳定工作量精确一致。普通计时为 2 个 discovery ABBA 区组，通过后再运行 3 个独立 confirmation 区组；门槛为 Mark 收益 95% 下界不低于 3%，RenderGraph 回退不超过 2%。

本机脚本和用例位于 `build/vsm-mark-ray-pages/`，不纳入源码。

## 验证结果

GPU 矩阵通过：82 个主状态、10 组控制、1–12 层各 2 个状态，共 106 个状态，耗时约 141 秒，validation 为 0。44 个主状态实际减少采样，72 个主状态保留 sweep，两条路径均被覆盖。实际启用的 Mark 日志全部为方向光。

Sponza 诊断通过，16/32/63 帧输出、16–63 帧 stats32、build diagnostics 和 13 项稳定工作量精确一致。后 32 帧工作量中位数：

| 计数 | A | B | 变化 |
| --- | ---: | ---: | ---: |
| markReceiverDispatches | 2,070,497 | 2,070,497 | 0 |
| markSamplePositions | 70,396,898 | 41,693,220 | −40.8% |
| markLevelTests | 236,441,542 | 97,694,628 | −58.7% |
| markPageRequests | 374,324 | 381,126 | 含调度差异 |

独立 confirmation 为 3 个 ABBA 区组、12 个进程，RTX 5070 Ti，每进程 96 帧、预热 32 帧。没有与 discovery 合并；discovery 期间存在进程间大幅波动，可能受同机 GPU 负载影响，确认阶段无 spread 警告。

| GPU 范围 | A ms | B ms | 区组收益 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mark | 1.4373 | 1.3103 | 8.84% | [7.84%, 9.83%] |
| Shadows | 3.6163 | 3.4274 | 4.81% | [1.93%, 7.61%] |
| RenderGraph | 7.3570 | 6.9675 | 4.80% | [−3.54%, 12.48%] |
| Projection | 1.0207 | 1.0151 | 0.49% | [−0.48%, 1.45%] |
| Build | 0.1512 | 0.1578 | −19.40% | [−59.97%, 10.88%] |
| Raster | 0.0469 | 0.0488 | −3.52% | [−6.73%, −0.41%] |

Mark 达到至少 3% 的门槛，但 RenderGraph 下界低于 −2%，不能声明整图加速。Mark 时间降幅远小于 level 测试降幅，计数不能换算为耗时；receiver 重建和保留 sweep 的成本可能是主要下限，本轮没有硬件计数器证明。约每 receiver 20 个位置说明相当比例的射线仍有至少一级 sweep。

Agent 环境下，仓库根目录的 Low 完整性标签由 `build-release` 继承，从该目录启动的测试程序无法写入未带该标签的 `build/`。本次 GPU 运行使用哈希一致的副本 `build/vsm-mark-ray-pages/bin/MetallicRHITests.exe`。
