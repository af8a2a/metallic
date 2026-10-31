# 方向光 Mark 页地址专用编译

2026-10-31：根据用户提供的 Nsight GPU Trace 和本机 Unreal Engine 5.7.4 源码，实现方向光 Mark 页地址移位候选。正确性通过；独立确认 Mark 收益 18.93%、Shadows 收益 7.08%、RenderGraph 收益 5.10%，预定门槛判定为 `accept / confirmed_gain / manual_promotion`。生产默认仍为关闭。

## Trace 与 Unreal 对照

输入为 `Captures/NsightGraphics/MetallicVSMSample_2026_10_31_19_52_43.ngfx-gputrace`，SHA256 `9412bdc5767b3ad004e568a2719f43cb26fdff96f0ec2f214f34b2b1d7ed4178`。匹配本机 WarpViz schema 的只读容器解析结果位于 `build/vsm-nsight-directional-20261031/Trace/`，原文件未修改。

捕获为 RelWithDebInfo MetallicVSMSample，Nsight 2026.3.1、RTX 5070 Ti、驱动 621.01，clocks unaltered、实时 shader profiling 和缓存 shader 重编译开启。内嵌截图明确显示 Render 1920×1080、窗口 2400×1350。相机向下倾斜，与受控 A/B 的固定 Sponza 相机不同；精确相机参数及捕获 shader 的字节身份未恢复。因此本轮 A/B 不是该捕获的精确重放，不能直接与截图时间相减。

截图中的 `vsmVirtualPage` / `vsmMarkPosition` 热点和 FP→INT64 依赖提示页地址计算值得检查，但依赖样本占比不是可回收 GPU 时间。容器的 processed PC ranges 为空，shader payload 为 SIMPLE obfuscation；本轮没有从原始 PC 数据复现截图的百分比，也没有给未确认时间域的 BOP 原始差附加毫秒单位。

本机 Unreal 5.7.4 的源码差异：

| 范围 | Unreal | 本轮取舍 |
| --- | --- | --- |
| 页地址 | `Engine/Shaders/Shared/VirtualShadowMapDefinitions.h:13` 固定 log2 page size=7；`Private/VirtualShadowMaps/VirtualShadowMapPageMarking.ush:106` 及 `VirtualShadowMapPageAccessCommon.ush:294` 使用移位取页、mask 取页内位置 | 采用整数页地址移位；保留 Metallic 可配置的 1–128 二次幂页尺寸 |
| 方向光标页 | `VirtualShadowMapPageMarking.ush:139` 选 biased clipmap，页边缘 dilation | 不移植 receiver-only 标页；Metallic 当前 SMRT 轨迹覆盖契约不同 |
| 射线计算 | `VirtualShadowMapProjectionDirectional.ush:84` 准备 UVZ 起点/步长，采样中线性递进 | 保留后续候选；先投影后插值与现有世界空间采样再投影的浮点舍入不同 |
| fallback | `VirtualShadowMapProjectionCommon.ush:224,237` 通过页表 LODOffset 和整数 clipmap 转换定位 fallback | 保留后续候选；需要页表及失效生命周期变更 |

Unreal 源码根目录为 `E:/UnrealEngine`，版本来自 `Engine/Build/Build.version`；未取得其 Git commit，不把版本号当成源码 commit 身份。

## 实现及契约

新增 `VirtualShadowMapSettings::specializeMarkPageSize=false`。两个 VSM Pass 均支持生产属性 `markPageSizeSpecialization`，显式布尔 `benchmarkMarkPageSizeSpecialization` 优先。

`validProjectionSettings` 已在资源与 shader 构建前保证 `pageSize` 非零、为二次幂且不超过 128。有效方向光开关据此计算 `markPageShift=std::countr_zero(pageSize)`，范围 0–7；关闭或局部光为 −1。该值进入 Programs 缓存键，仅 Mark entry 编译 `VSM_MARK_PAGE_SHIFT`。其他 entry 的日志值固定为 −1。

唯一算法改动是 `vsmVirtualPage` 中的 `uint2 page = texel / pageSize` 在有效宏下变为 `texel >> VSM_MARK_PAGE_SHIFT`。对全部 uint 输入，二次幂除法与逻辑右移整数结果一致。浮点投影、floor、uint 转换、采样、页集合、质量、资源布局及 Projection 均未改变，也未开启此前独立的方向光 Projection 候选。

## 正确性与编译证据

- 新 GPU 等价测试通过 98 个状态：82 主状态、页尺寸 1/2/4/8/16/32/64/128 的 16 个 Cold/Warm 状态。其中 10 个状态增加默认、production、benchmark 优先级和逐帧开关控制。
- 两个持久 VSM 对象执行页尺寸 1→128→16→1，共 8 帧，验证输出及 Programs 缓存切换；原 clipmap mutation 的 8 帧也通过。Validation 为 0。
- Sponza 第 16/32/63 帧 PNG、HDR、depth、shadow 逐字节相同；16–63 帧 stats32、45 项 build diagnostics、15 项稳定工作量精确一致。仅保留既有 Mark page-request 原子调度计数例外。
- Release Metallic/MetallicVSMSample/MetallicRHITests、RelWithDebInfo MetallicVSMSample 构建通过；shader ownership audit 388 文件通过。CPU runner 57/57、最终专项 3/3、attribution 8/8 通过。
- 实际普通 Mark SPIR-V：`OpUDiv` 4→0、`OpShiftRightLogical` 40→44、`OpConvertFToU` 4→4、`OpTypeInt 64` 两臂均为 0。10 个实际模块通过 Vulkan 1.4 `spirv-val`。
- Mark 原生代码 72448→64640 bytes（−10.78%），寄存器 40→40、shared 2048→2048 bytes。Build/Projection/raster mesh/depth fragment 的实际 SPIR-V 两臂相同。该静态数据不证明 occupancy 或 stall 原因。

独立 native representation 查询成功但驱动返回空列表，没有可读 SASS；因此不声称已经证明截图中的 FP→INT64 依赖消失。

首轮 `Regression/` 保留为无效尝试：并发构建触发 shader source epoch 更新；新增 pageSize=1 测试还暴露了不适用的 Warm 零重绘断言。现有方向光缓存规则使用 `2/pageSize` 边距，尺寸 1/2 的可复用区间为空，必然全重绘。测试改为严格断言该既有行为，其他尺寸仍检查 Warm 零重绘，未放宽 A/B 图像标准。停止构建后的 `Regression-v2/` 全部通过。受限环境 RelWithDebInfo C1902 在正常 MSVC 服务环境重建后解决，无编译配置变更。

## 独立确认性能

RTX 5070 Ti / 621.01，1920×1080，固定 Sponza 相机，6 层 clipmap、2048 页、SMRT 2×16、temporal 开启。两臂保留默认 Mark clipmap/ray-page 优化；本轮仅切换页尺寸专用编译。第 16 帧冻结 streaming 并 camera cut，每进程 96 帧，前 32 帧预热；计时关闭 validation、readback 与 native resource capture。shader/PSO 缓存已由独立诊断预热，未控制 clocks，不声明系统级 GPU 独占。

预算预定为 2 个 discovery ABBA 区组和条件触发的 3 个 confirmation 区组。以下仅为 12 个新确认进程，无排除、无追加、无 spread 警告。绝对时间为每臂 6 个进程中位数的中位数；收益与 95% CI 使用完整 ABBA 区组 log 比值，故不必等于表中绝对值直接比值。

| GPU 范围 | A ms | B ms | 区组收益 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Mark | 1.3186 | 1.0712 | 18.93% | [18.30%, 19.56%] |
| Shadows | 3.4293 | 3.1969 | 7.08% | [3.19%, 10.82%] |
| RenderGraph | 6.9019 | 6.6379 | 5.10% | [1.03%, 9.00%] |
| Projection | 1.0213 | 1.4169 | −35.30% | [−49.81%, −22.19%] |
| Allocator | 0.0853 | 0.0690 | 18.90% | [18.55%, 19.24%] |
| Build | 0.1588 | 0.1484 | 21.83% | [−27.74%, 52.16%] |
| Raster | 0.0477 | 0.0457 | 4.46% | [3.48%, 5.44%] |

Projection scope 增长是实测回退，不能因为 shader 未改就忽略。其范围包含同步，当前尚未确立 native 指令层面的因果；Build 也有较宽分布。独立阶段中位数不可相加解释 Shadows。预定验收针对 Mark 收益下界≥3%、RenderGraph 收益下界≥−2%，两项均通过；这不代表所有阶段都加速，也不改变默认值。后续应先定位 Projection 范围增长，再评估与正交 Projection 候选组合的收益。

没有测试移动相机整场景性能，也没有新采集同视角的前后 Nsight Source/IL。

## 证据与使用

启用生产属性：`markPageSizeSpecialization: true`。页尺寸变化会重新选择对应编译程序，局部光保持通用路径。

本机证据目录为 `build/vsm-mark-page-size-20261031/`：`SummaryReport/report.md` 与 `Confirmation.png` 为汇总及图表；`PlotReport.py`、`PlotProvenance.json` 保留复现方式、输入哈希和版本；`Diagnostic/Discovery/Confirmation` 为原始密封包；对应 `*Report` 由当前 verifier 重新核验生成；`CompilerReport/InstructionAudit.json` 保留静态 IR 与原生查询边界。`FinalVerification.json` 确认 816 个源文件、20 个二进制、12 个确认进程 shader 身份及原始 trace 哈希。

运行协议见 [VirtualShadowMapMarkPageSize.md](../Tools/Perf/VirtualShadowMapMarkPageSize.md)。原始 capture、生成图和实验包未加入源码。
