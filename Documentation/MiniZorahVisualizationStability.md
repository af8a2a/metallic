# MiniZorah：转视角时的可视化稳定性

日期：2026-09-13。

## 定位结论

全局跳色的直接原因是调试颜色使用了每帧临时 visibility record 槽位。候选压缩、active group 输出顺序、可见性和 LOD cut 改变时，仍然存在的同一 cluster 可能分配到不同槽位。原 Meshlet ID 将 `recordIndex + 1` 哈希成颜色；Triangle ID 将包含该槽位的 `packedVisibility` 哈希成颜色，因此编号重排会让大片未更换几何同时换色。

LOD Level 还存在另一处问题：只有 resident 路径读取了真实 `lod.x`，stream 路径落回临时 record 编号。本次一并修复。

这解释了视频中的大面积颜色变化，不能用原来的跳色程度推断整个场景都在更换 LOD。实际 LOD 切换和视锥覆盖变化仍然存在，应与调试编号不稳定分开观察。

## 与参考实现对照

本地找到的参考目录为 `E:/vk_lod_clusters`。

| 路径 | cluster 着色依据 |
| --- | --- |
| 原 Metallic VBuffer | 每帧可见记录槽位 |
| Nanite | page index + page 内 cluster index |
| vk_lod_clusters | cluster ID；Triangle 模式再加入 triangle ID |
| 修复后 Metallic Stream | 逻辑 page index + page 内 cluster index |

依据：[NaniteVisualize.usf](E:/UnrealEngine/Engine/Shaders/Private/Nanite/NaniteVisualize.usf:418)、[vk_lod_clusters pathtrace](E:/vk_lod_clusters/shaders/render_pathtrace_hit.glsl:182)、[cluster 着色](E:/vk_lod_clusters/shaders/render_shading.glsl:55)。这些是对本地版本源码的观察。vk_lod_clusters 的 resident cluster ID 不等同于永不改变的资产 ID；本次选用逻辑 page 身份，避免依赖页面物理驻留位置。

## 改动

[VisibilityBufferComposite.slang](E:/metallic/Shaders/Features/VisibilityBuffer/VisibilityBufferComposite.slang:63) 先将临时记录解析到几何身份，再着色：

- Meshlet：resident 使用全局 meshlet index；stream 使用逻辑 page index 与 page 内 cluster index。同一几何的多个实例共享颜色。
- Triangle：稳定 cluster 身份再组合局部 triangle index。
- LOD Level：resident 和 stream 均使用实际 LOD level，同一层级显示相同颜色。
- Depth、Coverage、Off 保持原有显示语义。

可视化采用独立的 32 字节 [CPU push 参数](E:/metallic/Source/Runtime/Render/GPUDrivenRaster.h:10)。[VisibilityBufferPass](E:/metallic/Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp:2759) 在自己的 heap 中借用 stream records 和 active groups，新增两个 descriptor，不复制几何或每帧读回数据。资源重分配时重建并延迟回收旧绑定；光栅写入完成后增加记录读取屏障。

没有改变 packed visibility ID、候选顺序、软硬分类、LOD/cut、页面调度和 1.5 px 目标。真实 cluster/LOD 更换时，调试图仍会反映变化。此前硬件唯一顶点优化保留。

## 验证

RelWithDebInfo 的 Metallic、MetallicGPUDrivenSample、MetallicRhiTests 构建成功。最终 13 项回归全部通过，耗时 52.822 秒，无跳过项，Vulkan 验证日志无告警/错误。覆盖可视化、混合 resident/stream、Alpha Mask、冻结相机、异步软硬光栅、双帧槽、resize、shader reload、MiniZorah VBuffer 和唯一顶点光栅对照。

[合成 GPU 测试](E:/metallic/tests/rhi/VisibilityDebugStabilityTests.cpp:16) 实际绘制 composite shader 的六种显示模式。保持几何相同，同时更换记录槽位、active group 槽位、resident/stream 命名空间分界和页面物理偏移，输出逐位相同；另验证不同 cluster/triangle 的区分、真实 LOD、实例共享颜色及无效记录回退。页面搬迁在此测试中通过修改物理偏移模拟。

[完整场景测试](E:/metallic/tests/rhi/MiniZorahDebugStabilityTests.cpp:78) 在 1920×1080、1.5 px 下，固定 eye，依次转向 0°、−3°、+3°、0°，测试 Meshlet/Triangle/LOD 共 12 个组合。每七个像素取样，按几何身份匹配颜色，避免把相机转动后的不同屏幕像素直接比较。

- 同一几何/LOD 的颜色不一致：**0**。
- 匹配颜色样本：**3,224,838**，包含同帧及跨帧重复身份，不是唯一 cluster 数。
- 临时记录槽位重映射：**218,842** 次，按三个模式、相邻视角累计，不是全局唯一实例数。

Meshlet 模式中，相邻视角共有身份占上一视角采样身份的 74.8%–82.5%，这些身份均保持同色：

| 当前 yaw | 当前采样 cluster 身份 | 与上一视角共有 | 占上一视角 | 改变槽位的实例记录 |
| --- | ---: | ---: | ---: | ---: |
| -3° | 11,521 | 9,755 | 82.5% | 14,415 |
| +3° | 11,819 | 8,618 | 74.8% | 11,291 |
| +0° | 11,830 | 9,591 | 81.1% | 14,896 |

剩余身份差异包含视锥覆盖、真实 LOD 选择和屏幕采样差异，不能直接作为 LOD 抖动率。本轮解决临时编号导致的全局换色；如后续仍要分析真实 cut 抖动，应基于稳定的 instance/page/cluster 身份统计。

最终日志：[final.log](E:/metallic/build-relwithdebinfo/visibility-stability/final.log)。结构化结果：[MiniZorahVisualizationStabilityResult.json](E:/metallic/Documentation/MiniZorahVisualizationStabilityResult.json)。最初新增测试的退出阶段存在读回缓冲晚于设备析构的问题，已修正；最终回归包含完整退出过程。

## 查看与复现

需重新启动本轮编译后的编辑器，以加载新增的可视化参数绑定；只热重载 shader 不足以更新正在运行的旧二进制。

```powershell
$env:METALLIC_TEST_MINIZORAH='1'
./build-relwithdebinfo/tests/MetallicRhiTests.exe '--gtest_filter=*visibility_debug_stable_geometry_identity:*minizorah_debug_identity_stability' --rhi-validation --rhi-async-compute --output-dir build-relwithdebinfo/visibility-stability/repeat
```

修复后入口视角：

![MiniZorah Meshlet ID](E:/metallic/build-relwithdebinfo/visibility-stability/final/MiniZorah-meshlet-0.png)

向左转 3°：

![MiniZorah Meshlet ID yaw -3](E:/metallic/build-relwithdebinfo/visibility-stability/final/MiniZorah-meshlet-1.png)
