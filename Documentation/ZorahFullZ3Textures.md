# ZorahFull Z3：受预算约束的纹理资源

2026-09-18，基于 `612b0b305`（Z2）。Z3 完成 KTX2/BC 纹理资源、预算选择、上传与寻址验收。真实 ZorahFull 的 4418 张纹理已完成 GPU 上传；未执行全场景几何 cook，也不将此记为完整材质首帧。机器可读结果见 [ZorahFullZ3Validation.json](E:/metallic/Documentation/ZorahFullZ3Validation.json)。

## 实现

- 新增静态 2D KTX2 reader，支持 BC4 UNORM、BC5 UNORM、BC7 UNORM/sRGB，以及 raw/Zstd mip 数据。按 magic/vkFormat 判断格式，检查范围、重叠、mip 块长度与解压结果；不依赖 glTF 的错误 MIME。Full 有 4069 个 `image/png` 标注，实际均进入 KTX2 路径。
- 先检查所有图像头与 mip index，再选择最长边上限以内的完整尾链。只创建尾链尺寸的 image，将选中首 mip 重设为运行时 mip 0；不分配原始全尺寸 image。每个 mip 独立读盘、Zstd 解压，BC 块直接上传。
- `materialTextureMaxDimension` 默认 **512**，`materialTextureBudgetMiB` 默认 **2048**。使用 Vulkan image memory requirements 预估所有 KTX2 的分配；超预算时全局逐次减半尺寸上限，最粗尾链仍超预算则明确失败。上传时再次校验 VMA allocation size。
- RHI 增加 BC 格式、块行距/切片距、边缘小块上传，以及 image view swizzle。普通 view 与 `VK_EXT_descriptor_heap` 的 view 重建均应用 swizzle。GPU 测试发现并修复了后者遗漏。
- 保留 KTX2 `111r` / `1rg1` 通道含义。BC5 normal 从 XY 重建 Z；BC7 sRGB 由硬件解码，线性 KTX2 与 sRGB 采样结果均不再重复做 shader sRGB 转换。旧 PNG 仍采用原有 UNORM view 与手动颜色解码。未改变 authored normal/TBN 的朝向规则。
- 消除共享材质资源及相关 shader 的 256 限制，描述符按实际 image view 数量分配。逻辑 texture ID 映射到本代稳定 view 索引，索引 0 是白色 fallback；同 image 的逻辑引用复用 view。path trace、deferred、RTXDI、材质可视化和 resident shadow 的描述符布局使用实际数量，管线缓存也纳入数量变化。
- StreamAsset raster 通过 `SceneResourceManager` 取得与 deferred 共用的纹理资源及逻辑映射；预算/尺寸参数进入缓存键。发布沿用 upload/acquire timeline，最多三个在途上传批次；消费帧保留资源所有权至完成。旧的 resident alpha/displacement preview 上传器保留，Full 的 stream 路径不再走它。

KTX2 的 mip index、Zstd 与 swizzle 语义依据 [Khronos KTX 2.0 规范](https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html)。Zstd 使用 [官方 1.5.7](https://github.com/facebook/zstd/releases/tag/v1.5.7)，由 CMake FetchContent 下载并核对 SHA-256；无新增 DLL 部署要求。

## Full 实测

RTX 5070 Ti，启用 Vulkan validation。只加载 stream metadata、实例 accessor 与纹理尾链，不读取完整顶点/索引 payload。全量纹理使用默认 512 / 2048 MiB 策略。

| 项目 | 实测 |
| --- | ---: |
| 逻辑纹理 / KTX2 image | 4418 / 4418 |
| 有效 descriptor（含 fallback） | 4419 |
| BC mip-tail payload（含 fallback） | 1,458,267,388 B，1.358 GiB |
| image allocation 总计 | 1,469,821,440 B，1.369 GiB |
| 纹理预算 | 2,147,483,648 B，2 GiB |
| 峰值 staging（最后一轮 / 多轮最大） | 134,217,728 B / 201,326,592 B，128 / 192 MiB |
| 上传批次 | 369 |

Allocation 指各 image 的 VMA 子分配大小，**不是整卡显存、VMA heap 总预留或总渲染内存**；不含 descriptor heap、环境图、几何、CLAS、历史帧、staging、旧资源代。预算按资源代计算，多场景缓存或尺寸策略切换时的并存内存尚未纳入全局准入，这部分属于 Z6。

所有选中 mip 完成解压/上传，逻辑映射逐项检查；GPU 精确像素对照在合成 BC 图案上进行，Full 另采样末尾 descriptor。此验收不代表全部 Full texel 已与独立解码器逐像素比对。采样期间整卡有其他渲染负载，因此不报告可比较的加载耗时或 FPS。

## 验证与复现

`MetallicGPUDrivenSample`、`MetallicRhiTests` Release 构建通过。最终一轮 12 项测试全部通过、无 skip，进程正常退出，日志无 Vulkan validation 错误。原始结果：[final-tests.json](E:/metallic/build-release/zorah-z3/final-tests.json)，[日志](E:/metallic/build-release/zorah-z3/final-tests.log)。覆盖：

1. 300 张合成 KTX2：NPOT 7×5→3×2→1×1，BC4 `111r`、BC5 `1rg1`、normal Z，BC7 sRGB/线性、索引 >255、共享 owner、尺寸策略切换和错误 mip 长度拒绝。
2. 1 MiB 预算：初始上限 1024 自动降到 512，实际分配 987,136 B，采样正确。
3. BC4/5/7 的 64 B padding 行上传与 GPU 字节回读；拒绝非块对齐 offset 和非边缘半块，包含 1×1 尾 mip。旧 RGBA streamer 上传也通过。
4. Full 全纹理资源验收；OpenPBR、RTXDI、guides、stream shader 编译；已有 PNG 材质、材质可视化、alpha mask 渲染及 stream metadata 回归。

```powershell
cmake -S . -B build-release -DMETALLIC_BUILD_TESTS=ON
cmake --build build-release --target MetallicRhiTests MetallicGPUDrivenSample -j 4
New-Item -ItemType Directory -Force build-release/zorah-z3/rhi | Out-Null
build-release/tests/MetallicRhiTests.exe --rhi-bindless --filter ktx2_texture_resources --output-dir build-release/zorah-z3/rhi
$env:METALLIC_ZORAH_Z3_FULL='1'
build-release/tests/MetallicRhiTests.exe --rhi-bindless --filter zorah_texture_resources --output-dir build-release/zorah-z3/rhi
```

在 MSVC 开发环境中构建。离线构建可用 `FETCHCONTENT_SOURCE_DIR_METALLIC_ZSTD` 指向已解压的 1.5.7 源码。新增 `--rhi-bindless` 可独立启用 heap，不启动 Streamline；本机早期 `--rhi-realtime` 测试断言通过后曾停在 Streamline teardown，最终验收使用独立入口并正常退出。

## Z4/Z6 边界

Z3 支持此资产所需的外部静态 BC KTX2 子集，不提供 Basis transcoding、cube/array/3D、embedded KTX2 或其他 orientation。PNG 仍走旧 decode/mip 路径，其分配受预算检查，但不会参与 KTX2 的全局降 mip 计划。单张 PNG 的原始解码峰值仍需后续处理。

Full 全部逻辑纹理先纳入资源准备，包含尚未消费的 specular 等引用。specular/unlit 材质导入与求值、stream UV/normal/tangent 消费、正确 footprint、MASK 写深度前采样与 RT alpha、BLEND/玻璃都继续归 **Z4**。`VisibilityBufferMaterialPass` 仍是明确拒绝纹理场景的标量诊断消费者。

当前没有基于屏幕需求升降纹理 mip；sampler/filter、纹理热重载及跨场景/几何/CLAS 的总预算协调也没有新增闭环。下一步用 Z1/Z2 小探针验证 resident/stream 的材质像素，再进入 Z5 全量 cook。按视角持续细化与回收属于 Z6。
