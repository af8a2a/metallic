# ZorahFull T1：减少首帧纹理驻留

2026-09-20。GPUDrivenSample 与 Full 预设已更新，MiniZorah 的纹理策略不变。

## 实现

- Full 普通纹理 cap 从 512 降为 **256**，纹理 allocation 上限从 2048 MiB 改为 **512 MiB**；统一预算不足时继续降低普通图的 cap。
- 新增 `materialTextureMaskMaxDimension`（默认 0，兼容原策略），Full 设置 **512**。按 source image 聚合 MASK 基色引用，同一 image 被不同材质或逻辑 texture 复用时采用较高需求。Full 共 **110 张**受保护图片；预算不足时不会偷偷降低它们，最小工作集仍放不下则返回明确错误。
- VBuffer、Shadows、Deferred 使用相同三项纹理配置与资源键。MASK 策略参与缓存键、配置匹配、VBuffer 编译复用及资源重建；变更材质 alpha 类型/引用也沿用已有 layout 失效判断。
- T1 开发时的完整场景验证发现：仅修改 VBuffer/Deferred 会使 Shadows 沿用默认 512/2048，产生第二套贴图。现已同步阴影配置，并用全设备 material-texture 域的 allocation 数/字节断言三个消费者没有重复副本。旧 T0 中三者默认值一致，不应把此次策略分歧描述为原先必然存在的重复加载。
- 小型 device-local 材质 image 使用按 memory type 隔离的 **16 MiB VMA 内存池**。只改变材质图片，保留大型图片和驱动要求 dedicated 的路径，普通帧资源、buffer、RTAS 不变。新开块前按整块 16 MiB 检查共享额度，复用已有空闲范围不重复扣 heap；池生命周期跟随 device，VMA 可保留空闲块用于后续场景。
- 小图规划预留最多 **32 MiB** 块余量，代替 T0 按 4419 张 dedicated 图片累计约 276 MiB 的估计。较大图片策略继续采用保守的逐图估计；最终仍以逐分配 heap admission 为准。
- Full 预设生成器同步新参数，避免下次生成覆盖优化。CPU 有界预取、三批上传和 GPU 完成后发布的流程保持不变。

## 数据

4418 张 Full KTX2、4419 个 GPU image（含 fallback）。普通/受保护 image 均加载选定 mip 到最小 mip 的完整尾链；不是仅调整 sampler。

| 方案 | 普通 cap | MASK cap | image allocation | 纹理测试 local heap usage |
| --- | ---: | ---: | ---: | ---: |
| T0 原 512 对照 | 512 | 512 | 1401.73 MiB | 1629.18 MiB |
| T1 默认 | **256** | **512** | **373.55 MiB** | **440.55 MiB** |
| T1 小预算回归 | **128** | **512** | **116.26 MiB** | **184.55 MiB** |

默认 image allocation 减少 **1028.18 MiB（73.35%）**。heap usage 还包含 driver、分配器和采样探针，不等于纯纹理，也不是整卡 NVML 占用；不同测试上下文使它只能作为辅助证据。

T1 默认纹理测试 local heap 为 **26 blocks / 4420 allocations**，小预算为 **10 blocks / 4420 allocations**：逻辑纹理和 image 数量保持完整，减少的是分配块及驻留 mip。默认测试新增 local 额度 1024 MiB、小预算测试 512 MiB，均另设 64 MiB safety 并持有 128 MiB future reservation。默认仍受 Full 配置的 512 MiB allocation 上限约束。

证据：[默认纹理](../build-release/first-frame-residency/normal/zorah-textures.json)、[小预算](../build-release/first-frame-residency/pressure/zorah-textures.json)、[T0 对照](../build-release/gpu-memory-budget/regression/zorah-textures.json)。加载耗时没有作为同条件性能提升结论：缓存和机器负载不同。

## 完整 Full 验收

开启 Vulkan validation、统一预算，使用既有 cfg 相机、全量 meshstream、960×540 原生渲染。测试不启用 DLSS，包含完整根级准备、材质/阴影图、base-color 覆盖与卸载重载。

| 入口 | 根级就绪帧 | 到根级就绪 | 材质域实际字节/图片 | 结果 |
| --- | ---: | ---: | --- | --- |
| asset | 397 | 31.91 s | 391696896 B / 4419 | 通过 |
| world | 397 | 31.38 s | 391696896 B / 4419 | 通过 |

两个入口都只保留一套材质图片，普通 cap 256、110 张 MASK 基色 cap 512；完整几何没有被传统 resident 导入。图移除后 stream session 数归零，第二轮再次完成。验证记录：[ZorahFullFirstFrame.json](../build-release/first-frame-residency/full-shared/ZorahFullFirstFrame.json)、[运行日志](../build-release/first-frame-residency/full-shared.log)。

![Full 完整根级首帧，256 普通纹理与 512 MASK 基色](../build-release/first-frame-residency/full-shared/ZorahFull-first-ready-0.png)

**2026-09-20 纠正：上图实际是环境背景，旧测试把背景误判为几何覆盖。** 上述纹理分配、页就绪与卸载数据仍有效，但不能证明 Full 几何着色成功。现已隐藏环境背景重新验收，定位并修复大页池容量查询导致的解码拒绝，见 [Full 几何解码修复](ZorahFullGeometryDecodeFix.md)。

## 回归

- Release Sample 和 RHI 测试构建通过；7 项核心回归通过，覆盖 KTX、BC 上传、资源生命周期、共享预算、上传流水线和图重建。
- 合成 MASK 场景验证：普通图片降尾链，MASK 图片独立保持更细尾链；GPU sRGB/swizzle/alpha 采样与原 mip 一致；相同策略共享 owner，改变 MASK 策略不会复用错误 owner。
- 预算装不下 MASK 下限时，返回带 `MASK quality floor` 的 `OutOfMemory`，不发布部分场景。[下限拒绝回归](../build-release/first-frame-residency/floor-tests.json)。
- Full 全部 4418 张图片逐项核对 first mip，校验解码 mip 总数、全部逻辑映射和末尾描述符采样；两档预算通过。完整 Full 两轮测试通过。
- 生成器输出到临时文件验证，三个消费者参数一致。`git diff --check` 通过。

## 后续

这版仍是**加载时静态尾链驻留**，普通纹理进入场景后不会自动恢复到 512 或源分辨率。可见性/UV 导数驱动的升级、冷纹理回收和迁移峰值控制继续归 T2/T3。MASK 的 512 是保持此前默认覆盖质量，不代表全部视角下达到源贴图精度。

完整 DLSS 图及连续自由漫游还需独立验收。geometry/CLAS 固定池容量未缩小，池内 stream 回收继续沿用既有机制；不能将纹理节省量当作整个场景显存缩减比例。
## MiniZorah → Full 切换补充验收

14:33 用户日志暴露了旧场景仍被已完成的帧槽持有的问题。已修复回收点，并补充原生两轮切换、1404×674 完整 DLSS 图两轮切换与失败重试验收，见 [场景切换修复记录](ZorahFullSceneSwitchFix.md)。上文的单独 Full 加载测试未覆盖这一生命周期缺口。

2026-09-20 T2/T3 更新：已接入 GPU 采样需求驱动的 mip 细化、预算内 image 替换与冷回收。Full 默认基础上限改为 128，可见图按需恢复至 512；MASK 保持 512。实现、实际往返数据及边界见 [按需细化与冷回收](ZorahFullTextureStreaming.md)。
