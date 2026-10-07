# Native resident 细分图像差异

日期：2026-09-26。该问题在 typed uint64 指针链修复解除管线编译崩溃后暴露，尚未修复。不要把它与混合位宽原子的 CPU 驱动异常混为一谈。

## 当前证据

- `tessellation_displacement_render` 和 `tessellation_recursive_render` 的 native 运行不再发生管线创建崩溃，但 resident 图像与显式位移参考几何不同；同轮 stream 组合没有报告图像差异。
- mapped 位移测试的 64 个 resident/stream 组合通过。native mixed-producer 测试及六种软件 raster 的实际 GPU 执行通过。
- 受影响的 resident mesh 模块不包含 uint64 类型。对同一原始模块分别应用本轮开始前的 normalizer 与新 normalizer，得到逐字节相同的 60,836 字节结果，SHA-256 为 `7b10c4138cf7c3ababecf107c33569269a112a7f8d8139a38738b366495b4826`。因此 typed uint64 策略没有改变该模块。

## 隔离对照

所有替换只发生在已备份的本地 shader cache；每次独立进程退出后立即恢复。没有把以下诊断变体纳入生产。

| 变体（其他阶段继续 native） | 位移渲染结果 |
| --- | --- |
| 原 native resident mesh | resident 图像失败 |
| resident mesh 使用原始 typed buffer 指针，跳过 untyped normalization | 相同图像失败 |
| 仅 resident task 改为已验证的 mapped 字节码 | 相同图像失败 |
| resident mesh 改为 mapped 字节码 | 64 个组合全部通过 |
| resident mesh 保留 native buffer，仅 height texture 使用 mapped descriptor lowering | 64 个组合全部通过 |
| native image 的 Depth=2 改为 Depth=0 | 失败 |
| 显式为 native image load、pointer 和 index 添加 NonUniform 装饰 | 失败 |
| native image 的 OpConstantSizeOfEXT 改为本机实测 stride 32 | 失败 |

本机 imageDescriptorSize=32、imageDescriptorAlignment=32，sampled/storage image 的具体 descriptor size 均为 32。单纯 stride 数值或 NonUniform 装饰无法解释上述差异。源码层 NonUniformResourceIndex 的第一次试验没有在此 native 路径生成装饰，因此额外做了显式 SPIR-V 装饰试验。

## 2026-10-07 驱动 617.42 复测

RTX 5070 Ti 从 NVIDIA 616.92 更新到 617.42 后，使用与更新前 SHA-256
完全相同的 17:49 RHI 测试程序及 Slang 2026.18.2 DLL 复测。保留生产
native pointer normalization、typed uint64 策略和现有 shader cache；没有
替换 cache 字节码或添加按 shader 入口切换 mapped 的特例。

| 图像对比矩阵 | native | mapped |
| --- | --- | --- |
| 位移，64 个组合 | 32 个 resident 组合失败；32 个 stream 组合无图像差异 | 64 个组合全部通过 |
| 递归位移，192 个组合 | 96 个 resident 组合失败；96 个 stream 组合无图像差异 | 192 个组合全部通过 |

差异仍覆盖 resident 的 `baseColor` 和 `shadingNormal`。普通位移各有 16
个失败组合，递归位移各有 48 个失败组合；包括不同投影、负 X 缩放镜像
（同时关闭 `clusterPrebin`）、edgePixels、递归拆分深度和 legacy depth
设置。native 测试在收集完矩阵差异后
返回失败，未执行后续 live material edit 检查；mapped 完成了这些检查。
这里的 64/192 是图像对比组合数，不是提交帧数。

这四项完整像素矩阵在 native/mapped 两边都关闭 VVL；首次开启官方 VVL
1.4.363 的 native 递归运行在 300 秒超时，不计作完成或通过。独立的
descriptor 布局、采样和 MixedProducer 核心检查仍使用 VVL，见
[literal stride 验证记录](NativeDescriptorHeapStrideWorkaround.md)。新驱动
未消除这类 resident 图像差异；本轮没有重新运行跳过 normalization 的
raw pointer 隔离变体，不能据此判断其兼容处理可否删除。

完整 XML、失败图像、参考图像和环境/hash 记录位于本地
`.tmp/spirv-removal-research/Driver61742/20261007-184901-959/`。初始 VVL
超时记录位于 `20261007-184136-876/`；这些是本地验证输出，不纳入源码。

## 判断与边界

差异与 native resident mesh 的图像 handle lowering 相关。仅改变该部分即可恢复图像，但也会改变 shader 代码生成；这还不是一个排除了其他代码生成影响的最小根因证明，不能直接断言为已确诊的驱动 image-load 缺陷。

后续应构造 mesh 阶段图像尺寸与逐 texel 值的独立 GPU 回读，对照 compute 阶段、非零 descriptor 索引和不同 heap 布局，再判断 Slang 输出或驱动执行是否违反契约。当前不添加按入口切 mapped、BDA 或 shader 名称识别的生产特例，保持用户要求的统一 shader 资源用法。

证据：`.cache/typed-atomic-implementation/native-tessellation_*.json` 与 `resident-isolation/` 中的 `normalizer-comparison.json`、`cache-map.json`、各变体 JSON/log/SPIR-V。失败图片也保留在该目录下。
