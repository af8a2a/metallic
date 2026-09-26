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

## 判断与边界

差异与 native resident mesh 的图像 handle lowering 相关。仅改变该部分即可恢复图像，但也会改变 shader 代码生成；这还不是一个排除了其他代码生成影响的最小根因证明，不能直接断言为已确诊的驱动 image-load 缺陷。

后续应构造 mesh 阶段图像尺寸与逐 texel 值的独立 GPU 回读，对照 compute 阶段、非零 descriptor 索引和不同 heap 布局，再判断 Slang 输出或驱动执行是否违反契约。当前不添加按入口切 mapped、BDA 或 shader 名称识别的生产特例，保持用户要求的统一 shader 资源用法。

证据：`.cache/typed-atomic-implementation/native-tessellation_*.json` 与 `resident-isolation/` 中的 `normalizer-comparison.json`、`cache-map.json`、各变体 JSON/log/SPIR-V。失败图片也保留在该目录下。
