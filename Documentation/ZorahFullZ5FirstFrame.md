# ZorahFull Z5：全场景首帧

2026-09-20。完整源实例、材质和 512 mip 尾链已进入现有 VBuffer / CLAS / realtime OpenPBR 管线，完成全量 cook、真实首帧和重复加载/释放验证。更细 mip 的动态流送与持续漫游仍归 Z6。

交互验收补充：本报告运行条件为原生 960×540、DLSS 关闭。随后用户 Mini→Full 的交互式运行在 DLSS-SR 返回预算警告时停止渲染，不能用本报告宣称该路径已通过。已确认的错误映射、显存差距和下一步验收见 [交互加载计划](ZorahFullInteractiveLoadPlan.md)。

## 全量数据

| 项目 | 本机实测 |
| --- | ---: |
| 几何 + material 去重后几何 / primitive 实例 | 5715 / 43068 |
| 全 LOD 页数 / 必须常驻的终止页 | 1676913 / 51764 |
| 终止 cut 的实例 group 数 | 512130 |
| 浮点磁盘缓存 | 191.226 GiB |
| 原始根页 / 紧凑 GPU 根页（256 B 对齐） | 5.232 / 2.956 GiB |
| 几何池 / CLAS 池 / 纹理分配预算 | 3.5 / 2 / 2 GiB |
| 4418 张纹理实际 image allocation（包含兜底 image） | 1.369 GiB |
| 全量 cook 用时 / 峰值进程提交内存 | 3770.63 s / 3.220 GiB |
| 紧凑布局全页校验用时 / 峰值进程提交内存 | 661.03 s / 0.617 GiB |

根页压缩了 43.51%，使完整保底 cut 能进入当前小于 4 GiB 的几何池地址范围。法线修复涉及 225 份 cooked geometry、12324 个零长度法线；不把这些无有效方向的数据当成源法线保真样本。

原始与紧凑布局的全部 1676913 页均已通过校验：[Full.cook.json](E:/metallic/build-release/zorah-z5/Full.cook.json)、[Full.runtime.json](E:/metallic/build-release/zorah-z5/Full.runtime.json)。生成器输出与提交的 Full graph 逐项一致。显存统计必须区分池容量、有效页字节、scratch/帧资源和整卡占用；上述三个预算的和不是程序总显存。

## 实现

- 新增 `GPUDriven / ZorahFull` 内置场景和 `MetallicGPUDrivenSample.exe --zorah-full` 入口，加载 metadata 与预先生成的 meshstream。源文件为 `Asset/ZorahFull/zorah_textured_public.v1.gltf`。GPUDrivenSample 中选择 ZorahFull、File/Open 或拖入该源文件均切换完整 Full 预设，使用对应的相机、缓存和预算。
- 使用 Full cfg 的相机；保留全部 43068 个 primitive 实例。当前不应用 cfg 的 `skipmeshes`，不能把这份首帧数据直接当作与参考程序相同工作量的性能对照。
- 全量 cook 暴露部分 authored NORMAL 为零。对有限的零长度法线按相邻三角形重建；已有有效法线保持不变，非有限值仍报错，完全退化/孤立顶点有确定性兜底。不丢弃细小三角形。属性 cook revision 升到 2，旧的 attributed 缓存需重建；旧 position-only revision 0 兼容保留。
- Full 上传启用 `compactShadingAttributes`：复用常驻路径的 octahedral normal/tangent 编码，P/N/UV/T 从 52 B/vertex 降到 28 B/vertex，不含页表与对齐。位置、UV、三角形及材质索引保持原值，切线手性独立保留。磁盘 cook 仍保存浮点属性。
- 普通 streamed surface、CLAS ray hit、旧 resolve 与 tessellation normal 消费者均识别紧凑格式。raw / ByteRle 页可在上传时打包；带属性的 GPU-tile 缓存不能在上传阶段改变布局，因此显式拒绝此组合。
- `PrepareZorahFullRuntime.py` 从经过全页验证的 runtime manifest 生成根页容量、实例 cut 容量和预算；根页必须完整驻留，不能用任意较小预算掩盖缺失几何。

## 验证方法

最终完整记录：[first-frame-entry/ZorahFullFirstFrame.json](E:/metallic/build-release/zorah-z5/first-frame-entry/ZorahFullFirstFrame.json)、[rhi.json](E:/metallic/build-release/zorah-z5/first-frame-entry/rhi.json)。测试进程和 PowerShell 驱动脚本均以 0 退出。

| 同一 device 上的加载 | 首次完整 ready | 首帧时间（从本轮 metadata 开始） | 退出/stream retirement |
| --- | ---: | ---: | --- |
| 独立 asset 入口 | 第 397 帧 | 64.15 s | 通过 |
| 编辑器 world 绑定入口 | 第 397 帧 | 69.18 s | 通过 |

两轮均有 512956 个非黑 baseColor 像素；全部逻辑纹理 descriptor 指向有效的非兜底 image。geometry 使用量上限 3.5 GiB，CLAS 使用量峰值 1748.16 MiB，页加载错误和请求缓冲溢出均为 0。两轮各有 10804 次受几何池容量限制的页面分配拒绝，保底 cut 始终 ready。

采样到的整卡峰值为 15603/16303 MiB，包含桌面和其他进程；它不是 Full 的独占显存。纹理、geometry/CLAS 池之外仍有目录、遍历、光栅、BLAS/TLAS、scratch 和帧资源。该配置在当前 16 GiB 卡上余量较小；Z6 需要继续降低根页和临时缓冲占用，并建立总预算协调。

![Full cfg 相机的完整材质首帧](E:/metallic/build-release/zorah-z5/first-frame-entry/ZorahFull-first-ready-0.png)

最终入口验收没有与全量 CPU 页面校验并行；缓存已由此前测试预热，且原生分辨率逐帧 readback 会改变帧调度。这些秒数是可复跑的功能验收结果，不是严格冷启动或与 vk_lod_clusters 的等条件性能基准。

`RunZorahFullFirstFrame.ps1` 执行可选的 `RhiRendering.zorah_full_first_frame`，默认在同一个 GPU device 上加载/释放两次，分别覆盖独立 asset 入口和编辑器 world 绑定入口。960×540 原生分辨率，不启用 DLSS；交互式图保留 DLSS-SR。因此这份带逐帧 readback 的结果不是编辑器 FPS。

测试核对 cook/metadata 实例数、无全量常驻 vertex/index upload、全部终止页的 geometry/CLAS 与 fallback BLAS readiness、512 cap 的 4418 张纹理、页加载错误、geometry/CLAS 预算，以及切走空图后的 stream retirement。首次 ready 和其后 120 帧保存 PNG；另保存 baseColor 图并检查有效覆盖，避免仅天空背景通过。

后 120 帧只表示固定相机继续渲染，不代表 1.5 px LOD 完全收敛。实测细化后几何池达到 3.5 GiB 上限，出现页面分配拒绝计数；这是驻留预算回压，不能把它记成 Vulkan 分配失败，也不能将未进一步细化的内容记为质量收敛。环境使用项目现有 HDRI 和测试定向光，不宣称与参考程序逐像素等价。

GPU 属性回归覆盖材质/TBN、MASK、BLEND/玻璃续追和阴影。八个真实材质探针共 32 条记录，其中 29 条可做等价像素对照：最大平均 RGB8 通道误差 0.255758/255，最大离群像素比例 0.2392%。玻璃/BLEND final 与缺源法线 unlit mappedNormal 的比较边界沿用 Z4。原始记录：`build-release/zorah-z5/compact-probes/ZorahZ4Probes.json`。

Scene 属性测试 10 项通过，另启用并通过 9 个小探针的独立 resident 导入对照。4 项 GPU 材质/透明/阴影/编码回归及 3 项 sample/stream 打开回归通过。Full 运行启用 Vulkan validation；没有 VUID 错误或 DeviceLost。部分 PSO 缓存文件写入警告仍存在，不影响本次图像验收，不能忽略它们对下一次启动耗时的影响。

过程证据保留在同目录：`first-frame` 在 00:28 系统关机时中断；`first-frame-resumed` 已渲染但验收断言漏计了兜底 image；`first-frame-final` 的 RHI 测试通过，但 Windows PowerShell 未保留进程 handle，导致包装脚本读到空 exit code。修正计数和 handle 后，`first-frame-verified` 与上述最终双入口验收均通过。前面的失败/中断记录没有改写为成功。

## 复跑

```powershell
build-release/Source/MetallicMeshletCook.exe --source Asset/ZorahFull/zorah_textured_public.v1.gltf --output Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin --report build-release/zorah-z5/Full.cook.json --workers 4 --memory-mib 8192 --checkpoint-interval 32 --validate-payloads
build-release/Source/MetallicMeshletCook.exe --inspect --compact-shading --source Asset/ZorahFull/zorah_textured_public.v1.gltf --output Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin --report build-release/zorah-z5/Full.runtime.json --validate-payloads --memory-mib 8192
python Tools/PrepareZorahFullRuntime.py
powershell -ExecutionPolicy Bypass -File Tools/RunZorahFullFirstFrame.ps1
build-release/Source/MetallicGPUDrivenSample.exe --zorah-full
```

完整缓存及测试图像属于本地输出，不纳入 Git。动态纹理 mip 提升/回收、三类资源的总预算、持续漫游与透明软阴影统一仍属于后续阶段。
