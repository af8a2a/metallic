# MiniZorah M2：完整缓存的流式首帧

日期：2026-09-12。场景：`Asset/MiniZorah/zorah_main_public.v2.gltf`。M1 产物及源覆盖见 [MiniZorahCook.md](MiniZorahCook.md)。

**M2 已完成。** 全部 3,163 个 primitive、19,144 个 primitive instance 进入独立流式流程；两次独立进程启动和原始、远景、近景三个视角验收通过。结构化结果见 [MiniZorahFirstFrameResult.json](MiniZorahFirstFrameResult.json)。

## 启动入口

在仓库根目录运行：

```powershell
build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe --minizorah
```

也可在编辑器的 GPUDriven 分类选择 **GPUDriven / MiniZorah**。缓存必须已通过 M1 全量验证；缺失或过期时启动报错，运行时不会自动重新 cook。

配置：[gpu_driven_minizorah.metallic_graph.json](../Pipelines/Samples/gpu_driven_minizorah.metallic_graph.json)。从完成的 manifest 重新生成：

```powershell
python Tools/PrepareMiniZorahRuntime.py --page-mib 1024
```

该入口使用资产原始相机、自动 LOD（目标 1.5 render px）、硬件 mesh shader 光栅和简单几何光照，关闭 DLSS 与 CLAS。目标像素误差是细化请求条件，不能由此断言已经收敛。当前颜色是独立 StreamAsset shader 的简单光照；源材质 ID 保留，源材质参数着色属于 M3。

## 加载与预算

仅设置 `loadSceneInEditor=false` 不足以绕过普通加载：旧 StreamAsset pass 声明 World 依赖，RenderGraph 会再次通过 SceneResourceManager 导入整个源场景。新增 `streamAssetOnly=true` 后，该 pass 明确声明无 Scene 依赖，也跳过自己的普通 Scene 解析和 GPUScene source lease。

实例变换、primitive/material ID 直接来自缓存。缓存实例索引映射到该 pass 专有视图的可见性槽；它不是全局 GPUScene resident instance identity。GPUScene 的空 DrawSet 显式初始化，仅用于视图资源的版本校验、剔除和 HZB，不添加常驻几何或材质。M3 需要将此入口接入统一 VBuffer 的实例与材质 ownership。

| 资源 | 配置或缓存实测 |
| --- | ---: |
| 全场景 primitive / primitive instance | 3,163 / 19,144 |
| 根页 / 根页容量 | 3,163 / 4,096 |
| 根 payload（256 B 对齐） | 5,731,584 B，5.47 MiB |
| 根页加一个最大流式页下界 | 5,806,336 B |
| GPU page pool | 1 GiB |
| 额外驻留页数上限 | `maxResidentPages=0`，由字节预算约束 |
| active group 容量 | 131,072 |
| 每帧上传页上限 / CPU 页加载线程 | 256 / 4 |

1 GiB 是页池预算，不是进程总显存。共享 LOD 拓扑、每实例 frontier 状态、页表、帧槽、HZB 和渲染附件另外分配。完整 terminal 集合始终锁定，不能通过省略实例满足预算。页 ID/更新缓冲容量仍有界：运行时取资产页数与 `pageBytes / 256` 的较小值，本资产为 1,356,959。

验收中修正了两个配置问题。第一，变长页不能用“预算 / 最大页大小”推导最大驻留页数：初始 16,384 槽在近景耗尽时仍有 161 MiB 空闲，单帧出现 6,513 次分配失败；最终采用已有的纯字节预算模式。第二，完整世界包围球半径约 76,861，远景总览需从世界范围推导裁面，不能照搬原视点约 30,000 的 far plane；原视点和近景仍使用原始裁面。

## 验收方法

```powershell
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.streamasset_only_first_frame --rhi-validation --output-dir build-relwithdebinfo/minizorah-m2/bunny
$env:METALLIC_TEST_MINIZORAH='1'
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_stream_first_frame --rhi-validation --output-dir build-relwithdebinfo/minizorah-m2/cold
build-relwithdebinfo/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.minizorah_stream_first_frame --rhi-validation --output-dir build-relwithdebinfo/minizorah-m2/warm
```

完整场景测试为显式 opt-in，日常 CI 使用同一路径的 Bunny 回归。测试通过真实 Vulkan 渲染与 GPU readback 验证，不操作主桌面窗口：

- 使用发布配置测首像素和全部 terminal 页完成时间。
- 强制完整最粗 cut，读取 GPU active header 与全部 19,144 条根记录，逐项核对唯一实例、primitive、material、根页和 4×4 变换。
- 恢复自动 LOD，在原视点、远景、近景各运行 48 帧，验证可见性覆盖、保存几何光照图，并确认全部根页仍可用。
- 每帧断言 GPUScene resident 几何、实例、材质均为空；确认 CLAS 未创建、页加载失败与非法 GPU 请求均为零。

启动计时从测试开始，包含独立的缓存元数据核对、第二个 Vulkan 设备初始化、管线初始化和同步 readback，不等同于编辑器点击后的用户感知延迟。cold/warm 是两个独立进程的先后启动；没有清除操作系统文件缓存，因此不能称为断电冷盘测试。进程 I/O 计数包含着色器等其他文件，而且不完整计入内存映射页面缺页读取，不能当成缓存物理读盘字节。

## 实测结果

设备为 RTX 5070 Ti（16,303 MiB），驱动 616.64，RelWithDebInfo，1920×1080，开启 Vulkan validation。两次测试均执行 173 帧，全部根记录核对成功，三个采样的 active header 均为 `overflowCount=0`、`invalidCapacity=0`；页加载错误及非法请求累计为零，日志无 Vulkan validation error。

| 指标 | 首次独立进程（cold 标签） | 随后独立进程（warm 标签） |
| --- | ---: | ---: |
| 首像素 | 5.624 s | 5.586 s |
| 全部根页就绪 | 7.782 s | 7.724 s |
| 独立元数据核对 | 0.657 s | 0.395 s |
| 峰值进程提交内存 | 3.810 GiB | 3.786 GiB |
| 峰值工作集 | 1.912 GiB | 1.951 GiB |
| 全部观测完成（含 readback / PNG） | 35.464 s | 34.898 s |

共享 LOD topology GPU buffer 为 170,438,640 B，实例 LOD state 为 162,274,640 B。NVIDIA 全局显存采样为启动前 10,685 MiB、运行中 12,906 MiB，增量约 2.17 GiB；其他进程继续运行，该增量不是精确的单进程显存峰值。

| 视角 | 几何覆盖像素（cold / warm） | active groups（cold / warm） |
| --- | ---: | ---: |
| 原始 | 2,032,808 / 2,032,808 | 25,445 / 25,433 |
| 完整世界总览 | 47,995 / 47,995 | 19,144 / 19,144 |
| 近景 | 2,024,988 / 2,024,975 | 51,234 / 49,936 |

异步页面到达时机不同，固定帧数后的细化程度允许不同；全部实例身份和 terminal 集合必须一致。总览包含外围山体，主体建筑在该尺度下很小。

![原始视角，几何简单光照](../build-relwithdebinfo/minizorah-m2/cold/MiniZorah-original.png)

[完整世界总览](../build-relwithdebinfo/minizorah-m2/cold/MiniZorah-far.png) · [近景](../build-relwithdebinfo/minizorah-m2/cold/MiniZorah-near.png) · [cold 原始报告](../build-relwithdebinfo/minizorah-m2/cold/MiniZorahFirstFrameReport.json) · [warm 原始报告](../build-relwithdebinfo/minizorah-m2/warm/MiniZorahFirstFrameReport.json)

相关 9 项回归全部通过：GPUScene CPU 核心、source lease、global/view GPU buffers、取消提交恢复、无 Scene 的灯光 world、sample 加载、原 StreamAsset smoke、Bunny 纯流式首帧。[回归日志](../build-relwithdebinfo/minizorah-m2/regression.log)。`Metallic`、`MetallicGPUDrivenSample`、`MetallicRhiTests` 均编译通过；profile 与 manifest 生成结果一致，5 MiB 的不足根预算被拒绝。本轮未通过桌面 UI 启动编辑器，图像来自同一 sample/profile 的 headless Vulkan 路径。

## M3 / M4 的明确边界

**这些图像尚未证明 1.5 px 收敛，也未证明交互帧时。** 1 GiB 测试在最后近景采样仍有约 8,800 页排队上传、768 页 pending，页池约 98% 已分配。固定帧数后的样本没有容量分配失败，不表示继续细化或长期漫游不会遇到预算压力。

额外的 512 MiB 纯字节预算测试也通过全部根覆盖和三个视角验证，但最后近景仅剩 120,576 B 空闲，单帧分配失败 9,590 次；全套观测耗时 170.15 s。预算不足时 `allocatePageStorage()` 每次尝试都线性扫描驻留页寻找可淘汰对象，且近用页受 age 门槛保护；重复失败的 CPU 开销是明确的 M4 调度问题。[512 MiB 压力结果](../build-relwithdebinfo/minizorah-m2/pressure-512.json)。

下一步 M3 接入统一 VBuffer 的纯流式 Scene/GPUScene ownership、稳定实例/材质身份和源材质参数着色。M4 同时需要减少不可见细节需求、对缺页与淘汰进行批处理和优先级调度，再测不带 debug readback 的实际帧时；扩大页池不能替代这些工作。
