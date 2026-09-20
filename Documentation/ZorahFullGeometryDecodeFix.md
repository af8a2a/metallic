# ZorahFull 已加载但仅显示背景：几何解码修复

2026-09-20，针对用户 15:01 日志与截图。

## 根因与证据

本次不是加载入口或纹理上传失败。Full 页数据成功准备，VBuffer meshlet 输出存在完整建筑几何；延迟着色重建表面时却全部被拒绝。

关闭环境背景（保留环境照明）后，修复前最终画面只有黑色，meshlet 输出仍有 103352 种颜色。逐阶段 GPU 诊断将拒绝点定位到 `decodeStreamTriangleAtPage` 的页容量检查：当前 descriptor-heap 路径对 3.5 GiB 的 `StructuredBuffer<uint>` 调用 `GetDimensions`，实测返回 count=0、stride=4，造成所有有效页面被当成越界。光栅路径已经使用显式 `pageBufferBytes`，因此仍能画出 meshlet。

这证明了当前 Full 配置上的查询异常，尚未将原因进一步归属到编译器或驱动，也未推定通用的失效尺寸阈值。

## 修复

- 共享几何解码函数改为接收流送页池的明确容量，不再依赖大 buffer 的维度查询。保留页表、页面范围、cluster/triangle 和顶点范围检查。
- 从已有每帧 `MeshletStreamGpuParams` 读取 `pageBufferBytes`；CPU 用 `offsetof` 静态断言对应偏移 100。复用既有每帧缓冲和 owner 生命周期。
- Deferred、简单材质预览以及流式 ray query / 阴影共同传入这一容量，避免只有主着色修好、其他路径仍漏几何。
- Statistics 在编辑器传统 scene 为空时读取 GPUScene 的流式 source metadata，避免显示误导性的 “No scene loaded”；常驻页数据仍在 Profiler / Streaming。

## 防止再次误验收

此前报告把森林环境背景误判为 Full 场景图像。页就绪、输出非黑和彩色像素数均不足以单独证明场景着色，原报告已纠正。

原生测试现在隐藏背景，同时捕获 meshlet、最终着色和 base color，分别检查材质分组及非分组路径。真实 Editor / DLSS 切换测试也隐藏背景，再读回实际 presentation output 检查表面覆盖。

## 原生测试结果

Release，Vulkan validation，960×540，MiniZorah 先提交帧后切入 Full；asset/world 两轮均通过。每轮第 397 帧达到 103528/103528 根页就绪，卸载后 stream session 清空。

两轮 base color 均覆盖 **389386 / 518400 像素（75.11%）**；关闭 material binning 后也为 389386。最终首帧颜色数分别为 52308 / 52304。已检查图片，建筑、地面、雕塑与材质实际可见，背景为黑色。

- [运行日志](../build-release/full-geometry-fix/after.log)
- [两轮数据](../build-release/full-geometry-fix/after/ZorahFullFirstFrame.json)
- [真实着色首帧](../build-release/full-geometry-fix/after/ZorahFull-first-ready-0.png)
- [Meshlet 对照](../build-release/full-geometry-fix/after/ZorahFull-meshlets-0.png)

此验收不代表长时间自由漫游或全部近景保真已经完成。

## 真实 Editor / DLSS 验收

通过应用隐藏窗口 CLI 运行真实编辑器，1404×674、完整 DLSS 与当前 HDR 输出。连续两轮 MiniZorah → Full，均在第 397 帧达到 103528/103528 页就绪，再继续渲染 60 帧。第二轮最终输出读回为 **603174 / 946296 有效着色像素**（环境背景隐藏；RGBA16F 检查有限正 RGB，alpha 不计入）。同时确认流式 metadata 有 43068 个 render nodes，Statistics 使用相同来源。

测试随后主动注入 UINT64_MAX reservation 触发一次预算拒绝，验证不会逐帧重试；恢复预算并显式 retry 后成功。这一条预期 OutOfMemory 不属于加载回归。

- [Editor / DLSS 最终日志](../build-release/full-geometry-fix/editor-final.log)
- 复现：`METALLIC_SMOKE_TEST_HIDDEN=1`、`METALLIC_SMOKE_TEST_ZORAH_FULL_SWITCH=1`，运行 `build-release/Source/MetallicGPUDrivenSample.exe --smoke-test`。

首次新增读回测试只接受 RGBA8，因此在当前 HDR 环境中提前退出；已支持实际 RGBA16F 输出，最终执行退出码为 0。

## 补充回归与边界

`stream_material_shading`、`stream_material_transmission` 通过。

扩展检查中的 `visibility_buffer_material_edit_refresh` 在材质 alpha 编辑阶段出现 SEH 0xc0000005；`realtime_ray_traced_sigma_shadows` 的 penumbra 断言为 0,0,0。为判断归因，仅撤回本次共享解码 shader、三处消费者绑定和容量偏移断言共 8 个文件，再构建运行同两项测试，仍分别以相同原因失败。未更改这些断言来取得通过；这两项现存回归不属于此次修复，仍待单独处理。对照后已恢复所有修复文件并重新构建。

- [修复版扩展回归](../build-release/full-geometry-fix/regression.log)
- [撤回解码修复的对照](../build-release/full-geometry-fix/baseline.log)

最终恢复构建成功；`stream_material_shadow`、`stream_metadata_contract` 两项也通过，覆盖 CLAS MASK 阴影和简单材质预览共用解码路径。`git diff --check` 通过。
