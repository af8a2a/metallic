# GPUDrivenSample 场景入口

`MetallicGPUDrivenSample` 聚焦两个正式流式场景，使用各自的完整实时渲染预设。

| 场景 | 启动参数 | 几何池 / CLAS 池 |
| --- | --- | --- |
| MiniZorah（默认） | 无参数或 `--minizorah` | 512 MiB / 256 MiB |
| ZorahFull | `--zorah-full` | 3.5 GiB / 2 GiB |

```powershell
.\build-release\Source\MetallicGPUDrivenSample.exe
.\build-release\Source\MetallicGPUDrivenSample.exe --zorah-full
.\build-release\Source\MetallicGPUDrivenSample.exe --list-scenes
```

在 File 菜单直接选择 MiniZorah / ZorahFull，也可在 Render Graph Editor 的 Scene 下拉框选择。File/Open、最近文件和拖入源文件仅接受仓库内的 `Asset/MiniZorah/zorah_main_public.v2.gltf` 与 `Asset/ZorahFull/zorah_textured_public.v1.gltf`，支持绝对路径和相对路径。切换会重载对应图、相机、缓存路径和预算；Reset Scene 恢复当前场景的预设。

两个场景均读取 metadata 和匹配的预构建 meshstream，不在启动时全量导入或自动 cook。Full 使用完整模型属性、512 mip 尾链与紧凑属性上传，准备和首帧证据见 [Z5](ZorahFullZ5FirstFrame.md)。池预算不等于程序总显存。

`--sample` 兼容两个 ID：`gpu-driven-sample`、`gpu-driven-zorah-full`。`--streamasset-path <file>` 仅覆盖所选启动场景的 cook，切换到另一个场景使用其自身的缓存。`--debug-control`、`--smoke-test` 与 `--wait-for-graphics-debugger` 保留。

在 GPUDriven pass 的 Runtime Settings 中启用 **Async Meshlet Feedback Readback**，或在图的 pass 属性中设置 `"asyncFeedbackReadback": true`。默认关闭。Vulkan 提供独立 Copy Queue 时，几何流式反馈会在生产者提交后由 Copy Queue 回读，CPU 在后续帧读取已完成的反馈。回读使用独立的完成信号，下一次写入同一反馈缓冲前由 GPU 等待。没有独立 Copy Queue 时，回读沿用 Graphics Queue。

Nsight Graphics 2026.3.1 在 MiniZorah 开始捕获时可能崩溃于
`ngfx-capture-interception.dll + 0xec137`。单独 reset 后释放仍可能崩溃。
捕获注入期间，RHI 现会 reset 并复用原生命令缓冲和命令池，延迟到设备
关闭时销毁，以处理捕获器的失效命令缓冲引用；保留 DLSS、
Streamline 注释和异步计算。复现、回放验证和回归命令见
[捕获崩溃调查](NsightCommandBufferCaptureInvestigation.md)。

任意 `--scene` 覆盖、旧诊断 ID、`--streamasset` 和 `--minizorah-vbuffer` 不再作为本程序入口。其他诊断样例与自定义图仍在通用 `Metallic` 编辑器中使用；共享样例目录和渲染 Pass 保留。

## 地形样例

在仓库根目录使用已有的编辑器构建。地形烘焙所需的 Python 依赖见 [requirements-terrain.txt](../Tools/requirements-terrain.txt)。

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainP0Smoke MetallicTerrainP1Smoke
cmake --build build-relwithdebinfo --target MetallicTerrainVSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainVSMSmoke
```

烘焙后，在通用 `Metallic.exe` 的内置样例列表选择 **GPUDrivenSample / Terrain OpenPBR + VSM**。样例将 Houdini 高度场、测量塔和桥烘焙进同一份 StreamAsset。材质走 OpenPBR 延迟着色，场景里的 Sun 通过 VSM 投影。默认机位同时观察测量塔投到地表的影子，以及山脊投到桥面的影子。

桥面原先有 5 个漏光点。朝向太阳的射线只命中地形背面，随后从西侧裙边下方穿出，单面材质的背面在 VSM 光栅中被剔除。这个样例的高度场材质现在启用双面。固定原机位和探针复验，塔投到地表 27/27，山脊投到桥面 30/30，直射对照 30/30，原漏光点的阴影值全部从 255 降到 0。可见性和深度缓冲与修复前逐字节一致，静态缓存与强制重绘的阴影和线性颜色也逐字节一致。

### 2×2 地形块

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainBlocksVSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainBlocksVSMSmoke
```

在通用编辑器选择 **GPUDrivenSample / Terrain Blocks + VSM**。同一高度场划成四个独立网格，共用原样例的相机、道具和渲染图。共享顶点直接从全局网格取位置、法线、切线和 UV。裙边沿地形外圈生成。分块 primitive 通过 `METALLIC_lockMeshletLODBoundary` 在 meshstream 简化中锁定开放边端点，使相邻地形块可以独立选择 LOD。

烘焙验证逐段比较了四条共享边的 LOD0、保底 LOD 及其混合配对。16 种配对和 124 次 DAG 细化替换均保留了接缝的解码位置及法线。原机位和两个掠射机位的 GPU 检查确认了相邻块的实际混合 LOD。接缝附近的两个背景像素位于山脊外轮廓，同机位的单块地形也复现了这两个像素。

32 页上限下，编辑器在原机位渲染了 60 帧并导出画面。同一上限的严格 RHI 验收触发了流式断言，在接缝捕获前退出。三机位接缝结论来自默认驻留预算。
