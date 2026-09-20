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

任意 `--scene` 覆盖、旧诊断 ID、`--streamasset` 和 `--minizorah-vbuffer` 不再作为本程序入口。其他诊断样例与自定义图仍在通用 `Metallic` 编辑器中使用；共享样例目录和渲染 Pass 保留。
