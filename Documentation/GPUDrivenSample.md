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

流式预算诊断可在 VBuffer 上设置 `maxResidentPages`，同时在 RHI 验收用例中指定 `"allowResidentBudgetDeferral": true`。用例会核对实际页上限和已占用页数 `activePages`，并要求观察到因预算耗尽且没有可淘汰页而产生的 `allocationBudgetDeferrals`。这些退避仍计入现有的 `allocationFailures` 汇总。其余分配失败、加载失败和请求溢出仍会中止验收，报告保留触发断言的帧。默认用例仍按原来的严格条件检查。

32 页上限的旧用例在第 1 帧触发断言。当时 32 个页槽全部占用，5 个保底页已经驻留，几何池使用约 2.93 MiB（上限 64 MiB）。唯一一次分配失败计数对应预算退避。启用预算诊断后，原三个机位的 60 帧验收通过。全程保底页 5/5 就绪，活跃页数峰值 32，记录到 59 次预算退避、38 次上传和 11 次淘汰，加载失败、其余分配失败和请求溢出均为 0。

紧预算下，在全景、西南、东北机位分别于第 19、39、59 帧采到 1,208、60、360 对跨块且 LOD 不同的相邻像素。结合首帧和两次切机后的首帧，检查覆盖了四条共享边，参考轮廓共同内域无漏像素。三个机位还分别以 32 页上限冷启动。首帧仅有 5 个保底页驻留，与解码后的保底网格 CPU 光栅结果逐像素对照，三个机位的内域覆盖均无缺失。

### 4×4 地形块和跨块飞行

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks4x4VSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks4x4VSMSmoke
```

在通用编辑器选择 **GPUDrivenSample / Terrain Blocks 4x4 + VSM**。样例在原高度场的 256×256 世界范围内细分到 16 块，沿用 OpenPBR、VSM、桥、塔和三个回归机位。预构建资产包含 160 页，其中保底页 17 个，占 315,648 字节。为最大细节页留出空间后，最低几何池容量为 468,480 字节。烘焙检查逐边对比了解码后的位置和法线，24 条共享边的 96 种 LOD 配对和 143 次 DAG 细化替换全部通过。

验收脚本生成三机位回归、三次独立冷启动和两档预算的连续飞行。飞行持续 105 帧，含终点停留的 20 帧，跨过 X、Z 方向的六条分块边界。两档预算用相同相机、640×360 分辨率和 LOD 设置。每次过界均保留前、中、后三帧，整个路线共采集 36 帧。`streamingEvidenceFrames` 对逐帧 GPU 请求、页表状态以及采集帧实际写入可见性缓冲的页和 LOD 做关联。

```powershell
python -B Tools/Perf/PrepareTerrainStreamingAcceptance.py
$terrainRun = 'build/terrain-vsm/scale/motion-tight'
$env:METALLIC_VSM_ACCEPTANCE_CASES = Join-Path $terrainRun 'Cases.json'
.\build-ci\tests\MetallicRHITests.exe --filter virtual_shadow_map_acceptance --rhi-bindless --rhi-validation --output-dir "$terrainRun/capture"
```

每个进程运行一个用例。将命令中的 `motion-tight` 换为 `motion-ample`、`fixed-tight`、`fixed-ample`、`cold-overview`、`cold-southwest` 或 `cold-northeast` 可执行对应验收。

| 飞行预算 | 活跃页峰值 | 上传 | 淘汰 | 预算退避 | 实际混合 LOD 的共享边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 页 | 32 | 40 | 25 | 104 | 20 / 24 |
| 256 页 | 160 | 143 | 0 | 0 | 24 / 24 |

两档预算均保持 17 个保底页全程就绪。32 页飞行中，首次跨界后有三个细节页记录了从请求到可绘制、再到可见像素的完整链路。256 页飞行在第 4 帧已发布全部 160 页。三个固定机位的冷启动均覆盖解码保底网格的内域。两档飞行的可见性与深度采样覆盖了全部 24 条共享边，参考轮廓共同内域没有漏像素。

#### 32 页定向接缝路线

将复验命令中的用例改为 `motion-directed-tight`，运行 282 帧并采集 33 帧。开场沿用原飞行的眼位，以 24° 视场角朝向南侧地形块 `(0,3)`，使它先获得细节页。镜头随后转向 `(0,2) / (0,3)` 共享边，在第 12 帧恢复 60° 视场角。第 12、13、14 帧分别观察过界前、中、后，第 13 帧检查同一条边两侧实际绘制的 LOD。后续镜头围绕这条边及 `(1,2) / (1,3)`、`(1,3) / (2,3)` 和 `(0,0) / (1,0)` 移动。用例目录的 `Targets.json` 记录首次过界、混合 LOD 验收帧和各观察段。原三个固定机位仍通过 `fixed-tight` 和 `fixed-ample` 运行。

定向路线在相机移动且 32 个页槽均占用的帧中补齐了四条边的混合 LOD 证据。合并两条 32 页飞行的捕获，共 24 条共享边实际出现跨块混合 LOD。第一次过界的第 12、13、14 帧分别测得 280、269、281 对跨边相邻像素，三帧均为北侧 LOD5、南侧 LOD1。定向路线 282 帧中保底页全程 17/17 就绪，采样的参考轮廓共同内域无漏像素。

#### 自动回归

安装 `Tools/requirements-terrain.txt` 中的依赖后，可在启用了测试的构建目录执行专用目标。它构建验收程序和 4×4 烘焙资产，然后运行 Vulkan 验证。

```powershell
cmake --build build-ci --target MetallicTerrainStreamingRegression
```

已有验收程序和烘焙资产时，可直接用 CTest 运行同一套回归。

```powershell
ctest --test-dir build-ci -R '^MetallicTerrainStreamingRegression$' --output-on-failure
```

每次运行生成独立采集目录，验收开场 15 帧和原三个固定机位的 60 帧。门槛要求流式上限为 32 页，第 13 帧占满 32 个页槽、眼位实际跨过共享边，并在深度和可见性确定的两侧相邻像素上测到不同 LOD。固定机位检查各自的相机参数，六次采集各自至少有 10% 的画面被地表覆盖，且地表上同时有受光和阴影样本。结果写入构建目录的 `tests/terrain-streaming-regression/run-*/TerrainStreamingRegression.json`。

### 8×8 地形块

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks8x8VSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks8x8VSMSmoke
python -B Tools/Perf/PrepareTerrainStreamingAcceptance.py --tiles-per-axis 8
```

在通用编辑器选择 **GPUDrivenSample / Terrain Blocks 8x8 + VSM**。样例在原高度场和 256×256 世界范围内细分到 64 块，烘焙为 342 页，含 65 个保底页。保底页使用 612,096 字节，与一个最大细节页合计 731,392 字节。112 条共享边的 448 种 LOD 配对和 277 次 DAG 细化替换均通过解码检查。

编辑器样例使用共享图的 256 页上限。8×8 验收清单写入 `build/terrain-vsm/scale-8x8/`，沿用三个固定机位和连续飞行路径，分别用 80 页和 512 页验收。80 页给保底 cut 留出完整覆盖，同时保留 15 个细节页槽。飞行在 14 个不同帧跨过 14 条网格分割线，每档预算采集 54 帧。

两档飞行均通过 105 帧验收，65 个保底页全程就绪，加载失败、请求溢出和 Vulkan 验证错误均为 0。80 页下有 59 次上传和 44 次淘汰，14 个过界帧都能看到保底页绘出的地表。512 页下第 6 帧全部 342 页可绘制，全程没有淘汰。80 页的原三机位 60 帧验收也通过。

80 页飞行与三个固定机位合计观察到全部 112 条共享边，其中 79 条实测混合 LOD。加入 512 页飞行后，混合 LOD 证据为 105 / 112。两档飞行的 14 个过界窗口中有 13 个出现跨边相邻像素，第 19 帧对应的共享边在整个过界窗口内处于视锥外。第 59 帧的四个轮廓候选与实际绘制页的解码重光栅结果一致，采样中无未解释的缺口。

#### 80 页定向接缝路线

8×8 清单生成器还会生成 `motion-directed-8x8-tight`。开场沿用原飞行眼位，先以窄视角取景东侧，再在第 18 帧恢复 60° 视场角并瞄准 `(1,5) / (2,5)` 共享边。第 18、19、20 帧分别采集过界前、中、后的画面。后续观察站先取景一个地形块，再展宽视野捕获它与相邻块的边界，部分边另有斜视观察。`Targets.json` 记录过界帧、待补边及采样帧。

首轮完整路线 1,116 帧、71 次采集通过 Vulkan 验收，65 个保底页全程就绪。33 条待补边均在 80 个页槽满额时测到混合 LOD，合并原 80 页用例后共有 112 / 112 条共享边得到满额时的混合证据。第 18、19、20 帧分别测到 152、121、147 对跨边像素，均为西侧 LOD2、东侧 LOD3。第 19 帧沿原眼位实际跨过该边。

共同内域核验无漏像素。第 762 帧的 12 个外轮廓候选经当帧可见页解码重光栅化后确认，整帧 230,400 个覆盖样本与 GPU 一致。

#### 8×8 自动回归

```powershell
cmake --build build-ci --target MetallicTerrain8x8StreamingRegression
ctest --test-dir build-ci -R '^MetallicTerrain8x8StreamingRegression$' --output-on-failure
```

CMake 目标会准备样例资产、构建 RHI 测试并运行回归。资产和测试程序就绪后也可直接用 CTest 执行。每次运行重新采集原飞行、定向路线和原三个固定机位。三组采集都检查 80 页上限和 65 个保底页的就绪状态。112 条共享边各自需要一组页槽满额时的混合 LOD 相邻像素。第 18、19、20 帧都需要看到 `(1,5) / (2,5)` 共享边，第 19 帧还要确认实际过界且两侧 LOD 不同。定向路线为 `(1,3) / (1,4)` 增加两个斜视机位，利用稳定驻留的 LOD2 和 LOD3 页取证。固定机位分别检查原相机参数、地表覆盖和 VSM 的受光与阴影样本。报告写入 `tests/terrain-streaming-regression-8x8/run-*/TerrainStreamingRegression.json`，4×4 仍由 `MetallicTerrainStreamingRegression` 独立验收。

### 16×16 地形块

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks16x16VSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainBlocks16x16VSMSmoke
python -B Tools/Perf/PrepareTerrainStreamingAcceptance.py --tiles-per-axis 16
```

在通用编辑器选择 **GPUDrivenSample / Terrain Blocks 16x16 + VSM**。原高度场和 256×256 世界范围细分为 256 块，烘焙资产有 1,025 页，含 257 个保底页。16×16 样例将几何页上限设为 512。

验收清单写入 `build/terrain-vsm/scale-16x16/`。272 页紧预算给保底 cut 留出完整覆盖并保留 15 个细节页槽，2,048 页用于充足预算对照。两档都沿 133 帧的飞行路线验收，在 30 个不同帧跨过全部网格分割线，采集每次过界的前、中、后帧。固定机位沿用原 overview、southwest 和 northeast 相机。

两档飞行的 30 个过界窗口均观察到跨边相邻像素。272 页下，257 个保底页全程就绪，30 个实际过界帧均绘出了保底地表，并记录到 108 次细节页上传和 93 次淘汰。2,048 页下，第 13 帧全部 1,025 页可绘制，全程无淘汰。两档飞行和 272 页的三个固定机位合计观察到 451 条共享边，其中 219 条出现混合 LOD，接缝共同内域无漏画。加载失败、请求溢出、VSM 工作溢出和 Vulkan 验证错误均为 0。本轮报告位于 `build/terrain-vsm/scale-16x16/Terrain16x16RuntimeAcceptance.json`。

### Houdini 岩石和草地权重

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainLayers16x16VSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainLayers16x16VSMSmoke
cmake --build build-ci --target MetallicTerrainLayerRegression
```

在编辑器选择 **GPUDrivenSample / Terrain Rock and Grass + VSM**。样例使用原 `Asset/simple_terrain_height.exr` 高度图和 `Asset/Terrain/TerrainRockWeight.exr`。权重图的线性 R 通道为岩石比例，草地比例为 `1 - R`。导出自己的权重时沿用高度图的数据窗口和行方向，像素值保持在 `[0, 1]`。`BuildTerrainVSMSample.py --weight-map <EXR>` 将它绑定到地形材质，随高度图应用 `flipZ`，并镜像扩展边缘供 mip 采样。

样例的权重图可用 Houdini 随附的 `hoiiotool` 重建。

```powershell
python -B Tools/ExportTerrainWeight.py Asset/simple_terrain_height.exr `
  --output Asset/Terrain/TerrainRockWeight.exr --oiiotool '<Houdini>/bin/hoiiotool.exe'
```

分块 glTF 的全局 `TEXCOORD_0` 对应高度场的世界 XZ 坐标。材质资源变换将它换算为草地每 2 世界单位、岩石每 6 世界单位的重复坐标。两层底色按线性颜色混合，法线解码后在共同的切线空间混合并归一化，线性粗糙度存于各层法线贴图的 alpha。三个结果进入同一套 OpenPBR 着色。VSM 样例仍与普通网格共用 Visibility Buffer 和光照图。

双层自动回归在 `(11,3)` 和 `(11,4)` 的共享边上检查同一条岩石带。近景、远景和 272 页飞行的实际混合 LOD 帧均采集底色、法线、粗糙度和最终着色。CMake 目标将每次运行的结果写入 `build-ci/tests/terrain-layer-regression/` 下独立的目录。直接执行 `TerrainLayerAcceptance.py run` 时，默认仍使用双层样例。

第 84 帧在 272 页满额时采到 87 对材质带内的 LOD 2/3 跨边像素。近景、远景和混合 LOD 的底色与粗糙度均通过世界坐标参考比较，两层法线均产生了可测的映射结果。另对烘焙页解码，480 条共享边和 768 次 LOD 细化替换的世界位置、法线、UV0 和切线均与源数据一致。

### Houdini RGBA 地形层资产

将 Houdini 的层遮罩导出为单张线性 OpenEXR，声明使用的 `R`、`G`、`B`、`A` 通道。每个通道是独立的 HALF 或 FLOAT 权重，取值为 `[0, 1]`。权重图的 `dataWindow`（含原点）和行方向与源高度图一致，导入时从烘焙地形读取 `flipZ` 并应用一次。

在 EXR 旁保存版本为 `1` 的 `Metallic.TerrainLayers` JSON。每层用一个稳定的名字和独占的 RGBA 通道，层名在忽略大小写时也须唯一。调整层列表顺序时沿用已有的名字和通道，`defaultLayer` 填写某个已声明的完整层名。标量 `worldTileSize` 同时用于 X/Z，`[x, z]` 分别指定两个方向的世界平铺周期。输入贴图路径相对于这份 JSON。

```json
{
  "type": "Metallic.TerrainLayers",
  "version": 1,
  "weightMap": "terrain_weights.exr",
  "defaultLayer": "grass",
  "layers": [
    {"name": "rock", "channel": "R", "worldTileSize": 6, "baseColor": "rock_color.png", "normal": "rock_normal.png", "roughness": "rock_roughness.png"},
    {"name": "grass", "channel": "G", "worldTileSize": 2, "baseColor": "grass_color.png", "normal": "grass_normal.png", "roughness": "grass_roughness.png"},
    {"name": "gravel", "channel": "B", "worldTileSize": [3, 5], "baseColor": "gravel_color.png", "normal": "gravel_normal.png", "roughness": "gravel_roughness.png"},
    {"name": "snow", "channel": "A", "worldTileSize": 8, "baseColor": "snow_color.png", "normal": "snow_normal.png", "roughness": "snow_roughness.png"}
  ]
}
```

```powershell
python -B Tools/ImportTerrainLayers.py Asset/MeshletCache/TerrainBlocks16x16/terrain_blocks_16x16_height.gltf `
  path/to/terrain.layers.json --output build/terrain-layers/terrain.imported.json
python -B tests/tools/TestTerrainLayerAsset.py -v
```

导入会在每个像素归一化已声明的权重，未声明的通道置零，已声明的权重全为零时令默认层占满。输出清单和镜像边缘的线性 RGBA8 权重图。四个字节逐像素合计为 255，A 通道仍是权重。清单记录全局权重 UV 变换和各层的世界平铺变换，标明底色按 sRGB 读取、法线与粗糙度按线性读取。

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainMultilayer16x16VSMSample
cmake --build build-relwithdebinfo --target MetallicTerrainMultilayer16x16VSMSmoke
```

在编辑器选择 **GPUDrivenSample / Terrain Four Layers + VSM**。样例生成岩石、草地、砾石和雪四层，并在原材质带旁加入四层交叠区。为自己的地形生成材质时，将同一份 Houdini `Metallic.TerrainLayers` JSON 传给 `BuildTerrainVSMSample.py --layer-asset`，导入权重并在场景中绑定 `OpenPBR.TerrainLayers` 材质。旧的 `--weight-map` 入口继续使用 `R / (1 - R)` 权重。

`OpenPBR.TerrainLayers` 在材质实例的 `terrainLayers` 中显式绑定 `weights` 和 `defaultLayer`，并在每层声明 `name`、`channel`、`baseColor`、`normal` 和 `roughness`。贴图使用常规材质资源 URI 和 UV 变换。运行时在采样后对已声明的权重再次归一化，全零时选择默认层。底色按工作空间线性混合，切线空间法线解码后加权并归一化，粗糙度读取各层贴图的 R 并沿用双层样例的线性加权语义。

#### 独立重导入地形材质

样例首次构建完成后，可以编辑四层样例 `source/` 中的权重 EXR 或某层贴图，再执行材质目标。完整样例目标也会按同一套依赖执行增量更新。旧岩草样例使用 `MetallicTerrainLayers16x16VSMMaterials`。

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainMultilayer16x16VSMMaterials
python -B Tools/BuildTerrainVSMMaterial.py path/to/sample.gltf --layer-asset path/to/terrain.layers.json
```

独立导入从已有样例 glTF 读取高度网格坐标，将层元数据写入 `.terrain-layers.json`。四层纹理文件按内容命名，素材有变化时生成对应的新纹理，再更新 `.material` 的绑定。重新载入场景即可读取新材质。老版本完整样例首次构建时会迁移到独立清单，并烘焙一次对应的几何基线。此后的权重和贴图编辑只触发材质导入。

独立重导入回归在专用运行目录内修改权重图和层贴图，用增量构建检查绑定更新以及几何产物的稳定性。回归需要 Ninja 和地形 Python 依赖，报告写入 `build-ci/tests/terrain-material-reimport/run-*/TerrainMaterialReimportRegression.json`。构建目标会先准备离线烘焙器。也可在烘焙器已构建后直接运行同名 CTest。

```powershell
cmake --build build-ci --target MetallicTerrainMaterialReimportRegression
ctest --test-dir build-ci -R '^MetallicTerrainMaterialReimportRegression$' --output-on-failure
```

四层自动回归沿用双层的机位和飞行路线，在 `(11,3)` 与 `(11,4)` 的共享边检查第 84 帧（272 页、混合 LOD）、第 94 帧（近景）和第 104 帧（远景）的底色、法线与粗糙度连续性。回归要求共享边两侧都有四层参与，还会根据抓帧的地表深度和相机重建纹理采样范围，比较滤波后的世界坐标参考和接缝残差。以下目标会构建样例与 `MetallicRHITests` 并运行回归，输出保存在 `build-ci/tests/terrain-multilayer-regression/`。各次运行独立保存抓帧、Vulkan 检查及 `TerrainLayerAcceptance.json`。

```powershell
cmake --build build-ci --target MetallicTerrainMultilayerRegression
```

如果已构建两个样例和 `MetallicRHITests`，可以由 CTest 依次执行双层和四层回归。

```powershell
ctest --test-dir build-ci -R '^MetallicTerrain(Layer|Multilayer)Regression$' --output-on-failure
```

### 地表拾取与摆放

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainPlacement16x16VSMSample
python -B tests/editor/RunTerrainPlacementSmoke.py --executable build-relwithdebinfo/Source/Metallic.exe --output build/terrain-placement/editor-smoke
```

在编辑器选择 **GPUDrivenSample / Terrain Placement + VSM**。在视口的 **Place on Terrain** 菜单选 **Rock** 或 **Vegetation**，然后左键点击地表。物件的 +Y 对齐源法线。按 Esc 返回选择模式，Ctrl+Z 和 Ctrl+Y 撤销及重做。选中已放置物件后，可在 **Terrain Anchor** 调整世界 X/Z，或用平移控件沿 X/Z 贴地移动。

场景文档保存摆放 ID、物件类别和世界 XZ。用 **Save Scene** 保存后，可在同一输出路径重新烘焙高度图并重新打开场景。摆放保留原 XZ，按新高度和法线更新姿态。如果新地形范围缩小，范围外的锚点保留在文档中，待地形重新覆盖该坐标时恢复。

`BuildTerrainMesh.py --height-field-output <path>` 从渲染网格使用的同一组缩放、偏移及翻转后的源样本导出 `.heightfield.bin`。`TerrainHeightField::sample` 提供世界 XZ 处的高度和插值法线，`raycast` 返回最近的世界交点、法线和沿归一化射线的距离。两者使用源网格三角形，独立于流式 LOD 和裙边。后续笔刷、散布或碰撞可直接使用此查询类型。

编辑器 smoke 用视口点击分别放置岩石和植被，再以同一路径重导入整体抬高 3 单位的测试地形。两个摆放 ID 和保存的 XZ 均保持一致，世界 Y 各增加 3，且 +Y 继续与地表法线对齐。重导入后的截图和运行日志保存在 `build/terrain-placement/editor-smoke/run-*/`。

### 独立区域流式加载

```powershell
cmake --build build-relwithdebinfo --target MetallicTerrainRegionsSample
cmake --build build-relwithdebinfo --target MetallicTerrainRegionsSmoke
cmake --build build-ci --target MetallicTerrainRegionStreamingRegression
```

编辑器样例 **GPUDrivenSample / Terrain Regions + VSM** 从西区起飞。样例将原 8×8 地形分为东西两个 4×8 区域。每区有独立的 glTF、顶点缓冲和 StreamAsset。聚合步骤将两份已烘焙的目录和页写入同一个寻址空间，保留原来的顶点属性、世界坐标和边界几何。生成文件位于 `Asset/MeshletCache/TerrainRegions/`。

相机与区域 XZ 包围盒的距离进入 44 个世界单位时开始锁定目标区的保底页，超过 56 时请求卸载。Visibility Buffer 和 VSM 共用已发布的区域集合。发布前等待所有目标保底页就绪，切换时等待读取旧实例数据的 GPU 命令完成，随后通过原有的页退休流程回收旧区独占分配。预算须容纳交接时两区的保底页。

视口漫游使用烘焙时导出的源高度场。在地形范围内，相机眼位至少高出地面 2 个世界单位。需要抬高时，眼位和观察目标一起移动以保留视线方向。`MetallicTerrainRegionsSmoke` 回放录屏中的西区斜坡机位，检查地表下方的候选眼位和越出地形范围的眼位。

分区样例的主视图以 0.05 显示像素为自动 LOD 误差上限，减小了主视图接收面与细化后 VSM 投影几何之间的偏差。地形材质为双面，保留掠射光下只命中开放高度场背面的投影。

分区样例的方向光 PCF 使用 8 毫米的阴影深度偏置，容纳接收面法线跨越相邻三角形时产生的深度误差。近景与远景机位都对照源地形的受光与遮挡采样。

回归在 80 页和 1 MiB 的几何页池里往返跨区。栅格和 VSM 都使用保底 LOD，以隔离细节页请求，并检查实际过界前、中、后的深度和可见性覆盖。编辑器样例使用自动 LOD、512 页上限和 64 MiB 页池，为交接时两区的细节页留出页槽。逐帧报告记录目标区、已发布区、GPU 页表锁定页、保底页占用和几何页池的已用字节。后两项计量的是页池内分配，页池预留容量由运行配置决定。报告位于构建目录的 `tests/terrain-region-streaming-regression/run-*/TerrainRegionStreamingAcceptance.json`。

首次通过的 204 帧往返中，双区分别在第 31 和 124 帧就绪并发布，先于第 59 和 154 帧的实际过界。旧区独占分配分别在第 81 和 175 帧完成回收。锁定页数随西区、双区、东区在 32、64、32 之间变化，对应几何页池已用字节为 306,944、611,072、304,128。回到西区后恢复为 306,944 字节。

### 跨区材质采样与成本

`MetallicTerrainRegionMaterialsSample` 为同一份区域 glTF 和 StreamAsset 生成双层与四层场景文档。每个变体的东西两区绑定同一个材质，并使用全局 XZ 映射和同一份线性 RGBA 权重图。两个变体都使用 OpenPBR 的显式地形层求值。双层保留 R/G 对应的岩石和草地，四层再启用 B/A 对应的碎石和雪。

```powershell
cmake --build build-ci --target MetallicTerrainRegionMaterialsSample MetallicRHITests
python -B Tools/Perf/TerrainRegionMaterialAcceptance.py --executable build-ci/tests/MetallicRHITests.exe
```

路线从西区起飞，在同一条材质带上停留于远、近两个机位，再横跨东西区边界。诊断读回底色、切线空间法线和粗糙度，并对照从 GPU 深度重建的世界坐标及纹理采样范围。过界前、中、后三帧均要求两区可见。额外远景扩大权重纹素覆盖，使用底色读回对照共享权重图的滤波结果和固定 mip 0。区域飞行沿用 80 页和 1 MiB 的保底几何设置。`--mode diagnostic` 可单独执行连续性验证。

普通计时在独立进程中以双层、四层交替运行，共用相机、640×360 内部尺寸和预热帧。报告使用固定机位上逐进程的 Deferred GPU 时间中位数，以及 VBuffer 报告的已驻留纹理分配字节。区域发布和释放的几何字节另有记录。`Continuity.json`、`Timing.json` 和原始读回写入 `build/terrain-region-material-validation/run-*/`。

### Ancient Rift Valley 中央 2×2

```powershell
cmake --build build-ci --target MetallicAncientRiftValley2x2Sample
cmake --build build-ci --target MetallicAncientRiftValley2x2Regression
cmake --build build-ci --target MetallicAncientRiftValley2x2MaterialReimportRegression
```

编辑器样例 **GPUDrivenSample / Ancient Rift Valley 2x2 + VSM** 使用原 1 km 高度场的中央 1025² 个米制样本。500 m × 500 m 的范围由四个独立烘焙的区域组成。几何法线使用相邻高度样本计算，权重和细节贴图沿用全局 1 km 的 XZ 坐标。样例启动时发布四区的保底几何，并按屏幕误差请求细节。

八层材质共用两张原尺寸的 2049² 线性 RGBA 权重图。每层按图索引和通道绑定，导入时联合量化，GPU 在两张图的相同源 mip 采样后联合归一化。新权重图采用边缘钳制，细节贴图按世界尺度平铺。此样例为 XZ 平面投影基线，DCC 预览中的陡壁三平面投影属于后续工作。

GPU 回归在四区交点检查深度覆盖和混合 LOD，并在冲积扇采集八层底色、切线法线和粗糙度供 CPU 参考比较。第二张权重图的增量回归在隔离的 Ninja 构建中分别调用完整目标和 `MetallicAncientRiftValley2x2Materials`，核对区域几何、StreamAsset 和场景文档的 SHA-256 及修改时间。报告和抓帧保存在 `build-ci/tests/ancient-rift-valley-2x2/` 和 `build-ci/tests/ancient-rift-valley-material-reimport/`。

### Ancient Rift Valley 全域 16 区

```powershell
cmake --build build-ci --target MetallicAncientRiftValley4x4Sample
cmake --build build-ci --target MetallicAncientRiftValley4x4Smoke
cmake --build build-ci --target MetallicAncientRiftValley4x4Regression
```

编辑器样例 **GPUDrivenSample / Ancient Rift Valley 4x4 + VSM** 使用完整的 2049² 源高度场，覆盖 1 km × 1 km。16 个区域独立烘焙，256 个渲染块共享全局边界采样和八层材质。全区保底用于启动与全景验证。普通漫游启用 `geometryRegionSelectionMode="viewAndDirectionalCasters"`，选区包括近眼滞回范围、与视锥相交的接收区，以及能沿太阳或月亮方向向这些接收区投影的候选区域。细节继续由现有页请求驱动。全景机位的远裁剪面为 3500 m，方向光 VSM 从 8 m 半宽开始，使用 9 级 clipmap。

GPU 回归固定在保底 LOD，在同一路线对照全区保底和按需选区，逐帧读取保底页与区域发布状态，在越过 X=250 m 的边界前、中、后对比深度、覆盖和方向光阴影。谷底探针另与纯距离选区对照，用于验证视锥外山脊的投影。分别在原切面、跨界后和全景就绪后对比 VSM 缓存与强制重画。俯视镜头跳转时，刚进入视野的区域要等保底页发布后才出现。回归记录首帧缺失像素和全区完成发布的帧号，并检查就绪后的完整覆盖。结果位于 `build-ci/tests/ancient-rift-valley-4x4/`，中央 2×2 回归仍有独立入口。
