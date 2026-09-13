# MiniZorah 编辑器漫游相机修复

2026-09-13。

## 原因与修复

编辑器输入更新共享 `RenderView`，但旧执行器将 `sceneBinding: asset` 无条件视为局部相机。MiniZorah VBuffer 因此一直使用节点中的固定相机；编辑器同时跳过该节点的初始相机导入，Inspector 显示默认 eye/FOV，而画面仍显示原始场景视点。

场景所有权与相机所有权现在可以分别指定：保留 `sceneBinding: asset` 的独立加载，通过显式 `viewBinding: global` 跟随视口。未声明 viewBinding 的旧 asset 图仍使用局部相机，场景输入消费者继承 producer 的局部约束。相同规则用于编译上下文、逐帧执行和编辑器相机控件。

MiniZorah 独立 StreamAsset 与统一 VBuffer 两个图都提供顶层 `view.camera`，初始化为资产原始视点。运行时平移、旋转及投影修改通过共享视图送入渲染，无需重编图或修改节点相机属性。

## 回归方式

此前 M4 自动路线直接修改节点 `camera`，验证了流式渲染与 cut，但没有验证编辑器的共享相机输入链路。现在完整场景首帧/视点测试和自动漫游均绑定外部 `RenderView`；漫游每 5 秒从 GPU 读回 `rasterInfo`，逐轴检查渲染使用的 eye/center 与输入一致，并断言没有通过节点 runtime camera 绕过共享视图。原有覆盖、DAG、预算检查继续执行。

通用回归增加独立 asset 使用 global view、下游消费者继承、移动不重编译，以及无场景依赖 pass 的 GPU 共享视图对照。保留旧 asset 的局部相机行为测试。

`Metallic`、`MetallicGPUDrivenSample` 和 `MetallicRhiTests` 的 RelWithDebInfo 构建成功。启用 Vulkan 验证和异步计算的 8 项回归全部通过（184.643 秒）：共享 GPU 视图、场景绑定契约、异步场景切换、Bunny 的两个流式入口、MiniZorah 的两个完整场景入口及持续漫游。日志中没有 VUID 或 Vulkan validation error。

共享视口的 60.008 秒、1 GiB、1920×1080 漫游完成 2,756 个计时帧；12 个检查点的渲染相机、cut、覆盖和页池检查全部通过。完整场景两个入口均通过原始/远景/近景检查；统一 VBuffer 另外通过 resize、释放和重载。此处验证相机链路；用户的旧编辑器仍同时运行，本轮不作为独占 GPU 性能对照。

[测试日志](../build-relwithdebinfo/minizorah-camera-fix/tests.log)、[漫游原始报告](../build-relwithdebinfo/minizorah-camera-fix/results/MiniZorahRoamingReport.json) 和图像保存在 `build-relwithdebinfo/minizorah-camera-fix/`。第 35 秒转向右侧建筑的输出已人工核对：

![共享视口转向结果](../build-relwithdebinfo/minizorah-camera-fix/results/roam-35.png)

## 使用修复后的程序

已运行的进程仍使用旧代码，需要重启。仓库根目录的启动命令：

```powershell
& .\build-relwithdebinfo\Source\MetallicGPUDrivenSample.exe --minizorah-vbuffer
```

在视口中按住右键拖动改变朝向；按住右键使用 W/A/S/D 移动。也可以修改 Inspector 的 Eye / Center。
