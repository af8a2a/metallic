# Metallic RHI Testbench 使用说明

M1/M2 已实现：在原 `MetallicRhiTests` 中增加声明式 requirements、三个设备配置、逐用例进程隔离、验证消息审查、完整产物检查和重放；M2 增加基础 RHI/Core/RenderGraph 覆盖、共享 fixture 和显式同步验证。旧 `--rhi-*` / GoogleTest 入口保持可用。设计背景见 [实现方案](RhiTestbenchPlan.md)。

## 构建和首次运行

复用已配置的 Windows x64 MSVC 构建树，先构建测试，再运行 CTest：

```powershell
cmake --build build-pass-stages-nrd --target MetallicRhiTests Metallic --parallel 8
ctest --test-dir build-pass-stages-nrd -R '^MetallicTestbench\.' --output-on-failure
```

实际 build 目录可替换为其他启用 `METALLIC_BUILD_TESTS` 的兼容配置。M1 的进程隔离暂限 Windows；其他平台旧测试入口不受影响。

| CTest | 内容 |
| --- | --- |
| `MetallicTestbench.Harness` | 无 GPU 的 recorder、requirements、协议及真实异常子进程自测 |
| `MetallicTestbench.Contract` | 无 SDL/Device 初始化的 BufferRange 和 RenderGraph 访问计划契约 |
| `MetallicTestbench.Core` | 资源/Result 契约，buffer/纹理复制，图形状态、depth、history、timestamp |
| `MetallicTestbench.Binding` | bindless、参数生命周期、BufferSlice、ComputeKernel 与 prepared dispatch |
| `MetallicTestbench.Sync` | 显式同步验证：Core barrier、RenderGraph、Joined/Pipelined、query ring、copy timestamp |
| `MetallicTestbench.SyncActivation` | 真实层的 core/sync 对照负例：只录制图像 WAW，不提交 |
| `MetallicTestbench.Async` | 两种提交模式的独立 copy 进度、跨队列依赖与 query ring |

所有 GPU testbench 作业与当前目录中已有 rhi/editor CTest 使用 `MetallicGpu` 锁。外部手工启动的编辑器和其他项目不受 CTest 锁控制。Contract/Core 要求所有选择用例通过；Binding/Sync/Async 允许缺少声明能力或队列时显式 skip。参考 GPU 验收必须加 `--tb-require-all`，禁止将 skip 视为通过。Sync 自动包含 Async 的三个提交/query ring 回归。

## CLI

```powershell
$test = '.\build-pass-stages-nrd\tests\MetallicRhiTests.exe'
& $test --tb-help
& $test --tb-plan --tb-suite core
& $test --tb-self-test
& $test --tb-run --tb-suite contract --tb-require-all
& $test --tb-run --tb-suite core --tb-require-all
& $test --tb-run --tb-suite binding --tb-require-all
& $test --tb-run --tb-suite sync --tb-validation sync --tb-require-all --tb-repeat 2
& $test --tb-run --tb-suite async --tb-require-all --tb-repeat 5
& $test --tb-run --tb-suite core --tb-profile binding --tb-filter '*timestamp*'
```

`--tb-plan` 不初始化 SDL、TaskSystem 或 GPU；它枚举已迁移 case，落盘配置计划、shader 输入指纹和构建身份。三个 profile 是 `core`（关闭可关闭的 RT/DGC/SDK、optimal layouts）、`binding`（加 bindless）、`async`（再请求独立 compute）。每个 case/iteration 都使用新子进程；尚未优化为同 profile 的多用例设备复用。

支持 `--tb-filter` 或 `--gtest_filter` 的 `*`、`?`、`:`、排除表达式；`--tb-repeat` 或 `--gtest_repeat` 范围 1–1000；`--tb-seed` 是 uint64。新模式拒绝与 `--rhi-*` 混用，以免配置不明确。未迁移用例只在旧入口运行，新模式匹配不到用例会失败。

默认输出位于工作目录 `.tmp/testbench/<run>/`，`--output-dir` 可指定其他空目录。已有产物不会覆盖，每个 iteration 单独保存。保持从仓库根目录运行。

## 验证层环境

支持 `--tb-validation core|sync|off`。GPU 用例至少要求 core；sync suite 的新增同步用例要求 sync。低于要求的模式为 EnvironmentFailure。Vulkan 的 `DeviceDesc::enableSynchronizationValidation` 显式增加 `VK_EXT_layer_settings` 的 `validate_sync=true`，要求 validation layer、扩展和 messenger 均可用；不满足时设备创建失败。该选项默认关闭，普通编辑器行为不变，不能与 ShaderPrintf 配置合用。

`capabilities.json` 记录后端配置模式、messenger、层版本和每队列 timestampValidBits。SyncActivation 用同一 executable/profile/layerPath 分别运行 core 与 sync：core 必须干净通过，sync 必须由 recorder 记录恰好一个 `SYNC-HAZARD-WRITE-AFTER-WRITE` 并判 Fail。CMake 检查具体状态、执行标志、消息 ID 和阶段；设备失败/崩溃/超时不能满足负例。这个作业证明层确实执行了同步检查，不能只凭配置布尔值声称开启。设置依据 [Khronos SyncVal 文档](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/syncval_usage.md#enabling-synchronization-validation)。GPU-assisted 尚未实现。

有失效的系统 layer 注册时，默认运行会如实记录 loader error 并失败。可显式指定安装好的 layer 目录：

```powershell
& $test --tb-run --tb-suite core --tb-require-all `
    --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin

# 可选：只配置当前构建树的 CTest，保留其其他编译选项。
cmake -S . -B build-pass-stages-nrd `
    -DMETALLIC_TESTBENCH_LAYER_PATH=C:/VulkanSDK/1.4.350.0/Bin
```

版本路径是本机示例，须替换成实际安装目录。该选项只在测试子进程中设置 `VK_LAYER_PATH`，并将 `VK_IMPLICIT_LAYER_PATH` 指向本用例的空目录，隔离 overlay 等隐式层；不会修改注册表或用户/系统环境变量。路径保存在 input，可重放。其含义依据 [Vulkan Loader layer discovery](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md#layer-discovery)。权限提升导致 loader 忽略这些路径时，错误仍会报告，不静默放宽检查。

runner 拒绝会覆盖验证行为的外部 `VK_LOADER_LAYERS_ENABLE/DISABLE`、`VK_INSTANCE_LAYERS`、`VK_LAYER_ENABLES/DISABLES`、`VK_LAYER_SETTINGS_PATH`，以及 `VK_VALIDATION_*` / `VK_KHRONOS_VALIDATION_*`。使用普通 conformance 环境；Nsight、SDK 和自定义 validation settings 不属于 conformance 作业。不存在通用忽略 VUID/loader error 的开关。

## 判定与产物

requirements 在运行用例前检查。profile 没有请求必要能力时为 SkipNotEnabled；请求后有效设备缺必要能力/queue 时为 SkipUnsupported。物理支持未知时保留 Unknown，不从 enabled=false 推断硬件事实。设备创建失败、验证层不可用是 EnvironmentFailure。requirements 已满足后，run() 内的 skip 视为 Fail。

所有收到的 validation warning/error、非 VUID 的同步 hazard、recorder 溢出或记录异常均失败。对象、字符串和消息上下文在回调内深拷贝；创建、run、等待、cleanup、析构、设备销毁阶段都在 recorder 生存期内。用例间不存在共用 GPU Device，因此 timeout/crash 后下一用例从干净进程启动。

```text
<run>/
    plan.json
    run.json                    # 配置时源码 revision、运行时 dirty 摘要、binary hash、编译器
    shader-inputs.json          # Shaders、tests/rhi/shaders 和本地 slang.dll（存在时）的指纹
    source-state/stdout.log      # git status --porcelain；缺 git 时状态为 Unknown
    results.json
    coverage.json               # 当前选择的 coverage observations，逐 iteration
    gtest.xml                   # 基于实际子进程/GoogleTest verdict 的聚合 JUnit 报告
    <profile>/<case>/<iteration>/
        input.json
        profile.json
        capabilities.json       # 成功创建 Device 后生成；CPU case 标记 deviceCreated=false
        journal.jsonl
        validation.json
        stdout.log
        stderr.log
        result.json
        parent-result.json
        gtest.xml
        ...原始 readback / oracle / 查询数据
```

`result.json` 在 cleanup 和 Device 销毁后原子提交。父进程检查身份、schema、退出码、executed 标志、必需产物清单、字节数和 FNV-1a64 内容指纹；缺失、截断、矛盾结果均为 InfrastructureFailure。FNV 指纹用于检测意外变化，不是安全认证。GPU driver 卡住时由父进程 Job Object watchdog 限时终止并保留已有 journal/log。原始运行错误与后续 cleanup 错误同时保留。

构建 revision 是 CMake 配置时的值，run.json 的 dirty 摘要是运行时状态，不声称二者一定代表一个干净源码提交。精确可执行文件由 binary hash 标识。shader 清单覆盖仓库 shader 与测试 probe，不是完整外部 SDK 依赖锁文件。

## 重放

```powershell
& $test --tb-replay .tmp/testbench/<run>/core/RhiCommand.buffer_copy_offset_readback/0
```

恢复 case、profile、validation、seed 和 layerPath，并写入新的输出目录。可执行文件或 shader 输入指纹变化时拒绝；修复后用 `--tb-allow-version-mismatch` 明确允许新版本，原始输入保留在新 run.json 中供比较。不会使用原目录覆盖结果。

覆盖率只报告已选择且已迁移的格子，不代表全部 RHI；多轮通过不会被描述为新增独立覆盖。尚未提供通用格式 requirements、完整物理能力 probe、跨设备 differential 调度、后端 trace 或 fuzz/shrinker。基础纹理用例限定 RGBA8/R32Uint/D32 等既有格式，不声称覆盖所有格式组合。

## M2 覆盖与边界

| 责任层 / 类别 | GPU 读回 | 契约 / 调度检查 |
| --- | --- | --- |
| RHI Resource / Copy | buffer 偏移、哨兵；13×9 纹理全部 mip/layer；3D 体积；带 padding 和局部覆盖 | Result 移动所有权、零大小、溢出、range、view、错误码 |
| RHI Binding / Compute | constant/structured/raw/RW/atomic；sampled/storage image；sampler 非一致索引；最后有效 descriptor 槽与 heap 切换 | descriptor 容量/回收、错误范围、无效 shader 输入；typed packet ABI |
| RHI Graphics | triangle、reversed Z、材质 shader object；pipeline → shader object → pipeline 全像素 oracle | 缺失 shader stages；viewport/scissor 和未绘制区域保持 |
| RHI Query / Queue | graphics timestamp；copy 队列六轮 timestamp+copy；局部 host reset | 非法 reset 不改变结果；完成后复用；独立 copy 不被阻塞 graphics 队列拖住 |
| Core | 参数提交与并行读回、BDA/indirect、prepared kernel/batch；history 跨帧 | BufferSlice provenance、跨设备、wrapper 释放后的 retained allocations、ABI/stale packet、错误原子性 |
| RenderGraph | fanout 多消费者、图像 clear/copy；Joined/Pipelined 多槽、多队列和 history 读回 | RAW/WAR/WAW/read-read 计划、别名、stage 录制边界、失败取消/部分提交、外部 consumer、profiling 不引入额外前驱、query ring |

`Fixtures.h` 共用测试设备借用/所有权、提交等待和读回保存。可信入口借用 runner 的 Device；旧入口仍按原配置创建 Device。跨设备契约所需的第二个 Device 使用相同配置与 recorder。mapped/native 双路径测试声明 native descriptor pointer 前置条件。GPU 数据保存为必需产物；内部多轮读回带编号保留；既有解析式预期和断言仍在测试代码中，新 probe 另存 expected/actual/diff。

测试分清证据类型：访问计划纯 CPU 检查和 barrier 编码计数不算 GPU 数据读回。同步验证也不覆盖所有访问：本机 1.4.350 层未报告初始 `vkCmdCopyMemoryKHR` 地址复制 WAW 探针，因此激活探针采用图像 clear WAW；地址复制、BDA、bindless 的正确性仍依赖数据 oracle 与计划/编码断言。不能把“零同步消息”解释为这些路径已被层完整检查。当前 API 没有独立的 blend state 配置，因此没有宣称任意混合状态覆盖。扩展 AS/OMM/PTLAS/DGC 对比留在 M3。

## 添加用例

现有 `RhiTest` 默认无 metadata，仍走旧 runner。要迁移单个用例：

1. 覆盖 `metadata()`，声明 suite、profile、责任层、requirements、timeout、coverage 和必须生成的 oracle 文件。
2. 保留原有 name、type 和 `run(RhiTestContext&)`。通过 `context.evidence` 写原始结果；旧模式中它为空。
3. Device 由 harness 提供；迁移旧用例使用 `bench::createTestDevice`，必要的第二个 Device 明确指定 `additionalDevice=true`。不要在 run() 内另建未接 sink 的 Device。测试自己等待所提交工作的完成并安全释放本地对象；harness 的末尾 drain 不能修补测试内部提前释放资源。
4. CPU-only 用例声明 `requiresDevice=false`、validation Off、空 queues，并实现 `runCpu(Evidence&)`；cleanup 用 `cleanupCpu()`。
5. 用 `--tb-plan` 检查 requirements，再运行样板、对应旧过滤测试与 harness 自测。

自测的 `harness-fixtures` suite 是显式负例集合：timeout、crash、cleanup exception、首错保留、unexpected skip、真实 shader 编译失败和正常通过。它默认不运行，也不向旧注册表添加故意失败的测试；自测会断言这些失败被正确归类。
