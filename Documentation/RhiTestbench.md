# Metallic RHI Testbench 使用说明

M1 已实现：在原 `MetallicRhiTests` 中增加声明式 requirements、三个设备配置、逐用例进程隔离、验证消息审查、完整产物检查和重放。旧 `--rhi-*` / GoogleTest 入口保持可用。设计背景见 [实现方案](RhiTestbenchPlan.md)。

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
| `MetallicTestbench.Contract` | 无 SDL/Device 初始化的 BufferRange 契约 |
| `MetallicTestbench.Core` | buffer 偏移复制/保留区域读回、提交完成、timestamp/host reset |
| `MetallicTestbench.Async` | 两种提交模式的独立 copy 进度、跨队列依赖与 query ring |

Core、Async 与当前目录中已有 rhi/editor CTest 使用 `MetallicGpu` 锁。外部手工启动的编辑器和其他项目不受 CTest 锁控制。Contract/Core 要求所有已选择样板通过；Async 默认允许硬件拓扑不支持时 skip，参考 GPU 可用 `--tb-require-all` 禁止 skip。

## CLI

```powershell
$test = '.\build-pass-stages-nrd\tests\MetallicRhiTests.exe'
& $test --tb-help
& $test --tb-plan --tb-suite core
& $test --tb-self-test
& $test --tb-run --tb-suite contract --tb-require-all
& $test --tb-run --tb-suite core --tb-require-all
& $test --tb-run --tb-suite async --tb-require-all --tb-repeat 5
& $test --tb-run --tb-suite core --tb-profile binding --tb-filter '*timestamp*'
```

`--tb-plan` 不初始化 SDL、TaskSystem 或 GPU；它枚举已迁移 case，落盘配置计划、shader 输入指纹和构建身份。三个 profile 是 `core`（关闭可关闭的 RT/DGC/SDK、optimal layouts）、`binding`（加 bindless）、`async`（再请求独立 compute）。M1 每个 case/iteration 都使用新子进程；尚未优化为同 profile 的多用例设备复用。

支持 `--tb-filter` 或 `--gtest_filter` 的 `*`、`?`、`:`、排除表达式；`--tb-repeat` 或 `--gtest_repeat` 范围 1–1000；`--tb-seed` 是 uint64。新模式拒绝与 `--rhi-*` 混用，以免配置不明确。未迁移用例只在旧入口运行，新模式匹配不到用例会失败。

默认输出位于工作目录 `.tmp/testbench/<run>/`，`--output-dir` 可指定其他空目录。已有产物不会覆盖，每个 iteration 单独保存。保持从仓库根目录运行。

## 验证层环境

M1 支持 `--tb-validation core|off`；GPU 样板要求 core validation，选择 off 会得到 EnvironmentFailure，不会把无验证运行算成 conformance 通过。读取 Vulkan 后端实际配置的 layer/messenger 状态，并记录发现的层版本。Synchronization/GPU-assisted 模式留待 M2。

有失效的系统 layer 注册时，默认运行会如实记录 loader error 并失败。可显式指定安装好的 layer 目录：

```powershell
& $test --tb-run --tb-suite core --tb-require-all `
    --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin

# 可选：只配置当前构建树的 CTest，保留其其他编译选项。
cmake -S . -B build-pass-stages-nrd `
    -DMETALLIC_TESTBENCH_LAYER_PATH=C:/VulkanSDK/1.4.350.0/Bin
```

版本路径是本机示例，须替换成实际安装目录。该选项只在测试子进程中设置 `VK_LAYER_PATH`，并将 `VK_IMPLICIT_LAYER_PATH` 指向本用例的空目录，隔离 overlay 等隐式层；不会修改注册表或用户/系统环境变量。路径保存在 input，可重放。其含义依据 [Vulkan Loader layer discovery](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md#layer-discovery)。权限提升导致 loader 忽略这些路径时，错误仍会报告，不静默放宽检查。

M1 拒绝会覆盖验证行为的外部 `VK_LOADER_LAYERS_ENABLE/DISABLE`、`VK_INSTANCE_LAYERS`、`VK_LAYER_ENABLES/DISABLES`、`VK_LAYER_SETTINGS_PATH`。使用普通 conformance 环境；Nsight、SDK 和自定义 validation settings 不属于 M1 作业。不存在通用忽略 VUID/loader error 的开关。

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
    coverage.json               # 当前选择的 M1 coverage observations，逐 iteration
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

覆盖率只报告已选择且已迁移的格子，不代表全部 RHI；多轮通过不会被描述为新增独立覆盖。M1 尚未提供跨设备 differential 调度、格式 requirements、完整物理能力 probe、后端 trace 或 fuzz/shrinker。

## 添加 M1 样板

现有 `RhiTest` 默认无 metadata，仍走旧 runner。要迁移单个用例：

1. 覆盖 `metadata()`，声明 suite、profile、责任层、requirements、timeout、coverage 和必须生成的 oracle 文件。
2. 保留原有 name、type 和 `run(RhiTestContext&)`。通过 `context.evidence` 写原始结果；旧模式中它为空。
3. Device 由 harness 提供，不在 run() 内另建未接 sink 的 Device。测试自己等待所提交工作的完成并安全释放本地对象；harness 的末尾 drain 不能修补测试内部提前释放资源。
4. CPU-only 用例声明 `requiresDevice=false`、validation Off、空 queues，并实现 `runCpu(Evidence&)`；cleanup 用 `cleanupCpu()`。
5. 用 `--tb-plan` 检查 requirements，再运行样板、对应旧过滤测试与 harness 自测。

自测的 `harness-fixtures` suite 是显式负例集合：timeout、crash、cleanup exception、首错保留、unexpected skip、真实 shader 编译失败和正常通过。它默认不运行，也不向旧注册表添加故意失败的测试；自测会断言这些失败被正确归类。
