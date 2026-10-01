# Metallic RHI Testbench 使用说明

M1/M2/M3/M4 的执行入口已实现：在原 `MetallicRHITests` 中增加声明式 requirements、设备配置、逐用例进程隔离、验证消息审查、完整产物检查和重放；M2 覆盖基础 RHI/Core/RenderGraph 和显式同步验证，M3 增加扩展的独立 reference/target 执行与父进程比较；M4 增加生产同步编码测试、可选有限 trace、合法 Buffer 序列和隔离缩减。旧 `--rhi-*` / GoogleTest 入口保持可用。设计背景见 [实现方案](RhiTestbenchPlan.md)。

## 构建和首次运行

复用已配置的 Windows x64 MSVC 构建树，先构建测试，再运行 CTest：

```powershell
cmake --build build-pass-stages-nrd --target MetallicRHITests Metallic --parallel 8
ctest --test-dir build-pass-stages-nrd -R '^MetallicTestbench\.' --output-on-failure
```

实际 build 目录可替换为其他启用 `METALLIC_BUILD_TESTS` 的兼容配置。M1 的进程隔离暂限 Windows；其他平台旧测试入口不受影响。

| CTest | 内容 |
| --- | --- |
| `MetallicTestbench.Harness` | 无 GPU 的 recorder、requirements、协议及真实异常子进程自测 |
| `MetallicTestbench.Contract` | 无 SDL/Device 初始化的 BufferRange、RenderGraph 访问计划、同步编码、trace 和序列契约 |
| `MetallicTestbench.Core` | 资源/Result 契约，buffer/纹理复制，图形状态、depth、history、timestamp |
| `MetallicTestbench.Binding` | bindless、参数生命周期、BufferSlice、ComputeKernel 与 prepared dispatch |
| `MetallicTestbench.Sync` | 显式同步验证：Core barrier、RenderGraph、Joined/Pipelined、query ring、copy timestamp |
| `MetallicTestbench.SyncActivation` | 真实层的 core/sync 对照负例：只录制图像 WAW，不提交 |
| `MetallicTestbench.Extensions` | 扩展 A/B、解析 AS、OMM bake、CPU/GPU 解压；允许能力缺失时显式 skip |
| `MetallicTestbench.ExtensionsRequired` | 可选参考机器策略，所选用例与 comparison 必须通过 |
| `MetallicTestbench.Async` | 两种提交模式的独立 copy 进度、跨队列依赖与 query ring |
| `MetallicTestbench.Property` | 固定 seed 的 4 个合法 Buffer 序列；同步验证和 CPU shadow readback |
| `MetallicTestbench.Shrinker` | 显式注入 CPU oracle 故障；真实 GPU 执行、两次确认、缩减、最小序列重放 |
| `MetallicTestbench.Trace` | 仅诊断构建注册；trace 开/关读回及图规划对照、溢出必须失败 |

所有 GPU testbench 作业与当前目录中已有 rhi/editor CTest 使用 `MetallicGPU` 锁。外部手工启动的编辑器和其他项目不受 CTest 锁控制。Contract/Core 要求所有选择用例通过；Binding/Sync/Async 允许缺少声明能力或队列时显式 skip。参考 GPU 验收必须加 `--tb-require-all`，禁止将 skip 视为通过。Sync 自动包含 Async 的三个提交/query ring 回归。

## CLI

```powershell
$test = '.\build-pass-stages-nrd\tests\MetallicRHITests.exe'
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

`--tb-plan` 不初始化 SDL、TaskSystem 或 GPU；它枚举已迁移 case，落盘配置计划、shader 输入指纹和构建身份。基础 profile 是 `core`（关闭可关闭的 RT/DGC/SDK、optimal layouts）、`binding`（加 bindless）、`async`（再请求独立 compute）。每个 case/iteration 都使用新子进程；尚未优化为同 profile 的多用例设备复用。

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

## HTML 报告

每次执行自动生成离线 HTML，无需额外开关、Python、网络或本地服务器。布局参考 [agfx test report](https://github.com/AmelieHeinrich/agfx/tree/main/tools/test_report)，结果仍以执行器核验后的 verdict 为准。

- `--tb-run`、`--tb-replay`：`<run>/report.html` 汇总，以及每个执行和 comparison 目录下的 `report.html`。
- `--tb-shrink`：缩减输出目录下的汇总，以及每个 attempt 的详情。故障复现的 Fail 是预期证据；缩减是否成功查看 `shrink.json`。
- `--tb-self-test` 和旧 GoogleTest 入口：`<output-dir>/reports/<timestamp>/report.html`，每个 case / iteration 都有独立详情。旧入口保留原有图片输出位置，只将本用例新增或改写的产物复制进报告目录；它使用 GoogleTest verdict，不提供隔离执行器的 validation 审计保证。

终端打印 HTML 路径，双击即可打开。页面支持名称/coverage/消息搜索，状态、产物类型、责任层、profile 筛选，名称/耗时排序；点击卡片查看实际结果、seed、设备/驱动、输出/预期预览、Buffer 差异、validation、trace 摘要和原始证据链接。JSON、图片和预览数据内嵌，所以单个 HTML 可以离线分享；要打开原始产物链接，需保留对应输出目录结构。

报告保留 `SkipUnsupported`、`SkipNotEnabled`、Crash、Timeout、InfrastructureFailure 等实际状态；require-all 导致的 skip 算失败并单独标注。每个用例结束后原子更新汇总，因此已完成用例不会因之后子进程崩溃而失去报告。父进程提前退出时最后一份汇总标为 incomplete；规划/列举用例和执行开始前的参数错误不生成执行报告。报告写入失败会记录错误并使运行非零退出，既有机器证据继续保留。

耗时是 CPU 墙钟：隔离子进程包括启动、设备创建、测试和 cleanup；comparison 单独计时；旧入口使用 GoogleTest 用例时间。它们不代表 GPU 时间，也不能直接作为性能对比。汇总墙钟包含报告生成和调度开销。

预览只展示实际可解释的数据：PNG，或 fixture 明确提供 `visuals.json` 的 RGBA8 区域。原始 RGBA 缩略图默认显示 RGB（忽略 alpha）；详情可切换 RGBA 透明度或 Alpha 灰度，原始像素保持不变。Buffer 的逐字节差异是展示信息，不替代浮点容差、解析 oracle 或扩展 comparison 的正式判断；没有加入未经计算的 FLIP 指标或 golden 图。单个预览读取上限 16 MiB，PNG 上限 256 KiB，RGBA 区域最大 512×512；每个 case 最多四组图片/Buffer 对比。汇总达到 24 MiB 展示预算后保留状态和详情链接，后续大预览只在各自 case 页显示。日志/诊断文本有展示截断，完整文件继续保留。

添加 RGBA8 预览时通过 `Evidence` 保存 `visuals.json` 数组，每项明确 `format: "rgba8"`、`actual` 文件名、`width`、`height`，可选 `expected`、`label`、`rowPitch`、`offset`。文件只能来自当前用例目录；只有格式和范围通过检查才嵌入预览。报告生成不执行 GPU 工作，也不会改变原有 oracle 或结果状态。

本机报告验证：Release/MSVC 构建通过；12 个 testbench CTest 作业通过，最终版本另完成 10 项 harness 自测、13 个 core GPU 用例、单用例 replay 和旧入口 3 个图像用例 × 2 轮。Edge 直接打开 `file://` 报告，验证筛选、详情/证据链接、RGB/RGBA/Alpha 像素、桌面/窄屏布局和脚本注入防护；真实 extensions 报告保留 OMM target / comparison 的两个 skip，未把它们算成 GPU 验证通过。

## 重放

```powershell
& $test --tb-replay .tmp/testbench/<run>/core/RHICommand.buffer_copy_offset_readback/0
```

恢复 case、profile、validation、seed 和 layerPath，并写入新的输出目录。可执行文件或 shader 输入指纹变化时拒绝；修复后用 `--tb-allow-version-mismatch` 明确允许新版本，原始输入保留在新 run.json 中供比较。不会使用原目录覆盖结果。

覆盖率只报告已选择且已迁移的格子，不代表全部 RHI；多轮通过不会被描述为新增独立覆盖。尚未提供通用格式 requirements、完整物理能力 probe、跨机器 differential 调度、完整 GPU trace 或通用 fuzz。基础纹理用例限定 RGBA8/R32Uint/D32 等既有格式，不声称覆盖所有格式组合。

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

测试分清证据类型：访问计划纯 CPU 检查和 barrier 编码计数不算 GPU 数据读回。同步验证也不覆盖所有访问：本机 1.4.350 层未报告初始 `vkCmdCopyMemoryKHR` 地址复制 WAW 探针，因此激活探针采用图像 clear WAW；地址复制、BDA、bindless 的正确性仍依赖数据 oracle 与计划/编码断言。不能把“零同步消息”解释为这些路径已被层完整检查。当前 API 没有独立的 blend state 配置，因此没有宣称任意混合状态覆盖。扩展对比的实际范围见下节。

## M3 扩展对比

`--tb-suite extensions` 包含 12 个逻辑用例：9 个 pair、2 个 GPU 解压用例、1 个 CPU OMM bake 用例。每个 pair 的 reference 和 target 使用同一 executable，在不同子进程创建 Device。目标能力不可用时保留已通过的 reference，target 与 comparison 单独 skip；任何一侧失败都会使 comparison 失败。

| 对比 / 检查 | 实际覆盖 | 责任层 |
| --- | --- | --- |
| optimal / unified layouts | 三次 pipeline → shader object → pipeline 绘制与全像素 copy 读回；lazy view、早释放、取消 | RHI |
| Standard TLAS / PTLAS，mapped + native | 共用 6 条解析射线，hit/miss、mask、instance/primitive ID、t、重心、正反面；变换后 Standard refit / PTLAS 同存储重建；typed backend 拒绝与分配退休 | RHI |
| 普通三角形 BLAS / CLAS | GPU 实际尺寸、两次 relocation；每次完成后释放旧 CLAS，再用新地址构建 BLAS/TLAS 并查询；旧 allocation 必须过期 | RHI |
| fallback positions / PositionFetch，默认 + authored tangent + native | 合成 glTF；BLAS 压缩、释放 build-only 几何、TLAS refit、变换、UV、normal、back-face TBN 的独立解析断言与 192 个浮点值对比 | Core scene integration |
| shader alpha / OMM | 合成 alpha texture；7 步 cutoff、UV、alpha、BLEND、变换；每步 4096 rays 对 CPU bilinear repeat oracle；父进程比较 visibility 并要求 shader alpha candidates 减少超过一半 | Core scene integration |
| direct / DGC | 固定 pipeline、execution set 切换、push constants、GPU count、显式 preprocess、执行后显式重绑；12 个整数结果 | Vulkan Backend |
| GPU 解压 | CPU 原始字节、mixed Raw/GDeflate tiles、尾块、取消、slot 复用、CPU/GPU 阈值切换 | Core streamer |
| OMM bake | 无 Device；subdivision 0–5 的 packed states、opaque/transparent/unknown、双线性边界、repeat seam、cutoff 等号、BLEND | Core CPU |

`RayQueryFixture.h` 的 CPU oracle 独立计算单位三角形的交点与重心，probe 在 `UnifiedTopLevelProbe.slang`。射线避开三角形边界，正反面判据依据 [Vulkan ray-space signed area](https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-traversal-culling-face)。测试不会调用生产场景交点代码来生成预期。

新增 profile：`core-unified`；`ray-query` 与 `ray-query-position/omm/ptlas/clas`；`binding-dgc`；`decompression`。每个 pair 只改变声明的一个 feature 请求。父进程核对 GPU UUID、driver、API/validation mode、seed/iteration、binary/shader 指纹、fixture 内容、目标能力 requested/enabled 和 `execution.json` 的实际分支标记。配置中其他已记录字段必须相同。整数严格比较，RT 浮点绝对容差为 1e-5，PositionFetch 为 1e-4；非有限值/空观察结果失败。OMM 的 visibility 不比较本来就应不同的 candidate 次数。

原始 `readback.bin`（内部轮次追加 `.1` 等）、`fixture.json`、`observations.json`、`execution.json` 是 pair 的必需产物，接受之前仍经过 M1 的 manifest/hash 检查。汇总额外生成 `comparisons.json` 和 `comparisons/<case>/<iteration>/diff.json`，记录容差、数值数量、最大绝对误差、首个不一致位置和两侧证据路径。JUnit/coverage 中 comparison 与两个执行结果分开，不能把相同 fallback 的两次运行判成扩展通过。上述扩展 fixture 是固定输入，seed 被记录并核对；M4 的随机生成仅适用于单独的 Buffer property suite。

```powershell
& $test --tb-plan --tb-suite extensions
& $test --tb-run --tb-suite extensions --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin
# 参考机器：根据它应支持的范围填写 filter，skip 将导致非零退出。
& $test --tb-run --tb-suite extensions --tb-filter '*-RHIRendering.opacity_micromap_ray_query' `
    --tb-require-all --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin
# 可选 CTest 必测策略；这里只修改该构建树的新测试选项。
cmake -S . -B build-pass-stages-nrd `
    '-DMETALLIC_TESTBENCH_REQUIRED_EXTENSIONS=*-RHIRendering.opacity_micromap_ray_query'
```

上面的排除项仅适用于当前验证层环境：本机 1.4.350 被生产后端的 KHR OMM 版本检查禁用，后端要求至少 1.4.357。OMM reference 与 CPU bake 可以通过，但 target 为 SkipUnsupported，不能据此宣称 OMM GPU 路径已验证。安装兼容层后应将必测 filter 改为 `*`，重新执行 OMM pair；测试没有关闭 validation 绕过这个限制。

pair 不允许 `--tb-profile` 覆盖，防止两侧配置含义漂移。`--tb-replay <variant-directory>` 只重放这一侧，保留其 target requirements；输出空的 comparison 集合，不声称重放了完整 pair。需要重新比较时用原 case filter 运行 extensions suite。

覆盖边界：PTLAS 当前公共调用重写完整实例，并非任意 partition operation/update 队列；布局 pair 尚不涵盖所有 sampled/storage/格式组合；DGC 是 native compute pipeline 探针，不覆盖公共 draw/mesh/shader-object execution set。OMM/PositionFetch 仍复用合成场景集成 fixture，未把两套完整 scene builder 搬入裸 RHI。高级 AS/扩展的全格式、容量边界、压力和随机序列测试不因本轮通过而视作完成。

### M3 本机验证记录（2026-09-27）

已在 NVIDIA GeForce RTX 5070 Ti / 616.92、Release/MSVC、验证层 1.4.350 上构建 `MetallicRHITests` 与 `Metallic`。全部 9 个 testbench CTest 作业通过；extensions 的 21 个 child 中 20 Pass、1 OMM target skip，9 个 comparison 中 8 Pass、1 OMM skip，全部 validation recorder 为零消息且无 overflow。必测策略排除上述 OMM pair 后 27 个执行/比较结果全部 Pass。

另外验证了 PTLAS target 单侧 replay（Pass，comparison 集合为空）、把不可用 OMM 设为 require-all（保留 reference Pass，整体退出 1），以及旧入口 14 个受影响用例（13 Pass、1 OMM skip）。编辑器 smoke 完成 recorded/submitted/presented frame；这只是启动与提交检查，不代表新做过长期视觉稳定性验收。

## M4 诊断与合法序列

生产 `VulkanSynchronization` 模块共用 stage/access/layout 翻译和 scope 合法性检查，CPU contract 直接覆盖该模块，包括 copy queue、execution-only、扩展 gating 和 unified layout。GPU 路径仍使用原同步规则；本轮没有改变 RenderGraph 的调度算法。

诊断开关默认关闭。可在已有兼容构建树中只增加此选项：

```powershell
cmake -S . -B build-pass-stages-nrd -DMETALLIC_RHI_DIAGNOSTICS=ON
cmake --build build-pass-stages-nrd --target MetallicRHITests Metallic --parallel 8
ctest --test-dir build-pass-stages-nrd -R '^MetallicTestbench\.(Trace|Property|Shrinker|Contract)$' --output-on-failure
& $test --tb-run --tb-suite async --tb-filter 'RHIRendering.frame_self_submit_two_slots*' `
    --tb-validation sync --tb-trace --tb-require-all --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin
```

`METALLIC_RHI_DIAGNOSTICS` 是共享 runtime 的 PUBLIC 编译定义，编辑器和测试使用同一套后端。公共 RHI 对象布局不变。OFF 时观察 hook 内联为空，不在生产路径分配、加锁或增加等待；ON 但未安装 sink 时只检查观察指针。启用捕获会复制数据并在 recorder 内加锁，因此不用于性能结论。sink 按 Device 过滤，每个进程只允许一个会话，安装/卸载时调用线程必须静止。

`trace.json` 的 schema 为 1，记录 CommandBuffer 录制代次、VulkanRHI/DGC helper 的实际 barrier 和最终 `vkQueueSubmit2` 的队列、commands、合并后的 semaphore waits/signals、stage/access/layout/range、返回码。句柄在回调内转换为本次捕获的逻辑 ID，退休后重用句柄会获得新 ID。默认最多 8192 个事件、8 MiB 序列化事件内容，单事件最多 1024 个 barrier/submit 元素；`--tb-trace-limit 1..8192` 可降低事件预算。溢出/捕获异常是 InfrastructureFailure；未编译时请求 trace 是 EnvironmentFailure，不能静默通过。输入和 replay 保留开关与预算。

范围有明确边界：只观察这些 RHI 的同步与提交出口，不记录完整 draw/dispatch/bind、驱动内部工作或 SDK 直接发出的 native 命令。Buffer scope 当前编码为可合并的全局 `VkMemoryBarrier2`，相同 native layout 的图像也可能如此；trace 同时保存 requested range 与 actual global scope，不把它冒充原生逐 Buffer barrier。不同线程记录的事件顺序是 CPU 观察顺序，不是 GPU 执行时间线。

两种 `frame_self_submit_two_slots` 用例现在使用明确支持流水提交的测试 clear/upload pass，并断言实际模式；此前 Triangle/transfer fixture 会把 Pipelined 请求回退到 Joined，不能将两个名字当作模式覆盖。用例生成 `graph.json`，包含资源、声明 uses、计划 barriers、segments、batches 和前驱。开启 trace 后增加 `graph-trace-links.json`、第一帧 `graph-trace.json`、`graph-trace-checks.json`，把图资源/队列映射到 trace ID，检查计划范围的编码覆盖和独立 copy 实际提交没有 graphics wait。配合原有 readback 和 CPU 访问计划测试，可分别检查声明、规划、后端编码、执行四层。关联目前限定于这两个合成图的输出资源，不声称支持任意图的内部资源/stage/subresource 自动关联。

Trace 作业在同一 binary 下各运行 ON/OFF：Joined/Pipelined 图、Buffer sequence、mip/layer/volume texture copy。逐字节核对读回、具体序列和图规划快照，并检查实际图像 barrier 的布局、访问和范围；不固定任意 barrier 数量。另用 1 个事件的预算验证完整执行仍会因 trace 截断失败。这个对照证明所测 workload 的结果与规划一致，不代表零诊断开销。

### 序列文件与缩减

```powershell
& $test --tb-run --tb-suite property --tb-validation sync --tb-seed 1592594996 `
    --tb-repeat 4 --tb-require-all --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin
& $test --tb-replay .tmp/testbench/<run>/core/RHICommand.buffer_sequence/0
# 外部 concrete 文件先验证 schema/预算，再生成隔离执行计划。
& $test --tb-run --tb-suite property --tb-validation sync --tb-sequence path/to/sequence.json `
    --tb-layer-path C:/VulkanSDK/1.4.350.0/Bin
# 只接受已确认的 readback mismatch；路径是原失败 case 目录。
& $test --tb-shrink .tmp/testbench/<run>/core/RHICommand.buffer_sequence/0
& $test --tb-replay .tmp/testbench/<shrink-run>/minimal
```

`input.json` 内嵌具体 sequence，执行时另存 `sequence.json`。schema/generatorVersion 都为 1，包含 seed、iteration、4 个 Buffer × 64 个 uint32、带稳定 ID 的 commands。SplitMix64 根据 seed/iteration 默认产生 48 条 upload/fill/copy/readback，最多接受 256 条。fill 通过常量 staging upload + RHI copy 实现，当前公共 RHI 没有 fillBuffer；这不算验证原生 fill 指令。

所有 Buffer 在固定前缀创建、清零，命令只使用初始化的合法范围，不生成同 Buffer copy 或越界访问；资源保持到提交完成。每次使用声明 transfer scope，末尾 transfer→host 可见性，一次提交/fence。CPU 数组独立执行命令，随机 checkpoint 与最终四个 Buffer 全量读回保存 `expected.bin`、`readback.bin`、`sequence-diff.json`。序列中的具体操作而非重新生成的随机数决定重放行为。外部 sequence 的 seed 同步到执行计划，覆盖 CLI 的生成 seed；其 iteration 表示生成序号，执行重复次数仍由 runner 独立编号。

缩减器使用删除块的 delta debugging；固定初始化和对象生命期使每个候选仍合法，stable checkpoint ID 与 `readback-mismatch` 类别必须保留。原例和每个接受的候选在两个新 child 中连续复现；沿用父进程 manifest/hash/退出码检查，且 validation 必须零消息。GPU UUID、driver/API/validation 与原例一致；binary/shader 指纹沿用 replay 检查。Timeout/Crash/DeviceLost/环境故障不被接受为缩减成功。

预算为 64 个 child、120 秒墙钟，每个 child 最多 15 秒且受剩余时间限制。保存 `original.json`、每步即时更新的 `minimal.json`、各 attempt 的完整证据、`attempts.json`、`shrink.json` 与可直接 replay 的 `minimal/input.json`。只有完整尝试了逐条删除才报告 `oneDeletionMinimal`，预算耗尽则明确标记，不声称全局最小；尚不缩减数值/范围参数，也不自动缩减 GPU hang 或验证错误。

`sequence-fixtures` 是单独显式选择的故障 suite，故意对指定 fill 触发的最终 CPU 预期值翻转一位，GPU 命令仍合法。`injectedOracle=true` 写入 diff/失败签名；不注册进普通旧入口。Shrinker CTest 要求 49 条命令最终剩 1 条触发命令，并再次从最小输入启动 fresh child 复现。它证明执行/归类/重放/缩减链路，不能当作发现硬件错误的证据。

### M4 本机验证记录（2026-09-27）

在上述 RTX 5070 Ti / 616.92 / validation 1.4.350 环境中完成诊断 OFF 和 ON 的 `MetallicRHITests`、`Metallic` 构建。12 个 testbench CTest 作业通过；其中修复实际提交模式后的 Contract/Sync/Async/Property/Shrinker/Trace 6 个作业重新通过。8 个 trace A/B child 均 Pass、validation 零消息，实际 Pipelined/Joined 布尔值分别为 true/false；8192 事件正常捕获与 1 事件溢出失败均验证。

注入的 49 条 Buffer 命令在 14 个隔离 child 内缩减为 1 条，报告逐条删除最小且预算未耗尽，最小输入再次重放相同故障。具体 sequence 文件导入的 seed 与计划一致，正常 replay 保留 trace，并与原例的 sequence/expected/readback 逐字节一致。DGC reference/target/comparison 带 trace 通过，实际 preprocess write→indirect read barrier 已被记录。

旧入口 6 个相关用例全部通过，包括真实流水提交的 GPU progress/失败取消回归。编辑器完成 recorded/submitted/presented frame；未做新的长期场景、视觉稳定性或性能结论。M3 的 OMM target 仍受 validation 版本限制。资产缓存未发生受控文件变更，原有 `External/microprofile` 工作区状态保持。

本轮本地产物索引为 `.tmp/testbench-m4-verification.json`，包含构建/CTest/旧入口/smoke 日志、trace 对照和 shrinker 目录。它是本机证据，不纳入源代码。构建树 `build-pass-stages-nrd` 当前保持诊断 ON；项目选项默认仍为 OFF。

## 添加用例

现有 `RHITest` 默认无 metadata，仍走旧 runner。要迁移单个用例：

1. 覆盖 `metadata()`，声明 suite、profile、责任层、requirements、timeout、coverage 和必须生成的 oracle 文件。
2. 保留原有 name、type 和 `run(RHITestContext&)`。通过 `context.evidence` 写原始结果；旧模式中它为空。
3. Device 由 harness 提供；迁移旧用例使用 `bench::createTestDevice`，必要的第二个 Device 明确指定 `additionalDevice=true`。不要在 run() 内另建未接 sink 的 Device。测试自己等待所提交工作的完成并安全释放本地对象；harness 的末尾 drain 不能修补测试内部提前释放资源。
4. CPU-only 用例声明 `requiresDevice=false`、validation Off、空 queues，并实现 `runCpu(Evidence&)`；cleanup 用 `cleanupCpu()`。
5. 用 `--tb-plan` 检查 requirements，再运行样板、对应旧过滤测试与 harness 自测。

自测的 `harness-fixtures` suite 是显式负例集合：timeout、crash、cleanup exception、首错保留、unexpected skip、真实 shader 编译失败和正常通过。它默认不运行，也不向旧注册表添加故意失败的测试；自测会断言这些失败被正确归类。
