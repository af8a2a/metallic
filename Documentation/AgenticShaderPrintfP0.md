# Shader Printf Agentic Debugger：P0 验收

2026-09-26：P0 已完成本机真实 GPU 验收。Slang → SPIR-V DebugPrintf → VVL 插桩 →
INFO callback → 原始证据的链路已跑通；普通 compute、生产 mapped descriptor heap
和实验性 native descriptor heap 分开验证。此结果是计算路径的能力基线，尚不是
WorkControl 生产 logpoint 或完整 Agent 调试会话。

## 实现

- [VulkanShaderPrintf](../Source/Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.h)：
  `DeviceDesc.shaderPrintf` 显式启用，默认空指针；session 必须活到 device 销毁之后。
  必须发现 Khronos validation、debug utils、layer settings，并成功创建 messenger；
  Printf 请求缺条件时返回失败，不按无插桩模式静默继续。
- 通过 `VK_EXT_layer_settings` 设置 Printf、INFO、stdout=false、64 KiB GPU buffer、
  无重复消息抑制。`printf_only_preset=false`，保留常规验证。
  本机 VVL 实際要求 `printf_buffer_size` 使用 **UINT32**；manifest 的 UI 类型 `INT`
  不能直接推导 Vulkan layer-setting 类型。
- 显式检查并开启本机 VVL 插桩需要的 stores/atomics、timeline、memory model、
  8/16-bit storage 特性。只在 Printf 请求时增加这些特性；仍保留 RHI 的其他要求。
  NvPerf 与 Printf 的同时请求在设备初始化中拒绝。
- INFO callback 在 instance 创建期和设备期均接入。Printf session 中的 INFO、Warning、
  Error 统一复制到 raw queue，替代原 `ValidationSink` 的即时日志/JSON 路径。
  回调不分配、不调用 Vulkan；try-lock 失败、容量耗尽均计入 dropped。
  队列最多 256 条，每条 message 4096 bytes、ID name 160 bytes（含终止符）；
  截断单独计数。队列存储在堆上，快照在完成后读取。
- [MetallicShaderPrintfProbe](../Source/Tools/ShaderPrintfProbe.cpp) 为手动构建工具，
  不属于默认 ALL，也不成为编辑器/sample 依赖。无需 Nsight、UI 自动化或窗口操作。
  探针使用 SDL Vulkan loader，但不创建呈现窗口。
- [Slang fixture](../tests/rhi/shaders/ShaderPrintfEcho.slang) 只选择 group.x=1、lane=3，
  dispatch 为两个 32-thread groups。heap 路径分配非零 slot，使用最终 descriptor 索引，
  读取输入、写入输出并在 GPU 完成后核对四个 uint。不会把只有 Printf 的空 heap 测试
  视为生产资源访问通过。

## 实测能力

[最终 Suite.json](../build-release/shader-printf-p0-verified-20260926/Suite.json)
中七项 `accepted=true`，`complete=true`、`passed=true`。
负用例的 accepted 表示正确拒绝，不表示 Printf 能力验证成功。

| 路径 / 故障 | 真实观察 | probe 状态 / 退出码 |
|---|---|---|
| ordinary compute | 1 条精确 echo | verified / 0 |
| mapped descriptor heap compute | 1 条精确 echo；回读 `[74,305397763,15,16]` | verified / 0 |
| native descriptor heap compute | 1 条精确 echo；同一回读结果 | verified / 0 |
| 缺 INFO 订阅 | GPU 完成，callback 0 条 echo | incomplete / 2 |
| stdout 重定向 | stdout 有精确 echo，callback 0 条 | incomplete / 2 |
| 128-byte GPU buffer，预期 128 条 | 仅 2 条；INFO 消息中明确报告 buffer truncation | incomplete / 2 |
| layer 搜索路径为空 | layerDiscovered=false；未创建 instance | failed / 2 |

实际加载与运行环境：

| 项目 | 本次记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Driver | NVIDIA 616.92；Vulkan API 1.4.351 |
| VVL | `C:/VulkanSDK/1.4.350.0/Bin/VkLayer_khronos_validation.dll`；声明 API 1.4.350 |
| Slang | 2026.18.2；实际 `build-release/Source/slang-compiler.dll` |
| 编译 | spirv_1_6，debug mode Disabled，disk cache=false |
| Printf 识别 | 实际 `OpExtInst` 的 NonSemantic.DebugPrintf；callback ID `0x4fe1fef9` + INFO |
| 来源 / 入口 | source `ordinaryMain` 或 `heapMain`；SPIR-V 入口均为 `main` |

每例保存 `Report.json`、`RawMessages.json`、`compiler.spv`、stdout/stderr。
`Report.capabilities` 区分发现、instance/messenger/device 配置；`smokeVerified` 必须由
精确 echo、GPU 完成、heap 回读和无丢失共同成立。发现 DLL 或成功编译都不算验证通过。
所有记录明确标注 `instrumentation=true`、`performanceEligible=false`。
`compiler.spv` 是编译器结果，不宣称是 VVL 内部插桩后的最终二进制。
[Hashes.json](../build-release/shader-printf-p0-verified-20260926/Hashes.json) 的所有文件哈希
已复核；Suite 同时记录 executable、实际 layer/Slang DLL、shader dependency 的 SHA-256。

## 复跑

使用现有 Release 配置，在 x64 Visual Studio developer PowerShell 中：

```powershell
cmake --build build-release --target MetallicShaderPrintfProbe -j 12
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
python Tools/RunShaderPrintfP0.py --exe build-release/Source/MetallicShaderPrintfProbe.exe --layer-path C:/VulkanSDK/1.4.350.0/Bin --output "build-release/shader-printf-p0-$stamp"
```

路径按本机实际 layer 目录调整。输出必须是新目录，工具不会覆盖旧证据。
[runner](../Tools/RunShaderPrintfP0.py) 串行启动独立进程，每例默认 90 秒上限；
崩溃或超时会停止后续 GPU 测试，不能视为预期负例通过。
GPU fence 等待限 10 秒；外层进程超时也覆盖 queue idle、编译和驱动停留。

runner 仅为子进程设置 `VK_LAYER_PATH` / `VK_IMPLICIT_LAYER_PATH` / layer settings 路径，
避免本机失效的 EOS overlay、`E:/Validation.json` 注册项污染本轮证据。
没有修改注册表、全局环境、系统 SDK 或 VkConfig。
`--layer-path` 省略时沿用显式 layer 发现路径；异常 loader 消息会使成功用例不合格。

## 回归与边界

已完成：

```powershell
cmake --build build-release --target MetallicShaderPrintfProbe Metallic -j 12
cmake --build build-pass-stages-nrd --target MetallicShaderPrintfTests -j 12
ctest --test-dir build-pass-stages-nrd -R '^MetallicShaderPrintf(Tests|EvidenceTests)$' --output-on-failure
.\build-release\Source\Metallic.exe --smoke-test
```

两项 CTest 均通过：4 个 C++ callback 用例、7 个 Python 证据用例。
覆盖 borrowed memory 复制、队列满、精确容量与截断、并发计数；以及空记录、重复 echo、
伪装消息、INFO 级溢出、错误容量、配置告警误判、heap 缺回读、重定向、崩溃等验收边界。
普通编辑器 smoke 退出 0，完成 RenderGraph 录制、提交及呈现，
[日志](../build-release/shader-printf-p0-default-smoke/Output.log)与
[结果](../build-release/shader-printf-p0-default-smoke/Result.json)已保存。
这只验证启动与一帧运行，未进行全场景图像、时序或显存压力验收。

初次开发失败证据保留在 `build-release/shader-printf-p0-first` 和 `*-suite-01`：
前者错误使用源函数名作为 Vulkan 入口，VVL 在解析阶段发生访问异常，已改为验证和使用
实际 SPIR-V `main` 入口；后者暴露容量 setting 类型错误与本机陈旧 layer 注册项。
这些失败结果没有改写成通过。当前最终证据以 `*-verified-20260926` 为准。

后续 P1 已实现 job/token/source mapping、typed event decoder、export/offline verifier，
并增加 `metallicctl shader watch`；详见 [P1 验收记录](AgenticShaderPrintfP1.md)。
P2 尚未实现：WorkControl 生产站点、variant lease 与生产恢复；editor 尚无 `--shader-trace`。
本轮未验证 graphics shader object、mesh/task、fragment、ray query/ray tracing 中的 Printf。

依据：[Slang printf](https://docs.shader-slang.org/en/latest/external/core-module-reference/global-decls/printf.html)、
[VVL Debug Printf](https://vulkan.lunarg.com/doc/view/latest/windows/debug_printf.html)、
[Vulkan loader layer 路径](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md)。
