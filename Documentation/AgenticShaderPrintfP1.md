# Shader Printf Agentic Debugger：P1 有界事件协议

日期：2026-09-26。P1 已完成控制面、原始证据、typed decoder、导出与离线复核。
真实 GPU 验证范围为独立 compute fixture 的 mapped/native descriptor heap。
生产 WorkControl logpoint、variant lease 和恢复确认继续属于 P2。

## 已交付

- `Source/Runtime/Debug/ShaderTraceCore.*`：有界记录、不可复用 token、MTS1 解码及完整性判定。
- `Source/Runtime/Render/Debug/ShaderTraceRuntime.*`：owner 线程桥接，复核 graph/generation；
  fixture 在编译前后与提交前复核 source hashes。Vulkan 对象不进入 DebugCore/IPC。
- `DebugCore` / `metallicctl`：`shader.capabilities`、`shader.sites`、`shader.watch`，
  复用 `jobs.get/cancel`、`capture export`、`eval`，新增离线 `shader verify`。
- `Shaders/Modules/ShaderTrace.slang` 与 `tests/rhi/shaders/ShaderTraceFixture.slang`：
  固定格式 BEGIN/DATA/END；GPU 内筛选 group 1 / local index 3，真实 heap 读取和输出回读。
- `Tools/RunShaderTraceP1.py`：串行真实 GPU、CLI、导出、在线/离线一致性和篡改拒绝验收。

Probe 服务先完成 P0 live echo/readback，才公布 `smokeVerified=true, fixtureOnly=true`。
普通编辑器 watch 返回 `RestartRequired`；未增加默认 SDK 依赖或生产插桩。
`MetallicShaderPrintfProbe` 保持 `EXCLUDE_FROM_ALL`，诊断编译关闭磁盘 cache。

## 运行

在兼容的 x64 MSVC 开发环境中：

```powershell
cmake --build build-release --target MetallicShaderPrintfProbe MetallicCtl -j 12
python Tools/RunShaderTraceP1.py --exe build-release/Source/MetallicShaderPrintfProbe.exe --cli build-release/Source/metallicctl.exe --output build-release/shader-trace-p1-new-run --layer-path C:/VulkanSDK/1.4.350.0/Bin
```

输出目录必须不存在。runner 仅隔离子进程 Vulkan layer 环境，两个服务串行运行，
每个服务有效期 25 秒、正常自然退出。失败保留证据并停止后续 GPU 测试。
手动服务支持 `--serve --mode heap-mapped|heap-native --serve-seconds 1..300 --output NEW_DIR`。
从 `Server.json` 获取当前 PID/session，也可用 `metallicctl list`。

```powershell
metallicctl --pid PID --session SESSION --json shader capabilities
metallicctl --pid PID --session SESSION --json shader sites
metallicctl --pid PID --session SESSION --json shader watch --spec watch.json --wait
metallicctl --pid PID --session SESSION --json jobs get JOB
metallicctl --pid PID --session SESSION capture export JOB --out NEW_CAPTURE_DIR
metallicctl --capture NEW_CAPTURE_DIR --json shader verify
metallicctl --capture NEW_CAPTURE_DIR --json eval 'shaderTrace.records[0].fields'
```

最小 `watch.json`（schema hash 取自当前 `shader sites`）：

```json
{
  "version": 1,
  "generation": 1,
  "target": {"site": "fixture.echo", "expectedSiteSchemaHash": "<current site schemaHash>"},
  "invocation": {"group": [1, 0, 0], "localIndex": 3},
  "limits": {"targetFrames": 1, "maxRecords": 16, "timeoutMs": 30000},
  "fixtureScenario": "matched"
}
```

P1 只接受 manifest 声明的固定 invocation 和完整字段 schema。未知参数、生产 pass/entry
选择、任意表达式、predicate/fields 筛选均拒绝。fixtureScenario 只在 fixture 站点开放，
支持 matched/no-match/site-not-reached/missing-end/quota，不代表生产谓词系统。

`Ready` 只表示 artifact 可用。Agent 必须检查 `outcome` 与 `selectedScopeComplete`。
离线 `shader verify` 退出 0 表示校验与解析成功，观测结果仍可能是 Incomplete。

## 协议、身份与生命周期

MTS1 全部用无符号十进制 u32：

```text
MTS1 sessionLo sessionHi runLo runHi dispatchLo dispatchHi siteId kind groupX groupY groupZ localIndex seq payloadWordCount [payload words]
```

kind 0=BEGIN，seq=0、空 payload；kind 1=DATA，按 schema 解码；kind 2=END，payload 为
`siteEvaluationCount matchedCount emittedCount exitReason budgetExceeded`，seq=emittedCount+1。
总配额最多 16 条，BEGIN/END 保留两条，DATA 最多 14 条。达到配额只抑制打印，继续计算。
f32 传原始 bits，另给 classification/value；NaN value=null、payload 仍保留。u64 用
low/high 和既有 lossless JSON，避免大整数精度丢失。

session token 为 session 的 SHA-256 前 64 位；run/dispatch token 单调增加、不回绕。
每个 token 绑定 graph/generation/execution、pass/phase、recording、dispatch ordinal、
queue family、提交 timeline、source/schema、输入指纹和编译 variant。提交后不能更改
variant。实际 compiler SPIR-V 的 SHA-256、字节、编译模式、宏与依赖 hash 同包导出。
`deviceSpirv=null`：没有声称取得 VVL/驱动内部最终插桩二进制。

回调池 256 slot，每条 text 最多 4095 字节、idName 最多 159 字节；回调无 JSON、
Vulkan 调用或等待。竞争/满队列记 host drop，截断独立计数。owner 排序/解码/封存，
arrival 仅供审计。未知/过期 token 保留为 orphan，不能套用当前帧身份或算作当前 DATA。

P1 每次观察创建独立 device/instance。tracked submit 完成、queue idle、资源及 instance
销毁后 drain 并封存。完整性同时要求 BEGIN/END、连续 seq、计数一致、回读正确、
backend 健康且无丢失/截断/重复/orphan/quota/停止原因；不以安静窗口证明回收。
此边界未推广到常驻生产 renderer。

一次只允许一个 active observation，复用 Queued/Planning/Compiling/Recording/
Submitted/Collecting/Ready 状态。取消/超时保持已取得的 reservation 和 token，直到 owner
结束 GPU/backend 生命周期。Cancelled 终态仍可携带 artifact；超时原因保留为 Timeout。
`--wait` 可能先返回错误，继续 `jobs get`，出现 artifactCount 后可导出部分证据。
排队阶段就取消、没有取得 reservation 的请求不会凭空生成 GPU artifact。

## 验收与证据

[最终 Suite.json](../build-release/shader-trace-p1-verified-20260926/Suite.json) 保存 runtime、
二进制 hash、layer 环境和每项结果；[Hashes.json](../build-release/shader-trace-p1-verified-20260926/Hashes.json)
覆盖原始文件。每个 capture 用现有 manifest-last 导出，`0.bin` 是 lossless JSON 原始包，
包含 request/site/dispatch/health/rawMessages/runtime。离线核 SHA-256 后以 parserVersion=1
重新判定完整性，忽略缓存的成功结论。SHA 是完整性校验，不是来源签名。

| 真实 GPU 场景 | mapped heap | native heap | selectedScopeComplete |
|---|---|---|---|
| matched | Matched | Matched | true |
| no-match：站点求值但无匹配 | NoMatch | NoMatch | true |
| site-not-reached：入口执行但站点未求值 | SiteNotReached | SiteNotReached | true |
| missing-end | Incomplete | Incomplete | false |
| quota：15 次匹配、14 条 DATA、两条控制记录 | Incomplete | Incomplete | false |

十项均检查 submit 完成、heap 回读、token、SPIR-V hash 和在线/离线相等；有 DATA 的场景
验证负零 0x80000000、NaN payload 0x7fc12345、u64=2305843009213693953 与 heap value=73。
两条路径各额外篡改独立副本，均被拒绝。环境：RTX 5070 Ti / NVIDIA 616.92、
Vulkan 1.4.351、SDK VVL 1.4.350、Slang 2026.18.2。

CPU 测试通过：22 个 `MetallicDebugTests`（含 12 个 ShaderTrace）、5 个 callback 测试、
7 个 Python evidence 测试。覆盖乱序/延迟 token、丢失/截断/重复、坏协议/数值溢出、
缺 END、计数矛盾、scope/schema/generation 失配、bounded queue、取消/超时保留资源和
部分导出、token 溢出、SHA 标准向量、离线篡改。乱序与取消/超时是确定性主机测试，
没有声称重现驱动随机乱序或真实 device lost。

```powershell
cmake --build build-pass-stages-nrd --target MetallicDebugTests MetallicShaderPrintfTests -j 12
ctest --test-dir build-pass-stages-nrd -R '^(MetallicDebugTests|MetallicShaderPrintfTests|MetallicShaderPrintfEvidenceTests)$' --output-on-failure
```

P0 [七项真实 GPU 回归](../build-release/shader-printf-p1-regression-20260926/Suite.json)均 accepted，
包括 INFO 关闭、stdout 重定向、GPU overflow 和 missing layer 的预期不完整/失败结果。
普通编辑器 [smoke 退出 0](../build-release/shader-trace-p1-default-smoke-20260926/Result.json)；
该结果只证明启动和一帧提交/呈现，不能证明全场景图像、时序或显存稳定性。

开发期失败保留在 `build-release/shader-trace-p1-first-20260926`：Slang 跨模块字段未声明
public，作业正确导出 ExecutionFailed 而非 NoMatch。修复后第二轮通过；最终证据以
verified 目录为准，原始失败未改写。

## P2 边界

尚未提供 WorkControl 生产站点、图像正确性对照、在线 variant 恢复、多 invocation、
graphics shader object、fragment/mesh/task/ray-tracing 的验证。所有 Printf capture 均为
performanceEligible=false，不能用于无插桩 A/B 性能结论。
P2 应在可信 WorkloadCase 的真实 bind 点接入具名站点与 completion lease，证明持久
instance 的消息回收边界，并验证取消后的下一次生产绑定。
