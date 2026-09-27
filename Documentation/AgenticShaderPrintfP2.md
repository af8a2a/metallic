# Shader Printf P2：WorkControl 生产 logpoint

日期：2026-09-27。P2 独立进程批处理目标已实现，真实 MiniZorah WorkControl 验收通过。

## 已实现的使用入口

从仓库根目录运行，要求启用 Streamline 的现有 Release sample/CLI、Vulkan validation layer，以及 MiniZorah 资产。
输出目录必须全新。每个进程只观察一个 dispatch、一个 invocation、一个目标帧。

```powershell
python -B -X utf8 Tools/Perf/ShaderTrace.py run --layer-path C:/VulkanSDK/1.4.350.0/Bin --phase early --group-x 0 --local-index 0 --output build-release/shader-watch-new
python -B -X utf8 Tools/Perf/ShaderTrace.py verify build-release/shader-watch-new
```

`--phase late` 选择 late WorkControl；`--triangle-id 0` 添加 `triangleId eq u32` 谓词。
支持第一行 workgroup 的 X=0..65534、localIndex=0..127；实际未执行的 invocation 不能凭空
产生完成证据。小于 32-lane wave 的 fallback 报告 UnsupportedPath，完整性为 false。
未指定谓词时观察站点实际求值。字段集固定，不执行任意表达式或源码行。

单次 `run` 的 verified 只表示该观察通过；`p2Acceptance=false`。
完整验收使用下面的固定顺序：无插桩控制进程，early 三次，late 三次，NoMatch，SiteNotReached。
后三种重复/阴性控制使用 group=(0,0,0)；SiteNotReached 选择 lane=127，要求该组实际三角形
数量小于 128，条件不成立会失败，不重试挑选能通过的数据。

```powershell
python -B -X utf8 Tools/Perf/ShaderTrace.py acceptance --layer-path C:/VulkanSDK/1.4.350.0/Bin --output build-release/shader-trace-p2-new
python -B -X utf8 Tools/Perf/ShaderTrace.py verify build-release/shader-trace-p2-new
```

首次运行只顺序读取 MiniZorah 声明资产闭包，当前 5 个文件共 70,928,795,715 字节。
可用 `--assets <已有 Assets.json>` 复用 SHA-256 清单；每次进程前后重新检查大小和 mtime。
资产保留在原位置，未复制完整可移植场景。进程默认限时 240 秒，超时只清理本次拥有的进程树。
共享 `build/shader-experiment.lock` 防止合作 runner 同时跑 GPU；遗留锁不会被自动抢占。

## 生产接入和证据

站点位于 `GPUDrivenStreamWorkRaster.slang` 的 `streamClusterRasterWorkControlMain`，
实际三点已经由生产路径准备好、即将调用 `hybridRasterPreparedTriangle` 的位置。
读取的是现有局部值，原来的 raster 调用保留。默认 `METALLIC_WORK_CONTROL_TRACE=0`。

13 个字段为 recordIndex、triangleId、instanceFlags、triangleCount，以及 a/b/c 各自的
PositionX、PositionY、Depth。坐标保留生产 `HybridScreenVertex.position` 的有符号整数值，
不是重新投影结果；depth 保留 f32 raw bits。BEGIN/DATA/END 使用 P1 MTS1 协议与完整性判定。
选择 lane=0、谓词 UINT32_MAX 时应得到 siteEvaluationCount=1 的 NoMatch；选择未到站点的
lane=127 时应得到 siteEvaluationCount=0 的 SiteNotReached。缺消息或缺 END 均不算空成功。

`WorkControlShaderTrace` 通过现有 BeforeStreamEarlySoftware/BeforeStreamLateSoftware checkpoint
在真实生产 bind 后、push data 和 indirect dispatch 前换绑一个独立编译的 compute pipeline。
只允许 VBuffer、graphics queue、mode 5、frozen snapshot。early/late 有不同 schema hash，
选中的 binding 记录诊断 compiler SPIR-V SHA-256，并另存原 production SPIR-V FNV1a64，
不把原 hash 当成诊断 pipeline 的 hash。

诊断 variant 使用原 Slang 优化策略、mapped heap、136 字节生产 push layout、实际 SPIR-V `main`
入口；不修改全局 -O0，不写诊断 Slang 磁盘缓存。记录编译器版本、宏、全部依赖源码 hash 和
compiler SPIR-V 字节；`deviceSpirv=null` 表示未取得 VVL/驱动最终内部二进制。

pipeline/shader lease 同时由 adapter 和实际 command recording 持有，配合 submission transaction
记录提交；tracked frame completion 后才认定目标 GPU 完成。execution 与 frameSlot 来自实际
DebugEvidenceStamp；当前 commandBufferRecording 保存所属 RenderFrameContext.frameIndex，
不代表全局唯一 VkCommandBuffer 句柄。观察由 session/dispatchToken 和实际 phase 绑定关联；
token 和 generation 不在录制身份更新接口中被替换。
提交证据保存真实 frame 聚合 timeline values；覆盖多个 timeline 时单值字段为 null，
没有将聚合完成点误标成某个单独 dispatch 的 semaphore/value。
恢复帧重新走原生产 bind，不将诊断 pipeline 写回生产 pipeline 数组。

case adapter 沿用 MiniZorah history recipe：1797×660 输出、1198×440 内部尺寸、15 秒 ready 后
warmup、冻结流送、禁用 jitter/异步软件 raster、每个 checkpoint 前四帧偏移相机 priming。
每个观察包含 before/diagnostic/restored 三份真实 depth/visibility 回读，并核对 cut、page mappings、
early/late bin list、分类计数及生产绑定。完整验收还与独立无插桩进程比较这些身份及输出。
诊断进程没有 timing frame 文件，所有 trace 产物 `performanceEligible=false`；没有性能收益结论。

## 收集边界与 Streamline 兼容

这是 **独立工作负载进程的批处理 adapter**。instance 在 warmup、baseline、诊断和恢复期间复用；
整次观察在资源/GPU 排空并销毁 device/instance 后才封存。没有用 queue idle 或短暂安静窗口
冒充 VVL 消息回收完成，也没有宣称支持任意常驻编辑器上的连续在线 watch。
内部复用 DebugCore shader.watch/jobs 与现有 artifact 导出格式；agent 当前通过 runner 发起生产观察，
最终可用 metallicctl 的现有离线接口查询。

本机 bundled Streamline interposer 只枚举全局 instance extensions，会错误拒绝 layer 提供的
VK_EXT_layer_settings。关闭 Streamline 又会使原图的 DLSS-SR 不可用。P2 因此在进程内创建
独立 `layer-settings/vk_layer_settings.txt`，临时设置本进程 VK_LAYER_SETTINGS_PATH，并使用
VVL 支持的文件配置方式；析构后还原原值，未修改系统注册表、ACL 或全局设置。
默认 P0/P1 仍使用原来的 VkLayerSettingsCreateInfoEXT。官方配置方式见
[VK_EXT_layer_settings proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_layer_settings.html)。

文件启用 printf、保留 validation、订阅 INFO/warn/error、禁用 stdout 输出、消息过滤及重复限制，
buffer=64 KiB。必须先在同一生产 instance 上完成真实 echo；配置文件存在本身不算能力通过。
当前 SDK VVL 1.4.350 会使 renderer 的 KHR OMM 分支使用已有兼容回退；该运行事实保留在日志中。
无插桩对照用于检验这组环境下 WorkControl 输入和 depth/visibility 是否受影响。

## 验收结果

最终环境：RTX 5070 Ti / NVIDIA 616.92、Vulkan 1.4.351、SDK VVL 1.4.350、Slang 2026.18.2。

[Manifest.json](../build-release/shader-trace-p2-verified-20260927/Manifest.json) 固定源码、工具、二进制、
资产、配置与原始文件 SHA-256；[Result.json](../build-release/shader-trace-p2-verified-20260927/Result.json)
为九个串行独立进程的汇总。`verify` 使用当前 CLI 从原始包重算，结果同样通过。

| 真实 GPU 场景 | 独立进程数 | 结果 |
|---|---:|---|
| 无 Printf/validation 控制进程 | 1 | 原 WorkControl 绑定与输出通过 |
| early group 0 / lane 0 | 3 | Matched；13 个字段的位型和语义完全一致 |
| late group 0 / lane 0 | 3 | Matched；13 个字段的位型和语义完全一致 |
| early + triangleId=4294967295 | 1 | NoMatch；求值 1 次，匹配/输出 0 次 |
| early group 0 / lane 127 | 1 | SiteNotReached；求值/匹配/输出均 0 次 |

八个观察均 selectedScopeComplete=true、hostDropped=0、hostTruncated=0、orphan=0；
每个 before/diagnostic/restored 回读一致，恢复后的 production binding 相同。所有进程的
冻结输入、depth/visibility、production binding、相机和 graph 配置还与无插桩控制进程一致。
early/late 各自独立复现，未将两阶段数据混在一起：

| phase | recordIndex | triangleId | triangleCount | instanceFlags | aDepth raw bits |
|---|---:|---:|---:|---:|---|
| early | 0 | 0 | 61 | 3 | 0x3a02f7a2 |
| late | 3410 | 0 | 87 | 1 | 0x3ae5cb4c |

[篡改与 timing 隔离测试](../build-release/shader-trace-p2-integrity-20260927/Result.json)确认：
真实 artifact 独立副本被改动一个字节后，CLI 以 Artifact size/hash mismatch 拒绝；
原始 P2 capture 和仅将标签改成 normal-timing 的 validation capture 均被旧 timing 门槛拒绝。

回归全部通过：

- 44 个主机用例：26 DebugCore/ShaderTrace、6 callback、7 P0 evidence、5 P2 evidence；
  包括 phase/schema 混用、invocation/predicate 越界、实际录制身份冻结、fallback 不得空成功、
  跨进程字段位型/输入漂移、重复 session 和重叠进程。
- [P0 七项真实 GPU 正负回归](../build-release/shader-printf-p2-regression-20260927/Suite.json)。
- [P1 mapped/native 共十项真实 GPU 回归](../build-release/shader-trace-p2-p1-regression-20260927/Suite.json)，
  覆盖 matched、no-match、site-not-reached、missing-end、quota；预期不完整也按其应有状态验收。

构建与测试命令（先进入现有 x64 MSVC developer shell）：

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicCtl MetallicShaderPrintfProbe -j 12
cmake --build build-pass-stages-nrd --target MetallicDebugTests MetallicShaderPrintfTests -j 12
ctest --test-dir build-pass-stages-nrd -R '^Metallic(DebugTests|ShaderPrintfTests|ShaderPrintfEvidenceTests|ShaderTraceP2EvidenceTests)$' --output-on-failure
```

开发期 evidence 未改写：`shader-trace-p2-pilot-01` 记录 Streamline 拒绝 layer extension；
`pilot-02` 记录关闭 Streamline 后 DLSS-SR 不可用；`pilot-03` 首次通过。
`shader-trace-p2-acceptance-20260927` 是首轮九进程通过记录。收尾修正聚合 timeline 为 null/真实
values，以及小 wave fallback 的 UnsupportedPath 判定后，重新构建并完整重跑，最终以
`shader-trace-p2-verified-20260927` 为准。这些目录都在 build-release 下，不提交生成证据。

## 证据复核与限制

每个 `app/shader-trace/capture` 都可独立用 `metallicctl --capture <目录> --json shader verify`
重新解析 rawMessages。suite verify 进一步校验 artifact SHA-256、源码快照、编译宏/token、
真实绑定、原始图像回读、串行进程区间、session 独立性及跨进程字段逐位一致；不执行归档脚本。
SHA-256 表示文件完整性，不是来源签名。

目前只验收上述固定 case 的 mapped heap / compute WorkControl 单 invocation。
没有声称完整最终彩色图像/时序质量、所有 renderer 场景、native production heap、其他 shader stage、
任意表达式、多个同时 watch、driver device-lost 恢复或常驻编辑器持续收集都已验证。
主机测试覆盖取消/超时与资源持有契约；实际 GPU 取消中途、热重载竞争尚未独立故障注入验收。
P3 已增加实际 triangle decision 站点和有界故障/修复闭环，见 [P3 入口与验收](AgenticShaderPrintfP3.md)。
HZB 站点仍未扩展。
