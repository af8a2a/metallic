# Shader Printf P3：有界 Agent 调试闭环

日期：2026-09-27。当前实现为 MiniZorah WorkControl 的声明式、独立进程批处理实验。
新增实际 triangle-decision 站点；闭环顺序为健康基线 → 声明故障 → 证据定位 → 声明修复 → 无插桩复验。
这是首个受限故障/修复 recipe，不是任意 shader 的自动程序修复器。

## 入口

复用现有启用 Streamline 的 Release sample/metallicctl、Vulkan validation layer 和 MiniZorah 资产。
从仓库根目录执行；plan 与 evidence 输出必须全新。

```powershell
python -B -X utf8 Tools/Perf/ShaderDebugger.py plan --output build-release/triangle-plan.json
python -B -X utf8 Tools/Perf/ShaderDebugger.py run --plan build-release/triangle-plan.json --assets build-release/shader-trace-p2-verified-20260927/Assets.json --layer-path C:/VulkanSDK/1.4.350.0/Bin --output build-release/triangle-debug-new
python -B -X utf8 Tools/Perf/ShaderDebugger.py verify build-release/triangle-debug-new
```

plan 固定 workload ID、观察顺序、early group 0 / lane 0、故障和修复的源码路径、SHA-256、
唯一替换上下文。P3 只接受注册的 `triangle-edge-collapse`，拒绝额外命令、任意表达式、
未登记选择器和其他路径。故障将准备完成的 `b` 覆盖为 `a`，发生在 prepare 站点之后、
实际 triangle decision 之前。它同样作用于无 Printf 的生产变体。
修复必须逐字节还原计划钉住的源码，不根据“日志少了”或时间变快自行接受。

单独观察新站点可使用现有 runner：

```powershell
python -B -X utf8 Tools/Perf/ShaderTrace.py run --site stream.triangle-decision --phase early --assets build-release/shader-trace-p2-verified-20260927/Assets.json --layer-path C:/VulkanSDK/1.4.350.0/Bin --output build-release/triangle-watch-new
python -B -X utf8 Tools/Perf/ShaderTrace.py verify build-release/triangle-watch-new
```

原 `stream.after-triangle-prepare` 仍为默认站点；新增站点注册 early/late 两个 schema。
每进程仍只有一个 dispatch、一个 invocation、一个 watch。未指定 triangleId 谓词时输出
该 invocation 实际求值记录；沿用 P1 BEGIN/DATA/END、位型解码、配额和覆盖完整性。

## 实际决策证据

`HybridRasterTriangle.slang` 的 `hybridRasterPreparedTriangleDecision` 在原执行分支中赋值
`HybridTriangleDecision`。原 `hybridRasterPreparedTriangle` 通过丢弃此返回值的包装保留调用接口。
原坐标取整、方向交换、像素覆盖和深度算式不变，不新增资源读取。
诊断 shader 在调用返回后打印该次实际决定，不在 CPU 或另一个 shader 中重新推导分支。

| reason | 含义 | 已求值字段 |
|---:|---|---|
| 0 | SetupAccepted | area、bounds、plane；不保证任何像素最终贡献 |
| 1 | DegenerateArea | area |
| 2 | Backface | area |
| 3 | EmptyBounds | area、bounds |
| 4 | DegenerateDepthPlane | area、bounds、plane |

12 个 DATA 字段：recordIndex、triangleId、instanceFlags、signedArea、doubleSided、reason、
lowerX/Y、upperX/Y、determinant、evaluatedStages。`evaluatedStages` 的位 1/2/4 分别表示
area/bounds/plane 已计算。未求值字段虽初始化为零，分析器不得当成已算出的零。
诊断编译宏 `TRACE_SITE_ID`、site ID/schema、源码依赖 hash、真实生产绑定和 token 必须一致。

## 闭环门禁

固定 12 个独立进程，串行执行，不自动重试、挑选成功样本或降低门槛：

| arm | 无插桩运行 | prepare 观察 | decision 观察 |
|---|---:|---:|---:|
| healthy | 3 | 1 | 1 |
| fault | 1 | 1 | 1 |
| repaired | 3 | 0 | 1 |

沿用 P2 的真实图、1797×660 输出 / 1198×440 内部尺寸、15 秒 warmup、32 帧 settle、
冻结流送与每 checkpoint 的四帧偏移相机 priming；正常进程采集 8 帧 timing。
每个诊断进程必须保持 before/diagnostic/restored 的真实 depth/visibility 和输入一致。
每个 arm 的所有进程也必须有相同冻结工作负载、图像及绑定。

定位要求健康/故障的 prepare 全部字段逐位一致，recordIndex/triangleId/flags 一致，
实际 decision 的非零面积变成零，退出原因成为 DegenerateArea。
结论仅为**这两个有序观察点中，所选三角形最早有证据的分歧**，不是 GPU 全局事件顺序。
故障可能改变 early 深度继而影响 late HZB/bin，所以跨故障 arm 不强制后续分类列表相同；
相机、图、内部尺寸及被选中的完整准备值仍必须相同。每个 arm 内的完整冻结身份不能放宽。

修复验收同时要求无插桩故障确实改变输出、三次无插桩修复输出和绑定精确等于 healthy、
正常运行 residency 一致，以及 repaired decision 有完整且非空的记录并还原全部字段。
“日志消失”、只修诊断宏、缺独立正常进程均会失败。
选中三角形的分支见证与全局输出退化分别证明；不能仅凭它宣称某个变化像素的因果归属。

性能筛查仅使用 healthy/repaired 各三个独立正常进程的 software pass 总时长与 RenderGraph
GPU envelope 中位数；A/A spread 门限 10%，独立进程配对 95% t 区间，最大回退 2%。
顺序运行并非 ABBA，也未收集 GPU 竞争覆盖；结果只作保守筛查，不是优化接受。
`candidateAccepted=false`、`performanceEligible=false` 始终成立，M3 仍需其原来的
ABBA、竞争环境和独立确认门禁。`shaderTraceRequested=true` 也会被 WorkloadCase 正常计时资格拒绝。

## 源码事务和离线复核

共享 `build/shader-experiment.lock`。进程运行期间不改源码；只在进程结束后的 arm 边界替换
声明的一个 shader。事务写 journal 后才修改，finally 还原原始字节，成功也不保留故障。
如果检测到第三方编辑，拒绝覆盖，保留基线和锁供人工处理。

中断后可用 `ShaderDebugger.py recover <目录>` 恢复匹配 journal 的已知源码版本；
运行中的拥有者、仍存活的 Metallic renderer/compiler（包括控制器退出后的孤儿进程）
或冲突版本会被拒绝；不自动终止其他进程。恢复操作不将未完成实验改成 verified。

每个 arm 保留源码快照、原始日志、配置、实际回读、raw trace 包及进程起止时间。
根 Manifest 记录工具/源码/二进制/资产身份，Plan、Baseline、Candidate、Transaction、Diagnosis
和 Result 可逐项复核。`verify` 校验完整文件清单与 SHA-256，并调用当前 metallicctl
重新解码原始记录、校验实际编译依赖、重算正确性与性能筛查，不执行归档脚本。
Hash 是完整性校验，不是来源签名。大资产原件不复制，复用既有内容 hash 后逐进程检查大小/mtime。

## 本机验收

[完整 GPU Result](../build-release/shader-debug-p3-acceptance-01/Result.json) 与
[Manifest](../build-release/shader-debug-p3-acceptance-01/Manifest.json)：12 个独立进程全部通过
正确性门禁，事务已恢复原始源码。repaired 的三次无插桩输出、生产绑定、residency 和一次
非空 decision 观察全部恢复到 healthy。环境为 RTX 5070 Ti / NVIDIA 616.92、SDK VVL
1.4.350、Slang 2026.18.2；兼容条件继承 P2。

性能筛查为 **inconclusive**：healthy/repaired 的 softwareTotalMs 跨进程 spread 为
2.86% / 2.03%，但 RenderGraph GPU envelope spread 为 24.76% / 91.60%，超过 10% 门限。
softwareTotalMs 的相对收益 95% 区间为 [−2.63%, +2.31%]，也跨越 −2% 回退门槛。
没有丢弃样本、重试挑选或把波动解释成加速；不接受性能候选。

本轮已经确认：健康输出逐位等于 P2 归档的 depth/visibility；新旧完整 snapshot 只有
production SPIR-V 绑定身份变化。无插桩故障改变 527,120 个像素中的 101,193 个
（depth、visibility 均为此数）。early recordIndex=0 / triangleId=0 / instanceFlags=3：

| 数据 | healthy | fault |
|---|---|---|
| prepare 13 字段 | 与 fault 逐位相同 | 与 healthy 逐位相同 |
| signedArea | −923 | 0 |
| reason | 3 / EmptyBounds | 1 / DegenerateArea |
| evaluatedStages | 3 / area+bounds | 1 / area |
| 像素 bounds | lower=(615,235)，upper=(614,234) | 未求值 |

该健康三角形本来就在 EmptyBounds 退出，因此这里证明的是顶点覆盖导致的真实分支分歧，
不能将 101,193 个像素变化归因于这一个 triangle。全局故障影响多个 WorkControl 三角形。


[离线复核](../build-release/shader-debug-p3-offline-verify.json) 使用当前 CLI 重解码，结论与
GPU 实时 Result 完全一致；工作区 shader 字节等于归档 Baseline，故障未遗留。
[既有 P2 档案兼容性复核](../build-release/shader-debug-p3-old-p2-verify.json) 也通过；旧站点
未记录 TRACE_SITE_ID 宏的历史包仍可按其原始 site ID=2 复核。

[P2 九进程 GPU 回归](../build-release/shader-trace-p3-p2-regression-20260927/Result.json)通过：
无插桩控制、early/late 各三次、NoMatch 和 SiteNotReached 均满足原门禁，离线重算一致。
[新 decision 站点 late 实测](../build-release/shader-trace-p3-decision-late-20260927/Result.json)及
[离线复核](../build-release/shader-trace-p3-late-offline-verify.json)也通过。最终固定运行共 22 个进程
（12 个闭环、9 个 P2 回归、1 个新站点 late），另有前述 early pilot。
late 实测 recordIndex=3410 / triangleId=0 / flags=1，signedArea=366706、doubleSided=0，
reason=2（Backface）、evaluatedStages=1，bounds/plane 均未求值。本轮目标记录覆盖
DegenerateArea、Backface、EmptyBounds；SetupAccepted 与 DegenerateDepthPlane 尚未做
独立目标记录的 GPU 分支验收。
[收尾检查汇总](../build-release/shader-debug-p3-checks-20260927/Result.json)保存输出对照、
像素差异、源码恢复和实时/离线 verdict 一致性。

CPU 回归：7 组 CTest 共 90 个用例通过（DebugCore/Trace 26、callback 6、P0 evidence 7、
P3 13、P2 evidence 5、M3 experiment 17、workload 16）。另外 `tests/perf` 目录的 94 个
Python 用例通过，其中上述 M3/workload 的 33 个有重叠；本轮合计 151 个不同主机用例。
新增负向覆盖日志消失、Printf-only 修复、正常输出不变却声称复现、较早分歧、错误 triangle、
未求值字段、计时插桩、噪声/回退、旧 session/重叠进程、计划越界、恢复冲突、孤儿 renderer
和未完成/篡改 evidence。CPU 测试不能代替上述 12 次 GPU 验收。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicCtl -j 12
cmake --build build-pass-stages-nrd --target MetallicDebugTests MetallicShaderPrintfTests -j 12
ctest --test-dir build-pass-stages-nrd -R '^Metallic(DebugTests|ShaderPrintfTests|ShaderPrintfEvidenceTests|ShaderTraceP2EvidenceTests|ShaderDebuggerP3Tests|WorkloadCaseTests|ExperimentRunnerTests)$' --output-on-failure
python -B -X utf8 -m unittest discover -s tests/perf -p 'Test*.py'
```

开发期原始证据保留：`shader-trace-p3-decision-pilot-01` 属于 JSON/string_view 比较编译
失败后误启动旧 executable 的无效实验，已停止本任务拥有的进程，未当成验收；修正后构建
成功，`shader-trace-p3-decision-pilot-02` 首次通过新站点。完整闭环以
`shader-debug-p3-acceptance-01` 为准。生成证据全部留在 build-release，不提交源码库。

## 范围

保留 P2 单进程 instance 销毁后封存的收集边界、mapped heap、compute WorkControl。
未扩展常驻编辑器并发 watch、HZB 站点、任意表达式或其他 stage。通过 depth/visibility
不代表最终彩色画面、所有场景、长期时序质量或全 renderer 性能都已验证。
