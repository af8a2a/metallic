# #71 / #10 验收补跑记录

运行日期：2026-11-13；记录及 issue 状态复核完成：2026-11-14（Asia/Shanghai）。目标提交 `02eea37cf6e2acaa86a8f7ec0e533ed24623e379`。

## 范围与授权

用户要求先检查兼容 MSVC、GPU 输出目录问题，再补跑对应验收；仅验收通过后关闭 issue。随后明确选择使用干净克隆中的独立 MSVC 14.52 构建树，不修改旧缓存或 Windows 权限，GPU 输出路径保持原路径。

本轮开始时两个 issue 均为 OPEN；最终决定及状态见第 9 节。#10 的原文 v3/23、Build v3/45 属于历史状态，本次按已审查并发布的当前 workload v6/31、Build v7/60 验证；保留历史文档和 #9 依赖，不把本次 ABI 验证当作关闭 #9 或完成全部 VSM 特性的证据。

## 1. MSVC 诊断（已完成）

- 旧 NRD 构建树 `build-pass-stages-nrd` 锁定的 `E:/VisualStudio/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe` 已不存在。
- 现有主 VS 安装具有 MSVC 14.44.35207 和 14.52.37025；x64 developer shell 实际验证 cl 19.52.37025、Windows SDK 10.0.26100.0、CMake 4.2.3 和 Ninja 可用。
- VS 安装器 `C:/ProgramData/Microsoft/VisualStudio/Packages/_Instances/5379629c/plan.xml` 记录了 11/6 更新时 14.51 家族组件的卸载、14.52 家族组件的安装。旧配置日志证实 14.51 曾经可用。这是升级后旧绝对路径失效，不是缺少全部 MSVC。
- 已有 14.52 树 NRD=OFF；没有修改它们的 NRD/SDK 选项，也没有迁移旧 NRD 缓存或复用旧 object/lib。

## 2. GPU 输出目录诊断（已完成，未修改权限）

按原路径复现命令：

```powershell
.\build-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.virtual_shadow_map_profiling --rhi-validation --output-dir E:/metallic/build/vsm-abi-docs-20261113/Profiling
```

结果：1 failed、0 passed，192 ms；在创建 `Profiling/VirtualShadowMapProfiling` 时 Access is denied，未执行 GPU fixture。HTML listener 创建 `Profiling/reports/...` 同样失败。

新增只读证据：

- `icacls E:/metallic` 显示 Low Mandatory Level 的可继承完整性标签；`build-release`、其 tests 子目录及现有 RHI EXE 均继承此标签。
- 独立原生探针以 CREATE_SUSPENDED 创建该 EXE，只读取 token 后终止未执行的子进程；实测完整性 SID 为 `S-1-16-4096`（Low）。没有恢复线程、运行测试或修改 token。
- `E:/metallic/build` 没有此 Low 标签（默认 Medium）。可见普通权限允许当前已启用的 Authenticated Users 组 Modify。Medium 的 shell/git 能在其下创建本次干净克隆，但 Low RHI 进程因 mandatory no-write-up 不能创建原 GPU 输出目录。这解释了可见允许写入但 EXE 仍被拒绝的现象。
- Defender Controlled Folder Access=0，近 20 分钟无匹配的 1123/1124 事件；没有把该防护当作原因，也未修改防护配置。
- 使用兼容旧 RHI 树运行 `cmake --build build-release --target MetallicRHITests --parallel 2` 成功（仅 glob/deploy，无重新链接）；不能称为 current HEAD 的全新 EXE，且标签保持不变。

此前权限脚本的报告 `E:/metallic-acl-recovery-20261113/acl-report-c80fe5a6f5294060b8c741007b3d6323.jsonl` 对不存在的叶目录停止，fixed=0、granted=0、rollback not-needed。本轮未重跑脚本，未改 owner、普通权限或完整性标签，无恢复命令；没有改变原 GPU 输出位置。

## 3. #71 干净递归初始化（已通过）

```powershell
git clone --branch modernization_/vsm --single-branch --depth 1 https://github.com/af8a2a/metallic.git E:/metallic/build/issue71-clean-20261113
git -C E:/metallic/build/issue71-clean-20261113 submodule update --init --recursive --jobs 2
git -C E:/metallic/build/issue71-clean-20261113 submodule status --recursive
git -C E:/metallic/build/issue71-clean-20261113 status --short
```

全部 exit 0；远端新克隆 HEAD 正是目标提交。全部 25 个递归子模块状态为已初始化且与索引一致（空格前缀，无 `-` / `+` / `U`）；工作树状态为空；无 fatal。NRD=`36183520b006b65860bebc96d3dd7185e5edeb45`，microprofile=`a469709c55b973099c1600b78a503912150e2c69`，并正确递归检出了其 distorm 等子模块。原工作区 microprofile 的既有脏状态未动。

## 4. 独立配置与构建（已通过）

- source：`E:/metallic/build/issue71-clean-20261113`。
- build：其下 `build-nrd-msvc1452`；位于没有继承 Low 标签的干净克隆中。
- Ninja / Release / MSVC 14.52.37025 / x64 Windows SDK 10.0.26100.0。
- tests=ON、NVIDIA_SDKS=ON、NRD=ON、OpenUSD/Streamline/NTC/NvPerf=OFF、RHI_DIAGNOSTICS=ON、sccache=OFF。
- 复用已安装 `E:/metallic/External/slang`；FetchContent 仅复用旧 NRD 树已解包的 GDeflate/Zstd 源码目录，不复用 object/lib；自动子模块、Slang、Streamline 下载关闭。
- Configure 已 exit 0。新 cache 已核对 NRD=ON/NVIDIA_SDKS=ON，生成编译规则包含真实 `METALLIC_HAS_NRD=1`，并编译本仓拥有的 NRD 集成源。
- `cmake --build E:/metallic/build/issue71-clean-20261113/build-nrd-msvc1452 --target MetallicNRDTests MetallicRHITests --parallel 2` 已完成全部目标链接，**exit 0**。新树独立编译了运行时、NRD integration 及两个测试目标，没有复用旧 object/lib。输出中的 vendor MathLib C4834 等为 warning，未修改 vendor。
- 构建后两个新 EXE 的 `icacls` 均无旧的 Low 完整性标签。干净克隆 `git status --short` 仍为空。
- 额外 NRD CTest 已完成：CTest exit 8、0/1 passed；GTest XML 实际为 **16 tests、13 passed、3 failed、0 skipped**，75.59 s。失败为 `NRDShaders.EverySupportedPermutationUsesNativeHandles`（首次 REBLUR_ClassifyTiles compile 返回 source epoch notification advanced）、`NRDWorkingColor.RTXDIRelaxSceneChromaticity` 和 `NRDWorkingColor.PhysicalAtmosphereHDRAndAerial`（MeshletStreamPageLoader requires an initialized TaskSystem）。NRDGPU 的5项以及 NRDRayTracingGPU 的1项全部通过。XML 为新 build 根下 `nrd-gtest.xml`，CTest XML 为 `nrd-ctest.xml`；未过滤失败或重写测试。独立只读分析已完成，见第 7 节；不把 C++ 构建成功等同于全套 NRD 运行时通过。
- 新 RHI EXE SHA-256=`0AAE49CB7CF6BE9DC9E11FB391E52966C6603C836D41CAD48FDC15C6B7ED4961`。23:44:58+08 在原输出位置启动真实 profiling；23:48:34+08 随后串行启动原 manifest acceptance。两者最终结果见第 8 节，issue 决定见第 9 节。

## 5. #10 后续验收门槛

新二进制需先核对完整性标签及编译 revision；继续使用原输出 `E:/metallic/build/vsm-abi-docs-20261113/Profiling`（Acceptance 使用同一原输出根下对应子目录），不搬迁输出规避问题。manifest 为原始 `E:/metallic/build/vsm-arc-20261101/diagnostic-2-BCases.json`。只读独立审计已确认 76/76 个显式输入可解析/存在：原始 manifest+graph，以及干净克隆内 74 个资产。74 对资产逐项 SHA-256 与主仓相同（缺少0、不同0），无 reparse point；graph 两处 scene 相对路径都按新 PROJECT_SOURCE_DIR 解析，没有硬编码旧资源路径。manifest SHA-256=`76A70A3BCDF87CAF18331414300EC551ACB86ED4DE84069D9D35B858A622EE23`，graph=`B218CC966598E08FDAD6037140CA69701AE3863DFD1791BB3E826534851D00F6`。这只证明输入准备好，不是 runtime pass。

- 显式传入该 manifest 和 `--rhi-validation`，不启用 realtime/Streamline；保存并恢复原 manifest 环境变量。
- 两份 GTest XML 都应恰好一个目标 testcase，零 failure/error/skipped；exit 0 不能代替此检查。
- Profiling：passed=true；20 个 variant/phase 帧；4 个 profiling arms；Work/Both 每 face workload31/build60，Off/Timing 为空，影像与 public32-word 不变，validation 无误。
- Acceptance：status=passed、validation=true、单 SponzaSMRT2 case；64 帧/32 warmup，真实 workload v6/31 与 Build v7/60 的 per-face 字节长度/名称/计数检查；全部18个 captures及其PNG/raw/receipt存在且长度匹配；无不允许的 ray query/RTAS 分配、overflow或validation错误。
- 本 manifest 是 diagnostic、comparisonFrames=[]、BuildSubstages=false、receiverCoverage=false；不能宣称完成 cached/forced-redraw A/B、性能、receiver coverage 或 build substage attribution 验收。影像还需检查运动、时序和 resize 帧。
- 同步补跑配置 Python 的 parser unittest 与 CTest。历史 work16 归档保留，不删除历史记录以凑 grep 为空。

## 6. 干净克隆 parser 验证（已完成）

- 配置的 Python=`D:/Scoop/apps/python/current/python.exe`（3.12.4）。
- 原 issue 写法 `python -B -m unittest tests/perf/TestVirtualShadowMapGPUProfile.py` 首次失败：环境已安装的 `D:/Scoop/apps/python/current/Lib/site-packages/tests/__init__.py` 抢占了 `tests` 包名，导致 `No module named tests.perf`，尚未加载实际测试；此失败原样保留，不算通过。
- 不修改源码或 Python 安装，改用等价、按目录发现的 `python -B -m unittest discover -s tests/perf -p TestVirtualShadowMapGPUProfile.py`：**48 tests passed，0 failed，0.994 s**。
- `ctest --test-dir E:/metallic/build/issue71-clean-20261113/build-nrd-msvc1452 -R '^MetallicVirtualShadowMapGPUProfileTests$' --output-on-failure --no-tests=error`：**1/1 passed，1.12 s**。
- 这些是 parser/历史布局单测，不替代第 8 节的真实 GPU ABI 读回。

## 7. 补充 NRD 运行时失败分析（不算全套通过）

- `NRDShaders.EverySupportedPermutationUsesNativeHandles` 的首次 `REBLUR_ClassifyTiles` 编译因 source epoch notification advanced 拒收；fatal assertion 中断后续全 permutations 验证。同进程稍后对同模块的 native PSO/cache 验证以及其他 REBLUR 编译成功，不能替代该失败测试，也不足以判定首次失败的通知来源。watcher 可因名称、属性、security 等事件推进 epoch；日志只给根目录，不证明 shader 字节变化或具体写入者。未宣称为既有基线错误。
- 两个 WorkingColor 测试在 residency 初始化失败，尚未到 chromaticity/HDR 断言。[NRD target](../../tests/CMakeLists.txt#L80-L100) 只链接 `gtest_main`；[普通 RHI main](../../tests/rhi/main.cpp#L429-L441) 才初始化/关闭全局 TaskSystem；[page loader](../../Source/Runtime/Render/Streamer/MeshletStreamPageLoader.cpp#L193-L196) 明确要求该服务已初始化。这是独立 NRD harness 的服务生命周期缺口，不是已验证的颜色/HDR 回归。未发现可用 CLI/env 补上该服务的初始化。
- 本轮未修改 harness、shader、vendor、测试过滤器或检查条件，未重跑来覆盖原失败；13 pass / 3 fail / 0 skip 保留。

## 8. #10 实际 GPU 验收结果

两个测试串行运行；新 EXE 在原输出位置成功创建 JSON、PNG、raw 和 HTML，没有改 ACL/完整性标签/防护配置，也没有搬迁 GPU 输出。输出访问缺口已解决，但不等于测试逻辑全部通过。

### 8.1 Profiling：失败，保留 4/20 部分证据

```powershell
& E:/metallic/build/issue71-clean-20261113/build-nrd-msvc1452/tests/MetallicRHITests.exe --gtest_filter=RHIRendering.virtual_shadow_map_profiling --rhi-validation --output-dir E:/metallic/build/vsm-abi-docs-20261113/Profiling --gtest_output=xml:E:/metallic/build/vsm-abi-docs-20261113/Profiling/profiling.xml --gtest_color=no
```

- exit **1**；[XML](../../build/vsm-abi-docs-20261113/Profiling/profiling.xml)：1 testcase、1 failure、0 errors、0 skipped，run/completed；测试 137.575 s。错误为 `Missing or overflowed profiling timestamp frame`。
- [JSON](../../build/vsm-abi-docs-20261113/Profiling/VirtualShadowMapProfiling/VirtualShadowMapProfiling.json)：`passed=false`；仅 SingleHard Cold/Warm/Dirty/Overflow **4/20** 个计划帧；[HTML](../../build/vsm-abi-docs-20261113/Profiling/reports/17945846983523742/report.html)。每帧四个 arms；所有 arms 的 public statistics 为 1×32；Work/Both workload=1×31、buildDiagnostics=1×60；Off/Timing 两种诊断为空。
- 已完成帧通过了测试内部的 shadow/debug/depth/public32 不变、确定性 workload（排除允许变化的 markPageRequests）及 build diagnostics 一致检查。验证 recorder `captureFailed=false,count=0,messages=[]`；仅代表已执行窗口，不可扩展到未跑的 variants。
- 后续失败发生在 **LegacyPCF-Cold** render 后、[timestamp gate](../../tests/rhi/VirtualShadowMapSceneTests.cpp#L3151-L3158)；LegacyPCF arms 尚未序列化，其后 CompactSMRT/StreamSingleHard/TinySingleHard 未完成。现有 gate 合并了 completed 数量、timing availability 与 profilingOverflow，失败日志没有分别保存这三项，不能声称直接观测到了其中哪一项。
- 静态审计确认正常 LegacyPCF-Cold 的多 arm/多 face scopes 超出当前每 frame/queue query 预算：[分配公式](../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1778-L1810) `(N+1)*2+512*(P+1)`，该图 N=9、P=0，因此 532 queries=266 intervals。Timing/Both 各 8 个实际 faces，即使仅计每 face 至少 22 个必经 scopes，也需 704 queries，至少超出 172。这是预算不匹配的源码证据，不是失败 frame 的直接 query-used 实测；确切先丢弃哪个 scope、是否同时缺 timing，现有产物无法恢复。
- 本用例名为 Overflow 的阶段是刻意限制 `maxWorkItems` 的 VSM 工作溢出（已完成 SingleHard public stats[5]=241），不要与失败 gate 的 timestamp query overflow 混同。
- 这 4 帧的新增 work29/30、build55..59 均零，只证明新长度尾部零值读回，不证明这些生产路径被命中或此前非零尾部已正确 reset。未关闭 validation、减少必需覆盖或跳过 gate 来凑 pass。

### 8.2 原 manifest acceptance：通过，另行量化验证

```powershell
# 运行前保存原环境变量，finally 中恢复；不改变 manifest 内容。
$env:METALLIC_VSM_ACCEPTANCE_CASES='E:/metallic/build/vsm-arc-20261101/diagnostic-2-BCases.json'
& E:/metallic/build/issue71-clean-20261113/build-nrd-msvc1452/tests/MetallicRHITests.exe --gtest_filter=RHIRendering.virtual_shadow_map_acceptance --rhi-validation --output-dir E:/metallic/build/vsm-abi-docs-20261113/Acceptance --gtest_output=xml:E:/metallic/build/vsm-abi-docs-20261113/Acceptance/acceptance.xml --gtest_color=no
```

- exit **0**；[XML](../../build/vsm-abi-docs-20261113/Acceptance/acceptance.xml)：1 suite/1 testcase，failures=0、errors=0、disabled=0、suite skipped=0，run/completed（root 无 skipped 属性，没有补默认值）；测试 61.848 s。
- [JSON](../../build/vsm-abi-docs-20261113/Acceptance/VirtualShadowMapAcceptance.json)：top 和唯一 SponzaSMRT2 case `status=passed`；top `validation=true`，validationMessages `captureFailed=false,count=0,messages=[]`。case 无独立 validation 字段；其 synchronizationValidation、noRayQueryOrRTAS、noRayTracingAllocations 均 true。嵌入 manifest 与原文件语义相同，原 manifest/graph 实算 SHA-256 仍与第 5 节完全一致。
- 64 帧严格 0..63；warmup 精确为 0..31。所有帧 requested/actual/呈现 extent 一致，0..59 为 1920×1080，60..63 为 127×73，extentObservedThisFrame 全 true；ready 全 true。streaming loadFailures/allocationFailures/requestOverflows、budgetDroppedCount、public stats[5] 工作 overflow、[9] invalid coarse 全零。
- 64 帧每帧 12 faces（capacity2×6）；gpuWork enabled/version= true/6，31 名称；gpuBuildDiagnostics true/7，60 名称；名称与当前声明逐项一致，face ID 和 uint64 值范围正确，totals 逐 counter 等于 per-face 求和。每 face 公共 statistics 仍 32 words；totals/lodStatistics 也是32。
- **raw diagnostics 字节长度未在此 JSON 提供**，对象只有 enabled/faces/totals/version；不能把测试代码硬编码的版本或解析后的数组当作 raw byte-size receipt。capacity2 的期望 workload=2976B、build=5760B（2×6×248/480）只是 ABI 公式，未实测验证原始诊断缓冲区字节长度。
- work29 allocatorLRUEvictions、work30 allocatorLRUEvictionAgeFrames 在此场景均有真实非零值：每项64帧 totals min0/max429/sum1210；非零帧为2–7、40–47、60（frame60=429）。但 build55..59 每项64×12=768 个 face 值全零；不宣称 layered/static split 等新增尾部生产路径已命中。
- 18 capture frames 精确为16、32、33、34、35、40–47、52–55、63。18 主 PNG + 36 shadow-layer PNG 存在且 PNG header 尺寸正确；18 主 PNG 解码 RGBA 的 SHA-256 与 receipts 一致。72 raw bins 全部存在，合计 **916,772,246 bytes**；72/72 实际长度、SHA-256、width/height/rowStride/layerStride/layerCount 公式匹配，重复 depth/shadow/HDR hash 一致。18 HDR 文件经只读 float32 扫描共 **141,041,884 components，非有限0**，与 receipts 一致。
- 已实际查看 [frame16](../../build/vsm-abi-docs-20261113/Acceptance/SponzaSMRT2-16.png)、[frame32](../../build/vsm-abi-docs-20261113/Acceptance/SponzaSMRT2-32.png)、40–47 运动帧、[frame55](../../build/vsm-abi-docs-20261113/Acceptance/SponzaSMRT2-55.png)、[frame63 resize](../../build/vsm-abi-docs-20261113/Acceptance/SponzaSMRT2-63.png)：场景完整、非空白，所抽查运动序列未见明显整帧损坏；没有据此宣称 cached/forced-redraw 等价或所有时序行为无误。[完整 HTML](../../build/vsm-abi-docs-20261113/Acceptance/reports/17945849148040460/report.html)。
- 所有帧 RT caps（含 rayQuery/RTAS/RT pipeline/CLAS）均 false；RT/CLAS/CLASScratch allocations/count/peak 全0；denied allocations0。
- 范围仍是 diagnostic：performanceEligible=false、comparisons=[]、BuildSubstages=false、receiverCoverage=false。无 #9 A/B 缓存等价或性能通过结论；acceptance 通过不能覆盖 profiling 的失败。

## 9. Issue 决定与工作区边界

- [#71](<https://github.com/af8a2a/metallic/issues/71>)：原定五项映射/递归初始化/子模块对齐/NRD configure-build/依赖文档验收全部满足，已于 **2026-11-13 23:55:18+08 关闭为 completed**。[关闭说明](<https://github.com/af8a2a/metallic/issues/71#issuecomment-6858034788>)明确披露补充 NRD runtime 的13通过/3失败/0跳过，不冒称运行时全绿。
- [#10](<https://github.com/af8a2a/metallic/issues/10>)：**保持 OPEN**；parser 和原 manifest acceptance 通过，但 profiling 仅4/20帧且失败，另有 raw diagnostics byte-size 与新增 build 尾部路径未验证的覆盖边界。[结果已回填](<https://github.com/af8a2a/metallic/issues/10#issuecomment-6858153703>)，GitHub 状态再次核对为 OPEN。
- 下一步建议单独修复 timestamp 的可观测性和 query 预算，保留 multi-arm/multi-face 覆盖及原检查，然后完整复验5×4；独立 NRD harness 需补 TaskSystem 生命周期，并定位 source-epoch 通知触发者。本轮没有为此修改源码或弱化测试。
- runtime 后干净克隆 HEAD 仍为 `02eea37cf6e2acaa86a8f7ec0e533ed24623e379`；递归25子模块全部对齐，tracked diff/status 均空。主工作区只保留原有 microprofile 与两份未跟踪报告，以及本份新增本地验收记录；本轮没有新增 commit/push、没有清理无关文件。

