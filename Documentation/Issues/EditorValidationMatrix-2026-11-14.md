# #63 编辑器 skip 与集成验证矩阵（2026-11-14）

关联 [#63](https://github.com/af8a2a/metallic/issues/63)。这是**源码清单，不是本轮运行结果**：只读枚举了编辑器 `GTEST_SKIP`、前置守卫、fixture、生产平台限制及 CTest 注册。没有运行编辑器、改变现有构建配置/ACL、启动 Nsight host/replay 或搬动 GPU 产物；没有新增跨平台实现。下文“替身候选”全部**尚未实现、尚未验证**，不是新增抽象层的设计批准。

CPU preset 的构建/回归另见[本轮记录](CIPresetFollowup-2026-11-14.md)。即使 CPU lane 全绿，也不完成本矩阵的编辑器、非 Windows 或 Nsight 验收，更不能据此关闭 #63。

## 记账边界

分别记录：①平台不注册，②生成 fixture/catalog 缺失不注册，③已注册但 preset 未选，④内部 GoogleTest runtime skip，⑤外层 CTest regex capability skip，⑥真实运行通过。前五种均不计作第六种；配置成功、编译成功、列举成功也不是 runtime 通过。

- [编辑器测试目录](../../tests/editor/) 共 **18 个 `GTEST_SKIP` 调用点 / 6 个文件 / 23 个不同用例**。13 个非 Windows 分支调用点影响 19 个用例；1 个额外 Windows symlink 失败调用点与其中一个用例重叠；另 4 个调用点是非平台配置/capability 条件。
- 这 18 个站点**没有**以缺少场景 asset 或 GPU/device capability 为条件的 skip。OS 目录 symlink 能力不是 GPU 能力；后文其他 editor smoke 的注册/skip 另计。
- 全部站点编入 [MetallicShaderRequestsTests](../../tests/CMakeLists.txt#L48-L74)，单个 CTest 聚合运行，labels=`editor;shader`、timeout=300s。其注册不受 Windows 守卫；61–71 行仅为 Windows Slang DLL post-build copy。无过滤 CTest 会选择该聚合测试，随后才可能发生内部 skip。
- 当前 [metallic-ci preset](../../CMakePresets.json) 不选择 `editor;shader`，其 build target 列表也不包含该 executable。本轮未提交的 tools/构建闭包增补不改变这点。`editor` label 后置获得 [MetallicGPU 锁](../../tests/CMakeLists.txt#L918-L925)，但锁本身不证明这些 compiler/文件系统测试执行 GPU 工作。
- 聚合 CTest exit 0 不代表 23 项均执行通过。运行时应另留 GoogleTest XML 的逐项 skipped/reason，与外层 CTest JUnit 分开统计，遵循[结果与 skip 策略](../RhiTestbench.md#普通-rhi--编辑器-required-run)。本清单不扩大 [KnownSkips.json](../../tests/KnownSkips.json) 的允许范围；已有 MiniZorah opt-in profile 不能豁免这里的 skip。

## 非 Windows 限制不是仅由测试文案推断

生产代码明确区分 fresh 与可复用 snapshot：

- [validateNotifications](../../Source/Runtime/Render/Core/SlangSnapshotFileSystem.h#L307-L321) 在非 Windows 返回 false，并给出 Windows notifications unavailable 原因；[watch](../../Source/Runtime/Render/Core/SlangSnapshotFileSystem.h#L672-L683) 在非 Windows 抛出 Windows-required 错误。
- [CompilerProducerSeal::create](../../Source/Runtime/Render/Core/SlangCompileContext.inc#L44-L80) 在非 Windows 抛出 source snapshot reuse 需要 Windows producer paths/notifications，而不是创建一个默认有效的 seal。
- [warmup batch](../../Source/Runtime/Render/Core/SlangWarmupBatch.inc#L99-L105) 在非 Windows 走 fresh batch 分支。该源码分支的存在不等于本轮实测，更不等于 Windows live epoch/cache reuse 的等价验证。这里不泛化宣称所有编译/cache 功能均不支持非 Windows。

### ShaderWarmupBatch：12 个站点 / 11 个用例

[WarmupDirectory fixture](../../tests/editor/ShaderWarmupBatchTests.cpp#L35-L55) 自建 source/cache，并清理 tracking；不是缺外部场景资产。下表除 900 行外，均在 `#ifndef _WIN32` 分支 skip，Windows 编译实际主体。用例名均属于 `ShaderWarmupBatch` suite。

| skip 位置 / 用例 | 保留原因、平台限制 | 替身或后续验证候选（未实现） |
| --- | --- | --- |
| [193](../../tests/editor/ShaderWarmupBatchTests.cpp#L190-L236) `RejectedOriginalAliasDoesNotRegisterRepresentativeIdentity` | Windows fixture 用 `_putenv_s` 在 preparing 回调中改变优化策略，验证 original alias 被拒绝且不注册 provisional identity；环境变量本身不是 Windows 独占能力。 | 可移植环境 RAII + 确定性 identity 改变合约；真实 epoch 集成仍单列。 |
| [242](../../tests/editor/ShaderWarmupBatchTests.cpp#L239-L250) `RealSlangColdAllHitMixedPrivateHeaderShadowAndOrdinaryCacheFailures` | 真实 Slang cold/all-hit/mixed、private header、shadow、缓存失败依赖 live epoch qualification。 | 可控 epoch/proof 的 cache-transition 合约；另补真实非 Windows backend，fresh fallback 不替代 reuse 断言。 |
| [400](../../tests/editor/ShaderWarmupBatchTests.cpp#L397-L417) `IndependentModulesKeepPrivateDependenciesThroughMixedPublication` | jobs=1/2、shareSessions=false/true 下 private dependency 隔离依赖当前 epoch 通知。 | 独立 invocation proof 的 provenance 合约；另跑真实 Slang/通知组合。 |
| [520](../../tests/editor/ShaderWarmupBatchTests.cpp#L517-L605) `IssuedTokensAcceptedHitUnionAndActualFilePublicationGates` | Windows snapshot 的 issued token、accepted-read union、事务发布、损坏 envelope/字节、编辑后恢复仍失效。 | 合成 token/envelope/发布断言；可控事件覆盖 edit-and-restore，真实 OS 通知不由 mock 替代。 |
| [654](../../tests/editor/ShaderWarmupBatchTests.cpp#L651-L708) `TransactionPreservesDistinctItemProofsWithoutBatchUnion` | A/B 独立 dependency proof 来自真实 snapshot；不得被 batch union 取代。 | 合成不同 item proof 的 transaction/envelope 合约；另测实际 snapshot provenance。 |
| [714](../../tests/editor/ShaderWarmupBatchTests.cpp#L711-L776) `NamespaceEnumerationPreservesHiddenUnicodeAndEmptyDirectories` | Windows HIDDEN/SYSTEM 属性与当前 snapshot 后端；同时覆盖 Unicode、空目录、递归枚举。 | 拆出可移植枚举；Windows 属性继续专测。POSIX dotfile 不等价于 HIDDEN/SYSTEM。 |
| [785](../../tests/editor/ShaderWarmupBatchTests.cpp#L782-L828) `CacheHitValidationReadsOnlyInvocationDependencies` | Windows writable mapping 修改字节但保留有效通知，以隔离 invocation read set 与 owner union。 | 可控字节/通知源分别改变 cached、delivered、unused 依赖；POSIX mmap 需另实测事件语义。 |
| [834](../../tests/editor/ShaderWarmupBatchTests.cpp#L831-L856) `CacheHitValidationChecksOriginalDependencyBytes` | original 与 frozen 分离，mapping 改 original 后必须因 original bytes 拒绝 hit。 | 合成 original/frozen provenance 合约；真实映射行为单列。 |
| [862](../../tests/editor/ShaderWarmupBatchTests.cpp#L859-L878) `CacheHitValidationRejectsNamespaceShadow` | negative lookup 后新建 header，当前 snapshot fixture 验证 namespace shadow 使 hit 失效。 | 注入 namespace-generation/negative-lookup 转移；真实目录事件另测。 |
| [884、900](../../tests/editor/ShaderWarmupBatchTests.cpp#L881-L904) `CacheHitValidationRejectsReparseReplacement` | 884 为非 Windows skip；900 为 Windows `CreateSymbolicLinkW` 失败，输出实际 `GetLastError()`。不能先验归咎 ACL/Developer Mode。 | 保留 Windows reparse/symlink 集成；另拟 POSIX replacement/通用目录身份改变合约，不宣称覆盖全部 reparse 类型；不为本清单改权限。 |
| [910](../../tests/editor/ShaderWarmupBatchTests.cpp#L907-L970) `GenericSnapshotsKeepPerFileProofQualification` | GenericSnapshot 无 notification API，但初始化 proof 仍来自真实 Windows snapshot；验证逐 envelope qualification 和后项失败时不注册。 | 用合成合法 proof 初始化 GenericSnapshot，隔离纯事务检查；真实 snapshot 初始化仍单列。 |

### CompilerProducerSealTest：1 个站点 / 1 个用例

[103 行](../../tests/editor/CompilerProducerSealTests.cpp#L55-L104) 的非 Windows 分支跳过 `CompilerProducerSealTest.CacheHitRetainsHostAndSDKWriteDeleteExclusion`。Windows 主体真实编译后命中 cache，对 compiler host 和三个 Slang SDK DLL 分别尝试 `GENERIC_WRITE`/`DELETE` open，要求 sharing violation。[helper](../../tests/editor/CompilerProducerSealTests.cpp#L37-L51) **只尝试打开，不写入或删除 producer**。

这是 Windows sharing/pinning 合约，不是 asset/GPU 条件。[生产 pin](../../Source/Runtime/Render/Core/SlangCompileContext.inc#L159-L170) 保留 read-only/share-read handle。未来可另测 producer 生命周期/identity/失效 seal 的纯合约；非 Windows 要先明确不可变 producer 方案，不能将 POSIX advisory lock 直接称为 share-deny 等价物，也不能把这个 skip 描述为仅换 fixture API 即可移植。

### SlangSourcePreparationTest：1 个 fixture 站点 / 7 个用例

[SetUp:74](../../tests/editor/SlangSourcePreparationTests.cpp#L68-L98) 在非 Windows 统一 skip，以下七个 body 都不执行。Windows 自建 source/cache、关闭 debug mode并恢复；虽有非 Windows `ofstream` [写入 helper](../../tests/editor/SlangSourcePreparationTests.cpp#L29-L51)，仅移植 helper 不能解除 epoch/producer seal 限制。

以下保留原因均是当前 live source epoch/producer 平台限制，不是资产或 GPU 缺失。替身候选只覆盖纯合约，不取代真实 Windows 集成。

| 用例（suite 均为 SlangSourcePreparationTest，共享 skip:74） | 当前断言 / 未来替身候选（未实现） |
| --- | --- |
| [SearchOrderKeepsRequestIdentityAndOutputWhileSharingSourceImage](../../tests/editor/SlangSourcePreparationTests.cpp#L155-L195) | 搜索顺序改变 identity/输出而可共享 source image；候选：合成 ordered request 与 unordered capture-root 共享边界。 |
| [PreparationDoesNotWriteCacheAndSharesOnlyEquivalentCaptureRoots](../../tests/editor/SlangSourcePreparationTests.cpp#L198-L238) | preparation 不写 cache、covered roots 复用而 expanded roots 重捕获；候选：可控 capture/cache backend 的无写入和分组合约。 |
| [ChangedDependencyInvalidatesPreparedImageEvenWithRestoredTimestamp](../../tests/editor/SlangSourcePreparationTests.cpp#L241-L265) | 同 timestamp 的新字节也必须 miss/re-capture/改变 SPIR-V；候选：同 metadata/不同 bytes 的依赖版本。 |
| [MissingSearchPathOutsideCapturedNamespaceRejectsReuse](../../tests/editor/SlangSourcePreparationTests.cpp#L268-L282) | 未捕获 ancestor 下的缺失搜索目录拒绝 reuse；候选：合成 namespace/ancestor proof。 |
| [NewShadowingHeaderInvalidatesPreparedNamespace](../../tests/editor/SlangSourcePreparationTests.cpp#L285-L311) | 新的高优先级 header 令 miss/re-capture/输出改变；候选：negative lookup 到 present/shadow 的转移。 |
| [ConcurrentPreparationAndOrdinaryAcquisitionCaptureOnce](../../tests/editor/SlangSourcePreparationTests.cpp#L314-L367) | 8 worker 只 capture 一次、reuse 七次；候选：可控 provider/barrier 的 single-flight 合约，不代表 OS notification 验证。 |
| [PreparedImageSurvivesWorkerExitAndStillDetectsDependencyChanges](../../tests/editor/SlangSourcePreparationTests.cpp#L370-L404) | 实际 thread 退出后仍复用、随后检测编辑；候选：pool 所有权/worker lifetime 合约；真实 overlapped-directory-I/O 生存期仍需独立集成。 |

### 其他 4 个非平台 skip

这些仍全部属于上述 `MetallicShaderRequestsTests` 聚合 CTest，而非新增 lane。

| 用例 / skip 位置 | 精确条件、原因 | 后续验证候选（未实现） |
| --- | --- | --- |
| `NativeShaderWarmupRequests.FoldedFamiliesHaveOnlyTheirClosedRuntimeTuples` [289](../../tests/editor/NativeShaderWarmupRequestsTests.cpp#L287-L314) | working color space 非 ACEScg；配置条件，不是 GPU/RTXCR 资产缺失。 | ACEScg/Rec709 独立进程和独立预期；[色域选择](../../Source/Runtime/Render/Core/ColorSpace.h#L36-L46) 进程静态，不能进程内改环境伪造双覆盖。 |
| `ShaderPostprocessTest.SlangAppendedPassesPreserveSelectedO0AndEliminateResolvedO1Branches` [205](../../tests/editor/ShaderPostprocessTests.cpp#L202-L215) | SlangDeadBranches 或 DeadBranches backend unavailable。 | 真实 SPIRV-Tools enabled 配置跑 opcode/bytecode 断言；fake optimizer 只能补 orchestration，不替代优化结果。 |
| `ShaderPostprocessTest.UnavailableOptimizerDoesNotFallBackToSourceOrWriteCache` [307](../../tests/editor/ShaderPostprocessTests.cpp#L293-L307) | 两种 mode 都 available，没有 unavailable backend 负向样本；不是“本机缺工具”。 | 独立 SPIRV-Tools OFF 配置，验证失败/diagnostics/无 cache；不改变现有树配置。 |
| `ShaderRequestCompileTest.ReusedCompilerReportsOnlyCurrentInvocationTiming` [432](../../tests/editor/ShaderRequestsTests.cpp#L405-L432) | 八次禁用 disk cache 的编译均无 SDK elapsed-time counter。已运行的编译/输出断言不等于 timing-bound 断言已验证。 | 支持 counters 的 SDK 实跑；fake timing 只能补传递/每 invocation 重置合约。 |

## Windows 条件注册与其他 editor smoke

这是 [CTest 注册源码](../../tests/CMakeLists.txt) 的事实，不是当前生成目录的注册快照或运行结果。表内“默认”指满足门槛后的无过滤 CTest；当前 CPU preset 均不选 `editor` 作业。

| 注册范围 | 门槛 / labels / 状态区分 |
| --- | --- |
| [MetallicSlangBootstrap.partial](../../tests/CMakeLists.txt#L10-L15) | Windows，`build`；CPU preset 选择。配置/引导检查不是 editor GPU 验证。 |
| [MetallicNsightCaptureLauncherTests](../../tests/CMakeLists.txt#L109-L124) | Windows，`editor;profiling`；专用 probe 假扮 ngfx UI/写假 capture，**不是实际 Nsight GPU capture**，见 [fixture](../../tests/editor/NsightCaptureLauncherTests.cpp#L16-L45)。 |
| [MetallicLookDevPathTracePipelineCacheSmoke](../../tests/CMakeLists.txt#L157-L166) | Windows，`editor;shader;gpu`；[脚本](../../tests/editor/RunPathTracePipelineCacheSmoke.cmake#L43-L79) 对 internal-cache control 缺失或 feedback unknown 输出 `SKIP:`，CTest regex 分类 skipped。 |
| [DLSSNR / StreamlineReflex / NVIDIASDK / NsightGraphics Configuration](../../tests/CMakeLists.txt#L167-L192) | Windows，均为 `build` 加专项 label，CPU preset 选择；配置成功不等于 SDK/GPU runtime 成功。 |
| [MetallicEditorReflex.off/on/boost](../../tests/CMakeLists.txt#L193-L206) | Windows + NV low latency，`editor;reflex`；[child 77](../../tests/editor/RunReflexSmoke.cmake#L19-L27) 被脚本转换为 skip 文案并由 CTest regex 分类。 |
| [MetallicEditorDLSSCameraSmoke](../../tests/CMakeLists.txt#L207-L218) | Windows + Streamline enabled，`editor;dlss`；无 skip property，脚本要求成功标记。当前 CI configure 禁用 Streamline。 |
| [MetallicLookDevMaterialInspectorSmoke / MaterialGraphSmoke](../../tests/CMakeLists.txt#L219-L235) | Windows；前者 `editor`，后者**未设置 LABELS**、RUN_SERIAL=TRUE/270s，不能以 `-L editor` 声称选全。源 asset 缺失为脚本错误，不是 skip。该标签缺口未在本文修复。 |
| [MetallicLookDevPainterSceneSwitchSmoke / StudioSceneSwitchSmoke](../../tests/CMakeLists.txt#L236-L258) | Windows + 各自本地 Catalog 存在才注册，`editor;painter` / `editor;studio`；缺生成 catalog 是不注册，不是 runtime skip。 |
| [MetallicLookDevSceneSwitchSmoke / VBufferSmoke / SliderSmoke / MetallicEditorFinalBlitSmoke / MultiViewportSmoke](../../tests/CMakeLists.txt#L259-L310) | Windows，`editor`；无额外注册资产存在守卫、无 skip property；注册本身不证明 runtime 前置条件满足。 |
| [MetallicShaderWarmupStartupTests](../../tests/CMakeLists.txt#L28-L45)、[MetallicNsightLaunchOptionsTests](../../tests/CMakeLists.txt#L103-L107) | 注册不受 Windows 守卫；分别 `editor;shader` / `editor;profiling`，当前 CPU preset 均不选。不能泛化为全部 editor 测试只在 Windows 注册。 |
| [MetallicResourceInspectorSmoke](../../tests/CMakeLists.txt#L741-L746) | 同样不受该 Windows 大段守卫，`editor;rhi;resource-inspector`；当前 CPU preset 不选。 |
| [Property / Shrinker / Trace testbench](../../tests/CMakeLists.txt#L927-L943) | Windows，Trace 另要求 diagnostics；`testbench`，CPU preset 不选。不是上述 editor GTEST_SKIP 的来源。 |

## Nsight opt-in、外部进程与 timeout

[注册](../../tests/CMakeLists.txt#L126-L144) 要求 Windows、capture SDK capability 和 `METALLIC_NSIGHT_INTEGRATION_TESTS=ON`；最后一项默认 OFF。检测到 SDK enabled 不等于集成选项 ON。loop 含 disabled/startup/export/collection/invalid-metrics，startup 另要求 Streamline，因此满足其余门槛时是**四或五项**，并非无条件五项。未注册不是 runtime skip。

全部 labels=`editor;profiling;gpu`，带 MetallicEditor 锁，后置循环另加 MetallicGPU。CPU preset 不选。执行均通过 [RunNsightGPUTraceStartup.cmake](../../tests/editor/RunNsightGPUTraceStartup.cmake) 启动 LookDev：

| case | 脚本 target-process 预算 / CTest 外层预算 | 实际验收要求，本文未运行 |
| --- | --- | --- |
| disabled | 180s / 200s | 普通启动呈现帧、明确 Nsight disabled、无 handshake/shutdown marker。 |
| startup | 180s / 200s | DLSS-RR、handshake、呈现帧、Streamline debug initialized/passed。 |
| export | 180s / 200s | device-ready → handshake → host launch → handshake-end 顺序、三份非空 gputrace 产物。 |
| collection | **600s / 620s** | graphics capture + replay collection、三份非空 capture 产物、不走 live-trace handshake。 |
| invalid-metrics | 75s / 200s | 特制无效 metric set，应用 exit 1 且有明确 startup diagnostic，不得挂起。 |

[STARTUP_TIMEOUT_SECONDS](../../tests/editor/RunNsightGPUTraceStartup.cmake#L52-L65) 可显式覆盖脚本完整进程预算，但当前注册未传它；不会修改 CTest 外层预算或应用 activation watchdog。静态读到 620s **不是观察到 collection 正确运行/超时终止的证据**。

实际外部进程由源码确认，不以 CPU script fixture 代替：

- [live trace](../../Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp#L295-L348)：安装目录下 ngfx host attach editor PID；general operation `--no-timeout` 与仍保留的 `--trace-timeout 120` 不同，launch 放入 kill-on-close Windows job。
- [graphics collection](../../Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp#L472-L488)：注入前启动独立、未受 capture 注入的 MetallicNsightReplay worker，使用 mailbox。
- [replay request](../../Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp#L831-L869)：ngfx host 启动 ngfx-replay、loop-count 3、trace-timeout 120，新建 `_Collected` 输出。应用 request deadline 180s 不能与脚本 600s/CTest 620s 混为一谈。

[历史 Nsight 调查](../NsightGPUTraceStartupInvestigation.md) 的特定日期、版本、hash、cache 和 capture 是历史证据，不是当前 HEAD rerun。尚需兼容的 integration-ON build（含 LookDev/replay 依赖）、记录 Streamline/startup 是否可用、真实 host/replay/driver/GPU 与原始日志、产物、退出码及 timeout 证据。本文不补写或虚构这些运行结果。
