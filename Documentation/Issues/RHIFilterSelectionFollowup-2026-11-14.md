# RHI legacy filter 零选择退出码（2026-11-14）

范围：[测试会计 #62](https://github.com/af8a2a/metallic/issues/62) 的相邻假绿缺口；
基线 `d77dad1a55a03c764ab8c705df2dcfae5d9d4b44`。不把 zero selected 混成 passed 或 skipped。
此前 [Debug source 文档对账](DebugSourceDocumentationFollowup-2026-11-14.md) 发现旧测试名已退休，
而 legacy filter 默认可以不执行任何测试却成功退出。

## 边界与策略

[legacy 参数解析](../../tests/rhi/main.cpp) 把 `--filter NAME` 和 `--filter=NAME` 转成
`--gtest_filter=*NAME*`；匹配、disabled tests、分片均仍由 GoogleTest 负责，不复制其选择算法。
[原生默认](../../External/googletest/googletest/src/gtest.cc#L273-L278) 允许 zero selected；
[现成 strict 分支](../../External/googletest/googletest/src/gtest.cc#L6122-L6135) 可返回失败。

修复契约：只有使用 legacy filter 时，才在程序名之后、所有用户参数之前注入
`--gtest_fail_if_no_test_selected`，作为可覆盖的 CLI 默认策略：

- 未知 legacy filter（包括文档曾使用的 retired mixed-producer 名称）默认退出 1，提示 No tests ran。
- 用户显式 `--gtest_fail_if_no_test_selected=false` 无论放在 legacy filter 前后，都可覆盖默认。
- 注入的是 CLI 默认，会覆盖 `GTEST_FAIL_IF_NO_TEST_SELECTED=0` 环境初值；需要 opt-out 时用上述显式 CLI。
- 不使用 legacy filter 时，不改变 native `--gtest_filter` 或无 filter 的 GoogleTest 默认。
- [list 分支](../../External/googletest/googletest/src/gtest.cc#L6004-L6017) 保持原生契约：
  可列出匹配项，也可成功列出空集合；不把 discovery 当作执行回归。
- [testbench 分流](../../tests/rhi/harness/Runner.cpp#L796-L805) 位于 legacy 解析之前，不改 `--tb-*` 策略。

通过 GoogleTest 已有 CLI 接口实现，不改 vendor，不新增 selected-count 检查或自写 filter matcher。
不在 `RUN_ALL_TESTS` 之前读取尚未建立的 `test_to_run_count()`。

## CPU 回归设计

[CLI 回归脚本](../../tests/cmake/RHIFilterSelectionTests.cmake) 启动真实 RHI executable 子进程，
清理 `GTEST_*` 及 TESTBRIDGE/XML/warnings 回退环境，避免 repeat、filter、分片等污染。
失败断言同时检查正常数字退出码 1 与原因文本，不用 `WILL_FAIL` 将崩溃、缺 DLL 或超时反转为通过。

13 个 case 覆盖两种 legacy 未知 filter、retired 名称、legacy/native 全量 list、两种匹配 list、
空 list、native 默认空执行、显式 opt-out 前后顺序、native 显式 strict，以及 legacy 缺值错误。
正向 list 验证当前 composed-stream sentinel，且过滤后的列表不含无关 ColorSpace sentinel。

这些 case 只使用零选择或 list；GoogleTest 仅在
[确有选中执行时调用全局 SetUp](../../External/googletest/googletest/src/gtest.cc#L6065-L6072)，
RHI device 创建位于 [RHI environment](../../tests/rhi/main.cpp) 的 SetUp 中。
即使某个 TEST 自身只做 CPU 数学，选中执行它也会调用全局 RHI environment，所以这里不用
运行 ColorSpace TEST 冒充无 GPU 测试。未验证选中 rendering test 后的 GPU 行为或图像。

## 实际验证

### 构建与红绿

复用现有 Release / MSVC 14.52 / Ninja 配置，先构建未修改入口的 `MetallicRHITests`，再运行脚本；之后应用入口和 CTest 注册修改并重建同一目标。两次构建均成功，没有修改编译器、生成器或 SDK 选项。保留既有 MathLib C4834 警告；CMake 自动再生成的第三方依赖提示不作为本修复新增错误。

```powershell
cmake --build build-release --target MetallicRHITests -j 4
cmake -DRHI_EXECUTABLE=E:/metallic/build-release/tests/MetallicRHITests.exe -P tests/cmake/RHIFilterSelectionTests.cmake
ctest --test-dir build-release -R '^MetallicRHIFilterSelectionTests$' -V
```

- **fresh baseline 红**：3/13 case 失败，其余 10 项通过。失败恰为 legacy 空格写法、等号写法和已退休名称，均返回 0 并输出 `Running 0 tests from 0 test suites`。首次试跑曾因 GoogleTest 日志内的分号被 CMake list 拆分而误报 6/13；将 failure 名称与详情分开存储后，在仍未修复的同一二进制上重跑，得到上述准确的 3/13。不是六个产品失败。
- **修复后绿**：注册的 `MetallicRHIFilterSelectionTests` 为 1/1 passed，内部 13/13 case 通过，3.90 秒。
- **环境隔离复验**：同时注入冲突的 `GTEST_FILTER`、strict=1、repeat=0、total-shards=2、无效 shard-index=7，以及两个 TESTBRIDGE selection/fail-fast 变量后，再跑同一 CTest，内部仍为 13/13，通过耗时 3.87 秒。另外也设置了 `GTEST_LIST_TESTS=1`，但当前 vendor 不读取该环境变量，不把它算作有效 discovery 冲突证据。此为同一覆盖集合的复验，不累计为 26 个不同 case。
- 正向匹配只列出测试；13 个子进程均不得出现 legacy RHI 的 global environment set-up 标记。没有执行正向 RHI/GPU 测试。

### 额外 testbench 检查的限制

额外执行原有 `MetallicTestbench.Harness` CTest：**4 passed / 6 failed，CTest exit 8**。六项均在创建 `.tmp` 下的测试输出目录时抛出 Windows `Access is denied`；HTML report 目录也创建失败。将同一可执行文件的 `--tb-self-test` 放到独立构建输出工作目录复跑，仍为 **4 passed / 6 failed，exit 1**，没有把换目录后的尝试算作成功。

这些失败必须保留，不能声称 testbench Harness 回归通过或全量 CTest 全绿。未诊断具体 Windows 文件权限机制，未运行权限修复，也未把失败改记为产品 skip。此检查走已有 `runIfRequested` 的 self-test 早退分支，不进入本次修改的 legacy `parseArguments`；本补丁不修改 testbench 实现或其输出目录约定。

### 审查与静态核对

- 独立只读审查批准源码与契约：legacy-only 默认、两种 CLI opt-out 顺序、native/list/testbench 边界，以及 failure 计数修正；审查者未代替 root 运行构建或测试。
- 文档的 9 个本地链接/行号范围核对通过，2 个坏链接负例均被拒绝；1 个外部 URL 仅识别，未将它记为网络可用性验证。
- scoped diff 只包含入口、CPU CMake 脚本、CTest 注册和本文；不修改 vendor 或无关工作树内容。

上述证据只证明本次 CLI 选择策略和脚本隔离，不证明 GPU 渲染、validation、skip 基线或全量 CI 验收通过。

## Issue 验收余额

这是显式 legacy 回归命令的 zero-selection 默认策略，不是 skip 严格模式。
#62 的普通 RHI/editor require-all、known skip baseline/新增 skip 判退，以及
final-blit 已知失败闭环仍需各自证据；本轮不勾选这些验收项，不关闭 issue。
