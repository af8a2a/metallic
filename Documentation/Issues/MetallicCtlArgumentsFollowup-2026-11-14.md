# metallicctl ROI/pixel 参数边界修复（2026-11-14）

范围：[#66](https://github.com/af8a2a/metallic/issues/66) 工具入口的已有参数校验，
不是 `perf` 新协议实现或整票验收。基线为
`8d5bf53824b6b9573d58d1e57579c9d6db9473c5`。

## 已确认的问题与修复

[CLI 参数入口](../../Source/Tools/MetallicCtl.cpp#L294-L311) 原来直接用
`istringstream >> uint32_t` 读坐标，不验证流是否读尽：负数可能被转换为 unsigned，
最后一项的 numeric suffix、多余数字或垃圾项则被忽略。

修复将每个坐标作为完整 token 交给已有 `parseUnsigned(token, UINT32_MAX)`，要求
pixel 恰好两个分量、ROI 恰好四个；保留逗号/空白分隔、pixel 默认 1×1。
非法用户输入通过现有 `InvalidArgument` / exit 1 路径报告；不新增错误协议，
不搬迁 capture provider 对资源 extent、INT32 范围和非空尺寸的校验。
参见[入口说明](../DebugControlPlane.md#discover-capture-analyze-export)。

## 确定性回归

[测试脚本](../../tests/cmake/MetallicCtlArgumentsTests.cmake) 通过 CMake 调用实际 CLI：
`--json` 放首位、`--help` 放末尾，合法输入必须到达 help 并 exit 0，非法输入必须
exit 1 且输出 `status=error` / `error.code=InvalidArgument` JSON，stderr 均应为空。
这些路径不会调用 `run()`、named pipe、debug service、capture 文件或 GPU。

共 43 个案例：9 个合法 ROI/pixel，27 个非法 ROI/pixel，7 个既有 unsigned 参数边界对照。
覆盖负数/显式正号、uint32/uint64 溢出、少/多分量、numeric suffix、逗号/空白、
UINT32_MAX 合法边界，以及 PID/count/timeout 的既有数值规则。

```powershell
# 复用已配置的 build-release；在匹配的 x64 VS developer shell 运行。
cmake --build build-release --target MetallicCtl MetallicDebugTests -j 4
ctest --test-dir build-release -R '^Metallic(CtlArguments|Debug)Tests$' --output-on-failure
ctest --test-dir build-release -R '^MetallicCtlArgumentsTests$' -V
```

结果：

- 先从基线源码重建 CLI，再运行新 oracle：**17/43 红例**，均是本应拒绝的 ROI/pixel 被接受。
- 应用修复后重建成功：**43/43 CLI 案例通过**；对应外层 CTest 为 **1 passed**，不能将内外层相加。
- 扩展 CPU 回归 `MetallicDebugTests`：**26 passed / 1 failed / 0 skipped**。
  失败为 `DebugCapture.ScopedRelationsMissingDependenciesAndNonFiniteStatistics`，
  [旧 fixture](../../tests/debug/DebugCoreTests.cpp#L184-L192) 期待 `Resident`，而
  [当前 decoder](../../Source/Runtime/Debug/DebugAnalysis.cpp#L138-L145) 对 source 0 给 `Unknown`。
  因此两项一起执行的外层 CTest 是 **1 passed / 1 failed**，不是全绿。
- `git diff --exit-code HEAD -- Source/Runtime/Debug tests/debug` 通过；这些 Debug 源码与测试
  未在本次修改。Ninja target dependency 核对确认 `MetallicDebugTests.exe` 只依赖原有
  DebugCore / GoogleTest，不链接 CLI main，不能把此失败归功于或掩盖在本修复中。
- 独立只读审查 `d2-cli-boundary` 对实现、43-case oracle、CTest 注册和跟进记录结论为
  **APPROVE**，无剩余阻断；本地链接及锚点已复核，`git diff --check` 通过。
- 未运行 editor / GPU；没有新增 GPU passed/skipped/failed 计数。

## 验收余额

本修复不提供 `perf capabilities`、case manifest、collect/analyze、四态能力或 Python
证据链迁移，不勾选 #66 任一 feature 验收，也不关闭 #66。
DebugCapture fixture/decoder 的既有不一致保留为单独对账事项；本记录不改变其预期，
不通过过滤该用例声称完整 Debug suite 通过。
