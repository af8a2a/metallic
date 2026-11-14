# DebugCapture source fixture 对账（2026-11-14）

范围：共享 task-12，修复已退休 `Resident` 语义导致的 CPU fixture 失败。
本次只修改测试，不修改 decoder、producer、ABI、shader 或 configured build options；
不是 [#63](https://github.com/af8a2a/metallic/issues/63) 整票验收。

## 现行契约与历史依据

- [C++ producer 定义](../../Source/Runtime/Render/GPUDrivenRaster.h#L93-L121)
  只声明 `VisibleClusterSource::StreamPage = 1`；`VisibleClusterRecord` 仍为 16-byte。
- [Slang producer](../../Shaders/Modules/GPUDriven/GPUDrivenRasterCommon.slang#L13-L40)
  仅定义 source 1 的 StreamPage。compact record 的 group 保存 index+1，清零记录无效；
  unpack 时 group 非零才写 source 1，否则 flags 为 0。source 0 不能再解释成受支持的
  Resident producer 路径。
- [现行 decoder](../../Source/Runtime/Debug/DebugAnalysis.cpp#L138-L156)
  将 source 1 标为 `StreamPage`，其余为 `Unknown`。
  `recordValidity` 独立由 captured visibility pixel 是否引用该 slot 决定；本 fixture
  没有 visibility artifact，所以两个 raw-storage record 均应为 `Unverified`。
  不能仅凭 source 标签或可解析的 bytes 宣称 live record。
- 历史提交
  [`b91fecf61ee722b3e7d2435b124a56dc7ff4c90b`](https://github.com/af8a2a/metallic/commit/b91fecf61ee722b3e7d2435b124a56dc7ff4c90b)
  （2026-10-11）同时删除 C++ 的 `Resident = 0`、Slang 的
  `kVisibleClusterSourceResident = 0`，并将 decoder 的 source 0 文本改为 `Unknown`。
  `git blame -L 184,193 -- tests/debug/DebugCoreTests.cpp` 显示原 fixture 的
  records 和 `Resident` 断言仍来自 `7ddabc4c00`（2026-09-04），未随语义迁移。

## 最小测试变更

[fixture 断言](../../tests/debug/DebugCoreTests.cpp#L184-L203) 现在要求 source 0 为
`Unknown` + `Unverified`；保留既有 raw-storage 的 geometry `Captured` / meshlet
`MissingDependency` 关联结果断言，不将它们描述为有效的 Resident producer 证据。
source 1 继续覆盖 `StreamPage`、activeGroup `MissingDependency`、instance `OutOfRange`，
并显式要求 `Unverified`。原有 counter overflow 和 scoped/non-finite statistics 断言不变。
没有新增重复内防，也没有顺手修改 decoder 中既有的 source 0 关联分支。

## 红绿回归

修改前先执行单例，重现原 line 190 的 `Unknown != Resident`：
**0 passed / 1 failed / 0 skipped**。
随后在既有 x64 MSVC 14.52 / `build-release` 中执行完整选定 CPU targets：

```powershell
cmd.exe /d /s /c '""E:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.52 && cmake --build build-release --target MetallicCtl MetallicDebugTests -j 4 && ctest --test-dir build-release -R "^Metallic(CtlArguments|Debug)Tests$" --output-on-failure"'
```

基线 build 显示 `ninja: no work to do`，重新确认既有失败：

| 层级 | 修改前结果 | 最终修改后结果 |
| --- | --- | --- |
| GoogleTest：MetallicDebugTests | 26 passed / 1 failed / 0 skipped | 27 passed / 0 failed / 0 skipped |
| CLI script：参数案例 | 43 passed / 0 failed / 0 skipped | 43 passed / 0 failed / 0 skipped |
| 外层 CTest：上述两个条目 | 1 passed / 1 failed / 0 skipped（exit 8） | 2 passed / 0 failed / 0 skipped（exit 0） |

最终 fixture 保留原关联断言后再次重建，并将同一 CTest 选择改用 `-V`，
输出明确给出 `27 tests from 7 test suites` 全过及 `43 CLI argument cases passed`。
内外层计数不能相加；没有删去或过滤失败用例来取得 green。
所有构建 job 均已收集，未运行 GPU targets 或 editor；CPU tests 中使用的 GPU completion
模拟或 ShaderTrace fixture 不是 Vulkan/GPU runtime 验证。

`git diff --check` 的本次 scoped 检查通过。root 已独立审查实际 fixture diff、完整记录与
当前 C++/Slang producer 契约，结论为 **APPROVE**。
root 随后对 `Metallic(Debug|CtlArguments|VirtualShadowMapGPUProfile)Tests` 三个条目运行
最终集成 verbose CTest：**3 passed / 0 failed / 0 skipped**；内层仍分别为 27 / 43 / 53，
不与外层相加。实施代理未自行 stage、commit、push 或修改远端 issue，由 root 统一集成。

## 验收余额

此修复只消除一处陈旧 CPU fixture，不能证明 #63 的 CI 配置、`metallic-ci` tools 标签、
平台 skip 矩阵、Nsight 集成或 ShaderPipelineBench lane 已经验收，不应关闭整票。
主说明中关于 Resident 的历史描述仍需单独对账，
例如[现存描述](../DebugControlPlane.md#L201-L209) 与[旧混合 producer 测试说明](../DebugControlPlane.md#L341-L343)；
它们不改变上面已由当前 producer、decoder 和历史提交共同确立的契约，也不在本次测试变更中顺手重写。
