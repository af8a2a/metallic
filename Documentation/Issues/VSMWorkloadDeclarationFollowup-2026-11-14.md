# VSM workload 顶层诊断声明的 readback 门槛（2026-11-14）

范围：[#10](https://github.com/af8a2a/metallic/issues/10) 的一个独立 CPU 证据读取边界。
基线 `5bd0ea5f4410d54759787fa5f3d003faa50c1b13`；不修改 C++/Slang、ABI 或 GPU 路径。

## 契约与实际缺口

[C++ writer](../../tests/rhi/VirtualShadowMapAcceptanceTests.cpp#L639-L674) 从
`benchmarkGPUWork` 获取 bool，禁止 ordinary timing 下启用，并把同一声明发布到 case 与
capture 两层。[读取规则](../VirtualShadowMapsGPUProfile.md#json-与历史包读取规则) 明确：
声明启用诊断却缺少 frame readback 不视为历史兼容。

原 [GPU profile 汇总入口](../../Tools/Perf/VirtualShadowMapGPUProfile.py#L415-L439)
的 workload 门槛仅检查 case 层：capture 声明 `gpuWorkDiagnostics:true`、case 没有该字段时，
所有帧都缺 `shadow.gpuWork` 的包仍被接受并汇总成 `workCounterFrames=0`。
这不是 null 或 disabled sidecar 争议，反例完全没有 workload 对象。

本修复只将该门槛改为 `case.get("gpuWorkDiagnostics", data.get("gpuWorkDiagnostics", False))`，
采用已有 [Build attribution 先例](../../Tools/Perf/VirtualShadowMapBuildAttribution.py#L108-L109)
和相邻 Build declaration 的 case 优先 / capture 回退规则。保留独立的 timing 声明禁令，
不新增声明类型校验或变更 null/disabled 语义。

## 红绿证据

[五个新测试方法](../../tests/perf/TestVirtualShadowMapGPUProfile.py#L368-L416) 均走真实
`package()` / `mutate_capture()`：临时目录写文件、重封装 Manifest，再调用
`summarize_package()` 的完整 sealed-package 验证，而不是 mock `load_package()`。

覆盖：

- capture 声明启用而全部缺 readback；
- 只缺一个 warmup 帧或一个 measured 帧，分别拒绝；
- capture 声明启用且 workload v6 完整，4 帧 available、3 帧 post-warmup 计数；
- case 声明显式覆盖 capture，两种 bool 方向均保留现有优先级；
- 无任何声明的历史包仍为 unavailable，缺字段不补零。

```powershell
python -B tests/perf/TestVirtualShadowMapGPUProfile.py
python -B tests/perf/TestVirtualShadowMapBuildAttribution.py
```

修改 parser 前：53 个测试方法运行，两个新方法产生 **3 条预期 failure**
（全部缺失、缺 warmup、缺 measured 均未抛异常）；其余新正例与兼容性检查通过。
单行修复后：**53 passed / 0 failed / 0 skipped**；邻接 Build attribution **8 passed**。
随后按实际文件名运行全部 VSM Python suites：

```powershell
python -B -m unittest discover -s tests/perf -p 'TestVirtualShadowMap*.py' -v
```

共 **157 passed / 0 failed / 0 skipped**，包含上述 53/8，不可重复累加。
初次邻接命令曾误引不存在的 `TestVirtualShadowMapAcceptance.py`（启动失败 exit 2），
经文件发现确认后改用上面的 discovery；不将该启动错误算作测试执行或验收证据。
外层入口也已执行：`ctest --test-dir build-release -R '^MetallicVirtualShadowMapGPUProfileTests$' -V`，
结果 **1 passed / 0 failed / 0 skipped**，verbose 输出再次确认内层 53 测试全过。
root 最终还对 `Metallic(Debug|CtlArguments|VirtualShadowMapGPUProfile)Tests` 三个条目运行
集成 verbose CTest：**3 passed / 0 failed / 0 skipped**，对应内层 27 / 43 / 53。
这些是不同层级和重跑，不与 Python 157 累加。

独立只读审查 `d2-vsm-oracle` 对单行实现、producer 契约及五个真实包测试结论为
**APPROVE**；没有顺带更改 timing 门槛、字段表、版本或偏移。

## 验收余额

本次没有运行 GPU / editor，不提供 raw readback 长度、真实 profiling 或数值正确性证据；
Python fixture 本身也不是 C++ writer/header 与 Python 字段表的跨语言 ABI 漂移 oracle。
#10 的 GPU profiling / acceptance 验收仍未满足，不能据此关闭整票。
