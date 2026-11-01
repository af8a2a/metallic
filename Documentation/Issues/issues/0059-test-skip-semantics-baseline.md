---
id: "0059"
title: "测试基建: skip 与通过无法区分，缺少 skip 基线与已知失败跟踪"
area: tooling
kind: task
priority: P1
state: open
milestone: M3-Hardening
labels: ["area:tooling", "kind:task", "priority:P1"]
evidence:
  - tests/rhi/main.cpp:325-327
  - tests/rhi/harness/Runner.cpp:309
  - tests/CMakeLists.txt:148
  - Documentation/ProjectArchitecture.md:700
  - Documentation/ProjectArchitecture.md:509-511
  - Documentation/RenderGraphAutomaticSynchronization.md:19
---

## 背景 / Context

仓库的 GPU 验证高度依赖「能力不足就跳过」的机制：RHI 用例按设备扩展/特性决定是否
运行，编辑器 smoke 允许脚本打印 `SKIP:`。这在没有统一 GPU 的机器上是必要的，但
当前会计口径把 skip 与通过混在同一个「通过」结论里。

## 现状 / Current state

- RHI 适配器在环境要求不满足时直接 `GTEST_SKIP() << gEnvironment->skipReason()`；
  用例自己也可返回 `skipped` 结果转成 `GTEST_SKIP()`。
- 同一框架内部对 skip 是**不一致**的：`Runner.cpp:309` 把「要求已通过后仍出现
  skip」判为 Fail（"unexpected skip after requirements passed"），而 GTEST 适配器
  路径接受任意 skip。
- CTest 侧用 `SKIP_REGULAR_EXPRESSION "SKIP:"`（编辑器 pipeline-cache smoke）和
  `SKIP_RETURN_CODE 77`（DLSS/NR 运行用例）把跳过标记为「未运行」，在 ctest 汇总
  里不进入失败，也不进入「通过」——但报告文本里常被读成通过。
- 文档已经在用「87 通过、3 跳过」这种并列表述，却没有一份**已知 skip 基线**：
  没人能判断某次运行多出来的 skip 是新增缺口还是环境差异。
- 更严重的是已知失败没有跟踪闭环：`render_graph_final_blit_pipelines` 的白名单
  缺少 `gpu_driven_realtime` 等样例、「另行执行仍失败」，测试本身未被修改，也没有
  对应的 issue 或已知失败清单。

## 问题或缺口 / Problem

1. 「测试全绿」可以完全由 skip 构成，尤其在新机器/CI 镜像上，GPU 能力不足会让
   整批 RHI 用例变成 skip，而总结仍然看起来是成功。
2. 反方向也有损失：框架把部分真实跳过当失败（`Runner.cpp:309`），维护者于是倾向
   放宽要求，进一步削弱信号。
3. 已知失败（如上述白名单用例）没有归属，长期停在「另行执行仍失败」的状态。

## 期望结果 / Expected outcome

- 统一 skip 会计口径：区分 **passed / skipped（带原因与类别）/ failed**，并在
  CTest 汇总和测试报告里分别计数。
- 维护一份已知 skip 基线（用例名 + 原因 + 所需能力），新增 skip 视为回归。
- 已知失败进入显式清单（issue 或 `KNOWN_FAILURES` 文件），并与 CI 的
  expected-failure 判定挂钩。

## 验收标准 / Acceptance criteria

- [ ] 新增脚本/CTest 报告汇总每一类结果的计数与明细，例如
      `ctest --test-dir build-ci -L rhi --output-junit .tmp/ctest-rhi.xml` 后由脚本
      输出 passed/skipped/failed 三列与 skip 原因直方图。
- [ ] 把 testbench 已有的 `--tb-require-all` 语义扩展到普通 RHI/编辑器用例路径：
      存在未满足能力要求的 skip 时以非零退出，供 CI 的强验证 lane 使用
      （现状：`add_test(... --tb-run --tb-suite core --tb-require-all ...)` 已用于
      testbench，普通 `--gtest_filter` 运行没有等价开关）。
- [ ] 仓库内提交 `tests/KnownSkips.md`（或 JSON），列出每个 skip 的用例、原因与
      所需扩展；CI 对比时对新增项失败。
- [ ] `render_graph_final_blit_pipelines` 的失败要么修复并纳入回归，要么进入
      已知失败清单并在报告中单列。
- [ ] `Documentation/ProjectArchitecture.md` 的验证记录段落改用 passed/skipped 分列。

## 相关文档 / References

- [ProjectArchitecture.md](../../ProjectArchitecture.md)
- [RenderGraphAutomaticSynchronization.md](../../RenderGraphAutomaticSynchronization.md)
- [RhiTestbench.md](../../RhiTestbench.md)
