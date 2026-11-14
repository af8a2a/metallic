# #62 skip 策略与 final-blit 清单后续验收

跟踪 [#62](https://github.com/af8a2a/metallic/issues/62)，承接
[D2 XML 汇总记录](D2Followup-2026-11-14.md)。本轮是局部实现，不关闭整票。

## 现状与实现

| 剩余验收 | 本轮落实 | 尚缺的整项证据 |
| --- | --- | --- |
| 普通 RHI / 编辑器强验证 | [required-run 入口](../../Tools/RunRequiredTests.py) 组合子进程退出码和全新 XML，选中 skip 非零；CPU 子进程及真实 CTest fixture 验证 | 裸二进制默认仍宽松；参考机器选择集、真实 GPU/editor 零 skip 与部署的 CI lane 未验 |
| 已知 skip / 新增 skip 失败 | [精确比较器](../../Tools/SummarizeTestResults.py) 与非空 [有限基线](../../tests/KnownSkips.json)，新 case/suite/reason 失败 | 初始仅两项 MiniZorah opt-in，source-backed、非 GPU 观测；尚无完整设备/驱动基线和实际 GPU CI 对比证据 |
| final-blit 已知失败 | 修复 [C++ 清单](../../tests/rhi/FinalBlitTests.cpp) 遗漏的 10 个当前资产；[CPU 回归](../../tests/perf/TestFinalBlitPipelineInventory.py) 覆盖全部 38 个图并证明遗漏/错误源失败 | 原 C++ 用例未重新构建/执行，不代表反射、内建样例变换、RenderGraph::validate 或 GPU 运行通过 |
| 架构验证记录分列 | [历史验证表](../ProjectArchitecture.md#10-rhi-与-vulkan-后端) 明列 passed/skipped/failed；无分类证据处写“未单列”，不补零；新记录注明粒度 | 本项文档已落实，不把历史 87/3、71/4 改写成当前能力基线 |

required-run 与基线比较是两种策略：前者拒绝所有 skip，后者仅允许所选 profile 的精确
已知 skip，绝不赦免真实失败。外层 CTest 和内部 GoogleTest 分别记账，不能相加。
CLI、限制和 CI 调用方式见 [RHI 说明](../RhiTestbench.md#普通-rhi--编辑器-required-run)。

## 已知失败单列：render_graph_final_blit_pipelines

- 原 issue 所提 `gpu_driven_realtime` 已在历史提交 `82adcc58b` 加入清单；不能据此判断
  当前已修复。检查当前资产发现清单 28 项、实际 38 个图，目录枚举/数量断言仍必失败。
- 本轮精确增加四个实时 VSM 图（预期 `DLSSSR.color`）、五个 virtual-shadow-map 图
  （预期 `Deferred.color`）和 native strands（预期 `Fiber.color`）。预期指 AutoExposure
  前的输入，既有曝光/NR/FinalBlit 连接断言完整保留；没有修改资产或放宽断言。
- 红测：原清单下 CPU 回归 17 passed / 0 skipped / 1 error，错误逐项列出 10 个遗漏；
  补全后 18 passed / 0 skipped / 0 failed，真实资产子例遍历全部 38 个图。
- **状态：静态清单缺陷已修；完整 C++ runtime 验收待执行。** 继续归属 #62，不能把
  未运行写成已通过，也不注册 `WILL_FAIL` 或通配 expected-failure 掩盖新的失败。

## 回归记录

主线程从仓库根运行，Python 使用当前已有解释器，CTest 复用 MSVC 14.52.37025 /
Ninja / SDK 10.0.26100.0 的已有 Release 配置；只重新生成测试注册，没有切换 SDK/功能选项。

| 检查 | passed | skipped | failed |
| --- | ---: | ---: | ---: |
| XML 汇总及精确基线 CPU unittest | 36 | 0 | 0 |
| required-run synthetic child / 真实 CTest fixture | 13 | 0 | 0 |
| final-blit 静态清单 / 负例 unittest | 18 | 0 | 0 |
| VSM GPU profile 工具 CPU/ABI 回归 | 48 | 0 | 0 |
| 注册 CTest（上述前三组，外层作业） | 3 | 0 | 0 |

注册 CTest 的本地忽略产物位于 `build/issue62-policy-20261114/`，不是可移植的仓库附件。
三项 CTest 原始返回 0，XML 再由汇总器 `--require-all` 检查为 3/0/0、返回 0。
wrapper 回归包含 pass/skip/fail、child 77、child 非零但 XML pass、缺失/损坏/空 XML、
零选择、disabled、旧报告拒绝、单轮约束以及带空格路径。它们是 CPU fixture，不冒充 GPU。

[CI preset](../../CMakePresets.json) 已包含专用 `skip-policy` 标签并要求 Python；缺 Python 会在
该 preset 配置阶段失败，不再静默省略工具回归。没有新增托管 CI workflow，也没有证明
仓库外 CI 已部署 GPU runner。尚缺项继续留在 #62，不以空基线、旧日志或 expected-failure
代替验收。
