# #63 Scene opt-in skip 原因回执

这是 [#63](https://github.com/af8a2a/metallic/issues/63) 的独立 D2 小切片，不关闭整票。
它不改变 SceneLoader、资产处理、opt-in 条件或已有 skip 数量。

## 真实缺口

修复前的真实 Windows hosted [run 43558571277](https://github.com/af8a2a/metallic/actions/runs/43558571277)
在 `cdff22f52692449cffb973c24f90a816963af371` 已通过原 61 项 CPU CTest；
但完整原始日志中 `SceneLoading.ZorahFullOpenReportsMissingCook` 只有位置和空白的 skip 正文。
这不是 outer JUnit 的失败，也不能写成已有充分的 skip reason。
当次 Scene 为 171 方法 = 162 PASS + 9 native SKIP；benchmark 另有 4 Python SKIP，二者不能混称。

[原生用例](../../tests/scene/main.cpp#L6855-L6875) 在
`METALLIC_TEST_ZORAH_FULL` 未设置时跳过；本切片只补充
`Set METALLIC_TEST_ZORAH_FULL=1 for the local asset`，与
[相同 opt-in 的兄弟用例](../../tests/scene/glTFInstancingTests.cpp#L193-L199) 一致。
原条件仍只检查环境变量是否存在；没有顺便将其改为布尔值解析。
设置 opt-in 后的真实大资产、missing-cook 及 resident-import 断言全部保留，未用 fixture 替代。

## 回归检查的边界

[行为回归](../../tests/perf/TestSceneSkipReceipt.py) 调用已经构建的原生 Scene executable，
不是搜索 C++ 字符串或读取手写 XML：

- 复制当前环境，仅从子进程环境中移除该 opt-in；不改父进程或机器环境。
- 只选择上述一个 CPU 方法，单轮运行；在新建的临时工作目录产出新的原生 GoogleTest XML。
- 要求原生退出 0，恰好一个指定 testcase、一个明确 skip、没有 failure/error；
  skip 的 `message` 必须包含完整的预期原因，只有源位置不算原因。
- 原生非零、缺失/损坏 XML、零选择、错误选择、意外 pass 或 skip 同时含 failure 均不能通过。
- 临时产物在检查后清理；原生输出及 XML 正文写入测试日志，便于审读实际回执。

[CTest 注册](../../tests/CMakeLists.txt) 使用目标生成表达式和 `--executable`，
沿用 [CPU executable closure 检查](../../tests/perf/TestCIPreset.py#L57-L81)，
无需额外 build target。标签为 `tools;cpu;skip-policy`：原 61 项保持，新增这一项后应为
62 个 outer CTest / 39 个 Python 命令。此 probe 会再次执行同一个预期跳过的 native case，
必须与主要 Scene suite 的 skip 分开记账，不能当成新的资产路径通过或新增的唯一 skip case。
没有扩充 [已知 skip 基线](../../tests/KnownSkips.json)，也没有放宽 required-run 或任何失败门槛。

## 验证记录

修复前，实际已有原生 executable 成功跳过该方法，但新检查退出 1：
原始 XML 的 `message` 只有源位置和换行，没有原因。
本地重建沿用原 Ninja Debug / MSVC 14.52.37025 / SDK 10.0.26100.0 配置，
没有为了通过修改 compiler、generator、SDK 或功能开关；这不是一次新的 hosted SDK-off 构建。

Scene 最小重建后，同一真实方法仍为 1 SKIP / 0 PASS，但 XML 含完整原因，新检查退出 0。
随后六个选中 target 的构建完成（不是 no-op；Ninja 进度分母195，末条为194/195的RHI链接，命令退出0），实际 CPU CTest
**62/62 PASS、0 outer failure、0 outer skip，93.00s**。离线核对确认原 61 项名字及逐项
labels 全保留，唯一新增项就是 receipt 检查；实际有 39 个 Python 命令。
本机 ignored 证据保留了 red/green、构建日志、完整原生输出、JUnit 和离线核对结果。

本地主要 Scene suite 为 **172 = 163 PASS + 9 native SKIP**；与此前 hosted 的171项相比，
逐项清点仅多 [RTXCRClairePonytailDots](../../tests/scene/main.cpp#L4935-L4942)，
该方法由 `METALLIC_HAS_RTXCR_GEOMETRY` 编译开关控制，没有少掉 hosted 方法。
这是兼容本地配置与 SDK-off hosted 的边界，不将172说成 hosted 实测。
新增 probe 另重复执行同一个 opt-in skip 一次；benchmark 仍有 4 个独立 Python SKIP。

独立审查另以隔离 fake-child 验证 20 项（2 个有效回执接受、18 个异常拒绝），
包括空原因、只有位置、错误选择、原生失败与 failure+skip；同时核对父环境不变、
新目录隔离和清理。它验证检查器的拒绝逻辑，不替代上述真实原生 red/green。

## 尚未满足的整票验收

本检查通过只代表 **预期跳过时诊断回执完整**，不代表该被跳过的大资产路径通过。
Windows editor / shared skip matrix、真实非 Windows cache policy 和 Nsight integration-ON
startup 严格验收仍未满足；本切片不追加 GPU 运行，不豁免既有 Nsight 失败。
原 61 项真实 hosted 成功也不自动成为本次新增第 62 项的 hosted 证据。
