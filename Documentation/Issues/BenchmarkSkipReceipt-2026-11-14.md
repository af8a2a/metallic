# #63 benchmark 内层 opt-in skip 逐项回执

这是 [#63](https://github.com/af8a2a/metallic/issues/63) 的独立 D2 可观测性小切片。
它接续 [Scene 原生 skip reason 修复](SceneSkipReceipt-2026-11-14.md)，不关闭整票。

## 真实日志缺口和最小 producer

此前真实 hosted [run 43558571277](https://github.com/af8a2a/metallic/actions/runs/43558571277)
在 `cdff22f52692449cffb973c24f90a816963af371` 已通过原61项；其中
`MetallicShaderPipelineBenchTests` 的原始输出只有 `..ssss....`、`Ran 10 tests` 和
`OK (skipped=4)`。它证明 6 PASS + 4 Python SKIP，但没有逐项打印方法名或原因。
源码可以解释这四项，却不能替代旧 run 没有留下的逐项实际回执。

唯一代码变化是 [该项 CTest 命令](../../tests/perf/CMakeLists.txt#L50-L55)
在 script 路径和 `--executable` 参数之后追加 `-v`，即 unittest verbose，
不是 Python interpreter 的 import verbosity，也不是给整个 CTest 加 verbose。
[现有入口](../../tests/perf/TestShaderPipelineBench.py#L310-L317) 已通过
`parse_known_args()` 保留参数并交给 `unittest.main()`，无需修复或改写 runner。
[原生调用](../../tests/perf/TestShaderPipelineBench.py#L31-L43) 显式构造各方法的 native 参数，
不转发这个 `-v`；原有 timeout、crash、非预期退出仍走 failure，而非 skip fallback。

名称、标签、180s timeout、10个方法、`--gpu` opt-in、断言与失败策略全部不变。
没有新增测试、GPU 运行、profile、skip baseline 或环境开关。

## 四项源码 skip 合同

四个方法都在进入原生 GPU 工作之前以 `if not self.gpu:` 主动跳过，
不是探测设备失败后豁免。以下精确名称和原因先作为源码合同列出，不能倒填旧 hosted 回执：

| 方法 | 完整 reason |
|---|---|
| [test_gpu_concurrent_pipeline_batches](../../tests/perf/TestShaderPipelineBench.py#L205-L212) | `enable --gpu for concurrent pipeline checks` |
| [test_gpu_explicit_modes_and_worker_matrix](../../tests/perf/TestShaderPipelineBench.py#L235-L249) | `enable --gpu for native pipeline mode checks` |
| [test_gpu_pipeline_cache_and_timing](../../tests/perf/TestShaderPipelineBench.py#L168-L180) | `enable --gpu for native pipeline checks` |
| [test_gpu_shard_merge_reloads_metadata](../../tests/perf/TestShaderPipelineBench.py#L284-L292) | `enable --gpu for shard merge/reload checks` |

## 回执保留边界

[现有 CPU workflow](../../.github/workflows/CPU.yml#L24-L40) 上传 JUnit 和原始测试输出；
必须按本次实际内容核对它们保留的逐项范围，不推测所有输出目的地都完整。
外层仍使用 `--output-on-failure`：成功测试不保证把这些逐项文本直接打印到
Actions console 或 lane 的汇总日志。内层 Python SKIP 也不会因此变成四个结构化的
outer JUnit skipped testcase。

## 实际本地验证

沿用原兼容 Ninja Debug / MSVC 14.52.37025 / SDK 10.0.26100.0，六个选中目标构建完成；
不是 no-op，原生对象确有重新编译与链接。没有改 compiler、generator、SDK 或 feature flags，
也不将这个本地配置冒充 hosted SDK-off 配置。
实际完整 CPU 回归为 **62/62 PASS、0 outer failure、0 outer skip，93.98s**。
对照上一轮生成的模型与原始 JUnit：62项名字、逐项 labels 和其他 properties 全不变；
其余61项命令不变，唯一命令差异是本项末尾的 `-v`，仍有39个实际 Python 命令。

完整原始测试输出实际留下10条方法结果：**6 PASS + 4 SKIP**，以及 `Ran 10 tests` /
`OK (skipped=4)`。四项 SKIP 的实际完整名称与 reason 均与上表逐项匹配；
这是本次本地运行新留下的回执，而非把源码文字补到旧 run。

实际 JUnit `system-out` 受现有1024-byte上限截断：保留前6条完整方法记录
（2 PASS + 4 SKIP，四项 skip 原因均完整），第7条 `test_help_and_invalid_arguments`
只有不完整文字；其余方法的完整结果与最终汇总要读原始测试输出，不能从 JUnit 缺失处推测。
外层 JUnit 的62项成功不受这项文本截断影响。没有为此扩大生产输出阈值或伪造结构化内层结果。
本机 ignored 证据保留了构建日志、原始完整输出、JUnit、生成模型和离线核对结果。

截至本切片本地审查时，尚无新的 hosted 结果；前一次原61项成功不是新增第62项的 hosted 证据。
本切片不代表被跳过的 GPU pipeline 路径、非 Windows policy、editor/shared matrix
或既有 Nsight integration-ON 严格验收通过，未追加任何 GPU 重试。
