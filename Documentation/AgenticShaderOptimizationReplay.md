# WorkControl isolated production replay

2026-09-27，基于 `515fbc11c4e3a57860ce6df514de930faf041240`。
实现入口：[WorkControlReplay.py](../Tools/Perf/WorkControlReplay.py)；
[最小合同](WorkControlReplayContract.md)；[运行与证据格式](../Tools/Perf/WorkControlReplay.md)。

## 已验收范围

MiniZorah history WorkloadCase，early/late 各三个独立串行进程。每进程从同一个
primed target frame 捕获真实生产 dispatch 的输入和紧邻 dispatch 的直接输出；
两次无 counter replay 逐字节一致后，才启动 NvPerf。保留生产 PreparedExecution，
保存 input/device SPIR-V 和完整 push 数据；私有 heap 保留相同 typed descriptor
索引，所有 shader 可达 buffer 指向独立副本，物理地址变化明确记录。

每个 SDK pass 之前，完整恢复并读回验证全部 scratch buffer；随后独立提交一个
间接 dispatch；之后核对完整 packed depth/visibility pixel allocation 及只读
scratch bytes。page table 全字段（包括 lastRequestFrame）、request、LOD、HZB
和实例可见性等生产状态也逐字节不变。最后正常 checkpoint 的 depth/visibility
与 control/before 相同。未将只检查 page mapping 或默认清零视为恢复。

counter 采集只使用原有两项指标。通过 SDK pass groups 将两项拆开，六个进程均由
SDK 实际返回 `requiredPasses=2`、完成 `passesCollected=2`，零 range/trace 丢失。
这是状态恢复验证用调度，不增加新指标，也不声称是最优 counter 调度。

- [early 三进程原始包](../build-release/work-control-replay-early-multipass-01/Manifest.json)
- [late 三进程原始包](../build-release/work-control-replay-late-multipass-01/Manifest.json)
- [单 pass isolated pilot](../build-release/work-control-replay-early-nvperf-01/Manifest.json)

各包包含源码、EXE/DLL/SDK 身份、原始输入/control/每 pass restore/output、生产
前后状态、进程生命周期、提交台账和 counter image。当前 verifier 已重新检查
整个文件清单/hash、原始 byte equality、nested binding、indirect、SPIR-V 身份、
相同冻结帧及跨进程 WorkloadCase 一致性；不执行归档脚本。

## 计数结果及边界

| 指标 | early 中位数 | early 相对极差 | late 中位数 | late 相对极差 |
| --- | ---: | ---: | ---: | ---: |
| gpu__time_duration.sum | 167416.67 ns | 2.80% | 42805.56 ns | **10.64%** |
| sm__cycles_active.avg.pct_of_peak_sustained_elapsed | 95.5225% | 0.19% | 81.3780% | 2.06% |

相对极差为 `(max-min)/median`。late duration 超过既有 10% 重复性门槛，保留为
不稳定；没有丢弃该进程或重跑直到通过。early 两项及 late SM active 稳定。上述
时间是诊断 range 时间，不是正常 production timing；SM active 也不是 occupancy。
没有性能候选、加速或整帧收益结论。

isolated 证据使用 `metallic-nvperf-isolated-v1`，范围为
`isolated-dispatch-RHI-exclusive`：目标 submission 只有一个 dispatch，采集时
其他 RHI queue 提交由 owner lease 排除。此范围不证明系统级 GPU 独占，不屏蔽
其他程序/native SDK 的 GPU 活动，也不是源码行或 PC 归因。现有 in-frame
`metallic-nvperf-v1` 继续保持原范围，verifier 拒绝直接将其改标为 isolated。

2025.1 SDK 配套头文件/运行库；Vulkan 1.4 的
`vulkanVersionOfficiallySupported=false` 与原始 SDK 警告保留。成功运行不等于
SDK 官方认证。离线检查仍不重新解码 CounterDataImage，
`counterImageReevaluated=false` 明确保留。

## 兼容性和未覆盖范围

[前一版 correctness-only 包](../build-release/work-control-replay-late-pilot-01/Manifest.json)
已由当前 verifier [离线复核](../build-release/work-control-replay-late-pilot-offline.json)，
仍为 correctness-only，没有提升为 counter evidence。
旧 in-frame parser 的兼容与 scope-promotion 拒绝有 CPU 回归；历史
`build/nvperf-20260925-02` 原始目录当前不存在，无法对该 9 月 25 日包重新完整复核。
已提交的历史汇总不会被当作原始包的替代品。

最初 early pilot 的 GPU 比较通过，但 Python verifier 将 pixel descriptor 偏移
误写为 bins[9] 而拒绝封包。核对 shader 后修正为 bins[11]（bins[9] 是 reversed-Z），
当前 verifier 对其原始 run 离线检查通过；失败 Manifest 原样保留。

首版限定 minimum subgroup >=32、冻结 MiniZorah、graphics queue、非 tessellation、
非异步路径，source snapshot 总量上限 1 GiB。低 subgroup fallback、Full Zorah、
跨进程资源重建及最终彩色画面/长期时序质量没有由本次验收覆盖。原 in-frame
backend 仍拒绝多 pass；只有 isolated owner 能进入恢复后多 pass 路径。

## 最终回归与失败门禁

[机器验收索引](AgenticShaderOptimizationReplay.json)记录各原始 Manifest 的 SHA-256、
当前 verifier 结果和逐进程指标。大体积原始包保留在本机 build-release，未加入源码。

- 最终 allocation identity / lease / queue identity 版本分别完成
  [early](../build-release/work-control-replay-final-early-01/Manifest.json) 和
  [late](../build-release/work-control-replay-final-late-01/Manifest.json) 单进程两 pass
  回归；它们没有混入前述三进程重复性统计。
- 七项独立进程负向测试全部 `expected-failure-verified`：提交前/后取消、提交失败、
  设备错误、超时、恢复失败、生产状态泄漏检查。均未启动 counter session。
  v2 提交/设备错误走实际 Result 错误分支；设备错误/超时保存 poisoned 状态并以 3
  退出。超时注入发生在 GPU 安全退休之后，不是实际 GPU 挂起或 TDR。
  恢复失败跳过第二次 pixel GPU reset，实际 readback 检出累积状态；状态泄漏测试
  修改 pageTable 比较观测字节，不破坏生产 GPU page table。
- [新的旧协议回归包](../build-release/work-control-replay-inframe-regression-01/Manifest.json)
  三进程单 pass 完成并由原 verifier 复核，仍为 in-frame scope。这不替代缺失的
  9 月 25 日原始证据。
- NvPerf ON：`cmake --build build-release --target MetallicGPUDrivenSample -j 12` 通过。
  NvPerf OFF：复用原有 `build-pass-stages-nrd` 配置构建 `MetallicRhiTests` 通过，
  未更改该树的编译器、SDK 或生成器选项。
- `python -B -X utf8 -m unittest discover -s tests/perf -p 'Test*.py'`：115 项通过。
  `ctest --test-dir build/perf-tests -C Debug --output-on-failure`：8/8 组通过。
  当前 RHI profiling、registry lifetime、submission transaction、indirect barrier、
  allocation identity、prepared execution 回归：14/14 通过，无跳过。

取消/失败包只证明所声明注入路径的拒绝、隔离及证据行为；没有将它们当成真实
驱动故障恢复认证。未修改 shader 算法，也没有接受任何性能优化候选。
