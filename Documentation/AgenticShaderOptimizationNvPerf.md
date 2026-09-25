# NvPerf backend 验收记录

日期：2026-09-25。实现由官方 SDK 直接采集的可选 Vulkan backend；不用 Nsight UI、
ngfx 导出或 ProfileMark CSV。公开使用说明见 [NvPerf.md](../Tools/Perf/NvPerf.md)。

## 环境与结果

- SDK：`D:/NVIDIA_Nsight_Perf_SDK_2025.1_Public_Windows.zip`，解压到忽略的
  `build/nvperf-sdk-2025.1`，没有混用 Nsight Graphics 2026 的 DLL。
- GPU：RTX 5070 Ti / GB203，驱动 616.92；Release，时钟 unaltered。
- 工作负载：MiniZorah WorkControl，1797×660 输出、1198×440 渲染，固定 cut/page
  mappings，四帧历史 priming，graphics queue，单目标帧，early/late 各一个 range。
- [正式批次](../build/nvperf-20260925-02/Manifest.json)：三次独立串行进程全部成功，
  每次单 pass、两个范围、零 ranges/trace bytes 丢失。三次完整 workload、绑定
  SPIR-V、depth/visibility readback 一致。

| 范围 | 指标 | 三次中位数 | (max−min)/median |
|---|---|---:|---:|
| early | GPU duration | 136.880 µs | 1.01% |
| late | GPU duration | 29.731 µs | 1.65% |
| early | SM cycles active / peak sustained elapsed | 95.758% | 0.88% |
| late | SM cycles active / peak sustained elapsed | 75.706% | 1.05% |

以上是诊断计数，不能与普通计时直接混用；SM active 指标不是 occupancy 或候选加速。
所有原始值与 evaluator 返回单位见 [Summary.json](../build/nvperf-20260925-02/Summary.json)。

2025.1 SDK 对 Vulkan 1.4 报出“未正式支持该版本”的警告，源码说明它按最新已知
版本处理。已保存 `vulkanVersionOfficiallySupported=false` 与原始 stderr。
这里证实的是该本机组合的实际采集结果，不是官方兼容性认证。

## 可复核边界

[离线复核](../build/nvperf-20260925-02-audit.json)通过：文件哈希、原始 workload
readback、进程串行性、范围/指标/单位、完整 pass、零丢失和汇总重算一致。
原始 availability、config、prefix 与 CounterDataImage 均保存；离线复核暂不再次
调用 SDK 解码该 image，明确返回 `counterImageReevaluated=false`。

`nvperf-20260925-01` 保留为失败批次：三个 GPU 采集均生成 complete，但第三次
renderer 正常退出和 psutil 子进程枚举发生竞态，导致 runner 没有记录 exitCode。
修复监控竞态后用新目录重跑三次，没有补写旧进程状态冒充成功。

[错误指标验证](../build/nvperf-invalid-metric-20260925/Validation.json)通过：未知名称
在 BeginSession 前拒绝，进程退出码 1，失败原因明确，没有 CounterDataImage。

Release ON/OFF 两条构建路径通过。OFF 构建使用不存在的 SDK 路径，
[普通工作负载](../build/nvperf-off-20260925/normal/Validation.json)仍通过且
`normalTiming=true`；[显式请求](../build/nvperf-off-20260925/requested/Validation.json)
失败并提示 backend 未编译。最终恢复本机 ON 配置。
OFF 普通运行与 ON 采集运行的完整 workload 身份及 readback 也相同。
摘要和证据文件哈希保存于 [机器结果](AgenticShaderOptimizationNvPerf.json)。

Perf CTest 7/7，共 94 项 Python 测试通过，包括缺失/重复范围、部分 pass、溢出、
错误指标与非有限值、仪器化运行不能进入普通计时，以及实际遇到的退出监控竞态。

多 pass 仍明确拒绝；后续需要恢复可写资源与历史状态才能开启。源码行/PC/SASS
归因、Full Zorah 驻留验收和 shader 候选性能结论均不在本次完成范围内。
