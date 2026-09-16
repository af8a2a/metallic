# Stream demand：纯 wave 边界分配

2026-09-16，基线 `21a8c16`；RTX 5060 / 驱动 610.47。

## 参考与实现

已通过本机 Git 认证取得 [Unreal PR 10648](https://github.com/EpicGames/UnrealEngine/pull/10648)，head 为 `42105bf1e959d8efc23c58ef9ab854b64a04d835`，父提交 `5d025e0cf62c324995a48ec95a13572865df7bd7`。浅层引用存放于 `.cache/stream-waveops/unreal-pr.git`，没有切换本地 Unreal 工作树。网页和 GitHub connector 的未授权入口无法读取该私有 PR，最终依据的是实际 Git 补丁。

PR 面向 Nanite Tessellation 的 `DistributeWork`：使用有效源位图、set-bit select、批内边界位图和 popcount，替代 LDS 队列压缩与边界表。PR 作者报告的是 RTX 3080 上 shader 指令计数的改善，不是 Metallic 的帧时间收益。

Metallic 原来的 demand 分配已经使用纯 wave 的二分 shuffle 查找源，因此本次收益来自减少每个输出批次的源查找工作，而不是把这一段从 LDS 改成 wave。Slang 实现独立编写，没有复制 Unreal 文件。

- `WaveWorkDistribution.slang` 将非空源压缩一次，保存源 lane 与累积结束位置。全满 wave 跳过压缩。
- 每个输出批次用 `WaveActiveBitOr` 汇总源的结束边界；popcount 得到输出 lane 对应的紧凑源序号，再 shuffle 原始源和局部 item 索引，取代原先每个批次的二分查找。
- 使用实际 `WaveGetLaneIndex()`。空源和最后一个不满批次的 lane 仍参与 shuffle，未使用的紧凑 lane 不发布边界。尾部 shuffle 索引限制在有效源范围。
- 掩码按 32 位分段；宽 wave 使用多 word ballot，不对 uint 做 32 位及以上位移，也不使用 workgroup index 代替 subgroup lane。实际硬件验证为 wave32；更宽 subgroup 尚未实机验证。
- 清理/入队阶段改为每个 wave 独立 prefix-count、申请队列区间并广播首地址。任务顺序不决定安全 cut 或预算，已有 dispatch barrier 继续保证 payload 完整发布后才消费。

安全 frontier、mask、稳定 emit 与全局预算扫描仍保留跨 wave 同步；LOD 层之间的设备内存发布也保持不变。本次没有调整可见性、LOD 阈值、输出容量、流送预算、worker 数或自适应调度阈值。

需求内核的生成 SPIR-V 检查结果：Workgroup 变量 0，`OpControlBarrier` 0，`OpMemoryBarrier` 0，包含 subgroup ballot、bitwise-or 和 shuffle；通过 `spirv-val --target-env vulkan1.3`。这项检查针对 `streamDistributedDemandMain`，不表示整个 VBuffer 没有共享内存或屏障。

## 验证与测量协议

沿用仓库 MiniZorah 路线 `.cache/gpudriven-four/Replay.json`，SHA-256 `feb79872154850af32db25a54ba3d22b48b9a04a10f7f2e8dadaf19f98f2f2d2`。每轮 3000 帧、10 个阶段，完整实时管线；1920×1080 输出、DLSS Quality 内部 1280×720、LOD 1.5 render px、4096 demand workers。预算和 CLAS/BLAS/光栅配置一致。

原始日志、帧数据、完整源文件哈希及 GPU 竞争记录在 `.cache/stream-waveops/`。`before` 为修改前的两轮，`ballot` 为首次实现的一轮探索数据，`final` 为最终两轮及独立质量回放。所有指定轮次和全部帧保留。新增测试会改变测试可执行文件哈希，运行时 C++、图配置与 replay harness 没有修改。

新 GPU oracle 使用 128 线程工作组，检查空源、单个首/末 lane、密集与稀疏源、64 个 item 的长源、交替 31/33 item、跨批次与尾部。CPU 根据 GPU 导出的实际 subgroup membership 构造串行展开结果，逐个核对 owner、源 lane、局部 item 与全局顺序，未使用输出必须保留 poison；不假定 workgroup/subgroup 的 lane 映射。

另运行既有 872 个 GPU/CPU cut 用例、24 个跨实例预算用例、Bunny 的光栅/cut/相机组合、软硬光栅等价、混合 resident/stream producer 和 CPU 拓扑验证。首次新增 probe 因 include 搜索路径错误而编译失败，改为相对包含生产 helper 后通过；失败日志保留在 `tests.log`。

## 结果

完整分布、所有重复、质量检查点和 manifest 见 [StreamWaveWorkDistributionResults.json](StreamWaveWorkDistributionResults.json)。下表为 GPU P50，单位 ms，未剔除慢帧。frontier envelope 包含 priority clear、state clear/seed、detail demand、frontier 和 mask。

| 阶段 | 基线 m1 | 基线 m2 | 最终 m1 | 最终 m2 |
| --- | ---: | ---: | ---: | ---: |
| warmup | 0.938 | 0.937 | 0.917 | 0.918 |
| start_hold | 0.939 | 0.937 | 0.919 | 0.917 |
| forward_1 | 0.935 | 0.934 | 0.912 | 0.907 |
| forward_2 | 0.714 | 0.709 | 0.702 | 0.718 |
| forward_3 | 0.525 | 0.523 | 0.528 | 0.526 |
| far_hold | 0.676 | 0.675 | 0.681 | 0.678 |
| return_1 | 0.524 | 0.523 | 0.528 | 0.527 |
| return_2 | 0.708 | 0.726 | 0.715 | 0.709 |
| return_3 | 0.941 | 0.936 | 0.915 | 0.911 |
| return_hold | 0.940 | 0.939 | 0.919 | 0.916 |

`return_hold` 的明细：

| 区间 | 基线 m1 / m2 | 最终 m1 / m2 |
| --- | ---: | ---: |
| Detail demand | 0.253920 / 0.255600 | 0.237968 / 0.237408 |
| LOD clear / demand seed | 0.210768 / 0.210688 | 0.202560 / 0.201200 |
| LOD frontier | 0.322928 / 0.321408 | 0.325984 / 0.325296 |
| Stream traversal | 1.578304 / 1.578848 | 1.582784 / 1.558128 |
| VBuffer | 4.145776 / 4.142304 | 4.125376 / 4.116368 |

这两轮中 demand 降低约 6–7%，入队降低约 4%，近景 frontier envelope 降低约 2–2.5%。safe frontier 本身增加约 3–4 μs；中远景使用 ordered 路径，没有预期中的 demand 加速，部分阶段略增。尚不能据此认为大部分 stream 开销已经解决，或把 PR 中的指令统计比例套用为整帧收益。

全路线逐帧 envelope 均值为 **0.816463 / 0.819460 → 0.825948 / 0.807070 ms**。最终 m1 的后台 Unity 引擎采样峰值达到约 80%，该轮 `start_hold` 的 host P50 从约 8.2 ms 增到 19.1 ms。报告保留这些波动、P95/P99/max 和竞争记录；本轮只报告本机观察，不能宣称隔离环境下的稳定整帧提速。

四个计时对照与独立质量回放的最终 cut 相同：19,394 active groups、162,989 selected clusters，early 26,806 HW / 19,172 SW，late 200 HW / 361 SW，容量回退实例数 0。最终需求工作也相同：105,483 个候选根、17,144 个任务、67,178 个节点、80,813 个 group。wave 槽位数随非确定性的入队/领取顺序变化，未将其误当作几何变化。

独立 3000 帧质量回放包含 15 个检查点；第 29 帧仍有 2,575 项待流送的可见细节，第 59 帧及之后可见超目标细化均为 0，最终最大可见误差 1.499928 px。ordered/distributed 的切换与此前相同，没有容量回退或覆盖退化。

Release 构建通过；最终 `validation.log` 的 9 项测试全部通过，并开启 Vulkan validation。新增分配 oracle 覆盖 6,144 个源、192 个 native wave32；原有 872 个 cut 用例及预算、光栅验证也通过。未复现上一轮记录的 CPU SEH，但本次没有修改其相关 C++ 路径，不能宣称已经修复该异常。

## 复现

```powershell
cmake --build build-release --target MetallicRhiTests MetallicGPUDrivenSample --parallel 6
./build-release/tests/MetallicRhiTests.exe '--gtest_filter=*stream_wave_work_distribution:*meshlet_lod_stream*:*hybrid_raster_scene_equivalence:*render_graph_gpu_driven_mixed_producer_render' --rhi-validation --output-dir .cache/stream-waveops/validation
./Tools/RunMetallicCfgReplay.ps1 -Replay .cache/gpudriven-four/Replay.json -OutputRoot .cache/stream-waveops/final -Realtime -QualityWithoutValidation
```
