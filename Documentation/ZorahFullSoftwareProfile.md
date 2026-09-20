# ZorahFull SW：寄存器、lane 与原子写入定位

2026-09-20，RTX 5070 Ti / GB203，驱动 616.64，Nsight Graphics 2026.3.1。

**下一步优先做 cluster 装载与同步前后工作分配；暂不先做全局三角形排序或 tile 原子合并。** 当前证据更支持“lane 利用率低、barrier/访存等待较多”的方向，没有原子吞吐饱和证据。尚未取得逐源码行归因，因此这是有依据的下一项 A/B 实验，不是已证明的唯一根因。

## 实测证据

### 编译资源

通过可选 `VK_KHR_pipeline_executable_properties` 查询真实生产入口；普通优化编译和优化加行号编译（CaptureSymbols）各测一次，结果一致：

| SW 入口 | 寄存器/线程 | 驱动报告 shared bytes/组 | binary bytes |
|---|---:|---:|---:|
| legacy | 96 | 9408 | 30208 |
| prepared | 96 | 6848 | 30720 |
| depth plane | 96 | 9408 | 30592 |

工作组 128 threads，subgroup 32。预计算降低了部分 shared 用量，但没有跨越寄存器分配档位；不能由减少算术直接推导 occupancy 提高。这里报告的是驱动对完整 shader 的实际资源分配，不能用源码中某一块 2048 B 顶点数组代替总 shared 用量。

驱动的 `Local Memory Size` 字段返回约 64 GiB/线程，小内核也有同一异常高位；原值保留在 JSON，**不用于判断 spill 大小，也没有截断高位来制造结论**。本次没有可靠的 spill 字节证据。

### 当前 Full 的 GPU Trace

两次有效 live GPU Trace，每次 3 帧；第二次开启 `time-every-action`。采用完整材质/阴影、编辑器输出 1797×660、DLSS Quality 内部 1198×440、LOD 1.5 px、8 px 分流，legacy SW、关闭 jitter、串行 HW/SW。启动/预热后冻结 camera、cut、几何/CLAS 与纹理发布，再进入仅渲染的专用 hold 窗口。没有诊断读回混入 hold；Nsight 在采集完毕后终止目标，未把退出后的应用 Capture 缺失当成采样失败。

两次 `ProfileReady.json` 的 cut 都是 `7619267120868061868`，页面映射都是 `10653295828724463040`。Nsight REPRO_INFO 报告显存 demotion 为 0，工具正常生成 trace 与非空表格，未发现硬件事件溢出或计数器不可用警告。系统 RAM 约 97%，故本轮不用于评价 CPU/加载性能。时钟 unaltered，不是锁频 P95 基准。

**导出范围限制：** 两次 CLI 导出都把相邻 raster 工作归入 early `Hybrid raster: stable cluster bins` 范围，未单列源代码中的 `stream software clusters` 标签；逐 action 选项也没有改善 TSV 的标签粒度。下表是这个合并范围（最新 TSV 第 34、57、80 行）的指标，不能称为纯 SW 内核计数。此前独立 RHI 时间戳已证明 SW 是光栅的主要时间项，但不能把两种工具、两次运行的时间直接相减。

| 指标 | 最新 3 帧范围 | 含义 |
|---|---:|---|
| predicated-on active lanes / warp | 15.12–15.49 / 32 | 约 47–48% 的指令 lane 有效；不等于三角形数量利用率 |
| compute resident warp occupancy | 29.75–31.55% | 实测驻留，不能单独归因于寄存器 |
| compute register-file allocated | 66.27–70.29% | 寄存器资源用量可观，尚不足以证明它是唯一限制 |
| barrier 样本占比 | 54.51–54.67% | warp 状态样本归一化，包含 selected/not-selected |
| L1TEX long scoreboard 样本占比 | 22.15–22.25% | 访存依赖等待，不能直接归为 atomic |
| SM throughput | 22.52–23.51% peak | 不是算术管线吞吐饱和表现 |
| L2 throughput | 6.34–6.68% peak | 不能排除访存延迟 |
| DRAM sectors | 13.59–14.26% peak | 不是 DRAM 带宽饱和表现 |
| L2 atomic input active | 0.0549–0.0585% peak | 未见原子单元吞吐饱和 |
| global atom/red write sectors | 17.38–17.43 M | 是 sectors，不能当作 atomic 指令数或有效像素数 |
| atom/red write throughput | 0.0157–0.0168% peak | 使用原始指标语义，不改称帧耗时占比 |
| atomic L2 hit rate | 79.23–79.60% | 仍不能区分局部地址竞争与其他内存依赖 |

首轮（非 time-every-action）active lanes 为 14.83–15.27，方向一致。**55% barrier 不等于可省下 55% 帧时间；0.055% atomic active 也不等于 atomic 只耗时 0.055%。** 本轮不能给出原子写入独立毫秒数、单条原子指令延迟或扫描循环每行 lane 利用率。CLI 无逐行表格，当前也没有可用的 cua-child 原生 UI 接口；没有操作用户主桌面或把旧 MiniZorah 结果替代这些缺项。

指标含义参照 NVIDIA 的 [Shader Profiler](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html) 和 [System Architecture Guide](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html)。原始字段全名及行号保存在结果 JSON，避免混淆 syslts/lts 或 throughput/sectors。

## 工作分配决策

当前 `rasterStreamCluster` 由 lane 0 完成 `loadStreamRasterCluster` 后全组同步，再按唯一顶点投影、再次同步，最后一 lane 扫描一三角形 bbox。装载函数经多级依赖访问 active header、group、instance、page table、payload 与 cluster 描述，并填充含整份相机参数和实例变换的共享结构。高 barrier 与低有效 lane 和这条路径相符；但也可能包含顶点阶段/扫描尾部差异，不能只凭计数把所有等待定位到第一个 barrier。

建议按以下顺序实施，每一步独立验收：

1. **精简 raster 描述与协作装载。** 将光栅所需字段从通用 cull/classify 数据中分出，缩短局部变量生命周期；首个 wave 协作读取独立字段，保留依赖顺序、页面/索引边界校验和 uniform 退出语义。比较沿用分类阶段描述与光栅阶段协作加载两种办法：前者新增显存写读，不能预设更快。目标是减少单 lane 阶段、shared 广播与同步等待，同时观察 register/shared 是否实际下降。
2. **仅在第一步后仍有扫描长尾时，做轻量工作量分桶。** 按已经算出的 bbox 面积/行数划分空或退化、微小矩形、较大扫描三类；优先 cluster 内或局部 wave 重排，避免先引入全局三角形队列及重复 setup。保留稳定 record/triangle ID，工作顺序与可见性 ID 分离；小三角形继续一 lane，较大 bbox 的多 lane 协作作为独立候选。
3. **原子合并暂缓。** 当前没有原子吞吐饱和证据。先取得 SW 独立指令/地址冲突或受控 atomic A/B 证据；只有证明它限制关键路径，才评估 tile 聚合、局部深度裁决与额外 shared/barrier 的净收益。不得用删掉 atomic 后输出无效的快路径当作同质量优化。

验收沿用同 camera/cut/驻留的 legacy 对照，depth/visibility/bin 逐位检查；记录寄存器、shared、active lanes、barrier/scoreboard 与完整光栅成本。构造破损页面、近裁剪、双面、零工作量和 early/late 回归；任何新增队列都须验证容量溢出与不丢工作。固定状态改进通过后再验收 Full 漫游与 30 fps 目标。

## 本次代码与复现

- `METALLIC_SHADER_CAPTURE_SYMBOLS=1`：只开启优化 shader 的路径/行号，不注入 Graphics Capture，也不启用 unoptimized ShaderDebug。
- raster comparison 配置 `profileHoldSeconds`：0 默认；仅第一轮 legacy SW、冻结状态与读回恢复之后等待，范围 0–300 秒，输出 `ProfileReady.json`。
- `METALLIC_VK_PIPELINE_STATISTICS=1`：可选 device feature 与 compute 创建统计 flag，默认关闭；不支持时报告 disabled，不阻止普通渲染。不修改公共 RHI 接口或生产 shader。
- RHI 分类回归在统计模式下额外建立三种真实 SW pipeline，分别以 Disabled/CaptureSymbols 编译并报告；正常回归路径保持原行为。

Release sample/RHI 构建通过；统计模式验证通过；正常模式的分类等价与 hybrid coverage/depth/overflow 两项回归通过。

```powershell
$env:METALLIC_VK_PIPELINE_STATISTICS='1'
build-release/tests/MetallicRhiTests.exe --gtest_filter=RhiRendering.stream_cluster_cull_classify_equivalence
Remove-Item Env:METALLIC_VK_PIPELINE_STATISTICS
python Tools/AnalyzeZorahFullSoftwareProfile.py Documentation/ZorahFullSoftwareProfileResult.json build-release/sw-pipeline-statistics-final.log build-release/full-sw-nsight-profile3 build-release/full-sw-nsight-profile4
```

采集使用已安装官方 CLI `GPU Trace Profiler`，`--architecture "Blackwell GB20x" --metric-set-name "Top-Level Triage" --real-time-shader-profiler --auto-export --set-gpu-clocks unaltered`。第一次未创建输出目录、第二次等待超时均无有效数据，已剔除；修正后两个采样退出码均为 0。最新使用 `--start-after-ms 150000 --limit-to-frames 3 --max-duration-ms 1000 --trace-timeout 360 --no-timeout --time-every-action`，外层进程有 480 秒上限；须核对实际触发落在 hold 窗口内。

证据：

- [分析 JSON](ZorahFullSoftwareProfileResult.json)
- [原始 Nsight Trace](../build-release/full-sw-nsight-profile4/trace/MetallicGPUDrivenSample_2026_09_20_19_45_08.ngfx-gputrace)
- [逐范围 counters](../build-release/full-sw-nsight-profile4/trace/BASE_UNLOCKED/GPUTRACE_REGIMES.xls)
- [采样环境与选项](../build-release/full-sw-nsight-profile4/trace/BASE_UNLOCKED/REPRO_INFO.xls)
- [冻结状态](../build-release/full-sw-nsight-profile4/app/ProfileReady.json)
- [编译统计与验证日志](../build-release/sw-pipeline-statistics-final.log)
- [上一轮精确 SW 对照](ZorahFullPreparedSoftwareRaster.md)
