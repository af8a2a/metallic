# ZorahFull：4 B 紧凑可见记录

日期：2026-09-29。接续逐实例 BLAS 复用和同尺寸重复 resize 修复。

## 结果与范围

streamed visible record 从 16 B 改为 **4 B**，Full 实际分配日志为：

```text
Visible records capacity=33554400 stride=4 bytes=134217600
```

同一逻辑容量下，原布局为 536,870,400 B，本次减少 **402,652,800 B，约 384 MiB（75%）**。不新增 ID 映射缓冲，不调整 candidate 上限、LOD、几何/CLAS/BLAS 预算或实例准入规则。BLAS 构建、复用和回退算法未修改。

这是 V2 的**记录布局压缩**：仍保留稀疏逻辑地址空间，并非此前设想的 8M 条稠密 append records。选择这一方案是因为记录里的实例身份可从已有 active group 取得，source 类型也已由 streamed 命名空间确定；先消除重复字段即可取得约 384 MiB 收益，无需引入稠密容量溢出策略和跨 early/late 的 ID 映射寿命。

## 数据契约

`CompactStreamVisibleRecord::packed`：

| 位 | 内容 |
|---|---|
| 0–4 | 页内 cluster index，0–31 |
| 5–29 | active group index + 1 |
| 30–31 | 原来的双面/绕序两个 raster flags |

group 字段为零表示无效；全零记录不解码成有效 stream group。最大合法 group 来自现有 25 位 visibility record 容量约束，CPU 静态断言覆盖最高 group、cluster 31、flags 3 和零值。全 `0xffffffff` 的 packed word 可以表示有效边界记录，不能把整个 word 当作 invalid sentinel。

消费者先验证记录与 group 范围，再从 group 取得 `gpuSceneInstanceIndex`。原有 visibility ID、resident base、triangle ID 和调试颜色种子均保留。resident `VisibleClusterRecord` 继续使用 16 B。

已同步：

- HW、SW、legacy queue、两遍 HZB 和候选溢出回退共用的记录发布入口。
- standalone stream 解码、共享三角形解码、Deferred 和材质可视化。
- 材质分桶增加已有 active group 的只读绑定，参数 ABI 升为 v3；没有新增 group 副本。
- Composite 调试可视化、CPU readback 测试和 Debug provider 的 4 B 布局/稀疏有效性说明。
- runtime snapshot 导出 `visibleRecordStride`，初始化日志记录实际容量、stride 和字节数。

record 不包含历史 epoch；有效性仍沿用现有契约：只解码当前 visibility attachment 引用的槽位，在对应帧的 group/page 资源寿命内消费。容量不代表 live count；未被当前像素引用的稀疏槽位不可作为有效记录枚举。

## 验证

Release sample 和 RHI tests 构建成功，`git diff --check` 通过。

以下 10 个不同 GPU 回归最终通过，开启 Vulkan validation：

- `stream_cluster_cull_classify_equivalence`：生产入口写入的 4 B 数据逐项核对 group/cluster/flags，保留 early/late、HW/SW、稳定分类比较。
- `stream_indexed_mesh_raster_equivalence`：HW/overflow/legacy queue 附件逐位比较；高位 visibility ID、primitive chunks、绕序、正反 Z、裁剪和 jitter。
- `visibility_debug_stable_geometry_identity`：记录、group、页面和 resident base 重排后的颜色不变，零值/越界记录正确拒绝。
- `material_binning_indirect_coverage`、`material_binning_typed_indirect_coverage`。
- `render_graph_gpu_driven_mixed_producer_render`、`mixed_producer_raster_work_bins`。
- `streamed_realtime_pipeline`：binned/unbinned 材质输出、阴影和 resize/recompile 连续性。
- `zorah_full_first_frame`：MiniZorah→Full、首帧准备和材质输出。
- `minizorah_debug_identity_stability`：1920×1080 实际转视角，meshlet/triangle/LOD 三种可视化；覆盖 221,306 次实例记录位置变化，身份对应颜色保持稳定。

首次九项运行有一项失败：调试稳定性测试已有条件赋值 `i == 0 ? vertex : fragment = ...` 未给 vertex 赋值，导致管线创建 InvalidArgument。补上括号 `(i == 0 ? vertex : fragment) = ...` 后重跑通过；没有降低颜色稳定性断言。

Full settled 与 base-color 两张 PNG 和上一版 `vbuffer-refresh-validation` **逐像素完全一致**，相机、cook 元数据、宽高、properties 和 DLSS 设置一致。这是 960×540、DLSS 关闭的确定性回归，不能外推为所有相机/长时漫游逐像素等价。

证据目录：

- `build-scheduling-release/compact-record-validation/`、同名 `.log`：首轮九项及 Full 输出。
- `build-scheduling-release/compact-record-debug-validation/`、同名 `.log`：修正测试夹具后重跑。
- `build-scheduling-release/compact-record-minizorah-debug/`、同名 `.log`：实际场景身份稳定性与图片。

## Full 编辑器漫游

180 帧绝对相机路线、逻辑 30 秒、预热 3 秒；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、8M candidates。已有 cook/shader/纹理磁盘缓存，新进程，关闭工作量读回与验证层，`diagnosticRun=false`。沿用 `Tools/RunZorahFullRoam.ps1` 和 `build-release/blas-selected-normal-route.json`。

| 指标 | 本轮单次观测 |
|---|---:|
| 帧均值 / P95 | 27.927 / 37.965 ms |
| >33.33 ms | 30/180 帧 |
| BLAS build 均值 | 0.408 ms |
| 动态 live references 峰值 | 5,048,719 |
| BLAS overflow / 页面 IO 失败 / 请求溢出 | 0 / 0 / 0 |
| NVML 整卡峰值 | 13,562 MiB |

本轮未出现 Validation Error、VUID 或 DeviceLost。**尚未达到持续 30 fps**。

此前 resize 修复后的单次记录为帧均值 24.024 ms、P95 34.132 ms；本轮普通计时较高，不能隐藏这一观察，也不能由两次非交错样本判定解码回归。进程监测中的非样例负载不同：dwm/msedge 有记录样本的平均利用率，前次约 0.74%/0.53%，本次约 3.67%/3.13%。这些不是帧级 GPU 独占证明。本轮没有做严格同 cut/residency 的解码耗时 A/B，**只验收记录容量下降与正确性，不宣称帧率收益**。

该缓冲实减约 384 MiB，不等于 NVML 整卡峰值必然降低同样数值；背景进程、分配重叠和其他池驻留仍影响峰值。进一步稠密化及解码成本对照可独立推进。本任务没有运行仅支持 WorkRaster 文本替换的 ExperimentRunner，也不作为 M3 候选验收。

证据：`build-release/compact-record-full-roam/run1/{Capture.json,Frames.jsonl,Summary.json,stdout.log,Gpu.csv,GpuProcesses.csv}`；上级 Manifest 记录二进制/shader 摘要及配置。
