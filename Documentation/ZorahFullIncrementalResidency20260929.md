# ZorahFull：增量驻留维护与根 cut 审计（2026-09-29）

## 已落地

资源维护仍由 Streamer 管理。`MeshletStreamResidencyManager` 将每批反馈遍历全驻留集合的需求更新改为：

- 完整反馈发布共享需求 epoch；持续热页查询年龄时继承该 epoch，无需每帧写入 `lastUsedFrame`。
- 显式 unused 集合与上一集合比较，只在冷热转换时物化时间和更新集合。仍需要读取反馈中的 unused ID、检查上一 unused 集合；不是 O(变化页数) 的零扫描算法。
- 预取命中维护独立的驻留预取集合，避免漏计或重复计数。
- 冷回收/预算驱逐候选来自已确认 unused 的集合，不再扫描全部驻留页。未收到需求反馈的旧式显式请求模式保留原来的候选策略。
- 页面离开驻留、重新驻留、释放和 reset 时同步维护集合与 epoch；卸载、CLAS/BLAS 退休和在途完成点规则不变。
- 过期生产者反馈不覆盖较新需求；截断反馈只能使明确 unused 的页成为候选，遗漏页受到保护且不虚构命中时间。无生产者编号的兼容请求按当前 CPU 帧规范化，防止与带编号反馈混用时年龄倒退。

新增 `demandTransitions / demandEpochUpdates / demandStaleBatches / demandMembershipTests / demandPrefetchVisited / coldCandidateTests`，导出到编辑器与漫游 JSON。`demandVisited` 表示实际需求分类访问，预取访问单独计数；原调试字段 `frameResidentDemandCount` 改名为 `frameResidentDemandTransitionCount`，避免将转换量误当成全部热页数。

代价：CPU `PageEntry` 从 64 B 增为 72 B，另有 unused/预取集合；约 61K 驻留页对应新增约 0.47 MiB 的逐页字段（不含非驻留跟踪项及集合分配）。不增加 GPU 工作缓冲。不改变 LOD、驻留预算、根集、几何或属性。

## 验证与计时

Release sample、MeshletCook 与 `build-scheduling-release/MetallicRhiTests` 均构建通过。

19 项 Streamer 回归通过，包括预算准入、延迟卸载、联合冷回收、页请求/补丁、上传完成和延迟统计。新增覆盖：

- 96 批完整/截断/重复反馈及同帧多批，与原逐页刷新年龄语义逐项对照。
- 稳定热集合零逐页需求访问；旧反馈不撤销新需求。
- 同一 page ID 卸载后重新驻留，旧视图既不能标冷，也不能刷新新驻留年龄。
- 预取页实际上传后，旧/截断/unused 反馈不产生命中；首次有效热反馈只计一次。

Full 首帧与 streamed realtime pipeline 两项验证通过。960×540、DLSS 关闭的 Full settled/base-color 图与上一版 `compact-record-validation` 逐像素一致。正常漫游与 Vulkan validation 回归分开执行。

最终源码二进制三轮普通计时：`build-release/incremental-residency-epoch-roam/run{1,2,3}`。每轮 180 帧，绝对相机路线、逻辑 30 秒、预热 3 秒；输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px、8M candidates，已有 cook/shader/纹理磁盘缓存，新进程。采样期间未运行本任务的编译或其他 GPU 测试；未取得整机独占保证。

| 指标 | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 驻留页均值 | 61,396 | 61,400 | 61,401 |
| 需求分类访问/帧均值 | 163.71 | 163.44 | 163.46 |
| 上一 unused 集合检查/帧均值 | 45.14 | 44.69 | 44.75 |
| 冷候选检查/帧均值 | 160.32 | 160.05 | 160.06 |
| Update resident demand 均值 ms | 0.0372 | 0.0317 | 0.0280 |
| Update resident demand P95 ms | 0.1283 | 0.0804 | 0.0749 |
| 整帧均值 ms | 25.701 | 25.020 | 25.299 |
| 整帧 P95 ms | 33.957 | 35.077 | 35.258 |
| 超过 33.33 ms 的帧数 | 12/180 | 21/180 | 24/180 |
| 页面驱逐计数合计 | 20,731 | 20,765 | 20,755 |
| BLAS overflow / IO 失败 / 请求溢出 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

历史同路线 `compact-record-full-roam/run1` 的 Update resident demand 为均值 1.388 ms、P95 1.941 ms。此次访问量与该范围的耗时明显降低，但未执行旧新二进制交错 A/B，且驻留集合、后台负载并非完全相同，不将该阶段的差值声明为整帧加速比。最终三轮仍未达到持续 30 fps。

保留前序证据而不挑选最好结果：

- `incremental-residency-full-roam` 与构建有重叠，P95 43.96 ms，不作为正常性能验收。
- `incremental-residency-final-roam` 是补齐无编号反馈规范化之前的三轮版本，P95 27.90/28.47/28.04 ms，540 帧中 2 帧超预算。最终版本使用上表，未用较好旧轮次覆盖最终结果。
- Manifest 保存每次二进制/shader/config 摘要。正常反馈工作量不依赖强制 shader 工作量读回。

原始结果：`build-scheduling-release/incremental-residency-epoch-validation.log`、`incremental-residency-epoch-full.log`；最终漫游的 `Capture.json/Frames.jsonl/Summary.json/Gpu.csv/GpuProcesses.csv`；工作量摘录 `build-release/incremental-residency-epoch-work-summary.json`。

## 根 cut 审计

`MetallicMeshletCook --inspect` 现导出 `rootCutAudit`：复用运行时 topology 校验与实际 device payload 大小函数，覆盖所有被实例引用的 primitive，独立于相机及 instance.visible。按 terminal flags 统计全部层级的根，不把最高 LOD 当成完整根 cut。

只读命令（`--output` 在 inspect 模式为输入 cook；compact shading 仅作用于私有内存目录）：

```powershell
.\build-release\Source\MetallicMeshletCook.exe --inspect --compact-shading `
  --output Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin `
  --report build-release/root-cut-audit-full.json
```

运行时 open 校验细化引用范围、所属 primitive、LOD/误差顺序及 terminal flags。另核对逐 primitive / 逐 LOD 汇总、实例乘数和根三角形不超过 LOD0 的一致性。未遍历解码所有 payload，未重 cook 或改写正式资产。

| 项目 | 审计值 |
|---|---:|
| 根页 / clusters | 51,764 / 1,154,901 |
| 根页三角形 | 50,326,944 |
| 实例化根 groups / clusters | 512,130 / 11,519,969 |
| 根页 device 字节，256 B 对齐 | 3,173,527,552 B = 2.956 GiB |
| LOD0 根字节 | 1,993,576,192 B，62.82% |
| 低于各自最高 LOD 的根字节 | 1,783,538,432 B，56.20% |
| 完全没有更高 LOD 的 primitive 根字节 | 1,091,563,776 B，约 1.017 GiB |
| OPAQUE / MASK 根字节 | 3,155,846,400 / 17,680,640 B |

LOD0 与“低于最高层级”集合有交集，不可相加。以上是 device payload 请求容量，不是实际整卡 VRAM；几何池仍为原配置 3.5 GiB，扣除根约剩 0.544 GiB。

| primitive（源索引） | 源模型 | 根 MiB | 根三角形 / LOD0 |
|---|---|---:|---:|
| 5152（6512） | Courtyard Dome A1 Top A4 | 146.71 | 99.57% |
| 4108（5046） | Courtyard Part0 Cylinder 022 | 110.54 | 81.53% |
| 2624（3013） | Courtyard Part0 Cylinder 007 | 110.54 | 81.52% |
| 820（905） | ThroneRoom CircularWindow A1 | 84.39 | 100% |
| 2186（2431） | ThroneRoom CircularWindow A2 | 84.39 | 100% |

前三项合计 367.79 MiB，均为 OPAQUE；本次根负担不能主要归因于透明叶片。按完全相同的源 accessor/index 身份统计，重复几何根字节的条件上界约 48.19 MiB，远小于早停根负担；材质/alpha/构建兼容仍需验证，不能直接算成可释放量。

当前 `scene.cpp` 在 group 简化输出仍大于输入 85% 时停止该分支；成功收敛到单 cluster 也产生 terminal。法线/UV seam、切线手性受到保护，有属性时关闭 sloppy fallback。**V9 没有序列化停止原因**，本次不能进一步断言某一根由 UV、法线或 partition 边界造成，更不能直接放宽这些保护。

## 下一步：带诊断的局部 cook

新增 `Tools/PrepareRootCutProbes.py`，基于审计的 sourcePrimitive 映射，复用已有 Zorah 探针 remapper。已生成前三项独立 glTF 到 `build-release/root-cut-probes/`，保留源节点变换、属性、材质与纹理引用，不复制/修改正式 payload。源三角形分别为 1,733,192 / 3,678,446 / 3,678,446；检查材质在索引还原后相等、属性 accessor 规格及外部资源路径有效。**本轮只生成探针，尚未重 cook。**

```powershell
python Tools/PrepareRootCutProbes.py `
  --source Asset/ZorahFull/zorah_textured_public.v1.gltf `
  --audit build-release/root-cut-audit-full.json `
  --directory build-release/root-cut-probes
```

推荐接着在这些探针的 cook 中记录每层简化尝试数、输入/目标/实际三角形数、85% 阈值失败数、单 cluster 收敛数、position remap 重合率、各属性 protect 位及 partition 边界锁定率。先定位“属性硬边过密 / 源几何连接性 / 分组边界”中的实际限制，再决定保真简化修正或独立粗代理。优先验证穹顶，再比较两组圆柱；不先下调几何池、不删除 terminal 分支。

汇总证据：[ZorahFullRootCutAudit20260929.json](ZorahFullRootCutAudit20260929.json)。全 primitive 明细保留在 build 输出。
