# Scene streaming ownership refactor

2026-09-23。目标是将场景资源加载和流送调度收束到 `StreamerSubsystem`，为后续优化建立清晰边界。

## 所有权与调用顺序

| 阶段 | 调用者 / 所有者 | 工作 |
|---|---|---|
| 声明需求 | RenderPass | `sceneResourcesRequired()` 声明资源特性、几何生产者和纹理反馈需求 |
| 准备场景 | RenderGraph → StreamerSubsystem | 获取场景快照、解析 StreamAsset 来源与配置、创建/复用视图流送会话 |
| 帧首 | RenderGraph → StreamerSubsystem | 查询就绪状态、处理完成事件、消费 GPU 请求、冷页回收、上传、纹理细化与反馈缓冲 |
| 渲染准备 | RenderPass `prepareExecution()` | 更新相机、HZB 历史、GPUScene 视图和渲染描述符；无资源加载 |
| 遍历 | RenderGraph → StreamerSubsystem | 页面表更新、LOD cut、需求/预取、CLAS 和 BLAS/TLAS 构建 |
| 绘制 | RenderPass `execute()` | 剔除、软硬分类、光栅、着色以及渲染资源屏障 |
| 帧末 | RenderGraph → StreamerSubsystem | 请求回读与流送统计发布 |

`RenderGraphCompileContext` 不再提供 `SceneResourceManager*`。pass 使用 `PreparedSceneResources`，不能再自行获取或加载场景。普通场景可视化、RTXDI、Shadows、Deferred 和 VBuffer 都改为消费预先准备的快照。光追 cluster 调试 AS 的构建也迁入 Streamer；独立 ImageSample 的图片解码与上传由 Streamer 完成。

渲染用的常量、描述符、HZB、驻留几何绘制布局及内置 OpenPBR 数值 LUT 仍由渲染端维护；这些不涉及场景文件加载、按需请求或驻留调度。保留既有图节点配置字段以兼容已有管线，实际配置解析与加载执行位于 Streamer。

## 生命周期约束

- 每个几何生产者有独立流送会话，防止不同视图消费对方的遍历反馈；材质纹理沿用场景快照共享。
- 图尺寸变化复用流送会话和驻留 cut；场景身份、结构、流送配置或读回模式变化会重建对应会话。
- 多个消费者在同一帧共享同一纹理反馈缓冲；纹理迁移发布和快照代际缓存保持原有顺序。
- 资源在上传完成前不会进入对应 pass 的录制；帧持有纹理代际、图片与几何会话直到 GPU 完成。
- shader 热重载为内部流送管线准备新会话，全部成功后才替换旧 pass；失败保留原图。新增热重载、会话回收检查，并释放 pass 在共享流送 heap 上拥有的描述符。
- 图片首帧上传注册取消回滚，避免未提交帧把上传状态错误标为完成。
- Streaming profiler 数据从子系统统一发布，每个几何生产者每帧一条；CPU/GPU 子 scope 仍嵌套在消费它的图节点下，表示实际命令依赖位置，不能重复相加。

## 验证记录

Release 构建目标：`MetallicGPUDrivenSample`、`MetallicRhiTests`。

验证层回归覆盖普通/流式材质、透射、阴影、场景绑定、同地址场景替换、材质代际、失败准备回滚、独立资产、描述符快照、图片、窗口缩放、帧提交事务、shader 热重载、MiniZorah 和流送会话回收。`SceneBindingTests` 增加了实际资源快照的身份/代际/共享断言；`RealtimePipelineTests` 增加了内部流送 shader 热重载及旧会话释放断言。

边界检查：`python Tools/CheckSceneStreamingBoundary.py`。

Full 实测输出：`build-release/streamer-extraction-final-full/run1/`。1797×660 输出，1198×440 渲染，DLSS Quality，10 秒预热 + 30 秒固定漫游，共 1353 帧。帧耗时 P50 22.03 ms，P95 28.70 ms，P99 31.83 ms，最大 56.48 ms；4 帧超过 33.33 ms。此轮用于重构功能回归，没有独立的重构前同条件对照，不作为性能收益或稳定 30 fps 验收结论。

漫游中每帧恰好一条流送样本，加载失败和请求溢出均为 0；几何驻留页数 61,324–62,605，单帧上传最多 172 页、回收最多 256 页。纹理升级累计 634→1326，降级 12→94，证明视角移动仍驱动几何/纹理加载与回收，未退化为冻结驻留。

日志：`build-release/streamer-extraction-regression.log`、`build-release/streamer-extraction-stream-lifecycle.log`，`build-release/streamer-extraction-final-regression.log`。最终补充批次 10/10 通过，合计 24 项不同测试通过。可选 Zorah Z4 探针没有提供环境变量而跳过；RTXDI/RELAX 整图因当前构建未启用 NRD 而跳过，均不计为通过。
