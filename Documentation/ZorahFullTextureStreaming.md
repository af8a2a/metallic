# ZorahFull T2/T3：按需纹理细化与冷回收

2026-09-20。普通 KTX2 材质图从有预算的基础尾链起步，由实际着色反馈升级；离开视野后退回基础尾链。MiniZorah 默认策略保持不变。

## 数据与调度

- Full：普通图初始上限 128，MASK 基色 512，纹理总 image allocation 预算 512 MiB；VBuffer、Shadows、Deferred 和生成脚本使用相同策略/资源缓存身份。
- Deferred 在材质采样时按 1/8 采样率记录原始 mip 需求和命中次数。使用源图尺寸及既有 ray-cone/UV transform 足迹，与当前尾链的尺寸无关；避免粗图永远无法请求高 mip。此版本使用 ray cone，尚非各向异性 UV 导数虚拟纹理。
- CPU 仅消费已完成 GPU 帧的反馈，最多四份在途缓冲，完成后复用。取消的未提交反馈不参与调度。资源 owner 独立持有异步请求，清理时取消并 join，旧场景结果不能发布到新 owner。
- 按命中数、缺失 mip 和增加字节排序，每次最多提升一级，最高 512。只有近期实际采样的图可升级；不会把全场景同步恢复至 512。
- 默认 180 帧未被使用则退回基础尾链，预算压力下可提前回收至少 30 帧未使用的图。升级/回收之间保留 30 帧冷却（测试可缩短），抑制视角边界抖动。

## 内存与生命周期

使用普通 image 尾链替换，不使用 sparse image。每批最多 8 张 / 4 MiB 新 image，CPU 解码至多两个 worker，解码及输入 scratch 总额 64 MiB；每帧尽量限制 image 准备阶段为 1 ms（单次驱动调用可能超过）。上传独立提交到 graphics queue，完成后才发布逻辑槽位的新物理 image。

总预算计算包含当前 image、待退休旧 image、待创建新 image 的完整大小；另留最多 16 MiB 稳态余量供迁移。每次创建前重新检查共享设备可用额度，分配失败保留已有尾链并延后重试。逻辑 texture ID 和 material buffer 不随 mip 改变。

帧保留不可变的 texture/view generation，新消费者加入上传完成点依赖；旧帧仍引用旧 image，全部引用退役后才实际释放。Raster 只绑定固定的 alpha/displacement 图和 fallback，避免闲置的 raster 描述符引用已回收的 shading image。MASK/BLEND 基色与位移图保持固定尾链，避免主视图反馈缺失导致离屏阴影或几何变化。

回收统计是 image allocation 归还分配器；VMA 可保留空块复用，因此不保证 NVML 整卡占用按相同字节立即下降。共享预算仍依据实时 heap usage。

## 观察入口

Profiler → Streaming → **Texture Residency**（折叠 scope）：常驻/待迁移预留/待退休 MiB、细化图片数、请求数、升级/降级次数、预算延后次数、反馈帧数、累计上传量及请求到发布的最大帧数。图中 pending 表示保守预留，resident/retiring 为实际 image allocation；延迟尚不包含候选等待预算的时间。

## 验证

小规模 GPU 反馈回归检查 128→256→512 的物理 image 替换、未见图片不升级、冷却后回到底图、MASK 稳定、BC5 采样、逻辑 ID 稳定、取消反馈、共享预算拒绝升级及旧新共存预算。Full 场景路线记录前进、离开视野、返回三阶段，并隐藏环境背景防止把背景当成几何。

完整实测结果见下方补充；此功能仍以 512 为细化上限，不等同于原始 4K/8K 全精度纹理或 sparse 虚拟纹理系统。

## Full 实测（960×540 原生，Vulkan validation）

| 阶段 | image 驻留 MiB | 已细化图片 | 累计升级 | 累计降级 |
| --- | ---: | ---: | ---: | ---: |
| 前进后（165 帧采样） | 167.86 | 275 | 429 | 8 |
| 离开视野（405 帧采样） | 116.26 | 0 | 429 | 283 |
| 返回（105 帧采样） | 133.07 | 136 | 599 | 283 |

冷阶段回收 **51.59 MiB**，所有动态细化图回到基础尾链，pending/retiring 都归零；返回后重新细化。该路线请求入队至发布最大 4 帧。全程逐阶段断言 old/new 共存不超过 512 MiB；完整根页仍在第 397 帧就绪。环境背景隐藏后的建筑/材质图像已检查。

路线为原 cfg 相机沿视线前进半个 eye→center 向量（约 4.5 场景单位），随后把相机移至场景上空并朝天以隔离冷回收，再返回原位。朝天阶段是可重复的不可见性试验，不等同于自然漫游路线的峰值性能；并未据此宣称漫游 FPS 提升。

- [Full 逐帧/分阶段数据](../build-release/texture-streaming/full/ZorahFullFirstFrame.json)
- [Full 日志](../build-release/texture-streaming/full.log)
- [前进后着色图](../build-release/texture-streaming/full/ZorahFull-refined-0.png)
- [返回图](../build-release/texture-streaming/full/ZorahFull-return-0.png)

真实编辑器隐藏窗口也通过两轮 MiniZorah→Full、1404×674 完整 DLSS/HDR：两轮分别已有 522 / 677 次升级；第二轮隐藏背景后的实际输出有 603250 / 946296 个有效着色像素。随后主动注入一次预算拒绝，自动重试暂停与显式恢复均通过。[Editor 日志](../build-release/texture-streaming/editor.log)。计数由共享纹理 owner 累计，第二轮可复用同一 owner；不是每轮清零的升级次数。

复现 Full 路线：设置 `METALLIC_TEST_ZORAH_FULL=1`、`METALLIC_ZORAH_FULL_CYCLES=1`、`METALLIC_TEST_TEXTURE_ROAM=1`，运行 `MetallicRHITests.exe --rhi-validation --rhi-bindless --gtest_filter=*zorah_full_first_frame --output-dir <目录>`。

最终定向回归共 6 项通过：KTX2 资源与 streaming、流式材质着色、透射、MASK 阴影、metadata/material preview。新增压力阶段强制 device-local heap 上限为 1 B，已有底图保持可用且没有升级；解除限制后正常细化与冷回收。初版测试误把 graphReserveBytes 配置当作已生效的 reservation，已改用实际 heap 上限注入。

- [五项材质/资源回归及初版压力测试记录](../build-release/texture-streaming/regression.log)
- [修正注入后的 streaming / 压力回归](../build-release/texture-streaming/pressure.log)
- [小场景驻留量](../build-release/texture-streaming/pressure/texture-streaming/residency.json)：495616 → 823296 → 495616 B，新旧共存峰值 912896 B，预算 4194304 B。

Release Sample/RHI 测试构建通过；Full 原生路线、Editor 双切换日志未出现 Vulkan Validation Error；生成配置与仓库预设的三个消费者策略一致；`git diff --check` 通过。

## 2026-09-30：纹理发布与重建输入稳定性

本轮对照的是本地 `E:/vk_lod_clusters` 的纹理与重建实现。参考程序在加载时按预算选择固定 mip 尾链，不在漫游中替换纹理；Metallic 会在上传完成后把 128→256→512 的新尾链发布给着色器。这是几何页换层之外的另一种细节变化，法线、粗糙度和基色都可能参与。Full 仍保留 512 MiB 纹理预算、4 MiB/8 张迁移批次与最高 512 的细化上限；这不等同于参考程序默认约 4 GiB 的固定纹理集。

### 实现

- 新尾链包含旧尾链的全部 mip。发布时在新纹理中将最细采样 LOD 限制到旧的可见 source mip，再用 150 ms 单调活动时间把限制降到零；连续升级沿用当前可见 source LOD。无需旧新两张纹理交叉采样，也不增加常驻尾链或迁移预算。冷纹理降档仍按原有不可见性/预算规则执行。
- 每纹理 8-word feedback 的 word 6 保存浮点采样下限。原图尺寸决定需求，采样下限只决定如何显露已经到达的细节，因此渐变不会抑制下一档需求。冻结和反馈积压时仍提供与当帧纹理 generation 匹配的采样元数据；此时 source dimensions 为零，只关闭需求写入。冻结暂停渐变，恢复首帧不补算冻结时间。
- 实时 Deferred 的普通纹理改用 repeat + 硬件三线性，消除 STF 单随机样本在普通材质上的纹理噪声输入；BC5 法线重建和 KTX sRGB/线性格式语义保留。路径追踪及神经纹理仍使用 STF。`Stochastic Texture Filtering` 可恢复普通实时材质的随机滤波作对照；玻璃显式最细采样始终保持确定性，并同样受发布下限约束。对照时保留后续光照与反馈的随机维度。
- 主射线锥采用解析的内部像素直径：透视 `2*tan(fovY/2)*max(aspect/renderWidth,1/renderHeight)`，正交为 `orthoHeight*max(aspect/renderWidth,1/renderHeight)`。旧邻射线 `acos(dot())*2` 在中心多出一倍足迹，并有高分辨率小角量化问题。采样与原图需求共用该公式；这里是纹理的内部渲染像素足迹，几何误差仍使用显示视口像素。

### 重建比较边界

| 项目 | Metallic Full | 参考程序 |
| --- | --- | --- |
| 当前管线 | SR Quality，NR 默认关闭 | 本地 Full 日志使用 RR |
| 已观察到的尺寸 | 2560×1440 输出，1707×960 输入 | 日志窗口 1648×536，RR 输出 3296×1072，输入 2197×715 |
| 运动矢量 | UV 位移，Streamline scale=1 | 像素位移，直接 NGX scale=1 |
| 色彩顺序 | 线性 HDR → SR → AutoExposure | 路径追踪色调映射 → RGBA8 → RR |

MV 单位遵循不同 SDK 契约，不能互相照抄。负 jitter、去 jitter 的 MV 和正常相机运动时保留历史的实现均已核对；纹理升级不触发全局 DLSS history reset。参考日志的输出目标是窗口线性尺寸的 2 倍，内部采样密度也高于窗口，因此不能仅凭同为 Quality 或窗口相同判断重建质量相当，也未把参考的 RGBA8/RR 顺序移入 Metallic。

Full 跑测配置新增 `stochasticTextureFiltering: false/true` 和 `dlssMode: "Quality"/"DLAA"/"Off"`（也接受现有其他质量枚举）。`Capture.json` 同时记录所选模式、纹理渐变时长和实际输出/内部尺寸。改变滤波或重建模式应重新预热；稳定采样、发布渐变、几何缺页追赶和 SR 历史收敛须分别分析。

### 本轮验证

Release 的 Editor、GPUDrivenSample 和 RHI tests 构建通过。开启 Vulkan validation、bindless 和 Streamline 的定向回归有 14 项通过，覆盖材质、透射、MASK、阴影历史、MV/历史契约及新增测试；另 1 项要求 `--rhi-realtime`，在该批次跳过。新增射线锥测试包含 32 个 GPU 投影用例及高分辨率单调性检查。新增 BC4 测试使用各 mip 颜色不同的真实 KTX2 尾链，验证 128→256→512 发布帧采样值连续、渐变中仍请求原图细节、冻结 180 ms 不推进、恢复首帧连续，以及冷回收后预算回到基线。

| Full 运行 | 固定路线采样帧 | 采样窗口内纹理升级 | resident + pending + retired 峰值 |
| --- | ---: | ---: | ---: |
| 随机滤波对照 | 360 | 712 | 297.07 MiB |
| 硬件三线性 | 360 | 720 | 286.57 MiB |
| 硬件三线性 + validation | 180 | 328 | 147.87 MiB |
| validation + 输出抓取 | 180 | 328 | 147.31 MiB |

四次运行均为 2560×1440 输出、1707×960 内部渲染、SR Quality、NR 关闭，正常结束并完成资源退出。加载失败、请求溢出、BLAS 溢出均为零，纹理共存量低于 512 MiB；两次 validation 日志无 VUID、DeviceLost 或 Streamline dump。360 帧对照使用相同可执行文件、shader 和资产，预热 10 秒，只切换滤波方式；两边都包含本轮射线锥和 150 ms 渐变，不能当作全部修改的旧版/新版对照。shader cache 已预热，未重置系统文件缓存。validation 运行预热 5 秒，验证层开销下的计时不用于性能结论。

已检查 FinalBlit 中央 1280×720 输出区域，RGBA16F 数据全部有限，建筑和材质正常显示。预览使用 Reinhard 映射后编码到 SDR，仅用于检查，原始 HDR buffer 仍保留。单幅图像与升级计数不能证明整段漫游的所有细节跳变已消除，也不能据此宣称帧率提升。

`realtime_clustered_dlss_pipeline` 另以 `--rhi-realtime` 运行时，渲染断言通过，但开启 NR 的变体在全局退出阶段触发 `sl.common.dll` 访问异常及 `VkSemaphore` 泄漏；9 月 26 日的既有日志已有相同签名。因此该组合测试不记为完整通过，NR/Streamline 退出生命周期问题仍需独立处理；Full 的 NR-off 正常退出不能替代它。

- [Full 配置、计数、预算与验证汇总](../build-release/texture-detail-comparison-20260930.json)
- [14 项通过的回归日志](../build-scheduling-release/texture-detail-20260930/regression.log)
- [逐帧纹理采样连续性数据](../build-scheduling-release/texture-detail-20260930/texture-sampling-stability/samples.json)
- [Full 输出预览](../build-release/texture-output-check-20260930/image/Preview-SDR.png) / [原始捕获元数据](../build-release/texture-output-check-20260930/image/manifest.json)
- [NR 组合退出异常日志](../build-scheduling-release/texture-detail-20260930/dlss-runtime.log) / [9 月 26 日同签名日志](../build-scheduling-release/pass-stages-realtime.log)
