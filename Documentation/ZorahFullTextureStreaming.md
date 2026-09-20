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

复现 Full 路线：设置 `METALLIC_TEST_ZORAH_FULL=1`、`METALLIC_ZORAH_FULL_CYCLES=1`、`METALLIC_TEST_TEXTURE_ROAM=1`，运行 `MetallicRhiTests.exe --rhi-validation --rhi-bindless --gtest_filter=*zorah_full_first_frame --output-dir <目录>`。

最终定向回归共 6 项通过：KTX2 资源与 streaming、流式材质着色、透射、MASK 阴影、metadata/material preview。新增压力阶段强制 device-local heap 上限为 1 B，已有底图保持可用且没有升级；解除限制后正常细化与冷回收。初版测试误把 graphReserveBytes 配置当作已生效的 reservation，已改用实际 heap 上限注入。

- [五项材质/资源回归及初版压力测试记录](../build-release/texture-streaming/regression.log)
- [修正注入后的 streaming / 压力回归](../build-release/texture-streaming/pressure.log)
- [小场景驻留量](../build-release/texture-streaming/pressure/texture-streaming/residency.json)：495616 → 823296 → 495616 B，新旧共存峰值 912896 B，预算 4194304 B。

Release Sample/RHI 测试构建通过；Full 原生路线、Editor 双切换日志未出现 Vulkan Validation Error；生成配置与仓库预设的三个消费者策略一致；`git diff --check` 通过。
