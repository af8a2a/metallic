# VSM resident caster CPU 优化

2026-10-28，基于 `f0bfd379a`。为 resident 场景缓存不可变 caster 几何包，避免未变化的几何每帧重复变换、hash 和上传。LookDev / Sponza 的完整候选图实测证实 CPU 收益，保留此实现。没有修改 shader、光源、阴影分辨率、页预算、LOD 或默认图；[P5](VirtualShadowMapsP5.md) 的其他默认提升门槛继续有效。

## 定位与实现

P5 的普通计时记录了 Shadows 总 CPU，但未导出已有 profile section 的 CPU 字段。本轮仅在验收 JSON 中补出 `section.cpuMs`，基线与候选均使用同一采集逻辑。初次 64 帧定位中，LookDev 的 caster 准备 / 逐面录制中位数为 13.50 / 15.61 ms，Sponza 为 38.70 / 1.49 ms。定位记录见 `.cache/vsm-cpu-profile-{lookdev,sponza}/`。

这两套 manifest 都是 `metadataOnly=false`：主视图可以走 meshstream，而 VSM 仍按 `scene.hasStreamGeometry()` 选择 resident caster。原路径每帧展开所有三角、变换顶点、更新 bounds、逐字节计算位置和 UV hash，再上传每三角 112 B 数据。LookDev 88,264 三角约 9.43 MiB，Sponza 262,267 三角约 28.01 MiB。直接改用 stream caster 会改变几何来源，本轮保持原来源和全部几何计算。

[VirtualShadowMapCasterCache](../Source/Runtime/Render/Shadow/VirtualShadowMapCasters.h) 由两个 VSM Pass 各自持有。resident 命中时复用完整不可变上传包；stream 每次清除 resident 缓存并调用原准备函数，保持 `cmdPrepareShadowGeometry` 的驻留检查、上传依赖和 usage tracking。原 `prepareVirtualShadowMapCasters` 保留无缓存语义，供独立参考和直接调用。

缓存 key 包含 device、scene resource identity、结构 / 几何变换 / 可见性 / 材质 revision、纹理 publication revision、位移开关 / factor / subdivision，以及完整 displacement texture 槽序列。Scene 的几何和材质公开 getter 为 const，正常 mutation API 发布上述 revision；Deferred meshlet 构建只修改派生数据，不修改原始顶点流。槽长度和缺失 view 改变也会 miss，由原准备函数处理错误边界。

每次命中仍向当前命令录制 retain owner 和全部 descriptor leases。上传包只含不可变 CPU 写入数据，不含阴影深度结果，因此取消录制后仍可复用；缓存替换不会覆盖在途包。资源随 Pass / cache 生命周期释放，device 必须比它们更长寿。失败不替换上次成功快照。两个 Pass 的 `compile()` 仅在 device 改变时清缓存，避免同一场景的相机 / 灯光编辑触发图重编后丢失复用；多灯 Pass 沿用 scene identity 边界释放旧场景包。

几何 / 材质等真实编辑仍重新构建。原 shadow-specific `contentKey` 算法保持不变：例如 RGB-only 材质编辑会重建上传包，但不会额外把原本有效的阴影页标脏。缓存不依赖相机、灯光、filter 或输出尺寸。

## 验证

Release / Streamline ON / NRD OFF 的 `MetallicRHITests` 与 `Metallic` 均构建成功；编辑器可执行文件为 `build-release/Source/Metallic.exe`。没有临时计时插桩留在生产代码中。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| VSM scene | 38 个用例通过 | [场景 JSON](../.cache/vsm-cpu-regression/VirtualShadowMapScene/VirtualShadowMapScene.json) |
| VSM local lights | 88 帧通过，validation 零消息 | [局部灯 JSON](../.cache/vsm-cpu-regression/VirtualShadowMapLocalLights/VirtualShadowMapLocalLights.json) |
| resident / stream content | 50 组 cached / uncached GPU 几何和阴影逐字节相同；14 项错误边界通过；core / synchronization validation 零消息 | [内容 JSON](../.cache/vsm-cpu-regression/VirtualShadowMapContent/VirtualShadowMapContent.json) |
| 编辑与寿命 | 相机 / 灯光复用；变换、visibility、材质、height / alpha publication、细分失效；RGB 不误脏页；取消后重试；timeline gate 下旧包替换和 cache 销毁后仍可正确执行 | 同上，`residentPackageLifetime` |
| 完整 DLSS 图 A/B | LookDev / Sponza 的 A、B 各 64 帧；第 0 / 63 帧最终输出原始 hash 相同，第 16 / 63 帧 shadow / depth hash 相同，Deferred HDR 无非有限分量 | [跨版本对照](../.cache/vsm-cpu-correctness/CrossVariant.json) |

完整图检查开启 core validation，因既有 Streamline 互操作限制关闭 synchronization validation；四个进程均零 validation / SDK 错误，RT / CLAS live 和 peak 分配为零。LookDev 在第 32 帧 resize，第 48 帧恢复并 reload / camera cut；两版本输出保持一致。以上不是动画密集场景或 Full Zorah 的性能验收，也不解除 P5 已记录的曲面质量限制。

## 固定预算 A/B

GPU 为 RTX 5070 Ti 16 GB。使用与 P5 相同的完整候选图、固定场景相机，开启 temporal jitter，输出 768×512、DLSS Quality，实际内部尺寸始终为 512×341。每场景预先固定 3 个 ABBA 区组，共 24 个独立进程；每进程 128 帧，仅排除前 16 帧预热。保留磁盘缓存，每进程新建 preview；没有锁定时钟、关闭用户程序或保证系统 GPU 独占。

初次执行会话中断，只有 3 个完成进程及一个不完整进程，没有完整 ABBA 区组。整个不完整区组排除且保存在 `.cache/vsm-cpu-abba/`，原因记录在 [Interruption.json](../.cache/vsm-cpu-build-identities/Interruption.json)。正式统计使用重新声明的原定 3 区组 / 场景预算 `.cache/vsm-cpu-abba-resumed/`，没有根据测量结果扩样或挑选轮次。

基线 EXE 在优化前归档，候选 EXE 在最终构建后归档，各自源码与二进制 SHA256 独立记录于 [Plan.json](../.cache/vsm-cpu-abba-resumed/Plan.json)。生产源码差异只有 caster cache 和两个 Pass 接入，shader 源文件 SHA256 相同。每次 capture 的 `Identity.json` 描述采集时工作区，不能误当成旧 EXE 的构建来源。大型 cooked assets 延续路径身份限制；小型 scene / graph 引用另记录 hash。

下表耗时为各侧 6 个进程中位数的中位数；收益用每个 ABBA 区组内两次 B 的平均 log 减两次 A 的平均 log，再以 3 个区组计算双侧 95% Student-t 区间（df=2）。帧不作为独立样本，表中收益不直接由两个展示中位数相除得到。

| 场景 | Shadows CPU，A → B | CPU 降幅 [95% CI] | 整图 CPU 录制，A → B | 串行 preview wall，A → B |
| --- | ---: | ---: | ---: | ---: |
| LookDev | 35.282 → 9.728 ms | 72.73% [69.77%, 75.40%] | 36.853 → 11.112 ms | 39.696 → 13.776 ms |
| Sponza | 40.306 → 1.341 ms | 96.68% [96.38%, 96.97%] | 41.899 → 2.608 ms | 44.239 → 4.927 ms |

caster 准备中位数为 LookDev 17.141 → 0.0014 ms、Sponza 38.722 → 0.0023 ms。LookDev 仍有多灯逐面资源和录制成本，未将其全部归因到单一操作，也未扩展为 RHI 资源池改造。普通计时保留 CPU execution snapshot 采集；wall 包含串行 preview 和验收 CPU 统计工作，不能推算编辑器 present loop 的 FPS 或 CPU/GPU 重叠收益。

整图 GPU 为 LookDev 2.205 → 2.175 ms、Sponza 1.891 → 1.907 ms。GPU 降幅区间分别为 [-7.78%, 8.71%]、[-6.24%, 4.63%]，均包含零，没有明确 GPU 改善或回退结论。预先记录的本次门槛是 CPU 降幅下界至少 10%、GPU 回退上界不超过 10%，两场景均满足；这不是 2% GPU 非劣性证明。全进程 live RHI 分配峰值两侧分别相同：LookDev 1092.78 MiB、Sponza 1435.05 MiB；不等同物理 VRAM。

LookDev 的 A 侧进程中位数在 29.60–39.58 ms、B 侧在 8.73–9.96 ms，存在明显运行波动。全部样本保留在区组计算中，完整进程点图和各侧 range/median 警告见 [最终报告](../.cache/vsm-cpu-abba-final-report/report.md)、[CPU/GPU 图](../.cache/vsm-cpu-abba-final-report/CPUAndGPU.png)。报告目录包含逐进程 CSV / JSON、版本信息、原始包 hash、[采集脚本](../.cache/vsm-cpu-abba-final-report/run_abba.py)和[绘图脚本](../.cache/vsm-cpu-abba-final-report/plot_report.py)。已实际运行 verifier 和绘图，检查图例、坐标轴和裁切。

本轮结论为所测 resident 完整图的 CPU 优化成立；动态几何仍按原路径重建，stream 场景性能不外推，默认阴影选择保持 P5 当前状态。
