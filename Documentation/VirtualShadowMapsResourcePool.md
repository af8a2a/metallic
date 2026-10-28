# VSM 资源复用与池化

2026-10-28。已实现 VSM 局部资源池；稳定负载下 LookDev 每帧 86 次、Sponza 每帧 10 次 buffer 申请全部命中复用，buffer / texture / view 创建计数为零。普通 CPU 与串行整图 wall time 收益已确认；**GPU 验收存在下述限制，不能将整轮标成无条件通过**。

## 实现与寿命

[VirtualShadowMaps.cpp](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp) 的 `ResourcePool` 按用途分槽，复用十个基础 buffers、点光六面索引表和私有纹理。Buffer 按容量、stride、usage、memory location 匹配；纹理按尺寸、格式和用途匹配。外部 graph 输出的 immutable texture view 按底层 allocation 身份和 layer 缓存。

- 复用同时要求所属 frame completion 完成、pool 是 buffer/texture wrapper 的唯一拥有者。上一代页表仍被 `Cache` 或在途 `State::previous` 持有时不能复用。槽数随实际寿命增长，不固定双缓冲。
- `VirtualShadowMapResult::resourceOwner` 保留独立 backing wrappers；其中没有 State/cache 反向链接。`project()` 在实际消费 command 上保留本面 State 和所有 point-face owners，覆盖跨面和延迟投影，避免引用环。
- 提交/取消结果解析后，先释放 coordinator 的旧 State，再选择可用槽。没有 recording frame completion 的独立调用继续直接分配，不将无效 completion 误当成可复用凭据。
- 复用前增加 HostWrite 可见性和 Device buffer 前次访问到 reset 的屏障。Shader、页分配算法、逐面录制顺序和阴影质量设置保持原样。
- Buffer 保留已申请容量；idle 且尺寸不匹配的纹理在每次录制时退役，包括 atlas cache-hit 路径。缩预算不能让旧的大 atlas 永久滞留。删除灯或 scene handoff 继续释放对应 cache/pool；在途命令仍保有必要资源。

`resourcePoolStats()` 区分 create、reuse、eviction 和 retained bytes。Create 是本 VSM 调用 `Device::createBuffer/createTexture/createTextureView` 的次数；eviction 只代表 pool 放弃拥有权，不能冒充实际 native free 次数。`memoryUsage()` 计入池内保留资源并去重；native padding 和整个设备的瞬时峰值由 RHI memory budget 数据补充。

## 稳态零创建与图像

[完整诊断对照](../.cache/vsm-pool-checks/Comparison.json)保留每一帧的计数，包括所有创建事件：

| 场景 | 初始创建 | 稳态每帧 buffer reuse | 之后的创建事件 |
| --- | --- | ---: | --- |
| LookDev，8 faces | frame 0：86 buffers / 8 textures / 24 views；frame 1：8 个第二代页表 | 86 | frame 32 resize：16 个输出 views；frame 48 scene reload：重新初始化；frame 49：8 个第二代页表 |
| Sponza，1 face | frame 0：10 buffers / 1 texture / 3 views；frame 1：1 个第二代页表 | 10 | 无 |

除上述初始化/resize/reload 帧外，LookDev 全部 59 帧、Sponza 全部 46 帧的 buffer / texture / view create 及 pool eviction 均为零。诊断最多 226 sections，低于每 node 256 上限。

两场景基线/候选最终颜色 raw SHA256 相同，比较帧的 depth / shadow hash 跨版本相同；各自 cached / force-redraw shadow 和 HDR 逐字节相同，HDR 无非有限分量。已查看输出 PNG。LookDev 检查 resize、恢复尺寸、scene reload；Sponza 在 frame 16 camera cut。两者 jitter OFF，core validation ON；完整 DLSS graph 的 sync validation 延续既有 Streamline 限制关闭。这不覆盖此前 jittered Sponza 历史差异，相关原始失败仍见 [CPU 细分记录](VirtualShadowMapsCPUProfile.md)。

## 普通性能对比

预先固定每场景 3 个 ABBA 区组，共 24 个独立进程；每进程 128 帧，仅排除前 16 帧。固定相机，jitter ON，768×512 输出，DLSS Quality 实际内部 512×341，磁盘缓存保留。Release / Streamline ON / NRD OFF；ordinary timing 关闭 validation 和 CPU 诊断。未控制 GPU 时钟或保证系统 GPU 独占。

表中 A/B 为进程中位数的中位数；收益与双侧 95% CI 从三个 ABBA 区组的平均 log(B/A) 计算，两者不直接相除。帧不是独立统计样本，没有删掉离散较大的进程或追加择优采样。

| 指标 | LookDev A → B | LookDev 降幅 / 95% CI | Sponza A → B | Sponza 降幅 / 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Shadows CPU | 9.2618 → 1.2136 ms | 86.72% / [85.85%, 87.53%] | 1.4018 → 0.2843 ms | 79.61% / [76.48%, 82.32%] |
| 整图 CPU | 10.5867 → 2.4745 ms | 76.34% / [75.05%, 77.57%] | 2.6543 → 1.4923 ms | 43.98% / [33.03%, 53.14%] |
| 串行 graph wall | 13.3866 → 5.2606 ms | 60.14% / [58.62%, 61.61%] | 4.9360 → 3.7667 ms | 24.33% / [13.15%, 34.07%] |
| Shadows GPU | 0.6974 → 0.7481 ms | **-7.45% / [-8.13%, -6.76%]** | 0.2331 → 0.2369 ms | -2.54% / [-5.79%, 0.61%] |
| 整图 GPU | 2.2844 → 2.2980 ms | -2.12% / [-4.76%, 0.46%] | 1.7489 → 1.7319 ms | 3.22% / **[-10.25%, 15.05%]** |

CPU 收益通过预设下界至少 10% 的门槛。LookDev 整图 GPU 通过回退上界不超过 10% 的门槛，但其 Shadows GPU 已测到约 0.051 ms 增加；新增同步是可能来源，未做独立归因。Sponza 整图 GPU 的回退 CI 上界为 **10.25%**，略超门槛，因此该场景完整 timing decision 为 **inconclusive_or_regressed**，不是已经证实 GPU 退化，也不能改阈值宣称通过。保留当前 CPU 实现及这项待确认边界，不声称 GPU 加速或编辑器 FPS 收益。

帧准备根区间 CPU 中位数 LookDev 0.14165 → 0.14467 ms、Sponza 0.13730 → 0.13627 ms；frame begin 均约 0.0003 ms，command pool reset 约 0.00213 → 0.00180 / 0.00165 → 0.00160 ms。完整准备、drain、slot wait 分布另列在报告，未观察到原先数毫秒释放成本被转移到这些边界。它们是整个 renderer 的共享准备，不能全部归给 VSM。

设备稳态 live allocation bytes 两版本一致：LookDev 1053.258 MiB、Sponza 1392.986 MiB；逐帧观测的 live 峰值也一致，为 1092.777 / 1435.053 MiB。`Other` domain 的 peak allocation counter 分别从 143.797 → 133.918 MiB、64.870 → 59.331 MiB；不能把不同 domain 的峰值相加当作同一时刻的物理 VRAM 峰值。

候选 Shadows CPU 的进程范围/中位数离散度为 13.6% / 13.2%，完整保留在 CI 中。[报告与过程数据](../.cache/vsm-pool-report/report.md)、[Summary](../.cache/vsm-pool-report/Summary.json)、[逐帧数据](../.cache/vsm-pool-report/Frames.csv)和[预声明 Plan](../.cache/vsm-pool-abba/Plan.json)可复核。

![普通 CPU 与 GPU 进程中位数](../.cache/vsm-pool-report/CPUAndGPU.png)

## 正确性与复现

`MetallicRHITests` 与 `Metallic` 构建通过。最终代码的七项 focused VSM tests 全部通过，core/synchronization validation 零消息：

- [scene](../.cache/vsm-pool-regression-scene-final/VirtualShadowMapScene/VirtualShadowMapScene.json)：resident/stream、真实 receiver、溢出和投影。
- [cache](../.cache/vsm-pool-regression-final-rest/VirtualShadowMapCache/VirtualShadowMapCache.json)、[filter demand](../.cache/vsm-pool-regression-final-rest/VirtualShadowMapFilterDemand/VirtualShadowMapFilterDemand.json)、[allocator](../.cache/vsm-pool-regression-final-rest/VirtualShadowMapAllocator/VirtualShadowMapAllocator.json)。
- [local lights](../.cache/vsm-pool-regression-final/VirtualShadowMapLocalLights/VirtualShadowMapLocalLights.json)、[compact](../.cache/vsm-pool-regression-final/VirtualShadowMapCompact/VirtualShadowMapCompact.json)：六面、seam、跨面投影、灯光身份和预算变化。
- [content / pool](../.cache/vsm-pool-regression-final/VirtualShadowMapContent/VirtualShadowMapContent.json)：50 组内容对照、14 个拒绝边界及新增池测试。8 个稳定帧零创建；扩大/缩小、取消重试；阻塞旧 submission 后录制新帧，确认已完成 producer 的页表仍被在途 reader 保护、上传未覆盖、renderer 销毁后两幅 GPU shadow 仍各自等于强制重绘。GPU statistics 确认两帧实际走缓存复用且有 dirty bounds。缩容量后 retained texture bytes 从 2,229,349 回到原来的 1,111,141。

早期尝试保留在 `.cache/vsm-pool-regression`（local-light 后续 render 失败，未捕获明确根因）；最终重新构建版本该测试通过。另一轮 scene 回归因并发新增 `.cache` 配置触发 `source epoch notification advanced: E:/metallic/.cache` 中止编译，日志保留在 `.cache/vsm-pool-regression-final-rest.log`。停止并发写入后单独 scene 回归通过。它们没有进入普通计时样本。

```powershell
cmake --build build-release --target MetallicRHITests Metallic -j 8
.\build-release\tests\MetallicRHITests.exe --gtest_filter="*virtual_shadow_map_scene*:*virtual_shadow_map_cache*:*virtual_shadow_map_local_lights*:*virtual_shadow_map_compact*:*virtual_shadow_map_filter_demand*:*virtual_shadow_map_allocator*:*virtual_shadow_map_content*" --rhi-bindless --rhi-validation --output-dir <new-directory>
python -B .cache/vsm-pool-scripts/RunABBA.py
python -B .cache/vsm-pool-scripts/ReportABBA.py
python -B .cache/vsm-pool-scripts/RunChecks.py
python -B .cache/vsm-pool-scripts/ReportChecks.py
```

脚本默认输出目录必须不存在；重跑应换新目录或复制脚本修改输出路径，不能覆盖现有证据。运行 shader 编译/计时期间不要并发改写源码或新增 `.cache` 目录。A/B 二进制、DLL、配置与源码 hash 均保留在 Plan 和 manifests；大型 cooked assets 延续路径身份限制。本轮没有改 shader 或全局 RHI allocator，没有编辑器交互及大规模动态场景性能验证。
