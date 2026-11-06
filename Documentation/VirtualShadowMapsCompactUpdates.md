# VSM 紧凑页变更上传

2026-10-28。本轮在[资源池化](VirtualShadowMapsResourcePool.md)之后，将 CPU 每帧生成的 dense remap 改成每 level 的整数页偏移与 inclusive dirty rectangles；在原有 reset pass 中解码，保持页表和 allocator 协议。CPU 与上传量收益已确认，8 项 GPU 回归通过；**LookDev 整图 GPU 置信区间未通过预设回退门槛，不能把整轮标成无条件通过**。

## 实现与契约

[VirtualShadowMaps.cpp](../Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp) 每个 fine level 上传一个 `uint4 {dx, dy, firstRectangle, rectangleCount}`，后接 `uint4 {minX, minY, maxX, maxY}` 矩形。偏移以有符号 32 位编码，先在 CPU 将原 64 位偏移限制在 `[-axis, axis]`；超出一整层的滚动没有可保留页。Bounds 投影、边界 clamp 与原来的 `±0.0001` rounding 均保留在 CPU。无复用或 force-redraw 时无需生成矩形。

[vsmResetMain](../Shaders/Features/VirtualShadowMaps/VirtualShadowMaps.slang) 用当前页加 offset 求旧页。方向光每层使用完整 axis；局部灯的 mip 仍在固定 `VPA²` stride 中只使用 `axis²` 前缀，其余槽保持 unmapped。支持任意合法的 VPA（包括非二次幂），未替换为仅适合二次幂的位运算。

- 只有旧页实际驻留时才检查 dirty rectangles。矩形使用新页坐标，保持原 dense producer 的失效范围。
- Coarse 页仍映射自身；任何 dirty bounds 或 force-redraw 都使它失效。其历史物理槽为 0，由现有 reset/allocator 生产者保证。
- Reset 将 dirty/remapped 两位暂存在现有 owners buffer 的页排名区间；物理 owners 位于另一段。Mark 不访问该区间，现有 reset barrier 保证 Count 可见。Count 读取标记后把同一位置覆盖成 scan rank。
- 未被当前 receiver 请求但仍驻留的页也保留统计标记。`statistics[18]/[19]` 与原算法一致；旧页本身内容无效和本帧 dirty mapping 仍是两个不同条件。
- 没有新增资源、dispatch、barrier 或 push layout。原资源池按 stride 和容量继续复用；取消、在途 reader 与旧页表寿命沿用原契约。

CPU scope `Build VSM page remapping` 保留名称以便对照，新增 `remapBytes`、`denseRemapBytes`、`remapLevels`、`dirtyRectangles` 工作量计数。紧凑格式大小为 `16 × (levelCount + rectangleCount)`；静态 6 层为 96 B/face，旧表为 `98,305 × 8 = 786,440 B/face`。这是 HostUpload 写入的数据量，不代表整个 pass 的 GPU 内存流量。

## 验证方法

[VirtualShadowMapRemapTests.cpp](../tests/rhi/VirtualShadowMapRemapTests.cpp) 用独立的原 dense CPU 算法对照 production reset/count，覆盖 15 组场景：cold/disabled、单页、正负/整层/超大滚动、非二次幂、inclusive 边缘与重叠矩形、coarse dirty、force redraw、spot/point mip 空槽及每层 128 个矩形。检查全部页表 words、reset scratch flags、全部统计和 indirect reset。原 allocator 测试仍独立检查页分配、owner 与 overflow。

该测试独立构造紧凑输入，不直接调用生产 CPU packer；CPU bounds 投影与 rounding 由 cache/content 场景集成测试补充。大量矩形的合成测试仅证明结果一致，不是动态场景性能验收。

普通计时以已池化实现为 A、紧凑实现为 B；每场景 3 个 ABBA 区组，共 24 个进程，每进程 128 帧，前 16 帧 warmup。CPU attribution 与 validation 关闭。固定相机、jitter ON、768×512 输出、DLSS Quality 内部 512×341，保留磁盘缓存。没有控制 GPU 时钟或保证系统 GPU 独占。

基线 EXE 在改动前归档，HEAD 为 `41d1d96c904e26601399eaebefbb04c0ad5136f0`。运行时 shader 从工作区编译，因此每个进程必须配对冻结的 baseline/candidate shader；仅在进程退出后切换文件，前后校验 hash，`finally` 恢复候选。Capture 的当前 C++ 源码快照不是历史基线二进制的构建证明；历史 receipt 单独保留。两 arm 的运行时源码快照只允许这一份 shader 的已声明差异。

计时统计单位为独立进程中位数，以三个 ABBA 区组的平均 log(B/A) 求双侧 95% Student-t CI（df=2）。预声明门槛为 Shadows CPU 收益下界至少 10%，整图 GPU 回退上界至多 10%；不追加择优样本。固定历史诊断另外检查 cached/force-redraw shadow/HDR、跨 arm 图像、resize/reload/cut，以及稳态池复用计数。

## 上传量与 CPU 归因

[完整诊断对照](../.cache/vsm-compact-check-report/Comparison.json)通过每个进程的 EXE/DLL、shader、其它源码和输入完整性校验。下表为每帧各 face 求和；两个静态场景均没有 dirty rectangles。A 的 dense remap 大小由已记录的 `totalPages × 8` 得到，A/B 的基础 HostUpload 总量均有运行时 counter。

| 工作量 | LookDev，8 faces：A → B | Sponza，1 face：A → B |
| --- | ---: | ---: |
| 页变更上传 | 6,291,520 → 768 B | 786,440 → 96 B |
| 基础 buffers HostUpload 总量 | 6,300,608 → 9,856 B | 791,032 → 4,688 B |
| 稳态 buffer reuse | 86 → 86 | 10 → 10 |
| 稳态 buffer / texture / view create | 0 → 0 | 0 → 0 |

页变更字节减少约 99.988%；基础 HostUpload 总量不包括上层 light table 等其它上传。全部创建事件仍保留在对照 JSON：LookDev 初始两代页表、frame 32 resize 的 views、frame 48/49 reload；Sponza 仅初始两帧。其余帧均为零创建/退役。

[CPU 归因](../.cache/vsm-compact-analysis/CPUAttribution.json)为开启 validation/CPU attribution 后的稳定帧中位数，排除上述初始化与 resize/reload 帧，包含 warmup 中的稳定帧；它用于解释成本变化，不作为普通性能收益估计。

| 诊断 CPU 区间 | LookDev A → B | Sponza A → B |
| --- | ---: | ---: |
| Build VSM page remapping | 0.4408 → 0.0028 ms | 0.1237 → 0.0005 ms |
| Upload VSM buffers | 0.2421 → 0.0129 ms | 0.02855 → 0.00190 ms |
| Prepare VSM buffers（包含上述 upload） | 0.3718 → 0.1329 ms | 0.0481 → 0.0198 ms |

## 普通性能结果与限制

全部 24 个预定进程保留，无失败重跑或样本剔除。Release / Streamline ON / NRD OFF。表中 A/B 为进程中位数的中位数，降幅来自区组 log ratio，两者不直接相除。

| 指标 | LookDev A → B | 降幅 / 95% CI | Sponza A → B | 降幅 / 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Shadows CPU | 1.22400 → 0.48350 ms | 61.61% / [56.78%, 65.89%] | 0.29170 → 0.12890 ms | 54.76% / [50.98%, 58.24%] |
| 整图 CPU | 2.53297 → 1.68255 ms | 34.63% / [24.62%, 43.32%] | 1.47635 → 1.31215 ms | 10.86% / [8.32%, 13.33%] |
| 串行 graph wall | 5.45690 → 4.52960 ms | 17.21% / [11.64%, 22.43%] | 3.73960 → 3.60055 ms | 3.89% / [-1.29%, 8.81%] |
| Shadows GPU | 0.74822 → 0.74239 ms | 0.66% / [0.31%, 1.02%] | 0.23430 → 0.24329 ms | -2.22% / [-14.71%, 8.91%] |
| VSM demand/cache/raster GPU | 0.51052 → 0.50724 ms | 0.45% / [-1.09%, 1.96%] | 0.13155 → 0.14108 ms | -3.81% / [-26.09%, 14.54%] |
| 整图 GPU | 2.32684 → 2.28055 ms | 1.69% / **[-13.61%, 14.94%]** | 1.73224 → 1.74749 ms | -2.64% / [-9.40%, 3.70%] |

两场景均通过 CPU 收益下界至少 10% 的门槛。LookDev 的串行整图 wall 收益也明确；Sponza wall 的 CI 包含零，不能声称已确认端到端提速。

LookDev 整图 GPU 的回退 CI 上界 **13.61%** 超过预设 10%，因此 timing decision 是 **inconclusive_or_regressed**。这是尚不能排除回退，不是已证实 13.61% 退化；也不能用 Shadows GPU 的较小变化替代整图门槛。Sponza 通过本轮预设 CPU/整图 GPU timing gates，但 VSM 构建区间的 CI 仍较宽。没有独立 reset/count 时间戳，不能将 demand/cache/raster 的差值直接当作它们的耗时；不声称 GPU 加速。本轮相对已池化基线，不替代上一轮池化的 GPU 验收保留项。

候选 Shadows CPU 进程范围 LookDev 为 0.4645–0.4905 ms（范围/中位数 5.4%），Sponza 为 0.1245–0.1506 ms（20.2%）。全部离散样本进入上述 CI。[逐 node GPU 进程分布](../.cache/vsm-compact-analysis/GPUDispersion.json)仅供后续定位，未据此删样本或推断原因。

共享 preparation 根区间 CPU 中位数为 0.14625 → 0.13733 ms / 0.15660 → 0.13562 ms；完整 frame begin、pool reset、drain、slot wait 数据保留在报告。它们不全属于 VSM，未把这些嵌套区间重复累加。

RHI 稳态 live allocation 为 LookDev 1053.258 → 1047.259 MiB、Sponza 1392.986 → 1392.236 MiB；包含 warmup 的逐帧观测峰值为 1092.777 → 1086.778 MiB / 1435.053 → 1434.303 MiB。各 domain 的 peak counter 未改变，allocation 数量也未改变。上述是 renderer 分配计数，不是物理 VRAM 测量或帧内瞬时峰值。

[预声明 Plan](../.cache/vsm-compact-abba/Plan.json)、[结果报告](../.cache/vsm-compact-report/report.md)、[Summary](../.cache/vsm-compact-report/Summary.json)与[逐帧数据](../.cache/vsm-compact-report/Frames.csv)可复核。每次进程前后 shader/EXE 配对均通过；[最终恢复记录](../.cache/vsm-compact-abba/ShaderRestoration.json)确认工作区恢复为候选 shader。

![普通 CPU 与 GPU 进程中位数](../.cache/vsm-compact-report/CPUAndGPU.png)

## 正确性结果

`MetallicRHITests` 与 `Metallic` 构建通过，Slang module ownership audit 通过（386 files）。最终 8 项 focused GPU tests 全部通过，core/synchronization validation 零消息：

- [remap](../.cache/vsm-compact-regression/VirtualShadowMapRemap/VirtualShadowMapRemap.json)：15 组 production reset/count 与独立 dense oracle 精确相等；[allocator](../.cache/vsm-compact-regression/VirtualShadowMapAllocator/VirtualShadowMapAllocator.json)：16 组配置、两次执行，页分配/owner/统计与原独立参考相等。
- [scene](../.cache/vsm-compact-regression/VirtualShadowMapScene/VirtualShadowMapScene.json)、[cache](../.cache/vsm-compact-regression/VirtualShadowMapCache/VirtualShadowMapCache.json)、[local lights](../.cache/vsm-compact-regression/VirtualShadowMapLocalLights/VirtualShadowMapLocalLights.json)、[compact lights](../.cache/vsm-compact-regression/VirtualShadowMapCompact/VirtualShadowMapCompact.json)、[filter demand](../.cache/vsm-compact-regression/VirtualShadowMapFilterDemand/VirtualShadowMapFilterDemand.json)：实际 receiver、滚动、灯光身份、局部 mip、六面和过滤等集成路径。
- [content / pool](../.cache/vsm-compact-regression/VirtualShadowMapContent/VirtualShadowMapContent.json)：50 组内容对照、14 个拒绝边界，以及扩大/缩小、取消重试、在途旧页表 reader 和上传保护；两幅在途 shadow 均与强制重绘一致。

四个固定历史诊断进程全部通过：两场景最终颜色 raw SHA256 跨 arm 相同；LookDev frame 16/63 和 Sponza frame 47 的 depth/shadow 跨 arm 相同，各自 cached/force-redraw shadow/HDR 逐字节相同，HDR 无非有限分量。LookDev 包含 resize、恢复尺寸与 scene reload；Sponza 包含 camera cut。已查看两场景候选 PNG。

这些诊断 jitter OFF、core validation ON，完整 DLSS graph 的 sync validation 延续既有 Streamline 边界关闭。它们不覆盖此前 jittered Sponza 的历史差异，原证据仍见 [CPU 细分记录](VirtualShadowMapsCPUProfile.md)。大量动态 caster/dirty rectangles 的 GPU 性能仍待独立验证；现有合成正确性测试不能替代该验收。

## 复现

```powershell
cmake --build build-release --target MetallicRHITests Metallic -j 8
python -B Tools/CheckShaderModules.py
.\build-release\tests\MetallicRHITests.exe --gtest_filter="*virtual_shadow_map_scene*:*virtual_shadow_map_cache*:*virtual_shadow_map_local_lights*:*virtual_shadow_map_compact*:*virtual_shadow_map_filter_demand*:*virtual_shadow_map_allocator*:*virtual_shadow_map_content*:*virtual_shadow_map_remap*" --rhi-bindless --rhi-validation --output-dir <new-directory>
python -B .cache/vsm-compact-scripts/RunABBA.py
python -B .cache/vsm-compact-scripts/ReportABBA.py
python -B .cache/vsm-compact-scripts/RunChecks.py
python -B .cache/vsm-compact-scripts/ReportChecks.py
```

构建需要兼容的 x64 MSVC developer shell。脚本输出目录必须是新目录，不能覆盖历史证据。Shader 编译和计时期间停止并发源码/`.cache` 配置写入。大型 cooked assets 延续路径身份限制；本轮不推断编辑器 FPS，也不覆盖大规模动态 caster 的性能。
