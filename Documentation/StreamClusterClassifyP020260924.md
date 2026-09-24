# streamClusterBinMain P0 实现与测量（2026-09-24）

P0 已作为生产 `streamClusterBinMain` 实现：分类专用共享描述符 + 首个 wave 协作加载。RTX 5060 / 616.92 上，同输入 GPU timestamp A/B 显示大批量分类耗时降低约 13%，1,284 条目降低约 10%。72 条目的小任务增加约 0.35–0.45 μs。这是受控 kernel 微基准结果，尚不是 GPUDrivenSample 整帧或 VisibilityBufferPass 的收益。

## 实现范围

- [StreamClusterClassify.slang](../Shaders/Features/GPUDriven/StreamClusterClassify.slang) 保留选中的五个相机 float4、四个实例变换 float4、五个几何 uint 与 renderCamera 标志，共 168 字节的逻辑字段。移除完整 Params/ActiveGroup 拷贝和通用结构清零；这不等于完整 shader 的共享内存分配量。
- 首个 wave 按字段分摊相机、变换及几何地址读取，每个字段只有一个写入者。字段循环也处理 subgroup lane 数小于字段数量的情况，现有首个组同步发布所有字段。
- 128 threads、三次组同步、coverage/tessellation HW 回退、四 word 分类输入、candidate tag 与 stable bin ABI 均保持不变。选取 render camera 提前到装载阶段，投影运算顺序、jitter、near/far、反射绕序及非法索引回退沿用旧算法。
- 生产入口仍由 `GPUDrivenStreamAsset.slang` 提供，调用端无需改动。旧入口冻结于测试 shader [StreamClusterClassificationProbe.slang](../tests/rhi/shaders/StreamClusterClassificationProbe.slang)，已机械核对其函数体与改动前 HEAD 一致。
- [StreamClusterClassificationTests.cpp](../tests/rhi/StreamClusterClassificationTests.cpp) 增加 legacy 对照调度及可选 GPU timestamp 导出。默认回归也执行 legacy、新版、独立参考三个调度，benchmark 默认为关闭。

## 同输入 A/B

Release C++ 测试程序；shader 两种模式均保留优化。每个模式三次独立进程运行；每个场景预热 8 对，然后测量 40 对新旧 dispatch，每对交替 AB/BA 顺序。计时关闭 Vulkan validation 和 pipeline statistics，使用 GPU timestamp，不含 CPU 提交与 shader 编译。

每对运行复用同一份 cull 输出、相机、页面、bin buffer、间接调度参数和阈值。分类仅覆盖写 candidate tags，重复调度不消费工作列表；每次 dispatch 间有 buffer barrier。最终正式运行分类，再与独立参考比较输出。

下表为三个进程各自中位数的中位数，单位 μs；降低率由表中两个中位数计算。

| 场景 / 有效分类条目 | 普通优化：旧 → P0 | 耗时降低 | Capture symbols：旧 → P0 | 耗时降低 |
|---|---:|---:|---:|---:|
| case 0：73,760，正交、反向 Z、密集 | 674.160 → 588.320 | 12.73% | 678.768 → 590.480 | 13.01% |
| case 5：1,284，透视、反向 Z、HZB | 17.232 → 15.456 | 10.31% | 17.344 → 15.520 | 10.52% |
| case 21：264，透视、float3 payload | 6.080 → 5.504 | 9.47% | 6.080 → 5.440 | 10.53% |
| case 4：72，正交、32px 阈值 | 4.288 → 4.704 | −9.70% | 4.416 → 4.864 | −10.14% |
| case 6：72，正交、render camera/jitter | 4.288 → 4.640 | −8.21% | 4.288 → 4.640 | −8.21% |

case 0 单次进程的降低率范围：普通优化 12.49–12.96%，capture symbols 13.01–13.65%。大任务约节省 86–88 μs；小任务的固定加载/分支成本仍值得后续关注，不能概括为所有 dispatch 都变快。本次未添加基于条目数量切换 kernel 的运行时策略。

边界：这是现有合成回归场景，几何复用单个 128 KiB 页面，输入缓冲采用 HostUpload，反复读取使缓存充分预热。73,760 是有效工作数；现有二维间接 dispatch 会向上补齐线程组，多余组直接返回。测试没有固定 GPU 时钟，也没有复现 MiniZorah capture 的实际几何工作集或 GPU 并行负载。收益不可直接外推到整帧。

[完整统计与源文件 SHA-256](StreamClusterClassifyP020260924.json)，[原始 CSV / 日志目录](../build/classify-p0-20260924/)。六份 timing CSV 共 2,400 个有效测量值。

## 驱动资源统计

通过现有 `VK_KHR_pipeline_executable_properties` 探针取得，普通优化与 capture symbols 结果一致，subgroup=32。

| 资源 | 旧分类器 | P0 |
|---|---:|---:|
| Register Count | 40 | 40 |
| Shared Memory Size | 3,696 B | 3,264 B |
| Binary Size | 22,656 B | 15,616 B |

共享内存减少 432 B（11.69%），二进制减少 31.07%，寄存器计数不变。没有重新测得 occupancy 或 barrier stall 分布，不能据此宣称 occupancy 提升，或将原 profile 的 Barrier 百分比当作可回收时间。探针返回的 Local Memory Size 约为 64 GiB/线程，明显不适合直接解释，未用它推断 spill。

原 Nsight GPU Trace CLI 的 metric-set 阻塞仍存在，详见前一份 [capture 报告](VisibilityBufferNsight20260924.md)。本次使用真实 GPU timestamp 验证收益，未声称已经取得新的 Shader Profiler/Trace。

## 正确性与集成验证

- 22 个用例 × early/late 两阶段，新版和 legacy 分别与独立参考比较，共 88 组结果比较通过。检查 header、candidate tags、稳定 bin 列表、record IDs、HZB retry、间接调度参数和 workload accounting。
- 覆盖正交/透视、正反 Z、render camera/jitter、反射/shear、float3/float4、异常 payload/索引、coverage/tessellation、空列表、纯 HW、metadata fast SW/HW，以及超过 65,535 的二维调度。普通优化和 capture symbols 都通过。
- `stream_metadata_vbuffer`、`stream_reflected_winding`、`stream_metadata_contract` 三个渲染集成回归在关闭 validation 后通过。Bunny 的 HW/async hybrid 比较：49,152 像素，coverage/visibility ID/interior ID mismatch 均为 0；resize/release/reopen 通过。
- 验证限制：本机 validation 层报告不识别 `VK_KHR_device_address_commands` 与相关结构。启用 validation 的分类回归通过但有这些初始化警告；三个渲染集成回归在场景初始化、P0 kernel 编译之前抛出 `0xc0000005`，关闭 validation 后通过。本次不宣称 validation-clean，也未修改 RHI/验证层来绕过该问题。

## 复现

使用 x64 VS Developer PowerShell 和项目已有 CLion CMake 4.3.1 配置、构建；本机普通 PATH 下 CMake 4.2.1 在既有缓存上未正确识别 MSVC features，初次配置失败的日志也保留于 build。

```powershell
& 'E:/VS2026/Common7/Tools/Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
& 'F:/CLion 2026.2.2/bin/cmake/win/x64/bin/cmake.exe' -S . -B build-release -DMETALLIC_BUILD_TESTS=ON
& 'F:/CLion 2026.2.2/bin/cmake/win/x64/bin/cmake.exe' --build build-release --target MetallicRhiTests -j 8

# 输出目录先创建；每次进程使用不同 CSV 文件。
$env:METALLIC_CLASSIFY_BENCHMARK = 'E:/metallic/build/classify-p0-20260924/reproduce.csv'
$env:METALLIC_CLASSIFY_CAPTURE_SYMBOLS = '0' # 改为 1 对照优化 + g2
& build-release/tests/MetallicRhiTests.exe --rhi-no-validation '--gtest_filter=*stream_cluster_cull_classify_equivalence'
Remove-Item Env:METALLIC_CLASSIFY_BENCHMARK
Remove-Item Env:METALLIC_CLASSIFY_CAPTURE_SYMBOLS

# 单独检查资源，勿把此运行作为性能采样。
$env:METALLIC_VK_PIPELINE_STATISTICS = '1'
& build-release/tests/MetallicRhiTests.exe --rhi-validation '--gtest_filter=*stream_cluster_cull_classify_equivalence'
Remove-Item Env:METALLIC_VK_PIPELINE_STATISTICS

& build-release/tests/MetallicRhiTests.exe --rhi-no-validation '--gtest_filter=*stream_metadata_vbuffer:*stream_reflected_winding:*stream_metadata_contract'
```

下一步应在同 camera/cut/驻留的实际 GPUDrivenSample 新帧上，测量 classifier 与完整 VisibilityBufferPass，并重新查看首个同步点的 profile；本次已有 kernel 层面的收益证据，尚未完成这一整帧测量。
