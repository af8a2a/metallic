# ZorahFull Shadows CPU 分解与描述符缓存

日期：2026-09-23。

## 结论

Shadows CPU 热点是 trace dispatch 中每帧重复写入材质纹理描述符。灯光记录、管线/图像检查不是主要成本。采用按纹理代际复用描述符后，Shadows CPU 均值由 3.406 ms 降至三轮 0.175 / 0.183 / 0.218 ms，降低 93.6–94.9%。整帧均值未明显改善，不能据此宣称已实现持续 30 fps。

## 新增 CPU scope

Shadows 下按顺序记录：Validate inputs and camera、Build light records、Resolve stream resources、Prepare and record shadows、Publish shadow image、Publish shadow parameters、Publish camera history。

Prepare and record shadows 进一步拆为资源有效性、trace pipeline、shadow images、参数准备、图像转换、dispatch bindings/material textures、trace dispatch、denoising、最终状态。trace dispatch 再拆为 Acquire dispatch tables、Update dispatch descriptors、Prepare dispatch constants、Record dispatch commands。CPU 子 scope 发布到现有 profiler/漫游导出，不创建额外 GPU timestamp。

## 缓存边界

| 部分 | 基准 CPU 均值 ms | 决策 |
|---|---:|---|
| 灯光记录 | 0.00079 | 保持逐帧构建；灯光/设置变化直接生效 |
| trace pipeline 检查 | 0.00025 | 已按 shader 变体、纹理数量缓存 |
| shadow images 检查 | 0.00016 | 已按尺寸及取消状态复用 |
| shadow parameters | 0.00225 | 已有安全的参数池；相机和灯光参数逐帧写入 |
| trace dispatch | 3.25697 | 本次缓存 sampled-image 描述符 |
| 输出图像发布 | 0.01156 | 保持每帧图像拷贝、状态转换与依赖 |
| 输出参数发布 | 0.07349 | 保持逐帧上传；目前不作为主要优化目标 |
| 相机历史发布 | 0.00013 | 必须随当前帧更新 |

ScenePathTraceResources 在纹理初始发布和 mip 升降级发布时生成不可变 ComputeSampledImageSnapshot。只有 Shadows 的材质 sampled-image 数组显式启用缓存：同代快照直接命中；新代逐项检查 view 指针及 shared ownership 身份，只重写变化项。普通调用仍逐项写入；它们会清除对应缓存状态。TLAS、深度、参数、输出等动态绑定保持更新。

描述符表沿用既有 completion 门控，GPU 未完成的表不会改写。同一提交中的多次 dispatch 仍使用独立可写表。缓存对快照及 view 仅持 weak_ptr，防止旧 mip 被缓存长期保活；在途帧保活快照及其底层 TextureGeneration。比较控制块身份防止指针地址复用导致错误命中。失败的部分写入不会被标记为整代完成。

## 同条件短程漫游

RTX 5070 Ti，驱动 616.92。输出 1797×660，内部 1198×440，DLSS Quality、LOD 1.5 px、完整材质与阴影；同一固定路线，预热 10 秒，采样 30 秒，性能采样关闭 validation。基准为一轮仅加计时的版本；缓存版三轮。时间驱动路线的实际帧数、流送进展和等待分布会变化，不属于冻结 camera/cut/residency 的逐帧 A/B。

| 指标（ms，除帧数） | 基准 | 缓存 1 | 缓存 2 | 缓存 3 |
|---|---:|---:|---:|---:|
| 样本帧数 | 1318 | 1296 | 1321 | 1325 |
| Shadows CPU mean | 3.406 | 0.175 | 0.183 | 0.218 |
| Shadows CPU P95 | 4.905 | 0.476 | 0.478 | 0.507 |
| trace dispatch CPU mean | 3.257 | 0.032 | 0.032 | 0.036 |
| 描述符更新 CPU mean | 未单独计时 | 0.0198 | 0.0192 | 0.0212 |
| 实际 dispatch 命令录制 CPU mean | 未单独计时 | 0.0088 | 0.0093 | 0.0109 |
| Frame mean | 22.772 | 23.157 | 22.712 | 22.658 |
| Frame P95 | 33.099 | 33.087 | 32.747 | 28.805 |
| Frame P99 | 36.985 | 36.701 | 36.648 | 31.777 |
| 超过 33.33 ms 帧数 | 62 | 56 | 58 | 5 |

Shadows GPU 均值基准为 0.0464 ms，前两轮缓存为 0.0452 / 0.0467 ms，符合本次主要消除 CPU 描述符写入的范围。整帧等待分布有明显波动：Reflex pacing CPU 均值为基准 2.126 ms，缓存三轮 3.849 / 3.339 / 1.107 ms；Deferred CPU 同时由基准 4.240 ms 变为 5.434 / 5.535 / 6.230 ms。因此不把单个 scope 节省的时间等同于帧时间收益，也不单凭第三轮较低 P95 判定稳定 30 fps。

下一步建议给 Deferred 的 dispatch 接入同样的细分计时，确认是否也有重复纹理描述符写入；有证据后复用本次快照机制。Shadows 剩余不到约 0.22 ms，继续缓存灯光记录的优先级很低。

## 验证与证据

Release MetallicGPUDrivenSample / MetallicRhiTests 构建通过。7 项开启 Vulkan validation 的测试通过，无跳过或验证错误：frame_sampled_image_cache、frame_descriptor_snapshots、frame_history_dependencies、frame_two_slot_graph_reuse、frame_submission_transactions、ktx2_texture_streaming、stream_material_shadow。

新增 GPU 回归读取实际 shader 输出，覆盖同代零改写、新代单项改写、普通数组使缓存失效、两个并行在途帧分别使用旧/新代，以及 caller 释放后在途保活、完成帧释放后缓存不阻止旧纹理回收。stream_material_shadow 验证流式材质阴影路径。当前构建 METALLIC_HAS_NRD=0，未验证 NRD/SIGMA 执行分支。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*frame_sampled_image_cache*:*frame_descriptor_snapshots*:*frame_history_dependencies*:*frame_two_slot_graph_reuse*:*frame_submission_transactions*:*ktx2_texture_streaming*:*stream_material_shadow*" --rhi-bindless --rhi-validation
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-shadow-cache-0923 -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

数据摘要：ZorahFullShadowCpuResult.json。原始数据：build-release/full-shadow-profile-0923、build-release/full-shadow-cache-0923（含配置、程序/shader 摘要、逐帧 scope、GPU 监视与日志）。最终回归日志：build-release/shadow-cache-final-tests.log；构建日志：shadow-cache-build.log、shadow-cache-final-build.log。
