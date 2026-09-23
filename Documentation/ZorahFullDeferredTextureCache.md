# ZorahFull Deferred 材质纹理代际缓存

日期：2026-09-23。

## 改动

ScenePathTracePass 的 Deferred 分支在 binding 9 上传入 sceneResources_.materialTextureSnapshot()，接入 Shadows 已使用的 sampled-image 描述符缓存。快照在本帧 beginTextureStreaming 发布后取得；直接 dispatch 和材质分桶 dispatchIndirectBatch 共用该绑定。普通 PathTrace/Realtime 分支保持原始数组路径；没有快照时也仍可使用原始数组。

沿用已有安全边界：同代快照命中直接跳过整批图像描述符写入；换代后逐项比较 view 及 shared ownership 身份，只更新变化项；完成点保护在途描述符表；当前帧保活快照和底层图像，缓存只持弱引用。TLAS、深度、输出、反馈与参数等动态绑定照常更新。

本次缓存接入只增加 binding 9 的 sampledImages 参数与注释；前一阶段新增的 CPU scope 保留。没有修改 shader、纹理预算、冷回收规则、材质分桶或间接 dispatch 数量。

## 三轮同条件对照

RTX 5070 Ti，驱动 616.92。输出 1797×660、内部 1198×440、DLSS Quality、LOD 1.5 px；相同 6 米固定路线，每轮预热 10 秒、采样 30 秒。基准为上一阶段仅加计时的三轮；缓存版三轮。两组 manifest 中配置和 shader SHA256 相同。时间驱动漫游并非逐帧冻结的 camera/cut/residency A/B，实际流送和等待分布会变化。

| CPU 均值 ms | 基准 1 / 2 / 3 | 缓存 1 / 2 / 3 |
|---|---|---|
| Deferred 总计 | 5.292 / 5.651 / 5.716 | 1.705 / 1.589 / 1.469 |
| 着色 dispatch | 3.786 / 4.073 / 4.171 | 0.085 / 0.083 / 0.076 |
| 描述符更新 | 3.736 / 4.023 / 4.118 | 0.037 / 0.036 / 0.033 |
| 纹理流送 | 1.359 / 1.421 / 1.387 | 1.455 / 1.349 / 1.249 |
| Reflex pacing | 5.239 / 4.907 / 2.703 | 7.653 / 7.729 / 10.184 |
| VBuffer | 9.363 / 9.605 / 9.896 | 10.647 / 10.184 / 9.759 |

描述符更新降低约 99%；其 P95 从 5.257 / 5.461 / 5.428 ms 降为 0.061 / 0.059 / 0.052 ms。Deferred 总 CPU P95 从 8.061 / 8.723 / 8.206 ms 降为 3.947 / 3.563 / 3.397 ms。父子计时为包含关系，不能叠加。

| 整帧 ms | 基准 1 / 2 / 3 | 缓存 1 / 2 / 3 |
|---|---|---|
| 均值 | 24.054 / 24.162 / 22.807 | 24.445 / 24.000 / 25.560 |
| P95 | 34.173 / 33.890 / 33.937 | 33.462 / 33.713 / 33.999 |
| P99 | 38.349 / 36.710 / 38.063 | 37.323 / 36.328 / 36.625 |

局部 CPU 收益明确，但本组没有整帧均值提升，也未达到持续 30 fps。Reflex pacing 等待增加，同时 VBuffer CPU 有波动；这些数据说明不能将描述符省下的时间直接视为整帧收益，并不足以单独确定等待增长的因果关系。后续应联看帧槽等待、Reflex pacing 与 GPU 完成时间；Deferred 剩余热点主要是约 1.25–1.46 ms 的纹理流送。

## 换代与回收验证

缓存组三轮采样共 3652 帧，期间纹理升级计数分别增加 664 / 663 / 642，降级计数增加 82 / 92 / 76，覆盖持续换代情形。待回收纹理字节三轮最低均为 0，峰值均为 2,813,952 字节；结束分别为 0 / 89,600 / 0 字节。第二轮结束仍有少量在途退休资源，不能要求在连续上传中每一帧都清零；数据未表现为旧代持续累积。

Release MetallicGPUDrivenSample 和 MetallicRhiTests 构建通过。7 项开启 Vulkan validation 的现有回归全部通过，无跳过、VUID 或验证错误：frame_sampled_image_cache、frame_descriptor_snapshots、ktx2_texture_streaming、material_binning_indirect_coverage、stream_material_shading、stream_material_transmission、stream_material_shadow。覆盖同代/换代、原始绑定失效、跨帧保活与释放、mip 流送、分桶以及材质渲染。Full 三轮运行未记录 error、VUID 或 DeviceLost。git diff --check 通过。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*frame_sampled_image_cache*:*frame_descriptor_snapshots*:*ktx2_texture_streaming*:*material_binning_indirect_coverage*:*stream_material_shading*:*stream_material_transmission*:*stream_material_shadow*" --rhi-bindless --rhi-validation
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-deferred-texture-cache-0923 -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

摘要数据：ZorahFullDeferredTextureCacheResult.json。基准原始数据：build-release/full-deferred-dispatch-profile-0923；缓存原始数据：build-release/full-deferred-texture-cache-0923。构建/回归日志：build-release/deferred-texture-cache-build.log、build-release/deferred-texture-cache-tests.log。
