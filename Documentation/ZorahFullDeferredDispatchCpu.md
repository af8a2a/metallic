# ZorahFull Deferred dispatch CPU 分解

日期：2026-09-23。

## 结论

三轮固定漫游中，Deferred CPU 均值为 5.292–5.716 ms。其中 Update dispatch descriptors 占 3.736–4.118 ms，约为着色 dispatch CPU 的 98.7–98.8%、Deferred 总 CPU 的 70.6–72.1%。实际命令录制仅 0.023–0.024 ms。下一步优先把 Deferred 材质纹理数组接入已有的不可变代际快照缓存。

本次仅增加计时，没有启用新的缓存或修改着色、纹理准入、材质分桶、GPU dispatch 数量。

## 计时边界

VisibilityBufferDeferredPass 复用 ScenePathTracePass 实现。execute 统一收集和发布 CPU scope，提前返回或报错时也会结束并发布已执行的 scope。普通 PathTrace/Realtime 路径不启用新增 recorder。

Deferred 外层拆分场景/环境验证、灯光与采样准备、纹理流送、输出和着色参数、可见性资源、相机和历史、纹理与 LUT 上传、绑定准备、辐射缓存参数、着色 dispatch、历史发布。材质分桶放在绑定准备的子 scope；旧图内联阴影分支单独标记 Record inline shadows，并转发阴影内部 profiler。既有 Texture streaming 子树保留在相同路径。

普通直接 dispatch 和材质分桶 dispatchIndirectBatch 接入 ComputeProgram 的四段 CPU scope：

- Acquire dispatch tables：验证、获取可安全写入的描述符表以及准备 push 数据。
- Update dispatch descriptors：所有资源绑定的描述符更新。
- Prepare dispatch constants：资源表/常量 packet 的分配检查、映射、写入及 flush。
- Record dispatch commands：绑定 heap/pipeline、push 数据、直接或间接 dispatch 及批内切换。

Full 当前使用材质分桶批次；四段计时覆盖整批，不是单个材质类的耗时。SHaRC/NRC 特殊缓存路径本次仍只包含在外层 Record shading dispatch 中，未深入其内部 dispatch。所有新增 scope 都是 CPU-only，不增加 GPU timestamp。

## 三轮结果

条件：RTX 5070 Ti，驱动 616.92；输出 1797×660，内部 1198×440，DLSS Quality、LOD 1.5 px；沿用 Full 编辑器固定 6 米漫游和转向路线，每轮预热 10 秒、采样 30 秒，关闭 validation。三轮均运行同一份加计时的程序，不是优化前后 A/B。

| CPU scope 均值（ms） | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Deferred 总计 | 5.2919 | 5.6507 | 5.7155 |
| Texture streaming | 1.3591 | 1.4209 | 1.3873 |
| Prepare dispatch bindings（含分桶） | 0.0404 | 0.0425 | 0.0429 |
| Record material binning | 0.0338 | 0.0355 | 0.0359 |
| Record shading dispatch | 3.7855 | 4.0727 | 4.1706 |
| └ Acquire dispatch tables | 0.0228 | 0.0227 | 0.0245 |
| └ Update dispatch descriptors | 3.7362 | 4.0226 | 4.1181 |
| └ Prepare dispatch constants | 0.0023 | 0.0023 | 0.0025 |
| └ Record dispatch commands | 0.0232 | 0.0241 | 0.0245 |

描述符更新 P95 为 5.257 / 5.461 / 5.428 ms；着色 dispatch P95 为 5.330 / 5.513 / 5.488 ms。

整帧均值 24.054 / 24.162 / 22.807 ms，P95 为 34.173 / 33.890 / 33.937 ms。未满足持续 30 fps。父子 scope 为包含关系，不应相加；本表各项都覆盖完整采样帧，纹理流送中条件执行子项的均值则需注意各自样本数。

## 下一步实现边界

1. 优先给 Deferred 的 binding 9（完整材质纹理数组）传入 ScenePathTraceResources::materialTextureSnapshot()。源码确认该绑定目前使用原始 view 数组，每次批次都更新描述符；这是结合计时与源码定位出的首要候选，本次没有进一步逐绑定计时。沿用 Shadows 已验证的同代快照命中、换代逐项比较、completion 门控与弱引用缓存。
2. 保持 TLAS、可见性/深度、输出、feedback、灯光及每帧参数等动态绑定的更新。纹理迁移发布新快照后必须采到新代，并继续让在途帧保活旧代。
3. 原有材质批次已经共用一张不可变描述符表，无需再次为每个材质类拆开或重排。表获取、常量准备和实际命令录制合计仅约 0.05 ms，优先级低。
4. 缓存后重跑同路线，检查描述符 CPU mean/P95、纹理换代正确性、冷回收和整帧时间。剩余 1.36–1.42 ms 的纹理调度可在此后单独优化；不能把描述符更新耗时全部视为可兑现的整帧收益。

## 验证与证据

Release MetallicGPUDrivenSample / MetallicRhiTests 构建通过。开启 Vulkan validation 的四项现有 GPU 回归通过：material_binning_indirect_coverage、stream_material_shading、stream_material_transmission、stream_material_shadow，无跳过或验证错误。

三轮共 3806 帧；逐帧验证 Record shading dispatch 及四个子 scope 都存在，GPU 时间为空，四个子项 CPU 总和不超过父项（允许导出精度 0.001 ms）。git diff --check 通过。

```powershell
cmake --build build-release --target MetallicGPUDrivenSample MetallicRhiTests -j 6
.\build-release\tests\MetallicRhiTests.exe --gtest_filter="*material_binning_indirect_coverage*:*stream_material_shading*:*stream_material_transmission*:*stream_material_shadow*" --rhi-bindless --rhi-validation
pwsh -NoProfile -File Tools/RunZorahFullRoam.ps1 -OutputRoot build-release/full-deferred-dispatch-profile-0923 -Runs 3 -DurationSeconds 30 -WarmupSeconds 10 -Width 1797 -Height 660 -TimeoutSeconds 900
```

摘要：ZorahFullDeferredDispatchCpuResult.json。原始数据：build-release/full-deferred-dispatch-profile-0923，含配置、程序/shader 摘要、Frames.jsonl、Summary.json 和 GPU 监视。构建与测试日志：build-release/deferred-dispatch-profile-build.log、build-release/deferred-dispatch-profile-tests.log。
