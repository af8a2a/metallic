---
id: "0018"
title: "Streamer: 初始加载仍在主线程同步完成（loadingSeconds 7.7 s，无进度/取消），未达可交互目标"
area: geometry
kind: feature
priority: P1
state: open
milestone: M3-Hardening
labels: ["area:geometry", "kind:feature", "priority:P1"]
depends_on: []
evidence:
  - Documentation/ZorahFullInitialLoading.md:69
  - Documentation/ZorahFullInitialLoading.md:65
  - Documentation/ZorahFullInitialLoading.md:87-92
  - Documentation/ZorahFullInitialLoading.md:94
  - Documentation/ZorahFullInitialLoading.md:116
  - Source/Runtime/Render/Streamer/MeshletStreamInitialLoader.h
---

## 背景 / Context

`initialLoad`（默认 `true`）把新 stream 的根资源加载从渲染帧中拆出，用独立命令批次上传根页、构建 CLAS 和 fallback BLAS，每批由 GPU 完成点推进。这个改动把 Full 的首次可用画面从 18.6 s 降到 7.7 s。

## 现状 / Current state

- 实测（RTX 5070 Ti、960×540 / DLSS 640×360、缓存已存在、非磁盘冷启动）：`initialLoad=false` 平均 18.618 s / 398 次加载绘制；`initialLoad=true` 平均 7.722 s / 1 次。减少 58.5%。
- 专用加载阶段本身 3312–3377 ms，148 个独立 GPU 提交，上传 3020.5 MiB 设备 payload，CPU 等待完成点累计约 397–400 ms；图编译另需 4060–4276 ms。
- 51,764 个根页与对应 CLAS 在首个图渲染帧之前完成，readiness 的 103,528 步全部完成。

## 问题或缺口 / Problem

1. **整次初始加载仍在主线程同步完成**：`pump()` 的 16 ms 预算是「一次调用追加批次」的上限，单批可超预算，外层循环持续调用直到完成或失败。因此加载期间没有异步 UI、没有交互、没有取消。
2. 每批等待的 30 秒超时、整轮 5 分钟未就绪即失败，这两个阈值不是 UI 响应时间上限；错误后的资源排空仍可能长时间占用主线程。
3. 文档自述「不承诺达到参考程序的三秒」，端到端目标（请求→首次有效画面→细节收敛）也没有统一口径。
4. 缺少「同一图自动 scene refresh 同时替换真实 stream」这一组合的验证，也未做强制 GPU 超时与 UI 异步交互验证。

## 期望结果 / Expected outcome

- 初始加载可显示进度、可取消，主线程单帧阻塞时间有明确上限；加载与图编译并行推进。
- `loadingSeconds` 与「首帧→细节收敛」两项分别记录并对齐成可比较的口径。

## 验收标准 / Acceptance criteria

- [ ] 引入可观测的进度与取消：取消后已接受 GPU 工作排空、未接受命令撤销，session 可重建且不泄漏页/CLAS/BLAS。
- [ ] 记录并断言主线程最长连续阻塞（含图编译），给出目标值；不得用「单批可超预算」作为解释留在文档里。
- [ ] `loadingSeconds` 在相同缓存热条件下与 7.72 s 直接可比，且新增「请求→首次有效画面→1.5 px 收敛」三段计时。
- [ ] `MetallicRHITests --rhi-validation --gtest_filter='RHIRendering.stream_initial_loading:*streamer*'` 全绿，覆盖未提交取消重试、GPU gate 前不发布就绪、reset/reload。
- [ ] `METALLIC_TEST_ZORAH_FULL=1 METALLIC_ZORAH_FULL_CYCLES=2` 的 `zorah_full_first_frame` 通过，asset/world 两入口均检查材质、唯一 stream 所有者与释放。
- [ ] 补跑「同一图 scene refresh 同时替换真实 stream」组合测试。

## 相关文档 / References

- [ZorahFullInitialLoading.md](../../ZorahFullInitialLoading.md)
- [ZorahFullLoadingComparison20260929.md](../../ZorahFullLoadingComparison20260929.md)
- [SceneStreamingSubsystemRefactor.md](../../SceneStreamingSubsystemRefactor.md)
