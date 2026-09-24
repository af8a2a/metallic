# RelWithDebInfo 下 Full 首帧 OOM 修复

2026-09-24。用户日志最后的致命错误为 `ClusterLightGrid buffer allocation failed / OutOfMemory`。这不是 C++ 符号或 preset 编译失败：Full 已完成 graph compile，首帧从 1797×660 输出尺寸切到 DLSS Quality 的 1198×440 后，显存预算器拒绝了 64 字节分配。

## 原因与修复

VisibilityBufferPass 原来只在宽高完全相等时复用 VisibilityHybridRasterizer。DLSS 缩小内部尺寸会重新分配整个 rasterizer，包括与分辨率无关的大型 cluster 列表、间接参数、队列和管线；旧 bundle 必须等已提交工作结束才退役。由此产生的额外峰值使 Nsight 注入下本已紧张的预算耗尽。

现在按实际像素缓冲容量和 cluster 容量判断能否复用。容量足够时只修改逻辑宽高，保留缓冲地址、描述符和资源状态。已录制命令的 push constants 保留原尺寸；新帧按新尺寸清理和访问同一缓冲。所有 bundle 准备成功后才更新尺寸。超出容量或 cluster 数增长仍走原来的重建和退役路径。

保留 RelWithDebInfo 的优化、CPU 调试信息、Nsight 默认注入和 CaptureSymbols；未降低几何/CLAS 预算、关闭 DLSS，亦未放宽显存预算保护。此修复同时适用于其他构建配置。

## 验证

- `cmake --preset metallic-relwithdebinfo -DMETALLIC_BUILD_TESTS=ON`，构建 `MetallicGPUDrivenSample`、`MetallicRhiTests` 成功。
- 3/3 Vulkan validation 光栅测试通过：深度/覆盖/队列溢出、稳定 cluster 分桶与间接参数、场景等价。新增检查模拟先按大尺寸分配再缩小/恢复，验证缓冲身份不变、拒绝零尺寸/超容量且不改变当前尺寸，并继续对照 HW 深度与覆盖结果。测试日志没有 Validation Error / VUID。
- 保持默认 `nsightCapture=true`、`shaderDebugMode=capture-symbols`，隐藏编辑器启动 ZorahFull；1797×660 输出、1198×440 内部，2 秒预热和 5 秒短程漫游。`capture_complete`，51 帧，GPU timing 无缺失，退出码 0，无 OOM。DLSS 首次初始化后可用预算为 2,290,220,432 字节（约 2.13 GiB）。这是捕获注入模式的启动验证，没有导出新的 `.ngfx-capture`，也不将其帧耗时当作常规性能基准。

- 同样开启 Nsight 注入，完成两轮 MiniZorah→ZorahFull DLSS 切换并正常退出（0）。两轮各加载 103528/103528 必需页；隐藏环境图后，实际 DLSS 输出中 604550/946296 像素通过几何着色覆盖检查。纹理按需升级和预算内驻留检查通过。末尾故意申请 UINT64_MAX 预算来验证失败后不自动重试及显式重试恢复，日志中的该条 Reservation denied / OutOfMemory 是预期注入故障，测试最终通过。

日志：

- [构建](E:/metallic/build-relwithdebinfo/relwithdebinfo-resize-fix-build-final.log)
- [光栅回归](E:/metallic/build-relwithdebinfo/resize-fix-tests.log)
- [Full 运行日志](E:/metallic/build-relwithdebinfo/resize-fix-full-capture/stdout.log)
- [两轮场景切换](E:/metallic/build-relwithdebinfo/resize-fix-full-switch/stdout.log)
- [Full 结果](E:/metallic/build-relwithdebinfo/resize-fix-full-capture/Capture.json)

边界：这消除了容量内缩放的重复分配；其他进程抢占显存或真正增大容量时仍可能触发预算保护。
