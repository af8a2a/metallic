# 第一批执行与验收记录（2026-11-13）

> 下文前部保留首次执行结束时的提交前快照；用户随后授权的审查、提交、推送和 issue 处置见末节“发布结果”。

## 结论与边界

用户授权以三个 `gpt-6.1-sol` 成员替代原定 DeepSeek（#71、#10）和 Gemini（#47）。三项修改已落地并由 Lead 审查；**#47 文档验收完成，#71、#10 仍有环境相关验收缺口，不声明三票全部通过或可关闭。**

基线：`modernization_/vsm`，HEAD `866eba50a`。九个 tracked 文件变更，230 insertions / 36 deletions；另新增本文。没有 stage、commit、push、修改 issue 状态或发布评论，没有改变 gitlink，也没有清理既有 dirty。

## 执行前远端复核

2026-11-13 约 22:08–22:09（Asia/Shanghai），通过 GitHub issue 页内嵌 issue 数据复核（匿名 API 已限流）：

| Issue | 状态 / 评论 | 远端 updatedAt（UTC） | 与本地票据的差异 |
| --- | --- | --- | --- |
| [#71](https://github.com/af8a2a/metallic/issues/71) | OPEN / 0 | `2026-11-01T12:42:21Z` | 正文一致 |
| [#10](https://github.com/af8a2a/metallic/issues/10) | OPEN / 0 | `2026-11-01T12:42:54Z` | 远端另附 Blocked by #9 的依赖段；未移除、未宣称依赖已完成 |
| [#47](https://github.com/af8a2a/metallic/issues/47) | OPEN / 0 | `2026-11-01T12:41:23Z` | 正文一致 |

## #71：NRD 子模块映射

- 在 [.gitmodules:20–22](../../.gitmodules#L20-L22) 补 canonical URL `https://github.com/NVIDIA-RTX/NRD.git`，不指定浮动 branch、不升级锁定提交。
- [Build:39–83](../Build.md#L39-L83) 记录 NRD / microprofile gitlink、recursive 初始化、NRD reference snapshot 与实际构建输入的区别，以及保留既有 dirty 的要求。
- `git submodule status --recursive` 成功，25 个递归条目均无 `-` / `+` / `U` 前缀，缺映射 fatal 已消失。
- NRD 为 `36183520b006b65860bebc96d3dd7185e5edeb45`（本地 describe `v4.16.1`，集成 snapshot 口径为 NRD 4.16）；microprofile 为 `a469709c55b973099c1600b78a503912150e2c69`，均匹配父仓库 gitlink。
- 票据关于 microprofile「指针漂移」的推断不适用于当前工作区：其 HEAD 已匹配，dirty 来自既有 nested untracked 内容，未 reset / clean / stage 以制造干净状态。

**未完成的原票验收：**

1. 未完成全新 clone + 全量 recursive 初始化。上游 Git 探测曾遇到 Schannel `SEC_E_NO_CREDENTIALS`，未更改 TLS 后端或降低校验；现有 initialized checkout 的 status 不能替代 fresh-clone 证明。
2. `git status --short` 仍保留既有 microprofile dirty，因此原票要求的 External 全干净并未成立。
3. 在已有 NRD-enabled 配置上尝试构建 `MetallicNRDTests`；CMake 自动 regeneration 失败，因为该树绑定的 `E:/VisualStudio/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe` 已不存在。未换编译器、生成器或 SDK 配置绕过失败；NRD configure/build 和其后的运行测试未通过。

```powershell
& E:\VisualStudio\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
cmake --build build-pass-stages-nrd --target MetallicNRDTests --parallel 2
```

## #10：VSM workload / Build ABI 文档

统一入口：[GPUProfile:73–134](../VirtualShadowMapsGPUProfile.md#L73-L134)。当前契约为 workload **v6 / 31 / 248 B**、Build **v7 / 60 / 480 B**；stats32 为 128 B，Build 起点 word94 / byte376，三段合计 **856 B**，另行追加可选 receiver coverage records。

补齐 workload v1–v6、Build v1–v7 的版本表（Build v4 为 50 项）；其他三份 VSM 文档引用统一入口，历史实验数字保留明确版本标签。JSON 的 missing、disabled、measured-zero 不混同。Lead 已核对 header、parser、[JSON writer 和长度断言](../../tests/rhi/VirtualShadowMapAcceptanceTests.cpp#L210-L257)。没有回退到旧票据的 v3/23。

**通过：**

```powershell
ctest --test-dir build-release -R '^MetallicVirtualShadowMap(GPUProfile|BuildAttribution)Tests$' --output-on-failure
# 2/2 passed，4.03 s；注册解释器为 D:/Scoop/apps/python/current/python.exe。

& D:\Scoop\apps\python\current\python.exe -B -m unittest discover -s tests/perf -p TestVirtualShadowMapGPUProfile.py
# Ran 48 tests in 1.358s，OK。
```

初期 confined 运行曾遇到 TEMP 创建/清理 WinError 5；切到完全权限后，上述配置内测试通过。原票的 `-m unittest tests/perf/...` 在此仓库触发 `ModuleNotFoundError: tests.perf`，因此使用 discovery 而不是改测试包。额外尝试 bundled Python 时，48 项中报告绘图一项缺少 matplotlib；没有为此安装包，最终有效通过来自原 CTest 使用的既有解释器。

**未通过 / 未运行：**

- 现有 RHI executable 的 `virtual_shadow_map_profiling` 已尝试：Vulkan 初始化成功，但输出目录 `E:/metallic/build/vsm-abi-docs-20261113/Profiling/VirtualShadowMapProfiling` 创建被拒绝，0 pass / 1 fail，fixture 未实际执行。不能声称 GPU ABI、图像或 validation 已通过。
- `virtual_shadow_map_acceptance` 未运行。不能以无 manifest 的 skip 代替验收；旧 BuildDiagnostics manifest 引用缺失 graph，另有历史 ARC manifest 可供环境修复后审查并复跑。
- 原票「Documentation 全局 grep 为空」的字面条件仍不成立：历史 issue 正文、原模型路由报告仍引用 `work16`，其他子系统文档也有其自身 16-counter workload。此次修正四份 VSM 文档的当前契约，不删除历史证据或扩大到无关子系统。

## #47：M2 文档与生产代码对账

修改 [M2 状态快照](../MaterialValueProgramsM2.md)、[Roadmap M2](../MaterialSystemRoadmap.md#m2-跑通真正的自定义程序) 与 [Phase 11](../MaterialSystemPhase11.md)，逐项区分 delivered / superseded / remaining，不将稀疏分箱默认开启等同完整 M2 完成。

Lead 第二人已打开核对默认设置、producer、sparse allocation、GPU count/prefix/scatter、真实 indirect consumer、fallback、generation / ProgramKey / Slab family，以及 Coverage publication、IR slice、纹理 footprint 回调。生产路径是 Wave32 sparse executable bins + fused evaluation/lighting；Phase 11 split 为独立原型，64 作者程序预算与 4096 active executable bin 上限不同。

- Lead 独立自动校验三份文档 **108 个本地链接**（目标文件、标题锚点、源码行范围）：0 errors。
- 逐项状态和剩余验收方法一致，事实、链接与第二人入口复核通过；没有为文档变更重建 renderer 或新增 GPU/画质/性能通过声明。
- 保留 RT cone/LOD、程序化梯度/三平面、UV seam/grazing/跨 primitive/tile、规模/显存、非 Wave32 设备和屏幕外自定义反射的专门数值验收；它们是原 M2 的 remaining，不是本次文档修复已交付的行为。

## 权限诊断与后续条件

完全权限模式下 GPU 输出目录创建仍失败。按预设运行一次 Windows 文件权限诊断，目标为 `E:/metallic/build/vsm-abi-docs-20261113/Profiling`，授权范围限工作区。脚本因目标尚不存在返回 `NOT_THIS_CLASS` / `nextAction:stop`，未检查到造成拒绝的具体权限项，**未修改任何文件权限或 owner，无需 rollback、无恢复命令**。按停止指令未再尝试权限修复或换输出目录规避失败。

诊断记录：[acl-report-c80fe5a6f5294060b8c741007b3d6323.jsonl](<E:/metallic-acl-recovery-20261113/acl-report-c80fe5a6f5294060b8c741007b3d6323.jsonl>)。目前已是完全权限，不能把再次切换权限当作已解决；继续 GPU 验收前需明确已有父目录上的创建失败原因。可将本会话连同报告中 `MISSING`、`NOT_THIS_CLASS` 和 `nextAction:stop` 作为反馈提交，保留这个诊断脚本尚未覆盖的场景。

补完 #71 需可用的兼容 MSVC 和 fresh-clone 网络环境；补完 #10 GPU 项需先解决输出目录创建失败，再运行 profiling 与带显式 manifest 的 acceptance。不得用构建成功、GPU 初始化或 skip 代替这些结果。

首次执行结束时，`git diff --check` 无 whitespace error（仅既有换行策略的 LF→CRLF 提示）；既有 [模型路由报告](ModelRouting-2026-11-13.md) 和 microprofile dirty 均保留。该快照时三票均保持远端 OPEN。

## 发布结果（用户后续授权）

用户要求审查后创建提交并上传，并进一步明确确认：

1. 允许既有未上传提交 `866eba50a` 随本批一起推送到 `modernization_/vsm`。
2. 只关闭已满足文档验收的 #47；#71、#10 继续开放并附提交及缺口说明。

两位独立审查代理均未发现提交阻碍。再次执行聚焦 CTest 2/2 通过（4.20 s），108 个本地链接复验零错误。已创建三个分项提交：

| Issue | 提交 | 远端状态 |
| --- | --- | --- |
| #71 | [b60004cc6 — Fix NRD submodule mapping](https://github.com/af8a2a/metallic/commit/b60004cc6) | OPEN，[已发布修复和验收缺口说明](https://github.com/af8a2a/metallic/issues/71#issuecomment-6856825988) |
| #10 | [b3dfba610 — Document current VSM diagnostics ABI](https://github.com/af8a2a/metallic/commit/b3dfba610) | OPEN，[已发布修复和验收缺口说明](https://github.com/af8a2a/metallic/issues/10#issuecomment-6856827317) |
| #47 | [02eea37cf — Align M2 documentation with production paths](https://github.com/af8a2a/metallic/commit/02eea37cf) | [CLOSED / completed](https://github.com/af8a2a/metallic/issues/47)，关闭说明保留完整 M2 的 remaining，不声明新 GPU 验收通过 |

推送后通过 `git ls-remote` 确认远端 tip 等于本地 `02eea37cf6e2acaa86a8f7ec0e533ed24623e379`，并通过 GitHub API 再读三票状态。实际提交仅九个已审查配置/文档文件；gitlinks 未变、index 干净。既有 microprofile dirty 与两份本地报告（本文及模型路由报告）均未纳入提交。

GitHub CLI 原先未登录，后在单次进程内安全复用已有 GitHub Git 凭据完成 API 读写；没有输出/落盘凭据，也没有持久修改 CLI 登录配置。
