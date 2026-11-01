---
id: "0069"
title: "Build: .gitmodules 缺少 External/nrd 映射，子模块初始化直接失败"
area: build
kind: defect
priority: P1
state: open
milestone: M2-Correctness
labels: ["area:build", "kind:defect", "priority:P1"]
evidence:
  - .gitmodules
  - External/nrd
---

## 背景 / Context

按 `Documentation/Build.md` 与 `AGENTS.md` 的约定，缺失依赖要用
`git submodule update --init --recursive -- <submodule-path>` 初始化。
但在当前检出上这条路径对 NRD 直接不可用。

## 现状 / Current state

索引里 `External/nrd` 是一个 gitlink：

```
160000 36183520b006b65860bebc96d3dd7185e5edeb45 0   External/nrd
160000 a469709c55b973099c1600b78a503912150e2c69 0   External/microprofile
```

而 `git submodule status` 在扫到它时直接报错：

```
fatal: no submodule mapping found in .gitmodules for path 'External/nrd'
```

`.gitmodules` 里确实没有 `[submodule "External/nrd"]` 条目。

同时 `External/microprofile` 的 gitlink 固定为 `a469709`，而工作区子模块处于
另一个提交（`git status` 显示 ` M External/microprofile`），即指针漂移且未被提交。

## 问题或缺口 / Problem

1. **新检出不可构建**：缺少 submodule 映射时，任何
   `git submodule update --init --recursive` 都会中止，`mapping` 错误会连带让
   后续子模块初始化流程中断，NRD 相关的构建选项（`Source/Runtime/Render/Denoising/`）
   无法被稳定配置。
2. **子模块指针漂移**：`microprofile` 的工作区提交与索引不一致，别人拉下来的
   版本与你本地的版本不同，profile 数据与产物可能不可比。
3. **文档与事实不符**：`Build.md` 描述的子模块初始化流程在这台机器上是坏的，
   而错误信息不会指向根因。

## 期望结果 / Expected outcome

1. 在 `.gitmodules` 中补上 `External/nrd` 条目（URL + 与 gitlink 一致的分支/提交），
   或在确认 NRD 不再需要时移除该 gitlink。
2. `External/microprofile` 的指针与工作区对齐，并把想要的提交正式提交进索引。
3. `Build.md` 记录 NRD 与 microprofile 的初始化与版本约束。

## 验收标准 / Acceptance criteria

- [ ] 干净的 `git clone` + `git submodule update --init --recursive` 无 `fatal:` 输出
- [ ] `git submodule status` 全部条目无 `-` / `+` 前缀（即全部已初始化且与索引一致）
- [ ] `git status --short` 不再出现 `External/` 下的子模块改动
- [ ] 使用 NRD 的构建配置（路径 tracing / 降噪相关 preset）能完成 configure 与 build
- [ ] `Documentation/Build.md` 的依赖小节与此一致

## 相关文档 / References

- [Build.md](../../Build.md)
- [DlssNr.md](../../DlssNr.md)、[NRDColorContract.md](../../NRDColorContract.md)
- [AGENTS.md](../../../AGENTS.md)（子模块初始化约定）
