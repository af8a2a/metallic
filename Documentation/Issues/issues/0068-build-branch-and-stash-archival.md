---
id: "0068"
title: "Build: 9 个长期未合并分支与 10 个 stash 缺少归档策略，工作正在腐烂"
area: build
kind: task
priority: P3
state: open
milestone: M3-Hardening
labels: ["area:build", "kind:task", "priority:P3"]
evidence:
  - "git:refs/heads — 9 branches ahead of modernization, 10 stashes"
  - Documentation/ProjectArchitecture.md
  - Documentation/Build.md
---

## 背景 / Context

当前检出的主线是 `modernization_/vsm`（HEAD 2026-11-01），仓库里还有一批
本地分支带着未合并提交，以及 10 个 stash。它们的最后提交时间跨度从
2026-03 到 2026-09，越旧的分支与主线的差异越大，越不可能再合并。

## 现状 / Current state

`git rev-list --count modernization..<branch>` 的结果（2026-11-01 采样）：

| 分支 | 领先提交 | 最后提交 | 主题 |
| --- | --- | --- | --- |
| `modernization_/meshlet-compression` | 3 | 2026-09-29 | 自适应位置编码、关节切线帧、BLAS 工作区内存 |
| `modernization_/gapi-bda` | 1 | 2026-09-26 | 提交信息即 `archived` |
| `modernization_/sky` | 1 | 2026-08-08 | 大气天空渲染通道（已单独立项） |
| `modernization-openpbr` | 9 | 2026-06-19 | NRD 子模块接入与 chess 路径追踪 |
| `editor` | 6 | 2026-05-29 | 管线编辑器整合、编辑器布局重构 |
| `master` | 1 | 2026-04-27 | 用 task shader 重写 meshlet 剔除 |
| `gpudriven` | 12 | 2026-04-19 | cluster streaming 拆分与任务流 |
| `PIX` / `archive` | 1 / 1 | 2026-04-05 / 2026-03-29 | 明确归档用途 |

`git stash list` 另有 10 条（`stash@{0}`–`stash@{9}`），最早可追溯到
`refactor` 与 `master` 分支时期的 WIP，其中若干条涉及已经重构掉的子系统
（HZB 金字塔、ClusterLOD refine info cache）。

## 问题或缺口 / Problem

1. **无法判断哪些工作还活着**：`gpudriven`、`editor`、`modernization-openpbr`
   的目标看起来已被主线现代化重写覆盖（NRD 子模块现已在主线
   `Source/Runtime/Render/Denoising/` 下），但没有书面结论。
2. **stash 沉默丢失风险**：stash 不参与任何测试、没有负责人，容易被误丢。
3. **新成员和 agent 会重复调研**：无法区分"已做过但没合"与"从没做过"，
   直接体现为重复实现与重复性能实验。

## 期望结果 / Expected outcome

对每个分支与 stash 给出三类结论之一，并落成一次清理：

- **rebase 并合并**（例如 `modernization_/meshlet-compression`、`modernization_/sky` 已有独立 issue）；
- **转为 issue 后删除**（保留可检索的意图，丢弃不可用的代码）；
- **直接删除**（确认被主线覆盖或方向已放弃，在提交信息/文档中写明原因）。

## 验收标准 / Acceptance criteria

- [ ] 每个领先分支都在本 issue 或对应 issue 中有结论（保留/转 issue/删除）
- [ ] `git rev-list --count modernization..<branch>` 为非零的分支只剩下有明确 issue 归属的
- [ ] 10 条 stash 全部清空，或被转成 issue/分支并注明来历
- [ ] `Documentation/` 新增一节说明分支命名与归档规则，供后续 agent 遵循

## 相关文档 / References

- [ProjectArchitecture.md](../../ProjectArchitecture.md)
- [Build.md](../../Build.md)
- 相关分支 issue：[0066](./0066-sky-atmospheric-pass-parked-branch.md)、[0067](./0067-geometry-meshlet-compression-parked-branch.md)
