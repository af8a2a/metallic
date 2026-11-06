# Issue 起草规范 / Issue authoring conventions

本目录把仓库里「正在推进」和「尚未解决」的功能与缺陷固化为可直接发布到
GitHub 的 issue 草稿。发布前 `Documentation/Issues/manifest.json` 是唯一
机器可读真相来源，`issues/` 下的 Markdown 是正文载体，发布脚本读取 manifest。

## 1. 目录布局

```
Documentation/Issues/
  ISSUE_CONVENTIONS.md   本文件（规范）
  README.md              索引 + 发布步骤（Lead 维护）
  Roadmap.md             长期里程碑路线图（Lead 维护）
  manifest.json          机器可读清单（由 issues/*.md 生成，勿手改）
  issues/NNNN-slug.md    单条 issue：YAML front matter + 正文
Tools/
  PublishGitHubIssues.py 幂等发布脚本（Lead 维护）
```

`manifest.json` 由 `python -B Tools/PublishGitHubIssues.py --sync` 从各 issue 文件
生成，`--check` 校验一致性，`--apply` 才真正调用 `gh` 推送。

## 2. 单条 issue 文件格式

文件名 `NNNN-slug.md`，`NNNN` 为四位编号，`slug` 为 kebab-case 英文短名。

```markdown
---
id: "0012"
title: "VSM: propagated projection mapping 保持默认关闭并缺独立统计确认"
area: shadow
kind: perf
priority: P1
state: in-progress
milestone: M1-Active
labels: ["area:shadow", "kind:perf", "priority:P1", "state:in-progress"]
depends_on: ["0008"]
evidence:
  - Documentation/VirtualShadowMapPropagatedMapping.md:147-189
  - Source/Runtime/Render/Shadow/VirtualShadowMaps.cpp:120
---

## 背景 / Context

## 现状 / Current state

## 问题或缺口 / Problem

## 期望结果 / Expected outcome

## 验收标准 / Acceptance criteria
- [ ] ...

## 相关文档 / References
- [VirtualShadowMapPropagatedMapping.md](../VirtualShadowMapPropagatedMapping.md)
```

字段约束：

- `id`：与本文件编号一致。
- `title`：中文为主、保留英文专有名词；前缀标出子系统，如 `VSM: `、`Streamer: `、`RHI: `。
- `area`：`shadow` | `geometry` | `rhi` | `material` | `editor` | `tooling` | `build`，取一个主值。
- `kind`：`defect` | `feature` | `perf` | `research` | `task`。
- `priority`：`P0`（阻塞全部工作或正确性已坏）| `P1`（近期处理：阻塞他人工作，或正在推进且需决策）| `P2`（已规划，下一步明确）| `P3`（探索性）。优先级只表示"现在要不要动手"的排序，严重程度写在正文，不靠优先级替代。
- `state`：`in-progress`（仓库里已有半成品代码/开关）| `blocked`（有明确阻塞条件）| `open`（确认存在未做）| `proposed`（尚未开工）。
- `milestone`：见第 4 节决策表，只允许 `M1-Active` | `M2-Correctness` | `M3-Hardening` | `M4-Performance` | `M5-Research`。
- `labels`：必须与 `area`/`kind`/`priority`/`state` 自洽，且 `state:in-progress` 只在该 state 下出现。
- `depends_on`：其他 issue 的 `id`，只写真实阻塞关系，不要写成"相关"。
- `evidence`：**必填至少一条**，格式 `路径` 或 `路径:起-止行`。指向证明该问题存在的代码或文档，不接受口头推断。

正文用中英混排，保持简短：每条 40-90 行以内，`验收标准` 必须是可核对的复选框。

## 3. 标签集（发布脚本会创建）

`area:shadow` `area:geometry` `area:rhi` `area:material` `area:editor` `area:tooling` `area:build`
`kind:defect` `kind:feature` `kind:perf` `kind:research` `kind:task`
`priority:P0` `priority:P1` `priority:P2` `priority:P3`
`state:in-progress` `state:blocked`

里程碑用 GitHub Milestone 承载，不再额外加 label。

## 4. 里程碑决策表（长期任务的骨架）

| 里程碑 | 含义 | 收录条件 |
| --- | --- | --- |
| `M1-Active` | 当前正在推进，需要收口或决策 | `state: in-progress`，或有未提交改动/默认关闭的实验分支 |
| `M2-Correctness` | 正确性、确定性、验证可信度 | `kind: defect`，或 `state: blocked` 且影响现有路径 |
| `M3-Hardening` | 生产化：默认开启、内存/生命周期、工具与测试基建 | `kind: task/feature` 已有半成品需要补完才能上线 |
| `M4-Performance` | 有测量支撑的性能跟进 | `kind: perf`，已有 profile 数据但未达标或未采纳 |
| `M5-Research` | 长期探索，尚无实现 | `kind: research`，或 `state: proposed` 的大特性 |

## 5. 质量门槛

1. **可验证**：每条 issue 的 evidence 都能被打开核对；不得虚构行号。
2. **不重复**：同一根因只出一条；跨子系统影响写进同一条的正文。
3. **不夸大**：文档里写"未验证/未采纳/默认关闭"的就照实写进 `现状`。
4. **可执行**：验收标准指向具体命令、测试名、开关或指标，不写"优化一下"。
5. **守住 ownership**：只对本规范指定的文档前缀和源码目录起草；跨界内容记在
   最终报告里交给 Lead 去重，不要自己另立 issue。
