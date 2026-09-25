# M3：单候选自动实验闭环

2026-09-25。实现入口为 [ExperimentRunner.py](../Tools/Perf/ExperimentRunner.py)，
使用已通过 M2 的 MiniZorah 非零 late case。Full Zorah 的纹理驻留不确定性仍是
独立阻塞项，不通过缩小质量或忽略身份字段把它纳入合格基线。

## 已实现的流程

候选 JSON 明确假设、单文件范围、基线 SHA-256 和带匹配次数的替换操作。runner
保存完整源码清单与候选 patch，构建宿主样例，通过生产 Slang 路径编译 shader，
检查每次真正绑定的 SPIR-V，运行交错 A/B，然后恢复基线文件。任何路径、上下文
或并发修改冲突都不能无条件覆盖原文件。

正常计时使用独立进程，默认三组 ABBA，共六个相邻进程对。比较对象是每个进程
内部各帧的中位数，未将帧数当成独立统计样本。初测通过后才运行另外两组 ABBA
进行独立确认。判定保留 `accept / reject / inconclusive`；无论结果如何，均不
自动把候选安装为生产版本。

接受门槛固定为：软件 early+late 总耗时改善的 95% 区间下界至少 3%，整图耗时
改善的区间下界至少 -2%，且每一侧均通过 case 原有的 A/A 稳定性限制。正确性
要求所有诊断检查点的 depth/visibility 与基线逐字节一致；已记录的输入、工作
列表、驻留和绑定字段必须一致，只允许候选 SPIR-V 指纹变化。具体故障分类、
命令、证据边界和恢复说明见[工具契约](../Tools/Perf/ExperimentRunner.md)。

首个候选为[共享顶点 uint3 暂存](../Tools/Perf/Candidate.CompactWorkVertices.json)。
它保留屏幕坐标及 depth 原始位模式，移除未读取的第四分量。减少数组声明大小
不等于已证明减少最终共享内存分配或提升 occupancy；这些只是待测假设。

该候选的源码证据 ID 为基线文件 SHA-256
`153788a450366a61c63ae1f1623ac66a5175f87bbba288c1917b523752b2bbac`，内容与
候选 diff 一同归档。此前 [M2 Nsight 映射](../build/nsight-m2-ui-20260925-01/Validation.json)
用于确认生产 WorkControl 入口；这个具体假设来自未读取分量的源码检查，没有
把 Nsight samples 当成可回收时间。输出不等价会直接证伪；性能收益不能通过预先
声明的 3% 门槛及独立确认，则不会接受。

## 实机验收

首次 [12 进程批次](../build/m3-compact-vertices-20260925-01/Decision.json)由机器判为
`inconclusive / process_or_environment_evidence_missing`：PDH 查询未包含随后创建
的 Metallic GPU 实例。12 次进程正常退出，输出一致，但不作为性能验收结果。

改为重新创建 PDH 查询后，[第二批](../build/m3-compact-vertices-20260925-02/Decision.json)
在首个基线进程即停止：查询重建约 5 秒，三个测量窗口仅覆盖一个。逐轮覆盖检查
没有放宽。前两批均已完成离线完整性复核并自动恢复原始 shader。

最终采样器等待目标 GPU 实例出现，再打开连续查询。目标零利用率样本也保留，
每个测量窗口必须有目标进程记录。[第三批完整实验](../build/m3-compact-vertices-20260925-03/Manifest.json)
完成 12 个独立进程、6 个相邻 A/B 对，每进程 3×32 个目标帧。全部进程 exit 0，
36 个测量窗口均有 PDH 覆盖，depth/visibility 在全部诊断检查点逐字节一致。
输入不变量一致；两侧绑定的 SPIR-V FNV 分别为 `14699073418322314354`、
`5624891983249778816`，侧内保持稳定。

| 指标 | 基线 A：6 个进程中位数的中位数 | 候选 B：同一统计量 |
| --- | --- | --- |
| Software early+late | 0.176192 ms | 0.196464 ms |
| Graph GPU | 4.040496 ms | 4.055240 ms |

按六个独立进程对计算，software 耗时平均增加 **11.4827%**，95% 区间为增加
**11.0588%–11.9066%**。Graph 配对收益区间为 -0.9196% 至 +0.0713%，通过 2%
回退预算。两侧原有 A/A 稳定性检查均通过；软件收益明显低于门槛，因此机器输出
[`reject / gain_below_threshold`](../build/m3-compact-vertices-20260925-03/Decision.json)。
这不是成功优化；闭环成功阻止了一个源码上看似合理、实际更慢的改动进入生产。

候选未进入独立确认阶段，符合预先声明的“初测通过才确认”规则。基线已自动
恢复，事务锁已释放。保留候选 patch 和全部原始数据，没有重写失败批次。
[当前离线审计](../build/m3-compact-vertices-20260925-03-audit.json)通过：重新读取原始
readback/frame 数据、检查进程生命周期及非重叠顺序，再计算出相同 reject。
采集器原始代码与 hash 保持归档，审计结果另外记录当前 verifier hash；不修改
原始 Manifest 来冒充由较新的 verifier 版本采集。

## 验证状态

- 宿主 Release 构建通过；新 shader 由生产 Slang 路径实际编译，绑定指纹已核对。
- Perf CTest 5/5，共 72 项 Python 测试通过，其中 M3 新增 17 项。覆盖错误输出、
  输入/身份漂移、空证据、仪器注入、无实际 shader 改动、噪声、回退、缺少逐轮
  监控、重复进程证据、Debug/错误构建目录、并发修改保护及异常恢复。
- 合成完整证据包验证了 accept 的离线重算和“缺独立确认不得接受”；这些测试
  不作为 GPU 性能证据。本轮实机执行了 reject 和 inconclusive 路径，未声称
  已获得或部署任何被接受的优化，也未声称实机触发过 confirmation 分支。
- 首版完成的是 MiniZorah 单候选闭环；Full 驻留恢复、多 case 验收与自动候选
  搜索仍按后续阶段推进。

## 范围限制

- 首版只允许 WorkRaster 单文件候选；不接受任意 C++、多文件或质量设置改动。
- 正确性是所选相机历史下相对于基线的 depth/visibility 等价；没有宣称独立
  reference renderer、最终 HDR 图像指标或全部场景正确性。
- 沿用 M2 已记录的输入不变量，不冒充所有 HZB/history 字节的完整重放包。
- 资产通过 M2 内容 hash 清单及运行前后 metadata 复查关联，未复制全部资源。
- GPU 后台活动被记录而非强制排除；时间噪声、输入变化与监控缺口均可使结果
  inconclusive，不自动重试到出现好结果。
- Nsight 数据用于形成诊断假设，正常计时接受门槛不依赖 UI；M1 通用 UI adapter
  的未完成部分没有被这个 runner 掩盖。
