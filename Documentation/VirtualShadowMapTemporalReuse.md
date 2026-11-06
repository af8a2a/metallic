# VSM Temporal 历史中心值与视图基底复用调研

2026-11-05 后续单项实验：仅复用首次历史中心读取的 packed uint，不改 view basis、
浮点运算或任何拒绝条件。该候选通过全部精确正确性检查，但一组 ABBA 未达到预定
性能保留门槛，**已恢复 Temporal 原实现**，没有新增运行时切换。

同一最终二进制的基线与单项各通过 lifecycle 和三类灯1/2 rays、360帧时域 oracle，
原帧 hash/count 和 MAE 一致。固定 Sponza1080p、SMRT2×16/history32/layered2048、
关闭 jitter，第16帧freeze/cut；16/47/79帧的全部 PNG/HDR/shadow/depth/visibility
逐字节一致，64个canonical帧的stats32、Build60与稳定Work一致，沿用两项Mark
调度计数例外。已查看候选输出，正确性运行validation为零。

普通计时每进程96帧/预热48，独立 ABBA，关闭validation、回读与work/build诊断；
保留磁盘cache，未锁频、不保证GPU独占。每臂为两个进程warm-frame中位数的中位数：

| GPU范围 | A 原实现 ms | T 仅中心值复用 ms | 收益点估计 |
| --- | ---: | ---: | ---: |
| Temporal parent | 0.849720 | 0.854672 | -0.58% |
| Shadows | 4.367552 | 4.364144 | 0.08% |
| RenderGraph | 10.725384 | 10.682256 | 0.40% |

目标阶段未达到至少3%改善，按计划撤回；两对Temporal点估计分别-0.27%、-0.90%。
仅一个ABBA区组，无置信区间或独立确认，不称为已确认回退。实际普通Temporal
59156→59032 B，tap-loop history read静态站点6→5；427项typed浮点opcode/
ExtInst序列及Normalize/Tan/Cross站点数保持相同。这不证明所有操作数、native
指令、寄存器、cache行为或收益；本轮不追加基底复用或其他候选。

候选与完整证据位于 `build/vsm-single-reuse-20261105/` 的CenterTemporal.slang、
Correctness-T、Summary-T、IR/ 和IRAuditReview.json。当前Temporal恢复SHA256为
`afe880bc43a0fdab8c5606cb7c619daefb2c6acaa8406abd9b71925f5f73774f`。
stepRandom单项独立保留，分层Build诊断独立保留；下文仍是此前双复用候选的失败记录。

2026-11-05：候选已实现并完成4次 GPU 运行，但未通过预先约定的逐字节一致门槛，
已恢复 Temporal 原实现，没有新增运行时切换。未执行计时，也没有性能收益结论。
默认继续保留已接受的投影空间射线状态和 receiver-mask 移位；此前已按用户指令
撤回 Projection receiver-plane / UE-style bias 候选。

## 本轮实施与验证结果

本轮将两个复用点作为一个候选实现，没有拆分收益归因：

- `temporalSurface` 返回首次历史读取的 packed uint，几何接受后直接解包 visibility
  和 sample count，去掉历史中心值的第二次读取。
- shader 内分别构造 current/previous 的 forward/right/up 和 tanHalfFov，并在位置
  重建、法线邻点和 pixelSize 中复用。有效 receiver 才构造 current，hasHistory
  才构造 previous；保留正交分支、近远平面拒绝、原运算顺序和所有历史拒绝条件。
- 未改 CPU 资源绑定、80-byte 参数 ABI、历史池、stats32 或时域算法。

候选通过 Slang O1 编译、模块所有权审计和独立代码审查。实际 diagnostic 进程
产生的候选 SPIR-V 为60440 B，SHA256：
`0708987b31b1f060753ed13ef2f8825561c564e9217f463e54a39bc6bc78f6f2`，
与提前编译的候选 payload 相同。静态 tap loop 内剩5个 history read站点，
没有 Normalize/Tan；整个模块4个 Normalize、2个 Tan。它们不代表动态执行量、
驱动原生指令、寄存器占用或实际性能收益。

固定计划原定6个 GPU 进程：Regression A/B、Diagnostic A/B、Timing A/B。
正确性要求包括所有 checkpoint 原始图像逐字节一致；性能保留条件事先定为
Temporal parent 至少改善3%、Shadows不回退、Graph回退不超过2%。正确性失败
即停止，不修改门槛或追加采样。本轮只完成前4个进程，Timing-A 在启动 GPU 前
被正确性检查拦截。

Regression A/B 各执行原有 `virtual_shadow_map_temporal_lifecycle` 和
`virtual_shadow_map_smrt_temporal`，均通过且没有 validation message。覆盖历史提交/
取消/重复帧、camera cut、深度显露、相机移动、resize、点光 compact/legacy，以及
directional/spot/point 各1/2 rays 的原几何与时域 oracle；两组报告的帧输出 hash、
计数完全一致，所有 MAE 指标未变差。

Diagnostic A/B 使用固定 Sponza 1920×1080、SMRT 2 rays×16 steps、history32、
layered cache2048、6 clipmaps、VPA128、page128、firstExtent1，关闭 jitter，
80帧/预热48，第16帧冻结流送并 camera cut，捕获16/47/79帧。使用同一可执行文件，
保留磁盘 cache；没有锁频或保证 GPU 独占。

| 检查 | 结果 |
| --- | --- |
| 第16帧 shadow/HDR | 完全一致 |
| 第47帧 shadow | layer0 的 (1164,95) 从219变218，仅1个R8值变化 |
| 第79帧 shadow | layer0 共3像素，各相差一个R8灰阶，完整坐标和值见下文 |
| HDR | 第47/79帧分别3/9个RGB分量变化，均与阴影差异像素重合；最大绝对差0.00042426586151123047，alpha一致，无非有限值 |
| PNG、depth、visibility | 三个checkpoint的文件实际字节完全一致；已查看候选第79帧画面 |
| stats32/totals、Build55、canonical输入 | 16..79帧全部一致 |
| stable GPU work | 剔除计划中既定的两项Mark调度计数后全部一致 |

第79帧的完整 shadow 差异为 (1164,95)216→215、(1089,263)164→163、
(1371,438)7→6；layer1始终一致。这里列的是该帧实际值，第47帧同坐标219→218。
微小差异可能与基底复用/函数拆分后编译器对浮点计算的处理有关，但没有 tap
中间值或原生指令证据，不能确定根因，也不能将差异单独归因于历史中心读取复用。
因此不能把这次候选称为已验证的等价默认实现；本轮撤回候选，不放宽门槛。

Temporal 源码已按原快照逐字节恢复，SHA256：
`afe880bc43a0fdab8c5606cb7c619daefb2c6acaa8406abd9b71925f5f73774f`。
候选源码保存在 `build/vsm-temporal-reuse-20261105/Candidate.slang`；同目录保留
Plan/Seal/Receipts、两组回归与诊断、DiagnosticDifferences.json、
IndependentDifferenceReview.json、实际候选 IR/Diagnostic-B/ 和 Withdrawal.json。
所有生成物留在忽略的 build 目录中。后续若继续，应先单独验证历史中心 packed
uint 复用，再定位基底复用引入差异的位置；当前没有单项收益或等价性结论。

## 当前热点与实际编译证据

复用已有固定 Sponza 1920×1080 计时：96 帧、预热48，未锁频、未保证 GPU 独占。
`build/vsm-directional-slope-clamp-20261105/Timing-A/` 的48个 eligible帧中位数：
Temporal parent 0.847200 ms，dispatch 0.837232 ms，prepare 0.000032 ms，publish
0.000288 ms。这是旧进程的范围计时，未重新采样，也不是孤立内存或 barrier 成本。

对应实际 O1 `vsmTemporalMain` payload 为59156 B，SHA256：
`5fba04385945855017eab47996d0f447607282b0d89af0062db77d077e63ea0f`。
从已绑定的 shader cache 解包，保存原始 SPIR-V、反汇编及复现脚本。结论：

| 项目 | 实际 O1 SPIR-V | 下一步 |
| --- | --- | --- |
| 历史中心重读 | tap loop内2456/3743行从同一 previousHistory、同一 pixel 读取两次；中间没有 image write，第二次只在几何接受后执行 | 复用第一次采样的 packed 值，每个 accepted tap 少一次 read |
| 视图基底 | tap loop内仍有5组 forward/right/up 构造，即10个 Normalize、5个 Tan 静态站点；loop外还有6组基底、7个 Tan | current/previous 各构造一次，复用原表达式 |

tap loop保留 Unroll 标记，并未在保存的模块中展开。裁剪 literal true/false 死分支后，
loop内有6个 history read站点：中心两次、四邻各一次；loop外有1个 shadow read、
5个 depth fetch。不能将静态站点或四tap上限当作每像素实际执行量，更不能推算
GPU加速百分比；驱动是否进一步 CSE、实际收益及寄存器影响仍未知。

## 建议的最小实现边界

1. 先复用历史中心值。`temporalSurface` 首次读取的 `stored.x` 已包含 visibility
   half与sample count；可通过额外 out 参数返回这个 packed uint，后续几何接受后
   直接解包。无需将未再次使用的 depth word 长期携带，也不改四邻采样或tap顺序。
2. 再引入 shader-local view basis，按原 normalize/cross 顺序构造 forward/right/up。
   current只在有效receiver路径建立，previous只在 hasHistory 路径建立；tan复用时
   保留正交/透视及投影有效性分支，避免冷帧、背景、正交相机增加无用计算。
3. 保留位置重建、投影、加权与容差的原运算次序。不把除法改倒数乘法，不将
   right/aspect/tan 重新组合成不同乘法顺序，不新增运行时切换。

历史复用有明确生命周期保证：CPU为稳定光源维护独立快照，pending只在提交后
成为committed，重复逻辑帧使用predecessor；三槽池由引用计数和GPU completion
控制复用。有效旧history在本次dispatch只读，并与当前写入槽分离，且被dependency、
barrier及frame retain保护。因此中心count/depth/visibility可来自同一次稳定采样。

current/previous不能合并为同一基底：sameProjection只约束投影参数，仍允许相机
平移、旋转及jitter变化；正确旧view来自历史snapshot，而非当前ViewConstants.previous。
不改CPU/header或80-byte参数ABI，不增加上传数据或改变历史失效规则。

必须保留bounds-before-load、count/clear-depth拒绝、near/far严格边界、pixel
[-0.5,size-0.5)、jitter符号、法线邻边严格小于的tie规则、cross次序、1e-20退化
阈值、0.9法线拒绝、双向平面tolerance、weightSum>=0.5、计数舍入与half packing。

本轮不推进12点邻域cache：静态、关闭jitter时有效tap分布尚未测量，预取可能增加
零权重或无效tap原本跳过的工作。也不改法线重建、disocclusion或时域算法。

调研阶段建议实施时先检查实际SPIR-V中的重复是否消失，再用已有
`virtual_shadow_map_temporal_lifecycle` 和 `virtual_shadow_map_smrt_temporal` 验证
失效、运动、显露及点光图层，随后在相同输入和预热条件下做有界A/B。两个复用点
分别评估。本轮为控制运行数合并了两个复用点；正确性失败后停止，没有计时或
单项性能收益结论。

本机证据：`build/vsm-temporal-reuse-investigation-20261105/` 中的 Withdrawal.json、
TemporalBaseline.slang、ExistingTiming.json 和 IR/Report.json；IR/ 包含raw payload、
反汇编、manifest及复现脚本。生成物不入库。
