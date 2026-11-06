# VSM Projection 下一步优化调查

2026-11-05 后续单项实验：**receiver 级 stepRandom 复用已保留**。只将同一
pixel/dimension2/samplePhase 的随机值移到 SMRT ray 循环之前，保留原 time 除法、
减法、平方与终点零值；没有合并 fine-overflow 标志、页缓存或其他采样改写。
PCF 在此之前返回，Hard 不进入该 helper。原像素参数入口保留给 Mark 和射线探针。

本轮使用同一最终二进制、独立 shader 快照，基线与 stepRandom 单项各通过射线状态、
SMRT search、三类灯时域和 receiver-mask 缓存原有 oracle。固定 Sponza 1080p、
SMRT2×16/history32/layered2048、6clipmaps/VPA128/page128、关闭 jitter，第16帧
freeze/cut；16/47/79帧的 PNG/HDR/shadow/depth/visibility 字节精确一致，64个
canonical 帧的 stats32、Build60 和稳定 Work 一致，保留两项既有 Mark 调度计数例外。
已查看场景输出；正确性运行 validation 为零。

普通计时按预声明的一组 ABBA 执行，每进程96帧/预热48，关闭 validation、回读、
GPUWork 和 BuildSubstages；保留磁盘缓存，未锁频、不保证 GPU 独占。下表每臂为
两个独立进程 warm-frame 中位数的中位数；阶段时间不等于呈现 FPS。

| GPU范围 | A 原实现 ms | S stepRandom复用 ms | 收益点估计 |
| --- | ---: | ---: | ---: |
| Projection | 2.091456 | 1.971000 | 5.76% |
| Shadows | 4.401064 | 4.372960 | 0.64% |
| RenderGraph | 10.681376 | 10.701216 | -0.19% |

通过目标阶段至少3%、Shadows不回退、Graph回退不超过2%的预定保留门槛；两对
Projection 收益分别8.85%和2.46%。只有一个 ABBA 区组，没有置信区间或独立确认，
不将点估计称为已确认的生产加速。实际普通 Projection 216692→216604 B，IR确认
两处 stepRandom hash 移到 ray/step 循环之外；其他六个实际 shader payload 相同。
SPIR-V 的位置变化不证明原生寄存器、带宽或 occupancy 收益。

同轮新增的 Build 分层诊断只在诊断排列执行，全部七个普通基线 payload 与旧归档
逐字节相同。证据在 `build/vsm-single-reuse-20261105/` 的 Plan/Seal/Receipts、
独立源码与EXE快照、Correctness-S、Summary-S、IR/ 和 IRAuditReview.json；生成物
不入库。下文为前序实验记录，stepRandom 不再是未执行候选，fineAvailable 尚未实验。

2026-11-05 更新：页尺寸整数寻址和 **32-bit紧凑mapping＋物理XY** 两项均已完成
实验。两项精确正确性通过，但本次各自普通计时未达到预设保留门槛，候选均已撤回。
生产与测试源码按实验前快照恢复，编辑器、VSMSample及RHI测试程序构建通过。
下一步可单独评估每receiver复用step随机offset及fine-overflow标志；最近页缓存
仍需先补命中率证据。后文保留实验前的UE对照，不把紧凑mapping称为已采用实现。

继续保留已接受的方向光 uniform 资源访问、投影空间射线状态、propagated mapping
及 Mark receiver-mask 移位；不恢复已经撤回的 slope / UE bias 候选。

## 32-bit紧凑mapping实验结果（未采用）

只实施紧凑采样表示和直接物理XY：physicalXY各6bit、target4bit、mode2bit、
parentXY各7bit。ownership prefix保持16*N；packed mapping使用同pages句柄的
ByteAddress scalar Load/Store，地址16*N+4*page；保留旧allocation、valid mask
的32*N偏移及physical history的actual-pool-end位置。slot4095/(63,63)仍有效，
mode0与coarse由mode/target识别。generic point查询传实际选中face的pagesHandle。
prefix/coarse回退保留原线性页地址换算；浮点表达式、ancestor clamp、SMRT history、
资源绑定与stats32不变。没有混入pageSize移位、随机偏移外提或生产切换。

本轮预先封存6进程：Regression A/B、Diagnostic A/B、Timing A/B；恰好全部执行，
没有重试/补采样。采用门槛为Projection至少改善3%、Shadows不回退、Graph回退
不超过2%。一个最终二进制用于所有进程并归档实际EXE；Main与测试Probe的A/B
快照通过逐进程before/after源码及binary记录绑定。测试CPP共同支持探针报告的
4-word/1-word原始记录，作为测试格式识别，不是生产实现切换。

验证结果：

- 两臂各6项原回归通过，无跳过，validation为0：mapping19案；projected ray9案/
  594样本及Mark区间；SMRT search17案；SMRT temporal6组合/360帧；页尺寸
  `1→128→16→1` 8帧；receiver-mask cache27帧。canonical签名精确一致。
- 在原mapping用例内新增7个双向codec样本。CPU独立编码/解码实际raw word与
  生产GPU pack/load交叉验证，覆盖mode0..3、slot4095、(63,63)、parent127、
  level11/coarse12；原完整CPU整数路由、prefix未变及物理atlas texel oracle保留。
- 固定Sponza1920×1080、SMRT2×16、history32、分层缓存2048页、6clipmaps、
  VPA128/page128、firstExtent1、关闭jitter。diagnostic为80帧/预热48，第16帧
  freeze/cut；16/47/79帧PNG/HDR/shadow/depth/visibility实读字节精确一致，
  已查看候选第79帧场景。64个canonical帧stats32、Build55、输入与稳定工作一致，
  保留原两项Mark调度计数例外。仅Projection/Propagation实际入口变化，另外
  Mark/Build/Raster的4入口和Temporal实际bytes保持相同。
- 候选probe、Projection、Propagation离线编译通过，Release构建及模块审计通过；
  撤回后3个源文件逐字节恢复、程序重建通过，没有额外GPU运行。

实际普通Projection模块216692→221660 B，Propagation13420→15656 B。IR确认
mapping读取由v4uint变为uint、producer写入由v4uint变为scalar uint，没有向量
read-modify-write；9份不同普通payload通过Vulkan1.4 spirv-val。回退除法仍存在。
静态体积、指令站点和16→4B逻辑条目变化都不证明native带宽或寄存器收益。
首阶段没有收缩allocation，分配容量节约为零，默认活跃mapping跨度从1,572,880
变为393,220 B；该跨度缩小仅描述候选，当前默认已恢复。

普通计时为96帧/预热48、相同二进制及场景/质量，validation/readback/GPUWork/
BuildSubstages关闭，磁盘cache保留，未锁频、不保证GPU独占。两臂80个canonical
帧输入一致；每列为该进程48个warm帧中位数：

| GPU范围 | A 原默认 ms | B 紧凑mapping ms | 耗时变化 |
| --- | ---: | ---: | ---: |
| Projection | 2.151680 | 2.215008 | +2.94% |
| Propagation | 0.007648 | 0.006400 | -16.32% |
| Shadows | 4.425664 | 4.497744 | +1.63% |
| RenderGraph | 10.652512 | 11.145536 | +4.63% |
| Mark | 1.041104 | 1.217888 | +16.98% |
| Allocator | 0.248288 | 0.053168 | -78.59% |

Propagation为其独立marker范围，包含在原父级工作范围内，不能再加到父级上。
三项保留门槛均未通过，按计划撤回。Mark实际shader未变化，Allocator也有明显
差异，原因未定位；保留这些原始计时，不将波动确定归因于候选。只有一个按A→B
顺序的配对，没有ABBA、置信区间或独立确认，不称为已确认回退或端到端FPS结论。
也不能把两次不同实验的绝对时长相减当作新A/B。

证据位于 `build/vsm-packed-mapping-20261105/`：Original/Baseline/Candidate快照、
Candidate.patch、共同测试CPP及Probe两种快照、Plan/Seal/Receipts、6份SourceIdentity
sidecar及其receipt SHA绑定、EXE归档/BinaryArchive.json、两组回归/诊断/普通计时、
Correctness.json、Summary.json、InstructionAudit.json、IR/实际payload/manifest、
Adoption.json、RestoreBuild.log、RestoredDefaultCheck.json、IndependentFinalReview.json。
生成物不入库。当前默认shader SHA仍为
`f4879af7974195c976cbec0262aeed3d494cacdc011478123ec3399f1f2b1983`。

## 第1项实验结果

本轮只实施整数地址变换：新增内部 `VSM_PROJECTION_PAGE_SHIFT` 编译常量，仅传给
真实方向光 per-face Projection；值来自同次校验设置及已有Programs页尺寸缓存键。
`vsmVirtualPage` 使用右移，三处atlas页内地址使用 `vsmPageLocalTexel` 的常量mask。
物理页宽度的除法/余数、浮点坐标与比较、采样、历史、资源及stats32保持原式/布局。
没有新增生产属性或benchmark开关。两臂使用同一最终二进制，baseline shader
忽略新增宏。射线探针的固定pageSize4显式传shift2，以实际覆盖候选地址路径。

在首个GPU进程前封存6阶段计划：Regression A/B、Diagnostic A/B、Timing A/B。
正确性要求原有oracle通过、原始图像/统计精确一致；性能保留门槛固定为Projection
至少改善3%、Shadows不回退、Graph回退不超过2%。本轮恰好执行6个GPU进程，
未追加样本、重试或参数扫描。

验证结果：

- A/B各运行5项已有回归，均通过、无跳过、validation为0：propagated mapping
  19案；projected ray state 9案/594样本及Mark区间；SMRT search 17案；真实
  SMRT temporal 6个灯类型/ray组合、360帧与移动/显露/点光发布oracle；真实
  SMRT页尺寸 `1→128→16→1` 的8帧持久对象/fresh对象和缓存miss/reuse检查。
  两臂的显式正确性签名精确一致，MAE没有变差。
- 固定Sponza1920×1080、SMRT2×16、history32、分层缓存2048页、关闭jitter。
  diagnostic为80帧/预热48，第16帧冻结流送并camera cut。第16/47/79帧
  PNG/HDR/shadow/depth/visibility文件实际字节完全一致，已查看候选第79帧画面。
- 16..79帧stats32、Build55、canonical输入和稳定GPU工作完全一致，继续明确
  排除原有 `markPageRequests` / `markReceiverMaskAtomicOrs` 两项调度计数。
  除Projection外的四个VSM入口及Temporal实际payload字节一致。
- 候选、恢复后均通过Release构建；模块所有权审计与diff检查通过。恢复后没有
  再启动GPU进程，不把重建当成额外运行验证。

实际普通Projection模块216692→216412 B；pageSize的静态向量 `OpUDiv` 10→0、
`OpUMod` 6→0，物理页宽度的标量 `OpUDiv` / `OpUMod` 各6个站点保持不变。
这些包含Hard/PCF/SMRT内联副本，不能当作每tap动态指令数量。普通A/B的7种不同
raw payload通过Vulkan1.4 spirv-val；没有原生寄存器、spill或occupancy证据。

普通计时每进程96帧/预热48，关闭validation/readback/GPUWork/BuildSubstages；
保留磁盘cache，不锁频、不保证GPU独占。使用同一二进制、相同场景/相机/质量，
并核对80个canonical帧输入一致。以下为各进程48个warm帧的中位数：

| GPU范围 | A 原默认 ms | B 整数寻址 ms | 耗时变化 |
| --- | ---: | ---: | ---: |
| Projection | 1.801968 | 1.828464 | +1.47% |
| Shadows | 3.834224 | 3.859648 | +0.66% |
| RenderGraph | 9.418496 | 9.446032 | +0.29% |

Projection最低收益及Shadows保护门槛未通过，Graph保护门槛通过。因此按预设计划
恢复候选。仅一个按A→B顺序的配对，没有ABBA、置信区间或独立确认；不能将
点估计称为已确认回退，更不能由SPIR-V中的除法消失宣称已加速。具体未获益原因
尚未定位，不把它确定归因于驱动除法优化、寄存器或指令调度。

本轮证据保留在 `build/vsm-projection-page-address-20261105/`：Baseline/Candidate
shader、Original/Candidate CPP及test快照、Candidate.patch、Plan/Seal/Receipts、
两组回归/诊断/普通计时、Correctness.json、Summary.json、InstructionAudit.json、
IR/中的实际payload和manifest、Adoption.json、RestoreBuild.log、RestoredDefaultCheck.json
及独立复核的IndependentFinalReview.json。源码恢复后的
默认shader SHA256仍为 `f4879af7974195c976cbec0262aeed3d494cacdc011478123ec3399f1f2b1983`。
生成物不入库；没有保留候选切换。

## 实验后与UE对照的下一轮方案

本节为整数寻址实验之后的只读分析，未修改生产、未编译或新增GPU运行。对照本机
UE5.7.4，提交 `8284a0e654e82cee0ed716f16704b1c897e330c3`。上轮无收益只说明
该单一A/B未达到保留门槛，不能反推硬件不受ALU、带宽或寄存器限制。

| 环节 | Unreal | Metallic当前默认 | 优化判断 |
| --- | --- | --- | --- |
| 采样页表 | 一个uint编码physicalXY、LOD offset和valid bits | derived mapping为uint4，实际SMRT IR保留整v4uint load | 首选紧凑记录及直接physicalXY |
| 随机step offset | receiver传入Noise，ray cast在循环前使用它 | 每个非终点step重复调用同pixel/dimension2/phase的hash | 小候选：仅复用原random结果，保留time表达式 |
| fine有效性 | 页表条目携带对应有效性 | step还重复读取固定statistics[5] | 可随step不变量复用，不删除overflow分支 |
| 起始clipmap | GetMappedClipmap可把ray基准抬至已映射层 | 每sample从receiverLevel重查，允许重新进入更细页 | 直接迁移会改变分辨率选择，后置 |
| ARC及循环展开 | 已有lit/umbra wave退出；静态sample排列仅2/4 | 已有同类ARC，当前默认16步 | 缺少支持重做ARC或展开16步的新证据 |

### 主候选：32-bit采样mapping与物理XY

UE `VirtualShadowMapPageAccessCommon.ush:237–285` 的页表是32位：XY各10bit、
LOD offset6bit及有效位；`VirtualShadowMapProjectionCommon.ush:279–284` 直接用
解码后的physicalXY构造atlas地址。Metallic不应复制UE全部bit分配，但可以采用
其紧凑采样表示。当前生产边界允许完整保留Metallic额外的整数parent归属信息：

| 字段 | 位数 | 依据 |
| --- | ---: | --- |
| physicalX / physicalY | 6+6 | physicalPageCount≤4096，现有atlas XY在0..63 |
| targetLevel | 4 | clipmapCount≤12，包含coarse标记level==count |
| mode | 2 | 原0/1/2/3四种语义 |
| ancestorPageX / ancestorPageY | 7+7 | virtualPagesPerAxis≤128，坐标在0..127 |
| 合计 | 32 | 不丢字段，无需扩大表或加入新资源 |

可行布局是XY位0..11、target12..15、mode16..17、parentX18..24、parentY25..31。
这只是建议编码，尚未写入shader。无效值不占用一个physical槽：mode0继续原查页；
mode1为own；mode2为directional ancestor，mode3为local ancestor；后两种的
target==clipmapCount继续走coarse。必须先判断mode/target，再使用物理坐标，
因此slot4095及physicalXY=(63,63)仍然可表示。不要用12bit的全1作为无效sentinel。

Producer可把合法linear physical slot预先转换成XY；consumer直接携带uint2到
atlas。若解码回linear index后又按physicalPagesX除/取模，就没有完成地址表达迁移。
原ownership/residency前缀及其linear index保持不变，原tag0和prefix/coarse fallback
仍保留检查与必要的linear→XY计算。parent坐标、continuous transform、clamp及
未dither比较深度均保持原式。通用/local、方向光、one-pass各job及probe必须使用
同一编码；one-pass按当前job的level/VPA推导表长，不能使用第一面的表长。

实际普通baseline IR的step loop在12693/12694行建立v4uint指针并整条加载mapping，
并非只由源码类型推断。最新warm帧每帧14,890,404次mapping查询；按逻辑记录宽度
相乘，16B为238.246MB，4B为59.562MB。它们是重复访问累计的逻辑记录请求量，
不是unique bytes、DRAM流量或4倍加速预测。driver标量化、cache line、合并访问和
复用率均未测量。源/target/selected projection字段也有重复load，但携带完整
ShadowProjection跨分支可能拉长状态寿命，不据此反转当前identity-only lookup设计。

建议分成两个实现阶段，首个实验只验证第一阶段：

1. **先紧凑读写，保留现有资源分配。** N为本面totalPages，ownership前缀仍16*N
   bytes，在旧mapping tail起点16*N连续存N个uint，地址为16*N+4*sourcePage。
   使用现有pages句柄的ByteAddressBuffer.Load / RWByteAddressBuffer.Store，避免
   多线程修改同一uint4分量形成向量read-modify-write竞争。每个invocation独占一个
   32-bit word；先确认实际SPIR-V为scalar load/store。暂保留旧allocation、
   valid receiver mask的32*N偏移和physical history的actual-pool-end位置。
   这时默认mapping活跃跨度1,572,880→393,220B，**分配容量节约仍为零**。
2. **有净收益再收缩布局。** 分配改为16*N+alignUp(4*N,16)+mask/history，同步
   `vsmValidReceiverMaskByteOffset`、CPU allocation和所有probe。默认可减少
   1,179,648B（1.125MiB）的逻辑所需容量；buffer pool可能保留更大allocation，
   不能把逻辑缩小当作已释放VRAM。physical history仍按实际pool buffer末端定位。

当前同pages句柄已有ByteAddress Load2/Store2用于receiver mask（shader:1733/1755/
2328/3144），首阶段无需新增RHI接口。Propagation始终在Finalize发布后重建所有
记录，写tail与读ownership前缀分离；它仍须独立dispatch并保持发布依赖。额外编码/
解码、Propagation中预计算XY及寄存器存活均有成本。此候选同时改变采样记录宽度
和物理地址表示，未来只能报告该表示变换的组合收益，不能分别归因。

### 小候选：每receiver复用stepRandom和fineAvailable

UE `VirtualShadowMapProjectionDirectional.ush:182,257` 将receiver的Noise传给
SMRTRayCast；`VirtualShadowMapSMRTTemplate.ush:41–55` 在step循环前建立时间参数。
Metallic `vsmRaySampleTime:902–906` 则在每个非终点step重算
`vsmSampleRandom(pixel,2,samplePhase)`。实际普通IR12320–12342行仍执行两轮pixel
hash，12343–12361行执行phase/Fract分支；这些输入不依赖ray或step。

可以在有效SMRT receiver路径、ray循环之前计算一次相同random值，再继续使用
原 `1-(float(step+1)-random)/float(samplesPerRay)` 及平方顺序，保留终点step返回0。
不照搬UE的TimeScale/TimeBias，不改倒数乘法、随机序列、ray-dependent dither或
Mark/query一致性。PCF、Hard和背景路径不应为此计算额外random。

同一receiver在IR9471行已读statistics[5]，step12395–12399行又读取和分支。Build/
Finalize已发布该槽，Projection只写其他统计槽，可一起携带shader-local
fineAvailable并保留原判断。局部点光跨面的标志仍属于各自资源，不能共享当前面值。
这仅增加一个random scalar与boolean，不改CPU参数ABI。实际驱动是否已进一步
外提、长寿命scalar是否影响寄存器仍未知；应与紧凑mapping分开实验。

### 后置方案与必要证据

- UE GetMappedClipmap（Directional:38–70,194–203）先选已映射层再建立整条ray。
  当前约50.4%初始receiver使用ancestor，确有潜在涉及范围，但Metallic沿ray重进
  finer页的行为会改变；这属于质量策略，不作为纯等价优化。UE自身也先做一次
  receiver查页，不能把Metallic全部初始查询一概称为可删除冗余；stats32及coarse
  域判定仍依赖它。
- 最近页cache仍缺连续source key或physical texel命中率。每实际ray平均3.17次
  atlas查询，不能按17次预估收益。先dry-run计数，不能把ancestor比例当作cache命中。
- UE没有在该SMRT路径使用2×2 gather。其gather用于page overlap/management；
  不改Metallic D32纹理格式或将任意ray样本当作2×2 footprint。
- 未来实验继续使用有界6进程、exact图像/统计及同输入约束，同时观察Propagation、
  Projection、Shadows和Graph。需要收益与原生瓶颈结论时，再依据实际counter或
  native profiling补证据；不由SPIR-V体积或逻辑字节数作结论。

本次只读证据保留在 `build/vsm-ue-next-20261105/`：UEComparison.md、IRReview.md/
json、IRBaseline.spvasm及LayoutAnalysis.json。未修改生产或新增GPU运行。

以下为实验前的源码与UE调研记录，不将建议解释为尚未执行的第1项实验。

## 当前证据

当前 `VirtualShadowMaps.slang` SHA256 为
`f4879af7974195c976cbec0262aeed3d494cacdc011478123ec3399f1f2b1983`。
最新相同源码的 Sponza diagnostic 为
`build/vsm-temporal-reuse-20261105/Diagnostic-A/`：1920×1080，SMRT 2×16，history32，
6 clipmaps，128 texels/page，128 pages/axis，2048 physical pages，关闭 jitter，
第16帧冻结流送并 camera cut。以下为48..79帧中位数；instrumented 数据只用于
工作量分析，不作为性能计时：

| 每帧工作 | 数量 |
| --- | ---: |
| Receivers | 2,070,497 |
| 实际 rays | 4,041,225 |
| Atlas loads | 12,819,907 |
| Mapping loads | 14,890,404 |
| Raw page-table loads / mapping slow paths | 0 / 0 |
| Adaptive saved rays | 99,769 |

所有 warm 帧满足 `mappingLoads = atlasLoads + receivers`，额外项是初始 receiver
查询。平均每实际 ray 约3.17228次 atlas 查询，不能用配置的17个step位置估计
实际读取量。Adaptive 节约配置总射线约2.4093%。初始 receiver 有50.3974%使用
fine ancestor；扣除初始查询后，ray ancestor 解析占 ray mapping reads 约19.3413%。
这些比例都不是连续同页命中率。当前没有 coarse fallback，2047个 admitted fine
pages 全部 clean/valid；只代表固定、静止、warm 场景。

最近普通计时沿用 `build/vsm-directional-slope-clamp-20261105/Timing-A/` 原默认臂：
Projection 1.804800 ms、Shadows 3.837216 ms、Graph 9.613504 ms，均为单进程
48个 warm 帧的中位数。它是背景参考，不与本轮 diagnostic 相减，不是新的收益结论。

对应实际普通 Projection payload 为216692 B，SHA256：
`4be3cb5b72ecbbffba87386a977f20878f2a2184371ec1a865eb264329f89ea0`，
runtime 日志/原 manifest 绑定在 `Timing-A/run-1/Process.log:398`。
原始 payload 已复制到 `build/vsm-projection-next-20261105/Baseline.spv`。
其反汇编中 filtered SMRT arm 的 ray/step/level loop 分别位于11704/12286/12407行。
以下按该 arm 定位，不把 Hard-inline 副本或全模块静态数量当作每样本执行量。
本轮对保存的原始模块运行 `spirv-val --target-env vulkan1.4`，验证通过。

## 1. Projection 页尺寸整数寻址专用化

当前 `vsmVirtualPage`（shader:494）只有 Mark 获得 `VSM_MARK_PAGE_SHIFT`；CPU
在 `VirtualShadowMaps.cpp:1302` 仅给 Mark entry 传这个宏。Projection 的
`texel / pageSize` 仍是运行时整数除法，`vsmDirectionalRayLookup:1328` 的页内
`virtualTexel % pageSize` 也没有消除。

实际 step 路径仍有：

| SPIR-V 行 | 运算 | 分母 |
| --- | --- | --- |
| 12668 | `OpUDiv %v2uint` | pageSize |
| 13179 | `OpUMod %v2uint` | pageSize |

`validProjectionSettings`（CPU:117）保证 pageSize 是1..128内的非零二次幂。
因此 uint 地址的除法可改为 `>> pageShift`，余数可改为 `& (pageSize - 1)`。
这与此前消除 Mark 的 mask 除法不同，目标是 Projection 的虚页索引及 atlas 页内
地址。仅对真实方向光 per-face entry 使用按实际设置生成的编译常量；现有 Programs
已经按方向光 `markPageShift` 区分缓存（CPU:1230附近），不需要新的功能开关。

只替换整数地址运算。保持 floor、float resolution、坐标投影、边界 inset、ancestor
clamp、比较深度及 Mark 共享浮点表达式原样，避免同时扩大数值变化范围。混合
one-pass 与局部灯先保留通用地址路径；generic probe 的256页尺寸不套用该常量。

离线已穷举8种支持页尺寸的32,640个标量虚 texel，shift/mask 与原除法/余数一致。
这只证明整数等式，尚未证明最终 GPU 图像或性能。优点是改动范围小、没有额外
buffer/read/长寿命 ray state；实际驱动可能已降低部分除法成本，不能预报收益百分比。

## 2. Mapping 预存物理页 XY

`vsmDirectionalRayLookup:1327` 每次成功查询仍将线性 physicalPage 拆成 atlas XY：

```text
physical = (physicalPage % physicalPagesX, physicalPage / physicalPagesX)
```

实际 step 中13172/13173行为标量 `OpUMod` / `OpUDiv`，分母是 physicalPagesX。
默认2048页的宽度是 `ceil(sqrt(2048)) = 46`（CPU:671），不能直接替换成位移。
不建议为了方便地址运算改成64页宽 atlas，这会改变纹理布局、占用和 raster 舍入。

本机 UE 5.7.4 的 `VirtualShadowMapPageAccessCommon.ush:245–285` 已在页表编码中
保存并解码 `uint2 PhysicalAddress`；`VirtualShadowMapProjectionCommon.ush:279–284`
直接使用 `PhysicalAddress * VSM_PAGE_SIZE + (VirtualTexelAddress & VSM_PAGE_SIZE_MASK)`。
可以迁移其地址表达方式，无须迁移整套UE页池或深度算法。

Metallic 的窄方案是只扩展 derived mapping tail 的编码：

- ownership prefix、mapping.x 的线性physicalPage、mapping.y 的targetLevel以及
  mapping.z 的整数parent page保持含义不变。
- mapping.w 低2位保存原mode0..3；在其高位附带6-bit物理X和6-bit物理Y。
  `mode | (x << 2) | (y << 8)` 是一个可用布局，并非已实施协议。
- `record` 的参数边界限制 physicalPageCount 为1..4096（CPU:570），当前 atlas
  构造保证X/Y均在0..63。已穷举全部预算和8,390,656个有效物理页索引，并对
  mode1/2/3验证编码、解码及线性地址回算。
- Propagation 在发布有效own/ancestor mapping时计算XY。方向光SMRT消费mapping
  时直接解码；tag0、无效/coarse及实际原始页表fallback先保留原计算与验证。

这不增加 `uint4` 大小，但改变derived记录格式，所有消费者都必须提取低2位后
判断mode，包括通用/local lookup、directional lookup、CPU oracle和GPU probe。
不能只改SMRT读取而留下现有 `mapping.w == 1/2/3`。Fine/coarse sentinel、跨面资源
选择、整数ancestor归属和target texel clamp必须保留。Tail仍由Finalize之后的独立
Propagation dispatch重建，沿用既有发布依赖（CPU:1526–1540），不能改写ownership
前缀或把祖先physicalPage误作所有者。

预计算将除法移到每虚页的Propagation，默认共有98,305条记录，不能把工作当成免费。
它可能减少每帧约12.82M次成功ray采样处的线性物理地址拆分，但这只是shader工作
位置的变化，不等于内存读取或时间同比下降。验收应同时观察Projection、Propagation、
Shadows和Graph；本轮没有native register、spill、occupancy或底层div成本证据。

## 3. 固定 fine-overflow 标志复用

`vsmProjectReceiver:3765` 已读取 `statistics[5]`，而
`vsmDirectionalRayLookup:1288` 每个step再次读取同一槽。实际 filtered SMRT arm
在9471行首次读取，在step loop内12395/12396行重新建立slot5指针并 `OpLoad`，
不是仅由源码猜测的重复。

Slot5只由Build的work overflow路径累加（shader:2799/3039/3048），Finalize读取
并失效fine页；进入Projection前已经发布，Projection本身只更新其他统计槽。
可以在原receiver读取位置保存 `fineAvailable`，沿shader-local参数传给方向光SMRT，
每次lookup继续执行相同判断。不删除overflow路径，不改public stats32或CPU参数。
点光跨面查询读取各自statistics，不能把当前面的标志当作所有面的标志。

这是低复杂度、无浮点重排的小候选，但标志可能命中缓存或被驱动进一步优化，
还需承担一个存活boolean；不能由SPIR-V多一次load就声称有可采用收益。建议与
地址优化分开评估，避免组合后无法归因。

## 暂不优先的方向

- **最近页/深度缓存**：缺少连续source page及physical texel统计。先做dry-run
  `(sourceLevel, sourcePage)` checks/hits，每ray重置，并保留原load。不同source页
  可能指向相同ancestor；同一physical page内的不同texel不能复用depth。不能沿用
  上次target level跳过更细搜索，reverse march可以重新进入fine clipmap。
- **继续优化祖先逐级搜索**：当前warm原始页表和mapping slow paths都为零。
- **更多采样数学专用化、receiver slope、改SMRT步进/阈值或ARC规则**：已有数值
  差异或算法变化风险；不与本轮整数地址候选混合。
- **重做wave统计**：此前精确版本Projection仅改善0.24%，未达到原采用门槛，
  见[统计实验](VirtualShadowMapProjectionStatistics.md)，当前没有支持重试的新证据。

## 实验前预定实施与验证边界

首个候选预定只实施第1项，不增加生产或benchmark开关。先核对最终实际SPIR-V中的
两处向量除法/余数变为shift/and；保留物理页宽度除法，不能冒称消除了所有整数除法。
再做有界验证：回归A/B、Sponza诊断A/B、普通计时A/B，共6个GPU进程；正确性
失败则提前停止，不在结果之后放宽门槛或追加计时。

整数地址候选应要求shadow/HDR/depth/visibility/PNG及stats32精确一致，保留原
mapping、SMRT质量/时域oracle、overflow/coarse与边界检查；canonical输入、Build55
和稳定GPU工作一致，继续披露既有两项Mark调度计数例外。普通计时固定同一二进制、
Sponza1920×1080、96帧/预热48，关闭validation/readback/GPUWork；预先固定收益和
整图保护门槛，保留缓存/频率条件与单配对局限。第2项必须另行验证mapping格式及
Propagation成本，不能复用第1项的通过结论。

本轮本机证据在 `build/vsm-projection-next-20261105/`：原始Baseline.spv及反汇编、
Analyze.py/IRFacts.json、WorkEvidence.json、AddressProof.py/json以及UEReview.md。
生成物不入库。本文记录可执行的候选与证据缺口，不声明已经得到新的Projection加速。
