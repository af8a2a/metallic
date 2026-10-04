# M4：Value IR → Closure IR → 场景执行

本补充接通 M1–M4 审查中的 P1。没有增加 MaterialGraph 编辑器，也没有扩大
[Slab 原型](MaterialSystemPhase10.md) 的散射模型或预算。

## 作者资产与 lowering

`.materialdef` 的 `implementation: "Slab.Surface"` 拥有 `surfaceProgram` 对象。
它使用 Value IR v3：`nodes` 提供可共享的数值表达式，`outputs` 只允许 `emissive`
和 `coverage`，`closure` 提供有序的 Slab/Mix/Layer 拓扑。例：

```json
{
  "version": 3,
  "nodes": {"color": {"op": "parameter", "index": 0}},
  "outputs": {},
  "closure": {
    "op": "mix",
    "a": {"op": "slab", "reflectance": {"ref": "color"}},
    "b": {"op": "slab", "reflectance": [0.8, 0.2, 0.05, 0]},
    "weight": {"op": "parameter", "index": 1}
  }
}
```

Slab 的 `reflectance`、可选 `opticalDepth`，以及 Mix 的 `weight` 都接受 Value
表达式，包括有显式 footprint 的纹理采样。Layer 的 `a/b` 分别是顶层和底层。
输入先经过 Value IR 的引用验证、常量折叠、CSE 和 DCE；Closure IR 检查拓扑与
实时预算并选择 SingleSlabClosure 或 DualSlabClosure。规范化身份包含拓扑和活跃
数值表达式；生成代码从同一 IR 取得动态输入，不把实例参数烘焙进代码。

作者色彩表达式使用 linear Rec.709，reflectance/emissive 在输出边界转换到当前
working RGB。reflectance 限制在 [0,1]，emissive 限制在 [0,1e6]。opticalDepth
是当前 working RGB 三个通道上的非负吸收系数，范围 [0,1e6]；它不是颜色，不进行
原色转换。Mix 使用 weight.x 并限制在 [0,1]。

定义的 `defaults.valueParameters` 和实例的 `valueParameters` 都是稀疏对象，键
为 `"0"` 到 `"3"`，值为四个有限浮点数；父实例按 float4 slot 覆盖，省略的 slot
继承，最底层缺省为零。纹理沿用语义资源槽、asset URI 和显式 footprint。

可直接绑定的例子是 [AbsorbingSlab.material](../Asset/Materials/Examples/AbsorbingSlab.material)，
资产根目录为仓库的 `Asset/`。它引用 [SlabLayer.materialdef](../Asset/Materials/SlabLayer.materialdef)。
通过现有 `SceneDocument::setMaterialAsset` 绑定和保存即可使用。sidecar 保存资产引用
与本地改动；重载继承新定义。切回 OpenPBR 资产会清除旧 Slab 资产拥有的程序，
旧版独立场景 Value 程序仍按原规则保留。

## 场景运行

MaterialGeneration 增加稳定 SingleSlab/DualSlab 类型 ID。动态值和 Coverage 字节码
继续共用原有 80-byte MaterialValueInstance 输入与 binding 97，legacy payload
仍为 720 bytes。资产、Value 源码、纹理引用与参数通过现有事务性场景发布进入同一代；
失败不能发布一半输入，在途帧继续持有旧代。

生产 SceneOpenPBRMaterialProgram 在命中点根据生成的 family dispatch 构造场景
Closure。材质纹理仅在 evaluate 阶段读取，prepare 和 lighting 只访问数值状态。
原 OpenPBR 和 Slab 都进入 SurfaceLighting 的 eval/PDF/weighted-sample 接口。
PT 每次命中重新选择程序，因此 secondary hit 不沿用 primary 的静态 ID。
Slab 的实时环境光也使用其实际 PreparedClosure 积分，而不是 OpenPBR split-sum
近似。调试/guide 的 Layer albedo 是法线入射反射率摘要，不能用于代替真实 Layer eval。

resident VBuffer 依据完整编译 ProgramKey 调度：每个 Slab 程序有自己的静态 ID
编译请求，Single 属于 SingleSlab family，Mix 与 Layer 都属于 DualSlab family。
默认仍是 fused 调度；没有新建全屏 packed Closure buffer。关闭 `materialBinning`
会走动态 family dispatch；关闭 `programBinning` 时，自定义程序同样回退通用路径，
避免被旧版仅看 glTF 参数的 BSDF 类别误分箱。

## 能力边界

- 支持 lit opaque / MASK Surface；Coverage 仍独立切片，在 VBuffer、ray query 和阴影中
  使用同代输入。Slab 不支持 BLEND、透射、Fiber 或 unlit。
- 当前只有共享法线、单面漫反射 Single，或两个直接 Slab 加一个 Mix/Layer；超预算
  拒绝。resolved family 的 48/96-byte 指标不是通用场景 tagged wrapper 的大小，
  也不是寄存器或显存性能测量。
- 支持 resident 场景 PT 和 VBuffer。stream materials、其他 BSDF 后端、RTXDI 的
  Slab 路径明确拒绝；没有宣称这些路径具备 Slab transport。
- 单独法线、镜面/折射、多次层间散射、任意深度图、MaterialGraph UI 和默认 split
  调度仍不在本次范围。

## 验收

新增 CPU 资产测试覆盖定义/父实例参数、保存与重载、定义默认值更新、本地覆盖、
失败保留和切回 OpenPBR。`material_value_closure_ir` 检查数值/拓扑身份、Coverage
分离、family manifest、动态值共享与预算拒绝；`material_value_closure_publication`
执行真实 GPU 上传与失败分配恢复，验证不可变代和参数更新。

`material_value_closure_scene` 使用真正的场景资产、生产 PT/VBuffer 与 HDR 回读：
三种程序映射到两个 family，分箱开关结果一致；透明 Layer 边界等于 Mix 端点；
动态 optical depth、Mix weight 和纹理改变实际散射。相机背后的 Slab 发光墙在
depth=1 时不可见，只能通过 secondary hit 照亮 receiver；发光量翻倍应使间接
辐射翻倍。直接 BSDF 对照显式关闭阴影，避免该墙遮住测试太阳；间接测试关闭全部
灯光与环境。有限值、图像与误差记录保存在本地构建输出。

`material_coverage_slab_winner_shadow_ray` 将现有 winner/shadow/ray 夹具的共享数值图
切换到 v3 Slab，逐像素检查纹理 alpha、动态阈值、蓝色/红色 Surface 值和恢复 legacy
路径；每次运行比较 30,752 个内部像素。新场景测试还检查实际 Layer IBL 的吸收变化。
PT 发光线性对照在新 renderer 的 frame 0 重放，避免全局 sampleFrame 改变随机序列。

从仓库根目录、兼容的 x64 MSVC 开发环境运行：

```powershell
cmake --build build-scheduling-release --target MetallicRHITests MetallicSceneTests Metallic LookDev -j 8
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter='MaterialAssets.*'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_value_closure*:*material_coverage_slab*' --output-dir build/material-m4-new
```

同一个测试程序可在启动前设置 `METALLIC_WORKING_COLOR_SPACE=acescg` 或 `rec709`。
历史 Phase 0 冻结图是 Rec.709；不能直接把当前 ACEScg raw buffer 与它比较。

### 2026-10-04 实测记录

使用 `build-scheduling-release` 的最终二进制，GPU 验收启用 Vulkan validation，
隔离无关 implicit layers。默认 working space 为 ACEScg；冻结图比较单独使用
Rec.709。原始日志和图片保留在本地 `build/`，未纳入源码。

| 验收 | 结果 | 本地证据 |
| --- | --- | --- |
| MetallicRHITests / MetallicSceneTests / Metallic / LookDev | 构建成功 | `build/material-m4-verified-build.log` |
| Scene 全部 131 项 | 122 通过、9 跳过、0 失败 | `build/material-m4-scene-all.xml` |
| 材质/GPU/编译/热重载综合 60 项 | 58 通过、1 跳过、1 失败，详见下文 | `build/material-m4-verified/Tests.xml`、`build/material-m4-verified.log` |
| P1 场景执行 | 3 个 Program / 2 个 Family；分箱、PT/VBuffer Layer 边界和二次命中发光线性误差均为 0 | `build/material-m4-verified/ClosureSceneAcceptance.json` |
| Slab MASK winner / shadow / ray | 30,752 个内部像素检查通过 | `build/material-m4-verified/Coverage.txt` |
| Slab 数学/能量/采样/PDF | 23 材质 × 3 角度 × 4096 样本通过 | `build/material-m4-verified/SlabEnergy.txt` |
| 失败用例独立复测 | 后台热重载连续 5 次通过 | `build/material-m4-hot-reload-repeat.log` |
| Rec.709 回归 | OpenPBR stages / Value ABI / 场景发布 / Phase 0 基线 4/4 通过 | `build/material-m4-rec709/Tests.xml` |
| OpenPBR PT / Deferred / RTXCR Chiang 冻结 HDR | 三图与 Phase 0 run-0 逐位一致，有限值检查通过，RGB RMSE/max=0 | `build/material-m4-frozen-comparison.json` |
| LookDev Inspector 严格 CTest | 最终二进制通过真实拖动、撤销/重做、保存/重载和比较帧，64.08 s | `build/material-m4-inspector.log`、`build-scheduling-release/tests/lookdev-material-inspector/editor.log` |

综合组唯一失败是未改动的 `slang_shader_background_hot_reload`，报错为
`deleted shader dependency was not reported`；该断言同时涵盖文件删除失败和通知
等待失败，现有日志不足以区分根因。独立 5 次复测通过不能抹去整组失败，因此不将
综合组标为全绿。新增 P1 用例与其余材质用例均通过；该热重载不稳定项仍待独立定位。

首轮综合测试还暴露旧测试对当前 ACEScg 的期望值及 Coverage 诊断输出编码不匹配。
已修正测试的色彩转换期望与 PT/Deferred 成对诊断配置，没有放宽数值容差或改变
生产色彩行为；修复后 9 项重点回归全部通过，证据为 `build/material-m4-repaired.log`。

Scene 的 9 个跳过项涉及 Zorah、USD/SuperSponza 内容或可选依赖；GPU 跳过项是
缺少 cooked Zorah probes。当前 validation SDK 对 OMM 使用 shader alpha traversal，
本次 Slab MASK 结果不代表 OMM 硬件路径验收。未验证多设备、长时间编辑会话、完整
大场景 VRAM 或 Slab 性能收益。
