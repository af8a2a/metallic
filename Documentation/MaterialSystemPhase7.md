# Material System Phase 7 — Feature System

Phase 7 将材质语义、编译策略和最终 shader 选择分开。入口是
[MaterialFeatures](../Source/Runtime/Material/MaterialFeatures.h)，资产、场景、生产 Program 分箱和编辑器共享同一个分析器。
GPU 材质 payload 仍为 720 字节；Feature 元数据属于不可变 CPU material generation。

## 分类与策略

| 语义 | 类别 | 当前处理 |
| --- | --- | --- |
| baseColor、roughness、unlit | Dynamic | 参数/纹理求值，不进入 ProgramSignature |
| metalness | Closure | Auto 分析是否可消除 dielectric/conductor 分支 |
| transmission | Closure、Visibility | 透射存在性影响 closure 与可见性语义 |
| alphaMode | Visibility | opaque/mask/blend 独立签名，不参与 Program 选择 |
| doubleSided | PipelineState | 背面剔除语义独立签名 |
| BSDFImplementation | Specialization | 内置 OpenPBRComposite / RTXCRChiang |
| graphTopology | Structural | 现有 M2 Value Program 的规范化 JSON 代码，排除实例 valueParameters |

`FeatureCategory` 是语义分类位集，与 `SlangMacroDefine` 没有一一对应关系。
现有可配置策略是 metalness/transmission 的 `Auto`、`Dynamic`、`Specialization`、`Closure`：

- Auto 由 compiler 根据目标、静态因子和纹理存在性选择动态求值、静态特化或 closure 分类。
- Dynamic 保留相应动态分支：metalness 使用 Opaque 或 General；transmission 使用 General。
- Specialization / Closure 是优化意图，遵守可用 kernel 和正确性约束，不强行消除可达分支。
  当前 builtin compiler 与 Auto 使用相同的保守分类；它们不是用户指定的 variant ID。
- 金属度为 0 可选 Dielectric；金属度为 1 且没有普通或 NTC 金属度纹理时可选 Conductor；其余非透射表面可选 Opaque。
- 透射、M2 Value Program、Fiber 和 RayHit 目标采用 General。M2 继续使用已有未分箱执行路径。

其他参数没有开放任意特化策略，未知 feature/policy 被拒绝。分类表不表示已经新增 coat 实现或任意 graph 编译器。

## 资产与继承

`.materialdef` 的 `defaults.featurePolicies` 和 `.material` 的 `featurePolicies` 使用语义名称：

```json
"featurePolicies": {
    "metalness": "Auto",
    "transmission": "Dynamic"
}
```

缺省为 Auto，按 definition → parent → child 逐项覆盖；显式 Auto 可以覆盖父级 Dynamic。
旧 v1 资产仍可读取。资产不保存最终 compiler decision、signature、variant ID、GPU handle；未知派生字段仍由严格反序列化拒绝。
`ResolvedMaterialInstance` 包含分析结果，资源只按存在性分析，无需加载纹理。

Scene material 保存作者策略，因此编辑、撤销和重载可以重建分析结果。场景 sidecar 仅保存每个 feature 的局部覆盖，
修改一个 feature 不会冻结另一 feature 的 definition 默认值。失败的解析/上传不会发布部分状态。

## 三种签名与生产接入

`MaterialFeatureResolution` 提供三个 64 位确定性语义签名：

- ProgramSignature：模型、Feature compiler 版本、目标、已选择的 closure kernel、M2 结构代码。
- VisibilitySignature：AlphaMode 与透射存在性；颜色、roughness、alphaCutoff 等运行时数值不在其中。
- PipelineSignature：当前材质所需的背面剔除语义。

三者是当前 compiler 上下文中的语义身份，不是完整 Vulkan pipeline key。
已有 `MaterialProgramKey` 继续包含实际 IR、宏特化、layout、quality、设备能力等 executable 身份。
相同上下文与 ProgramSignature 的实例选择相同 kernel，并复用 executable/cache；不同语义签名也可能复用同一个 general kernel。
当前签名版本为 v1；改变 Feature lowering 规则时需更新版本域。

[ScenePathTraceResources](../Source/Runtime/Render/Streamer/ScenePathTraceResources.cpp) 在同步、异步、材质热更新路径中，
将作者策略和结构代码与 GPU payload 的同一 revision 一起交给 `MaterialGeneration`。
generation 按实际上传的普通/NTC 纹理存在性分析，避免纹理重映射导致错误特化。
[ScenePathTracePass](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) 消费结果选 executable，
仍按完整 ProgramKey 去重和执行 Phase 6 稀疏分箱，不再自行读取 AlphaMode、metalness、transmission 做新路径分类。

`programBinning=false` 的固定五分类路径保留为历史对照，它不应用新 Feature 策略。
`materialBinning=false` 使用通用 kernel。Visibility/Pipeline 签名描述现有 alpha 和 culling 路径，
没有引入 Phase 8 的独立 Coverage IR；VBuffer 仍跳过 alpha blend。

编辑器材质面板的 **Feature diagnostics** 展示 Deferred 与 Ray-hit 两个 compiler 的三个签名，
以及作者策略 → 编译决策。该面板使用同一个分析器，不把派生结果写回资产。

## 验证（2026-10-04）

复用 MSVC/Ninja Release `build-scheduling-release`，构建 `Metallic`、`LookDev`、`MetallicSceneTests`、`MetallicRHITests`。

- `MaterialAssets.*`：13 项通过，覆盖独立签名、Auto/纹理保守分类、结构代码、策略继承、错误拒绝、局部覆盖与保存重载。
- `ctest --test-dir build-scheduling-release -R '^MetallicSceneTests$' --output-on-failure`：通过。
- GPU 材质相关回归：37 项中 36 通过，1 跳过。跳过的 Zorah probe 需要外部 `METALLIC_ZORAH_Z4_PROBES` 数据。
- 新增 `material_feature_policy_rendering`：生产 Deferred、256×256、8 帧，Auto 与 Dynamic 原始 RGBA32F 逐位一致，relative RMSE 和最大绝对误差均为 0；连续帧没有新增 executable pipeline。
- generation 测试覆盖普通/NTC 分类、源数量不匹配拒绝、策略热更新及旧 snapshot 保持；GPU cache 测试覆盖相同 ProgramSignature、不同 AlphaMode/doubleSided/动态值实例的 artifact 复用。
- 实际 LookDev 编辑器 smoke 的拖动、undo/redo、保存重载、双路径渲染和签名 UI 文本断言全部通过。
  CTest 严格的零 Vulkan 消息门禁仍失败：loader 报缺失 EOS overlay / `E:\Validation.json`，以及实例函数通过 `vkGetDeviceProcAddr` 查询的警告。
  没有修改过滤规则或关闭验证来隐藏这些消息。

完整 GPU 测试集的 256 帧冻结 HDR 对比中，Deferred 和 RTXCR Chiang 与 Phase 0 逐位一致；
PathTrace 的 RGB RMSE 为 0.0005241747、最大绝对误差为 0.05389053。完整套件上下文中的 PT 差异原因尚未定位，不能据此宣称所有输出逐位一致。
随后单独运行 `*material_asset_phase0_equivalence*`：三项原始 RGBA32F 均与 Phase 0 逐位一致。
最后新增的签名/cache 断言由 `*material_runtime_inflight_reload*` 单独重跑通过。
原始输出、日志、XML 与预览位于本地 `build/material-phase7-*`，不纳入源代码。
