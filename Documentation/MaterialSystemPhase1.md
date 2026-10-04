# Phase 1：Material Definition / Instance / Program

本阶段实现外部材质路线图 Phase 1 的 CPU/runtime 资产层，接入已有 M1 的程序和 GPU generation。它与仓库旧 M1/M2 里程碑编号不同。现有 shader、720 字节模型 payload、材质分箱、Slang 请求和 Program key 算法保持不变。

## 三层职责

- `material::MaterialDefinition` 是作者定义：版本、注册实现名与 `MaterialSchema`。Schema 分开描述参数、资源、feature 的类型、范围和默认值，不保存 GPU offset。
- `material::MaterialInstance` 是可序列化的稀疏覆盖：Definition URI、可选 Parent URI、参数/资源/feature overrides。解析结果为 `ResolvedMaterialInstance`，每次解析重新读取依赖，不缓存过时默认值。
- `render::MaterialProgram` 是已有共享执行程序；`MaterialProgramKey` 命名其现有语义 key。Definition 的 `implementation` 通过 `findMaterialProgram(string_view)` 对应既有注册程序。当前注册的资产实现是 `OpenPBRComposite.Legacy`；实例值不进入 key、不增加 variant。

现有 `render::MaterialDefinition/MaterialSchema/MaterialInstance` 是 **GPU 兼容执行描述与快照条目**；它们与新的作者资产分处命名空间，继续管理已发布 generation 与 GPU 布局。作者 schema 不取代 720 字节 ABI。转换链为：

```text
.materialdef defaults -> Parent overrides -> Child overrides
    -> ResolvedMaterialInstance -> scene::RenderMaterial
    -> existing ScenePathTraceResources upload
    -> MaterialGeneration / shared MaterialProgram / existing GPU ABI
```

## 文件与实例继承

入口：[MaterialAsset.h](../Source/Runtime/Material/MaterialAsset.h)。实例示例：[RedPlastic](../Asset/Materials/Examples/RedPlastic.material)、[PolishedRedPlastic](../Asset/Materials/Examples/PolishedRedPlastic.material)；Definition 为 [OpenPBR.materialdef](../Asset/Materials/OpenPBR.materialdef)。这三份文件使用仓库 `Asset/` 作为 asset root。

`.materialdef` 的 `implementation` 选择已注册模型及其固定语义 schema。`defaults.parameters/resources/features` 只需列出模型默认值以外的覆盖。Phase 1 不接受任意 shader 实现名或自定义 schema 类型；新增模型和编译器属于后续阶段。

`version` 是文件格式版本，`definitionVersion` 是语义定义版本，二者独立。更改兼容的默认值后，未 override 的实例在下一次 resolve/场景载入/`reloadMaterialAsset()` 时继承新值；不需要重写实例。显式 override 即使等于当前默认值，也会保留。Parent 必须引用相同 Definition，且每层的 definitionVersion 都匹配；不匹配明确拒绝，不静默迁移。

参数覆盖包括现有 OpenPBR 的 baseColor/opacity、metalness、roughness、emission、normalScale、occlusionStrength、specularWeight/specularColor、transmission/ior/thickness/attenuationDistance/attenuationColor、diffuseTransmission/diffuseTransmissionColor、alphaCutoff 和 displacementMagnitude/displacementCenter。feature 为 alphaMode（opaque/mask/blend）、doubleSided 和 unlit。

资源覆盖支持全部 12 个现有纹理槽（含 displacement）。写法：

```json
{
    "baseColorTexture": "asset://Textures/Wood.ktx2",
    "normalTexture": {
        "uri": "asset://Textures/WoodNormal.png",
        "texCoord": 0,
        "transform": [2, 0, 0, 0, 2, 0]
    },
    "emissiveTexture": null
}
```

null 显式移除继承的绑定；省略则继承。URI 不保存 descriptor 或运行时 textureIndex。`imported://baseColorTexture` 等命名引用目标材质原始导入槽，可保留 glTF/GLB/USD 的嵌入纹理、NTC、采样器身份而不复制资源；这种引用需要同一导入材质上下文，不宣称是独立可分发的贴图文件。UV set/transform 随语义资源值保存。外部 `asset://` 纹理由场景注册并经现有纹理上传/解码路径消费。

资产根内路径 canonicalize 后解析，拒绝逃逸、缺失文件、父链循环、超过 32 层继承、超过 1 MiB 或过深的 JSON、未知字段、非有限/越界值和非法类型。失败解析不替换调用者输出；缺失资源不发布材质或新增纹理。`.material` 保存先验证整个继承链，再用临时文件原子替换。

`upgradeMaterial(document, from, to, error)` 支持明示的 v0 原型到 v1（`metallic` 重命名为 `metalness`，歧义拒绝）；v0 仅为记录的原型格式，不表示仓库曾发布此资产格式。未来版本/降级拒绝。Definition 语义版本不通过该文件格式迁移函数自动转换；现有 GPU schema migration 仍由 M1 的 `migrateMaterialParameters` 负责。

## 使用与场景持久化

```cpp
material::MaterialAssetLibrary library(assetRoot);
material::ResolvedMaterialInstance resolved;
std::string error;
if (!library.resolve("asset://Materials/Examples/RedPlastic.material", resolved, error)) {
    // Report error; published scene remains unchanged.
}

// SceneDocument resolves, binds and remembers the source asset reference.
document.setMaterialAsset(materialIndex,
    "asset://Materials/Examples/RedPlastic.material", assetRoot, error);
document.save(error);
```

Scene sidecar 的 material entry 保留已有 sourceId/local materialIndex/sourceName 校验，增加：

```json
"materialAsset": {
    "uri": "asset://Materials/Examples/RedPlastic.material",
    "root": "relative/path/to/Asset"
}
```

root 相对 sidecar 目录保存。`properties` 只写入相对 resolved asset 的场景局部编辑，因此普通 scene save 不会把 Definition/Parent 默认值冻结成全量 overrides。重新载入先应用资产再应用局部编辑；`reloadMaterialAsset()` 保留局部编辑，重读 Definition/Parent/Instance。这里没有自动文件监视器；调用者选择 reload 边界。

参数编辑沿用 materialRevision 与既有增量上传；切换纹理绑定会改变 scene resource identity，让现有资源系统重建一致的纹理/光追资源。旧引用、GPU 生命周期和失败 GPU 发布继续由原有 generation/frame 机制管理。旧场景的 properties-only material overrides 无需迁移。

`createMaterialInstance()` 将现有 OpenPBR RenderMaterial 导出为相对 Definition 的稀疏资产；调用者通过 ResourceEncoder 提供语义 URI。M2 的自定义 Value Program 不在实例里序列化代码，导出会明确拒绝；给已有带 Value Program 的材质设置模型参数仍保留原有代码/参数。Fiber 本阶段不转成 Surface 资产，现有 RTXCR 路径保留。

## 验证

测试入口为 `MaterialAssets.*`（CPU）、`material_asset_upload_equivalence`（resident / materials-only GPU payload 与共享 Program）、`material_asset_phase0_equivalence`（Phase 0 三条 256 帧 HDR，OpenPBR 先经资产保存/解析/绑定）。

```powershell
cmake --build build-scheduling-release --target MetallicSceneTests MetallicRHITests -j 8
.\build-scheduling-release\tests\MetallicSceneTests.exe --gtest_filter='MaterialAssets.*:SceneEditing.*:SceneComposition.*'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='*material_asset*:*material_runtime_gpu_abi*:*material_runtime_generations*:*material_runtime_scene_publication*:*material_value*' --rhi-validation --output-dir build/material-phase1-gpu
```

Phase 0 的 hardware counters 与 PT A/A 非逐位一致仍按 [Phase 0](MaterialSystemPhase0.md) 记录，不由本阶段的 CPU/ABI 验证代替。

### 2026-10-04 实测结果

- MSVC Release 的 `MetallicSceneTests`、`MetallicRHITests` 构建成功。
- CPU **30/30 通过**：8 项 MaterialAssets、20 项 SceneEditing、2 项 SceneComposition，包含已有 scene sidecar 与 M2 程序持久化兼容性。
- RHI **9 个不同测试最终均通过，无 skip**：GPU ABI、generation、scene publication、4 项 Value Program、三条 Phase 0 图，以及资产上传等价。初次执行的上传测试错误地 map Device 内存；修正为 compute probe 后，又修正了探针缺少 bindless / 默认 requiresRayQuery 的设备配置。最终专门复跑上传测试通过；失败日志保留，没有把它们当作产品路径通过证据。
- 上传测试在 resident / materials-only 两种模式中，用 GPU probe 逐字节读回 720 字节参数，覆盖资产往返、参数更新及真实 2×2 PNG 的 URI 注册/解码/上传，并确认沿用现有共享 Program。
- 三条固定图各运行 **256 帧**，OpenPBR 先导出、保存、解析并绑定 `.material`；RTXCR 保持原路径作为回归对照。与冻结的 Phase 0 `run-0` 比较，OpenPBR PT、OpenPBR Deferred、RTXCR Chiang 的 **RGBA32F 均逐位一致，RGB RMSE / max absolute 均为 0**，且全部有限值。已检查相同曝光与 sRGB 显示变换下的三张预览。这是本次对照结果，不取消 Phase 0 记录的跨进程 PT 非确定性。
- 最终 GPU 上传及其余 8 项运行未出现 Vulkan VUID 错误；环境仍报告缺失的 EOS / `E:\Validation.json` layer manifest，以及旧 validation layer 下 OMM 使用已有 shader alpha traversal。

本地证据在忽略的 `build/` 中：`material-phase1-cpu-final.{log,xml}`；`material-phase1-gpu-final.log`（8 项通过、上传探针旧失败）；`material-phase1-gpu-upload.log` / `material-phase1-gpu-upload/Tests.xml`（修正后的上传测试通过）；`material-phase1-report/comparison.json` 与三张预览。生产 shader、pipeline assets 和 shader warmup 请求未修改；新增 Slang 文件仅用于测试读回。

Phase 1 六项验收（已有 OpenPBR 字段表达、资产 round-trip、父链/稀疏覆盖、默认值继承、旧 GPU 上传等价、无新增生产 shader variant）已覆盖。当前交付是 CPU/runtime API 与 sidecar 接入；没有新增编辑器资产选择面板或自动文件监视器。未执行交互式编辑器操作和大型场景长时间流送/显存压力测试，不将 LookDev 结果扩展为这些路径的验证。
