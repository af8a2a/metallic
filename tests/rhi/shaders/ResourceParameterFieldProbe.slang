import Core;
using Metallic;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    // The short-root variant retains these two constant-span words in its
    // 20-byte allocation. Address the test output independently of the full
    // root guard so the field reads can prove that they return zero.
    let constantAddress = resolveUniform(gParameterRoot.resource).Load<uint2>(
        gParameterRoot.byteOffset + sizeof(BufferSpan<uint>));
    let outputHandle = resolveUniform(ResourceHandle<ByteAddressBuffer>(constantAddress.x))
        .Load<ResourceHandle<ByteAddressBuffer>>(constantAddress.y);
    let output = resolveBuffer<RWByteAddressBuffer>(outputHandle);
    let guides = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
    let sharc = getResourceParameterField<SceneResourceParameters, SceneSharcResourceParameters>(kSceneSharcOffset);
    output.Store(0, getResourceParameterField<SceneResourceParameters, ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationCurrentOffset).index);
    output.Store(4, getResourceParameterField<SceneResourceParameters, ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationPreviousOffset).index);
    output.Store(8, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneCacheParamsOffset).index);
    output.Store(12, getResourceParameterField<SceneResourceParameters, ResourceHandle<RWTexture2D<float4>>>(kSceneOutputOffset).index);
    output.Store(16, guides.guideAlbedo.index);
    output.Store(20, guides.guideDepth.index);
    output.Store(24, guides.guideLinearDepth.index);
    output.Store(28, guides.guideMotion.index);
    output.Store(32, guides.guideNormalRoughness.index);
    output.Store(36, guides.guideSpecularAlbedo.index);
    output.Store(40, guides.guideSpecularHitDistance.index);
    output.Store(44, sharc.sharcAccumulation.index);
    output.Store(48, sharc.sharcHashEntries.index);
    output.Store(52, sharc.sharcResolved.index);
    output.Store(56, getResourceParameterField<SceneResourceParameters, uint>(sizeof(SceneResourceParameters) - sizeof(uint)));
    output.Store(60, getResourceParameterField<SceneResourceParameters, uint>(sizeof(SceneResourceParameters)));
    output.Store(64, getResourceParameterField<SceneResourceParameters, uint>(0xfffffffc));
    output.Store(68, getResourceParameterField<SceneResourceParameters, uint>(1));
    let scene = getResourceParameterField<SceneResourceParameters, uint64_t>(kSceneSceneOffset);
    output.Store2(72, uint2(uint(scene), uint(scene >> 32)));

    let materialTextures = getResourceParameterField<SceneResourceParameters, SceneMaterialTextureResourceParameters>(kSceneMaterialSamplerOffset);
    output.Store(80, materialTextures.materialSampler.index);
    output.Store(84, materialTextures.materialTextures.resource.index);
    output.Store(88, materialTextures.materialTextures.byteOffset);
    output.Store(92, materialTextures.materialTextures.count);
    let openPBRLuts = getResourceParameterField<SceneResourceParameters, SceneOpenPBRLUTResourceParameters>(kSceneOpenPBRLUTsOffset);
    output.Store(96, openPBRLuts.openPBRLut2D.resource.index);
    output.Store(100, openPBRLuts.openPBRLut2D.byteOffset);
    output.Store(104, openPBRLuts.openPBRLut2D.count);
    output.Store(108, openPBRLuts.openPBRLut3D.resource.index);
    output.Store(112, openPBRLuts.openPBRLut3D.byteOffset);
    output.Store(116, openPBRLuts.openPBRLut3D.count);
    output.Store(120, openPBRLuts.openPBRLutSampler.index);
    let rayStream = getResourceParameterField<SceneResourceParameters, SceneRayStreamResourceParameters>(kSceneRayStreamOffset);
    output.Store(124, rayStream.rayStreamHeader.index);
    output.Store(128, rayStream.rayStreamInstances.index);
    output.Store(132, rayStream.rayStreamPageTable.index);
    output.Store(136, rayStream.rayStreamPages.index);
    let lights = getResourceParameterField<SceneResourceParameters, SceneLightResourceParameters>(kSceneLightsOffset);
    output.Store(140, lights.lights.index);
    output.Store(144, lights.lightsPdf.index);
    let environment = getResourceParameterField<SceneResourceParameters, EnvironmentResourceParameters>(kSceneEnvironmentOffset);
    output.Store(148, environment.celestialLights.index);
    output.Store(152, environment.atmosphereParameters.index);
    output.Store(156, environment.transmittance.index);
    output.Store(160, environment.multiScattering.index);
    output.Store(164, environment.skyView.index);
    output.Store(168, environment.aerialPerspective.index);
    output.Store(172, environment.cloudShadow.index);

    output.Store(176, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset).index);
    output.Store(180, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneMaterialValuesOffset).index);
    let textureSpan = getResourceParameterField<SceneResourceParameters, BufferSpan<uint>>(kSceneMaterialTexturesOffset);
    output.Store(184, textureSpan.resource.index);
    output.Store(188, textureSpan.byteOffset);
    output.Store(192, textureSpan.count);
    output.Store(196, getResourceParameterField<SceneResourceParameters, SamplerHandle>(kSceneMaterialSamplerOffset).index);
    output.Store(200, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamHeaderOffset).index);
    output.Store(204, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamInstancesOffset).index);
    output.Store(208, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPageTableOffset).index);
    output.Store(212, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamPagesOffset).index);
    output.Store(216, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightsOffset).index);
    output.Store(220, getResourceParameterField<SceneResourceParameters, ResourceHandle<Texture2D<float>>>(kSceneLightsPdfOffset).index);
    output.Store(224, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneReGIRGridOffset).index);
    output.Store(228, getResourceParameterField<SceneResourceParameters, ResourceHandle<Texture2D<float4>>>(kSceneEnvironmentMapOffset).index);
    output.Store(232, getResourceParameterField<SceneResourceParameters, ResourceHandle<Texture2D<float>>>(kSceneEnvironmentPdfOffset).index);
    output.Store(236, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamChangesOffset).index);
    output.Store(240, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneStreamPageCountOffset).index);
    output.Store(244, getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureFeedbackOffset).index);
}
