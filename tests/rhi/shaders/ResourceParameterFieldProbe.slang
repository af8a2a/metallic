import Core;
using Metallic;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    let output = resolveBuffer<RWByteAddressBuffer>(getConstants<ResourceHandle<ByteAddressBuffer>>());
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
    output.Store(56, getResourceParameterField<SceneResourceParameters, uint>(452));
    output.Store(60, getResourceParameterField<SceneResourceParameters, uint>(456));
    output.Store(64, getResourceParameterField<SceneResourceParameters, uint>(0xfffffffc));
    output.Store(68, getResourceParameterField<SceneResourceParameters, uint>(1));
}
