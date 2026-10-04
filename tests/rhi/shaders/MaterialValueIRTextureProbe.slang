// Uses the production Value IR dispatch and deterministic footprint adapter.
#include "../../../Shaders/Features/PathTracing/SceneSurface.slang"
[shader("compute")]
[numthreads(1, 1, 1)]
void materialValueIRTextureProbeMain(uint3 id : SV_DispatchThreadID)
{
    PathTraceMaterial material = (PathTraceMaterial)0;
    material.baseColorTexture.textureIndex = 0;
    material.baseColorTexture.ntcTextureSetIndex = 0xffffffffu;
    material.baseColorTexture.transform0 = float4(1, 0, 0, 0);
    material.baseColorTexture.transform1 = float4(0, 1, 0, 0);
    MaterialValueTextureContext context;
    context.textureCount = 1; context.ntcCount = 0; context.normalizedRayLod = 0;
    PathTraceMaterial result = evaluateMaterialValue(id.x, 0, float3(0, 0, 1), float2(0.25), material, context);
    resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().probeOutput)[id.x] = result.baseColor;
}
