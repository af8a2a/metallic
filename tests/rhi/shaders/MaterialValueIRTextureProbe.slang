// Uses the production Value IR dispatch and deterministic footprint adapter.
#include "../../../Shaders/Features/PathTracing/SceneSurface.slang"
[shader("compute")]
[numthreads(1, 1, 1)]
void materialValueIRTextureProbeMain(uint3 id : SV_DispatchThreadID)
{
    PathTraceMaterial material = (PathTraceMaterial)0;
    material.baseColorTexture.textureIndex = 0;
    material.baseColorTexture.ntcTextureSetIndex = 0xffffffffu;
    MaterialValueInstance instance = resolveBuffer<StructuredBuffer<MaterialValueInstance>>(getResourceParameters<SceneResourceParameters>().materialValues)[id.x];
    material.baseColorTexture.transform0 = float4(1, 0, 0, instance.parameters[3].x);
    material.baseColorTexture.transform1 = float4(0, 1, 0, 0);
    material.emissiveTexture = material.baseColorTexture;
    MaterialValueTextureContext context;
    context.textureCount = 1; context.ntcCount = 0; context.normalizedRayLod = 0;
    PathTraceMaterial result = evaluateMaterialValue(id.x, 0, float3(0, 0, 1), float2(0.25), material, context);
    resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().probeOutput)[id.x] =
        id.x == 11u || id.x == 12u ? result.emissive : result.baseColor;
}
