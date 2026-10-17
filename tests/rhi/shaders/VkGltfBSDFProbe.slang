#language slang 2026
module VkGltfBSDFProbe;

import VkGltfBSDF;
using Metallic.VkGltf;

// Pure BSDF probe. No production resource packet or OpenPBR dependency.
StructuredBuffer<PreparedMaterial> probeMaterials;
StructuredBuffer<float4> probeDirections;
RWStructuredBuffer<float4> probeOutput;

[shader("compute")]
[numthreads(64, 1, 1)]
void vkGltfBSDFProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    PreparedMaterial mat = probeMaterials[lane];
    float3 k1 = probeDirections[lane * 3].xyz;
    float3 k2 = probeDirections[lane * 3 + 1].xyz;
    float3 xi = probeDirections[lane * 3 + 2].xyz;
    BSDFEvaluation eval = evaluateBSDF(mat, k1, k2, xi);
    BSDFSample sample = sampleBSDF(mat, k1, xi);
    probeOutput[lane * 5] = float4(eval.projectedDiffuse, eval.pdf);
    probeOutput[lane * 5 + 1] = float4(eval.projectedGlossy, float(sample.eventType));
    probeOutput[lane * 5 + 2] = float4(sample.direction, sample.pdf);
    probeOutput[lane * 5 + 3] = float4(sample.weight, 0);
    probeOutput[lane * 5 + 4] = float4(volumeExtinctionCoefficient(mat),
        henyeyGreenstein(dot(k1, k2), mat.scatterAnisotropy));
}
