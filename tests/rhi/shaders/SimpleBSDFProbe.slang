#language slang 2026
module SimpleBSDFProbe;

import MaterialProgram;
import VkGltfBSDF;
#if METALLIC_SIMPLE_BSDF_PROBE_GENERIC
import SimpleBSDF;
#endif

using Metallic.Material;

StructuredBuffer<Metallic.VkGltf.PreparedMaterial> probeMaterials;
StructuredBuffer<float4> probeDirections;
RWStructuredBuffer<float4> probeOutput;

#if METALLIC_SIMPLE_BSDF_PROBE_GENERIC
public struct ProbeSimpleBSDFSource : ISimpleBSDFSource
{
    public Metallic.VkGltf.PreparedMaterial inputs;
    public uint resolveCount;
    public uint seed;
    public uint resolvedOffset;
    public float resolvedPosition;

    [mutating]
    public Metallic.VkGltf.PreparedMaterial resolve(
        SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        resolveCount++;
        seed++;
        resolvedOffset = instance.parameterOffset + instance.resourceOffset;
        resolvedPosition = context.positionWS.x;
        return inputs;
    }
};
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void simpleBSDFProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    let inputs = probeMaterials[lane];
    float3 wo = probeDirections[lane * 3].xyz;
    float3 wi = probeDirections[lane * 3 + 1].xyz;
    float3 random = probeDirections[lane * 3 + 2].xyz;
    Metallic.VkGltf.BSDFEvaluation evaluated;
    Metallic.VkGltf.BSDFSample sampled;
    float4 emissionAndFlags;
    float4 sourceState;
    float4 contract;
#if METALLIC_SIMPLE_BSDF_PROBE_GENERIC
    Material<SimpleBSDFMaterialProgram<ProbeSimpleBSDFSource>> material = {};
    material.program.source.inputs = inputs;
    material.program.source.seed = lane & 255u;
    SurfaceMaterialContext context = {};
    context.positionWS.x = float(lane) + 0.25;
    MaterialInstanceRef instance;
    instance.parameterOffset = 17u;
    instance.resourceOffset = 23u;
    let resolved = material.evaluate(context, instance);
    let prepared = prepareSimpleBSDF(resolved.closure, wo, TransportMode.Radiance);
    evaluated = evaluateSimpleBSDF(prepared, wi, random);
    sampled = sampleSimpleBSDF(prepared, random);
    let importance = prepareSimpleBSDF(resolved.closure, wo, TransportMode.Importance);
    let invalidEval = evaluateSimpleBSDF(importance, wi, random);
    let invalidSample = sampleSimpleBSDF(importance, random);
    bool rejected = all(invalidEval.projectedDiffuse == float3(0)) &&
        all(invalidEval.projectedGlossy == float3(0)) && invalidEval.pdf == 0.0 &&
        all(invalidSample.direction == float3(0)) && all(invalidSample.weight == float3(0)) &&
        invalidSample.pdf == 0.0 && invalidSample.eventType == Metallic.VkGltf.kEventAbsorb;
    emissionAndFlags = float4(resolved.emission, float(resolved.flags));
    sourceState = float4(float(material.program.source.resolveCount), float(material.program.source.seed),
        float(material.program.source.resolvedOffset), material.program.source.resolvedPosition);
    contract = float4(float(rejected), prepared.shadingNormal());
#else
    evaluated = Metallic.VkGltf.evaluateBSDF(inputs, wo, wi, random);
    sampled = Metallic.VkGltf.sampleBSDF(inputs, wo, random);
    emissionAndFlags = float4(inputs.emissive, 0);
    sourceState = float4(1, float((lane & 255u) + 1u), 40, float(lane) + 0.25);
    contract = float4(1, inputs.N);
#endif
    probeOutput[lane * 8] = float4(evaluated.projectedDiffuse, evaluated.pdf);
    probeOutput[lane * 8 + 1] = float4(evaluated.projectedGlossy, float(sampled.eventType));
    probeOutput[lane * 8 + 2] = float4(sampled.direction, sampled.pdf);
    probeOutput[lane * 8 + 3] = float4(sampled.weight, 0);
    probeOutput[lane * 8 + 4] = float4(Metallic.VkGltf.volumeExtinctionCoefficient(inputs),
        Metallic.VkGltf.henyeyGreenstein(dot(wo, wi), inputs.scatterAnisotropy));
    probeOutput[lane * 8 + 5] = emissionAndFlags;
    probeOutput[lane * 8 + 6] = sourceState;
    probeOutput[lane * 8 + 7] = contract;
}
