import ShaderCore;
using Metallic;

#if FIBER_REFERENCE
#include "../../../External/RTXCR-Material/shaders/include/rtxcr/HairChiangBSDF.hlsli"
#include "../../../External/RTXCR-Material/shaders/include/rtxcr/HairFarFieldBCSDF.hlsli"
#define HairMaterialData RTXCR_HairMaterialData
#define HairInteractionSurface RTXCR_HairInteractionSurface
#define HairMaterialSeparateChiangData RTXCR_HairMaterialSeparateChiangData
#define createHairMaterialInteraction RTXCR_CreateHairMaterialInteraction
#define createHairMaterialInteractionBCSDF RTXCR_CreateHairMaterialInteractionBcsdf
#define hairChiangBSDFEval RTXCR_HairChiangBsdfEval
#define hairSeparateChiangBSDFEval RTXCR_HairSeparateChiangBsdfEval
#define hairFarFieldBCSDFEval RTXCR_HairFarFieldBcsdfEval
#define sampleChiangBSDF RTXCR_SampleChiangBsdf
#define sampleSeparateChiangBSDF RTXCR_SampleSeparateChiangBsdf
#define sampleFarFieldBCSDF RTXCR_SampleFarFieldBcsdf
#else
import RTXCRHair;
import RTXCRFiber;
import FiberMaterial;
import FiberLighting;
import MaterialProgram;
using Metallic.RTXCR;
using Metallic.Material;
using Metallic.Lighting;
#endif

struct Params { RWBufferSpan<float4> output; uint lightCount; };
[[vk::push_constant]] ConstantBuffer<Params> gProbe;
float random(inout uint state)
{
    state ^= state >> 16; state *= 0x7feb352du; state ^= state >> 15;
    state *= 0x846ca68bu; state ^= state >> 16;
    return (float(state >> 8) + 0.5) * (1.0 / 16777216.0);
}

#if !FIBER_REFERENCE
struct Parameters : IRTXCRHairParameterProvider
{
    HairMaterialData material;
    uint loads;
    uint offset;
    [mutating] HairMaterialData load(MaterialInstanceRef instance)
    {
        ++loads; offset = instance.parameterOffset;
        return material;
    }
};
struct Lights : IFiberLightProvider
{
    float3 wi;
    uint samples;
    uint count() { return gProbe.lightCount; }
    [mutating] FiberLightSample sample(uint index)
    {
        ++samples;
        FiberLightSample light;
        light.direction = wi;
        light.illuminance = float3(2, 3, 4) / float(gProbe.lightCount);
        light.inverseSourcePdf = 1;
        return light;
    }
    float3 visibility(FiberLightSample light) { return 1; }
};
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void fiberProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x, state = lane + 12345, base = lane * 16;
    HairMaterialData material;
    material.baseColor = float3(random(state), random(state), random(state));
    material.melanin = random(state);
    material.melaninRedness = random(state);
    material.absorptionModel = lane % 3;
    material.fresnelApproximation = lane % 2;
    material.ior = 1.01 + random(state) * 1.99;
    material.eta = 1 / material.ior;
    material.longitudinalRoughness = lane % 4 == 0 ? 0.02 : 0.05 + random(state) * 0.95;
    material.azimuthalRoughness = lane % 4 == 1 ? 0.02 : 0.05 + random(state) * 0.95;
    material.cuticleAngleInDegrees = random(state) * 20 - 10;
    float3 wo = normalize(float3(random(state) * 2 - 1, random(state) * 2 - 1, 0.001 + random(state)));
    float3 wi = normalize(float3(random(state) * 2 - 1, random(state) * 2 - 1, random(state) * 2 - 1));
    HairInteractionSurface surface;
    surface.shadingNormal = lane % 2 == 0 ? float3(0, 0, 1) : normalize(float3(0.3, 0.4, 1));
    surface.tangent = normalize(cross(float3(0, 1, 0), surface.shadingNormal));
    surface.incidentRayDirection = wo;
    float2 u[2] = {float2(random(state), random(state)), float2(random(state), random(state))};
    float3 direction, numerator; float density; uint event;
    bool valid = sampleChiangBSDF(material, surface, u, direction, density, numerator, event);
    if (!valid) { direction = numerator = 0; density = 0; event = 0; }
    float3 value = hairChiangBSDFEval(material, surface, wi);
    gProbe.output.store(base, float4(value, density));
    gProbe.output.store(base + 1, float4(direction, float(event)));
    gProbe.output.store(base + 2, float4(numerator, float(valid)));

    HairMaterialSeparateChiangData separate;
    separate.base = material;
    separate.longitudinalRoughnessTT = 0.15;
    separate.longitudinalRoughnessTRT = 0.4;
    separate.azimuthalRoughnessTT = 0.3;
    separate.azimuthalRoughnessTRT = 0.6;
    value = hairSeparateChiangBSDFEval(separate, surface, wi);
    valid = sampleSeparateChiangBSDF(separate, surface, u, direction, density, numerator, event);
    if (!valid) { direction = numerator = 0; density = 0; event = 0; }
    gProbe.output.store(base + 3, float4(value, density));
    gProbe.output.store(base + 4, float4(direction, float(event)));
    gProbe.output.store(base + 5, float4(numerator, float(valid)));

    let far = createHairMaterialInteractionBCSDF(material, float3(0.2, 0.4, 0.8), 0.25, 0.3);
    float3 diffuse;
    hairFarFieldBCSDFEval(surface, far, wi, wo, value, diffuse, density);
    gProbe.output.store(base + 6, float4(value, density));
    gProbe.output.store(base + 7, float4(diffuse, 0));
    valid = sampleFarFieldBCSDF(surface, far, wo, 0.2, random(state), u, direction, numerator, diffuse, density);
    if (!valid) { direction = numerator = diffuse = 0; density = 0; }
    gProbe.output.store(base + 8, float4(direction, density));
    gProbe.output.store(base + 9, float4(numerator, float(valid)));
    gProbe.output.store(base + 10, float4(diffuse, 0));

#if FIBER_REFERENCE
    gProbe.output.store(base + 11, float4(hairChiangBSDFEval(material, surface, wi) * float3(2, 3, 4), 1));
    valid = sampleChiangBSDF(material, surface, u, direction, density, numerator, event);
    if (!valid) { direction = numerator = 0; density = 0; event = 0; }
    gProbe.output.store(base + 12, float4(direction, density));
    gProbe.output.store(base + 13, float4(max(numerator, 0) / max(density, 1e-6), float(event)));
    gProbe.output.store(base + 14, float4(1, float(gProbe.lightCount), float(lane * 64), 0));
    gProbe.output.store(base + 15, 0);
#else
    RTXCRChiangMaterialProgram<Parameters> program;
    program.parameters.material = material;
    program.parameters.loads = 0;
    Lights lights; lights.wi = wi; lights.samples = 0;
    FiberMaterialContext context = (FiberMaterialContext)0;
    context.normalWS = surface.shadingNormal;
    context.tangentWS = surface.tangent;
    context.bitangentWS = cross(surface.shadingNormal, surface.tangent);
    MaterialInstanceRef instance = (MaterialInstanceRef)0;
    instance.parameterOffset = lane * 64;
    let shaded = shadeFiber(program, context, instance, wo, TransportMode.Radiance, lights);
    let scatter = shaded.prepared.sampleWeighted(float4(u[0], u[1]));
    gProbe.output.store(base + 11, float4(shaded.radiance, 1));
    gProbe.output.store(base + 12, float4(scatter.wiWS, scatter.pdf));
    gProbe.output.store(base + 13, float4(scatter.weight, float(scatter.eventType)));
    gProbe.output.store(base + 14, float4(float(program.parameters.loads), float(lights.samples), float(program.parameters.offset), 0));
    let unsupported = lane % 2 == 0
        ? shaded.material.closure.prepare(wo, TransportMode.Importance)
        : shaded.material.closure.prepare(surface.tangent, TransportMode.Radiance);
    gProbe.output.store(base + 15, float4(unsupported.evalProjected(wi), unsupported.sampleWeighted(float4(u[0], u[1])).pdf));
#endif
}
