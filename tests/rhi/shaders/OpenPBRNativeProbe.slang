import ShaderCore;
using Metallic;

struct ProbeParams
{
    BufferSpan<float4> parameters;
    RWBufferSpan<float4> output;
    ResourceHandle<Texture2D<float4>> texture;
    ResourceHandle<RWByteAddressBuffer> counters;
    uint lightCount;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gProbe;

static const uint kOffsets[8] = {2, 32770, 33794, 34818, 67586, 68610, 69634, 69666};
float4 lutTexel(int id, int3 p, int3 size)
{
    return gProbe.parameters.load(kOffsets[id] + uint((p.z * size.y + p.y) * size.x + p.x));
}
float4 lut(int id, float3 uvw, int3 size)
{
    float3 p = clamp(uvw * float3(size) - 0.5, float3(0), float3(size - 1));
    int3 lo = int3(floor(p));
    int3 hi = min(lo + 1, size - 1);
    float3 f = p - float3(lo);
    float4 a = lerp(lerp(lutTexel(id, lo, size), lutTexel(id, int3(hi.x, lo.yz), size), f.x),
        lerp(lutTexel(id, int3(lo.x, hi.y, lo.z), size), lutTexel(id, int3(hi.xy, lo.z), size), f.x), f.y);
    float4 b = lerp(lerp(lutTexel(id, int3(lo.xy, hi.z), size), lutTexel(id, int3(hi.x, lo.y, hi.z), size), f.x),
        lerp(lutTexel(id, int3(lo.x, hi.yz), size), lutTexel(id, hi, size), f.x), f.y);
    return lerp(a, b, f.z);
}

#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0
#define OPENPBR_FEATURE_EnableSheenAndCoat true
#define OPENPBR_FEATURE_EnableDispersion true
#define OPENPBR_FEATURE_EnableTranslucency true
#define OPENPBR_FEATURE_EnableMetallic true
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) OPENPBR_FEATURE_##name
#define OPENPBR_SAMPLE_2D_TEXTURE(id, uv) lut(id, float3(uv, 0.5), int3(32, id == 6 ? 1 : 32, 1))
#define OPENPBR_SAMPLE_3D_TEXTURE(id, uvw) lut(id, uvw, int3(32))
#if METALLIC_OPENPBR_REFERENCE
#include "../../../External/openpbr-bsdf/openpbr.h"
#else
#include "../../../Shaders/Interop/OpenPBRModule.hlsli"
#endif

float random(inout uint state)
{
    state ^= state >> 16; state *= 0x7feb352du; state ^= state >> 15;
    state *= 0x846ca68bu; state ^= state >> 16;
    return float(state >> 8) * (1.0 / 16777216.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRNativeProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    // Both divergent waves and coherent material regions use the same cases.
    uint variant = gProbe.lightCount == 0 ? lane % 16 : lane / 2048;
    uint state = lane + 12345;
    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();
    inputs.geometry_basis = openpbr_make_basis(float3(0, 0, 1));
    inputs.geometry_coat_basis = openpbr_make_basis(normalize(float3(0.12, -0.08, 1)));
    inputs.base_color = float3(random(state), random(state), random(state));
    inputs.specular_roughness = ((lane & 256) != 0) ? max(0.001, random(state)) : 0.001;
    inputs.base_diffuse_roughness = random(state);
    inputs.specular_ior = ((lane & 512) != 0) ? 1.5 : 3.1;
    inputs.specular_weight = ((lane & 1024) != 0) ? 0.4 : 1.0;
    inputs.base_metalness = variant == 1 || variant == 9 ? 1 : variant == 14 ? 0.5 : 0;
    inputs.specular_roughness_anisotropy = variant >= 8 ? 0.85 : 0;
    inputs.specular_anisotropy_rotation_cos_sin = float2(0.6, 0.8);
    inputs.transmission_weight = variant == 2 || variant == 3 || variant == 7 || variant >= 10 ? 0.9 : 0;
    inputs.transmission_color = float3(0.8, 0.5, 0.3);
    inputs.transmission_depth = variant == 10 ? 0.4 : 0;
    inputs.transmission_scatter = variant == 10 ? float3(0.1, 0.2, 0.3) : float3(0);
    inputs.transmission_scatter_anisotropy = 0.3;
    inputs.geometry_thin_walled = variant == 3 || variant == 12;
    inputs.coat_weight = variant == 4 || variant >= 13 ? 0.8 : 0;
    inputs.coat_color = float3(0.7, 0.85, 1);
    inputs.coat_roughness = 0.18;
    inputs.coat_roughness_anisotropy = 0.4;
    inputs.fuzz_weight = variant == 5 || variant >= 13 ? 0.7 : 0;
    inputs.fuzz_color = float3(0.6, 0.8, 0.9);
    inputs.fuzz_roughness = 0.4;
    inputs.thin_film_weight = variant == 6 || variant == 9 || variant == 14 ? 1 : 0;
    inputs.thin_film_thickness = 0.42;
    inputs.transmission_dispersion_scale = variant == 7 || variant == 14 ? 1 : 0;
    inputs.subsurface_weight = variant == 11 || variant == 12 ? 0.8 : 0;
    inputs.emission_luminance = 3;
    if (variant == 15) {
        inputs.base_weight = inputs.specular_weight = inputs.transmission_weight = 0;
        inputs.fuzz_weight = inputs.coat_weight = inputs.emission_luminance = 0;
    }
    float3 wo = normalize(float3(random(state) * 2 - 1, random(state) * 2 - 1,
        ((lane & 32) != 0) ? 0.001 : 1));
    if ((lane & 64) != 0) { wo = -wo; }
    float3 wi = normalize(float3(random(state) * 2 - 1, random(state) * 2 - 1, random(state) * 2 - 1));
    float3 throughput = float3(0.3, 0.6, 0.9);
    let bsdf = openpbr_prepare(inputs, throughput, float3(610, 550, 460), 1.1, wo);
    let value = openpbr_eval(bsdf, wi);
    float pdf = openpbr_pdf(bsdf, wi);
    float3 direction;
    OpenPBR_DiffuseSpecular weight;
    float density;
    OpenPBR_BsdfLobeType eventType;
    float3 rand = float3(random(state), random(state), random(state));
    openpbr_sample(bsdf, rand, direction, weight, density, eventType);
    if (density <= 0) { direction = 0; weight = (OpenPBR_DiffuseSpecular)0; eventType = 0; }
    uint base = lane * 8;
    gProbe.output.store(base, float4(value.diffuse, pdf));
    gProbe.output.store(base + 1, float4(value.specular, density));
    gProbe.output.store(base + 2, float4(direction, float(eventType)));
    gProbe.output.store(base + 3, float4(weight.diffuse, 0));
    gProbe.output.store(base + 4, float4(weight.specular, 0));
    gProbe.output.store(base + 5, float4(bsdf.emission, 0));
    gProbe.output.store(base + 6, float4(bsdf.volume.extinction_coefficient, bsdf.volume.anisotropy));
    gProbe.output.store(base + 7, float4(bsdf.volume.albedo, 0));
}
