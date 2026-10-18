#language slang 2026
module OpenPBRFresnelProbe;

import ShaderCore;
import OpenPBR;
using Metallic;
using Metallic.OpenPBR;

struct FresnelProbeParams
{
    BufferSpan<float4> inputs;
    RWBufferSpan<float4> output;
};
[[vk::push_constant]] ConstantBuffer<FresnelProbeParams> gProbe;

// Frozen pre-optimization formula. Keep this independent of production Fresnel
// and its square/sqrt helpers so a future edit cannot change both sides.
float originalFresnel(float eta, float cosine)
{
    if (eta == 1.0f) { return 0.0f; }
    float sinI2 = 1.0f - cosine * cosine;
    float sinT2 = sinI2 / (eta * eta);
    if (sinT2 >= 1.0f) { return 1.0f; }
    float cosT = sqrt(1.0f - sinT2);
    float etaCosI = eta * cosine;
    float etaCosT = eta * cosT;
    float rs = (cosine - etaCosT) / (cosine + etaCosT);
    float rp = (cosT - etaCosI) / (cosT + etaCosI);
    return 0.5f * (rs * rs + rp * rp);
}

float originalTIR(float eta, float cosine)
{
    return eta != 1.0f && (1.0f - cosine * cosine) / (eta * eta) >= 1.0f ? 1.0f : 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRFresnelMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gProbe.inputs.count) { return; }
    float4 input = gProbe.inputs.load(tid.x);
    float3 eta = input.xyz;
    float cosine = input.w;
    float3 reference = float3(originalFresnel(eta.x, cosine), originalFresnel(eta.y, cosine),
        originalFresnel(eta.z, cosine));
    uint offset = tid.x * 5;
    gProbe.output.store(offset, float4(openPBRFresnelRgb(eta, cosine, true), openPBRThinWallFresnel(eta.x, cosine)));
    gProbe.output.store(offset + 1, float4(reference, (2.0f * reference.x) / (1.0f + reference.x)));
    gProbe.output.store(offset + 2, float4(openPBRFresnelRgb(eta, cosine, false), openPBRFresnel(eta.x, cosine)));
    gProbe.output.store(offset + 3, float4(float3(reference.x), reference.x));
    gProbe.output.store(offset + 4, float4(originalTIR(eta.x, cosine), originalTIR(eta.y, cosine),
        originalTIR(eta.z, cosine), 1.0f));
}

// Both entry points use the same uploaded inputs and loop. The constant argument
// removes the unused Fresnel implementation from each compiled entry point.
[ForceInline]
void benchmarkFresnel(uint lane, bool production)
{
    float4 input = gProbe.inputs.load(lane);
    float cosine = input.w;
    float sum = 0;
    [loop]
    for (uint iteration = 0; iteration < 128; ++iteration) {
        cosine = frac(cosine + 0.61803398875f);
        sum += production ? openPBRFresnel(input.x, cosine) : originalFresnel(input.x, cosine);
    }
    gProbe.output.store(lane, float4(sum, sum / 128.0f, cosine, 1.0f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRFresnelBenchmarkOriginal(uint3 tid : SV_DispatchThreadID)
{
    benchmarkFresnel(tid.x, false);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRFresnelBenchmarkProduction(uint3 tid : SV_DispatchThreadID)
{
    benchmarkFresnel(tid.x, true);
}
