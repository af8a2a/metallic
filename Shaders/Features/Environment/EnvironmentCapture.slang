import Atmosphere;
using Metallic;
using Metallic.Atmosphere;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

[numthreads(8, 8, 1)]
void environmentCaptureMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    float3 radiance = float3(0.0);
    if (gParams.dimensions.w == 0u) {
        float2 uv = (float2(id.xy) + 0.5) / float2(gParams.dimensions.xy);
        GPUAtmosphereParameters p = loadParameters(gParams.parameters);
        radiance = sampleSkyView(p, resolveUniform(gParams.skyView), latLongDirection(uv));
    } else {
        uint width, height; resolveUniform(gParams.sourceMip).GetDimensions(width, height);
        uint2 first = id.xy * 2u;
        float total = 0.0;
        for (uint y = first.y; y < min(first.y + 2u, height); ++y) {
            float weight = cos(float(y) * kPi / float(height)) - cos(float(y + 1u) * kPi / float(height));
            for (uint x = first.x; x < min(first.x + 2u, width); ++x) {
                radiance += resolveUniform(gParams.sourceMip).Load(int3(x, y, 0)).xyz * weight;
                total += weight;
            }
        }
        radiance /= max(total, 1e-8);
    }
    resolveUniform(gParams.output)[id.xy] = float4(clamp(radiance, 0.0, 65504.0), 1.0);
}
