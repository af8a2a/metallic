import ShaderCore;
using Metallic;

struct CloudIBLProbeParams
{
    ResourceHandle<Texture2D<float4>> radiance;
    ResourceHandle<Texture2D<float>> pdf;
    BufferSpan<float4> harmonics;
    BufferSpan<float4> specular;
    RWBufferSpan<float4> output;
    uint reserved;
};
[[vk::push_constant]] ConstantBuffer<CloudIBLProbeParams> gProbe;

[numthreads(1, 1, 1)]
void cloudIBLProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id != 0u)) { return; }
    let radiance = resolveUniform(gProbe.radiance);
    let pdf = resolveUniform(gProbe.pdf);
    float3 captureMean = float3(0.0), specularSharp = float3(0.0), specularRough = float3(0.0);
    float solidWeight = 0.0;
    uint width, height; radiance.GetDimensions(width, height);
    for (uint sample = 0; sample < 2048u; ++sample) {
        uint x = sample % 64u, y = sample / 64u;
        float weight = sin((float(y) + 0.5) * 3.14159265358979323846 / 32.0);
        int2 pixel = int2((x * width + width / 2u) / 64u, (y * height + height / 2u) / 32u);
        captureMean += radiance.Load(int3(pixel, 0)).xyz * weight;
        uint specularPixel = ((y * 128u + 64u) / 32u) * 256u + (x * 256u + 128u) / 64u;
        specularSharp += gProbe.specular[specularPixel].xyz * weight;
        specularRough += gProbe.specular[7u * 256u * 128u + specularPixel].xyz * weight;
        solidWeight += weight;
    }
    gProbe.output.store(0u, float4(captureMean / solidWeight, 1.0));
    gProbe.output.store(1u, float4(specularSharp / solidWeight, 1.0));
    gProbe.output.store(2u, float4(specularRough / solidWeight, 1.0));
    for (uint index = 0u; index < 9u; ++index) {
        gProbe.output.store(index + 3u, float4(gProbe.harmonics[index].xyz, 1.0));
    }
    uint levels; pdf.GetDimensions(0u, width, height, levels);
    float sum = 0.0, xMoment = 0.0, yMoment = 0.0;
    for (uint index = 0u; index < width * height; ++index) {
        float value = pdf.Load(int3(index % width, index / width, 0));
        sum += value;
        xMoment += value * float(index % width) / float(width);
        yMoment += value * float(index / width) / float(height);
    }
    gProbe.output.store(12u, float4(sum, xMoment / max(sum, 1e-30), yMoment / max(sum, 1e-30), 1.0));
    gProbe.output.store(13u, float4(pdf.Load(int3(0, 0, levels - 1u)), width, height, 1.0));
}
