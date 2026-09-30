import Core;
using Metallic;
import Material;
using Metallic.Material;

struct Push { uint slot; uint visible; float nearSourceLod; float farSourceLod; };

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Push push = getConstants<Push>();
    Texture2D<float4> texture = getResourceArray<Texture2D<float4>>(0, push.slot);
    RWStructuredBuffer<uint> feedback = getResource<RWStructuredBuffer<uint>>(1);
    RWStructuredBuffer<float4> output = getResource<RWStructuredBuffer<float4>>(2);
    SamplerState sampler = getResource<SamplerState>(3);
    uint count, stride;
    feedback.GetDimensions(count, stride);
    uint offset = push.slot * 8u;
    if (offset + 7u >= count) {
        output[0] = float4(-1.0);
        output[1] = float4(-1.0);
        return;
    }
    // Source footprint remains fixed while physical tail index and clamp change.
    // Sample the production helper; do not duplicate its filtering or LOD clamp.
    uint firstMip = feedback[offset + 3u];
    float minimumLod = asfloat(feedback[offset + 6u]);
    float2 uv = float2(0.375, 0.625);
    float nearValue = sampleResidentMaterialTexture(texture, sampler, uv,
        push.nearSourceLod - float(firstMip), minimumLod).r;
    float farValue = sampleResidentMaterialTexture(texture, sampler, uv,
        push.farSourceLod - float(firstMip), minimumLod).r;
    if (push.visible != 0u && feedback[offset] != 0u) {
        uint ignored;
        InterlockedMin(feedback[offset + 4u], uint(max(floor(push.nearSourceLod), 0.0)), ignored);
        InterlockedAdd(feedback[offset + 5u], 1000u, ignored);
    }
    uint width, height;
    texture.GetDimensions(width, height);
    output[0] = float4(nearValue, farValue, minimumLod, float(firstMip));
    output[1] = float4(float(feedback[offset + 4u]), float(feedback[offset + 5u]),
        float(feedback[offset]), float(width));
}
