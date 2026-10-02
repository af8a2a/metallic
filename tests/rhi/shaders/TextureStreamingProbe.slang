#include "../../../Source/Runtime/Render/Core/TextureProbeParameters.h"
using Metallic;
import Material;
using Metallic.Material;

[[vk::push_constant]] ConstantBuffer<TextureStreamingProbeParameters> push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    if (!push.output.contains(1u)) { return; }
    Texture2D<float4> texture = resolveDescriptor(push.texture);
    uint* feedback = push.feedback.data;
    float4* output = push.output.data;
    SamplerState sampler = resolveDescriptor(push.sampler);
    if (push.slot >= push.feedback.length() / 8u) {
        output[0] = float4(-1.0);
        output[1] = float4(-1.0);
        return;
    }
    uint offset = push.slot * 8u;
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
