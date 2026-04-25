// Meshlet debug visualization — color-codes pixels by meshlet, instance, or triangle ID.
#include "../../Source/Rendering/visibility_constants.h"

struct MeshletVisUniforms {
    uint screenWidth;
    uint screenHeight;
    uint colorMode; // 0=Meshlet, 1=Instance, 2=Triangle
    uint pad;
};

static const uint INVALID_VIS = 0xFFFFFFFF;

// Integer hash (Wang) for good color distribution
float3 hashColor(uint id) {
    uint h = id;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    float r = float((h >>  0u) & 0xFFu) / 255.0;
    float g = float((h >>  8u) & 0xFFu) / 255.0;
    float b = float((h >> 16u) & 0xFFu) / 255.0;
    float mx = max(r, max(g, b));
    if (mx > 0.0) { r /= mx; g /= mx; b /= mx; }
    return float3(r, g, b);
}

[[vk::push_constant]] ConstantBuffer<MeshletVisUniforms> uniforms;
Texture2D<uint> visibilityBuffer;
RWTexture2D<float4> outputTexture;

[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= uniforms.screenWidth || tid.y >= uniforms.screenHeight) return;

    uint2 pixel = uint2(tid.x, tid.y);
    uint vis = visibilityBuffer[pixel];

    if (vis == INVALID_VIS) {
        outputTexture[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    uint id;
    if (uniforms.colorMode == 1u) {
        id = vis & VISIBILITY_INSTANCE_MASK;
    } else if (uniforms.colorMode == 2u) {
        id = (vis >> VISIBILITY_INSTANCE_BITS) & VISIBILITY_TRIANGLE_MASK;
    } else {
        id = (vis >> VISIBILITY_MESHLET_SHIFT) & VISIBILITY_MESHLET_MASK;
    }

    outputTexture[pixel] = float4(hashColor(id), 1.0);
}
