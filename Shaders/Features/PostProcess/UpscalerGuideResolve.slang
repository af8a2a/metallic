#include "../../Libraries/Resources/ComputeResources.slang"

#define gMotion METALLIC_RESOURCE(Texture2D<float2>, 0)
#define gDepth METALLIC_RESOURCE(Texture2D<float>, 1)
#define gOutputMotion METALLIC_RESOURCE(RWTexture2D<float2>, 2)
#define gOutputDepth METALLIC_RESOURCE(RWTexture2D<float>, 3)

struct UpscalerGuideResolvePush { float2 jitter; };

[shader("compute")]
[numthreads(8, 8, 1)]
void upscalerGuideResolveMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height, sourceWidth, sourceHeight;
    gOutputDepth.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    gDepth.GetDimensions(sourceWidth, sourceHeight);
    UpscalerGuideResolvePush push = METALLIC_CONSTANTS(UpscalerGuideResolvePush);
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    int2 base = int2(floor(uv * float2(sourceWidth, sourceHeight) - push.jitter - 0.5));
    int2 selected = clamp(base, int2(0), int2(sourceWidth, sourceHeight) - 1);
    float nearest = 1.0;
    // Standard depth: choose the foreground sample across an edge. Motion stays
    // in UV units, so changing resolution must not multiply it by the SR ratio.
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            int2 p = clamp(base + int2(x, y), int2(0), int2(sourceWidth, sourceHeight) - 1);
            float depth = gDepth.Load(int3(p, 0));
            if (depth < nearest) { nearest = depth; selected = p; }
        }
    }
    gOutputDepth[id.xy] = nearest;
    gOutputMotion[id.xy] = gMotion.Load(int3(selected, 0));
}
