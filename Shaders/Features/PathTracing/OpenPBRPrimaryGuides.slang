#language slang 2026
module OpenPBRPrimaryGuides;

import Core;
import ScenePathTypes;
import ScenePathMath;
import OpenPBRPathTracing;
import OpenPBRPrimaryHit;

using Metallic;
using Metallic.ScenePath;
using Metallic.OpenPBRPathTracing;
using Metallic.OpenPBRPrimaryHit;

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrPrimaryGuidesMain(uint3 id : SV_DispatchThreadID)
{
    ScenePathTracePush push = withTextureFeedbackSampling(getConstants<ScenePathTracePush>(), id.xy);
    uint2 pixel = id.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 jitter = float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
    float2 samplePixel = float2(pixel) + jitter;
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
    HitInfo hit;
    bool found = queryOpenPbrPrimaryStage(pixel, ray, rayCone, push, hit);
    storeOpenPbrPrimaryHit(pixel, push, found, hit);
}
