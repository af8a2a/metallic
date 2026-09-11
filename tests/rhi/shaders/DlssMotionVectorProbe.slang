#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

#include "../../../Shaders/Features/PathTracing/ScenePathTrace.slang"

#define gMotionResults METALLIC_RESOURCE(RWStructuredBuffer<float4>, 63)

[shader("compute")]
[numthreads(32, 1, 1)]
void dlssMotionVectorProbeMain(uint3 dispatchId : SV_DispatchThreadID)
{
    uint index = dispatchId.x;
    ScenePathTracePush push = {};
    push.width = 640;
    push.height = 360;
    push.eye = float4(0, 0, 0, 1);
    push.center = float4(0, 0, -1, 1);
    push.upProjection = float4(0, 1, 0, index / 16);
    push.viewport = float4(640.0 / 360.0, 0, 0, kPi * 0.5);
    push.clipOrtho = float4(0.01, 100, 4, 0);
    push.previousEye = push.eye;
    push.previousCenter = push.center;
    push.previousUpProjection = push.upProjection;
    push.previousViewport = push.viewport;
    push.previousClipOrtho = push.clipOrtho;
    push.previousCameraValid = 1;
    push.temporalJitter = 1;
    push.jitterOffsetX = float(index % 4) * 0.25 - 0.375;
    push.jitterOffsetY = float((index / 4) % 4) * 0.25 - 0.375;
    float2 pixelCenter = float2(219.5, 173.5);
    float2 samplePixel = pixelCenter + float2(push.jitterOffsetX, push.jitterOffsetY);
    float3 direction = rayDirectionFromPixel(samplePixel, push);
    float3 hit = rayOriginFromPixel(samplePixel, push) + direction * (4.0 / -direction.z);
    gMotionResults[index * 4] = float4(
        motionVectorToPrevious(hit, pixelCenter, push),
        skyMotionVectorToPrevious(direction, pixelCenter, push));

    // Translate the camera without rotating it. Geometry moves in screen space;
    // an infinitely distant environment must not move with camera translation.
    push.eye.xy += float2(0.5, 0.25);
    push.center.xy += float2(0.5, 0.25);
    hit = rayOriginFromPixel(samplePixel, push) + direction * (4.0 / -direction.z);
    gMotionResults[index * 4 + 1] = float4(
        motionVectorToPrevious(hit, pixelCenter, push),
        skyMotionVectorToPrevious(direction, pixelCenter, push));

    push.previousCameraValid = 0;
    gMotionResults[index * 4 + 2] = float4(
        motionVectorToPrevious(hit, pixelCenter, push),
        skyMotionVectorToPrevious(direction, pixelCenter, push));

    // Rotation must move both the geometry and the sky, with matching motion
    // for points on the same perspective ray.
    push.previousCameraValid = index < 16 ? 1 : 0;
    push.eye = float4(0, 0, 0, 1);
    push.center = float4(sin(0.1), 0, -cos(0.1), 1);
    direction = rayDirectionFromPixel(samplePixel, push);
    hit = direction * 4.0;
    gMotionResults[index * 4 + 3] = float4(
        motionVectorToPrevious(hit, pixelCenter, push),
        skyMotionVectorToPrevious(direction, pixelCenter, push));
}
