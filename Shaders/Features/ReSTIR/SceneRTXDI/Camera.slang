#language slang 2026
implementing SceneRTXDI;


float3 rayDirectionFromPixel(float2 pixel, SceneRTXDIPush push)
{
    float aspect = max(push.viewport.x, 0.001);
    float2 uv = pixel / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    if (push.upProjection.w > 0.5) {
        return forward;
    }
    float halfHeight = tan(clamp(push.viewport.w, 0.017453292, 3.12413936) * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

float3 rayOriginFromPixel(float2 pixel, SceneRTXDIPush push)
{
    if (push.upProjection.w <= 0.5) {
        return push.eye.xyz;
    }
    float aspect = max(push.viewport.x, 0.001);
    float2 uv = pixel / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float halfHeight = max(push.clipOrtho.z * 0.5, 0.0001);
    return push.eye.xyz + right * (ndc.x * halfHeight * aspect) + up * (ndc.y * halfHeight);
}

bool projectWorldToPreviousUv(float3 worldPosition, SceneRTXDIPush push, out float2 uv)
{
    float3 forward = safeNormalize(
        push.previousCenter.xyz - push.previousEye.xyz,
        float3(0.0, 0.0, -1.0));
    float3 right = safeNormalize(
        cross(forward, safeNormalize(push.previousUpProjection.xyz, float3(0.0, 1.0, 0.0))),
        float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 view = worldPosition - push.previousEye.xyz;
    float aspect = max(push.previousViewport.x, 0.001);
    float2 ndc;
    if (push.previousUpProjection.w > 0.5) {
        float halfHeight = max(push.previousClipOrtho.z * 0.5, 0.0001);
        ndc = float2(
            dot(view, right) / max(halfHeight * aspect, 0.0001),
            dot(view, up) / halfHeight);
    } else {
        float viewZ = dot(view, forward);
        if (viewZ <= max(push.previousClipOrtho.x, 0.0001)) {
            uv = 0.0;
            return false;
        }
        float halfHeight = max(tan(push.previousViewport.w * 0.5), 0.0001);
        ndc = float2(
            dot(view, right) / max(viewZ * aspect * halfHeight, 0.0001),
            dot(view, up) / max(viewZ * halfHeight, 0.0001));
    }
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return all(uv >= 0.0) && all(uv < 1.0);
}

float cameraLinearDepth(float3 worldPosition, SceneRTXDIPush push)
{
    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    return max(dot(worldPosition - push.eye.xyz, forward), 0.0);
}

float2 motionVectorToPrevious(float3 worldPosition, float2 currentSamplePixel, SceneRTXDIPush push)
{
    if (push.hasHistory == 0u) {
        return float2(0.0, 0.0);
    }
    float2 previousUv;
    if (!projectWorldToPreviousUv(worldPosition, push, previousUv)) {
        return float2(0.0, 0.0);
    }
    float2 currentUv = currentSamplePixel / float2(max(push.width, 1u), max(push.height, 1u));
    return previousUv - currentUv;
}
