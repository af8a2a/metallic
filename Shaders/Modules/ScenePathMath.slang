#language slang 2026
module ScenePathMath;

import ScenePathTypes;
import Material;

namespace Metallic.ScenePath
{
using Metallic.Material;

internal static const float kPi = 3.14159265358979323846;

public float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

public float3 safeTriangleNormal(float3 edge1, float3 edge2, float3 fallback)
{
    float3 unnormalizedNormal = cross(edge1, edge2);
    float normalLengthSquared = dot(unnormalizedNormal, unnormalizedNormal);
    float edgeLengthProduct = dot(edge1, edge1) * dot(edge2, edge2);
    if (edgeLengthProduct <= 0.0 ||
        normalLengthSquared <= edgeLengthProduct * 1e-12) {
        return fallback;
    }
    return unnormalizedNormal * rsqrt(normalLengthSquared);
}

public uint hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

public float rand(inout uint state)
{
    state = hash(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

public float4 rand4(inout uint state)
{
    return float4(rand(state), rand(state), rand(state), rand(state));
}

public float safeLog2(float value)
{
    return log2(max(value, 0.0000001));
}

public RayCone makeRayCone(float width, float spreadAngle)
{
    RayCone rayCone;
    rayCone.width = width;
    rayCone.spreadAngle = spreadAngle;
    return rayCone;
}

public RayCone propagateRayCone(RayCone rayCone, float distance)
{
    rayCone.width = abs(rayCone.width) + abs(rayCone.spreadAngle) * max(distance, 0.0);
    return rayCone;
}

public float rayConeSpreadAngleExpansionByScatterPdf(float bsdfScatterPdf)
{
    float inversePdfSolidAngle = (1.0 / max(bsdfScatterPdf, 0.000001)) / (2.0 * kPi);
    return 0.3 * 2.0 * acos(max(-1.0, 1.0 - inversePdfSolidAngle));
}

public RayCone expandRayConeAfterScatter(RayCone rayCone, float bsdfScatterPdf)
{
    rayCone.spreadAngle = min(
        rayCone.spreadAngle + rayConeSpreadAngleExpansionByScatterPdf(bsdfScatterPdf),
        2.0 * kPi);
    return rayCone;
}

public float3 rayDirectionFromPixel(float2 pixel, ScenePathTracePush push)
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

    float fovRadians = clamp(push.viewport.w, 0.017453292, 3.12413936);
    float halfHeight = tan(fovRadians * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

public float3 rayOriginFromPixel(float2 pixel, ScenePathTracePush push)
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

public RayCone primaryRayConeFromPixel(float2 pixel, ScenePathTracePush push)
{
    // One internal-render pixel, matching the reference ray-cone diameter.
    // Analytic projection avoids acos precision steps at small pixel angles.
    float2 cone = primaryTextureRayCone(uint2(push.width, push.height), push.viewport.x,
        push.viewport.w, push.clipOrtho.z, push.upProjection.w > 0.5);
    return makeRayCone(cone.x, cone.y);
}

public float3 offsetRay(float3 position, float3 normal)
{
    return position + normal * 0.001;
}

public float3 cameraForwardFrom(float4 eye, float4 center)
{
    return safeNormalize(center.xyz - eye.xyz, float3(0.0, 0.0, -1.0));
}

public float3 cameraRightFrom(float3 forward, float3 cameraUp)
{
    return safeNormalize(cross(forward, safeNormalize(cameraUp, float3(0.0, 1.0, 0.0))), float3(1.0, 0.0, 0.0));
}

public float3 cameraUpFrom(float3 forward, float3 right)
{
    return safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
}

public bool projectWorldToUv(
    float3 worldPosition,
    float4 eye,
    float4 center,
    float4 upProjection,
    float4 viewport,
    float4 clipOrtho,
    out float2 uv)
{
    float3 forward = cameraForwardFrom(eye, center);
    float3 right = cameraRightFrom(forward, upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    float3 view = worldPosition - eye.xyz;
    float aspect = max(viewport.x, 0.001);
    float2 ndc;
    if (upProjection.w > 0.5) {
        float halfHeight = max(clipOrtho.z * 0.5, 0.0001);
        ndc = float2(
            dot(view, right) / max(halfHeight * aspect, 0.0001),
            dot(view, up) / halfHeight);
    } else {
        float viewZ = dot(view, forward);
        if (viewZ <= max(clipOrtho.x, 0.0001)) {
            uv = float2(0.0, 0.0);
            return false;
        }
        float fovRadians = clamp(viewport.w, 0.017453292, 3.12413936);
        float halfHeight = max(tan(fovRadians * 0.5), 0.0001);
        ndc = float2(
            dot(view, right) / max(viewZ * aspect * halfHeight, 0.0001),
            dot(view, up) / max(viewZ * halfHeight, 0.0001));
    }
    uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return true;
}

public float cameraLinearDepth(float3 worldPosition, ScenePathTracePush push)
{
    float3 forward = cameraForwardFrom(push.eye, push.center);
    return max(dot(worldPosition - push.eye.xyz, forward), 0.0);
}

// SDK guide textures use standard depth (near 0, far 1), independently of the
// renderer's reversed-Z attachments and projection.
public float cameraHardwareDepth(float linearDepth, ScenePathTracePush push)
{
    float zNear = max(push.clipOrtho.x, 0.0001);
    float zFar = max(push.clipOrtho.y, zNear + 0.0001);
    float viewDepth = clamp(linearDepth, zNear, zFar);
    if (push.upProjection.w > 0.5) {
        return saturate((viewDepth - zNear) / (zFar - zNear));
    }
    return saturate(
        zFar / (zFar - zNear) -
        (zNear * zFar) / ((zFar - zNear) * viewDepth));
}

public float2 motionVectorToPrevious(float3 worldPosition, float2 currentPixelCenter, ScenePathTracePush push)
{
    if (push.previousCameraValid == 0u) {
        return float2(0.0, 0.0);
    }

    float2 previousUv;
    if (!projectWorldToUv(
        worldPosition,
        push.previousEye,
        push.previousCenter,
        push.previousUpProjection,
        push.previousViewport,
        push.previousClipOrtho,
        previousUv)) {
        return float2(0.0, 0.0);
    }
    // The hit came from the jittered primary ray. Subtract that same sample
    // position so a stationary camera produces zero motion, as required by
    // Streamline's motionVectorsJittered = false contract.
    float2 samplePixel = currentPixelCenter + float2(push.jitterOffsetX, push.jitterOffsetY);
    float2 currentUv = samplePixel / float2(max(push.width, 1u), max(push.height, 1u));
    return previousUv - currentUv;
}

public float2 skyMotionVectorToPrevious(float3 direction, float2 currentPixelCenter, ScenePathTracePush push)
{
    // A sky direction is a point at infinity (w = 0). Project it relative to
    // the previous eye so camera translation cannot create sky parallax.
    if (push.upProjection.w > 0.5) {
        return float2(0.0, 0.0);
    }
    return motionVectorToPrevious(
        push.previousEye.xyz + direction * max(push.previousClipOrtho.y, 1.0),
        currentPixelCenter,
        push);
}

public void computeTriangleTangentSpace(
    float3 p0,
    float3 p1,
    float3 p2,
    float2 uv0,
    float2 uv1,
    float2 uv2,
    float3 normal,
    out float3 tangent,
    out float3 bitangent)
{
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 deltaUv1 = uv1 - uv0;
    float2 deltaUv2 = uv2 - uv0;
    float determinant = deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;

    if (abs(determinant) <= 0.0000001) {
        float3 referenceAxis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
        tangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
        bitangent = safeNormalize(cross(normal, tangent), float3(0.0, 1.0, 0.0));
        return;
    }

    float invDeterminant = 1.0 / determinant;
    tangent = (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * invDeterminant;
    bitangent = (edge2 * deltaUv1.x - edge1 * deltaUv2.x) * invDeterminant;
}

} // namespace Metallic.ScenePath
