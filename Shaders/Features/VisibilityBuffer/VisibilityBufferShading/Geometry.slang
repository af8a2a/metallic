implementing VisibilityBufferShading;

import GPUDriven;
using Metallic.GPUDriven;
import Core;
import MaterialTextureDecode;
using Metallic;
import Material;
using Metallic.Material;

import OpenPBR;
import OpenPBRClosure;
import OpenPBRTextureLUT;
import MaterialProgram;
using Metallic.OpenPBR;

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 transformPoint(SceneTransform transform, float3 localPosition)
{
    return transform.world0.xyz * localPosition.x +
        transform.world1.xyz * localPosition.y +
        transform.world2.xyz * localPosition.z +
        transform.world3.xyz;
}

float3 transformDirection(SceneTransform transform, float3 localDirection)
{
    return transform.world0.xyz * localDirection.x +
        transform.world1.xyz * localDirection.y +
        transform.world2.xyz * localDirection.z;
}

float3 transformNormal(SceneTransform transform, float3 localNormal)
{
    float3 cofactor0 = cross(transform.world1.xyz, transform.world2.xyz);
    float3 cofactor1 = cross(transform.world2.xyz, transform.world0.xyz);
    float3 cofactor2 = cross(transform.world0.xyz, transform.world1.xyz);
    return safeNormalize(
        cofactor0 * localNormal.x + cofactor1 * localNormal.y + cofactor2 * localNormal.z,
        float3(0.0, 0.0, 1.0));
}

float3 worldToView(float3 worldPosition, GPUDrivenPreviewParams params)
{
    float3 forward = safeNormalize(params.renderCenter.xyz - params.renderEye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(params.renderUpProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - params.renderEye.xyz;
    return float3(dot(cameraVector, right), dot(cameraVector, up), dot(cameraVector, forward));
}

float projectDepth(float viewZ, GPUDrivenPreviewParams params)
{
    float zNear = max(params.renderClipOrtho.x, 0.0001);
    float zFar = max(params.renderClipOrtho.y, zNear + 0.0001);
    return conservativeProjectDepth(viewZ, zNear, zFar, params.renderUpProjection.w > 0.5);
}

float4 projectPosition(float3 worldPosition, GPUDrivenPreviewParams params)
{
    float3 viewPosition = worldToView(worldPosition, params);
    float ndcZ = projectDepth(viewPosition.z, params);
    float aspect = max(params.renderViewport.x, 0.001);
    if (params.renderUpProjection.w > 0.5) {
        float halfHeight = max(params.renderClipOrtho.z * 0.5, 0.0001);
        return float4(
            viewPosition.x / (halfHeight * aspect),
            -viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    }
    float fovRadians = clamp(params.renderViewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    return float4(
        viewPosition.x * yScale / aspect,
        -viewPosition.y * yScale,
        ndcZ * w,
        w);
}

float2 screenPosition(float4 clipPosition, GPUDrivenPreviewParams params)
{
    float reciprocalW = 1.0 / max(abs(clipPosition.w), 0.000001);
    float2 ndc = clipPosition.xy * reciprocalW;
    return (ndc * 0.5 + 0.5) * float2(params.width, params.height);
}

float cross2(float2 left, float2 right)
{
    return left.x * right.y - left.y * right.x;
}

float3 perspectiveBarycentrics(
    float2 pixel,
    float4 clip0,
    float4 clip1,
    float4 clip2,
    GPUDrivenPreviewParams params)
{
    float2 p0 = screenPosition(clip0, params);
    float2 p1 = screenPosition(clip1, params);
    float2 p2 = screenPosition(clip2, params);
    float denominator = cross2(p1 - p0, p2 - p0);
    if (abs(denominator) <= 0.000001) {
        return float3(1.0, 0.0, 0.0);
    }
    float3 screenWeights = float3(
        cross2(p1 - pixel, p2 - pixel),
        cross2(p2 - pixel, p0 - pixel),
        cross2(p0 - pixel, p1 - pixel)) / denominator;
    float3 weights = screenWeights / float3(clip0.w, clip1.w, clip2.w);
    float weightSum = weights.x + weights.y + weights.z;
    return abs(weightSum) > 0.000001 ? weights / weightSum : float3(1.0, 0.0, 0.0);
}

float3 cameraRayDirection(float2 pixel, GPUDrivenPreviewParams params)
{
    float2 uv = pixel / float2(max(params.width, 1u), max(params.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 forward = safeNormalize(params.renderCenter.xyz - params.renderEye.xyz, float3(0.0, 0.0, -1.0));
    if (params.renderUpProjection.w > 0.5) {
        return forward;
    }
    float3 cameraUp = safeNormalize(params.renderUpProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float halfHeight = tan(clamp(params.renderViewport.w, 0.017453292, 3.12413936) * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * max(params.renderViewport.x, 0.001) * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}
