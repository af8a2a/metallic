#ifndef GPU_DRIVEN_CULLING_COMMON_SLANG
#define GPU_DRIVEN_CULLING_COMMON_SLANG

#include "GPUDrivenSceneCommon.slang"

static const uint kCullInstanceFrustum = 1u << 0u;
static const uint kCullInstanceHzb = 1u << 1u;
static const uint kCullMeshletFrustum = 1u << 2u;
static const uint kCullMeshletNormalCone = 1u << 3u;
static const uint kInstanceVisible = 1u << 0u;
static const uint kInstanceDoubleSided = 1u << 1u;
static const uint kInstanceAlphaMasked = 1u << 2u;
static const uint kInstanceAlphaBlend = 1u << 3u;
static const uint kDrawBucketCount = 4u;

struct CameraData
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 transformPoint(SceneTransform transform, float3 localPosition)
{
    return
        transform.world0.xyz * localPosition.x +
        transform.world1.xyz * localPosition.y +
        transform.world2.xyz * localPosition.z +
        transform.world3.xyz;
}

float transformMaxScale(SceneTransform transform)
{
    return max(
        length(transform.world0.xyz),
        max(length(transform.world1.xyz), length(transform.world2.xyz)));
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

CameraData cullingCamera(GPUDrivenPreviewParams params)
{
    CameraData camera;
    camera.eye = params.eye;
    camera.center = params.center;
    camera.upProjection = params.upProjection;
    camera.viewport = params.viewport;
    camera.clipOrtho = params.clipOrtho;
    return camera;
}

CameraData renderCamera(GPUDrivenPreviewParams params)
{
    CameraData camera;
    camera.eye = params.renderEye;
    camera.center = params.renderCenter;
    camera.upProjection = params.renderUpProjection;
    camera.viewport = params.renderViewport;
    camera.clipOrtho = params.renderClipOrtho;
    return camera;
}

CameraData previousCullingCamera(GPUDrivenPreviewParams params)
{
    CameraData camera;
    camera.eye = params.previousEye;
    camera.center = params.previousCenter;
    camera.upProjection = params.previousUpProjection;
    camera.viewport = params.previousViewport;
    camera.clipOrtho = params.previousClipOrtho;
    return camera;
}

float3 worldToView(float3 worldPosition, CameraData camera)
{
    float3 forward = safeNormalize(camera.center.xyz - camera.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(camera.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - camera.eye.xyz;
    return float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));
}

float projectDepth(float viewZ, CameraData camera)
{
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    float depth;
    if (camera.upProjection.w > 0.5) {
        depth = (viewZ - zNear) / (zFar - zNear);
    } else {
        // Keep the hardware perspective depth distribution; linear view-space depth
        // loses most of D32 precision near the camera when zFar is large.
        float safeViewZ = abs(viewZ) > 0.000001 ? viewZ :
            (viewZ < 0.0 ? -0.000001 : 0.000001);
        depth = zFar / (zFar - zNear) -
            (zFar * zNear) / ((zFar - zNear) * safeViewZ);
    }
    return camera.clipOrtho.w > 0.5 ? 1.0 - depth : depth;
}

float4 projectPosition(float3 worldPosition, CameraData camera)
{
    float3 viewPosition = worldToView(worldPosition, camera);
    float ndcZ = projectDepth(viewPosition.z, camera);
    float aspect = max(camera.viewport.x, 0.001);

    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        return float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    }

    float fovRadians = clamp(camera.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    return float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
}

bool sphereInsideFrustum(float3 center, float radius, CameraData camera)
{
    float3 viewCenter = worldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    if (viewCenter.z + radius < zNear || viewCenter.z - radius > zFar) {
        return false;
    }

    float aspect = max(camera.viewport.x, 0.001);
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * aspect;
        return abs(viewCenter.x) <= halfWidth + radius &&
            abs(viewCenter.y) <= halfHeight + radius;
    }

    if (viewCenter.z + radius <= 0.0) {
        return false;
    }
    float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
    float tanX = tanY * aspect;
    float radiusX = radius * sqrt(1.0 + tanX * tanX);
    float radiusY = radius * sqrt(1.0 + tanY * tanY);
    return abs(viewCenter.x) <= viewCenter.z * tanX + radiusX &&
        abs(viewCenter.y) <= viewCenter.z * tanY + radiusY;
}

uint mipDimension(uint value, uint mipLevel)
{
    for (uint level = 0; level < mipLevel; ++level) {
        value = max(1u, (value + 1u) >> 1u);
    }
    return value;
}

uint hzbMipOffset(GPUDrivenPreviewParams params, uint mipLevel)
{
    uint offset = 0;
    uint width = params.width;
    uint height = params.height;
    for (uint level = 0; level < mipLevel; ++level) {
        offset += width * height;
        width = max(1u, (width + 1u) >> 1u);
        height = max(1u, (height + 1u) >> 1u);
    }
    return offset;
}

float loadHzb(
    GPUDrivenPreviewPush push,
    uint hzbBufferIndex,
    uint elementIndex)
{
    uint handleIndex = hzbBufferIndex == 0 ? push.hzbBuffer0 : push.hzbBuffer1;
    DescriptorHandle<RWStructuredBuffer<float>> hzbHandle =
        DescriptorHandle<RWStructuredBuffer<float>>(uint2(push.bufferShaderIndexBase + handleIndex, 0));
    return hzbHandle[elementIndex];
}

bool sphereOccluded(
    float3 center,
    float radius,
    CameraData camera,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push,
    uint hzbBufferIndex)
{
    float3 viewCenter = worldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    if (viewCenter.z - radius <= zNear) {
        return false;
    }

    float2 centerPixels;
    float2 radiusPixels;
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * max(camera.viewport.x, 0.001);
        centerPixels = float2(
            (viewCenter.x / halfWidth * 0.5 + 0.5) * float(params.width),
            (-viewCenter.y / halfHeight * 0.5 + 0.5) * float(params.height));
        radiusPixels = float2(
            radius / halfWidth * 0.5 * float(params.width),
            radius / halfHeight * 0.5 * float(params.height));
    } else {
        float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
        float tanX = tanY * max(camera.viewport.x, 0.001);
        centerPixels = float2(
            (viewCenter.x / (viewCenter.z * tanX) * 0.5 + 0.5) * float(params.width),
            (-viewCenter.y / (viewCenter.z * tanY) * 0.5 + 0.5) * float(params.height));
        float closestZ = max(viewCenter.z - radius, zNear);
        radiusPixels = float2(
            radius / (closestZ * tanX) * 0.5 * float(params.width),
            radius / (closestZ * tanY) * 0.5 * float(params.height));
    }

    float2 minPixels = clamp(centerPixels - radiusPixels, 0.0, float2(params.width - 1u, params.height - 1u));
    float2 maxPixels = clamp(centerPixels + radiusPixels, 0.0, float2(params.width - 1u, params.height - 1u));
    float maxExtent = max(maxPixels.x - minPixels.x, maxPixels.y - minPixels.y);
    // At this mip the projected rectangle spans at most one texel per axis, so
    // the four corner loads conservatively cover every contributing depth texel.
    uint mipLevel = min(
        params.hzbMipCount - 1u,
        maxExtent > 1.0 ? uint(ceil(log2(maxExtent))) : 0u);
    uint mipWidth = mipDimension(params.width, mipLevel);
    uint mipHeight = mipDimension(params.height, mipLevel);
    float mipScale = exp2(-float(mipLevel));
    uint2 minTexel = min(uint2(minPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint2 maxTexel = min(uint2(maxPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint mipOffset = hzbMipOffset(params, mipLevel);
    float depth00 = loadHzb(push, hzbBufferIndex, mipOffset + minTexel.y * mipWidth + minTexel.x);
    float depth10 = loadHzb(push, hzbBufferIndex, mipOffset + minTexel.y * mipWidth + maxTexel.x);
    float depth01 = loadHzb(push, hzbBufferIndex, mipOffset + maxTexel.y * mipWidth + minTexel.x);
    float depth11 = loadHzb(push, hzbBufferIndex, mipOffset + maxTexel.y * mipWidth + maxTexel.x);
    float nearestZ = max(viewCenter.z - radius, zNear);
    // Pull the test sphere slightly toward the camera and require a strict depth
    // separation to avoid toggling instances on quantized HZB boundaries.
    float depthBias = max(0.001, radius * 0.01);
    float nearestDepth = projectDepth(max(nearestZ - depthBias, zNear), camera);
    if (camera.clipOrtho.w > 0.5) {
        float occluderDepth = min(min(depth00, depth10), min(depth01, depth11));
        return nearestDepth < occluderDepth;
    }
    float occluderDepth = max(max(depth00, depth10), max(depth01, depth11));
    return nearestDepth > occluderDepth;
}

bool meshletVisible(
    GPUDrivenPreviewMeshlet meshlet,
    uint meshletFlags,
    SceneTransform transform,
    GPUDrivenPreviewParams params)
{
    CameraData camera = cullingCamera(params);
    float radius = max(meshlet.boundingSphere.w * transformMaxScale(transform), 0.0);
    float3 center = transformPoint(transform, meshlet.boundingSphere.xyz);
    if ((params.cullingFlags & kCullMeshletFrustum) != 0 &&
        radius > 0.0 &&
        !sphereInsideFrustum(center, radius, camera)) {
        return false;
    }

    if ((params.cullingFlags & kCullMeshletNormalCone) != 0 &&
        (meshletFlags & kInstanceDoubleSided) == 0 &&
        meshlet.coneApexCutoff.w < 0.9999) {
        float3 coneApex = transformPoint(transform, meshlet.coneApexCutoff.xyz);
        float3 coneAxis = transformNormal(transform, meshlet.coneAxisLodError.xyz);
        float3 viewDirection = safeNormalize(coneApex - camera.eye.xyz, -coneAxis);
        if (dot(viewDirection, coneAxis) >= meshlet.coneApexCutoff.w) {
            return false;
        }
    }
    return true;
}

#endif
