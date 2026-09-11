#ifndef GPU_DRIVEN_CULLING_COMMON_SLANG
#define GPU_DRIVEN_CULLING_COMMON_SLANG

#include "GPUDrivenSceneCommon.slang"
#include "ConservativeOcclusion.slang"

static const uint kCullInstanceFrustum = 1u << 0u;
static const uint kCullInstanceHzb = 1u << 1u;
static const uint kCullMeshletFrustum = 1u << 2u;
static const uint kCullMeshletNormalCone = 1u << 3u;
static const uint kCullMeshletHzb = 1u << 4u;
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
    return conservativeTransformScale(transform.world0.xyz, transform.world1.xyz, transform.world2.xyz);
}

float3 transformNormal(SceneTransform transform, float3 localNormal)
{
    float3 cofactor0 = cross(transform.world1.xyz, transform.world2.xyz);
    float3 cofactor1 = cross(transform.world2.xyz, transform.world0.xyz);
    float3 cofactor2 = cross(transform.world0.xyz, transform.world1.xyz);
    float3 normal = cofactor0 * localNormal.x + cofactor1 * localNormal.y + cofactor2 * localNormal.z;
    // Cofactors scale quadratically with the object. Sponza's 0.008 scale
    // produces a squared length of 4.096e-9: the camera-vector epsilon would
    // replace its valid cone axis with +Z. Rescale before normalizing so the
    // decision depends on direction, including the cofactor's winding sign.
    float maxComponent = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    return maxComponent > 0.0 && isfinite(maxComponent)
        ? normalize(normal / maxComponent) : float3(0.0);
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
    return conservativeProjectDepth(viewZ, zNear, zFar, camera.upProjection.w > 0.5, camera.clipOrtho.w > 0.5);
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
    float2 guard = 1.0 + 2.0 / max(camera.viewport.yz, 1.0);
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * aspect;
        return abs(viewCenter.x) <= halfWidth * guard.x + radius &&
            abs(viewCenter.y) <= halfHeight * guard.y + radius;
    }

    if (viewCenter.z + radius <= 0.0) {
        return false;
    }
    float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
    float tanX = tanY * aspect * guard.x;
    tanY *= guard.y;
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
    float4 rect;
    if (params.hzbMipCount == 0u || !conservativeSphereScreenRect(viewCenter, radius,
        float2(params.width, params.height), max(camera.viewport.x, 0.001),
        clamp(camera.viewport.w, 0.017453292, 3.12413936),
        camera.upProjection.w > 0.5 ? max(camera.clipOrtho.z, 0.0002) : 0.0, zNear, rect)) {
        return false;
    }
    float2 minPixels = rect.xy, maxPixels = rect.zw;
    float maxExtent = max(maxPixels.x - minPixels.x, maxPixels.y - minPixels.y);
    // At most two texels per axis. Verify the integer span as well, so rounding
    // at power-of-two boundaries cannot skip a middle texel in a four-tap test.
    uint mipLevel = min(
        params.hzbMipCount - 1u,
        maxExtent > 1.0 ? uint(ceil(log2(maxExtent))) : 0u);
    uint mipWidth = mipDimension(params.width, mipLevel);
    uint mipHeight = mipDimension(params.height, mipLevel);
    float mipScale = exp2(-float(mipLevel));
    uint2 minTexel = min(uint2(minPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint2 maxTexel = min(uint2(maxPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    if (any(maxTexel - minTexel > 1u)) { return false; }
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
    return conservativeDepthOccluded(nearestDepth, float4(depth00, depth10, depth01, depth11), camera.clipOrtho.w > 0.5);
}

bool meshletVisibleInOcclusionPhase(float3 center, float radius, uint instanceState,
    GPUDrivenPreviewParams params, GPUDrivenPreviewPush push)
{
    bool early = push.passIndex == 0u;
    if ((params.cullingFlags & kCullMeshletHzb) == 0u) {
        return instanceState == (early ? 1u : 3u);
    }
    if (early ? instanceState != 1u : (instanceState != 1u && instanceState != 3u)) { return false; }
    uint currentHzb = params.frameIndex & 1u;
    // Re-evaluate the immutable early predicate in the second task-shader pass.
    // This reconstructs the deferred set without a bounded append queue (and
    // without dropping geometry if a queue would overflow).
    bool deferred = instanceState == 3u || (params.hzbValid != 0u &&
        sphereOccluded(center, radius, previousCullingCamera(params), params, push, currentHzb ^ 1u));
    if (early) { return !deferred; }
    return deferred && !sphereOccluded(center, radius, cullingCamera(params), params, push, currentHzb);
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
        // A degenerate transform/axis cannot prove that the meshlet is back-facing.
        if (dot(coneAxis, coneAxis) > 0.0 && dot(viewDirection, coneAxis) >= meshlet.coneApexCutoff.w) {
            return false;
        }
    }
    return true;
}

#endif
