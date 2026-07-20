static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kMeshletTriangleChunkSize = 64;
static const uint kMeshletChunkCount = 2;
static const uint kMaxMeshOutputVertices = kMeshletTriangleChunkSize * 3;

static const uint kCullInstanceFrustum = 1u << 0u;
static const uint kCullInstanceHzb = 1u << 1u;
static const uint kCullMeshletFrustum = 1u << 2u;
static const uint kCullMeshletNormalCone = 1u << 3u;
static const uint kMeshletDoubleSided = 1u << 0u;

struct GPUDrivenPreviewPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint positionBuffer;
    uint meshletBuffer;
    uint meshletVertexBuffer;
    uint meshletTriangleBuffer;
    uint paramsBuffer;
    uint transformBuffer;
    uint instanceBuffer;
    uint instanceVisibilityBuffer;
    uint visibleMeshletBuffer0;
    uint visibleMeshletBuffer1;
    uint indirectBuffer0;
    uint indirectBuffer1;
    uint hzbBuffer0;
    uint hzbBuffer1;
    uint deferredColorBuffer;
    uint depthImage;
    uint visibilityImage;
    uint passIndex;
    uint mipLevel;
    uint projectWithCullingCamera;
    uint materialBuffer;
    uint materialTextureImageBase;
    uint environmentImage;
    uint padding;
};

struct GPUDrivenPreviewVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 texcoord;
};

struct SceneTransform
{
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenPreviewMeshlet
{
    uint vertexOffset;
    uint vertexCount;
    uint triangleOffset;
    uint triangleCount;
    uint primitiveIndex;
    uint materialIndex;
    uint lodLevel;
    uint lodGroupIndex;
    uint transformIndex;
    uint instanceIndex;
    uint flags;
    uint padding0;
    float4 boundingSphere;
    float4 coneApexCutoff;
    float4 coneAxis;
};

struct GPUDrivenPreviewInstance
{
    float4 boundingSphere;
    uint transformIndex;
    uint primitiveIndex;
    uint padding0;
    uint padding1;
};

struct GPUDrivenPreviewParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
    float4 renderEye;
    float4 renderCenter;
    float4 renderUpProjection;
    float4 renderViewport;
    float4 renderClipOrtho;
    uint mode;
    uint meshletOffset;
    uint meshletCount;
    uint selectedLodLevel;
    uint instanceCount;
    uint width;
    uint height;
    uint hzbMipCount;
    uint frameIndex;
    uint hzbValid;
    uint cullingFlags;
    uint materialTextureCount;
    float environmentIntensity;
    float environmentRotationRadians;
    uint environmentMode;
    uint environmentVisible;
    uint materialCount;
    uint shadingPadding0;
    uint shadingPadding1;
    uint shadingPadding2;
    float4 environmentSH[9];
};

struct CameraData
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
};

struct GPUDrivenPreviewVertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint visibilityId : TEXCOORD0;
};

struct FullscreenVertexOutput
{
    float4 position : SV_Position;
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
        (meshlet.flags & kMeshletDoubleSided) == 0 &&
        meshlet.coneApexCutoff.w < 0.9999) {
        float3 coneApex = transformPoint(transform, meshlet.coneApexCutoff.xyz);
        float3 coneAxis = transformNormal(transform, meshlet.coneAxis.xyz);
        float3 viewDirection = safeNormalize(coneApex - camera.eye.xyz, -coneAxis);
        if (dot(viewDirection, coneAxis) >= meshlet.coneApexCutoff.w) {
            return false;
        }
    }
    return true;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void gpuDrivenPreviewResetMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    uint handleIndex = push.passIndex == 0 ? push.indirectBuffer0 : push.indirectBuffer1;
    DescriptorHandle<RWStructuredBuffer<uint>> indirectHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + handleIndex, 0));
    indirectHandle[0] = 0u;
    indirectHandle[1] = 1u;
    indirectHandle[2] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenPreviewInstanceCullMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<SceneTransform>> transformHandle =
        DescriptorHandle<StructuredBuffer<SceneTransform>>(uint2(push.bufferShaderIndexBase + push.transformBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> visibilityHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.instanceVisibilityBuffer, 0));

    GPUDrivenPreviewParams params = paramsHandle[0];
    uint instanceIndex = dispatchThreadId.x;
    if (instanceIndex >= params.instanceCount) {
        return;
    }

    RWStructuredBuffer<uint> visibility = visibilityHandle;
    if (push.passIndex != 0 && visibility[instanceIndex] != 2u) {
        return;
    }

    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    StructuredBuffer<SceneTransform> transforms = transformHandle;
    GPUDrivenPreviewInstance instance = instances[instanceIndex];
    SceneTransform transform = transforms[instance.transformIndex];
    float3 center = transformPoint(transform, instance.boundingSphere.xyz);
    float radius = max(instance.boundingSphere.w * transformMaxScale(transform), 0.000001);
    CameraData camera = cullingCamera(params);

    if ((params.cullingFlags & kCullInstanceFrustum) != 0 &&
        !sphereInsideFrustum(center, radius, camera)) {
        visibility[instanceIndex] = 0u;
        return;
    }

    if ((params.cullingFlags & kCullInstanceHzb) == 0 || params.hzbValid == 0) {
        visibility[instanceIndex] = push.passIndex == 0 ? 1u : 3u;
        return;
    }

    uint currentHzb = params.frameIndex & 1u;
    uint hzbIndex = push.passIndex == 0 ? (currentHzb ^ 1u) : currentHzb;
    CameraData hzbCamera = push.passIndex == 0 ? previousCullingCamera(params) : camera;
    bool occluded = sphereOccluded(center, radius, hzbCamera, params, push, hzbIndex);
    visibility[instanceIndex] = occluded ? (push.passIndex == 0 ? 2u : 0u) :
        (push.passIndex == 0 ? 1u : 3u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenPreviewCompactMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> visibilityHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.instanceVisibilityBuffer, 0));

    GPUDrivenPreviewParams params = paramsHandle[0];
    uint localMeshletIndex = dispatchThreadId.x;
    if (localMeshletIndex >= params.meshletCount) {
        return;
    }

    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<uint> visibility = visibilityHandle;
    uint meshletIndex = params.meshletOffset + localMeshletIndex;
    uint expectedVisibility = push.passIndex == 0 ? 1u : 3u;
    if (visibility[meshlets[meshletIndex].instanceIndex] != expectedVisibility) {
        return;
    }

    uint visibleHandleIndex = push.passIndex == 0 ? push.visibleMeshletBuffer0 : push.visibleMeshletBuffer1;
    uint indirectHandleIndex = push.passIndex == 0 ? push.indirectBuffer0 : push.indirectBuffer1;
    DescriptorHandle<RWStructuredBuffer<uint>> visibleHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + visibleHandleIndex, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> indirectHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + indirectHandleIndex, 0));

    uint chunkCount = min(
        kMeshletChunkCount,
        (meshlets[meshletIndex].triangleCount + kMeshletTriangleChunkSize - 1u) /
            kMeshletTriangleChunkSize);
    uint outputIndex = 0;
    InterlockedAdd(indirectHandle[0], chunkCount, outputIndex);
    for (uint chunkIndex = 0u; chunkIndex < kMeshletChunkCount; ++chunkIndex) {
        if (chunkIndex < chunkCount) {
            visibleHandle[outputIndex + chunkIndex] =
                (meshletIndex << 1u) | chunkIndex;
        }
    }
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenPreviewMeshMain(
    uint3 groupID : SV_GroupID,
    out vertices GPUDrivenPreviewVertexOutput vertices[kMaxMeshOutputVertices],
    out indices uint3 triangles[kMeshletTriangleChunkSize],
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> positionHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletVertexHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletVertexBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletTriangleHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletTriangleBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<SceneTransform>> transformHandle =
        DescriptorHandle<StructuredBuffer<SceneTransform>>(uint2(push.bufferShaderIndexBase + push.transformBuffer, 0));
    uint visibleHandleIndex = push.passIndex == 0 ? push.visibleMeshletBuffer0 : push.visibleMeshletBuffer1;
    DescriptorHandle<StructuredBuffer<uint>> visibleHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + visibleHandleIndex, 0));

    StructuredBuffer<GPUDrivenPreviewVertex> verticesBuffer = positionHandle;
    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<uint> meshletVertices = meshletVertexHandle;
    StructuredBuffer<uint> meshletTriangles = meshletTriangleHandle;
    StructuredBuffer<GPUDrivenPreviewParams> paramsBuffer = paramsHandle;
    StructuredBuffer<SceneTransform> transforms = transformHandle;
    StructuredBuffer<uint> visibleMeshlets = visibleHandle;

    uint meshletChunk = visibleMeshlets[groupID.x];
    uint meshletIndex = meshletChunk >> 1u;
    uint chunkIndex = meshletChunk & 1u;
    GPUDrivenPreviewMeshlet meshlet = meshlets[meshletIndex];
    SceneTransform transform = transforms[meshlet.transformIndex];
    GPUDrivenPreviewParams params = paramsBuffer[0];
    if (!meshletVisible(meshlet, transform, params)) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint triangleStart = chunkIndex * kMeshletTriangleChunkSize;
    if (triangleStart >= meshlet.triangleCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint triangleCount = min(
        meshlet.triangleCount - triangleStart,
        kMeshletTriangleChunkSize);
    uint vertexCount = triangleCount * 3u;
    SetMeshOutputCounts(vertexCount, triangleCount);

    CameraData projectionCamera = push.projectWithCullingCamera != 0u ?
        cullingCamera(params) : renderCamera(params);
    for (uint vertexIndex = 0; vertexIndex < kMaxMeshOutputVertices; ++vertexIndex) {
        if (vertexIndex < vertexCount) {
            uint chunkTriangleIndex = vertexIndex / 3u;
            uint triangleCorner = vertexIndex % 3u;
            uint meshletTriangleIndex = triangleStart + chunkTriangleIndex;
            uint triangleBase = meshlet.triangleOffset + meshletTriangleIndex * 3u;
            uint localVertexIndex = meshletTriangles[triangleBase + triangleCorner];
            uint positionIndex = meshletVertices[meshlet.vertexOffset + localVertexIndex];
            float3 worldPosition = transformPoint(transform, verticesBuffer[positionIndex].position.xyz);
            float4 position = projectPosition(worldPosition, projectionCamera);
            position.y = -position.y;
            vertices[vertexIndex].position = position;
            vertices[vertexIndex].visibilityId =
                ((meshletIndex + 1u) << 7u) | meshletTriangleIndex;
        }
    }

    for (uint triangleIndex = 0; triangleIndex < kMeshletTriangleChunkSize; ++triangleIndex) {
        if (triangleIndex < triangleCount) {
            triangles[triangleIndex] = uint3(
                triangleIndex * 3u,
                triangleIndex * 3u + 1u,
                triangleIndex * 3u + 2u);
        }
    }
}

[shader("fragment")]
uint gpuDrivenPreviewFragmentMain(
    GPUDrivenPreviewVertexOutput input,
    uniform GPUDrivenPreviewPush push) : SV_Target0
{
    return input.visibilityId;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenPreviewHzbMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    uint mipWidth = mipDimension(params.width, push.mipLevel);
    uint mipHeight = mipDimension(params.height, push.mipLevel);
    if (dispatchThreadId.x >= mipWidth || dispatchThreadId.y >= mipHeight) {
        return;
    }

    uint currentHzb = params.frameIndex & 1u;
    uint hzbHandleIndex = currentHzb == 0 ? push.hzbBuffer0 : push.hzbBuffer1;
    DescriptorHandle<RWStructuredBuffer<float>> hzbHandle =
        DescriptorHandle<RWStructuredBuffer<float>>(uint2(push.bufferShaderIndexBase + hzbHandleIndex, 0));
    RWStructuredBuffer<float> hzb = hzbHandle;
    uint destinationIndex = hzbMipOffset(params, push.mipLevel) +
        dispatchThreadId.y * mipWidth + dispatchThreadId.x;

    if (push.mipLevel == 0) {
        DescriptorHandle<Texture2D<float>> depthHandle =
            DescriptorHandle<Texture2D<float>>(uint2(push.imageShaderIndexBase + push.depthImage, 0));
        Texture2D<float> depth = depthHandle;
        hzb[destinationIndex] = depth.Load(int3(dispatchThreadId.xy, 0));
        return;
    }

    uint sourceMip = push.mipLevel - 1u;
    uint sourceWidth = mipDimension(params.width, sourceMip);
    uint sourceHeight = mipDimension(params.height, sourceMip);
    uint sourceOffset = hzbMipOffset(params, sourceMip);
    uint2 sourceBase = dispatchThreadId.xy * 2u;
    uint2 source0 = min(sourceBase, uint2(sourceWidth - 1u, sourceHeight - 1u));
    uint2 source1 = min(sourceBase + uint2(1u, 0u), uint2(sourceWidth - 1u, sourceHeight - 1u));
    uint2 source2 = min(sourceBase + uint2(0u, 1u), uint2(sourceWidth - 1u, sourceHeight - 1u));
    uint2 source3 = min(sourceBase + uint2(1u, 1u), uint2(sourceWidth - 1u, sourceHeight - 1u));
    float depth0 = hzb[sourceOffset + source0.y * sourceWidth + source0.x];
    float depth1 = hzb[sourceOffset + source1.y * sourceWidth + source1.x];
    float depth2 = hzb[sourceOffset + source2.y * sourceWidth + source2.x];
    float depth3 = hzb[sourceOffset + source3.y * sourceWidth + source3.x];
    hzb[destinationIndex] = params.clipOrtho.w > 0.5 ?
        min(min(depth0, depth1), min(depth2, depth3)) :
        max(max(depth0, depth1), max(depth2, depth3));
}

static const float2 kFullscreenTrianglePositions[3] =
{
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};

[shader("vertex")]
FullscreenVertexOutput gpuDrivenPreviewCompositeVertexMain(uint vertexIndex : SV_VertexID)
{
    FullscreenVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 gpuDrivenPreviewCompositeFragmentMain(
    FullscreenVertexOutput input,
    uniform GPUDrivenPreviewPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> colorHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.deferredColorBuffer, 0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    StructuredBuffer<uint> colors = colorHandle;
    uint2 pixel = min(uint2(input.position.xy), uint2(params.width - 1u, params.height - 1u));
    uint packed = colors[pixel.y * params.width + pixel.x];
    return float4(
        float(packed & 0xffu),
        float((packed >> 8u) & 0xffu),
        float((packed >> 16u) & 0xffu),
        float((packed >> 24u) & 0xffu)) / 255.0;
}
