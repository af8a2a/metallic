#include "../../Libraries/GPUDriven/GPUDrivenCullingCommon.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void gpuDrivenPreviewResetMain(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uniform GPUDrivenPreviewPush push)
{
    if (push.passIndex == 0u) {
        DescriptorHandle<RWStructuredBuffer<uint>> visibleInstanceCounterHandle =
            DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
                push.bufferShaderIndexBase + push.visibleInstanceCounterBuffer,
                0));
        visibleInstanceCounterHandle[0] = 0u;
    }
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
    DescriptorHandle<RWStructuredBuffer<uint>> visibilityHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.instanceVisibilityBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> visibleInstanceIdsHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.visibleInstanceIdsBuffer,
            0));
    DescriptorHandle<RWStructuredBuffer<uint>> visibleInstanceCounterHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.visibleInstanceCounterBuffer,
            0));

    GPUDrivenPreviewParams params = paramsHandle[0];
    uint instanceIndex = dispatchThreadId.x;
    if (instanceIndex >= params.instanceCount) {
        return;
    }

    RWStructuredBuffer<uint> visibility = visibilityHandle;
    if (push.streamOwnerMaskBuffer != kInvalidTextureIndex) {
        DescriptorHandle<StructuredBuffer<uint>> streamOwnerMaskHandle =
            DescriptorHandle<StructuredBuffer<uint>>(uint2(
                push.bufferShaderIndexBase + push.streamOwnerMaskBuffer,
                0));
        StructuredBuffer<uint> streamOwnerMask = streamOwnerMaskHandle;
        if (streamOwnerMask[instanceIndex] != 0u) {
            // Early culling clears stale shared state before the Stream
            // producer runs. Late culling must preserve Stream's early state
            // (2 == deferred/occluded) so its late HZB pass can recover it.
            if (push.passIndex == 0u) {
                visibility[instanceIndex] = 0u;
            }
            return;
        }
    }
    if (push.passIndex != 0 && visibility[instanceIndex] != 2u) {
        return;
    }

    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    GPUDrivenPreviewInstance instance = instances[instanceIndex];
    if ((instance.identity.w & kInstanceVisible) == 0u) {
        visibility[instanceIndex] = 0u;
        return;
    }
    SceneTransform transform = instanceTransform(instance);
    float3 center = transformPoint(transform, instance.localBoundingSphere.xyz);
    float radius = max(
        instance.localBoundingSphere.w * transformMaxScale(transform),
        0.000001);
    CameraData camera = cullingCamera(params);

    if ((params.cullingFlags & kCullInstanceFrustum) != 0 &&
        !sphereInsideFrustum(center, radius, camera)) {
        visibility[instanceIndex] = 0u;
        return;
    }

    if ((params.cullingFlags & kCullInstanceHzb) == 0 || params.hzbValid == 0) {
        visibility[instanceIndex] = push.passIndex == 0 ? 1u : 3u;
        uint outputIndex = 0u;
        InterlockedAdd(visibleInstanceCounterHandle[0], 1u, outputIndex);
        if (outputIndex < params.instanceCount) {
            visibleInstanceIdsHandle[outputIndex] = instanceIndex;
        }
        return;
    }

    uint currentHzb = params.frameIndex & 1u;
    uint hzbIndex = push.passIndex == 0 ? (currentHzb ^ 1u) : currentHzb;
    CameraData hzbCamera = push.passIndex == 0 ? previousCullingCamera(params) : camera;
    bool occluded = sphereOccluded(center, radius, hzbCamera, params, push, hzbIndex);
    visibility[instanceIndex] = occluded ? (push.passIndex == 0 ? 2u : 0u) :
        (push.passIndex == 0 ? 1u : 3u);
    if (!occluded) {
        uint outputIndex = 0u;
        InterlockedAdd(visibleInstanceCounterHandle[0], 1u, outputIndex);
        if (outputIndex < params.instanceCount) {
            visibleInstanceIdsHandle[outputIndex] = instanceIndex;
        }
    }
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
