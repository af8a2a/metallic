#include "../../../Source/Runtime/Render/Core/InstanceCullParameters.h"
[[vk::push_constant]] ConstantBuffer<InstanceCullParameters> gInstanceCull;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

[shader("compute")]
[numthreads(1, 1, 1)]
void gpuDrivenPreviewResetMain(uint3 id : SV_DispatchThreadID)
{
    InstanceCullParameters p = gInstanceCull;
    if (p.phase == 0u) { resolveDescriptor(p.counter)[0] = 0u; }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenPreviewInstanceCullMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    InstanceCullParameters p = gInstanceCull;
    if (!p.settings.contains(0u)) { return; }
    RWStructuredBuffer<uint> visibleInstanceCounterHandle = resolveDescriptor(p.counter);
    GPUDrivenPreviewParams params = p.settings.data[0];
    uint instanceIndex = dispatchThreadId.x;
    if (instanceIndex >= params.instanceCount || !p.instances.contains(instanceIndex) || !p.visibility.contains(instanceIndex)) {
        return;
    }

    if (p.streamOwners.contains(instanceIndex) && p.streamOwners.data[instanceIndex] != 0u) {
        // Preserve Stream's early state during the late resident pass.
        if (p.phase == 0u) { p.visibility.data[instanceIndex] = 0u; }
        return;
    }
    if (p.phase != 0 && p.visibility.data[instanceIndex] != 2u) {
        return;
    }

    GPUDrivenPreviewInstance instance = p.instances.data[instanceIndex];
    if ((instance.identity.w & kInstanceVisible) == 0u) {
        p.visibility.data[instanceIndex] = 0u;
        return;
    }
    SceneTransform transform = instanceTransform(instance);
    float3 center = transformPoint(transform, instance.localBoundingSphere.xyz);
    float radius = max(
        instance.localBoundingSphere.w * transformMaxScale(transform),
        0.000001) + params.displacementBound;
    CameraData camera = cullingCamera(params);

    if ((params.cullingFlags & kCullInstanceFrustum) != 0 &&
        !sphereInsideFrustum(center, radius, camera)) {
        p.visibility.data[instanceIndex] = 0u;
        return;
    }

    if ((params.cullingFlags & kCullInstanceHZB) == 0 || (p.phase == 0u && params.hzbValid == 0u)) {
        p.visibility.data[instanceIndex] = p.phase == 0 ? 1u : 3u;
        uint outputIndex = 0u;
        InterlockedAdd(visibleInstanceCounterHandle[0], 1u, outputIndex);
        if (outputIndex < params.instanceCount && p.visibleIds.contains(outputIndex)) {
            p.visibleIds.data[outputIndex] = instanceIndex;
        }
        return;
    }

    CameraData hzbCamera = p.phase == 0 ? previousCullingCamera(params) : camera;
    bool occluded = sphereOccluded(center, radius, hzbCamera, params, p.hzb);
    p.visibility.data[instanceIndex] = occluded ? (p.phase == 0 ? 2u : 0u) :
        (p.phase == 0 ? 1u : 3u);
    if (!occluded) {
        uint outputIndex = 0u;
        InterlockedAdd(visibleInstanceCounterHandle[0], 1u, outputIndex);
        if (outputIndex < params.instanceCount && p.visibleIds.contains(outputIndex)) {
            p.visibleIds.data[outputIndex] = instanceIndex;
        }
    }
}
