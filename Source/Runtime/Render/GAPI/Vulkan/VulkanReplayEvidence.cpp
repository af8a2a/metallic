#include "VulkanReplayEvidence.h"
#include "VulkanNative.h"
namespace metallic::render::vulkan {
std::vector<uint8_t> replayComputeSpirv(ComputePipeline& pipeline, bool deviceCode)
{
    const auto code = nativeComputeSpirv(pipeline, deviceCode);
    return {code.begin(), code.end()};
}
bool replayUsesNativeDescriptorHeap(Device& device)
{
    return nativeDevice(device).descriptorHeapEnabled;
}
ReplayQueueIdentity replayQueueIdentity(Queue& queue)
{
    const auto native = nativeQueue(queue);
    return {native.familyIndex, reinterpret_cast<uintptr_t>(native.queue)};
}
} // namespace metallic::render::vulkan
