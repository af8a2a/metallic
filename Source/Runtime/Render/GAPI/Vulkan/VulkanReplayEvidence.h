#pragma once
#include <cstdint>
#include <vector>
namespace metallic::render { class ComputePipeline; class Device; class Queue; }
namespace metallic::render::vulkan {
std::vector<uint8_t> replayComputeSpirv(ComputePipeline& pipeline, bool deviceCode);
bool replayUsesNativeDescriptorHeap(Device& device);
struct ReplayQueueIdentity { uint32_t familyIndex; uintptr_t identity; };
ReplayQueueIdentity replayQueueIdentity(Queue& queue);
} // namespace metallic::render::vulkan
