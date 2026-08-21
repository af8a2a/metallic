#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#ifndef METALLIC_HAS_NRC
#define METALLIC_HAS_NRC 0
#endif

#if METALLIC_HAS_NRC
#include <volk.h>

#include <NrcCommon.h>
#include <NrcVk.h>
#endif

namespace metallic::render::vulkan {

// Vulkan integration for NVIDIA's Neural Radiance Cache (NRC) SDK, following
// the RTXGI v2 Pathtracer sample's NrcVulkanIntegration. Metallic allocates
// the NRC buffers itself (GlobalSettings::enableGPUMemoryAllocation = false)
// so they can be bound directly through the regular RHI binding model; the
// native VkBuffer handles and device addresses are handed to the SDK via
// Context::Configure.
#if METALLIC_HAS_NRC

class NrcIntegration {
public:
    static constexpr uint32_t kBufferCount = static_cast<uint32_t>(nrc::BufferIdx::Count);

    NrcIntegration();
    ~NrcIntegration();

    NrcIntegration(NrcIntegration&&) noexcept;
    NrcIntegration& operator=(NrcIntegration&&) noexcept;

    NrcIntegration(const NrcIntegration&) = delete;
    NrcIntegration& operator=(const NrcIntegration&) = delete;

    // Initializes the NRC library (process-wide, reference counted) and
    // creates a context on the device.
    Result initialize(Device& device, std::string& log);
    void clear();
    bool valid() const;

    // (Re)configures the context and (re)allocates the shared buffers.
    // Call whenever the context settings change; may stall.
    Result configure(const nrc::ContextSettings& settings, Device& device, std::string& log);
    const nrc::ContextSettings& contextSettings() const { return contextSettings_; }

    Result beginFrame(CommandBuffer& commandBuffer, const nrc::FrameSettings& frameSettings);
    Result populateShaderConstants(::NrcConstants& outConstants) const;
    Result queryAndTrain(CommandBuffer& commandBuffer, float* trainingLoss);
    Result resolve(CommandBuffer& commandBuffer, TextureView& outputView);
    Result endFrame(Queue& queue);

    Buffer* buffer(uint32_t index) const
    {
        return index < kBufferCount && buffers_[index] ? buffers_[index].get() : nullptr;
    }

private:
    nrc::vulkan::Context* context_ = nullptr;
    std::array<std::unique_ptr<Buffer>, kBufferCount> buffers_;
    nrc::vulkan::Buffers nativeBuffers_ {};
    nrc::ContextSettings contextSettings_ {};
};

#endif // METALLIC_HAS_NRC

} // namespace metallic::render::vulkan
