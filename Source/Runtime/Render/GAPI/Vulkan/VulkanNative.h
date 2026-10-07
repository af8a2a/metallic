#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include "VulkanDeviceProperties.h"

namespace metallic::render::vulkan {

struct NativeDevice {
    // Borrowed immutable dispatch table; valid for the lifetime of the Device.
    const VolkDeviceTable* functions = nullptr;
    const VolkInstanceTable* instanceFunctions = nullptr;
    const VulkanDeviceProperties* properties = nullptr;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t apiVersion = 0;
    bool descriptorHeapEnabled = false;
    bool shaderUntypedPointersEnabled = false;
    bool validationEnabled = false;
    bool synchronizationValidationEnabled = false;
    bool validationMessengerActive = false;
};

struct NativeQueue {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
};

struct NativeBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint64_t size = 0;
    VkDevice device = VK_NULL_HANDLE;
};

struct NativePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

struct NativeGraphicsShaders {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderEXT vertex = VK_NULL_HANDLE;
    VkShaderEXT fragment = VK_NULL_HANDLE;
};

struct NativeTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t mipCount = 0;
    uint32_t layerCount = 0;
    VkImageCreateFlags flags = 0;
    VkImageUsageFlags usage = 0;
};

NativeDevice nativeDevice(Device& device);
NativeQueue nativeQueue(Queue& queue);
NativeBuffer nativeBuffer(Buffer& buffer);
// Borrowed module; the owning ShaderModule must outlive native creation.
VkShaderModule nativeShaderModule(ShaderModule& shader);
// For external ABIs (e.g. ImGui). Lock creation and record a successful stable
// state key in the same RHI cache. The caller owns the resulting native pipeline.
Result<> createCachedGraphicsPipeline(PipelineCache& cache, const VkGraphicsPipelineCreateInfo& info,
    uint64_t stateHash, VkPipeline& pipeline);
NativePipeline nativePipeline(ComputePipeline& pipeline);
// Available only when diagnostic replay was enabled at shader creation.
std::vector<uint8_t> nativeComputeSpirv(ComputePipeline& pipeline, bool deviceCode);
NativePipeline nativePipeline(GraphicsPipeline& pipeline);
NativeGraphicsShaders nativeShaders(GraphicsShaderObjectProgram& program);
NativeTexture nativeTexture(Texture& texture);
// Lexically bound native recording. Do not retain exported handles/function tables
// or interleave RHI state/draw commands before the scope ends. Commands must outlive
// the scope. Every exit invalidates all cached RHI execution and dynamic state.
class ExternalCommandScope final {
public:
    explicit ExternalCommandScope(CommandBuffer& commands);
    ~ExternalCommandScope();
    ExternalCommandScope(const ExternalCommandScope&) = delete;
    ExternalCommandScope& operator=(const ExternalCommandScope&) = delete;
    explicit operator bool() const { return commands_ != nullptr; }
    VkCommandBuffer commandBuffer() const;
    const VolkDeviceTable& functions() const;
    // Materialize and retain both the view and image for this recording.
    [[nodiscard]] Result<VkImageView> imageView(TextureView& view) const;
private:
    CommandBuffer* commands_ = nullptr;
};

// Read-only device identity, with no recording handle export.
VkDevice nativeCommandBufferDevice(CommandBuffer& commandBuffer);
VkFormat nativeSwapchainFormat(Swapchain& swapchain);
// Exporters use the backend policy, rather than hard-coding an optimal layout.
VkImageLayout nativeImageLayout(TextureView& view, TextureLayout layout);
// Backend diagnostic query; does not materialize a view.
bool hasNativeImageView(const TextureView& view);
// For host-side descriptor creation with an external lifetime owner only.
// During recording, export through ExternalCommandScope::imageView instead.
VkImageView nativeImageView(TextureView& view);
} // namespace metallic::render::vulkan
