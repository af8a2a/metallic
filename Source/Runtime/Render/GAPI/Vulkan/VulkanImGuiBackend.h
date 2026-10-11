#pragma once

#include "Runtime/Render/GAPI/RHI.h"

struct ImDrawList;
struct ImDrawData;
struct ImGuiViewport;

namespace metallic::render::vulkan {

struct ImGuiShaderRequest {
    bool hdr = false;
    bool scRgbImage = false;
    bool targetSrgb = false;
    bool encodePQ = false;
    float paperWhiteNits = 203.0f;
};

// The application layer supplies compilation/cache policy; GAPI owns the native ABI.
struct ImGuiShaderServices {
    std::function<Result<std::unique_ptr<ShaderModule>>(Device&, const ImGuiShaderRequest&, bool vertex)> load;
    std::function<Result<>(Device&, uint64_t, const std::function<Result<>(PipelineCache&)>&)> cache;
};

struct ImGuiDisplayDesc {
    Format colorFormat = Format::RGBA16Sfloat;
    Format pqOutputFormat = Format::Unknown;
    bool hdr = true;
    float paperWhiteNits = 203.0f;
    uint32_t imageCount = 2;
};

using ImGuiTexture = uint64_t;

// ImGui context-thread object. Shut down before destroying that context or Device.
class VulkanImGuiBackend {
public:
    VulkanImGuiBackend();
    ~VulkanImGuiBackend();
    VulkanImGuiBackend(const VulkanImGuiBackend&) = delete;
    VulkanImGuiBackend& operator=(const VulkanImGuiBackend&) = delete;
    Result<> init(Device& device, Queue& queue, Swapchain& swapchain,
        ImGuiShaderServices shaders, float paperWhiteNits);
    // Also supports offscreen composition tests.
    Result<> init(Device& device, Queue& queue, const ImGuiDisplayDesc& desc, ImGuiShaderServices shaders);
    Result<> configure(Swapchain& swapchain, float paperWhiteNits);
    Result<> configure(const ImGuiDisplayDesc& desc);
    void shutdown();
    Result<> newFrame();
    Result<ImGuiTexture> addTexture(TextureView& view);
    void removeTexture(ImGuiTexture texture);
    void beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport);
    // Retains registered views/descriptors for this recording, including textures
    // used only in platform windows. Call even when the main window is minimized.
    Result<> retainTextures(CommandBuffer& commands);
    // Explicit draw data enables native offscreen panel captures without desktop automation.
    Result<> render(CommandBuffer& commands, ImDrawData* drawData = nullptr);
    // Source is a retained DR sampled-image handle; independent of ImGui texture IDs.
    Result<> encodeHDR10(CommandBuffer& commands, BindlessHeap& heap, BindlessHandle source,
        uint32_t width, uint32_t height);
    // The caller seals its frame completion after this method, covering all windows.
    Result<> renderPlatformWindows();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
