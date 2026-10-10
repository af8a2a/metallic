#pragma once

#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanImGuiBackend.h"

#include <array>

namespace metallic {

// A bounded, asynchronous viewer of pass-boundary snapshots. No graph resource
// pointers survive a boundary or enter the UI.
class EditorResourceInspector {
public:
    render::RenderDebugRuntime& runtime() { return runtime_; }
    void select(std::string resource);
    debug::DebugValue diagnostics() const;
    void draw(render::RenderDebugRuntime& runtime, render::Device& device,
        render::vulkan::VulkanImGuiBackend& backend, float scale);
    render::Result<> upload(render::CommandBuffer& commands);
    void shutdown(render::vulkan::VulkanImGuiBackend& backend);

private:
    friend class EditorApplication;
    void cancel(debug::DebugCore& core);
    void request(render::RenderDebugRuntime& runtime, const debug::DebugSnapshot& snapshot,
        const debug::DebugValue& resource);
    void drawTexture(render::Device& device, render::vulkan::VulkanImGuiBackend& backend);
    void drawCube();
    void drawBuffer();
    bool rebuildImage(render::Device& device, render::vulkan::VulkanImGuiBackend& backend);

    render::RenderDebugRuntime runtime_;
    std::string selected_, job_, status_, graph_;
    std::string bufferLayout_;
    uint64_t generation_ = 0;
    char filter_[192] = {};
    bool live_ = false, refresh_ = true, imageDirty_ = false, fit_ = true, hex_ = false;
    double nextRefresh_ = 0;
    int channel_ = 0, scalar_ = 0, columns_ = 4;
    int sourceFilter_ = 0;
    bool rawBuffer_ = false;
    int fieldPage_ = 0;
    float exposure_ = 0, rangeMin_ = 0, rangeMax_ = 1, zoom_ = 1;
    uint64_t offset_ = 0;
    int count_ = 256;
    std::array<int, 4> roi_ = {0, 0, 0, 0};
    int slice_ = 0;
    bool cube_ = true;
    float cubeYaw_ = 0.65f, cubePitch_ = 0.45f, cubeZoom_ = 1.f;
    std::shared_ptr<const debug::DebugCapture> capture_;
    std::unique_ptr<render::Texture> image_;
    std::unique_ptr<render::TextureView> view_;
    std::unique_ptr<render::Buffer> upload_;
    render::vulkan::ImGuiTexture descriptor_ = 0;
};

} // namespace metallic
