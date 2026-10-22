#pragma once

#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/GAPI/TextureFormat.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

namespace metallic {

// Used only outside timed frames. The owner drains submitted work before export
// and before destroying this observer, including when a capture fails.
class RayQualityCapture final : public render::IRenderDebugObserver {
public:
    using Json = nlohmann::json;
    bool capture = false;
    bool captureReflections = false;
    void compiled(Json) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override
    {
        device_ = &device;
    }
    void endExecution(bool) override {}
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view, std::span<const render::DebugResourceBinding> resources, const Json&) override
    {
        using namespace render;
        if (!capture || checkpoint != "AfterPass") { return; }
        for (const auto& resource : resources) {
            if (resource.id != "Shadows.shadow" && resource.id != "Deferred.color" && resource.id != "FinalBlit.color" &&
                !(captureReflections && resource.id == "ReflectionQuality.color")) { continue; }
            if (!resource.texture || copies_.contains(resource.id)) { continue; }
            const auto& desc = resource.texture->desc();
            const auto info = formatInfo(desc.format);
            const char* format = nullptr;
            switch (desc.format) {
            case Format::R8Unorm: format = "unorm8x1"; break;
            case Format::R16Sfloat: format = "float16x1"; break;
            case Format::R32Sfloat: format = "float32x1"; break;
            case Format::RG16Sfloat: format = "float16x2"; break;
            case Format::RGBA16Sfloat: format = "float16x4"; break;
            case Format::RGBA32Sfloat: format = "float32x4"; break;
            case Format::RGBA8Unorm: format = "unorm8x4"; break;
            case Format::RGBA8sRGB: format = "srgb8x4"; break;
            default: throw std::runtime_error("Unsupported quality capture texture format: " + resource.id);
            }
            Copy copy;
            copy.metadata = {{"resource", resource.id}, {"width", desc.width}, {"height", desc.height}, {"format", format}};
            const uint64_t bytes = uint64_t(desc.width) * desc.height * info.bytesPerBlock;
            if (!device_->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { copy.buffer = std::move(value); })) {
                throw std::runtime_error("Quality readback allocation failed");
            }
            TextureBarrierDesc barrier{.texture = resource.texture,
                .oldLayout = textureLayoutForResourceState(resource.state), .newLayout = TextureLayout::TransferSource,
                .before = resourceSyncScope(resource.state, PipelineStageBits::AllCommands),
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
            if (!commands.synchronize({.textures = {&barrier, 1}})) { throw std::runtime_error("Quality capture barrier failed"); }
            if (!copy.buffer->slice().and_then([&](const auto& slice) {
                return commands.copyTextureToBuffer({.texture = resource.texture, .buffer = slice, .width = desc.width, .height = desc.height});
            })) { throw std::runtime_error("Quality capture copy failed"); }
            std::swap(barrier.before, barrier.after);
            std::swap(barrier.oldLayout, barrier.newLayout);
            if (!commands.synchronize({.textures = {&barrier, 1}})) { throw std::runtime_error("Quality capture restore failed"); }
            copies_.emplace(resource.id, std::move(copy));
        }
    }
    Json exportAfterDrain(const std::filesystem::path& directory)
    {
        if (!copies_.contains("Shadows.shadow") || !copies_.contains("Deferred.color") || !copies_.contains("FinalBlit.color")) {
            throw std::runtime_error("Quality capture is missing a required scene output");
        }
        if (captureReflections && !copies_.contains("ReflectionQuality.color")) {
            throw std::runtime_error("Quality capture is missing the path-traced reflection diagnostic");
        }
        std::filesystem::create_directories(directory);
        Json report = Json::array();
        for (auto& [name, copy] : copies_) {
            copy.buffer->invalidate();
            const void* data = copy.buffer->map();
            if (!data) { throw std::runtime_error("Quality readback map failed"); }
            std::ofstream file(directory / (name + ".bin"), std::ios::binary);
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(copy.buffer->desc().size));
            copy.buffer->unmap();
            if (!file) { throw std::runtime_error("Quality capture export failed"); }
            copy.metadata["file"] = name + ".bin";
            report.push_back(copy.metadata);
        }
        std::ofstream manifest(directory / "Images.json");
        manifest << report.dump(2) << '\n';
        if (!manifest) { throw std::runtime_error("Quality manifest export failed"); }
        return report;
    }
private:
    struct Copy { std::unique_ptr<render::Buffer> buffer; Json metadata; };
    render::Device* device_ = nullptr;
    std::map<std::string, Copy> copies_;
};

} // namespace metallic
