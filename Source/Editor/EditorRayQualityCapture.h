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
    bool captureRasterInputs = false;
    bool capturePathDiagnostics = false;
    bool capturePostprocessInputs = false;
    uint32_t diagnosticRows = 0;
    void compiled(Json) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override
    {
        device_ = &device;
    }
    void endExecution(bool) override {}
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t passId,
        std::string_view pass, std::span<const render::DebugResourceBinding> resources, const Json& values) override
    {
        using namespace render;
        if (!capture) { return; }
        if (capturePostprocessInputs && (pass == "DLSSSR" || pass == "AutoExposure" ||
            pass == "DLSSNR" || pass == "ColorGrading" || pass == "FinalBlit")) {
            postprocessBoundaries_.push_back({{"pass", pass}, {"passId", passId},
                {"checkpoint", checkpoint}, {"values", values}});
        }
        if (checkpoint != "AfterPass") { return; }
        for (const auto& resource : resources) {
            const bool postprocessBuffer = capturePostprocessInputs && resource.id == "AutoExposure.exposure";
            if (resource.buffer && (postprocessBuffer || (capturePathDiagnostics &&
                (resource.id == "streaming.VBuffer.rtRepresentations" ||
                 resource.id == "streaming.VBuffer.rtRepresentationState" ||
                 resource.id == "streaming.VBuffer.tlasInstances")))) {
                if (copies_.contains(resource.id)) { continue; }
                const uint64_t bytes = resource.size ? resource.size : resource.buffer->desc().size - resource.offset;
                Copy copy;
                copy.postprocess = postprocessBuffer;
                copy.diagnostic = !postprocessBuffer;
                copy.metadata = {{"resource", resource.id}, {"bytes", bytes}, {"layout", resource.layout},
                    {"allocation", resource.allocation}, {"metadata", resource.metadata}};
                if (!device_->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { copy.buffer = std::move(value); })) {
                    throw std::runtime_error("RT diagnostic readback allocation failed");
                }
                BufferBarrierDesc barrier{.buffer = resource.buffer,
                    .before = resourceSyncScope(resource.state, PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                    .range = {.offset = resource.offset, .size = bytes}};
                if (!commands.synchronize({.buffers = {&barrier, 1}})) { throw std::runtime_error("RT diagnostic barrier failed"); }
                if (!resource.buffer->slice({resource.offset, bytes}).and_then([&](const auto& source) {
                    return copy.buffer->slice().and_then([&](const auto& destination) { return commands.copyBuffer(source, destination); });
                })) { throw std::runtime_error("RT diagnostic copy failed"); }
                std::swap(barrier.before, barrier.after);
                if (!commands.synchronize({.buffers = {&barrier, 1}})) { throw std::runtime_error("RT diagnostic restore failed"); }
                copies_.emplace(resource.id, std::move(copy));
                continue;
            }
            const bool postprocessTexture = capturePostprocessInputs && (resource.id == "DLSSSR.color" ||
                resource.id == "AutoExposure.color" || resource.id == "DLSSNR.color" || resource.id == "ColorGrading.lut");
            if (!postprocessTexture && resource.id != "Shadows.shadow" && resource.id != "Deferred.color" && resource.id != "FinalBlit.color" &&
                !(captureReflections && resource.id == "ReflectionQuality.color") &&
                !(capturePathDiagnostics && resource.id == "ReflectionQuality.pathDiagnostics") &&
                !(captureRasterInputs && (resource.id == "VBuffer.depth" || resource.id == "VBuffer.visibility"))) { continue; }
            if (!resource.texture || copies_.contains(resource.id)) { continue; }
            const auto& desc = resource.texture->desc();
            const auto info = formatInfo(desc.format);
            const char* format = nullptr;
            switch (desc.format) {
            case Format::R8Unorm: format = "unorm8x1"; break;
            case Format::R16Sfloat: format = "float16x1"; break;
            case Format::R32Sfloat: format = "float32x1"; break;
            case Format::D32Sfloat: format = "float32x1"; break;
            case Format::R32Uint: format = "uint32x1"; break;
            case Format::RG16Sfloat: format = "float16x2"; break;
            case Format::RGBA16Sfloat: format = "float16x4"; break;
            case Format::RGBA32Sfloat: format = "float32x4"; break;
            case Format::RGBA8Unorm: format = "unorm8x4"; break;
            case Format::RGBA8sRGB: format = "srgb8x4"; break;
            default: throw std::runtime_error("Unsupported quality capture texture format: " + resource.id);
            }
            Copy copy;
            copy.diagnostic = resource.id == "ReflectionQuality.pathDiagnostics";
            copy.postprocess = postprocessTexture;
            const uint32_t width = copy.diagnostic ? 104u : desc.width;
            const uint32_t height = copy.diagnostic ? diagnosticRows : desc.height;
            const uint32_t depth = desc.type == TextureType::Texture3D ? desc.depth : 1u;
            copy.metadata = {{"resource", resource.id}, {"width", width}, {"height", height}, {"format", format}};
            if (postprocessTexture) { copy.metadata["depth"] = depth; }
            const uint64_t bytes = uint64_t(width) * height * depth * info.bytesPerBlock;
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
                return commands.copyTextureToBuffer({.texture = resource.texture, .buffer = slice,
                    .bufferRowPitch = width * info.bytesPerBlock, .bufferSlicePitch = width * height * info.bytesPerBlock,
                    .width = width, .height = height, .depth = depth});
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
        if (captureRasterInputs && (!copies_.contains("VBuffer.depth") || !copies_.contains("VBuffer.visibility"))) {
            throw std::runtime_error("Quality capture is missing raster diagnostic inputs");
        }
        if (capturePathDiagnostics && (!copies_.contains("ReflectionQuality.pathDiagnostics") ||
            !copies_.contains("streaming.VBuffer.rtRepresentations") ||
            !copies_.contains("streaming.VBuffer.rtRepresentationState") || !copies_.contains("streaming.VBuffer.tlasInstances"))) {
            throw std::runtime_error("Quality capture is missing path or RT owner diagnostics");
        }
        if (capturePostprocessInputs) {
            for (const char* resource : {"DLSSSR.color", "AutoExposure.color", "AutoExposure.exposure",
                "DLSSNR.color", "ColorGrading.lut"}) {
                if (!copies_.contains(resource)) {
                    throw std::runtime_error(std::string("Quality capture is missing postprocess input ") + resource);
                }
            }
            for (const char* pass : {"AutoExposure", "ColorGrading", "FinalBlit"}) {
                bool parametersFound = false;
                for (const auto& boundary : postprocessBoundaries_) {
                    parametersFound |= boundary.at("pass") == pass &&
                        boundary.at("checkpoint") == "PostprocessParameters";
                }
                if (!parametersFound) {
                    throw std::runtime_error(std::string("Quality capture is missing postprocess parameters for ") + pass);
                }
            }
        }
        std::filesystem::create_directories(directory);
        Json report = Json::array();
        Json diagnostics = Json::array();
        Json postprocess = Json::array();
        for (auto& [name, copy] : copies_) {
            copy.buffer->invalidate();
            const void* data = copy.buffer->map();
            if (!data) { throw std::runtime_error("Quality readback map failed"); }
            std::ofstream file(directory / (name + ".bin"), std::ios::binary);
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(copy.buffer->desc().size));
            copy.buffer->unmap();
            if (!file) { throw std::runtime_error("Quality capture export failed"); }
            copy.metadata["file"] = name + ".bin";
            (copy.postprocess ? postprocess : (copy.diagnostic ? diagnostics : report)).push_back(copy.metadata);
        }
        std::ofstream manifest(directory / "Images.json");
        manifest << report.dump(2) << '\n';
        if (!manifest) { throw std::runtime_error("Quality manifest export failed"); }
        if (capturePathDiagnostics) {
            std::ofstream diagnosticManifest(directory / "Diagnostics.json");
            diagnosticManifest << diagnostics.dump(2) << '\n';
            if (!diagnosticManifest) { throw std::runtime_error("Diagnostic manifest export failed"); }
        }
        if (capturePostprocessInputs) {
            std::ofstream postprocessManifest(directory / "Postprocess.json");
            postprocessManifest << Json{{"resources", postprocess}, {"boundaries", postprocessBoundaries_}}.dump(2) << '\n';
            if (!postprocessManifest) { throw std::runtime_error("Postprocess manifest export failed"); }
        }
        return report;
    }
private:
    struct Copy { std::unique_ptr<render::Buffer> buffer; Json metadata; bool diagnostic = false; bool postprocess = false; };
    render::Device* device_ = nullptr;
    std::map<std::string, Copy> copies_;
    Json postprocessBoundaries_ = Json::array();
};

} // namespace metallic
