#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <cstring>

namespace metallic::render::builtin_pass {
namespace {

class VisibilityBufferMaterialPass final : public ComputePass {
public:
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"visibility", "rasterInfo"}};
    }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("visibility", "Unified resident/stream visibility IDs").sampledRead().format = Format::R32Uint;
        reflection.addBufferInput("rasterInfo", "Raster scene, view and frame identity")
            .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
        reflection.addTextureOutput("color", "Scalar materials with geometric normals and directional lighting")
            .storageReadWrite().format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr || context.runtimeScene == nullptr) { return makeError(Error::InvalidArgument); }
        // Texture sampling/transmission belong to the full OpenPBR consumer.
        // Reject these explicitly instead of silently displaying scalar substitutes.
        for (const auto& material : context.runtimeScene->materials()) {
            if (material.transmissionFactor > 0 || material.diffuseTransmissionFactor > 0 ||
                (material.alphaMode == "BLEND" && material.baseColorFactor.w < 1)) {
                log = "VisibilityBufferMaterialPass supports opaque scalar materials only";
                return makeError(Error::Unsupported);
            }
        }
        if (!context.runtimeScene->textures().empty()) {
            log = "VisibilityBufferMaterialPass does not sample material textures";
            return makeError(Error::Unsupported);
        }
        if (program_.valid()) { return {}; }
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({
            .moduleName = "Features/VisibilityBuffer/VisibilityBufferMaterial",
            .entryPointName = "visibilityBufferMaterialMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        std::array<ComputeProgramBindingDesc, 15> bindings;
        for (uint32_t slot = 0; slot < bindings.size(); ++slot) {
            bindings[slot] = {.binding = slot, .kind = ComputeResourceBindingKind::StorageBuffer};
        }
        bindings[0].kind = ComputeResourceBindingKind::StorageImage;
        bindings[1].kind = ComputeResourceBindingKind::SampledImage;
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = 32,
            .bindings = bindings.data(), .bindingCount = uint32_t(bindings.size()),
            .debugName = "VisibilityBufferMaterial", .requiresRayQuery = false}, log);
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        const auto visibility = context.inputTexture("visibility");
        const auto infoBuffer = context.inputBuffer("rasterInfo");
        const auto color = context.outputTexture("color");
        auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!visibility.valid() || !infoBuffer.valid() || !color.valid() || !gpuScene ||
            infoBuffer.desc().size != sizeof(VisibilityBufferFrameInfo) ||
            infoBuffer.desc().memoryLocation != MemoryLocation::HostUpload) { return makeError(Error::InvalidArgument); }
        VisibilityBufferFrameInfo info;
        const void* mapped = infoBuffer.buffer()->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&info, mapped, sizeof(info));
        infoBuffer.buffer()->unmap();
        if (!context.runtimeScene() || info.sceneIdentity != context.runtimeScene()->resourceIdentity() ||
            info.frameIndex != context.frameIndex() || info.width != context.width() || info.height != context.height()) {
            return makeError(Error::InvalidArgument);
        }
        const auto& views = gpuScene->globalBufferViews();
        if (!views.validFor(gpuScene->drawSet().generation, gpuScene->drawSet().revision)) { return makeError(Error::InvalidArgument); }
        const auto* stream = gpuScene->visibilityStream({info.lightGridViewIndex, info.lightGridViewGeneration},
            info.frameIndex, info.sceneIdentity);
        if (info.hasStreamGeometry && !stream) { return makeError(Error::InvalidArgument); }
        // Unused optional descriptors point at a valid metadata buffer; the shader
        // selects its producer before accessing any geometry descriptor.
        Buffer* fallback = views.geometries.buffer;
        const auto optional = [fallback](Buffer* buffer) { return buffer ? buffer : fallback; };
        TextureView* image = visibility.view();
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureView = color.view()},
            {.binding = 1, .textureViews = &image, .textureViewCount = 1},
            {.binding = 2, .buffer = views.instances.buffer},
            {.binding = 3, .buffer = views.materials.buffer},
            {.binding = 4, .buffer = optional(views.meshletDraws.buffer)},
            {.binding = 5, .buffer = optional(views.meshlets.buffer)},
            {.binding = 6, .buffer = optional(views.vertices.buffer)},
            {.binding = 7, .buffer = optional(views.meshletVertices.buffer)},
            {.binding = 8, .buffer = optional(views.meshletTriangleWords.buffer)},
            {.binding = 9, .buffer = views.geometries.buffer},
            {.binding = 10, .buffer = stream ? stream->visibleClusterBuffer : fallback},
            {.binding = 11, .buffer = stream ? stream->activeGroupBuffer : fallback},
            {.binding = 12, .buffer = stream ? stream->pageBuffer : fallback},
            {.binding = 13, .buffer = stream ? stream->pageTableBuffer : fallback},
            {.binding = 14, .buffer = stream ? stream->paramsBuffer : fallback},
        };
        struct Push { uint32_t width, height, residentCount, streamCount, mode; float eye[3]; };
        const auto mode = properties().value("visualization", "shaded");
        const Push push{info.width, info.height, info.residentRecordCount, stream ? stream->visibleRecordCapacity : 0u,
            mode == "baseColor" ? 1u : mode == "normal" ? 2u : mode == "instance" ? 3u : 0u,
            {info.eye[0], info.eye[1], info.eye[2]}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .bindingCount = uint32_t(std::size(bindings)), .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (info.width + 7) / 8, .groupCountY = (info.height + 7) / 8});
    }

private:
    ComputeProgram program_;
};
} // namespace

std::unique_ptr<RenderGraphPass> createVisibilityBufferMaterialPass()
{
    return std::make_unique<VisibilityBufferMaterialPass>();
}
} // namespace metallic::render::builtin_pass
