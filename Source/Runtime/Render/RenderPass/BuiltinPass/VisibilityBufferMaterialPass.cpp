#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/VisibilityMaterialParameters.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"
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
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite).format = Format::RGBA8Unorm;
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
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
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        device_ = context.device;
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .parameters = parameterAbi<VisibilityMaterialParams>(kVisibilityMaterialABI, ParameterTransport::InlinePush),
            .debugName = "VisibilityBufferMaterial",
        }, log);
    }

    Result<> prepareExecution(RenderGraphExecutionContext& context) override
    {
        prepared_ = {};
        return context.commandBuffer().frameContext() ? prepareMaterial(context, true) : Result<>{};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        return context.commandBuffer().frameContext() ? prepared_.record(context.commandBuffer()) : prepareMaterial(context, false);
    }

private:
    Result<> prepareMaterial(RenderGraphExecutionContext& context, bool prepare)
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
        const auto mode = properties().value("visualization", "shaded");
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const auto optional = [&writer](Buffer* buffer) { return buffer ? writer.buffer(buffer) : ShaderBuffer{}; };
        const VisibilityMaterialParams params{
            .output = writer.storageImage(color.view()),
            .visibility = writer.sampledImage(visibility.view()),
            .instances = optional(views.instances.buffer),
            .materials = optional(views.materials.buffer),
            .records = optional(views.meshletDraws.buffer),
            .meshlets = optional(views.meshlets.buffer),
            .vertices = optional(views.vertices.buffer),
            .vertexIndices = optional(views.meshletVertices.buffer),
            .triangles = optional(views.meshletTriangleWords.buffer),
            .geometries = optional(views.geometries.buffer),
            .streamRecords = optional(stream ? stream->visibleClusterBuffer : nullptr),
            .groups = optional(stream ? stream->activeGroupBuffer : nullptr),
            .pages = optional(stream ? stream->pageBuffer : nullptr),
            .pageTable = optional(stream ? stream->pageTableBuffer : nullptr),
            .streamParams = optional(stream ? stream->paramsBuffer : nullptr),
            .settings = {info.width, info.height, info.residentRecordCount, stream ? stream->visibleRecordCapacity : 0u,
                mode == "baseColor" ? 1u : mode == "normal" ? 2u : mode == "instance" ? 3u : 0u,
                info.eye[0], info.eye[1], info.eye[2]},
        };
        auto encoded = writer.encode(params, kVisibilityMaterialABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        auto dispatch = program_.prepareDispatch(*encoded, (info.width + 7) / 8, (info.height + 7) / 8);
        if (!dispatch) { return makeError(dispatch.error()); }
        if (prepare) { prepared_ = std::move(*dispatch); return {}; }
        return dispatch->record(commands);
    }

    PreparedComputeDispatch prepared_;
    Device* device_ = nullptr;
    ComputeKernel program_;
};
} // namespace

std::unique_ptr<RenderGraphPass> createVisibilityBufferMaterialPass()
{
    return std::make_unique<VisibilityBufferMaterialPass>();
}
} // namespace metallic::render::builtin_pass
