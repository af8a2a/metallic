#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/NamedComputeParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <cstring>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kNamedComputeABI = 0x4e43500000000004ull;

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
        reflection.addTextureInput("visibility", "Stream visibility IDs").sampledRead().format = Format::R32Uint;
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
        auto result = ShaderRegistry::instance().getShader({
            .moduleName = "Features/VisibilityBuffer/VisibilityBufferMaterial",
            .entryPointName = "visibilityBufferMaterialMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        device_ = context.device;
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .parameters = parameterAbi<NamedComputeParameters>(kNamedComputeABI),
            .debugName = "VisibilityBufferMaterial",
        }, log);
    }

    Result<> prepareExecution(RenderGraphExecutionContext& context) override
    {
        prepared_ = {};
        return metallic::render::RenderFrameContext::from(context.commandBuffer()) ? prepareMaterial(context, true) : Result<>{};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        return metallic::render::RenderFrameContext::from(context.commandBuffer()) ? prepared_.record(context.commandBuffer()) : prepareMaterial(context, false);
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
        if (!stream) { return makeError(Error::InvalidArgument); }
        TextureView* image = visibility.view();
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(context.commandBuffer()));
        VisibilityMaterialResourceParameters resources{};
        resources.output = writer.storageImageHandle(color.view());
        resources.visibility = writer.sampledImageHandle(image);
        resources.instances = writer.buffer(views.instances.buffer);
        resources.materials = writer.buffer(views.materials.buffer);
        resources.streamRecords = writer.buffer(stream->visibleClusterBuffer);
        resources.groups = writer.buffer(stream->activeGroupBuffer);
        resources.pages = writer.buffer(stream->pageBuffer);
        resources.pageTable = writer.buffer(stream->pageTableBuffer);
        resources.pageCount = writer.buffer(stream->paramsBuffer);
        struct Push { uint32_t width, height, streamCount, mode; float eye[3]; };
        const auto mode = properties().value("visualization", "shaded");
        const Push push{info.width, info.height, stream->visibleRecordCapacity,
            mode == "baseColor" ? 1u : mode == "normal" ? 2u : mode == "instance" ? 3u : 0u,
            {info.eye[0], info.eye[1], info.eye[2]}};
        auto encoded = encodeNamedParameters(writer, resources, push, kNamedComputeABI);
        if (!encoded) { return makeError(encoded.error()); }
        auto packet = program_.prepareDispatch(*encoded, (info.width + 7) / 8, (info.height + 7) / 8);
        if (!packet) { return makeError(packet.error()); }
        if (prepare) { prepared_ = std::move(*packet); return {}; }
        return packet->record(context.commandBuffer());
    }

    PreparedComputeDispatch prepared_;
    ComputeKernel program_;
    Device* device_ = nullptr;
};
} // namespace

std::unique_ptr<RenderGraphPass> createVisibilityBufferMaterialPass()
{
    return std::make_unique<VisibilityBufferMaterialPass>();
}
} // namespace metallic::render::builtin_pass
