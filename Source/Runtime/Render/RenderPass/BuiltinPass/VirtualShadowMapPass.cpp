#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

namespace metallic::render::builtin_pass {
namespace {

constexpr uint32_t kVirtualShadowMapStatisticsSize = 32 * sizeof(uint32_t);

VirtualShadowMapSettings shadowSettings(const RenderGraphProperties& properties)
{
    VirtualShadowMapSettings settings;
    settings.pageSize = properties.value("pageSize", settings.pageSize);
    settings.virtualPagesPerAxis = properties.value("virtualPagesPerAxis", settings.virtualPagesPerAxis);
    settings.clipmapCount = properties.value("clipmapCount", settings.clipmapCount);
    settings.physicalPageCount = properties.value("physicalPageCount", settings.physicalPageCount);
    settings.maxWorkItems = properties.value("maxWorkItems", settings.maxWorkItems);
    settings.firstClipmapExtent = properties.value("firstClipmapExtent", settings.firstClipmapExtent);
    settings.shadowBias = properties.value("shadowBias", settings.shadowBias);
    settings.cacheEnabled = boolProperty(&properties, "cacheEnabled", false);
    settings.forceRedraw = boolProperty(&properties, "forceRedraw", false);
    settings.enabled = boolProperty(&properties, "enabled", true) &&
        !boolProperty(&properties, "debugDisableShadows", false);
    return settings;
}

class VirtualShadowMapPass final : public UnsafePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        // Borrow the producer's complete geometry publication. A second geometry
        // requirement would prepare another camera-driven streaming traversal.
        return {.features = SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures,
            .textureFeedback = true};
    }

    bool supportsFrameOverlap() const override { return false; }

    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }

    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo"}};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Receiver depth from the matching visibility raster")
            .sampledRead().format = Format::D32Sfloat;
        auto& info = reflection.addBufferInput("rasterInfo", "Matching raster camera and scene identity")
            .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
        info.memoryLocation = MemoryLocation::HostUpload;
        reflection.addTextureOutput("shadow", "Selected celestial light's hard visibility for deferred direct lighting")
            .transferWrite().transient(RenderGraphInitialization::FullOverwrite).format = Format::R8Unorm;
        reflection.addBufferOutput("parameters", "Legacy single-light shadow identity and camera")
            .buffer(sizeof(ScreenSpaceShadowParameters), sizeof(ScreenSpaceShadowParameters)).transferWrite();
        reflection.addTextureOutput("debug", "Clipmap, page allocation and coarse fallback visualization")
            .transferWrite().transient(RenderGraphInitialization::FullOverwrite).format = Format::RGBA8Unorm;
        reflection.addBufferOutput("statistics", "VSM demand, allocation, raster work and fallback counters")
            .buffer(kVirtualShadowMapStatisticsSize, sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeBoolSetting("enabled", "Celestial Shadows", true),
            runtimeIntSetting("lightIndex", "Celestial Light (0 Sun, 1 Moon)", 0, 0, 1),
            runtimeBoolSetting("cacheEnabled", "Cache Shadow Pages", false),
            runtimeBoolSetting("forceRedraw", "Force Shadow Redraw", false),
            runtimeIntSetting("pageSize", "Page Size (px, power of two)", 128, 16, 128),
            runtimeIntSetting("virtualPagesPerAxis", "Virtual Pages per Axis", 16, 1, 128),
            runtimeIntSetting("clipmapCount", "Clipmap Levels", 4, 1, 12),
            runtimeIntSetting("physicalPageCount", "Physical Page Budget", 256, 1, 4096),
            runtimeIntSetting("maxWorkItems", "Raster Work Budget", 65535, 1, 4 * 1024 * 1024),
            runtimeFloatSetting("firstClipmapExtent", "First Clipmap Half-width (m)", 4.0f, 0.01f, 10000.0f),
            runtimeFloatSetting("shadowBias", "Shadow Bias (m)", 0.002f, 0.0f, 1.0f),
            runtimeIntSetting("tessellationFactor", "Shadow Displacement Subdivisions", 4, 1, 8),
            runtimeIntSetting("tessellationSubdivisionDepth", "Shadow Displacement Split Depth", 0, 0, 2),
        };
    }

    std::vector<std::string> debugCheckpoints() const override
    {
        return {"VirtualShadowMapPages", "AfterPass"};
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (properties().value("filter", std::string("hard")) != "hard") {
            log = "VirtualShadowMapPass publishes the legacy hard-shadow interface; use VirtualShadowMapLightsPass for PCF/SMRT";
            return makeError(Error::Unsupported);
        }
        if (context.device == nullptr || !context.preparedScene || !context.preparedScene->snapshot ||
            !context.preparedScene->snapshot->pathTraceResources) {
            log = "VirtualShadowMapPass requires a device and prepared material resources";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().meshShader || !context.device->capabilities().bindlessDescriptorHeap) {
            log = "VirtualShadowMapPass requires mesh shaders and bindless descriptors";
            return makeError(Error::Unsupported);
        }
        device_ = context.device;
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto depth = context.inputTexture("depth");
        const auto metadata = context.inputBuffer("rasterInfo");
        const auto output = context.outputTexture("shadow");
        const auto debugOutput = context.outputTexture("debug");
        const auto parametersOutput = context.outputBuffer("parameters");
        const auto statisticsOutput = context.outputBuffer("statistics");
        const auto* scene = context.runtimeScene();
        const auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!device_ || !context.streamer() || !scene || !gpuScene ||
            !depth.valid() || !depth.view() || !metadata.valid() || !output.valid() ||
            !debugOutput.valid() || !parametersOutput.valid() || !statisticsOutput.valid() ||
            metadata.desc().size != sizeof(VisibilityBufferFrameInfo) ||
            metadata.desc().memoryLocation != MemoryLocation::HostUpload ||
            depth.desc().width != context.width() || depth.desc().height != context.height()) {
            return makeError(Error::InvalidArgument);
        }
        VisibilityBufferFrameInfo info;
        const void* mapped = metadata.buffer()->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&info, mapped, sizeof(info));
        metadata.buffer()->unmap();
        if (info.width != context.width() || info.height != context.height() ||
            info.frameIndex != context.frameIndex() || info.sceneIdentity != scene->resourceIdentity()) {
            spdlog::error("VirtualShadowMapPass requires current matching raster scene/view metadata");
            return makeError(Error::InvalidArgument);
        }

        ViewConstants view{};
        static_assert(offsetof(VisibilityBufferFrameInfo, width) == sizeof(ViewCameraConstants));
        std::memcpy(&view.current, info.eye, sizeof(ViewCameraConstants));
        // The raster ABI stores jitter in these two otherwise unused lanes.
        view.current.eye[3] = view.current.center[3] = 0.0f;
        view.previous = view.current;
        view.jitter[0] = view.jitter[2] = info.jitter[0];
        view.jitter[1] = view.jitter[3] = info.jitter[1];
        view.frame[0] = uint32_t(info.frameIndex);
        view.frame[2] = info.temporalJitter;
        if (const auto* sharedView = context.viewConstants()) { view.frame[3] = sharedView->frame[3]; }

        auto& commands = context.commandBuffer();
        auto profile = context.profileScope("VSM casters");
        const auto* stream = gpuScene->shadowGeometry({info.lightGridViewIndex, info.lightGridViewGeneration},
            info.frameIndex, info.sceneIdentity);
        if (scene->hasStreamGeometry() && stream == nullptr) {
            spdlog::error("VirtualShadowMapPass requires the visibility producer's complete shadow geometry publication");
            return makeError(Error::InvalidArgument);
        }
        std::string log;
        const auto& materialResources = *context.preparedScene()->snapshot->pathTraceResources;
        auto textures = materialResources.materialTextureSnapshot();
        if (!textures) { return makeError(Error::InvalidArgument); }
        auto result = commands.retainResource(std::make_shared<decltype(textures)>(textures));
        if (!result) { return result; }
        const auto logicalTextures = materialResources.logicalTextureIndices();
        std::vector<TextureView*> alphaTextures(scene->materials().size());
        std::vector<TextureView*> displacementTextures(scene->materials().size());
        for (size_t materialIndex = 0; materialIndex < scene->materials().size(); ++materialIndex) {
            const auto& material = scene->materials()[materialIndex];
            const auto logical = material.baseColorTexture.textureIndex;
            if (material.alphaMode == "MASK" && logical >= 0 && size_t(logical) < logicalTextures.size()) {
                const uint32_t slot = logicalTextures[logical];
                if (slot < textures->views.size()) { alphaTextures[materialIndex] = textures->views[slot].get(); }
            }
            const auto displacement = material.displacementTexture.textureIndex;
            if (material.displacementMagnitude != 0 && displacement >= 0 && size_t(displacement) < logicalTextures.size()) {
                const uint32_t slot = logicalTextures[displacement];
                if (slot < textures->views.size()) { displacementTextures[materialIndex] = textures->views[slot].get(); }
            }
        }
        const auto hasTexture = [](const auto* texture) { return texture != nullptr; };
        const bool textureDependent = std::any_of(alphaTextures.begin(), alphaTextures.end(), hasTexture) ||
            std::any_of(displacementTextures.begin(), displacementTextures.end(), hasTexture);
        const VirtualShadowMapCasterSettings casterSettings{
            .tessellationEnabled = (info.reserved & 1u) != 0,
            .tessellationFactor = context.properties().value("tessellationFactor", 4u),
            .subdivisionDepth = context.properties().value("tessellationSubdivisionDepth", 0u), .displacementTextures = displacementTextures,
            .texturePublicationRevision = textureDependent ? materialResources.materialTexturePublicationRevision() : 0};
        auto casters = prepareVirtualShadowMapCasters(*device_, commands, *scene,
            scene->hasStreamGeometry() ? stream : nullptr, log, casterSettings);
        if (!casters) {
            spdlog::error("VirtualShadowMapPass casters: {}", log);
            return makeError(casters.error());
        }
        const auto lightIndex = context.properties().value("lightIndex", 0u);
        if (lightIndex >= environment::kCelestialLightCount) {
            spdlog::error("VirtualShadowMapPass lightIndex must be 0 (Sun) or 1 (Moon)");
            return makeError(Error::InvalidArgument);
        }
        const auto light = buildCelestialLightRecords(resolveWorldEnvironment(scene, context.world()))[lightIndex];
        const auto settings = shadowSettings(context.properties());
        profile.next("VSM demand, raster and resolve");
        auto shadow = shadows_.record(*device_, commands, *depth.view(), view, light, *casters,
            scene->materials(), alphaTextures, settings, log);
        if (!shadow) {
            spdlog::error("VirtualShadowMapPass: {}", log);
            return makeError(shadow.error());
        }

        ScreenSpaceShadowParameters parameters{};
        parameters.view = view;
        for (size_t i = 0; i < 3; ++i) {
            parameters.light.directionType[i] = light.direction[i];
            parameters.light.colorIntensity[i] = light.irradiance[i];
        }
        parameters.light.colorIntensity[3] = 1.0f;
        parameters.trace[2] = settings.shadowBias;
        parameters.control[0] = settings.enabled && (light.flags & kGPUCelestialLightEnabled) != 0 &&
            (light.flags & kGPUCelestialLightCastsShadow) != 0;
        parameters.control[1] = 1;
        parameters.control[2] = kCelestialLightSourceTag | lightIndex;

        profile.next("Publish VSM outputs");
        auto copyBatch = context.streamer()->beginCopyBatch();
        if (!copyBatch) { return makeError(copyBatch.error()); }
        const StreamDataChunk chunk{&parameters, sizeof(parameters)};
        const auto uploaded = context.streamer()->streamBufferData({.dataChunks = {&chunk, 1},
            .placementAlignment = 16, .dstBuffer = parametersOutput.buffer(), .copyBatch = *copyBatch});
        if (!uploaded.valid()) {
            (void)context.streamer()->cancelCopyBatch(*copyBatch);
            return makeError(Error::OutOfMemory);
        }
        auto sourceStats = shadow->statistics->slice({0, kVirtualShadowMapStatisticsSize});
        auto destinationStats = statisticsOutput.buffer()->slice({0, kVirtualShadowMapStatisticsSize});
        if (!sourceStats || !destinationStats) {
            (void)context.streamer()->cancelCopyBatch(*copyBatch);
            return makeError(!sourceStats ? sourceStats.error() : destinationStats.error());
        }
        const std::array textureImports{
            RenderGraphTextureImport{"privateShadow", shadow->texture, shadow->shadow,
                ResourceState::ShaderRead, ResourceState::ShaderRead},
            RenderGraphTextureImport{"privateDebug", shadow->debugTexture, shadow->debug,
                ResourceState::ShaderRead, ResourceState::ShaderRead}};
        const std::array bufferImports{RenderGraphBufferImport{"privateStatistics", *sourceStats,
            RenderGraphResourceAccess::BufferShaderRead}};
        const std::array uses{
            RenderGraphStageUse{"privateShadow", RenderGraphResourceAccess::TextureTransferRead},
            RenderGraphStageUse{"privateDebug", RenderGraphResourceAccess::TextureTransferRead},
            RenderGraphStageUse{"privateStatistics", RenderGraphResourceAccess::BufferTransferRead},
            RenderGraphStageUse{"shadow", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"debug", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"parameters", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"statistics", RenderGraphResourceAccess::BufferTransferWrite}};
        const std::array stages{RenderGraphStage{"Publish VSM outputs", uses,
            [&](CommandBuffer& stageCommands) -> Result<> {
                auto copied = stageCommands.copyTexture({.source = shadow->texture, .destination = output.texture(),
                    .width = context.width(), .height = context.height(), .depth = 1});
                if (!copied) { return copied; }
                copied = stageCommands.copyTexture({.source = shadow->debugTexture, .destination = debugOutput.texture(),
                    .width = context.width(), .height = context.height(), .depth = 1});
                if (!copied) { return copied; }
                copied = stageCommands.copyBuffer(*sourceStats, *destinationStats);
                if (!copied) { return copied; }
                return context.streamer()->copyStreamedData(stageCommands, *copyBatch);
            }}};
        result = context.executeStages(stages, bufferImports, textureImports);
        if (!result) {
            (void)context.streamer()->cancelCopyBatch(*copyBatch);
            return result;
        }
        const std::array resources{
            DebugResourceBinding{.id = "atlas", .texture = shadow->atlas, .state = ResourceState::ShaderRead},
            DebugResourceBinding{.id = "pageTable", .buffer = shadow->pageTable, .state = ResourceState::ShaderRead},
            DebugResourceBinding{.id = "projections", .buffer = shadow->projections, .state = ResourceState::ShaderRead}};
        context.debugCheckpoint("VirtualShadowMapPages", resources, {{"frame", info.frameIndex},
            {"sceneIdentity", info.sceneIdentity}, {"streamed", casters->streamed},
            {"triangleCapacity", casters->triangleCapacity}, {"clipmapCount", settings.clipmapCount}, {"lightIndex", lightIndex},
            {"pageSize", settings.pageSize}, {"physicalPageCount", settings.physicalPageCount},
            {"cacheEnabled", settings.cacheEnabled}, {"forceRedraw", settings.forceRedraw}});
        return {};
    }

private:
    Device* device_ = nullptr;
    VirtualShadowMaps shadows_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVirtualShadowMapPass()
{
    return std::make_unique<VirtualShadowMapPass>();
}

} // namespace metallic::render::builtin_pass
