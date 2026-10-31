#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
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
    settings.receiverFootprint = boolProperty(&properties, "receiverFootprint", true);
    settings.cacheEnabled = boolProperty(&properties, "cacheEnabled", false);
    settings.cacheLayering = boolProperty(&properties, "cacheLayering", settings.cacheLayering);
    settings.shadowLOD = boolProperty(&properties, "shadowLOD", settings.shadowLOD);
    settings.shadowLODHierarchy = boolProperty(&properties, "shadowLODHierarchy", settings.shadowLODHierarchy);
    settings.shadowLODErrorTexels = properties.value("shadowLODErrorTexels", settings.shadowLODErrorTexels);
    settings.shadowLODSmartBias = boolProperty(&properties, "shadowLODSmartBias", settings.shadowLODSmartBias);
    settings.shadowLODSmartBiasRefine = boolProperty(&properties, "shadowLODSmartBiasRefine", settings.shadowLODSmartBiasRefine);
    settings.shadowLODScreenRayLength = properties.value("shadowLODScreenRayLength", settings.shadowLODScreenRayLength);
    settings.forceRedraw = boolProperty(&properties, "forceRedraw", false);
    settings.collectGPUWork = boolProperty(&properties, "benchmarkGPUWork", false);
    settings.profileBuildSubstages = boolProperty(&properties, "benchmarkBuildSubstages", false);
    settings.optimizeProjectionStatistics = boolProperty(&properties, "benchmarkProjectionStatistics", settings.optimizeProjectionStatistics);
    settings.directionalProjection = boolProperty(&properties, "benchmarkDirectionalProjection",
        boolProperty(&properties, "directionalProjection", settings.directionalProjection));
    settings.markRayPages = boolProperty(&properties, "benchmarkMarkRayPages",
        boolProperty(&properties, "markRayPages", settings.markRayPages));
    settings.compactBuild = boolProperty(&properties, "benchmarkCompactBuild", settings.compactBuild);
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
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo", "visibility"}};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Receiver depth from the matching visibility raster")
            .sampledRead().format = Format::D32Sfloat;
        reflection.addTextureInput("visibility", "Optional same-raster triangle IDs for refined LOD receiver bias")
            .sampledRead().setOptional().format = Format::R32Uint;
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
        reflection.addBufferOutput("cacheDiagnostics", "Cache invalidation reasons, caster changes and static/dynamic page work")
            .buffer(kVirtualShadowMapCacheDiagnosticSize, sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        reflection.addBufferOutput("lodStatistics", "Shadow LOD selection, page demand and selected cluster level histogram")
            .buffer(32 * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        if (properties().value("benchmarkGPUWork", false)) {
            reflection.addBufferOutput("workload", "VSM GPU workload counters; sixteen low/high uint pairs")
                .buffer(kVirtualShadowMapWorkloadSize, sizeof(uint32_t)).bufferLayout("u32").transferWrite();
            reflection.addBufferOutput("buildDiagnostics", "Cluster build counters and actual per-level page-scan histogram")
                .buffer(kVirtualShadowMapBuildDiagnosticSize, sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        }
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeBoolSetting("enabled", "Celestial Shadows", true),
            runtimeIntSetting("lightIndex", "Celestial Light (0 Sun, 1 Moon)", 0, 0, 1),
            runtimeBoolSetting("cacheEnabled", "Cache Shadow Pages", false),
            runtimeBoolSetting("cacheLayering", "Separate Static and Dynamic Depth", false),
            runtimeBoolSetting("shadowLOD", "Shadow-specific Geometry LOD", false),
            runtimeBoolSetting("shadowLODHierarchy", "Adaptive Shadow LOD Hierarchy", true),
            runtimeBoolSetting("shadowLODDepthGuard", "Preserve Shadow LOD Depth Agreement", true),
            runtimeBoolSetting("shadowLODSmartBias", "Shadow LOD Screen Depth Bias", false),
            runtimeBoolSetting("shadowLODSmartBiasRefine", "Refine Shadow LOD Receiver Bias", true),
            runtimeFloatSetting("shadowLODScreenRayLength", "Shadow LOD Screen Ray Length", .015f, 0, 1),
            runtimeFloatSetting("shadowLODErrorTexels", "Shadow LOD Error (texels)", 1, 0, 16),
            runtimeBoolSetting("forceRedraw", "Force Shadow Redraw", false),
            runtimeIntSetting("pageSize", "Page Size (px, power of two)", 128, 16, 128),
            runtimeIntSetting("virtualPagesPerAxis", "Virtual Pages per Axis", 16, 1, 128),
            runtimeIntSetting("clipmapCount", "Clipmap Levels", 4, 1, 12),
            runtimeIntSetting("physicalPageCount", "Physical Page Budget", 256, 1, 4096),
            runtimeIntSetting("maxWorkItems", "Raster Work Budget", 65535, 1, 4 * 1024 * 1024),
            runtimeFloatSetting("firstClipmapExtent", "First Clipmap Half-width (m)", 4.0f, 0.01f, 10000.0f),
            runtimeFloatSetting("shadowBias", "Shadow Bias (m)", 0.002f, 0.0f, 1.0f),
            runtimeBoolSetting("receiverFootprint", "Match Receiver Footprint", true),
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
        collectGPUWork_ = properties().value("benchmarkGPUWork", false);
        if (properties().value("benchmarkCPUProfile", false) && properties().value("benchmarkGPUProfile", false)) {
            log = "VSM CPU and detailed GPU profiling must run separately";
            return makeError(Error::InvalidArgument);
        }
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
        if (device_ != context.device) { casterCache_ = {}; }
        device_ = context.device;
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        if ((context.properties().value("benchmarkCPUProfile", false) &&
            context.properties().value("benchmarkGPUProfile", false)) ||
            context.properties().value("benchmarkGPUWork", false) != collectGPUWork_) {
            spdlog::error("VSM CPU and GPU profiling must run separately; changing benchmarkGPUWork requires graph recompilation");
            return makeError(Error::InvalidArgument);
        }
        CPUProfileRecorder cpuRecorder;
        auto* cpu = context.properties().value("benchmarkCPUProfile", false) ? &cpuRecorder : nullptr;
        cpuRecorder.threadDiagnostics = context.properties().value("benchmarkCPUThreadDiagnostics", false);
        const auto result = executeProfiled(context, cpu);
        if (cpu) { context.publishCpuProfile(cpuRecorder.sections); }
        return result;
    }

    Result<> executeProfiled(RenderGraphExecutionContext& context, CPUProfileRecorder* cpu)
    {
        CPUProfileScope cpuTotal(cpu, "VSM single light CPU");
        CPUProfileScope cpuPhase(cpu, "Validate inputs and prepare view");
        const auto depth = context.inputTexture("depth");
        const auto visibility = context.inputTexture("visibility");
        const auto metadata = context.inputBuffer("rasterInfo");
        const auto output = context.outputTexture("shadow");
        const auto debugOutput = context.outputTexture("debug");
        const auto parametersOutput = context.outputBuffer("parameters");
        const auto statisticsOutput = context.outputBuffer("statistics");
        const auto cacheDiagnosticsOutput = context.outputBuffer("cacheDiagnostics");
        const auto lodStatisticsOutput = context.outputBuffer("lodStatistics");
        const auto workloadOutput = context.outputBuffer("workload");
        const auto buildDiagnosticsOutput = context.outputBuffer("buildDiagnostics");
        auto* gpuProfiler = context.properties().value("benchmarkGPUProfile", false) ? &context : nullptr;
        const auto* scene = context.runtimeScene();
        const auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!device_ || !context.streamer() || !scene || !gpuScene ||
            !depth.valid() || !depth.view() || !metadata.valid() || !output.valid() ||
            !debugOutput.valid() || !parametersOutput.valid() || !statisticsOutput.valid() || !cacheDiagnosticsOutput.valid() || !lodStatisticsOutput.valid() ||
            (collectGPUWork_ && (!workloadOutput.valid() || !buildDiagnosticsOutput.valid())) ||
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
        cpuPhase.next("Caster textures and geometry");
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
        auto casters = casterCache_.prepare(*device_, commands, *scene,
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
        VirtualShadowMapReceiverResources receiver{};
        if (settings.shadowLOD && settings.shadowLODSmartBias && settings.shadowLODSmartBiasRefine &&
            visibility.valid() && (info.reserved & 1u) == 0) {
            if (visibility.desc().width != info.width || visibility.desc().height != info.height) {
                return makeError(Error::InvalidArgument);
            }
            const auto* receiverStream = gpuScene->visibilityStream({info.lightGridViewIndex, info.lightGridViewGeneration},
                info.frameIndex, info.sceneIdentity);
            if (receiverStream) { receiver = {visibility.view(), receiverStream}; }
        }
        std::array<uint32_t, 32> lodStatistics{};
        lodStatistics[0] = 2;
        lodStatistics[1] = settings.shadowLOD;
        lodStatistics[10] = casters->triangleCapacity;
        if (settings.shadowLOD && settings.enabled &&
            (light.flags & (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow)) ==
                (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow)) {
            VirtualShadowMapLight shadowLight;
            std::copy_n(light.direction, 3, shadowLight.direction.begin());
            shadowLight.angularRadius = light.angularRadius;
            const auto projections = VirtualShadowMaps::makeShadowLODProjections(casters->bounds, view, shadowLight, settings, log);
            if (!projections) {
                spdlog::error("VirtualShadowMapPass shadow LOD projections: {}", log);
                return makeError(projections.error());
            }
            lodStatistics[2] = uint32_t(projections->size());
            casters = prepareVirtualShadowMapShadowLOD(*device_, commands, *scene,
                scene->hasStreamGeometry() ? stream : nullptr, *casters, casterSettings,
                *projections, settings.shadowLODErrorTexels, log, settings.shadowLODHierarchy,
                boolProperty(&context.properties(), "shadowLODDepthGuard", true) ?
                    0.5f * settings.shadowBias : std::numeric_limits<float>::max());
            if (!casters) {
                spdlog::error("VirtualShadowMapPass shadow LOD: {}", log);
                return makeError(casters.error());
            }
        }
        const auto& lod = casters->shadowLOD;
        lodStatistics[3] = lod.testedGroups;
        lodStatistics[4] = lod.selectedClusters;
        lodStatistics[5] = casters->triangleCapacity;
        lodStatistics[6] = lod.requestedPages;
        lodStatistics[7] = lod.blockedPages;
        lodStatistics[8] = lod.fallbackCasters;
        lodStatistics[9] = lod.protectedCasters;
        lodStatistics[11] = lod.metricGroups;
        lodStatistics[12] = lod.hierarchyNodes;
        lodStatistics[13] = lod.prunedGroups;
        lodStatistics[14] = lod.metadataBuilds;
        lodStatistics[15] = lod.metadataCacheHits;
        for (size_t level = 0; level < lod.selectedLevels.size(); ++level) {
            lodStatistics[16 + std::min(level, size_t(15))] += lod.selectedLevels[level];
        }
        profile.next("VSM demand, raster and resolve");
        cpuPhase.next("Record shadow face");
        auto shadow = shadows_.record(*device_, commands, *depth.view(), view, light, *casters,
            scene->materials(), alphaTextures, settings, log, false, nullptr, cpu, gpuProfiler,
            receiver.stream ? &receiver : nullptr);
        if (!shadow) {
            spdlog::error("VirtualShadowMapPass: {}", log);
            return makeError(shadow.error());
        }

        ScreenSpaceShadowParameters parameters{};
        cpuPhase.next("Publication setup and commands");
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
        const StreamDataChunk lodChunk{lodStatistics.data(), sizeof(lodStatistics)};
        const auto lodUploaded = context.streamer()->streamBufferData({.dataChunks = {&lodChunk, 1},
            .placementAlignment = 16, .dstBuffer = lodStatisticsOutput.buffer(), .copyBatch = *copyBatch});
        if (!lodUploaded.valid()) {
            (void)context.streamer()->cancelCopyBatch(*copyBatch);
            return makeError(Error::OutOfMemory);
        }
        auto sourceStats = shadow->statistics->slice({0, kVirtualShadowMapStatisticsSize});
        auto destinationStats = statisticsOutput.buffer()->slice({0, kVirtualShadowMapStatisticsSize});
        auto sourceCacheDiagnostics = shadow->cacheDiagnostics->slice({0, kVirtualShadowMapCacheDiagnosticSize});
        auto destinationCacheDiagnostics = cacheDiagnosticsOutput.buffer()->slice({0, kVirtualShadowMapCacheDiagnosticSize});
        auto importedStats = shadow->statistics->slice({0, kVirtualShadowMapStatisticsSize +
            (collectGPUWork_ ? kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize : 0)});
        if (!sourceStats || !destinationStats || !importedStats || !sourceCacheDiagnostics || !destinationCacheDiagnostics) {
            (void)context.streamer()->cancelCopyBatch(*copyBatch);
            return makeError(Error::InvalidArgument);
        }
        BufferSlice sourceWorkload, destinationWorkload;
        BufferSlice sourceBuildDiagnostics, destinationBuildDiagnostics;
        if (collectGPUWork_) {
            auto source = shadow->statistics->slice({kVirtualShadowMapStatisticsSize, kVirtualShadowMapWorkloadSize});
            auto destination = workloadOutput.buffer()->slice({0, kVirtualShadowMapWorkloadSize});
            if (!source || !destination) {
                (void)context.streamer()->cancelCopyBatch(*copyBatch);
                return makeError(Error::InvalidArgument);
            }
            sourceWorkload = *source; destinationWorkload = *destination;
            auto buildSource = shadow->statistics->slice({kVirtualShadowMapStatisticsSize + kVirtualShadowMapWorkloadSize,
                kVirtualShadowMapBuildDiagnosticSize});
            auto buildDestination = buildDiagnosticsOutput.buffer()->slice({0, kVirtualShadowMapBuildDiagnosticSize});
            if (!buildSource || !buildDestination) {
                (void)context.streamer()->cancelCopyBatch(*copyBatch);
                return makeError(Error::InvalidArgument);
            }
            sourceBuildDiagnostics = *buildSource; destinationBuildDiagnostics = *buildDestination;
        }
        const std::array textureImports{
            RenderGraphTextureImport{"privateShadow", shadow->texture, shadow->shadow,
                ResourceState::ShaderRead, ResourceState::ShaderRead},
            RenderGraphTextureImport{"privateDebug", shadow->debugTexture, shadow->debug,
                ResourceState::ShaderRead, ResourceState::ShaderRead}};
        const std::array bufferImports{RenderGraphBufferImport{"privateStatistics", *importedStats,
            RenderGraphResourceAccess::BufferStorageReadWrite},
            RenderGraphBufferImport{"privateCacheDiagnostics", *sourceCacheDiagnostics,
                RenderGraphResourceAccess::BufferStorageReadWrite}};
        std::vector<RenderGraphStageUse> uses{
            RenderGraphStageUse{"privateShadow", RenderGraphResourceAccess::TextureTransferRead},
            RenderGraphStageUse{"privateDebug", RenderGraphResourceAccess::TextureTransferRead},
            RenderGraphStageUse{"privateStatistics", RenderGraphResourceAccess::BufferTransferRead},
            RenderGraphStageUse{"privateCacheDiagnostics", RenderGraphResourceAccess::BufferTransferRead},
            RenderGraphStageUse{"shadow", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"debug", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"parameters", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"statistics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"cacheDiagnostics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"lodStatistics", RenderGraphResourceAccess::BufferTransferWrite}};
        if (collectGPUWork_) {
            uses.push_back({"workload", RenderGraphResourceAccess::BufferTransferWrite});
            uses.push_back({"buildDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
        }
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
                copied = stageCommands.copyBuffer(*sourceCacheDiagnostics, *destinationCacheDiagnostics);
                if (!copied) { return copied; }
                if (collectGPUWork_) {
                    copied = stageCommands.copyBuffer(sourceWorkload, destinationWorkload);
                    if (!copied) { return copied; }
                    copied = stageCommands.copyBuffer(sourceBuildDiagnostics, destinationBuildDiagnostics);
                    if (!copied) { return copied; }
                }
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
            {"cacheEnabled", settings.cacheEnabled}, {"cacheLayering", settings.cacheLayering}, {"forceRedraw", settings.forceRedraw}});
        return {};
    }

private:
    bool collectGPUWork_ = false;
    Device* device_ = nullptr;
    VirtualShadowMapCasterCache casterCache_;
    VirtualShadowMaps shadows_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVirtualShadowMapPass()
{
    return std::make_unique<VirtualShadowMapPass>();
}

} // namespace metallic::render::builtin_pass
