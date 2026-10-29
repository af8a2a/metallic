#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLights.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace metallic::render::builtin_pass {
namespace {

struct ManagedLight
{
    uint64_t stableId = 0;
    VirtualShadowMapLight light;
    VirtualShadowMapLightEntry entry{};
    std::array<uint32_t, 6> faceBudgets{};
};

uint64_t temporalContentKey(const VirtualShadowMapCasterResources& casters, const VisibilityBufferFrameInfo& raster)
{
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&hash](const auto& value) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) { hash = (hash ^ bytes[i]) * 1099511628211ull; }
    };
    append(casters.sceneIdentity); append(casters.structuralRevision); append(casters.dependencyKey);
    append(raster.sceneIdentity); append(raster.rasterSettingsRevision);
    for (const auto& caster : casters.states) {
        append(caster.id); append(caster.contentKey); append(caster.bounds.valid);
        append(caster.bounds.min.x); append(caster.bounds.min.y); append(caster.bounds.min.z);
        append(caster.bounds.max.x); append(caster.bounds.max.y); append(caster.bounds.max.z);
    }
    return hash;
}

class VirtualShadowMapLightsPass final : public UnsafePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
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
        const uint32_t capacity = properties().value("lightCapacity", 8u);
        const bool compact = properties().value("compactOutputs", false);
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Receiver depth from the matching visibility raster")
            .sampledRead().format = Format::D32Sfloat;
        auto& info = reflection.addBufferInput("rasterInfo", "Matching raster camera and scene identity")
            .buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo)).shaderRead();
        info.memoryLocation = MemoryLocation::HostUpload;
        auto& shadows = reflection.addTextureOutput("shadows", "Linear visibility; one compact layer or six face layers per light")
            .transferWrite().transient(RenderGraphInitialization::FullOverwrite);
        shadows.format = Format::R8Unorm;
        shadows.layerCount = compact ? capacity : capacity * 6;
        // Temporal stabilization is a runtime toggle and reads/writes the
        // published layer after every face has finished projection.
        shadows.stageAccess(RenderGraphResourceAccess::TextureStorageReadWrite, RenderGraphPassKind::Compute);
        auto& debug = reflection.addTextureOutput("debug", "Per-face page selection; magenta marks unavailable layers")
            .transferWrite().transient(RenderGraphInitialization::FullOverwrite);
        debug.format = Format::RGBA8Unorm;
        debug.layerCount = compact ? capacity : capacity * 6;
        if (compact) { debug.stageAccess(RenderGraphResourceAccess::TextureStorageWrite, RenderGraphPassKind::Compute); }
        reflection.addBufferOutput("lights", "Current source-to-layer table and explicit light admission")
            .buffer(sizeof(VirtualShadowMapLightsHeader) + uint64_t(capacity) * sizeof(VirtualShadowMapLightEntry), 0)
            .transferWrite();
        reflection.addBufferOutput("statistics", "Six 32-uint face blocks per light entry")
            .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapFaceStatisticsSize, sizeof(uint32_t))
            .bufferLayout("u32").transferWrite();
        if (properties().value("benchmarkGPUWork", false)) {
            reflection.addBufferOutput("workload", "Six GPU workload blocks per light entry; sixteen low/high uint pairs per face")
                .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapWorkloadSize, sizeof(uint32_t))
                .bufferLayout("u32").transferWrite();
            reflection.addBufferOutput("buildDiagnostics", "Six cluster build counter and page-scan histogram blocks per light entry")
                .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapBuildDiagnosticSize, sizeof(uint32_t))
                .bufferLayout("u32").transferWrite();
        }
        reflection.addBufferOutput("memory", "Logical owned bytes: atlas, private output, buffers, public output, faces, atlas pages, temporal history")
            .buffer(8 * sizeof(uint64_t), sizeof(uint64_t)).transferWrite();
        return reflection;
    }
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto capacity = runtimeIntSetting("lightCapacity", "Light Table Capacity", 8, 2, 64);
        capacity.rebuildGraph = true;
        auto compact = runtimeBoolSetting("compactOutputs", "One Output Layer per Light", false);
        compact.rebuildGraph = true;
        return {capacity, compact,
            runtimeBoolSetting("enabled", "Virtual Shadows", true),
            runtimeBoolSetting("cacheEnabled", "Cache Shadow Pages", true),
            runtimeBoolSetting("forceRedraw", "Force Shadow Redraw", false),
            runtimeIntSetting("physicalPageBudget", "Total Physical Page Budget", 256, 0, 262144),
            runtimeIntSetting("pageSize", "Page Size", 128, 16, 128),
            runtimeIntSetting("virtualPagesPerAxis", "Virtual Pages per Axis", 16, 1, 128),
            runtimeIntSetting("clipmapCount", "Clipmap / Local Mip Levels", 4, 1, 12),
            runtimeIntSetting("maxWorkItems", "Per-face Raster Work Budget", 65535, 1, 4 * 1024 * 1024),
            runtimeFloatSetting("firstClipmapExtent", "First Directional Half-width (m)", 4, 0.01f, 10000),
            runtimeFloatSetting("shadowBias", "Shadow Bias (m)", 0.002f, 0, 1),
            runtimeEnumSetting("filter", "Shadow Filter", "hard", {{"Hard", "hard"}, {"PCF", "pcf"}, {"SMRT", "smrt"}}),
            runtimeFloatSetting("pcfRadius", "PCF Radius (texels)", 1, 0, 8),
            runtimeIntSetting("smrtRayCount", "SMRT Rays", 8, 1, 32),
            runtimeIntSetting("smrtSamplesPerRay", "SMRT Samples per Ray", 8, 1, 32),
            runtimeFloatSetting("smrtMaxRayLength", "SMRT Soft Search Distance (m)", 10, 0.001f, 10000),
            runtimeFloatSetting("smrtRayLengthScale", "SMRT View Distance Scale (0 = fixed)", 0, 0, 1000),
            runtimeFloatSetting("smrtMaxRayAngle", "SMRT Local Max Angle (rad)", 0.03f, 0.001f, 0.35f),
            runtimeBoolSetting("smrtTemporalEnabled", "SMRT Temporal Stabilization", false),
            runtimeIntSetting("smrtTemporalMaxFrames", "SMRT History Frames", 32, 1, 64),
            runtimeBoolSetting("receiverFootprint", "Match Receiver Footprint", true),
            runtimeIntSetting("tessellationFactor", "Shadow Displacement Subdivisions", 4, 1, 8),
            runtimeIntSetting("tessellationSubdivisionDepth", "Shadow Displacement Split Depth", 0, 0, 2),
            runtimeFloatSetting("nearPlane", "Local Near Plane (m)", 0.001f, 0.00001f, 1)};
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        capacity_ = properties().value("lightCapacity", 8u);
        compact_ = properties().value("compactOutputs", false);
        collectGPUWork_ = properties().value("benchmarkGPUWork", false);
        if (properties().value("benchmarkCPUProfile", false) && properties().value("benchmarkGPUProfile", false)) {
            log = "VSM CPU and detailed GPU profiling must run separately";
            return makeError(Error::InvalidArgument);
        }
        if (capacity_ < 2 || capacity_ > 64 || !context.device || !context.preparedScene ||
            !context.preparedScene->snapshot || !context.preparedScene->snapshot->pathTraceResources) {
            log = "VirtualShadowMapLightsPass requires material resources and lightCapacity in [2,64]";
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().meshShader || !context.device->capabilities().bindlessDescriptorHeap) {
            log = "VirtualShadowMapLightsPass requires mesh shaders and bindless descriptors";
            return makeError(Error::Unsupported);
        }
        if (device_ != context.device) { casterCache_ = {}; histories_.clear(); }
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
        CPUProfileScope cpuTotal(cpu, "VSM lights CPU");
        CPUProfileScope cpuPhase(cpu, "Validate inputs and resolve settings");
        const auto depth = context.inputTexture("depth");
        const auto metadata = context.inputBuffer("rasterInfo");
        const auto output = context.outputTexture("shadows");
        const auto debug = context.outputTexture("debug");
        const auto table = context.outputBuffer("lights");
        const auto statistics = context.outputBuffer("statistics");
        const auto workload = context.outputBuffer("workload");
        const auto buildDiagnostics = context.outputBuffer("buildDiagnostics");
        auto* gpuProfiler = context.properties().value("benchmarkGPUProfile", false) ? &context : nullptr;
        const auto memory = context.outputBuffer("memory");
        const auto* scene = context.runtimeScene();
        const auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!scene || !gpuScene || !context.streamer() || !depth.valid() || !depth.view() ||
            !metadata.valid() || metadata.desc().size != sizeof(VisibilityBufferFrameInfo) ||
            metadata.desc().memoryLocation != MemoryLocation::HostUpload ||
            !output.valid() || !debug.valid() || !table.valid() || !statistics.valid() || !memory.valid() ||
            (collectGPUWork_ && (!workload.valid() || !buildDiagnostics.valid()))) {
            return makeError(Error::InvalidArgument);
        }
        VisibilityBufferFrameInfo info;
        const void* mapped = metadata.buffer()->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&info, mapped, sizeof(info));
        metadata.buffer()->unmap();
        if (info.width != context.width() || info.height != context.height() ||
            info.frameIndex != context.frameIndex() || info.sceneIdentity != scene->resourceIdentity()) {
            spdlog::error("VirtualShadowMapLightsPass requires current matching raster metadata");
            return makeError(Error::InvalidArgument);
        }
        const auto sources = gpuScene->scene().lights();
        if (sources.size() + 2 > capacity_) {
            spdlog::error("VirtualShadowMapLightsPass lightCapacity {} cannot hold {} source slots",
                capacity_, sources.size() + 2);
            return makeError(Error::Unsupported);
        }
        VirtualShadowMapSettings settings;
        const auto& properties = context.properties();
        settings.pageSize = properties.value("pageSize", settings.pageSize);
        settings.virtualPagesPerAxis = properties.value("virtualPagesPerAxis", settings.virtualPagesPerAxis);
        settings.clipmapCount = properties.value("clipmapCount", settings.clipmapCount);
        settings.maxWorkItems = properties.value("maxWorkItems", settings.maxWorkItems);
        settings.firstClipmapExtent = properties.value("firstClipmapExtent", settings.firstClipmapExtent);
        settings.shadowBias = properties.value("shadowBias", settings.shadowBias);
        const auto filter = properties.value("filter", std::string("hard"));
        if (filter != "hard" && filter != "pcf" && filter != "smrt") { return makeError(Error::InvalidArgument); }
        settings.filter = filter == "smrt" ? VirtualShadowMapFilter::SMRT :
            filter == "pcf" ? VirtualShadowMapFilter::PCF : VirtualShadowMapFilter::Hard;
        settings.pcfRadius = properties.value("pcfRadius", settings.pcfRadius);
        settings.smrtRayCount = properties.value("smrtRayCount", settings.smrtRayCount);
        settings.smrtSamplesPerRay = properties.value("smrtSamplesPerRay", settings.smrtSamplesPerRay);
        settings.smrtMaxRayLength = properties.value("smrtMaxRayLength", settings.smrtMaxRayLength);
        settings.smrtRayLengthScale = properties.value("smrtRayLengthScale", settings.smrtRayLengthScale);
        settings.smrtMaxRayAngle = properties.value("smrtMaxRayAngle", settings.smrtMaxRayAngle);
        settings.smrtTemporalSampling = settings.filter == VirtualShadowMapFilter::SMRT &&
            boolProperty(&properties, "smrtTemporalEnabled", false);
        const uint32_t historyFrames = properties.value("smrtTemporalMaxFrames", 32u);
        if (historyFrames == 0 || historyFrames > 64) { return makeError(Error::InvalidArgument); }
        settings.receiverFootprint = boolProperty(&properties, "receiverFootprint", true);
        settings.cacheEnabled = boolProperty(&properties, "cacheEnabled", true);
        settings.forceRedraw = boolProperty(&properties, "forceRedraw", false);
        settings.collectGPUWork = collectGPUWork_;
        settings.optimizeMark = !boolProperty(&properties, "benchmarkLegacyMark", !settings.optimizeMark);
        settings.clusterCulling = !boolProperty(&properties, "benchmarkDisableClusterCull", false);
        settings.enabled = boolProperty(&properties, "enabled", true) && !boolProperty(&properties, "debugDisableShadows", false);
        const uint32_t budget = properties.value("physicalPageBudget", 256u);
        const float nearPlane = properties.value("nearPlane", 0.001f);
        if (budget > 262144 || !std::isfinite(nearPlane) || nearPlane <= 0) { return makeError(Error::InvalidArgument); }
        const auto celestials = buildCelestialLightRecords(resolveWorldEnvironment(scene, context.world()));
        cpuPhase.next("Light admission and cache pruning");
        const auto lighting = resolveSceneLighting(scene, context.world());
        std::vector<ManagedLight> lights(sources.size() + 2);
        for (uint32_t index = 0; index < lights.size(); ++index) {
            auto& item = lights[index];
            item.entry.baseLayer = compact_ ? index : index * 6;
            item.light.nearPlane = nearPlane;
            if (index < 2) {
                const auto& light = celestials[index];
                item.stableId = (uint64_t(1) << 62) | index;
                item.entry.source = kCelestialLightSourceTag | index;
                std::copy_n(light.direction, 3, item.light.direction.begin());
                item.light.angularRadius = light.angularRadius;
                item.light.enabled = settings.enabled && (light.flags & kGPUCelestialLightEnabled) &&
                    (light.flags & kGPUCelestialLightCastsShadow);
                item.entry.faceCount = 1;
            } else {
                const auto& source = sources[index - 2].source;
                const auto& light = source.gpu;
                item.stableId = source.stableId;
                item.entry.source = index - 2;
                bool point = light.directionType[3] < 1.5f;
                if (!source.enabled) {
                    if (source.sourceVirtualLightIndex >= 0) {
                        point = lighting.lights[source.sourceVirtualLightIndex].properties.type == "point";
                    } else {
                        point = scene->lights()[source.sourceRenderLightIndex].type == "point";
                    }
                }
                item.entry.kind = point ? 2 : 1;
                item.entry.faceCount = point ? 6 : 1;
                item.light.type = point ? VirtualShadowMapLightType::PointFace : VirtualShadowMapLightType::Spot;
                std::copy_n(light.positionRange, 3, item.light.position.begin());
                std::copy_n(light.directionType, 3, item.light.direction.begin());
                std::copy_n(light.positionRange, 4, item.entry.positionRange);
                item.light.range = light.positionRange[3];
                item.light.sourceRadius = light.spot[2];
                item.light.outerConeAngle = std::acos(std::clamp(light.spot[1], -1.0f, 1.0f));
                item.light.enabled = settings.enabled && source.enabled;
            }
            item.entry.stableId[0] = uint32_t(item.stableId);
            item.entry.stableId[1] = uint32_t(item.stableId >> 32);
        }
        std::vector<uint32_t> order(lights.size());
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return lights[a].stableId < lights[b].stableId; });
        VirtualShadowMapLightsHeader header{};
        header.version = kVirtualShadowMapLightsVersion;
        header.count = uint32_t(lights.size());
        header.capacity = capacity_;
        header.reserved[0] = compact_ ? 1u : 0u;
        header.physicalPageBudget = budget;
        header.sceneIdentity[0] = uint32_t(info.sceneIdentity);
        header.sceneIdentity[1] = uint32_t(info.sceneIdentity >> 32);
        header.frameIndex[0] = uint32_t(info.frameIndex);
        header.frameIndex[1] = uint32_t(info.frameIndex >> 32);
        header.lightGeneration = gpuScene->scene().drawSet().lightGeneration;
        header.sourceSlotCount = uint32_t(sources.size());
        uint32_t remaining = budget, admittedFaces = 0;
        for (uint32_t index : order) {
            auto& item = lights[index];
            header.totalFaces += item.entry.faceCount;
            if (!item.light.enabled) { ++header.disabledCount; continue; }
            if (remaining < item.entry.faceCount) {
                item.entry.status = kVirtualShadowMapLightBudgetDropped;
                ++header.budgetDroppedCount;
                continue;
            }
            item.entry.status = kVirtualShadowMapLightAdmitted;
            ++header.admittedCount;
            remaining -= item.entry.faceCount;
            admittedFaces += item.entry.faceCount;
        }
        const uint32_t extra = admittedFaces ? remaining / admittedFaces : 0;
        uint32_t remainder = admittedFaces ? remaining % admittedFaces : 0;
        for (uint32_t index : order) {
            auto& item = lights[index];
            if (item.entry.status != kVirtualShadowMapLightAdmitted) { continue; }
            for (uint32_t face = 0; face < item.entry.faceCount; ++face) {
                item.faceBudgets[face] = std::min(4096u, 1 + extra + (remainder ? (--remainder, 1u) : 0u));
                item.entry.pageBudget += item.faceBudgets[face];
            }
        }
        if (sceneIdentity_ != info.sceneIdentity) {
            caches_.clear();
            histories_.clear();
            casterCache_ = {};
            sceneIdentity_ = info.sceneIdentity;
        }
        std::unordered_set<uint64_t> identities;
        for (const auto& item : lights) {
            if (item.entry.status == kVirtualShadowMapLightAdmitted) { identities.insert(item.stableId); }
        }
        // Admission owns persistent pool memory. Disabled/rejected lights must
        // release their old pools when the global budget shrinks.
        std::erase_if(caches_, [&](const auto& cache) { return !identities.contains(cache.first); });
        std::unordered_set<uint64_t> temporalIdentities;
        if (settings.smrtTemporalSampling) {
            for (const auto& item : lights) {
                const float size = item.light.type == VirtualShadowMapLightType::Directional ? item.light.angularRadius : item.light.sourceRadius;
                if (item.entry.status == kVirtualShadowMapLightAdmitted && size > 0.0f) { temporalIdentities.insert(item.stableId); }
            }
        }
        std::erase_if(histories_, [&](const auto& history) { return !temporalIdentities.contains(history.first); });
        cpuTotal.work("sourceSlots", sources.size());
        cpuTotal.work("admittedLights", header.admittedCount);
        cpuTotal.work("admittedFaces", admittedFaces);
        cpuPhase.next("Light table and statistics upload");

        struct FacePublication
        {
            VirtualShadowMapResult shadow;
            BufferSlice sourceStats;
            BufferSlice destinationStats;
            BufferSlice importedStats;
            BufferSlice sourceWorkload;
            BufferSlice destinationWorkload;
            BufferSlice sourceBuildDiagnostics;
            BufferSlice destinationBuildDiagnostics;
            uint32_t layer;
            std::array<std::string, 3> names;
        };
        std::vector<FacePublication> publications;
        publications.reserve(admittedFaces);
        std::vector<std::byte> tableData(table.desc().size);
        std::memcpy(tableData.data(), &header, sizeof(header));
        for (size_t index = 0; index < lights.size(); ++index) {
            std::memcpy(tableData.data() + sizeof(header) + index * sizeof(VirtualShadowMapLightEntry),
                &lights[index].entry, sizeof(VirtualShadowMapLightEntry));
        }
        std::vector<uint32_t> zeroStatistics(statistics.desc().size / sizeof(uint32_t), 0);
        std::vector<uint32_t> zeroWorkload(collectGPUWork_ ? workload.desc().size / sizeof(uint32_t) : 0, 0);
        std::vector<uint32_t> zeroBuildDiagnostics(collectGPUWork_ ? buildDiagnostics.desc().size / sizeof(uint32_t) : 0, 0);
        auto batch = context.streamer()->beginCopyBatch();
        if (!batch) { return makeError(batch.error()); }
        std::vector<StreamDataChunk> chunks{{tableData.data(), tableData.size()},
            {zeroStatistics.data(), zeroStatistics.size() * sizeof(uint32_t)}};
        std::vector<Buffer*> destinations{table.buffer(), statistics.buffer()};
        if (collectGPUWork_) {
            chunks.push_back({zeroWorkload.data(), zeroWorkload.size() * sizeof(uint32_t)});
            destinations.push_back(workload.buffer());
            chunks.push_back({zeroBuildDiagnostics.data(), zeroBuildDiagnostics.size() * sizeof(uint32_t)});
            destinations.push_back(buildDiagnostics.buffer());
        }
        for (size_t index = 0; index < chunks.size(); ++index) {
            const auto upload = context.streamer()->streamBufferData({.dataChunks = {&chunks[index], 1},
                .placementAlignment = 16, .dstBuffer = destinations[index], .copyBatch = *batch});
            if (!upload.valid()) { (void)context.streamer()->cancelCopyBatch(*batch); return makeError(Error::OutOfMemory); }
        }
        std::vector<RenderGraphStageUse> initializeUses{RenderGraphStageUse{"shadows", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"debug", RenderGraphResourceAccess::TextureTransferWrite},
            RenderGraphStageUse{"lights", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"statistics", RenderGraphResourceAccess::BufferTransferWrite}};
        if (collectGPUWork_) {
            initializeUses.push_back({"workload", RenderGraphResourceAccess::BufferTransferWrite});
            initializeUses.push_back({"buildDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        struct ProjectionGroup
        {
            std::array<VirtualShadowMaps*, 6> faces{};
            std::array<VirtualShadowMapResult, 6> results{};
            uint32_t faceCount = 0;
            bool filteredPoint = false;
        };
        std::vector<ProjectionGroup> projections;
        ViewConstants view{};
        uint64_t contentKey = 0;
        auto recordFaces = [&](CommandBuffer&) -> Result<> {
            cpuPhase.next("Caster textures and geometry");
            auto profile = context.profileScope("VSM caster preparation");
            if (admittedFaces != 0) {
                static_assert(offsetof(VisibilityBufferFrameInfo, width) == sizeof(ViewCameraConstants));
                std::memcpy(&view.current, info.eye, sizeof(ViewCameraConstants));
                view.current.eye[3] = view.current.center[3] = 0;
                view.previous = view.current;
                view.jitter[0] = view.jitter[2] = info.jitter[0];
                view.jitter[1] = view.jitter[3] = info.jitter[1];
                view.frame[0] = uint32_t(info.frameIndex);
                view.frame[2] = info.temporalJitter;
                if (const auto* shared = context.viewConstants()) { view.frame[3] = shared->frame[3]; }
                auto& commands = context.commandBuffer();
                const auto* stream = gpuScene->shadowGeometry({info.lightGridViewIndex, info.lightGridViewGeneration},
                    info.frameIndex, info.sceneIdentity);
                if (scene->hasStreamGeometry() && !stream) { return makeError(Error::InvalidArgument); }
                std::string log;
                const auto& materialResources = *context.preparedScene()->snapshot->pathTraceResources;
                auto textures = materialResources.materialTextureSnapshot();
                if (!textures) { return makeError(Error::InvalidArgument); }
                auto result = commands.retainResource(std::make_shared<decltype(textures)>(textures));
                if (!result) { return result; }
                const auto logicalTextures = materialResources.logicalTextureIndices();
                std::vector<TextureView*> alphaTextures(scene->materials().size());
                std::vector<TextureView*> displacementTextures(scene->materials().size());
                for (size_t index = 0; index < scene->materials().size(); ++index) {
                    const auto& material = scene->materials()[index];
                    const auto logical = material.baseColorTexture.textureIndex;
                    if (material.alphaMode == "MASK" && logical >= 0 && size_t(logical) < logicalTextures.size()) {
                        const uint32_t slot = logicalTextures[logical];
                        if (slot < textures->views.size()) { alphaTextures[index] = textures->views[slot].get(); }
                    }
                    const auto displacement = material.displacementTexture.textureIndex;
                    if (material.displacementMagnitude != 0 && displacement >= 0 && size_t(displacement) < logicalTextures.size()) {
                        const uint32_t slot = logicalTextures[displacement];
                        if (slot < textures->views.size()) { displacementTextures[index] = textures->views[slot].get(); }
                    }
                }
                const auto hasTexture = [](const auto* texture) { return texture != nullptr; };
                const bool textureDependent = std::any_of(alphaTextures.begin(), alphaTextures.end(), hasTexture) ||
                    std::any_of(displacementTextures.begin(), displacementTextures.end(), hasTexture);
                const VirtualShadowMapCasterSettings casterSettings{
                    .tessellationEnabled = (info.reserved & 1u) != 0,
                    .tessellationFactor = properties.value("tessellationFactor", 4u),
                    .subdivisionDepth = properties.value("tessellationSubdivisionDepth", 0u), .displacementTextures = displacementTextures,
                    .texturePublicationRevision = textureDependent ? materialResources.materialTexturePublicationRevision() : 0};
                auto casters = casterCache_.prepare(*device_, commands, *scene,
                    scene->hasStreamGeometry() ? stream : nullptr, log, casterSettings);
                if (!casters) { spdlog::error("VirtualShadowMapLightsPass casters: {}", log); return makeError(casters.error()); }
                if (!temporalIdentities.empty()) { contentKey = temporalContentKey(*casters, info); }
                profile.next("VSM page demand, cache and raster");
                cpuPhase.next("Record all faces");
                for (uint32_t index : order) {
                    auto& item = lights[index];
                    if (item.entry.status != kVirtualShadowMapLightAdmitted) { continue; }
                    auto& faces = caches_[item.stableId];
                    for (uint32_t face = item.entry.faceCount; face < faces.size(); ++face) { faces[face].reset(); }
                    const bool filteredPoint = item.entry.faceCount == 6 && settings.filter != VirtualShadowMapFilter::Hard;
                    const bool deferredProjection = compact_ || filteredPoint;
                    std::array<VirtualShadowMapResult, 6> pointResults;
                    for (uint32_t face = 0; face < item.entry.faceCount; ++face) {
                        if (!faces[face]) { faces[face] = std::make_unique<VirtualShadowMaps>(); }
                        item.light.face = face;
                        settings.physicalPageCount = item.faceBudgets[face];
                        const VirtualShadowMapProjectionTarget target{output.texture(), debug.texture(), item.entry.baseLayer, item.entry.faceCount == 6};
                        auto shadow = faces[face]->record(*device_, commands, *depth.view(), view, item.light, *casters,
                            scene->materials(), alphaTextures, settings, log, deferredProjection, compact_ ? &target : nullptr, cpu, gpuProfiler);
                        if (!shadow) { spdlog::error("VirtualShadowMapLightsPass: {}", log); return makeError(shadow.error()); }
                        pointResults[face] = *shadow;
                        const uint32_t layer = index * 6 + face;
                        auto sourceStats = shadow->statistics->slice({0, kVirtualShadowMapFaceStatisticsSize});
                        auto destinationStats = statistics.buffer()->slice({uint64_t(layer) * kVirtualShadowMapFaceStatisticsSize,
                            kVirtualShadowMapFaceStatisticsSize});
                        auto importedStats = shadow->statistics->slice({0, kVirtualShadowMapFaceStatisticsSize +
                            (collectGPUWork_ ? kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize : 0)});
                        if (!sourceStats || !destinationStats || !importedStats) { return makeError(Error::InvalidArgument); }
                        BufferSlice sourceWorkload, destinationWorkload;
                        BufferSlice sourceBuildDiagnostics, destinationBuildDiagnostics;
                        if (collectGPUWork_) {
                            auto source = shadow->statistics->slice({kVirtualShadowMapFaceStatisticsSize, kVirtualShadowMapWorkloadSize});
                            auto destination = workload.buffer()->slice({uint64_t(layer) * kVirtualShadowMapWorkloadSize,
                                kVirtualShadowMapWorkloadSize});
                            if (!source || !destination) { return makeError(Error::InvalidArgument); }
                            sourceWorkload = *source; destinationWorkload = *destination;
                            auto buildSource = shadow->statistics->slice({kVirtualShadowMapFaceStatisticsSize + kVirtualShadowMapWorkloadSize,
                                kVirtualShadowMapBuildDiagnosticSize});
                            auto buildDestination = buildDiagnostics.buffer()->slice({uint64_t(layer) * kVirtualShadowMapBuildDiagnosticSize,
                                kVirtualShadowMapBuildDiagnosticSize});
                            if (!buildSource || !buildDestination) { return makeError(Error::InvalidArgument); }
                            sourceBuildDiagnostics = *buildSource; destinationBuildDiagnostics = *buildDestination;
                        }
                        const auto suffix = std::to_string(layer);
                        publications.push_back({*shadow, *sourceStats, *destinationStats, *importedStats,
                            sourceWorkload, destinationWorkload, sourceBuildDiagnostics, destinationBuildDiagnostics, layer,
                            {"faceShadow" + suffix, "faceDebug" + suffix, "faceStatistics" + suffix}});
                    }
                    if (compact_) {
                        ProjectionGroup group;
                        group.results = pointResults;
                        group.faceCount = item.entry.faceCount;
                        group.filteredPoint = filteredPoint;
                        for (uint32_t face = 0; face < group.faceCount; ++face) { group.faces[face] = faces[face].get(); }
                        projections.push_back(group);
                    } else if (deferredProjection) {
                        for (auto& face : faces) {
                            if (auto projected = face->project(commands, pointResults, log, cpu, gpuProfiler); !projected) {
                                spdlog::error("VirtualShadowMapLightsPass projection: {}", log);
                                return projected;
                            }
                        }
                    }
                }
            }
            return {};
        };
        if (auto r = recordFaces(context.commandBuffer()); !r) { (void)context.streamer()->cancelCopyBatch(*batch); return r; }
        cpuPhase.next("Publication stage setup");
        // Build views only after the owning names have reached their final
        // addresses. Every face's private outputs remain ShaderRead until this
        // single graph-stage sequence transitions them for publication.
        std::vector<RenderGraphTextureImport> textureImports;
        std::vector<RenderGraphBufferImport> bufferImports;
        std::vector<RenderGraphStageUse> projectionUses;
        if (compact_) {
            projectionUses = {{"shadows", RenderGraphResourceAccess::TextureStorageWrite},
                {"debug", RenderGraphResourceAccess::TextureStorageWrite},
                {"depth", RenderGraphResourceAccess::TextureSampleRead}};
        }
        std::vector<RenderGraphStageUse> publishUses{{"statistics", RenderGraphResourceAccess::BufferTransferWrite}};
        if (collectGPUWork_) {
            publishUses.push_back({"workload", RenderGraphResourceAccess::BufferTransferWrite});
            publishUses.push_back({"buildDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        if (!compact_) {
            publishUses.push_back({"shadows", RenderGraphResourceAccess::TextureTransferWrite});
            publishUses.push_back({"debug", RenderGraphResourceAccess::TextureTransferWrite});
        }
        textureImports.reserve(publications.size() * 2);
        bufferImports.reserve(publications.size());
        publishUses.reserve(3 + publications.size() * 3);
        for (const auto& publication : publications) {
            const auto& shadow = publication.shadow;
            const auto& names = publication.names;
            if (!compact_) {
                textureImports.push_back({names[0], shadow.texture, shadow.shadow,
                    ResourceState::ShaderRead, ResourceState::ShaderRead});
                textureImports.push_back({names[1], shadow.debugTexture, shadow.debug,
                    ResourceState::ShaderRead, ResourceState::ShaderRead});
                publishUses.push_back({names[0], RenderGraphResourceAccess::TextureTransferRead});
                publishUses.push_back({names[1], RenderGraphResourceAccess::TextureTransferRead});
            }
            bufferImports.push_back({names[2], publication.importedStats, RenderGraphResourceAccess::BufferStorageReadWrite});
            publishUses.push_back({names[2], RenderGraphResourceAccess::BufferTransferRead});
            if (compact_) { projectionUses.push_back({names[2], RenderGraphResourceAccess::BufferStorageReadWrite}); }
        }

        auto memoryBatch = context.streamer()->beginCopyBatch();
        if (!memoryBatch) { (void)context.streamer()->cancelCopyBatch(*batch); return makeError(memoryBatch.error()); }
        std::vector<RenderGraphStageUse> temporalUses{{"memory", RenderGraphResourceAccess::BufferTransferWrite}};
        if (!temporalIdentities.empty()) {
            temporalUses.push_back({"shadows", RenderGraphResourceAccess::TextureStorageReadWrite});
            temporalUses.push_back({"depth", RenderGraphResourceAccess::TextureSampleRead});
        }
        std::array<uint64_t, 8> memoryData{};
        const std::array stages{RenderGraphStage{"Initialize VSM light publication", initializeUses,
            [&](CommandBuffer& cmd) -> Result<> {
                CPUProfileScope stageCPU(cpu, "Initialize publication commands");
                auto cleared = cmd.clearColorTexture(*output.texture(), TextureLayout::TransferDestination, {1, 1, 1, 1});
                if (!cleared) { return cleared; }
                cleared = cmd.clearColorTexture(*debug.texture(), TextureLayout::TransferDestination, {1, 0, 1, 1});
                if (!cleared) { return cleared; }
                return context.streamer()->copyStreamedData(cmd, *batch);
            }}, RenderGraphStage{"Project compact VSM lights", projectionUses,
            [&](CommandBuffer& cmd) -> Result<> {
                CPUProfileScope stageCPU(cpu, "Project all faces");
                std::string log;
                for (const auto& group : projections) {
                    for (uint32_t face = 0; face < group.faceCount; ++face) {
                        const auto pointFaces = group.filteredPoint ? std::span<const VirtualShadowMapResult>(group.results) :
                            std::span<const VirtualShadowMapResult>{};
                        if (auto r = group.faces[face]->project(cmd, pointFaces, log, cpu, gpuProfiler); !r) {
                            spdlog::error("VirtualShadowMapLightsPass projection: {}", log);
                            return r;
                        }
                    }
                }
                return {};
            }}, RenderGraphStage{"Publish VSM faces", publishUses,
            [&](CommandBuffer& cmd) -> Result<> {
                CPUProfileScope stageCPU(cpu, "Memory statistics and publication commands");
                for (const auto& [identity, faces] : caches_) {
                    for (const auto& face : faces) {
                        if (!face) { continue; }
                        const auto usage = face->memoryUsage();
                        memoryData[0] += usage.atlasBytes;
                        memoryData[1] += usage.privateOutputBytes;
                        memoryData[2] += usage.bufferBytes;
                        ++memoryData[4];
                        memoryData[5] += usage.allocatedAtlasPages;
                    }
                }
                memoryData[3] = uint64_t(context.width()) * context.height() * output.desc().layerCount * 5;
                for (const auto& publication : publications) {
                    const auto& shadow = publication.shadow;
                    if (!compact_) {
                        auto copied = cmd.copyTexture({.source = shadow.texture, .destination = output.texture(),
                            .width = context.width(), .height = context.height(), .destinationBaseLayer = publication.layer});
                        if (!copied) { return copied; }
                        copied = cmd.copyTexture({.source = shadow.debugTexture, .destination = debug.texture(),
                            .width = context.width(), .height = context.height(), .destinationBaseLayer = publication.layer});
                        if (!copied) { return copied; }
                    }
                    auto copied = cmd.copyBuffer(publication.sourceStats, publication.destinationStats);
                    if (!copied) { return copied; }
                    if (collectGPUWork_) {
                        copied = cmd.copyBuffer(publication.sourceWorkload, publication.destinationWorkload);
                        if (!copied) { return copied; }
                        copied = cmd.copyBuffer(publication.sourceBuildDiagnostics, publication.destinationBuildDiagnostics);
                        if (!copied) { return copied; }
                    }
                }
                return {};
            }}, RenderGraphStage{"Stabilize SMRT shadows", temporalUses,
            [&](CommandBuffer& cmd) -> Result<> {
                const uint64_t revision = context.historyResources() ? context.historyResources()->geometryReprojectionInvalidationRevision() : 0;
                std::string log;
                for (uint32_t index : order) {
                    const auto& item = lights[index];
                    if (!temporalIdentities.contains(item.stableId)) { continue; }
                    auto& history = histories_[item.stableId];
                    if (!history) { history = std::make_unique<VirtualShadowMapTemporal>(); }
                    auto temporalSettings = settings;
                    temporalSettings.physicalPageCount = item.faceBudgets[0];
                    uint64_t lightContentKey = contentKey;
                    for (uint32_t faceBudget : item.faceBudgets) {
                        lightContentKey = (lightContentKey ^ faceBudget) * 1099511628211ull;
                    }
                    if (auto r = history->record(*device_, cmd, *depth.view(), *output.view(), view, info.frameIndex,
                        revision, lightContentKey, item.light, temporalSettings, historyFrames,
                        item.entry.baseLayer, item.entry.faceCount, compact_, log); !r) {
                        spdlog::error("VirtualShadowMapLightsPass temporal: {}", log);
                        return r;
                    }
                    memoryData[6] += history->memoryBytes();
                }
                const StreamDataChunk memoryChunk{memoryData.data(), sizeof(memoryData)};
                const auto memoryUpload = context.streamer()->streamBufferData({.dataChunks = {&memoryChunk, 1},
                    .placementAlignment = 16, .dstBuffer = memory.buffer(), .copyBatch = *memoryBatch});
                if (!memoryUpload.valid()) { return makeError(Error::OutOfMemory); }
                return context.streamer()->copyStreamedData(cmd, *memoryBatch);
            }}};
        cpuPhase.next("Execute publication stages");
        auto result = context.executeStages(stages, bufferImports, textureImports);
        if (!result) {
            (void)context.streamer()->cancelCopyBatch(*batch);
            (void)context.streamer()->cancelCopyBatch(*memoryBatch);
            return result;
        }
        return {};
    }
private:
    Device* device_ = nullptr;
    VirtualShadowMapCasterCache casterCache_;
    uint32_t capacity_ = 8;
    bool compact_ = false;
    bool collectGPUWork_ = false;
    uint64_t sceneIdentity_ = 0;
    std::unordered_map<uint64_t, std::array<std::unique_ptr<VirtualShadowMaps>, 6>> caches_;
    std::unordered_map<uint64_t, std::unique_ptr<VirtualShadowMapTemporal>> histories_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVirtualShadowMapLightsPass()
{
    return std::make_unique<VirtualShadowMapLightsPass>();
}

} // namespace metallic::render::builtin_pass
