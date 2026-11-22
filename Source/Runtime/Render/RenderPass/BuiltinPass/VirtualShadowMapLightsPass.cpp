#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLights.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProjectionGrid.h"
#include "Runtime/Render/Shadow/VirtualShadowMapResolution.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <bit>
#include <cassert>
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
};

uint64_t temporalContentKey(const VirtualShadowMapCasterResources& casters, const VisibilityBufferFrameInfo& raster)
{
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&hash](const auto& value) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(value); ++i) { hash = (hash ^ bytes[i]) * 1099511628211ull; }
    };
    append(casters.sceneIdentity); append(casters.dependencyKey);
    append(raster.sceneIdentity); append(raster.rasterSettingsRevision);
    for (const auto& caster : casters.states) {
        append(caster.id); append(caster.contentKey); append(caster.lodKey);
        append(caster.textureKey); append(caster.tessellationKey); append(caster.bounds.valid);
        if (caster.tessellationKey != 0) { append(caster.displacementMaterialKey); }
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
    // VSM uploads and feedback slots follow their recording completion. The
    // graph orders GPU access to shared cache and output resources across frames.
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }
    std::optional<RenderGraphExternalDomainContract> externalDomainContract() const override
    {
        return RenderGraphExternalDomainContract{
            .domainName = "render.pass.virtual-shadow-map-lights",
            .ownership = RenderGraphResourceOwnership::ExternallyOwned,
            .hasUndeclaredImplicitAccesses = true,
            .conservativeReason = "Persistent shared VSM physical page pool, per-light caches, and per-light SMRT temporal histories",
        };
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo", "visibility"}};
    }
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        const uint32_t capacity = properties().value("lightCapacity", 8u);
        const bool compact = properties().value("compactOutputs", false);
        const bool debugOutput = properties().value("debugOutput", true);
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Receiver depth from the matching visibility raster")
            .sampledRead().format = Format::D32Sfloat;
        reflection.addTextureInput("visibility", "Optional same-raster triangle IDs for refined LOD receiver bias")
            .sampledRead().setOptional().format = Format::R32Uint;
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
        if (debugOutput) {
            auto& debug = reflection.addTextureOutput("debug", "Per-face page selection; magenta marks unavailable layers")
                .transferWrite().transient(RenderGraphInitialization::FullOverwrite);
            debug.format = Format::RGBA8Unorm;
            debug.layerCount = compact ? capacity : capacity * 6;
            debug.stageAccess(RenderGraphResourceAccess::TextureStorageWrite, RenderGraphPassKind::Compute);
        }
        reflection.addBufferOutput("lights", "Current source-to-layer table and explicit light admission")
            .buffer(sizeof(VirtualShadowMapLightsHeader) + uint64_t(capacity) * sizeof(VirtualShadowMapLightEntry), 0)
            .transferWrite()
            .stageAccess(RenderGraphResourceAccess::BufferStorageReadWrite, RenderGraphPassKind::Compute);
        reflection.addBufferOutput("statistics", "Six 32-uint face blocks per light entry")
            .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapFaceStatisticsSize, sizeof(uint32_t))
            .bufferLayout("u32").transferWrite();
        reflection.addBufferOutput("cacheDiagnostics", "Six cache invalidation and static/dynamic page-work blocks per light entry")
            .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapCacheDiagnosticSize, sizeof(uint32_t))
            .bufferLayout("u32").transferWrite();
        reflection.addBufferOutput("lodStatistics", "Per-view shadow LOD selection, page demand and selected cluster level histogram")
            .buffer(32 * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        reflection.addBufferOutput("budgetDiagnostics", "Current GPU shared-pool allocation header and eight uints per source face")
            .buffer(uint64_t(8 + capacity * 6 * 8) * sizeof(uint32_t), sizeof(uint32_t)).bufferLayout("u32").transferWrite()
            .stageAccess(RenderGraphResourceAccess::BufferStorageReadWrite, RenderGraphPassKind::Compute);
        reflection.addBufferOutput("resolutionDiagnostics", "Completed pool demand and current receiver resolution LOD bias")
            .buffer(sizeof(VirtualShadowMapResolutionController::Diagnostics), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        reflection.addBufferOutput("rasterResolutionDiagnostics", "Completed GPU shadow raster time and resolution feedback")
            .buffer(sizeof(VirtualShadowMapResolutionController::Diagnostics), sizeof(uint32_t)).bufferLayout("u32").transferWrite();
        if (properties().value("benchmarkGPUWork", false)) {
            reflection.addBufferOutput("projectionGridDiagnostics", "One-pass light grid and shared receiver counters in low/high uint pairs")
                .buffer(kVirtualShadowMapProjectionGridDiagnosticSize, sizeof(uint32_t))
                .bufferLayout("u32").transferWrite()
                .stageAccess(RenderGraphResourceAccess::BufferStorageReadWrite, RenderGraphPassKind::Compute);
            reflection.addBufferOutput("workload", "Six GPU workload blocks per light entry; sixteen low/high uint pairs per face")
                .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapWorkloadSize, sizeof(uint32_t))
                .bufferLayout("u32").transferWrite();
            reflection.addBufferOutput("buildDiagnostics", "Six cluster build counter and page-scan histogram blocks per light entry")
                .buffer(uint64_t(capacity) * 6 * kVirtualShadowMapBuildDiagnosticSize, sizeof(uint32_t))
                .bufferLayout("u32").transferWrite();
        }
        if (properties().value("benchmarkReceiverCoverage", false)) {
            const auto levels = properties().value("clipmapCount", 4u);
            const auto axis = properties().value("virtualPagesPerAxis", 16u);
            reflection.addBufferOutput("receiverCoverageSettings", "Receiver coverage protocol and actual face settings")
                .buffer(16 * sizeof(uint32_t), sizeof(uint32_t)).transferWrite();
            reflection.addBufferOutput("receiverCoverageProjections", "Complete directional projections including coarse fallback")
                .buffer(uint64_t(levels + 1) * 112, 4).transferWrite();
            reflection.addBufferOutput("receiverCoveragePages", "Actual directional page table after allocation and projection")
                .buffer((uint64_t(levels) * axis * axis + 1) * 16, 4).transferWrite();
            const VirtualShadowMapSettings defaults;
            if (properties().value("cacheEnabled", true) && properties().value("cacheLayering", defaults.cacheLayering)) {
                reflection.addBufferOutput("receiverCoverageMask", "Current requested 8x8 receiver coverage for every virtual page")
                    .buffer((uint64_t(levels) * axis * axis + 1) * 8, 4).transferWrite();
            }
            if (properties().value("benchmarkReceiverCoverageRecords", false)) {
                reflection.addBufferOutput("receiverCoverage", "Per-pixel receiver coverage diagnostic records")
                    .buffer(uint64_t(context.width) * context.height * kVSMReceiverCoverageRecordWords * sizeof(uint32_t), 4)
                    .transferWrite();
            }
        }
        reflection.addBufferOutput("memory", "Logical owned bytes: total atlas, private output, buffers, public output, faces, atlas pages, temporal history, static atlas subset")
            .buffer(8 * sizeof(uint64_t), sizeof(uint64_t)).transferWrite();
        return reflection;
    }
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto capacity = runtimeIntSetting("lightCapacity", "Light Table Capacity", 8, 2, 64);
        capacity.rebuildGraph = true;
        auto compact = runtimeBoolSetting("compactOutputs", "One Output Layer per Light", false);
        compact.rebuildGraph = true;
        auto debugOutput = runtimeBoolSetting("debugOutput", "Publish Page Debug", true);
        debugOutput.rebuildGraph = true;
        return {capacity, compact, debugOutput,
            runtimeBoolSetting("enabled", "Virtual Shadows", true),
            runtimeBoolSetting("benchmarkGPUProfile", "Detailed GPU Profile", false),
            runtimeBoolSetting("benchmarkBuildSubstages", "Profile Build Substages", false),
            runtimeBoolSetting("cacheEnabled", "Cache Shadow Pages", true),
            runtimeBoolSetting("cacheLayering", "Separate Static and Dynamic Depth", false),
            runtimeIntSetting("framesStaticThreshold", "Frames Before Caster Becomes Static (0 Off)", 100, 0, 65535),
            runtimeBoolSetting("shadowLOD", "Shadow-specific Geometry LOD", false),
            runtimeBoolSetting("shadowLODSmartBias", "Shadow LOD Screen Depth Bias", false),
            runtimeFloatSetting("shadowLODScreenRayLength", "Shadow LOD Screen Ray Length", .015f, 0, 1),
            runtimeFloatSetting("shadowLODErrorTexels", "Shadow LOD Error (texels)", 1, 0, 16),
            runtimeBoolSetting("forceRedraw", "Force Shadow Redraw", false),
            runtimeIntSetting("physicalPageBudget", "Total Physical Page Budget", 256, 0, 262144),
            runtimeBoolSetting("demandDrivenBudget", "Distribute Pages by Receiver Demand", true),
            runtimeBoolSetting("unifiedMark", "Mark Shadow Pages Per View", false),
            runtimeBoolSetting("pointLightFaceSelection", "Select Point Shadow Face Per Light", true),
            runtimeBoolSetting("localSMRTStateReuse", "Reuse Local SMRT Ray State", true),
            runtimeBoolSetting("batchBuildRaster", "Batch Shadow Build and Raster", true),
            runtimeBoolSetting("compactShadowViews", "Compact Active Shadow Views", false),
            runtimeBoolSetting("distantLightSinglePage", "Single Page for Distant Local Lights", false),
            runtimeIntSetting("maxLightCacheAge", "Absent Light Cache Frames", VirtualShadowMapSettings{}.maxLightCacheAge, 0, 65535),
            runtimeBoolSetting("clusterHybridRaster", "Hybrid Shadow Cluster Raster", false),
            runtimeBoolSetting("twoPassHZB", "Two-Pass Shadow HZB", false),
            runtimeBoolSetting("hierarchyHZB", "Two-Pass Shadow Caster Hierarchy HZB", false),
            runtimeIntSetting("pageSize", "Page Size", 128, 16, 128),
            runtimeIntSetting("virtualPagesPerAxis", "Virtual Pages per Axis", 16, 1, 128),
            runtimeIntSetting("clipmapCount", "Clipmap / Local Mip Levels", 4, 1, 12),
            runtimeIntSetting("maxWorkItems", "Per-face Raster Work Budget", 65535, 1, 4 * 1024 * 1024),
            runtimeFloatSetting("firstClipmapExtent", "First Directional Half-width (m)", 4, 0.01f, 10000),
            runtimeFloatSetting("maxPagePoolLoadFactor", "Target Page Pool Load (0 Off)", VirtualShadowMapResolutionController::kDefaultLoadFactor, 0, 1),
            runtimeFloatSetting("maxResolutionLodBias", "Maximum Resolution LOD Bias", VirtualShadowMapResolutionController::kDefaultMaxBias, 0, 12),
            runtimeFloatSetting("rasterTimeBudgetMs", "GPU Shadow Raster Budget in ms (0 Off)", 0, 0, 100),
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
        debugOutput_ = properties().value("debugOutput", true);
        collectGPUWork_ = properties().value("benchmarkGPUWork", false);
        receiverCoverage_ = properties().value("benchmarkReceiverCoverage", false);
        receiverCoverageRecords_ = properties().value("benchmarkReceiverCoverageRecords", false);
        if (receiverCoverageRecords_ && (!receiverCoverage_ || !collectGPUWork_)) {
            log = "Receiver coverage records require benchmarkReceiverCoverage and benchmarkGPUWork";
            return makeError(Error::InvalidArgument);
        }
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
        if (device_ != context.device) {
            casterCache_ = {}; histories_.clear(); caches_.clear();
            physicalPool_ = {};
            resolution_ = {};
        }
        device_ = context.device;
        return {};
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        if ((context.properties().value("benchmarkCPUProfile", false) &&
            context.properties().value("benchmarkGPUProfile", false)) ||
            context.properties().value("benchmarkGPUWork", false) != collectGPUWork_ ||
            context.properties().value("benchmarkReceiverCoverage", false) != receiverCoverage_ ||
            context.properties().value("benchmarkReceiverCoverageRecords", false) != receiverCoverageRecords_) {
            spdlog::error("VSM CPU and GPU profiling must run separately; changing diagnostic resource settings requires graph recompilation");
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
        const auto visibility = context.inputTexture("visibility");
        const auto metadata = context.inputBuffer("rasterInfo");
        const auto output = context.outputTexture("shadows");
        const auto debug = context.outputTexture("debug");
        const auto table = context.outputBuffer("lights");
        const auto statistics = context.outputBuffer("statistics");
        const auto cacheDiagnostics = context.outputBuffer("cacheDiagnostics");
        const auto lodStatisticsOutput = context.outputBuffer("lodStatistics");
        const auto budgetDiagnostics = context.outputBuffer("budgetDiagnostics");
        const auto resolutionOutput = context.outputBuffer("resolutionDiagnostics");
        const auto rasterResolutionOutput = context.outputBuffer("rasterResolutionDiagnostics");
        const auto workload = context.outputBuffer("workload");
        const auto buildDiagnostics = context.outputBuffer("buildDiagnostics");
        const auto projectionGridDiagnostics = context.outputBuffer("projectionGridDiagnostics");
        const auto coverageSettings = context.outputBuffer("receiverCoverageSettings");
        const auto coverageProjections = context.outputBuffer("receiverCoverageProjections");
        const auto coveragePages = context.outputBuffer("receiverCoveragePages");
        const auto coverageMask = context.outputBuffer("receiverCoverageMask");
        const auto coverageRecords = context.outputBuffer("receiverCoverage");
        auto* gpuProfiler = context.properties().value("benchmarkGPUProfile", false) ? &context : nullptr;
        // Multi-light projection experiments need only these scopes; profiling
        // every face can exhaust the graph's per-pass timestamp section budget.
        auto* projectionGpuProfiler = gpuProfiler || context.properties().value("benchmarkProjectionProfile", false) ? &context : nullptr;
        const auto memory = context.outputBuffer("memory");
        const auto* scene = context.runtimeScene();
        const auto* gpuScene = context.subsystem<GPUSceneSubsystem>();
        if (!scene || !gpuScene || !context.streamer() || !depth.valid() || !depth.view() ||
            !metadata.valid() || metadata.desc().size != sizeof(VisibilityBufferFrameInfo) ||
            metadata.desc().memoryLocation != MemoryLocation::HostUpload ||
            !output.valid() || (debugOutput_ && !debug.valid()) || !table.valid() || !statistics.valid() || !cacheDiagnostics.valid() ||
            !lodStatisticsOutput.valid() || !budgetDiagnostics.valid() || !resolutionOutput.valid() ||
            !rasterResolutionOutput.valid() || !memory.valid() ||
            (collectGPUWork_ && (!workload.valid() || !buildDiagnostics.valid() || !projectionGridDiagnostics.valid())) ||
            (receiverCoverage_ && (!coverageSettings.valid() || !coverageProjections.valid() || !coveragePages.valid())) ||
            (receiverCoverageRecords_ && !coverageRecords.valid())) {
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
        settings.cacheLayering = boolProperty(&properties, "cacheLayering", settings.cacheLayering);
        settings.unifiedMark = boolProperty(&properties, "unifiedMark", settings.unifiedMark);
        settings.pointLightFaceSelection = boolProperty(&properties, "pointLightFaceSelection", settings.pointLightFaceSelection);
        settings.localSMRTStateReuse = boolProperty(&properties, "localSMRTStateReuse", settings.localSMRTStateReuse);
        settings.batchBuildRaster = boolProperty(&properties, "batchBuildRaster", settings.batchBuildRaster);
        settings.compactShadowViews = boolProperty(&properties, "compactShadowViews", settings.compactShadowViews);
        settings.distantLightSinglePage = boolProperty(&properties, "distantLightSinglePage", settings.distantLightSinglePage);
        settings.maxLightCacheAge = properties.value("maxLightCacheAge", settings.maxLightCacheAge);
        settings.clusterHybridRaster = boolProperty(&properties, "clusterHybridRaster", settings.clusterHybridRaster);
        settings.twoPassHZB = boolProperty(&properties, "twoPassHZB", settings.twoPassHZB);
        settings.hierarchyHZB = boolProperty(&properties, "hierarchyHZB", settings.hierarchyHZB);
        settings.framesStaticThreshold = properties.value("framesStaticThreshold", settings.framesStaticThreshold);
        settings.shadowLOD = boolProperty(&properties, "shadowLOD", settings.shadowLOD);
        settings.shadowLODErrorTexels = properties.value("shadowLODErrorTexels", settings.shadowLODErrorTexels);
        settings.shadowLODSmartBias = boolProperty(&properties, "shadowLODSmartBias", settings.shadowLODSmartBias);
        settings.shadowLODScreenRayLength = properties.value("shadowLODScreenRayLength", settings.shadowLODScreenRayLength);
        VirtualShadowMapReceiverResources receiver{};
        if (settings.shadowLOD && settings.shadowLODSmartBias &&
            visibility.valid() && (info.reserved & 1u) == 0) {
            if (visibility.desc().width != info.width || visibility.desc().height != info.height) {
                return makeError(Error::InvalidArgument);
            }
            // Optional publications may be absent on non-visibility producers.
            // Tessellated primary surfaces require their final domain normal.
            const auto* receiverStream = gpuScene->visibilityStream({info.lightGridViewIndex, info.lightGridViewGeneration},
                info.frameIndex, info.sceneIdentity);
            if (receiverStream) { receiver = {visibility.view(), receiverStream}; }
        }
        settings.forceRedraw = boolProperty(&properties, "forceRedraw", false);
        settings.collectGPUWork = collectGPUWork_;
        settings.collectReceiverCoverage = receiverCoverageRecords_;
        settings.profileBuildSubstages = boolProperty(&properties, "benchmarkBuildSubstages", false);
        settings.optimizeProjectionStatistics = boolProperty(&properties, "benchmarkProjectionStatistics", settings.optimizeProjectionStatistics);
        settings.dirtyPageHierarchy = boolProperty(&properties, "benchmarkDirtyPageHierarchy", settings.dirtyPageHierarchy);
        VirtualShadowMapProjectionGridResources projectionGrid;
        if (!sources.empty()) {
            // Mark and projection consume the same primary view and GPUScene light slots.
            projectionGrid.grid = gpuScene->lightGrid({info.lightGridViewIndex, info.lightGridViewGeneration}, info.lightGridFrameSlot);
        }
        projectionGrid.diagnostics = collectGPUWork_ ? projectionGridDiagnostics.buffer() : nullptr;
        settings.compactBuild = boolProperty(&properties, "benchmarkCompactBuild", settings.compactBuild);
        settings.clusterPageRaster = boolProperty(&properties, "benchmarkClusterPageRaster", settings.clusterPageRaster);
        settings.pageWindowRaster = boolProperty(&properties, "benchmarkPageWindowRaster", settings.pageWindowRaster);
        settings.instanceBuild = boolProperty(&properties, "benchmarkInstanceBuild", settings.instanceBuild);
        settings.enabled = boolProperty(&properties, "enabled", true) && !boolProperty(&properties, "debugDisableShadows", false);
        const uint32_t budget = properties.value("physicalPageBudget", 256u);
        const bool demandDriven = boolProperty(&properties, "demandDrivenBudget", true);
        const float nearPlane = properties.value("nearPlane", 0.001f);
        if (budget > 262144 || settings.maxLightCacheAge > 65535 || !std::isfinite(nearPlane) || nearPlane <= 0) {
            return makeError(Error::InvalidArgument);
        }
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
        if (sceneIdentity_ != info.sceneIdentity) {
            caches_.clear();
            histories_.clear();
            physicalPool_ = {};
            resolution_ = {};
            casterCache_ = {};
            sceneIdentity_ = info.sceneIdentity;
        }
        std::unordered_set<uint64_t> identities;
        for (const auto& item : lights) {
            if (item.entry.status == kVirtualShadowMapLightAdmitted) { identities.insert(item.stableId); }
        }
        std::erase_if(caches_, [&](const auto& cache) {
            if (identities.contains(cache.first)) { return false; }
            if (!settings.cacheEnabled || settings.maxLightCacheAge == 0 || budget == 0) { return true; }
            bool retained = false;
            for (const auto& face : cache.second) {
                if (face && face->retainInactiveCache(info.frameIndex, settings.maxLightCacheAge)) { retained = true; }
            }
            return !retained;
        });
        if (caches_.empty()) { physicalPool_ = {}; }
        auto resolutionWords = resolution_.update(
            properties.value("maxPagePoolLoadFactor", VirtualShadowMapResolutionController::kDefaultLoadFactor),
            properties.value("maxResolutionLodBias", VirtualShadowMapResolutionController::kDefaultMaxBias), admittedFaces != 0,
            properties.value("rasterTimeBudgetMs", 0.f), device_->capabilities().timestampQueries);
        if (!resolutionWords) { return makeError(resolutionWords.error()); }
        settings.resolutionLodBias = resolution_.bias();
        const auto& rasterResolutionWords = resolution_.rasterDiagnostics();
        std::vector<uint32_t> budgetData(8 + capacity_ * 6 * 8, 0);
        // Version 2 describes current-frame GPU demand and grants. The shared
        // allocator publishes totals and face records after this zeroed header
        // is uploaded; no delayed CPU demand sample controls physical capacity.
        budgetData[0] = 2;
        budgetData[1] = demandDriven;
        budgetData[2] = admittedFaces;
        budgetData[4] = budget;
        std::vector<uint32_t> publicationFaces;
        publicationFaces.reserve(admittedFaces);
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
        std::array<uint32_t, 16> coverageHeader{};
        if (receiverCoverage_) {
            const auto selected = std::find_if(lights.begin(), lights.end(), [](const auto& item) {
                return item.entry.status == kVirtualShadowMapLightAdmitted;
            });
            if (admittedFaces != 1 || selected == lights.end() ||
                selected->light.type != VirtualShadowMapLightType::Directional || !settings.enabled ||
                settings.filter != VirtualShadowMapFilter::SMRT || settings.smrtRayCount != 2) {
                spdlog::error("Receiver coverage export requires one admitted enabled directional SMRT face with two rays and ray-page marking");
                return makeError(Error::InvalidArgument);
            }
            const uint64_t recordBytes = uint64_t(info.width) * info.height * kVSMReceiverCoverageRecordWords * sizeof(uint32_t);
            if (coverageMask.valid() != (settings.cacheEnabled && settings.cacheLayering) ||
                (receiverCoverageRecords_ && coverageRecords.desc().size != recordBytes) ||
                coverageProjections.desc().size != uint64_t(settings.clipmapCount + 1) * 112 ||
                coveragePages.desc().size != (uint64_t(settings.clipmapCount) * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1) * 16) {
                spdlog::error("Receiver coverage export dimensions/settings changed without graph recompilation");
                return makeError(Error::InvalidArgument);
            }
            coverageHeader = {1, info.width, info.height, settings.pageSize, settings.virtualPagesPerAxis,
                settings.clipmapCount, budget, settings.smrtRayCount, settings.smrtSamplesPerRay,
                std::bit_cast<uint32_t>(selected->light.angularRadius), uint32_t(selected - lights.begin()),
                uint32_t(receiverCoverageRecords_), kVSMReceiverCoverageRecordWords, uint32_t(coverageProjections.desc().size),
                uint32_t(coveragePages.desc().size), coverageMask.valid() ? uint32_t(coverageMask.desc().size) : 0u};
        }
        cpuPhase.next("Light table and statistics upload");

        struct FacePublication
        {
            BufferSlice sourceStats;
            BufferSlice destinationStats;
            BufferSlice importedStats;
            BufferSlice sourceWorkload;
            BufferSlice destinationWorkload;
            BufferSlice sourceBuildDiagnostics;
            BufferSlice destinationBuildDiagnostics;
            BufferSlice sourceCacheDiagnostics;
            BufferSlice destinationCacheDiagnostics;
            std::string name;
            std::string cacheName;
        };
        struct CoveragePublication
        {
            BufferSlice source;
            BufferSlice destination;
            std::string name;
            std::string output;
        };
        std::vector<CoveragePublication> coveragePublications;
        std::vector<FacePublication> publications;
        publications.reserve(admittedFaces);
        std::vector<std::byte> tableData(table.desc().size);
        std::memcpy(tableData.data(), &header, sizeof(header));
        for (size_t index = 0; index < lights.size(); ++index) {
            std::memcpy(tableData.data() + sizeof(header) + index * sizeof(VirtualShadowMapLightEntry),
                &lights[index].entry, sizeof(VirtualShadowMapLightEntry));
        }
        std::vector<uint32_t> zeroStatistics(statistics.desc().size / sizeof(uint32_t), 0);
        std::vector<uint32_t> zeroCacheDiagnostics(cacheDiagnostics.desc().size / sizeof(uint32_t), 0);
        std::vector<uint32_t> zeroWorkload(collectGPUWork_ ? workload.desc().size / sizeof(uint32_t) : 0, 0);
        std::vector<uint32_t> zeroBuildDiagnostics(collectGPUWork_ ? buildDiagnostics.desc().size / sizeof(uint32_t) : 0, 0);
        std::array<uint32_t, kVirtualShadowMapProjectionGridDiagnosticSize / sizeof(uint32_t)> zeroProjectionGrid{};
        auto batch = context.streamer()->beginCopyBatch();
        if (!batch) { return makeError(batch.error()); }
        std::vector<StreamDataChunk> chunks{{tableData.data(), tableData.size()},
            {zeroStatistics.data(), zeroStatistics.size() * sizeof(uint32_t)},
            {zeroCacheDiagnostics.data(), zeroCacheDiagnostics.size() * sizeof(uint32_t)},
            {budgetData.data(), budgetData.size() * sizeof(uint32_t)}, {resolutionWords->data(), sizeof(*resolutionWords)},
            {rasterResolutionWords.data(), sizeof(rasterResolutionWords)}};
        std::vector<Buffer*> destinations{table.buffer(), statistics.buffer(), cacheDiagnostics.buffer(),
            budgetDiagnostics.buffer(), resolutionOutput.buffer(), rasterResolutionOutput.buffer()};
        if (collectGPUWork_) {
            chunks.push_back({zeroWorkload.data(), zeroWorkload.size() * sizeof(uint32_t)});
            destinations.push_back(workload.buffer());
            chunks.push_back({zeroBuildDiagnostics.data(), zeroBuildDiagnostics.size() * sizeof(uint32_t)});
            destinations.push_back(buildDiagnostics.buffer());
            chunks.push_back({zeroProjectionGrid.data(), sizeof(zeroProjectionGrid)});
            destinations.push_back(projectionGridDiagnostics.buffer());
        }
        if (receiverCoverage_) {
            chunks.push_back({coverageHeader.data(), sizeof(coverageHeader)});
            destinations.push_back(coverageSettings.buffer());
        }
        for (size_t index = 0; index < chunks.size(); ++index) {
            const auto upload = context.streamer()->streamBufferData({.dataChunks = {&chunks[index], 1},
                .placementAlignment = 16, .dstBuffer = destinations[index], .copyBatch = *batch});
            if (!upload.valid()) { (void)context.streamer()->cancelCopyBatch(*batch); return makeError(Error::OutOfMemory); }
        }
        std::vector<RenderGraphStageUse> initializeUses{
            RenderGraphStageUse{"lights", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"statistics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"cacheDiagnostics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"resolutionDiagnostics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"rasterResolutionDiagnostics", RenderGraphResourceAccess::BufferTransferWrite},
            RenderGraphStageUse{"budgetDiagnostics", RenderGraphResourceAccess::BufferTransferWrite}};
        if (collectGPUWork_) {
            initializeUses.push_back({"workload", RenderGraphResourceAccess::BufferTransferWrite});
            initializeUses.push_back({"buildDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
            initializeUses.push_back({"projectionGridDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        if (receiverCoverage_) {
            initializeUses.push_back({"receiverCoverageSettings", RenderGraphResourceAccess::BufferTransferWrite});
        }
        std::vector<VirtualShadowMapProjectionGroup> projections;
        projections.reserve(header.admittedCount);
        ViewConstants view{};
        uint64_t contentKey = 0;
        std::shared_ptr<Buffer> gpuLODStatistics;
        std::array<uint32_t, 32> lodStatistics{};
        lodStatistics[0] = 3;
        lodStatistics[1] = settings.shadowLOD;
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
                std::vector<VirtualShadowMapMaterialTextureRevisions> textureRevisions(scene->materials().size());
                for (size_t index = 0; index < scene->materials().size(); ++index) {
                    const auto& material = scene->materials()[index];
                    const auto logical = material.baseColorTexture.textureIndex;
                    if (material.alphaMode == "MASK" && logical >= 0 && size_t(logical) < logicalTextures.size()) {
                        const uint32_t slot = logicalTextures[logical];
                        if (slot < textures->views.size()) {
                            alphaTextures[index] = textures->views[slot].get();
                            textureRevisions[index].alpha = textures->slotPublicationRevisions[slot];
                        }
                    }
                    const auto displacement = material.displacementTexture.textureIndex;
                    if (material.displacementMagnitude != 0 && displacement >= 0 && size_t(displacement) < logicalTextures.size()) {
                        const uint32_t slot = logicalTextures[displacement];
                        if (slot < textures->views.size()) {
                            displacementTextures[index] = textures->views[slot].get();
                            textureRevisions[index].height = textures->slotPublicationRevisions[slot];
                        }
                    }
                }
                const auto hasTexture = [](const auto* texture) { return texture != nullptr; };
                const bool textureDependent = std::any_of(alphaTextures.begin(), alphaTextures.end(), hasTexture) ||
                    std::any_of(displacementTextures.begin(), displacementTextures.end(), hasTexture);
                const bool casterHierarchy = settings.hierarchyHZB && settings.cacheEnabled && !settings.cacheLayering &&
                    settings.batchBuildRaster && device_->capabilities().computeSubgroupBallotArithmetic &&
                    std::ranges::any_of(lights, [](const auto& item) {
                        return item.entry.status == kVirtualShadowMapLightAdmitted &&
                            item.light.type == VirtualShadowMapLightType::Directional;
                    });
                const VirtualShadowMapCasterSettings casterSettings{
                    .tessellationEnabled = (info.reserved & 1u) != 0,
                    .tessellationFactor = properties.value("tessellationFactor", 4u),
                    .subdivisionDepth = properties.value("tessellationSubdivisionDepth", 0u), .displacementTextures = displacementTextures,
                    .texturePublicationRevision = textureDependent ? materialResources.materialTexturePublicationRevision() : 0,
                    .instanceBuild = settings.instanceBuild || casterHierarchy, .hierarchyHZB = casterHierarchy,
                    .materialTextureRevisions = textureRevisions};
                auto casters = casterCache_.prepare(*device_, commands, *scene,
                    scene->hasStreamGeometry() ? stream : nullptr, log, casterSettings);
                if (!casters) { spdlog::error("VirtualShadowMapLightsPass casters: {}", log); return makeError(casters.error()); }
                lodStatistics[5] = casters->triangleCapacity;
                lodStatistics[10] = casters->triangleCapacity;
                if (settings.shadowLOD) {
                    casters = casterCache_.prepareGPUShadowLOD(*device_, commands, *scene,
                        scene->hasStreamGeometry() ? stream : nullptr, *casters, casterSettings, log);
                    if (!casters) { spdlog::error("VirtualShadowMapLightsPass shadow LOD: {}", log); return makeError(casters.error()); }
                }
                if (!temporalIdentities.empty()) { contentKey = temporalContentKey(*casters, info); }
                profile.next("VSM page demand, cache and raster");
                cpuPhase.next("Record shared physical pool");
                std::vector<VirtualShadowMapFaceRequest> faceRequests;
                faceRequests.reserve(admittedFaces);
                for (uint32_t index : order) {
                    const auto& item = lights[index];
                    if (item.entry.status != kVirtualShadowMapLightAdmitted) { continue; }
                    auto& faces = caches_[item.stableId];
                    for (uint32_t face = item.entry.faceCount; face < faces.size(); ++face) { faces[face].reset(); }
                    for (uint32_t face = 0; face < item.entry.faceCount; ++face) {
                        if (!faces[face]) { faces[face] = std::make_unique<VirtualShadowMaps>(); }
                        auto light = item.light;
                        light.face = face;
                        faceRequests.push_back({faces[face].get(), light,
                            {output.texture(), debugOutput_ ? debug.texture() : nullptr,
                                item.entry.baseLayer + (compact_ ? 0u : face), compact_ && item.entry.faceCount == 6}, index});
                        publicationFaces.push_back(index * 6 + face);
                    }
                }
                auto rasterFeedback = resolution_.prepareRaster(*device_, commands, settings.batchBuildRaster ? 1u : admittedFaces);
                if (!rasterFeedback) { return std::unexpected(rasterFeedback.error()); }
                auto sharedResults = VirtualShadowMaps::recordShared(physicalPool_, *device_, commands,
                    *depth.view(), view, faceRequests, *casters, scene->materials(), alphaTextures, settings,
                    budget, demandDriven, log, cpu, gpuProfiler, receiver.stream ? &receiver : nullptr, projectionGrid.grid,
                    *rasterFeedback);
                if (!sharedResults) { spdlog::error("VirtualShadowMapLightsPass: {}", log); return makeError(sharedResults.error()); }
                if (auto recorded = resolution_.record(*device_, commands, *sharedResults, budget, &physicalPool_); !recorded) { return recorded; }
                auto aggregatedLOD = VirtualShadowMaps::aggregateLODStatistics(*device_, commands, *sharedResults, log);
                if (!aggregatedLOD) { spdlog::error("VirtualShadowMapLightsPass LOD statistics: {}", log); return makeError(aggregatedLOD.error()); }
                gpuLODStatistics = *aggregatedLOD;
                size_t resultIndex = 0;
                for (uint32_t index : order) {
                    const auto& item = lights[index];
                    if (item.entry.status != kVirtualShadowMapLightAdmitted) { continue; }
                    auto& faces = caches_[item.stableId];
                    const bool filteredPoint = item.entry.faceCount == 6 && settings.filter != VirtualShadowMapFilter::Hard;
                    std::array<VirtualShadowMapResult, 6> pointResults;
                    for (uint32_t face = 0; face < item.entry.faceCount; ++face) {
                        const auto* shadow = &(*sharedResults)[resultIndex++];
                        pointResults[face] = *shadow;
                        if (receiverCoverage_) {
                            auto appendCoverage = [&](Buffer* source, uint64_t offset, const auto& destination,
                                const std::string& name, const std::string& outputName) -> Result<> {
                                auto input = source->slice({offset, destination.desc().size});
                                auto targetSlice = destination.buffer()->slice();
                                if (!input || !targetSlice) { return makeError(Error::InvalidArgument); }
                                coveragePublications.push_back({*input, *targetSlice, name, outputName});
                                return {};
                            };
                            if (auto r = appendCoverage(shadow->projections, 0, coverageProjections,
                                "coverageProjectionSource", "receiverCoverageProjections"); !r) { return r; }
                            if (auto r = appendCoverage(shadow->pageTable, 0, coveragePages,
                                "coveragePageSource", "receiverCoveragePages"); !r) { return r; }
                            if (coverageMask.valid()) {
                                if (!shadow->requestedReceiverMasks) {
                                    spdlog::error("Receiver mask export requires active directional SMRT layered caching");
                                    return makeError(Error::InvalidArgument);
                                }
                                if (auto r = appendCoverage(shadow->requestedReceiverMasks, shadow->requestedReceiverMaskOffset,
                                    coverageMask, "coverageMaskSource", "receiverCoverageMask"); !r) { return r; }
                            }
                            if (receiverCoverageRecords_) {
                                if (auto r = appendCoverage(shadow->statistics,
                                    uint64_t(kVSMReceiverCoverageStatisticsWordOffset) * sizeof(uint32_t), coverageRecords,
                                    "coverageRecordSource", "receiverCoverage"); !r) { return r; }
                            }
                        }
                        const uint32_t layer = index * 6 + face;
                        auto sourceStats = shadow->statistics->slice({0, kVirtualShadowMapFaceStatisticsSize});
                        auto destinationStats = statistics.buffer()->slice({uint64_t(layer) * kVirtualShadowMapFaceStatisticsSize,
                            kVirtualShadowMapFaceStatisticsSize});
                        // Stage imports merge aliases by allocation and retain the first slice.
                        // Include the diagnostic tail here so its transfer barrier covers every record.
                        auto importedStats = shadow->statistics->slice({0, kVirtualShadowMapFaceStatisticsSize +
                            (collectGPUWork_ ? kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize : 0) +
                            (receiverCoverageRecords_ ? coverageRecords.desc().size : 0)});
                        auto sourceCacheDiagnostics = shadow->cacheDiagnostics->slice({0, kVirtualShadowMapCacheDiagnosticSize});
                        auto destinationCacheDiagnostics = cacheDiagnostics.buffer()->slice({uint64_t(layer) * kVirtualShadowMapCacheDiagnosticSize,
                            kVirtualShadowMapCacheDiagnosticSize});
                        if (!sourceStats || !destinationStats || !importedStats || !sourceCacheDiagnostics || !destinationCacheDiagnostics) {
                            return makeError(Error::InvalidArgument);
                        }
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
                        publications.push_back({*sourceStats, *destinationStats, *importedStats,
                            sourceWorkload, destinationWorkload, sourceBuildDiagnostics, destinationBuildDiagnostics,
                            *sourceCacheDiagnostics, *destinationCacheDiagnostics,
                            "faceStatistics" + std::to_string(layer), "faceCacheDiagnostics" + std::to_string(layer)});
                    }
                    VirtualShadowMapProjectionGroup group;
                    group.results = std::move(pointResults);
                    group.faceCount = item.entry.faceCount;
                    group.filteredPoint = filteredPoint;
                    group.lightIndex = index;
                    for (uint32_t face = 0; face < group.faceCount; ++face) { group.faces[face] = faces[face].get(); }
                    projections.push_back(std::move(group));
                }
            }
            return {};
        };
        if (auto r = recordFaces(context.commandBuffer()); !r) { (void)context.streamer()->cancelCopyBatch(*batch); return r; }
        BufferSlice sourceLODStatistics, destinationLODStatistics;
        if (gpuLODStatistics) {
            auto source = gpuLODStatistics->slice();
            auto destination = lodStatisticsOutput.buffer()->slice({0, sizeof(lodStatistics)});
            if (!source || !destination) {
                (void)context.streamer()->cancelCopyBatch(*batch);
                return makeError(Error::InvalidArgument);
            }
            sourceLODStatistics = *source; destinationLODStatistics = *destination;
        } else {
            const StreamDataChunk lodChunk{lodStatistics.data(), sizeof(lodStatistics)};
            const auto lodUpload = context.streamer()->streamBufferData({.dataChunks = {&lodChunk, 1},
                .placementAlignment = 16, .dstBuffer = lodStatisticsOutput.buffer(), .copyBatch = *batch});
            if (!lodUpload.valid()) { (void)context.streamer()->cancelCopyBatch(*batch); return makeError(Error::OutOfMemory); }
            initializeUses.push_back({"lodStatistics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        cpuPhase.next("Publication stage setup");
        // Build views only after the owning names have reached their final
        // addresses. Projection writes the public arrays directly; only face
        // statistics require imports for projection and publication.
        std::vector<RenderGraphBufferImport> bufferImports;
        std::vector<RenderGraphStageUse> projectionUses{{"shadows", RenderGraphResourceAccess::TextureStorageWrite},
            {"depth", RenderGraphResourceAccess::TextureSampleRead}};
        if (receiver.stream) { projectionUses.push_back({"visibility", RenderGraphResourceAccess::TextureSampleRead}); }
        if (debugOutput_) { projectionUses.push_back({"debug", RenderGraphResourceAccess::TextureStorageWrite}); }
        if (collectGPUWork_) { projectionUses.push_back({"projectionGridDiagnostics", RenderGraphResourceAccess::BufferStorageReadWrite}); }
        std::vector<RenderGraphStageUse> publishUses{{"statistics", RenderGraphResourceAccess::BufferTransferWrite},
            {"cacheDiagnostics", RenderGraphResourceAccess::BufferTransferWrite}};
        if (collectGPUWork_) {
            publishUses.push_back({"workload", RenderGraphResourceAccess::BufferTransferWrite});
            publishUses.push_back({"buildDiagnostics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        bufferImports.reserve(publications.size() * 2);
        publishUses.reserve(4 + publications.size() * 2);
        projectionUses.reserve(3 + publications.size());
        if (gpuLODStatistics) {
            bufferImports.push_back({"privateLODStatistics", sourceLODStatistics, RenderGraphResourceAccess::BufferTransferRead});
            publishUses.push_back({"privateLODStatistics", RenderGraphResourceAccess::BufferTransferRead});
            publishUses.push_back({"lodStatistics", RenderGraphResourceAccess::BufferTransferWrite});
        }
        if (projectionGrid.grid) {
            const auto& grid = *projectionGrid.grid;
            const std::array gridBuffers{
                std::pair{"vsmGridParameters", grid.parameters}, std::pair{"vsmGridCells", grid.cells},
                std::pair{"vsmGridIndices", grid.lightIndices}, std::pair{"vsmGridCandidates", grid.candidates}};
            for (const auto& [name, buffer] : gridBuffers) {
                auto slice = buffer->slice();
                if (!slice) { (void)context.streamer()->cancelCopyBatch(*batch); return std::unexpected(slice.error()); }
                bufferImports.push_back({name, *slice, RenderGraphResourceAccess::BufferStorageRead});
                projectionUses.push_back({name, RenderGraphResourceAccess::BufferStorageRead});
            }
        }
        for (const auto& publication : publications) {
            bufferImports.push_back({publication.name, publication.importedStats, RenderGraphResourceAccess::BufferStorageReadWrite});
            bufferImports.push_back({publication.cacheName, publication.sourceCacheDiagnostics, RenderGraphResourceAccess::BufferStorageReadWrite});
            publishUses.push_back({publication.name, RenderGraphResourceAccess::BufferTransferRead});
            publishUses.push_back({publication.cacheName, RenderGraphResourceAccess::BufferTransferRead});
            projectionUses.push_back({publication.name, RenderGraphResourceAccess::BufferStorageReadWrite});
        }

        for (const auto& publication : coveragePublications) {
            bufferImports.push_back({publication.name, publication.source, RenderGraphResourceAccess::BufferStorageReadWrite});
            publishUses.push_back({publication.name, RenderGraphResourceAccess::BufferTransferRead});
            publishUses.push_back({publication.output, RenderGraphResourceAccess::BufferTransferWrite});
        }

        auto memoryBatch = context.streamer()->beginCopyBatch();
        if (!memoryBatch) { (void)context.streamer()->cancelCopyBatch(*batch); return makeError(memoryBatch.error()); }
        std::vector<RenderGraphStageUse> temporalUses{{"memory", RenderGraphResourceAccess::BufferTransferWrite}};
        if (!temporalIdentities.empty()) {
            temporalUses.push_back({"shadows", RenderGraphResourceAccess::TextureStorageReadWrite});
            temporalUses.push_back({"depth", RenderGraphResourceAccess::TextureSampleRead});
            if (gpuLODStatistics) {
                for (const auto& publication : publications) {
                    temporalUses.push_back({publication.cacheName, RenderGraphResourceAccess::BufferStorageRead});
                }
            }
        }
        std::array<uint64_t, 8> memoryData{};
        const uint32_t outputLayerCount = output.desc().layerCount;
        uint32_t projectedFaceCount = 0;
        for (const auto& group : projections) { projectedFaceCount += group.faceCount; }
        const bool onePassProjection = projectedFaceCount > 1 && !settings.optimizeProjectionStatistics &&
            device_->capabilities().computeSubgroupBallotArithmetic;
        std::vector<uint8_t> coveredLayers(outputLayerCount, 0);
        if (!onePassProjection) {
            for (const auto& group : projections) {
                if (group.lightIndex >= lights.size()) { continue; }
                const auto& entry = lights[group.lightIndex].entry;
                const bool sharedPointLayer = compact_ && entry.faceCount == 6;
                if (sharedPointLayer) { continue; }
                for (uint32_t face = 0; face < group.faceCount; ++face) {
                    const uint32_t layer = entry.baseLayer + face;
                    if (layer < outputLayerCount) { coveredLayers[layer] = 1; }
                }
            }
        }
        std::vector<TextureSubresourceRange> clearLayerRanges;
        for (uint32_t layer = 0; layer < outputLayerCount;) {
            if (coveredLayers[layer]) { ++layer; continue; }
            const uint32_t start = layer;
            while (layer < outputLayerCount && !coveredLayers[layer]) { ++layer; }
            clearLayerRanges.push_back({0, output.desc().mipCount, start, layer - start});
        }
        if (!clearLayerRanges.empty()) {
            initializeUses.push_back({"shadows", RenderGraphResourceAccess::TextureTransferWrite});
            if (debugOutput_) { initializeUses.push_back({"debug", RenderGraphResourceAccess::TextureTransferWrite}); }
        }
        const std::array allocationUses{
            RenderGraphStageUse{"lights", RenderGraphResourceAccess::BufferStorageReadWrite},
            RenderGraphStageUse{"budgetDiagnostics", RenderGraphResourceAccess::BufferStorageReadWrite}};
        const std::array stages{RenderGraphStage{"Initialize VSM light publication", initializeUses,
            [&](CommandBuffer& cmd) -> Result<> {
                CPUProfileScope stageCPU(cpu, "Initialize publication commands");
                for (const auto& range : clearLayerRanges) {
                    auto cleared = cmd.clearColorTexture(*output.texture(), TextureLayout::TransferDestination, {1, 1, 1, 1}, range);
                    if (!cleared) { return cleared; }
                    if (debugOutput_) {
                        cleared = cmd.clearColorTexture(*debug.texture(), TextureLayout::TransferDestination, {1, 0, 1, 1}, range);
                        if (!cleared) { return cleared; }
                    }
                }
                return context.streamer()->copyStreamedData(cmd, *batch);
            }}, RenderGraphStage{"Publish shared VSM allocations", allocationUses,
            [&](CommandBuffer& cmd) -> Result<> {
                if (publicationFaces.empty()) { return {}; }
                CPUProfileScope stageCPU(cpu, "Publish current GPU page grants");
                std::string log;
                auto result = physicalPool_.publishAllocations(cmd, *table.buffer(), *budgetDiagnostics.buffer(), publicationFaces, log);
                if (!result) { spdlog::error("VirtualShadowMapLightsPass allocation publication: {}", log); }
                return result;
            }}, RenderGraphStage{"Project VSM lights", projectionUses,
            [&](CommandBuffer& cmd) -> Result<> {
                CPUProfileScope stageCPU(cpu, "Project all faces");
                std::string log;
                auto result = VirtualShadowMaps::projectShared(cmd, projections, log, cpu, projectionGpuProfiler, &projectionGrid);
                if (!result) { spdlog::error("VirtualShadowMapLightsPass projection: {}", log); }
                return result;
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
                        memoryData[7] += usage.staticAtlasBytes;
                    }
                }
                const auto poolUsage = physicalPool_.memoryUsage();
                memoryData[0] += poolUsage.atlasBytes;
                memoryData[1] += poolUsage.privateOutputBytes;
                memoryData[2] += poolUsage.bufferBytes;
                memoryData[5] += poolUsage.allocatedAtlasPages;
                memoryData[7] += poolUsage.staticAtlasBytes;
                // R8 visibility and optional RGBA8 page-debug arrays.
                memoryData[3] = uint64_t(context.width()) * context.height() * output.desc().layerCount;
                if (debugOutput_) {
                    memoryData[3] += uint64_t(context.width()) * context.height() * debug.desc().layerCount * 4;
                }
                for (const auto& publication : coveragePublications) {
                    if (auto copied = cmd.copyBuffer(publication.source, publication.destination); !copied) { return copied; }
                }
                if (gpuLODStatistics) {
                    if (auto copied = cmd.copyBuffer(sourceLODStatistics, destinationLODStatistics); !copied) { return copied; }
                }
                for (const auto& publication : publications) {
                    auto copied = cmd.copyBuffer(publication.sourceStats, publication.destinationStats);
                    if (!copied) { return copied; }
                    copied = cmd.copyBuffer(publication.sourceCacheDiagnostics, publication.destinationCacheDiagnostics);
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
                    temporalSettings.physicalPageCount = budget;
                    const auto group = std::find_if(projections.begin(), projections.end(),
                        [index](const auto& projection) { return projection.lightIndex == index; });
                    assert(group != projections.end());
                    temporalSettings.clipmapCount = group->results[0].clipmapCount;
                    const uint64_t lightContentKey = (contentKey ^ uint32_t(demandDriven)) * 1099511628211ull;
                    std::array<Buffer*, 6> lodCaches{};
                    uint32_t lodCacheCount = 0;
                    if (gpuLODStatistics) {
                        lodCacheCount = group->faceCount;
                        for (uint32_t face = 0; face < lodCacheCount; ++face) {
                            lodCaches[face] = group->results[face].cacheDiagnostics;
                        }
                    }
                    if (auto r = history->record(*device_, cmd, *depth.view(), *output.view(), view, info.frameIndex,
                        revision, lightContentKey, item.light, temporalSettings, historyFrames,
                        item.entry.baseLayer, item.entry.faceCount, compact_, log, gpuProfiler,
                        {lodCaches.data(), lodCacheCount}); !r) {
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
        auto result = context.executeStages(stages, bufferImports);
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
    bool debugOutput_ = true;
    bool collectGPUWork_ = false;
    bool receiverCoverage_ = false;
    bool receiverCoverageRecords_ = false;
    uint64_t sceneIdentity_ = 0;
    VirtualShadowMapPhysicalPool physicalPool_;
    VirtualShadowMapResolutionController resolution_;
    std::unordered_map<uint64_t, std::array<std::unique_ptr<VirtualShadowMaps>, 6>> caches_;
    std::unordered_map<uint64_t, std::unique_ptr<VirtualShadowMapTemporal>> histories_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createVirtualShadowMapLightsPass()
{
    return std::make_unique<VirtualShadowMapLightsPass>();
}

} // namespace metallic::render::builtin_pass
