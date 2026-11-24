#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenLightingHistory.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenProbePlacement.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kUpdateABI = 0x4c554d454e050001ull;
constexpr uint64_t kTraceABI = 0x4c554d454e050002ull;
constexpr uint64_t kCompactMarkABI = 0x4c554d454e050003ull;
constexpr uint64_t kPriorityABI = 0x4c554d454e050004ull;
constexpr uint32_t kGlobalLightingTraceBudgetScale = 4;
constexpr uint32_t kMaxClipmaps = 6;
constexpr uint32_t kTraceGroupEdge = 8;
constexpr uint32_t kUpdateThreads = 64;
constexpr uint32_t kUpdateStateWords = 32;
constexpr uint32_t kIndirectDispatchCount = 4;
constexpr uint32_t kProbePositionStride = 2 * sizeof(uint32_t);
constexpr uint64_t kMaxAtlasBufferBytes = 128ull * 1024 * 1024;

struct RadianceCacheClipmap {
    std::array<float, 4> cornerAndCellSize;
    std::array<float, 4> traceSettings;
    std::array<uint32_t, 4> layout;
};
static_assert(sizeof(RadianceCacheClipmap) == 48);

struct RadianceCacheUpdateParams {
    ShaderSampledImage position;
    ShaderBuffer screenProbePositions, clipmaps, previousClipmaps, indirection, previousIndirection;
    ShaderBuffer freeList, probeFrames, updateState, traceData, indirectArgs;
    uint32_t gridResolution, clipmapCount, maxProbes, screenProbeCount;
    uint32_t frameNumber, historyValid, probeTraceBudget, probeResolution;
    std::array<float, 3> cameraPosition;
};
static_assert(sizeof(RadianceCacheUpdateParams) == 88);

struct CompactRadianceCacheMarkParams {
    RadianceCacheUpdateParams update;
    ShaderBuffer adaptiveCount;
    uint32_t uniformCount;
};
static_assert(sizeof(CompactRadianceCacheMarkParams) == 96 && offsetof(CompactRadianceCacheMarkParams, adaptiveCount) == 88);

struct RadianceCachePriorityParams {
    ShaderBuffer updateState;
    uint32_t probeTraceBudget, globalLightingChanged;
};
static_assert(sizeof(RadianceCachePriorityParams) == 12);

struct RadianceCacheTraceParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderBuffer clipmaps, indirection, traceData, updateState;
    ShaderBuffer traceTiles, indirectArgs, rawAtlas, filteredAtlas;
    ShaderStorageImage radianceAtlas;
    uint32_t gdfClipmapCount;
    std::array<float, 3> cameraPosition;
    float maxTraceDistance;
};
static_assert(sizeof(RadianceCacheTraceParams) == 100);

struct RadianceCacheLayout {
    uint32_t gridResolution = 48;
    uint32_t clipmapCount = 4;
    uint32_t probeResolution = 32;
    uint32_t atlasProbesPerAxis = 128;
    uint32_t traceBudget = 100;
    float worldExtent = 25.0f;
    float maxTraceDistance = 200.0f;

    uint32_t gridCells() const { return gridResolution * gridResolution * gridResolution * clipmapCount; }
    uint32_t probeCapacity() const { return atlasProbesPerAxis * atlasProbesPerAxis; }
    uint32_t borderedAtlasEdge() const { return (probeResolution + 2) * atlasProbesPerAxis; }
    uint64_t atlasTexels() const { return uint64_t(probeCapacity()) * probeResolution * probeResolution; }
    uint64_t maxTraceTiles() const
    {
        const uint32_t tilesPerAxis = 2 * probeResolution / kTraceGroupEdge;
        return uint64_t(probeCapacity()) * tilesPerAxis * tilesPerAxis;
    }
};

std::optional<RadianceCacheLayout> readLayout(const RenderGraphProperties& properties)
{
    try {
        for (const auto* name : {"gridResolution", "clipmapCount", "probeResolution", "atlasProbesPerAxis", "traceBudget"}) {
            const auto value = properties.find(name);
            if (value != properties.end() && (!value->is_number_integer() || value->get<int64_t>() < 0 ||
                    uint64_t(value->get<int64_t>()) > UINT32_MAX)) {
                return std::nullopt;
            }
        }
        RadianceCacheLayout layout{
            .gridResolution = properties.value("gridResolution", 48u),
            .clipmapCount = properties.value("clipmapCount", 4u),
            .probeResolution = properties.value("probeResolution", 32u),
            .atlasProbesPerAxis = properties.value("atlasProbesPerAxis", 128u),
            .traceBudget = properties.value("traceBudget", 100u),
            .worldExtent = properties.value("worldExtent", 25.0f),
            .maxTraceDistance = properties.value("maxTraceDistance", 200.0f),
        };
        if (layout.gridResolution < 2 || layout.gridResolution > 128 || layout.clipmapCount == 0 ||
            layout.clipmapCount > kMaxClipmaps || !std::has_single_bit(layout.probeResolution) ||
            layout.probeResolution < kTraceGroupEdge || layout.probeResolution > 64 ||
            !std::has_single_bit(layout.atlasProbesPerAxis) || layout.atlasProbesPerAxis > 128 ||
            layout.atlasTexels() * 2u * sizeof(uint32_t) > kMaxAtlasBufferBytes ||
            layout.gridCells() > UINT16_MAX * kUpdateThreads ||
            layout.traceBudget > UINT32_MAX / kGlobalLightingTraceBudgetScale ||
            !std::isfinite(layout.worldExtent) || layout.worldExtent <= 0.0f ||
            !std::isfinite(layout.maxTraceDistance) || layout.maxTraceDistance <= 0.05f || layout.maxTraceDistance > 65504.0f) {
            return std::nullopt;
        }
        return layout;
    } catch (const RenderGraphProperties::exception&) {
        return std::nullopt;
    }
}

std::array<RadianceCacheClipmap, kMaxClipmaps> makeClipmaps(const RadianceCacheLayout& layout,
    const std::array<float, 3>& eye)
{
    std::array<RadianceCacheClipmap, kMaxClipmaps> clipmaps{};
    // UE 5.7.4 FClipmapGeometry and UpdateRadianceCacheState.
    const double levelZeroCellSize = double(layout.worldExtent * 2.0f) / layout.gridResolution;
    for (uint32_t level = 0; level < layout.clipmapCount; ++level) {
        const double cellSize = std::ldexp(levelZeroCellSize, int(level));
        const double gridSize = cellSize * layout.gridResolution;
        auto& clipmap = clipmaps[level];
        for (size_t axis = 0; axis < 3; ++axis) {
            const double snapped = std::floor(double(eye[axis]) / cellSize) * cellSize;
            clipmap.cornerAndCellSize[axis] = float((snapped - 0.5 * cellSize) - gridSize / 2.0);
        }
        clipmap.cornerAndCellSize[3] = float(cellSize);
        clipmap.traceSettings = {float(std::sqrt(cellSize * cellSize + cellSize * cellSize + cellSize * cellSize)),
            1.0f, 1.5f, 0.0f};
        clipmap.layout = {layout.gridResolution, layout.clipmapCount, layout.probeResolution, layout.atlasProbesPerAxis};
    }
    return clipmaps;
}

struct CacheContent {
    uint64_t sceneIdentity, sceneContent, transform, visibility, nonSunLighting, environment;
    bool operator==(const CacheContent&) const = default;
};

struct SunLightingSample {
    const RenderWorld* world = nullptr;
    const scene::Scene* scene = nullptr;
    uint64_t sceneLifetime = 0;
    float maximum = 0.0f;
};

class LumenRadianceCachePass final : public ComputePass {
public:
    std::optional<RenderGraphExternalDomainContract> externalDomainContract() const override
    {
        return RenderGraphExternalDomainContract{
            .domainName = "render.pass.lumen-radiance-cache",
            .ownership = RenderGraphResourceOwnership::ExternallyOwned,
            .hasUndeclaredImplicitAccesses = true,
            .conservativeReason = "Persistent probe atlas, probeFrames, and freeList remain conservative; indirection history is declaratively TemporalOwned",
        };
    }
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    RadianceCacheLayout layout_;
    bool compactProbeAtlas_ = false;
    std::optional<CacheContent> previousContent_;
    std::shared_ptr<SubmissionTransaction> previousSubmission_;
    SunLightingSample recordedSunLighting_;
    std::optional<SunLightingSample> submittedSunLighting_;
    std::array<RadianceCacheClipmap, kMaxClipmaps> previousClipmaps_{};
    uint64_t previousFrame_ = 0;
    ComputeKernel place_, clear_, mark_, remap_, allocate_, priority_, select_, setupArgs_, save_;
    ComputeKernel generateTiles_, traceArgs_, trace_, filter_, borders_;
};

RenderPassReflection LumenRadianceCachePass::reflect(const RenderGraphCompileContext& context) const
{
    const auto layout = readLayout(properties()).value_or(RadianceCacheLayout{});
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    auto& position = reflection.addTextureInput("position", "Receiver world position and depth").sampledRead();
    position.format = Format::RGBA32Sfloat;
    position.matchOutputExtent = false;
    const auto compact = properties().find("compactProbeAtlas");
    if (compact != properties().end() && *compact == true) {
        reflection.addBufferInput("probePositions", "Shared compact screen probe positions")
            .buffer(0, kProbePositionStride).storageRead();
        reflection.addBufferInput("probeAdaptiveCount", "Attempted adaptive allocations for shared placement")
            .buffer(sizeof(uint32_t), sizeof(uint32_t)).storageRead();
    } else {
        auto& normal = reflection.addTextureInput("normal", "Receiver shading normal").sampledRead();
        normal.format = Format::RGBA16Sfloat;
        normal.matchOutputExtent = false;
        const uint64_t placements = uint64_t(lumenScreenProbeTiles(context.width)) *
            lumenScreenProbeTiles(context.height) * kLumenScreenProbeSlotsPerTile;
        reflection.addBufferOutput("placedProbes", "Uniform and adaptive screen probe positions")
            .buffer(placements * kProbePositionStride, kProbePositionStride).storageReadWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
    }
    for (const auto* name : {"cacheClipmaps", "previousClipmaps"}) {
        reflection.addBufferOutput(name, "Camera-centered radiance cache clipmap descriptors")
            .buffer(layout.clipmapCount * sizeof(RadianceCacheClipmap), sizeof(RadianceCacheClipmap)).transferWrite()
            .stageAccess(RenderGraphResourceAccess::BufferStorageRead);
    }
    const auto addBuffer = [&](const char* name, const char* description, uint64_t count, uint32_t stride) -> RenderGraphField& {
        return reflection.addBufferOutput(name, description).buffer(count * stride, stride).storageReadWrite();
    };
    addBuffer("cacheIndirection", "World probe cell to physical atlas slot", layout.gridCells(), 4).bufferLayout("u32");
    reflection.addBufferInput("previousIndirection", "Last frame's world probe cell to atlas slot")
        .buffer(uint64_t(layout.gridCells()) * 4u, 4).storageRead().temporalRead("indirection", -1, /*graphOwned=*/true);
    reflection.addBufferOutput("savedIndirection", "Saved world probe cell to atlas slot for next frame")
        .buffer(uint64_t(layout.gridCells()) * 4u, 4).storageWrite().temporalWrite("indirection", /*graphOwned=*/true);
    addBuffer("freeList", "Freed atlas slots", layout.probeCapacity(), 4);
    addBuffer("probeFrames", "Last traced and last used frame per slot", layout.probeCapacity(), 8);
    addBuffer("updateState", "Allocators, update budget, priority histogram and residency counters", kUpdateStateWords, 4)
        .bufferLayout("u32");
    addBuffer("traceData", "Selected world-space probes and packed tracing state", layout.probeCapacity(), 16);
    addBuffer("traceTiles", "Compacted directional trace tiles", layout.maxTraceTiles(), 8);
    addBuffer("indirectArgs", "Per-stage dispatch arguments", kIndirectDispatchCount * 3u, 4)
        .stageAccess(RenderGraphResourceAccess::BufferIndirectRead);
    addBuffer("rawAtlas", "Raw directional radiance and hit distance in FP16", layout.atlasTexels(), 8);
    addBuffer("filteredAtlas", "Spatially filtered probe radiance", layout.atlasTexels(), 8);
    auto& radiance = reflection.addTextureOutput("cacheRadiance", "Persistent bordered directional radiance atlas")
        .texture2D(layout.borderedAtlasEdge(), layout.borderedAtlasEdge()).storageWrite();
    radiance.format = Format::RGBA16Sfloat;
    radiance.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenRadianceCachePass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto layout = readLayout(properties());
    if (!layout) {
        log = "Invalid Lumen Radiance Cache clipmap layout, atlas size, distance or update budget";
        return makeError(Error::InvalidArgument);
    }
    const auto compact = properties().find("compactProbeAtlas");
    if (compact != properties().end() && !compact->is_boolean()) {
        log = "Lumen Radiance Cache compactProbeAtlas must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    device_ = context.device;
    layout_ = *layout;
    compactProbeAtlas_ = compact != properties().end() && *compact == true;
    // A cache rebuild ends residency without creating a new Sun lighting change.
    previousContent_.reset();
    const auto compileKernel = [&](const char* module, const char* entry, ParameterABI abi, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = module, .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = abi, .debugName = entry}, kernel, log);
    };
    if (!compactProbeAtlas_) {
        if (auto result = compileKernel("Features/DistanceField/LumenScreenProbeFilter", "lumenPlaceAdaptiveProbesMain",
            parameterAbi<LumenScreenProbeFilterParams>(kScreenProbeFilterABI), place_); !result) { return result; }
    }

    constexpr char update[] = "Features/DistanceField/LumenRadianceCacheUpdate";
    constexpr char trace[] = "Features/DistanceField/LumenRadianceCacheTrace";
    const auto markResult = compactProbeAtlas_ ?
        compileKernel(update, "lumenMarkCompactRadianceCacheProbesMain",
            parameterAbi<CompactRadianceCacheMarkParams>(kCompactMarkABI), mark_) :
        compileKernel(update, "lumenMarkRadianceCacheProbesMain", parameterAbi<RadianceCacheUpdateParams>(kUpdateABI), mark_);
    if (!markResult) { return markResult; }
    if (auto result = compileKernel(update, "lumenSelectRadianceCachePriorityMain",
            parameterAbi<RadianceCachePriorityParams>(kPriorityABI), priority_); !result) {
        return result;
    }
    using KernelEntry = std::pair<const char*, ComputeKernel*>;
    const KernelEntry updates[]{
        {"lumenClearRadianceCacheUpdateMain", &clear_},
        {"lumenRemapRadianceCacheProbesMain", &remap_}, {"lumenAllocateRadianceCacheProbesMain", &allocate_},
        {"lumenSelectRadianceCacheTraceProbesMain", &select_},
        {"lumenSetupRadianceCacheDispatchMain", &setupArgs_}, {"lumenSaveRadianceCacheIndirectionMain", &save_},
    };
    for (const auto& [entry, kernel] : updates) {
        if (auto result = compileKernel(update, entry, parameterAbi<RadianceCacheUpdateParams>(kUpdateABI), *kernel); !result) {
            return result;
        }
    }
    const KernelEntry traces[]{
        {"lumenRadianceCacheGenerateTilesMain", &generateTiles_},
        {"lumenRadianceCacheTraceArgsMain", &traceArgs_}, {"lumenRadianceCacheTraceMain", &trace_},
        {"lumenRadianceCacheFilterMain", &filter_}, {"lumenRadianceCacheBordersMain", &borders_},
    };
    for (const auto& [entry, kernel] : traces) {
        if (auto result = compileKernel(trace, entry, parameterAbi<RadianceCacheTraceParams>(kTraceABI), *kernel); !result) {
            return result;
        }
    }
    return {};
}

Result<> LumenRadianceCachePass::execute(RenderGraphExecutionContext& context)
{
    const auto* view = context.viewConstants();
    if (!view || view->current.upProjection[3] > 0.5f) {
        spdlog::error("Software Lumen Radiance Cache requires a perspective graph camera");
        return makeError(Error::InvalidArgument);
    }
    const auto* world = context.world();
    if (world && (world->environment().enabled ||
        world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
        spdlog::error("This Lumen Radiance Cache permutation requires skylight to be disabled");
        return makeError(Error::InvalidArgument);
    }
    const auto gdfClipmaps = lumenGlobalTraceClipmapCount(context);
    if (!gdfClipmaps) { return makeError(Error::InvalidArgument); }
    const auto* scene = world ? world->scene() : context.runtimeScene();
    const CacheContent content{scene ? scene->resourceIdentity() : 0, scene ? scene->contentRevision() : 0,
        scene ? scene->geometryTransformRevision() : 0, scene ? scene->visibilityRevision() : 0,
        world ? world->nonSunLightingRevision() : 0, world ? world->environmentRevision() : 0};
    const bool previousSubmitted = previousSubmission_ && previousSubmission_->resolved() && !previousSubmission_->cancelled();
    if (previousSubmitted) { submittedSunLighting_ = recordedSunLighting_; }
    const SunLightingSample currentSunLighting{world, scene, scene ? scene->sceneGraph().lifetimeRevision() : 0,
        world ? world->sunIrradianceMax() : 0.0f};
    const bool sameLightingScene = submittedSunLighting_ && submittedSunLighting_->world == world &&
        submittedSunLighting_->scene == scene && submittedSunLighting_->sceneLifetime == currentSunLighting.sceneLifetime;
    const float previousSunMax = sameLightingScene ? submittedSunLighting_->maximum : 0.0f;
    // UE UpdateGlobalLightingState also compares Sky Light. This software permutation has no skylight.
    const bool globalLightingChanged = world && world->worldEnvironment().astronomy.mode == environment::AstronomyMode::Manual &&
        !world->worldEnvironment().moon.enabled && lumenGlobalSunLightingChanged(previousSunMax, currentSunLighting.maximum);
    const bool historyValid = previousContent_ == content && previousFrame_ + 1u == context.frameIndex() &&
        context.temporalValid("previousIndirection") && previousSubmitted && !globalLightingChanged;
    const std::array eye{view->current.eye[0], view->current.eye[1], view->current.eye[2]};
    const auto clipmaps = makeClipmaps(layout_, eye);
    const uint64_t clipmapBytes = layout_.clipmapCount * sizeof(RadianceCacheClipmap);
    auto uploadResult = device_->createBuffer({.size = 2 * clipmapBytes, .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload});
    if (!uploadResult) { return makeError(uploadResult.error()); }
    std::shared_ptr<Buffer> upload(std::move(*uploadResult));
    auto* mapped = static_cast<std::byte*>(upload->map());
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, clipmaps.data(), clipmapBytes);
    std::memcpy(mapped + clipmapBytes, previousClipmaps_.data(), clipmapBytes);
    upload->flush();
    upload->unmap();

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    const auto buffer = [&](const char* name) -> Buffer* {
        if (std::string_view(name) == "previousIndirection" || std::string_view(name) == "savedIndirection") {
            return context.temporalBuffer(name).buffer.buffer();
        }
        return context.outputBuffer(name).buffer();
    };
    const auto bind = [&](const char* name) { return writer.buffer(buffer(name)); };
    if (buffer("previousIndirection") == nullptr || buffer("savedIndirection") == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    const auto position = context.inputTexture("position");
    const uint32_t width = position.desc().width;
    const uint32_t height = position.desc().height;
    const uint32_t probeWidth = lumenScreenProbeTiles(width);
    const uint32_t probeHeight = lumenScreenProbeTiles(height);
    if (width != uint32_t(view->current.viewport[1]) || height != uint32_t(view->current.viewport[2])) {
        spdlog::error("Lumen Radiance Cache guides must match the graph render extent");
        return makeError(Error::InvalidArgument);
    }
    const uint32_t uniformCount = probeWidth * probeHeight;
    const uint32_t denseCount = uniformCount * kLumenScreenProbeSlotsPerTile;
    const auto probePositions = compactProbeAtlas_ ? context.inputBuffer("probePositions") : context.outputBuffer("placedProbes");
    const uint64_t probeCount = probePositions.desc().size / kProbePositionStride;
    if (probePositions.desc().structureStride != kProbePositionStride ||
        probePositions.desc().size % kProbePositionStride != 0 ||
        probeCount < (compactProbeAtlas_ ? uniformCount : denseCount) || probeCount > denseCount) {
        spdlog::error("Lumen Radiance Cache {} probe positions are incompatible with receiver extent {}x{} "
            "(graph extent {}x{}, {} bytes with stride {}, expected {} to {} probes)",
            compactProbeAtlas_ ? "shared" : "dense", width, height, context.width(), context.height(),
            probePositions.desc().size, probePositions.desc().structureStride, compactProbeAtlas_ ? uniformCount : denseCount, denseCount);
        return makeError(Error::InvalidArgument);
    }
    const auto positionHandle = writer.sampledImageHandle(position.view());
    const auto screenProbeHandle = writer.buffer(probePositions.buffer());
    EncodedParameters placement;
    if (!compactProbeAtlas_) {
        const auto encodedPlacement = writer.encode(LumenScreenProbeFilterParams{
            positionHandle, writer.sampledImageHandle(context.inputTexture("normal").view()),
            screenProbeHandle, {}, {}, {}, {}, width, height, probeWidth, probeHeight,
            uint32_t(context.frameIndex())}, kScreenProbeFilterABI);
        if (!encodedPlacement) { return makeError(encodedPlacement.error()); }
        placement = *encodedPlacement;
    }
    const RadianceCacheUpdateParams updateParams{
        positionHandle, screenProbeHandle, bind("cacheClipmaps"), bind("previousClipmaps"),
        bind("cacheIndirection"), bind("previousIndirection"), bind("freeList"), bind("probeFrames"),
        bind("updateState"), bind("traceData"), bind("indirectArgs"),
        layout_.gridResolution, layout_.clipmapCount, layout_.probeCapacity(), uint32_t(probeCount),
        uint32_t(context.frameIndex() + 1u), historyValid, layout_.traceBudget, layout_.probeResolution, eye};
    const auto update = writer.encode(updateParams, kUpdateABI);
    if (!update) { return makeError(update.error()); }
    // UE boosts the priority histogram only. New-probe downsampling still uses the input budget.
    const uint32_t priorityBudget = layout_.traceBudget * (globalLightingChanged ? kGlobalLightingTraceBudgetScale : 1u);
    const auto priority = writer.encode(RadianceCachePriorityParams{
        updateParams.updateState, priorityBudget, static_cast<uint32_t>(globalLightingChanged)}, kPriorityABI);
    if (!priority) { return makeError(priority.error()); }
    const auto mark = compactProbeAtlas_ ? writer.encode(CompactRadianceCacheMarkParams{updateParams,
        writer.buffer(context.inputBuffer("probeAdaptiveCount").buffer()), uniformCount}, kCompactMarkABI) : update;
    if (!mark) { return makeError(mark.error()); }
    auto saveParams = updateParams;
    saveParams.previousIndirection = bind("savedIndirection");
    const auto saveUpdate = writer.encode(saveParams, kUpdateABI);
    if (!saveUpdate) { return makeError(saveUpdate.error()); }
    const auto trace = writer.encode(RadianceCacheTraceParams{
        bindLumenGlobalTrace(context, writer), bind("cacheClipmaps"), bind("cacheIndirection"), bind("traceData"),
        bind("updateState"), bind("traceTiles"), bind("indirectArgs"),
        bind("rawAtlas"), bind("filteredAtlas"), writer.storageImageHandle(context.outputTexture("cacheRadiance").view()),
        *gdfClipmaps, eye, layout_.maxTraceDistance}, kTraceABI);
    if (!trace) { return makeError(trace.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse uploadUses[]{{"cacheClipmaps", Access::BufferTransferWrite},
        {"previousClipmaps", Access::BufferTransferWrite}};
    const RenderGraphStageUse placeUses[]{{"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"placedProbes", Access::BufferStorageWrite}};
    std::vector<RenderGraphStage> stages;
    stages.reserve(15);
    stages.push_back({"Upload world probe clipmaps", uploadUses, [&](CommandBuffer& cmd) -> Result<> {
        if (auto result = cmd.retainResource(upload); !result) { return result; }
        cmd.hostWriteBarrier();
        for (size_t index = 0; index < std::size(uploadUses); ++index) {
            const auto source = upload->slice({index * clipmapBytes, clipmapBytes});
            const auto destination = buffer(index == 0 ? "cacheClipmaps" : "previousClipmaps")->slice({});
            if (!source || !destination) { return makeError(Error::InvalidArgument); }
            if (auto result = cmd.copyBuffer(*source, *destination); !result) { return result; }
        }
        return {};
    }});
    if (!compactProbeAtlas_) {
        stages.push_back({"Place screen probes", placeUses, [&](CommandBuffer& cmd) {
            return place_.dispatch(cmd, placement, probeWidth, probeHeight, 1);
        }});
    }
    const auto groups = (layout_.gridCells() + kUpdateThreads - 1u) / kUpdateThreads;
    const uint32_t markGroups = (uint32_t(probeCount) + kUpdateThreads - 1u) / kUpdateThreads;
    using Use = RenderGraphStageUse;
    constexpr Access read = Access::BufferStorageRead;
    constexpr Access write = Access::BufferStorageWrite;
    constexpr Access rw = Access::BufferStorageReadWrite;
    constexpr Access indirect = Access::BufferIndirectRead;
    constexpr Access sample = Access::TextureSampleRead;

    // Store the owners of the stage-use spans until all stages have been recorded.
    std::array<std::vector<Use>, 13> stageUses;
    size_t useIndex = 0;
    const auto dispatch = [&](const char* name, ComputeKernel& kernel, const EncodedParameters& encoded,
        uint32_t groupCount, std::initializer_list<Use> uses) {
        auto& ownedUses = stageUses[useIndex++];
        ownedUses.assign(uses);
        stages.push_back({name, ownedUses, [&kernel, &encoded, groupCount](CommandBuffer& cmd) {
            return kernel.dispatch(cmd, encoded, groupCount, 1, 1);
        }});
    };
    const auto dispatchIndirect = [&](const char* name, ComputeKernel& kernel, uint32_t argumentIndex,
        std::initializer_list<Use> uses) {
        auto& ownedUses = stageUses[useIndex++];
        ownedUses.assign(uses);
        ownedUses.push_back({"indirectArgs", indirect});
        stages.push_back({name, ownedUses, [&, argumentIndex, kernelPtr = &kernel](CommandBuffer& cmd) -> Result<> {
            constexpr uint32_t stride = 3 * sizeof(uint32_t);
            const auto args = buffer("indirectArgs")->slice({argumentIndex * stride, stride});
            if (!args) { return makeError(args.error()); }
            return kernelPtr->dispatchIndirect(cmd, *trace, *args);
        }});
    };
    dispatch("Clear world probe requests", clear_, *update, groups, {{"cacheIndirection", write}, {"updateState", rw}});
    if (compactProbeAtlas_) {
        dispatch("Mark world probes from screen probes", mark_, *mark, markGroups,
            {{"cacheClipmaps", read}, {"probePositions", read}, {"probeAdaptiveCount", read}, {"position", sample},
                {"cacheIndirection", rw}, {"updateState", rw}});
    } else {
        dispatch("Mark world probes from screen probes", mark_, *mark, markGroups,
            {{"cacheClipmaps", read}, {"placedProbes", read}, {"position", sample}, {"cacheIndirection", rw}, {"updateState", rw}});
    }
    if (historyValid) {
        dispatch("Remap persistent world probes", remap_, *update, groups,
            {{"cacheClipmaps", read}, {"previousClipmaps", read}, {"previousIndirection", read},
                {"cacheIndirection", rw}, {"updateState", rw}, {"probeFrames", rw}, {"freeList", write}});
    }
    dispatch("Allocate requested world probes", allocate_, *update, groups,
        {{"cacheClipmaps", read}, {"cacheIndirection", rw}, {"updateState", rw}, {"probeFrames", rw}, {"freeList", read}});
    dispatch("Select probe priority budget", priority_, *priority, 1, {{"updateState", rw}});
    dispatch("Select probes to trace", select_, *update, groups,
        {{"cacheClipmaps", read}, {"cacheIndirection", read}, {"updateState", rw}, {"probeFrames", rw}, {"traceData", write}});
    dispatch("Set up probe dispatches", setupArgs_, *update, 1, {{"updateState", rw}, {"indirectArgs", write}});
    dispatchIndirect("Generate directional trace tiles", generateTiles_, 0,
        {{"cacheClipmaps", read}, {"traceData", read}, {"traceTiles", write}, {"updateState", rw}});
    dispatch("Set up directional trace dispatch", traceArgs_, *trace, 1, {{"updateState", read}, {"indirectArgs", rw}});
    dispatchIndirect("Trace world probe radiance", trace_, 3,
        {{"cacheClipmaps", read}, {"traceData", read}, {"traceTiles", read}, {"updateState", read}, {"rawAtlas", write},
            {"pageTable", sample}, {"pageAtlas", sample}, {"coarseMip", sample}, {"domain", read},
            {"objectGrid", read}, {"objectGridScene", read}, {"cards", read}, {"cardScene", read},
            {"cardDepth", sample}, {"cardLighting", sample}});
    dispatchIndirect("Filter directional probe radiance", filter_, 1,
        {{"cacheClipmaps", read}, {"cacheIndirection", read}, {"traceData", read},
            {"rawAtlas", read}, {"filteredAtlas", write}});
    dispatchIndirect("Fix up octahedral atlas borders", borders_, 2,
        {{"cacheClipmaps", read}, {"traceData", read}, {"filteredAtlas", read}, {"cacheRadiance", Access::TextureStorageWrite}});
    dispatch("Save world probe indirection", save_, *saveUpdate, groups,
        {{"cacheIndirection", read}, {"savedIndirection", write}});

    if (auto result = context.executeStages(stages); !result) { return result; }
    context.setTemporalWritten("savedIndirection", true);
    auto submission = std::make_shared<SubmissionTransaction>([] {}, [] {});
    if (auto result = commands.addSubmissionTransaction(submission); !result) { return result; }
    previousSubmission_ = std::move(submission);
    recordedSunLighting_ = currentSunLighting;
    previousContent_ = content;
    previousClipmaps_ = clipmaps;
    previousFrame_ = context.frameIndex();
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenRadianceCachePass()
{
    return std::make_unique<LumenRadianceCachePass>();
}

} // namespace metallic::render::builtin_pass
