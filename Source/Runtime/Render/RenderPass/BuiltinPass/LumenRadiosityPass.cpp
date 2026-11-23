#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/Subsystem/GPUSceneLightFrustum.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kRadiosityABI = 0x4c554d454e040004ull;
constexpr uint64_t kRadiosityPersistentABI = 0x4c554d454e040005ull;
constexpr uint64_t kRadiosityScheduleABI = 0x4c554d454e040003ull;
constexpr uint32_t kProbeSpacing = 4;
constexpr uint32_t kScheduleStateWords = 24;
constexpr uint32_t kIndirectDispatchCount = 3;
constexpr uint32_t kDefaultRadiosityUpdateFactor = 64;
constexpr std::array kSHNames{"probeSHRed", "probeSHGreen", "probeSHBlue"};
constexpr std::array kHistoryNames{"indirectIrradiance", "lighting"};

struct LumenRadiosityParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderSampledImage cardNormal, cardAlbedo, cardEmissive, cardDirectLighting, previousIndirect;
    ShaderSampledImage traceRadiance, filteredRadiance;
    ShaderBuffer probeSHRed, probeSHGreen, probeSHBlue;
    ShaderStorageImage outputTrace, outputFiltered;
    ShaderBuffer outputSHRed, outputSHGreen, outputSHBlue;
    ShaderStorageImage outputIndirect, finalLighting, currentIndirect, currentLighting;
    uint32_t atlasSize, tileResolution, clipmapCount, historyValid, temporalIndex;
    float maxTraceDistance;
    uint32_t viewFrameIndex, traceValid;
};
static_assert(sizeof(LumenRadiosityParams) == 152 && offsetof(LumenRadiosityParams, atlasSize) == 120);

struct PersistentRadiosityParams {
    LumenRadiosityParams lighting;
    ShaderBuffer scheduledCards, refreshCards, updateState;
};
static_assert(sizeof(PersistentRadiosityParams) == 164 &&
    offsetof(PersistentRadiosityParams, scheduledCards) == 152);

struct RadiosityScheduleParams {
    ShaderBuffer cards, cardScene, domains, cardUpdates;
    ShaderBuffer cardState, scheduledCards, refreshCards, updateState, indirectArgs, updateStatistics;
    uint32_t capacity, maxCardsPerFrame, tileResolution, cacheInitialized, frameIndex, hasCamera;
    std::array<float, 4> cameraEye;
    std::array<std::array<float, 4>, 6> cameraPlanes;
};
static_assert(sizeof(RadiosityScheduleParams) == 176 && offsetof(RadiosityScheduleParams, cameraEye) == 64);

struct RadiosityLayout {
    uint32_t atlasSize = 256;
    uint32_t tileResolution = 32;
    float maxTraceDistance = 200.0f;
    bool persistentCards = false;
    uint32_t maxCardsPerFrame = 1;

    uint32_t capacity() const
    {
        const uint32_t tiles = atlasSize / tileResolution;
        return tiles * tiles;
    }
};

std::optional<RadiosityLayout> readLayout(const RenderGraphProperties& properties)
{
    try {
        RadiosityLayout layout{properties.value("atlasSize", 256u), properties.value("tileResolution", 32u),
            properties.value("maxTraceDistance", 200.0f), properties.value("persistentCards", false)};
        if (layout.tileResolution < kProbeSpacing || layout.tileResolution % kProbeSpacing != 0 ||
            layout.atlasSize < layout.tileResolution || layout.atlasSize > 2048 ||
            layout.atlasSize % layout.tileResolution != 0 || !std::isfinite(layout.maxTraceDistance) ||
            layout.maxTraceDistance <= 0.1f) {
            return std::nullopt;
        }
        // UE LumenSceneLighting.cpp SetLightingUpdateAtlasSize, with one tile per physical page.
        const float scale = 1.0f / std::sqrt(float(kDefaultRadiosityUpdateFactor));
        const uint32_t updateAtlasSize = uint32_t(float(layout.atlasSize) * scale + 0.5f);
        const uint32_t updateTiles = std::max(1u, (updateAtlasSize + layout.tileResolution - 1u) / layout.tileResolution);
        const auto budget = properties.find("maxCardsPerFrame");
        if (budget != properties.end() && (!budget->is_number_integer() ||
            budget->get<uint64_t>() == 0 || budget->get<uint64_t>() > layout.capacity())) {
            return std::nullopt;
        }
        layout.maxCardsPerFrame = properties.value("maxCardsPerFrame", updateTiles * updateTiles);
        return layout;
    } catch (const RenderGraphProperties::exception&) {
        return std::nullopt;
    }
}

struct RadiosityContent {
    uint64_t sceneIdentity, sceneContent, transform, visibility, lighting, environment;
    bool operator==(const RadiosityContent&) const = default;
};

class LumenRadiosityPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

private:
    LumenRadiosityParams bindParams(RenderGraphExecutionContext& context, ParameterWriter& writer,
        uint32_t clipmapCount) const;
    Result<> executePersistent(RenderGraphExecutionContext& context, ParameterWriter& writer, uint32_t clipmapCount);

    Device* device_ = nullptr;
    RadiosityLayout layout_;
    ComputeKernel refreshDenseLighting_, trace_, filter_, convertToSH_, integrate_;
    ComputeKernel clearSchedule_, buildPriority_, selectPriority_, buildCardLists_, finishSchedule_, refreshCards_;
    std::shared_ptr<uint64_t> initializedAtlas_;
    HistoryResourceManager history_;
    std::optional<RadiosityContent> lastContent_;
    uint64_t lastFrame_ = 0;
    uint64_t lastLightingRevision_ = 0;
    uint32_t lastTemporalIndex_ = 0;
};

RenderPassReflection LumenRadiosityPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    const auto layout = readLayout(properties()).value_or(RadiosityLayout{});
    const uint32_t edge = layout.atlasSize;
    reflectLumenGlobalTraceInputs(reflection);
    for (std::string_view name : {"cardNormal", "cardAlbedo", "cardEmissive", "cardDirectLighting"}) {
        auto& input = reflection.addTextureInput(std::string(name), "Surface Cache input for radiosity")
            .texture2D(edge, edge).sampledRead();
        input.matchOutputExtent = false;
        input.format = name == "cardDirectLighting" ? Format::RGBA32Sfloat :
            (name == "cardEmissive" ? Format::RGBA16Sfloat : Format::RGBA8Unorm);
    }
    for (auto name : {"traceRadiance", "filteredRadiance"}) {
        auto& output = reflection.addTextureOutput(name, "Hemisphere trace radiance")
            .texture2D(edge, edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead);
        output.format = Format::RGBA16Sfloat;
        if (!layout.persistentCards) { output.transient(RenderGraphInitialization::FullOverwrite); }
    }
    const uint32_t probesPerAxis = edge / kProbeSpacing;
    constexpr uint32_t packedSHStride = 2u * sizeof(uint32_t);
    for (auto name : kSHNames) {
        auto& output = reflection.addBufferOutput(name, "Two-band spherical harmonics per card probe packed as four half floats")
            .buffer(uint64_t(probesPerAxis) * probesPerAxis * packedSHStride, packedSHStride).storageReadWrite();
        if (!layout.persistentCards) { output.transient(RenderGraphInitialization::FullOverwrite); }
    }
    for (auto name : kHistoryNames) {
        auto& output = reflection.addTextureOutput(name, "Scene-linear Surface Cache lighting")
            .texture2D(edge, edge).storageWrite();
        output.format = Format::RGBA16Sfloat;
        output.colorEncoding = DisplayColorEncoding::SceneLinear;
        if (!layout.persistentCards) {
            output.transient(RenderGraphInitialization::FullOverwrite);
        } else if (name == kHistoryNames[0]) {
            output.storageReadWrite();
        } else {
            output.stageAccess(RenderGraphResourceAccess::TextureSampleRead);
        }
    }
    if (layout.persistentCards) {
        reflection.addBufferInput("cardUpdates", "Surface Cache capture and direct-light flags per slot")
            .buffer(uint64_t(layout.capacity()) * sizeof(uint32_t), sizeof(uint32_t)).storageRead();
        const auto listSize = uint64_t(layout.capacity()) * 2u * sizeof(uint32_t);
        for (const auto name : {"cardState", "scheduledCards", "refreshCards"}) {
            reflection.addBufferOutput(name, "Fixed-atlas radiosity slot state or compacted slot list")
                .buffer(listSize, 2u * sizeof(uint32_t)).storageReadWrite();
        }
        reflection.addBufferOutput("updateState", "Radiosity update counters and priority histogram")
            .buffer(kScheduleStateWords * sizeof(uint32_t), sizeof(uint32_t)).storageReadWrite().bufferLayout("u32");
        reflection.addBufferOutput("indirectArgs", "Refresh, trace and SH work group counts")
            .buffer(kIndirectDispatchCount * 3u * sizeof(uint32_t), sizeof(uint32_t)).storageWrite()
            .stageAccess(RenderGraphResourceAccess::BufferIndirectRead);
        reflection.addBufferOutput("updateStatistics", "Updated resident, reset or retired, resident, budget")
            .buffer(4u * sizeof(uint32_t), sizeof(uint32_t)).storageWrite().bufferLayout("u32");
    }
    return reflection;
}

Result<> LumenRadiosityPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto layout = readLayout(properties());
    if (!layout || !context.runtimeScene) {
        log = "Lumen Radiosity requires a world scene, a tile divisible by four, a matching Surface Cache atlas, "
            "and any maxCardsPerFrame setting within the atlas capacity";
        return makeError(Error::InvalidArgument);
    }
    layout_ = *layout;
    device_ = context.device;
    lastContent_.reset();
    if (layout_.persistentCards) {
        if (!initializedAtlas_) { initializedAtlas_ = std::make_shared<uint64_t>(0); }
    } else {
        initializedAtlas_.reset();
        if (auto result = history_.initialize(*device_); !result) { return result; }
    }
    const auto compileKernel = [&](const char* module, const char* entry, ComputeKernel& kernel) {
        constexpr SlangMacroDefine persistentDefine[]{{"METALLIC_LUMEN_PERSISTENT_CARDS", "1"}};
        const auto defines = layout_.persistentCards ? std::span<const SlangMacroDefine>{persistentDefine} :
            std::span<const SlangMacroDefine>{};
        const auto abi = layout_.persistentCards ? parameterAbi<PersistentRadiosityParams>(kRadiosityPersistentABI) :
            parameterAbi<LumenRadiosityParams>(kRadiosityABI);
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = module, .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .macroDefines = defines},
            {.parameters = abi, .debugName = entry}, kernel, log);
    };
    constexpr char traceModule[] = "Features/DistanceField/LumenRadiosityTrace";
    constexpr char integrateModule[] = "Features/DistanceField/LumenRadiosityIntegrate";
    if (auto result = compileKernel(traceModule, "lumenTraceRadiosityMain", trace_); !result) { return result; }
    if (auto result = compileKernel(traceModule, "lumenFilterRadiosityMain", filter_); !result) { return result; }
    if (auto result = compileKernel(integrateModule, "lumenRadiosityConvertToSHMain", convertToSH_); !result) { return result; }
    if (auto result = compileKernel(integrateModule, "lumenRadiosityIntegrateMain", integrate_); !result) { return result; }
    if (!layout_.persistentCards) {
        return compileKernel(integrateModule, "lumenRadiosityRefreshDenseLightingMain", refreshDenseLighting_);
    }
    if (auto result = compileKernel(integrateModule, "lumenRadiosityRefreshCardsMain", refreshCards_); !result) { return result; }
    constexpr char scheduleModule[] = "Features/DistanceField/LumenRadiositySchedule";
    const std::pair<const char*, ComputeKernel*> entries[]{
        {"lumenRadiosityClearScheduleMain", &clearSchedule_},
        {"lumenRadiosityBuildPriorityMain", &buildPriority_},
        {"lumenRadiositySelectPriorityMain", &selectPriority_},
        {"lumenRadiosityBuildCardListsMain", &buildCardLists_},
        {"lumenRadiosityFinishScheduleMain", &finishSchedule_},
    };
    for (const auto& [entry, kernel] : entries) {
        if (auto result = ShaderRegistry::instance().getComputeKernel(*device_,
                {.moduleName = scheduleModule, .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                {.parameters = parameterAbi<RadiosityScheduleParams>(kRadiosityScheduleABI), .debugName = entry},
                *kernel, log); !result) {
            return result;
        }
    }
    return {};
}

Result<> LumenRadiosityPass::execute(RenderGraphExecutionContext& context)
{
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }

    const auto edge = layout_.atlasSize;
    for (auto name : {"cardDepth", "cardLighting", "cardNormal", "cardAlbedo", "cardEmissive", "cardDirectLighting"}) {
        const auto& desc = context.inputTexture(name).desc();
        if (desc.width != edge || desc.height != edge) {
            spdlog::error("Lumen Radiosity input {} must match atlasSize {}", name, edge);
            return makeError(Error::InvalidArgument);
        }
    }
    if (context.inputBuffer("cards").desc().size != uint64_t(layout_.capacity()) * kLumenGlobalTraceCardStride) {
        spdlog::error("Lumen Radiosity tileResolution must match Surface Cache card capacity");
        return makeError(Error::InvalidArgument);
    }

    const auto* world = context.world();
    const auto* scene = world ? world->scene() : context.runtimeScene();
    if (world && (world->environment().enabled ||
        world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
        spdlog::error("The opaque Lumen Radiosity GDF branch requires skylight to be disabled");
        return makeError(Error::InvalidArgument);
    }
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    if (layout_.persistentCards) {
        return executePersistent(context, writer, *clipmapCount);
    }

    // Card history is world-space. Camera cuts affect the screen histories only.
    const RadiosityContent content{scene->resourceIdentity(), scene->contentRevision(),
        scene->geometryTransformRevision(), scene->visibilityRevision(), world ? world->nonSunLightingRevision() : 0,
        world ? world->environmentRevision() : 0};
    const uint64_t lightingRevision = world ? world->lightingRevision() : 0;
    history_.beginFrame(context.frameIndex());
    // Match UE's direct-only bootstrap. Later feedback invalidation can still
    // trace the freshly relit Surface Cache supplied by the graph.
    const bool traceValid = history_.texture(kHistoryNames[1], HistorySlot::Previous).valid;
    if (lastContent_ != content || lastFrame_ + 1 != context.frameIndex()) { history_.invalidateAll(); }
    const TextureDesc historyDesc{
        .type = TextureType::Texture2D, .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
        .format = Format::RGBA16Sfloat, .width = edge, .height = edge, .depth = 1, .mipCount = 1, .layerCount = 1,
        .memoryLocation = MemoryLocation::Device};
    for (const auto name : kHistoryNames) {
        if (auto result = history_.ensureTexture(name, historyDesc, TextureViewDesc{.format = historyDesc.format}); !result) {
            return result;
        }
    }
    const auto previousIndirect = history_.texture(kHistoryNames[0], HistorySlot::Previous);
    const auto previousLighting = history_.texture(kHistoryNames[1], HistorySlot::Previous);
    const auto currentIndirect = history_.texture(kHistoryNames[0], HistorySlot::Current);
    const auto currentLighting = history_.texture(kHistoryNames[1], HistorySlot::Current);
    const bool historyValid = previousIndirect.valid && previousLighting.valid;
    const bool refreshLighting = historyValid && lastLightingRevision_ != lightingRevision;
    const uint32_t temporalIndex = historyValid ? lastTemporalIndex_ + 1u : 1u;

    auto params = bindParams(context, writer, *clipmapCount);
    if (historyValid) {
        params.scene.cardLighting = writer.sampledImageHandle(refreshLighting ? currentLighting.view : previousLighting.view);
        params.previousIndirect = writer.sampledImageHandle(previousIndirect.view);
    }
    params.currentIndirect = writer.storageImageHandle(currentIndirect.view);
    params.currentLighting = writer.storageImageHandle(currentLighting.view);
    params.historyValid = historyValid;
    params.traceValid = traceValid;
    params.temporalIndex = temporalIndex;
    const auto encoded = writer.encode(params, kRadiosityABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse refreshUses[]{{"cardDepth", Access::TextureSampleRead},
        {"cardAlbedo", Access::TextureSampleRead}, {"cardEmissive", Access::TextureSampleRead},
        {"cardDirectLighting", Access::TextureSampleRead}, {"previousIndirect", Access::TextureSampleRead},
        {"currentLighting", Access::TextureStorageWrite}};
    const char* traceSource = refreshLighting ? "currentLighting" : historyValid ? "previousLighting" : "cardLighting";
    const RenderGraphStageUse traceUses[] = {{"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead}, {"objectGrid", Access::BufferStorageRead},
        {"objectGridScene", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead}, {"cards", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardNormal", Access::TextureSampleRead},
        {"traceRadiance", Access::TextureStorageWrite},
        {traceSource, Access::TextureSampleRead}};
    const RenderGraphStageUse filterUses[] = {{"cards", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardNormal", Access::TextureSampleRead},
        {"traceRadiance", Access::TextureSampleRead}, {"filteredRadiance", Access::TextureStorageWrite}};
    const RenderGraphStageUse shUses[] = {{"cards", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
        {"cardNormal", Access::TextureSampleRead}, {"filteredRadiance", Access::TextureSampleRead},
        {"probeSHRed", Access::BufferStorageWrite}, {"probeSHGreen", Access::BufferStorageWrite},
        {"probeSHBlue", Access::BufferStorageWrite}};
    const RenderGraphStageUse integrateUses[] = {{"cards", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
        {"cardNormal", Access::TextureSampleRead}, {"cardAlbedo", Access::TextureSampleRead},
        {"cardEmissive", Access::TextureSampleRead}, {"cardDirectLighting", Access::TextureSampleRead},
        {"probeSHRed", Access::BufferStorageRead}, {"probeSHGreen", Access::BufferStorageRead}, {"probeSHBlue", Access::BufferStorageRead},
        {"indirectIrradiance", Access::TextureStorageWrite}, {"lighting", Access::TextureStorageWrite},
        {"currentIndirect", Access::TextureStorageWrite}, {"currentLighting", Access::TextureStorageWrite},
        {"previousIndirect", Access::TextureSampleRead}};
    const auto import = [](std::string_view name, const HistoryTextureRef& texture) {
        return RenderGraphTextureImport{name, texture.texture, texture.view, texture.state, ResourceState::ShaderRead};
    };
    const RenderGraphTextureImport imports[] = {import("currentIndirect", currentIndirect), import("currentLighting", currentLighting),
        import("previousIndirect", previousIndirect), import("previousLighting", previousLighting)};
    const uint32_t groups = (edge + 7u) / 8u;
    const uint32_t probeGroups = (edge / kProbeSpacing + 7u) / 8u;
    const RenderGraphStage stages[] = {
        {"Combine fresh card direct lighting with retained irradiance", refreshUses,
            [&](CommandBuffer& cmd) { return refreshDenseLighting_.dispatch(cmd, *encoded, groups, groups, 1); }},
        {"Trace card probes", traceUses, [&](CommandBuffer& cmd) { return trace_.dispatch(cmd, *encoded, groups, groups, 1); }},
        {"Filter card probes", filterUses, [&](CommandBuffer& cmd) { return filter_.dispatch(cmd, *encoded, groups, groups, 1); }},
        {"Convert card probes to SH", shUses,
            [&](CommandBuffer& cmd) { return convertToSH_.dispatch(cmd, *encoded, probeGroups, probeGroups, 1); }},
        {"Accumulate card irradiance and combine outgoing lighting",
            std::span{integrateUses}.first(std::size(integrateUses) - (historyValid ? 0u : 1u)),
            [&](CommandBuffer& cmd) { return integrate_.dispatch(cmd, *encoded, groups, groups, 1); }},
    };
    const size_t importCount = historyValid ? (refreshLighting ? 3u : 4u) : 2u;
    if (auto result = context.executeStages(std::span{stages}.subspan(refreshLighting ? 0u : 1u), {},
            std::span{imports}.first(importCount)); !result) {
        return result;
    }
    for (const auto name : kHistoryNames) {
        auto result = history_.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
        if (!result) { return result; }
    }
    lastContent_ = content;
    lastFrame_ = context.frameIndex();
    lastLightingRevision_ = lightingRevision;
    lastTemporalIndex_ = temporalIndex;
    return {};
}

LumenRadiosityParams LumenRadiosityPass::bindParams(RenderGraphExecutionContext& context, ParameterWriter& writer,
    uint32_t clipmapCount) const
{
    const auto sampled = [&](const char* name) { return writer.sampledImageHandle(context.inputTexture(name).view()); };
    const auto readOutput = [&](const char* name) { return writer.sampledImageHandle(context.outputTexture(name).view()); };
    const auto writeOutput = [&](const char* name) { return writer.storageImageHandle(context.outputTexture(name).view()); };
    const auto outputBuffer = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
    LumenRadiosityParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    params.cardNormal = sampled("cardNormal");
    params.cardAlbedo = sampled("cardAlbedo");
    params.cardEmissive = sampled("cardEmissive");
    params.cardDirectLighting = sampled("cardDirectLighting");
    params.traceRadiance = readOutput("traceRadiance");
    params.filteredRadiance = readOutput("filteredRadiance");
    params.probeSHRed = outputBuffer(kSHNames[0]);
    params.probeSHGreen = outputBuffer(kSHNames[1]);
    params.probeSHBlue = outputBuffer(kSHNames[2]);
    params.outputTrace = writeOutput("traceRadiance");
    params.outputFiltered = writeOutput("filteredRadiance");
    params.outputSHRed = params.probeSHRed;
    params.outputSHGreen = params.probeSHGreen;
    params.outputSHBlue = params.probeSHBlue;
    params.outputIndirect = writeOutput(kHistoryNames[0]);
    params.finalLighting = writeOutput(kHistoryNames[1]);
    params.atlasSize = layout_.atlasSize;
    params.tileResolution = layout_.tileResolution;
    params.clipmapCount = clipmapCount;
    params.maxTraceDistance = layout_.maxTraceDistance;
    params.viewFrameIndex = static_cast<uint32_t>(context.frameIndex());
    return params;
}

Result<> LumenRadiosityPass::executePersistent(RenderGraphExecutionContext& context, ParameterWriter& writer,
    uint32_t clipmapCount)
{
    auto& commands = context.commandBuffer();
    if (!RenderFrameContext::from(commands)) {
        spdlog::error("Persistent Lumen Radiosity requires a tracked render frame");
        return makeError(Error::Unsupported);
    }

    const auto outputBuffer = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
    const auto atlasAllocation = context.outputTexture("indirectIrradiance").texture()->memoryInfo().allocationId;
    const bool cacheInitialized = *initializedAtlas_ == atlasAllocation;
    PersistentRadiosityParams params{
        .lighting = bindParams(context, writer, clipmapCount),
        .scheduledCards = outputBuffer("scheduledCards"),
        .refreshCards = outputBuffer("refreshCards"),
        .updateState = outputBuffer("updateState"),
    };
    params.lighting.scene.cardLighting = writer.sampledImageHandle(context.outputTexture("lighting").view());
    // Fresh/reset cards use the direct-only prepass until they have a valid phase.
    params.lighting.historyValid = 1u;
    params.lighting.traceValid = 1u;
    RadiosityScheduleParams schedule{
        .cards = params.lighting.scene.cards,
        .cardScene = params.lighting.scene.cardScene,
        .domains = params.lighting.scene.domains,
        .cardUpdates = writer.buffer(context.inputBuffer("cardUpdates").buffer()),
        .cardState = outputBuffer("cardState"),
        .scheduledCards = params.scheduledCards,
        .refreshCards = params.refreshCards,
        .updateState = params.updateState,
        .indirectArgs = outputBuffer("indirectArgs"),
        .updateStatistics = outputBuffer("updateStatistics"),
        .capacity = layout_.capacity(),
        .maxCardsPerFrame = layout_.maxCardsPerFrame,
        .tileResolution = layout_.tileResolution,
        .cacheInitialized = cacheInitialized,
        // UE reserves zero for a card that has never received lighting.
        .frameIndex = static_cast<uint32_t>(context.frameIndex() % UINT32_MAX) + 1u,
    };
    if (const auto* view = context.viewConstants()) {
        schedule.hasCamera = 1u;
        std::ranges::copy(view->current.eye, schedule.cameraEye.begin());
        const auto asVector = [](const float (&xyz)[4]) { return float3(xyz[0], xyz[1], xyz[2]); };
        const auto& camera = view->current;
        const auto planes = gpuSceneLightFrustumPlanes(asVector(camera.eye), asVector(camera.center),
            asVector(camera.upProjection), camera.viewport[0], camera.viewport[3],
            camera.clipOrtho[0], camera.clipOrtho[1], camera.upProjection[3] > 0.5f ? camera.clipOrtho[2] : 0.0f);
        for (size_t i = 0; i < planes.size(); ++i) {
            const auto& plane = planes[i];
            const float length = std::sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);
            const float scale = length > 0.0f ? -1.0f / length : 0.0f;
            schedule.cameraPlanes[i] = {plane.x * scale, plane.y * scale, plane.z * scale, plane.w * scale};
        }
    }

    const auto encodedSchedule = writer.encode(schedule, kRadiosityScheduleABI);
    if (!encodedSchedule) { return makeError(encodedSchedule.error()); }
    const auto encodedLighting = writer.encode(params, kRadiosityPersistentABI);
    if (!encodedLighting) { return makeError(encodedLighting.error()); }
    const auto indirect = [&](ComputeKernel& kernel, uint32_t argumentIndex, CommandBuffer& cmd) -> Result<> {
        constexpr uint32_t stride = 3u * sizeof(uint32_t);
        const auto args = context.outputBuffer("indirectArgs").buffer()->slice({argumentIndex * stride, stride});
        if (!args) { return makeError(args.error()); }
        return kernel.dispatchIndirect(cmd, *encodedLighting, *args);
    };

    using Access = RenderGraphResourceAccess;
    using Use = RenderGraphStageUse;
    constexpr Access read = Access::BufferStorageRead;
    constexpr Access write = Access::BufferStorageWrite;
    constexpr Access rw = Access::BufferStorageReadWrite;
    constexpr Access sample = Access::TextureSampleRead;
    constexpr Access imageWrite = Access::TextureStorageWrite;
    constexpr Access imageRW = Access::TextureStorageReadWrite;
    constexpr Access args = Access::BufferIndirectRead;
    const Use clearUses[]{{"updateState", write}};
    const Use buildPriorityUses[]{{"cards", read}, {"cardScene", read}, {"domain", read}, {"cardUpdates", read},
        {"cardState", read}, {"updateState", rw}};
    const Use selectPriorityUses[]{{"updateState", rw}};
    const Use listUses[]{{"cards", read}, {"cardScene", read}, {"domain", read}, {"cardUpdates", read},
        {"cardState", rw}, {"updateState", rw}, {"scheduledCards", write}, {"refreshCards", write}};
    const Use finishUses[]{{"cardScene", read}, {"updateState", read}, {"indirectArgs", write}, {"updateStatistics", write}};
    const Use refreshUses[]{{"indirectArgs", args}, {"updateState", read}, {"refreshCards", read},
        {"cardDepth", sample}, {"cardAlbedo", sample}, {"cardEmissive", sample}, {"cardDirectLighting", sample},
        {"traceRadiance", imageWrite}, {"filteredRadiance", imageWrite},
        {"probeSHRed", write}, {"probeSHGreen", write}, {"probeSHBlue", write},
        {"indirectIrradiance", imageRW}, {"lighting", imageWrite}};
    const Use traceUses[]{{"indirectArgs", args}, {"updateState", read}, {"scheduledCards", read},
        {"pageTable", sample}, {"pageAtlas", sample}, {"coarseMip", sample}, {"domain", read}, {"objectGrid", read},
        {"objectGridScene", read}, {"cardScene", read}, {"cards", read}, {"cardDepth", sample}, {"cardNormal", sample},
        {"lighting", sample}, {"traceRadiance", imageWrite}};
    const Use filterUses[]{{"indirectArgs", args}, {"updateState", read}, {"scheduledCards", read},
        {"cards", read}, {"cardDepth", sample}, {"cardNormal", sample},
        {"traceRadiance", sample}, {"filteredRadiance", imageWrite}};
    const Use shUses[]{{"indirectArgs", args}, {"updateState", read}, {"scheduledCards", read},
        {"cards", read}, {"cardDepth", sample}, {"cardNormal", sample}, {"filteredRadiance", sample},
        {"probeSHRed", write}, {"probeSHGreen", write}, {"probeSHBlue", write}};
    const Use integrateUses[]{{"indirectArgs", args}, {"updateState", read}, {"scheduledCards", read},
        {"cards", read}, {"cardDepth", sample}, {"cardNormal", sample}, {"cardAlbedo", sample}, {"cardEmissive", sample},
        {"cardDirectLighting", sample}, {"probeSHRed", read}, {"probeSHGreen", read}, {"probeSHBlue", read},
        {"indirectIrradiance", imageRW}, {"lighting", imageWrite}};
    const uint32_t slotGroups = (schedule.capacity + 63u) / 64u;
    const RenderGraphStage stages[]{
        {"Clear card update counters", clearUses,
            [&](CommandBuffer& cmd) { return clearSchedule_.dispatch(cmd, *encodedSchedule, 1); }},
        {"Build card update priority histogram", buildPriorityUses,
            [&](CommandBuffer& cmd) { return buildPriority_.dispatch(cmd, *encodedSchedule, slotGroups); }},
        {"Select card priority budget", selectPriorityUses,
            [&](CommandBuffer& cmd) { return selectPriority_.dispatch(cmd, *encodedSchedule, 1); }},
        {"Compact card update lists", listUses,
            [&](CommandBuffer& cmd) { return buildCardLists_.dispatch(cmd, *encodedSchedule, slotGroups); }},
        {"Publish card update dispatches and statistics", finishUses,
            [&](CommandBuffer& cmd) { return finishSchedule_.dispatch(cmd, *encodedSchedule, 1); }},
        {"Refresh invalidated and directly relit card lighting", refreshUses,
            [&](CommandBuffer& cmd) { return indirect(refreshCards_, 0, cmd); }},
        {"Trace selected card probes", traceUses, [&](CommandBuffer& cmd) { return indirect(trace_, 1, cmd); }},
        {"Filter selected card probes", filterUses, [&](CommandBuffer& cmd) { return indirect(filter_, 1, cmd); }},
        {"Convert selected card probes to SH", shUses, [&](CommandBuffer& cmd) { return indirect(convertToSH_, 2, cmd); }},
        {"Accumulate selected card irradiance and combine outgoing lighting", integrateUses,
            [&](CommandBuffer& cmd) { return indirect(integrate_, 1, cmd); }},
    };
    if (auto result = context.executeStages(stages); !result) { return result; }
    if (cacheInitialized) { return {}; }
    return commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
        [initialized = initializedAtlas_, atlasAllocation]() { *initialized = atlasAllocation; }, [] {}));
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenRadiosityPass()
{
    return std::make_unique<LumenRadiosityPass>();
}

} // namespace metallic::render::builtin_pass
