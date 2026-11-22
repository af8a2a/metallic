#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kRadiosityABI = 0x4c554d454e040001ull;
constexpr uint32_t kProbeSpacing = 4;
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
    uint32_t viewFrameIndex;
};
static_assert(sizeof(LumenRadiosityParams) == 148 && offsetof(LumenRadiosityParams, atlasSize) == 120);

struct RadiosityLayout {
    uint32_t atlasSize = 256;
    uint32_t tileResolution = 32;
    float maxTraceDistance = 200.0f;
};

std::optional<RadiosityLayout> readLayout(const RenderGraphProperties& properties)
{
    try {
        const RadiosityLayout layout{properties.value("atlasSize", 256u), properties.value("tileResolution", 32u),
            properties.value("maxTraceDistance", 200.0f)};
        if (layout.tileResolution < kProbeSpacing || layout.tileResolution % kProbeSpacing != 0 ||
            layout.atlasSize < layout.tileResolution || layout.atlasSize > 2048 ||
            layout.atlasSize % layout.tileResolution != 0 || !std::isfinite(layout.maxTraceDistance) ||
            layout.maxTraceDistance <= 0.1f) {
            return std::nullopt;
        }
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
    Device* device_ = nullptr;
    RadiosityLayout layout_;
    ComputeKernel trace_, filter_, convertToSH_, integrate_;
    HistoryResourceManager history_;
    std::optional<RadiosityContent> lastContent_;
    uint64_t lastFrame_ = 0;
    uint32_t lastTemporalIndex_ = 0;
};

RenderPassReflection LumenRadiosityPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    const uint32_t edge = readLayout(properties()).value_or(RadiosityLayout{}).atlasSize;
    reflectLumenGlobalTraceInputs(reflection);
    for (std::string_view name : {"cardNormal", "cardAlbedo", "cardEmissive", "cardDirectLighting"}) {
        auto& input = reflection.addTextureInput(std::string(name), "Surface Cache input for radiosity")
            .texture2D(edge, edge).sampledRead();
        input.matchOutputExtent = false;
        input.format = name == "cardDirectLighting" ? Format::RGBA32Sfloat :
            (name == "cardEmissive" ? Format::RGBA16Sfloat : Format::RGBA8Unorm);
    }
    for (auto name : {"traceRadiance", "filteredRadiance"}) {
        reflection.addTextureOutput(name, "Hemisphere trace radiance")
            .texture2D(edge, edge).storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead)
            .transient(RenderGraphInitialization::FullOverwrite).format = Format::RGBA16Sfloat;
    }
    const uint32_t probesPerAxis = edge / kProbeSpacing;
    constexpr uint32_t packedSHStride = 2u * sizeof(uint32_t);
    for (auto name : kSHNames) {
        reflection.addBufferOutput(name, "Two-band spherical harmonics per card probe packed as four half floats")
            .buffer(uint64_t(probesPerAxis) * probesPerAxis * packedSHStride, packedSHStride).storageReadWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
    }
    for (auto name : kHistoryNames) {
        auto& output = reflection.addTextureOutput(name, "Scene-linear Surface Cache lighting")
            .texture2D(edge, edge).storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        output.format = Format::RGBA16Sfloat;
        output.colorEncoding = DisplayColorEncoding::SceneLinear;
    }
    return reflection;
}

Result<> LumenRadiosityPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto layout = readLayout(properties());
    if (!layout || !context.runtimeScene) {
        log = "Lumen Radiosity requires a world scene, a tile divisible by four and a matching fixed Surface Cache atlas";
        return makeError(Error::InvalidArgument);
    }
    layout_ = *layout;
    device_ = context.device;
    lastContent_.reset();
    if (auto result = history_.initialize(*device_); !result) { return result; }
    const auto compileKernel = [&](const char* module, const char* entry, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = module, .entryPointName = entry, .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<LumenRadiosityParams>(kRadiosityABI), .debugName = entry}, kernel, log);
    };
    constexpr char traceModule[] = "Features/DistanceField/LumenRadiosityTrace";
    constexpr char integrateModule[] = "Features/DistanceField/LumenRadiosityIntegrate";
    if (auto result = compileKernel(traceModule, "lumenTraceRadiosityMain", trace_); !result) { return result; }
    if (auto result = compileKernel(traceModule, "lumenFilterRadiosityMain", filter_); !result) { return result; }
    if (auto result = compileKernel(integrateModule, "lumenRadiosityConvertToSHMain", convertToSH_); !result) { return result; }
    return compileKernel(integrateModule, "lumenRadiosityIntegrateMain", integrate_);
}

Result<> LumenRadiosityPass::execute(RenderGraphExecutionContext& context)
{
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }

    const auto [edge, tile, maxDistance] = layout_;
    for (auto name : {"cardDepth", "cardLighting", "cardNormal", "cardAlbedo", "cardEmissive", "cardDirectLighting"}) {
        const auto& desc = context.inputTexture(name).desc();
        if (desc.width != edge || desc.height != edge) {
            spdlog::error("Lumen Radiosity input {} must match atlasSize {}", name, edge);
            return makeError(Error::InvalidArgument);
        }
    }
    const uint32_t tiles = edge / tile;
    if (context.inputBuffer("cards").desc().size != uint64_t(tiles) * tiles * kLumenGlobalTraceCardStride) {
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
    // Card history is world-space. Camera cuts affect the screen histories only.
    const RadiosityContent content{scene->resourceIdentity(), scene->contentRevision(),
        scene->geometryTransformRevision(), scene->visibilityRevision(), world ? world->lightingRevision() : 0,
        world ? world->environmentRevision() : 0};
    history_.beginFrame(context.frameIndex());
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
    const uint32_t temporalIndex = historyValid ? lastTemporalIndex_ + 1u : 1u;

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    const auto sampled = [&](const char* name) { return writer.sampledImageHandle(context.inputTexture(name).view()); };
    const auto readOutput = [&](const char* name) { return writer.sampledImageHandle(context.outputTexture(name).view()); };
    const auto writeOutput = [&](const char* name) { return writer.storageImageHandle(context.outputTexture(name).view()); };
    const auto outputBuffer = [&](const char* name) { return writer.buffer(context.outputBuffer(name).buffer()); };
    LumenRadiosityParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    if (historyValid) { params.scene.cardLighting = writer.sampledImageHandle(previousLighting.view); }
    params.cardNormal = sampled("cardNormal");
    params.cardAlbedo = sampled("cardAlbedo");
    params.cardEmissive = sampled("cardEmissive");
    params.cardDirectLighting = sampled("cardDirectLighting");
    if (historyValid) { params.previousIndirect = writer.sampledImageHandle(previousIndirect.view); }
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
    params.currentIndirect = writer.storageImageHandle(currentIndirect.view);
    params.currentLighting = writer.storageImageHandle(currentLighting.view);
    params.atlasSize = edge;
    params.tileResolution = tile;
    params.clipmapCount = *clipmapCount;
    params.historyValid = historyValid;
    params.temporalIndex = temporalIndex;
    params.maxTraceDistance = maxDistance;
    params.viewFrameIndex = static_cast<uint32_t>(context.frameIndex());
    const auto encoded = writer.encode(params, kRadiosityABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse traceUses[] = {{"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead}, {"objectGrid", Access::BufferStorageRead},
        {"objectGridScene", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead}, {"cards", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardNormal", Access::TextureSampleRead},
        {"traceRadiance", Access::TextureStorageWrite},
        {historyValid ? "previousLighting" : "cardLighting", Access::TextureSampleRead}};
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
        {"Trace card probes", traceUses, [&](CommandBuffer& cmd) { return trace_.dispatch(cmd, *encoded, groups, groups, 1); }},
        {"Filter card probes", filterUses, [&](CommandBuffer& cmd) { return filter_.dispatch(cmd, *encoded, groups, groups, 1); }},
        {"Convert card probes to SH", shUses,
            [&](CommandBuffer& cmd) { return convertToSH_.dispatch(cmd, *encoded, probeGroups, probeGroups, 1); }},
        {"Accumulate card irradiance and combine outgoing lighting",
            std::span{integrateUses}.first(std::size(integrateUses) - (historyValid ? 0u : 1u)),
            [&](CommandBuffer& cmd) { return integrate_.dispatch(cmd, *encoded, groups, groups, 1); }},
    };
    if (auto result = context.executeStages(stages, {}, std::span{imports}.first(historyValid ? 4u : 2u)); !result) {
        return result;
    }
    for (const auto name : kHistoryNames) {
        auto result = history_.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
        if (!result) { return result; }
    }
    lastContent_ = content;
    lastFrame_ = context.frameIndex();
    lastTemporalIndex_ = temporalIndex;
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenRadiosityPass()
{
    return std::make_unique<LumenRadiosityPass>();
}

} // namespace metallic::render::builtin_pass
