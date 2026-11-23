#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenTraceHistory.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <limits>
#include <optional>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenTemporalGatherABI = 0x4c554d454e030005ull;

struct LumenTemporalGatherParams {
    ShaderSampledImage indirectDiffuse, position, normal, directColor;
    ShaderSampledImage previousDiffuse, previousGeometry;
    ShaderStorageImage indirectOutput, color, historyWeight, currentDiffuse, currentGeometry;
    uint32_t historyValid;
    uint32_t width, height;
    float previousJitter[2];
    ViewCameraConstants currentCamera, previousCamera;
    ShaderSampledImage sceneDepth;
    ShaderStorageImage screenHistoryDepth, screenHistoryColor;
    uint32_t publishScreenTraceHistory;
    float maxFramesAccumulated;
    ShaderSampledImage blueNoiseScalar;
    uint32_t frameIndex;
    ShaderSampledImage previousPosition;
};
static_assert(sizeof(LumenTemporalGatherParams) == 256 &&
    offsetof(LumenTemporalGatherParams, maxFramesAccumulated) == 240 &&
    offsetof(LumenTemporalGatherParams, previousPosition) == 252);

struct HistoryContent {
    uint64_t reprojectionRevision = 0;
    uint64_t sceneIdentity = 0;
    uint64_t sceneLifetimeRevision = 0;
    uint64_t sceneStructuralRevision = 0;
    uint64_t sceneContentRevision = 0;
    uint64_t geometryTransformRevision = 0;
    uint64_t visibilityRevision = 0;
    uint64_t lightingRevision = 0;
    uint64_t environmentRevision = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t cameraCutSerial = 0;
    bool operator==(const HistoryContent&) const = default;
};

class LumenTemporalGatherPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

private:
    bool dynamicReprojection() const
    {
        const auto value = properties().find("dynamicReprojection");
        return value != properties().end() && *value == true;
    }

    bool publishesScreenTraceHistory() const
    {
        const auto value = properties().find("publishScreenTraceHistory");
        return value != properties().end() && *value == true;
    }

    Device* device_ = nullptr;
    ComputeKernel kernel_;
    std::unique_ptr<TextureView> blueNoiseScalar_;
    float maxFramesAccumulated_ = 10.0f;
    // The shared manager invalidates its resources during ordinary editor camera
    // motion. Retain this pass's reprojectable history across that event.
    HistoryResourceManager history_;
    std::optional<HistoryContent> lastContent_;
    uint64_t lastFrame_ = 0;
};

RenderPassReflection LumenTemporalGatherPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    reflection.addTextureInput("currentIndirectDiffuse", "Current unexposed indirect diffuse after receiver response")
        .sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("position", "Receiver world position and positive linear view depth")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("normal", "Receiver shading normal").sampledRead().format = Format::RGBA16Sfloat;
    if (dynamicReprojection()) {
        reflection.addTextureInput("previousPosition", "Previous world position of the current receiver and validity")
            .sampledRead().format = Format::RGBA32Sfloat;
    }
    reflection.addTextureInput("directColor", "Unexposed scene color before indirect diffuse")
        .sampledRead().format = Format::RGBA32Sfloat;
    if (publishesScreenTraceHistory()) {
        reflection.addTextureInput("sceneDepth", "Reversed-Z depth of the composited frame")
            .sampledRead().format = Format::D32Sfloat;
    }

    auto& indirect = reflection.addTextureOutput("indirectDiffuse", "Accumulated unexposed indirect diffuse")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    indirect.format = Format::RGBA16Sfloat;
    indirect.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& color = reflection.addTextureOutput("color", "Unexposed scene color with accumulated indirect diffuse")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    reflection.addTextureOutput("historyWeight", "Accumulated history contribution after geometry rejection")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite).format = Format::R32Sfloat;
    return reflection;
}

Result<> LumenTemporalGatherPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto publish = properties().find("publishScreenTraceHistory");
    if (publish != properties().end() && !publish->is_boolean()) {
        log = "Lumen Temporal Gather publishScreenTraceHistory must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    const auto dynamic = properties().find("dynamicReprojection");
    if (dynamic != properties().end() && !dynamic->is_boolean()) {
        log = "Lumen Temporal Gather dynamicReprojection must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    const auto frames = properties().find("maxFramesAccumulated");
    if (frames != properties().end() && (!frames->is_number_integer() || *frames < 1 ||
        *frames > std::numeric_limits<int32_t>::max())) {
        log = "Lumen maxFramesAccumulated must be a positive 32-bit integer";
        return makeError(Error::InvalidArgument);
    }
    maxFramesAccumulated_ = static_cast<float>(properties().value("maxFramesAccumulated", 10u));
    const auto quantization = properties().find("stochasticHistoryQuantization");
    if (quantization != properties().end() && !quantization->is_boolean()) {
        log = "Lumen stochasticHistoryQuantization must be a boolean";
        return makeError(Error::InvalidArgument);
    }
    device_ = context.device;
    const bool ditherHistory = properties().value("stochasticHistoryQuantization", false);
    if (ditherHistory) {
        if (!context.graphicsQueue) {
            log = "Lumen temporal quantization requires a graphics queue to upload blue noise";
            return makeError(Error::InvalidArgument);
        }
        auto loaded = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Scalar, log);
        if (!loaded) { return makeError(loaded.error()); }
        blueNoiseScalar_ = std::move(*loaded);
    } else {
        blueNoiseScalar_.reset();
    }
    std::array<SlangMacroDefine, 2> defines{};
    size_t defineCount = 0;
    if (ditherHistory) { defines[defineCount++] = {"METALLIC_LUMEN_DITHER_HISTORY", "1"}; }
    if (dynamicReprojection()) { defines[defineCount++] = {"METALLIC_LUMEN_DYNAMIC_REPROJECTION", "1"}; }
    lastContent_.reset();
    if (auto result = history_.initialize(*device_); !result) { return result; }
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = "Features/DistanceField/LumenTemporalGather", .entryPointName = "lumenTemporalGatherMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = {defines.data(), defineCount}},
        {.parameters = parameterAbi<LumenTemporalGatherParams>(kLumenTemporalGatherABI),
            .debugName = "LumenTemporalGather"}, kernel_, log);
}

Result<> LumenTemporalGatherPass::execute(RenderGraphExecutionContext& context)
{
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    const auto* view = context.viewConstants();
    const auto* world = context.world();
    const auto* scene = world ? world->scene() : context.runtimeScene();
    const auto* globalHistory = context.historyResources();
    const bool dynamic = dynamicReprojection();
    const uint64_t reprojectionRevision = globalHistory
        ? (dynamic ? globalHistory->motionReprojectionInvalidationRevision() : globalHistory->reprojectionInvalidationRevision())
        : 0;
    const HistoryContent content{
        .reprojectionRevision = reprojectionRevision,
        .sceneIdentity = scene ? scene->resourceIdentity() : 0,
        .sceneLifetimeRevision = dynamic && scene ? scene->sceneGraph().lifetimeRevision() : 0,
        .sceneStructuralRevision = dynamic && scene ? scene->sceneGraph().structuralRevision() : 0,
        .sceneContentRevision = scene ? scene->contentRevision() : 0,
        .geometryTransformRevision = !dynamic && scene ? scene->geometryTransformRevision() : 0,
        .visibilityRevision = scene ? scene->visibilityRevision() : 0,
        .lightingRevision = world ? world->lightingRevision() : 0,
        .environmentRevision = world ? world->environmentRevision() : 0,
        .width = context.width(),
        .height = context.height(),
        .cameraCutSerial = view ? view->frame[3] : 0,
    };

    const std::string historyScope = "LumenTemporalGather." + context.passName() + ".";
    const std::array names{historyScope + "diffuse", historyScope + "geometry"};
    const std::array formats{Format::RGBA16Sfloat, Format::RGBA32Sfloat};
    history_.beginFrame(context.frameIndex());
    if (lastContent_ != content || lastFrame_ + 1 != context.frameIndex() || !view || view->frame[1] == 0) {
        history_.invalidateAll();
    }

    for (size_t index = 0; index < names.size(); ++index) {
        const TextureDesc desc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = formats[index],
            .width = content.width,
            .height = content.height,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        auto result = history_.ensureTexture(names[index], desc, TextureViewDesc{.format = desc.format});
        if (!result) { return result; }
    }

    const auto currentDiffuse = history_.texture(names[0], HistorySlot::Current);
    const auto currentGeometry = history_.texture(names[1], HistorySlot::Current);
    const auto previousDiffuse = history_.texture(names[0], HistorySlot::Previous);
    const auto previousGeometry = history_.texture(names[1], HistorySlot::Previous);
    const bool historyValid = previousDiffuse.valid && previousGeometry.valid;

    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenTemporalGatherParams params{};
    params.maxFramesAccumulated = maxFramesAccumulated_;
    if (blueNoiseScalar_) {
        params.blueNoiseScalar = writer.sampledImageHandle(blueNoiseScalar_.get());
        params.frameIndex = static_cast<uint32_t>(context.frameIndex());
    }
    params.indirectDiffuse = writer.sampledImageHandle(context.inputTexture("currentIndirectDiffuse").view());
    params.position = writer.sampledImageHandle(context.inputTexture("position").view());
    params.normal = writer.sampledImageHandle(context.inputTexture("normal").view());
    params.directColor = writer.sampledImageHandle(context.inputTexture("directColor").view());
    params.previousDiffuse = historyValid ? writer.sampledImageHandle(previousDiffuse.view) : params.indirectDiffuse;
    params.previousGeometry = historyValid ? writer.sampledImageHandle(previousGeometry.view) : params.position;
    params.indirectOutput = writer.storageImageHandle(context.outputTexture("indirectDiffuse").view());
    params.color = writer.storageImageHandle(context.outputTexture("color").view());
    params.historyWeight = writer.storageImageHandle(context.outputTexture("historyWeight").view());
    params.currentDiffuse = writer.storageImageHandle(currentDiffuse.view);
    params.currentGeometry = writer.storageImageHandle(currentGeometry.view);
    params.historyValid = historyValid;
    params.width = content.width;
    params.height = content.height;
    if (view) {
        params.currentCamera = view->current;
        params.previousCamera = view->previous;
        params.previousJitter[0] = view->jitter[2];
        params.previousJitter[1] = view->jitter[3];
    }
    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> uses{
        {"currentIndirectDiffuse", Access::TextureSampleRead}, {"position", Access::TextureSampleRead},
        {"normal", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
        {"indirectDiffuse", Access::TextureStorageWrite}, {"color", Access::TextureStorageWrite},
        {"historyWeight", Access::TextureStorageWrite}, {"currentDiffuse", Access::TextureStorageWrite},
        {"currentGeometry", Access::TextureStorageWrite},
    };
    if (dynamic) {
        params.previousPosition = writer.sampledImageHandle(context.inputTexture("previousPosition").view());
        uses.push_back({"previousPosition", Access::TextureSampleRead});
    }
    const auto import = [](std::string_view name, const HistoryTextureRef& texture) {
        return RenderGraphTextureImport{name, texture.texture, texture.view, texture.state, ResourceState::ShaderRead};
    };
    std::vector<RenderGraphTextureImport> textures{
        import("currentDiffuse", currentDiffuse), import("currentGeometry", currentGeometry),
    };
    if (historyValid) {
        uses.insert(uses.end(), {{"previousDiffuse", Access::TextureSampleRead}, {"previousGeometry", Access::TextureSampleRead}});
        textures.push_back(import("previousDiffuse", previousDiffuse));
        textures.push_back(import("previousGeometry", previousGeometry));
    }
    const bool publishScreenHistory = publishesScreenTraceHistory();
    const auto screenNames = publishScreenHistory ? lumenScreenTraceHistoryNames(context.passName()) : std::array<std::string, 2>{};
    auto& screenHistory = context.graphHistoryResources();
    if (publishScreenHistory) {
        if (auto result = ensureLumenScreenTraceHistory(screenHistory, screenNames, params.width, params.height); !result) {
            return result;
        }
        const auto depth = screenHistory.texture(screenNames[0], HistorySlot::Current);
        const auto color = screenHistory.texture(screenNames[1], HistorySlot::Current);
        params.sceneDepth = writer.sampledImageHandle(context.inputTexture("sceneDepth").view());
        params.screenHistoryDepth = writer.storageImageHandle(depth.view);
        params.screenHistoryColor = writer.storageImageHandle(color.view);
        params.publishScreenTraceHistory = 1;
        uses.insert(uses.end(), {{"sceneDepth", Access::TextureSampleRead},
            {"screenHistoryDepth", Access::TextureStorageWrite}, {"screenHistoryColor", Access::TextureStorageWrite}});
        textures.push_back(import("screenHistoryDepth", depth));
        textures.push_back(import("screenHistoryColor", color));
    }
    const auto encoded = writer.encode(params, kLumenTemporalGatherABI);
    if (!encoded) { return makeError(encoded.error()); }
    const RenderGraphStage stages[] = {
        {"Reproject and accumulate diffuse", uses,
            [&](CommandBuffer& cmd) {
                return kernel_.dispatch(cmd, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
            }},
    };
    auto result = context.executeStages(stages, {}, textures);
    if (!result) { return result; }

    for (const auto& name : names) {
        result = history_.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
        if (!result) { return result; }
    }
    if (publishScreenHistory) {
        for (const auto& name : screenNames) {
            result = screenHistory.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
            if (!result) { return result; }
        }
    }
    lastContent_ = content;
    lastFrame_ = context.frameIndex();
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenTemporalGatherPass()
{
    return std::make_unique<LumenTemporalGatherPass>();
}

} // namespace metallic::render::builtin_pass
