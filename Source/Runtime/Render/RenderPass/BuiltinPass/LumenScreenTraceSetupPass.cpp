#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenTraceHistory.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <bit>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kSetupABI = 0x4c554d454e020003ull;
constexpr uint64_t kReduceABI = 0x4c554d454e020002ull;
constexpr uint32_t kUEHZBMipBatchSize = 4u;
using Extent = std::array<uint32_t, 2>;

struct LumenScreenTraceView {
    ViewConstants cameras;
    std::array<uint32_t, 4> state;
};
static_assert(sizeof(LumenScreenTraceView) == 224);

struct LumenScreenTraceSetupParams {
    LumenScreenTraceView traceView;
    ShaderSampledImage depth, color, historyDepth, historyColor;
    ShaderStorageImage previousDepth, previousColor, currentDepth, currentColor;
    ShaderBuffer closestHZB, view;
    uint32_t writeCurrentHistory, padding;
};
static_assert(sizeof(LumenScreenTraceSetupParams) == 272);

struct LumenClosestHZBReduceParams {
    ShaderBuffer closestHZB;
    Extent sourceExtent, destinationExtent;
    uint32_t sourceOffset, destinationOffset;
    uint32_t roundSource;
};
static_assert(sizeof(LumenClosestHZBReduceParams) == 32);

struct HZBLayout {
    Extent baseExtent;
    uint32_t mipCount;
    uint32_t texelCount;
};

HZBLayout hzbLayout(Extent fullExtent)
{
    // UE 5.7.4 SceneTextureReductions.cpp BuildHZB.
    const Extent baseExtent{std::max(1u, std::bit_ceil(fullExtent[0]) >> 1),
        std::max(1u, std::bit_ceil(fullExtent[1]) >> 1)};
    const uint32_t mipCount = std::max(1u, std::bit_width(std::max(baseExtent[0], baseExtent[1])) - 1u);
    uint32_t texelCount = fullExtent[0] * fullExtent[1];
    for (uint32_t mip = 0; mip < mipCount; ++mip) {
        texelCount += std::max(baseExtent[0] >> mip, 1u) * std::max(baseExtent[1] >> mip, 1u);
    }
    return {baseExtent, mipCount, texelCount};
}

using HistoryContent = std::array<uint64_t, 9>;

class LumenScreenTraceSetupPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

private:
    bool usesCompositedHistory() const
    {
        const auto source = properties().find("historySource");
        return source != properties().end() && *source == "composited";
    }

    Device* device_ = nullptr;
    HistoryResourceManager history_;
    std::array<std::string, 2> historyNames_;
    HistoryContent previousContent_{};
    uint64_t previousFrame_ = 0;
    ComputeKernel snapshot_, reduce_;
};

RenderPassReflection LumenScreenTraceSetupPass::reflect(const RenderGraphCompileContext& context) const
{
    RenderPassReflection reflection;
    reflection.addTextureInput("depth", "Current reversed-Z scene depth").sampledRead().format = Format::D32Sfloat;
    if (!usesCompositedHistory()) {
        reflection.addTextureInput("color", "Scene-linear color before indirect diffuse").sampledRead().format = Format::RGBA32Sfloat;
    }
    const auto layout = hzbLayout({context.width, context.height});
    reflection.addBufferOutput("closestHZB", "Closest HZB with full-resolution scene depth followed by packed mips")
        .buffer(uint64_t(layout.texelCount) * sizeof(float), sizeof(float)).storageReadWrite()
        .transient(RenderGraphInitialization::FullOverwrite);
    reflection.addBufferOutput("traceView", "Camera and screen trace history validity")
        .buffer(sizeof(LumenScreenTraceView), sizeof(LumenScreenTraceView)).storageWrite()
        .transient(RenderGraphInitialization::FullOverwrite);
    reflection.addTextureOutput("previousDepth", "Previous reversed-Z scene depth")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite).format = Format::R32Sfloat;
    auto& previousColor = reflection.addTextureOutput("previousColor", "Previous scene-linear color for screen trace hits")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    previousColor.format = Format::RGBA32Sfloat;
    previousColor.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenScreenTraceSetupPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    const auto source = properties().find("historySource");
    if (source != properties().end() && *source != "deferred" && *source != "composited") {
        log = "Lumen screen trace historySource must be deferred or composited";
        return makeError(Error::InvalidArgument);
    }
    if (usesCompositedHistory()) {
        const auto producer = properties().find("historyProducer");
        if (producer == properties().end() || !producer->is_string() || producer->get_ref<const std::string&>().empty()) {
            log = "Lumen composited screen history requires the name of a historyProducer pass";
            return makeError(Error::InvalidArgument);
        }
        historyNames_ = lumenScreenTraceHistoryNames(producer->get_ref<const std::string&>());
    } else {
        historyNames_ = {"sceneDepth", "directColor"};
    }

    device_ = context.device;
    if (!usesCompositedHistory()) {
        if (auto result = history_.initialize(*device_); !result) { return result; }
    }
    previousFrame_ = 0;
    previousContent_ = {};
    constexpr char module[] = "Features/DistanceField/LumenScreenTraceSetup";
    constexpr char shaderPath[] = PROJECT_SOURCE_DIR "/Shaders";
    if (auto result = ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = module, .entryPointName = "lumenScreenTraceSetupMain", .searchPath = shaderPath},
        {.parameters = parameterAbi<LumenScreenTraceSetupParams>(kSetupABI), .debugName = "LumenScreenTraceSetup"},
        snapshot_, log); !result) {
        return result;
    }
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = module, .entryPointName = "lumenClosestHZBReduceMain", .searchPath = shaderPath},
        {.parameters = parameterAbi<LumenClosestHZBReduceParams>(kReduceABI), .debugName = "LumenClosestHZBReduce"},
        reduce_, log);
}

Result<> LumenScreenTraceSetupPass::execute(RenderGraphExecutionContext& context)
{
    const auto* view = context.viewConstants();
    if (!view) { return makeError(Error::InvalidArgument); }
    const Extent extent{context.width(), context.height()};
    const auto layout = hzbLayout(extent);
    const auto* world = context.world();
    const auto* scene = world ? world->scene() : context.runtimeScene();
    const auto* globalHistory = context.historyResources();
    const HistoryContent content{
        globalHistory ? globalHistory->reprojectionInvalidationRevision() : 0,
        scene ? scene->resourceIdentity() : 0,
        scene ? scene->contentRevision() : 0,
        scene ? scene->geometryTransformRevision() : 0,
        scene ? scene->visibilityRevision() : 0,
        world ? world->lightingRevision() : 0,
        world ? world->environmentRevision() : 0,
        extent[0], extent[1],
    };
    const bool compositedHistory = usesCompositedHistory();
    auto& history = compositedHistory ? context.graphHistoryResources() : history_;
    const bool previousContentValid = previousContent_ == content && previousFrame_ + 1 == context.frameIndex() &&
        view->frame[1] != 0u;
    if (!compositedHistory) {
        // Keep the Deferred comparison path reprojectable during ordinary camera motion.
        history.beginFrame(context.frameIndex());
        if (!previousContentValid) { history.invalidateAll(); }
        if (auto result = ensureLumenScreenTraceHistory(history, historyNames_, extent[0], extent[1]); !result) {
            return result;
        }
    }
    const auto previousDepth = history.texture(historyNames_[0], HistorySlot::Previous);
    const auto previousColor = history.texture(historyNames_[1], HistorySlot::Previous);
    const auto matchesFrame = [&](const HistoryTextureRef& texture) {
        return texture.valid && texture.desc->width == extent[0] && texture.desc->height == extent[1];
    };
    const bool historyValid = previousContentValid && matchesFrame(previousDepth) && matchesFrame(previousColor);

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenScreenTraceSetupParams params{};
    params.traceView.cameras = *view;
    params.traceView.state = {historyValid, layout.baseExtent[0], layout.baseExtent[1], layout.mipCount};
    params.depth = writer.sampledImageHandle(context.inputTexture("depth").view());
    if (historyValid) {
        params.historyDepth = writer.sampledImageHandle(previousDepth.view);
        params.historyColor = writer.sampledImageHandle(previousColor.view);
    }
    params.previousDepth = writer.storageImageHandle(context.outputTexture("previousDepth").view());
    params.previousColor = writer.storageImageHandle(context.outputTexture("previousColor").view());
    params.closestHZB = writer.buffer(context.outputBuffer("closestHZB").buffer());
    params.view = writer.buffer(context.outputBuffer("traceView").buffer());
    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> snapshotUses{
        {"depth", Access::TextureSampleRead},
        {"closestHZB", Access::BufferStorageWrite}, {"traceView", Access::BufferStorageWrite},
        {"previousDepth", Access::TextureStorageWrite}, {"previousColor", Access::TextureStorageWrite},
    };
    std::vector<RenderGraphTextureImport> textures;
    if (!compositedHistory) {
        const auto currentDepth = history.texture(historyNames_[0], HistorySlot::Current);
        const auto currentColor = history.texture(historyNames_[1], HistorySlot::Current);
        params.color = writer.sampledImageHandle(context.inputTexture("color").view());
        params.currentDepth = writer.storageImageHandle(currentDepth.view);
        params.currentColor = writer.storageImageHandle(currentColor.view);
        params.writeCurrentHistory = 1;
        snapshotUses.insert(snapshotUses.end(), {{"color", Access::TextureSampleRead},
            {"historyCurrentDepth", Access::TextureStorageWrite}, {"historyCurrentColor", Access::TextureStorageWrite}});
        textures.push_back({"historyCurrentDepth", currentDepth.texture, currentDepth.view, currentDepth.state, ResourceState::ShaderRead});
        textures.push_back({"historyCurrentColor", currentColor.texture, currentColor.view, currentColor.state, ResourceState::ShaderRead});
    }
    if (historyValid) {
        snapshotUses.push_back({"historyPreviousDepth", Access::TextureSampleRead});
        snapshotUses.push_back({"historyPreviousColor", Access::TextureSampleRead});
        textures.push_back({"historyPreviousDepth", previousDepth.texture, previousDepth.view, previousDepth.state, ResourceState::ShaderRead});
        textures.push_back({"historyPreviousColor", previousColor.texture, previousColor.view, previousColor.state, ResourceState::ShaderRead});
    }
    const auto encoded = writer.encode(params, kSetupABI);
    if (!encoded) { return makeError(encoded.error()); }
    std::vector<RenderGraphStage> stages;
    stages.reserve(1u + layout.mipCount);
    stages.push_back({"Snapshot scene history and seed HZB", snapshotUses, [&](CommandBuffer& cmd) {
        return snapshot_.dispatch(cmd, *encoded, (extent[0] + 7u) / 8u, (extent[1] + 7u) / 8u);
    }});

    const RenderGraphStageUse reduceUses[] = {{"closestHZB", Access::BufferStorageReadWrite}};
    LumenClosestHZBReduceParams reduceParams{.closestHZB = params.closestHZB, .sourceExtent = extent,
        .destinationExtent = layout.baseExtent, .sourceOffset = 0, .destinationOffset = extent[0] * extent[1]};
    for (uint32_t mip = 0; mip < layout.mipCount; ++mip) {
        reduceParams.roundSource = mip != 0u && mip % kUEHZBMipBatchSize == 0u;
        const auto reduction = writer.encode(reduceParams, kReduceABI);
        if (!reduction) { return makeError(reduction.error()); }
        const auto destination = reduceParams.destinationExtent;
        stages.push_back({"Reduce closest HZB", reduceUses, [&, reduction, destination](CommandBuffer& cmd) {
            return reduce_.dispatch(cmd, *reduction, (destination[0] + 7u) / 8u, (destination[1] + 7u) / 8u);
        }});
        reduceParams.sourceExtent = destination;
        reduceParams.sourceOffset = reduceParams.destinationOffset;
        reduceParams.destinationOffset += destination[0] * destination[1];
        reduceParams.destinationExtent = {std::max(1u, destination[0] >> 1), std::max(1u, destination[1] >> 1)};
    }

    if (auto result = context.executeStages(stages, {}, textures); !result) { return result; }
    for (const auto& name : historyNames_) {
        if (!compositedHistory) {
            if (auto result = history.publishTextureState(commands, name,
                HistorySlot::Current, ResourceState::ShaderRead, true); !result) { return result; }
        }
        if (historyValid) {
            if (auto result = history.publishTextureState(commands, name,
                HistorySlot::Previous, ResourceState::ShaderRead); !result) { return result; }
        }
    }
    previousContent_ = content;
    previousFrame_ = context.frameIndex();
    return {};
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenScreenTraceSetupPass()
{
    return std::make_unique<LumenScreenTraceSetupPass>();
}

} // namespace metallic::render::builtin_pass
