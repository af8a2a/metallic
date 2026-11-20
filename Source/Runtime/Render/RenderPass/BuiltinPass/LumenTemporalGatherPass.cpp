#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <optional>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenTemporalGatherABI = 0x4c554d454e030001ull;

struct LumenTemporalGatherParams {
    ShaderSampledImage indirectDiffuse, position, normal, directColor;
    ShaderSampledImage previousDiffuse, previousGeometry;
    ShaderStorageImage indirectOutput, color, historyWeight, currentDiffuse, currentGeometry;
    uint32_t historyValid;
    uint32_t width, height;
    float previousJitter[2];
    ViewCameraConstants currentCamera, previousCamera;
};
static_assert(sizeof(LumenTemporalGatherParams) == 224);

struct HistoryContent {
    uint64_t reprojectionRevision = 0;
    uint64_t sceneIdentity = 0;
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

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
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
    reflection.addTextureInput("directColor", "Unexposed scene color before indirect diffuse")
        .sampledRead().format = Format::RGBA32Sfloat;

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
    device_ = context.device;
    lastContent_.reset();
    if (auto result = history_.initialize(*device_); !result) { return result; }
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = "Features/DistanceField/LumenTemporalGather", .entryPointName = "lumenTemporalGatherMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
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
    const HistoryContent content{
        .reprojectionRevision = globalHistory ? globalHistory->reprojectionInvalidationRevision() : 0,
        .sceneIdentity = scene ? scene->resourceIdentity() : 0,
        .sceneContentRevision = scene ? scene->contentRevision() : 0,
        .geometryTransformRevision = scene ? scene->geometryTransformRevision() : 0,
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
    const auto encoded = writer.encode(params, kLumenTemporalGatherABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse uses[] = {
        {"currentIndirectDiffuse", Access::TextureSampleRead}, {"position", Access::TextureSampleRead},
        {"normal", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
        {"indirectDiffuse", Access::TextureStorageWrite}, {"color", Access::TextureStorageWrite},
        {"historyWeight", Access::TextureStorageWrite}, {"currentDiffuse", Access::TextureStorageWrite},
        {"currentGeometry", Access::TextureStorageWrite}, {"previousDiffuse", Access::TextureSampleRead},
        {"previousGeometry", Access::TextureSampleRead},
    };
    const auto import = [](std::string_view name, const HistoryTextureRef& texture) {
        return RenderGraphTextureImport{name, texture.texture, texture.view, texture.state, ResourceState::ShaderRead};
    };
    const RenderGraphTextureImport textures[] = {
        import("currentDiffuse", currentDiffuse), import("currentGeometry", currentGeometry),
        import("previousDiffuse", previousDiffuse), import("previousGeometry", previousGeometry),
    };
    const RenderGraphStage stages[] = {
        {"Reproject and accumulate diffuse", std::span{uses}.first(historyValid ? std::size(uses) : std::size(uses) - 2),
            [&](CommandBuffer& cmd) {
                return kernel_.dispatch(cmd, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
            }},
    };
    auto result = context.executeStages(stages, {},
        std::span{textures}.first(historyValid ? std::size(textures) : std::size(textures) - 2));
    if (!result) { return result; }

    for (const auto& name : names) {
        result = history_.publishTextureState(commands, name, HistorySlot::Current, ResourceState::ShaderRead, true);
        if (!result) { return result; }
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
