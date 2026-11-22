#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenProbePlacement.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kProbeVisualizeABI = 0x4c554d454e090001ull;
constexpr uint32_t kProbeGatherResolution = 8;
constexpr uint32_t kRadianceStride = 4 * sizeof(float);
constexpr float kDefaultDistanceRange = 4.0f;

struct LumenProbeVisualizeParams {
    ShaderBuffer probePositions, probeRadiance, filteredProbeRadiance, probeDistance;
    ShaderSampledImage sceneColor, historyWeight;
    ShaderStorageImage placement, traceSource, rawRadiance, filteredRadiance, hitDistance, historyWeightView;
    uint32_t width, height, probeWidth, probeHeight;
    uint32_t probeSlot, traceResolution;
    float distanceRange;
};
static_assert(sizeof(LumenProbeVisualizeParams) == 76 && offsetof(LumenProbeVisualizeParams, width) == 48);

class LumenProbeVisualizePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
};

RenderPassReflection LumenProbeVisualizePass::reflect(const RenderGraphCompileContext& context) const
{
    RenderPassReflection reflection;
    const uint64_t probeCount = uint64_t(lumenScreenProbeTiles(context.width)) * lumenScreenProbeTiles(context.height) *
        kLumenScreenProbeSlotsPerTile;
    constexpr uint32_t positionStride = 2 * sizeof(uint32_t);
    reflection.addBufferInput("probePositions", "Final Gather probe pixel positions by tile and slot")
        .buffer(probeCount * positionStride, positionStride).storageRead();
    reflection.addBufferInput("probeRadiance", "Unfiltered probe trace radiance and source classification")
        .buffer(0, kRadianceStride).storageRead();
    reflection.addBufferInput("filteredProbeRadiance", "Filtered screen probe radiance in 8×8 octahedral order")
        .buffer(probeCount * kProbeGatherResolution * kProbeGatherResolution * kRadianceStride, kRadianceStride).storageRead();
    reflection.addBufferInput("probeDistance", "Unfiltered probe trace distances in meters")
        .buffer(0, sizeof(float)).storageRead();
    reflection.addTextureInput("sceneColor", "Scene-linear color behind the probe placement markers")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("historyWeight", "Temporal history contribution")
        .sampledRead().format = Format::R32Sfloat;

    const auto addOutput = [&](const char* name, const char* description, Format format, DisplayColorEncoding encoding) {
        auto& field = reflection.addTextureOutput(name, description)
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        field.format = format;
        field.colorEncoding = encoding;
    };
    constexpr auto diagnostic = DisplayColorEncoding::DisplayLinearRec709;
    constexpr auto radiance = DisplayColorEncoding::SceneLinear;
    addOutput("placement", "Yellow uniform and magenta adaptive screen probes", Format::RGBA8Unorm, diagnostic);
    addOutput("traceSource", "Probe trace source and miss state in octahedral lane order", Format::RGBA8Unorm, diagnostic);
    addOutput("rawRadiance", "Scene-linear unfiltered probe radiance", Format::RGBA32Sfloat, radiance);
    addOutput("filteredRadiance", "Scene-linear spatially filtered probe radiance", Format::RGBA32Sfloat, radiance);
    addOutput("hitDistance", "Stored probe distance mapped from black to white over the distance range", Format::RGBA8Unorm, diagnostic);
    addOutput("historyWeightView", "Temporal history weight as grayscale", Format::RGBA8Unorm, diagnostic);
    return reflection;
}

std::vector<RenderGraphRuntimeSetting> LumenProbeVisualizePass::runtimeSettings() const
{
    return {runtimeIntSetting("probeSlot", "Probe Slot (0 Uniform)", 0, 0, kLumenScreenProbeSlotsPerTile - 1),
        runtimeFloatSetting("distanceRange", "Distance Range (m)", kDefaultDistanceRange, 0.01f, 1000.0f)};
}

Result<> LumenProbeVisualizePass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = "Features/DistanceField/LumenProbeVisualize", .entryPointName = "lumenProbeVisualizeMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenProbeVisualizeParams>(kProbeVisualizeABI), .debugName = "LumenProbeVisualize"}, kernel_, log);
}

Result<> LumenProbeVisualizePass::execute(RenderGraphExecutionContext& context)
{
    const auto& properties = context.properties();
    const auto slot = properties.find("probeSlot");
    const auto range = properties.find("distanceRange");
    if ((slot != properties.end() && (!slot->is_number_integer() || *slot < 0 || *slot >= kLumenScreenProbeSlotsPerTile)) ||
        (range != properties.end() && !range->is_number())) {
        spdlog::error("Lumen probe visualization requires a probeSlot in [0, 8] and a numeric distanceRange");
        return makeError(Error::InvalidArgument);
    }
    const float distanceRange = properties.value("distanceRange", kDefaultDistanceRange);
    if (!std::isfinite(distanceRange) || distanceRange <= 0.0f) {
        spdlog::error("Lumen probe visualization distanceRange must be positive and finite");
        return makeError(Error::InvalidArgument);
    }

    const uint32_t probeWidth = lumenScreenProbeTiles(context.width());
    const uint32_t probeHeight = lumenScreenProbeTiles(context.height());
    const uint64_t probeCount = uint64_t(probeWidth) * probeHeight * kLumenScreenProbeSlotsPerTile;
    const auto probeDistance = context.inputBuffer("probeDistance");
    const auto probeRadiance = context.inputBuffer("probeRadiance");
    const uint64_t distanceBytes = probeDistance.desc().size;
    const uint64_t rayCount = distanceBytes / sizeof(float);
    uint32_t traceResolution;
    if (rayCount == probeCount * 8 * 8) {
        traceResolution = 8;
    } else if (rayCount == probeCount * 16 * 16) {
        traceResolution = 16;
    } else {
        spdlog::error("Lumen probe visualization requires an 8×8 or 16×16 trace buffer for every probe slot");
        return makeError(Error::InvalidArgument);
    }
    if (distanceBytes != rayCount * sizeof(float) || probeRadiance.desc().size != rayCount * kRadianceStride) {
        spdlog::error("Lumen probe visualization requires corresponding trace distance and radiance buffers");
        return makeError(Error::InvalidArgument);
    }

    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    const auto storage = [&](const char* name) { return writer.storageImageHandle(context.outputTexture(name).view()); };
    const LumenProbeVisualizeParams params{
        .probePositions = writer.buffer(context.inputBuffer("probePositions").buffer()),
        .probeRadiance = writer.buffer(probeRadiance.buffer()),
        .filteredProbeRadiance = writer.buffer(context.inputBuffer("filteredProbeRadiance").buffer()),
        .probeDistance = writer.buffer(probeDistance.buffer()),
        .sceneColor = writer.sampledImageHandle(context.inputTexture("sceneColor").view()),
        .historyWeight = writer.sampledImageHandle(context.inputTexture("historyWeight").view()),
        .placement = storage("placement"), .traceSource = storage("traceSource"),
        .rawRadiance = storage("rawRadiance"), .filteredRadiance = storage("filteredRadiance"),
        .hitDistance = storage("hitDistance"), .historyWeightView = storage("historyWeightView"),
        .width = context.width(), .height = context.height(), .probeWidth = probeWidth, .probeHeight = probeHeight,
        .probeSlot = properties.value("probeSlot", 0u), .traceResolution = traceResolution, .distanceRange = distanceRange,
    };
    const auto encoded = writer.encode(params, kProbeVisualizeABI);
    if (!encoded) { return makeError(encoded.error()); }
    return kernel_.dispatch(commands, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenProbeVisualizePass()
{
    return std::make_unique<LumenProbeVisualizePass>();
}

} // namespace metallic::render::builtin_pass
