#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <cmath>
#include <array>
#include <vector>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kScreenProbeGatherABI = 0x4c554d454e010006ull;
constexpr uint64_t kScreenProbeFilterABI = 0x4c554d454e010007ull;
constexpr uint32_t kProbeDownsampleFactor = 16;
constexpr uint32_t kRaysPerProbe = 8 * 8;
constexpr uint32_t kTraceStride = 4 * sizeof(float);
constexpr uint32_t kSlotsPerTile = 9;
constexpr uint32_t kScreenTraceViewStride = 224;
constexpr uint32_t kSpatialPassCount = 3;

uint32_t screenProbeCount(uint32_t pixels)
{
    return (pixels + kProbeDownsampleFactor - 1) / kProbeDownsampleFactor;
}

struct LumenScreenProbeGatherParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderSampledImage position, normal, diffuse, directColor;
    ShaderBuffer traceOutput, traceInput;
    ShaderStorageImage indirectDiffuse, color;
    uint32_t width, height, probeWidth, probeHeight, clipmapCount;
    float maxTraceDistance;
    ShaderBuffer probePositions, traceDistance, closestHZB;
    ShaderSampledImage previousDepth, previousColor;
    ShaderBuffer traceView;
    uint32_t frameIndex;
};
static_assert(sizeof(LumenScreenProbeGatherParams) == 128);

struct LumenScreenProbeFilterParams {
    ShaderSampledImage position, normal;
    ShaderBuffer probeOutput, probeInput, inputRadiance, inputDistance, outputRadiance;
    uint32_t width, height, probeWidth, probeHeight, frameIndex;
};
static_assert(sizeof(LumenScreenProbeFilterParams) == 48);

class LumenScreenProbeGatherPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    bool optionEnabled(std::string_view name) const
    {
        const auto value = properties().find(name);
        return value != properties().end() && value->is_boolean() && value->get<bool>();
    }

    Device* device_ = nullptr;
    ComputeKernel trace_;
    ComputeKernel integrate_;
    ComputeKernel place_;
    ComputeKernel filter_;
    float maxTraceDistance_ = 4.0f;
};

RenderPassReflection LumenScreenProbeGatherPass::reflect(const RenderGraphCompileContext& context) const
{
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    reflection.addTextureInput("position", "Receiver world position and positive linear view depth")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("normal", "Receiver shading normal").sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("diffuse", "Receiver diffuse response").sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("directColor", "Unexposed deferred scene color before indirect diffuse")
        .sampledRead().format = Format::RGBA32Sfloat;

    if (optionEnabled("screenTraces")) {
        reflection.addBufferInput("closestHZB", "Current closest reversed-Z depth hierarchy")
            .buffer(0, sizeof(float)).storageRead();
        reflection.addTextureInput("previousDepth", "Previous device depth").sampledRead().format = Format::R32Sfloat;
        reflection.addTextureInput("previousColor", "Previous scene-linear direct and emissive color")
            .sampledRead().format = Format::RGBA32Sfloat;
        reflection.addBufferInput("traceView", "Screen trace view and history validity")
            .buffer(kScreenTraceViewStride, kScreenTraceViewStride).storageRead();
    }

    const uint64_t probeCount = uint64_t(screenProbeCount(context.width)) * screenProbeCount(context.height);
    const bool finalGather = optionEnabled("finalGather");
    const uint64_t rayCount = probeCount * (finalGather ? kSlotsPerTile : 1u) * kRaysPerProbe;
    reflection.addBufferOutput("probeRadiance", "Probe ray RGB and trace classification in tile order")
        .buffer(rayCount * kTraceStride, kTraceStride).storageReadWrite()
        .transient(RenderGraphInitialization::FullOverwrite);
    if (finalGather) {
        reflection.addBufferOutput("probePositions", "Uniform and adaptive receiver pixels by tile and slot")
            .buffer(probeCount * kSlotsPerTile * 2 * sizeof(uint32_t), 2 * sizeof(uint32_t)).storageReadWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        reflection.addBufferOutput("probeDistance", "Per-direction trace distance in meters")
            .buffer(rayCount * sizeof(float), sizeof(float)).storageReadWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        reflection.addBufferOutput("filteredProbeRadiance", "Spatially filtered probe ray RGB")
            .buffer(rayCount * kTraceStride, kTraceStride).storageReadWrite().transient(RenderGraphInitialization::FullOverwrite);
        reflection.addBufferOutput("probeFilterScratch", "Spatial filtering ping-pong buffer")
            .buffer(rayCount * kTraceStride, kTraceStride).storageReadWrite().transient(RenderGraphInitialization::FullOverwrite);
    }

    auto& indirect = reflection.addTextureOutput("indirectDiffuse", "Unexposed indirect diffuse after receiver response")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    indirect.format = Format::RGBA16Sfloat;
    indirect.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& color = reflection.addTextureOutput("color", "Unexposed scene color with indirect diffuse")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenScreenProbeGatherPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    for (std::string_view name : {"finalGather", "screenTraces", "stochasticInterpolation"}) {
        const auto value = properties().find(name);
        if (value != properties().end() && !value->is_boolean()) {
            log += "Lumen Screen Probe Gather requires a boolean " + std::string(name);
            return makeError(Error::InvalidArgument);
        }
    }
    const bool finalGather = optionEnabled("finalGather");
    const bool screenTraces = optionEnabled("screenTraces");
    const bool stochasticInterpolation = optionEnabled("stochasticInterpolation");
    if (screenTraces && !finalGather) {
        log += "Lumen screen tracing requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    if (stochasticInterpolation && !finalGather) {
        log += "Lumen stochastic probe interpolation requires Final Gather";
        return makeError(Error::InvalidArgument);
    }
    const auto distance = properties().find("maxTraceDistance");
    if (distance != properties().end()) {
        if (!distance->is_number() || !std::isfinite(distance->get<float>()) || distance->get<float>() <= 0.0001f) {
            log += "Lumen Screen Probe Gather maxTraceDistance must be finite and greater than 0.0001 meters";
            return makeError(Error::InvalidArgument);
        }
        maxTraceDistance_ = distance->get<float>();
    }

    std::vector<SlangMacroDefine> defines;
    if (finalGather) { defines.push_back({"METALLIC_LUMEN_FINAL_GATHER", "1"}); }
    if (screenTraces) { defines.push_back({"METALLIC_LUMEN_SCREEN_TRACES", "1"}); }
    if (stochasticInterpolation) { defines.push_back({"METALLIC_LUMEN_STOCHASTIC_INTERPOLATION", "1"}); }
    const auto createGather = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeGather", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .macroDefines = defines},
            {.parameters = parameterAbi<LumenScreenProbeGatherParams>(kScreenProbeGatherABI), .debugName = name}, kernel, log);
    };
    if (auto result = createGather("lumenTraceScreenProbesMain", "LumenTraceScreenProbes", trace_); !result) {
        return result;
    }
    if (auto result = createGather("lumenIntegrateScreenProbesMain", "LumenIntegrateScreenProbes", integrate_); !result || !finalGather) {
        return result;
    }

    const auto createFilter = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeFilter", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<LumenScreenProbeFilterParams>(kScreenProbeFilterABI), .debugName = name}, kernel, log);
    };
    if (auto result = createFilter("lumenPlaceAdaptiveProbesMain", "LumenPlaceAdaptiveProbes", place_); !result) {
        return result;
    }
    return createFilter("lumenFilterScreenProbesMain", "LumenFilterScreenProbes", filter_);
}

Result<> LumenScreenProbeGatherPass::execute(RenderGraphExecutionContext& context)
{
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenScreenProbeGatherParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    params.position = writer.sampledImageHandle(context.inputTexture("position").view());
    params.normal = writer.sampledImageHandle(context.inputTexture("normal").view());
    params.diffuse = writer.sampledImageHandle(context.inputTexture("diffuse").view());
    params.directColor = writer.sampledImageHandle(context.inputTexture("directColor").view());
    params.traceOutput = writer.buffer(context.outputBuffer("probeRadiance").buffer());
    params.traceInput = params.traceOutput;
    params.indirectDiffuse = writer.storageImageHandle(context.outputTexture("indirectDiffuse").view());
    params.color = writer.storageImageHandle(context.outputTexture("color").view());
    params.width = context.width();
    params.height = context.height();
    params.probeWidth = screenProbeCount(params.width);
    params.probeHeight = screenProbeCount(params.height);
    params.clipmapCount = *clipmapCount;
    params.maxTraceDistance = maxTraceDistance_;
    params.frameIndex = static_cast<uint32_t>(context.frameIndex());
    const bool finalGather = optionEnabled("finalGather");
    const bool screenTraces = optionEnabled("screenTraces");
    ShaderBuffer scratch;
    if (finalGather) {
        params.probePositions = writer.buffer(context.outputBuffer("probePositions").buffer());
        params.traceDistance = writer.buffer(context.outputBuffer("probeDistance").buffer());
        params.traceInput = writer.buffer(context.outputBuffer("filteredProbeRadiance").buffer());
        scratch = writer.buffer(context.outputBuffer("probeFilterScratch").buffer());
    }
    if (screenTraces) {
        params.closestHZB = writer.buffer(context.inputBuffer("closestHZB").buffer());
        params.previousDepth = writer.sampledImageHandle(context.inputTexture("previousDepth").view());
        params.previousColor = writer.sampledImageHandle(context.inputTexture("previousColor").view());
        params.traceView = writer.buffer(context.inputBuffer("traceView").buffer());
    }
    const auto encoded = writer.encode(params, kScreenProbeGatherABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    std::vector<RenderGraphStageUse> traceUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead},
        {"objectGrid", Access::BufferStorageRead}, {"objectGridScene", Access::BufferStorageRead},
        {"cards", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardLighting", Access::TextureSampleRead},
        {"probeRadiance", Access::BufferStorageWrite},
    };
    std::vector<RenderGraphStageUse> integrateUses{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"diffuse", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
        {finalGather ? "filteredProbeRadiance" : "probeRadiance", Access::BufferStorageRead},
        {"indirectDiffuse", Access::TextureStorageWrite},
        {"color", Access::TextureStorageWrite},
    };
    if (screenTraces) {
        traceUses.insert(traceUses.end(), {{"closestHZB", Access::BufferStorageRead},
            {"previousDepth", Access::TextureSampleRead}, {"previousColor", Access::TextureSampleRead},
            {"traceView", Access::BufferStorageRead}});
    }

    const RenderGraphStageUse placeUses[] = {{"position", Access::TextureSampleRead},
        {"normal", Access::TextureSampleRead}, {"probePositions", Access::BufferStorageWrite}};
    std::array<std::array<RenderGraphStageUse, 5>, kSpatialPassCount> filterUses;
    EncodedParameters placeParams;
    std::array<EncodedParameters, kSpatialPassCount> filteredParams;
    std::vector<RenderGraphStage> stages;
    if (finalGather) {
        traceUses.insert(traceUses.end(), {{"probePositions", Access::BufferStorageRead},
            {"probeDistance", Access::BufferStorageWrite}});
        integrateUses.push_back({"probePositions", Access::BufferStorageRead});
        LumenScreenProbeFilterParams filterParams{params.position, params.normal, params.probePositions,
            params.probePositions, {}, params.traceDistance, {}, params.width, params.height, params.probeWidth,
            params.probeHeight, params.frameIndex};
        auto placement = writer.encode(filterParams, kScreenProbeFilterABI);
        if (!placement) { return makeError(placement.error()); }
        placeParams = std::move(*placement);
        stages.push_back({"Place uniform and adaptive probes", placeUses, [&](CommandBuffer& cmd) {
            return place_.dispatch(cmd, placeParams, params.probeWidth, params.probeHeight, 1);
        }});

        constexpr std::array inputNames{"probeRadiance", "filteredProbeRadiance", "probeFilterScratch"};
        constexpr std::array outputNames{"filteredProbeRadiance", "probeFilterScratch", "filteredProbeRadiance"};
        const std::array inputs{params.traceOutput, params.traceInput, scratch};
        const std::array outputs{params.traceInput, scratch, params.traceInput};
        for (uint32_t index = 0; index < kSpatialPassCount; ++index) {
            filterUses[index] = {{{"position", Access::TextureSampleRead}, {"probePositions", Access::BufferStorageRead},
                {"probeDistance", Access::BufferStorageRead}, {inputNames[index], Access::BufferStorageRead},
                {outputNames[index], Access::BufferStorageWrite}}};
            filterParams.inputRadiance = inputs[index];
            filterParams.outputRadiance = outputs[index];
            auto filter = writer.encode(filterParams, kScreenProbeFilterABI);
            if (!filter) { return makeError(filter.error()); }
            filteredParams[index] = std::move(*filter);
        }
    }
    stages.push_back({"Trace screen probes", traceUses, [&](CommandBuffer& cmd) {
        return trace_.dispatch(cmd, *encoded, params.probeWidth, params.probeHeight, finalGather ? kSlotsPerTile : 1);
    }});
    for (uint32_t index = 0; finalGather && index < kSpatialPassCount; ++index) {
        stages.push_back({"Spatially filter probe radiance", filterUses[index], [&, index](CommandBuffer& cmd) {
            return filter_.dispatch(cmd, filteredParams[index], params.probeWidth, params.probeHeight, kSlotsPerTile);
        }});
    }
    stages.push_back({"Integrate and composite diffuse", integrateUses, [&](CommandBuffer& cmd) {
        return integrate_.dispatch(cmd, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
    }});
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenScreenProbeGatherPass()
{
    return std::make_unique<LumenScreenProbeGatherPass>();
}

} // namespace metallic::render::builtin_pass
