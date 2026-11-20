#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kScreenProbeGatherABI = 0x4c554d454e010005ull;
constexpr uint32_t kProbeDownsampleFactor = 16;
constexpr uint32_t kRaysPerProbe = 8 * 8;
constexpr uint32_t kTraceStride = 4 * sizeof(float);

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
};
static_assert(sizeof(LumenScreenProbeGatherParams) == 100);

class LumenScreenProbeGatherPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    ComputeKernel trace_;
    ComputeKernel integrate_;
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

    const uint64_t probeCount = uint64_t(screenProbeCount(context.width)) * screenProbeCount(context.height);
    reflection.addBufferOutput("probeRadiance", "Probe ray RGB and GDF classification in tile order")
        .buffer(probeCount * kRaysPerProbe * kTraceStride, kTraceStride).storageReadWrite()
        .transient(RenderGraphInitialization::FullOverwrite);

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
    const auto distance = properties().find("maxTraceDistance");
    if (distance != properties().end()) {
        if (!distance->is_number() || !std::isfinite(distance->get<float>()) || distance->get<float>() <= 0.0001f) {
            log += "Lumen Screen Probe Gather maxTraceDistance must be finite and greater than 0.0001 meters";
            return makeError(Error::InvalidArgument);
        }
        maxTraceDistance_ = distance->get<float>();
    }

    const auto create = [&](const char* entry, const char* name, ComputeKernel& kernel) {
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenScreenProbeGather", .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<LumenScreenProbeGatherParams>(kScreenProbeGatherABI), .debugName = name}, kernel, log);
    };
    if (auto result = create("lumenTraceScreenProbesMain", "LumenTraceScreenProbes", trace_); !result) {
        return result;
    }
    return create("lumenIntegrateScreenProbesMain", "LumenIntegrateScreenProbes", integrate_);
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
    const auto encoded = writer.encode(params, kScreenProbeGatherABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse traceUses[] = {
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"pageTable", Access::TextureSampleRead}, {"pageAtlas", Access::TextureSampleRead},
        {"coarseMip", Access::TextureSampleRead}, {"domain", Access::BufferStorageRead},
        {"objectGrid", Access::BufferStorageRead}, {"objectGridScene", Access::BufferStorageRead},
        {"cards", Access::BufferStorageRead}, {"cardScene", Access::BufferStorageRead},
        {"cardDepth", Access::TextureSampleRead}, {"cardLighting", Access::TextureSampleRead},
        {"probeRadiance", Access::BufferStorageWrite},
    };
    const RenderGraphStageUse integrateUses[] = {
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"diffuse", Access::TextureSampleRead}, {"directColor", Access::TextureSampleRead},
        {"probeRadiance", Access::BufferStorageRead}, {"indirectDiffuse", Access::TextureStorageWrite},
        {"color", Access::TextureStorageWrite},
    };
    const RenderGraphStage stages[] = {
        {"Trace uniform screen probes", traceUses, [&](CommandBuffer& cmd) {
            return trace_.dispatch(cmd, *encoded, params.probeWidth, params.probeHeight, 1);
        }},
        {"Integrate and composite diffuse", integrateUses, [&](CommandBuffer& cmd) {
            return integrate_.dispatch(cmd, *encoded, (params.width + 7u) / 8u, (params.height + 7u) / 8u, 1);
        }},
    };
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenScreenProbeGatherPass()
{
    return std::make_unique<LumenScreenProbeGatherPass>();
}

} // namespace metallic::render::builtin_pass
