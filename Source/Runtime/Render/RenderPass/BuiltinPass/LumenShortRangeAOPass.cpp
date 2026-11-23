#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <spdlog/spdlog.h>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenShortRangeAOABI = 0x4c554d454e080001ull;

struct LumenShortRangeAOParams {
    ShaderSampledImage depth, normal, blueNoiseVec2;
    ShaderBuffer view;
    ShaderStorageImage shortRangeAORaw;
};
static_assert(sizeof(LumenShortRangeAOParams) == 20);

class LumenShortRangeAOPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsExternalFrameOverlap() const override { return false; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("depth", "Current full-resolution reversed-Z depth")
            .sampledRead().format = Format::D32Sfloat;
        reflection.addTextureInput("normal", "Current receiver world shading normal")
            .sampledRead().format = Format::RGBA16Sfloat;
        auto& raw = reflection.addTextureOutput("shortRangeAORaw", "Half-resolution packed world-space bent normal")
            .texture2D((context.width + 1u) / 2u, (context.height + 1u) / 2u)
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        raw.format = Format::R32Uint;
        raw.matchOutputExtent = false;
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.graphicsQueue) {
            log = "Lumen Short Range AO requires a graphics queue to upload blue noise";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
        if (!blueNoise) { return makeError(blueNoise.error()); }
        blueNoise_ = std::move(*blueNoise);
        return ShaderRegistry::instance().getComputeKernel(*device_,
            {.moduleName = "Features/DistanceField/LumenShortRangeAO", .entryPointName = "lumenShortRangeAOMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<LumenShortRangeAOParams>(kLumenShortRangeAOABI),
                .debugName = "LumenShortRangeAO"}, kernel_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto* view = context.viewConstantsBuffer();
        if (!view) {
            spdlog::error("Lumen Short Range AO requires graph view constants");
            return makeError(Error::InvalidArgument);
        }
        auto registry = ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
        const auto output = context.outputTexture("shortRangeAORaw");
        const LumenShortRangeAOParams params{
            writer.sampledImageHandle(context.inputTexture("depth").view()),
            writer.sampledImageHandle(context.inputTexture("normal").view()),
            writer.sampledImageHandle(blueNoise_.get()), writer.buffer(view), writer.storageImageHandle(output.view()),
        };
        const auto encoded = writer.encode(params, kLumenShortRangeAOABI);
        if (!encoded) { return makeError(encoded.error()); }
        return kernel_.dispatch(commands, *encoded, (output.desc().width + 7u) / 8u, (output.desc().height + 7u) / 8u, 1);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
    std::unique_ptr<TextureView> blueNoise_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createLumenShortRangeAOPass()
{
    return std::make_unique<LumenShortRangeAOPass>();
}

} // namespace metallic::render::builtin_pass
