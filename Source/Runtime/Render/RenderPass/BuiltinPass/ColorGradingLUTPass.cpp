#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Core/ColorGrading.h"

namespace metallic::render::builtin_pass {
namespace {

class ColorGradingLUTPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& lut = reflection.addTextureOutput("lut", "Log-shaped RGB cube; display sRGB or absolute scRGB");
        lut.texture3D(64, 64, 64).storageWrite();
        lut.format = Format::RGBA16Sfloat;
        lut.colorEncoding =
            isHDROutput(context.displayOutput.mode) ? DisplayColorEncoding::scRGB : DisplayColorEncoding::sRGB;
        // Persistent graph allocation. Rebuild every execution to keep settings,
        // graph reload and overlapping frames coherent without a private cache.
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        auto transform = runtimeEnumSetting("toneCurve", "Display Transform", "aces2",
                                            {{"ACES 2.0", "aces2"},
                                             {"Unreal Film / ACES 1.3", "unreal"},
                                             {"Reinhard", "reinhard"},
                                             {"Exponential", "exponential"},
                                             {"None", "none"}});
        transform.rebuildGraph = true;
        auto settings = colorGradingSettings();
        settings.insert(settings.begin(), transform);
        return settings;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (!context.device) {
            return makeError(Error::InvalidArgument);
        }
        output_ = context.displayOutput;
        const auto transform = properties().value("toneCurve", "aces2");
        transform_ =
            transform == "aces2"
                ? 4
                : (transform == "unreal" ? 3 : (transform == "none" ? 2 : (transform == "exponential" ? 1 : 0)));
        auto result = resources_.initialize(
            *context.device, properties(), isHDROutput(output_.mode) ? output_.peakNits : 100.0f, transform_ == 4, log);
        if (!result) {
            return result;
        }
        return ShaderRegistry::instance().getComputeKernel(*context.device,
            {.moduleName = "Features/PostProcess/ColorGradingLUT",
                .entryPointName = "composeColorGradingLUT", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<ColorGradingLUTParams>(kColorGradingLUTABI), .debugName = "ColorGradingLUT"},
            program_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto lut = context.outputTexture("lut");
        if (!lut.valid() || !lut.view() || lut.desc().type != TextureType::Texture3D) {
            return makeError(Error::InvalidArgument);
        }
        const GradingPush push{transform_, isHDROutput(output_.mode) ? 1u : 0u,
                        isHDROutput(output_.mode) ? output_.peakNits : 100.0f, output_.paperWhiteNits,
                        colorGradingParameters(context.properties())};
        auto views = resources_.views();
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(commands));
        ColorGradingLUTParams params{};
        params.output = writer.storageImageHandle(lut.view());
        params.custom0 = writer.sampledImageHandle(views[0]);
        params.custom1 = writer.sampledImageHandle(views[1]);
        params.custom2 = writer.sampledImageHandle(views[2]);
        params.custom3 = writer.sampledImageHandle(views[3]);
        params.reach = writer.sampledImageHandle(views[4]);
        params.gamut = writer.sampledImageHandle(views[5]);
        params.gammaTable = writer.sampledImageHandle(views[6]);
        params.sampler = writer.samplerHandle(SamplerDesc{});
        params.display = push;
        auto encoded = writer.encode(params, kColorGradingLUTABI);
        if (!encoded) { return makeError(encoded.error()); }
        return program_.dispatch(commands, *encoded, 16, 16, 16);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel program_;
    ColorGradingResources resources_;
    DisplayOutputParameters output_;
    uint32_t transform_ = 4;
};
} // namespace

std::unique_ptr<RenderGraphPass> createColorGradingLUTPass()
{
    return std::make_unique<ColorGradingLUTPass>();
}
} // namespace metallic::render::builtin_pass
