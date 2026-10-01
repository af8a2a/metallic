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
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/PostProcess/ColorGradingLUT",
                                            .entryPointName = "composeColorGradingLUT",
                                            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
                                           shader.diagnostics)
                     .transform([&](auto value) { shader = std::move(value); });
        if (!result) {
            log += shader.diagnostics;
            return result;
        }
        std::vector<ComputeProgramBindingDesc> bindings{
            {.binding = 0, .kind = ComputeResourceBindingKind::StorageImage}};
        for (uint32_t i = 1; i < 8; ++i) {
            bindings.push_back({.binding = i, .kind = ComputeResourceBindingKind::SampledImage});
        }
        bindings.push_back({.binding = 8, .kind = ComputeResourceBindingKind::Sampler});
        return program_.initialize(*context.device,
                                   {.spirv = shader.spirv,
                                    .pushConstantSize = sizeof(Push),
                                    .bindings = bindings,
                                    .debugName = "ColorGradingLUT",
                                    .requiresRayQuery = false},
                                   log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto lut = context.outputTexture("lut");
        if (!lut.valid() || !lut.view() || lut.desc().type != TextureType::Texture3D) {
            return makeError(Error::InvalidArgument);
        }
        const Push push{transform_, isHDROutput(output_.mode) ? 1u : 0u,
                        isHDROutput(output_.mode) ? output_.peakNits : 100.0f, output_.paperWhiteNits,
                        colorGradingParameters(context.properties())};
        auto views = resources_.views();
        const SamplerDesc sampler{};
        std::vector<ComputeDispatchBinding> bindings{{.binding = 0, .textureView = lut.view()}};
        for (uint32_t i = 0; i < views.size(); ++i) {
            bindings.push_back({.binding = i + 1, .textureViews = {&views[i], 1}});
        }
        bindings.push_back({.binding = 8, .sampler = &sampler});
        return program_.dispatch({.commandBuffer = &context.commandBuffer(),
                                  .bindings = bindings,
                                  .pushData = &push,
                                  .pushDataSize = sizeof(push),
                                  .groupCountX = 16,
                                  .groupCountY = 16,
                                  .groupCountZ = 16});
    }

private:
    struct Push {
        uint32_t transform, hdr;
        float peak, paperWhite;
        ColorGradingParameters grade;
    };
    static_assert(sizeof(Push) == 144);
    ComputeProgram program_;
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
