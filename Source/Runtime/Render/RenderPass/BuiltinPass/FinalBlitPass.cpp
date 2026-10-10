#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::render::builtin_pass {
namespace {

bool isColorSource(TextureHandle source)
{
    if (!source.valid() || source.view() == nullptr || source.desc().type != TextureType::Texture2D ||
        source.desc().width == 0 || source.desc().height == 0) {
        return false;
    }
    // Integer IDs and depth require their own visualization before presentation.
    switch (source.desc().format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::RG8Unorm:
    case Format::RG8Snorm:
    case Format::BGRA8Unorm:
    case Format::BGRA8sRGB:
    case Format::RGBA8Unorm:
    case Format::RGBA8Snorm:
    case Format::RGBA8sRGB:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Sfloat:
    case Format::RG16Unorm:
    case Format::RG16Snorm:
    case Format::RG16Sfloat:
    case Format::RGBA16Unorm:
    case Format::RGBA16Snorm:
    case Format::RGBA16Sfloat:
    case Format::R32Sfloat:
    case Format::RG32Sfloat:
    case Format::RGB32Sfloat:
    case Format::RGBA32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::BGRA4Unorm:
        return true;
    default:
        return false;
    }
}

class FinalBlitPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override
    {
        RenderPassReflection reflection;
        auto& source = reflection.addTextureInput("source", "Color RT; unavailable input displays UV");
        source.sampledRead().setOptional();
        source.matchOutputExtent = false;
        auto& lut = reflection.addTextureInput("lut", "3D display LUT from ColorGradingLUTPass");
        lut.texture3D(0, 0, 0).sampledRead().setOptional();
        lut.format = Format::RGBA16Sfloat;
        auto& color = reflection.addTextureOutput("color", "Final image presented by the viewport to the swapchain");
        color.texture2D(context.presentationWidth != 0 ? context.presentationWidth : context.width,
            context.presentationHeight != 0 ? context.presentationHeight : context.height);
        color.storageWrite().transient(RenderGraphInitialization::FullOverwrite);
        // Both HDR profiles retain linear scRGB until UI composition is complete.
        const bool hdr = isHDROutput(context.displayOutput.mode);
        color.format = hdr ? Format::RGBA16Sfloat : Format::RGBA8Unorm;
        color.colorEncoding = hdr ? DisplayColorEncoding::scRGB : DisplayColorEncoding::sRGB;
        color.presentationOutput = true;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeEnumSetting("inputEncoding", "Input Color", "auto",
                {{"Automatic", "auto"}, {"sRGB display color", "srgb"},
                    {"Exposed scene-linear", "linear"}, {"Linear display Rec.709", "displayLinearRec709"},
                    {"scRGB (absolute)", "scrgb"}}),
            runtimeBoolSetting("calibrationPattern", "HDR Calibration Pattern", false),
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        displayOutput_ = context.displayOutput;
        Result<> result = initializeProgram(*context.device, "finalBlitUvMain", false, uvProgram_, log);
        if (!result) {
            return result;
        }
        result = initializeProgram(*context.device, "finalBlitMain", false, blitProgram_, log);
        if (!result) { return result; }
        return initializeProgram(*context.device, "finalBlitMain", true, lutProgram_, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const TextureHandle color = context.outputTexture("color");
        if (!color.valid() || color.view() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const TextureHandle source = context.inputTexture("source");
        const bool calibration = displayOutput_.calibrationPattern ||
            context.properties().value("calibrationPattern", false);
        const bool sampleSource = !calibration && isColorSource(source);
        auto encoding = context.input("source") != nullptr
            ? context.input("source")->colorEncoding : DisplayColorEncoding::sRGB;
        const auto inputEncoding = context.properties().value("inputEncoding", "auto");
        if (inputEncoding == "srgb") { encoding = DisplayColorEncoding::sRGB; }
        if (inputEncoding == "linear") { encoding = DisplayColorEncoding::ExposedLinear; }
        if (inputEncoding == "scrgb") { encoding = DisplayColorEncoding::scRGB; }
        if (inputEncoding == "displayLinearRec709") { encoding = DisplayColorEncoding::DisplayLinearRec709; }
        // sRGB texture sampling already decodes the transfer function.
        const bool sampledSrgb = sampleSource &&
            (source.desc().format == Format::RGBA8sRGB || source.desc().format == Format::BGRA8sRGB);
        const auto toneCurve = context.properties().value("toneCurve", "reinhard");
        const auto lut = context.inputTexture("lut");
        if (lut.valid() && (lut.desc().type != TextureType::Texture3D ||
            lut.desc().format != Format::RGBA16Sfloat || lut.desc().width < 2 ||
            lut.desc().width != lut.desc().height || lut.desc().width != lut.desc().depth)) {
            return makeError(Error::InvalidArgument);
        }
        const bool hasLut = lut.valid() && lut.view() && lut.desc().type == TextureType::Texture3D;
        const bool sceneInput = encoding == DisplayColorEncoding::SceneLinear || encoding == DisplayColorEncoding::ExposedLinear;
        if (!hasLut && sampleSource && sceneInput && (toneCurve == "aces2" || toneCurve == "unreal")) {
            return makeError(Error::InvalidArgument); // Migrate grading to ColorGradingLUTPass.
        }
        const DisplayOutputPush push{isHDROutput(displayOutput_.mode) ? 1u : 0u,
            static_cast<uint32_t>(encoding), calibration ? 1u : 0u, sampledSrgb ? 1u : 0u,
            displayOutput_.paperWhiteNits, displayOutput_.peakNits, std::exp2(displayOutput_.exposureEV),
            toneCurve == "aces2" ? 4u : (toneCurve == "unreal" ? 3u : (toneCurve == "none" ? 2u : (toneCurve == "exponential" ? 1u : 0u))),
            hasLut ? 1u : 0u};
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(commands));
        FinalBlitParams params{};
        params.output = writer.storageImageHandle(color.view());
        params.display = push;
        if (sampleSource) { params.source = writer.sampledImageHandle(source.view()); }
        if (hasLut && sampleSource) {
            params.lut = writer.sampledImageHandle(lut.view());
            params.lutSampler = writer.samplerHandle(SamplerDesc{});
        }
        auto encoded = writer.encode(params, kFinalBlitABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        ComputeKernel& program = sampleSource ? (hasLut ? lutProgram_ : blitProgram_) : uvProgram_;
        return program.dispatch(commands, *encoded, (context.width() + 7) / 8, (context.height() + 7) / 8);
    }

private:
    Device* device_ = nullptr;

    static Result<> initializeProgram(
        Device& device, const char* entryPoint, bool withLut,
        ComputeKernel& program, std::string& log)
    {
        if (program.valid()) {
            return {};
        }
        ShaderCompileResult shader;
        const SlangMacroDefine defines[] = {{"FINAL_USE_LUT", withLut ? "1" : "0"}};
        Result<> result = ShaderRegistry::instance().getShader(SlangShaderDesc{
            .moduleName = "Features/PostProcess/FinalBlit",
            .entryPointName = entryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .macroDefines = defines,
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) {
            log += std::string("FinalBlit shader compilation failed: ") + shader.diagnostics;
            return result;
        }
        return program.initialize(device, ComputeKernelDesc{
            .spirv = shader.spirv,
            .parameters = parameterAbi<FinalBlitParams>(kFinalBlitABI, ParameterTransport::InlinePush),
            .debugName = entryPoint,
        }, log);
    }

    ComputeKernel uvProgram_;
    ComputeKernel blitProgram_;
    ComputeKernel lutProgram_;
    DisplayOutputParameters displayOutput_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createFinalBlitPass()
{
    return std::make_unique<FinalBlitPass>();
}

} // namespace metallic::render::builtin_pass
