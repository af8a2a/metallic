#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/SlangCompiler.h"

namespace metallic::render::builtin_pass {
namespace {

bool isColorSource(TextureHandle source)
{
    if (!source.valid() || source.view() == nullptr ||
        source.desc().width == 0 || source.desc().height == 0) {
        return false;
    }
    // Integer IDs and depth require their own visualization before presentation.
    switch (source.desc().format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::Rg8Unorm:
    case Format::Rg8Snorm:
    case Format::Bgra8Unorm:
    case Format::Bgra8Srgb:
    case Format::Rgba8Unorm:
    case Format::Rgba8Snorm:
    case Format::Rgba8Srgb:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Sfloat:
    case Format::Rg16Unorm:
    case Format::Rg16Snorm:
    case Format::Rg16Sfloat:
    case Format::Rgba16Unorm:
    case Format::Rgba16Snorm:
    case Format::Rgba16Sfloat:
    case Format::R32Sfloat:
    case Format::Rg32Sfloat:
    case Format::Rgb32Sfloat:
    case Format::Rgba32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::Bgra4Unorm:
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
        auto& color = reflection.addTextureOutput("color", "Final image presented by the viewport to the swapchain");
        color.storageReadWrite();
        const bool hdr = context.displayOutput.mode == DisplayOutputMode::HdrScRgb;
        color.format = hdr ? Format::Rgba16Sfloat : Format::Rgba8Unorm;
        color.colorEncoding = hdr ? DisplayColorEncoding::ScRgb : DisplayColorEncoding::Srgb;
        color.presentationOutput = true;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeEnumSetting("inputEncoding", "Input Color", "auto",
                {{"Automatic", "auto"}, {"sRGB display color", "srgb"},
                    {"Exposed scene-linear", "linear"}, {"scRGB (absolute)", "scrgb"}}),
            runtimeBoolSetting("calibrationPattern", "HDR Calibration Pattern", false),
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        displayOutput_ = context.displayOutput;
        Result result = initializeProgram(*context.device, "finalBlitUvMain", false, uvProgram_, log);
        if (!result) {
            return result;
        }
        return initializeProgram(*context.device, "finalBlitMain", true, blitProgram_, log);
    }

    Result execute(RenderGraphExecutionContext& context) override
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
            ? context.input("source")->colorEncoding : DisplayColorEncoding::Srgb;
        const auto inputEncoding = context.properties().value("inputEncoding", "auto");
        if (inputEncoding == "srgb") { encoding = DisplayColorEncoding::Srgb; }
        if (inputEncoding == "linear") { encoding = DisplayColorEncoding::ExposedLinear; }
        if (inputEncoding == "scrgb") { encoding = DisplayColorEncoding::ScRgb; }
        // sRGB texture sampling already decodes the transfer function.
        const bool sampledSrgb = sampleSource &&
            (source.desc().format == Format::Rgba8Srgb || source.desc().format == Format::Bgra8Srgb);
        const Push push{displayOutput_.mode == DisplayOutputMode::HdrScRgb ? 1u : 0u,
            static_cast<uint32_t>(encoding), calibration ? 1u : 0u, sampledSrgb ? 1u : 0u,
            displayOutput_.paperWhiteNits, displayOutput_.peakNits, std::exp2(displayOutput_.exposureEV), 0.0f};
        TextureView* sourceView = sampleSource ? source.view() : nullptr;
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureView = color.view()},
            {.binding = 1, .textureViews = &sourceView, .textureViewCount = 1},
        };
        ComputeProgram& program = sampleSource ? blitProgram_ : uvProgram_;
        return program.dispatch(ComputeDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = sampleSource ? 2u : 1u,
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
        });
    }

private:
    struct Push {
        uint32_t hdr, inputEncoding, calibration, sampledSrgb;
        float paperWhiteNits, peakNits, exposure, padding;
    };
    static_assert(sizeof(Push) == 32);

    static Result initializeProgram(
        Device& device, const char* entryPoint, bool sampleSource,
        ComputeProgram& program, std::string& log)
    {
        if (program.valid()) {
            return {};
        }
        ShaderCompileResult shader;
        Result result = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = "Features/PostProcess/FinalBlit",
            .entryPointName = entryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        }, shader);
        if (!result) {
            log += std::string("FinalBlit shader compilation failed: ") + shader.diagnostics;
            return result;
        }
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = 1, .kind = ComputeResourceBindingKind::SampledImage},
        };
        return program.initialize(device, ComputeProgramDesc{
            .spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t),
            .pushConstantSize = sizeof(Push),
            .bindings = bindings,
            .bindingCount = sampleSource ? 2u : 1u,
            .debugName = entryPoint,
            .requiresRayQuery = false,
        }, log);
    }

    ComputeProgram uvProgram_;
    ComputeProgram blitProgram_;
    DisplayOutputParameters displayOutput_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createFinalBlitPass()
{
    return std::make_unique<FinalBlitPass>();
}

} // namespace metallic::render::builtin_pass
