#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

bool isComparisonSource(TextureHandle source, uint32_t width, uint32_t height)
{
    if (!source.valid() || source.view() == nullptr ||
        source.desc().width != width || source.desc().height != height) {
        return false;
    }
    // Both inputs must contain color in the same space, at the same pixel resolution.
    switch (source.desc().format) {
    case Format::RGBA8Unorm:
    case Format::BGRA8Unorm:
    case Format::RGBA16Sfloat:
    case Format::RGBA32Sfloat:
    case Format::RG32Sfloat:
    case Format::R32Sfloat:
    case Format::B10G11R11UfloatPack32:
        return true;
    default:
        return false;
    }
}

class SliderDebugPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("sourceA", "A: left / top; same extent and color space as B").sampledRead();
        reflection.addTextureInput("sourceB", "B: right / bottom; same extent and color space as A").sampledRead();
        reflection.addTextureOutput("color", "Pixel-aligned comparison; preserves HDR and alpha")
            .storageWrite().transient(RenderGraphInitialization::FullOverwrite)
            .format = Format::RGBA32Sfloat;
        return reflection;
    }

    void prepareResourceMetadata(RenderGraphExecutionContext& context) const override
    {
        if (const auto* source = context.input("sourceA")) {
            if (auto* color = context.output("color")) { color->colorEncoding = source->colorEncoding; }
        }
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeFloatSetting("splitPosition", "Split Position", 0.5f, 0.0f, 1.0f),
            runtimeEnumSetting("orientation", "Divider", "vertical",
                {{"Vertical (left / right)", "vertical"}, {"Horizontal (top / bottom)", "horizontal"}}),
            runtimeBoolSetting("swapSides", "Swap A / B", false),
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (context.device == nullptr) { return makeError(Error::InvalidArgument); }
        if (program_.valid()) { return {}; }
        ShaderCompileResult shader;
        Result<> result = ShaderRegistry::instance().getShader({.moduleName = "Features/Debug/SliderDebug",
            .entryPointName = "sliderDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .parameters = parameterAbi<SliderDebugParams>(kSliderDebugABI, ParameterTransport::InlinePush),
            .debugName = "SliderDebug",
        }, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto sourceA = context.inputTexture("sourceA");
        const auto sourceB = context.inputTexture("sourceB");
        const auto color = context.outputTexture("color");
        if (!isComparisonSource(sourceA, context.width(), context.height()) ||
            !isComparisonSource(sourceB, context.width(), context.height()) ||
            !color.valid() || color.view() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const auto* inputA = context.input("sourceA");
        const auto* inputB = context.input("sourceB");
        if (!inputA || !inputB || inputA->colorEncoding != inputB->colorEncoding) {
            return makeError(Error::InvalidArgument);
        }
        const auto& properties = context.properties();
        const float split = properties.value("splitPosition", 0.5f);
        const SliderDebugPush push{
            std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f,
            properties.value("orientation", "vertical") == "horizontal" ? 1u : 0u,
            properties.value("swapSides", false) ? 1u : 0u,
        };
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(commands));
        SliderDebugParams params{};
        params.sourceA = writer.sampledImageHandle(sourceA.view());
        params.sourceB = writer.sampledImageHandle(sourceB.view());
        params.output = writer.storageImageHandle(color.view());
        params.display = push;
        auto encoded = writer.encode(params, kSliderDebugABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return program_.dispatch(commands, *encoded, (context.width() + 7) / 8, (context.height() + 7) / 8);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSliderDebugPass()
{
    return std::make_unique<SliderDebugPass>();
}

} // namespace metallic::render::builtin_pass
