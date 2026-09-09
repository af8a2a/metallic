#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

struct SliderDebugPush {
    float splitPosition;
    uint32_t horizontal;
    uint32_t swapSides;
};
static_assert(sizeof(SliderDebugPush) == 12);

bool isComparisonSource(TextureHandle source, uint32_t width, uint32_t height)
{
    if (!source.valid() || source.view() == nullptr ||
        source.desc().width != width || source.desc().height != height) {
        return false;
    }
    // Both inputs must contain color in the same space, at the same pixel resolution.
    switch (source.desc().format) {
    case Format::Rgba8Unorm:
    case Format::Bgra8Unorm:
    case Format::Rgba16Sfloat:
    case Format::Rgba32Sfloat:
    case Format::Rg32Sfloat:
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
            .storageReadWrite().format = Format::Rgba32Sfloat;
        return reflection;
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

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) { return makeError(Error::InvalidArgument); }
        if (program_.valid()) { return {}; }
        ShaderCompileResult shader;
        Result result = compileSlangShaderToSpirv({.moduleName = "Features/Debug/SliderDebug",
            .entryPointName = "sliderDebugMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log += shader.diagnostics; return result; }
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = ComputeResourceBindingKind::SampledImage},
            {.binding = 2, .kind = ComputeResourceBindingKind::StorageImage},
        };
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = sizeof(SliderDebugPush),
            .bindings = bindings, .bindingCount = 3, .debugName = "SliderDebug", .requiresRayQuery = false}, log);
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        const auto sourceA = context.inputTexture("sourceA");
        const auto sourceB = context.inputTexture("sourceB");
        const auto color = context.outputTexture("color");
        if (!isComparisonSource(sourceA, context.width(), context.height()) ||
            !isComparisonSource(sourceB, context.width(), context.height()) ||
            !color.valid() || color.view() == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        const auto& properties = context.properties();
        const float split = properties.value("splitPosition", 0.5f);
        const SliderDebugPush push{
            std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f,
            properties.value("orientation", "vertical") == "horizontal" ? 1u : 0u,
            properties.value("swapSides", false) ? 1u : 0u,
        };
        TextureView* viewA = sourceA.view();
        TextureView* viewB = sourceB.view();
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = &viewA, .textureViewCount = 1},
            {.binding = 1, .textureViews = &viewB, .textureViewCount = 1},
            {.binding = 2, .textureView = color.view()},
        };
        return program_.dispatch({.commandBuffer = &context.commandBuffer(),
            .bindings = bindings, .bindingCount = 3, .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8, .groupCountY = (context.height() + 7) / 8});
    }

private:
    ComputeProgram program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createSliderDebugPass()
{
    return std::make_unique<SliderDebugPass>();
}

} // namespace metallic::render::builtin_pass
