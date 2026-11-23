#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenBlueNoise.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <cmath>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenReflectionABI = 0x4c554d454e050001ull;
constexpr char kShaderModule[] = "Features/DistanceField/LumenReflection";

struct LumenReflectionParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderSampledImage position, normal, specular, sceneColor, blueNoise;
    ShaderStorageImage outputReflections, trace, color;
    ShaderSampledImage reflections;
    ViewCameraConstants camera;
    uint32_t width, height, clipmapCount, frameIndex;
    float maxTraceDistance, maxRoughness;
};
static_assert(offsetof(LumenReflectionParams, camera) == 80 && offsetof(LumenReflectionParams, width) == 160 &&
    offsetof(LumenReflectionParams, maxTraceDistance) == 176 && sizeof(LumenReflectionParams) == 192);

class LumenReflectionPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;
    bool supportsFrameOverlap() const override { return true; }

private:
    Device* device_ = nullptr;
    std::unique_ptr<TextureView> blueNoise_;
    ComputeKernel trace_;
    ComputeKernel composite_;
    float maxTraceDistance_ = 200.0f;
    float maxRoughness_ = 0.4f;
};

RenderPassReflection LumenReflectionPass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    reflection.addTextureInput("position", "World receiver position and positive linear depth")
        .sampledRead().format = Format::RGBA32Sfloat;
    reflection.addTextureInput("normal", "World receiver shading normal")
        .sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("specular", "Integrated receiver specular response and perceptual roughness")
        .sampledRead().format = Format::RGBA16Sfloat;
    reflection.addTextureInput("sceneColor", "Scene-linear color before specular reflections")
        .sampledRead().format = Format::RGBA32Sfloat;

    auto& reflections = reflection.addTextureOutput("reflections", "Raw scene-linear reflection radiance and roughness fade")
        .storageWrite().stageAccess(RenderGraphResourceAccess::TextureSampleRead);
    reflections.format = Format::RGBA16Sfloat;
    reflections.colorEncoding = DisplayColorEncoding::SceneLinear;
    reflection.addTextureOutput("trace", "Hit distance in meters, GDF classification, clipmap index and card cache classification")
        .storageWrite().format = Format::RGBA32Sfloat;
    auto& color = reflection.addTextureOutput("color", "Scene-linear color with the reflection response applied")
        .storageWrite();
    color.format = Format::RGBA32Sfloat;
    color.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenReflectionPass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    if (!context.device || !context.graphicsQueue) {
        log = "Lumen Reflection requires a rendering device and a graphics queue";
        return makeError(Error::InvalidArgument);
    }
    try {
        // UE 5.7.4 FPostProcessSettings defaults to 20000 cm and roughness 0.4.
        maxTraceDistance_ = properties().value("maxTraceDistance", 200.0f);
        maxRoughness_ = properties().value("maxRoughness", 0.4f);
    } catch (const RenderGraphProperties::exception& error) {
        log = "Invalid Lumen Reflection properties: " + std::string(error.what());
        return makeError(Error::InvalidArgument);
    }
    if (!(std::isfinite(maxTraceDistance_) && maxTraceDistance_ >= 0.0001f) ||
        !(maxRoughness_ >= 0.0f && maxRoughness_ <= 1.0f)) {
        log = "Lumen Reflection requires a finite trace distance of at least 0.0001 m and roughness in [0, 1]";
        return makeError(Error::InvalidArgument);
    }

    device_ = context.device;
    auto blueNoise = loadLumenBlueNoise(*device_, *context.graphicsQueue, LumenBlueNoise::Vec2, log);
    if (!blueNoise) { return makeError(blueNoise.error()); }
    blueNoise_ = std::move(*blueNoise);
    auto& registry = ShaderRegistry::instance();
    auto result = registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionTraceMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionTrace"},
        trace_, log);
    if (!result) { return result; }
    return registry.getComputeKernel(*device_,
        {.moduleName = kShaderModule, .entryPointName = "lumenReflectionCompositeMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenReflectionParams>(kLumenReflectionABI), .debugName = "LumenReflectionComposite"},
        composite_, log);
}

Result<> LumenReflectionPass::execute(RenderGraphExecutionContext& context)
{
    if (const auto* world = context.world(); world && (world->environment().enabled ||
        world->worldEnvironment().source == environment::EnvironmentSource::PhysicalAtmosphere)) {
        spdlog::error("The opaque Lumen Reflection GDF branch requires skylight to be disabled");
        return makeError(Error::Unsupported);
    }
    const auto* view = context.viewConstants();
    if (!view) {
        spdlog::error("Lumen Reflection requires the graph camera");
        return makeError(Error::InvalidArgument);
    }
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(context.commandBuffer()));
    const auto sampled = [&](const char* name) { return writer.sampledImageHandle(context.inputTexture(name).view()); };
    const auto storage = [&](const char* name) { return writer.storageImageHandle(context.outputTexture(name).view()); };
    const auto& output = context.outputTexture("color").desc();
    const LumenReflectionParams parameters{
        .scene = bindLumenGlobalTrace(context, writer),
        .position = sampled("position"),
        .normal = sampled("normal"),
        .specular = sampled("specular"),
        .sceneColor = sampled("sceneColor"),
        .blueNoise = writer.sampledImageHandle(blueNoise_.get()),
        .outputReflections = storage("reflections"),
        .trace = storage("trace"),
        .color = storage("color"),
        .reflections = writer.sampledImageHandle(context.outputTexture("reflections").view()),
        .camera = view->current,
        .width = output.width,
        .height = output.height,
        .clipmapCount = *clipmapCount,
        .frameIndex = view->frame[0],
        .maxTraceDistance = maxTraceDistance_,
        .maxRoughness = maxRoughness_,
    };
    const auto encoded = writer.encode(parameters, kLumenReflectionABI);
    if (!encoded) { return makeError(encoded.error()); }

    using Access = RenderGraphResourceAccess;
    const RenderGraphStageUse traceUses[]{
        {"position", Access::TextureSampleRead}, {"normal", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"pageTable", Access::TextureSampleRead},
        {"pageAtlas", Access::TextureSampleRead}, {"coarseMip", Access::TextureSampleRead},
        {"domain", Access::BufferStorageRead}, {"objectGrid", Access::BufferStorageRead},
        {"objectGridScene", Access::BufferStorageRead}, {"cards", Access::BufferStorageRead},
        {"cardScene", Access::BufferStorageRead}, {"cardDepth", Access::TextureSampleRead},
        {"cardLighting", Access::TextureSampleRead}, {"reflections", Access::TextureStorageWrite},
        {"trace", Access::TextureStorageWrite},
    };
    const RenderGraphStageUse compositeUses[]{
        {"sceneColor", Access::TextureSampleRead}, {"reflections", Access::TextureSampleRead},
        {"specular", Access::TextureSampleRead}, {"color", Access::TextureStorageWrite},
    };
    const uint32_t groupsX = (parameters.width + 7u) / 8u;
    const uint32_t groupsY = (parameters.height + 7u) / 8u;
    const RenderGraphStage stages[]{
        {"Trace software reflections", traceUses, [&](CommandBuffer& cmd) {
            return trace_.dispatch(cmd, *encoded, groupsX, groupsY);
        }},
        {"Composite reflections", compositeUses, [&](CommandBuffer& cmd) {
            return composite_.dispatch(cmd, *encoded, groupsX, groupsY);
        }},
    };
    return context.executeStages(stages);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenReflectionPass()
{
    return std::make_unique<LumenReflectionPass>();
}

} // namespace metallic::render::builtin_pass
