#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenGlobalTraceBindings.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <cstddef>

namespace metallic::render::builtin_pass {
namespace {

constexpr uint64_t kLumenSceneVisualizeABI = 0x4c554d454e070001ull;

struct LumenSceneVisualizeParams {
    ShaderLumenGlobalTraceBindings scene;
    ShaderStorageImage sceneRadiance, geometryNormals, surfaceCache;
    uint32_t clipmapCount, padding;
    ViewCameraConstants camera;
};
static_assert(sizeof(LumenSceneVisualizeParams) == 144 && offsetof(LumenSceneVisualizeParams, camera) == 64);

class LumenSceneVisualizePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override;
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override;
    Result<> execute(RenderGraphExecutionContext& context) override;

private:
    Device* device_ = nullptr;
    ComputeKernel program_;
};

RenderPassReflection LumenSceneVisualizePass::reflect(const RenderGraphCompileContext&) const
{
    RenderPassReflection reflection;
    reflectLumenGlobalTraceInputs(reflection);
    auto& scene = reflection.addTextureOutput("scene", "Scene-linear Surface Cache radiance at the software trace hit")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    scene.format = Format::RGBA16Sfloat;
    scene.colorEncoding = DisplayColorEncoding::SceneLinear;
    auto& geometryNormals = reflection.addTextureOutput("geometryNormals", "World-space GDF hit normals remapped to RGB")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    geometryNormals.format = Format::RGBA16Sfloat;
    geometryNormals.colorEncoding = DisplayColorEncoding::sRGB;
    auto& surfaceCache = reflection.addTextureOutput("surfaceCache", "Surface Cache radiance with uncovered hits in magenta")
        .storageWrite().transient(RenderGraphInitialization::FullOverwrite);
    surfaceCache.format = Format::RGBA16Sfloat;
    surfaceCache.colorEncoding = DisplayColorEncoding::SceneLinear;
    return reflection;
}

Result<> LumenSceneVisualizePass::compile(const RenderGraphCompileContext& context, std::string& log)
{
    device_ = context.device;
    return ShaderRegistry::instance().getComputeKernel(*device_,
        {.moduleName = "Features/DistanceField/LumenSceneVisualize", .entryPointName = "lumenSceneVisualizeMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.parameters = parameterAbi<LumenSceneVisualizeParams>(kLumenSceneVisualizeABI), .debugName = "LumenSceneVisualize"},
        program_, log);
}

Result<> LumenSceneVisualizePass::execute(RenderGraphExecutionContext& context)
{
    const auto* view = context.viewConstants();
    if (!view) {
        spdlog::error("Lumen Scene visualization requires a graph camera");
        return makeError(Error::InvalidArgument);
    }
    const auto clipmapCount = lumenGlobalTraceClipmapCount(context);
    if (!clipmapCount) { return makeError(Error::InvalidArgument); }
    auto registry = ResourceRegistry::forDevice(*device_);
    if (!registry) { return makeError(registry.error()); }
    auto& commands = context.commandBuffer();
    ParameterWriter writer(*device_, **registry, RenderFrameContext::from(commands));
    LumenSceneVisualizeParams params{};
    params.scene = bindLumenGlobalTrace(context, writer);
    params.sceneRadiance = writer.storageImageHandle(context.outputTexture("scene").view());
    params.geometryNormals = writer.storageImageHandle(context.outputTexture("geometryNormals").view());
    params.surfaceCache = writer.storageImageHandle(context.outputTexture("surfaceCache").view());
    params.clipmapCount = *clipmapCount;
    params.camera = view->current;
    const auto encoded = writer.encode(params, kLumenSceneVisualizeABI);
    if (!encoded) { return makeError(encoded.error()); }
    return program_.dispatch(commands, *encoded, (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
}

} // namespace

std::unique_ptr<RenderGraphPass> createLumenSceneVisualizePass()
{
    return std::make_unique<LumenSceneVisualizePass>();
}

} // namespace metallic::render::builtin_pass
