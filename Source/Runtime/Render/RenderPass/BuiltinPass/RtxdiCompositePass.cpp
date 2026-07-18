#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class RtxdiCompositePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("denoisedDiffuse", "RELAX denoised diffuse radiance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("denoisedSpecular", "RELAX denoised specular radiance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        reflection.addTextureInput("emissive", "Emissive and background radiance")
            .storageReadWrite()
            .format = Format::Rgba16Sfloat;
        reflection.addTextureOutput("color", "Composited RELAX-denoised RTXDI color")
            .storageReadWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeFloatSetting("exposure", "Exposure", 1.0f, 0.05f, 8.0f),
        };
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            log = "RtxdiCompositePass requires a device";
            return makeError(Error::InvalidArgument);
        }
        if (program_.valid()) {
            return {};
        }

        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kRtxdiCompositeShaderModuleName,
                .entryPointName = kRtxdiCompositeEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
        if (!result) {
            log = resultMessage("compileSlangShaderToSpirv(RtxdiComposite.rtxdiCompositeMain)", result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }

        const SceneRayQueryBindingDesc bindings[] = {
            {.binding = 0, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 1, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 2, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 3, .kind = SceneRayQueryBindingKind::StorageImage},
            {.binding = 4, .kind = SceneRayQueryBindingKind::StorageImage},
        };
        std::string programLog;
        result = program_.initialize(
            *context.device,
            SceneRayQueryProgramDesc{
                .spirv = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
                .pushConstantSize = sizeof(RtxdiCompositePush),
                .bindings = bindings,
                .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                .debugName = "RtxdiCompositePass",
            },
            programLog);
        if (!programLog.empty()) {
            log += programLog;
        }
        if (!result) {
            program_.clear();
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle denoisedDiffuse = context.inputTexture("denoisedDiffuse");
        TextureHandle denoisedSpecular = context.inputTexture("denoisedSpecular");
        TextureHandle baseColorMetalness = context.inputTexture("baseColorMetalness");
        TextureHandle emissive = context.inputTexture("emissive");
        TextureHandle color = context.outputTexture("color");
        if (!validTexture(denoisedDiffuse) ||
            !validTexture(denoisedSpecular) ||
            !validTexture(baseColorMetalness) ||
            !validTexture(emissive) ||
            !validTexture(color) ||
            !program_.valid()) {
            return makeError(Error::InvalidArgument);
        }

        RtxdiCompositePush push;
        push.width = context.width();
        push.height = context.height();
        push.exposure = floatProperty(context.properties(), "exposure", 1.0f, 0.05f, 8.0f);
        const SceneRayQueryDispatchBinding bindings[] = {
            {.binding = 0, .textureView = denoisedDiffuse.view()},
            {.binding = 1, .textureView = denoisedSpecular.view()},
            {.binding = 2, .textureView = baseColorMetalness.view()},
            {.binding = 3, .textureView = emissive.view()},
            {.binding = 4, .textureView = color.view()},
        };
        return program_.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
            .groupCountZ = 1,
        });
    }

private:
    static bool validTexture(TextureHandle texture)
    {
        return texture.valid() && texture.texture() != nullptr && texture.view() != nullptr;
    }

    static float floatProperty(
        const RenderGraphProperties& properties,
        const char* key,
        float fallback,
        float minimum,
        float maximum)
    {
        if (!properties.is_object()) {
            return fallback;
        }
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number()) {
            return fallback;
        }
        const float value = iter->get<float>();
        return std::isfinite(value) ? std::clamp(value, minimum, maximum) : fallback;
    }

    SceneRayQueryProgram program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createRtxdiCompositePass()
{
    return std::make_unique<RtxdiCompositePass>();
}

} // namespace metallic::render::builtin_pass
