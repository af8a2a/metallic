#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RTXDIPostProcessParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class RTXDICompositePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("denoisedDiffuse", "RELAX denoised diffuse radiance")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        reflection.addTextureInput("denoisedSpecular", "RELAX denoised specular radiance")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        reflection.addTextureInput("baseColorMetalness", "Base color and metalness")
            .storageRead()
            .format = Format::RGBA8Unorm;
        reflection.addTextureInput("emissive", "Emissive and background radiance")
            .storageRead()
            .format = Format::RGBA16Sfloat;
        auto& color = reflection.addTextureOutput("color", "Composited RELAX-denoised RTXDI color")
            .storageWrite()
            // The full-resolution dispatch writes every in-bounds output texel.
            .transient(RenderGraphInitialization::FullOverwrite);
        color.format = Format::RGBA32Sfloat;
        color.colorEncoding = DisplayColorEncoding::SceneLinear;
        return reflection;
    }

    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override
    {
        return {
            runtimeFloatSetting("exposure", "Exposure", 1.0f, 0.05f, 8.0f),
        };
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            log = "RTXDICompositePass requires a device";
            return makeError(Error::InvalidArgument);
        }
        if (program_.valid()) {
            return {};
        }

        device_ = context.device;
        ShaderCompileResult compileResult;
        Result<> result = ShaderRegistry::instance().getShader(SlangShaderDesc{
                .moduleName = kRTXDICompositeShaderModuleName,
                .entryPointName = kRTXDICompositeEntryPoint,
                .searchPath = kTriangleShaderSearchPath,
            }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            log = resultMessage("compileSlangShaderToSpirv(RTXDIComposite.rtxdiCompositeMain)", result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            return result;
        }

        std::string programLog;
        result = program_.initialize(
            *context.device,
            ComputeKernelDesc{
                .spirv = compileResult.spirv,
                .parameters = parameterAbi<RTXDICompositeParams>(kRTXDICompositeABI, ParameterTransport::InlinePush),
                .debugName = "RTXDICompositePass",
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

    Result<> execute(RenderGraphExecutionContext& context) override
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

        RTXDICompositePush push{};
        push.width = context.width();
        push.height = context.height();
        push.exposure = floatProperty(context.properties(), "exposure", 1.0f, 0.05f, 8.0f);
        push.outputLinear = 1u;
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(context.commandBuffer()));
        RTXDICompositeParams params{
            .denoisedDiffuse = writer.storageImage(denoisedDiffuse.view()),
            .denoisedSpecular = writer.storageImage(denoisedSpecular.view()),
            .baseColorMetalness = writer.storageImage(baseColorMetalness.view()),
            .emissive = writer.storageImage(emissive.view()),
            .output = writer.storageImage(color.view()),
            .settings = push,
        };
        auto encoded = writer.encode(params, kRTXDICompositeABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return program_.dispatch(context.commandBuffer(), *encoded,
            (context.width() + 7u) / 8u, (context.height() + 7u) / 8u);
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

    Device* device_ = nullptr;
    ComputeKernel program_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createRtxdiCompositePass()
{
    return std::make_unique<RTXDICompositePass>();
}

} // namespace metallic::render::builtin_pass
