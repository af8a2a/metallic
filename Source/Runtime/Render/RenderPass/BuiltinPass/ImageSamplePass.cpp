#include "Runtime/Render/Core/ImageSampleParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class ImageSamplePass final : public UnsafePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.sampledImage = true};
    }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Fullscreen sampled image")
            .transient(RenderGraphInitialization::Clear)
            .format = Format::RGBA8Unorm;
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "ImageSamplePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        if (!context.preparedScene || !context.preparedScene->imageView) {
            log = "Image resource was not prepared by StreamerSubsystem";
            return makeError(Error::InvalidArgument);
        }
        device_ = context.device;
        Result<> result;
        result = createShaderModule(*context.device, kImageSampleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kImageSampleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(GraphicsPipelineDesc{
            .vertexShader = {vertexShader_.get()},
            .fragmentShader = {fragmentShader_.get()},
            .colorFormat = Format::RGBA8Unorm,
            .topology = PrimitiveTopology::TriangleList,
            .usesBindlessHeap = true,
        }).transform([&](auto rhiValue) { pipeline_ = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createGraphicsPipeline(ImageSamplePass)", result);
            log += '\n';
        }
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() ||
            device_ == nullptr ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const auto* prepared = context.preparedScene();
        if (!prepared || !prepared->imageView) { return makeError(Error::InvalidArgument); }
        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const ImageSampleParams params{.source = writer.sampledImage(prepared->imageView)};
        auto encoded = writer.encode(params, kImageSampleABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        if (auto result = encoded->bindResources(commands); !result) { return result; }
        const auto bytes = encoded->inlineData();

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        if (auto rendering = context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
        }); !rendering) { return rendering; }
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        if (auto result = commands.bindExecution(pipeline_->execution(), bytes.data(), uint32_t(bytes.size())); !result) {
            commands.endRendering();
            return result;
        }
        context.commandBuffer().draw(3);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    static Result<> createShaderModule(
        Device& device,
        const char* entryPointName,
        std::unique_ptr<ShaderModule>& outShaderModule,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
                .moduleName = kImageSampleShaderModuleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
            }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += entryPointName;
            log += ") returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }

        const std::string shaderDebugName =
            std::string(kImageSampleShaderModuleName) + "." + entryPointName;
        result = device.createShaderModule(ShaderModuleDesc{
            .spirv = compileResult.spirv,
            .debugName = shaderDebugName.c_str(),
        }).transform([&](auto rhiValue) { outShaderModule = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createShaderModule(ImageSamplePass)", result);
            log += '\n';
        }
        return result;
    }

    Device* device_ = nullptr;
    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;

};

} // namespace

std::unique_ptr<RenderGraphPass> createImageSamplePass()
{
    return std::make_unique<ImageSamplePass>();
}

} // namespace metallic::render::builtin_pass
