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
        Result<> result;
        result = context.device->createBindlessHeap(BindlessHeapDesc{
                .maxSampledImages = 1,
            }).transform([&](auto rhiValue) { bindlessHeap_ = std::move(rhiValue); });
        if (!result || bindlessHeap_ == nullptr) {
            log += resultMessage("createBindlessHeap(ImageSamplePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = bindlessHeap_->allocate(BindlessHandleKind::SampledImage).transform([&](auto rhiValue) { imageHandle_ = std::move(rhiValue); });
        if (!result || !imageHandle_.valid()) {
            log += resultMessage("allocateSampledImage(ImageSamplePass)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        result = bindlessHeap_->writeSampledImage(
            imageHandle_,
            *context.preparedScene->imageView,
            TextureLayout::ShaderRead);
        if (!result) {
            log += resultMessage("writeSampledImage(ImageSamplePass)", result);
            log += '\n';
            return result;
        }

        result = createShaderModule(*context.device, kImageSampleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kImageSampleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = ShaderRegistry::instance().getGraphicsPipeline(*context.device, GraphicsPipelineDesc{
            .vertexShader = {vertexShader_.get()},
            .fragmentShader = {fragmentShader_.get()},
            .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1,
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
            bindlessHeap_ == nullptr ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color.view(),
            .layout = TextureLayout::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        if (auto rendering = context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
        }); !rendering) { return rendering; }
        if (auto commandResult = context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        }); !commandResult) { return commandResult; }
        context.commandBuffer().setScissor(renderArea);
        if (auto commandResult = context.commandBuffer().bindBindlessHeap(*bindlessHeap_); !commandResult) { return commandResult; }
        if (auto commandResult = context.commandBuffer().bindExecution((pipeline_)->execution(), &imageHandle_.shaderIndex, sizeof(imageHandle_.shaderIndex)); !commandResult) { return commandResult; }
        if (auto commandResult = context.commandBuffer().draw(3); !commandResult) { return commandResult; }
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
        Result<> result = ShaderRegistry::instance().getShader(SlangShaderDesc{
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
        result = ShaderRegistry::instance().getShaderModule(device, ShaderModuleDesc{
            .spirv = compileResult.spirv,
            .debugName = shaderDebugName.c_str(),
        }).transform([&](auto rhiValue) { outShaderModule = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createShaderModule(ImageSamplePass)", result);
            log += '\n';
        }
        return result;
    }

    std::unique_ptr<BindlessHeap> bindlessHeap_;
    BindlessHandle imageHandle_;
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
