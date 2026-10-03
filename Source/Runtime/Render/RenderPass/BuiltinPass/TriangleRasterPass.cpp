#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class TriangleRasterPass final : public RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Rasterized triangle color")
            .transient(RenderGraphInitialization::Clear)
            .format = Format::RGBA8Unorm;
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result<> result = createShaderModule(*context.device, kTriangleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kTriangleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(GraphicsPipelineDesc{
            .vertexShader = {vertexShader_.get()},
            .fragmentShader = {fragmentShader_.get()},
            .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1,
            .topology = PrimitiveTopology::TriangleList,
        }).transform([&](auto rhiValue) { pipeline_ = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createGraphicsPipeline", result);
            log += '\n';
        }
        if (result) { execution_ = pipeline_->execution(); }
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle color = context.outputTexture("color");
        if (!color.valid() || pipeline_ == nullptr) {
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
            .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
        };
        auto rendering = context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = {&attachment, 1},
        });
        if (!rendering) { return rendering; }
        if (auto commandResult = context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        }); !commandResult) { return commandResult; }
        context.commandBuffer().setScissor(renderArea);
        auto bound = context.commandBuffer().bindExecution(execution_);
        if (!bound) { context.commandBuffer().endRendering(); return bound; }
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
        Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
                .moduleName = kTriangleShaderModuleName,
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
            std::string(kTriangleShaderModuleName) + "." + entryPointName;
        result = device.createShaderModule(ShaderModuleDesc{
            .spirv = compileResult.spirv,
            .debugName = shaderDebugName.c_str(),
        }).transform([&](auto rhiValue) { outShaderModule = std::move(rhiValue); });
        if (!result) {
            log += resultMessage("createShaderModule", result);
            log += '\n';
        }
        return result;
    }

    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
    PreparedExecution execution_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createTriangleRasterPass()
{
    return std::make_unique<TriangleRasterPass>();
}

} // namespace metallic::render::builtin_pass
