#include "Runtime/Render/RenderGraph/render_graph.h"

#include "Runtime/Render/slang_compiler.h"

#include <memory>
#include <string>
#include <string_view>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";

std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

class ClearColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addOutput("color", "Cleared color target")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* color = context.output("color");
        if (color == nullptr || color->view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        ColorValue clear{0.04f, 0.06f, 0.09f, 1.0f};
        const RenderGraphProperties& props = context.properties();
        if (props.contains("color") && props["color"].is_array() && props["color"].size() >= 4) {
            clear.r = props["color"][0].get<float>();
            clear.g = props["color"][1].get<float>();
            clear.b = props["color"][2].get<float>();
            clear.a = props["color"][3].get<float>();
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color->view,
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = clear,
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().endRendering();
        return {};
    }
};

class CopyColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        RenderGraphField& source = reflection.addInput("source", "Source color texture");
        source.usage = TextureUsageBits::TransferSource;
        source.state = ResourceState::TransferSource;

        RenderGraphField& color = reflection.addOutput("color", "Copied color texture");
        color.usage = TextureUsageBits::TransferDestination;
        color.state = ResourceState::TransferDestination;
        color.format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* source = context.input("source");
        RenderGraphResource* color = context.output("color");
        if (source == nullptr ||
            source->texture == nullptr ||
            color == nullptr ||
            color->texture == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        context.commandBuffer().copyTexture(TextureCopyDesc{
            .source = source->texture,
            .destination = color->texture,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .sourceMipLevel = 0,
            .sourceBaseLayer = 0,
            .destinationMipLevel = 0,
            .destinationBaseLayer = 0,
        });
        return {};
    }
};

class TriangleRasterPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addOutput("color", "Rasterized triangle color")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createShaderModule(*context.device, kTriangleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kTriangleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createGraphicsPipeline", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* color = context.output("color");
        if (color == nullptr || color->view == nullptr || pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color->view,
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
        context.commandBuffer().draw(3);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    static Result createShaderModule(
        Device& device,
        const char* entryPointName,
        std::unique_ptr<ShaderModule>& outShaderModule,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kTriangleShaderModuleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
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

        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            },
            outShaderModule);
        if (!result) {
            log += resultMessage("createShaderModule", result);
            log += '\n';
        }
        return result;
    }

    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
};

} // namespace

void registerBuiltInRenderGraphPasses()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    registerRenderGraphPassType(
        "ClearColorPass",
        "Clear a color texture",
        []() { return std::make_unique<ClearColorPass>(); });
    registerRenderGraphPassType(
        "CopyColorPass",
        "Copy a color texture",
        []() { return std::make_unique<CopyColorPass>(); });
    registerRenderGraphPassType(
        "TriangleRasterPass",
        "Rasterize the built-in triangle shader",
        []() { return std::make_unique<TriangleRasterPass>(); });
}

} // namespace metallic::render
