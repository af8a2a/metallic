#include "rhi_test.h"

#include "Runtime/Render/RenderGraph/render_graph.h"
#include "Runtime/Render/slang_compiler.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::tests {
namespace {

constexpr const char* kShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kBindlessSmokeShaderModuleName = "bindless_smoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";

render::Result createSlangShaderModule(
    render::Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<render::ShaderModule>& outShaderModule,
    std::string& log)
{
    render::ShaderCompileResult compileResult;
    render::Result result = render::compileSlangShaderToSpirv(
        render::SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kShaderSearchPath,
        },
        compileResult);
    if (!result) {
        log += std::string("compileSlangShaderToSpirv(") + moduleName + "." + entryPointName + ") returned ";
        log += toString(result);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        log += '\n';
        return result;
    }

    return device.createShaderModule(
        render::ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
}

class TestInputOutputPass final : public render::RenderGraphPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addInput("input", "Required input");
        reflection.addOutput("color", "Output color");
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext&) override
    {
        return {};
    }
};

class TestBindlessSamplePass final : public render::RenderGraphPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBindlessSampledInput("source", "Source bindless sampled texture");
        reflection.addOutput("color", "Bindless sampled output")
            .format = render::Format::Rgba8Unorm;
        return reflection;
    }

    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }

        render::Result result = createSlangShaderModule(
            *context.device,
            kBindlessSmokeShaderModuleName,
            kBindlessSmokeVertexEntryPoint,
            vertexShader_,
            log);
        if (!result) {
            return result;
        }
        result = createSlangShaderModule(
            *context.device,
            kBindlessSmokeShaderModuleName,
            kBindlessSmokeFragmentEntryPoint,
            fragmentShader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            render::GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = render::Format::Rgba8Unorm,
                .topology = render::PrimitiveTopology::TriangleList,
                .usesBindlessHeap = true,
            },
            pipeline_);
        if (!result) {
            log += std::string("createGraphicsPipeline(bindless graph pass) returned ") + toString(result) + '\n';
        }
        return result;
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::BindlessHandle* sourceHandle = context.bindlessInput("source");
        render::RenderGraphResource* color = context.output("color");
        if (sourceHandle == nullptr ||
            sourceHandle->kind != render::BindlessHandleKind::SampledImage ||
            sourceHandle->index != 0 ||
            color == nullptr ||
            color->view == nullptr ||
            pipeline_ == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }

        const render::Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        render::RenderingAttachmentDesc attachment{
            .view = color->view,
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        context.commandBuffer().beginRendering(render::RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(render::Viewport{
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
    std::unique_ptr<render::ShaderModule> vertexShader_;
    std::unique_ptr<render::ShaderModule> fragmentShader_;
    std::unique_ptr<render::GraphicsPipeline> pipeline_;
};

void registerTestPass()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;
    render::registerRenderGraphPassType(
        "TestInputOutputPass",
        "Test-only pass with one required input and one output",
        []() { return std::make_unique<TestInputOutputPass>(); });
    render::registerRenderGraphPassType(
        "TestBindlessSamplePass",
        "Test-only pass that samples a RenderGraph input through bindless",
        []() { return std::make_unique<TestBindlessSamplePass>(); });
}

uint32_t countBrightPixels(const std::vector<uint32_t>& pixels)
{
    uint32_t brightPixelCount = 0;
    for (uint32_t pixel : pixels) {
        const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
        const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
        const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
        if (r > 120 || g > 120 || b > 120) {
            ++brightPixelCount;
        }
    }
    return brightPixelCount;
}

uint32_t countVisiblePixels(const std::vector<uint32_t>& pixels)
{
    uint32_t visiblePixelCount = 0;
    for (uint32_t pixel : pixels) {
        const uint8_t r = static_cast<uint8_t>(pixel & 0xffu);
        const uint8_t g = static_cast<uint8_t>((pixel >> 8u) & 0xffu);
        const uint8_t b = static_cast<uint8_t>((pixel >> 16u) & 0xffu);
        if (r > 8 || g > 8 || b > 8) {
            ++visiblePixelCount;
        }
    }
    return visiblePixelCount;
}

class RenderGraphSerializationTest : public RhiTest {
public:
    RenderGraphSerializationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_json_roundtrip";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        render::RenderGraphNode* node = graph.findNode("Triangle");
        if (node == nullptr) {
            return RhiTestResult::fail("default graph did not create Triangle node");
        }
        graph.setNodePosition(node->id, 123.0f, 456.0f);

        const std::string json = render::serializeRenderGraphToString(graph);
        render::RenderGraph loaded;
        std::string message;
        if (!render::deserializeRenderGraphFromString(json, loaded, message)) {
            return RhiTestResult::fail(message);
        }

        if (loaded.nodes().size() != 1 || loaded.edges().size() != 0 || loaded.outputs().size() != 1) {
            return RhiTestResult::fail("round-trip changed graph topology");
        }
        const render::RenderGraphNode* loadedNode = loaded.findNode("Triangle");
        if (loadedNode == nullptr ||
            loadedNode->type != "TriangleRasterPass" ||
            loadedNode->uiX != 123.0f ||
            loadedNode->uiY != 456.0f) {
            return RhiTestResult::fail("round-trip changed node data");
        }
        if (loaded.firstOutputName() != "Triangle.color") {
            return RhiTestResult::fail("round-trip changed marked output");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphValidationTest : public RhiTest {
public:
    RenderGraphValidationTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_validation";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();

        std::string log;
        render::RenderGraph missingOutput;
        missingOutput.addNode("TriangleRasterPass", "Triangle");
        if (missingOutput.validate(log)) {
            return RhiTestResult::fail("graph without outputs validated successfully");
        }

        render::RenderGraph badEndpoint = render::RenderGraph::createDefaultTriangleGraph();
        badEndpoint.addEdge("Triangle.color", "Triangle.missing");
        if (badEndpoint.validate(log)) {
            return RhiTestResult::fail("graph with invalid edge endpoint validated successfully");
        }

        render::RenderGraph cyclic;
        cyclic.addNode("TestInputOutputPass", "A");
        cyclic.addNode("TestInputOutputPass", "B");
        cyclic.addEdge("A.color", "B.input");
        cyclic.addEdge("B.color", "A.input");
        cyclic.markOutput("A.color");
        if (cyclic.validate(log)) {
            return RhiTestResult::fail("cyclic graph validated successfully");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphPreviewTest : public RhiTest {
public:
    RenderGraphPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_triangle_preview";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        result = preview.render(graph, 128, 96);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RhiTestResult::fail("default triangle graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 64, 64);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 64 || preview.height() != 64) {
            return RhiTestResult::fail("preview resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 64) {
            return RhiTestResult::fail("resized default triangle graph produced too few bright pixels");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphCopyColorWorkflowTest : public RhiTest {
public:
    RenderGraphCopyColorWorkflowTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_copy_color_workflow";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("CopyColorWorkflow");
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("CopyColorPass", "Copy");
        graph.addEdge("Triangle.color", "Copy.source");
        graph.markOutput("Copy.color");

        result = preview.render(graph, 128, 96);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result));
        }
        if (countBrightPixels(preview.pixels()) < 128) {
            return RhiTestResult::fail("copy color graph produced too few bright pixels");
        }

        graph.markDirty();
        result = preview.render(graph, 80, 80);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render resize returned ") + toString(result));
        }
        if (preview.width() != 80 || preview.height() != 80) {
            return RhiTestResult::fail("copy color graph resize did not update output dimensions");
        }
        if (countBrightPixels(preview.pixels()) < 80) {
            return RhiTestResult::fail("resized copy color graph produced too few bright pixels");
        }

        return RhiTestResult::pass();
    }
};

class RenderGraphBindlessTextureWorkflowTest : public RhiTest {
public:
    RenderGraphBindlessTextureWorkflowTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_bindless_texture_workflow";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerTestPass();

        constexpr uint32_t kWidth = 16;
        constexpr uint32_t kHeight = 16;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4ull;

        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RenderGraph Bindless Texture Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("bindless test device has no graphics queue");
        }

        render::RenderGraphProperties sourceProperties = render::RenderGraphProperties::object();
        sourceProperties["color"] = {0.25f, 0.50f, 0.75f, 1.0f};

        render::RenderGraph graph;
        graph.setName("BindlessTextureWorkflow");
        graph.addNode("ClearColorPass", "Source", sourceProperties);
        graph.addNode("TestBindlessSamplePass", "Sample");
        graph.addEdge("Source.color", "Sample.source");
        graph.markOutput("Sample.color");

        render::RenderGraphExecutor executor;
        std::string log;
        result = executor.compile(*device, graph, kWidth, kHeight, log);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(log);
            }
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = device->createCommandPool(*graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(
            render::BufferDesc{
                .size = kReadbackByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        result = executor.execute(*commandBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }

        render::RenderGraphResource* output = executor.outputResource("Sample.color");
        if (output == nullptr || output->texture == nullptr) {
            return RhiTestResult::fail("bindless graph output resource is missing");
        }

        result = executor.transitionOutput(*commandBuffer, "Sample.color", render::ResourceState::TransferSource);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionOutput returned ") + toString(result));
        }
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = output->texture,
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
            .mipLevel = 0,
            .baseLayer = 0,
        });

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(render::QueueSubmitDesc{
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalFence = fence.get(),
        });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        readbackBuffer->invalidate();
        void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kReadbackByteSize));
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        uint32_t matchedPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = pixels[index * 4 + 0];
            const uint8_t g = pixels[index * 4 + 1];
            const uint8_t b = pixels[index * 4 + 2];
            const uint8_t a = pixels[index * 4 + 3];
            if (r >= 48 && r <= 80 && g >= 112 && g <= 144 && b >= 176 && b <= 208 && a >= 240) {
                ++matchedPixelCount;
            }
        }

        if (matchedPixelCount < (kWidth * kHeight) / 2) {
            return RhiTestResult::fail(
                std::string("bindless graph sampled too few source pixels: ") +
                std::to_string(matchedPixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_bindless_texture_workflow.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        (void)device->waitIdle();
        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class RenderGraphImageSamplePassPreviewTest : public RhiTest {
public:
    RenderGraphImageSamplePassPreviewTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_image_sample_pass_preview";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(false);
        if (!result) {
            return RhiTestResult::skip(std::string("RenderGraphPreviewRenderer::initialize returned ") + toString(result));
        }

        render::RenderGraph graph;
        graph.setName("ImageSamplePreview");
        graph.addNode("ImageSamplePass", "Image");
        graph.markOutput("Image.color");

        result = preview.render(graph, 160, 120);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(preview.lastLog());
            }
            return RhiTestResult::fail(std::string("RenderGraphPreviewRenderer::render returned ") + toString(result) + ": " + preview.lastLog());
        }

        const uint32_t visiblePixelCount = countVisiblePixels(preview.pixels());
        if (visiblePixelCount < 160 * 120 / 2) {
            return RhiTestResult::fail(
                std::string("image sample pass produced too few visible pixels: ") +
                std::to_string(visiblePixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "render_graph_image_sample_pass_preview.png";
        const auto* bytes = reinterpret_cast<const uint8_t*>(preview.pixels().data());
        if (!saveRgba8Png(outputPath, bytes, preview.width(), preview.height(), outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphSerializationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphValidationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphPreviewTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphCopyColorWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBindlessTextureWorkflowTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphImageSamplePassPreviewTest);

} // namespace
} // namespace metallic::tests
