#include "rhi_test.h"

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

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 96;
constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";

RhiTestResult createTriangleShaderModule(
    render::Device& device,
    const char* entryPointName,
    std::unique_ptr<render::ShaderModule>& outShaderModule)
{
    render::ShaderCompileResult compileResult;
    render::Result result = render::compileSlangShaderToSpirv(
        render::SlangShaderDesc{
            .moduleName = kTriangleShaderModuleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        compileResult);
    if (result != render::Result::Success) {
        std::string message = std::string("compileSlangShaderToSpirv(") + entryPointName + ") returned " + toString(result);
        if (!compileResult.diagnostics.empty()) {
            message += ": ";
            message += compileResult.diagnostics;
        }
        return RhiTestResult::fail(std::move(message));
    }

    result = device.createShaderModule(
        render::ShaderModuleDesc{
            .code = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
        },
        outShaderModule);
    if (result != render::Result::Success || outShaderModule == nullptr) {
        return RhiTestResult::fail(std::string("createShaderModule returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

class OffscreenTriangleTest : public RhiTest {
public:
    OffscreenTriangleTest()
    {
        type = RhiTestType::Rendering;
        name = "offscreen_triangle_readback";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::ShaderModule> vertexShader;
        RhiTestResult testResult = createTriangleShaderModule(
            context.device,
            kTriangleVertexEntryPoint,
            vertexShader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ShaderModule> fragmentShader;
        testResult = createTriangleShaderModule(
            context.device,
            kTriangleFragmentEntryPoint,
            fragmentShader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::GraphicsPipeline> pipeline;
        render::Result result = context.device.createGraphicsPipeline(
            render::GraphicsPipelineDesc{
                .vertexShader = vertexShader.get(),
                .fragmentShader = fragmentShader.get(),
                .colorFormat = render::Format::Rgba8Unorm,
                .topology = render::PrimitiveTopology::TriangleList,
            },
            pipeline);
        if (result != render::Result::Success || pipeline == nullptr) {
            return RhiTestResult::fail(std::string("createGraphicsPipeline returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> colorTexture;
        result = context.device.createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = render::Format::Rgba8Unorm,
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            },
            colorTexture);
        if (result != render::Result::Success || colorTexture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture returned ") + toString(result));
        }

        std::unique_ptr<render::TextureView> colorTextureView;
        result = context.device.createTextureView(
            *colorTexture,
            render::TextureViewDesc{
                .format = render::Format::Rgba8Unorm,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            colorTextureView);
        if (result != render::Result::Success || colorTextureView == nullptr) {
            return RhiTestResult::fail(std::string("createTextureView returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(kWidth) * static_cast<uint64_t>(kHeight) * 4ull,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (result != render::Result::Success || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue, commandPool);
        if (result != render::Result::Success || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (result != render::Result::Success || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (result != render::Result::Success) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        render::TextureBarrierDesc toColor{
            .texture = colorTexture.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::ColorAttachment,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{.textures = &toColor, .textureCount = 1});

        const render::Rect renderArea{
            .x = 0,
            .y = 0,
            .width = kWidth,
            .height = kHeight,
        };
        render::RenderingAttachmentDesc colorAttachment{
            .view = colorTextureView.get(),
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
        };
        commandBuffer->beginRendering(
            render::RenderingDesc{
                .renderArea = renderArea,
                .colorAttachments = &colorAttachment,
                .colorAttachmentCount = 1,
            });
        commandBuffer->setViewport(
            render::Viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(kWidth),
                .height = static_cast<float>(kHeight),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            });
        commandBuffer->setScissor(renderArea);
        commandBuffer->bindGraphicsPipeline(*pipeline);
        commandBuffer->draw(3);
        commandBuffer->endRendering();

        render::TextureBarrierDesc toTransfer{
            .texture = colorTexture.get(),
            .before = render::ResourceState::ColorAttachment,
            .after = render::ResourceState::TransferSource,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{.textures = &toTransfer, .textureCount = 1});
        commandBuffer->copyTextureToBuffer(
            render::TextureBufferCopyDesc{
                .texture = colorTexture.get(),
                .buffer = readbackBuffer.get(),
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipLevel = 0,
                .baseLayer = 0,
            });

        result = commandBuffer->end();
        if (result != render::Result::Success) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (result != render::Result::Success || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = commandBuffers,
                .commandBufferCount = 1,
                .signalFence = fence.get(),
            });
        if (result != render::Result::Success) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (result != render::Result::Success) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        readbackBuffer->invalidate();
        const void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight) * 4u);
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "offscreen_triangle_readback.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        const auto* bytes = pixels.data();
        uint32_t brightPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = bytes[index * 4 + 0];
            const uint8_t g = bytes[index * 4 + 1];
            const uint8_t b = bytes[index * 4 + 2];
            if (r > 120 || g > 120 || b > 120) {
                ++brightPixelCount;
            }
        }

        if (brightPixelCount < 128) {
            return RhiTestResult::fail(
                std::string("triangle readback found too few bright pixels: ") +
                std::to_string(brightPixelCount));
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

METALLIC_REGISTER_RHI_TEST(OffscreenTriangleTest);

} // namespace
} // namespace metallic::tests
