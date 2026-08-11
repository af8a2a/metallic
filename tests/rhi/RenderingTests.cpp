#include "RhiTest.h"

#include "Runtime/Render/SlangCompiler.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
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
constexpr const char* kTriangleShaderModuleName = "Triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
constexpr const char* kReversedDepthNearVertexEntryPoint = "reversedDepthNearVertexMain";
constexpr const char* kReversedDepthFarVertexEntryPoint = "reversedDepthFarVertexMain";
constexpr const char* kSolidGreenFragmentEntryPoint = "solidGreenFragmentMain";
constexpr const char* kSolidRedFragmentEntryPoint = "solidRedFragmentMain";
constexpr const char* kMaterialShaderModuleName = "MaterialShaderObject";
constexpr const char* kMaterialVertexEntryPoint = "materialShaderObjectVertexMain";
constexpr const char* kMaterialFragmentEntryPoint = "materialShaderObjectFragmentMain";
constexpr const char* kMaterialAlternateFragmentEntryPoint = "materialShaderObjectAlternateFragmentMain";

struct MaterialGpuParams {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
};

struct MaterialUserPush {
    uint32_t positionBuffer = 0;
    uint32_t materialIndexBuffer = 0;
    uint32_t materialBuffer = 0;
    uint32_t paramsBuffer = 0;
    uint32_t vertexOffset = 0;
    uint32_t materialVariant = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

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
    if (!result) {
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
    if (!result || outShaderModule == nullptr) {
        return RhiTestResult::fail(std::string("createShaderModule returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult compileMaterialShader(
    const char* entryPointName,
    render::ShaderCompileResult& outCompileResult)
{
    render::Result result = render::compileSlangShaderToSpirv(
        render::SlangShaderDesc{
            .moduleName = kMaterialShaderModuleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        },
        outCompileResult);
    if (!result) {
        std::string message = std::string("compileSlangShaderToSpirv(") + entryPointName + ") returned ";
        message += toString(result);
        if (!outCompileResult.diagnostics.empty()) {
            message += ": ";
            message += outCompileResult.diagnostics;
        }
        return RhiTestResult::fail(std::move(message));
    }
    return RhiTestResult::pass();
}

RhiTestResult createUploadStorageBuffer(
    render::Device& device,
    const void* data,
    uint64_t byteSize,
    const char* label,
    std::unique_ptr<render::Buffer>& outBuffer)
{
    render::Result result = device.createBuffer(
        render::BufferDesc{
            .size = byteSize,
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostUpload,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        return RhiTestResult::fail(std::string("createBuffer(") + label + ") returned " + toString(result));
    }

    void* mapped = outBuffer->map();
    if (mapped == nullptr) {
        return RhiTestResult::fail(std::string("map(") + label + ") returned null");
    }
    std::memcpy(mapped, data, static_cast<size_t>(byteSize));
    outBuffer->flush(0, byteSize);
    outBuffer->unmap();
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
        if (!result || pipeline == nullptr) {
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
        if (!result || colorTexture == nullptr) {
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
        if (!result || colorTextureView == nullptr) {
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
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
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
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
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

class ReversedZDepthRenderingTest : public RhiTest {
public:
    ReversedZDepthRenderingTest()
    {
        type = RhiTestType::Rendering;
        name = "reversed_z_depth_readback";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::ShaderModule> nearVertexShader;
        RhiTestResult testResult = createTriangleShaderModule(
            context.device,
            kReversedDepthNearVertexEntryPoint,
            nearVertexShader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ShaderModule> farVertexShader;
        testResult = createTriangleShaderModule(
            context.device,
            kReversedDepthFarVertexEntryPoint,
            farVertexShader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ShaderModule> greenFragmentShader;
        testResult = createTriangleShaderModule(
            context.device,
            kSolidGreenFragmentEntryPoint,
            greenFragmentShader);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::ShaderModule> redFragmentShader;
        testResult = createTriangleShaderModule(
            context.device,
            kSolidRedFragmentEntryPoint,
            redFragmentShader);
        if (!testResult.passed) {
            return testResult;
        }

        auto createDepthPipeline =
            [&](render::ShaderModule& vertexShader,
                render::ShaderModule& fragmentShader,
                std::unique_ptr<render::GraphicsPipeline>& outPipeline) -> RhiTestResult {
            render::Result result = context.device.createGraphicsPipeline(
                render::GraphicsPipelineDesc{
                    .vertexShader = &vertexShader,
                    .fragmentShader = &fragmentShader,
                    .colorFormat = render::Format::Rgba8Unorm,
                    .depthStencilFormat = render::Format::D32Sfloat,
                    .topology = render::PrimitiveTopology::TriangleList,
                    .depthStencil = render::DepthStencilState{
                        .depthTestEnable = true,
                        .depthWriteEnable = true,
                        .depthCompareOp = render::CompareOp::GreaterEqual,
                    },
                },
                outPipeline);
            if (!result || outPipeline == nullptr) {
                return RhiTestResult::fail(std::string("createGraphicsPipeline(depth) returned ") + toString(result));
            }
            return RhiTestResult::pass();
        };

        std::unique_ptr<render::GraphicsPipeline> nearGreenPipeline;
        testResult = createDepthPipeline(*nearVertexShader, *greenFragmentShader, nearGreenPipeline);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::GraphicsPipeline> farRedPipeline;
        testResult = createDepthPipeline(*farVertexShader, *redFragmentShader, farRedPipeline);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::Texture> colorTexture;
        render::Result result = context.device.createTexture(
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
        if (!result || colorTexture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture(color) returned ") + toString(result));
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
        if (!result || colorTextureView == nullptr) {
            return RhiTestResult::fail(std::string("createTextureView(color) returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> depthTexture;
        result = context.device.createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::DepthStencilAttachment,
                .format = render::Format::D32Sfloat,
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            },
            depthTexture);
        if (!result || depthTexture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture(depth) returned ") + toString(result));
        }

        std::unique_ptr<render::TextureView> depthTextureView;
        result = context.device.createTextureView(
            *depthTexture,
            render::TextureViewDesc{
                .format = render::Format::D32Sfloat,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            depthTextureView);
        if (!result || depthTextureView == nullptr) {
            return RhiTestResult::fail(std::string("createTextureView(depth) returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(kWidth) * static_cast<uint64_t>(kHeight) * 4ull,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        render::TextureBarrierDesc renderBarriers[] = {
            render::TextureBarrierDesc{
                .texture = colorTexture.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::ColorAttachment,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            render::TextureBarrierDesc{
                .texture = depthTexture.get(),
                .before = render::ResourceState::Undefined,
                .after = render::ResourceState::DepthStencilAttachment,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
        };
        commandBuffer->barrier(render::BarrierDesc{
            .textures = renderBarriers,
            .textureCount = 2,
        });

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
            .clearColor = render::ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
        };
        render::RenderingAttachmentDesc depthAttachment{
            .view = depthTextureView.get(),
            .state = render::ResourceState::DepthStencilAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearDepth = 0.0f,
        };
        commandBuffer->beginRendering(
            render::RenderingDesc{
                .renderArea = renderArea,
                .colorAttachments = &colorAttachment,
                .colorAttachmentCount = 1,
                .depthStencilAttachment = &depthAttachment,
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
        commandBuffer->bindGraphicsPipeline(*nearGreenPipeline);
        commandBuffer->draw(3);
        commandBuffer->bindGraphicsPipeline(*farRedPipeline);
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
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
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
        const void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight) * 4u);
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        const size_t centerIndex =
            (static_cast<size_t>(kHeight / 2) * static_cast<size_t>(kWidth) + static_cast<size_t>(kWidth / 2)) * 4u;
        const uint8_t r = pixels[centerIndex + 0];
        const uint8_t g = pixels[centerIndex + 1];
        const uint8_t b = pixels[centerIndex + 2];
        if (g < 180 || r > 80 || b > 80) {
            return RhiTestResult::fail(
                "reversed-z center pixel was not preserved by depth test: rgba=(" +
                std::to_string(r) +
                ", " +
                std::to_string(g) +
                ", " +
                std::to_string(b) +
                ")");
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "reversed_z_depth_readback.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class ShaderObjectMaterialRenderingTest : public RhiTest {
public:
    ShaderObjectMaterialRenderingTest()
    {
        type = RhiTestType::Rendering;
        name = "shader_object_material_readback";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Shader Object Material Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().shaderObject || !device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("shaderObject or bindlessDescriptorHeap capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::skip("shader object test device has no graphics queue");
        }

        render::ShaderCompileResult vertexCompile;
        RhiTestResult testResult = compileMaterialShader(kMaterialVertexEntryPoint, vertexCompile);
        if (!testResult.passed) {
            return testResult;
        }
        render::ShaderCompileResult fragmentCompile;
        testResult = compileMaterialShader(kMaterialFragmentEntryPoint, fragmentCompile);
        if (!testResult.passed) {
            return testResult;
        }
        render::ShaderCompileResult alternateFragmentCompile;
        testResult = compileMaterialShader(kMaterialAlternateFragmentEntryPoint, alternateFragmentCompile);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::GraphicsShaderObjectProgram> defaultProgram;
        result = device->createGraphicsShaderObjectProgram(
            render::GraphicsShaderObjectProgramDesc{
                .vertexCode = vertexCompile.spirv.data(),
                .vertexByteSize = static_cast<uint64_t>(vertexCompile.spirv.size() * sizeof(uint32_t)),
                .fragmentCode = fragmentCompile.spirv.data(),
                .fragmentByteSize = static_cast<uint64_t>(fragmentCompile.spirv.size() * sizeof(uint32_t)),
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MaterialUserPush),
            },
            defaultProgram);
        if (!result || defaultProgram == nullptr) {
            return RhiTestResult::fail(std::string("createGraphicsShaderObjectProgram(default) returned ") + toString(result));
        }

        std::unique_ptr<render::GraphicsShaderObjectProgram> alternateProgram;
        result = device->createGraphicsShaderObjectProgram(
            render::GraphicsShaderObjectProgramDesc{
                .vertexCode = vertexCompile.spirv.data(),
                .vertexByteSize = static_cast<uint64_t>(vertexCompile.spirv.size() * sizeof(uint32_t)),
                .fragmentCode = alternateFragmentCompile.spirv.data(),
                .fragmentByteSize = static_cast<uint64_t>(alternateFragmentCompile.spirv.size() * sizeof(uint32_t)),
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(MaterialUserPush),
            },
            alternateProgram);
        if (!result || alternateProgram == nullptr) {
            return RhiTestResult::fail(std::string("createGraphicsShaderObjectProgram(alternate) returned ") + toString(result));
        }

        constexpr std::array<float, 24> kPositions{
            -0.9f, -0.7f, 0.0f, 1.0f,
            -0.15f, -0.7f, 0.0f, 1.0f,
            -0.52f, 0.7f, 0.0f, 1.0f,
            0.15f, -0.7f, 0.0f, 1.0f,
            0.9f, -0.7f, 0.0f, 1.0f,
            0.52f, 0.7f, 0.0f, 1.0f,
        };
        constexpr std::array<uint32_t, 6> kMaterialIndices{0, 0, 0, 1, 1, 1};
        constexpr std::array<float, 8> kMaterials{
            1.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
        };
        const MaterialGpuParams params{
            .eye = {0.0f, 0.0f, 2.0f, 0.0f},
            .center = {0.0f, 0.0f, 0.0f, 0.0f},
            .upProjection = {0.0f, 1.0f, 0.0f, 0.0f},
            .viewport = {
                static_cast<float>(kWidth) / static_cast<float>(kHeight),
                static_cast<float>(kWidth),
                static_cast<float>(kHeight),
                1.0471975512f,
            },
            .clipOrtho = {0.1f, 10.0f, 2.0f, 0.0f},
        };

        std::unique_ptr<render::Buffer> positionBuffer;
        testResult = createUploadStorageBuffer(
            *device,
            kPositions.data(),
            static_cast<uint64_t>(kPositions.size() * sizeof(float)),
            "positions",
            positionBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> materialIndexBuffer;
        testResult = createUploadStorageBuffer(
            *device,
            kMaterialIndices.data(),
            static_cast<uint64_t>(kMaterialIndices.size() * sizeof(uint32_t)),
            "material indices",
            materialIndexBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> materialBuffer;
        testResult = createUploadStorageBuffer(
            *device,
            kMaterials.data(),
            static_cast<uint64_t>(kMaterials.size() * sizeof(float)),
            "materials",
            materialBuffer);
        if (!testResult.passed) {
            return testResult;
        }
        std::unique_ptr<render::Buffer> paramsBuffer;
        testResult = createUploadStorageBuffer(
            *device,
            &params,
            sizeof(params),
            "params",
            paramsBuffer);
        if (!testResult.passed) {
            return testResult;
        }

        std::unique_ptr<render::BindlessHeap> bindlessHeap;
        result = device->createBindlessHeap(render::BindlessHeapDesc{.maxBuffers = 4}, bindlessHeap);
        if (!result || bindlessHeap == nullptr) {
            return RhiTestResult::fail(std::string("createBindlessHeap returned ") + toString(result));
        }

        render::BindlessHandle positionHandle;
        render::BindlessHandle materialIndexHandle;
        render::BindlessHandle materialHandle;
        render::BindlessHandle paramsHandle;
        result = bindlessHeap->allocateBuffer(positionHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(position) returned ") + toString(result));
        }
        result = bindlessHeap->allocateBuffer(materialIndexHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(materialIndex) returned ") + toString(result));
        }
        result = bindlessHeap->allocateBuffer(materialHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(material) returned ") + toString(result));
        }
        result = bindlessHeap->allocateBuffer(paramsHandle);
        if (!result) {
            return RhiTestResult::fail(std::string("allocateBuffer(params) returned ") + toString(result));
        }

        result = bindlessHeap->writeStorageBuffer(positionHandle, *positionBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(position) returned ") + toString(result));
        }
        result = bindlessHeap->writeStorageBuffer(materialIndexHandle, *materialIndexBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(materialIndex) returned ") + toString(result));
        }
        result = bindlessHeap->writeStorageBuffer(materialHandle, *materialBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(material) returned ") + toString(result));
        }
        result = bindlessHeap->writeStorageBuffer(paramsHandle, *paramsBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("writeStorageBuffer(params) returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> colorTexture;
        result = device->createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = render::Format::Rgba8Unorm,
                .width = kWidth,
                .height = kHeight,
                .memoryLocation = render::MemoryLocation::Device,
            },
            colorTexture);
        if (!result || colorTexture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture returned ") + toString(result));
        }
        std::unique_ptr<render::TextureView> colorTextureView;
        result = device->createTextureView(
            *colorTexture,
            render::TextureViewDesc{
                .format = render::Format::Rgba8Unorm,
                .mipCount = 1,
                .layerCount = 1,
            },
            colorTextureView);
        if (!result || colorTextureView == nullptr) {
            return RhiTestResult::fail(std::string("createTextureView returned ") + toString(result));
        }
        std::unique_ptr<render::Buffer> readbackBuffer;
        result = device->createBuffer(
            render::BufferDesc{
                .size = static_cast<uint64_t>(kWidth) * static_cast<uint64_t>(kHeight) * 4ull,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
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

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        render::TextureBarrierDesc toColor{
            .texture = colorTexture.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::ColorAttachment,
            .mipCount = 1,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{.textures = &toColor, .textureCount = 1});

        const render::Rect renderArea{.x = 0, .y = 0, .width = kWidth, .height = kHeight};
        render::RenderingAttachmentDesc colorAttachment{
            .view = colorTextureView.get(),
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.02f, 0.02f, 0.02f, 1.0f},
        };
        commandBuffer->beginRendering(
            render::RenderingDesc{
                .renderArea = renderArea,
                .colorAttachments = &colorAttachment,
                .colorAttachmentCount = 1,
            });
        commandBuffer->bindBindlessHeap(*bindlessHeap);
        commandBuffer->bindGraphicsShaderObjectProgram(*defaultProgram);
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
        commandBuffer->setGraphicsShaderObjectState();

        MaterialUserPush push{
            .positionBuffer = positionHandle.index,
            .materialIndexBuffer = materialIndexHandle.index,
            .materialBuffer = materialHandle.index,
            .paramsBuffer = paramsHandle.index,
            .vertexOffset = 0,
        };
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->draw(3);

        commandBuffer->bindGraphicsShaderObjectProgram(*alternateProgram);
        push.vertexOffset = 3;
        push.materialVariant = 1;
        commandBuffer->pushBindlessData(&push, sizeof(push));
        commandBuffer->draw(3);
        commandBuffer->endRendering();

        render::TextureBarrierDesc toTransfer{
            .texture = colorTexture.get(),
            .before = render::ResourceState::ColorAttachment,
            .after = render::ResourceState::TransferSource,
            .mipCount = 1,
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
            });

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }
        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = graphicsQueue->submit(
            render::QueueSubmitDesc{
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
        const void* mapped = readbackBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("readback buffer did not map");
        }

        std::vector<uint8_t> pixels(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight) * 4u);
        std::memcpy(pixels.data(), mapped, pixels.size());
        readbackBuffer->unmap();

        uint32_t redPixelCount = 0;
        uint32_t cyanPixelCount = 0;
        for (uint32_t index = 0; index < kWidth * kHeight; ++index) {
            const uint8_t r = pixels[index * 4 + 0];
            const uint8_t g = pixels[index * 4 + 1];
            const uint8_t b = pixels[index * 4 + 2];
            if (r > 160 && g < 80 && b < 80) {
                ++redPixelCount;
            }
            if (r < 80 && g > 80 && b > 100) {
                ++cyanPixelCount;
            }
        }

        if (redPixelCount < 64 || cyanPixelCount < 64) {
            return RhiTestResult::fail(
                "shader object material readback did not find both material colors: red=" +
                std::to_string(redPixelCount) +
                " cyan=" +
                std::to_string(cyanPixelCount));
        }

        std::string outputMessage;
        const std::filesystem::path outputPath = context.outputDirectory / "shader_object_material_readback.png";
        if (!saveRgba8Png(outputPath, pixels.data(), kWidth, kHeight, outputMessage)) {
            return RhiTestResult::fail(outputMessage);
        }

        return RhiTestResult::pass(std::string("wrote ") + outputPath.string());
    }
};

class PipelineCachePersistenceTest : public RhiTest {
public:
    PipelineCachePersistenceTest()
    {
        type = RhiTestType::Rendering;
        name = "pipeline_cache_persistence_and_shader_invalidation";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        const std::filesystem::path cachePath =
            context.outputDirectory / "pipeline_cache_persistence.pso";
        std::error_code fileError;
        std::filesystem::remove(cachePath, fileError);
        if (fileError) {
            return RhiTestResult::fail(
                "failed to remove previous .pso test file: " + fileError.message());
        }
        const std::string cachePathString = cachePath.string();

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
        std::unique_ptr<render::ShaderModule> changedFragmentShader;
        testResult = createTriangleShaderModule(
            context.device,
            kSolidGreenFragmentEntryPoint,
            changedFragmentShader);
        if (!testResult.passed) {
            return testResult;
        }
        if (fragmentShader->contentHash() == 0 ||
            changedFragmentShader->contentHash() == 0 ||
            fragmentShader->contentHash() == changedFragmentShader->contentHash()) {
            return RhiTestResult::fail("shader content hashes did not distinguish changed SPIR-V");
        }

        const auto createPipeline = [&](
                                        render::PipelineCache& cache,
                                        render::ShaderModule& fragment,
                                        std::unique_ptr<render::GraphicsPipeline>& outPipeline,
                                        render::RasterizationState rasterization = {}) {
            return context.device.createGraphicsPipeline(
                render::GraphicsPipelineDesc{
                    .vertexShader = vertexShader.get(),
                    .fragmentShader = &fragment,
                    .colorFormat = render::Format::Rgba8Unorm,
                    .topology = render::PrimitiveTopology::TriangleList,
                    .rasterization = rasterization,
                    .pipelineCache = &cache,
                },
                outPipeline);
        };

        std::unique_ptr<render::PipelineCache> firstCache;
        render::Result result = context.device.createPipelineCache(
            render::PipelineCacheDesc{
                .filePath = cachePathString.c_str(),
                .saveOnDestroy = false,
            },
            firstCache);
        if (!result || firstCache == nullptr) {
            return RhiTestResult::fail(
                std::string("createPipelineCache(first) returned ") + toString(result));
        }
        if (firstCache->stats().loadStatus != render::PipelineCacheLoadStatus::NotFound) {
            return RhiTestResult::fail("first pipeline cache did not report a cold load");
        }

        std::unique_ptr<render::GraphicsPipeline> firstPipeline;
        result = createPipeline(*firstCache, *fragmentShader, firstPipeline);
        if (!result || firstPipeline == nullptr || firstPipeline->psoHash() == 0) {
            return RhiTestResult::fail(
                std::string("createGraphicsPipeline(first) returned ") + toString(result));
        }
        if (firstPipeline->pipelineCacheHit()) {
            return RhiTestResult::fail("cold pipeline creation unexpectedly reported a cache hit");
        }
        const uint64_t originalPsoHash = firstPipeline->psoHash();
        result = firstCache->save();
        if (!result || !std::filesystem::exists(cachePath) ||
            std::filesystem::file_size(cachePath) <= 80u) {
            return RhiTestResult::fail("pipeline cache did not save a non-empty .pso file");
        }
        firstPipeline.reset();
        firstCache.reset();

        std::unique_ptr<render::PipelineCache> loadedCache;
        result = context.device.createPipelineCache(
            render::PipelineCacheDesc{
                .filePath = cachePathString.c_str(),
                .saveOnDestroy = false,
            },
            loadedCache);
        if (!result || loadedCache == nullptr) {
            return RhiTestResult::fail(
                std::string("createPipelineCache(loaded) returned ") + toString(result));
        }
        const render::PipelineCacheStats loadedStats = loadedCache->stats();
        if (loadedStats.loadStatus != render::PipelineCacheLoadStatus::Loaded ||
            loadedStats.storedPsoCount != 1) {
            return RhiTestResult::fail("saved .pso file did not reload its PSO hash table");
        }

        std::unique_ptr<render::GraphicsPipeline> cachedPipeline;
        result = createPipeline(*loadedCache, *fragmentShader, cachedPipeline);
        if (!result || cachedPipeline == nullptr ||
            !cachedPipeline->pipelineCacheHit() ||
            cachedPipeline->psoHash() != originalPsoHash) {
            return RhiTestResult::fail("unchanged shader did not reuse the saved PSO cache entry");
        }

        std::unique_ptr<render::GraphicsPipeline> explicitDefaultPipeline;
        result = createPipeline(
            *loadedCache,
            *fragmentShader,
            explicitDefaultPipeline,
            render::RasterizationState{
                .cullMode = render::CullMode::None,
                .frontFace = render::FrontFace::CounterClockwise,
            });
        if (!result || explicitDefaultPipeline == nullptr ||
            !explicitDefaultPipeline->pipelineCacheHit() ||
            explicitDefaultPipeline->psoHash() != originalPsoHash) {
            return RhiTestResult::fail("explicit default rasterization state changed the PSO hash");
        }

        std::unique_ptr<render::GraphicsPipeline> frontCullPipeline;
        result = createPipeline(
            *loadedCache,
            *fragmentShader,
            frontCullPipeline,
            render::RasterizationState{
                .cullMode = render::CullMode::Front,
                .frontFace = render::FrontFace::CounterClockwise,
            });
        if (!result || frontCullPipeline == nullptr || frontCullPipeline->pipelineCacheHit() ||
            frontCullPipeline->psoHash() == originalPsoHash) {
            return RhiTestResult::fail("front-face culling did not invalidate the PSO hash");
        }

        std::unique_ptr<render::GraphicsPipeline> backCullPipeline;
        result = createPipeline(
            *loadedCache,
            *fragmentShader,
            backCullPipeline,
            render::RasterizationState{
                .cullMode = render::CullMode::Back,
                .frontFace = render::FrontFace::CounterClockwise,
            });
        if (!result || backCullPipeline == nullptr || backCullPipeline->pipelineCacheHit() ||
            backCullPipeline->psoHash() == originalPsoHash ||
            backCullPipeline->psoHash() == frontCullPipeline->psoHash()) {
            return RhiTestResult::fail("back-face culling did not produce a distinct PSO hash");
        }

        std::unique_ptr<render::GraphicsPipeline> clockwisePipeline;
        result = createPipeline(
            *loadedCache,
            *fragmentShader,
            clockwisePipeline,
            render::RasterizationState{
                .cullMode = render::CullMode::Back,
                .frontFace = render::FrontFace::Clockwise,
            });
        if (!result || clockwisePipeline == nullptr || clockwisePipeline->pipelineCacheHit() ||
            clockwisePipeline->psoHash() == backCullPipeline->psoHash()) {
            return RhiTestResult::fail("front-face winding did not invalidate the PSO hash");
        }

        std::unique_ptr<render::GraphicsPipeline> changedPipeline;
        result = createPipeline(*loadedCache, *changedFragmentShader, changedPipeline);
        if (!result || changedPipeline == nullptr) {
            return RhiTestResult::fail(
                std::string("createGraphicsPipeline(changed shader) returned ") + toString(result));
        }
        if (changedPipeline->pipelineCacheHit() ||
            changedPipeline->psoHash() == originalPsoHash) {
            return RhiTestResult::fail("changed shader did not invalidate the PSO hash");
        }

        result = loadedCache->save();
        const render::PipelineCacheStats finalStats = loadedCache->stats();
        if (!result || finalStats.hitCount != 2 || finalStats.missCount != 4 ||
            finalStats.storedPsoCount != 5) {
            return RhiTestResult::fail("pipeline cache hit/miss statistics are inconsistent");
        }

        changedPipeline.reset();
        clockwisePipeline.reset();
        backCullPipeline.reset();
        frontCullPipeline.reset();
        explicitDefaultPipeline.reset();
        cachedPipeline.reset();
        loadedCache.reset();
        {
            std::ofstream corruptStream(cachePath, std::ios::binary | std::ios::trunc);
            corruptStream.put('\0');
            if (!corruptStream) {
                return RhiTestResult::fail("failed to create a corrupt .pso test file");
            }
        }

        std::unique_ptr<render::PipelineCache> recoveredCache;
        result = context.device.createPipelineCache(
            render::PipelineCacheDesc{
                .filePath = cachePathString.c_str(),
                .saveOnDestroy = false,
            },
            recoveredCache);
        if (!result || recoveredCache == nullptr ||
            recoveredCache->stats().loadStatus != render::PipelineCacheLoadStatus::Invalid) {
            return RhiTestResult::fail("corrupt .pso file did not fall back to an empty cache");
        }
        std::unique_ptr<render::GraphicsPipeline> recoveredPipeline;
        result = createPipeline(*recoveredCache, *fragmentShader, recoveredPipeline);
        if (!result || recoveredPipeline == nullptr || recoveredPipeline->pipelineCacheHit()) {
            return RhiTestResult::fail("recovered cache did not rebuild the PSO after corruption");
        }
        result = recoveredCache->save();
        if (!result || std::filesystem::file_size(cachePath) <= 80u) {
            return RhiTestResult::fail("recovered cache did not replace the corrupt .pso file");
        }
        recoveredPipeline.reset();
        recoveredCache.reset();

        std::unique_ptr<render::PipelineCache> repairedCache;
        result = context.device.createPipelineCache(
            render::PipelineCacheDesc{
                .filePath = cachePathString.c_str(),
                .saveOnDestroy = false,
            },
            repairedCache);
        if (!result || repairedCache == nullptr ||
            repairedCache->stats().loadStatus != render::PipelineCacheLoadStatus::Loaded) {
            return RhiTestResult::fail("repaired .pso file could not be loaded");
        }
        std::unique_ptr<render::GraphicsPipeline> repairedPipeline;
        result = createPipeline(*repairedCache, *fragmentShader, repairedPipeline);
        if (!result || repairedPipeline == nullptr || !repairedPipeline->pipelineCacheHit()) {
            return RhiTestResult::fail("repaired .pso file did not contain the rebuilt PSO hash");
        }

        const std::string invalidPath =
            (context.outputDirectory / "pipeline_cache_invalid.bin").string();
        std::unique_ptr<render::PipelineCache> invalidCache;
        result = context.device.createPipelineCache(
            render::PipelineCacheDesc{.filePath = invalidPath.c_str()},
            invalidCache);
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidCache != nullptr) {
            return RhiTestResult::fail("pipeline cache accepted a file without the .pso extension");
        }

        return RhiTestResult::pass(
            "validated raster state hashing, cold miss, warm hit, shader invalidation, corruption recovery, and .pso persistence");
    }
};

METALLIC_REGISTER_RHI_TEST(OffscreenTriangleTest);
METALLIC_REGISTER_RHI_TEST(ReversedZDepthRenderingTest);
METALLIC_REGISTER_RHI_TEST(ShaderObjectMaterialRenderingTest);
METALLIC_REGISTER_RHI_TEST(PipelineCachePersistenceTest);

} // namespace
} // namespace metallic::tests
