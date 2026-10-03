#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/Core/RHISmokeTests.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <utility>

namespace metallic::render {

namespace {

int resultToExitCode(Result<> result)
{
    return result ? 0 : 1;
}

bool checkResult(Result<> result, const char* label)
{
    if (result) {
        return true;
    }

    spdlog::error("{} failed with Result {}", label, resultToString(result));
    return false;
}

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "Features/Samples/Triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";
constexpr const char* kBindlessSmokeShaderModuleName = "Features/SmokeTests/BindlessSmoke";
constexpr const char* kBindlessSmokeVertexEntryPoint = "bindlessSmokeVertexMain";
constexpr const char* kBindlessSmokeFragmentEntryPoint = "bindlessSmokeFragmentMain";

Result<> createSlangShaderModule(
    Device& device,
    const char* moduleName,
    const char* entryPointName,
    std::unique_ptr<ShaderModule>& outShaderModule)
{
    ShaderCompileResult compileResult;
    Result<> result = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = moduleName,
            .entryPointName = entryPointName,
            .searchPath = kTriangleShaderSearchPath,
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
    if (!result) {
        spdlog::error("Slang compile failed for {}.{}", moduleName, entryPointName);
        if (!compileResult.diagnostics.empty()) {
            spdlog::error("{}", compileResult.diagnostics);
        }
        return result;
    }
    if (!compileResult.diagnostics.empty()) {
        spdlog::warn("{}", compileResult.diagnostics);
    }

    const std::string shaderDebugName = std::string(moduleName) + "." + entryPointName;
    return device.createShaderModule(ShaderModuleDesc{
        .spirv = compileResult.spirv,
        .debugName = shaderDebugName.c_str(),
    }).transform([&](auto rhiValue) { outShaderModule = std::move(rhiValue); });
}

Result<> createTriangleShaderModule(
    Device& device,
    const char* entryPointName,
    std::unique_ptr<ShaderModule>& outShaderModule)
{
    return createSlangShaderModule(device, kTriangleShaderModuleName, entryPointName, outShaderModule);
}

} // namespace

namespace detail {

struct TrianglePreviewRendererImpl {
    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    std::unique_ptr<ShaderModule> vertexShader;
    std::unique_ptr<ShaderModule> fragmentShader;
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::unique_ptr<Texture> colorTexture;
    std::unique_ptr<TextureView> colorTextureView;
    std::unique_ptr<Buffer> readbackBuffer;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;

    Result<> initialize(bool enableValidation);
    Result<> ensureResources(uint32_t newWidth, uint32_t newHeight);
    Result<> render(uint32_t newWidth, uint32_t newHeight);
};

Result<> TrianglePreviewRendererImpl::initialize(bool enableValidation)
{
    Result<> result = createDevice(DeviceDesc{
            .applicationName = "Metallic Triangle Preview",
            .enableValidation = enableValidation,
            .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{
                .enableAftermath = true,
            },
        }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    graphicsQueue = device->getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
    if (!result) {
        return result;
    }
    result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
    if (!result) {
        return result;
    }
    result = device->createFence(true).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    result = createTriangleShaderModule(*device, kTriangleVertexEntryPoint, vertexShader);
    if (!result) {
        return result;
    }
    result = createTriangleShaderModule(*device, kTriangleFragmentEntryPoint, fragmentShader);
    if (!result) {
        return result;
    }

    return device->createGraphicsPipeline(GraphicsPipelineDesc{
        .vertexShader = {vertexShader.get()},
        .fragmentShader = {fragmentShader.get()},
        .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1,
        .topology = PrimitiveTopology::TriangleList,
    }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); });
}

Result<> TrianglePreviewRendererImpl::ensureResources(uint32_t newWidth, uint32_t newHeight)
{
    if (newWidth == 0 || newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    if (newWidth == width && newHeight == height && colorTexture != nullptr && readbackBuffer != nullptr) {
        return {};
    }

    if (device == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    (void)device->waitIdle();
    colorTextureView.reset();
    colorTexture.reset();
    readbackBuffer.reset();

    Result<> result = device->createTexture(TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
            .format = Format::RGBA8Unorm,
            .width = newWidth,
            .height = newHeight,
            .depth = 1,
            .mipCount = 1,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        }).transform([&](auto rhiValue) { colorTexture = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    result = device->createTextureView(*colorTexture,
        TextureViewDesc{
            .format = Format::RGBA8Unorm,
            .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
        }).transform([&](auto rhiValue) { colorTextureView = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
    result = device->createBuffer(BufferDesc{
            .size = byteSize,
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
        }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    width = newWidth;
    height = newHeight;
    pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    return {};
}

Result<> TrianglePreviewRendererImpl::render(uint32_t newWidth, uint32_t newHeight)
{
    Result<> result = ensureResources(newWidth, newHeight);
    if (!result) {
        return result;
    }

    result = fence->wait();
    if (!result) {
        return result;
    }
    result = fence->reset();
    if (!result) {
        return result;
    }
    result = commandPool->reset();
    if (!result) {
        return result;
    }

    result = commandBuffer->begin();
    if (!result) {
        return result;
    }

    TextureBarrierDesc toColor{
        .texture = colorTexture.get(),
        .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::ColorAttachment,
        .before = {},
        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&toColor, 1}}); !commandResult) { return commandResult; }

    const Rect renderArea{
        .x = 0,
        .y = 0,
        .width = width,
        .height = height,
    };
    RenderingAttachmentDesc colorAttachment{
        .view = colorTextureView.get(),
        .layout = TextureLayout::ColorAttachment,
        .loadOp = LoadOp::Clear,
        .storeOp = StoreOp::Store,
        .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
    };
    if (auto commandResult = commandBuffer->beginRendering(RenderingDesc{
        .renderArea = renderArea,
        .colorAttachments = {&colorAttachment, 1},
    }); !commandResult) { return commandResult; }
    if (auto commandResult = commandBuffer->setViewport(Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(width),
        .height = static_cast<float>(height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    }); !commandResult) { return commandResult; }
    commandBuffer->setScissor(renderArea);
    if (auto commandResult = commandBuffer->bindExecution((pipeline)->execution()); !commandResult) { return commandResult; }
    if (auto commandResult = commandBuffer->draw(3); !commandResult) { return commandResult; }
    commandBuffer->endRendering();

    TextureBarrierDesc toTransfer{
        .texture = colorTexture.get(),
        .oldLayout = TextureLayout::ColorAttachment,
        .newLayout = TextureLayout::TransferSource,
        .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&toTransfer, 1}}); !commandResult) { return commandResult; }
    if (auto commandResult = (readbackBuffer.get())->slice().and_then([&](const auto& bufferSlice) { return commandBuffer->copyTextureToBuffer(BufferTextureRegion{
        .texture = colorTexture.get(),
        .buffer = bufferSlice,
        .width = width,
        .height = height,
        .depth = 1,
        .mipLevel = 0,
        .baseLayer = 0,
    }); }); !commandResult) { return commandResult; }

    result = commandBuffer->end();
    if (!result) {
        return result;
    }

    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = graphicsQueue->submit(QueueSubmitDesc{
        .commandBuffers = {commandBuffers, 1},
        .signalFence = fence.get(),
    });
    if (!result) {
        return result;
    }
    result = fence->wait();
    if (!result) {
        return result;
    }

    readbackBuffer->invalidate();
    void* mapped = readbackBuffer->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }

    const uint64_t byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ull;
    std::memcpy(pixels.data(), mapped, static_cast<size_t>(byteSize));
    readbackBuffer->unmap();
    return {};
}

} // namespace detail

TrianglePreviewRenderer::TrianglePreviewRenderer()
    : impl_(std::make_unique<detail::TrianglePreviewRendererImpl>())
{
}

TrianglePreviewRenderer::~TrianglePreviewRenderer() = default;
TrianglePreviewRenderer::TrianglePreviewRenderer(TrianglePreviewRenderer&&) noexcept = default;
TrianglePreviewRenderer& TrianglePreviewRenderer::operator=(TrianglePreviewRenderer&&) noexcept = default;

Result<> TrianglePreviewRenderer::initialize(bool enableValidation)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->initialize(enableValidation);
}

Result<> TrianglePreviewRenderer::render(uint32_t width, uint32_t height)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->render(width, height);
}

const std::vector<uint32_t>& TrianglePreviewRenderer::pixels() const
{
    static const std::vector<uint32_t> emptyPixels;
    return impl_ != nullptr ? impl_->pixels : emptyPixels;
}

uint32_t TrianglePreviewRenderer::width() const
{
    return impl_ != nullptr ? impl_->width : 0;
}

uint32_t TrianglePreviewRenderer::height() const
{
    return impl_ != nullptr ? impl_->height : 0;
}

int runRhiTrianglePreviewTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    int exitCode = 0;
    {
        TrianglePreviewRenderer previewRenderer;
        Result<> result = previewRenderer.initialize(enableValidation);
        if (!checkResult(result, "TrianglePreviewRenderer::initialize")) {
            exitCode = resultToExitCode(result);
        } else {
            result = previewRenderer.render(320, 240);
            if (!checkResult(result, "TrianglePreviewRenderer::render")) {
                exitCode = resultToExitCode(result);
            } else {
                uint32_t brightPixelCount = 0;
                const std::vector<uint32_t>& pixels = previewRenderer.pixels();
                const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
                for (size_t index = 0; index < pixels.size(); ++index) {
                    const uint8_t r = bytes[index * 4 + 0];
                    const uint8_t g = bytes[index * 4 + 1];
                    const uint8_t b = bytes[index * 4 + 2];
                    if (r > 120 || g > 120 || b > 120) {
                        ++brightPixelCount;
                    }
                }

                if (brightPixelCount < 256) {
                    spdlog::error(
                        "Triangle preview pixel check failed: only {} bright pixels found.",
                        brightPixelCount);
                    exitCode = 1;
                }
            }
        }
    }

    SDL_Quit();
    return exitCode;
}

int runRhiBindlessDescriptorHeapSmokeTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    int exitCode = 0;
    {
        constexpr uint32_t kWidth = 16;
        constexpr uint32_t kHeight = 16;
        constexpr uint64_t kReadbackByteSize = static_cast<uint64_t>(kWidth) * kHeight * 4ull;

        std::unique_ptr<Device> device;
        std::unique_ptr<CommandPool> commandPool;
        std::unique_ptr<CommandBuffer> commandBuffer;
        std::unique_ptr<Fence> fence;
        std::unique_ptr<Texture> sourceTexture;
        std::unique_ptr<TextureView> sourceTextureView;
        std::unique_ptr<Texture> outputTexture;
        std::unique_ptr<TextureView> outputTextureView;
        std::unique_ptr<Buffer> readbackBuffer;
        std::unique_ptr<BindlessHeap> bindlessHeap;
        std::unique_ptr<ShaderModule> vertexShader;
        std::unique_ptr<ShaderModule> fragmentShader;
        std::unique_ptr<GraphicsPipeline> pipeline;

        Result<> result = createDevice(DeviceDesc{
                .applicationName = "Metallic RHI Bindless Descriptor Heap Smoke Test",
                .enableValidation = enableValidation,
                .enableBindlessDescriptorHeap = true,
                .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{
                    .enableAftermath = true,
                },
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!checkResult(result, "createDevice")) {
            exitCode = resultToExitCode(result);
        } else {
            const BindlessHeapDesc bindlessHeapDesc{
                .maxSampledImages = 1,
            };
            result = device->createBindlessHeap(bindlessHeapDesc).transform([&](auto rhiValue) { bindlessHeap = std::move(rhiValue); });
            if (!device->capabilities().bindlessDescriptorHeap) {
                if (hasError(result, Error::Unsupported)) {
                    spdlog::info("VK_EXT_descriptor_heap unsupported; bindless smoke test skipped.");
                } else {
                    spdlog::error(
                        "createBindlessHeap was expected to return Unsupported, got {}",
                        resultToString(result));
                    exitCode = 1;
                }
            } else if (!checkResult(result, "createBindlessHeap")) {
                exitCode = resultToExitCode(result);
            } else {
                Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
                if (graphicsQueue == nullptr) {
                    spdlog::error("No graphics queue available.");
                    exitCode = 1;
                }

                if (exitCode == 0) {
                    result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
                    if (!checkResult(result, "createCommandPool")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
                    if (!checkResult(result, "createCommandBuffer")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createFence(true).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
                    if (!checkResult(result, "createFence")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTexture(TextureDesc{
                            .type = TextureType::Texture2D,
                            .usage = TextureUsageBits::Sampled | TextureUsageBits::ColorAttachment,
                            .format = Format::RGBA8Unorm,
                            .width = kWidth,
                            .height = kHeight,
                            .depth = 1,
                            .mipCount = 1,
                            .layerCount = 1,
                            .memoryLocation = MemoryLocation::Device,
                        }).transform([&](auto rhiValue) { sourceTexture = std::move(rhiValue); });
                    if (!checkResult(result, "createTexture(source)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTextureView(*sourceTexture,
                        TextureViewDesc{
                            .format = Format::RGBA8Unorm,
                            .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                        }).transform([&](auto rhiValue) { sourceTextureView = std::move(rhiValue); });
                    if (!checkResult(result, "createTextureView(source)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTexture(TextureDesc{
                            .type = TextureType::Texture2D,
                            .usage = TextureUsageBits::ColorAttachment | TextureUsageBits::TransferSource,
                            .format = Format::RGBA8Unorm,
                            .width = kWidth,
                            .height = kHeight,
                            .depth = 1,
                            .mipCount = 1,
                            .layerCount = 1,
                            .memoryLocation = MemoryLocation::Device,
                        }).transform([&](auto rhiValue) { outputTexture = std::move(rhiValue); });
                    if (!checkResult(result, "createTexture(output)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createTextureView(*outputTexture,
                        TextureViewDesc{
                            .format = Format::RGBA8Unorm,
                            .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                        }).transform([&](auto rhiValue) { outputTextureView = std::move(rhiValue); });
                    if (!checkResult(result, "createTextureView(output)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createBuffer(BufferDesc{
                            .size = kReadbackByteSize,
                            .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback,
                        }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
                    if (!checkResult(result, "createBuffer(readback)")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                BindlessHandle sourceImageHandle;
                if (exitCode == 0) {
                    result = bindlessHeap->allocate(BindlessHandleKind::SampledImage).transform([&](auto rhiValue) { sourceImageHandle = std::move(rhiValue); });
                    if (!checkResult(result, "allocateSampledImage")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = bindlessHeap->writeSampledImage(
                        sourceImageHandle,
                        *sourceTextureView,
                        TextureLayout::ShaderRead);
                    if (!checkResult(result, "writeSampledImage")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = createSlangShaderModule(
                        *device,
                        kBindlessSmokeShaderModuleName,
                        kBindlessSmokeVertexEntryPoint,
                        vertexShader);
                    if (!checkResult(result, "createSlangShaderModule(vertex)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = createSlangShaderModule(
                        *device,
                        kBindlessSmokeShaderModuleName,
                        kBindlessSmokeFragmentEntryPoint,
                        fragmentShader);
                    if (!checkResult(result, "createSlangShaderModule(fragment)")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = device->createGraphicsPipeline(GraphicsPipelineDesc{
                        .vertexShader = {vertexShader.get()},
                        .fragmentShader = {fragmentShader.get()},
                        .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1,
                        .topology = PrimitiveTopology::TriangleList,
                        .usesBindlessHeap = true,
                    }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); });
                    if (!checkResult(result, "createGraphicsPipeline(bindless)")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    result = fence->wait();
                    if (!checkResult(result, "fence wait")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = fence->reset();
                    if (!checkResult(result, "fence reset")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandPool->reset();
                    if (!checkResult(result, "commandPool reset")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = commandBuffer->begin();
                    if (!checkResult(result, "commandBuffer begin")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    TextureBarrierDesc sourceToColor{
                        .texture = sourceTexture.get(),
                        .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::ColorAttachment,
                        .before = {},
                        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                    };
                    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&sourceToColor, 1}}); !commandResult) { return 1; }

                    const Rect renderArea{
                        .x = 0,
                        .y = 0,
                        .width = kWidth,
                        .height = kHeight,
                    };
                    RenderingAttachmentDesc sourceAttachment{
                        .view = sourceTextureView.get(),
                        .layout = TextureLayout::ColorAttachment,
                        .loadOp = LoadOp::Clear,
                        .storeOp = StoreOp::Store,
                        .clearColor = ColorValue{0.25f, 0.50f, 0.75f, 1.0f},
                    };
                    if (auto commandResult = commandBuffer->beginRendering(RenderingDesc{
                        .renderArea = renderArea,
                        .colorAttachments = {&sourceAttachment, 1},
                    }); !commandResult) { return 1; }
                    commandBuffer->endRendering();

                    TextureBarrierDesc sourceToShaderRead{
                        .texture = sourceTexture.get(),
                        .oldLayout = TextureLayout::ColorAttachment,
                        .newLayout = TextureLayout::ShaderRead,
                        .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                    };
                    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&sourceToShaderRead, 1}}); !commandResult) { return 1; }

                    TextureBarrierDesc outputToColor{
                        .texture = outputTexture.get(),
                        .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::ColorAttachment,
                        .before = {},
                        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                    };
                    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&outputToColor, 1}}); !commandResult) { return 1; }

                    RenderingAttachmentDesc outputAttachment{
                        .view = outputTextureView.get(),
                        .layout = TextureLayout::ColorAttachment,
                        .loadOp = LoadOp::Clear,
                        .storeOp = StoreOp::Store,
                        .clearColor = ColorValue{0.0f, 0.0f, 0.0f, 1.0f},
                    };
                    if (auto commandResult = commandBuffer->beginRendering(RenderingDesc{
                        .renderArea = renderArea,
                        .colorAttachments = {&outputAttachment, 1},
                    }); !commandResult) { return 1; }
                    if (auto commandResult = commandBuffer->setViewport(Viewport{
                        .x = 0.0f,
                        .y = 0.0f,
                        .width = static_cast<float>(kWidth),
                        .height = static_cast<float>(kHeight),
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f,
                    }); !commandResult) { return 1; }
                    commandBuffer->setScissor(renderArea);
                    if (auto commandResult = commandBuffer->bindExecution((pipeline)->execution()); !commandResult) { return 1; }
                    commandBuffer->bindBindlessHeap(*bindlessHeap);
                    commandBuffer->pushBindlessData(&sourceImageHandle.shaderIndex, sizeof(sourceImageHandle.shaderIndex));
                    if (auto commandResult = commandBuffer->draw(3); !commandResult) { return 1; }
                    commandBuffer->endRendering();

                    TextureBarrierDesc outputToTransfer{
                        .texture = outputTexture.get(),
                        .oldLayout = TextureLayout::ColorAttachment,
                        .newLayout = TextureLayout::TransferSource,
                        .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
                    };
                    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&outputToTransfer, 1}}); !commandResult) { return 1; }
                    if (auto commandResult = (readbackBuffer.get())->slice().and_then([&](const auto& bufferSlice) { return commandBuffer->copyTextureToBuffer(BufferTextureRegion{
                        .texture = outputTexture.get(),
                        .buffer = bufferSlice,
                        .width = kWidth,
                        .height = kHeight,
                        .depth = 1,
                        .mipLevel = 0,
                        .baseLayer = 0,
                    }); }); !commandResult) { return 1; }

                    result = commandBuffer->end();
                    if (!checkResult(result, "commandBuffer end")) {
                        exitCode = resultToExitCode(result);
                    }
                }

                if (exitCode == 0) {
                    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
                    result = graphicsQueue->submit(QueueSubmitDesc{
                        .commandBuffers = {commandBuffers, 1},
                        .signalFence = fence.get(),
                    });
                    if (!checkResult(result, "graphicsQueue submit")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    result = fence->wait();
                    if (!checkResult(result, "fence wait after submit")) {
                        exitCode = resultToExitCode(result);
                    }
                }
                if (exitCode == 0) {
                    readbackBuffer->invalidate();
                    std::vector<uint32_t> pixels(static_cast<size_t>(kWidth) * kHeight);
                    void* mapped = readbackBuffer->map();
                    if (mapped == nullptr) {
                        spdlog::error("Failed to map bindless smoke readback buffer.");
                        exitCode = 1;
                    } else {
                        std::memcpy(pixels.data(), mapped, static_cast<size_t>(kReadbackByteSize));
                        readbackBuffer->unmap();

                        uint32_t matchedPixelCount = 0;
                        const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
                        for (size_t index = 0; index < pixels.size(); ++index) {
                            const uint8_t r = bytes[index * 4 + 0];
                            const uint8_t g = bytes[index * 4 + 1];
                            const uint8_t b = bytes[index * 4 + 2];
                            const uint8_t a = bytes[index * 4 + 3];
                            if (r >= 48 && r <= 80 && g >= 112 && g <= 144 && b >= 176 && b <= 208 && a >= 240) {
                                ++matchedPixelCount;
                            }
                        }

                        if (matchedPixelCount < pixels.size() / 2) {
                            const uint8_t r = bytes[0];
                            const uint8_t g = bytes[1];
                            const uint8_t b = bytes[2];
                            const uint8_t a = bytes[3];
                            spdlog::error(
                                "Bindless descriptor heap pixel check failed: {} matching pixels. First pixel RGBA=({}, {}, {}, {}).",
                                matchedPixelCount,
                                static_cast<uint32_t>(r),
                                static_cast<uint32_t>(g),
                                static_cast<uint32_t>(b),
                                static_cast<uint32_t>(a));
                            exitCode = 1;
                        }
                    }
                }
            }

            if (device != nullptr) {
                (void)device->waitIdle();
            }
        }
    }

    SDL_Quit();
    return exitCode;
}

int runRhiSmokeTest(bool enableValidation)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("SDL_Init failed: {}", SDL_GetError());
        return 1;
    }

    const SDL_WindowFlags windowFlags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    SDL_Window* window = SDL_CreateWindow("Metallic RHI Smoke Test", 1280, 720, windowFlags);
    if (window == nullptr) {
        spdlog::error("SDL_CreateWindow failed: {}", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    std::unique_ptr<Device> device;
    std::unique_ptr<Swapchain> swapchain;
    std::vector<std::unique_ptr<TextureView>> swapchainViews;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<SwapchainSemaphore> imageAvailable;
    std::vector<std::unique_ptr<SwapchainSemaphore>> renderFinishedSemaphores;
    std::unique_ptr<Fence> frameFence;

    auto cleanup = [&]() {
        if (device != nullptr) {
            (void)device->waitIdle();
        }

        swapchainViews.clear();
        renderFinishedSemaphores.clear();
        commandBuffer.reset();
        commandPool.reset();
        imageAvailable.reset();
        frameFence.reset();
        swapchain.reset();
        device.reset();

        if (window != nullptr) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        SDL_Quit();
    };

    int pixelWidth = 0;
    int pixelHeight = 0;
    if (!SDL_GetWindowSizeInPixels(window, &pixelWidth, &pixelHeight)) {
        spdlog::error("SDL_GetWindowSizeInPixels failed: {}", SDL_GetError());
        cleanup();
        return 1;
    }

    Result<> result = createDevice(DeviceDesc{
            .applicationName = "Metallic RHI Smoke Test",
            .enableValidation = enableValidation,
            .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{
                .enableAftermath = true,
            },
        }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
    if (!checkResult(result, "createDevice")) {
        cleanup();
        return resultToExitCode(result);
    }

    Queue* graphicsQueue = device->getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        spdlog::error("No graphics queue available.");
        cleanup();
        return 1;
    }

    result = device->createSwapchain(SwapchainDesc{
            .window = {
                .system = WindowSystem::SDL3,
                .nativeWindow = window,
            },
            .width = static_cast<uint32_t>(std::max(pixelWidth, 1)),
            .height = static_cast<uint32_t>(std::max(pixelHeight, 1)),
            .imageCount = 3,
            .framesInFlight = 2,
            .format = Format::BGRA8sRGB,
            .vsync = true,
        }).transform([&](auto rhiValue) { swapchain = std::move(rhiValue); });
    if (!checkResult(result, "createSwapchain")) {
        cleanup();
        return resultToExitCode(result);
    }

    swapchainViews.reserve(swapchain->imageCount());
    renderFinishedSemaphores.reserve(swapchain->imageCount());
    for (uint32_t imageIndex = 0; imageIndex < swapchain->imageCount(); ++imageIndex) {
        std::unique_ptr<TextureView> view;
        result = device->createTextureView(*swapchain->texture(imageIndex),
            TextureViewDesc{
                .format = swapchain->format(),
                .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
            }).transform([&](auto rhiValue) { view = std::move(rhiValue); });
        if (!checkResult(result, "createTextureView")) {
            cleanup();
            return resultToExitCode(result);
        }
        swapchainViews.push_back(std::move(view));

        std::unique_ptr<SwapchainSemaphore> renderFinished;
        result = device->createSwapchainSemaphore().transform([&](auto rhiValue) { renderFinished = std::move(rhiValue); });
        if (!checkResult(result, "createSwapchainSemaphore(renderFinished)")) {
            cleanup();
            return resultToExitCode(result);
        }
        renderFinishedSemaphores.push_back(std::move(renderFinished));
    }

    result = device->createCommandPool(*graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
    if (!checkResult(result, "createCommandPool")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
    if (!checkResult(result, "createCommandBuffer")) {
        cleanup();
        return resultToExitCode(result);
    }

    if (!checkResult(device->createSwapchainSemaphore().transform([&](auto rhiValue) { imageAvailable = std::move(rhiValue); }), "createSwapchainSemaphore(imageAvailable)") ||
        !checkResult(device->createFence(false).transform([&](auto rhiValue) { frameFence = std::move(rhiValue); }), "createFence")) {
        cleanup();
        return 1;
    }

    uint32_t imageIndex = 0;
    result = swapchain->acquireNextImage(*imageAvailable).transform([&](auto rhiValue) { imageIndex = std::move(rhiValue); });
    if (!checkResult(result, "acquireNextImage")) {
        cleanup();
        return resultToExitCode(result);
    }
    if (imageIndex >= renderFinishedSemaphores.size() || renderFinishedSemaphores[imageIndex] == nullptr) {
        spdlog::error("acquireNextImage returned invalid image index.");
        cleanup();
        return 1;
    }

    result = commandBuffer->begin();
    if (!checkResult(result, "CommandBuffer::begin")) {
        cleanup();
        return resultToExitCode(result);
    }

    TextureBarrierDesc toColor{
        .texture = swapchain->texture(imageIndex),
        .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::ColorAttachment,
        .before = {},
        .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&toColor, 1}}); !commandResult) { return 1; }

    const Rect renderArea{
        .x = 0,
        .y = 0,
        .width = swapchain->width(),
        .height = swapchain->height(),
    };
    const ColorValue clearColor{0.04f, 0.08f, 0.13f, 1.0f};
    RenderingAttachmentDesc colorAttachment{
        .view = swapchainViews[imageIndex].get(),
        .layout = TextureLayout::ColorAttachment,
        .loadOp = LoadOp::DontCare,
        .storeOp = StoreOp::Store,
        .clearColor = clearColor,
    };
    if (auto commandResult = commandBuffer->beginRendering(RenderingDesc{
        .renderArea = renderArea,
        .colorAttachments = {&colorAttachment, 1},
    }); !commandResult) { return 1; }
    commandBuffer->clearColorAttachment(0, clearColor, renderArea);
    commandBuffer->endRendering();

    TextureBarrierDesc toPresent{
        .texture = swapchain->texture(imageIndex),
        .oldLayout = TextureLayout::ColorAttachment,
        .newLayout = TextureLayout::Present,
        .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
        .after = {},
        .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer->synchronize(BarrierDesc{.textures = {&toPresent, 1}}); !commandResult) { return 1; }

    result = commandBuffer->end();
    if (!checkResult(result, "CommandBuffer::end")) {
        cleanup();
        return resultToExitCode(result);
    }

    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    SwapchainSemaphoreSubmitDesc waitSemaphore{
        .semaphore = imageAvailable.get(),
        .stages = PipelineStageBits::ColorAttachment,
    };
    SwapchainSemaphoreSubmitDesc signalSemaphore{
        .semaphore = renderFinishedSemaphores[imageIndex].get(),
        .stages = PipelineStageBits::AllCommands,
    };
    result = graphicsQueue->submit(QueueSubmitDesc{
        .waitSwapchainSemaphores = {&waitSemaphore, 1},
        .commandBuffers = {commandBuffers, 1},
        .signalSwapchainSemaphores = {&signalSemaphore, 1},
        .signalFence = frameFence.get(),
    });
    if (!checkResult(result, "Queue::submit")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = swapchain->present(*graphicsQueue, imageIndex, *renderFinishedSemaphores[imageIndex]);
    if (!checkResult(result, "Swapchain::present")) {
        cleanup();
        return resultToExitCode(result);
    }

    result = frameFence->wait();
    if (!checkResult(result, "Fence::wait")) {
        cleanup();
        return resultToExitCode(result);
    }

    cleanup();
    return 0;
}

} // namespace metallic::render
