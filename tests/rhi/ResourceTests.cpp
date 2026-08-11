#include "RhiTest.h"

#include <cstdint>
#include <cstring>
#include <memory>

namespace metallic::tests {
namespace {

class ResourceLifecycleTest : public RhiTest {
public:
    ResourceLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "resource_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Buffer> invalidBuffer;
        render::Result result = context.device.createBuffer(
            render::BufferDesc{
                .size = 0,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            invalidBuffer);
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidBuffer != nullptr) {
            return RhiTestResult::fail("zero-sized buffer was not rejected");
        }

        std::unique_ptr<render::Buffer> uploadBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = 256,
                .structureStride = 16,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            },
            uploadBuffer);
        if (!result || uploadBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer returned ") + toString(result));
        }
        if (uploadBuffer->desc().size != 256 || uploadBuffer->desc().structureStride != 16) {
            return RhiTestResult::fail("created buffer descriptor does not match request");
        }

        void* mapped = uploadBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("host upload buffer did not map");
        }
        std::memset(mapped, 0x5a, static_cast<size_t>(uploadBuffer->desc().size));
        uploadBuffer->unmap();

        std::unique_ptr<render::Texture> invalidTexture;
        result = context.device.createTexture(
            render::TextureDesc{
                .format = render::Format::Unknown,
            },
            invalidTexture);
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidTexture != nullptr) {
            return RhiTestResult::fail("texture with unknown format was not rejected");
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = render::Format::Rgba8Unorm,
                .width = 16,
                .height = 16,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            },
            texture);
        if (!result || texture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture returned ") + toString(result));
        }
        if (texture->desc().format != render::Format::Rgba8Unorm ||
            texture->desc().width != 16 ||
            texture->desc().height != 16) {
            return RhiTestResult::fail("created texture descriptor does not match request");
        }

        std::unique_ptr<render::TextureView> textureView;
        result = context.device.createTextureView(
            *texture,
            render::TextureViewDesc{
                .format = render::Format::Rgba8Unorm,
                .baseMip = 0,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            textureView);
        if (!result || textureView == nullptr) {
            return RhiTestResult::fail(std::string("createTextureView returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(true, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }
        if (!fence->isSignaled()) {
            return RhiTestResult::fail("signaled fence reported unsignaled");
        }
        result = fence->wait(1'000'000);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }
        result = fence->reset();
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::reset returned ") + toString(result));
        }
        if (fence->isSignaled()) {
            return RhiTestResult::fail("reset fence reported signaled");
        }

        std::unique_ptr<render::Semaphore> semaphore;
        result = context.device.createSemaphore(render::SemaphoreDesc{.initialValue = 2}, semaphore);
        if (!result || semaphore == nullptr) {
            return RhiTestResult::fail(std::string("createSemaphore returned ") + toString(result));
        }
        if (semaphore->currentValue() != 2) {
            return RhiTestResult::fail("timeline semaphore initial value does not match request");
        }
        result = semaphore->wait(2, 1'000'000);
        if (!result) {
            return RhiTestResult::fail(std::string("Semaphore::wait returned ") + toString(result));
        }
        result = semaphore->signal(3);
        if (!result) {
            return RhiTestResult::fail(std::string("Semaphore::signal returned ") + toString(result));
        }
        if (semaphore->currentValue() != 3) {
            return RhiTestResult::fail("timeline semaphore host signal did not update value");
        }

        std::unique_ptr<render::SwapchainSemaphore> swapchainSemaphore;
        result = context.device.createSwapchainSemaphore(swapchainSemaphore);
        if (!result || swapchainSemaphore == nullptr) {
            return RhiTestResult::fail(std::string("createSwapchainSemaphore returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

class IndirectArgumentBarrierTest : public RhiTest {
public:
    IndirectArgumentBarrierTest()
    {
        type = RhiTestType::Command;
        name = "indirect_argument_barrier";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Buffer> indirectBuffer;
        render::Result result = context.device.createBuffer(
            render::BufferDesc{
                .size = 16,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
                .memoryLocation = render::MemoryLocation::Device,
            },
            indirectBuffer);
        if (!result || indirectBuffer == nullptr) {
            return RhiTestResult::fail(
                std::string("createBuffer(indirect) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue, commandPool);
        if (!result || commandPool == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer(commandBuffer);
        if (!result || commandBuffer == nullptr) {
            return RhiTestResult::fail(
                std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(
                std::string("CommandBuffer::begin returned ") + toString(result));
        }

        const render::BufferBarrierDesc toIndirect{
            .buffer = indirectBuffer.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::IndirectArgument,
        };
        commandBuffer->barrier(
            render::BarrierDesc{.buffers = &toIndirect, .bufferCount = 1});

        const render::BufferBarrierDesc toGeneral{
            .buffer = indirectBuffer.get(),
            .before = render::ResourceState::IndirectArgument,
            .after = render::ResourceState::General,
        };
        commandBuffer->barrier(
            render::BarrierDesc{.buffers = &toGeneral, .bufferCount = 1});

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(
                std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(
                std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = commandBuffers,
                .commandBufferCount = 1,
                .signalFence = fence.get(),
            });
        if (!result) {
            return RhiTestResult::fail(
                std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(
                std::string("Fence::wait returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ResourceLifecycleTest);
METALLIC_REGISTER_RHI_TEST(IndirectArgumentBarrierTest);

} // namespace
} // namespace metallic::tests
