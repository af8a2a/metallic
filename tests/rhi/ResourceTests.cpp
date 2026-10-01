#include "RHITest.h"
#include "harness/Fixtures.h"

#include <cstdint>
#include <cstring>
#include <memory>

namespace metallic::tests {
namespace {

class ResourceLifecycleTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"resource.lifecycle.contract"}, bench::Layer::RHI, "core", "core");
    }

    ResourceLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "resource_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Buffer> invalidBuffer;
        render::Result<> result = context.device.createBuffer(render::BufferDesc{
                .size = 0,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { invalidBuffer = std::move(rhiValue); });
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidBuffer != nullptr) {
            return RHITestResult::fail("zero-sized buffer was not rejected");
        }

        std::unique_ptr<render::Buffer> uploadBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = 256,
                .structureStride = 16,
                .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload,
            }).transform([&](auto rhiValue) { uploadBuffer = std::move(rhiValue); });
        if (!result || uploadBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer returned ") + toString(result));
        }
        if (uploadBuffer->desc().size != 256 || uploadBuffer->desc().structureStride != 16) {
            return RHITestResult::fail("created buffer descriptor does not match request");
        }
        if (uploadBuffer->deviceAddress() == 0) {
            return RHITestResult::fail("transfer buffer requires an implicit device address");
        }
        // These usages must work without Storage or an explicit ShaderDeviceAddress request.
        for (auto usage : {render::BufferUsageBits::Indirect, render::BufferUsageBits::TransferDestination}) {
            std::unique_ptr<render::Buffer> addressBuffer;
            result = context.device.createBuffer({.size = 16, .usage = usage, .memoryLocation = render::MemoryLocation::Device}).transform([&](auto rhiValue) { addressBuffer = std::move(rhiValue); });
            if (!result || addressBuffer == nullptr || addressBuffer->deviceAddress() == 0) {
                return RHITestResult::fail("command buffer usage requires an implicit device address");
            }
        }

        void* mapped = uploadBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("host upload buffer did not map");
        }
        std::memset(mapped, 0x5a, static_cast<size_t>(uploadBuffer->desc().size));
        uploadBuffer->unmap();

        std::unique_ptr<render::Texture> invalidTexture;
        result = context.device.createTexture(render::TextureDesc{
                .format = render::Format::Unknown,
            }).transform([&](auto rhiValue) { invalidTexture = std::move(rhiValue); });
        if (!render::hasError(result, render::Error::InvalidArgument) || invalidTexture != nullptr) {
            return RHITestResult::fail("texture with unknown format was not rejected");
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = render::Format::RGBA8Unorm,
                .width = 16,
                .height = 16,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            }).transform([&](auto rhiValue) { texture = std::move(rhiValue); });
        if (!result || texture == nullptr) {
            return RHITestResult::fail(std::string("createTexture returned ") + toString(result));
        }
        if (texture->desc().format != render::Format::RGBA8Unorm ||
            texture->desc().width != 16 ||
            texture->desc().height != 16) {
            return RHITestResult::fail("created texture descriptor does not match request");
        }

        std::unique_ptr<render::TextureView> textureView;
        result = context.device.createTextureView(*texture,
            render::TextureViewDesc{
                .format = render::Format::RGBA8Unorm,
                .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
            }).transform([&](auto rhiValue) { textureView = std::move(rhiValue); });
        if (!result || textureView == nullptr) {
            return RHITestResult::fail(std::string("createTextureView returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(true).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }
        if (!fence->isSignaled()) {
            return RHITestResult::fail("signaled fence reported unsignaled");
        }
        result = fence->wait(1'000'000);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }
        result = fence->reset();
        if (!result) {
            return RHITestResult::fail(std::string("Fence::reset returned ") + toString(result));
        }
        if (fence->isSignaled()) {
            return RHITestResult::fail("reset fence reported signaled");
        }

        std::unique_ptr<render::Semaphore> semaphore;
        result = context.device.createSemaphore(render::SemaphoreDesc{.initialValue = 2}).transform([&](auto rhiValue) { semaphore = std::move(rhiValue); });
        if (!result || semaphore == nullptr) {
            return RHITestResult::fail(std::string("createSemaphore returned ") + toString(result));
        }
        if (semaphore->currentValue() != 2) {
            return RHITestResult::fail("timeline semaphore initial value does not match request");
        }
        result = semaphore->wait(2, 1'000'000);
        if (!result) {
            return RHITestResult::fail(std::string("Semaphore::wait returned ") + toString(result));
        }
        result = semaphore->signal(3);
        if (!result) {
            return RHITestResult::fail(std::string("Semaphore::signal returned ") + toString(result));
        }
        if (semaphore->currentValue() != 3) {
            return RHITestResult::fail("timeline semaphore host signal did not update value");
        }

        std::unique_ptr<render::SwapchainSemaphore> swapchainSemaphore;
        result = context.device.createSwapchainSemaphore().transform([&](auto rhiValue) { swapchainSemaphore = std::move(rhiValue); });
        if (!result || swapchainSemaphore == nullptr) {
            return RHITestResult::fail(std::string("createSwapchainSemaphore returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

class IndirectArgumentBarrierTest : public RHITest {
public:
    IndirectArgumentBarrierTest()
    {
        type = RHITestType::Command;
        name = "indirect_argument_barrier";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Buffer> indirectBuffer;
        render::Result<> result = context.device.createBuffer(render::BufferDesc{
                .size = 16,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
                .memoryLocation = render::MemoryLocation::Device,
            }).transform([&](auto rhiValue) { indirectBuffer = std::move(rhiValue); });
        if (!result || indirectBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createBuffer(indirect) returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(
                std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(
                std::string("CommandBuffer::begin returned ") + toString(result));
        }

        const render::BufferBarrierDesc toIndirect{
            .buffer = indirectBuffer.get(),
            .before = {},
            .after = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{.buffers = {&toIndirect, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        const render::BufferBarrierDesc toGeneral{
            .buffer = indirectBuffer.get(),
            .before = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{.buffers = {&toGeneral, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }

        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(
                std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(
                std::string("createFence returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = {commandBuffers, 1},
                .signalFence = fence.get(),
            });
        if (!result) {
            return RHITestResult::fail(
                std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(
                std::string("Fence::wait returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ResourceLifecycleTest);
METALLIC_REGISTER_RHI_TEST(IndirectArgumentBarrierTest);

} // namespace
} // namespace metallic::tests
