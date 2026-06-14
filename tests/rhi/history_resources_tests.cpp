#include "rhi_test.h"

#include "Runtime/Render/history_resources.h"

#include <memory>
#include <string>

namespace metallic::tests {
namespace {

render::TextureDesc makeHistoryTextureDesc(uint32_t width, uint32_t height)
{
    return render::TextureDesc{
        .type = render::TextureType::Texture2D,
        .usage = render::TextureUsageBits::Sampled |
            render::TextureUsageBits::Storage |
            render::TextureUsageBits::TransferSource,
        .format = render::Format::Rgba8Unorm,
        .width = width,
        .height = height,
        .depth = 1,
        .mipCount = 1,
        .layerCount = 1,
        .memoryLocation = render::MemoryLocation::Device,
    };
}

class HistoryTextureLifecycleTest : public RhiTest {
public:
    HistoryTextureLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "history_texture_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::TextureDesc desc = makeHistoryTextureDesc(16, 16);

        render::Result result = manager.ensureTexture("uninitialized", desc);
        if (!render::hasError(result, render::Error::InvalidArgument)) {
            return RhiTestResult::fail("ensureTexture succeeded before initialize");
        }

        result = manager.initialize(context.device);
        if (!result) {
            return RhiTestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        manager.beginFrame(0);
        result = manager.ensureTexture("color", desc);
        if (!result) {
            return RhiTestResult::fail(std::string("ensureTexture returned ") + toString(result));
        }

        render::HistoryTextureRef current = manager.texture("color", render::HistorySlot::Current);
        render::HistoryTextureRef previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr || current.desc == nullptr) {
            return RhiTestResult::fail("current history texture ref is missing handles");
        }
        if (current.valid || previous.valid || manager.hasPrevious("color")) {
            return RhiTestResult::fail("new history texture unexpectedly started valid");
        }

        manager.markWritten("color");
        if (!manager.texture("color", render::HistorySlot::Current).valid) {
            return RhiTestResult::fail("markWritten did not validate the current texture slot");
        }

        manager.beginFrame(1);
        current = manager.texture("color", render::HistorySlot::Current);
        previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.valid) {
            return RhiTestResult::fail("beginFrame left stale current texture data valid");
        }
        if (!previous.valid || !manager.hasPrevious("color")) {
            return RhiTestResult::fail("previous texture slot was not valid on the next frame");
        }

        desc.width = 32;
        result = manager.ensureTexture("color", desc);
        if (!result) {
            return RhiTestResult::fail(std::string("ensureTexture(resized) returned ") + toString(result));
        }

        current = manager.texture("color", render::HistorySlot::Current);
        previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.texture == nullptr || current.desc == nullptr || current.desc->width != 32) {
            return RhiTestResult::fail("resized current texture ref is invalid");
        }
        if (current.valid || previous.valid || manager.hasPrevious("color")) {
            return RhiTestResult::fail("texture descriptor change did not invalidate history");
        }

        if (manager.texture("missing", render::HistorySlot::Current).texture != nullptr) {
            return RhiTestResult::fail("missing texture returned a handle");
        }

        return RhiTestResult::pass();
    }
};

class HistoryBufferLifecycleTest : public RhiTest {
public:
    HistoryBufferLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "history_buffer_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::Result result = manager.initialize(context.device);
        if (!result) {
            return RhiTestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        render::BufferDesc desc{
            .size = 64,
            .structureStride = 16,
            .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::Device,
        };

        manager.beginFrame(0);
        result = manager.ensureBuffer("moments", desc);
        if (!result) {
            return RhiTestResult::fail(std::string("ensureBuffer returned ") + toString(result));
        }

        render::HistoryBufferRef current = manager.buffer("moments", render::HistorySlot::Current);
        render::HistoryBufferRef previous = manager.buffer("moments", render::HistorySlot::Previous);
        if (current.buffer == nullptr || current.desc == nullptr || current.view != nullptr || current.viewDesc != nullptr) {
            return RhiTestResult::fail("buffer without view returned invalid ref metadata");
        }
        if (current.valid || previous.valid || manager.hasPrevious("moments")) {
            return RhiTestResult::fail("new history buffer unexpectedly started valid");
        }

        manager.markWritten("moments");
        manager.beginFrame(1);
        if (!manager.buffer("moments", render::HistorySlot::Previous).valid ||
            !manager.hasPrevious("moments")) {
            return RhiTestResult::fail("previous buffer slot was not valid on the next frame");
        }

        manager.invalidate("moments");
        if (manager.hasPrevious("moments") ||
            manager.buffer("moments", render::HistorySlot::Previous).valid) {
            return RhiTestResult::fail("invalidate did not clear buffer history validity");
        }

        manager.markWritten("moments");
        manager.beginFrame(2);
        if (!manager.hasPrevious("moments")) {
            return RhiTestResult::fail("buffer was not valid before descriptor change");
        }

        desc.size = 128;
        result = manager.ensureBuffer("moments", desc);
        if (!result) {
            return RhiTestResult::fail(std::string("ensureBuffer(resized) returned ") + toString(result));
        }

        current = manager.buffer("moments", render::HistorySlot::Current);
        previous = manager.buffer("moments", render::HistorySlot::Previous);
        if (current.buffer == nullptr || current.desc == nullptr || current.desc->size != 128) {
            return RhiTestResult::fail("resized current buffer ref is invalid");
        }
        if (current.valid || previous.valid || manager.hasPrevious("moments")) {
            return RhiTestResult::fail("buffer descriptor change did not invalidate history");
        }

        return RhiTestResult::pass();
    }
};

class HistoryBufferViewLifecycleTest : public RhiTest {
public:
    HistoryBufferViewLifecycleTest()
    {
        type = RhiTestType::Resource;
        name = "history_buffer_view_lifecycle";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic History Buffer View Test",
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
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::HistoryResourceManager manager;
        result = manager.initialize(*device);
        if (!result) {
            return RhiTestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        render::BufferDesc desc{
            .size = 64,
            .structureStride = 16,
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::Device,
        };
        render::BufferViewDesc viewDesc{
            .type = render::BufferViewType::ReadWriteStructured,
            .offset = 0,
            .size = UINT64_MAX,
            .structureStride = 0,
        };

        manager.beginFrame(0);
        result = manager.ensureBuffer("structured", desc, &viewDesc);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("ensureBuffer(view) returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("ensureBuffer(view) returned ") + toString(result));
        }

        render::HistoryBufferRef current = manager.buffer("structured", render::HistorySlot::Current);
        if (current.buffer == nullptr ||
            current.view == nullptr ||
            current.viewDesc == nullptr ||
            current.viewDesc->size != 64 ||
            current.viewDesc->structureStride != 16) {
            return RhiTestResult::fail("buffer view ref did not expose normalized view metadata");
        }

        manager.markWritten("structured");
        manager.beginFrame(1);
        if (!manager.buffer("structured", render::HistorySlot::Previous).valid ||
            !manager.hasPrevious("structured")) {
            return RhiTestResult::fail("previous buffer view slot was not valid on the next frame");
        }

        result = manager.ensureBuffer("structured", desc, &viewDesc);
        if (!result) {
            return RhiTestResult::fail(std::string("ensureBuffer(view repeat) returned ") + toString(result));
        }
        if (!manager.hasPrevious("structured")) {
            return RhiTestResult::fail("unchanged normalized buffer view descriptor invalidated history");
        }

        (void)device->waitIdle();
        return RhiTestResult::pass();
    }
};

class HistoryResourceTransitionSmokeTest : public RhiTest {
public:
    HistoryResourceTransitionSmokeTest()
    {
        type = RhiTestType::Command;
        name = "history_resource_transition_smoke";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::Result result = manager.initialize(context.device);
        if (!result) {
            return RhiTestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        manager.beginFrame(0);
        result = manager.ensureTexture("color", makeHistoryTextureDesc(8, 8));
        if (!result) {
            return RhiTestResult::fail(std::string("ensureTexture returned ") + toString(result));
        }

        result = manager.ensureBuffer(
            "historyData",
            render::BufferDesc{
                .size = 64,
                .structureStride = 16,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::Device,
            });
        if (!result) {
            return RhiTestResult::fail(std::string("ensureBuffer returned ") + toString(result));
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

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::General);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionTexture(General) returned ") + toString(result));
        }
        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::General,
            true);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionTexture(force General) returned ") + toString(result));
        }
        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::ShaderRead);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionTexture(ShaderRead) returned ") + toString(result));
        }

        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::General);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionBuffer(General) returned ") + toString(result));
        }
        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::General,
            true);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionBuffer(force General) returned ") + toString(result));
        }
        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::TransferSource);
        if (!result) {
            return RhiTestResult::fail(std::string("transitionBuffer(TransferSource) returned ") + toString(result));
        }

        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(render::QueueSubmitDesc{
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

        result = context.graphicsQueue.waitIdle();
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::waitIdle returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(HistoryTextureLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryBufferLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryBufferViewLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryResourceTransitionSmokeTest);

} // namespace
} // namespace metallic::tests
