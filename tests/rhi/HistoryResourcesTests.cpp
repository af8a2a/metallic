#include "Runtime/Render/Core/ResourceSynchronization.h"
#include <stdexcept>
#include <string>

#include "RHITest.h"
#include "harness/Fixtures.h"

#include "Runtime/Render/Core/HistoryResources.h"

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
        .format = render::Format::RGBA8Unorm,
        .width = width,
        .height = height,
        .depth = 1,
        .mipCount = 1,
        .layerCount = 1,
        .memoryLocation = render::MemoryLocation::Device,
    };
}

class HistoryTextureLifecycleTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"history.texture.lifecycle.contract"}, bench::Layer::Core, "core", "core");
    }

    HistoryTextureLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "history_texture_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::TextureDesc desc = makeHistoryTextureDesc(16, 16);

        render::Result<> result = manager.ensureTexture("uninitialized", desc);
        if (!render::hasError(result, render::Error::InvalidArgument)) {
            return RHITestResult::fail("ensureTexture succeeded before initialize");
        }

        result = manager.initialize(context.device);
        if (!result) {
            return RHITestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        const uint64_t initialInvalidationRevision = manager.invalidationRevision();
        manager.invalidateAll();
        if (manager.invalidationRevision() == initialInvalidationRevision) {
            return RHITestResult::fail("invalidateAll did not advance the history invalidation revision");
        }

        manager.beginFrame(0);
        result = manager.ensureTexture("color", desc);
        if (!result) {
            return RHITestResult::fail(std::string("ensureTexture returned ") + toString(result));
        }

        render::HistoryTextureRef current = manager.texture("color", render::HistorySlot::Current);
        render::HistoryTextureRef previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.texture == nullptr || current.view == nullptr || current.desc == nullptr) {
            return RHITestResult::fail("current history texture ref is missing handles");
        }
        if (current.valid || previous.valid || manager.hasPrevious("color")) {
            return RHITestResult::fail("new history texture unexpectedly started valid");
        }

        manager.markWritten("color");
        if (!manager.texture("color", render::HistorySlot::Current).valid) {
            return RHITestResult::fail("markWritten did not validate the current texture slot");
        }

        manager.beginFrame(1);
        current = manager.texture("color", render::HistorySlot::Current);
        previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.valid) {
            return RHITestResult::fail("beginFrame left stale current texture data valid");
        }
        if (!previous.valid || !manager.hasPrevious("color")) {
            return RHITestResult::fail("previous texture slot was not valid on the next frame");
        }

        desc.width = 32;
        result = manager.ensureTexture("color", desc);
        if (!result) {
            return RHITestResult::fail(std::string("ensureTexture(resized) returned ") + toString(result));
        }

        current = manager.texture("color", render::HistorySlot::Current);
        previous = manager.texture("color", render::HistorySlot::Previous);
        if (current.texture == nullptr || current.desc == nullptr || current.desc->width != 32) {
            return RHITestResult::fail("resized current texture ref is invalid");
        }
        if (current.valid || previous.valid || manager.hasPrevious("color")) {
            return RHITestResult::fail("texture descriptor change did not invalidate history");
        }

        if (manager.texture("missing", render::HistorySlot::Current).texture != nullptr) {
            return RHITestResult::fail("missing texture returned a handle");
        }

        return RHITestResult::pass();
    }
};

class HistoryBufferLifecycleTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"history.buffer.lifecycle.contract"}, bench::Layer::Core, "core", "core");
    }

    HistoryBufferLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "history_buffer_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::Result<> result = manager.initialize(context.device);
        if (!result) {
            return RHITestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
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
            return RHITestResult::fail(std::string("ensureBuffer returned ") + toString(result));
        }

        render::HistoryBufferRef current = manager.buffer("moments", render::HistorySlot::Current);
        render::HistoryBufferRef previous = manager.buffer("moments", render::HistorySlot::Previous);
        if (current.buffer == nullptr || current.desc == nullptr || current.view != nullptr || current.viewDesc != nullptr) {
            return RHITestResult::fail("buffer without view returned invalid ref metadata");
        }
        if (current.valid || previous.valid || manager.hasPrevious("moments")) {
            return RHITestResult::fail("new history buffer unexpectedly started valid");
        }

        manager.markWritten("moments");
        manager.beginFrame(1);
        if (!manager.buffer("moments", render::HistorySlot::Previous).valid ||
            !manager.hasPrevious("moments")) {
            return RHITestResult::fail("previous buffer slot was not valid on the next frame");
        }

        manager.invalidate("moments");
        if (manager.hasPrevious("moments") ||
            manager.buffer("moments", render::HistorySlot::Previous).valid) {
            return RHITestResult::fail("invalidate did not clear buffer history validity");
        }

        manager.markWritten("moments");
        manager.beginFrame(2);
        if (!manager.hasPrevious("moments")) {
            return RHITestResult::fail("buffer was not valid before descriptor change");
        }

        desc.size = 128;
        result = manager.ensureBuffer("moments", desc);
        if (!result) {
            return RHITestResult::fail(std::string("ensureBuffer(resized) returned ") + toString(result));
        }

        current = manager.buffer("moments", render::HistorySlot::Current);
        previous = manager.buffer("moments", render::HistorySlot::Previous);
        if (current.buffer == nullptr || current.desc == nullptr || current.desc->size != 128) {
            return RHITestResult::fail("resized current buffer ref is invalid");
        }
        if (current.valid || previous.valid || manager.hasPrevious("moments")) {
            return RHITestResult::fail("buffer descriptor change did not invalidate history");
        }

        return RHITestResult::pass();
    }
};

class HistoryBufferViewLifecycleTest : public RHITest {
public:
    HistoryBufferViewLifecycleTest()
    {
        type = RHITestType::Resource;
        name = "history_buffer_view_lifecycle";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic History Buffer View Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("DeviceCapabilities::bindlessDescriptorHeap is false");
        }

        render::HistoryResourceManager manager;
        result = manager.initialize(*device);
        if (!result) {
            return RHITestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        render::BufferDesc desc{
            .size = 64,
            .structureStride = 16,
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::Device,
        };
        render::BufferViewDesc viewDesc{
            .type = render::BufferViewType::ReadWriteStructured,
            .range = {.offset = 0, .size = UINT64_MAX},
            .structureStride = 0,
        };

        manager.beginFrame(0);
        result = manager.ensureBuffer("structured", desc, &viewDesc);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::skip(std::string("ensureBuffer(view) returned ") + toString(result));
            }
            return RHITestResult::fail(std::string("ensureBuffer(view) returned ") + toString(result));
        }

        render::HistoryBufferRef current = manager.buffer("structured", render::HistorySlot::Current);
        if (current.buffer == nullptr ||
            current.view == nullptr ||
            current.viewDesc == nullptr ||
            current.viewDesc->range.size != 64 ||
            current.viewDesc->structureStride != 16) {
            return RHITestResult::fail("buffer view ref did not expose normalized view metadata");
        }

        manager.markWritten("structured");
        manager.beginFrame(1);
        if (!manager.buffer("structured", render::HistorySlot::Previous).valid ||
            !manager.hasPrevious("structured")) {
            return RHITestResult::fail("previous buffer view slot was not valid on the next frame");
        }

        result = manager.ensureBuffer("structured", desc, &viewDesc);
        if (!result) {
            return RHITestResult::fail(std::string("ensureBuffer(view repeat) returned ") + toString(result));
        }
        if (!manager.hasPrevious("structured")) {
            return RHITestResult::fail("unchanged normalized buffer view descriptor invalidated history");
        }

        (void)device->waitIdle();
        return RHITestResult::pass();
    }
};

class HistoryResourceTransitionSmokeTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"history.transition.rollback.contract"}, bench::Layer::Core, "core", "core");
    }

    HistoryResourceTransitionSmokeTest()
    {
        type = RHITestType::Command;
        name = "history_resource_transition_smoke";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::HistoryResourceManager manager;
        render::Result<> result = manager.initialize(context.device);
        if (!result) {
            return RHITestResult::fail(std::string("HistoryResourceManager::initialize returned ") + toString(result));
        }

        manager.beginFrame(0);
        result = manager.ensureTexture("color", makeHistoryTextureDesc(8, 8));
        if (!result) {
            return RHITestResult::fail(std::string("ensureTexture returned ") + toString(result));
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
            return RHITestResult::fail(std::string("ensureBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::CommandPool> commandPool;
        result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }

        // Failed recording must not publish a state that never reached the GPU.
        const auto oldTextureState = manager.texture("color", render::HistorySlot::Current).state;
        if (!render::hasError(manager.transitionTexture(*commandBuffer, "color", render::HistorySlot::Current,
                render::ResourceState::General), render::Error::InvalidArgument) ||
            !render::hasError(manager.transitionBuffer(*commandBuffer, "historyData", render::HistorySlot::Current,
                render::ResourceState::ShaderRead), render::Error::InvalidArgument) ||
            !render::hasError(manager.transitionBuffer(*commandBuffer, "historyData", render::HistorySlot::Current,
                render::ResourceState::ShaderRead), render::Error::InvalidArgument) ||
            manager.texture("color", render::HistorySlot::Current).state != oldTextureState) {
            return RHITestResult::fail("Failed barrier advanced history resource state");
        }

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }

        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::General);
        if (!result) {
            return RHITestResult::fail(std::string("transitionTexture(General) returned ") + toString(result));
        }
        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::General,
            true);
        if (!result) {
            return RHITestResult::fail(std::string("transitionTexture(force General) returned ") + toString(result));
        }
        result = manager.transitionTexture(
            *commandBuffer,
            "color",
            render::HistorySlot::Current,
            render::ResourceState::ShaderRead);
        if (!result) {
            return RHITestResult::fail(std::string("transitionTexture(ShaderRead) returned ") + toString(result));
        }

        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::General);
        if (!result) {
            return RHITestResult::fail(std::string("transitionBuffer(General) returned ") + toString(result));
        }
        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::General,
            true);
        if (!result) {
            return RHITestResult::fail(std::string("transitionBuffer(force General) returned ") + toString(result));
        }
        result = manager.transitionBuffer(
            *commandBuffer,
            "historyData",
            render::HistorySlot::Current,
            render::ResourceState::TransferSource);
        if (!result) {
            return RHITestResult::fail(std::string("transitionBuffer(TransferSource) returned ") + toString(result));
        }

        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        result = context.graphicsQueue.submit(render::QueueSubmitDesc{
            .commandBuffers = {commandBuffers, 1},
            .signalFence = fence.get(),
        });
        if (!result) {
            return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }

        result = context.graphicsQueue.waitIdle();
        if (!result) {
            return RHITestResult::fail(std::string("Queue::waitIdle returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

class HistoryTexturePlannedStateTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"history.plannedState.transaction.contract"}, bench::Layer::Core, "core", "core");
    }

    HistoryTexturePlannedStateTest()
    {
        type = RHITestType::Command;
        name = "history_texture_planned_state_transactions";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        HistoryResourceManager manager;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        if (!manager.initialize(context.device) ||
            !manager.ensureTexture("color", makeHistoryTextureDesc(8, 8)) ||
            !context.device.createCommandPool(context.graphicsQueue).transform(
                [&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !context.device.createFence(false).transform([&](auto value) { fence = std::move(value); })) {
            return RHITestResult::fail("planned history setup failed");
        }
        const auto record = [&](ResourceState after, bool written) {
            const auto texture = manager.texture("color", HistorySlot::Current);
            const TextureBarrierDesc barrier{
                .texture = texture.texture,
                .oldLayout = metallic::render::textureLayoutForResourceState(texture.state),
                .newLayout = metallic::render::textureLayoutForResourceState(after),
                .before = metallic::render::resourceSyncScope(texture.state, metallic::render::PipelineStageBits::AllCommands),
                .after = metallic::render::resourceSyncScope(after, metallic::render::PipelineStageBits::AllCommands),
                .range = {.mipCount = 1, .layerCount = 1},
            };
            if (auto commandResult = commands->synchronize({.textures = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
            return manager.publishTextureState(*commands, "color", HistorySlot::Current, after, written);
        };
        manager.beginFrame(0);
        if (!commands->begin() || !record(ResourceState::General, true) ||
            !record(ResourceState::ShaderRead, false) || !commands->end()) {
            return RHITestResult::fail("could not record planned history states");
        }
        auto texture = manager.texture("color", HistorySlot::Current);
        if (texture.state != ResourceState::ShaderRead || !texture.valid) {
            return RHITestResult::fail("planned state was not visible to subsequent recordings");
        }
        // Cancellation must address the captured physical slot, after a frame
        // flip, and undo publications in reverse recording order.
        manager.beginFrame(1);
        if (!pool->reset()) { return RHITestResult::fail("planned history cancellation failed"); }
        texture = manager.texture("color", HistorySlot::Previous);
        if (texture.state != ResourceState::Undefined || texture.valid) {
            return RHITestResult::fail("cancel did not restore the original slot layout and invalidate history");
        }
        if (!commands->begin() || !record(ResourceState::General, true) || !commands->end()) {
            return RHITestResult::fail("could not record accepted history state");
        }
        CommandBuffer* submitted[] = {commands.get()};
        if (!context.graphicsQueue.submit({
            .commandBuffers = {submitted, 1},
            .signalFence = fence.get(),
        }) || !fence->wait(5'000'000'000ull) || !pool->reset()) {
            (void)context.graphicsQueue.waitIdle();
            return RHITestResult::fail("accepted history submission failed");
        }
        texture = manager.texture("color", HistorySlot::Current);
        if (texture.state != ResourceState::General || !texture.valid) {
            return RHITestResult::fail("reset rolled back an accepted history publication");
        }
        if (!commands->begin() || !record(ResourceState::ShaderRead, false) ||
            !commands->end() || !pool->reset()) {
            return RHITestResult::fail("could not cancel a history layout update");
        }
        texture = manager.texture("color", HistorySlot::Current);
        if (texture.state != ResourceState::General || texture.valid) {
            return RHITestResult::fail("cancel lost the last accepted layout or retained speculative contents");
        }
        if (!commands->begin() || !record(ResourceState::ShaderRead, true) ||
            !manager.ensureTexture("color", makeHistoryTextureDesc(16, 8)) ||
            !commands->end() || !pool->reset()) {
            return RHITestResult::fail("could not replace pending history storage");
        }
        texture = manager.texture("color", HistorySlot::Current);
        if (texture.state != ResourceState::Undefined || texture.valid || texture.desc->width != 16) {
            return RHITestResult::fail("old transaction modified the replacement history allocation");
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(HistoryTexturePlannedStateTest);
METALLIC_REGISTER_RHI_TEST(HistoryTextureLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryBufferLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryBufferViewLifecycleTest);
METALLIC_REGISTER_RHI_TEST(HistoryResourceTransitionSmokeTest);

} // namespace
} // namespace metallic::tests
