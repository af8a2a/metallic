#include "RhiTest.h"

#include <memory>

namespace metallic::tests {
namespace {

class SubmitEmptyCommandBufferTest : public RhiTest {
public:
    SubmitEmptyCommandBufferTest()
    {
        type = RhiTestType::Command;
        name = "submit_empty_command_buffer";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::CommandPool> commandPool;
        render::Result result = context.device.createCommandPool(context.graphicsQueue, commandPool);
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
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false, fence);
        if (!result || fence == nullptr) {
            return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Semaphore> semaphore;
        result = context.device.createSemaphore(semaphore);
        if (!result || semaphore == nullptr) {
            return RhiTestResult::fail(std::string("createSemaphore returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        render::SemaphoreSubmitDesc signalSemaphore{
            .semaphore = semaphore.get(),
            .value = 1,
            .stages = render::PipelineStageBits::AllCommands,
        };
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = commandBuffers,
                .commandBufferCount = 1,
                .signalSemaphores = &signalSemaphore,
                .signalSemaphoreCount = 1,
                .signalFence = fence.get(),
            });
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }
        if (!fence->isSignaled()) {
            return RhiTestResult::fail("submitted fence reported unsignaled after wait");
        }
        result = semaphore->wait(1, 5'000'000'000ull);
        if (!result) {
            return RhiTestResult::fail(std::string("Semaphore::wait returned ") + toString(result));
        }
        if (semaphore->currentValue() != 1) {
            return RhiTestResult::fail("submitted timeline semaphore did not reach signal value");
        }

        result = context.graphicsQueue.waitIdle();
        if (!result) {
            return RhiTestResult::fail(std::string("Queue::waitIdle returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(SubmitEmptyCommandBufferTest);

} // namespace
} // namespace metallic::tests
