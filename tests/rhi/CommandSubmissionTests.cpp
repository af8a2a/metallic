#include "RhiTest.h"

#include <array>
#include <cmath>
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

class TimestampQueryTest : public RhiTest {
public:
    TimestampQueryTest()
    {
        type = RhiTestType::Command;
        name = "timestamp_query";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries ||
            context.graphicsQueue.timestampValidBits() == 0) {
            return RhiTestResult::skip("graphics queue does not support timestamp queries");
        }

        std::unique_ptr<render::TimestampQueryPool> queryPool;
        render::Result result = context.device.createTimestampQueryPool(
            context.graphicsQueue,
            render::TimestampQueryPoolDesc{.queryCount = 2},
            queryPool);
        if (!result || queryPool == nullptr) {
            return RhiTestResult::fail(std::string("createTimestampQueryPool returned ") + toString(result));
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
        result = commandBuffer->resetTimestampQueries(*queryPool, 0, 2);
        if (!result) {
            return RhiTestResult::fail(std::string("resetTimestampQueries returned ") + toString(result));
        }
        result = commandBuffer->writeTimestamp(*queryPool, 0, render::PipelineStageBits::TopOfPipe);
        if (!result) {
            return RhiTestResult::fail(std::string("writeTimestamp(begin) returned ") + toString(result));
        }
        commandBuffer->hostWriteBarrier();
        result = commandBuffer->writeTimestamp(*queryPool, 1, render::PipelineStageBits::BottomOfPipe);
        if (!result) {
            return RhiTestResult::fail(std::string("writeTimestamp(end) returned ") + toString(result));
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

        std::array<render::TimestampQueryResult, 2> timestamps{};
        result = queryPool->readResults(0, static_cast<uint32_t>(timestamps.size()), timestamps.data());
        if (!result) {
            return RhiTestResult::fail(std::string("TimestampQueryPool::readResults returned ") + toString(result));
        }
        if (!timestamps[0].available || !timestamps[1].available) {
            return RhiTestResult::fail("timestamp results were unavailable after the submission fence completed");
        }

        const double milliseconds = queryPool->durationMilliseconds(
            timestamps[0].value,
            timestamps[1].value);
        if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
            return RhiTestResult::fail("timestamp duration was invalid");
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(TimestampQueryTest);

} // namespace
} // namespace metallic::tests
