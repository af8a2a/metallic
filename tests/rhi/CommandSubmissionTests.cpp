#include "RHITest.h"
#include "Runtime/Render/GAPI/QueueSubmissionIsolation.h"
#include <future>
#include <stdexcept>
#include "harness/Evidence.h"

#include <array>
#include <cmath>
#include <memory>

namespace metallic::tests {
namespace {

class QueueSubmissionIsolationTest final : public RHITest {
public:
    QueueSubmissionIsolationTest() { type = RHITestType::Command; name = "queue_submission_isolation"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        using namespace std::chrono_literals;
        std::future<Result<>> pending;
        bool sharedProgress = false;
        bool upgradeRejected = false;
        {
            const detail::QueueSubmissionAccess reader;
            pending = std::async(std::launch::async, [&] { return context.graphicsQueue.submit({}); });
            sharedProgress = pending.wait_for(2s) == std::future_status::ready;
            try { const QueueSubmissionIsolation invalidUpgrade; }
            catch (const std::logic_error&) { upgradeRejected = true; }
        }
        if (!pending.get() || !sharedProgress || !upgradeRejected) {
            return RHITestResult::fail("Ordinary submissions did not share access or isolation upgrade was accepted");
        }
        bool excluded = false, nestedExcluded = false;
        Result<> ownerResult;
        std::promise<void> attempted;
        auto started = attempted.get_future();
        {
            const QueueSubmissionIsolation isolation;
            pending = std::async(std::launch::async, [&] {
                attempted.set_value();
                return context.graphicsQueue.submit({});
            });
            started.wait();
            excluded = pending.wait_for(50ms) == std::future_status::timeout;
            {
                const QueueSubmissionIsolation nested;
                ownerResult = context.graphicsQueue.submit({});
            }
            nestedExcluded = pending.wait_for(50ms) == std::future_status::timeout;
            // Validation failure must also release its access scope.
            Queue invalid;
            if (!hasError(invalid.submit({}), Error::InvalidArgument)) {
                ownerResult = makeError(Error::Failure);
            }
        }
        const auto resumed = pending.get();
        const auto idle = context.graphicsQueue.waitIdle();
        if (!excluded || !nestedExcluded || !ownerResult || !resumed || !idle) {
            return RHITestResult::fail("Isolation failed to exclude other threads, permit owner submission or release access");
        }
        return RHITestResult::pass("Shared ordinary access, exclusive/nested owner submission, upgrade rejection and release");
    }
};
METALLIC_REGISTER_RHI_TEST(QueueSubmissionIsolationTest);

class SubmitEmptyCommandBufferTest : public RHITest {
public:
    SubmitEmptyCommandBufferTest()
    {
        type = RHITestType::Command;
        name = "submit_empty_command_buffer";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.coverage = {"queue.submit.completion"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::CommandPool> commandPool;
        render::Result<> result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
        if (!result || commandPool == nullptr) {
            return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
        }

        std::unique_ptr<render::CommandBuffer> commandBuffer;
        result = commandPool->createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
        if (!result || commandBuffer == nullptr) {
            return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
        }

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
        }

        std::unique_ptr<render::Semaphore> semaphore;
        result = context.device.createSemaphore().transform([&](auto rhiValue) { semaphore = std::move(rhiValue); });
        if (!result || semaphore == nullptr) {
            return RHITestResult::fail(std::string("createSemaphore returned ") + toString(result));
        }

        render::CommandBuffer* commandBuffers[] = {commandBuffer.get()};
        render::SemaphoreSubmitDesc signalSemaphore{
            .semaphore = semaphore.get(),
            .value = 1,
            .stages = render::PipelineStageBits::AllCommands,
        };
        result = context.graphicsQueue.submit(
            render::QueueSubmitDesc{
                .commandBuffers = {commandBuffers, 1},
                .signalSemaphores = {&signalSemaphore, 1},
                .signalFence = fence.get(),
            });
        if (!result) {
            return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
        }

        result = fence->wait(5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
        }
        if (!fence->isSignaled()) {
            return RHITestResult::fail("submitted fence reported unsignaled after wait");
        }
        result = semaphore->wait(1, 5'000'000'000ull);
        if (!result) {
            return RHITestResult::fail(std::string("Semaphore::wait returned ") + toString(result));
        }
        if (semaphore->currentValue() != 1) {
            return RHITestResult::fail("submitted timeline semaphore did not reach signal value");
        }

        result = context.graphicsQueue.waitIdle();
        if (!result) {
            return RHITestResult::fail(std::string("Queue::waitIdle returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(SubmitEmptyCommandBufferTest);

class TimestampQueryTest : public RHITest {
public:
    TimestampQueryTest()
    {
        type = RHITestType::Command;
        name = "timestamp_query";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.requirements = {.capabilities = {bench::Capability::TimestampQueries}, .timestampQueues = {render::QueueType::Graphics}},
            .coverage = {"query.timestamp.readback", "query.timestamp.hostReset"}, .artifacts = {"timestamps.json"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries ||
            context.graphicsQueue.timestampValidBits() == 0) {
            return RHITestResult::skip("graphics queue does not support timestamp queries");
        }

        std::unique_ptr<render::TimestampQueryPool> queryPool;
        render::Result<> result = context.device.createTimestampQueryPool(context.graphicsQueue,
            render::TimestampQueryPoolDesc{.queryCount = 2}).transform([&](auto rhiValue) { queryPool = std::move(rhiValue); });
        if (!result || queryPool == nullptr) {
            return RHITestResult::fail(std::string("createTimestampQueryPool returned ") + toString(result));
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

        result = commandBuffer->begin();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = commandBuffer->resetTimestampQueries(*queryPool, 0, 2);
        if (!result) {
            return RHITestResult::fail(std::string("resetTimestampQueries returned ") + toString(result));
        }
        result = commandBuffer->writeTimestamp(*queryPool, 0, render::PipelineStageBits::TopOfPipe);
        if (!result) {
            return RHITestResult::fail(std::string("writeTimestamp(begin) returned ") + toString(result));
        }
        commandBuffer->hostWriteBarrier();
        result = commandBuffer->writeTimestamp(*queryPool, 1, render::PipelineStageBits::BottomOfPipe);
        if (!result) {
            return RHITestResult::fail(std::string("writeTimestamp(end) returned ") + toString(result));
        }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        std::unique_ptr<render::Fence> fence;
        result = context.device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
        if (!result || fence == nullptr) {
            return RHITestResult::fail(std::string("createFence returned ") + toString(result));
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

        std::array<render::TimestampQueryResult, 2> timestamps{};
        result = queryPool->readResults(0, timestamps);
        if (!result) {
            return RHITestResult::fail(std::string("TimestampQueryPool::readResults returned ") + toString(result));
        }
        if (!timestamps[0].available || !timestamps[1].available) {
            return RHITestResult::fail("timestamp results were unavailable after the submission fence completed");
        }

        const double milliseconds = queryPool->durationMilliseconds(
            timestamps[0].value,
            timestamps[1].value);
        if (context.evidence) {
            context.evidence->json("timestamps.json", {{"begin", timestamps[0].value}, {"end", timestamps[1].value},
                {"milliseconds", milliseconds}, {"beginAvailable", timestamps[0].available}, {"endAvailable", timestamps[1].available}});
        }
        if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
            return RHITestResult::fail("timestamp duration was invalid");
        }
        if (queryPool->reset(0, 0) || queryPool->reset(2, 1) || queryPool->reset(1, UINT32_MAX)) {
            return RHITestResult::fail("host timestamp reset accepted an invalid range");
        }
        result = queryPool->reset(0, 1);
        if (!result) {
            return RHITestResult::fail(std::string("TimestampQueryPool::reset returned ") + toString(result));
        }
        result = queryPool->readResults(0, timestamps);
        if (!result || timestamps[0].available || !timestamps[1].available) {
            return RHITestResult::fail("host reset did not invalidate only the selected completed query");
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(TimestampQueryTest);

} // namespace
} // namespace metallic::tests
