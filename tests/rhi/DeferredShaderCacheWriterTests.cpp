#include "Runtime/Render/Core/DeferredShaderCacheWriter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

using namespace std::chrono_literals;
using render::DeferredShaderCacheWriter;

class CallbackGate {
public:
    void block()
    {
        std::unique_lock lock(mutex_);
        entered_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
    }

    bool waitForEntry()
    {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, 2s, [this] { return entered_; });
    }

    void release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool entered_ = false;
    bool released_ = false;
};

TEST(DeferredShaderCacheWriter, CoalescesRevisionsAndIgnoresUnchangedRequests)
{
    DeferredShaderCacheWriter writer(1h, 1h);
    std::vector<int> saved;
    writer.request("PT", 1, [&] { saved.push_back(1); return true; });
    writer.request("PT", 2, [&] { saved.push_back(2); return true; });
    writer.request("PT", 2, [&] { saved.push_back(20); return true; });
    writer.request("PT", 1, [&] { saved.push_back(10); return true; });
    writer.flush();
    EXPECT_EQ(saved, std::vector<int>({2}));
    writer.request("PT", 2, [&] { saved.push_back(200); return true; });
    writer.flush();
    EXPECT_EQ(saved, std::vector<int>({2}));
}

TEST(DeferredShaderCacheWriter, RepeatedRevisionDoesNotPostponeQuietDeadline)
{
    DeferredShaderCacheWriter writer(200ms, 2s);
    std::mutex mutex;
    std::condition_variable completed;
    bool saved = false;
    const auto start = std::chrono::steady_clock::now();
    auto callback = [&] {
        std::lock_guard lock(mutex);
        saved = true;
        completed.notify_all();
        return true;
    };
    writer.request("PT", 1, callback);
    std::this_thread::sleep_until(start + 150ms);
    writer.request("PT", 1, callback);
    std::unique_lock lock(mutex);
    EXPECT_TRUE(completed.wait_until(lock, start + 300ms, [&] { return saved; }));
    lock.unlock();
    writer.flush();
}

TEST(DeferredShaderCacheWriter, ContinuousRequestsHaveBoundedMaximumDelay)
{
    DeferredShaderCacheWriter writer(400ms, 500ms);
    std::mutex mutex;
    std::condition_variable completed;
    bool saved = false;
    const auto start = std::chrono::steady_clock::now();
    auto callback = [&] {
        std::lock_guard lock(mutex);
        saved = true;
        completed.notify_all();
        return true;
    };
    writer.request("PT", 1, callback);
    for (uint64_t revision = 2; revision <= 5; ++revision) {
        std::this_thread::sleep_until(start + 100ms * (revision - 1));
        writer.request("PT", revision, callback);
    }
    std::unique_lock lock(mutex);
    EXPECT_TRUE(completed.wait_until(lock, start + 650ms, [&] { return saved; }));
    lock.unlock();
    writer.flush();
}

TEST(DeferredShaderCacheWriter, StalledFailedCallbackAllowsRequestAndRetainsNewRevision)
{
    DeferredShaderCacheWriter writer(1h, 1h);
    CallbackGate gate;
    std::vector<int> saved;
    writer.request("PT", 1, [&] { gate.block(); saved.push_back(1); return false; });
    auto firstFlush = std::async(std::launch::async, [&] { writer.flush(); });
    EXPECT_TRUE(gate.waitForEntry());
    auto request = std::async(std::launch::async, [&] {
        writer.request("PT", 2, [&] { saved.push_back(2); return true; });
    });
    EXPECT_EQ(request.wait_for(250ms), std::future_status::ready);
    gate.release();
    request.get();
    firstFlush.get();
    writer.flush();
    EXPECT_EQ(saved, std::vector<int>({1, 2}));
}

TEST(DeferredShaderCacheWriter, ConcurrentFlushesWaitForInFlightCallback)
{
    DeferredShaderCacheWriter writer(1h, 1h);
    CallbackGate gate;
    std::atomic<int> calls = 0;
    writer.request("PT", 1, [&] { ++calls; gate.block(); return true; });
    auto firstFlush = std::async(std::launch::async, [&] { writer.flush(); });
    EXPECT_TRUE(gate.waitForEntry());
    auto secondFlush = std::async(std::launch::async, [&] { writer.flush(); });
    EXPECT_EQ(firstFlush.wait_for(50ms), std::future_status::timeout);
    EXPECT_EQ(secondFlush.wait_for(50ms), std::future_status::timeout);
    gate.release();
    firstFlush.get();
    secondFlush.get();
    EXPECT_EQ(calls, 1);
}

TEST(DeferredShaderCacheWriter, FlushAndDestructionBoundFailureAttempts)
{
    std::atomic<int> failures = 0;
    std::atomic<int> exceptions = 0;
    std::atomic<int> successes = 0;
    {
        DeferredShaderCacheWriter writer(1h, 1h);
        writer.request("failed", 1, [&] { ++failures; return false; });
        writer.request("exception", 1, [&]() -> bool { ++exceptions; throw std::runtime_error("fixture"); });
        writer.request("good", 1, [&] { ++successes; return true; });
        writer.flush();
        EXPECT_EQ(failures, 1);
        EXPECT_EQ(exceptions, 1);
        EXPECT_EQ(successes, 1);
        writer.request("good", 2, [&] { ++successes; return true; });
    }
    EXPECT_EQ(failures, 2);
    EXPECT_EQ(exceptions, 2);
    EXPECT_EQ(successes, 2);
}

TEST(DeferredShaderCacheWriter, RuntimeFailureRetriesWithoutAnotherRequest)
{
    std::mutex mutex;
    std::condition_variable completed;
    int attempts = 0;
    DeferredShaderCacheWriter writer(1ms, 10ms);
    writer.request("PT", 1, [&] {
        std::lock_guard lock(mutex);
        ++attempts;
        completed.notify_all();
        return attempts > 1;
    });
    std::unique_lock lock(mutex);
    EXPECT_TRUE(completed.wait_for(lock, 2s, [&] { return attempts >= 2; }));
    EXPECT_EQ(attempts, 2);
    lock.unlock();
    writer.flush();
}

TEST(DeferredShaderCacheWriter, NestedDeferralsRetainLatestRevisionsUntilFinalRelease)
{
    std::atomic<int> calls = 0;
    std::atomic<int> revision = 0;
    std::promise<void> saved;
    auto completed = saved.get_future();
    DeferredShaderCacheWriter writer(1h, 1h);
    {
        auto outer = writer.deferSaves();
        writer.request("PT", 1, [&] { ++calls; revision = 1; return true; });
        {
            auto inner = writer.deferSaves();
            writer.request("PT", 2, [&] { ++calls; revision = 2; saved.set_value(); return true; });
            writer.request("PT", 1, [&] { ++calls; revision = -1; return true; });
        }
        EXPECT_EQ(completed.wait_for(50ms), std::future_status::timeout);
        EXPECT_EQ(calls, 0);
    }
    EXPECT_EQ(completed.wait_for(2s), std::future_status::ready);
    writer.flush();
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(revision, 2);
}

TEST(DeferredShaderCacheWriter, DeferralSuppressesExpiredAutomaticDeadlines)
{
    std::atomic<int> calls = 0;
    DeferredShaderCacheWriter writer(0ms, 0ms);
    {
        auto scope = writer.deferSaves();
        writer.request("PT", 1, [&] { ++calls; return true; });
        std::this_thread::sleep_for(50ms);
        EXPECT_EQ(calls, 0);
    }
    writer.flush();
    EXPECT_EQ(calls, 1);
}

TEST(DeferredShaderCacheWriter, DeferralEntryWaitsForInFlightCallback)
{
    CallbackGate gate;
    DeferredShaderCacheWriter writer(0ms, 0ms);
    writer.request("PT", 1, [&] { gate.block(); return true; });
    EXPECT_TRUE(gate.waitForEntry());
    auto entry = std::async(std::launch::async, [&] { return writer.deferSaves(); });
    EXPECT_EQ(entry.wait_for(50ms), std::future_status::timeout);
    gate.release();
    auto scope = entry.get();
    std::atomic<int> calls = 0;
    writer.request("PT", 2, [&] { ++calls; return true; });
    std::this_thread::sleep_for(50ms);
    EXPECT_EQ(calls, 0);
    writer.flush();
    EXPECT_EQ(calls, 1);
}

TEST(DeferredShaderCacheWriter, ExplicitFlushOverridesDeferralAndRetriesFailureOnce)
{
    std::atomic<int> attempts = 0;
    DeferredShaderCacheWriter writer(0ms, 0ms);
    {
        auto scope = writer.deferSaves();
        writer.request("PT", 1, [&] { return ++attempts >= 2; });
        writer.flush();
        EXPECT_EQ(attempts, 1);
        std::this_thread::sleep_for(50ms);
        EXPECT_EQ(attempts, 1);
        writer.flush();
        EXPECT_EQ(attempts, 2);
    }
    writer.flush();
    EXPECT_EQ(attempts, 2);
}

TEST(DeferredShaderCacheWriter, FinalReleaseRetriesPendingFailure)
{
    std::atomic<int> attempts = 0;
    std::promise<void> saved;
    auto completed = saved.get_future();
    DeferredShaderCacheWriter writer(1h, 1h);
    {
        auto scope = writer.deferSaves();
        writer.request("PT", 1, [&] {
            if (++attempts == 1) { return false; }
            saved.set_value();
            return true;
        });
        writer.flush();
        EXPECT_EQ(attempts, 1);
    }
    EXPECT_EQ(completed.wait_for(2s), std::future_status::ready);
    writer.flush();
    EXPECT_EQ(attempts, 2);
}

TEST(DeferredShaderCacheWriter, MovedDeferralRetainsContainingOwnerUntilRelease)
{
    struct Owner {
        DeferredShaderCacheWriter writer{1h, 1h};
    };
    auto owner = std::make_shared<Owner>();
    std::weak_ptr<Owner> weak = owner;
    bool saved = false;
    {
        auto first = owner->writer.deferSaves(owner);
        owner->writer.request("PT", 1, [&] { saved = true; return true; });
        auto second = std::move(first);
        owner.reset();
        EXPECT_FALSE(weak.expired());
        EXPECT_FALSE(saved);
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(saved);
}

TEST(DeferredShaderCacheWriter, DestructionDrainsBeforeCallbackResourcesAreReleased)
{
    bool resourcesAlive = true;
    bool saved = false;
    {
        DeferredShaderCacheWriter writer(1h, 1h);
        writer.request("PT", 1, [&] {
            saved = resourcesAlive;
            return true;
        });
    }
    resourcesAlive = false;
    EXPECT_TRUE(saved);
}

} // namespace
} // namespace metallic::tests
