#include "Runtime/Render/Core/NativePipelineCache.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace metallic::tests {
namespace {

using namespace render;
using detail::NativePipelineCache;
using detail::NativePipelineKey;
using detail::NativePipelineType;
using namespace std::chrono_literals;

const NativePipelineKey kComputeA{NativePipelineType::Compute, 1};
const NativePipelineKey kComputeB{NativePipelineType::Compute, 2};
const NativePipelineKey kComputeC{NativePipelineType::Compute, 3};

class CreatorGate {
public:
    void block()
    {
        std::unique_lock lock(mutex_);
        ++entered_;
        condition_.notify_all();
        condition_.wait(lock, [&] { return released_; });
    }

    bool waitForEntries(size_t count)
    {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, 2s, [&] { return entered_ >= count; });
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
    size_t entered_ = 0;
    bool released_ = false;
};

template <typename Predicate>
bool waitUntil(Predicate&& predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::yield();
    }
    return true;
}

Result<std::shared_ptr<void>> createOwner(int value = 1)
{
    return std::make_shared<int>(value);
}

TEST(NativePipelineCache, FullSpecializationEqualitySurvivesForcedHashCollision)
{
    NativePipelineCache cache;
    NativePipelineKey first{NativePipelineType::Compute, 42,
        {{1000, ShaderSpecializationType::Int32, 0}, {1001, ShaderSpecializationType::Bool32, 1}}};
    auto second = first; second.specializationConstants[0].bits = 1;
    auto a = cache.acquire(first, [] { return createOwner(1); });
    auto b = cache.acquire(second, [] { return createOwner(2); });
    ASSERT_TRUE(a); ASSERT_TRUE(b);
    EXPECT_NE(a->owner, b->owner);
    // The cache owns its key; mutating the caller's vector cannot relabel A.
    first.specializationConstants[0].bits = 9;
    first = second; first.specializationConstants[0].bits = 0;
    auto again = cache.acquire(first, [] { return createOwner(99); });
    ASSERT_TRUE(again);
    EXPECT_EQ(again->owner, a->owner);
    EXPECT_TRUE(again->nativeObjectReused);
}

TEST(NativePipelineCache, SameSpecializationWaitersShareExactlyOneSuccessfulCreation)
{
    NativePipelineCache cache;
    const NativePipelineKey key{NativePipelineType::Compute, 7,
        {{1000, ShaderSpecializationType::Int32, 0xffffffffu}, {1001, ShaderSpecializationType::Bool32, 0}}};
    CreatorGate gate;
    std::atomic_uint calls = 0;
    auto create = [&] { ++calls; gate.block(); return createOwner(31); };
    auto first = std::async(std::launch::async, [&] { return cache.acquire(key, create); });
    const bool entered = gate.waitForEntries(1);
    auto second = std::async(std::launch::async, [&] { return cache.acquire(key, create); });
    const bool joined = waitUntil([&] { return cache.stats().singleFlightWaitCount == 1; });
    gate.release();
    auto a = first.get(); auto b = second.get();
    ASSERT_TRUE(a); ASSERT_TRUE(b);
    EXPECT_TRUE(entered); EXPECT_TRUE(joined);
    EXPECT_EQ(calls, 1); EXPECT_EQ(a->owner, b->owner);
}

TEST(NativePipelineCache, SameKeyWaitersShareExactlyOneSuccessfulCreation)
{
    NativePipelineCache cache;
    CreatorGate gate;
    std::atomic<uint32_t> calls = 0;
    std::barrier start(9);
    std::array<std::future<Result<detail::NativePipelineAcquisition>>, 8> workers;
    for (auto& worker : workers) {
        worker = std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return cache.acquire(kComputeA, [&] {
                ++calls;
                gate.block();
                return createOwner(17);
            });
        });
    }
    start.arrive_and_wait();
    const bool creatorEntered = gate.waitForEntries(1);
    const bool allWaitersJoined = waitUntil([&] { return cache.stats().singleFlightWaitCount == 7; });
    const auto blocked = cache.stats();
    gate.release();
    EXPECT_TRUE(creatorEntered);
    EXPECT_TRUE(allWaitersJoined);
    EXPECT_EQ(blocked.inFlightCount, 1);
    EXPECT_EQ(blocked.nativeCreateCount, 1);
    EXPECT_EQ(blocked.residentCount, 0);
    void* owner = nullptr;
    uint32_t reused = 0;
    uint64_t totalWaitTime = 0;
    for (auto& worker : workers) {
        auto acquired = worker.get();
        ASSERT_TRUE(acquired);
        if (!owner) { owner = acquired->owner.get(); }
        EXPECT_EQ(acquired->owner.get(), owner);
        EXPECT_EQ(*std::static_pointer_cast<int>(acquired->owner), 17);
        reused += uint32_t(acquired->nativeObjectReused);
        totalWaitTime += acquired->singleFlightWaitTimeNanoseconds;
    }
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(reused, 7);
    EXPECT_GT(totalWaitTime, 0);
    const auto after = cache.stats();
    EXPECT_EQ(after.lookupCount, 8);
    EXPECT_EQ(after.inFlightCount, 0);
    EXPECT_EQ(after.failedCreateCount, 0);
    EXPECT_EQ(after.residentCount, 1);
}

TEST(NativePipelineCache, DistinctKeysEnterTheirFactoriesWhileAnotherCreationIsBlocked)
{
    NativePipelineCache cache;
    CreatorGate gate;
    auto create = [&] {
        gate.block();
        return createOwner();
    };
    auto first = std::async(std::launch::async, [&] { return cache.acquire(kComputeA, create); });
    auto second = std::async(std::launch::async, [&] { return cache.acquire(kComputeB, create); });
    const bool overlapped = gate.waitForEntries(2);
    const auto blocked = cache.stats();
    gate.release();
    EXPECT_TRUE(overlapped);
    EXPECT_EQ(blocked.inFlightCount, 2);
    auto firstResult = first.get();
    auto secondResult = second.get();
    ASSERT_TRUE(firstResult);
    ASSERT_TRUE(secondResult);
    EXPECT_NE(firstResult->owner.get(), secondResult->owner.get());
    EXPECT_EQ(cache.stats().nativeCreateCount, 2);
    EXPECT_EQ(cache.stats().singleFlightWaitCount, 0);
}

TEST(NativePipelineCache, FailedFlightPublishesSameErrorToWaitersAndAllowsRetry)
{
    NativePipelineCache cache;
    CreatorGate gate;
    std::atomic<uint32_t> calls = 0;
    std::barrier start(9);
    std::array<std::future<Result<detail::NativePipelineAcquisition>>, 8> workers;
    for (auto& worker : workers) {
        worker = std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return cache.acquire(kComputeA, [&]() -> Result<std::shared_ptr<void>> {
                ++calls;
                gate.block();
                return makeError(Error::DeviceLost);
            });
        });
    }
    start.arrive_and_wait();
    const bool creatorEntered = gate.waitForEntries(1);
    const bool allWaitersJoined = waitUntil([&] { return cache.stats().singleFlightWaitCount == 7; });
    gate.release();
    EXPECT_TRUE(creatorEntered);
    EXPECT_TRUE(allWaitersJoined);
    for (auto& worker : workers) {
        EXPECT_TRUE(hasError(worker.get(), Error::DeviceLost));
    }
    const auto failed = cache.stats();
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(failed.nativeCreateCount, 1);
    EXPECT_EQ(failed.failedCreateCount, 1);
    EXPECT_EQ(failed.inFlightCount, 0);
    EXPECT_EQ(failed.readyEntryCount, 0);
    auto retried = cache.acquire(kComputeA, [&] { ++calls; return createOwner(); });
    ASSERT_TRUE(retried);
    EXPECT_FALSE(retried->nativeObjectReused);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(cache.stats().failedCreateCount, 1);
}

TEST(NativePipelineCache, FactoryExceptionsArePublishedAsErrorsAndDoNotPoisonTheKey)
{
    NativePipelineCache cache;
    auto allocation = cache.acquire(kComputeA, []() -> Result<std::shared_ptr<void>> { throw std::bad_alloc(); });
    EXPECT_TRUE(hasError(allocation, Error::OutOfMemory));
    auto exception = cache.acquire(kComputeA, []() -> Result<std::shared_ptr<void>> {
        throw std::runtime_error("native test factory");
    });
    EXPECT_TRUE(hasError(exception, Error::Failure));
    EXPECT_EQ(cache.stats().inFlightCount, 0);
    auto recovered = cache.acquire(kComputeA, [] { return createOwner(); });
    EXPECT_TRUE(recovered);
    EXPECT_EQ(cache.stats().nativeCreateCount, 3);
    EXPECT_EQ(cache.stats().failedCreateCount, 2);
}

TEST(NativePipelineCache, ZeroCapacityReusesLiveWeakOwnersAndRecreatesAfterAllReferencesExpire)
{
    NativePipelineCache cache(0);
    uint32_t calls = 0;
    auto create = [&] { ++calls; return createOwner(); };
    auto first = cache.acquire(kComputeA, create);
    auto reused = cache.acquire(kComputeA, create);
    ASSERT_TRUE(first);
    ASSERT_TRUE(reused);
    EXPECT_FALSE(first->nativeObjectReused);
    EXPECT_TRUE(reused->nativeObjectReused);
    EXPECT_EQ(first->owner, reused->owner);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(cache.stats().residentCount, 0);
    std::weak_ptr<void> weak = first->owner;
    first->owner.reset();
    reused->owner.reset();
    EXPECT_TRUE(weak.expired());
    auto recreated = cache.acquire(kComputeA, create);
    EXPECT_TRUE(recreated);
    EXPECT_FALSE(recreated->nativeObjectReused);
    EXPECT_EQ(calls, 2);
}

TEST(NativePipelineCache, StrongLRUProtectsRecentlyTouchedObjectsAcrossExternalRelease)
{
    NativePipelineCache cache(2);
    uint32_t calls = 0;
    auto create = [&] { ++calls; return createOwner(); };
    auto first = cache.acquire(kComputeA, create);
    auto second = cache.acquire(kComputeB, create);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    std::weak_ptr<void> firstWeak = first->owner;
    std::weak_ptr<void> secondWeak = second->owner;
    first->owner.reset();
    second->owner.reset();
    EXPECT_FALSE(firstWeak.expired());
    EXPECT_FALSE(secondWeak.expired());
    auto touched = cache.acquire(kComputeA, create);
    ASSERT_TRUE(touched);
    EXPECT_TRUE(touched->nativeObjectReused);
    touched->owner.reset();
    auto third = cache.acquire(kComputeC, create);
    ASSERT_TRUE(third);
    EXPECT_FALSE(firstWeak.expired());
    EXPECT_TRUE(secondWeak.expired());
    EXPECT_EQ(cache.stats().residentCount, 2);
    EXPECT_EQ(cache.stats().evictionCount, 1);
    auto recreated = cache.acquire(kComputeB, create);
    ASSERT_TRUE(recreated);
    EXPECT_FALSE(recreated->nativeObjectReused);
    EXPECT_EQ(calls, 4);
}

TEST(NativePipelineCache, EvictionPreservesWeakReuseOfObjectsHeldByExternalExecutions)
{
    NativePipelineCache cache(1);
    uint32_t calls = 0;
    auto create = [&] { ++calls; return createOwner(); };
    auto first = cache.acquire(kComputeA, create);
    auto second = cache.acquire(kComputeB, create);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    auto reused = cache.acquire(kComputeA, create);
    ASSERT_TRUE(reused);
    EXPECT_TRUE(reused->nativeObjectReused);
    EXPECT_EQ(first->owner, reused->owner);
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(cache.stats().residentCount, 1);
    cache.setCapacity(0);
    EXPECT_EQ(cache.stats().residentCount, 0);
    EXPECT_EQ(cache.stats().strongCapacity, 0);
    auto weakReused = cache.acquire(kComputeA, create);
    ASSERT_TRUE(weakReused);
    EXPECT_TRUE(weakReused->nativeObjectReused);
    EXPECT_EQ(weakReused->owner, first->owner);
    EXPECT_EQ(calls, 2);
}

TEST(NativePipelineCache, CapacityChangesTrimStrongOwnersAndCanRestoreRetention)
{
    NativePipelineCache cache(3);
    auto first = cache.acquire(kComputeA, [] { return createOwner(); });
    auto second = cache.acquire(kComputeB, [] { return createOwner(); });
    auto third = cache.acquire(kComputeC, [] { return createOwner(); });
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    ASSERT_TRUE(third);
    std::weak_ptr<void> firstWeak = first->owner;
    std::weak_ptr<void> secondWeak = second->owner;
    std::weak_ptr<void> thirdWeak = third->owner;
    first->owner.reset();
    second->owner.reset();
    third->owner.reset();
    cache.setCapacity(1);
    EXPECT_TRUE(firstWeak.expired());
    EXPECT_TRUE(secondWeak.expired());
    EXPECT_FALSE(thirdWeak.expired());
    EXPECT_EQ(cache.stats().residentCount, 1);
    EXPECT_EQ(cache.stats().strongCapacity, 1);
    cache.setCapacity(0);
    EXPECT_TRUE(thirdWeak.expired());
    cache.setCapacity(2);
    auto recreated = cache.acquire(kComputeA, [] { return createOwner(); });
    ASSERT_TRUE(recreated);
    firstWeak = recreated->owner;
    recreated->owner.reset();
    EXPECT_FALSE(firstWeak.expired());
    EXPECT_EQ(cache.stats().residentCount, 1);
}

TEST(NativePipelineCache, PipelineKindAndDeviceLocalCacheInstancesDoNotAlias)
{
    NativePipelineCache firstDevice;
    NativePipelineCache secondDevice;
    auto compute = firstDevice.acquire(kComputeA, [] { return createOwner(); });
    auto graphics = firstDevice.acquire({NativePipelineType::Graphics, kComputeA.psoHash}, [] { return createOwner(); });
    auto independent = secondDevice.acquire(kComputeA, [] { return createOwner(); });
    ASSERT_TRUE(compute);
    ASSERT_TRUE(graphics);
    ASSERT_TRUE(independent);
    EXPECT_NE(compute->owner, graphics->owner);
    EXPECT_NE(compute->owner, independent->owner);
    EXPECT_EQ(firstDevice.stats().nativeCreateCount, 2);
    EXPECT_EQ(secondDevice.stats().nativeCreateCount, 1);
}

TEST(NativePipelineCache, CacheDestructionDropsResidentsWhileExternalOwnersRemainAlive)
{
    struct CountedOwner {
        uint32_t& destroyed;
        ~CountedOwner() { ++destroyed; }
    };
    uint32_t destroyed = 0;
    auto cache = std::make_unique<NativePipelineCache>(2);
    auto create = [&]() -> Result<std::shared_ptr<void>> {
        return std::make_shared<CountedOwner>(destroyed);
    };
    auto external = cache->acquire(kComputeA, create);
    auto resident = cache->acquire(kComputeB, create);
    ASSERT_TRUE(external);
    ASSERT_TRUE(resident);
    resident->owner.reset();
    EXPECT_EQ(destroyed, 0);
    cache.reset();
    EXPECT_EQ(destroyed, 1);
    external->owner.reset();
    EXPECT_EQ(destroyed, 2);
}

TEST(NativePipelineCache, EmptySuccessfulOwnersAreRejectedAndCanBeRetried)
{
    NativePipelineCache cache;
    auto empty = cache.acquire(kComputeA, []() -> Result<std::shared_ptr<void>> { return std::shared_ptr<void>{}; });
    EXPECT_TRUE(hasError(empty, Error::Failure));
    EXPECT_EQ(cache.stats().inFlightCount, 0);
    EXPECT_EQ(cache.stats().readyEntryCount, 0);
    auto retry = cache.acquire(kComputeA, [] { return createOwner(); });
    EXPECT_TRUE(retry);
    EXPECT_EQ(cache.stats().nativeCreateCount, 2);
    EXPECT_EQ(cache.stats().failedCreateCount, 1);
}

} // namespace
} // namespace metallic::tests
