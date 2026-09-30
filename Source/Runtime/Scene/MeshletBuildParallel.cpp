#include "Runtime/Scene/MeshletBuildParallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace metallic::scene {
namespace {

thread_local const void* activePool = nullptr;
thread_local size_t activeWorker = 0;

struct ActivePoolScope {
    ActivePoolScope(const void* pool, size_t worker) : previousPool(activePool), previousWorker(activeWorker)
    {
        activePool = pool;
        activeWorker = worker;
    }

    ~ActivePoolScope()
    {
        activePool = previousPool;
        activeWorker = previousWorker;
    }

    const void* previousPool;
    size_t previousWorker;
};

} // namespace

struct MeshletBuildParallel::Impl {
    explicit Impl(size_t count) : workerCount(std::max<size_t>(1, count))
    {
        workers.reserve(workerCount - 1);
        try {
            for (size_t worker = 1; worker < workerCount; ++worker) {
                workers.emplace_back([this, worker] {
                    size_t observedEpoch = 0;
                    for (;;) {
                        std::unique_lock lock(workMutex);
                        workChanged.wait(lock, [&] { return stopping || epoch != observedEpoch; });
                        if (stopping) { return; }
                        observedEpoch = epoch;
                        lock.unlock();
                        runTasks(worker);
                        lock.lock();
                        if (--remainingWorkers == 0) { finished.notify_one(); }
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~Impl()
    {
        stop();
    }

    void stop()
    {
        {
            std::lock_guard lock(workMutex);
            stopping = true;
        }
        workChanged.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) { worker.join(); }
        }
    }

    void runTasks(size_t workerIndex)
    {
        ActivePoolScope scope(this, workerIndex);
        try {
            for (;;) {
                const size_t taskIndex = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIndex >= taskCount) { break; }
                task(taskIndex, workerIndex);
            }
        } catch (...) {
            std::lock_guard lock(workMutex);
            if (!failure) { failure = std::current_exception(); }
            nextTask.store(taskCount, std::memory_order_relaxed);
        }
    }

    const size_t workerCount;
    std::mutex submissionMutex;
    std::mutex workMutex;
    std::condition_variable workChanged;
    std::condition_variable finished;
    std::atomic_size_t nextTask{0};
    size_t taskCount = 0;
    size_t epoch = 0;
    size_t remainingWorkers = 0;
    bool stopping = false;
    Task task;
    std::exception_ptr failure;
    std::vector<std::thread> workers;
};

MeshletBuildParallel::MeshletBuildParallel(size_t workerCount)
    : impl_(std::make_unique<Impl>(workerCount))
{
}

MeshletBuildParallel::~MeshletBuildParallel() = default;

size_t MeshletBuildParallel::workerCount() const noexcept
{
    return impl_->workerCount;
}

void MeshletBuildParallel::forEach(size_t taskCount, const Task& task)
{
    if (taskCount == 0) { return; }
    if (activePool == impl_.get()) {
        // A nested submission must not wait for the worker currently making it.
        for (size_t index = 0; index < taskCount; ++index) { task(index, activeWorker); }
        return;
    }
    std::lock_guard submissionLock(impl_->submissionMutex);
    if (taskCount == 1 || impl_->workerCount == 1) {
        ActivePoolScope scope(impl_.get(), 0);
        for (size_t index = 0; index < taskCount; ++index) { task(index, 0); }
        return;
    }
    {
        std::lock_guard lock(impl_->workMutex);
        impl_->task = task;
        impl_->taskCount = taskCount;
        impl_->nextTask.store(0, std::memory_order_relaxed);
        impl_->failure = nullptr;
        impl_->remainingWorkers = impl_->workerCount - 1;
        ++impl_->epoch;
    }
    impl_->workChanged.notify_all();
    impl_->runTasks(0);
    std::unique_lock lock(impl_->workMutex);
    impl_->finished.wait(lock, [&] { return impl_->remainingWorkers == 0; });
    const std::exception_ptr failure = impl_->failure;
    impl_->task = {};
    lock.unlock();
    if (failure) { std::rethrow_exception(failure); }
}

MeshletBuildParallel& MeshletBuildParallel::shared(size_t workerCount)
{
    static std::mutex poolsMutex;
    static std::unordered_map<size_t, std::unique_ptr<MeshletBuildParallel>> pools;
    workerCount = std::max<size_t>(1, workerCount);
    std::lock_guard lock(poolsMutex);
    auto& pool = pools[workerCount];
    if (!pool) { pool = std::make_unique<MeshletBuildParallel>(workerCount); }
    return *pool;
}

} // namespace metallic::scene
