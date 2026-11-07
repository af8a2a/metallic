#include "Runtime/Scene/MeshletBuildParallel.h"
#include "Runtime/Task/ParallelFor.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

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
    }

    const size_t workerCount;
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
    if (activePool == impl_.get() || task::isInsideTaskCallback()) {
        // The Meshlet callback can publish results in task-index order. A child
        // submission must finish inline instead of waiting on occupied workers.
        const size_t worker = activePool == impl_.get() ? activeWorker : 0;
        ActivePoolScope scope(impl_.get(), worker);
        for (size_t index = 0; index < taskCount; ++index) { task(index, worker); }
        return;
    }
    if (taskCount == 1) {
        ActivePoolScope scope(impl_.get(), 0);
        task(0, 0);
        return;
    }
    if (taskCount > UINT32_MAX || impl_->workerCount > UINT32_MAX) {
        throw std::length_error("Meshlet parallel range exceeds the uint32 index contract");
    }
    std::mutex failureMutex;
    std::exception_ptr failure;
    const auto outcome = task::parallelFor(0, static_cast<uint32_t>(taskCount),
        {.minGrain = 1, .maxWorkers = static_cast<uint32_t>(impl_->workerCount),
            .strategy = task::PartitionStrategy::Dynamic},
        [&](uint32_t begin, uint32_t end, task::WorkerContext& worker) {
            ActivePoolScope scope(impl_.get(), worker.workerIndex);
            try {
                for (uint32_t index = begin; index < end; ++index) { task(index, worker.workerIndex); }
            } catch (...) {
                std::lock_guard lock(failureMutex);
                if (!failure) { failure = std::current_exception(); }
                throw;
            }
        });
    if (failure) { std::rethrow_exception(failure); }
    if (!outcome) { throw std::runtime_error(outcome.error()); }
}

MeshletBuildParallel& MeshletBuildParallel::shared(size_t workerCount)
{
    static std::mutex adaptersMutex;
    static std::unordered_map<size_t, std::unique_ptr<MeshletBuildParallel>> adapters;
    workerCount = std::max<size_t>(1, workerCount);
    std::lock_guard lock(adaptersMutex);
    auto& adapter = adapters[workerCount];
    if (!adapter) { adapter = std::make_unique<MeshletBuildParallel>(workerCount); }
    return *adapter;
}

} // namespace metallic::scene
