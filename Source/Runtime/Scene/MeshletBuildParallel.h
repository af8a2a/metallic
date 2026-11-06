#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace metallic::scene {

// Uses the shared TaskSystem workers across LOD levels and primitive builds.
// Worker indices are dense, exclusive scratch slots within an invocation;
// tasks may finish in any order. Slots do not imply OS thread affinity.
class MeshletBuildParallel {
public:
    using Task = std::function<void(size_t taskIndex, size_t workerIndex)>;

    explicit MeshletBuildParallel(size_t workerCount);
    ~MeshletBuildParallel();
    MeshletBuildParallel(const MeshletBuildParallel&) = delete;
    MeshletBuildParallel& operator=(const MeshletBuildParallel&) = delete;

    size_t workerCount() const noexcept;
    void forEach(size_t taskCount, const Task& task);

    // Lightweight adapters are retained by worker limit; they own no threads.
    static MeshletBuildParallel& shared(size_t workerCount);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::scene
