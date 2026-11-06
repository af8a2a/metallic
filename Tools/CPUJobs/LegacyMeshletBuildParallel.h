#pragma once

// Benchmark-only snapshot of MeshletBuildParallel at 8e0d94587.

#include <cstddef>
#include <functional>
#include <memory>

namespace metallic::bench {

// Reuses workers across LOD levels and primitive builds. Worker indices are
// stable, dense scratch-buffer indices; tasks may finish in any order.
class MeshletBuildParallel {
public:
    using Task = std::function<void(size_t taskIndex, size_t workerIndex)>;

    explicit MeshletBuildParallel(size_t workerCount);
    ~MeshletBuildParallel();
    MeshletBuildParallel(const MeshletBuildParallel&) = delete;
    MeshletBuildParallel& operator=(const MeshletBuildParallel&) = delete;

    size_t workerCount() const noexcept;
    void forEach(size_t taskCount, const Task& task);

    // Pools are retained for subsequent geometry builds with the same limit.
    static MeshletBuildParallel& shared(size_t workerCount);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::bench
