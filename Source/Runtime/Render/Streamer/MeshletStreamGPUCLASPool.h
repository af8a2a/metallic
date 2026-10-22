#pragma once

#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"

namespace metallic::render {

// GPU owns compact offsets, the free-gap topology and the authoritative page
// publication. CPU observations are delayed statistics, never build inputs.
class MeshletStreamGPUCLASPool {
public:
    MeshletStreamGPUCLASPool();
    ~MeshletStreamGPUCLASPool();
    Result<> initialize(Device&, const MeshletStreamCLASPoolDesc&, std::string&);
    void beginFrame(CPUProfileRecorder* profiler = nullptr);
    Result<> cmdBuildPages(CommandBuffer&, Buffer&, std::span<const MeshletStreamCLASPageBuild>, std::string&,
        const std::function<void(std::string_view)>& checkpoint = {});
    void retirePages(std::span<const uint32_t>);
    bool ready() const;
    bool pageHasClas(uint32_t) const;
    bool pageBuildPending(uint32_t) const;
    uint64_t pageStorageBytes(uint32_t) const;
    uint32_t pageClasAddressOffset(uint32_t) const;
    uint64_t clusterAddress(uint32_t, uint32_t) const;
    Buffer* pageStorageBuffer(uint32_t) const;
    Buffer* clusterAddressBuffer() const;
    Buffer* pageTableBuffer() const;
    uint64_t publicationRevision() const;
    MeshletStreamCLASPoolStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
