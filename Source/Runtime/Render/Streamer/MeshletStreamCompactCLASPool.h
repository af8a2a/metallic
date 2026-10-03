#pragma once
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"

namespace metallic::render {

struct CPUProfileRecorder;
// Optional compact backend. The existing pool remains the temporary builder and
// the compatibility path; this backend only owns relocation and publication.
class MeshletStreamCompactCLASPool {
  public:
    MeshletStreamCompactCLASPool();
    ~MeshletStreamCompactCLASPool();
    Result<> initialize(Device&, const MeshletStreamCLASPoolDesc&, std::string&);
    void beginFrame(CPUProfileRecorder* profiler = nullptr);
    Result<> cmdBuildPages(CommandBuffer&, Buffer&, std::span<const MeshletStreamCLASPageBuild>, std::string&);
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
    MeshletStreamCLASPoolStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace metallic::render
