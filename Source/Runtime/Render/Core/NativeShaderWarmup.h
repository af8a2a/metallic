#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {

struct NativeShaderWarmupReport
{
    size_t sourceCount = 0;
    size_t sourceCacheHits = 0;
    size_t sourceCompilerExecutions = 0;
    size_t computeSourceCount = 0;
    size_t nativeCreateCount = 0;
    size_t nativeObjectReuseCount = 0;
    size_t skippedGraphicsCount = 0;
    size_t skippedUnsupportedCount = 0;
    size_t skippedUnownedABICount = 0;
    size_t retainedPipelineCount = 0;
    uint64_t totalTimeNanoseconds = 0;
    uint64_t nativeCreateTimeNanoseconds = 0;
    uint64_t sourceAcquireTimeNanoseconds = 0;
    uint64_t sourceSnapshotTimeNanoseconds = 0;
    uint64_t sourceBoundaryValidationTimeNanoseconds = 0;
    uint64_t sourcePublicationTimeNanoseconds = 0;
    bool sourceBatchUsed = false;
    bool sourceEpochValidated = false;
};

// Uses this rendering Device's shader ABI and a fully validated source batch.
// Installed artifact overrides retain ordinary acquisition with a stable producer.
// Successful native objects remain pinned until Device teardown, independently
// of the bounded native-cache LRU. Unknown source ABIs are explicitly skipped.
[[nodiscard]] Result<NativeShaderWarmupReport> warmupNativeShaders(Device& device,
    std::span<const ShaderRequest> requests, std::string& log);

// Only a successful, enabled application SPIR-V startup warmup opens this gate.
// Standalone shader compilation and ordinary RHI/test devices do not opt in.
[[nodiscard]] Result<> warmupNativeShadersForStartup(Device& device);

} // namespace metallic::render
