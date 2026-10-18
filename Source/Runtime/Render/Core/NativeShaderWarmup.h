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
    size_t nativeTaskCount = 0;
    size_t nativeWorkerCount = 0;
    uint64_t totalTimeNanoseconds = 0;
    // Wall time for the parallel native pipeline phase; creation time below is
    // cumulative across tasks and can exceed wall time.
    uint64_t nativePipelineTimeNanoseconds = 0;
    uint64_t nativeCreateTimeNanoseconds = 0;
    uint64_t sourceAcquireTimeNanoseconds = 0;
    uint64_t sourceSnapshotTimeNanoseconds = 0;
    uint64_t sourceBoundaryValidationTimeNanoseconds = 0;
    uint64_t sourcePublicationTimeNanoseconds = 0;
    bool sourceBatchUsed = false;
    bool sourceEpochValidated = false;
    bool pipelined = false;
    size_t nativeTasksBeforeSourceAcceptance = 0;
};

// Uses this rendering Device's shader ABI and a fully validated source batch.
// Installed artifact overrides retain ordinary acquisition with a stable producer.
// Successful native objects remain pinned until Device teardown, independently
// of the bounded native-cache LRU. Unknown source ABIs are explicitly skipped.
[[nodiscard]] Result<NativeShaderWarmupReport> warmupNativeShaders(Device& device,
    std::span<const ShaderRequest> requests, std::string& log);

// On a fresh device, overlap provisional source results with isolated native
// compilation. Publish native objects only after the complete source batch is
// accepted. Existing device caches and diagnostic backends use the batch path.
[[nodiscard]] Result<NativeShaderWarmupReport> warmupNativeShadersPipelined(Device& device,
    std::span<const ShaderRequest> requests, std::string& log);

// Application startup preparation opens this gate; device creation returns
// successfully only after the complete source/native warmup is accepted.
// Standalone shader compilation and ordinary RHI/test devices do not opt in.
[[nodiscard]] Result<> warmupNativeShadersForStartup(Device& device);

} // namespace metallic::render
