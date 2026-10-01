#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace metallic::render {

enum class RenderGraphExecutionSnapshotStatus : uint8_t { Planned, Recording, Recorded, Submitted, Failed };
enum class RenderGraphSegmentRole : uint8_t { Pass, Prologue, ComputeBranch, GraphicsBranch, Join, Epilogue };

struct RenderGraphTextureAliasSlotMemoryStats {
    bool complete = false;
    uint64_t backingAllocationId = 0;
    uint64_t logicalBytes = 0;
    uint64_t backingBytes = 0;
    uint64_t savedBytes = 0;
    uint64_t overheadBytes = 0;
    std::vector<std::string> resources;
    bool operator==(const RenderGraphTextureAliasSlotMemoryStats&) const = default;
};

// Compile-generation graph-owned texture capacities, including native alignment.
// logicalBytes is the independent allocation counterfactual; alias members query
// their original descriptors without VK_IMAGE_CREATE_ALIAS_BIT. backingBytes
// counts each actual allocation owner once. Buffers, scene/SDK allocations,
// private imports, history and driver heap residency are outside this scope.
struct RenderGraphTextureMemoryStats {
    bool aliasingEnabled = false;
    bool complete = false;
    uint32_t textureCount = 0;
    uint32_t transientTextureCount = 0;
    uint32_t pinnedTextureCount = 0;
    // Native-qualified candidates; zero when aliasing is disabled (no query).
    uint32_t eligibleTextureCount = 0;
    uint32_t aliasedTextureCount = 0;
    uint32_t aliasSlotCount = 0;
    uint32_t backingAllocationCount = 0;
    uint32_t unknownTextureCount = 0;
    uint64_t logicalBytes = 0;
    uint64_t backingBytes = 0;
    uint64_t savedBytes = 0;
    uint64_t overheadBytes = 0;
    // Savings/overhead are zero when incomplete; partial capacities alone do
    // not establish a valid comparison. Slots list actual shared allocations.
    std::vector<RenderGraphTextureAliasSlotMemoryStats> slots;
    bool operator==(const RenderGraphTextureMemoryStats&) const = default;
};

using RenderGraphBufferAliasSlotMemoryStats = RenderGraphTextureAliasSlotMemoryStats;

// All graph-owned buffers, including independent host readbacks. Only Device
// buffers can alias. Native backing capacities are counted once per owner;
// private imports, history, scene/SDK allocations and heap residency are excluded.
struct RenderGraphBufferMemoryStats {
    bool aliasingEnabled = false;
    bool complete = false;
    uint32_t bufferCount = 0;
    uint32_t transientBufferCount = 0;
    uint32_t pinnedBufferCount = 0;
    uint32_t eligibleBufferCount = 0;
    uint32_t aliasedBufferCount = 0;
    uint32_t aliasSlotCount = 0;
    uint32_t backingAllocationCount = 0;
    uint32_t unknownBufferCount = 0;
    uint64_t logicalBytes = 0;
    uint64_t backingBytes = 0;
    uint64_t savedBytes = 0;
    uint64_t overheadBytes = 0;
    // Incomplete comparisons never report savings or overhead.
    std::vector<RenderGraphBufferAliasSlotMemoryStats> slots;
    bool operator==(const RenderGraphBufferMemoryStats&) const = default;
};

struct RenderGraphExecutionQueueSnapshot {
    uint32_t id = 0;
    QueueType type = QueueType::Graphics;
};

struct RenderGraphExecutionResourceSnapshot {
    uint64_t id = 0; // Allocation generation, or a dynamic AS slot generation.
    std::string name;
    std::vector<std::string> aliases;
    RenderGraphResourceType type = RenderGraphResourceType::Texture2D;
    TextureDesc textureDesc;
    BufferDesc bufferDesc;
    RayTracingAccelerationStructureDesc accelerationStructureDesc;
    ResourceMemoryInfo memory;
    bool privateResource = false;
};

struct RenderGraphExecutionUseSnapshot {
    uint64_t resourceId = 0;
    ResourceState state = ResourceState::Undefined;
    SyncScope scope;
    bool reads = false;
    bool writes = false;
    bool exclusive = false; // Planner hazard ownership, including internal layout changes.
};

struct RenderGraphExecutionBarrierSnapshot {
    uint64_t resourceId = 0;
    ResourceState before = ResourceState::Undefined;
    ResourceState after = ResourceState::Undefined;
    SyncScope beforeScope;
    SyncScope afterScope;
    bool executionOnly = false;
    bool memoryAliasing = false;
};

struct RenderGraphExecutionStageSnapshot {
    std::string name;
    RenderGraphPassKind kind = RenderGraphPassKind::Compute;
    std::vector<RenderGraphExecutionUseSnapshot> uses;
    std::vector<RenderGraphExecutionBarrierSnapshot> barriers;
    bool allowParallelCompute = false;
    bool recorded = false;
    bool restoreBoundary = false;
    SynchronizationStats synchronization; // Encoded planner boundary only, not opaque SDK internals.
};

struct RenderGraphExecutionPassSnapshot {
    uint32_t id = 0;
    std::string name;
    std::string type;
    RenderGraphPassKind kind = RenderGraphPassKind::Unsafe;
    QueueType logicalQueue = QueueType::Graphics;
    uint32_t actualQueueId = UINT32_MAX; // Unknown for external command recording.
    std::vector<RenderGraphExecutionUseSnapshot> uses;
    std::vector<RenderGraphExecutionBarrierSnapshot> barriers;
    std::vector<uint32_t> predecessors; // Pass IDs, including dependencies with reused RAW visibility.
    std::vector<RenderGraphExecutionStageSnapshot> stages;
    bool recorded = false;
    SynchronizationStats synchronization;
};

struct RenderGraphExecutionSegmentSnapshot {
    uint32_t id = 0;
    uint32_t passId = UINT32_MAX;
    uint32_t queueId = UINT32_MAX;
    RenderGraphSegmentRole role = RenderGraphSegmentRole::Pass;
    std::vector<uint32_t> predecessors; // Recording dependencies; same-queue edges need no semaphore.
    bool recorded = false;
    bool accepted = false;
    bool completionKnown = false; // An accepted submission has a trackable completion point.
    bool completed = false; // Observed frame completion; no per-segment GPU timing is inferred.
};

struct RenderGraphExecutionBatchSnapshot {
    uint32_t id = 0;
    uint32_t queueId = UINT32_MAX;
    std::vector<uint32_t> segmentIds;
    std::vector<uint32_t> waitPredecessors; // Only actual cross-queue producer segments.
    uint32_t externalWaitCount = 0;
    uint32_t semaphoreWaitCount = 0; // Explicit executor waits, before frame/command dependency merging.
    bool waitDetailsComplete = false; // Opaque command dependencies are not inspected by this capture.
    bool accepted = false;
};

// Immutable diagnostic values. Enabling capture does not attach a debug observer,
// change scheduling, wait for the GPU, or retain GPU resources in the UI.
struct RenderGraphExecutionSnapshot {
    uint64_t executionId = 0;
    uint64_t graphGeneration = 0;
    std::string graphName;
    RenderGraphExecutionSnapshotStatus status = RenderGraphExecutionSnapshotStatus::Planned;
    bool externalRecording = false;
    bool success = false;
    bool pipelinedSubmission = false;
    std::vector<RenderGraphExecutionQueueSnapshot> queues;
    std::vector<RenderGraphExecutionResourceSnapshot> resources;
    std::vector<RenderGraphExecutionPassSnapshot> passes;
    std::vector<RenderGraphExecutionSegmentSnapshot> segments;
    std::vector<RenderGraphExecutionBatchSnapshot> batches;
    RenderGraphTextureMemoryStats textureMemory;
    RenderGraphBufferMemoryStats bufferMemory;
};

} // namespace metallic::render
