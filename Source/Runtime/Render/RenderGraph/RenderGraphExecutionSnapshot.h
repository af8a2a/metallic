#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace metallic::render {

enum class RenderGraphExecutionSnapshotStatus : uint8_t { Planned, Recording, Recorded, Submitted, Failed };
enum class RenderGraphSegmentRole : uint8_t { Pass, Prologue, ComputeBranch, GraphicsBranch, Join, Epilogue };

struct RenderGraphExecutionQueueSnapshot {
    uint32_t id = 0;
    QueueType type = QueueType::Graphics;
};

struct RenderGraphExecutionResourceSnapshot {
    uint64_t id = 0; // Allocation generation, not a pointer or physical memory block.
    std::string name;
    std::vector<std::string> aliases;
    RenderGraphResourceType type = RenderGraphResourceType::Texture2D;
    TextureDesc textureDesc;
    BufferDesc bufferDesc;
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
};

} // namespace metallic::render
