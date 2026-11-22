#pragma once

#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace metallic::render {

enum class RenderGraphResourceType : uint8_t;

enum class RenderGraphResourceOwnership : uint8_t {
    GraphOwned,
    TemporalOwned,
    ProviderOwned,
    ExternallyOwned,
    OpaqueSDK,
};

enum class RenderGraphMemoryPolicy : uint8_t {
    // Default conservative mode: reuse minimal physical backings with GPU waits.
    MinimizeMemory,
    // Opt-in budgeted multi-version pooling when an eligible physical backing is
    // still in flight on the GPU. Never eliminates algorithmic RAW dependencies.
    MaximizeOverlap,
};

const char* renderGraphResourceOwnershipName(RenderGraphResourceOwnership ownership);
const char* renderGraphMemoryPolicyName(RenderGraphMemoryPolicy policy);

// Stable cross-frame identity for a logical resource. Separates owner domain,
// logical name, view identity (0 = shared across views), and content identity
// (0 = mutable logical resource, non-zero = content-addressed immutable data).
struct LogicalResourceId {
    uint64_t value = 0;
    uint64_t viewId = 0;
    uint64_t contentKey = 0;
    std::string ownerDomain;
    std::string name;

    bool valid() const { return value != 0; }
    bool operator==(const LogicalResourceId& other) const { return value == other.value; }
};

LogicalResourceId makeLogicalResourceId(
    std::string_view ownerDomain,
    std::string_view name,
    uint64_t viewId = 0,
    uint64_t contentKey = 0);

// Separates logical production version, semantic content revision, and physical
// backing allocation generation. Rebuilding an evicted allocation for unchanged
// content preserves contentRevision while advancing allocationGeneration.
struct ResourceVersionId {
    LogicalResourceId logicalId;
    uint64_t version = 0;
    uint64_t contentRevision = 0;
    uint64_t allocationGeneration = 0;

    bool valid() const { return logicalId.valid() && version != 0; }
    bool operator==(const ResourceVersionId& other) const
    {
        return logicalId == other.logicalId &&
            version == other.version &&
            contentRevision == other.contentRevision &&
            allocationGeneration == other.allocationGeneration;
    }
};

// Externally-owned or provider-published physical backing with explicit state,
// queue ownership, and producer completion. RenderGraph never aliases, copies,
// or relocates externally-owned native/SDK allocations.
struct ExternalResourcePublication {
    LogicalResourceId logicalId;
    ResourceVersionId version;
    RenderGraphResourceType type{};
    RenderGraphResourceOwnership ownership = RenderGraphResourceOwnership::ExternallyOwned;
    Texture* texture = nullptr;
    TextureView* textureView = nullptr;
    TextureSubresourceRange textureRange{};
    Buffer* buffer = nullptr;
    BufferView* bufferView = nullptr;
    BufferRange bufferRange{};
    RayTracingAccelerationStructure* accelerationStructure = nullptr;
    std::shared_ptr<void> owner;
    ResourceState initialState = ResourceState::Undefined;
    ResourceState finalState = ResourceState::Undefined;
    Queue* ownerQueue = nullptr;
    GPUCompletionPoint producerCompletion;
    uint64_t backingAllocationId = 0;
    uint64_t sizeBytes = 0;
    // Conservative boundary for #60: exclusive queue-family ownership transfers
    // are not yet auto-emitted across families and fall back conservatively.
    bool exclusiveQueueOwnership = false;
    bool supportsCrossFrameReadSharing = true;
    bool needsHostWriteBarrier = false;
};

// Explicit contract for external/native/SDK domains (e.g. GPUScene bindless/BDA,
// VSM, Radiance Cache, NRD, Streamline). Undeclared implicit accesses remain
// strictly conservative and report their reason in execution diagnostics.
struct RenderGraphExternalDomainContract {
    std::string domainName;
    RenderGraphResourceOwnership ownership = RenderGraphResourceOwnership::ExternallyOwned;
    std::vector<ExternalResourcePublication> declaredResources;
    bool hasUndeclaredImplicitAccesses = true;
    std::string conservativeReason;
};

enum class FrontierEntryStatus : uint8_t {
    Recording,
    Accepted,
    Complete,
};

struct ResourceFrontierAccess {
    uint64_t frameIndex = 0;
    Queue* queue = nullptr;
    SyncScope scope;
    GPUCompletionPoint completion;
    FrontierEntryStatus status = FrontierEntryStatus::Recording;
    std::string passName;

    bool isActive() const;
};

// Per-physical-backing cross-frame hazard frontier. Tracks the last writer,
// layout anchor, and ALL overlapping readers (never just the last reader).
struct ResourceAccessFrontier {
    uint64_t backingAllocationId = 0;
    LogicalResourceId logicalId;
    ResourceVersionId version;
    RenderGraphResourceType type{};
    ResourceState state = ResourceState::Undefined;
    bool exclusiveQueueOwnership = false;
    std::optional<ResourceFrontierAccess> lastWriter;
    std::optional<ResourceFrontierAccess> layoutAnchor;
    std::vector<ResourceFrontierAccess> outstandingReaders;

    void pruneCompleted();
    bool isPhysicalBackingIdle() const;
    size_t activeReaderCount() const;
    std::vector<GPUCompletionPoint> activeCompletions() const;

    void collectWaitsForRead(
        Queue* consumerQueue,
        bool layoutChange,
        std::vector<GPUCompletionPoint>& outWaits,
        std::vector<std::string>* outReasons = nullptr) const;
    void collectWaitsForWrite(
        Queue* writerQueue,
        std::vector<GPUCompletionPoint>& outWaits,
        std::vector<std::string>* outReasons = nullptr) const;
};

struct CrossFrameAccessPlanResult {
    ResourceState initialState = ResourceState::Undefined;
    SyncScope initialScope;
    std::vector<GPUCompletionPoint> waits;
    std::vector<std::string> waitReasons;
    bool requiresConservativeFallback = false;
    std::string conservativeReason;
};

struct AcceptedCrossFrameAccess {
    uint64_t backingAllocationId = 0;
    LogicalResourceId logicalId;
    ResourceVersionId version;
    RenderGraphResourceType type{};
    uint64_t frameIndex = 0;
    Queue* queue = nullptr;
    ResourceState state = ResourceState::Undefined;
    SyncScope scope;
    bool writes = false;
    bool layoutChanged = false;
    bool exclusiveQueueOwnership = false;
    GPUCompletionPoint completion;
    std::string passName;
};

// Coordinator-owned ledger of accepted cross-frame physical backing frontiers.
// Updated ONLY upon queue acceptance (SubmissionReceipt), never during CPU
// planning/recording and never for cancelled or rejected submissions.
class CrossFrameResourceTracker {
public:
    void clear();
    void pruneCompleted();

    [[nodiscard]] CrossFrameAccessPlanResult planAccess(
        uint64_t backingAllocationId,
        std::string_view debugName,
        RenderGraphResourceType type,
        Queue* targetQueue,
        ResourceState nextState,
        bool writes,
        bool exclusiveQueueOwnership = false) const;

    void commitAcceptedAccess(const AcceptedCrossFrameAccess& access);
    void forgetBacking(uint64_t backingAllocationId);

    bool isBackingIdle(uint64_t backingAllocationId);
    bool isBackingIdle(uint64_t backingAllocationId) const;
    std::vector<GPUCompletionPoint> activeCompletions(uint64_t backingAllocationId) const;
    const ResourceAccessFrontier* findFrontier(uint64_t backingAllocationId) const;
    size_t trackedBackingCount() const { return frontiers_.size(); }

private:
    std::unordered_map<uint64_t, ResourceAccessFrontier> frontiers_;
};

// Immutable per-execution logical -> version -> physical binding entry frozen
// before parallel or serial pass recording begins.
struct ResolvedLogicalBinding {
    LogicalResourceId logicalId;
    ResourceVersionId version;
    RenderGraphResourceOwnership ownership = RenderGraphResourceOwnership::GraphOwned;
    RenderGraphResourceType type{};
    Texture* texture = nullptr;
    TextureView* textureView = nullptr;
    Buffer* buffer = nullptr;
    BufferView* bufferView = nullptr;
    RayTracingAccelerationStructure* accelerationStructure = nullptr;
    uint64_t backingAllocationId = 0;
    uint64_t sizeBytes = 0;
    uint32_t physicalVersionCount = 1;
    bool extraPhysicalVersion = false;
    bool temporalValid = false;
    uint64_t invalidationRevision = 0;
    GPUCompletionPoint producerCompletion;
    std::shared_ptr<void> owner;
    std::string producerName;
};

// Frozen per-execution resource snapshot. Once freeze() is called, subsequent
// beginFrame(N+1) or cache evictions cannot mutate or release Frame N's bindings.
class FrameExecutionResourceSnapshot {
public:
    uint64_t frameIndex() const { return frameIndex_; }
    uint64_t executionGeneration() const { return executionGeneration_; }
    bool frozen() const { return frozen_; }

    void beginBuild(uint64_t frameIndex, uint64_t executionGeneration);
    Result<> bindLogicalResource(std::string_view key, ResolvedLogicalBinding binding);
    Result<> bindResourceSetRaw(
        std::type_index type,
        std::shared_ptr<const void> data,
        std::shared_ptr<void> owner,
        LogicalResourceId setId,
        ResourceVersionId version);

    template <typename T>
    Result<> bindResourceSet(
        std::shared_ptr<const T> data,
        std::shared_ptr<void> owner,
        LogicalResourceId setId,
        ResourceVersionId version)
    {
        return bindResourceSetRaw(
            std::type_index(typeid(T)),
            std::static_pointer_cast<const void>(std::move(data)),
            std::move(owner),
            std::move(setId),
            std::move(version));
    }

    Result<> retainOwner(std::shared_ptr<void> owner);
    void freeze();

    const ResolvedLogicalBinding* findLogicalResource(std::string_view key) const;

    template <typename T>
    const T* resourceSet() const
    {
        return static_cast<const T*>(findResourceSetRaw(std::type_index(typeid(T))));
    }

    const std::vector<ResolvedLogicalBinding>& allBindings() const { return bindings_; }

private:
    struct ResourceSetEntry {
        LogicalResourceId setId;
        ResourceVersionId version;
        std::shared_ptr<const void> data;
        std::shared_ptr<void> owner;
    };

    const void* findResourceSetRaw(std::type_index type) const;

    uint64_t frameIndex_ = 0;
    uint64_t executionGeneration_ = 0;
    bool frozen_ = false;
    std::vector<ResolvedLogicalBinding> bindings_;
    std::unordered_map<std::string, size_t> bindingIndex_;
    std::unordered_map<std::type_index, ResourceSetEntry> resourceSets_;
    std::vector<std::shared_ptr<void>> retainedOwners_;
};

// Conditional hidden Producer work registered into RenderGraph execution by a
// subsystem Resource Provider. Never forces graph template recompilation.
struct RenderGraphProviderProducerWork {
    std::string debugName;
    RenderSubsystemId providerId;
    uint64_t dedupKey = 0;
    QueueType preferredQueue = QueueType::Graphics;
    std::vector<LogicalResourceId> producedResources;
    std::vector<ExternalResourcePublication> publications;
    // External completion dependencies required before this producer runs
    // (for example, shared static medium LUTs produced by an earlier submission).
    std::vector<GPUCompletionPoint> waitCompletions;
    // Same-frame prerequisite provider producers (for example, another view's
    // producer that computes shared static medium LUTs in the current frame).
    std::vector<uint64_t> prerequisiteDedupKeys;
    std::function<Result<>(CommandBuffer& commands, std::string& log)> record;
    std::function<void(const GPUCompletionPoint& completion)> onAccepted;
    std::function<void()> onCancelled;
};

// Optional extension interface for subsystems that act as RenderGraph Resource
// Providers. Does not break IRenderSubsystem compatibility.
class IRenderGraphResourceProvider {
public:
    virtual ~IRenderGraphResourceProvider() = default;
    virtual RenderSubsystemId providerSubsystemId() const = 0;
    virtual std::optional<RenderGraphExternalDomainContract> externalDomainContract() const
    {
        return std::nullopt;
    }
};

// Budgeted physical version pool for eligible graph-owned textures and buffers.
// Honors MinimizeMemory (default) and MaximizeOverlap within a byte budget,
// retiring replaced backings against all active frontier completions.
class RenderGraphResourceVersionPool {
public:
    struct Policy {
        RenderGraphMemoryPolicy mode = RenderGraphMemoryPolicy::MinimizeMemory;
        uint64_t extraVersionBudgetBytes = 0;
        uint32_t maxPhysicalVersionsPerResource = 2;
    };

    struct AcquiredTexture {
        Texture* texture = nullptr;
        TextureView* view = nullptr;
        uint64_t backingAllocationId = 0;
        uint64_t allocationGeneration = 0;
        uint64_t sizeBytes = 0;
        uint32_t physicalVersionCount = 1;
        bool extraVersion = false;
        ResourceState state = ResourceState::Undefined;
        std::shared_ptr<void> owner;
    };

    struct AcquiredBuffer {
        Buffer* buffer = nullptr;
        BufferView* view = nullptr;
        uint64_t backingAllocationId = 0;
        uint64_t allocationGeneration = 0;
        uint64_t sizeBytes = 0;
        uint32_t physicalVersionCount = 1;
        bool extraVersion = false;
        ResourceState state = ResourceState::Undefined;
        std::shared_ptr<void> owner;
    };

    struct Stats {
        uint32_t pooledLogicalResources = 0;
        uint32_t totalPhysicalVersions = 0;
        uint32_t extraVersionCount = 0;
        uint64_t totalManagedBytes = 0;
        uint64_t extraVersionBytes = 0;
        uint64_t peakManagedBytes = 0;
    };

    void setPolicy(Policy policy) { policy_ = policy; }
    const Policy& policy() const { return policy_; }

    Result<AcquiredTexture> acquireTexture(
        Device& device,
        uint64_t logicalId,
        const TextureDesc& desc,
        const TextureViewDesc& viewDesc,
        uint64_t frameIndex,
        const CrossFrameResourceTracker& tracker,
        DeferredReleaseQueue& deferredReleases);

    Result<AcquiredBuffer> acquireBuffer(
        Device& device,
        uint64_t logicalId,
        const BufferDesc& desc,
        const BufferViewDesc* viewDesc,
        uint64_t frameIndex,
        const CrossFrameResourceTracker& tracker,
        DeferredReleaseQueue& deferredReleases);

    void updateBackingState(uint64_t backingAllocationId, ResourceState state);
    void retireAll(const CrossFrameResourceTracker& tracker, DeferredReleaseQueue& deferredReleases);
    void clear();
    Stats stats() const;

private:
    struct TextureBacking {
        std::shared_ptr<Texture> texture;
        std::shared_ptr<TextureView> view;
        uint64_t backingAllocationId = 0;
        uint64_t allocationGeneration = 0;
        uint64_t sizeBytes = 0;
        uint64_t lastUsedFrame = 0;
        ResourceState state = ResourceState::Undefined;
    };

    struct BufferBacking {
        std::shared_ptr<Buffer> buffer;
        std::shared_ptr<BufferView> view;
        uint64_t backingAllocationId = 0;
        uint64_t allocationGeneration = 0;
        uint64_t sizeBytes = 0;
        uint64_t lastUsedFrame = 0;
        ResourceState state = ResourceState::Undefined;
    };

    struct TexturePoolEntry {
        TextureDesc desc{};
        TextureViewDesc viewDesc{};
        std::vector<TextureBacking> backings;
    };

    struct BufferPoolEntry {
        BufferDesc desc{};
        std::optional<BufferViewDesc> viewDesc;
        std::vector<BufferBacking> backings;
    };

    Policy policy_;
    uint64_t nextAllocationGeneration_ = 1;
    uint64_t peakManagedBytes_ = 0;
    std::unordered_map<uint64_t, TexturePoolEntry> textures_;
    std::unordered_map<uint64_t, BufferPoolEntry> buffers_;

    void updatePeakManagedBytes();
    uint64_t currentExtraBytes() const;
    uint64_t currentTotalBytes() const;
};

} // namespace metallic::render
