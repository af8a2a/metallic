#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/scene.h"

#include <array>
#include <cstddef>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace metallic::render {

template <typename Tag>
struct GPUSceneId {
    uint32_t index = std::numeric_limits<uint32_t>::max();
    uint32_t generation = 0;

    bool valid() const
    {
        return index != std::numeric_limits<uint32_t>::max() && generation != 0;
    }

    explicit operator bool() const { return valid(); }
    auto operator<=>(const GPUSceneId&) const = default;
};

struct GPUSceneGeometryTag;
struct GPUSceneMaterialTag;
struct GPUSceneInstanceTag;
struct GPUSceneViewTag;

using GPUSceneGeometryId = GPUSceneId<GPUSceneGeometryTag>;
using GPUSceneMaterialId = GPUSceneId<GPUSceneMaterialTag>;
using GPUSceneInstanceId = GPUSceneId<GPUSceneInstanceTag>;
using GPUSceneViewId = GPUSceneId<GPUSceneViewTag>;

enum class GPUSceneDrawBucket : uint8_t {
    OpaqueSingleSided,
    OpaqueDoubleSided,
    MaskedSingleSided,
    MaskedDoubleSided,
    Blend,
    Count,
};

inline constexpr size_t kGPUSceneDrawBucketCount =
    static_cast<size_t>(GPUSceneDrawBucket::Count);
inline constexpr size_t kGPUSceneRasterDrawBucketCount =
    static_cast<size_t>(GPUSceneDrawBucket::Blend);

const char* gpuSceneDrawBucketName(GPUSceneDrawBucket bucket);
GPUSceneDrawBucket classifyGPUSceneMaterial(const scene::RenderMaterial& material);

struct GPUSceneDrawKey {
    GPUSceneDrawBucket bucket = GPUSceneDrawBucket::OpaqueSingleSided;
    GPUSceneMaterialId material;
    GPUSceneGeometryId geometry;

    auto operator<=>(const GPUSceneDrawKey&) const = default;
};

// Fixed C++/Slang ABI records uploaded by GPUSceneSubsystem. Keep these records
// free of compiler-specific vector and matrix types so their layout is stable.
inline constexpr uint32_t kGPUSceneMaterialTextureSlotCount = 9;

enum GPUSceneGpuInstanceFlags : uint32_t {
    GPUSceneGpuInstanceVisible = 1u << 0,
    GPUSceneGpuInstanceDoubleSided = 1u << 1,
    GPUSceneGpuInstanceMasked = 1u << 2,
    GPUSceneGpuInstanceBlend = 1u << 3,
};

struct alignas(16) GPUSceneGpuGeometryRecord {
    // sourceRenderPrimitiveIndex, meshIndex, primitiveIndex, primitive mode.
    std::array<uint32_t, 4> source{};
    // vertexCount, indexCount, triangleCount, meshlet LOD level count.
    std::array<uint32_t, 4> counts{};
    // vertexOffset, indexOffset, base meshlet offset/count.
    std::array<uint32_t, 4> payload{};
    // meshletVertexOffset/count and meshletTriangleWordOffset/count across all
    // base+LOD meshlets.
    // Vertex references within the span are geometry-local and must be added
    // to payload.x by the raster consumer.
    std::array<uint32_t, 4> meshletPayload{};
    std::array<float, 4> localBoundingSphere{};
    // geometry ID index/generation and payload fingerprint low/high words.
    std::array<uint32_t, 4> identity{};
};

struct alignas(16) GPUSceneGpuMaterialTextureInfo {
    // Index into GPUSceneGpuDescriptorRemapRecord, not a consumer descriptor.
    uint32_t textureIndex = std::numeric_limits<uint32_t>::max();
    uint32_t texCoord = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    // Row-major 2x3 UV transform, padded to two float4 rows.
    std::array<float, 4> transform0{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> transform1{0.0f, 1.0f, 0.0f, 0.0f};
};

struct alignas(16) GPUSceneGpuMaterialRecord {
    // The first 544 bytes intentionally match GPUDrivenPreviewMaterial.
    std::array<float, 4> baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> emissive{};
    // metallic, roughness, alpha cutoff, double-sided.
    std::array<float, 4> params{1.0f, 1.0f, 0.5f, 0.0f};
    // normal scale, occlusion strength, reserved, reserved.
    std::array<float, 4> textureParams{1.0f, 1.0f, 0.0f, 0.0f};
    // transmission, ior, thickness, attenuation distance.
    std::array<float, 4> glassParams{0.0f, 1.5f, 0.0f, 0.0f};
    std::array<float, 4> attenuationColor{1.0f, 1.0f, 1.0f, 0.0f};
    // diffuse-transmission color rgb and factor.
    std::array<float, 4> diffuseTransmission{1.0f, 1.0f, 1.0f, 0.0f};
    GPUSceneGpuMaterialTextureInfo baseColorTexture;
    GPUSceneGpuMaterialTextureInfo metallicRoughnessTexture;
    GPUSceneGpuMaterialTextureInfo normalTexture;
    GPUSceneGpuMaterialTextureInfo occlusionTexture;
    GPUSceneGpuMaterialTextureInfo emissiveTexture;
    GPUSceneGpuMaterialTextureInfo transmissionTexture;
    GPUSceneGpuMaterialTextureInfo thicknessTexture;
    GPUSceneGpuMaterialTextureInfo diffuseTransmissionTexture;
    GPUSceneGpuMaterialTextureInfo diffuseTransmissionColorTexture;
    // material ID index/generation, source material index, material flags.
    std::array<uint32_t, 4> identity{};
};

struct alignas(16) GPUSceneGpuInstanceRecord {
    std::array<float, 16> worldMatrix{};
    std::array<float, 16> previousWorldMatrix{};
    std::array<float, 4> localBoundingSphere{};
    // geometry index, material index, source RenderNode index, instance flags.
    std::array<uint32_t, 4> identity{};
};

struct alignas(16) GPUSceneGpuDrawKeyRecord {
    // bucket, material index, geometry index, first DrawSet instance index.
    std::array<uint32_t, 4> key{};
    // instance count, material generation, geometry generation, reserved.
    std::array<uint32_t, 4> range{};
};

struct alignas(16) GPUSceneGpuVertexRecord {
    std::array<float, 4> position{};
    std::array<float, 4> normal{};
    std::array<float, 4> tangent{};
    std::array<float, 2> texcoord{};
    uint32_t flags = 0;
    uint32_t reserved = 0;
};

struct alignas(16) GPUSceneGpuMeshletRecord {
    // Global meshlet-vertex word offset/count and packed-triangle word
    // offset/triangle count. Meshlet vertex values remain geometry-local.
    std::array<uint32_t, 4> ranges{};
    // lodLevel, lodGroupIndex, reserved, reserved.
    std::array<uint32_t, 4> lod{};
    std::array<float, 4> boundingSphere{};
    // cone apex xyz and cone cutoff.
    std::array<float, 4> coneApexCutoff{};
    // cone axis xyz and LOD error.
    std::array<float, 4> coneAxisLodError{};
};

struct alignas(16) GPUSceneGpuMeshletDrawRecord {
    uint32_t meshletIndex = 0;
    uint32_t instanceIndex = 0;
    uint32_t geometryIndex = 0;
    uint32_t drawBucket = 0;
};

struct alignas(16) GPUSceneGpuDescriptorRemapRecord {
    // Logical texture ID from the material system. descriptorIndex remains
    // UINT32_MAX until a consumer maps that logical ID into its own heap.
    int32_t logicalTextureId = scene::kInvalidSceneIndex;
    uint32_t descriptorIndex = std::numeric_limits<uint32_t>::max();
    uint32_t materialIndex = std::numeric_limits<uint32_t>::max();
    uint32_t textureSlot = std::numeric_limits<uint32_t>::max();
};

static_assert(sizeof(GPUSceneGpuGeometryRecord) == 96);
static_assert(sizeof(GPUSceneGpuMaterialTextureInfo) == 48);
static_assert(sizeof(GPUSceneGpuMaterialRecord) == 560);
static_assert(offsetof(GPUSceneGpuMaterialRecord, identity) == 544);
static_assert(sizeof(GPUSceneGpuInstanceRecord) == 160);
static_assert(sizeof(GPUSceneGpuDrawKeyRecord) == 32);
static_assert(sizeof(GPUSceneGpuVertexRecord) == 64);
static_assert(sizeof(GPUSceneGpuMeshletRecord) == 80);
static_assert(sizeof(GPUSceneGpuMeshletDrawRecord) == 16);
static_assert(sizeof(GPUSceneGpuDescriptorRemapRecord) == 16);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuGeometryRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuMaterialTextureInfo>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuMaterialRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuInstanceRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuDrawKeyRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuVertexRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuMeshletRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuMeshletDrawRecord>);
static_assert(std::is_trivially_copyable_v<GPUSceneGpuDescriptorRemapRecord>);

struct GPUSceneSourceView {
    std::span<const scene::RenderPrimitive> renderPrimitives;
    std::span<const scene::RenderNode> renderNodes;
    std::span<const scene::RenderMaterial> materials;
    uint64_t lifetimeRevision = 0;
    uint64_t structuralRevision = 0;
    uint64_t contentRevision = 0;
    uint64_t transformRevision = 0;
    uint64_t visibilityRevision = 0;
    uint64_t externalRevision = 0;

    static GPUSceneSourceView fromScene(
        const scene::Scene& scene,
        uint64_t externalRevision = 0);
};

struct GPUSceneGeometryRecord {
    GPUSceneGeometryId id;
    int32_t sourceRenderPrimitiveIndex = scene::kInvalidSceneIndex;
    int32_t meshIndex = scene::kInvalidSceneIndex;
    int32_t primitiveIndex = scene::kInvalidSceneIndex;
    int32_t mode = 4;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t triangleCount = 0;
    scene::Bounds localBounds;
    uint64_t payloadFingerprint = 0;
};

struct GPUSceneMaterialRecord {
    GPUSceneMaterialId id;
    int32_t sourceMaterialIndex = scene::kInvalidSceneIndex;
    scene::RenderMaterial material;
    GPUSceneDrawBucket bucket = GPUSceneDrawBucket::OpaqueSingleSided;
    uint64_t payloadFingerprint = 0;
    bool fallback = false;
};

struct GPUSceneInstanceRecord {
    GPUSceneInstanceId id;
    int32_t sourceRenderNodeIndex = scene::kInvalidSceneIndex;
    scene::SceneEntity sourceObject = scene::kNullSceneEntity;
    int32_t sourceNodeIndex = scene::kInvalidSceneIndex;
    int32_t sourceRenderPrimitiveIndex = scene::kInvalidSceneIndex;
    int32_t sourceMaterialIndex = scene::kInvalidSceneIndex;
    GPUSceneGeometryId geometry;
    GPUSceneMaterialId material;
    GPUSceneDrawKey drawKey;
    float4x4 worldMatrix = float4x4::Identity();
    float4x4 previousWorldMatrix = float4x4::Identity();
    float4 localBoundingSphere{0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t transformRevision = 0;
    bool visible = true;
};

struct GPUSceneDrawSet {
    uint32_t generation = 0;
    uint64_t revision = 0;
    std::vector<GPUSceneInstanceId> instances;
    std::array<std::vector<GPUSceneInstanceId>, kGPUSceneDrawBucketCount> buckets;

    std::span<const GPUSceneInstanceId> instancesForBucket(GPUSceneDrawBucket bucket) const;
};

struct GPUSceneRasterDrawRange {
    uint32_t offset = 0;
    uint32_t count = 0;

    auto operator<=>(const GPUSceneRasterDrawRange&) const = default;
};

struct GPUSceneRasterDrawLayout {
    GPUSceneRasterDrawRange baseRange;
    std::vector<GPUSceneRasterDrawRange> lodRanges;
    uint32_t maxRangeCount = 0;
    uint32_t drawSetGeneration = 0;
    uint64_t drawSetRevision = 0;

    bool validFor(uint32_t generation, uint64_t revision) const;
};

struct GPUSceneBufferView {
    Buffer* buffer = nullptr;
    BufferView* view = nullptr;
    BindlessHandle bindless;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t structureStride = 0;
    uint32_t generation = 0;
    uint64_t revision = 0;

    bool valid() const;
    bool validFor(uint32_t expectedGeneration, uint64_t expectedRevision) const;
};

struct GPUSceneGlobalBufferViews {
    GPUSceneBufferView geometries;
    GPUSceneBufferView materials;
    GPUSceneBufferView instances;
    GPUSceneBufferView drawKeys;
    GPUSceneBufferView drawInstanceIds;
    GPUSceneBufferView vertices;
    GPUSceneBufferView indices;
    GPUSceneBufferView meshlets;
    GPUSceneBufferView meshletDraws;
    GPUSceneBufferView meshletVertices;
    GPUSceneBufferView meshletTriangleWords;
    GPUSceneBufferView descriptorRemap;
    uint32_t drawSetGeneration = 0;
    uint64_t drawSetRevision = 0;

    bool validFor(uint32_t generation, uint64_t revision) const;
};

enum class GPUSceneGlobalBufferKind : uint8_t {
    Geometries,
    Materials,
    Instances,
    DrawKeys,
    DrawInstanceIds,
    Vertices,
    Indices,
    Meshlets,
    MeshletDraws,
    MeshletVertices,
    MeshletTriangleWords,
    DescriptorRemap,
    Count,
};

inline constexpr size_t kGPUSceneGlobalBufferKindCount =
    static_cast<size_t>(GPUSceneGlobalBufferKind::Count);

struct GPUSceneConsumerBindings {
    std::array<BindlessHandle, kGPUSceneGlobalBufferKindCount> buffers{};
    // Descriptor handles remain valid across in-place revision updates. A
    // DrawSet generation change is the resource-allocation boundary.
    uint32_t drawSetGeneration = 0;
    // Revision captured when the handles were created, for diagnostics only.
    uint64_t drawSetRevision = 0;

    BindlessHandle operator[](GPUSceneGlobalBufferKind kind) const
    {
        return buffers[static_cast<size_t>(kind)];
    }

    bool validFor(const GPUSceneGlobalBufferViews& views) const;
};

enum class GPUSceneCullPhase : uint8_t {
    Early,
    Late,
    Count,
};

inline constexpr size_t kGPUSceneCullPhaseCount =
    static_cast<size_t>(GPUSceneCullPhase::Count);

struct GPUSceneBucketGpuView {
    GPUSceneBufferView indirectArguments;
    GPUSceneBufferView overflow;
    // Element offset into the phase visibleMeshletIds worklist. Consumers
    // with uniformly sized buckets may derive this as bucket * capacity;
    // GPUScene-owned View resources publish the explicit packed offset.
    uint32_t visibleMeshletOffset = 0;
    uint32_t visibleMeshletCapacity = 0;
};

struct GPUSceneCullPhaseGpuView {
    GPUSceneBufferView visibleMeshletIds;
    std::array<GPUSceneBucketGpuView, kGPUSceneRasterDrawBucketCount> buckets;
};

struct GPUSceneHzbGpuView {
    std::array<GPUSceneBufferView, 2> history;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0;
    uint32_t writeIndex = 0;
    uint64_t historyEpoch = 0;
    bool valid = false;
};

struct GPUSceneVisibleGpuResources {
    // Per-instance 0/1/2/3 culling state used by the early/late passes.
    GPUSceneBufferView instanceVisibilityStates;
    // Optional compact instance output for consumers that need one. GPUScene
    // never maps this buffer back to the CPU.
    GPUSceneBufferView visibleInstanceIds;
    GPUSceneBufferView visibleInstanceCounter;
    std::array<GPUSceneCullPhaseGpuView, kGPUSceneCullPhaseCount> phases;
    GPUSceneHzbGpuView hzb;
    GPUSceneViewId sourceView;
    uint32_t frameSlot = 0;
    uint32_t sourceDrawSetGeneration = 0;
    uint64_t sourceDrawSetRevision = 0;

    bool validFor(uint32_t generation, uint64_t revision) const;
};

struct GPUSceneVisibleDrawSetStats {
    uint32_t sourceInstanceCount = 0;
    uint32_t visibleInstanceCount = 0;
    std::array<uint32_t, kGPUSceneDrawBucketCount> bucketInstanceCounts{};
    uint32_t sourceDrawSetGeneration = 0;
    uint64_t sourceDrawSetRevision = 0;
    uint64_t prepareCount = 0;
    uint64_t hzbHistoryEpoch = 0;
    bool hzbValid = false;
};

struct GPUSceneVisibleDrawSet {
    std::vector<GPUSceneInstanceId> instances;
    std::array<std::vector<GPUSceneInstanceId>, kGPUSceneDrawBucketCount> buckets;
    GPUSceneVisibleGpuResources gpu;
    GPUSceneVisibleDrawSetStats stats;

    std::span<const GPUSceneInstanceId> instancesForBucket(GPUSceneDrawBucket bucket) const;
};

struct GPUSceneViewDesc {
    uint32_t frameSlotCount = 0;
    uint64_t userTag = 0;
    // Optional GPU resource capacities. A CPU-only GPUScene ignores these;
    // GPUSceneSubsystem allocates a persistent per-View bundle when all of
    // them are non-zero.
    uint32_t instanceCapacity = 0;
    std::array<uint32_t, kGPUSceneRasterDrawBucketCount> visibleMeshletCapacity{};
    uint32_t hzbWidth = 0;
    uint32_t hzbHeight = 0;
    uint32_t hzbMipCount = 0;
    uint64_t hzbElementCount = 0;
};

struct GPUSceneViewPrepareInfo {
    // Zero leaves the previously prepared extent unchanged.
    uint32_t width = 0;
    uint32_t height = 0;
    bool cameraCut = false;
    bool freezeCullingCamera = false;
};

struct GPUSceneStats {
    uint32_t geometryCount = 0;
    uint32_t materialCount = 0;
    uint32_t instanceCount = 0;
    uint32_t viewCount = 0;
    std::array<uint32_t, kGPUSceneDrawBucketCount> bucketInstanceCounts{};
    uint32_t drawSetGeneration = 0;
    uint64_t drawSetRevision = 0;
    uint64_t fullRebuildCount = 0;
    uint64_t incrementalSyncCount = 0;
    uint64_t unchangedSyncCount = 0;
    uint64_t deduplicatedGeometryCount = 0;
    uint64_t geometryPayloadConflictCount = 0;
    uint64_t skippedRenderNodeCount = 0;
    uint64_t invalidPrimitiveCount = 0;
    uint64_t invalidIndexCountPrimitiveCount = 0;
    uint64_t outOfRangeIndexPrimitiveCount = 0;
};

enum class GPUSceneInvalidPrimitiveReason : uint8_t {
    UnsupportedMode,
    InsufficientVertices,
    IndexCountNotMultipleOfThree,
    IndexOutOfRange,
};

struct GPUSceneInvalidPrimitiveDiagnostic {
    uint32_t sourceRenderPrimitiveIndex = 0;
    GPUSceneInvalidPrimitiveReason reason = GPUSceneInvalidPrimitiveReason::UnsupportedMode;
    uint64_t indexOffset = std::numeric_limits<uint64_t>::max();
    uint32_t vertexIndex = std::numeric_limits<uint32_t>::max();
};

enum class GPUSceneSyncResult : uint8_t {
    Unchanged,
    HistoryUpdated,
    Updated,
    RebuildRequired,
};

class GPUScene {
public:
    using VisibilityPredicate = std::function<bool(const GPUSceneInstanceRecord&)>;

    GPUScene() = default;
    ~GPUScene() = default;

    void setDefaultFrameSlotCount(uint32_t frameSlotCount);
    Result rebuild(const GPUSceneSourceView& source, std::string& log);
    GPUSceneSyncResult sync(const GPUSceneSourceView& source);
    void clearSource();
    void shutdown();

    const GPUSceneDrawSet& drawSet() const { return drawSet_; }
    std::span<const GPUSceneGeometryRecord> geometries() const { return geometries_; }
    std::span<const GPUSceneMaterialRecord> materials() const { return materials_; }
    std::span<const GPUSceneInstanceRecord> instances() const { return instances_; }
    std::span<const GPUSceneInvalidPrimitiveDiagnostic> invalidPrimitiveDiagnostics() const
    {
        return invalidPrimitiveDiagnostics_;
    }
    const GPUSceneStats& stats() const { return stats_; }

    const GPUSceneGeometryRecord* geometry(GPUSceneGeometryId id) const;
    const scene::RenderPrimitive* geometrySourcePrimitive(GPUSceneGeometryId id) const;
    const GPUSceneMaterialRecord* material(GPUSceneMaterialId id) const;
    const GPUSceneInstanceRecord* instance(GPUSceneInstanceId id) const;

    GPUSceneGeometryId geometryForRenderPrimitive(uint32_t renderPrimitiveIndex) const;
    GPUSceneMaterialId materialForSourceMaterial(uint32_t materialIndex) const;
    GPUSceneInstanceId instanceForRenderNode(uint32_t renderNodeIndex) const;
    std::span<const GPUSceneInstanceId> instancesForObject(scene::SceneEntity object) const;
    GPUSceneMaterialId fallbackMaterial() const { return fallbackMaterial_; }

    GPUSceneViewId createView(const GPUSceneViewDesc& desc = {});
    bool destroyView(GPUSceneViewId view);
    uint32_t viewFrameSlotCount(GPUSceneViewId view) const;
    bool invalidateViewGpuResources(
        GPUSceneViewId view,
        bool invalidateHzbHistory = false);
    bool prepareView(
        GPUSceneViewId view,
        uint32_t frameSlot,
        const VisibilityPredicate& predicate = {});
    bool prepareView(
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUSceneViewPrepareInfo& info,
        const VisibilityPredicate& predicate = {});
    bool markViewHzbValid(GPUSceneViewId view, uint32_t frameSlot, bool valid = true);
    const GPUSceneVisibleDrawSet* visibleDrawSet(GPUSceneViewId view, uint32_t frameSlot) const;
    GPUSceneVisibleDrawSet* visibleDrawSetForUpdate(GPUSceneViewId view, uint32_t frameSlot);
    bool setVisibleGpuResources(
        GPUSceneViewId view,
        uint32_t frameSlot,
        GPUSceneVisibleGpuResources resources);

    const GPUSceneGlobalBufferViews& globalBufferViews() const { return globalBufferViews_; }
    bool setGlobalBufferViews(GPUSceneGlobalBufferViews views);
    void invalidateGpuResources();

private:
    struct ViewSlot {
        GPUSceneViewDesc desc;
        std::vector<GPUSceneVisibleDrawSet> frameSlots;
        uint32_t generation = 1;
        bool occupied = false;
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t temporalHistoryEpoch = 0;
        uint64_t hzbHistoryEpoch = 1;
        bool freezeCullingCamera = false;
        bool freezeStateValid = false;
        bool hzbValid = false;
    };

    static uint32_t advanceGeneration(uint32_t generation);
    void invalidateSourceIds();
    void rebuildDrawSet();
    void invalidateVisibleDrawSets();
    bool validView(GPUSceneViewId view) const;

    uint32_t defaultFrameSlotCount_ = 1;
    uint32_t geometryGeneration_ = 1;
    uint32_t materialGeneration_ = 1;
    uint32_t instanceGeneration_ = 1;
    uint32_t drawSetGeneration_ = 0;
    uint64_t nextDrawSetRevision_ = 1;
    uint64_t temporalHistoryEpoch_ = 1;
    std::vector<GPUSceneGeometryRecord> geometries_;
    // Canonical, deduplicated CPU backing for persistent raster uploads.
    // Entries are aligned one-to-one with geometries_.
    std::vector<scene::RenderPrimitive> geometrySourcePrimitives_;
    std::vector<GPUSceneMaterialRecord> materials_;
    std::vector<GPUSceneInstanceRecord> instances_;
    GPUSceneDrawSet drawSet_;
    std::vector<GPUSceneGeometryId> geometryForRenderPrimitive_;
    std::vector<GPUSceneMaterialId> materialForSourceMaterial_;
    std::vector<GPUSceneInstanceId> instanceForRenderNode_;
    std::unordered_map<uint32_t, std::vector<GPUSceneInstanceId>> instancesForObject_;
    GPUSceneMaterialId fallbackMaterial_;
    std::vector<ViewSlot> views_;
    std::vector<uint32_t> freeViews_;
    GPUSceneGlobalBufferViews globalBufferViews_;
    GPUSceneStats stats_;
    std::vector<uint64_t> primitiveFingerprints_;
    std::vector<uint64_t> materialFingerprints_;
    std::vector<uint64_t> renderNodeTopologyFingerprints_;
    std::vector<GPUSceneInvalidPrimitiveDiagnostic> invalidPrimitiveDiagnostics_;
    uint64_t sourceLifetimeRevision_ = 0;
    uint64_t sourceStructuralRevision_ = 0;
    uint64_t sourceContentRevision_ = 0;
    uint64_t sourceTransformRevision_ = 0;
    uint64_t sourceVisibilityRevision_ = 0;
    uint64_t sourceExternalRevision_ = 0;
    bool hasSource_ = false;
};

} // namespace metallic::render
