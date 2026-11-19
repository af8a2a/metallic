#pragma once

#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"

#include <memory>
#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene { struct RenderMaterial; struct Bounds; }

namespace metallic::render {

struct VirtualShadowMapCasterResources;
struct MeshletStreamDeferredGPUResourcesView;
struct CPUProfileRecorder;
struct ClusterLightGridSnapshot;
class RenderGraphExecutionContext;
class VirtualShadowMaps;
class VirtualShadowMapPhysicalPool;

enum class VirtualShadowMapLightType : uint32_t { Directional, Spot, PointFace };
enum class VirtualShadowMapFilter : uint32_t { Hard, PCF, SMRT };

struct ShadowPush
{
    uint32_t geometry, materials, pages, coverageInputs;
    uint32_t workItems, statistics, projections, atlas;
    uint32_t depth, shadow, debug, view;
    uint32_t arguments, triangleCount, clipmapCount, virtualPagesPerAxis;
    uint32_t physicalPagesX, physicalPagesY, physicalPageCount, pageSize;
    uint32_t workCapacity, batchCount, workBase, rasterCoarse;
    uint32_t sampler, enabled, totalPages, lightType;
    uint32_t previousPages, reuse, owners, cacheMode;
    uint32_t coarseBatchCount, cancelledCount, cacheEpoch, dirtyBoundsCount;
    float lightRange;
    uint32_t projectionFace, filterMode, pointFaces;
    float pcfRadius, rayLength, sourceRadius, angularRadius;
    uint32_t rayCount, samplesPerRay, receiverLOD, projectionFlags;
    float rayLengthScale, maxRayAngle;
    uint32_t samplePhase, sampleReserved;
    float lodScreenRayLength;
    uint32_t receiverGeometry;
    uint32_t physicalPool, poolFace;
};
static_assert(sizeof(ShadowPush) == 224);

// Optional, same-frame primary visibility from the rasterInfo producer. The
// pass supplies only ordinary (non-tessellated) stream publications; missing
// inputs leave refined receiver correction disabled. record() retains every
// referenced native allocation through its descriptor leases.
struct VirtualShadowMapReceiverResources
{
    TextureView* visibility = nullptr;
    const MeshletStreamDeferredGPUResourcesView* stream = nullptr;
};

// Point faces use +X, -X, +Y, -Y, +Z, -Z. Position and the complete light
// projection identify depth content; source identities belong to the manager.
struct VirtualShadowMapLight
{
    VirtualShadowMapLightType type = VirtualShadowMapLightType::Directional;
    std::array<float, 3> position{};
    std::array<float, 3> direction{0, -1, 0};
    float outerConeAngle = 0.7853981634f;
    float range = 0; // Zero means unbounded influence; scene bounds determine far.
    float nearPlane = 0.001f;
    float angularRadius = 0; // Directional source half-angle in radians.
    float sourceRadius = 0; // Local disk source radius in world meters.
    uint32_t face = 0;
    bool enabled = true;
};

// One projection view: directional clipmaps, a spot frustum or one point face.
struct VirtualShadowMapSettings
{
    bool unifiedMark = false; // Reuse the main-view receiver across shared shadow faces.
    bool batchBuildRaster = true; // Shared pools batch compatible faces; false retains the serial reference path.
    bool collectReceiverCoverage = false; // Diagnostic per-pixel records; requires GPU work diagnostics.
    bool collectGPUWork = false; // Diagnostic shader permutation; never use its timings as ordinary performance.
    bool profileBuildSubstages = false; // Optional nested timestamps for cost attribution, not ordinary A/B timing.
    bool optimizeProjectionStatistics = false; // Request grouped statistics for compatible filters and per-face projection.
    bool compactBuild = false; // Experimental cluster candidate compaction and indirect fine build.
    bool pageWindowRaster = false; // Share directional fine geometry across bounded page windows.
    bool clusterPageRaster = false; // Pack fine work into masked cluster/page mesh groups.
    bool clusterHybridRaster = false; // Route eligible fine cluster/page work to software depth rasterization.
    bool twoPassHZB = false; // Defer history-occluded fine tickets and retest after the main raster.
    bool hierarchyHZB = false; // Defer directional instance subtrees before fine-page Build.
    bool instanceBuild = false; // Directional instance culling and indirect fine Build.
    bool dirtyPageHierarchy = false; // Experimental bounds/HMip cluster rejection; triangle tests remain exact.
    bool shadowLOD = false; // GPU hierarchy cuts for each face's clipmap/mip and coarse view.
    float shadowLODErrorTexels = 1.0f; // Zero requests the finest available complete cut.
    bool shadowLODSmartBias = false; // Hard directional shadows: screen-depth constrained receiver bias.
    float shadowLODScreenRayLength = 0.015f; // Fraction of vertical view extent at the receiver.
    uint32_t pageSize = 128;
    uint32_t virtualPagesPerAxis = 16;
    uint32_t clipmapCount = 4;
    uint32_t physicalPageCount = 256;
    uint32_t maxWorkItems = 65535;
    float firstClipmapExtent = 4.0f; // Half-width in world meters, doubles per level.
    float shadowBias = 0.002f; // World meters toward the light.
    VirtualShadowMapFilter filter = VirtualShadowMapFilter::Hard;
    float pcfRadius = 1.0f; // Radius in selected virtual texels.
    uint32_t smrtRayCount = 8;
    uint32_t smrtSamplesPerRay = 8;
    float smrtMaxRayLength = 10.0f; // Soft search in meters; the first reverse sample retains farther blockers.
    float smrtRayLengthScale = 0.0f; // Zero keeps fixed search; otherwise cap by camera distance times this scale.
    float smrtMaxRayAngle = 0.03f; // Maximum local-light angular span, in radians.
    bool smrtTemporalSampling = false; // Rotate raw samples; the lights pass owns the separate history resolve.
    bool receiverFootprint = true; // All filters share the projected receiver footprint when enabled.
    float resolutionLodBias = 0.0f; // Additional clipmap/mip levels selected under physical page pressure.
    bool enabled = true;
    bool cacheEnabled = false;
    bool cacheLayering = false; // Separate static base depth from dynamic composition; requires caching.
    uint32_t framesStaticThreshold = 100; // Submitted unchanged frames before a dynamic caster rejoins static depth. Zero disables.
    bool forceRedraw = false;
};

struct VirtualShadowMapResult
{
    Texture* texture = nullptr;
    TextureView* shadow = nullptr; // Linear visibility. Only Hard is compatible with legacy sqrt encoding.
    Texture* debugTexture = nullptr;
    TextureView* debug = nullptr;
    Buffer* statistics = nullptr;
    Texture* atlas = nullptr;
    TextureView* atlasView = nullptr;
    Buffer* pageTable = nullptr;
    Buffer* projections = nullptr;
    uint32_t physicalPagesX = 0;
    uint32_t clipmapCount = 0;
    uint32_t pageSize = 0;
    // Keeps backing wrappers alive for deferred projection/readback. It owns no
    // State or cache links, so cross-face projection cannot form reference cycles.
    std::shared_ptr<void> resourceOwner;
    Buffer* cacheDiagnostics = nullptr; // Independent kVirtualShadowMapCacheDiagnosticSize public prefix.
    // Current-frame requested subtile words in allocator scratch, before admission restricts residency.
    Buffer* requestedReceiverMasks = nullptr;
    uint64_t requestedReceiverMaskOffset = 0;
    Buffer* lodSelection = nullptr; // View-major uint[clusterCount], including the coarse view.
    Buffer* lodStatistics = nullptr; // View-major 32-word v3 diagnostics.
    Buffer* lodChanges = nullptr; // View-major uint[instanceCount], compared with the committed cut.
    uint32_t lodClusterCount = 0;
    uint32_t lodInstanceCount = 0;
};

// Demand before physical-page admission, including each face's coarse page.
struct VirtualShadowMapPressureSample
{
    uint64_t sequence = 0, frameIndex = 0;
    uint32_t requestedPages = 0, capacity = 0;
    float appliedBias = 0;
};

// Pass-owned across frames and reset with its device. Only completed copies
// from submitted command buffers can contribute a pressure observation.
class VirtualShadowMapPressureFeedback
{
public:
    Result<> record(Device& device, CommandBuffer& commands, std::span<const VirtualShadowMapResult> faces,
        uint32_t capacity, float appliedBias, const VirtualShadowMapPhysicalPool* pool = nullptr);
    Result<std::optional<VirtualShadowMapPressureSample>> poll();

private:
    struct Slot;
    std::vector<std::shared_ptr<Slot>> slots_;
    uint64_t nextSequence_ = 1, consumedSequence_ = 0;
};

// The caller owns these graph textures through GPU completion and transitions
// them to General for projection. Each view covers one layer. Point faces may
// share that layer because only the canonical face writes a receiver pixel.
struct VirtualShadowMapProjectionTarget
{
    Texture* shadow = nullptr;
    Texture* debug = nullptr; // Optional: null omits the debug allocation, binding and writes.
    uint32_t layer = 0;
    bool sharedPointLayer = false;
};

struct VirtualShadowMapFaceRequest
{
    VirtualShadowMaps* shadow = nullptr;
    VirtualShadowMapLight light;
    VirtualShadowMapProjectionTarget output;
    uint32_t sourceSlot = UINT32_MAX; // 0/1 for Sun/Moon, 2 + local GPUScene index for grid Mark.
};

// Ordered per-light projections. Point lights contain all six canonical faces;
// filtered point groups share one prepared face table. Other lights have one face.
struct VirtualShadowMapProjectionGroup
{
    std::array<VirtualShadowMaps*, 6> faces{};
    std::array<VirtualShadowMapResult, 6> results{};
    uint32_t faceCount = 0;
    bool filteredPoint = false;
    uint32_t lightIndex = UINT32_MAX; // Publication slot; required by the optional grid projection.
};

// The raster producer owns the matching grid through command completion.
// Without a grid, projection visits every admitted job. Optional diagnostics
// use the low/high uint pairs defined in VirtualShadowMapProjectionGrid.h.
struct VirtualShadowMapProjectionGridResources
{
    const ClusterLightGridSnapshot* grid = nullptr;
    Buffer* diagnostics = nullptr;
};

struct VirtualShadowMapMemoryUsage
{
    uint64_t atlasBytes = 0;
    uint64_t staticAtlasBytes = 0; // Subset of atlasBytes, not additional memory.
    uint64_t privateOutputBytes = 0;
    uint64_t bufferBytes = 0;
    uint64_t allocatedAtlasPages = 0;
};

struct VirtualShadowMapResourcePoolStats
{
    uint64_t bufferCreates = 0, bufferReuses = 0, bufferEvictions = 0;
    uint64_t textureCreates = 0, textureReuses = 0, textureEvictions = 0;
    uint64_t viewCreates = 0;
    uint64_t retainedBuffers = 0, retainedBufferBytes = 0;
    uint64_t retainedTextures = 0, retainedTextureBytes = 0;
};

// One physical address space for all admitted lights. Per-face virtual tables
// retain their identities; the GPU transfers free/evictable slots between them.
class VirtualShadowMapPhysicalPool
{
public:
    VirtualShadowMapPhysicalPool();
    ~VirtualShadowMapPhysicalPool();
    VirtualShadowMapPhysicalPool(VirtualShadowMapPhysicalPool&&) noexcept;
    VirtualShadowMapPhysicalPool& operator=(VirtualShadowMapPhysicalPool&&) noexcept;
    VirtualShadowMapMemoryUsage memoryUsage() const;
    Result<> publishAllocations(CommandBuffer& commands, Buffer& lights, Buffer& budgetDiagnostics,
        std::span<const uint32_t> publicationFaces, std::string& log);

private:
    friend class VirtualShadowMaps;
    friend class VirtualShadowMapPressureFeedback;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A complete coarse projection plus directional clipmaps or local mip levels.
// A physical page is reserved for each face's complete coarse coverage (slot
// zero only in standalone mode). Allocation failure or work overflow queries
// that rendered coarse page, never uninitialized depth.
class VirtualShadowMaps
{
public:
    VirtualShadowMaps();
    ~VirtualShadowMaps();
    // Records all requests before global allocation, then renders each face.
    // Results follow input order; projection is deferred until project/projectShared.
    // The pool and every face commit or cancel in the same tracked frame.
    static Result<std::vector<VirtualShadowMapResult>> recordShared(VirtualShadowMapPhysicalPool& pool,
        Device& device, CommandBuffer& commands, TextureView& depth, const ViewConstants& view,
        std::span<const VirtualShadowMapFaceRequest> faces, const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, uint32_t physicalPageBudget, bool demandDriven,
        std::string& log, CPUProfileRecorder* profiler = nullptr, RenderGraphExecutionContext* gpuProfiler = nullptr,
        const VirtualShadowMapReceiverResources* receiver = nullptr, const ClusterLightGridSnapshot* markGrid = nullptr);
    static Result<std::vector<VirtualShadowMapLODProjection>> makeShadowLODProjections(
        const scene::Bounds& bounds, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapSettings& settings, std::string& log);
    static Result<std::shared_ptr<Buffer>> aggregateLODStatistics(Device& device, CommandBuffer& commands,
        std::span<const VirtualShadowMapResult> faces, std::string& log);
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false,
        const VirtualShadowMapProjectionTarget* output = nullptr, CPUProfileRecorder* profiler = nullptr,
        RenderGraphExecutionContext* gpuProfiler = nullptr,
        const VirtualShadowMapReceiverResources* receiver = nullptr);
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false,
        const VirtualShadowMapProjectionTarget* output = nullptr, CPUProfileRecorder* profiler = nullptr,
        RenderGraphExecutionContext* gpuProfiler = nullptr,
        const VirtualShadowMapReceiverResources* receiver = nullptr);

    // A filtered point light records all six faces before projecting any face.
    // The ordered face results keep each tap's page table, projection and atlas
    // explicit; no tap may wrap into an unrelated physical atlas tile.
    Result<> project(CommandBuffer& commands, std::span<const VirtualShadowMapResult> pointFaces, std::string& log,
        CPUProfileRecorder* profiler = nullptr, RenderGraphExecutionContext* gpuProfiler = nullptr);

    // Shares CPU point-table preparation and its resource dependencies. The
    // GPU batch requires a common view/depth and filter settings, as supplied
    // by LightsPass. Empty input is valid; one face uses project().
    static Result<> projectShared(CommandBuffer& commands, std::span<const VirtualShadowMapProjectionGroup> groups,
        std::string& log, CPUProfileRecorder* profiler = nullptr, RenderGraphExecutionContext* gpuProfiler = nullptr,
        const VirtualShadowMapProjectionGridResources* grid = nullptr);

    // Logical resource bytes currently owned by this cache, recording and pool.
    // Native padding and retired recordings outside this cache/pool are excluded;
    // use Device::memoryBudget() for whole-device live/peak allocation evidence.
    VirtualShadowMapMemoryUsage memoryUsage() const;
    // Creation calls and pool hits/evictions, cumulative for the current device.
    // Eviction releases the pool owner, not necessarily the last native lease.
    VirtualShadowMapResourcePoolStats resourcePoolStats() const;

private:
    struct CasterChangeContext;
    Result<VirtualShadowMapResult> recordInternal(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
        std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings,
        std::string& log, bool deferProjection, const VirtualShadowMapProjectionTarget* output,
        CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
        const VirtualShadowMapReceiverResources* receiver, VirtualShadowMapPhysicalPool* physicalPool, uint32_t poolFace,
        CasterChangeContext* casterChangeContext = nullptr);
    Result<> preparePointFaces(CommandBuffer& commands, std::span<const VirtualShadowMapResult> pointFaces, std::string& log);
    Result<> projectPrepared(CommandBuffer& commands, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler);
    static Result<> projectOnePassPrepared(CommandBuffer& commands, std::span<const VirtualShadowMapProjectionGroup> groups,
        uint32_t faceCount, std::string& log, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
        const VirtualShadowMapProjectionGridResources* grid);
    struct Programs;
    struct State;
    static Result<> recordBatched(VirtualShadowMapPhysicalPool& physicalPool, CommandBuffer& commands,
        std::span<const std::shared_ptr<State>> states, const VirtualShadowMapSettings& settings,
        std::string& log, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler);
    struct Cache;
    struct ResourcePool;
    std::shared_ptr<ResourcePool> pool_;
    std::shared_ptr<Programs> programs_;
    std::shared_ptr<State> state_;
    std::shared_ptr<Cache> cache_;
    std::shared_ptr<Cache> pending_;
};

} // namespace metallic::render
