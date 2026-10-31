#pragma once

#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"

#include <memory>
#include <array>
#include <span>
#include <string>

namespace metallic::scene { struct RenderMaterial; struct Bounds; }

namespace metallic::render {

struct VirtualShadowMapCasterResources;
struct CPUProfileRecorder;
class RenderGraphExecutionContext;
class VirtualShadowMaps;

enum class VirtualShadowMapLightType : uint32_t { Directional, Spot, PointFace };
enum class VirtualShadowMapFilter : uint32_t { Hard, PCF, SMRT };

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
    bool collectGPUWork = false; // Diagnostic shader permutation; never use its timings as ordinary performance.
    bool profileBuildSubstages = false; // Optional nested timestamps for cost attribution, not ordinary A/B timing.
    bool optimizeMark = true;
    bool optimizeProjectionStatistics = false; // Experimental grouped PCF/SMRT projection statistics.
    // Multi-face GPU projection; single faces and grouped projection statistics retain the original kernel.
    bool onePassProjection = true;
    bool optimizeMarkSamples = true; // Reuse invariant Mark work for local lights; directional uses the original path.
    bool directionalMark = true; // Mark-only orthographic specialization; preserves receiver/sample demands.
    bool clusterCulling = true;
    bool clusterPageBitset = true; // Production cluster-page query; effective only with clusterCulling.
    bool trianglePageBitset = true; // Enumerate dirty fine pages; requires the effective cluster-page bitset.
    bool directionalBuild = true; // Build-only orthographic shader specialization; local lights stay generic.
    bool compactBuild = false; // Experimental cluster candidate compaction and indirect fine build.
    bool shadowLOD = false; // Shared caster cut driven by all admitted shadow projections.
    bool shadowLODHierarchy = true; // Cached adaptive traversal; false retains the flat reference for A/B.
    float shadowLODErrorTexels = 1.0f; // Zero requests the finest available complete cut.
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
    bool receiverFootprint = true; // All filters share receiver-selected resolution; false starts at the finest level.
    bool enabled = true;
    bool cacheEnabled = false;
    bool cacheLayering = false; // Separate static base depth from dynamic composition; requires caching.
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

// Ordered per-light projections. Point lights contain all six canonical faces;
// filtered point groups share one prepared face table. Other lights have one face.
struct VirtualShadowMapProjectionGroup
{
    std::array<VirtualShadowMaps*, 6> faces{};
    std::array<VirtualShadowMapResult, 6> results{};
    uint32_t faceCount = 0;
    bool filteredPoint = false;
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

// A complete coarse projection plus directional clipmaps or local mip levels.
// Physical page zero is reserved for complete coarse coverage. Fine allocation
// failure or work overflow queries that rendered coarse page, never uninitialized depth.
class VirtualShadowMaps
{
public:
    VirtualShadowMaps();
    ~VirtualShadowMaps();
    static Result<std::vector<VirtualShadowMapLODProjection>> makeShadowLODProjections(
        const scene::Bounds& bounds, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapSettings& settings, std::string& log);
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false,
        const VirtualShadowMapProjectionTarget* output = nullptr, CPUProfileRecorder* profiler = nullptr,
        RenderGraphExecutionContext* gpuProfiler = nullptr);
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false,
        const VirtualShadowMapProjectionTarget* output = nullptr, CPUProfileRecorder* profiler = nullptr,
        RenderGraphExecutionContext* gpuProfiler = nullptr);

    // A filtered point light records all six faces before projecting any face.
    // The ordered face results keep each tap's page table, projection and atlas
    // explicit; no tap may wrap into an unrelated physical atlas tile.
    Result<> project(CommandBuffer& commands, std::span<const VirtualShadowMapResult> pointFaces, std::string& log,
        CPUProfileRecorder* profiler = nullptr, RenderGraphExecutionContext* gpuProfiler = nullptr);

    // Shares CPU point-table preparation and its resource dependencies. The
    // optional GPU batch requires a common view/depth and filter settings, as
    // supplied by LightsPass. Empty input is valid; one face uses project().
    static Result<> projectShared(CommandBuffer& commands, std::span<const VirtualShadowMapProjectionGroup> groups,
        std::string& log, CPUProfileRecorder* profiler = nullptr, RenderGraphExecutionContext* gpuProfiler = nullptr,
        bool onePassProjection = true);

    // Logical resource bytes currently owned by this cache, recording and pool.
    // Native padding and retired recordings outside this cache/pool are excluded;
    // use Device::memoryBudget() for whole-device live/peak allocation evidence.
    VirtualShadowMapMemoryUsage memoryUsage() const;
    // Creation calls and pool hits/evictions, cumulative for the current device.
    // Eviction releases the pool owner, not necessarily the last native lease.
    VirtualShadowMapResourcePoolStats resourcePoolStats() const;

private:
    Result<> preparePointFaces(CommandBuffer& commands, std::span<const VirtualShadowMapResult> pointFaces, std::string& log);
    Result<> projectPrepared(CommandBuffer& commands, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler);
    static Result<> projectOnePassPrepared(CommandBuffer& commands, std::span<const VirtualShadowMapProjectionGroup> groups,
        uint32_t faceCount, std::string& log, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler);
    struct Programs;
    struct State;
    struct Cache;
    struct ResourcePool;
    std::shared_ptr<ResourcePool> pool_;
    std::shared_ptr<Programs> programs_;
    std::shared_ptr<State> state_;
    std::shared_ptr<Cache> cache_;
    std::shared_ptr<Cache> pending_;
};

} // namespace metallic::render
