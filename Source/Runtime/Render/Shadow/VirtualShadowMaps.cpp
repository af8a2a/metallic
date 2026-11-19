#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProjectionGrid.h"
#include "Runtime/Render/Shadow/VirtualShadowMapPhysicalPoolInterop.h"
#include "Runtime/Render/Shadow/VirtualShadowMapBatchInterop.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLODInterop.h"
#include "Runtime/Render/ClusterLightGrid.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <unordered_set>
#include <vector>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kDrawBatchSize = 65535;

uint32_t hybridScratchPageCount(uint32_t physicalPages, uint32_t pageSize)
{
    // The remaining physical slots use the regular D32 path. A large virtual
    // shadow pool therefore does not require a second full-sized depth atlas.
    constexpr uint32_t kScratchBytes = 64u << 20;
    constexpr uint32_t kScratchTexels = kScratchBytes / sizeof(uint32_t);
    return std::min(physicalPages, kScratchTexels / (pageSize * pageSize));
}


struct ShadowProjectionJob
{
    uint32_t pages, statistics, projections, atlas;
    uint32_t filterMode, shadow, debug, physicalPagesX;
    uint32_t clipmapCount, enabled, lightType, projectionFace;
    uint32_t pointFaces, projectionFlags, samplePhase;
    float lightRange, sourceRadius, angularRadius, lodScreenRayLength;
    uint32_t receiverGeometry;
};
static_assert(sizeof(ShadowProjectionJob) == 80);

struct ShadowReceiverGeometry
{
    uint32_t visibility, records, groups, pages;
    uint32_t pageTable, params, recordCapacity, flags;
};
static_assert(sizeof(ShadowReceiverGeometry) == 32);

struct ShadowFace
{
    uint32_t atlas, pages, projections, statistics;
    uint32_t physicalPagesX, clipmapCount, pageSize, reserved;
};
static_assert(sizeof(ShadowFace) == 32);

struct ShadowProjection
{
    std::array<float, 4> rowX, rowY, rowZ, rowW, metric;
    std::array<uint32_t, 4> page;
    std::array<float, 4> originNear;
};
static_assert(sizeof(ShadowProjection) == 112);

struct ShadowMaterial
{
    std::array<float, 4> alpha;
    std::array<uint32_t, 4> resources;
    std::array<float, 4> uvX, uvY;
};
static_assert(sizeof(ShadowMaterial) == 64);

using Vector = std::array<float, 3>;
float dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vector normalized(Vector value)
{
    const float length = std::sqrt(dot(value, value));
    for (auto& component : value) { component /= length; }
    return value;
}

bool validProjectionSettings(const VirtualShadowMapSettings& settings)
{
    return settings.pageSize != 0 && settings.pageSize <= 128 && !(settings.pageSize & (settings.pageSize - 1)) &&
        settings.virtualPagesPerAxis != 0 && settings.virtualPagesPerAxis <= 128 &&
        settings.clipmapCount != 0 && settings.clipmapCount <= 12 &&
        std::isfinite(settings.firstClipmapExtent) && settings.firstClipmapExtent > 0 &&
        std::isfinite(settings.shadowBias) && settings.shadowBias >= 0 &&
        std::isfinite(settings.resolutionLodBias) && settings.resolutionLodBias >= 0 &&
        std::isfinite(settings.shadowLODErrorTexels) && settings.shadowLODErrorTexels >= 0 &&
        std::isfinite(settings.shadowLODScreenRayLength) && settings.shadowLODScreenRayLength >= 0 &&
        settings.shadowLODScreenRayLength <= 1;
}

Result<> validateProjectionLight(const VirtualShadowMapLight& light, std::string& log)
{
    const bool directional = light.type == VirtualShadowMapLightType::Directional;
    const auto finite = [](const Vector& value) {
        return std::all_of(value.begin(), value.end(), [](float v) { return std::isfinite(v); });
    };
    if (uint32_t(light.type) > uint32_t(VirtualShadowMapLightType::PointFace) ||
        !std::isfinite(light.angularRadius) || light.angularRadius < 0 || light.angularRadius >= 1.5707963267948966f ||
        !std::isfinite(light.sourceRadius) || light.sourceRadius < 0 ||
        (light.type != VirtualShadowMapLightType::PointFace && (!finite(light.direction) || dot(light.direction, light.direction) < 1e-12f)) ||
        (!directional && (!finite(light.position) || !std::isfinite(light.range) || light.range < 0 ||
            !std::isfinite(light.nearPlane) || light.nearPlane <= 0)) ||
        (light.type == VirtualShadowMapLightType::PointFace && light.face >= 6)) {
        log += "Invalid VSM light projection.\n";
        return makeError(Error::InvalidArgument);
    }
    if (light.type == VirtualShadowMapLightType::Spot && (!std::isfinite(light.outerConeAngle) ||
        light.outerConeAngle <= 0 || light.outerConeAngle >= 1.5707963267948966f)) {
        log += "VSM spot projection requires an outer half-angle strictly between zero and pi/2.\n";
        return makeError(Error::Unsupported);
    }
    return {};
}

std::vector<ShadowProjection> makeProjections(const scene::Bounds& bounds, const ViewConstants& view,
    Vector direction, const VirtualShadowMapSettings& settings)
{
    const auto z = normalized(direction);
    const auto x = normalized(cross(std::abs(z[1]) < 0.95f ? Vector{0, 1, 0} : Vector{1, 0, 0}, z));
    const auto y = cross(z, x);
    Vector low{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    Vector high{-low[0], -low[1], -low[2]};
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const Vector position = bounds.valid ? Vector{corner & 1 ? bounds.max.x : bounds.min.x,
            corner & 2 ? bounds.max.y : bounds.min.y, corner & 4 ? bounds.max.z : bounds.min.z} : Vector{0, 0, 0};
        const Vector light{dot(x, position), dot(y, position), dot(z, position)};
        for (size_t axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], light[axis]); high[axis] = std::max(high[axis], light[axis]);
        }
    }
    const float margin = std::max(0.01f, settings.shadowBias * 2) +
        (settings.cacheEnabled ? std::max(high[2] - low[2], 1.f) * .25f : 0.f);
    low[2] -= margin; high[2] += margin;
    const float depthRange = std::max(high[2] - low[2], 0.01f);
    const Vector eye{view.current.eye[0], view.current.eye[1], view.current.eye[2]};
    const Vector eyeLight{dot(x, eye), dot(y, eye), dot(z, eye)};
    std::vector<ShadowProjection> result;
    for (uint32_t level = 0; level <= settings.clipmapCount; ++level) {
        const bool coarse = level == settings.clipmapCount;
        const uint32_t axis = coarse ? 1 : settings.virtualPagesPerAxis;
        const float extent = coarse ? std::max({(high[0] - low[0]) * .5f, (high[1] - low[1]) * .5f, 0.01f}) *
            (1.0f + 4.0f / settings.pageSize + (settings.cacheEnabled ? .25f : 0.f)) + margin : std::ldexp(settings.firstClipmapExtent, int(level));
        const float pageWorldSize = 2 * extent / axis;
        const float pageOriginX = coarse ? 0.f : std::floor(eyeLight[0] / pageWorldSize);
        const float pageOriginY = coarse ? 0.f : std::floor(eyeLight[1] / pageWorldSize);
        const float centerX = coarse ? (low[0] + high[0]) * .5f : pageOriginX * pageWorldSize;
        const float centerY = coarse ? (low[1] + high[1]) * .5f : pageOriginY * pageWorldSize;
        // Fine projection offsets share the canonical page grid used by the
        // rasterizer; avoid a world-size multiply/divide round trip.
        result.push_back({{x[0] / extent, x[1] / extent, x[2] / extent,
                coarse ? -centerX / extent : -pageOriginX * (2.f / axis)},
            {y[0] / extent, y[1] / extent, y[2] / extent,
                coarse ? -centerY / extent : -pageOriginY * (2.f / axis)},
            {-z[0] / depthRange, -z[1] / depthRange, -z[2] / depthRange, high[2] / depthRange},
            {0, 0, 0, 1},
            {extent, pageWorldSize / settings.pageSize, settings.shadowBias / depthRange, 0},
            {level * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis, axis, level, coarse ? 1u : 0u},
            {pageOriginX, pageOriginY, 0, 0}});
    }
    return result;
}

std::pair<Vector, Vector> projectedBounds(const scene::Bounds& bounds, const ShadowProjection& projection)
{
    Vector low{FLT_MAX, FLT_MAX, FLT_MAX}, high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    const std::array rows{projection.rowX, projection.rowY, projection.rowZ};
    float lowW = FLT_MAX, highW = -FLT_MAX;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const Vector p{corner & 1 ? bounds.max.x : bounds.min.x, corner & 2 ? bounds.max.y : bounds.min.y,
            corner & 4 ? bounds.max.z : bounds.min.z};
        const auto& w = projection.rowW;
        const float clipW = w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + w[3];
        lowW = std::min(lowW, clipW); highW = std::max(highW, clipW);
        for (size_t axis = 0; axis < 3; ++axis) {
            const auto& row = rows[axis];
            const float value = (row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]) /
                (projection.metric[3] == 0 ? 1.f : std::max(clipW, projection.originNear[3]));
            low[axis] = std::min(low[axis], value); high[axis] = std::max(high[axis], value);
        }
    }
    if (projection.metric[3] != 0 && lowW <= projection.originNear[3]) {
        // A near-plane crossing can cover any part of this face after clipping.
        // A box entirely behind it contributes nothing to this face.
        return highW < projection.originNear[3] ? std::pair{Vector{2, 2, 2}, Vector{2, 2, 2}} :
            std::pair{Vector{-1, -1, 0}, Vector{1, 1, 1}};
    }
    return {low, high};
}

float localFarExtent(const scene::Bounds& bounds, const VirtualShadowMapLight& light)
{
    if (light.range > 0) { return light.range; }
    float distance = light.nearPlane;
    if (bounds.valid) {
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const Vector delta{(corner & 1 ? bounds.max.x : bounds.min.x) - light.position[0],
                (corner & 2 ? bounds.max.y : bounds.min.y) - light.position[1],
                (corner & 4 ? bounds.max.z : bounds.min.z) - light.position[2]};
            distance = std::max(distance, std::sqrt(dot(delta, delta)));
        }
    }
    return distance;
}

bool useDistantLightSinglePage(const ViewConstants& view, const VirtualShadowMapLight& light,
    const VirtualShadowMapSettings& settings)
{
    if (!settings.distantLightSinglePage || light.type == VirtualShadowMapLightType::Directional || light.range == 0.f) {
        return false;
    }
    // A narrow spot cone needs more shadow texels per unit of world space.
    const float maxRadius = float(settings.pageSize) * (light.type == VirtualShadowMapLightType::Spot ?
        std::min(1.f, std::tan(light.outerConeAngle)) : 1.f);
    const auto& camera = view.current;
    if (camera.upProjection[3] != 0.f) {
        return light.range * camera.viewport[2] <= maxRadius * camera.clipOrtho[2];
    }
    const Vector delta{light.position[0] - camera.eye[0], light.position[1] - camera.eye[1], light.position[2] - camera.eye[2]};
    const Vector forward = normalized({camera.center[0] - camera.eye[0], camera.center[1] - camera.eye[1],
        camera.center[2] - camera.eye[2]});
    const float depth = dot(delta, forward);
    const float nearest = depth - light.range;
    if (nearest <= camera.clipOrtho[0]) { return false; }
    // Bound the radius along either image axis. Off-axis spheres can have
    // larger projected extents than an on-axis sphere at the same view depth.
    const float offAxisScale = std::sqrt(dot(delta, delta)) / depth;
    const float screenRadius = light.range / nearest * offAxisScale * camera.viewport[2] /
        (2.f * std::tan(camera.viewport[3] * .5f));
    return screenRadius <= maxRadius;
}

std::vector<ShadowProjection> makeLocalProjections(const VirtualShadowMapLight& light,
    float farPlane, const VirtualShadowMapSettings& settings)
{
    Vector x, y, z;
    if (light.type == VirtualShadowMapLightType::PointFace) {
        constexpr std::array<Vector, 6> forward{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
        constexpr std::array<Vector, 6> right{{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}}};
        z = forward[light.face]; x = right[light.face]; y = cross(z, x);
    } else {
        z = normalized(light.direction);
        x = normalized(cross(std::abs(z[1]) < .95f ? Vector{0, 1, 0} : Vector{1, 0, 0}, z));
        y = cross(z, x);
    }
    const float tangent = light.type == VirtualShadowMapLightType::PointFace ? 1.f : std::tan(light.outerConeAngle);
    const float depthScale = -light.nearPlane / (farPlane - light.nearPlane);
    const float depthOffset = light.nearPlane * farPlane / (farPlane - light.nearPlane);
    std::vector<ShadowProjection> result;
    for (uint32_t level = 0; level <= settings.clipmapCount; ++level) {
        const bool coarse = level == settings.clipmapCount;
        const uint32_t axis = coarse ? 1u : std::max(1u, settings.virtualPagesPerAxis >> level);
        result.push_back({{x[0] / tangent, x[1] / tangent, x[2] / tangent, -dot(x, light.position) / tangent},
            {y[0] / tangent, y[1] / tangent, y[2] / tangent, -dot(y, light.position) / tangent},
            {z[0] * depthScale, z[1] * depthScale, z[2] * depthScale, depthOffset - depthScale * dot(z, light.position)},
            {z[0], z[1], z[2], -dot(z, light.position)},
            {tangent, 0, settings.shadowBias, float(light.type)},
            {level * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis, axis, level, coarse ? 1u : 0u},
            {light.position[0], light.position[1], light.position[2], light.nearPlane}});
    }
    return result;
}

bool compatibleLight(const VirtualShadowMapLight& a, const VirtualShadowMapLight& b)
{
    if (a.type != b.type || a.enabled != b.enabled) { return false; }
    if (a.type == VirtualShadowMapLightType::Directional) { return a.direction == b.direction; }
    return a.position == b.position && a.range == b.range && a.nearPlane == b.nearPlane &&
        (a.type == VirtualShadowMapLightType::PointFace ? a.face == b.face :
            a.direction == b.direction && a.outerConeAngle == b.outerConeAngle);
}

bool compatibleSettings(const VirtualShadowMapSettings& a, const VirtualShadowMapSettings& b)
{
    return a.pageSize == b.pageSize && a.virtualPagesPerAxis == b.virtualPagesPerAxis &&
        a.clipmapCount == b.clipmapCount && a.physicalPageCount == b.physicalPageCount &&
        a.firstClipmapExtent == b.firstClipmapExtent && a.shadowBias == b.shadowBias && a.enabled == b.enabled &&
        a.clusterHybridRaster == b.clusterHybridRaster && a.twoPassHZB == b.twoPassHZB &&
        a.hierarchyHZB == b.hierarchyHZB;
}

std::vector<std::array<int64_t, 2>> pageOrigins(const ViewConstants& view, Vector direction,
    const VirtualShadowMapSettings& settings)
{
    const auto z = normalized(direction);
    const auto x = normalized(cross(std::abs(z[1]) < .95f ? Vector{0, 1, 0} : Vector{1, 0, 0}, z));
    const auto y = cross(z, x);
    const Vector eye{view.current.eye[0], view.current.eye[1], view.current.eye[2]};
    std::vector<std::array<int64_t, 2>> origins;
    for (uint32_t level = 0; level < settings.clipmapCount; ++level) {
        const float pageWorldSize = 2 * std::ldexp(settings.firstClipmapExtent, int(level)) / settings.virtualPagesPerAxis;
        origins.push_back({int64_t(std::floor(dot(x, eye) / pageWorldSize)), int64_t(std::floor(dot(y, eye) / pageWorldSize))});
    }
    return origins;
}

struct ShadowAtlas
{
    std::shared_ptr<Texture> texture;
    std::shared_ptr<TextureView> view;
};

struct SharedShadowPoolStorage
{
    Device* device = nullptr;
    uint32_t capacity = 0, pageSize = 0, pagesX = 0, pagesY = 0, layers = 0;
    std::shared_ptr<ShadowAtlas> atlas, staticAtlas;
    std::vector<std::shared_ptr<Buffer>> hzbBanks;
    std::shared_ptr<Buffer> hzbTable;
    std::shared_ptr<Buffer> hybridDepth, rasterHZB;
    std::vector<ResourceLease> leases;
};

struct SharedShadowPoolFrame
{
    std::shared_ptr<SharedShadowPoolStorage> storage;
    std::shared_ptr<const std::vector<VirtualShadowMapCasterState>> casters;
    std::weak_ptr<void> casterOwner;
    std::shared_ptr<Buffer> metadata, history, jobs;
    std::shared_ptr<Buffer> batchJobs, batchArguments;
    std::shared_ptr<Buffer> publicationFaces;
    std::shared_ptr<Buffer> markGrid;
    std::vector<std::shared_ptr<void>> pipelineOwners;
    std::vector<ResourceLease> leases;
    std::shared_ptr<SubmissionTransaction> submission;
    GPUCompletionPoint completion;
    uint32_t faceCount = 0, frameIndex = 0;
    uint64_t allocationSerial = 0;
};

void setGridJobs(VirtualShadowMapProjectionGrid& grid, uint32_t slot, uint32_t first, uint32_t count)
{
    grid.jobs[slot] = {first, count};
    if (slot >= 2u) {
        const uint32_t local = slot - 2u;
        auto& mask = local < 32u ? grid.admittedLow : grid.admittedHigh;
        mask |= 1u << (local & 31u);
        grid.localJobCount += count;
    }
}

Result<> bindGridBuffer(ResourceRegistry& registry, std::vector<ResourceLease>& leases, Buffer& buffer, uint32_t& index)
{
    auto lease = registry.storageBuffer(buffer);
    if (!lease) { return std::unexpected(lease.error()); }
    index = lease->shaderIndex();
    leases.push_back(*lease);
    return {};
}

Result<> bindLightGrid(ResourceRegistry& registry, std::vector<ResourceLease>& leases,
    const ClusterLightGridSnapshot& snapshot, VirtualShadowMapProjectionGrid& grid)
{
    if (auto r = bindGridBuffer(registry, leases, *snapshot.parameters, grid.parameters); !r) { return r; }
    if (auto r = bindGridBuffer(registry, leases, *snapshot.cells, grid.cells); !r) { return r; }
    if (auto r = bindGridBuffer(registry, leases, *snapshot.lightIndices, grid.lightIndices); !r) { return r; }
    return bindGridBuffer(registry, leases, *snapshot.candidates, grid.candidates);
}

} // namespace

struct VirtualShadowMaps::CasterChangeContext
{
    static constexpr size_t kMissingCaster = std::numeric_limits<size_t>::max();
    struct Change
    {
        size_t previous, current;
        bool content = false, transform = false, material = false;
        bool lod = false, texture = false, tessellation = false;
    };
    struct Diff
    {
        std::vector<Change> changes;
        std::vector<scene::Bounds> dirtyBounds;
        std::vector<size_t> currentForPrevious;

        Diff(std::span<const VirtualShadowMapCasterState> previous, std::span<const VirtualShadowMapCasterState> current,
            bool layered)
        {
            if (layered) { currentForPrevious.resize(previous.size(), kMissingCaster); }
            size_t oldIndex = 0, newIndex = 0;
            while (oldIndex < previous.size() || newIndex < current.size()) {
                if (newIndex == current.size() || (oldIndex < previous.size() && previous[oldIndex].id < current[newIndex].id)) {
                    dirtyBounds.push_back(previous[oldIndex].bounds);
                    changes.push_back({oldIndex++, kMissingCaster});
                    continue;
                }
                if (oldIndex == previous.size() || current[newIndex].id < previous[oldIndex].id) {
                    dirtyBounds.push_back(current[newIndex].bounds);
                    changes.push_back({kMissingCaster, newIndex++});
                    continue;
                }
                const auto& old = previous[oldIndex];
                const auto& next = current[newIndex];
                if (layered) { currentForPrevious[oldIndex] = newIndex; }
                const bool displacement = (old.tessellationKey != 0 || next.tessellationKey != 0) &&
                    old.displacementMaterialKey != next.displacementMaterialKey;
                const bool content = old.contentKey != next.contentKey || displacement;
                const bool lod = old.lodKey != next.lodKey;
                const bool texture = old.textureKey != next.textureKey;
                const bool tessellation = old.tessellationKey != next.tessellationKey;
                if (content || lod || texture || tessellation) {
                    changes.push_back({oldIndex, newIndex, content, old.transformKey != next.transformKey,
                        old.materialKey != next.materialKey || displacement, lod, texture, tessellation});
                    dirtyBounds.push_back(old.bounds);
                    dirtyBounds.push_back(next.bounds);
                }
                ++oldIndex;
                ++newIndex;
            }
        }
    };
    std::unordered_map<const std::vector<VirtualShadowMapCasterState>*, Diff> diffs;
};

struct VirtualShadowMapPhysicalPool::Impl
{
    struct MarkPermutation
    {
        int pageShift;
        bool lightGrid, receiverMask, gpuWork;
        bool operator==(const MarkPermutation&) const = default;
    };
    struct MarkPipeline
    {
        MarkPermutation permutation;
        std::unique_ptr<ShaderModule> shader;
        std::unique_ptr<ComputePipeline> pipeline;
    };
    Device* device = nullptr;
    std::shared_ptr<SharedShadowPoolFrame> committed, pending, recording;
    std::array<std::shared_ptr<ShaderModule>, 4> shaders;
    std::array<std::shared_ptr<ComputePipeline>, 4> pipelines;
    std::vector<std::shared_ptr<SharedShadowPoolFrame>> frames;
    std::shared_ptr<MarkPipeline> viewMark;
    Result<> recordViewMark(CommandBuffer& commands, ResourceRegistry& registry, ShadowPush common, uint32_t jobTable,
        std::span<const VirtualShadowMapFaceRequest> faces, const VirtualShadowMapSettings& settings,
        const ClusterLightGridSnapshot* grid, const ViewConstants& view, std::string& log, CPUProfileRecorder* profiler);
};

struct VirtualShadowMaps::Programs
{
    Device* device = nullptr;
    uint64_t coverageKey = 0;
    bool collectReceiverCoverage = false;
    bool collectGPUWork = false;
    bool optimizeProjectionStatistics = false;
    bool groupProjectionStatistics = false;
    bool directional = false;
    bool adaptiveRayCount = false;
    bool localSMRTStateReuse = false;
    uint32_t markCacheLevels = 12;
    int markPageShift = -1;
    bool compactBuild = false;
    bool pageWindowRaster = false;
    bool clusterPageRaster = false;
    bool clusterHybridRaster = false;
    bool twoPassHZB = false;
    bool hierarchyHZB = false;
    bool instanceBuild = false;
    bool dirtyPageHierarchy = false;
    bool receiverMask = false;
    bool staticPageHZB = false;
    bool refinedReceiver = false;
    bool sharedPhysicalPool = false;
    bool batchBuildRaster = false;
    bool compactShadowViews = false;
    bool gpuLOD = false;
    std::array<std::unique_ptr<ShaderModule>, 30> shaders;
    std::array<std::unique_ptr<ComputePipeline>, 16> compute;
    std::unique_ptr<ComputePipeline> selectLOD;
    std::unique_ptr<ComputePipeline> hybridClear, hybridRaster;
    std::unique_ptr<ComputePipeline> buildRasterHZB, retestRasterHZB;
    std::unique_ptr<ComputePipeline> prepareHierarchyHZB, resetHierarchyPost, retestHierarchyHZB, finalizeHierarchyPost;
    std::array<std::unique_ptr<ShaderModule>, 2> onePassShaders;
    std::array<std::unique_ptr<ComputePipeline>, 2> onePassComputes;
    std::unique_ptr<GraphicsPipeline> raster;
    std::unique_ptr<GraphicsPipeline> packedRaster;
    std::unique_ptr<GraphicsPipeline> clear;
};

struct VirtualShadowMaps::Cache
{
    Device* device = nullptr;
    std::shared_ptr<ShadowAtlas> atlas;
    std::shared_ptr<ShadowAtlas> staticAtlas;
    std::shared_ptr<Buffer> staticHZB;
    struct DynamicCaster { size_t index; uint64_t lastChanged; };
    std::vector<DynamicCaster> dynamic; // Sorted indices into this submitted snapshot.
    uint64_t cacheFrame = 0, recordedFrame = 0;
    std::shared_ptr<Buffer> pages;
    std::shared_ptr<Buffer> lodSelection;
    uint64_t lodLayoutKey = 0;
    std::vector<ShadowProjection> projections;
    std::vector<std::array<int64_t, 2>> origins;
    std::weak_ptr<void> casterOwner;
    std::shared_ptr<const std::vector<VirtualShadowMapCasterState>> casters;
    VirtualShadowMapSettings settings;
    VirtualShadowMapLight light;
    float farPlane = 0;
    uint64_t sceneIdentity = 0, dependencyKey = 0;
    uint64_t geometryRevision = 0, texturePublicationRevision = 0, tessellationKey = 0;
    uint32_t epoch = 0;
    bool enabled = false;
    bool receiverMask = false;
    GPUCompletionPoint completion;
    std::shared_ptr<SubmissionTransaction> submission;
    std::weak_ptr<SharedShadowPoolStorage> physicalPoolStorage;
    uint64_t poolSerial = 0;
};

struct VirtualShadowMaps::ResourcePool
{
    struct BufferEntry
    {
        std::shared_ptr<Buffer> buffer;
        GPUCompletionPoint completion;
    };
    struct TextureEntry
    {
        ShadowAtlas resource;
        GPUCompletionPoint completion;
        uint32_t pageSize = 0;
    };
    Device* device = nullptr;
    std::array<std::vector<BufferEntry>, 25> buffers;
    std::array<std::vector<TextureEntry>, 4> textures;
    std::array<std::shared_ptr<TextureView>, 2> externalViews;
    VirtualShadowMapResourcePoolStats stats;

    Result<std::shared_ptr<Buffer>> buffer(size_t slot, const BufferDesc& desc, const GPUCompletionPoint& completion)
    {
        BufferEntry* vacant = nullptr;
        if (completion.valid()) {
            for (auto& entry : buffers[slot]) {
                // Descriptor leases retain allocations, not wrappers. Completion
                // covers those GPU users; wrapper owners protect cache readers
                // and deferred projection results independently of that point.
                if (entry.buffer.use_count() != 1 || !entry.completion.isComplete()) { continue; }
                const auto& existing = entry.buffer->desc();
                if (existing.size >= desc.size && existing.structureStride == desc.structureStride &&
                    existing.usage == desc.usage && existing.memoryLocation == desc.memoryLocation) {
                    entry.completion = completion;
                    ++stats.bufferReuses;
                    return entry.buffer;
                }
                vacant = &entry;
            }
        }
        if (vacant) { vacant->buffer.reset(); ++stats.bufferEvictions; }
        auto created = device->createBuffer(desc);
        if (!created) {
            if (vacant) {
                std::erase_if(buffers[slot], [](const auto& entry) { return !entry.buffer; });
            }
            return std::unexpected(created.error());
        }
        ++stats.bufferCreates;
        std::shared_ptr<Buffer> result = std::move(*created);
        if (completion.valid()) {
            if (vacant) { *vacant = {result, completion}; }
            else { buffers[slot].push_back({result, completion}); }
        }
        return result;
    }

    Result<ShadowAtlas> texture(size_t slot, const TextureDesc& desc, const GPUCompletionPoint& completion, uint32_t pageSize)
    {
        TextureEntry* vacant = nullptr;
        if (completion.valid()) {
            for (auto& entry : textures[slot]) {
                if (entry.resource.texture.use_count() != 1 || !entry.completion.isComplete()) { continue; }
                const auto& existing = entry.resource.texture->desc();
                if (existing.width == desc.width && existing.height == desc.height && existing.format == desc.format &&
                    existing.usage == desc.usage) {
                    entry.completion = completion;
                    entry.pageSize = pageSize;
                    ++stats.textureReuses;
                    return entry.resource;
                }
                vacant = &entry;
            }
        }
        if (vacant) { vacant->resource = {}; ++stats.textureEvictions; }
        auto created = device->createTexture(desc);
        if (!created) {
            if (vacant) {
                std::erase_if(textures[slot], [](const auto& entry) { return !entry.resource.texture; });
            }
            return std::unexpected(created.error());
        }
        ++stats.textureCreates;
        auto view = device->createTextureView(**created, {.format = desc.format});
        if (!view) {
            if (vacant) {
                std::erase_if(textures[slot], [](const auto& entry) { return !entry.resource.texture; });
            }
            return std::unexpected(view.error());
        }
        ++stats.viewCreates;
        ShadowAtlas result{std::move(*created), std::move(*view)};
        if (completion.valid()) {
            if (vacant) { *vacant = {result, completion, pageSize}; }
            else { textures[slot].push_back({result, completion, pageSize}); }
        }
        return result;
    }
};

struct VirtualShadowMaps::State
{
    // Backing owners contain no State/cache links. Results and projection
    // commands may retain these across faces without forming frame chains.
    struct Resources
    {
        // materials, page table, coverage, work, statistics, projections, view, indirect, reuse, owners.
        std::array<std::shared_ptr<Buffer>, 10> buffers;
        std::shared_ptr<Buffer> pointFaces;
        std::shared_ptr<Buffer> projectionJobs;
        std::shared_ptr<Buffer> projectionGrid;
        std::shared_ptr<Buffer> receiverGeometry;
        std::shared_ptr<Buffer> cacheDiagnostics;
        std::shared_ptr<Buffer> hybridDepth, rasterHZB, hierarchyHZBQueue;
        std::shared_ptr<ShadowAtlas> staticAtlas;
        std::shared_ptr<Buffer> staticHZB, staticHZBJobs, clusterVisibility;
        std::shared_ptr<Buffer> lodContext, lodGroupStates, lodSelection, lodChanges, lodStatistics;
        std::shared_ptr<VirtualShadowMapGPUHierarchy> gpuLOD;
        std::shared_ptr<SharedShadowPoolStorage> physicalPoolStorage;
        std::array<std::shared_ptr<void>, 3> casterLayers;
        uint64_t layerBufferBytes = 0;
        std::array<std::shared_ptr<Texture>, 3> textures; // atlas, shadow, debug
        std::array<std::shared_ptr<TextureView>, 3> views;
    };
    std::shared_ptr<Resources> resources = std::make_shared<Resources>();
    std::shared_ptr<ResourcePool> pool;
    ShadowPush push{};
    ViewConstants receiverView{};
    std::array<float, 3> projectionOrigin{};
    bool pointLightFaceSelection = false;
    std::shared_ptr<ResourceRegistry> registry;
    std::shared_ptr<Programs> programs;
    std::vector<ResourceLease> leases;
    std::array<Texture*, 2> outputTextures{};
    bool externalOutput = false;
    std::shared_ptr<Cache> previous;
    std::shared_ptr<Cache> next;
    bool projectionPending = true;
    struct BatchWork
    {
        uint32_t hierarchyGroups = 0, buildGroups = 0, compactBatches = 0;
        uint32_t hierarchyHZBBatches = 0;
        uint32_t visibilityBatches = 0, mergeDrawOffset = 0, hybridDrawOffset = 0;
        uint32_t staticGeometry = 0, dynamicGeometry = 0, hzbJobs = UINT32_MAX;
        bool mergeStatic = false;
    } batch;
    std::function<Result<>()> finish;
};

VirtualShadowMaps::VirtualShadowMaps() = default;

VirtualShadowMapPhysicalPool::VirtualShadowMapPhysicalPool() = default;
VirtualShadowMapPhysicalPool::~VirtualShadowMapPhysicalPool() = default;
VirtualShadowMapPhysicalPool::VirtualShadowMapPhysicalPool(VirtualShadowMapPhysicalPool&&) noexcept = default;
VirtualShadowMapPhysicalPool& VirtualShadowMapPhysicalPool::operator=(VirtualShadowMapPhysicalPool&&) noexcept = default;

struct VirtualShadowMapPressureFeedback::Slot
{
    std::shared_ptr<Buffer> fineDemand;
    std::shared_ptr<SubmissionTransaction> submission;
    GPUCompletionPoint completion;
    VirtualShadowMapPressureSample sample;
};

struct VirtualShadowMapRasterFeedback::Slot
{
    std::unique_ptr<TimestampQueryPool> queries;
    std::shared_ptr<SubmissionTransaction> submission;
    GPUCompletionPoint completion;
    uint64_t sequence = 0;
    uint32_t queryCount = 0, written = 0;
    float appliedBias = 0;
};

Result<> VirtualShadowMapRasterFeedback::prepare(Device& device, CommandBuffer& commands, uint32_t rangeCount, float appliedBias)
{
    auto* frame = RenderFrameContext::from(commands);
    assert(frame && frame->recording() && rangeCount != 0);
    const uint32_t queryCount = rangeCount * 2u;
    const auto available = std::ranges::find_if(slots_, [&](const auto& slot) {
        return !slot->submission || (slot->completion.isComplete() &&
            (slot->submission->cancelled() || slot->sequence <= consumedSequence_));
    });
    const auto slot = available == slots_.end() ? slots_.emplace_back(std::make_shared<Slot>()) : *available;
    if (!slot->queries || slot->queries->desc().queryCount < queryCount) {
        auto queries = device.createTimestampQueryPool(*device.getQueue(QueueType::Graphics), {.queryCount = queryCount});
        if (!queries) { return std::unexpected(queries.error()); }
        slot->queries = std::move(*queries);
    }
    auto transaction = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
    if (auto r = commands.retainResource(slot); !r) { return r; }
    if (auto r = commands.addSubmissionTransaction(transaction); !r) { return r; }
    slot->submission = std::move(transaction);
    slot->completion = frame->completion();
    slot->sequence = 0;
    slot->written = 0;
    slot->queryCount = queryCount;
    slot->appliedBias = appliedBias;
    recording_ = slot;
    return commands.resetTimestampQueries(*slot->queries, 0, queryCount);
}

Result<> VirtualShadowMapRasterFeedback::timestamp(CommandBuffer& commands)
{
    assert(recording_ && recording_->written < recording_->queryCount);
    auto& slot = *recording_;
    // Both timestamps wait for prior commands to finish, excluding the Build
    // that precedes each raster interval on the sequential path.
    if (auto r = commands.writeTimestamp(*slot.queries, slot.written, PipelineStageBits::BottomOfPipe); !r) { return r; }
    if (++slot.written == slot.queryCount) { slot.sequence = nextSequence_++; }
    return {};
}

Result<std::optional<VirtualShadowMapRasterSample>> VirtualShadowMapRasterFeedback::poll()
{
    Slot* newest = nullptr;
    uint64_t sequence = consumedSequence_;
    for (const auto& slot : slots_) {
        if (slot->sequence > sequence && slot->submission->resolved() && !slot->submission->cancelled() &&
            slot->completion.isSubmitted() && slot->completion.isComplete()) {
            newest = slot.get();
            sequence = slot->sequence;
        }
    }
    if (!newest) { return std::nullopt; }

    std::vector<TimestampQueryResult> results(newest->queryCount);
    if (auto r = newest->queries->readResults(0, results); !r) { return std::unexpected(r.error()); }
    consumedSequence_ = sequence;
    double milliseconds = 0;
    for (size_t i = 0; i < results.size(); i += 2) {
        const auto& begin = results[i];
        const auto& end = results[i + 1];
        if (!begin.available || !end.available) { return std::nullopt; }
        milliseconds += newest->queries->durationMilliseconds(begin.value, end.value);
    }
    return VirtualShadowMapRasterSample{float(milliseconds), newest->appliedBias};
}

Result<> VirtualShadowMapPressureFeedback::record(Device& device, CommandBuffer& commands,
    std::span<const VirtualShadowMapResult> faces, uint32_t capacity, float appliedBias,
    const VirtualShadowMapPhysicalPool* pool)
{
    auto* frame = RenderFrameContext::from(commands);
    if (!frame || !frame->recording()) { return makeError(Error::InvalidArgument); }
    Buffer* source = nullptr;
    uint64_t offset = 0;
    if (!faces.empty()) {
        if (pool) {
            if (!pool->impl_ || !pool->impl_->pending) { return makeError(Error::InvalidArgument); }
            const auto& allocation = pool->impl_->pending;
            if (allocation->faceCount != faces.size() || allocation->storage->capacity != capacity) {
                return makeError(Error::InvalidArgument);
            }
            source = allocation->metadata.get();
            offset = (1u + faces.size() + capacity * 2u) * 16u;
            if (auto r = commands.retainResource(allocation); !r) { return r; }
        } else {
            if (faces.size() != 1 || !faces.front().statistics) { return makeError(Error::InvalidArgument); }
            source = faces.front().statistics; // statistics[0] counts all unique requested fine pages.
            if (auto r = commands.retainResource(faces.front().resourceOwner); !r) { return r; }
        }
    }

    // A cancelled tail can share a frame completion with a submitted prefix.
    // Reuse the host buffer only after every accepted use has finished.
    auto available = std::ranges::find_if(slots_, [&](const auto& slot) {
        return !slot->submission || (slot->completion.isComplete() &&
            (slot->submission->cancelled() || slot->sample.sequence <= consumedSequence_));
    });
    const auto slot = available == slots_.end() ? slots_.emplace_back(std::make_shared<Slot>()) : *available;
    if (source && !slot->fineDemand) {
        auto buffer = device.createBuffer({.size = sizeof(uint32_t), .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback, .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
        if (!buffer) { return std::unexpected(buffer.error()); }
        slot->fineDemand = std::move(*buffer);
    }
    auto transaction = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
    if (auto r = commands.retainResource(slot); !r) { return r; }
    if (auto r = commands.addSubmissionTransaction(transaction); !r) { return r; }
    slot->submission = std::move(transaction);
    slot->completion = frame->completion();
    // The slot stays in flight even if recording the copy fails partway.
    slot->sample = {};
    if (source) {
        const BufferRange range{offset, sizeof(uint32_t)};
        auto from = source->slice(range);
        if (!from) { return std::unexpected(from.error()); }
        auto to = slot->fineDemand->slice();
        if (!to) { return std::unexpected(to.error()); }
        const BufferBarrierDesc before{.buffer = source,
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = range};
        if (auto r = commands.synchronize({.buffers = {&before, 1}}); !r) { return r; }
        if (auto r = commands.copyBuffer(*from, *to); !r) { return r; }
        const BufferBarrierDesc after[]{
            {.buffer = source, .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}, .range = range},
            {.buffer = slot->fineDemand.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}}};
        if (auto r = commands.synchronize({.buffers = after}); !r) { return r; }
    }
    slot->sample = {nextSequence_++, frame->frameIndex(), uint32_t(faces.size()), capacity, appliedBias};
    return {};
}

Result<std::optional<VirtualShadowMapPressureSample>> VirtualShadowMapPressureFeedback::poll()
{
    Slot* newest = nullptr;
    uint64_t sequence = consumedSequence_;
    for (const auto& slot : slots_) {
        if (slot->sample.sequence > sequence && slot->submission && slot->submission->resolved() &&
            !slot->submission->cancelled() && slot->completion.isSubmitted() && slot->completion.isComplete()) {
            newest = slot.get();
            sequence = slot->sample.sequence;
        }
    }
    if (!newest) { return std::optional<VirtualShadowMapPressureSample>{}; }
    auto sample = newest->sample;
    if (sample.requestedPages != 0) {
        auto* fine = static_cast<const uint32_t*>(newest->fineDemand->map());
        if (!fine) { return makeError(Error::Failure); }
        newest->fineDemand->invalidate();
        sample.requestedPages += *fine;
        newest->fineDemand->unmap();
    }
    consumedSequence_ = sequence;
    return std::optional{sample};
}

Result<> VirtualShadowMapPhysicalPool::Impl::recordViewMark(CommandBuffer& commands, ResourceRegistry& registry,
    ShadowPush common, uint32_t jobTable, std::span<const VirtualShadowMapFaceRequest> faces,
    const VirtualShadowMapSettings& settings, const ClusterLightGridSnapshot* grid, const ViewConstants& view,
    std::string& log, CPUProfileRecorder* profiler)
{
    CPUProfileScope scope(profiler, "Record VSM view mark");
    VirtualShadowMapProjectionGrid gridData{};
    gridData.diagnostics = UINT32_MAX;
    if (!device->capabilities().computeSubgroupBallotArithmetic) { grid = nullptr; }
    // The public shared-pool path also accepts arbitrary face order and faces
    // without a GPUScene identity. Those callers traverse the same jobs directly.
    for (uint32_t first = 0; grid && first < faces.size();) {
        const uint32_t slot = faces[first].sourceSlot;
        if (slot >= std::size(gridData.jobs) || gridData.jobs[slot].count != 0u) {
            grid = nullptr;
            break;
        }
        uint32_t end = first + 1;
        while (end < faces.size() && faces[end].sourceSlot == slot) { ++end; }
        setGridJobs(gridData, slot, first, end - first);
        first = end;
    }
    const MarkPermutation permutation{std::countr_zero(settings.pageSize), grid != nullptr,
        settings.cacheEnabled && settings.cacheLayering && settings.filter == VirtualShadowMapFilter::SMRT, settings.collectGPUWork};
    if (!viewMark || viewMark->permutation != permutation) {
        const auto shift = std::to_string(permutation.pageShift);
        const std::array macros{SlangMacroDefine{"VSM_VIEW_MARK", "1"}, SlangMacroDefine{"VSM_LOCAL_MARK", "1"},
            SlangMacroDefine{"VSM_MARK_PAGE_SHIFT", shift.c_str()}, SlangMacroDefine{"VSM_MARK_CACHE_LEVELS", "12"},
            SlangMacroDefine{"VSM_RECEIVER_MASK", permutation.receiverMask ? "1" : "0"},
            SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", permutation.gpuWork ? "1" : "0"},
            SlangMacroDefine{"VSM_LIGHT_GRID_PROJECTION", permutation.lightGrid ? "1" : "0"},
            SlangMacroDefine{"VSM_SHARED_PHYSICAL_POOL", "1"}};
        auto& shaders = ShaderRegistry::instance();
        auto shader = shaders.getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
            .entryPointName = "vsmMarkViewMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = macros}, log);
        if (!shader) { return std::unexpected(shader.error()); }
        auto module = shaders.getShaderModule(*device, {.spirv = shader->spirv, .debugName = "vsmMarkViewMain"});
        if (!module) { return std::unexpected(module.error()); }
        auto pipeline = shaders.getComputePipeline(*device, {.computeShader = {module->get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
        if (!pipeline) { return std::unexpected(pipeline.error()); }
        viewMark = std::make_shared<MarkPipeline>(MarkPipeline{permutation, std::move(*module), std::move(*pipeline)});
    }
    recording->pipelineOwners.push_back(viewMark);
    common.geometry = jobTable;
    common.triangleCount = uint32_t(faces.size());
    common.arguments = UINT32_MAX;
    if (grid) {
        if (auto r = bindLightGrid(registry, recording->leases, *grid, gridData); !r) { return r; }
        if (!recording->markGrid) {
            auto buffer = device->createBuffer({.size = sizeof(gridData), .structureStride = sizeof(gridData),
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
            if (!buffer) { return std::unexpected(buffer.error()); }
            recording->markGrid = std::move(*buffer);
        }
        auto* mapped = recording->markGrid->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, &gridData, sizeof(gridData));
        recording->markGrid->flush(); recording->markGrid->unmap();
        if (auto r = bindGridBuffer(registry, recording->leases, *recording->markGrid, common.arguments); !r) { return r; }
        if (auto r = registry.bind(commands, recording->leases); !r) { return r; }
        commands.hostWriteBarrier();
    }
    if (auto r = commands.bindExecution(viewMark->pipeline->execution(), &common, sizeof(common)); !r) { return r; }
    if (auto r = commands.dispatch((uint32_t(view.current.viewport[1]) + 7u) / 8u,
        (uint32_t(view.current.viewport[2]) + 7u) / 8u); !r) { return r; }
    scope.work("markDispatches", 1);
    scope.work("faces", faces.size());
    scope.work("lightGrid", grid != nullptr);
    return {};
}

Result<std::vector<VirtualShadowMapResult>> VirtualShadowMaps::recordShared(VirtualShadowMapPhysicalPool& physicalPool,
    Device& device, CommandBuffer& commands, TextureView& depth, const ViewConstants& view,
    std::span<const VirtualShadowMapFaceRequest> faces, const VirtualShadowMapCasterResources& casters,
    std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
    const VirtualShadowMapSettings& inputSettings, uint32_t capacity, bool demandDriven,
    std::string& log, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapReceiverResources* receiver, const ClusterLightGridSnapshot* markGrid,
    VirtualShadowMapRasterFeedback* rasterFeedback)
{
    auto* context = RenderFrameContext::from(commands);
    if (!context || !context->recording() || faces.empty() || faces.size() > 384 ||
        capacity < faces.size() || capacity > 262144 || !validProjectionSettings(inputSettings)) {
        log += "Shared VSM requires a recording frame, valid settings and enough physical pages for every face.\n";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().meshShader || !device.capabilities().shaderOutputLayer) {
        log += "Shared VSM requires mesh shaders with layered rendering support.\n";
        return makeError(Error::Unsupported);
    }
    std::unordered_set<VirtualShadowMaps*> uniqueFaces;
    for (const auto& face : faces) {
        if (!face.shadow || !face.light.enabled || !uniqueFaces.insert(face.shadow).second) {
            log += "Shared VSM faces must be enabled, unique cache objects.\n";
            return makeError(Error::InvalidArgument);
        }
    }
    if (!physicalPool.impl_ || physicalPool.impl_->device != &device) {
        physicalPool.impl_ = std::make_unique<VirtualShadowMapPhysicalPool::Impl>();
        physicalPool.impl_->device = &device;
    }
    auto& pool = *physicalPool.impl_;
    if (pool.pending) {
        if (pool.pending->submission->cancelled() || pool.pending->completion.isCancelled()) { pool.pending.reset(); }
        else if (pool.pending->submission->resolved() && pool.pending->completion.isSubmitted()) { pool.committed = std::move(pool.pending); }
        else {
            log += "Previous shared VSM recording must be submitted or cancelled.\n";
            return makeError(Error::InvalidArgument);
        }
    }
    auto registryResult = ResourceRegistry::forDevice(device);
    if (!registryResult) { return std::unexpected(registryResult.error()); }
    auto registry = *registryResult;
    const bool layered = inputSettings.cacheEnabled && inputSettings.cacheLayering;
    std::shared_ptr<SharedShadowPoolStorage> storage = pool.committed ? pool.committed->storage : nullptr;
    const bool reuseStorage = storage && storage->capacity == capacity && storage->pageSize == inputSettings.pageSize &&
        bool(storage->staticAtlas) == layered;
    if (pool.committed) {
        if (auto r = addCommandDependency(commands, pool.committed->completion); !r) { return std::unexpected(r.error()); }
    }
    if (!reuseStorage) {
        storage = std::make_shared<SharedShadowPoolStorage>();
        storage->device = &device; storage->capacity = capacity; storage->pageSize = inputSettings.pageSize;
        storage->pagesX = capacity > 4096 ? 64u : uint32_t(std::ceil(std::sqrt(float(capacity))));
        storage->pagesY = capacity > 4096 ? 64u : (capacity + storage->pagesX - 1u) / storage->pagesX;
        storage->layers = (capacity + 4095u) / 4096u;
        auto createAtlas = [&]() -> Result<std::shared_ptr<ShadowAtlas>> {
            auto texture = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferSource |
                TextureUsageBits::DepthStencilAttachment, .format = Format::D32Sfloat,
                .width = storage->pagesX * storage->pageSize, .height = storage->pagesY * storage->pageSize,
                .layerCount = storage->layers});
            if (!texture) { return std::unexpected(texture.error()); }
            auto atlasView = device.createTextureView(**texture, {.format = Format::D32Sfloat,
                .range = {0, 1, 0, storage->layers}, .forceArray = true});
            if (!atlasView) { return std::unexpected(atlasView.error()); }
            return std::make_shared<ShadowAtlas>(ShadowAtlas{std::move(*texture), std::move(*atlasView)});
        };
        auto atlas = createAtlas();
        if (!atlas) { return std::unexpected(atlas.error()); }
        storage->atlas = *atlas;
        if (layered) {
            atlas = createAtlas();
            if (!atlas) { return std::unexpected(atlas.error()); }
            storage->staticAtlas = *atlas;
            const uint32_t wordsPerPage = std::max(1u, (storage->pageSize * storage->pageSize - 1u) / 3u);
            std::vector<uint32_t> bankHandles;
            for (uint32_t first = 0; first < capacity; first += 4096u) {
                auto bank = device.createBuffer({.size = uint64_t(std::min(4096u, capacity - first)) * wordsPerPage * 4u,
                    .structureStride = 4, .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device});
                if (!bank) { return std::unexpected(bank.error()); }
                auto lease = registry->storageBuffer(**bank);
                if (!lease) { return std::unexpected(lease.error()); }
                bankHandles.push_back(lease->shaderIndex()); storage->leases.push_back(*lease);
                storage->hzbBanks.push_back(std::move(*bank));
            }
            auto table = device.createBuffer({.size = bankHandles.size() * sizeof(uint32_t), .structureStride = 4,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
            if (!table) { return std::unexpected(table.error()); }
            auto* mapped = (*table)->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, bankHandles.data(), bankHandles.size() * sizeof(uint32_t));
            (*table)->flush(); (*table)->unmap();
            storage->hzbTable = std::move(*table);
        }
    }
    pool.recording.reset();
    std::erase_if(pool.frames, [&](const auto& candidate) {
        return candidate.use_count() == 1 && candidate->completion.isComplete() && candidate->storage != storage;
    });
    for (auto& candidate : pool.frames) {
        if (candidate.use_count() == 1 && candidate->completion.isComplete()) {
            pool.recording = candidate;
            break;
        }
    }
    if (!pool.recording) {
        pool.recording = std::make_shared<SharedShadowPoolFrame>();
        pool.frames.push_back(pool.recording);
    }
    auto recording = pool.recording;
    recording->storage = storage; recording->leases.clear(); recording->pipelineOwners.clear();
    if (inputSettings.cacheEnabled) {
        recording->casters = pool.committed && casters.owner && pool.committed->casterOwner.lock() == casters.owner
            ? pool.committed->casters : std::make_shared<const std::vector<VirtualShadowMapCasterState>>(casters.states);
        recording->casterOwner = casters.owner;
    } else {
        recording->casters.reset();
        recording->casterOwner.reset();
    }
    recording->completion = context->completion();
    recording->submission = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
    recording->faceCount = uint32_t(faces.size()); recording->frameIndex = uint32_t(context->frameIndex());
    recording->allocationSerial = reuseStorage ? pool.committed->allocationSerial + 1u : 1u;
    auto acquire = [&](std::shared_ptr<Buffer>& buffer, uint64_t size, uint32_t stride, MemoryLocation location,
        BufferUsageBits extraUsage = BufferUsageBits::None) -> Result<> {
        if (!buffer || buffer->desc().size < size || buffer->desc().structureStride != stride ||
            !hasFlag(buffer->desc().usage, extraUsage)) {
            auto created = device.createBuffer({.size = size, .structureStride = stride,
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource | extraUsage, .memoryLocation = location});
            if (!created) { return std::unexpected(created.error()); }
            buffer = std::move(*created);
        }
        return {};
    };
    if (auto r = acquire(recording->metadata,
        uint64_t(vsmSharedPoolTotalMetadataQuads(uint32_t(faces.size()), capacity)) * 16u,
        16, MemoryLocation::Device, BufferUsageBits::Indirect); !r) { return std::unexpected(r.error()); }
    if (auto r = acquire(recording->history, uint64_t(capacity) * 16u, 8, MemoryLocation::Device); !r) { return std::unexpected(r.error()); }
    if (auto r = acquire(recording->jobs, faces.size() * sizeof(ShadowPush), sizeof(ShadowPush), MemoryLocation::HostUpload); !r) { return std::unexpected(r.error()); }
    const char* entries[]{"vsmSharedPoolPrepareMain", "vsmSharedPoolPhysicalMain", "vsmSharedPoolPublishMain", "vsmBatchPrepareMain"};
    for (size_t index = 0; index < (inputSettings.batchBuildRaster ? 4u : 3u); ++index) {
        if (!pool.pipelines[index]) {
            const std::array macros{SlangMacroDefine{"VSM_SHARED_PHYSICAL_POOL", "1"}};
            auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = macros}, log);
            if (!shader) { return std::unexpected(shader.error()); }
            auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entries[index]});
            if (!module) { return std::unexpected(module.error()); }
            auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {module->get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = index < 2 ? sizeof(VirtualShadowMapPhysicalPoolPush) :
                index == 2 ? sizeof(VirtualShadowMapPhysicalPoolPublishPush) : sizeof(VirtualShadowMapBatchPreparePush)});
            if (!pipeline) { return std::unexpected(pipeline.error()); }
            pool.shaders[index] = std::move(*module); pool.pipelines[index] = std::move(*pipeline);
        }
        recording->pipelineOwners.push_back(pool.shaders[index]);
        recording->pipelineOwners.push_back(pool.pipelines[index]);
    }
    if (auto r = commands.retainResource(recording); !r) { return std::unexpected(r.error()); }
    if (auto r = commands.addSubmissionTransaction(recording->submission); !r) { return std::unexpected(r.error()); }
    // Publish pending now: any failure still leaves a transaction that must be
    // cancelled with the failed recording, never a partially usable pool state.
    pool.pending = recording;
    std::vector<VirtualShadowMapResult> results;
    std::vector<std::shared_ptr<State>> states;
    auto settings = inputSettings;
    settings.unifiedMark = settings.unifiedMark && faces.size() > 1 && !settings.collectReceiverCoverage;
    settings.compactShadowViews = settings.compactShadowViews && settings.batchBuildRaster &&
        device.capabilities().computeSubgroupBallotArithmetic;
    settings.physicalPageCount = capacity;
    std::vector<BufferBarrierDesc> previousReady;
    for (const auto& bank : storage->hzbBanks) {
        previousReady.push_back({.buffer = bank.get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}});
    }
    if (reuseStorage) {
        previousReady.push_back({.buffer = pool.committed->history.get(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}});
    }
    if (!previousReady.empty()) {
        if (auto r = commands.synchronize({.buffers = previousReady}); !r) { return std::unexpected(r.error()); }
    }
    CasterChangeContext casterChanges;
    for (uint32_t index = 0; index < faces.size(); ++index) {
        const auto& face = faces[index];
        auto result = face.shadow->recordInternal(device, commands, depth, view, face.light, casters,
            materials, alphaTextures, settings, log, true, &face.output, profiler, gpuProfiler, receiver, &physicalPool, index,
            &casterChanges, rasterFeedback);
        if (!result) { return std::unexpected(result.error()); }
        results.push_back(*result); states.push_back(face.shadow->state_);
    }
    auto* jobs = static_cast<ShadowPush*>(recording->jobs->map());
    if (!jobs) { return makeError(Error::Failure); }
    for (size_t index = 0; index < states.size(); ++index) { jobs[index] = states[index]->push; }
    if (settings.unifiedMark && settings.pointLightFaceSelection) {
        for (size_t first = 0; first + 6 <= faces.size(); ++first) {
            if (jobs[first].projectionFace != 0u || jobs[first].lightType != uint32_t(VirtualShadowMapLightType::PointFace)) { continue; }
            const auto& light = faces[first];
            uint32_t face = 0;
            for (; face < 6; ++face) {
                const auto& candidate = faces[first + face];
                const auto& job = jobs[first + face];
                if (job.lightType != uint32_t(VirtualShadowMapLightType::PointFace) || job.projectionFace != face ||
                    job.filterMode != uint32_t(VirtualShadowMapFilter::Hard) ||
                    candidate.sourceSlot != light.sourceSlot || candidate.light.position != light.light.position) { break; }
            }
            // PCF/SMRT Mark must still request pages for samples on neighboring faces.
            if (face == 6) { jobs[first].projectionFlags |= kVSMPointGroupFirst; }
        }
    }
    recording->jobs->flush(); recording->jobs->unmap();
    VirtualShadowMapPhysicalPoolPush push{};
    push.faceCount = uint32_t(faces.size()); push.capacity = capacity;
    push.demandDriven = demandDriven; push.frameIndex = recording->frameIndex; push.previousHistory = UINT32_MAX;
    push.allocationSerialLow = uint32_t(recording->allocationSerial);
    push.allocationSerialHigh = uint32_t(recording->allocationSerial >> 32u);
    push.preserveDormant = settings.cacheEnabled && settings.maxLightCacheAge != 0;
    auto bindBuffer = [&](Buffer& buffer, uint32_t& handle) -> Result<> {
        auto lease = registry->storageBuffer(buffer);
        if (!lease) { return std::unexpected(lease.error()); }
        handle = lease->shaderIndex(); recording->leases.push_back(*lease);
        return {};
    };
    if (auto r = bindBuffer(*recording->jobs, push.jobs); !r) { return std::unexpected(r.error()); }
    if (auto r = bindBuffer(*recording->metadata, push.metadata); !r) { return std::unexpected(r.error()); }
    if (auto r = bindBuffer(*recording->history, push.history); !r) { return std::unexpected(r.error()); }
    if (reuseStorage) {
        if (auto r = bindBuffer(*pool.committed->history, push.previousHistory); !r) { return std::unexpected(r.error()); }
    }
    if (auto r = registry->bind(commands, recording->leases); !r) { return std::unexpected(r.error()); }
    auto publish = [&](bool activeViews = false) -> Result<> {
        std::vector<BufferBarrierDesc> barriers;
        auto append = [&](Buffer* buffer) { barriers.push_back({.buffer = buffer,
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}}); };
        append(recording->metadata.get()); append(recording->history.get());
        for (const auto& state : states) {
            for (size_t index : {1u, 4u, 9u}) { append(state->resources->buffers[index].get()); }
            if (activeViews) { append(state->resources->cacheDiagnostics.get()); }
        }
        return commands.synchronize({.buffers = barriers});
    };
    auto global = [&](size_t pipeline) -> Result<> {
        if (auto r = commands.bindExecution(pool.pipelines[pipeline]->execution(), &push, sizeof(push)); !r) { return r; }
        if (auto r = commands.dispatch(1); !r) { return r; }
        return publish();
    };
    auto perFace = [&](size_t pipeline, bool singleGroup = false, bool activeViews = false) -> Result<> {
        for (const auto& state : states) {
            if (auto r = commands.bindExecution(state->programs->compute[pipeline]->execution(), &state->push, sizeof(ShadowPush)); !r) { return r; }
            if (auto r = commands.dispatch(singleGroup ? 1u : (state->push.totalPages + 255u) / 256u); !r) { return r; }
        }
        return publish(activeViews);
    };
    commands.hostWriteBarrier();
    if (auto r = publish(); !r) { return std::unexpected(r.error()); }
    if (settings.unifiedMark) {
        auto markGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM view mark") : RenderGraphExecutionContext::ProfileScope{};
        if (std::ranges::any_of(states, [](const auto& state) { return state->push.clipmapCount != 0; })) {
            if (auto r = pool.recordViewMark(commands, *registry, states.front()->push, push.jobs, faces, settings,
                markGrid, view, log, profiler); !r) { return std::unexpected(r.error()); }
            if (auto r = publish(); !r) { return std::unexpected(r.error()); }
        }
        markGpu.next("VSM allocator count");
        if (auto r = perFace(6); !r) { return std::unexpected(r.error()); }
        markGpu.next("VSM allocator allocate");
        if (auto r = perFace(2, true); !r) { return std::unexpected(r.error()); }
    }
    auto allocationGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM shared physical pool") : RenderGraphExecutionContext::ProfileScope{};
    if (auto r = global(0); !r) { return std::unexpected(r.error()); }
    if (auto r = perFace(7, false, settings.compactShadowViews); !r) { return std::unexpected(r.error()); }
    if (auto r = global(1); !r) { return std::unexpected(r.error()); }
    if (auto r = perFace(9, false, settings.compactShadowViews); !r) { return std::unexpected(r.error()); }
    allocationGpu.end();
    if (!reuseStorage) {
        for (const auto& atlas : {storage->atlas, storage->staticAtlas}) {
            if (!atlas) { continue; }
            const TextureBarrierDesc begin{.texture = atlas->texture.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::DepthStencilAttachment,
                .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite}, .range = {0, 1, 0, storage->layers}};
            if (auto r = commands.synchronize({.textures = {&begin, 1}}); !r) { return std::unexpected(r.error()); }
            const RenderingAttachmentDesc attachment{.view = atlas->view.get(), .layout = TextureLayout::DepthStencilAttachment,
                .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = 0};
            if (auto r = commands.beginRendering({.renderArea = {.width = atlas->texture->desc().width, .height = atlas->texture->desc().height},
                .depthStencilAttachment = &attachment, .layerCount = storage->layers}); !r) { return std::unexpected(r.error()); }
            commands.endRendering();
            const TextureBarrierDesc ready{.texture = atlas->texture.get(), .oldLayout = TextureLayout::DepthStencilAttachment,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
                .after = {PipelineStageBits::ComputeShader | PipelineStageBits::FragmentShader, AccessBits::ShaderRead}, .range = {0, 1, 0, storage->layers}};
            if (auto r = commands.synchronize({.textures = {&ready, 1}}); !r) { return std::unexpected(r.error()); }
        }
    }
    if (settings.batchBuildRaster) {
        if (auto r = recordBatched(physicalPool, commands, states, settings, log, profiler, gpuProfiler, rasterFeedback); !r) {
            return std::unexpected(r.error());
        }
    } else {
        for (const auto& state : states) {
            if (auto r = state->finish(); !r) { return std::unexpected(r.error()); }
        }
    }
    for (const auto& state : states) { state->finish = {}; }
    if (casters.gpuLOD && casters.gpuLOD->stream) {
        if (auto r = casters.gpuLOD->stream->cmdCollectShadowPageRequests(commands,
            *casters.gpuLOD->pageRequests, casters.gpuLOD->pageCount); !r) { return std::unexpected(r.error()); }
    }
    pool.recording.reset();
    return results;
}

Result<> VirtualShadowMaps::recordBatched(VirtualShadowMapPhysicalPool& physicalPool, CommandBuffer& commands,
    std::span<const std::shared_ptr<State>> states, const VirtualShadowMapSettings& settings,
    std::string& log, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    VirtualShadowMapRasterFeedback* rasterFeedback)
{
    auto& pool = *physicalPool.impl_;
    auto& frame = *pool.recording;
    auto& storage = *frame.storage;
    auto& first = *states.front();
    const bool layered = bool(storage.staticAtlas);
    CPUProfileScope total(profiler, "Record VSM build and raster batches");
    uint64_t computeDispatches = 0, prepareDispatches = 0, meshDraws = 0, renderingPasses = 0;
    uint64_t computeBins = 0, rasterBins = 0;
    struct Bin
    {
        PreparedExecution execution;
        std::vector<ShadowBatchJob> jobs;
        uint32_t firstJob = 0, argumentWord = 0, upperBound = 0;
        bool indirect = false;
        bool compactRows = false;
    };
    using Stage = std::map<uint64_t, Bin>;
    Stage hierarchy, initialBuild, indirectBuild, finalize, propagate, hzb, dynamicCull;
    Stage prepareHierarchyHZB, hierarchyHistory, resetHierarchyPost, retestHierarchy, buildHierarchyPost,
        finalizeHierarchyPost, drawHierarchyPost, propagateHierarchyPost;
    std::array<Stage, 2> raster, classify, historyHZB, currentHZB, retest, postRaster;
    const auto add = [](Stage& stage, auto& pipeline, ShadowPush push, uint32_t upperBound,
        uint32_t countBuffer = UINT32_MAX, uint32_t countWord = 0) {
        if (!upperBound) { return; }
        auto& bin = stage[pipeline.psoHash()];
        bin.execution = pipeline.execution();
        bin.jobs.push_back({push, countBuffer, countWord, upperBound, 0});
        bin.upperBound = std::max(bin.upperBound, upperBound);
        bin.indirect |= countBuffer != UINT32_MAX;
    };
    bool hasCompact = false, hasInstance = false;
    const State* hybrid = nullptr;
    for (const auto& state : states) {
        const auto& programs = *state->programs;
        if (!hybrid && programs.clusterHybridRaster) { hybrid = state.get(); }
        const auto& work = state->batch;
        auto push = state->push;
        push.workBase = 0;
        const bool hierarchyRetest = programs.hierarchyHZB && (push.cacheMode & 1u) && push.enabled;
        hasCompact |= programs.compactBuild;
        hasInstance |= programs.instanceBuild;
        if (programs.dirtyPageHierarchy) { add(hierarchy, *programs.compute[12], push, work.hierarchyGroups); }
        if (push.enabled && work.buildGroups) {
            add(initialBuild, *programs.compute[programs.compactBuild ? 10 : 3], push, work.buildGroups);
            if (programs.compactBuild) {
                for (uint32_t batch = 0; batch < work.compactBatches; ++batch) {
                    push.workBase = batch * kDrawBatchSize;
                    add(indirectBuild, *programs.compute[3], push, kDrawBatchSize, push.arguments,
                        (push.batchCount + push.coarseBatchCount + batch) * 3u);
                    if (hierarchyRetest) {
                        add(buildHierarchyPost, *programs.compute[3], push, kDrawBatchSize, push.arguments,
                            (push.batchCount + push.coarseBatchCount + batch) * 3u);
                    }
                }
            }
        }
        push.workBase = 0;
        const uint32_t pageGroups = (push.totalPages + 63u) / 64u;
        add(finalize, *programs.compute[4], push, (std::max(push.totalPages, push.batchCount + push.coarseBatchCount) + 63u) / 64u);
        if (push.clipmapCount != 0) {
            add(hierarchyRetest ? propagateHierarchyPost : propagate, *programs.compute[11], push, pageGroups);
        }
        if (hierarchyRetest) {
            add(prepareHierarchyHZB, *programs.prepareHierarchyHZB, push, pageGroups);
            add(resetHierarchyPost, *programs.resetHierarchyPost, push, (work.compactBatches + 63u) / 64u);
            add(finalizeHierarchyPost, *programs.finalizeHierarchyPost, push, (std::max(push.totalPages, push.batchCount) + 63u) / 64u);
            const uint32_t maxPages = std::min(push.totalPages - 1u, push.physicalPageCount);
            const uint32_t pageBatches = (maxPages + kDrawBatchSize - 1u) / kDrawBatchSize;
            const uint32_t retestOffset = work.hybridDrawOffset + 6u * push.batchCount + 2u * pageBatches;
            for (uint32_t batch = 0; batch < work.hierarchyHZBBatches; ++batch) {
                push.workBase = batch * kDrawBatchSize;
                add(retestHierarchy, *programs.retestHierarchyHZB, push, kDrawBatchSize, push.arguments,
                    (retestOffset + batch) * 3u);
            }
        }
        if (programs.staticPageHZB) {
            add(hzb, *programs.compute[13], push, push.physicalPageCount, work.hzbJobs, 0);
        }
        for (uint32_t batch = 0; batch < work.visibilityBatches; ++batch) {
            push.geometry = work.dynamicGeometry;
            push.workBase = batch * kDrawBatchSize;
            add(dynamicCull, *programs.compute[14], push, kDrawBatchSize, work.hzbJobs,
                6u + push.physicalPageCount + batch * 3u);
        }
        if (!push.enabled) { continue; }
        for (uint32_t layer = 0; layer < (layered ? 2u : 1u); ++layer) {
            const bool staticPass = layered && layer == 0;
            push = state->push;
            push.geometry = layered ? (staticPass ? work.staticGeometry : work.dynamicGeometry) : push.geometry;
            const uint32_t rasterLayer = staticPass ? 2u : (work.visibilityBatches ? 4u : 0u);
            const uint32_t drawOffset = work.mergeStatic && !staticPass ? work.mergeDrawOffset : 0u;
            push.rasterCoarse = rasterLayer | 1u;
            for (uint32_t batch = 0; batch < push.coarseBatchCount; ++batch) {
                push.workBase = batch * kDrawBatchSize;
                add(raster[layer], *programs.raster, push, kDrawBatchSize, push.arguments,
                    (drawOffset + push.batchCount + batch) * 3u);
            }
            push.rasterCoarse = rasterLayer;
            auto& finePipeline = programs.packedRaster ? *programs.packedRaster : *programs.raster;
            const bool classifyFine = programs.clusterHybridRaster || programs.twoPassHZB;
            const uint32_t fineDrawOffset = classifyFine ?
                work.hybridDrawOffset + (staticPass ? push.batchCount : 0u) : drawOffset;
            const bool retestHistory = programs.twoPassHZB && (push.cacheMode & 1u);
            if (retestHistory) {
                const uint32_t maxPages = std::min(push.totalPages - 1u, push.physicalPageCount);
                const uint32_t pageBatches = (maxPages + kDrawBatchSize - 1u) / kDrawBatchSize;
                const uint32_t hzbOffset = work.hybridDrawOffset + 6u * push.batchCount + (staticPass ? pageBatches : 0u);
                for (uint32_t batch = 0; batch < pageBatches; ++batch) {
                    push.workBase = batch * kDrawBatchSize;
                    const uint32_t groups = std::min(kDrawBatchSize, maxPages - push.workBase);
                    add(hierarchyRetest ? hierarchyHistory : historyHZB[layer], *programs.buildRasterHZB, push, groups,
                        push.arguments, (hzbOffset + batch) * 3u);
                    auto current = push;
                    current.rasterCoarse |= 8u;
                    add(currentHZB[layer], *programs.buildRasterHZB, current, groups, push.arguments, (hzbOffset + batch) * 3u);
                }
            }
            for (uint32_t batch = 0; batch < push.batchCount; ++batch) {
                push.workBase = batch * kDrawBatchSize;
                add(raster[layer], finePipeline, push, kDrawBatchSize, push.arguments, (fineDrawOffset + batch) * 3u);
                if (classifyFine) {
                    add(classify[layer], *programs.hybridRaster, push, kDrawBatchSize,
                        push.arguments, (drawOffset + batch) * 3u);
                }
                if (retestHistory) {
                    add(retest[layer], *programs.retestRasterHZB, push, kDrawBatchSize, push.arguments,
                        (fineDrawOffset + 2u * push.batchCount + batch) * 3u);
                    add(postRaster[layer], finePipeline, push, kDrawBatchSize, push.arguments,
                        (fineDrawOffset + 4u * push.batchCount + batch) * 3u);
                }
                if (hierarchyRetest) {
                    auto recovered = push;
                    recovered.rasterCoarse |= kVSMHierarchyHZBPostRaster;
                    add(drawHierarchyPost, finePipeline, recovered, kDrawBatchSize, push.arguments, batch * 3u);
                }
            }
        }
    }
    std::vector<ShadowBatchJob> jobs;
    uint32_t argumentWords = 0;
    auto appendStage = [&](Stage& stage, bool mesh, bool compactRows = false) {
        for (auto& [key, bin] : stage) {
            bin.firstJob = uint32_t(jobs.size());
            bin.argumentWord = argumentWords;
            bin.compactRows = compactRows;
            jobs.insert(jobs.end(), bin.jobs.begin(), bin.jobs.end());
            const uint32_t rows = uint32_t(bin.jobs.size());
            if (mesh) { argumentWords += rows * 3u; }
            else if (bin.indirect || compactRows) {
                const uint32_t xBlocks = (bin.upperBound + kDrawBatchSize - 1u) / kDrawBatchSize;
                const uint32_t yBlocks = (rows + kDrawBatchSize - 1u) / kDrawBatchSize;
                argumentWords += xBlocks * (yBlocks * 3u + (compactRows ? rows : 0u));
            }
        }
    };
    for (auto* stage : {&hierarchy, &finalize, &propagate, &hzb, &dynamicCull,
             &prepareHierarchyHZB, &hierarchyHistory, &resetHierarchyPost,
             &finalizeHierarchyPost, &propagateHierarchyPost}) { appendStage(*stage, false); }
    for (auto* stage : {&initialBuild, &indirectBuild, &retestHierarchy, &buildHierarchyPost}) {
        appendStage(*stage, false, settings.compactShadowViews);
    }
    for (auto& stage : raster) { appendStage(stage, true); }
    for (auto& stage : postRaster) { appendStage(stage, true); }
    appendStage(drawHierarchyPost, true);
    for (auto* stages : {&classify, &historyHZB, &currentHZB, &retest}) {
        for (auto& stage : *stages) { appendStage(stage, false); }
    }
    auto acquire = [&](std::shared_ptr<Buffer>& buffer, uint64_t size, uint32_t stride, MemoryLocation memory) -> Result<> {
        if (!buffer || buffer->desc().size < size) {
            auto created = pool.device->createBuffer({.size = size, .structureStride = stride,
                .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect, .memoryLocation = memory});
            if (!created) { return std::unexpected(created.error()); }
            buffer = std::move(*created);
        }
        return {};
    };
    if (auto r = acquire(frame.batchJobs, jobs.size() * sizeof(ShadowBatchJob), sizeof(ShadowBatchJob), MemoryLocation::HostUpload); !r) { return r; }
    if (argumentWords) {
        if (auto r = acquire(frame.batchArguments, uint64_t(argumentWords) * 4u, 4, MemoryLocation::Device); !r) { return r; }
    }
    auto* mapped = frame.batchJobs->map();
    if (!mapped) { log += "Unable to map VSM batch jobs.\n"; return makeError(Error::Failure); }
    std::memcpy(mapped, jobs.data(), jobs.size() * sizeof(ShadowBatchJob));
    frame.batchJobs->flush(); frame.batchJobs->unmap();
    auto jobLease = first.registry->storageBuffer(*frame.batchJobs);
    if (!jobLease) { return std::unexpected(jobLease.error()); }
    frame.leases.push_back(*jobLease);
    uint32_t argumentHandle = UINT32_MAX;
    if (argumentWords) {
        auto lease = first.registry->storageBuffer(*frame.batchArguments);
        if (!lease) { return std::unexpected(lease.error()); }
        argumentHandle = lease->shaderIndex(); frame.leases.push_back(*lease);
        const BufferBarrierDesc ready{.buffer = frame.batchArguments.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
        if (auto r = commands.synchronize({.buffers = {&ready, 1}}); !r) { return r; }
    }
    if (auto r = first.registry->bind(commands, frame.leases); !r) { return r; }
    commands.hostWriteBarrier();
    const auto buffersReady = [&](std::initializer_list<size_t> indices, bool diagnostics, bool hzbJobs,
        PipelineStageBits stages = PipelineStageBits::ComputeShader,
        AccessBits access = AccessBits::ShaderRead | AccessBits::ShaderWrite) -> Result<> {
        std::vector<BufferBarrierDesc> barriers;
        auto append = [&](Buffer* buffer) {
            barriers.push_back({.buffer = buffer, .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {stages, access}});
        };
        for (const auto& state : states) {
            for (auto index : indices) { append(state->resources->buffers[index].get()); }
            if (diagnostics) { append(state->resources->cacheDiagnostics.get()); }
            if (hzbJobs && state->resources->staticHZBJobs) { append(state->resources->staticHZBJobs.get()); }
        }
        return commands.synchronize({.buffers = barriers});
    };
    auto banksReady = [&]() -> Result<> {
        std::vector<BufferBarrierDesc> barriers;
        for (const auto& bank : storage.hzbBanks) {
            barriers.push_back({.buffer = bank.get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
        }
        return barriers.empty() ? Result<>{} : commands.synchronize({.buffers = barriers});
    };
    auto prepare = [&](const Bin& bin, uint32_t groupBase, uint32_t outputWord, bool mesh,
        uint32_t firstRow, uint32_t rowCount, bool collectBuildRows = false) -> Result<> {
        const VirtualShadowMapBatchPreparePush push{jobLease->shaderIndex(), bin.firstJob + firstRow, rowCount,
            argumentHandle, outputWord, groupBase, mesh ? 1u : bin.compactRows ? kVSMBatchCompactBuild : 0u,
            collectBuildRows};
        if (auto r = commands.bindExecution(pool.pipelines[3]->execution(), &push, sizeof(push)); !r) { return r; }
        if (auto r = commands.dispatch(1); !r) { return r; }
        ++prepareDispatches;
        // Each bin writes a disjoint argument slice. Publish only that range;
        // later preparations never overwrite a draw already in the command stream.
        const uint32_t outputWords = mesh ? rowCount * 3u : 3u + (bin.compactRows ? rowCount : 0u);
        const BufferBarrierDesc ready{.buffer = frame.batchArguments.get(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = bin.compactRows ? SyncScope{PipelineStageBits::DrawIndirect | PipelineStageBits::ComputeShader,
                AccessBits::IndirectRead | AccessBits::ShaderRead} : SyncScope{PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
            .range = {uint64_t(outputWord) * 4u, uint64_t(outputWords) * 4u}};
        return commands.synchronize({.buffers = {&ready, 1}});
    };
    auto compute = [&](Stage& stage) -> Result<> {
        const bool collectBuildRows = &stage == &initialBuild && settings.collectGPUWork;
        for (const auto& [key, bin] : stage) {
            ++computeBins;
            uint32_t argument = bin.argumentWord;
            for (uint32_t row = 0; row < bin.jobs.size(); row += kDrawBatchSize) {
                const uint32_t rows = std::min(kDrawBatchSize, uint32_t(bin.jobs.size()) - row);
                for (uint32_t base = 0; base < bin.upperBound; base += kDrawBatchSize) {
                    auto push = first.push;
                    push.physicalPool = jobLease->shaderIndex(); push.poolFace = bin.firstJob + row; push.workBase = base;
                    if (bin.indirect || bin.compactRows) {
                        if (auto r = prepare(bin, base, argument, false, row, rows, collectBuildRows); !r) { return r; }
                        if (bin.compactRows) {
                            push.rasterCoarse |= kVSMBatchCompactRows;
                            push.arguments = argumentHandle;
                            push.geometry = argument + 3u;
                        }
                        if (auto r = commands.bindExecution(bin.execution, &push, sizeof(push)); !r) { return r; }
                        auto args = frame.batchArguments->slice({uint64_t(argument) * 4u, 12});
                        if (!args) { return std::unexpected(args.error()); }
                        if (auto r = commands.dispatchIndirect(*args); !r) { return r; }
                        argument += 3u + (bin.compactRows ? rows : 0u);
                    } else {
                        if (auto r = commands.bindExecution(bin.execution, &push, sizeof(push)); !r) { return r; }
                        if (auto r = commands.dispatch(std::min(kDrawBatchSize, bin.upperBound - base), rows); !r) { return r; }
                    }
                    ++computeDispatches;
                }
            }
        }
        return {};
    };
    auto buildRasterHZB = [&](Stage& stage) -> Result<> {
        BufferBarrierDesc pyramid{.buffer = storage.rasterHZB.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
        if (auto r = commands.synchronize({.buffers = {&pyramid, 1}}); !r) { return r; }
        if (auto r = buffersReady({4}, true, false); !r) { return r; }
        if (auto r = compute(stage); !r) { return r; }
        pyramid.before = pyramid.after;
        pyramid.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
        return commands.synchronize({.buffers = {&pyramid, 1}});
    };
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM build") : RenderGraphExecutionContext::ProfileScope{};
    if (!hierarchy.empty()) {
        auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM build page hierarchy") : RenderGraphExecutionContext::ProfileScope{};
        if (auto r = compute(hierarchy); !r) { return r; }
        if (auto r = buffersReady({9}, false, false); !r) { return r; }
    }
    if (!hierarchyHistory.empty()) {
        auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM hierarchy history HZB") : RenderGraphExecutionContext::ProfileScope{};
        if (auto r = compute(prepareHierarchyHZB); !r) { return r; }
        if (auto r = buffersReady({3, 7}, true, false); !r) { return r; }
        if (auto r = buildRasterHZB(hierarchyHistory); !r) { return r; }
    }
    auto buildGpu = gpuProfiler && settings.profileBuildSubstages ? gpuProfiler->profileScope(commands,
        hasInstance ? "VSM build instance cull" : hasCompact ? "VSM build compact" : "VSM build direct") : RenderGraphExecutionContext::ProfileScope{};
    if (auto r = compute(initialBuild); !r) { return r; }
    if (hasCompact) {
        buildGpu.next("VSM build compact sync");
        if (auto r = buffersReady({9, 4, 7}, false, false); !r) { return r; }
        buildGpu.next("VSM build indirect");
        if (auto r = compute(indirectBuild); !r) { return r; }
    }
    buildGpu.next("VSM build finalize sync");
    if (auto r = buffersReady({3, 4}, false, true); !r) { return r; }
    buildGpu.next("VSM build finalize");
    if (auto r = compute(finalize); !r) { return r; }
    if (auto r = buffersReady({1}, false, true); !r) { return r; }
    {
        auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM propagate mappings") : RenderGraphExecutionContext::ProfileScope{};
        if (auto r = compute(propagate); !r) { return r; }
    }
    if (auto r = buffersReady({3, 1, 9, 4, 7}, true, true,
        PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader | PipelineStageBits::ComputeShader,
        AccessBits::ShaderRead | AccessBits::ShaderWrite); !r) { return r; }
    const BufferBarrierDesc ownershipReady{.buffer = frame.metadata.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader | PipelineStageBits::DrawIndirect,
            AccessBits::ShaderRead | AccessBits::IndirectRead}};
    if (auto r = commands.synchronize({.buffers = {&ownershipReady, 1}}); !r) { return r; }
    buildGpu.end(); gpu.end();
    const uint32_t clearBatches = vsmSharedPoolClearBatches(storage.capacity);
    const uint32_t faceCount = frame.faceCount;
    // The clear list spans all faces; the software depth buffer belongs to the pool.
    const auto& clearState = hybrid ? *hybrid : first;
    gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM raster") : RenderGraphExecutionContext::ProfileScope{};
    if (rasterFeedback) {
        if (auto r = rasterFeedback->timestamp(commands); !r) { return r; }
    }
    for (uint32_t layer = 0; layer < (layered ? 2u : 1u); ++layer) {
        const bool staticPass = layered && layer == 0;
        auto layerGpu = gpuProfiler && layered ? gpuProfiler->profileScope(commands,
            staticPass ? "VSM static raster" : "VSM dynamic raster and restore") : RenderGraphExecutionContext::ProfileScope{};
        auto rasterGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM raster prepare") : RenderGraphExecutionContext::ProfileScope{};
        if (!historyHZB[layer].empty()) {
            rasterGpu.next("VSM history page HZB");
            if (auto r = buildRasterHZB(historyHZB[layer]); !r) { return r; }
        }
        BufferBarrierDesc scratch{.buffer = storage.hybridDepth.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
        if (hybrid) {
            rasterGpu.next("VSM cluster software raster");
            if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
            auto clearPush = hybrid->push;
            clearPush.rasterCoarse = staticPass ? 2u : 0u;
            for (uint32_t batch = 0; batch < clearBatches; ++batch) {
                clearPush.workBase = batch * kDrawBatchSize;
                if (auto r = commands.bindExecution(hybrid->programs->hybridClear->execution(), &clearPush, sizeof(clearPush)); !r) { return r; }
                const uint64_t argOffset = uint64_t(vsmSharedPoolClearArgumentQuad(faceCount, storage.capacity, staticPass ? 1u : 0u, batch)) * 16u;
                auto args = frame.metadata->slice({argOffset, 12u});
                if (!args) { return std::unexpected(args.error()); }
                if (auto r = commands.dispatchIndirect(*args); !r) { return r; }
                ++computeDispatches;
            }
            scratch.before = scratch.after;
            scratch.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
            if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
        }
        if (!classify[layer].empty()) {
            if (!hybrid) { rasterGpu.next("VSM main cluster HZB cull"); }
            if (auto r = buffersReady({3, 4, 7}, true, false); !r) { return r; }
            if (auto r = compute(classify[layer]); !r) { return r; }
            if (hybrid) {
                scratch.before = scratch.after;
                scratch.after = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead};
                if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
            }
            if (auto r = buffersReady({3, 4, 7}, true, false,
                PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader,
                AccessBits::ShaderRead | AccessBits::ShaderWrite); !r) { return r; }
            rasterGpu.next("VSM raster prepare");
        }
        const auto draw = [&](const Stage& stage, bool mainPass) -> Result<> {
            for (const auto& [key, bin] : stage) {
                if (auto r = prepare(bin, 0, bin.argumentWord, true, 0, uint32_t(bin.jobs.size())); !r) { return r; }
            }
            auto& atlas = staticPass ? *storage.staticAtlas : *storage.atlas;
            const TextureBarrierDesc begin{.texture = atlas.texture.get(), .oldLayout = TextureLayout::ShaderRead,
                .newLayout = TextureLayout::DepthStencilAttachment,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
                .range = {0, 1, 0, storage.layers}};
            if (auto r = commands.synchronize({.textures = {&begin, 1}}); !r) { return r; }
            const Rect area{.width = storage.pagesX * storage.pageSize, .height = storage.pagesY * storage.pageSize};
            const RenderingAttachmentDesc attachment{.view = atlas.view.get(), .layout = TextureLayout::DepthStencilAttachment,
                .loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
            if (auto r = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &attachment, .layerCount = storage.layers}); !r) { return r; }
            ++renderingPasses;
            if (auto r = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1}); !r) { return r; }
            commands.setScissor(area);
            auto common = clearState.push;
            if (mainPass) {
                rasterGpu.next("VSM raster clear and restore");
                common.rasterCoarse = staticPass ? 2u : 0u;
                for (uint32_t batch = 0; batch < clearBatches; ++batch) {
                    common.workBase = batch * kDrawBatchSize;
                    if (auto r = commands.bindExecution(clearState.programs->clear->execution(), &common, sizeof(common)); !r) { return r; }
                    const uint64_t argOffset = uint64_t(vsmSharedPoolClearArgumentQuad(faceCount, storage.capacity, staticPass ? 1u : 0u, batch)) * 16u;
                    auto args = frame.metadata->slice({argOffset, 12u});
                    if (!args) { return std::unexpected(args.error()); }
                    if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return r; }
                    ++meshDraws;
                }
            }
            rasterGpu.next(mainPass ? "VSM raster coarse and fine batches" : "VSM post raster fine batches");
            common.physicalPool = jobLease->shaderIndex();
            for (const auto& [key, bin] : stage) {
                ++rasterBins;
                const uint32_t limit = pool.device->capabilities().maxDrawIndirectCount;
                for (uint32_t start = 0; start < bin.jobs.size();) {
                    const uint32_t count = std::min(limit, uint32_t(bin.jobs.size()) - start);
                    common.poolFace = bin.firstJob + start;
                    if (auto r = commands.bindExecution(bin.execution, &common, sizeof(common)); !r) { return r; }
                    auto args = frame.batchArguments->slice({uint64_t(bin.argumentWord + start * 3u) * 4u, uint64_t(count) * 12u});
                    if (!args) { return std::unexpected(args.error()); }
                    if (auto r = commands.drawMeshTasksIndirect(*args, count); !r) { return r; }
                    ++meshDraws; start += count;
                }
            }
            commands.endRendering();
            rasterGpu.next("VSM raster publish");
            const TextureBarrierDesc ready{.texture = atlas.texture.get(), .oldLayout = TextureLayout::DepthStencilAttachment,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
                .after = {PipelineStageBits::ComputeShader | PipelineStageBits::FragmentShader, AccessBits::ShaderRead},
                .range = {0, 1, 0, storage.layers}};
            return commands.synchronize({.textures = {&ready, 1}});
        };
        if (auto r = draw(raster[layer], true); !r) { return r; }
        if (!retest[layer].empty()) {
            rasterGpu.next("VSM current page HZB");
            if (auto r = buildRasterHZB(currentHZB[layer]); !r) { return r; }
            rasterGpu.next("VSM post cluster HZB retest");
            if (auto r = buffersReady({3, 4, 7}, true, false); !r) { return r; }
            if (auto r = compute(retest[layer]); !r) { return r; }
            if (auto r = buffersReady({3, 4, 7}, true, false,
                PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader,
                AccessBits::ShaderRead | AccessBits::ShaderWrite); !r) { return r; }
            if (auto r = draw(postRaster[layer], false); !r) { return r; }
        }
        if (!retestHierarchy.empty()) {
            rasterGpu.next("VSM hierarchy post retest");
            std::vector<BufferBarrierDesc> queues;
            for (const auto& state : states) {
                if (const auto& queue = state->resources->hierarchyHZBQueue) {
                    queues.push_back({.buffer = queue.get(),
                        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}});
                }
            }
            if (auto r = commands.synchronize({.buffers = queues}); !r) { return r; }
            if (auto r = buffersReady({3, 4, 7, 9}, true, false); !r) { return r; }
            if (auto r = compute(resetHierarchyPost); !r) { return r; }
            if (auto r = buffersReady({7, 9}, true, false); !r) { return r; }
            if (auto r = compute(retestHierarchy); !r) { return r; }
            if (auto r = buffersReady({4, 7, 9}, true, false); !r) { return r; }
            rasterGpu.next("VSM hierarchy post build");
            if (auto r = compute(buildHierarchyPost); !r) { return r; }
            if (auto r = buffersReady({3, 4, 7}, false, false); !r) { return r; }
            if (auto r = compute(finalizeHierarchyPost); !r) { return r; }
            if (auto r = buffersReady({1, 3, 7}, true, false,
                PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader,
                AccessBits::ShaderRead | AccessBits::ShaderWrite); !r) { return r; }
            if (auto r = draw(drawHierarchyPost, false); !r) { return r; }
            rasterGpu.next("VSM hierarchy post propagate mappings");
            if (auto r = compute(propagateHierarchyPost); !r) { return r; }
            if (auto r = buffersReady({1}, false, false, PipelineStageBits::ComputeShader | PipelineStageBits::Transfer,
                AccessBits::ShaderRead | AccessBits::TransferRead); !r) { return r; }
        }
        rasterGpu.end(); layerGpu.end();
        if (staticPass) {
            if (auto r = buffersReady({4}, true, true); !r) { return r; }
            if (auto r = banksReady(); !r) { return r; }
            {
                auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM static page HZB") : RenderGraphExecutionContext::ProfileScope{};
                if (auto r = compute(hzb); !r) { return r; }
            }
            if (auto r = banksReady(); !r) { return r; }
            if (auto r = buffersReady({4}, true, false); !r) { return r; }
            std::vector<BufferBarrierDesc> masks;
            for (const auto& state : states) {
                if (state->batch.visibilityBatches) {
                    masks.push_back({.buffer = state->resources->clusterVisibility.get(),
                        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}});
                }
            }
            if (!masks.empty()) {
                if (auto r = commands.synchronize({.buffers = masks}); !r) { return r; }
                auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM dynamic cluster HZB cull") : RenderGraphExecutionContext::ProfileScope{};
                if (auto r = compute(dynamicCull); !r) { return r; }
                for (auto& mask : masks) {
                    mask.before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite};
                    mask.after = {PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
                }
                if (auto r = commands.synchronize({.buffers = masks}); !r) { return r; }
            }
            if (auto r = buffersReady({4}, true, false,
                PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader,
                AccessBits::ShaderRead | AccessBits::ShaderWrite); !r) { return r; }
        }
    }
    if (rasterFeedback) {
        if (auto r = rasterFeedback->timestamp(commands); !r) { return r; }
    }
    {
        auto scope = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM raster diagnostics publish") : RenderGraphExecutionContext::ProfileScope{};
        if (auto r = buffersReady({4}, true, false, PipelineStageBits::ComputeShader | PipelineStageBits::Transfer,
            AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::TransferRead); !r) { return r; }
    }
    total.work("faces", states.size()); total.work("computeDispatches", computeDispatches);
    total.work("prepareDispatches", prepareDispatches); total.work("meshDraws", meshDraws);
    total.work("renderingPasses", renderingPasses); total.work("computeBins", computeBins); total.work("rasterBins", rasterBins);
    return {};
}

Result<> VirtualShadowMapPhysicalPool::publishAllocations(CommandBuffer& commands, Buffer& lights,
    Buffer& budgetDiagnostics, std::span<const uint32_t> publicationFaces, std::string& log)
{
    if (!impl_ || !impl_->pending || publicationFaces.size() != impl_->pending->faceCount) {
        log += "VSM allocation publication must match the recorded shared faces.\n";
        return makeError(Error::InvalidArgument);
    }
    auto frame = impl_->pending;
    auto registryResult = ResourceRegistry::forDevice(*impl_->device);
    if (!registryResult) { return std::unexpected(registryResult.error()); }
    auto registry = *registryResult;
    const uint64_t bytes = publicationFaces.size_bytes();
    if (!frame->publicationFaces || frame->publicationFaces->desc().size < bytes) {
        auto buffer = impl_->device->createBuffer({.size = bytes, .structureStride = 4,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
        if (!buffer) { return std::unexpected(buffer.error()); }
        frame->publicationFaces = std::move(*buffer);
    }
    auto* mapped = frame->publicationFaces->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, publicationFaces.data(), bytes);
    frame->publicationFaces->flush(); frame->publicationFaces->unmap();
    VirtualShadowMapPhysicalPoolPublishPush push{};
    push.faceCount = frame->faceCount; push.frameIndex = frame->frameIndex;
    Buffer* buffers[]{frame->jobs.get(), frame->metadata.get(), frame->publicationFaces.get(), &lights, &budgetDiagnostics};
    uint32_t* handles[]{&push.jobs, &push.metadata, &push.publicationFaces, &push.lights, &push.budgetDiagnostics};
    std::vector<ResourceLease> leases;
    for (size_t index = 0; index < std::size(buffers); ++index) {
        auto lease = registry->storageBuffer(*buffers[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *handles[index] = lease->shaderIndex(); leases.push_back(*lease);
    }
    if (auto r = commands.retainResource(frame); !r) { return r; }
    if (auto r = registry->bind(commands, leases); !r) { return r; }
    commands.hostWriteBarrier();
    const BufferBarrierDesc before[]{
        {.buffer = frame->metadata.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
        {.buffer = &lights, .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}},
        {.buffer = &budgetDiagnostics, .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}}};
    if (auto r = commands.synchronize({.buffers = before}); !r) { return r; }
    if (auto r = commands.bindExecution(impl_->pipelines[2]->execution(), &push, sizeof(push)); !r) { return r; }
    if (auto r = commands.dispatch(1); !r) { return r; }
    const BufferBarrierDesc after[]{
        {.buffer = &lights, .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead}},
        {.buffer = &budgetDiagnostics, .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead}}};
    return commands.synchronize({.buffers = after});
}

VirtualShadowMapMemoryUsage VirtualShadowMapPhysicalPool::memoryUsage() const
{
    VirtualShadowMapMemoryUsage usage;
    if (!impl_) { return usage; }
    std::unordered_set<const Texture*> textures;
    std::unordered_set<const Buffer*> buffers;
    auto addBuffer = [&](const std::shared_ptr<Buffer>& buffer) {
        if (buffer && buffers.insert(buffer.get()).second) { usage.bufferBytes += buffer->desc().size; }
    };
    for (const auto& frame : impl_->frames) {
        if (frame->storage) {
            const auto& storage = *frame->storage;
            for (const auto& atlas : {storage.atlas, storage.staticAtlas}) {
                if (!atlas || !textures.insert(atlas->texture.get()).second) { continue; }
                const auto& desc = atlas->texture->desc();
                const uint64_t bytes = uint64_t(desc.width) * desc.height * desc.layerCount * 4;
                usage.atlasBytes += bytes;
                if (atlas == storage.staticAtlas) { usage.staticAtlasBytes += bytes; }
                usage.allocatedAtlasPages += uint64_t(desc.width / storage.pageSize) * (desc.height / storage.pageSize) * desc.layerCount;
            }
            for (const auto& bank : storage.hzbBanks) { addBuffer(bank); }
            addBuffer(storage.hzbTable);
            addBuffer(storage.hybridDepth);
            addBuffer(storage.rasterHZB);
        }
        addBuffer(frame->metadata); addBuffer(frame->history); addBuffer(frame->jobs); addBuffer(frame->publicationFaces);
        addBuffer(frame->batchJobs); addBuffer(frame->batchArguments); addBuffer(frame->markGrid);
    }
    return usage;
}

Result<std::vector<VirtualShadowMapLODProjection>> VirtualShadowMaps::makeShadowLODProjections(
    const scene::Bounds& bounds, const ViewConstants& view, const VirtualShadowMapLight& light,
    const VirtualShadowMapSettings& settings, std::string& log)
{
    if (!validProjectionSettings(settings)) {
        log += "Invalid VSM shadow LOD projection settings.\n";
        return makeError(Error::InvalidArgument);
    }
    if (auto valid = validateProjectionLight(light, log); !valid) { return std::unexpected(valid.error()); }
    if (!light.enabled || !settings.enabled) { return std::vector<VirtualShadowMapLODProjection>{}; }
    const bool directional = light.type == VirtualShadowMapLightType::Directional;
    auto faceSettings = settings;
    if (useDistantLightSinglePage(view, light, settings)) { faceSettings.clipmapCount = 0; }
    // Only X/Y/W are consumed by LOD; local far depth does not affect them.
    auto projections = directional ? makeProjections(bounds, view, light.direction, settings) :
        makeLocalProjections(light, light.nearPlane * 2, faceSettings);
    if (directional && bounds.valid) {
        auto& coarse = projections.back();
        const auto [low, high] = projectedBounds(bounds, coarse);
        const float minimumExtent = std::max(.01f, .5f * coarse.metric[0] *
            std::max(high[0] - low[0], high[1] - low[1]));
        // A reused coarse cache may have a smaller padded domain than a newly
        // constructed projection. Every valid cache contains these full bounds;
        // their unpadded extent conservatively bounds its texel density.
        const float scale = coarse.metric[0] / minimumExtent;
        for (auto& component : coarse.rowX) { component *= scale; }
        for (auto& component : coarse.rowY) { component *= scale; }
    }
    std::vector<VirtualShadowMapLODProjection> result;
    result.reserve(projections.size());
    for (const auto& projection : projections) {
        result.push_back({projection.rowX, projection.rowY, projection.rowW,
            projection.page[1] * settings.pageSize, directional ? 0.f : light.nearPlane});
    }
    return result;
}
VirtualShadowMaps::~VirtualShadowMaps() = default;

bool VirtualShadowMaps::resolvePendingCache()
{
    if (!pending_) { return false; }
    if (pending_->submission->cancelled() || pending_->completion.isCancelled()) {
        pending_.reset();
        return true;
    }
    if (pending_->submission->resolved() && pending_->completion.isSubmitted()) {
        cache_ = std::move(pending_);
    }
    return false;
}

bool VirtualShadowMaps::retainInactiveCache(uint64_t frameIndex, uint32_t maxAge)
{
    resolvePendingCache();
    if (!pending_ && (!cache_ || cache_->physicalPoolStorage.expired() || frameIndex - cache_->recordedFrame > maxAge)) {
        return false;
    }
    state_.reset();
    pool_.reset();
    return true;
}

Result<VirtualShadowMapResult> VirtualShadowMaps::record(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection,
    const VirtualShadowMapProjectionTarget* output, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapReceiverResources* receiver, VirtualShadowMapRasterFeedback* rasterFeedback)
{
    CPUProfileScope total(profiler, "VSM sun record");
    VirtualShadowMapLight light;
    light.enabled = (sun.flags & (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow)) ==
        (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow);
    if (light.enabled) { light.direction = {sun.direction[0], sun.direction[1], sun.direction[2]}; }
    light.angularRadius = sun.angularRadius;
    return record(device, commands, depth, view, light, casters, materials, alphaTextures, settings, log, deferProjection,
        output, profiler, gpuProfiler, receiver, rasterFeedback);
}

Result<VirtualShadowMapResult> VirtualShadowMaps::record(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& inputLight,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection,
    const VirtualShadowMapProjectionTarget* output, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapReceiverResources* receiver, VirtualShadowMapRasterFeedback* rasterFeedback)
{
    return recordInternal(device, commands, depth, view, inputLight, casters, materials, alphaTextures,
        settings, log, deferProjection, output, profiler, gpuProfiler, receiver, nullptr, 0, nullptr, rasterFeedback);
}

Result<VirtualShadowMapResult> VirtualShadowMaps::recordInternal(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& inputLight,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& requestedSettings, std::string& log, bool deferProjection,
    const VirtualShadowMapProjectionTarget* output, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapReceiverResources* receiver, VirtualShadowMapPhysicalPool* physicalPool, uint32_t poolFace,
    CasterChangeContext* casterChangeContext, VirtualShadowMapRasterFeedback* rasterFeedback)
{
    const bool sharedPhysicalPool = physicalPool != nullptr;
    const bool compactShadowViews = sharedPhysicalPool && requestedSettings.compactShadowViews;
    const auto poolFrame = sharedPhysicalPool ? physicalPool->impl_->recording : nullptr;
    auto settings = requestedSettings;
    const bool gpuLOD = bool(casters.gpuLOD);
    CPUProfileScope total(profiler, "VSM face record");
    total.work("lightType", uint32_t(inputLight.type));
    total.work("face", inputLight.face);
    total.work("casters", casters.states.size());
    total.work("triangles", casters.triangleCapacity);
    CPUProfileScope phase(profiler, "Validate VSM inputs");
    if (!device.capabilities().meshShader) { return makeError(Error::Unsupported); }
    // Scene geometry uses material slot zero when no material was authored,
    // matching the renderer's ordinary default-material publication.
    static const scene::RenderMaterial defaultMaterial;
    if (materials.empty()) { materials = {&defaultMaterial, 1}; }
    total.work("materials", materials.size());
    if (!validProjectionSettings(settings) || settings.physicalPageCount == 0 || settings.physicalPageCount > (sharedPhysicalPool ? 262144u : 4096u) ||
        settings.maxWorkItems == 0 || settings.maxWorkItems > 4 * 1024 * 1024 ||
        uint32_t(settings.filter) > uint32_t(VirtualShadowMapFilter::SMRT) ||
        !std::isfinite(settings.pcfRadius) || settings.pcfRadius < 0 || settings.pcfRadius > 8 ||
        settings.smrtRayCount == 0 || settings.smrtRayCount > 32 ||
        settings.smrtSamplesPerRay == 0 || settings.smrtSamplesPerRay > 64 ||
        !std::isfinite(settings.smrtMaxRayLength) || settings.smrtMaxRayLength <= 0 || settings.smrtMaxRayLength > 10000 ||
        !std::isfinite(settings.smrtRayLengthScale) || settings.smrtRayLengthScale < 0 || settings.smrtRayLengthScale > 1000 ||
        !std::isfinite(settings.smrtMaxRayAngle) || settings.smrtMaxRayAngle < 0.001f || settings.smrtMaxRayAngle > 0.35f ||
        (!alphaTextures.empty() && alphaTextures.size() != materials.size()) ||
        view.current.viewport[1] < 1 || view.current.viewport[2] < 1 ||
        (view.current.clipOrtho[3] != 0 && view.current.clipOrtho[3] != 1)) {
        log += "Invalid VSM settings, material bindings or receiver depth convention.\n";
        return makeError(Error::InvalidArgument);
    }
    auto light = inputLight;
    if (auto valid = validateProjectionLight(light, log); !valid) { return std::unexpected(valid.error()); }
    const bool directional = light.type == VirtualShadowMapLightType::Directional;
    const bool singlePage = useDistantLightSinglePage(view, light, settings);
    if (singlePage) { settings.clipmapCount = 0; }
    const auto& caps = device.capabilities();
    const bool clusterHybridRaster = !singlePage && settings.clusterHybridRaster && caps.computeSubgroupBallotArithmetic &&
        caps.subPixelPrecisionBits >= 1 && caps.subPixelPrecisionBits <= 8 && !casters.displacedGeometry;
    settings.clusterHybridRaster = clusterHybridRaster;
    const bool hierarchyHZB = settings.hierarchyHZB && sharedPhysicalPool && settings.batchBuildRaster &&
        settings.cacheEnabled && !settings.cacheLayering && directional &&
        casters.hierarchyRootCount != 0u && caps.computeSubgroupBallotArithmetic && !casters.displacedGeometry;
    settings.hierarchyHZB = hierarchyHZB;
    const bool twoPassHZB = !singlePage && (settings.twoPassHZB || hierarchyHZB) &&
        caps.computeSubgroupBallotArithmetic && !casters.displacedGeometry;
    settings.twoPassHZB = twoPassHZB;
    const bool classifyFine = clusterHybridRaster || twoPassHZB;
    const bool enabled = settings.enabled && light.enabled;
    if (settings.collectReceiverCoverage && (!directional || !enabled || !settings.collectGPUWork ||
        settings.filter != VirtualShadowMapFilter::SMRT || settings.smrtRayCount != 2)) {
        log = "Receiver coverage diagnostics require enabled directional SMRT with two rays, ray-page marking and GPU work counters";
        return makeError(Error::InvalidArgument);
    }
    const bool adaptiveRayCount = device.capabilities().computeSubgroupBallotArithmetic;
    const uint32_t markCacheLevels = directional ? settings.clipmapCount : 12u;
    // validProjectionSettings guarantees a nonzero power-of-two page size, at most 128.
    const int markPageShift = directional ? std::countr_zero(settings.pageSize) : -1;
    const bool instanceBuild = directional && (settings.instanceBuild || hierarchyHZB) && casters.instanceTable != UINT32_MAX;
    const bool compactBuild = !singlePage && (settings.compactBuild || instanceBuild);
    // Keep scalar rasterization for nonbinary grids: window rasterization at
    // 127 pages per axis changed D32 values after clipmap scrolling.
    // A uint4 header, one uint4 per window and four per admitted triangle.
    // Keep the bindless view inside Vulkan's guaranteed storage-buffer range.
    const uint64_t windowBytes = 16u + uint64_t(settings.maxWorkItems) * 16u +
        uint64_t(std::min(casters.triangleCapacity, settings.maxWorkItems)) * 64u;
    const bool pageWindowRaster = !classifyFine && directional && settings.pageWindowRaster &&
        std::has_single_bit(settings.virtualPagesPerAxis) && windowBytes <= (128ull << 20);
    const bool clusterPageRaster = classifyFine ||
        (!singlePage && !settings.pageWindowRaster && settings.clusterPageRaster && caps.computeSubgroupBallotArithmetic);
    const bool groupedFineRaster = pageWindowRaster || clusterPageRaster;
    const bool dirtyPageHierarchy = !singlePage && (settings.dirtyPageHierarchy || instanceBuild);
    if (light.type == VirtualShadowMapLightType::PointFace && settings.filter != VirtualShadowMapFilter::Hard && !deferProjection) {
        log += "Filtered point VSM requires six deferred faces before project().\n";
        return makeError(Error::InvalidArgument);
    }
    const auto direction = light.direction;
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto& material = materials[index];
        if ((material.alphaMode != "OPAQUE" && material.alphaMode != "MASK") ||
            (!alphaTextures.empty() && alphaTextures[index] && material.baseColorTexture.texCoord != 0)) {
            log += "VSM supports opaque/MASK caster Coverage with UV0 alpha textures.\n";
            return makeError(Error::Unsupported);
        }
    }
    phase.next("Prepare VSM material programs");
    const auto coverage = MaterialValueProgramSet::create(materials, log);
    if (!coverage) { return makeError(Error::InvalidArgument); }
    phase.next("Promote VSM cache and prepare state");
    auto* frame = RenderFrameContext::from(commands);
    if (settings.cacheEnabled && (!frame || !frame->recording())) {
        log += "Persistent VSM requires a recording RenderFrameContext.\n";
        return makeError(Error::InvalidArgument);
    }
    const uint32_t cancelledCount = uint32_t(resolvePendingCache());
    if (pending_) {
        log += "Previous VSM recording must be submitted or cancelled before recording again.\n";
        return makeError(Error::InvalidArgument);
    }
    if (!settings.cacheEnabled) { cache_.reset(); }
    phase.next("Replace VSM recording state");
    // Commands and returned results retain their backing independently. Release
    // the coordinator's old snapshot before choosing an idle resource slot.
    state_.reset();
    if (!pool_ || pool_->device != &device) {
        pool_ = std::make_shared<ResourcePool>();
        pool_->device = &device;
    }
    const auto poolBefore = pool_->stats;
    const auto completion = frame && frame->recording() ? frame->completion() : GPUCompletionPoint{};
    phase.next("Prepare VSM recording resources");
    auto state = std::make_shared<State>();
    state->receiverView = view;
    state->projectionOrigin = light.position;
    state->pointLightFaceSelection = settings.pointLightFaceSelection;
    state->pool = pool_;
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    state->registry = *registry;
    auto& push = state->push;
    push.receiverGeometry = UINT32_MAX;
    push.physicalPool = UINT32_MAX;
    push.poolFace = poolFace;
    if (sharedPhysicalPool) {
        auto lease = state->registry->storageBuffer(*poolFrame->metadata);
        if (!lease) { return std::unexpected(lease.error()); }
        push.physicalPool = lease->shaderIndex();
        state->leases.push_back(*lease);
        state->resources->physicalPoolStorage = poolFrame->storage;
        state->leases.insert(state->leases.end(), poolFrame->storage->leases.begin(), poolFrame->storage->leases.end());
    }
    state->externalOutput = output != nullptr;
    push.projectionFlags = output && output->sharedPointLayer ? kVSMSharedPointLayer : 0u;
    push.triangleCount = casters.triangleCapacity;
    push.clipmapCount = settings.clipmapCount;
    push.virtualPagesPerAxis = settings.virtualPagesPerAxis;
    push.physicalPagesX = static_cast<uint32_t>(std::ceil(std::sqrt(float(settings.physicalPageCount))));
    push.physicalPagesY = (settings.physicalPageCount + push.physicalPagesX - 1) / push.physicalPagesX;
    if (sharedPhysicalPool) {
        push.physicalPagesX = poolFrame->storage->pagesX;
        push.physicalPagesY = poolFrame->storage->pagesY;
    }
    push.physicalPageCount = settings.physicalPageCount;
    push.pageSize = settings.pageSize;
    push.workCapacity = singlePage ? 0u : settings.maxWorkItems;
    push.batchCount = (push.workCapacity + kDrawBatchSize - 1) / kDrawBatchSize;
    push.coarseBatchCount = (casters.triangleCapacity + kDrawBatchSize - 1) / kDrawBatchSize;
    push.cancelledCount = cancelledCount;
    push.lightRange = light.range;
    push.lightType = uint32_t(light.type);
    push.projectionFace = light.face;
    push.filterMode = uint32_t(settings.filter);
    if (settings.filter == VirtualShadowMapFilter::SMRT && (directional ? light.angularRadius == 0 : light.sourceRadius == 0)) {
        push.filterMode = uint32_t(VirtualShadowMapFilter::Hard);
    }
    if (settings.filter == VirtualShadowMapFilter::PCF && settings.pcfRadius == 0) {
        push.filterMode = uint32_t(VirtualShadowMapFilter::Hard);
    }
    push.pointFaces = UINT32_MAX;
    push.pcfRadius = settings.pcfRadius;
    push.rayLength = settings.smrtMaxRayLength;
    push.rayLengthScale = settings.smrtRayLengthScale;
    push.maxRayAngle = settings.smrtMaxRayAngle;
    push.samplePhase = settings.smrtTemporalSampling && push.filterMode == 2u ? (view.frame[0] & 63u) + 1u : 0u;
    push.sourceRadius = light.sourceRadius;
    push.angularRadius = light.angularRadius;
    push.rayCount = settings.smrtRayCount;
    push.samplesPerRay = settings.smrtSamplesPerRay;
    // A nonnegative float needs 31 bits. The low bit retains the footprint flag,
    // preserving the original push word for zero bias.
    push.receiverLOD = (std::bit_cast<uint32_t>(settings.resolutionLodBias) << 1u) | uint32_t(settings.receiverFootprint);
    push.lodScreenRayLength = settings.shadowLOD && settings.shadowLODSmartBias && directional && push.filterMode == 0u ?
        settings.shadowLODScreenRayLength : 0.f;
    push.enabled = enabled;
    push.totalPages = settings.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1;
    total.work("totalPages", push.totalPages);
    state->leases = casters.leases;
    auto geometry = state->registry->storageBuffer(*casters.parameters);
    if (!geometry) { return std::unexpected(geometry.error()); }
    push.geometry = geometry->shaderIndex(); state->leases.push_back(*geometry);
    phase.next("Prepare VSM projections and dirty bounds");
    const float farExtent = directional ? 0.f : localFarExtent(casters.bounds, light);
    const float farMargin = std::max({.01f, light.nearPlane, settings.shadowBias * 2});
    float farPlane = std::max(light.nearPlane * 2, farExtent * (settings.cacheEnabled ? 1.25f : 1.05f) + farMargin);
    if (!directional && !std::isfinite(farPlane)) {
        log += "VSM local projection exceeds the finite depth range.\n";
        return makeError(Error::InvalidArgument);
    }
    auto projections = directional ? makeProjections(casters.bounds, view, direction, settings) :
        makeLocalProjections(light, farPlane, settings);
    // A local coarse page uses the level-zero frustum for changed-bound tests.
    const uint32_t remapLevelCount = std::max(1u, settings.clipmapCount);
    const auto origins = directional ? pageOrigins(view, direction, settings) :
        std::vector<std::array<int64_t, 2>>(remapLevelCount, {0, 0});
    const bool layered = settings.cacheEnabled && settings.cacheLayering;
    const bool staticPageHZB = layered && directional;
    const bool receiverMask = layered && directional &&
        push.filterMode == uint32_t(VirtualShadowMapFilter::SMRT);
    std::array<uint32_t, kVSMHybridHardwareTriangles + 1> cacheDiagnostics{};
    cacheDiagnostics[50] = UINT32_MAX;
    cacheDiagnostics[kVSMCacheInstanceTable] = casters.instanceTable;
    cacheDiagnostics[kVSMCacheInstanceCount] = static_cast<uint32_t>(casters.states.size());
    auto& reasons = cacheDiagnostics[kVSMCacheReasons];
    if (!settings.cacheEnabled) { reasons |= kVSMCacheReasonCacheDisabled; }
    if (!cache_) { reasons |= kVSMCacheReasonCold; }
    if (settings.forceRedraw) { reasons |= kVSMCacheReasonForceRedraw; }
    if (cache_) {
        if (cache_->device != &device) { reasons |= kVSMCacheReasonDevice; }
        if (!compatibleLight(cache_->light, light) || cache_->enabled != enabled) { reasons |= kVSMCacheReasonLight; }
        if (!compatibleSettings(cache_->settings, settings) || cache_->receiverMask != receiverMask) { reasons |= kVSMCacheReasonSettings; }
        if (cache_->sceneIdentity != casters.sceneIdentity) { reasons |= kVSMCacheReasonScene; }
        if (cache_->geometryRevision != casters.geometryRevision) { reasons |= kVSMCacheReasonGeometryPublication; }
        if (cache_->texturePublicationRevision != casters.texturePublicationRevision) { reasons |= kVSMCacheReasonTexturePublication; }
        if (cache_->tessellationKey != casters.tessellationKey) { reasons |= kVSMCacheReasonTessellation; }
        if (cache_->settings.cacheLayering != settings.cacheLayering) { reasons |= kVSMCacheReasonLayerMode; }
        if (bool(cache_->lodSelection) != gpuLOD) { reasons |= kVSMCacheReasonSettings; }
    }
    bool reuse = cache_ && cache_->receiverMask == receiverMask && cache_->settings.cacheLayering == settings.cacheLayering && cache_->device == &device && compatibleLight(cache_->light, light) && cache_->enabled == enabled &&
        compatibleSettings(cache_->settings, settings) && cache_->sceneIdentity == casters.sceneIdentity &&
        cache_->dependencyKey == casters.dependencyKey &&
        bool(cache_->lodSelection) == gpuLOD;
    bool inactiveSnapshot = false;
    if (reuse && sharedPhysicalPool) {
        const auto& previous = physicalPool->impl_->committed;
        const bool latest = previous && cache_->poolSerial == previous->allocationSerial;
        const bool recent = frame->frameIndex() - cache_->recordedFrame <= uint64_t(settings.maxLightCacheAge) + 1u;
        reuse = cache_->physicalPoolStorage.lock() == poolFrame->storage && (settings.maxLightCacheAge ? recent : latest);
        inactiveSnapshot = !latest;
    } else if (reuse) {
        reuse = cache_->poolSerial == 0;
    }
    const bool compatibleCache = reuse;
    if (reuse && directional && casters.bounds.valid) {
        const auto [low, high] = projectedBounds(casters.bounds, cache_->projections.back());
        const float xyMargin = 2.f / settings.pageSize;
        const float zMargin = cache_->projections.back().metric[2];
        reuse = low[0] > -1 + xyMargin && high[0] < 1 - xyMargin &&
            low[1] > -1 + xyMargin && high[1] < 1 - xyMargin && low[2] > zMargin && high[2] < 1 - zMargin;
    } else if (reuse && !directional) {
        reuse = farExtent + farMargin <= cache_->farPlane;
    }
    if (compatibleCache && !reuse) { reasons |= kVSMCacheReasonProjectionRange; }
    const bool restoreInactive = reuse && inactiveSnapshot;
    if (restoreInactive) {
        auto assignmentLease = state->registry->storageBuffer(*physicalPool->impl_->committed->history);
        if (!assignmentLease) { return std::unexpected(assignmentLease.error()); }
        cacheDiagnostics[kVSMCachePreviousSlotAssignments] = assignmentLease->shaderIndex();
        cacheDiagnostics[kVSMCachePoolSerialLow] = uint32_t(cache_->poolSerial);
        cacheDiagnostics[kVSMCachePoolSerialHigh] = uint32_t(cache_->poolSerial >> 32u);
        state->leases.push_back(*assignmentLease);
    }
    const uint64_t cacheFrame = reuse ? cache_->cacheFrame + 1u : 1u;
    std::span<const scene::Bounds> dirtyBounds;
    std::vector<scene::Bounds> staticDirtyBounds, mergeBounds;
    CasterChangeContext localChanges;
    size_t comparedCasterStates = 0;
    bool reusedCasterChanges = false;
    std::vector<Cache::DynamicCaster> dynamic;
    std::vector<uint64_t> dynamicCasters, demotedCasters;
    std::vector<std::array<uint32_t, 2>> mergeTriangleRanges;
    // A retained resident cut owns immutable geometry, authored keys and bounds.
    // A different upload (including a changed cut) must compare both snapshots.
    const bool unchangedCasters = reuse && casters.owner && cache_->casterOwner.lock() == casters.owner;
    const auto trackDynamic = [&](size_t index, uint64_t lastChanged) {
        const auto& caster = casters.states[index];
        if (settings.framesStaticThreshold != 0 && cacheFrame - lastChanged >= settings.framesStaticThreshold) {
            demotedCasters.push_back(caster.id);
            mergeBounds.push_back(caster.bounds);
            mergeTriangleRanges.push_back({caster.triangleFirst, caster.triangleCount});
        } else {
            dynamic.push_back({index, lastChanged});
            dynamicCasters.push_back(caster.id);
        }
    };
    if (reuse) {
        if (!directional) {
            projections = cache_->projections;
            farPlane = cache_->farPlane;
        }
        for (size_t index = 0; index < projections.size(); ++index) {
            projections[index].rowZ = cache_->projections[index].rowZ;
            projections[index].metric[2] = cache_->projections[index].metric[2];
        }
        projections.back() = cache_->projections.back();
        if (unchangedCasters) {
            for (const auto& caster : cache_->dynamic) { trackDynamic(caster.index, caster.lastChanged); }
        } else {
            auto& diffs = casterChangeContext ? casterChangeContext->diffs : localChanges.diffs;
            const auto [found, inserted] = diffs.try_emplace(cache_->casters.get(), *cache_->casters, casters.states, layered);
            const auto& diff = found->second;
            reusedCasterChanges = !inserted;
            comparedCasterStates = inserted ? casters.states.size() : 0;
            dirtyBounds = diff.dirtyBounds;
            const auto& previous = *cache_->casters;
            size_t dynamicIndex = 0;
            // Unchanged dynamic casters still age and may have shifted indices.
            // Interleave them with sparse edits to keep layer IDs sorted.
            const auto retainDynamicBefore = [&](size_t limit) {
                while (dynamicIndex < cache_->dynamic.size() && cache_->dynamic[dynamicIndex].index < limit) {
                    const auto& caster = cache_->dynamic[dynamicIndex++];
                    trackDynamic(diff.currentForPrevious[caster.index], caster.lastChanged);
                }
            };
            for (const auto& change : diff.changes) {
                if (change.previous == CasterChangeContext::kMissingCaster) {
                    ++cacheDiagnostics[kVSMCacheAddedCasters];
                    if (layered) { staticDirtyBounds.push_back(casters.states[change.current].bounds); }
                    continue;
                }
                retainDynamicBefore(change.previous);
                const bool wasDynamic = dynamicIndex < cache_->dynamic.size() &&
                    cache_->dynamic[dynamicIndex].index == change.previous;
                uint64_t lastChanged = wasDynamic ? cache_->dynamic[dynamicIndex++].lastChanged : 0u;
                const auto& old = previous[change.previous];
                if (change.current == CasterChangeContext::kMissingCaster) {
                    ++cacheDiagnostics[kVSMCacheRemovedCasters];
                    if (layered && !wasDynamic) { staticDirtyBounds.push_back(old.bounds); }
                    continue;
                }
                if (change.texture) { reasons |= kVSMCacheReasonTexturePublication; }
                if (change.tessellation) { reasons |= kVSMCacheReasonTessellation; }
                cacheDiagnostics[kVSMCacheLODChangedCasters] += change.lod;
                if (change.content) {
                    ++cacheDiagnostics[kVSMCacheChangedCasters];
                    cacheDiagnostics[kVSMCacheTransformChanges] += change.transform;
                    cacheDiagnostics[kVSMCacheMaterialChanges] += change.material;
                    if (layered && lastChanged == 0) {
                        staticDirtyBounds.push_back(old.bounds);
                        ++cacheDiagnostics[kVSMCachePromotedCasters];
                    }
                    if (layered) { lastChanged = cacheFrame; }
                } else {
                    if (layered && lastChanged == 0) {
                        staticDirtyBounds.push_back(old.bounds);
                        staticDirtyBounds.push_back(casters.states[change.current].bounds);
                    }
                    if (lastChanged != 0) { lastChanged = cacheFrame; }
                }
                if (lastChanged != 0) { trackDynamic(change.current, lastChanged); }
            }
            retainDynamicBefore(previous.size());
        }
        state->previous = cache_;
        if (auto r = addCommandDependency(commands, cache_->completion); !r) { return std::unexpected(r.error()); }
        auto lease = state->registry->storageBuffer(*cache_->pages);
        if (!lease) { return std::unexpected(lease.error()); }
        push.previousPages = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    const bool mergeStatic = !demotedCasters.empty() && !settings.forceRedraw;
    push.cacheMode = (reuse ? 1u : 8u) | (settings.cacheEnabled ? 2u : 0u) | (settings.forceRedraw ? 4u : 0u) |
        (layered ? 16u : 0u) | (mergeStatic ? 32u : 0u) | (restoreInactive ? kVSMCacheRestoreInactive : 0u);
    push.cacheEpoch = reuse ? cache_->epoch : (cache_ ? cache_->epoch + 1 : 1);
    push.dirtyBoundsCount = static_cast<uint32_t>(dirtyBounds.size());
    total.work("reuse", reuse);
    phase.work("comparedCasterStates", comparedCasterStates);
    phase.work("reusedCasterChanges", reusedCasterChanges);
    phase.work("dirtyBounds", dirtyBounds.size());
    phase.work("demotedCasters", demotedCasters.size());
    phase.next("Build VSM page remapping");
    // Upload each changed world bound once per layer. The GPU emits compact
    // per-level rectangle lists into a disjoint, uninitialized buffer tail.
    std::vector<std::array<uint32_t, 4>> pageChanges;
    std::vector<uint32_t> rectangleHeaders;
    uint64_t rectangleCapacity = 0;
    const auto appendRemap = [&](std::span<const scene::Bounds> changedBounds) {
        const uint32_t base = uint32_t(pageChanges.size());
        pageChanges.resize(pageChanges.size() + remapLevelCount + 1);
        const auto boundBase = uint32_t(pageChanges.size());
        pageChanges[base + remapLevelCount] = {boundBase, uint32_t(changedBounds.size()), 0u, 0u};
        for (const auto& bound : changedBounds) {
            pageChanges.push_back({std::bit_cast<uint32_t>(bound.min.x), std::bit_cast<uint32_t>(bound.min.y),
                std::bit_cast<uint32_t>(bound.min.z), uint32_t(bound.valid)});
            pageChanges.push_back({std::bit_cast<uint32_t>(bound.max.x), std::bit_cast<uint32_t>(bound.max.y),
                std::bit_cast<uint32_t>(bound.max.z), 0u});
        }
        if (reuse) {
            for (uint32_t level = 0; level < remapLevelCount; ++level) {
                const int64_t axis = projections[level].page[1];
                const auto dx = origins[level][0] - cache_->origins[level][0];
                const auto dy = origins[level][1] - cache_->origins[level][1];
                const auto offsetX = int32_t(std::clamp(dx, -axis, axis));
                const auto offsetY = int32_t(std::clamp(dy, -axis, axis));
                pageChanges[base + level] = {uint32_t(offsetX), uint32_t(offsetY), 0u, 0u};
                if (changedBounds.empty() || std::abs(offsetX) == axis || std::abs(offsetY) == axis || settings.forceRedraw) { continue; }
                rectangleHeaders.push_back(base + level);
                pageChanges[base + level][2] = uint32_t(rectangleCapacity);
                rectangleCapacity += changedBounds.size();
            }
        }
        return base;
    };
    appendRemap(dirtyBounds);
    const uint32_t staticRemapBase = layered ? appendRemap(staticDirtyBounds) : 0u;
    if (mergeStatic) { cacheDiagnostics[kVSMCacheMergeRemap] = appendRemap(mergeBounds); }
    cacheDiagnostics[kVSMCacheLayered] = layered;
    cacheDiagnostics[kVSMCacheReused] = reuse;
    cacheDiagnostics[kVSMCacheDirtyBounds] = uint32_t(dirtyBounds.size());
    cacheDiagnostics[kVSMCacheStaticDirtyBounds] = layered ? uint32_t(staticDirtyBounds.size()) : 0u;
    cacheDiagnostics[kVSMCacheCancelledRecordings] = cancelledCount;
    cacheDiagnostics[kVSMCacheRasterWorkAvailable] = layered && settings.collectGPUWork;
    cacheDiagnostics[kVSMCacheDynamicCasters] = uint32_t(dynamicCasters.size());
    cacheDiagnostics[kVSMCacheStaticCasters] = uint32_t(casters.states.size() - dynamicCasters.size());
    cacheDiagnostics[35] = uint32_t(staticDirtyBounds.size());
    cacheDiagnostics[36] = staticRemapBase;
    cacheDiagnostics[39] = pageChanges[remapLevelCount][0];
    if (mergeStatic) {
        // Build and coarse raster use original 64-slot cluster identities. Distinct
        // casters can share a cluster, including under the selected shadow LOD.
        std::ranges::sort(mergeTriangleRanges);
        const size_t base = pageChanges.size();
        for (const auto& [first, count] : mergeTriangleRanges) {
            const uint32_t end = first + count;
            for (uint32_t triangle = first; triangle < end;) {
                const uint32_t lane = triangle & 63u;
                const uint32_t clusterFirst = triangle - lane;
                const uint32_t length = std::min(64u - lane, end - triangle);
                const uint64_t mask = (UINT64_MAX >> (64u - length)) << lane;
                if (pageChanges.size() == base || pageChanges.back()[0] != clusterFirst) {
                    pageChanges.push_back({clusterFirst, 0u, 0u, 0u});
                }
                pageChanges.back()[1] |= uint32_t(mask);
                pageChanges.back()[2] |= uint32_t(mask >> 32u);
                triangle += length;
            }
        }
        cacheDiagnostics[kVSMCacheMergeClustersBase] = uint32_t(base);
        cacheDiagnostics[kVSMCacheMergeClusterCount] = uint32_t(pageChanges.size() - base);
    }
    if (rectangleCapacity + pageChanges.size() > UINT32_MAX) {
        log += "VSM invalidation rectangle capacity exceeds the GPU address range.\n";
        return makeError(Error::InvalidArgument);
    }
    for (const auto header : rectangleHeaders) { pageChanges[header][2] += uint32_t(pageChanges.size()); }
    const uint64_t remapUploadBytes = pageChanges.size() * sizeof(pageChanges[0]);
    phase.work("remapBytes", remapUploadBytes);
    phase.work("denseRemapBytes", uint64_t(push.totalPages) * 8);
    phase.work("remapLevels", settings.clipmapCount);
    phase.work("rectangleCapacity", rectangleCapacity);
    phase.next("Prepare VSM materials and alpha leases");
    std::vector<ShadowMaterial> gpuMaterials;
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto& material = materials[index];
        uint32_t alpha = UINT32_MAX;
        if (!alphaTextures.empty() && alphaTextures[index]) {
            auto lease = state->registry->sampledImage(*alphaTextures[index]);
            if (!lease) { return std::unexpected(lease.error()); }
            alpha = lease->shaderIndex(); state->leases.push_back(*lease);
        }
        const auto& transform = material.baseColorTexture.uvTransform;
        gpuMaterials.push_back({{material.baseColorFactor.w, material.alphaCutoff, material.alphaMode == "MASK" ? 1.f : 0.f, 0},
            {alpha, material.doubleSided ? 1u : 0u, 0, 0},
            {transform[0], transform[1], transform[2], 0}, {transform[3], transform[4], transform[5], 0}});
    }
    const uint32_t allocationBlocks = (push.totalPages + 255u) / 256u;
    const uint64_t allocationWords = uint64_t(push.physicalPageCount) * 2 + push.totalPages + allocationBlocks;
    // Append after owner/rank/block/candidate scratch. No allocation reader
    // aliases these words while Admit and Assign publish the dirty-page mask.
    const uint32_t clusterPageBitsetWords = (push.totalPages + 31u) / 32u;
    const uint32_t receiverMaskWords = receiverMask ? push.totalPages * 2u : 0u;
    const uint32_t hierarchyAxis = std::bit_ceil(push.virtualPagesPerAxis);
    const uint32_t hierarchyWords = dirtyPageHierarchy ?
        push.clipmapCount * (4u + (hierarchyAxis * hierarchyAxis - 1u) / 3u) : 0u;
    const uint32_t clusterCapacity = push.triangleCount / 64u + (push.triangleCount % 64u != 0u ? 1u : 0u);
    const uint32_t visibilityBatches = staticPageHZB && enabled &&
        cacheDiagnostics[kVSMCacheDynamicCasters] && cacheDiagnostics[kVSMCacheStaticCasters] ?
        (clusterCapacity + kDrawBatchSize - 1u) / kDrawBatchSize : 0u;
    // Splitting an original cluster at each instance boundary adds at most one
    // ticket per boundary. Every ticket owns at least one distinct triangle slot.
    const uint64_t candidateCapacity = instanceBuild ?
        std::min<uint64_t>(push.triangleCount, uint64_t(clusterCapacity) + casters.states.size()) : clusterCapacity;
    const uint32_t compactBatches = compactBuild ? static_cast<uint32_t>((candidateCapacity + kDrawBatchSize - 1) / kDrawBatchSize) : 0u;
    const uint64_t compactWords = compactBuild ? 1u + candidateCapacity * 2u : 0u;
    const uint64_t compactArgumentOffset = uint64_t(push.batchCount + push.coarseBatchCount) * 12u;
    const uint32_t mergeDrawOffset = push.batchCount + push.coarseBatchCount + compactBatches;
    const uint32_t clearDrawOffset = mergeDrawOffset + (mergeStatic ? push.batchCount + push.coarseBatchCount : 0u);
    const uint32_t hybridDrawOffset = clearDrawOffset + uint32_t(clusterPageRaster);
    cacheDiagnostics[kVSMCacheMergeDrawOffset] = mergeDrawOffset;
    // The original range is the allocator/cache ownership table. Projection
    // mappings are rebuilt into a disjoint tail after Finalize publishes validity.
    const uint64_t validReceiverMaskBytes = receiverMask ? ((uint64_t(push.totalPages) + 1u) / 2u) * 16u : 0u;
    // uint2(physicalSlot, lastRequestedFrame) in oldest-first order. Place this at
    // the end of the actual pooled page buffer; cached and current sizes can differ.
    const uint64_t physicalHistoryBytes = settings.cacheEnabled && !sharedPhysicalPool ? ((push.physicalPageCount + 1u) / 2u) * 16u : 0u;
    // Both layers reuse the compact HW list. Hybrid entries also store their
    // lane mask. History-occluded tickets use a separate single-word list.
    const uint32_t indexLists = uint32_t(classifyFine) + uint32_t(clusterHybridRaster) + uint32_t(twoPassHZB);
    const uint32_t maxHZBPages = twoPassHZB ? std::min(push.totalPages - 1u, push.physicalPageCount) : 0u;
    const uint32_t hzbPageBatches = (maxHZBPages + kDrawBatchSize - 1u) / kDrawBatchSize;
    const uint32_t instanceCount = cacheDiagnostics[kVSMCacheInstanceCount];
    const uint32_t hierarchyHZBBatches = hierarchyHZB ? (instanceCount + kDrawBatchSize - 1u) / kDrawBatchSize : 0u;
    const uint64_t indexWords = uint64_t(push.workCapacity) * indexLists + 2u * maxHZBPages;
    const uint64_t indexBytes = ((indexWords + 3u) / 4u) * 16u;
    const uint64_t workBytes = pageWindowRaster ? windowBytes :
        clusterPageRaster ? (uint64_t(push.workCapacity) + 1u) * 16u + indexBytes : uint64_t(std::max(push.workCapacity, 1u)) * 8u;
    const uint32_t argumentBatches = hybridDrawOffset + (twoPassHZB ? 6u : clusterHybridRaster ? 2u : 0u) * push.batchCount +
        2u * hzbPageBatches + hierarchyHZBBatches;
    const uint64_t sizes[]{gpuMaterials.size() * sizeof(ShadowMaterial), uint64_t(push.totalPages) * 32u + validReceiverMaskBytes + physicalHistoryBytes,
        coverage->inputBytes().size(), workBytes, 128 + (settings.collectGPUWork ?
            kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize : 0u) +
            (settings.collectReceiverCoverage ? uint64_t(view.current.viewport[1]) * uint64_t(view.current.viewport[2]) *
                kVSMReceiverCoverageRecordWords * sizeof(uint32_t) : 0u),
        projections.size() * sizeof(ShadowProjection), sizeof(ViewConstants),
        uint64_t(std::max(1u, argumentBatches)) * 12u,
        remapUploadBytes + rectangleCapacity * sizeof(pageChanges[0]), (allocationWords + clusterPageBitsetWords + receiverMaskWords + hierarchyWords + compactWords) * sizeof(uint32_t)};
    const uint32_t strides[]{64, 16, 4, groupedFineRaster ? 16u : 8u, 4, 112, 208, 4, 16, 4};
    const void* data[]{gpuMaterials.data(), nullptr, coverage->inputBytes().data(), nullptr, nullptr, projections.data(), &view, nullptr,
        pageChanges.data(), nullptr};
    uint32_t* indices[]{&push.materials, &push.pages, &push.coverageInputs, &push.workItems,
        &push.statistics, &push.projections, &push.view, &push.arguments, &push.reuse, &push.owners};
    phase.next("Prepare VSM buffers");
    constexpr const char* bufferPhases[]{"Create VSM buffers", "Upload VSM buffers", "Acquire VSM buffer descriptors"};
    CPUProfileRecorder bufferProfile;
    uint64_t bufferBytes = 0, hostUploadBytes = 0, hostUploadBuffers = 0;
    for (size_t index = 0; index < state->resources->buffers.size(); ++index) {
        bufferBytes += sizes[index];
        const uint64_t uploadBytes = index == 8 ? remapUploadBytes : sizes[index];
        if (data[index]) { hostUploadBytes += uploadBytes; ++hostUploadBuffers; }
        CPUProfileScope bufferPhase(profiler ? &bufferProfile : nullptr, bufferPhases[0]);
        auto buffer = pool_->buffer(index, {.size = sizes[index], .structureStride = strides[index],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (index == 7 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .memoryLocation = data[index] ? MemoryLocation::HostUpload : MemoryLocation::Device}, completion);
        if (!buffer) { return std::unexpected(buffer.error()); }
        state->resources->buffers[index] = std::move(*buffer);
        bufferPhase.next(bufferPhases[1]);
        if (data[index]) {
            void* mapped = state->resources->buffers[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data[index], uploadBytes);
            state->resources->buffers[index]->flush({0, uploadBytes}); state->resources->buffers[index]->unmap();
        }
        bufferPhase.next(bufferPhases[2]);
        auto lease = state->registry->storageBuffer(*state->resources->buffers[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *indices[index] = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    if (profiler) {
        // Each buffer records the same three ordered phases. Publish their
        // sums so multiple light faces fit the graph's CPU section capacity.
        // Thread diagnostics belong to the enclosing uninterrupted scope.
        std::array<double, 3> milliseconds{};
        for (size_t index = 0; index < bufferProfile.sections.size(); ++index) {
            milliseconds[index % milliseconds.size()] += bufferProfile.sections[index].cpuMilliseconds;
        }
        for (size_t index = 0; index < milliseconds.size(); ++index) {
            profiler->sections.push_back({.name = bufferPhases[index], .parent = profiler->parent,
                .cpuMilliseconds = milliseconds[index], .cpuOnly = true});
        }
    }
    phase.work("buffers", state->resources->buffers.size());
    phase.work("bufferBytes", bufferBytes);
    phase.work("hostUploadBuffers", hostUploadBuffers);
    phase.work("hostUploadBytes", hostUploadBytes);
    phase.work("deviceBuffers", state->resources->buffers.size() - hostUploadBuffers);
    phase.work("deviceBytes", bufferBytes - hostUploadBytes);
    phase.work("uploadBytes", hostUploadBytes);
    phase.work("bufferCreates", pool_->stats.bufferCreates - poolBefore.bufferCreates);
    phase.work("bufferReuses", pool_->stats.bufferReuses - poolBefore.bufferReuses);
    phase.work("bufferEvictions", pool_->stats.bufferEvictions - poolBefore.bufferEvictions);
    total.work("bufferBytes", bufferBytes);
    phase.next("Prepare VSM textures and descriptors");
    const uint32_t width = static_cast<uint32_t>(view.current.viewport[1]), height = static_cast<uint32_t>(view.current.viewport[2]);
    const Format formats[]{Format::D32Sfloat, Format::R8Unorm, Format::RGBA8Unorm};
    uint32_t* textureIndices[]{&push.atlas, &push.shadow, &push.debug};
    for (size_t index = 0; index < 3; ++index) {
        const uint32_t textureWidth = index == 0 ? push.physicalPagesX * push.pageSize : width;
        const uint32_t textureHeight = index == 0 ? push.physicalPagesY * push.pageSize : height;
        // Cache hits bypass texture acquisition. Still retire obsolete extents
        // once their GPU work and readers finish, so shrinking a page budget
        // does not retain the previous large atlas indefinitely.
        std::erase_if(pool_->textures[index], [&](const auto& entry) {
            const auto& desc = entry.resource.texture->desc();
            const bool obsolete = (index != 0 && output) || desc.width != textureWidth || desc.height != textureHeight;
            if (!obsolete || entry.resource.texture.use_count() != 1 || !entry.completion.isComplete()) { return false; }
            ++pool_->stats.textureEvictions;
            return true;
        });
        if (index != 0 && output) {
            auto* target = index == 1 ? output->shadow : output->debug;
            if (index == 2 && !target) {
                pool_->externalViews[1].reset();
                push.debug = UINT32_MAX;
                continue;
            }
            if (!target || target->desc().format != formats[index] || target->desc().width != width ||
                target->desc().height != height || output->layer >= target->desc().layerCount ||
                !hasFlag(target->desc().usage, TextureUsageBits::Storage)) {
                log += "VSM projection target must be a matching storage texture layer.\n";
                return makeError(Error::InvalidArgument);
            }
            state->outputTextures[index - 1] = target;
            auto& cached = pool_->externalViews[index - 1];
            if (!cached || cached->retainTexture() != target->retainAllocation() || cached->desc().range.baseLayer != output->layer) {
                auto textureView = device.createTextureView(*target, {.format = formats[index], .range = {.baseLayer = output->layer}});
                if (!textureView) { return std::unexpected(textureView.error()); }
                cached = std::move(*textureView);
                ++pool_->stats.viewCreates;
            }
            state->resources->views[index] = cached;
        } else if (index == 0 && sharedPhysicalPool) {
            state->resources->textures[index] = poolFrame->storage->atlas->texture;
            state->resources->views[index] = poolFrame->storage->atlas->view;
        } else if (index == 0 && reuse) {
            state->resources->textures[index] = cache_->atlas->texture;
            state->resources->views[index] = cache_->atlas->view;
        } else {
            auto texture = pool_->texture(index, {.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferSource |
                (index == 0 ? TextureUsageBits::DepthStencilAttachment : TextureUsageBits::Storage),
                .format = formats[index], .width = textureWidth, .height = textureHeight}, completion, push.pageSize);
            if (!texture) { return std::unexpected(texture.error()); }
            state->resources->textures[index] = std::move(texture->texture);
            state->resources->views[index] = std::move(texture->view);
        }
        if (index != 0 && !output) { state->outputTextures[index - 1] = state->resources->textures[index].get(); }
        auto lease = index == 0 ? state->registry->sampledImage(*state->resources->views[index]) : state->registry->storageImage(*state->resources->views[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *textureIndices[index] = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    // Static atlases obey the same completion/reader retirement rule as the composite atlas.
    // Run this on cache hits and after disabling layering, not only on pool acquisition.
    std::erase_if(pool_->textures[3], [&](const auto& entry) {
        const auto& desc = entry.resource.texture->desc();
        const bool obsolete = !layered || desc.width != push.physicalPagesX * push.pageSize ||
            desc.height != push.physicalPagesY * push.pageSize;
        if (!obsolete || entry.resource.texture.use_count() != 1 || !entry.completion.isComplete()) { return false; }
        ++pool_->stats.textureEvictions;
        return true;
    });
    if (layered) {
        if (sharedPhysicalPool) {
            state->resources->staticAtlas = poolFrame->storage->staticAtlas;
        } else if (reuse) {
            state->resources->staticAtlas = cache_->staticAtlas;
        } else {
            auto atlas = pool_->texture(3, {.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferSource |
                TextureUsageBits::DepthStencilAttachment, .format = Format::D32Sfloat,
                .width = push.physicalPagesX * push.pageSize, .height = push.physicalPagesY * push.pageSize}, completion, push.pageSize);
            if (!atlas) { return std::unexpected(atlas.error()); }
            state->resources->staticAtlas = std::make_shared<ShadowAtlas>(std::move(*atlas));
        }
        auto atlasLease = state->registry->sampledImage(*state->resources->staticAtlas->view);
        if (!atlasLease) { return std::unexpected(atlasLease.error()); }
        cacheDiagnostics[32] = atlasLease->shaderIndex(); state->leases.push_back(*atlasLease);
        for (uint32_t layer = 0; layer < (mergeStatic ? 3u : 2u); ++layer) {
            const bool mergeLayer = layer == 2u;
            auto geometryLayer = prepareVirtualShadowMapCasterLayer(device, commands, casters,
                mergeLayer ? demotedCasters : dynamicCasters, layer == 0, log);
            if (!geometryLayer) { return std::unexpected(geometryLayer.error()); }
            auto lease = state->registry->storageBuffer(*geometryLayer->parameters);
            if (!lease) { return std::unexpected(lease.error()); }
            cacheDiagnostics[mergeLayer ? kVSMCacheMergeGeometry : 33u + layer] = lease->shaderIndex();
            state->leases.push_back(*lease);
            state->leases.insert(state->leases.end(), geometryLayer->leases.begin(), geometryLayer->leases.end());
            state->resources->casterLayers[layer] = geometryLayer->owner;
            state->resources->layerBufferBytes += geometryLayer->layerBufferBytes;
        }
    }
    if (staticPageHZB) {
        if (sharedPhysicalPool) {
            state->resources->staticHZB = poolFrame->storage->hzbTable;
        } else if (reuse) {
            // Compatible layered directional snapshots always own the pyramid;
            // it is allocated before the cache snapshot is published.
            state->resources->staticHZB = cache_->staticHZB;
        } else {
            const uint32_t wordsPerPage = std::max(1u, (push.pageSize * push.pageSize - 1u) / 3u);
            auto pyramid = device.createBuffer({.size = uint64_t(push.physicalPageCount) * wordsPerPage * sizeof(float),
                .structureStride = sizeof(float), .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device});
            if (!pyramid) { return std::unexpected(pyramid.error()); }
            state->resources->staticHZB = std::move(*pyramid);
        }
        // Finalize writes the HZB job count/list. Propagate publishes its 2D
        // dispatch, followed by disjoint cluster-visibility indirect arguments.
        auto jobs = pool_->buffer(15, {.size = uint64_t(6u + push.physicalPageCount + visibilityBatches * 3u) * sizeof(uint32_t), .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect, .memoryLocation = MemoryLocation::Device}, completion);
        if (!jobs) { return std::unexpected(jobs.error()); }
        state->resources->staticHZBJobs = *jobs;
        auto pyramidLease = state->registry->storageBuffer(*state->resources->staticHZB);
        if (!pyramidLease) { return std::unexpected(pyramidLease.error()); }
        auto jobsLease = state->registry->storageBuffer(**jobs);
        if (!jobsLease) { return std::unexpected(jobsLease.error()); }
        cacheDiagnostics[37] = pyramidLease->shaderIndex(); state->leases.push_back(*pyramidLease);
        cacheDiagnostics[38] = jobsLease->shaderIndex(); state->leases.push_back(*jobsLease);
        cacheDiagnostics[40] = UINT32_MAX;
        if (visibilityBatches) {
            auto masks = pool_->buffer(16, {.size = uint64_t(clusterCapacity) * sizeof(uint32_t), .structureStride = 4,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device}, completion);
            if (!masks) { return std::unexpected(masks.error()); }
            state->resources->clusterVisibility = *masks;
            auto maskLease = state->registry->storageBuffer(**masks);
            if (!maskLease) { return std::unexpected(maskLease.error()); }
            cacheDiagnostics[40] = maskLease->shaderIndex(); state->leases.push_back(*maskLease);
        }
    }
    if (gpuLOD) {
        state->resources->gpuLOD = casters.gpuLOD;
        auto context = casters.gpuLOD->context;
        context.viewCount = push.clipmapCount + 1;
        context.maxErrorTexels = settings.shadowLODErrorTexels;
        context.maxWorldError = 0.5f * settings.shadowBias;
        context.previousValid = reuse && cache_->lodSelection && cache_->lodLayoutKey == casters.gpuLOD->layoutKey;
        context.previous = UINT32_MAX;
        auto acquireLOD = [&](size_t slot, uint64_t words, std::shared_ptr<Buffer>& target,
            uint32_t& handle, bool upload = false, uint32_t stride = sizeof(uint32_t)) -> Result<> {
            auto buffer = pool_->buffer(slot, {.size = std::max<uint64_t>(words, 1) * sizeof(uint32_t), .structureStride = stride,
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = upload ? MemoryLocation::HostUpload : MemoryLocation::Device}, completion);
            if (!buffer) { return std::unexpected(buffer.error()); }
            target = *buffer;
            auto lease = state->registry->storageBuffer(*target);
            if (!lease) { return std::unexpected(lease.error()); }
            handle = lease->shaderIndex(); state->leases.push_back(*lease);
            return {};
        };
        if (auto r = acquireLOD(18, uint64_t(context.viewCount) * context.groupStateCount,
            state->resources->lodGroupStates, context.groupStates); !r) { return std::unexpected(r.error()); }
        if (auto r = acquireLOD(19, uint64_t(context.viewCount) * context.clusterCount,
            state->resources->lodSelection, context.selected); !r) { return std::unexpected(r.error()); }
        if (auto r = acquireLOD(20, uint64_t(context.viewCount) * context.instanceCount,
            state->resources->lodChanges, context.changed); !r) { return std::unexpected(r.error()); }
        if (auto r = acquireLOD(21, uint64_t(context.viewCount) * kVirtualShadowMapLODStatisticsWords,
            state->resources->lodStatistics, context.statistics, true); !r) { return std::unexpected(r.error()); }
        auto* statistics = static_cast<uint32_t*>(state->resources->lodStatistics->map());
        if (!statistics) { return makeError(Error::Failure); }
        std::fill_n(statistics, size_t(context.viewCount) * kVirtualShadowMapLODStatisticsWords, 0u);
        for (uint32_t viewIndex = 0; viewIndex < context.viewCount; ++viewIndex) {
            auto* row = statistics + viewIndex * kVirtualShadowMapLODStatisticsWords;
            row[0] = 3; row[1] = 1; row[2] = 1;
        }
        state->resources->lodStatistics->flush(); state->resources->lodStatistics->unmap();
        if (context.previousValid) {
            auto lease = state->registry->storageBuffer(*cache_->lodSelection);
            if (!lease) { return std::unexpected(lease.error()); }
            context.previous = lease->shaderIndex(); state->leases.push_back(*lease);
        }
        if (auto r = acquireLOD(17, sizeof(context) / sizeof(uint32_t),
            state->resources->lodContext, cacheDiagnostics[50], true, sizeof(context)); !r) { return std::unexpected(r.error()); }
        auto* contextData = state->resources->lodContext->map();
        if (!contextData) { return makeError(Error::Failure); }
        std::memcpy(contextData, &context, sizeof(context));
        state->resources->lodContext->flush(); state->resources->lodContext->unmap();
    }
    if (clusterHybridRaster) {
        const uint32_t pageCount = hybridScratchPageCount(push.physicalPageCount, push.pageSize);
        const BufferDesc desc{.size = uint64_t(pageCount) * push.pageSize * push.pageSize * sizeof(uint32_t),
            .structureStride = sizeof(uint32_t), .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device};
        if (sharedPhysicalPool) {
            // The pool submission dependency also orders reuse of its scratch across frames.
            auto& scratch = poolFrame->storage->hybridDepth;
            if (!scratch) {
                auto buffer = device.createBuffer(desc);
                if (!buffer) { return std::unexpected(buffer.error()); }
                scratch = std::move(*buffer);
            }
            state->resources->hybridDepth = scratch;
        } else {
            auto scratch = pool_->buffer(22, desc, completion);
            if (!scratch) { return std::unexpected(scratch.error()); }
            state->resources->hybridDepth = std::move(*scratch);
        }
        auto lease = state->registry->storageBuffer(*state->resources->hybridDepth);
        if (!lease) { return std::unexpected(lease.error()); }
        state->leases.push_back(*lease);
        cacheDiagnostics[kVSMHybridDepth] = lease->shaderIndex();
        cacheDiagnostics[kVSMHybridPageCount] = pageCount;
        cacheDiagnostics[kVSMHybridSubpixelBits] = caps.subPixelPrecisionBits;
    }
    cacheDiagnostics[kVSMHybridDrawOffset] = hybridDrawOffset;
    if (twoPassHZB) {
        const BufferDesc desc{.size = uint64_t(push.physicalPageCount) * kVSMRasterHZBWordsPerSlot * sizeof(float),
            .structureStride = sizeof(float), .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device};
        if (sharedPhysicalPool) {
            auto& pyramid = poolFrame->storage->rasterHZB;
            if (!pyramid) {
                auto buffer = device.createBuffer(desc);
                if (!buffer) { return std::unexpected(buffer.error()); }
                pyramid = std::move(*buffer);
            }
            state->resources->rasterHZB = pyramid;
        } else {
            auto pyramid = pool_->buffer(23, desc, completion);
            if (!pyramid) { return std::unexpected(pyramid.error()); }
            state->resources->rasterHZB = std::move(*pyramid);
        }
        auto lease = state->registry->storageBuffer(*state->resources->rasterHZB);
        if (!lease) { return std::unexpected(lease.error()); }
        cacheDiagnostics[kVSMRasterHZB] = lease->shaderIndex();
        state->leases.push_back(*lease);
    }
    if (hierarchyHZB) {
        auto queue = pool_->buffer(24, {.size = uint64_t(instanceCount) * sizeof(uint32_t),
            .structureStride = sizeof(uint32_t), .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::Device}, completion);
        if (!queue) { return std::unexpected(queue.error()); }
        state->resources->hierarchyHZBQueue = std::move(*queue);
        auto lease = state->registry->storageBuffer(*state->resources->hierarchyHZBQueue);
        if (!lease) { return std::unexpected(lease.error()); }
        state->leases.push_back(*lease);
        cacheDiagnostics[kVSMHierarchyHZBQueue] = lease->shaderIndex();
        cacheDiagnostics[kVSMHierarchyHZBNodes] = casters.hierarchyNodes;
        cacheDiagnostics[kVSMHierarchyHZBRootCount] = casters.hierarchyRootCount;
    }
    auto diagnostics = pool_->buffer(12, {.size = sizeof(cacheDiagnostics), .structureStride = 4,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload}, completion);
    if (!diagnostics) { return std::unexpected(diagnostics.error()); }
    state->resources->cacheDiagnostics = *diagnostics;
    void* diagnosticData = (*diagnostics)->map();
    if (!diagnosticData) { return makeError(Error::Failure); }
    std::memcpy(diagnosticData, cacheDiagnostics.data(), sizeof(cacheDiagnostics));
    (*diagnostics)->flush(); (*diagnostics)->unmap();
    auto diagnosticLease = state->registry->storageBuffer(**diagnostics);
    if (!diagnosticLease) { return std::unexpected(diagnosticLease.error()); }
    push.sampleReserved = diagnosticLease->shaderIndex(); state->leases.push_back(*diagnosticLease);
    auto depthLease = state->registry->sampledImage(depth);
    if (!depthLease) { return std::unexpected(depthLease.error()); }
    push.depth = depthLease->shaderIndex(); state->leases.push_back(*depthLease);
    // A view supplies the same receiver resources for every light, including
    // lights that do not use the directional bias, so one-pass jobs share a permutation.
    if (receiver) {
        ShadowReceiverGeometry receiverData{};
        const auto& stream = *receiver->stream;
        auto visibilityLease = state->registry->sampledImage(*receiver->visibility);
        if (!visibilityLease) { return std::unexpected(visibilityLease.error()); }
        receiverData.visibility = visibilityLease->shaderIndex();
        state->leases.push_back(*visibilityLease);
        const std::array receiverBuffers{stream.visibleClusterBuffer, stream.activeGroupBuffer,
            stream.pageBuffer, stream.pageTableBuffer, stream.paramsBuffer};
        const std::array receiverIndices{&receiverData.records, &receiverData.groups,
            &receiverData.pages, &receiverData.pageTable, &receiverData.params};
        for (size_t index = 0; index < receiverBuffers.size(); ++index) {
            auto lease = state->registry->storageBuffer(*receiverBuffers[index]);
            if (!lease) { return std::unexpected(lease.error()); }
            *receiverIndices[index] = lease->shaderIndex();
            state->leases.push_back(*lease);
        }
        receiverData.recordCapacity = stream.visibleRecordCapacity;
        auto table = pool_->buffer(13, {.size = sizeof(receiverData), .structureStride = sizeof(receiverData),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}, completion);
        if (!table) { return std::unexpected(table.error()); }
        state->resources->receiverGeometry = std::move(*table);
        void* mapped = state->resources->receiverGeometry->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, &receiverData, sizeof(receiverData));
        state->resources->receiverGeometry->flush();
        state->resources->receiverGeometry->unmap();
        auto lease = state->registry->storageBuffer(*state->resources->receiverGeometry);
        if (!lease) { return std::unexpected(lease.error()); }
        push.receiverGeometry = lease->shaderIndex();
        state->leases.push_back(*lease);
    }
    const bool refinedReceiver = push.receiverGeometry != UINT32_MAX;
    if (settings.shadowLODSmartBias && !refinedReceiver) {
        // Missing optional primary geometry must not enable the unrefined correction.
        push.lodScreenRayLength = 0.f;
    }
    auto sampler = state->registry->sampler({});
    if (!sampler) { return std::unexpected(sampler.error()); }
    push.sampler = sampler->shaderIndex(); state->leases.push_back(*sampler);
    phase.work("textureCreates", pool_->stats.textureCreates - poolBefore.textureCreates);
    phase.work("textureReuses", pool_->stats.textureReuses - poolBefore.textureReuses);
    phase.work("textureEvictions", pool_->stats.textureEvictions - poolBefore.textureEvictions);
    phase.work("viewCreates", pool_->stats.viewCreates - poolBefore.viewCreates);
    phase.next("Prepare VSM pipeline cache");
    // Grouped filtered permutations change boundary taps; keep regular PCF/SMRT counters.
    // The requested flag still selects per-face projection for the statistics control.
    const bool groupProjectionStatistics = settings.optimizeProjectionStatistics &&
        push.filterMode == uint32_t(VirtualShadowMapFilter::Hard);
    const bool programMiss = !programs_ || programs_->device != &device || programs_->coverageKey != coverage->coverageKey() ||
        programs_->collectGPUWork != settings.collectGPUWork ||
        programs_->collectReceiverCoverage != settings.collectReceiverCoverage ||
        programs_->optimizeProjectionStatistics != settings.optimizeProjectionStatistics ||
        programs_->groupProjectionStatistics != groupProjectionStatistics ||
        programs_->directional != directional ||
        programs_->adaptiveRayCount != adaptiveRayCount ||
        programs_->localSMRTStateReuse != settings.localSMRTStateReuse ||
        programs_->markCacheLevels != markCacheLevels || programs_->markPageShift != markPageShift ||
        programs_->compactBuild != compactBuild || programs_->instanceBuild != instanceBuild || programs_->clusterPageRaster != clusterPageRaster ||
        programs_->pageWindowRaster != pageWindowRaster || programs_->clusterHybridRaster != clusterHybridRaster ||
        programs_->twoPassHZB != twoPassHZB ||
        programs_->hierarchyHZB != hierarchyHZB ||
        programs_->dirtyPageHierarchy != dirtyPageHierarchy ||
        programs_->receiverMask != receiverMask ||
        programs_->staticPageHZB != staticPageHZB ||
        programs_->refinedReceiver != refinedReceiver || programs_->sharedPhysicalPool != sharedPhysicalPool ||
        programs_->batchBuildRaster != (sharedPhysicalPool && settings.batchBuildRaster) ||
        programs_->compactShadowViews != compactShadowViews || programs_->gpuLOD != gpuLOD;
    phase.work("cacheMiss", programMiss);
    if (programMiss) {
        std::filesystem::path generatedDirectory;
        if (!coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log)) {
            return makeError(Error::Failure);
        }
        auto programs = std::make_shared<Programs>();
        programs->device = &device;
        programs->coverageKey = coverage->coverageKey();
        programs->collectGPUWork = settings.collectGPUWork;
        programs->collectReceiverCoverage = settings.collectReceiverCoverage;
        programs->optimizeProjectionStatistics = settings.optimizeProjectionStatistics;
        programs->groupProjectionStatistics = groupProjectionStatistics;
        programs->directional = directional;
        programs->adaptiveRayCount = adaptiveRayCount;
        programs->localSMRTStateReuse = settings.localSMRTStateReuse;
        programs->markCacheLevels = markCacheLevels;
        programs->markPageShift = markPageShift;
        programs->compactBuild = compactBuild;
        programs->clusterPageRaster = clusterPageRaster;
        programs->clusterHybridRaster = clusterHybridRaster;
        programs->twoPassHZB = twoPassHZB;
        programs->hierarchyHZB = hierarchyHZB;
        programs->pageWindowRaster = pageWindowRaster;
        programs->instanceBuild = instanceBuild;
        programs->dirtyPageHierarchy = dirtyPageHierarchy;
        programs->receiverMask = receiverMask;
        programs->staticPageHZB = staticPageHZB;
        programs->refinedReceiver = refinedReceiver;
        programs->sharedPhysicalPool = sharedPhysicalPool;
        programs->batchBuildRaster = sharedPhysicalPool && settings.batchBuildRaster;
        programs->compactShadowViews = compactShadowViews;
        programs->gpuLOD = gpuLOD;
        const char* entries[]{"vsmResetMain", "vsmMarkMain", "vsmAllocateMain", "vsmBuildMain", "vsmFinalizeMain", "vsmProjectMain",
            "vsmRasterMeshMain", "vsmDepthFragmentMain", "vsmClearMeshMain", "vsmClearFragmentMain",
            "vsmCountMain", "vsmAdmitMain", "vsmPhysicalMain", "vsmAssignMain", "vsmCompactMain", "vsmPropagateMappingMain",
            "vsmBuildPageHierarchyMain", "vsmBuildStaticPageHZBMain", "vsmCullDynamicClustersHZBMain", "vsmProjectChangedBoundsMain",
            pageWindowRaster ? "vsmRasterWindowMeshMain" : "vsmRasterPackedMeshMain", "vsmLODSelectMain",
            "vsmHybridClearMain", "vsmHybridRasterMain", "vsmBuildRasterHZBMain", "vsmRetestRasterHZBMain",
            "vsmPrepareHierarchyHZBMain", "vsmResetHierarchyPostMain", "vsmRetestHierarchyHZBMain", "vsmFinalizeHierarchyPostMain"};
        const auto generated = generatedDirectory.string();
        const std::array paths{generated.c_str()};
        const std::array capabilities{"spvMeshShadingEXT"};
        // Reuse Mark invariants only for local lights; directional hoisting regressed Mark.
        const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", coverage->coverageProgramCount() ? "1" : "0"},
            SlangMacroDefine{"VSM_SHARED_PHYSICAL_POOL", sharedPhysicalPool ? "1" : "0"},
            SlangMacroDefine{"VSM_GPU_LOD", gpuLOD ? "1" : "0"},
            SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", settings.collectGPUWork ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_PROJECTION_STATISTICS", groupProjectionStatistics ? "1" : "0"},
            SlangMacroDefine{"VSM_LOCAL_MARK", !directional ? "1" : "0"},
            SlangMacroDefine{"VSM_RECEIVER_MASK", receiverMask ? "1" : "0"},
            SlangMacroDefine{"VSM_CLUSTER_PAGE_BITSET", "1"}};
        const std::string markCacheLevelsDefine = std::to_string(markCacheLevels);
        const std::string markPageShiftDefine = std::to_string(markPageShift);
        for (size_t index = 0; index < programs->shaders.size(); ++index) {
            if (index == 14 && !compactBuild) { continue; }
            if (index == 16 && !dirtyPageHierarchy) { continue; }
            if (index == 17 && !staticPageHZB) { continue; }
            if (index == 18 && !staticPageHZB) { continue; }
            if (index == 20 && !groupedFineRaster) { continue; }
            if (index == 21 && !gpuLOD) { continue; }
            if (index == 22 && !clusterHybridRaster) { continue; }
            if (index == 23 && !classifyFine) { continue; }
            if (index >= 24 && !twoPassHZB) { continue; }
            if (index >= 26 && !hierarchyHZB) { continue; }
            const bool buildEntry = index == 3 || index == 14;
            const bool compactEntry = index == 0 || buildEntry || index >= 26;
            std::vector<SlangMacroDefine> entryMacros(macros.begin(), macros.end());
            if (compactShadowViews && (buildEntry || index == 11 || index == 13 || index == 28)) {
                entryMacros.push_back({"VSM_COMPACT_SHADOW_VIEWS", "1"});
            }
            if (programs->batchBuildRaster && (buildEntry || index == 4 || index == 6 || index == 8 ||
                index == 15 || index == 16 || index == 17 || index == 18 || index == 20 || index >= 22)) {
                entryMacros.push_back({"VSM_BATCHED_BUILD_RASTER", "1"});
            }
            if (staticPageHZB && (index == 0 || buildEntry || index == 4 || index == 15 || index == 17 || index == 18 || index == 19)) {
                entryMacros.push_back({"VSM_STATIC_PAGE_HZB", "1"});
            }
            if (staticPageHZB && (index == 4 || index == 6 || index == 18 || index == 20 || index == 23)) {
                entryMacros.push_back({"VSM_CLUSTER_HZB", "1"});
            }
            if (groupedFineRaster && (index == 0 || index == 3 || index == 4 || index == 20 || index == 23)) {
                entryMacros.push_back({pageWindowRaster ? "VSM_PAGE_WINDOW_RASTER" : "VSM_CLUSTER_PAGE_RASTER", "1"});
            }
            if (clusterHybridRaster && (index == 4 || index == 9 || index == 20 || (index >= 22 && index <= 26))) {
                entryMacros.push_back({"VSM_CLUSTER_HYBRID_RASTER", "1"});
            }
            if (twoPassHZB && (index == 0 || index == 4 || index == 20 || index >= 23)) {
                entryMacros.push_back({"VSM_TWO_PASS_HZB", "1"});
            }
            if (hierarchyHZB && (index == 4 || index == 14 || index == 20 || index == 24 || index >= 26)) {
                entryMacros.push_back({"VSM_HIERARCHY_HZB", "1"});
                if (index == 14) { entryMacros.push_back({"VSM_TWO_PASS_HZB", "1"}); }
            }
            if (dirtyPageHierarchy && (compactEntry || index == 16)) {
                entryMacros.push_back({"VSM_DIRTY_PAGE_HIERARCHY", "1"});
            }
            if (index == 1 && settings.collectReceiverCoverage) {
                entryMacros.push_back({"VSM_RECEIVER_COVERAGE_DIAGNOSTICS", "1"});
            }
            if (index == 1) {
                entryMacros.push_back({"VSM_DIRECTIONAL_MARK", directional ? "1" : "0"});
                entryMacros.push_back({"VSM_MARK_CACHE_LEVELS", markCacheLevelsDefine.c_str()});
                entryMacros.push_back({"VSM_MARK_PAGE_SHIFT", markPageShiftDefine.c_str()});
            }
            if (index == 5) {
                entryMacros.push_back({"VSM_REFINED_RECEIVER", refinedReceiver ? "1" : "0"});
                entryMacros.push_back({"VSM_DIRECTIONAL_PROJECTION", directional ? "1" : "0"});
                // Programs already keys directional page sizes through markPageShift.
                entryMacros.push_back({"VSM_PROJECTION_PAGE_SHIFT", markPageShiftDefine.c_str()});
                entryMacros.push_back({"VSM_ADAPTIVE_RAY_COUNT", adaptiveRayCount ? "1" : "0"});
                if (!directional && programs->localSMRTStateReuse) {
                    entryMacros.push_back({"VSM_LOCAL_SMRT_STATE_REUSE", "1"});
                }
            }
            if (buildEntry) { entryMacros.push_back({"VSM_DIRECTIONAL_BUILD", directional ? "1" : "0"}); }
            if (instanceBuild && compactEntry) { entryMacros.push_back({"VSM_INSTANCE_BUILD", "1"}); }
            if (compactEntry) { entryMacros.push_back({"VSM_COMPACT_BUILD", compactBuild ? "1" : "0"}); }
            auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                .capabilities = capabilities, .macroDefines = entryMacros}, log);
            if (!shader) { return std::unexpected(shader.error()); }
            auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entries[index]});
            if (!module) { return std::unexpected(module.error()); }
            programs->shaders[index] = std::move(*module);
            if (index == 16) {
                const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                spdlog::info("[VSMHierarchyShader] entry={} spirvSHA256={} receiverMask={}", entries[index], debug::debugSha256(bytes), receiverMask);
            }
            if (receiverMask && (index == 0 || index == 4 || index == 11 || index == 13)) {
                const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                spdlog::info("[VSMReceiverMaskShader] entry={} persistentCoverage=true propagatedMapping={} spirvSHA256={}",
                    entries[index], true, debug::debugSha256(bytes));
            }
            if (index == 5 || index == 15) {
                const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                spdlog::info("[VSMMappingShader] entry={} propagatedMapping={} spirvSHA256={}",
                    entries[index], true, debug::debugSha256(bytes));
            }
            if (index == 1 || index == 3 || index == 5 || index == 6 || index == 7 || index == 14) {
                const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                spdlog::info("[VSMShader] entry={} spirvSHA256={} optimizeMark={} clusterCull={} clusterPageBitset={} work={} markSamples={} lightType={} trianglePageBitset={} projectionStatistics={} directionalBuild={} compactBuild={} directionalMark={} refinedReceiver={} markCacheLevels={} markRayPages={} directionalProjection={} markPageShift={} projectionPageShift={} adaptiveRayCount={} dirtyPageHierarchy={} receiverMask={} aggregateReceiverMask={}", entries[index],
                    debug::debugSha256(bytes), true,
                    true, true, settings.collectGPUWork, !directional, uint32_t(light.type), true, groupProjectionStatistics, buildEntry && directional, compactEntry && compactBuild, index == 1 && directional, index == 5 && refinedReceiver, index == 1 ? markCacheLevels : 12u, index == 1 && directional, index == 5 && directional, index == 1 ? markPageShift : -1, index == 5 ? markPageShift : -1, index == 5 && adaptiveRayCount, dirtyPageHierarchy && compactEntry, receiverMask, index == 1 && receiverMask);
            }
            if (index == 20) {
                const auto bytes = std::as_bytes(std::span(shader->spirv));
                spdlog::info("[VSMPackedRasterShader] entry={} spirvSHA256={}", entries[index],
                    debug::debugSha256({reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()}));
            }
            if (index < 6 || (index >= 10 && index < 20)) {
                auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {programs->shaders[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
                if (!pipeline) { return std::unexpected(pipeline.error()); }
                programs->compute[index < 6 ? index : index - 4] = std::move(*pipeline);
            }
            if (index >= 21) {
                auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {programs->shaders[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
                if (!pipeline) { return std::unexpected(pipeline.error()); }
                const std::array destinations{&programs->selectLOD, &programs->hybridClear, &programs->hybridRaster,
                    &programs->buildRasterHZB, &programs->retestRasterHZB,
                    &programs->prepareHierarchyHZB, &programs->resetHierarchyPost, &programs->retestHierarchyHZB,
                    &programs->finalizeHierarchyPost};
                *destinations[index - 21] = std::move(*pipeline);
            }
        }
        GraphicsPipelineDesc rasterDesc{.meshShader = {programs->shaders[6].get()},
            .fragmentShader = {programs->shaders[7].get()}, .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::GreaterEqual}, .usesBindlessHeap = true};
        if (auto r = ShaderRegistry::instance().getGraphicsPipeline(device, rasterDesc)) { programs->raster = std::move(*r); }
        else { return std::unexpected(r.error()); }
        if (groupedFineRaster) {
            rasterDesc.meshShader = {programs->shaders[20].get()};
            auto packed = ShaderRegistry::instance().getGraphicsPipeline(device, rasterDesc);
            if (!packed) { return std::unexpected(packed.error()); }
            programs->packedRaster = std::move(*packed);
        }
        auto clear = ShaderRegistry::instance().getGraphicsPipeline(device, {.meshShader = {programs->shaders[8].get()},
            .fragmentShader = {programs->shaders[9].get()},
            .depthStencilFormat = Format::D32Sfloat, .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::Always}, .usesBindlessHeap = true});
        if (!clear) { return std::unexpected(clear.error()); }
        programs->clear = std::move(*clear);
        programs_ = std::move(programs);
    }
    // Every submitted frame keeps its exact pipelines alive across a Coverage
    // edit. Cache snapshots own only the atlas and page table, never prior frame
    // State objects, so retaining a previous snapshot cannot form a frame chain.
    phase.next("Retain and bind VSM resources");
    total.work("leases", state->leases.size());
    state->programs = programs_;
    if (auto result = commands.retainResource(state); !result) { return std::unexpected(result.error()); }
    if (auto result = state->registry->bind(commands, state->leases); !result) { return std::unexpected(result.error()); }
    auto dispatch = [&](size_t pipeline, uint32_t x, uint32_t y = 1) -> Result<> {
        if (auto result = commands.bindExecution(state->programs->compute[pipeline]->execution(), &push, sizeof(push)); !result) { return result; }
        return commands.dispatch(x, y);
    };
    auto barrier = [&](std::initializer_list<size_t> buffers, PipelineStageBits stages = PipelineStageBits::ComputeShader,
        AccessBits access = AccessBits::ShaderRead | AccessBits::ShaderWrite) -> Result<> {
        std::vector<BufferBarrierDesc> barriers;
        for (auto index : buffers) {
            barriers.push_back({.buffer = state->resources->buffers[index].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {stages, access}});
        }
        return commands.synchronize({.buffers = barriers});
    };
    phase.next("Record VSM demand and build commands");
    if (gpuLOD) {
        auto lodGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM per-view hierarchy LOD") : RenderGraphExecutionContext::ProfileScope{};
        commands.hostWriteBarrier();
        std::vector<BufferBarrierDesc> lodReady;
        for (const auto& buffer : {state->resources->lodGroupStates, state->resources->lodSelection, state->resources->lodChanges}) {
            lodReady.push_back({.buffer = buffer.get(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
        }
        if (reuse && cache_->lodSelection) {
            lodReady.push_back({.buffer = cache_->lodSelection.get(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}});
        }
        if (auto r = commands.synchronize({.buffers = lodReady}); !r) { return std::unexpected(r.error()); }
        for (uint32_t base = 0; base < casters.gpuLOD->context.instanceCount; base += kDrawBatchSize) {
            auto lodPush = push; lodPush.workBase = base;
            if (auto r = commands.bindExecution(state->programs->selectLOD->execution(), &lodPush, sizeof(lodPush)); !r) { return std::unexpected(r.error()); }
            if (auto r = commands.dispatch(std::min(kDrawBatchSize, casters.gpuLOD->context.instanceCount - base), push.clipmapCount + 1); !r) { return std::unexpected(r.error()); }
        }
        lodReady.clear();
        for (const auto& buffer : {state->resources->lodSelection, state->resources->lodChanges, state->resources->lodStatistics}) {
            lodReady.push_back({.buffer = buffer.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::Transfer,
                    AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::TransferRead}});
        }
        if (auto r = commands.synchronize({.buffers = lodReady}); !r) { return std::unexpected(r.error()); }
    }
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM reset") : RenderGraphExecutionContext::ProfileScope{};
    auto resetGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM reset prepare") : RenderGraphExecutionContext::ProfileScope{};
    commands.hostWriteBarrier();
    std::array<BufferBarrierDesc, 5> recycledReady;
    constexpr std::array<size_t, 5> deviceSlots{1, 3, 4, 7, 9};
    for (size_t index = 0; index < deviceSlots.size(); ++index) {
        recycledReady[index] = {.buffer = state->resources->buffers[deviceSlots[index]].get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    }
    if (auto r = commands.synchronize({.buffers = recycledReady}); !r) { return std::unexpected(r.error()); }
    if (staticPageHZB) {
        const BufferBarrierDesc ready[]{
            {.buffer = state->resources->staticHZB.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
            {.buffer = state->resources->staticHZBJobs.get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}}};
        if (auto r = commands.synchronize({.buffers = ready}); !r) { return std::unexpected(r.error()); }
    }
    if (reuse) {
        const BufferBarrierDesc previousReady{.buffer = cache_->pages.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite}, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
        if (auto r = commands.synchronize({.buffers = {&previousReady, 1}}); !r) { return std::unexpected(r.error()); }
    }
    if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM: demand and cache allocation"}); }
    if (rectangleCapacity != 0) {
        resetGpu.next("VSM project changed caster bounds");
        if (auto r = dispatch(15, remapLevelCount, mergeStatic ? 3u : layered ? 2u : 1u); !r) { return std::unexpected(r.error()); }
        if (auto r = barrier({8}); !r) { return std::unexpected(r.error()); }
    }
    // Reset writes diagnostics at statistics[32 + id.x], independently of the public 32 words.
    const uint32_t statisticsResetThreads = settings.collectGPUWork ?
        (kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / sizeof(uint32_t) : 32u;
    resetGpu.next("VSM reset dispatch");
    if (auto r = dispatch(0, (std::max({push.totalPages, push.physicalPageCount, statisticsResetThreads, hierarchyWords, receiverMaskWords,
        (push.batchCount + push.coarseBatchCount + compactBatches) * 3}) + 63) / 64); !r) { return std::unexpected(r.error()); }
    resetGpu.next("VSM reset publish");
    if (auto r = barrier(groupedFineRaster ? std::initializer_list<size_t>{1, 3, 4, 7, 9} :
        std::initializer_list<size_t>{1, 4, 7, 9}); !r) { return std::unexpected(r.error()); }
    resetGpu.end();
    if (!sharedPhysicalPool || !settings.unifiedMark) {
        gpu.next("VSM mark");
        auto markGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM mark dispatch") : RenderGraphExecutionContext::ProfileScope{};
        if (!singlePage) {
            if (auto r = dispatch(1, (width + 7) / 8, (height + 7) / 8); !r) { return std::unexpected(r.error()); }
        }
        markGpu.next("VSM mark publish");
        if (receiverMask) {
            if (auto r = barrier({9}); !r) { return std::unexpected(r.error()); }
        }
        if (auto r = barrier(settings.collectGPUWork ? std::initializer_list<size_t>{1, 4} : std::initializer_list<size_t>{1});
            !r) { return std::unexpected(r.error()); }
        markGpu.end();
        gpu.next("VSM allocator");
        // Each allocator child includes its publication dependency. Keep the
        // aggregate names stable for existing stage reports; never sum parent + child.
        auto allocatorGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM allocator count") : RenderGraphExecutionContext::ProfileScope{};
        if (auto r = dispatch(6, allocationBlocks); !r) { return std::unexpected(r.error()); }
        if (auto r = barrier({4, 9}); !r) { return std::unexpected(r.error()); }
        allocatorGpu.next("VSM allocator allocate");
        if (auto r = dispatch(2, 1); !r) { return std::unexpected(r.error()); }
        if (auto r = barrier({4, 9}); !r) { return std::unexpected(r.error()); }
        if (!sharedPhysicalPool) {
            allocatorGpu.next("VSM allocator admit");
            if (auto r = dispatch(7, allocationBlocks); !r) { return std::unexpected(r.error()); }
            if (auto r = barrier({1, 4, 9}); !r) { return std::unexpected(r.error()); }
            allocatorGpu.next("VSM allocator physical");
            if (auto r = dispatch(8, 1); !r) { return std::unexpected(r.error()); }
            if (auto r = barrier({1, 4, 9}); !r) { return std::unexpected(r.error()); }
            allocatorGpu.next("VSM allocator assign");
            if (auto r = dispatch(9, allocationBlocks); !r) { return std::unexpected(r.error()); }
            if (auto r = barrier({1, 9}); !r) { return std::unexpected(r.error()); }
        }
        allocatorGpu.end();
    }
    gpu.end();
    if (!gpuProfiler) { commands.endDebugLabel(); }
    // Store values, never references to this record() stack. State owns the
    // callback and commands retain State until completion; the raw self pointer
    // avoids a shared_ptr cycle. Shared allocation resolves before it is called.
    state->batch = {.hierarchyGroups = (clusterPageBitsetWords + 63u) / 64u,
        .buildGroups = singlePage ? 0u : hierarchyHZB && reuse ? casters.hierarchyRootCount :
            instanceBuild ? std::max(instanceCount, cacheDiagnostics[kVSMCacheMergeClusterCount]) : clusterCapacity,
        .compactBatches = compactBatches, .hierarchyHZBBatches = hierarchyHZBBatches,
        .visibilityBatches = visibilityBatches, .mergeDrawOffset = mergeDrawOffset,
        .hybridDrawOffset = hybridDrawOffset,
        .staticGeometry = cacheDiagnostics[33], .dynamicGeometry = cacheDiagnostics[34], .hzbJobs = cacheDiagnostics[38],
        .mergeStatic = mergeStatic};
    state->finish = [=, state = state.get(), &commands]() -> Result<> {
        auto& push = state->push;
        CPUProfileScope totals(profiler, "Record VSM build and raster");
        CPUProfileScope phase(profiler, "Record VSM build stages");
        uint64_t computeDispatches = 0, meshDraws = 0, renderingPasses = 0;
        auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM build") : RenderGraphExecutionContext::ProfileScope{};
        if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM: build"}); }
        auto dispatch = [&](size_t pipeline, uint32_t x, uint32_t y = 1) -> Result<> {
            if (auto r = commands.bindExecution(state->programs->compute[pipeline]->execution(), &push, sizeof(push)); !r) { return r; }
            if (auto r = commands.dispatch(x, y); !r) { return r; }
            ++computeDispatches;
            return {};
        };
        auto barrier = [&](std::initializer_list<size_t> buffers, PipelineStageBits stages = PipelineStageBits::ComputeShader,
            AccessBits access = AccessBits::ShaderRead | AccessBits::ShaderWrite) -> Result<> {
            std::vector<BufferBarrierDesc> barriers;
            for (auto index : buffers) {
                barriers.push_back({.buffer = state->resources->buffers[index].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {stages, access}});
            }
            return commands.synchronize({.buffers = barriers});
        };

        auto syncSharedHZB = [&]() -> Result<> {
            if (!sharedPhysicalPool) { return {}; }
            std::vector<BufferBarrierDesc> barriers;
            for (const auto& bank : state->resources->physicalPoolStorage->hzbBanks) {
                barriers.push_back({.buffer = bank.get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
            }
            return commands.synchronize({.buffers = barriers});
        };
        if (dirtyPageHierarchy) {
            auto hierarchyGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM build page hierarchy") :
                RenderGraphExecutionContext::ProfileScope{};
            if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM: build page hierarchy"}); }
            if (auto r = dispatch(12, (clusterPageBitsetWords + 63u) / 64u); !r) { return std::unexpected(r.error()); }
            if (auto r = barrier({9}); !r) { return std::unexpected(r.error()); }
            if (!gpuProfiler) { commands.endDebugLabel(); }
        }
        auto buildGpu = gpuProfiler && settings.profileBuildSubstages ?
            gpuProfiler->profileScope(commands, instanceBuild ? "VSM build instance cull" : compactBuild ? "VSM build compact" : "VSM build direct") :
            RenderGraphExecutionContext::ProfileScope{};
        const bool buildTriangles = !singlePage && push.triangleCount && enabled;
        if (buildTriangles) {
            // A merge-only frame can directly select its short cluster list on GPU.
            const uint32_t groups = instanceBuild ? std::max(cacheDiagnostics[kVSMCacheInstanceCount],
                cacheDiagnostics[kVSMCacheMergeClusterCount]) : clusterCapacity;
            for (uint32_t base = 0; base < groups;) {
                push.workBase = instanceBuild ? base : base * 64u;
                const uint32_t count = std::min(groups - base, kDrawBatchSize);
                if (auto r = dispatch(compactBuild ? 10 : 3, count); !r) { return std::unexpected(r.error()); }
                base += count;
                if (base < groups) {
                    if (auto r = barrier(compactBuild ? std::initializer_list<size_t>{4, 7, 9} : std::initializer_list<size_t>{3, 4});
                        !r) { return std::unexpected(r.error()); }
                }
            }
        }
        if (compactBuild) {
            buildGpu.next("VSM build compact sync");
            if (buildTriangles) {
                // Both consumers share the compute-write source scope, so one
                // publication dependency can cover shader and indirect reads.
                if (auto r = barrier({9, 4, 7}, PipelineStageBits::ComputeShader | PipelineStageBits::DrawIndirect,
                        AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::IndirectRead);
                    !r) { return std::unexpected(r.error()); }
            }
            buildGpu.next("VSM build indirect");
            if (buildTriangles) {
                for (uint32_t batch = 0; batch < compactBatches; ++batch) {
                    push.workBase = batch * kDrawBatchSize;
                    if (auto r = commands.bindExecution(state->programs->compute[3]->execution(), &push, sizeof(push));
                        !r) { return std::unexpected(r.error()); }
                    auto arguments = state->resources->buffers[7]->slice({compactArgumentOffset + uint64_t(batch) * 12u, 12});
                    if (!arguments) { return std::unexpected(arguments.error()); }
                    if (auto r = commands.dispatchIndirect(*arguments); !r) { return std::unexpected(r.error()); }
                    ++computeDispatches;
                    if (batch + 1 < compactBatches) {
                        if (auto r = barrier({3, 4}); !r) { return std::unexpected(r.error()); }
                    }
                }
                // Finalize's draw arguments are disjoint from these compact dispatch arguments.
            }
        }
        buildGpu.next("VSM build finalize sync");
        if (auto r = barrier(groupedFineRaster ? std::initializer_list<size_t>{3, 4} :
            std::initializer_list<size_t>{4}); !r) { return std::unexpected(r.error()); }
        buildGpu.next("VSM build finalize");
        if (staticPageHZB) {
            const BufferBarrierDesc jobsReady{.buffer = state->resources->staticHZBJobs.get(),
                .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
            if (auto r = commands.synchronize({.buffers = {&jobsReady, 1}}); !r) { return std::unexpected(r.error()); }
        }
        if (auto r = dispatch(4, (std::max(push.totalPages, push.batchCount + push.coarseBatchCount) + 63) / 64); !r) { return std::unexpected(r.error()); }
        if (!singlePage) {
            // Finalize may invalidate fine pages on work overflow. Read only its
            // completed ownership entries; no propagation thread consumes the tail.
            if (auto r = barrier({1}); !r) { return std::unexpected(r.error()); }
            if (staticPageHZB) {
                const BufferBarrierDesc jobsReady{.buffer = state->resources->staticHZBJobs.get(),
                    .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                if (auto r = commands.synchronize({.buffers = {&jobsReady, 1}}); !r) { return std::unexpected(r.error()); }
            }
            auto mappingGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM propagate mappings") : RenderGraphExecutionContext::ProfileScope{};
            if (auto r = dispatch(11, (push.totalPages + 63u) / 64u); !r) { return std::unexpected(r.error()); }
        }
        if (auto r = barrier({3, 1, 9}, PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderRead); !r) { return std::unexpected(r.error()); }
        if (auto r = barrier({7}, PipelineStageBits::DrawIndirect, AccessBits::IndirectRead); !r) { return std::unexpected(r.error()); }
        const auto rasterStages = classifyFine ? PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization :
            PipelineStageBits::PreRasterization;
        if (auto r = barrier({4}, settings.collectGPUWork ? rasterStages : PipelineStageBits::ComputeShader);
            !r) { return std::unexpected(r.error()); }
        buildGpu.end();
        if (!gpuProfiler) { commands.endDebugLabel(); }
        gpu.next("VSM raster");
        phase.next("Record VSM raster commands");
        if (rasterFeedback) {
            if (auto r = rasterFeedback->timestamp(commands); !r) { return r; }
        }
        phase.work("coarseBatches", push.coarseBatchCount);
        phase.work("fineBatches", push.batchCount);
        const uint32_t fullGeometry = push.geometry;
        const BufferBarrierDesc diagnosticReady{.buffer = state->resources->cacheDiagnostics.get(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {rasterStages | PipelineStageBits::FragmentShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        if (auto r = commands.synchronize({.buffers = {&diagnosticReady, 1}}); !r) { return std::unexpected(r.error()); }
        // Cache compatibility includes enabled state. A previously disabled face
        // already owns cleared, shader-readable atlases. Cold faces still clear them.
        const uint32_t rasterLayers = reuse && !enabled ? 0u : (layered ? 2u : 1u);
        phase.work("recordedRasterLayers", rasterLayers);
        for (uint32_t layer = 0; layer < rasterLayers; ++layer) {
            const bool staticPass = layered && layer == 0;
            auto layerGpu = gpuProfiler && layered ? gpuProfiler->profileScope(commands,
                staticPass ? "VSM static raster" : "VSM dynamic raster and restore") : RenderGraphExecutionContext::ProfileScope{};
            auto rasterGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM raster prepare") : RenderGraphExecutionContext::ProfileScope{};
            auto* rasterTexture = staticPass ? state->resources->staticAtlas->texture.get() : state->resources->textures[0].get();
            auto* rasterView = staticPass ? state->resources->staticAtlas->view.get() : state->resources->views[0].get();
            push.geometry = layered ? cacheDiagnostics[33 + layer] : fullGeometry;
            const uint32_t rasterLayer = staticPass ? 2u : (visibilityBatches ? 4u : 0u);
            const uint32_t drawOffset = mergeStatic && !staticPass ? mergeDrawOffset : 0u;
            const uint32_t mainFineOffset = classifyFine ?
                hybridDrawOffset + (staticPass ? push.batchCount : 0u) : drawOffset;
            const bool retestHistory = twoPassHZB && reuse && enabled;
            const auto dispatchBatches = [&](ComputePipeline& pipeline, uint32_t count, uint32_t offset) -> Result<> {
                for (uint32_t batch = 0; batch < count; ++batch) {
                    push.workBase = batch * kDrawBatchSize;
                    if (auto r = commands.bindExecution(pipeline.execution(), &push, sizeof(push)); !r) { return r; }
                    auto args = state->resources->buffers[7]->slice({uint64_t(offset + batch) * 12u, 12});
                    if (!args) { return std::unexpected(args.error()); }
                    if (auto r = commands.dispatchIndirect(*args); !r) { return r; }
                    ++computeDispatches;
                }
                return {};
            };
            // The same queue is consumed by mesh draws and then reused by the
            // post classifier and by the following layer. Arguments are read by
            // DrawIndirect while classifiers append to disjoint argument slices.
            const auto syncTicketBuffers = [&]() -> Result<> {
                const SyncScope before{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
                const SyncScope shaders{PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader,
                    AccessBits::ShaderRead | AccessBits::ShaderWrite};
                const BufferBarrierDesc buffers[]{
                    {.buffer = state->resources->buffers[3].get(), .before = before, .after = shaders},
                    {.buffer = state->resources->buffers[4].get(), .before = before, .after = shaders},
                    {.buffer = state->resources->cacheDiagnostics.get(), .before = before, .after = shaders},
                    {.buffer = state->resources->buffers[7].get(), .before = before,
                        .after = {PipelineStageBits::ComputeShader | PipelineStageBits::DrawIndirect,
                            AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::IndirectRead}}};
                return commands.synchronize({.buffers = buffers});
            };
            const auto buildRasterHZB = [&](bool current) -> Result<> {
                rasterGpu.next(current ? "VSM current page HZB" : "VSM history page HZB");
                BufferBarrierDesc hzb{.buffer = state->resources->rasterHZB.get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
                if (auto r = commands.synchronize({.buffers = {&hzb, 1}}); !r) { return r; }
                const uint32_t maxPages = std::min(push.totalPages - 1u, push.physicalPageCount);
                const uint32_t pageBatches = (maxPages + kDrawBatchSize - 1u) / kDrawBatchSize;
                const uint32_t pageOffset = hybridDrawOffset + 6u * push.batchCount + (staticPass ? pageBatches : 0u);
                push.rasterCoarse = rasterLayer | (current ? 8u : 0u);
                if (auto r = dispatchBatches(*state->programs->buildRasterHZB, pageBatches, pageOffset); !r) { return r; }
                push.rasterCoarse = rasterLayer;
                hzb.before = hzb.after;
                hzb.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead};
                return commands.synchronize({.buffers = {&hzb, 1}});
            };
            push.rasterCoarse = rasterLayer;
            if (classifyFine && enabled) {
                if (auto r = syncTicketBuffers(); !r) { return r; }
                if (retestHistory) {
                    if (auto r = buildRasterHZB(false); !r) { return r; }
                }
                BufferBarrierDesc scratch{.buffer = state->resources->hybridDepth.get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}};
                if (clusterHybridRaster) {
                    rasterGpu.next("VSM software cluster raster");
                    if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
                    const uint32_t scratchPages = hybridScratchPageCount(push.physicalPageCount, push.pageSize);
                    for (uint32_t base = 0; base < scratchPages; base += kDrawBatchSize) {
                        push.workBase = base;
                        if (auto r = commands.bindExecution(state->programs->hybridClear->execution(), &push, sizeof(push)); !r) { return r; }
                        if (auto r = commands.dispatch(std::min(scratchPages - base, kDrawBatchSize)); !r) { return r; }
                        ++computeDispatches;
                    }
                    scratch.before = scratch.after;
                    scratch.after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
                    if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
                } else { rasterGpu.next("VSM main cluster HZB cull"); }
                if (auto r = dispatchBatches(*state->programs->hybridRaster, push.batchCount, drawOffset); !r) { return r; }
                if (clusterHybridRaster) {
                    scratch.before = scratch.after;
                    scratch.after = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead};
                    if (auto r = commands.synchronize({.buffers = {&scratch, 1}}); !r) { return r; }
                }
                if (auto r = syncTicketBuffers(); !r) { return r; }
            }
            const auto draw = [&](bool mainPass) -> Result<> {
                const bool loadDepth = !mainPass || reuse || sharedPhysicalPool;
                const TextureBarrierDesc depthBegin{.texture = rasterTexture, .oldLayout = loadDepth ? TextureLayout::ShaderRead : TextureLayout::Undefined,
                    .newLayout = TextureLayout::DepthStencilAttachment,
                    .before = {PipelineStageBits::AllCommands, loadDepth ? AccessBits::MemoryRead | AccessBits::MemoryWrite : AccessBits::None},
                    .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
                    .range = {0, 1, 0, rasterTexture->desc().layerCount}};
                if (auto r = commands.synchronize({.textures = {&depthBegin, 1}}); !r) { return std::unexpected(r.error()); }
                const Rect area{.width = push.physicalPagesX * push.pageSize, .height = push.physicalPagesY * push.pageSize};
                const RenderingAttachmentDesc attachment{.view = rasterView, .layout = TextureLayout::DepthStencilAttachment,
                    .loadOp = loadDepth ? LoadOp::Load : LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = 0};
                if (!gpuProfiler) { commands.beginDebugLabel({.name = mainPass ? "VSM: dirty coarse and fine depth" : "VSM: recovered fine depth"}); }
                if (auto r = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &attachment, .layerCount = rasterTexture->desc().layerCount}); !r) { return std::unexpected(r.error()); }
                ++renderingPasses;
                if (auto r = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1}); !r) { return std::unexpected(r.error()); }
                commands.setScissor(area);
                if (enabled && mainPass) {
                    rasterGpu.next("VSM raster clear and restore");
                    if (clusterHybridRaster || reuse || sharedPhysicalPool || (layered && !staticPass)) {
                        push.rasterCoarse = staticPass ? 2u : 0u;
                        push.workBase = 0;
                        if (auto r = commands.bindExecution(state->programs->clear->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                        if (clusterPageRaster && !sharedPhysicalPool) {
                            auto args = state->resources->buffers[7]->slice({uint64_t(clearDrawOffset) * 12, 12});
                            if (!args) { return std::unexpected(args.error()); }
                            if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
                            ++meshDraws;
                        } else {
                            for (uint32_t base = 0; base < push.physicalPageCount; base += kDrawBatchSize) {
                                push.workBase = base;
                                if (auto r = commands.bindExecution(state->programs->clear->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                                if (auto r = commands.drawMeshTasks(std::min(kDrawBatchSize, push.physicalPageCount - base)); !r) { return std::unexpected(r.error()); }
                                ++meshDraws;
                            }
                        }
                    }
                    rasterGpu.next("VSM raster coarse");
                    push.rasterCoarse = rasterLayer | 1u;
                    for (uint32_t batch = 0; batch < push.coarseBatchCount; ++batch) {
                        push.workBase = batch * kDrawBatchSize;
                        if (auto r = commands.bindExecution(state->programs->raster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                        auto args = state->resources->buffers[7]->slice({uint64_t(drawOffset + push.batchCount + batch) * 12, 12});
                        if (!args) { return std::unexpected(args.error()); }
                        if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
                        ++meshDraws;
                    }
                }
                if (enabled) {
                    rasterGpu.next(mainPass ? "VSM raster fine" : "VSM post raster fine");
                    push.rasterCoarse = rasterLayer;
                    const auto& fineRaster = groupedFineRaster ? state->programs->packedRaster : state->programs->raster;
                    const uint32_t fineDrawOffset = mainPass ? mainFineOffset : mainFineOffset + 4u * push.batchCount;
                    for (uint32_t batch = 0; batch < push.batchCount; ++batch) {
                        push.workBase = batch * kDrawBatchSize;
                        if (auto r = commands.bindExecution(fineRaster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                        auto args = state->resources->buffers[7]->slice({uint64_t(fineDrawOffset + batch) * 12, 12});
                        if (!args) { return std::unexpected(args.error()); }
                        if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
                        ++meshDraws;
                    }
                }
                rasterGpu.next("VSM raster publish");
                commands.endRendering();
                if (!gpuProfiler) { commands.endDebugLabel(); }
                const TextureBarrierDesc atlasReady{.texture = rasterTexture,
                    .oldLayout = TextureLayout::DepthStencilAttachment, .newLayout = TextureLayout::ShaderRead,
                    .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
                    .after = {PipelineStageBits::ComputeShader | PipelineStageBits::FragmentShader, AccessBits::ShaderRead},
                    .range = {0, 1, 0, rasterTexture->desc().layerCount}};
                return commands.synchronize({.textures = {&atlasReady, 1}});
            };
            if (auto r = draw(true); !r) { return r; }
            if (retestHistory) {
                if (auto r = syncTicketBuffers(); !r) { return r; }
                if (auto r = buildRasterHZB(true); !r) { return r; }
                rasterGpu.next("VSM post cluster HZB retest");
                if (auto r = dispatchBatches(*state->programs->retestRasterHZB, push.batchCount,
                    mainFineOffset + 2u * push.batchCount); !r) { return r; }
                if (auto r = syncTicketBuffers(); !r) { return r; }
                if (auto r = draw(false); !r) { return r; }
            }
            rasterGpu.end();
            layerGpu.end();
            if (staticPass && staticPageHZB) {
                auto hzbGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM static page HZB") : RenderGraphExecutionContext::ProfileScope{};
                if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM static page HZB"}); }
                // Reset and Build have finished querying the old pyramid. All static
                // depth writes are available via atlasReady, including clears.
                const BufferBarrierDesc ready[]{
                    {.buffer = state->resources->staticHZB.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
                    {.buffer = state->resources->staticHZBJobs.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader | PipelineStageBits::DrawIndirect, AccessBits::ShaderRead | AccessBits::IndirectRead}},
                    {.buffer = state->resources->cacheDiagnostics.get(), .before = {PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}}};
                if (auto r = commands.synchronize({.buffers = ready}); !r) { return std::unexpected(r.error()); }
                if (auto r = syncSharedHZB(); !r) { return r; }
                auto args = state->resources->staticHZBJobs->slice({uint64_t(3u + push.physicalPageCount) * sizeof(uint32_t), 12});
                if (!args) { return std::unexpected(args.error()); }
                if (auto r = commands.bindExecution(state->programs->compute[13]->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                if (auto r = commands.dispatchIndirect(*args); !r) { return std::unexpected(r.error()); }
                ++computeDispatches;
                if (auto r = syncSharedHZB(); !r) { return r; }
                if (!gpuProfiler) { commands.endDebugLabel(); }
            }
            if (staticPass && visibilityBatches) {
                auto cullGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM dynamic cluster HZB cull") : RenderGraphExecutionContext::ProfileScope{};
                if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM dynamic cluster HZB cull"}); }
                std::vector<BufferBarrierDesc> cullReady{
                    {.buffer = state->resources->staticHZB.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
                    {.buffer = state->resources->clusterVisibility.get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
                    {.buffer = state->resources->cacheDiagnostics.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}}};
                if (settings.collectGPUWork) {
                    cullReady.push_back({.buffer = state->resources->buffers[4].get(),
                        .before = {PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
                }
                if (auto r = commands.synchronize({.buffers = cullReady}); !r) { return std::unexpected(r.error()); }
                push.geometry = cacheDiagnostics[34]; // The dynamic caster layer preserves the original cluster indices.
                for (uint32_t batch = 0; batch < visibilityBatches; ++batch) {
                    push.workBase = batch * kDrawBatchSize;
                    if (auto r = commands.bindExecution(state->programs->compute[14]->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                    auto args = state->resources->staticHZBJobs->slice({uint64_t(6u + push.physicalPageCount + batch * 3u) * sizeof(uint32_t), 12});
                    if (!args) { return std::unexpected(args.error()); }
                    if (auto r = commands.dispatchIndirect(*args); !r) { return std::unexpected(r.error()); }
                    ++computeDispatches;
                }
                const BufferBarrierDesc masksReady{.buffer = state->resources->clusterVisibility.get(),
                    .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {rasterStages, AccessBits::ShaderRead}};
                if (auto r = commands.synchronize({.buffers = {&masksReady, 1}}); !r) { return std::unexpected(r.error()); }
                if (!gpuProfiler) { commands.endDebugLabel(); }
            }
            if (layered && staticPass) {
                std::vector<BufferBarrierDesc> layerReady{{.buffer = state->resources->cacheDiagnostics.get(),
                    .before = {PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {rasterStages | PipelineStageBits::FragmentShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}}};
                if (settings.collectGPUWork) {
                    layerReady.push_back({.buffer = state->resources->buffers[4].get(),
                        .before = {PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {rasterStages, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
                }
                if (auto r = commands.synchronize({.buffers = layerReady}); !r) { return std::unexpected(r.error()); }
            }
        }
        if (rasterFeedback) {
            if (auto r = rasterFeedback->timestamp(commands); !r) { return r; }
        }
        push.geometry = fullGeometry;
        push.rasterCoarse = 0;
        auto diagnosticGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM raster diagnostics publish") : RenderGraphExecutionContext::ProfileScope{};
        const BufferBarrierDesc diagnosticPublished{.buffer = state->resources->cacheDiagnostics.get(),
            .before = {PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
        if (auto r = commands.synchronize({.buffers = {&diagnosticPublished, 1}}); !r) { return std::unexpected(r.error()); }
        if (settings.collectGPUWork) {
            const BufferBarrierDesc workReady{.buffer = state->resources->buffers[4].get(),
                .before = {PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::ComputeShader | PipelineStageBits::Transfer,
                    AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::TransferRead}};
            if (auto r = commands.synchronize({.buffers = {&workReady, 1}}); !r) { return std::unexpected(r.error()); }
        }
        diagnosticGpu.end();
        gpu.end();
        totals.work("faces", 1); totals.work("computeDispatches", computeDispatches);
        totals.work("prepareDispatches", 0); totals.work("meshDraws", meshDraws);
        totals.work("renderingPasses", renderingPasses);
        totals.work("computeBins", 0); totals.work("rasterBins", 0);
        return {};
    };

    if (!sharedPhysicalPool) {
        if (auto r = state->finish(); !r) { return std::unexpected(r.error()); }
        state->finish = {};
        if (gpuLOD && casters.gpuLOD->stream) {
            if (auto r = casters.gpuLOD->stream->cmdCollectShadowPageRequests(commands,
                *casters.gpuLOD->pageRequests, casters.gpuLOD->pageCount); !r) { return std::unexpected(r.error()); }
        }
    }
    phase.next("Prepare VSM cache snapshot");
    if (settings.cacheEnabled) {
        auto next = std::make_shared<Cache>();
        next->device = &device;
        if (sharedPhysicalPool) {
            next->physicalPoolStorage = poolFrame->storage;
            next->poolSerial = poolFrame->allocationSerial;
        } else {
            next->atlas = std::make_shared<ShadowAtlas>(ShadowAtlas{state->resources->textures[0], state->resources->views[0]});
            next->staticAtlas = state->resources->staticAtlas;
            next->staticHZB = state->resources->staticHZB;
        }
        next->pages = state->resources->buffers[1]; next->projections = std::move(projections); next->origins = origins;
        next->lodSelection = state->resources->lodSelection;
        next->lodLayoutKey = gpuLOD ? casters.gpuLOD->layoutKey : 0;
        next->casterOwner = casters.owner;
        next->casters = sharedPhysicalPool ? poolFrame->casters : unchangedCasters ? cache_->casters :
            std::make_shared<const std::vector<VirtualShadowMapCasterState>>(casters.states);
        next->settings = settings; next->light = light; next->farPlane = farPlane;
        next->receiverMask = receiverMask;
        next->sceneIdentity = casters.sceneIdentity;
        next->dynamic = std::move(dynamic); next->cacheFrame = cacheFrame;
        next->recordedFrame = frame->frameIndex();
        next->geometryRevision = casters.geometryRevision; next->texturePublicationRevision = casters.texturePublicationRevision;
        next->tessellationKey = casters.tessellationKey;
        next->dependencyKey = casters.dependencyKey; next->epoch = push.cacheEpoch; next->enabled = enabled;
        next->completion = frame->completion();
        next->submission = sharedPhysicalPool ? poolFrame->submission : std::make_shared<SubmissionTransaction>(nullptr, nullptr);
        if (auto r = sharedPhysicalPool ? Result<>{} : commands.addSubmissionTransaction(next->submission); !r) { return std::unexpected(r.error()); }
        state->next = next;
        pending_ = std::move(next);
    }
    phase.next("Publish VSM recording state");
    state_ = state;
    phase.next("Project VSM immediately");
    if (!deferProjection) {
        if (auto r = project(commands, {}, log, profiler, gpuProfiler); !r) { return std::unexpected(r.error()); }
    }
    phase.next("Finish VSM record");
    return VirtualShadowMapResult{state->outputTextures[0], state->resources->views[1].get(), state->outputTextures[1], state->resources->views[2].get(),
        state->resources->buffers[4].get(), state->resources->textures[0].get(), state->resources->views[0].get(), state->resources->buffers[1].get(), state->resources->buffers[5].get(), push.physicalPagesX, push.clipmapCount, push.pageSize,
        state->resources, state->resources->cacheDiagnostics.get(),
        receiverMask ? state->resources->buffers[9].get() : nullptr,
        (allocationWords + clusterPageBitsetWords) * sizeof(uint32_t),
        state->resources->lodSelection.get(), state->resources->lodStatistics.get(), state->resources->lodChanges.get(),
        gpuLOD ? casters.gpuLOD->context.clusterCount : 0, gpuLOD ? casters.gpuLOD->context.instanceCount : 0};
}

Result<> VirtualShadowMaps::preparePointFaces(CommandBuffer& commands,
    std::span<const VirtualShadowMapResult> pointFaces, std::string& log)
{
    auto& state = *state_;
    auto& push = state.push;
    if (!pointFaces.empty()) {
        if (pointFaces.size() != 6) {
            log += "Point VSM projection requires all six ordered faces.\n";
            return makeError(Error::InvalidArgument);
        }
        std::array<ShadowFace, 6> faces{};
        std::vector<BufferBarrierDesc> ready;
        for (size_t index = 0; index < faces.size(); ++index) {
            const auto& result = pointFaces[index];
            if (auto r = commands.retainResource(result.resourceOwner); !r) { return r; }
            auto atlas = state.registry->sampledImage(*result.atlasView);
            auto pages = state.registry->storageBuffer(*result.pageTable);
            auto projections = state.registry->storageBuffer(*result.projections);
            auto statistics = state.registry->storageBuffer(*result.statistics);
            if (!atlas) { return std::unexpected(atlas.error()); }
            if (!pages) { return std::unexpected(pages.error()); }
            if (!projections) { return std::unexpected(projections.error()); }
            if (!statistics) { return std::unexpected(statistics.error()); }
            state.leases.insert(state.leases.end(), {*atlas, *pages, *projections, *statistics});
            faces[index] = {atlas->shaderIndex(), pages->shaderIndex(), projections->shaderIndex(), statistics->shaderIndex(),
                result.physicalPagesX, result.clipmapCount, result.pageSize, 0};
            ready.push_back({.buffer = result.statistics,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
        }
        if (auto r = commands.synchronize({.buffers = ready}); !r) { return r; }
        auto* frame = RenderFrameContext::from(commands);
        auto table = state.pool->buffer(10, {.size = sizeof(faces), .structureStride = sizeof(ShadowFace),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload},
            frame && frame->recording() ? frame->completion() : GPUCompletionPoint{});
        if (!table) { return std::unexpected(table.error()); }
        state.resources->pointFaces = std::move(*table);
        void* mapped = state.resources->pointFaces->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, faces.data(), sizeof(faces));
        state.resources->pointFaces->flush(); state.resources->pointFaces->unmap();
        auto lease = state.registry->storageBuffer(*state.resources->pointFaces);
        if (!lease) { return std::unexpected(lease.error()); }
        push.pointFaces = lease->shaderIndex(); state.leases.push_back(*lease);
        commands.hostWriteBarrier();
    }
    return {};
}

Result<> VirtualShadowMaps::project(CommandBuffer& commands,
    std::span<const VirtualShadowMapResult> pointFaces, std::string& log, CPUProfileRecorder* profiler,
    RenderGraphExecutionContext* gpuProfiler)
{
    CPUProfileScope total(profiler, "VSM face project");
    total.work("pointFaces", pointFaces.size());
    if (!state_ || !state_->projectionPending) {
        log += "VSM project requires a recorded, unresolved face.\n";
        return makeError(Error::InvalidArgument);
    }
    auto& state = *state_;
    auto& push = state.push;
    total.work("lightType", push.lightType);
    total.work("face", push.projectionFace);
    total.work("triangles", push.triangleCount);
    total.work("totalPages", push.totalPages);
    if (push.lightType == uint32_t(VirtualShadowMapLightType::PointFace) &&
        push.filterMode != uint32_t(VirtualShadowMapFilter::Hard) && pointFaces.size() != 6) {
        log += "Filtered point VSM projection requires all six ordered faces.\n";
        return makeError(Error::InvalidArgument);
    }
    CPUProfileScope phase(profiler, "Prepare VSM point face table");
    const auto poolBefore = state.pool->stats;
    if (auto r = preparePointFaces(commands, pointFaces, log); !r) { return r; }
    phase.work("bufferCreates", state.pool->stats.bufferCreates - poolBefore.bufferCreates);
    phase.work("bufferReuses", state.pool->stats.bufferReuses - poolBefore.bufferReuses);
    phase.work("bufferEvictions", state.pool->stats.bufferEvictions - poolBefore.bufferEvictions);
    phase.end();
    total.work("pointTables", pointFaces.empty() ? 0u : 1u);
    total.work("leases", state.leases.size());
    return projectPrepared(commands, profiler, gpuProfiler);
}

Result<> VirtualShadowMaps::projectPrepared(CommandBuffer& commands, CPUProfileRecorder* profiler,
    RenderGraphExecutionContext* gpuProfiler)
{
    auto& state = *state_;
    auto& push = state.push;
    if (auto r = commands.retainResource(state_); !r) { return r; }
    CPUProfileScope phase(profiler, "Bind VSM projection resources");
    if (auto r = state.registry->bind(commands, state.leases); !r) { return r; }
    phase.next("Record VSM projection barriers and dispatch");
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM projection") : RenderGraphExecutionContext::ProfileScope{};
    auto projectionGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM projection prepare") : RenderGraphExecutionContext::ProfileScope{};
    if (state.externalOutput) {
        // Point faces share one layer. Canonical ownership makes their writes
        // disjoint, while the explicit dependency also satisfies whole-view
        // synchronization validation and orders any prior initialization.
        std::vector<TextureBarrierDesc> sharedReady;
        for (size_t index = 0; index < state.outputTextures.size(); ++index) {
            if (!state.outputTextures[index]) { continue; }
            sharedReady.push_back({.texture = state.outputTextures[index], .oldLayout = TextureLayout::General, .newLayout = TextureLayout::General,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .range = state.resources->views[index + 1]->desc().range});
        }
        if (auto r = commands.synchronize({.textures = sharedReady}); !r) { return r; }
    }
    const TextureBarrierDesc outputBegin[]{
        {.texture = state.resources->textures[1].get(), .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::General,
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
        {.texture = state.resources->textures[2].get(), .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::General,
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}}};
    if (!state.externalOutput) {
        if (auto r = commands.synchronize({.textures = outputBegin}); !r) { return r; }
    }
    projectionGpu.next("VSM projection dispatch");
    commands.beginDebugLabel({.name = "VSM: receiver projection"});
    if (auto r = commands.bindExecution(state.programs->compute[5]->execution(), &push, sizeof(push)); !r) { return r; }
    const auto& desc = state.outputTextures[0]->desc();
    if (auto r = commands.dispatch((desc.width + 7) / 8, (desc.height + 7) / 8); !r) { return r; }
    commands.endDebugLabel();
    projectionGpu.next("VSM projection publish");
    const TextureBarrierDesc outputEnd[]{
        {.texture = state.resources->textures[1].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}},
        {.texture = state.resources->textures[2].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}}};
    const BufferBarrierDesc statisticsReady{.buffer = state.resources->buffers[4].get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead}};
    if (auto r = commands.synchronize({.textures = state.externalOutput ? std::span<const TextureBarrierDesc>{} : std::span<const TextureBarrierDesc>{outputEnd},
        .buffers = {&statisticsReady, 1}}); !r) { return r; }
    state.projectionPending = false;
    return {};
}

Result<> VirtualShadowMaps::projectShared(CommandBuffer& commands,
    std::span<const VirtualShadowMapProjectionGroup> groups, std::string& log,
    CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapProjectionGridResources* grid)
{
    CPUProfileScope total(profiler, "VSM shared projection");
    uint32_t faceCount = 0;
    for (const auto& group : groups) {
        assert(group.faceCount == 1 || group.faceCount == 6);
        faceCount += group.faceCount;
        for (uint32_t face = 0; face < group.faceCount; ++face) {
            const auto& state = group.faces[face]->state_;
            if (!state || !state->projectionPending) {
                log += "VSM shared projection requires recorded, unresolved faces.\n";
                return makeError(Error::InvalidArgument);
            }
        }
    }
    total.work("faces", faceCount);
    total.work("lights", groups.size());
    if (faceCount <= 1) {
        total.work("pointTables", 0);
        total.work("onePass", 0);
        total.work("dispatches", faceCount);
        return faceCount == 0 ? Result<>{} : groups.front().faces[0]->project(commands, {}, log, profiler, gpuProfiler);
    }

    // Grouped projection statistics retain the original per-face kernel: the
    // combined one-pass permutation does not preserve receiver ownership.
    const auto& programs = *groups.front().faces[0]->state_->programs;
    const bool onePass = !programs.optimizeProjectionStatistics &&
        programs.device->capabilities().computeSubgroupBallotArithmetic;
    total.work("onePass", onePass);
    total.work("dispatches", onePass ? 1u : faceCount);
    uint32_t pointTables = 0;
    for (const auto& group : groups) {
        if (group.filteredPoint) {
            CPUProfileScope phase(profiler, "Prepare VSM shared point table");
            auto& owner = *group.faces[0];
            const auto poolBefore = owner.state_->pool->stats;
            if (auto r = owner.preparePointFaces(commands, {group.results.data(), group.faceCount}, log); !r) { return r; }
            ++pointTables;
            const auto& poolAfter = owner.state_->pool->stats;
            phase.work("bufferCreates", poolAfter.bufferCreates - poolBefore.bufferCreates);
            phase.work("bufferReuses", poolAfter.bufferReuses - poolBefore.bufferReuses);
            phase.work("bufferEvictions", poolAfter.bufferEvictions - poolBefore.bufferEvictions);
            for (uint32_t face = 1; face < group.faceCount; ++face) {
                group.faces[face]->state_->push.pointFaces = owner.state_->push.pointFaces;
            }
        }
        // The first face owns the table and cross-face leases. The GPU batch
        // retains/binds all states together; fallback keeps the original kernel.
        if (!onePass) {
            for (uint32_t face = 0; face < group.faceCount; ++face) {
                if (auto r = group.faces[face]->projectPrepared(commands, profiler, gpuProfiler); !r) { return r; }
            }
        }
    }
    total.work("pointTables", pointTables);
    if (onePass) { return projectOnePassPrepared(commands, groups, faceCount, log, profiler, gpuProfiler, grid); }
    return {};
}

Result<> VirtualShadowMaps::projectOnePassPrepared(CommandBuffer& commands,
    std::span<const VirtualShadowMapProjectionGroup> groups, uint32_t faceCount, std::string& log,
    CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler,
    const VirtualShadowMapProjectionGridResources* grid)
{
    auto& first = *groups.front().faces[0]->state_;
    auto& programs = *first.programs;
    auto& device = *programs.device;
    const bool lightGrid = grid && grid->grid;
    const uint32_t variant = lightGrid ? 1u : 0u;
    CPUProfileScope phase(profiler, "Prepare VSM one-pass pipeline");
    if (!programs.onePassComputes[variant]) {
        const std::array macros{SlangMacroDefine{"VSM_ONE_PASS_PROJECTION", "1"},
            SlangMacroDefine{"VSM_SHARED_PHYSICAL_POOL", programs.sharedPhysicalPool ? "1" : "0"},
            SlangMacroDefine{"VSM_LIGHT_GRID_PROJECTION", lightGrid ? "1" : "0"},
            SlangMacroDefine{"VSM_ADAPTIVE_RAY_COUNT", programs.adaptiveRayCount ? "1" : "0"},
            SlangMacroDefine{"VSM_LOCAL_SMRT_STATE_REUSE", programs.localSMRTStateReuse ? "1" : "0"},
            SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", programs.collectGPUWork ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_PROJECTION_STATISTICS", programs.groupProjectionStatistics ? "1" : "0"},
            SlangMacroDefine{"VSM_REFINED_RECEIVER", programs.refinedReceiver ? "1" : "0"}};
        const std::array capabilities{"spvMeshShadingEXT"};
        auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
            .entryPointName = "vsmProjectOnePassMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = capabilities, .macroDefines = macros}, log);
        if (!shader) { return std::unexpected(shader.error()); }
        auto module = ShaderRegistry::instance().getShaderModule(device,
            {.spirv = shader->spirv, .debugName = "vsmProjectOnePassMain"});
        if (!module) { return std::unexpected(module.error()); }
        auto pipeline = ShaderRegistry::instance().getComputePipeline(device,
            {.computeShader = {module->get()}, .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
        if (!pipeline) { return std::unexpected(pipeline.error()); }
        programs.onePassShaders[variant] = std::move(*module);
        programs.onePassComputes[variant] = std::move(*pipeline);
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
        spdlog::info("[VSMMappingShader] entry=vsmProjectOnePassMain propagatedMapping={} spirvSHA256={}",
            true, debug::debugSha256(bytes));
        spdlog::info("[VSMOnePassShader] entry=vsmProjectOnePassMain spirvSHA256={} work={} projectionStatistics={} refinedReceiver={} lightGrid={} adaptiveRayCount={}",
            debug::debugSha256(bytes), programs.collectGPUWork, programs.groupProjectionStatistics, programs.refinedReceiver, lightGrid, programs.adaptiveRayCount);
    }

    phase.next("Upload VSM projection jobs");
    std::vector<ShadowProjectionJob> jobs;
    jobs.reserve(faceCount);
    const bool gridContext = lightGrid || (grid && grid->diagnostics);
    VirtualShadowMapProjectionGrid gridData{};
    gridData.parameters = gridData.cells = gridData.lightIndices = gridData.candidates = UINT32_MAX;
    gridData.diagnostics = UINT32_MAX;
    ShadowPush push = first.push;
    const auto& outputDesc = first.outputTextures[0]->desc();
    // LightsPass records every job with the same view/depth and filter settings.
    // Effective filtering, ownership, phase and light-specific inputs remain in
    // each job; all other projection inputs keep their push-constant storage.
    for (const auto& group : groups) {
        const auto& base = *group.faces[0]->state_;
        bool selectPointFace = group.faceCount == 6 && base.pointLightFaceSelection;
        for (uint32_t face = 0; selectPointFace && face < group.faceCount; ++face) {
            const auto& state = *group.faces[face]->state_;
            selectPointFace = state.push.lightType == uint32_t(VirtualShadowMapLightType::PointFace) &&
                state.push.projectionFace == face && state.projectionOrigin == base.projectionOrigin;
        }
        if (gridContext) {
            // LightsPass validates the complete publication domain before
            // admission: two celestial slots plus at most 62 local sources.
            assert(group.lightIndex < std::size(gridData.jobs));
            setGridJobs(gridData, group.lightIndex, uint32_t(jobs.size()), group.faceCount);
        }
        for (uint32_t face = 0; face < group.faceCount; ++face) {
            const auto& state = *group.faces[face]->state_;
            const auto& p = state.push;
            assert(state.registry == first.registry && state.programs->device == &device);
            // Every recorded face receives the same immutable raster view.
            // Sharing receiver reconstruction is valid even though its per-face
            // descriptor handles differ.
            assert(std::memcmp(&state.receiverView, &first.receiverView, sizeof(ViewConstants)) == 0);
            assert(state.programs->collectGPUWork == programs.collectGPUWork &&
                state.programs->optimizeProjectionStatistics == programs.optimizeProjectionStatistics &&
                state.programs->adaptiveRayCount == programs.adaptiveRayCount &&
                state.programs->localSMRTStateReuse == programs.localSMRTStateReuse &&
                state.programs->refinedReceiver == programs.refinedReceiver);
            assert(p.depth == push.depth && p.pageSize == push.pageSize && p.virtualPagesPerAxis == push.virtualPagesPerAxis && p.pcfRadius == push.pcfRadius &&
                p.rayLength == push.rayLength && p.rayLengthScale == push.rayLengthScale && p.maxRayAngle == push.maxRayAngle &&
                p.rayCount == push.rayCount && p.samplesPerRay == push.samplesPerRay && p.receiverLOD == push.receiverLOD);
            assert(state.outputTextures[0]->desc().width == outputDesc.width &&
                state.outputTextures[0]->desc().height == outputDesc.height);
            jobs.push_back({p.pages, p.statistics, p.projections, p.atlas,
                p.filterMode, p.shadow, p.debug, p.physicalPagesX,
                p.clipmapCount, p.enabled, p.lightType, p.projectionFace,
                p.pointFaces, p.projectionFlags | (face == 0 && selectPointFace ? kVSMPointGroupFirst : 0u), p.samplePhase,
                p.lightRange, p.sourceRadius, p.angularRadius, p.lodScreenRayLength, p.receiverGeometry});
        }
    }
    auto* frame = RenderFrameContext::from(commands);
    auto table = first.pool->buffer(11, {.size = jobs.size() * sizeof(ShadowProjectionJob),
        .structureStride = sizeof(ShadowProjectionJob), .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload},
        frame && frame->recording() ? frame->completion() : GPUCompletionPoint{});
    if (!table) { return std::unexpected(table.error()); }
    first.resources->projectionJobs = std::move(*table);
    void* mapped = first.resources->projectionJobs->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, jobs.data(), jobs.size() * sizeof(ShadowProjectionJob));
    first.resources->projectionJobs->flush();
    first.resources->projectionJobs->unmap();
    auto lease = first.registry->storageBuffer(*first.resources->projectionJobs);
    if (!lease) { return std::unexpected(lease.error()); }
    first.leases.push_back(*lease);
    // These fields are unused by the original projection helpers.
    push.geometry = lease->shaderIndex();
    push.triangleCount = faceCount;
    push.arguments = UINT32_MAX;
    if (gridContext) {
        if (lightGrid) {
            if (auto r = bindLightGrid(*first.registry, first.leases, *grid->grid, gridData); !r) { return r; }
        }
        if (grid->diagnostics) {
            if (auto r = bindGridBuffer(*first.registry, first.leases, *grid->diagnostics, gridData.diagnostics); !r) { return r; }
        }
        auto context = first.pool->buffer(14, {.size = sizeof(gridData), .structureStride = sizeof(gridData),
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload},
            frame && frame->recording() ? frame->completion() : GPUCompletionPoint{});
        if (!context) { return std::unexpected(context.error()); }
        first.resources->projectionGrid = std::move(*context);
        void* destination = first.resources->projectionGrid->map();
        if (!destination) { return makeError(Error::Failure); }
        std::memcpy(destination, &gridData, sizeof(gridData));
        first.resources->projectionGrid->flush();
        first.resources->projectionGrid->unmap();
        if (auto r = bindGridBuffer(*first.registry, first.leases, *first.resources->projectionGrid, push.arguments); !r) { return r; }
    }
    commands.hostWriteBarrier();
    phase.work("lightGrid", lightGrid);
    phase.work("jobs", faceCount);
    phase.work("jobBytes", jobs.size() * sizeof(ShadowProjectionJob));

    phase.next("Bind VSM one-pass resources");
    std::vector<ResourceLease> leases;
    std::vector<TextureBarrierDesc> outputBegin, outputEnd;
    std::vector<BufferBarrierDesc> statisticsBegin, statisticsEnd;
    std::unordered_set<Texture*> outputs;
    std::unordered_set<Buffer*> statistics;
    for (const auto& group : groups) {
        for (uint32_t face = 0; face < group.faceCount; ++face) {
            const auto& stateOwner = group.faces[face]->state_;
            const auto& state = *stateOwner;
            if (auto r = commands.retainResource(stateOwner); !r) { return r; }
            leases.insert(leases.end(), state.leases.begin(), state.leases.end());
            for (auto* texture : state.outputTextures) {
                if (!texture || !outputs.insert(texture).second) { continue; }
                // Graph-owned output arrays are already wholly General. One
                // dependency covers all face views, including compact aliases.
                const TextureSubresourceRange range{0u, texture->desc().mipCount, 0u, texture->desc().layerCount};
                outputBegin.push_back({.texture = texture,
                    .oldLayout = state.externalOutput ? TextureLayout::General : TextureLayout::Undefined,
                    .newLayout = TextureLayout::General,
                    .before = state.externalOutput ? SyncScope{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite} : SyncScope{},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .range = range});
                if (!state.externalOutput) {
                    outputEnd.push_back({.texture = texture, .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
                        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}, .range = range});
                }
            }
            auto* buffer = state.resources->buffers[4].get();
            if (statistics.insert(buffer).second) {
                statisticsBegin.push_back({.buffer = buffer,
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
                statisticsEnd.push_back({.buffer = buffer,
                    .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead}});
            }
        }
    }
    if (auto r = first.registry->bind(commands, leases); !r) { return r; }
    phase.next("Dispatch VSM one-pass projection");
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM one-pass projection") : RenderGraphExecutionContext::ProfileScope{};
    auto projectionGpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM projection prepare") : RenderGraphExecutionContext::ProfileScope{};
    if (auto r = commands.synchronize({.textures = outputBegin, .buffers = statisticsBegin}); !r) { return r; }
    projectionGpu.next("VSM projection dispatch");
    commands.beginDebugLabel({.name = "VSM: one-pass receiver projection"});
    if (auto r = commands.bindExecution(programs.onePassComputes[variant]->execution(), &push, sizeof(push)); !r) { return r; }
    if (auto r = commands.dispatch((outputDesc.width + 7u) / 8u, (outputDesc.height + 7u) / 8u); !r) { return r; }
    commands.endDebugLabel();
    projectionGpu.next("VSM projection publish");
    if (auto r = commands.synchronize({.textures = outputEnd, .buffers = statisticsEnd}); !r) { return r; }
    for (const auto& group : groups) {
        for (uint32_t face = 0; face < group.faceCount; ++face) { group.faces[face]->state_->projectionPending = false; }
    }
    return {};
}

VirtualShadowMapMemoryUsage VirtualShadowMaps::memoryUsage() const
{
    VirtualShadowMapMemoryUsage usage;
    std::unordered_set<const Texture*> textures;
    std::unordered_set<const Buffer*> buffers;
    const auto addAtlas = [&](const Texture* texture, uint32_t pageSize, bool staticLayer = false) {
        if (texture && textures.insert(texture).second) {
            const auto& desc = texture->desc();
            usage.atlasBytes += uint64_t(desc.width) * desc.height * 4;
            if (staticLayer) { usage.staticAtlasBytes += uint64_t(desc.width) * desc.height * 4; }
            usage.allocatedAtlasPages += uint64_t(desc.width / pageSize) * (desc.height / pageSize);
        }
    };
    const auto addBuffer = [&](const Buffer* buffer) {
        if (buffer && buffers.insert(buffer).second) { usage.bufferBytes += buffer->desc().size; }
    };
    const auto addCache = [&](const std::shared_ptr<Cache>& cache) {
        if (cache) {
            if (cache->poolSerial == 0) {
                addAtlas(cache->atlas->texture.get(), cache->settings.pageSize);
                if (cache->staticAtlas) { addAtlas(cache->staticAtlas->texture.get(), cache->settings.pageSize, true); }
                addBuffer(cache->staticHZB.get());
            }
            addBuffer(cache->pages.get());
            addBuffer(cache->lodSelection.get());
        }
    };
    addCache(cache_);
    addCache(pending_);
    if (state_) {
        if (!state_->resources->physicalPoolStorage) {
            addAtlas(state_->resources->textures[0].get(), state_->push.pageSize);
            if (state_->resources->staticAtlas) { addAtlas(state_->resources->staticAtlas->texture.get(), state_->push.pageSize, true); }
            addBuffer(state_->resources->staticHZB.get());
            addBuffer(state_->resources->hybridDepth.get());
            addBuffer(state_->resources->rasterHZB.get());
        }
        addBuffer(state_->resources->cacheDiagnostics.get());
        addBuffer(state_->resources->hierarchyHZBQueue.get());
        addBuffer(state_->resources->lodContext.get());
        addBuffer(state_->resources->lodGroupStates.get());
        addBuffer(state_->resources->lodSelection.get());
        addBuffer(state_->resources->lodChanges.get());
        addBuffer(state_->resources->lodStatistics.get());
        addBuffer(state_->resources->staticHZBJobs.get());
        addBuffer(state_->resources->clusterVisibility.get());
        usage.bufferBytes += state_->resources->layerBufferBytes;
        for (size_t index = 1; index < state_->resources->textures.size(); ++index) {
            if (const auto& texture = state_->resources->textures[index]; texture && textures.insert(texture.get()).second) {
                const auto& desc = texture->desc();
                usage.privateOutputBytes += uint64_t(desc.width) * desc.height * (index == 1 ? 1 : 4);
            }
        }
        for (const auto& buffer : state_->resources->buffers) { addBuffer(buffer.get()); }
        addBuffer(state_->resources->pointFaces.get());
        addBuffer(state_->resources->projectionJobs.get());
        addBuffer(state_->resources->projectionGrid.get());
        addBuffer(state_->resources->receiverGeometry.get());
        addCache(state_->previous);
        addCache(state_->next);
    }
    if (pool_) {
        for (const auto& slot : pool_->buffers) {
            for (const auto& entry : slot) { addBuffer(entry.buffer.get()); }
        }
        for (size_t slot = 0; slot < pool_->textures.size(); ++slot) {
            for (const auto& entry : pool_->textures[slot]) {
                const auto* texture = entry.resource.texture.get();
                if (slot == 0 || slot == 3) { addAtlas(texture, entry.pageSize, slot == 3); }
                else if (textures.insert(texture).second) {
                    const auto& desc = texture->desc();
                    usage.privateOutputBytes += uint64_t(desc.width) * desc.height * (slot == 1 ? 1 : 4);
                }
            }
        }
    }
    return usage;
}

VirtualShadowMapResourcePoolStats VirtualShadowMaps::resourcePoolStats() const
{
    if (!pool_) { return {}; }
    auto stats = pool_->stats;
    for (const auto& slot : pool_->buffers) {
        for (const auto& entry : slot) {
            ++stats.retainedBuffers;
            stats.retainedBufferBytes += entry.buffer->desc().size;
        }
    }
    for (const auto& slot : pool_->textures) {
        for (const auto& entry : slot) {
            ++stats.retainedTextures;
            const auto& desc = entry.resource.texture->desc();
            stats.retainedTextureBytes += uint64_t(desc.width) * desc.height * (desc.format == Format::R8Unorm ? 1 : 4);
        }
    }
    return stats;
}

} // namespace metallic::render
