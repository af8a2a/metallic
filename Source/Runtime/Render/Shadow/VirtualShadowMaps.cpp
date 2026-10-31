#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_set>
#include <vector>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kDrawBatchSize = 65535;

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
    uint32_t rayCount, samplesPerRay, receiverFootprint, filterReserved;
    float rayLengthScale, maxRayAngle;
    uint32_t samplePhase, sampleReserved;
    float lodScreenRayLength;
};
static_assert(sizeof(ShadowPush) == 212);

struct ShadowProjectionJob
{
    uint32_t pages, statistics, projections, atlas;
    uint32_t filterMode, shadow, debug, physicalPagesX;
    uint32_t clipmapCount, enabled, lightType, projectionFace;
    uint32_t pointFaces, filterReserved, samplePhase, view;
    float lightRange, sourceRadius, angularRadius, lodScreenRayLength;
};
static_assert(sizeof(ShadowProjectionJob) == 80);

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
        a.firstClipmapExtent == b.firstClipmapExtent && a.shadowBias == b.shadowBias && a.enabled == b.enabled;
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

} // namespace

struct VirtualShadowMaps::Programs
{
    Device* device = nullptr;
    uint64_t coverageKey = 0;
    bool collectGPUWork = false;
    bool optimizeMark = true;
    bool optimizeProjectionStatistics = false;
    bool optimizeMarkSamples = false;
    bool directionalMark = false;
    bool clusterCulling = true;
    bool clusterPageBitset = false;
    bool trianglePageBitset = false;
    bool directionalBuild = false;
    bool compactBuild = false;
    std::array<std::unique_ptr<ShaderModule>, 15> shaders;
    std::array<std::unique_ptr<ComputePipeline>, 11> compute;
    std::unique_ptr<ShaderModule> onePassShader;
    std::unique_ptr<ComputePipeline> onePassCompute;
    std::unique_ptr<GraphicsPipeline> raster;
    std::unique_ptr<GraphicsPipeline> clear;
};

struct VirtualShadowMaps::Cache
{
    Device* device = nullptr;
    std::shared_ptr<ShadowAtlas> atlas;
    std::shared_ptr<ShadowAtlas> staticAtlas;
    std::vector<uint64_t> dynamicCasters;
    std::shared_ptr<Buffer> pages;
    std::vector<ShadowProjection> projections;
    std::vector<std::array<int64_t, 2>> origins;
    std::vector<VirtualShadowMapCasterState> casters;
    VirtualShadowMapSettings settings;
    VirtualShadowMapLight light;
    float farPlane = 0;
    uint64_t sceneIdentity = 0, structuralRevision = 0, dependencyKey = 0;
    uint64_t geometryRevision = 0, texturePublicationRevision = 0, tessellationKey = 0;
    uint32_t epoch = 0;
    bool enabled = false;
    GPUCompletionPoint completion;
    std::shared_ptr<SubmissionTransaction> submission;
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
    std::array<std::vector<BufferEntry>, 13> buffers;
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
        std::shared_ptr<Buffer> cacheDiagnostics;
        std::shared_ptr<ShadowAtlas> staticAtlas;
        std::array<std::shared_ptr<void>, 2> casterLayers;
        uint64_t layerBufferBytes = 0;
        std::array<std::shared_ptr<Texture>, 3> textures; // atlas, shadow, debug
        std::array<std::shared_ptr<TextureView>, 3> views;
    };
    std::shared_ptr<Resources> resources = std::make_shared<Resources>();
    std::shared_ptr<ResourcePool> pool;
    ShadowPush push{};
    std::shared_ptr<ResourceRegistry> registry;
    std::shared_ptr<Programs> programs;
    std::vector<ResourceLease> leases;
    std::array<Texture*, 2> outputTextures{};
    bool externalOutput = false;
    std::shared_ptr<Cache> previous;
    std::shared_ptr<Cache> next;
    bool projectionPending = true;
};

VirtualShadowMaps::VirtualShadowMaps() = default;

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
    // Only X/Y/W are consumed by LOD; local far depth does not affect them.
    auto projections = directional ? makeProjections(bounds, view, light.direction, settings) :
        makeLocalProjections(light, light.nearPlane * 2, settings);
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

Result<VirtualShadowMapResult> VirtualShadowMaps::record(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection,
    const VirtualShadowMapProjectionTarget* output, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler)
{
    CPUProfileScope total(profiler, "VSM sun record");
    VirtualShadowMapLight light;
    light.enabled = (sun.flags & (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow)) ==
        (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow);
    if (light.enabled) { light.direction = {sun.direction[0], sun.direction[1], sun.direction[2]}; }
    light.angularRadius = sun.angularRadius;
    return record(device, commands, depth, view, light, casters, materials, alphaTextures, settings, log, deferProjection, output, profiler, gpuProfiler);
}

Result<VirtualShadowMapResult> VirtualShadowMaps::record(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& inputLight,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection,
    const VirtualShadowMapProjectionTarget* output, CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler)
{
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
    if (!validProjectionSettings(settings) || settings.physicalPageCount == 0 || settings.physicalPageCount > 4096 ||
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
    const bool directional = light.type == VirtualShadowMapLightType::Directional;
    const bool enabled = settings.enabled && light.enabled;
    const bool clusterPageBitset = settings.clusterCulling && settings.clusterPageBitset;
    const bool trianglePageBitset = clusterPageBitset && settings.trianglePageBitset;
    const bool directionalBuild = directional && settings.directionalBuild;
    const bool directionalMark = directional && settings.directionalMark;
    const bool compactBuild = settings.clusterCulling && settings.compactBuild;
    // The local-light permutation has confirmed gains; directional hoisting regressed Mark.
    const bool optimizeMarkSamples = settings.optimizeMarkSamples && !directional;
    if (auto valid = validateProjectionLight(light, log); !valid) { return std::unexpected(valid.error()); }
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
    uint32_t cancelledCount = 0;
    if (pending_) {
        if (pending_->submission->cancelled() || pending_->completion.isCancelled()) {
            pending_.reset();
            cancelledCount = 1;
        } else if (pending_->submission->resolved() && pending_->completion.isSubmitted()) {
            cache_ = std::move(pending_);
        } else {
            log += "Previous VSM recording must be submitted or cancelled before recording again.\n";
            return makeError(Error::InvalidArgument);
        }
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
    state->pool = pool_;
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    state->registry = *registry;
    auto& push = state->push;
    state->externalOutput = output != nullptr;
    push.filterReserved = output && output->sharedPointLayer ? 1u : 0u;
    push.triangleCount = casters.triangleCapacity;
    push.clipmapCount = settings.clipmapCount;
    push.virtualPagesPerAxis = settings.virtualPagesPerAxis;
    push.physicalPagesX = static_cast<uint32_t>(std::ceil(std::sqrt(float(settings.physicalPageCount))));
    push.physicalPagesY = (settings.physicalPageCount + push.physicalPagesX - 1) / push.physicalPagesX;
    push.physicalPageCount = settings.physicalPageCount;
    push.pageSize = settings.pageSize;
    push.workCapacity = settings.maxWorkItems;
    push.batchCount = (settings.maxWorkItems + kDrawBatchSize - 1) / kDrawBatchSize;
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
    push.receiverFootprint = settings.receiverFootprint;
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
    const auto origins = directional ? pageOrigins(view, direction, settings) :
        std::vector<std::array<int64_t, 2>>(settings.clipmapCount, {0, 0});
    const bool layered = settings.cacheEnabled && settings.cacheLayering;
    std::array<uint32_t, 48> cacheDiagnostics{};
    auto& reasons = cacheDiagnostics[kVSMCacheReasons];
    if (!settings.cacheEnabled) { reasons |= kVSMCacheReasonCacheDisabled; }
    if (!cache_) { reasons |= kVSMCacheReasonCold; }
    if (settings.forceRedraw) { reasons |= kVSMCacheReasonForceRedraw; }
    if (cache_) {
        if (cache_->device != &device) { reasons |= kVSMCacheReasonDevice; }
        if (!compatibleLight(cache_->light, light) || cache_->enabled != enabled) { reasons |= kVSMCacheReasonLight; }
        if (!compatibleSettings(cache_->settings, settings)) { reasons |= kVSMCacheReasonSettings; }
        if (cache_->sceneIdentity != casters.sceneIdentity) { reasons |= kVSMCacheReasonScene; }
        if (cache_->structuralRevision != casters.structuralRevision) { reasons |= kVSMCacheReasonStructure; }
        if (cache_->geometryRevision != casters.geometryRevision) { reasons |= kVSMCacheReasonGeometryPublication; }
        if (cache_->texturePublicationRevision != casters.texturePublicationRevision) { reasons |= kVSMCacheReasonTexturePublication; }
        if (cache_->tessellationKey != casters.tessellationKey) { reasons |= kVSMCacheReasonTessellation; }
        if (cache_->settings.cacheLayering != settings.cacheLayering) { reasons |= kVSMCacheReasonLayerMode; }
    }
    bool reuse = cache_ && cache_->settings.cacheLayering == settings.cacheLayering && cache_->device == &device && compatibleLight(cache_->light, light) && cache_->enabled == enabled &&
        compatibleSettings(cache_->settings, settings) && cache_->sceneIdentity == casters.sceneIdentity &&
        cache_->structuralRevision == casters.structuralRevision && cache_->dependencyKey == casters.dependencyKey;
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
    std::vector<scene::Bounds> dirtyBounds, staticDirtyBounds;
    std::vector<uint64_t> dynamicCasters = reuse && layered ? cache_->dynamicCasters : std::vector<uint64_t>{};
    const auto isDynamic = [&](uint64_t id) { return std::binary_search(dynamicCasters.begin(), dynamicCasters.end(), id); };
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
        size_t oldIndex = 0, newIndex = 0;
        while (oldIndex < cache_->casters.size() || newIndex < casters.states.size()) {
            const auto* old = oldIndex < cache_->casters.size() ? &cache_->casters[oldIndex] : nullptr;
            const auto* next = newIndex < casters.states.size() ? &casters.states[newIndex] : nullptr;
            if (old && (!next || old->id < next->id)) {
                dirtyBounds.push_back(old->bounds);
                ++cacheDiagnostics[kVSMCacheRemovedCasters];
                if (!isDynamic(old->id)) { staticDirtyBounds.push_back(old->bounds); }
                ++oldIndex;
            } else if (next && (!old || next->id < old->id)) {
                dirtyBounds.push_back(next->bounds);
                ++cacheDiagnostics[kVSMCacheAddedCasters];
                if (!isDynamic(next->id)) { staticDirtyBounds.push_back(next->bounds); }
                ++newIndex;
            } else {
                const bool lodChanged = old->lodKey != next->lodKey;
                cacheDiagnostics[kVSMCacheLODChangedCasters] += lodChanged;
                if (old->contentKey != next->contentKey) {
                    dirtyBounds.push_back(old->bounds); dirtyBounds.push_back(next->bounds);
                    ++cacheDiagnostics[kVSMCacheChangedCasters];
                    cacheDiagnostics[kVSMCacheTransformChanges] += old->transformKey != next->transformKey;
                    cacheDiagnostics[kVSMCacheMaterialChanges] += old->materialKey != next->materialKey;
                    if (layered && !isDynamic(old->id)) {
                        staticDirtyBounds.push_back(old->bounds);
                        dynamicCasters.insert(std::lower_bound(dynamicCasters.begin(), dynamicCasters.end(), old->id), old->id);
                        ++cacheDiagnostics[kVSMCachePromotedCasters];
                    }
                } else if (lodChanged) {
                    dirtyBounds.push_back(old->bounds); dirtyBounds.push_back(next->bounds);
                    if (!isDynamic(old->id)) {
                        staticDirtyBounds.push_back(old->bounds); staticDirtyBounds.push_back(next->bounds);
                    }
                }
                ++oldIndex; ++newIndex;
            }
        }
        state->previous = cache_;
        if (auto r = addCommandDependency(commands, cache_->completion); !r) { return std::unexpected(r.error()); }
        auto lease = state->registry->storageBuffer(*cache_->pages);
        if (!lease) { return std::unexpected(lease.error()); }
        push.previousPages = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    push.cacheMode = (reuse ? 1u : 8u) | (settings.cacheEnabled ? 2u : 0u) | (settings.forceRedraw ? 4u : 0u) | (layered ? 16u : 0u);
    push.cacheEpoch = reuse ? cache_->epoch : (cache_ ? cache_->epoch + 1 : 1);
    push.dirtyBoundsCount = static_cast<uint32_t>(dirtyBounds.size());
    total.work("reuse", reuse);
    phase.work("dirtyBounds", dirtyBounds.size());
    phase.next("Build VSM page remapping");
    // Each remap table begins with one uint4 per level followed by inclusive rectangles.
    // Static and composite tables share snapped offsets and the physical page mapping.
    std::vector<std::array<uint32_t, 4>> pageChanges;
    const auto appendRemap = [&](std::span<const scene::Bounds> changedBounds) {
        const uint32_t base = uint32_t(pageChanges.size());
        pageChanges.resize(pageChanges.size() + settings.clipmapCount);
        if (reuse) {
            for (uint32_t level = 0; level < settings.clipmapCount; ++level) {
                const int64_t axis = projections[level].page[1];
                const auto dx = origins[level][0] - cache_->origins[level][0];
                const auto dy = origins[level][1] - cache_->origins[level][1];
                const auto offsetX = int32_t(std::clamp(dx, -axis, axis));
                const auto offsetY = int32_t(std::clamp(dy, -axis, axis));
                const auto firstRectangle = uint32_t(pageChanges.size());
                pageChanges[base + level] = {uint32_t(offsetX), uint32_t(offsetY), firstRectangle, 0};
                if (std::abs(offsetX) == axis || std::abs(offsetY) == axis || settings.forceRedraw) { continue; }
                for (const auto& bounds : changedBounds) {
                    if (!bounds.valid) { continue; }
                    const auto [low, high] = projectedBounds(bounds, projections[level]);
                    if (high[0] < -1 || low[0] > 1 || high[1] < -1 || low[1] > 1) { continue; }
                    const auto page = [axis](float v, float epsilon) {
                        return uint32_t(std::clamp(int(std::floor((std::clamp(v, -1.f, 1.f) * .5f + .5f) * axis + epsilon)), 0, int(axis - 1)));
                    };
                    pageChanges.push_back({page(low[0], -.0001f), page(low[1], -.0001f),
                        page(high[0], .0001f), page(high[1], .0001f)});
                }
                pageChanges[base + level][3] = uint32_t(pageChanges.size()) - firstRectangle;
            }
        }
        return base;
    };
    appendRemap(dirtyBounds);
    cacheDiagnostics[kVSMCacheDirtyRectangles] = uint32_t(pageChanges.size()) - settings.clipmapCount;
    const uint32_t staticRemapBase = layered ? appendRemap(staticDirtyBounds) : 0u;
    cacheDiagnostics[kVSMCacheStaticDirtyRectangles] = layered ? uint32_t(pageChanges.size()) - staticRemapBase - settings.clipmapCount : 0u;
    cacheDiagnostics[kVSMCacheLayered] = layered;
    cacheDiagnostics[kVSMCacheReused] = reuse;
    cacheDiagnostics[kVSMCacheDirtyBounds] = uint32_t(dirtyBounds.size());
    cacheDiagnostics[kVSMCacheStaticDirtyBounds] = layered ? uint32_t(staticDirtyBounds.size()) : 0u;
    cacheDiagnostics[kVSMCacheCancelledRecordings] = cancelledCount;
    cacheDiagnostics[kVSMCacheRasterWorkAvailable] = layered && settings.collectGPUWork;
    for (const auto& caster : casters.states) {
        ++cacheDiagnostics[layered && isDynamic(caster.id) ? kVSMCacheDynamicCasters : kVSMCacheStaticCasters];
    }
    cacheDiagnostics[35] = uint32_t(staticDirtyBounds.size());
    cacheDiagnostics[36] = staticRemapBase;
    phase.work("remapBytes", pageChanges.size() * sizeof(pageChanges[0]));
    phase.work("denseRemapBytes", uint64_t(push.totalPages) * 8);
    phase.work("remapLevels", settings.clipmapCount);
    phase.work("dirtyRectangles", cacheDiagnostics[kVSMCacheDirtyRectangles]);
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
    const uint32_t clusterPageBitsetWords = clusterPageBitset ? (push.totalPages + 31u) / 32u : 0u;
    const uint32_t clusterCapacity = push.triangleCount / 64u + (push.triangleCount % 64u != 0u ? 1u : 0u);
    const uint32_t compactBatches = compactBuild ? (clusterCapacity + kDrawBatchSize - 1) / kDrawBatchSize : 0u;
    const uint64_t compactWords = compactBuild ? 1u + uint64_t(clusterCapacity) * 2u : 0u;
    const uint64_t compactArgumentOffset = uint64_t(push.batchCount + push.coarseBatchCount) * 12u;
    const uint64_t sizes[]{gpuMaterials.size() * sizeof(ShadowMaterial), uint64_t(push.totalPages) * 16,
        coverage->inputBytes().size(), uint64_t(push.workCapacity) * 8, 128 + (settings.collectGPUWork ?
            kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize : 0u),
        projections.size() * sizeof(ShadowProjection), sizeof(ViewConstants), compactArgumentOffset + uint64_t(compactBatches) * 12u,
        pageChanges.size() * sizeof(pageChanges[0]), (allocationWords + clusterPageBitsetWords + compactWords) * sizeof(uint32_t)};
    const uint32_t strides[]{64, 16, 4, 8, 4, 112, 208, 4, 16, 4};
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
        if (data[index]) { hostUploadBytes += sizes[index]; ++hostUploadBuffers; }
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
            std::memcpy(mapped, data[index], sizes[index]);
            state->resources->buffers[index]->flush(); state->resources->buffers[index]->unmap();
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
        if (reuse) {
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
        for (uint32_t layer = 0; layer < 2; ++layer) {
            auto geometryLayer = prepareVirtualShadowMapCasterLayer(device, commands, casters, dynamicCasters, layer == 0, log);
            if (!geometryLayer) { return std::unexpected(geometryLayer.error()); }
            auto lease = state->registry->storageBuffer(*geometryLayer->parameters);
            if (!lease) { return std::unexpected(lease.error()); }
            cacheDiagnostics[33 + layer] = lease->shaderIndex(); state->leases.push_back(*lease);
            state->leases.insert(state->leases.end(), geometryLayer->leases.begin(), geometryLayer->leases.end());
            state->resources->casterLayers[layer] = geometryLayer->owner;
            state->resources->layerBufferBytes += geometryLayer->layerBufferBytes;
        }
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
    auto sampler = state->registry->sampler({});
    if (!sampler) { return std::unexpected(sampler.error()); }
    push.sampler = sampler->shaderIndex(); state->leases.push_back(*sampler);
    phase.work("textureCreates", pool_->stats.textureCreates - poolBefore.textureCreates);
    phase.work("textureReuses", pool_->stats.textureReuses - poolBefore.textureReuses);
    phase.work("textureEvictions", pool_->stats.textureEvictions - poolBefore.textureEvictions);
    phase.work("viewCreates", pool_->stats.viewCreates - poolBefore.viewCreates);
    phase.next("Prepare VSM pipeline cache");
    const bool programMiss = !programs_ || programs_->device != &device || programs_->coverageKey != coverage->coverageKey() ||
        programs_->collectGPUWork != settings.collectGPUWork ||
        programs_->optimizeProjectionStatistics != settings.optimizeProjectionStatistics ||
        programs_->optimizeMark != settings.optimizeMark || programs_->optimizeMarkSamples != optimizeMarkSamples ||
        programs_->directionalMark != directionalMark ||
        programs_->clusterCulling != settings.clusterCulling ||
        programs_->clusterPageBitset != clusterPageBitset || programs_->trianglePageBitset != trianglePageBitset ||
        programs_->directionalBuild != directionalBuild || programs_->compactBuild != compactBuild;
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
        programs->optimizeProjectionStatistics = settings.optimizeProjectionStatistics;
        programs->optimizeMark = settings.optimizeMark;
        programs->optimizeMarkSamples = optimizeMarkSamples;
        programs->directionalMark = directionalMark;
        programs->clusterCulling = settings.clusterCulling;
        programs->clusterPageBitset = clusterPageBitset;
        programs->trianglePageBitset = trianglePageBitset;
        programs->directionalBuild = directionalBuild;
        programs->compactBuild = compactBuild;
        const char* entries[]{"vsmResetMain", "vsmMarkMain", "vsmAllocateMain", "vsmBuildMain", "vsmFinalizeMain", "vsmProjectMain",
            "vsmRasterMeshMain", "vsmDepthFragmentMain", "vsmClearMeshMain", "vsmClearFragmentMain",
            "vsmCountMain", "vsmAdmitMain", "vsmPhysicalMain", "vsmAssignMain", "vsmCompactMain"};
        const auto generated = generatedDirectory.string();
        const std::array paths{generated.c_str()};
        const std::array capabilities{"spvMeshShadingEXT"};
        const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", coverage->coverageProgramCount() ? "1" : "0"},
            SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", settings.collectGPUWork ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_PROJECTION_STATISTICS", settings.optimizeProjectionStatistics ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_MARK", settings.optimizeMark ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_MARK_SAMPLES", optimizeMarkSamples ? "1" : "0"},
            SlangMacroDefine{"VSM_CLUSTER_CULL", settings.clusterCulling ? "1" : "0"},
            SlangMacroDefine{"VSM_CLUSTER_PAGE_BITSET", clusterPageBitset ? "1" : "0"},
            SlangMacroDefine{"VSM_TRIANGLE_PAGE_BITSET", trianglePageBitset ? "1" : "0"}};
        for (size_t index = 0; index < programs->shaders.size(); ++index) {
            if (index == 14 && !compactBuild) { continue; }
            const bool buildEntry = index == 3 || index == 14;
            const bool compactEntry = index == 0 || buildEntry;
            std::vector<SlangMacroDefine> entryMacros(macros.begin(), macros.end());
            if (index == 1) { entryMacros.push_back({"VSM_DIRECTIONAL_MARK", directionalMark ? "1" : "0"}); }
            if (buildEntry) { entryMacros.push_back({"VSM_DIRECTIONAL_BUILD", directionalBuild ? "1" : "0"}); }
            if (compactEntry) { entryMacros.push_back({"VSM_COMPACT_BUILD", compactBuild ? "1" : "0"}); }
            auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                .capabilities = capabilities, .macroDefines = entryMacros}, log);
            if (!shader) { return std::unexpected(shader.error()); }
            auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entries[index]});
            if (!module) { return std::unexpected(module.error()); }
            programs->shaders[index] = std::move(*module);
            if (index == 1 || index == 3 || index == 5 || index == 14) {
                const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                spdlog::info("[VSMShader] entry={} spirvSHA256={} optimizeMark={} clusterCull={} clusterPageBitset={} work={} markSamples={} lightType={} trianglePageBitset={} projectionStatistics={} directionalBuild={} compactBuild={} directionalMark={}", entries[index],
                    debug::debugSha256(bytes), settings.optimizeMark,
                    settings.clusterCulling, clusterPageBitset, settings.collectGPUWork, optimizeMarkSamples, uint32_t(light.type), trianglePageBitset, settings.optimizeProjectionStatistics, buildEntry && directionalBuild, compactEntry && compactBuild, index == 1 && directionalMark);
            }
            if (index < 6 || index >= 10) {
                auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {programs->shaders[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
                if (!pipeline) { return std::unexpected(pipeline.error()); }
                programs->compute[index < 6 ? index : index - 4] = std::move(*pipeline);
            }
        }
        auto raster = ShaderRegistry::instance().getGraphicsPipeline(device, {.meshShader = {programs->shaders[6].get()},
            .fragmentShader = {programs->shaders[7].get()}, .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::GreaterEqual}, .usesBindlessHeap = true});
        if (!raster) { return std::unexpected(raster.error()); }
        programs->raster = std::move(*raster);
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
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM reset") : RenderGraphExecutionContext::ProfileScope{};
    commands.hostWriteBarrier();
    std::array<BufferBarrierDesc, 5> recycledReady;
    constexpr std::array<size_t, 5> deviceSlots{1, 3, 4, 7, 9};
    for (size_t index = 0; index < deviceSlots.size(); ++index) {
        recycledReady[index] = {.buffer = state->resources->buffers[deviceSlots[index]].get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    }
    if (auto r = commands.synchronize({.buffers = recycledReady}); !r) { return std::unexpected(r.error()); }
    if (reuse) {
        const BufferBarrierDesc previousReady{.buffer = cache_->pages.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite}, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
        if (auto r = commands.synchronize({.buffers = {&previousReady, 1}}); !r) { return std::unexpected(r.error()); }
    }
    if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM: demand and cache allocation"}); }
    // Reset writes diagnostics at statistics[32 + id.x], independently of the public 32 words.
    const uint32_t statisticsResetThreads = settings.collectGPUWork ?
        (kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / sizeof(uint32_t) : 32u;
    if (auto r = dispatch(0, (std::max({push.totalPages, push.physicalPageCount, statisticsResetThreads,
        (push.batchCount + push.coarseBatchCount + compactBatches) * 3}) + 63) / 64); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1, 4, 7, 9}); !r) { return std::unexpected(r.error()); }
    gpu.next("VSM mark");
    if (auto r = dispatch(1, (width + 7) / 8, (height + 7) / 8); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier(settings.collectGPUWork ? std::initializer_list<size_t>{1, 4} : std::initializer_list<size_t>{1});
        !r) { return std::unexpected(r.error()); }
    gpu.next("VSM allocator");
    if (auto r = dispatch(6, allocationBlocks); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({4, 9}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(2, 1); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({4, 9}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(7, allocationBlocks); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1, 4, 9}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(8, 1); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({4, 9}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(9, allocationBlocks); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1, 9}); !r) { return std::unexpected(r.error()); }
    gpu.next("VSM build");
    auto buildGpu = gpuProfiler && settings.profileBuildSubstages ?
        gpuProfiler->profileScope(commands, compactBuild ? "VSM build compact" : "VSM build direct") :
        RenderGraphExecutionContext::ProfileScope{};
    const bool buildTriangles = push.triangleCount && enabled;
    if (buildTriangles) {
        // Compute dispatch limit is 65535 workgroups; split immutable triangle ranges.
        for (uint32_t base = 0; base < push.triangleCount;) {
            push.workBase = base;
            const uint32_t count = std::min(push.triangleCount - base, kDrawBatchSize * 64);
            if (auto r = dispatch(compactBuild ? 10 : 3, (count + 63) / 64); !r) { return std::unexpected(r.error()); }
            base += count;
            if (base < push.triangleCount) {
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
                if (batch + 1 < compactBatches) {
                    if (auto r = barrier({3, 4}); !r) { return std::unexpected(r.error()); }
                }
            }
            // Finalize writes the original draw-argument prefix only. The
            // indirect reads above address the disjoint appended dispatch tail.
        }
    }
    buildGpu.next("VSM build finalize sync");
    if (auto r = barrier({4}); !r) { return std::unexpected(r.error()); }
    buildGpu.next("VSM build finalize");
    if (auto r = dispatch(4, (std::max(push.totalPages, push.batchCount + push.coarseBatchCount) + 63) / 64); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({3, 1, 9}, PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader, AccessBits::ShaderRead); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({7}, PipelineStageBits::DrawIndirect, AccessBits::IndirectRead); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({4}, settings.collectGPUWork ? PipelineStageBits::PreRasterization : PipelineStageBits::ComputeShader);
        !r) { return std::unexpected(r.error()); }
    buildGpu.end();
    if (!gpuProfiler) { commands.endDebugLabel(); }
    gpu.next("VSM raster");
    phase.next("Record VSM raster commands");
    phase.work("coarseBatches", push.coarseBatchCount);
    phase.work("fineBatches", push.batchCount);
    const uint32_t fullGeometry = push.geometry;
    const BufferBarrierDesc diagnosticReady{.buffer = state->resources->cacheDiagnostics.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    if (auto r = commands.synchronize({.buffers = {&diagnosticReady, 1}}); !r) { return std::unexpected(r.error()); }
    for (uint32_t layer = 0; layer < (layered ? 2u : 1u); ++layer) {
        const bool staticPass = layered && layer == 0;
        auto layerGpu = gpuProfiler && layered ? gpuProfiler->profileScope(commands,
            staticPass ? "VSM static raster" : "VSM dynamic raster and restore") : RenderGraphExecutionContext::ProfileScope{};
        auto* rasterTexture = staticPass ? state->resources->staticAtlas->texture.get() : state->resources->textures[0].get();
        auto* rasterView = staticPass ? state->resources->staticAtlas->view.get() : state->resources->views[0].get();
        push.geometry = layered ? cacheDiagnostics[33 + layer] : fullGeometry;
        const TextureBarrierDesc depthBegin{.texture = rasterTexture, .oldLayout = reuse ? TextureLayout::ShaderRead : TextureLayout::Undefined,
            .newLayout = TextureLayout::DepthStencilAttachment,
            .before = {PipelineStageBits::AllCommands, reuse ? AccessBits::MemoryRead | AccessBits::MemoryWrite : AccessBits::None},
            .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite}};
        if (auto r = commands.synchronize({.textures = {&depthBegin, 1}}); !r) { return std::unexpected(r.error()); }
        const Rect area{.width = push.physicalPagesX * push.pageSize, .height = push.physicalPagesY * push.pageSize};
        const RenderingAttachmentDesc attachment{.view = rasterView, .layout = TextureLayout::DepthStencilAttachment,
            .loadOp = reuse ? LoadOp::Load : LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = 0};
        if (!gpuProfiler) { commands.beginDebugLabel({.name = "VSM: dirty coarse and fine depth"}); }
        if (auto r = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &attachment}); !r) { return std::unexpected(r.error()); }
        if (auto r = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1}); !r) { return std::unexpected(r.error()); }
        commands.setScissor(area);
        if (enabled) {
            if (reuse || (layered && !staticPass)) {
                push.rasterCoarse = staticPass ? 2u : 0u;
                if (auto r = commands.bindExecution(state->programs->clear->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                if (auto r = commands.drawMeshTasks(push.physicalPageCount); !r) { return std::unexpected(r.error()); }
            }
            push.rasterCoarse = staticPass ? 3u : 1u;
            for (uint32_t batch = 0; batch < push.coarseBatchCount; ++batch) {
                push.workBase = batch * kDrawBatchSize;
                if (auto r = commands.bindExecution(state->programs->raster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                auto args = state->resources->buffers[7]->slice({uint64_t(push.batchCount + batch) * 12, 12});
                if (!args) { return std::unexpected(args.error()); }
                if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
            }
            push.rasterCoarse = staticPass ? 2u : 0u;
            for (uint32_t batch = 0; batch < push.batchCount; ++batch) {
                push.workBase = batch * kDrawBatchSize;
                if (auto r = commands.bindExecution(state->programs->raster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
                auto args = state->resources->buffers[7]->slice({uint64_t(batch) * 12, 12});
                if (!args) { return std::unexpected(args.error()); }
                if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
            }
        }
        commands.endRendering();
        if (!gpuProfiler) { commands.endDebugLabel(); }
        const TextureBarrierDesc atlasReady{.texture = rasterTexture,
            .oldLayout = TextureLayout::DepthStencilAttachment, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
            .after = {PipelineStageBits::ComputeShader | PipelineStageBits::FragmentShader, AccessBits::ShaderRead}};
        if (auto r = commands.synchronize({.textures = {&atlasReady, 1}}); !r) { return std::unexpected(r.error()); }
        if (layered && staticPass) {
            std::vector<BufferBarrierDesc> layerReady{{.buffer = state->resources->cacheDiagnostics.get(),
                .before = {PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::PreRasterization | PipelineStageBits::FragmentShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}}};
            if (settings.collectGPUWork) {
                layerReady.push_back({.buffer = state->resources->buffers[4].get(),
                    .before = {PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
                    .after = {PipelineStageBits::PreRasterization, AccessBits::ShaderRead | AccessBits::ShaderWrite}});
            }
            if (auto r = commands.synchronize({.buffers = layerReady}); !r) { return std::unexpected(r.error()); }
        }
    }
    push.geometry = fullGeometry;
    push.rasterCoarse = 0;
    const BufferBarrierDesc diagnosticPublished{.buffer = state->resources->cacheDiagnostics.get(),
        .before = {PipelineStageBits::ComputeShader | PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
    if (auto r = commands.synchronize({.buffers = {&diagnosticPublished, 1}}); !r) { return std::unexpected(r.error()); }
    if (settings.collectGPUWork) {
        const BufferBarrierDesc workReady{.buffer = state->resources->buffers[4].get(),
            .before = {PipelineStageBits::PreRasterization, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader | PipelineStageBits::Transfer,
                AccessBits::ShaderRead | AccessBits::ShaderWrite | AccessBits::TransferRead}};
        if (auto r = commands.synchronize({.buffers = {&workReady, 1}}); !r) { return std::unexpected(r.error()); }
    }
    gpu.end();
    phase.next("Prepare VSM cache snapshot");
    if (settings.cacheEnabled) {
        auto next = std::make_shared<Cache>();
        next->device = &device;
        next->atlas = std::make_shared<ShadowAtlas>(ShadowAtlas{state->resources->textures[0], state->resources->views[0]});
        next->pages = state->resources->buffers[1]; next->projections = std::move(projections); next->origins = origins;
        next->casters = casters.states; next->settings = settings; next->light = light; next->farPlane = farPlane;
        next->sceneIdentity = casters.sceneIdentity; next->structuralRevision = casters.structuralRevision;
        next->staticAtlas = state->resources->staticAtlas; next->dynamicCasters = std::move(dynamicCasters);
        next->geometryRevision = casters.geometryRevision; next->texturePublicationRevision = casters.texturePublicationRevision;
        next->tessellationKey = casters.tessellationKey;
        next->dependencyKey = casters.dependencyKey; next->epoch = push.cacheEpoch; next->enabled = enabled;
        next->completion = frame->completion();
        next->submission = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
        if (auto r = commands.addSubmissionTransaction(next->submission); !r) { return std::unexpected(r.error()); }
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
        state->resources, state->resources->cacheDiagnostics.get()};
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
    commands.beginDebugLabel({.name = "VSM: receiver projection"});
    if (auto r = commands.bindExecution(state.programs->compute[5]->execution(), &push, sizeof(push)); !r) { return r; }
    const auto& desc = state.outputTextures[0]->desc();
    if (auto r = commands.dispatch((desc.width + 7) / 8, (desc.height + 7) / 8); !r) { return r; }
    commands.endDebugLabel();
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
    CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler, bool onePassProjection)
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
    const bool onePass = onePassProjection && !programs.optimizeProjectionStatistics &&
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
    if (onePass) { return projectOnePassPrepared(commands, groups, faceCount, log, profiler, gpuProfiler); }
    return {};
}

Result<> VirtualShadowMaps::projectOnePassPrepared(CommandBuffer& commands,
    std::span<const VirtualShadowMapProjectionGroup> groups, uint32_t faceCount, std::string& log,
    CPUProfileRecorder* profiler, RenderGraphExecutionContext* gpuProfiler)
{
    auto& first = *groups.front().faces[0]->state_;
    auto& programs = *first.programs;
    auto& device = *programs.device;
    CPUProfileScope phase(profiler, "Prepare VSM one-pass pipeline");
    if (!programs.onePassCompute) {
        const std::array macros{SlangMacroDefine{"VSM_ONE_PASS_PROJECTION", "1"},
            SlangMacroDefine{"VSM_WORKLOAD_DIAGNOSTICS", programs.collectGPUWork ? "1" : "0"},
            SlangMacroDefine{"VSM_OPTIMIZE_PROJECTION_STATISTICS", programs.optimizeProjectionStatistics ? "1" : "0"}};
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
        programs.onePassShader = std::move(*module);
        programs.onePassCompute = std::move(*pipeline);
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
        spdlog::info("[VSMOnePassShader] entry=vsmProjectOnePassMain spirvSHA256={} work={} projectionStatistics={}",
            debug::debugSha256(bytes), programs.collectGPUWork, programs.optimizeProjectionStatistics);
    }

    phase.next("Upload VSM projection jobs");
    std::vector<ShadowProjectionJob> jobs;
    jobs.reserve(faceCount);
    ShadowPush push = first.push;
    const auto& outputDesc = first.outputTextures[0]->desc();
    // LightsPass records every job with the same view/depth and filter settings.
    // Effective filtering, ownership, phase and light-specific inputs remain in
    // each job; all other projection inputs keep their push-constant storage.
    for (const auto& group : groups) {
        for (uint32_t face = 0; face < group.faceCount; ++face) {
            const auto& state = *group.faces[face]->state_;
            const auto& p = state.push;
            assert(state.registry == first.registry && state.programs->device == &device);
            assert(state.programs->collectGPUWork == programs.collectGPUWork &&
                state.programs->optimizeProjectionStatistics == programs.optimizeProjectionStatistics);
            assert(p.depth == push.depth && p.pageSize == push.pageSize && p.pcfRadius == push.pcfRadius &&
                p.rayLength == push.rayLength && p.rayLengthScale == push.rayLengthScale && p.maxRayAngle == push.maxRayAngle &&
                p.rayCount == push.rayCount && p.samplesPerRay == push.samplesPerRay && p.receiverFootprint == push.receiverFootprint);
            assert(state.outputTextures[0]->desc().width == outputDesc.width &&
                state.outputTextures[0]->desc().height == outputDesc.height);
            jobs.push_back({p.pages, p.statistics, p.projections, p.atlas,
                p.filterMode, p.shadow, p.debug, p.physicalPagesX,
                p.clipmapCount, p.enabled, p.lightType, p.projectionFace,
                p.pointFaces, p.filterReserved, p.samplePhase, p.view,
                p.lightRange, p.sourceRadius, p.angularRadius, p.lodScreenRayLength});
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
    commands.hostWriteBarrier();
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
    if (auto r = commands.synchronize({.textures = outputBegin, .buffers = statisticsBegin}); !r) { return r; }
    commands.beginDebugLabel({.name = "VSM: one-pass receiver projection"});
    if (auto r = commands.bindExecution(programs.onePassCompute->execution(), &push, sizeof(push)); !r) { return r; }
    if (auto r = commands.dispatch((outputDesc.width + 7u) / 8u, (outputDesc.height + 7u) / 8u); !r) { return r; }
    commands.endDebugLabel();
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
            addAtlas(cache->atlas->texture.get(), cache->settings.pageSize);
            if (cache->staticAtlas) { addAtlas(cache->staticAtlas->texture.get(), cache->settings.pageSize, true); }
            addBuffer(cache->pages.get());
        }
    };
    addCache(cache_);
    addCache(pending_);
    if (state_) {
        addAtlas(state_->resources->textures[0].get(), state_->push.pageSize);
        if (state_->resources->staticAtlas) { addAtlas(state_->resources->staticAtlas->texture.get(), state_->push.pageSize, true); }
        addBuffer(state_->resources->cacheDiagnostics.get());
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
