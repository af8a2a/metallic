#include "Runtime/Render/RealtimeDecals.h"

#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <tuple>

namespace metallic::render {
namespace {

struct TileRect {
    uint32_t x0, y0, x1, y1; // Inclusive tile coordinates.
};

struct Projection {
    float3 eye, forward, right, up;
    float halfWidth, halfHeight, nearZ, farZ, guard;
    uint32_t width, height;
    bool orthographic;
};

Projection projection(const builtin_pass::ScenePathTracePush& camera)
{
    // Same ray basis as makeCameraRayBasis in VisibilityBufferDeferred.slang.
    const auto normalized = [](float3 v, float3 fallback) {
        const float lengthSquared = dot(v, v);
        return lengthSquared > 1e-8f ? v / std::sqrt(lengthSquared) : fallback;
    };
    const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
    const float3 center(camera.center[0], camera.center[1], camera.center[2]);
    const float3 cameraUp(camera.upProjection[0], camera.upProjection[1], camera.upProjection[2]);
    const float3 forward = normalized(center - eye, float3(0, 0, -1));
    const float3 right = normalized(cross(forward, normalized(cameraUp, float3(0, 1, 0))), float3(1, 0, 0));
    const float3 up = normalized(cross(right, forward), float3(0, 1, 0));
    const bool orthographic = camera.upProjection[3] > 0.5f;
    const float halfHeight = orthographic ? std::max(camera.clipOrtho[2] * 0.5f, 0.0001f)
        : std::tan(std::clamp(camera.viewport[3], 0.017453292f, 3.12413936f) * 0.5f);
    const float nearZ = std::max(camera.clipOrtho[0], 0.0001f);
    // The camera's temporal sample offsets stay within one render pixel.
    return {eye, forward, right, up, halfHeight * std::max(camera.viewport[0], 0.001f), halfHeight,
        nearZ, std::max(camera.clipOrtho[1], nearZ + 0.0001f), camera.temporalJitter ? 1.0f : 0.0f,
        camera.width, camera.height, orthographic};
}

std::optional<TileRect> projectedTiles(const float4x4& worldFromDecal, const Projection& view)
{
    float minX = std::numeric_limits<float>::max(), minY = minX;
    float maxX = -minX, maxY = -minY;
    auto include = [&](float3 p) {
        const float z = view.orthographic ? 1.0f : p.z;
        const float x = (0.5f + 0.5f * p.x / (view.halfWidth * z)) * view.width;
        const float y = (0.5f - 0.5f * p.y / (view.halfHeight * z)) * view.height;
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
    };
    std::array<float3, 8> corners;
    float minZ = std::numeric_limits<float>::max(), maxZ = -minZ;
    for (uint32_t i = 0; i < corners.size(); ++i) {
        const float3 local((i & 1u) ? 0.5f : -0.5f, (i & 2u) ? 0.5f : -0.5f, (i & 4u) ? 0.5f : -0.5f);
        const float3 offset = worldFromDecal * local - view.eye;
        const float3 p(dot(offset, view.right), dot(offset, view.up), dot(offset, view.forward));
        corners[i] = p;
        minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
        if (p.z >= view.nearZ) { include(p); }
    }
    if (maxZ < view.nearZ || minZ > view.farZ) { return {}; }
    if (minZ < view.nearZ) {
        // Clipping creates new projected extrema on the twelve box edges,
        // including when the eye lies inside the projector.
        for (uint32_t i = 0; i < corners.size(); ++i) {
            for (uint32_t bit : {1u, 2u, 4u}) {
                if ((i & bit) != 0) { continue; }
                const float3 a = corners[i], b = corners[i | bit];
                if ((a.z < view.nearZ) == (b.z < view.nearZ)) { continue; }
                float3 p = a + (b - a) * ((view.nearZ - a.z) / (b.z - a.z));
                p.z = view.nearZ;
                include(p);
            }
        }
    }
    minX -= view.guard; minY -= view.guard;
    maxX += view.guard; maxY += view.guard;
    if (maxX < 0 || maxY < 0 || minX >= view.width || minY >= view.height) { return {}; }
    const auto tile = [](float pixel, uint32_t extent) {
        return static_cast<uint32_t>(std::clamp(pixel, 0.0f, float(extent - 1))) / kDeferredDecalTileSize;
    };
    return TileRect{tile(minX, view.width), tile(minY, view.height),
        tile(maxX, view.width), tile(maxY, view.height)};
}

DeferredDecalData packDecal(const scene::RenderDecal& decal, uint32_t materialIndex)
{
    DeferredDecalData record{};
    auto inverse = decal.worldFromDecal;
    inverse.Invert(); // The scene authoring boundary accepts only invertible affine transforms.
    for (uint32_t row = 0; row < 3; ++row) {
        const float4 values = inverse.Row(row);
        std::copy_n(values.a, 4, record.worldToDecal[row]);
    }
    record.receiverMask = decal.receiverMask;
    record.materialIndex = materialIndex;
    return record;
}

DeferredDecalMaterial packMaterial(const scene::RenderDecal& decal, const ScenePathTraceResources& resources)
{
    DeferredDecalMaterial material{};
    std::copy_n(decal.baseColor.a, 4, material.baseColor);
    material.roughness = decal.roughness;
    material.roughnessOpacity = decal.roughnessOpacity;
    material.baseColorTexture = resources.decalTextureInfo(decal.baseColorTexture, ksRGBColorTexture,
        material.baseColorSourceExtent);
    material.roughnessTexture = resources.decalTextureInfo(decal.roughnessTexture, {}, material.roughnessSourceExtent);
    return material;
}

} // namespace

RealtimeDecalView buildRealtimeDecalView(std::span<const scene::RenderDecal> decals,
    const builtin_pass::ScenePathTracePush& camera, const ScenePathTraceResources& resources, bool fullScan)
{
    struct Candidate {
        const scene::RenderDecal* decal;
        TileRect tiles;
    };
    const auto view = projection(camera);
    std::vector<Candidate> candidates;
    for (const auto& decal : decals) {
        if (!decal.enabled) { continue; }
        if (fullScan) { candidates.push_back({&decal, {}}); }
        else if (auto tiles = projectedTiles(decal.worldFromDecal, view)) { candidates.push_back({&decal, *tiles}); }
    }
    std::ranges::sort(candidates, {}, [](const Candidate& c) { return std::tie(c.decal->priority, c.decal->id); });
    RealtimeDecalView result;
    result.records.reserve(candidates.size());
    using MaterialKey = std::tuple<std::array<float, 6>, int32_t, int32_t>;
    std::map<MaterialKey, uint32_t> materialIndices;
    for (const auto& c : candidates) {
        const auto& decal = *c.decal;
        const auto& color = decal.baseColor;
        const MaterialKey key{{color.x, color.y, color.z, color.w, decal.roughness, decal.roughnessOpacity},
            decal.baseColorTexture, decal.roughnessTexture};
        const auto [entry, inserted] = materialIndices.try_emplace(key, uint32_t(result.materials.size()));
        if (inserted) { result.materials.push_back(packMaterial(decal, resources)); }
        result.records.push_back(packDecal(decal, entry->second));
    }
    if (fullScan || candidates.empty()) { return result; }

    uint64_t entries = 0;
    for (const auto& c : candidates) {
        entries += uint64_t(c.tiles.x1 - c.tiles.x0 + 1) * (c.tiles.y1 - c.tiles.y0 + 1);
    }
    // A single typed shader buffer must fit Vulkan's portable 128 MiB
    // storage-buffer range. Dense views can evaluate the same ordered records.
    if (entries * sizeof(uint32_t) > 128ull * 1024 * 1024) { return result; }
    result.tilesX = (camera.width + kDeferredDecalTileSize - 1) / kDeferredDecalTileSize;
    const auto tilesY = (camera.height + kDeferredDecalTileSize - 1) / kDeferredDecalTileSize;
    result.tileRanges.resize(size_t(result.tilesX) * tilesY);
    const auto visitTiles = [&](const TileRect& rect, auto visit) {
        for (uint32_t y = rect.y0; y <= rect.y1; ++y) {
            for (uint32_t x = rect.x0; x <= rect.x1; ++x) { visit(result.tileRanges[y * result.tilesX + x]); }
        }
    };
    for (const auto& c : candidates) { visitTiles(c.tiles, [](DecalTileRange& range) { ++range.count; }); }
    uint32_t first = 0;
    for (auto& range : result.tileRanges) {
        range.first = first;
        first += range.count;
        range.count = 0;
    }
    result.tileIndices.resize(size_t(entries));
    for (uint32_t index = 0; index < candidates.size(); ++index) {
        visitTiles(candidates[index].tiles, [&](DecalTileRange& range) {
            result.tileIndices[range.first + range.count++] = index;
        });
    }
    return result;
}

} // namespace metallic::render
