#include "Runtime/Render/RealtimeDecals.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <gtest/gtest.h>

namespace metallic::tests {
namespace {

render::builtin_pass::ScenePathTracePush decalCamera(bool orthographic = false)
{
    render::builtin_pass::ScenePathTracePush camera;
    camera.width = camera.height = 64;
    camera.center[2] = -1.0f;
    camera.upProjection[1] = 1.0f;
    camera.upProjection[3] = orthographic ? 1.0f : 0.0f;
    camera.viewport[0] = 1.0f;
    camera.viewport[3] = 1.5707963f;
    camera.clipOrtho[0] = 0.1f;
    camera.clipOrtho[1] = 20.0f;
    camera.clipOrtho[2] = 4.0f;
    return camera;
}

scene::RenderDecal projector(uint32_t id, int priority, float3 center, float3 size)
{
    scene::RenderDecal decal;
    decal.id = id;
    decal.priority = priority;
    decal.receiverMask = id;
    decal.worldFromDecal.AddScale(size);
    decal.worldFromDecal.SetTranslation(center);
    return decal;
}

TEST(RealtimeDecals, BothModesPreservePriorityAndPersistentIdOrder)
{
    const float3 center(0, 0, -4), size(0.5f);
    std::array decals{
        projector(6, 5, center, size), projector(5, 2, center, size),
        projector(4, 2, center, size), projector(7, 4, float3(50, 0, -4), size),
        projector(8, -1, center, size)};
    decals[4].enabled = false;
    decals[0].roughnessOpacity = 0.75f;
    const render::ScenePathTraceResources resources;
    const auto full = render::buildRealtimeDecalView(decals, decalCamera(), resources, true);
    ASSERT_EQ(full.records.size(), 4u);
    EXPECT_EQ(full.records[0].receiverMask, 4u);
    EXPECT_EQ(full.records[1].receiverMask, 5u);
    EXPECT_EQ(full.records[2].receiverMask, 7u);
    EXPECT_EQ(full.records[3].receiverMask, 6u);
    EXPECT_TRUE(full.tileRanges.empty());
    const auto tiled = render::buildRealtimeDecalView(decals, decalCamera(), resources);
    ASSERT_EQ(tiled.records.size(), 3u);
    EXPECT_EQ(tiled.records[2].receiverMask, 6u);
    ASSERT_EQ(tiled.tileRanges.size(), 16u);
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            const auto range = tiled.tileRanges[y * tiled.tilesX + x];
            const uint32_t expected = (x == 1 || x == 2) && (y == 1 || y == 2) ? 3u : 0u;
            ASSERT_EQ(range.count, expected);
            for (uint32_t i = 0; i < expected; ++i) {
                EXPECT_EQ(tiled.tileIndices.at(range.first + i), i);
            }
        }
    }
    ASSERT_EQ(tiled.materials.size(), 2u);
    EXPECT_EQ(tiled.records[0].materialIndex, tiled.records[1].materialIndex);
    EXPECT_NE(tiled.records[0].materialIndex, tiled.records[2].materialIndex);
    EXPECT_EQ(tiled.materials[0].baseColorTexture.textureIndex, UINT32_MAX);
    EXPECT_EQ(tiled.materials[0].roughnessTexture.transform0[3], 5.0f);
}

TEST(RealtimeDecals, ClipsProjectorEdgesAtTheNearPlane)
{
    const std::array decals{
        projector(1, 0, float3(0.0f), float3(2.0f)),
        projector(2, 0, float3(0.25f, 0, -0.05f), float3(0.2f, 0.06f, 0.4f)),
        projector(3, 0, float3(0, 0, 1), float3(0.2f)),
        projector(4, 0, float3(0, 0, -30), float3(1.0f))};
    const auto view = render::buildRealtimeDecalView(decals, decalCamera(), render::ScenePathTraceResources{});
    ASSERT_EQ(view.records.size(), 2u);
    ASSERT_EQ(view.tileRanges.size(), 16u);
    for (uint32_t tile = 0; tile < view.tileRanges.size(); ++tile) {
        const auto range = view.tileRanges[tile];
        ASSERT_NE(range.count, 0u);
        EXPECT_EQ(view.tileIndices.at(range.first), 0u);
        EXPECT_EQ(range.count, tile == 7 || tile == 11 ? 2u : 1u);
    }
}

TEST(RealtimeDecals, OrthographicProjectionAccountsForJitterAtATileBoundary)
{
    // Under the 4-unit ortho camera, x=-0.975 projects to pixel 16.4.
    const std::array decals{projector(1, 0, float3(-0.975f, 0.5f, -12.0f), float3(0.01f))};
    auto camera = decalCamera(true);
    render::ScenePathTraceResources resources;
    const auto fixed = render::buildRealtimeDecalView(decals, camera, resources);
    ASSERT_EQ(fixed.tileRanges.size(), 16u);
    EXPECT_EQ(fixed.tileRanges[4].count, 0u);
    EXPECT_EQ(fixed.tileRanges[5].count, 1u);
    camera.temporalJitter = 1;
    const auto jittered = render::buildRealtimeDecalView(decals, camera, resources);
    ASSERT_EQ(jittered.tileRanges.size(), 16u);
    EXPECT_EQ(jittered.tileRanges[4].count, 1u);
    EXPECT_EQ(jittered.tileRanges[5].count, 1u);
}

TEST(RealtimeDecals, PackedAffineRowsRecoverLocalCoordinates)
{
    auto decal = projector(1, 0, float3(3, 2, -5), float3(2, 3, -4));
    decal.worldFromDecal.a01 = 0.75f;
    const float3 local(0.25f, -0.3f, 0.45f);
    const float3 world = decal.worldFromDecal * local;
    const auto view = render::buildRealtimeDecalView(std::span(&decal, 1), decalCamera(),
        render::ScenePathTraceResources{}, true);
    ASSERT_EQ(view.records.size(), 1u);
    for (uint32_t row = 0; row < 3; ++row) {
        const float* r = view.records[0].worldToDecal[row];
        EXPECT_NEAR(r[0] * world.x + r[1] * world.y + r[2] * world.z + r[3], local.a[row], 1e-5f);
    }
}

TEST(RealtimeDecals, DenseViewFallsBackToOrderedRecordsWithoutTruncation)
{
    auto camera = decalCamera();
    camera.width = camera.height = 8192;
    std::vector<scene::RenderDecal> decals;
    for (uint32_t id = 129; id > 0; --id) {
        decals.push_back(projector(id, 0, float3(0.0f), float3(2.0f)));
    }
    const auto view = render::buildRealtimeDecalView(decals, camera, render::ScenePathTraceResources{});
    EXPECT_EQ(view.tilesX, 0u);
    EXPECT_TRUE(view.tileRanges.empty());
    EXPECT_TRUE(view.tileIndices.empty());
    ASSERT_EQ(view.records.size(), decals.size());
    for (uint32_t i = 0; i < view.records.size(); ++i) {
        EXPECT_EQ(view.records[i].receiverMask, i + 1);
    }
    EXPECT_EQ(view.materials.size(), 1u);
}

} // namespace
} // namespace metallic::tests
