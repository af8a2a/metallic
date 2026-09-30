#include "Runtime/Scene/MeshletBuildParallel.h"
#include "Runtime/Scene/Scene.h"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace metallic::scene;

TEST(MeshletBuildParallel, ReusesDenseWorkerIndicesAndThreadsAcrossSubmissions)
{
    MeshletBuildParallel& pool = MeshletBuildParallel::shared(4);
    EXPECT_EQ(&pool, &MeshletBuildParallel::shared(4));
    std::array<std::thread::id, 4> first{}, second{};
    std::barrier start(4);
    pool.forEach(4, [&](size_t, size_t worker) {
        first.at(worker) = std::this_thread::get_id();
        start.arrive_and_wait();
    });
    pool.forEach(4, [&](size_t, size_t worker) {
        second.at(worker) = std::this_thread::get_id();
        start.arrive_and_wait();
    });
    EXPECT_EQ(first, second);
    for (size_t a = 0; a < first.size(); ++a) {
        EXPECT_NE(first[a], std::thread::id{});
        for (size_t b = a + 1; b < first.size(); ++b) { EXPECT_NE(first[a], first[b]); }
    }
}

TEST(MeshletBuildParallel, PropagatesFailureAndRemainsReusable)
{
    MeshletBuildParallel pool(4);
    EXPECT_THROW(pool.forEach(1024, [](size_t index, size_t) {
        if (index == 5) { throw std::runtime_error("meshlet worker failure"); }
    }), std::runtime_error);
    std::array<std::atomic_uint32_t, 2048> visits{};
    pool.forEach(visits.size(), [&](size_t index, size_t) { ++visits[index]; });
    for (const auto& count : visits) { EXPECT_EQ(count.load(), 1u); }
}

TEST(MeshletBuildParallel, ExecutesNestedSubmissionsWithoutPoolStarvation)
{
    for (size_t workers : {1u, 4u}) {
        MeshletBuildParallel pool(workers);
        std::atomic_uint32_t visits{0};
        pool.forEach(64, [&](size_t, size_t outerWorker) {
            pool.forEach(8, [&](size_t, size_t innerWorker) {
                if (innerWorker != outerWorker) { throw std::runtime_error("nested scratch worker changed"); }
                ++visits;
            });
        });
        EXPECT_EQ(visits.load(), 64u * 8u);
    }
}

RenderPrimitive makeGrid(bool attributes, bool protectedSeams)
{
    RenderPrimitive primitive;
    constexpr uint32_t cells = 64;
    for (uint32_t y = 0; y <= cells; ++y) {
        for (uint32_t x = 0; x <= cells; ++x) {
            const float u = float(x) / cells, v = float(y) / cells;
            primitive.positions.emplace_back(u, v, .01f * std::sin(u * 6.f) * std::cos(v * 6.f));
            if (attributes) {
                primitive.normals.emplace_back(0.f, 0.f, 1.f);
                primitive.texcoords0.emplace_back(u, v);
                primitive.tangents.emplace_back(1.f, 0.f, 0.f, 1.f);
            }
        }
    }
    for (uint32_t y = 0; y < cells; ++y) {
        for (uint32_t x = 0; x < cells; ++x) {
            const uint32_t a = y * (cells + 1) + x, b = a + 1, c = a + cells + 1, d = c + 1;
            primitive.indices.insert(primitive.indices.end(), {a, b, c, b, d, c});
        }
    }
    if (protectedSeams) {
        RenderPrimitive soup;
        for (size_t index = 0; index < primitive.indices.size(); ++index) {
            const uint32_t vertex = primitive.indices[index];
            soup.indices.push_back(static_cast<uint32_t>(soup.positions.size()));
            soup.positions.push_back(primitive.positions[vertex]);
            soup.normals.push_back(primitive.normals[vertex]);
            soup.texcoords0.push_back(primitive.texcoords0[vertex]);
            soup.tangents.push_back(primitive.tangents[vertex]);
            // Distinct chart values and handedness at otherwise identical
            // positions force protected terminal branches in the hierarchy.
            const uint32_t chart = static_cast<uint32_t>(index / 3u) % 4u;
            soup.texcoords0.back().x += float(chart);
            soup.tangents.back().w = (chart & 1u) ? -1.f : 1.f;
        }
        primitive = std::move(soup);
    }
    primitive.vertexCount = primitive.positions.size();
    primitive.indexCount = primitive.indices.size();
    primitive.triangleCount = primitive.indices.size() / 3;
    primitive.hasAuthoredNormals = attributes;
    primitive.hasAuthoredTangents = attributes;
    for (const auto& position : primitive.positions) { primitive.localBounds.include(position); }
    return primitive;
}

void expectFloat3Equal(const float3& a, const float3& b)
{
    EXPECT_EQ(a.x, b.x);
    EXPECT_EQ(a.y, b.y);
    EXPECT_EQ(a.z, b.z);
}

void expectBoundsEqual(const Bounds& a, const Bounds& b)
{
    EXPECT_EQ(a.valid, b.valid);
    expectFloat3Equal(a.min, b.min);
    expectFloat3Equal(a.max, b.max);
}

void expectLodEqual(const RenderPrimitive& a, const RenderPrimitive& b)
{
    EXPECT_EQ(a.meshletLodVertices, b.meshletLodVertices);
    EXPECT_EQ(a.meshletLodTriangles, b.meshletLodTriangles);
    ASSERT_EQ(a.meshletLodLevels.size(), b.meshletLodLevels.size());
    ASSERT_EQ(a.meshletLodGroups.size(), b.meshletLodGroups.size());
    ASSERT_EQ(a.meshletLodClusters.size(), b.meshletLodClusters.size());
    for (size_t index = 0; index < a.meshletLodLevels.size(); ++index) {
        const auto& x = a.meshletLodLevels[index];
        const auto& y = b.meshletLodLevels[index];
        EXPECT_EQ(x.groupOffset, y.groupOffset);
        EXPECT_EQ(x.groupCount, y.groupCount);
        EXPECT_EQ(x.clusterOffset, y.clusterOffset);
        EXPECT_EQ(x.clusterCount, y.clusterCount);
        EXPECT_EQ(x.minBoundingSphereRadius, y.minBoundingSphereRadius);
        EXPECT_EQ(x.minMaxQuadricError, y.minMaxQuadricError);
    }
    for (size_t index = 0; index < a.meshletLodGroups.size(); ++index) {
        const auto& x = a.meshletLodGroups[index];
        const auto& y = b.meshletLodGroups[index];
        EXPECT_EQ(x.clusterOffset, y.clusterOffset);
        EXPECT_EQ(x.clusterCount, y.clusterCount);
        EXPECT_EQ(x.lodLevel, y.lodLevel);
        expectBoundsEqual(x.bounds, y.bounds);
        expectFloat3Equal(x.boundingSphereCenter, y.boundingSphereCenter);
        EXPECT_EQ(x.boundingSphereRadius, y.boundingSphereRadius);
        EXPECT_EQ(x.maxQuadricError, y.maxQuadricError);
    }
    for (size_t index = 0; index < a.meshletLodClusters.size(); ++index) {
        const auto& x = a.meshletLodClusters[index];
        const auto& y = b.meshletLodClusters[index];
        EXPECT_EQ(x.vertexOffset, y.vertexOffset);
        EXPECT_EQ(x.vertexCount, y.vertexCount);
        EXPECT_EQ(x.triangleOffset, y.triangleOffset);
        EXPECT_EQ(x.triangleCount, y.triangleCount);
        EXPECT_EQ(x.lodLevel, y.lodLevel);
        EXPECT_EQ(x.lodGroupChildIndex, y.lodGroupChildIndex);
        EXPECT_EQ(x.lodGroupIndex, y.lodGroupIndex);
        EXPECT_EQ(x.refinedGroupIndex, y.refinedGroupIndex);
        EXPECT_EQ(x.lodError, y.lodError);
        expectBoundsEqual(x.bounds, y.bounds);
        expectFloat3Equal(x.boundingSphereCenter, y.boundingSphereCenter);
        EXPECT_EQ(x.boundingSphereRadius, y.boundingSphereRadius);
        expectFloat3Equal(x.coneApex, y.coneApex);
        expectFloat3Equal(x.coneAxis, y.coneAxis);
        EXPECT_EQ(x.coneCutoff, y.coneCutoff);
        EXPECT_EQ(x.packedCone, y.packedCone);
    }
}

TEST(MeshletBuildParallel, OneAndFourWorkersProduceIdenticalLodData)
{
    for (const auto settings : {std::array{false, false}, std::array{true, false}, std::array{true, true}}) {
        const RenderPrimitive source = makeGrid(settings[0], settings[1]);
        RenderPrimitive serial = source, parallel = source, repeated = source;
        MeshletLodBuildStats serialStats, parallelStats;
        ASSERT_TRUE(buildStreamMeshletsForPrimitive(serial, {.maxWorkers = 1, .lodStats = &serialStats}));
        ASSERT_TRUE(buildStreamMeshletsForPrimitive(parallel, {.maxWorkers = 4, .lodStats = &parallelStats}));
        ASSERT_TRUE(buildStreamMeshletsForPrimitive(repeated, {.maxWorkers = 4}));
        ASSERT_GT(serial.meshletLodGroups.size(), 1u);
        expectLodEqual(serial, parallel);
        expectLodEqual(parallel, repeated);
        EXPECT_EQ(serialStats.protectedVertexCount, parallelStats.protectedVertexCount);
        if (settings[1]) {
            EXPECT_GT(serialStats.protectedVertexCount, 0u);
        } else {
            EXPECT_GT(serial.meshletLodLevels.size(), 1u);
        }
        ASSERT_EQ(serialStats.depths.size(), parallelStats.depths.size());
        for (size_t depth = 0; depth < serialStats.depths.size(); ++depth) {
            const auto& a = serialStats.depths[depth];
            const auto& b = parallelStats.depths[depth];
            EXPECT_EQ(a.inputGroupCount, b.inputGroupCount);
            EXPECT_EQ(a.inputClusterCount, b.inputClusterCount);
            EXPECT_EQ(a.inputTriangleCount, b.inputTriangleCount);
            EXPECT_EQ(a.targetTriangleCount, b.targetTriangleCount);
            EXPECT_EQ(a.simplifiedTriangleCount, b.simplifiedTriangleCount);
            EXPECT_EQ(a.terminalGroupCount, b.terminalGroupCount);
            EXPECT_EQ(a.emptyResultGroupCount, b.emptyResultGroupCount);
            EXPECT_EQ(a.noReductionGroupCount, b.noReductionGroupCount);
        }
    }
}

} // namespace
