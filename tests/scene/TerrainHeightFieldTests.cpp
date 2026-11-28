#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/ScenePicker.h"
#include "Runtime/Scene/TerrainHeightField.h"

#include <gtest/gtest.h>
#include <json.hpp>

#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <limits>

namespace {

using namespace metallic::scene;

TEST(TerrainHeightField, QueriesWorldHeightAndNormalsOnClosedBounds)
{
    std::string error;
    auto field = TerrainHeightField::create(3, 2, {-2.0f, 1.0f}, {2.0f, 5.0f},
        {8.0f, 12.0f, 16.0f, -4.0f, 0.0f, 4.0f}, error);
    ASSERT_TRUE(field) << error;
    const auto point = field->sample(0.5f, 2.25f);
    ASSERT_TRUE(point);
    EXPECT_FLOAT_EQ(point->position.y, 9.25f);
    const float3 normal = normalize(float3(-2.0f, 1.0f, 3.0f));
    EXPECT_NEAR(point->normal.x, normal.x, 1e-6f);
    EXPECT_NEAR(point->normal.y, normal.y, 1e-6f);
    EXPECT_NEAR(point->normal.z, normal.z, 1e-6f);
    for (const auto& corner : {float3(-2.0f, 8.0f, 1.0f), float3(2.0f, 4.0f, 5.0f)}) {
        const auto boundary = field->sample(corner.x, corner.z);
        ASSERT_TRUE(boundary);
        EXPECT_FLOAT_EQ(boundary->position.y, corner.y);
    }
    EXPECT_FALSE(field->sample(2.001f, 2.0f));
    EXPECT_FALSE(field->sample(0.0f, std::numeric_limits<float>::quiet_NaN()));
}

TEST(TerrainHeightField, UsesTheSourceDiagonalAndVertexNormalGradient)
{
    std::string error;
    auto field = TerrainHeightField::create(2, 2, {0.0f, 0.0f}, {1.0f, 1.0f}, {0, 0, 0, 8}, error);
    ASSERT_TRUE(field) << error;
    EXPECT_FLOAT_EQ(field->sample(0.25f, 0.25f)->position.y, 0.0f);
    EXPECT_FLOAT_EQ(field->sample(0.5f, 0.5f)->position.y, 0.0f);
    EXPECT_FLOAT_EQ(field->sample(0.75f, 0.75f)->position.y, 4.0f);
    EXPECT_GT(field->sample(0.75f, 0.75f)->normal.y, 0.0f);

    field = TerrainHeightField::create(3, 3, {-1.0f, -1.0f}, {1.0f, 1.0f},
        {2, 1, 2, 1, 0, 1, 2, 1, 2}, error);
    ASSERT_TRUE(field) << error;
    const auto corner = field->sample(-1.0f, -1.0f);
    ASSERT_TRUE(corner);
    EXPECT_NEAR(corner->normal.x, 2.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(corner->normal.y, 1.0f / 3.0f, 1e-6f);
    EXPECT_NEAR(corner->normal.z, 2.0f / 3.0f, 1e-6f);
}

TEST(TerrainHeightField, CastsVerticalRaysIncludingBoundsAndFiniteIntervals)
{
    std::string error;
    auto field = TerrainHeightField::create(2, 2, {-2.0f, -1.0f}, {2.0f, 1.0f}, {4, 4, 4, 4}, error);
    ASSERT_TRUE(field) << error;
    ScenePickRay ray{.origin = {2.0f, 10.0f, -1.0f}, .direction = {0.0f, -4.0f, 0.0f}};
    ray.minimumDistance = 6.0f;
    ray.maximumDistance = 6.0f;
    const auto boundary = field->raycast(ray);
    ASSERT_TRUE(boundary);
    EXPECT_FLOAT_EQ(boundary->distance, 6.0f);
    EXPECT_FLOAT_EQ(boundary->position.y, 4.0f);
    EXPECT_FLOAT_EQ(boundary->normal.y, 1.0f);
    ray.minimumDistance = 0.0f;
    ray.maximumDistance = 5.99f;
    EXPECT_FALSE(field->raycast(ray));
    ray.origin.x = 2.001f;
    ray.maximumDistance = 10.0f;
    EXPECT_FALSE(field->raycast(ray));
    ray.origin = {0.0f, 3.0f, 0.0f};
    ray.direction = {0.0f, 1.0f, 0.0f};
    const auto below = field->raycast(ray);
    ASSERT_TRUE(below);
    EXPECT_FLOAT_EQ(below->distance, 1.0f);
    ray.direction = float3(0.0f);
    EXPECT_FALSE(field->raycast(ray));
}

TEST(TerrainHeightField, TraversesSeveralCellsAndFindsTheNearestGrazingHit)
{
    std::string error;
    auto field = TerrainHeightField::create(5, 2, {0.0f, 0.0f}, {4.0f, 1.0f},
        {0, 4, 0, 4, 0, 0, 4, 0, 4, 0}, error);
    ASSERT_TRUE(field) << error;
    ScenePickRay ray{.origin = {-1.0f, 2.0f, 0.5f}, .direction = {1.0f, 0.0f, 0.0f}};
    auto hit = field->raycast(ray);
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->distance, 1.5f);
    ray.origin.x = 0.75f;
    hit = field->raycast(ray);
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->position.x, 1.5f);
    ray.origin = {-1.0f, 4.0f, 0.0f};
    hit = field->raycast(ray);
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->position.x, 1.0f);
    ray.minimumDistance = 2.01f;
    hit = field->raycast(ray);
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->position.x, 3.0f);
    ray.origin = {5.0f, 4.0f, 1.0f};
    ray.direction.x = -1.0f;
    hit = field->raycast(ray);
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->position.x, 1.0f);
}

TEST(TerrainHeightField, FindsTheFirstIntersectionWhenBothCellTrianglesIntersect)
{
    std::string error;
    auto field = TerrainHeightField::create(2, 2, {0.0f, 0.0f}, {1.0f, 1.0f}, {0, 4, 4, 0}, error);
    ASSERT_TRUE(field) << error;
    const auto hit = field->raycast({.origin = {-1.0f, 3.0f, 0.5f}, .direction = {1.0f, 0.0f, 0.0f}});
    ASSERT_TRUE(hit);
    EXPECT_FLOAT_EQ(hit->position.x, 0.25f);

    field = TerrainHeightField::create(3, 3, {0.0f, 0.0f}, {2.0f, 2.0f},
        {0, 1, 2, 1, 2, 3, 2, 3, 4}, error);
    ASSERT_TRUE(field) << error;
    const auto corner = field->raycast({.origin = {-1.0f, 2.0f, -1.0f}, .direction = {1.0f, 0.0f, 1.0f}});
    ASSERT_TRUE(corner);
    EXPECT_FLOAT_EQ(corner->position.x, 1.0f);
    EXPECT_FLOAT_EQ(corner->position.z, 1.0f);
}

class TerrainHeightFieldAsset : public testing::Test {
protected:
    void SetUp() override
    {
        const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() / ("metallic-terrain-query-" + std::to_string(timestamp));
        ASSERT_TRUE(std::filesystem::create_directory(directory_));
    }

    void TearDown() override
    {
        for (const auto& file : std::filesystem::directory_iterator(directory_)) {
            std::filesystem::remove(file.path());
        }
        std::filesystem::remove(directory_);
    }

    std::filesystem::path directory_;
};

TEST_F(TerrainHeightFieldAsset, ValidatesSidecarVersionSizeAndFiniteHeights)
{
    const auto path = directory_ / "Terrain.heightfield";
    std::array<uint32_t, 12> words{0x4648544Du, 1, 2, 2, 0, 0,
        std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(1.0f),
        std::bit_cast<uint32_t>(0.0f), std::bit_cast<uint32_t>(1.0f),
        std::bit_cast<uint32_t>(2.0f), std::bit_cast<uint32_t>(3.0f)};
    const auto write = [&](size_t count) {
        std::ofstream file(path, std::ios::binary);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t word = words[i];
            const std::array<char, 4> bytes{char(word), char(word >> 8u), char(word >> 16u), char(word >> 24u)};
            file.write(bytes.data(), bytes.size());
        }
    };
    std::string error;
    write(words.size());
    auto field = TerrainHeightField::load(path, error);
    ASSERT_TRUE(field) << error;
    EXPECT_FLOAT_EQ(field->sample(0.5f, 0.5f)->position.y, 1.5f);
    write(words.size() - 1);
    EXPECT_FALSE(TerrainHeightField::load(path, error));
    words[1] = 2;
    write(words.size());
    EXPECT_FALSE(TerrainHeightField::load(path, error));
    words[1] = 1;
    words.back() = std::bit_cast<uint32_t>(std::numeric_limits<float>::infinity());
    write(words.size());
    EXPECT_FALSE(TerrainHeightField::load(path, error));
}

TEST_F(TerrainHeightFieldAsset, RoundTripsNavigationHeightfieldWithoutPlacementPrototypes)
{
    const auto gltf = directory_ / "Terrain.gltf";
    const auto heightPath = directory_ / "Terrain.heightfield";
    constexpr std::array<uint32_t, 12> words{0x4648544Du, 1, 2, 2, 0, 0,
        std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(1.0f),
        std::bit_cast<uint32_t>(4.0f), std::bit_cast<uint32_t>(4.0f),
        std::bit_cast<uint32_t>(4.0f), std::bit_cast<uint32_t>(4.0f)};
    {
        std::ofstream file(heightPath, std::ios::binary);
        file.write(reinterpret_cast<const char*>(words.data()), sizeof(words));
        ASSERT_TRUE(file);
    }
    std::ofstream(gltf) << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"Terrain"}]})";
    const auto sidecar = SceneDocument::sidecarPathForSource(gltf);
    std::ofstream(sidecar) << nlohmann::json{{"version", 3}, {"source", gltf.filename().string()}, {"sceneIndex", 0},
        {"terrain", {{"heightField", heightPath.filename().string()}, {"cameraClearance", 2.0}}}};

    SceneDocument document;
    ASSERT_TRUE(document.load(sidecar)) << document.documentWarning();
    ASSERT_TRUE(document.terrainHeightField());
    EXPECT_FLOAT_EQ(document.terrainHeightField()->sample(0.5f, 0.5f)->position.y, 4.0f);
    EXPECT_FLOAT_EQ(document.terrainCameraClearance(), 2.0f);
    EXPECT_FALSE(document.supportsTerrainPlacement());
    std::string message;
    EXPECT_FALSE(document.addTerrainPlacement(TerrainPlacementKind::Rock, 0.5f, 0.5f, message));
    ASSERT_TRUE(document.save(message)) << message;

    SceneDocument reloaded;
    ASSERT_TRUE(reloaded.load(sidecar)) << reloaded.documentWarning();
    ASSERT_TRUE(reloaded.terrainHeightField());
    EXPECT_FLOAT_EQ(reloaded.terrainCameraClearance(), 2.0f);
    EXPECT_FALSE(reloaded.supportsTerrainPlacement());
}

TEST_F(TerrainHeightFieldAsset, MatchesSourcePickingAndCookedLOD0Triangles)
{
    using Json = nlohmann::json;
    const auto gltf = directory_ / "Terrain.gltf";
    constexpr std::array heights{0.0f, 4.0f, 0.0f, 1.0f, 3.0f, 2.0f, 0.0f, 2.0f, 0.0f};
    std::array<std::array<float, 3>, 9> vertices;
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i] = {static_cast<float>(i % 3), heights[i], static_cast<float>(i / 3)};
    }
    constexpr std::array<uint16_t, 24> indices{0, 3, 1, 1, 3, 4, 1, 4, 2, 2, 4, 5,
        3, 6, 4, 4, 6, 7, 4, 7, 5, 5, 7, 8};
    {
        std::ofstream stream(directory_ / "Terrain.bin", std::ios::binary);
        stream.write(reinterpret_cast<const char*>(vertices.data()), sizeof(vertices));
        stream.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
        ASSERT_TRUE(stream);
    }
    const Json document{
        {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", Json::array({{{"nodes", {0}}}})},
        {"nodes", Json::array({{{"mesh", 0}}})},
        {"meshes", Json::array({{{"primitives", Json::array({{{"attributes", {{"POSITION", 0}}}, {"indices", 1}}})}}})},
        {"buffers", Json::array({{{"uri", "Terrain.bin"}, {"byteLength", sizeof(vertices) + sizeof(indices)}}})},
        {"bufferViews", Json::array({{{"buffer", 0}, {"byteLength", sizeof(vertices)}},
            {{"buffer", 0}, {"byteOffset", sizeof(vertices)}, {"byteLength", sizeof(indices)}}})},
        {"accessors", Json::array({{{"bufferView", 0}, {"componentType", 5126}, {"count", 9},
            {"type", "VEC3"}, {"min", {0, 0, 0}}, {"max", {2, 4, 2}}},
            {{"bufferView", 1}, {"componentType", 5123}, {"count", 24}, {"type", "SCALAR"}}})},
    };
    std::ofstream(gltf) << document;

    std::string error;
    const auto cooked = directory_ / "Terrain.meshstream.bin";
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = gltf, .outputPath = cooked}, error)) << error;
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(cooked, error)) << error;
    std::vector<MeshletStreamAttributeValidation> validated;
    ASSERT_TRUE(validateMeshletStreamAttributes(asset, gltf, validated, error)) << error;
    ASSERT_EQ(validated.size(), 1u);
    EXPECT_EQ(validated.front().lod0Triangles, indices.size() / 3u);

    const auto field = TerrainHeightField::create(3, 3, {0.0f, 0.0f}, {2.0f, 2.0f},
        std::vector<float>(heights.begin(), heights.end()), error);
    ASSERT_TRUE(field) << error;
    Scene scene;
    ASSERT_TRUE(scene.load(gltf)) << scene.lastLoadResult().error;
    ScenePicker picker;
    for (const ScenePickRay& ray : {ScenePickRay{.origin = {0.35f, 10.0f, 0.7f}, .direction = {0.0f,-1.0f,0.0f}},
            ScenePickRay{.origin = {-1.0f, 5.0f, 1.63f}, .direction = {1.0f,-2.0f,0.0f}},
            ScenePickRay{.origin = {1.8f, -1.0f, 0.25f}, .direction = {-1.0f,3.0f,0.5f}}}) {
        const auto reference = picker.pick(scene, ray);
        const auto source = field->raycast(ray);
        ASSERT_TRUE(reference.hit());
        ASSERT_TRUE(source);
        EXPECT_NEAR(source->distance, reference.distance, 1e-5f);
    }
}

} // namespace
