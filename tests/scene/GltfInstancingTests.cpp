#include "Runtime/Scene/Scene.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "json.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace metallic::scene;
using Json = nlohmann::json;

struct InstanceFixture {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("MetallicGltfInstances-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Json root;
    InstanceFixture()
    {
        std::filesystem::create_directories(directory);
        // Interleaved translations prove that accessor offsets and stride are honored.
        const float translations[] = {99, 1, 0, 0, 99, 3, 0, 0};
        const int8_t rotations[] = {0, 0, 0, 127, 0, 0, 127, 0};
        const float scales[] = {1, 1, 1, -1, 2, 1};
        std::ofstream file(directory / "instances.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(translations), sizeof(translations));
        file.write(reinterpret_cast<const char*>(rotations), sizeof(rotations));
        file.write(reinterpret_cast<const char*>(scales), sizeof(scales));
        root = {
            {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0}}}}},
            {"extensionsRequired", {"EXT_mesh_gpu_instancing"}},
            {"buffers", {{{"uri", "geometry.bin"}, {"byteLength", 36}}, {{"uri", "instances.bin"}, {"byteLength", 64}}}},
            {"bufferViews", {{{"buffer", 0}, {"byteLength", 36}},
                {{"buffer", 1}, {"byteLength", 32}, {"byteStride", 16}},
                {{"buffer", 1}, {"byteOffset", 32}, {"byteLength", 8}},
                {{"buffer", 1}, {"byteOffset", 40}, {"byteLength", 24}}}},
            {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}, {"min", {0,0,0}}, {"max", {1,1,0}}},
                {{"bufferView", 1}, {"byteOffset", 4}, {"componentType", 5126}, {"count", 2}, {"type", "VEC3"}},
                {{"bufferView", 2}, {"componentType", 5120}, {"normalized", true}, {"count", 2}, {"type", "VEC4"}},
                {{"bufferView", 3}, {"componentType", 5126}, {"count", 2}, {"type", "VEC3"}}}},
            {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}, {"material", 0}},
                {{"attributes", {{"POSITION", 0}}}, {"material", 1}}}}}}},
            {"materials", {{{"name", "Opaque"}, {"pbrMetallicRoughness", {{"baseColorTexture", {{"index", 0}}}}}},
                {{"name", "Masked"}, {"alphaMode", "MASK"}, {"alphaCutoff", 0.25}, {"doubleSided", true}}}},
            {"images", {{{"uri", "does-not-exist.ktx2"}, {"mimeType", "image/ktx2"}}}},
            {"textures", {{{"source", 0}}}},
            {"nodes", {{{"name", "Parent"}, {"translation", {10,0,0}}, {"scale", {2,3,1}}, {"children", {1}}},
                {{"name", "Instances"}, {"mesh", 0}, {"translation", {0,4,0}}, {"children", {2}},
                    {"extensions", {{"EXT_mesh_gpu_instancing", {{"attributes", {{"TRANSLATION", 1}, {"ROTATION", 2}, {"SCALE", 3}}}}}}}},
                {{"name", "Ordinary child"}, {"mesh", 0}, {"translation", {0,0,5}}}}}
        };
    }
    std::filesystem::path save()
    {
        const auto path = directory / "scene.gltf";
        std::ofstream file(path); file << root.dump(); return path;
    }
    void geometry()
    {
        const float positions[] = {0,0,0, 1,0,0, 0,1,0};
        std::ofstream file(directory / "geometry.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(positions), sizeof(positions));
    }
};

TEST(GltfInstancing, MetadataReadsOnlyInstanceRangesAndPreservesHierarchy)
{
    InstanceFixture fixture;
    Scene scene;
    ASSERT_TRUE(scene.loadStreamMetadata(fixture.save())) << scene.lastLoadResult().error;
    ASSERT_EQ(scene.renderNodes().size(), 6u); // 2 expanded instances + one ordinary child, 2 primitives each.
    ASSERT_EQ(scene.nodes().size(), 5u);
    EXPECT_EQ(scene.nodes()[1].meshIndex, -1);
    EXPECT_EQ(scene.nodes()[2].parent, 1);
    EXPECT_EQ(scene.nodes()[3].parent, 1);
    const auto& info = scene.lastLoadResult().gpuInstancing;
    ASSERT_EQ(info.instances.size(), 2u);
    EXPECT_EQ(info.sourceNodeCount, 3u);
    EXPECT_EQ(info.rangeReadBytes, 60u);
    EXPECT_EQ(info.instances[1].nodeIndex, 4u);
    EXPECT_EQ(info.instances[1].sourceNodeIndex, 1u);
    EXPECT_EQ(info.instances[1].instanceIndex, 1u);
    const auto point = [](const float4x4& m, float4 p) { return m * p; };
    auto first = point(scene.renderNodes()[0].worldMatrix, float4(0,0,0,1));
    EXPECT_FLOAT_EQ(first.x, 12); EXPECT_FLOAT_EQ(first.y, 12);
    auto mirrored = point(scene.renderNodes()[2].worldMatrix, float4(1,1,0,1));
    EXPECT_NEAR(mirrored.x, 18, 1e-5); EXPECT_NEAR(mirrored.y, 6, 1e-5);
    auto child = point(scene.renderNodes()[4].worldMatrix, float4(0,0,0,1));
    EXPECT_FLOAT_EQ(child.x, 10); EXPECT_FLOAT_EQ(child.y, 12); EXPECT_FLOAT_EQ(child.z, 5);
    for (size_t i = 0; i < scene.renderNodes().size(); ++i) {
        EXPECT_EQ(scene.renderNodes()[i].materialIndex, int(i % 2));
        EXPECT_TRUE(scene.renderPrimitives()[i].positions.empty());
        EXPECT_EQ(scene.renderPrimitives()[i].storage, GeometryStorage::StreamAsset);
    }
    ASSERT_EQ(scene.images().size(), 1u);
    EXPECT_EQ(scene.images()[0].uri, "does-not-exist.ktx2");
    EXPECT_EQ(scene.images()[0].mimeType, "image/ktx2");
    EXPECT_TRUE(scene.images()[0].encodedData.empty());
    EXPECT_TRUE(scene.images()[0].decodedMips.empty());
    EXPECT_EQ(scene.materials()[0].baseColorTexture.textureIndex, 0);
    EXPECT_EQ(scene.materials()[1].alphaMode, "MASK");
    EXPECT_TRUE(scene.materials()[1].doubleSided);
    ASSERT_EQ(scene.lastLoadResult().gltfMaterialDescriptions.size(), 2u);
    EXPECT_EQ(Json::parse(scene.lastLoadResult().gltfMaterialDescriptions[1]), fixture.root["materials"][1]);
    Scene repeat;
    ASSERT_TRUE(repeat.loadStreamMetadata(fixture.save())) << repeat.lastLoadResult().error;
    EXPECT_EQ(repeat.renderNodes()[2].nodeIndex, scene.renderNodes()[2].nodeIndex);
}

TEST(GltfInstancing, ResidentAndCookMatchMetadataInstances)
{
    InstanceFixture fixture;
    fixture.geometry();
    const auto path = fixture.save();
    Scene metadata, resident;
    ASSERT_TRUE(metadata.loadStreamMetadata(path)) << metadata.lastLoadResult().error;
    ASSERT_TRUE(resident.load(path)) << resident.lastLoadResult().error;
    ASSERT_EQ(resident.renderNodes().size(), metadata.renderNodes().size());
    const auto output = fixture.directory / "scene.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = output}, reason)) << reason;
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    ASSERT_EQ(asset.instanceCount(), metadata.renderNodes().size());
    for (size_t i = 0; i < asset.instances().size(); ++i) {
        const auto& instance = asset.instances()[i];
        EXPECT_EQ(instance.renderNodeIndex, i);
        EXPECT_EQ(instance.materialIndex, metadata.renderNodes()[i].materialIndex);
        for (size_t c = 0; c < 16; ++c) {
            EXPECT_NEAR(instance.worldMatrix[c], metadata.renderNodes()[i].worldMatrix.a[c], 1e-5);
            EXPECT_NEAR(resident.renderNodes()[i].worldMatrix.a[c], metadata.renderNodes()[i].worldMatrix.a[c], 1e-5);
        }
    }
}

TEST(GltfInstancing, RejectsMalformedRangesAndCountsWithoutStaleScene)
{
    InstanceFixture fixture;
    const auto original = fixture.root;
    Scene scene;
    const auto reject = [&](std::string_view expected) {
        ASSERT_FALSE(scene.loadStreamMetadata(fixture.save()));
        EXPECT_FALSE(scene.valid()); EXPECT_TRUE(scene.renderNodes().empty());
        EXPECT_NE(scene.lastLoadResult().error.find(expected), std::string::npos) << scene.lastLoadResult().error;
    };
    ASSERT_TRUE(scene.loadStreamMetadata(fixture.save()));
    fixture.root["accessors"][3]["count"] = 1; reject("count mismatch");
    fixture.root = original; fixture.root["accessors"][1]["byteOffset"] = 20; reject("range/alignment");
    fixture.root = original; fixture.root["buffers"][1]["byteLength"] = 40; reject("range/alignment");
    fixture.root = original; fixture.root["accessors"][2]["normalized"] = false; reject("normalization");
    fixture.root = original; fixture.root["images"][0]["uri"] = "data:image/png;base64,AA=="; reject("external image");
}

TEST(GltfInstancing, ZorahFullMetadata)
{
    if (!std::getenv("METALLIC_TEST_ZORAH_FULL")) { GTEST_SKIP() << "Set METALLIC_TEST_ZORAH_FULL=1 for the local asset"; }
    Scene scene;
    const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ZorahFull/zorah_textured_public.v1.gltf";
    ASSERT_TRUE(scene.loadStreamMetadata(path)) << scene.lastLoadResult().error;
    EXPECT_EQ(scene.renderNodes().size(), 43068u);
    EXPECT_EQ(scene.stats().triangleCount, 18930392835ull);
    EXPECT_EQ(scene.materials().size(), 1514u);
    EXPECT_EQ(scene.images().size(), 4418u);
    EXPECT_EQ(scene.textures().size(), 4418u);
    ASSERT_EQ(scene.lastLoadResult().gltfMaterialDescriptions.size(), 1514u);
    EXPECT_TRUE(Json::parse(scene.lastLoadResult().gltfMaterialDescriptions[1513])["extensions"].contains("KHR_materials_unlit"));
    EXPECT_EQ(scene.lastLoadResult().gpuInstancing.sourceNodeCount, 13079u);
    EXPECT_EQ(scene.lastLoadResult().gpuInstancing.instances.size(), 7654u);
    const auto meshInstances = std::count_if(scene.nodes().begin(), scene.nodes().end(),
        [](const auto& node) { return node.meshIndex >= 0; });
    EXPECT_EQ(meshInstances, 16118);
    EXPECT_LE(scene.lastLoadResult().gpuInstancing.rangeReadBytes, 306160u);
    for (const auto& primitive : scene.renderPrimitives()) {
        EXPECT_TRUE(primitive.positions.empty()); EXPECT_TRUE(primitive.indices.empty());
    }
    for (const auto& image : scene.images()) {
        EXPECT_TRUE(image.encodedData.empty()); EXPECT_TRUE(image.decodedMips.empty());
    }
    const char* reportPath = std::getenv("METALLIC_ZORAH_Z1_REPORT");
    if (reportPath) {
        Json report{{"source", path.generic_string()}, {"primitiveInstances", scene.renderNodes().size()},
            {"meshInstances", meshInstances},
            {"trianglesWithInstancing", scene.stats().triangleCount}, {"materials", scene.materials().size()},
            {"images", scene.images().size()}, {"expandedInstances", scene.lastLoadResult().gpuInstancing.instances.size()},
            {"instanceRangeReadBytes", scene.lastLoadResult().gpuInstancing.rangeReadBytes},
            {"warnings", scene.lastLoadResult().warning}};
        std::ofstream file(reportPath); file << report.dump(2) << '\n';
        ASSERT_TRUE(file.good());
    }
}

TEST(GltfInstancing, ZorahFullProbes)
{
    const char* manifestPath = std::getenv("METALLIC_ZORAH_Z1_PROBES");
    if (!manifestPath) { GTEST_SKIP() << "Set METALLIC_ZORAH_Z1_PROBES to generated probes.json"; }
    std::ifstream file(manifestPath);
    const auto manifest = Json::parse(file);
    for (const auto& probe : manifest.at("probes")) {
        SCOPED_TRACE(probe.at("name").get<std::string>());
        const std::filesystem::path path(probe.at("path").get<std::string>());
        Scene scene;
        ASSERT_TRUE(scene.loadStreamMetadata(path)) << scene.lastLoadResult().error;
        ASSERT_EQ(scene.renderNodes().size(), probe.at("primitiveInstances").get<size_t>());
        EXPECT_EQ(scene.materials().size(), probe.at("materials").get<size_t>());
        EXPECT_EQ(scene.images().size(), probe.at("images").get<size_t>());
        EXPECT_EQ(scene.textures().size(), probe.at("textures").get<size_t>());
        const auto& expected = probe.at("expectedInstances");
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(scene.renderNodes()[i].materialIndex, expected[i].at("material").get<int>());
            for (size_t c = 0; c < 16; ++c) {
                const float reference = expected[i].at("worldMatrix").at(c).get<float>();
                EXPECT_NEAR(scene.renderNodes()[i].worldMatrix.a[c], reference, 1e-4f * std::max(1.0f, std::abs(reference)));
            }
        }
        // LargestMesh only exercises metadata until Z2 establishes its cook budget.
        if (!probe.at("cookRecommended").get<bool>()) { continue; }
        auto output = path; output.replace_extension("meshstream.bin");
        MeshletStreamAsset asset;
        std::string reason;
        ASSERT_TRUE(asset.open(output, reason)) << reason;
        ASSERT_EQ(asset.instanceCount(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(asset.instances()[i].renderNodeIndex, i);
            EXPECT_EQ(asset.instances()[i].materialIndex, scene.renderNodes()[i].materialIndex);
            for (size_t c = 0; c < 16; ++c) {
                EXPECT_NEAR(asset.instances()[i].worldMatrix[c], scene.renderNodes()[i].worldMatrix.a[c], 1e-5f);
            }
        }
    }
}
} // namespace
