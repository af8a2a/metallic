#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/MeshletStreamReferenceCodec.h"
#include "json.hpp"

#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>

namespace {
using namespace metallic::scene;
using Json = nlohmann::json;

TEST(MeshletReferenceSceneBuild, ZeroNormalRepairUsesReferencePositionsConsistently)
{
    // Found with a deterministic random search: truncating these positions
    // changes the repaired normal across the reference octahedral grid.
    const std::array<float, 9> positions{
        1.9618778228759766f, 1.0056099891662598f, 1.2756950855255127f,
        1.1533211469650269f, 1.9595096111297607f, 1.7317547798156738f,
        1.4703302383422852f, 1.7548472881317139f, 1.8948696851730347f};
    const auto directory = std::filesystem::temp_directory_path() /
        ("MetallicReferenceZeroNormal-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto source = directory / "triangle.gltf";
    const std::array<float, 9> normals{};
    {
        std::ofstream binary(directory / "triangle.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
        binary.write(reinterpret_cast<const char*>(normals.data()), sizeof(normals));
        const Json root = {
            {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0}}}}},
            {"buffers", {{{"uri", "triangle.bin"}, {"byteLength", 72}}}},
            {"bufferViews", {{{"buffer", 0}, {"byteLength", 36}},
                {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}}}},
            {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"},
                    {"min", {1, 1, 1}}, {"max", {2, 2, 2}}},
                {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}}}},
            {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}, {"NORMAL", 1}}}}}}}}},
            {"nodes", {{{"mesh", 0}}}}
        };
        std::ofstream json(source);
        json << root.dump();
    }
    SCOPED_TRACE(Json(positions).dump());
    Scene resident;
    ASSERT_TRUE(resident.load(source)) << resident.lastLoadResult().error;
    const auto genericPath = directory / "generic.meshstream.bin";
    const auto offlinePath = directory / "offline.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAsset({.scene = &resident, .sourcePath = source,
        .outputPath = genericPath, .compressionMode = MeshletStreamPayloadCompression::Reference}, reason)) << reason;
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = offlinePath,
        .compressionMode = MeshletStreamPayloadCompression::Reference}, reason)) << reason;
    MeshletStreamAsset generic, offline;
    ASSERT_TRUE(generic.open(genericPath, reason)) << reason;
    ASSERT_TRUE(offline.open(offlinePath, reason)) << reason;
    std::vector<MeshletStreamAttributeValidation> validation;
    // The offline path already repairs zero authored normals after truncating
    // positions; the resident-to-generic path must follow the same order.
    ASSERT_TRUE(validateMeshletStreamAttributes(offline, source, validation, reason)) << reason;
    validation.clear();
    EXPECT_TRUE(validateMeshletStreamAttributes(generic, source, validation, reason)) << reason;
}

TEST(MeshletReferenceSceneBuild, MirroredGeneratedTangentsMatchSourceAndEditedGeometryStaysAuthoritative)
{
    const auto directory = std::filesystem::temp_directory_path() /
        ("MetallicReferenceMirrored-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto source = directory / "mirrored.gltf";
    const float positions[12] = {0.00001234f, 0, 0, 1.00023456f, 0, 0,
        0.00001234f, 1.0005734f, 0, -1.001234f, 0, 0};
    const float normals[12] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    const float texcoords[8] = {0.00002343f, 0, 1.0001234f, 0,
        0.00002343f, 1.0007234f, 1.0001234f, 0};
    const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    {
        std::ofstream binary(directory / "mirrored.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
        binary.write(reinterpret_cast<const char*>(normals), sizeof(normals));
        binary.write(reinterpret_cast<const char*>(texcoords), sizeof(texcoords));
        binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        const Json root = {
            {"asset", {{"version", "2.0"}}}, {"scene", 0}, {"scenes", {{{"nodes", {0}}}}},
            {"buffers", {{{"uri", "mirrored.bin"}, {"byteLength", 140}}}},
            {"bufferViews", {{{"buffer", 0}, {"byteLength", 48}},
                {{"buffer", 0}, {"byteOffset", 48}, {"byteLength", 48}},
                {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 32}},
                {{"buffer", 0}, {"byteOffset", 128}, {"byteLength", 12}}}},
            {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", 4}, {"type", "VEC3"},
                    {"min", {-2, 0, 0}}, {"max", {2, 2, 0}}},
                {{"bufferView", 1}, {"componentType", 5126}, {"count", 4}, {"type", "VEC3"}},
                {{"bufferView", 2}, {"componentType", 5126}, {"count", 4}, {"type", "VEC2"}},
                {{"bufferView", 3}, {"componentType", 5123}, {"count", 6}, {"type", "SCALAR"}}}},
            {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}}, {"indices", 3}}}}}}},
            {"nodes", {{{"mesh", 0}}}}
        };
        std::ofstream json(source);
        json << root.dump();
    }
    Scene resident;
    ASSERT_TRUE(resident.load(source)) << resident.lastLoadResult().error;
    ASSERT_EQ(resident.renderPrimitives().size(), 1);
    ASSERT_GT(resident.renderPrimitives().front().positions.size(), 4);
    ASSERT_FALSE(resident.renderPrimitives().front().hasAuthoredTangents);
    const auto genericPath = directory / "generic.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAsset({.scene = &resident, .sourcePath = source,
        .outputPath = genericPath, .compressionMode = MeshletStreamPayloadCompression::Reference}, reason)) << reason;
    MeshletStreamAsset generic;
    ASSERT_TRUE(generic.open(genericPath, reason)) << reason;
    std::vector<MeshletStreamAttributeValidation> validation;
    ASSERT_TRUE(validateMeshletStreamAttributes(generic, source, validation, reason)) << reason;
    ASSERT_EQ(validation.size(), 1);
    EXPECT_GT(validation.front().negativeTangentVertices, 0);

    // Emulate a caller's in-memory geometry edit. Source reimport is permitted
    // only after exact attribute/index comparison, so this edit must survive.
    auto& edited = const_cast<RenderPrimitive&>(resident.renderPrimitives().front());
    for (auto& position : edited.positions) { position.x += 0.125f; }
    float expectedPosition[3] = {edited.positions.front().x, edited.positions.front().y, edited.positions.front().z};
    canonicalizeMeshletStreamReferenceVertex(expectedPosition, nullptr, nullptr, nullptr);
    const auto editedPath = directory / "edited.meshstream.bin";
    ASSERT_TRUE(buildMeshletStreamAsset({.scene = &resident, .sourcePath = source,
        .outputPath = editedPath, .compressionMode = MeshletStreamPayloadCompression::Reference}, reason)) << reason;
    MeshletStreamAsset editedAsset;
    ASSERT_TRUE(editedAsset.open(editedPath, reason)) << reason;
    bool foundEditedPosition = false;
    for (uint32_t p = 0; p < editedAsset.pageCount(); ++p) {
        std::vector<uint8_t> scratch;
        std::span<const uint8_t> bytes;
        ASSERT_TRUE(decodeMeshletStreamPayloadForDevice(editedAsset.pages()[p], editedAsset.pagePayload(p), scratch, bytes, reason)) << reason;
        MeshletStreamPayloadHeader header;
        std::memcpy(&header, bytes.data(), sizeof(header));
        for (uint32_t v = 0; v < header.vertexCount; ++v) {
            foundEditedPosition |= std::memcmp(bytes.data() + header.positionOffsetBytes + v * 12, expectedPosition, 12) == 0;
        }
    }
    EXPECT_TRUE(foundEditedPosition);
}
} // namespace
