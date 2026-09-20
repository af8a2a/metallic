#include "Runtime/Scene/GeometryAttributes.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "json.hpp"
#include "meshoptimizer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <cstdlib>

namespace {
using namespace metallic::scene;
using Json = nlohmann::json;

RenderPrimitive mirroredQuad()
{
    RenderPrimitive p;
    p.positions = {{0,0,0}, {1,0,0}, {0,1,0}, {-1,0,0}};
    p.normals.assign(4, float3(0,0,1));
    p.texcoords0 = {{0,0}, {1,0}, {0,1}, {1,0}};
    p.indices = {0,1,2, 0,2,3};
    p.vertexCount = 4; p.indexCount = 6; p.triangleCount = 2;
    p.hasAuthoredNormals = true;
    for (const auto& v : p.positions) { p.localBounds.include(v); }
    return p;
}

TEST(GeometryAttributes, SplitsMirroredUvCornersAndPreservesAuthoredData)
{
    auto p = mirroredQuad();
    const auto original = p;
    generateMissingTangents(p);
    EXPECT_EQ(p.positions.size(), 6u);
    EXPECT_FALSE(p.hasAuthoredTangents);
    for (size_t i = 0; i < 6; ++i) {
        const auto v = p.indices[i];
        const auto& t = p.tangents[v];
        EXPECT_NEAR(t.x, i < 3 ? 1.f : -1.f, 1e-6f);
        EXPECT_NEAR(t.y, 0.f, 1e-6f); EXPECT_NEAR(t.z, 0.f, 1e-6f);
        EXPECT_EQ(t.w, i < 3 ? 1.f : -1.f);
        EXPECT_FLOAT_EQ(p.positions[v].x, original.positions[original.indices[i]].x);
        EXPECT_FLOAT_EQ(p.texcoords0[v].x, original.texcoords0[original.indices[i]].x);
    }
    const auto tangents = p.tangents;
    generateMissingTangents(p);
    EXPECT_EQ(p.positions.size(), 6u);
    EXPECT_EQ(std::memcmp(tangents.data(), p.tangents.data(), tangents.size() * sizeof(float4)), 0);
    p = original; p.texcoords0.clear(); generateMissingTangents(p);
    EXPECT_TRUE(p.tangents.empty());
}

TEST(GeometryAttributes, RepairsZeroNormalsWithoutChangingValidAuthoredValues)
{
    auto p = mirroredQuad();
    p.normals[0] = float3(0, 0, 0);
    p.normals[1] = float3(0, 0, .9997f);
    // Tiny but nondegenerate source triangles must retain their orientation.
    for (auto& v : p.positions) { v *= 1e-5f; }
    EXPECT_EQ(repairZeroGeometryNormals(p), 1u);
    EXPECT_FLOAT_EQ(p.normals[0].x, 0); EXPECT_FLOAT_EQ(p.normals[0].y, 0); EXPECT_FLOAT_EQ(p.normals[0].z, 1);
    EXPECT_FLOAT_EQ(p.normals[1].x, 0); EXPECT_FLOAT_EQ(p.normals[1].y, 0); EXPECT_FLOAT_EQ(p.normals[1].z, .9997f);
    EXPECT_EQ(repairZeroGeometryNormals(p), 0u);
    generateMissingTangents(p);
    std::string reason;
    EXPECT_TRUE(validateGeometryAttributes(p, reason)) << reason;
    p.normals[0].x = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(repairZeroGeometryNormals(p), 0u);
    EXPECT_FALSE(validateGeometryAttributes(p, reason));
    p = mirroredQuad(); p.indices = {0, 0, 0}; p.normals[0] = float3(0.0f);
    EXPECT_EQ(repairZeroGeometryNormals(p), 1u);
    EXPECT_FLOAT_EQ(p.normals[0].x, 0); EXPECT_FLOAT_EQ(p.normals[0].y, 1); EXPECT_FLOAT_EQ(p.normals[0].z, 0);
}

TEST(GeometryAttributes, CoarseLodsPreserveChartsAndUseUvError)
{
    RenderPrimitive p;
    constexpr uint32_t n = 24;
    for (uint32_t chart = 0; chart < 2; ++chart) {
        const uint32_t base = static_cast<uint32_t>(p.positions.size());
        for (uint32_t y = 0; y <= n; ++y) {
            for (uint32_t x = 0; x <= n; ++x) {
                const float u = float(x) / n, v = float(y) / n;
                p.positions.emplace_back(float(chart) + u - 1, v, 0.f);
                p.normals.emplace_back(0.f,0.f,1.f);
                const float wave = 0.08f * std::sin(u * 6.2831853f) * std::sin(v * 6.2831853f);
                p.texcoords0.emplace_back(chart == 0 ? u + wave : 5 - u - wave, v);
            }
        }
        for (uint32_t y = 0; y < n; ++y) {
            for (uint32_t x = 0; x < n; ++x) {
                const uint32_t a = base + y * (n + 1) + x, b = a + 1, c = a + n + 1, d = c + 1;
                p.indices.insert(p.indices.end(), {a,b,c, b,d,c});
            }
        }
    }
    p.vertexCount = p.positions.size(); p.indexCount = p.indices.size(); p.triangleCount = p.indices.size() / 3;
    generateMissingTangents(p);
    ASSERT_TRUE(buildStreamMeshletsForPrimitive(p, {.maxWorkers = 2}));
    ASSERT_GT(p.meshletLodLevels.size(), 1u);
    uint64_t coarseTriangles = 0;
    bool uvError = false;
    for (const auto& group : p.meshletLodGroups) {
        uvError |= group.maxQuadricError > 1e-4f && group.maxQuadricError < 1e10f;
    }
    EXPECT_TRUE(uvError); // A flat, constant-normal surface has zero geometric error.
    for (const auto& cluster : p.meshletLodClusters) {
        if (cluster.lodLevel == 0) { continue; }
        coarseTriangles += cluster.triangleCount;
        for (uint32_t t = 0; t < cluster.triangleCount; ++t) {
            float minU = 10, maxU = -10, sign = 0;
            for (uint32_t c = 0; c < 3; ++c) {
                const auto v = p.meshletLodVertices[cluster.vertexOffset + p.meshletLodTriangles[cluster.triangleOffset + t*3+c]];
                minU = std::min(minU, p.texcoords0[v].x); maxU = std::max(maxU, p.texcoords0[v].x);
                if (c == 0) { sign = p.tangents[v].w; }
                EXPECT_EQ(p.tangents[v].w, sign);
                // A constant tangent-space normal map remains in the authored
                // hemisphere on both mirrored charts, including coarse faces.
                const auto& t = p.tangents[v];
                const float3 tangent(t.x, t.y, t.z);
                const float3 bitangent = cross(p.normals[v], tangent) * t.w;
                const float3 mapped = tangent * 0.2f + bitangent * 0.3f + p.normals[v] * std::sqrt(0.87f);
                EXPECT_NEAR(mapped.x, p.texcoords0[v].x < 2.f ? 0.2f : -0.2f, 1e-4f);
                EXPECT_NEAR(mapped.y, 0.3f, 1e-4f);
                EXPECT_GT(mapped.z, 0.9f);
            }
            EXPECT_TRUE(maxU <= 1.1f || minU >= 3.9f) << minU << ", " << maxU;
        }
    }
    EXPECT_GT(coarseTriangles, 0u);
}

struct AttributeFixture {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("MetallicAttributes-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Json root;
    AttributeFixture()
    {
        std::filesystem::create_directories(directory);
        auto p = mirroredQuad();
        std::ofstream binary(directory / "mesh.bin", std::ios::binary);
        for (const auto& v : p.positions) { binary.write(reinterpret_cast<const char*>(&v.x), 12); }
        for (const auto& v : p.normals) { binary.write(reinterpret_cast<const char*>(&v.x), 12); }
        for (const auto& v : p.texcoords0) { binary.write(reinterpret_cast<const char*>(&v.x), 8); }
        binary.write(reinterpret_cast<const char*>(p.indices.data()), p.indices.size() * 4);
        root = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1,2,3]}],
            "nodes":[{"mesh":0},{"mesh":1},{"mesh":2},{"mesh":3}],
            "buffers":[{"uri":"mesh.bin","byteLength":152}],
            "bufferViews":[{"buffer":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},
                           {"buffer":0,"byteOffset":96,"byteLength":32},{"buffer":0,"byteOffset":128,"byteLength":24}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1,0,0],"max":[1,1,0]},
                         {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
                         {"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},
                         {"bufferView":3,"componentType":5125,"count":6,"type":"SCALAR"}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]}],
            "materials":[{"name":"first"},{"name":"second"}]})");
        const auto mesh = root["meshes"][0];
        for (int i = 1; i < 4; ++i) { root["meshes"].push_back(mesh); }
        root["meshes"][2]["primitives"][0]["material"] = 1;
        root["accessors"].push_back(root["accessors"][2]);
        root["meshes"][3]["primitives"][0]["attributes"]["TEXCOORD_0"] = 4;
    }
    std::filesystem::path save()
    {
        const auto path = directory / "scene.gltf";
        std::ofstream file(path); file << root.dump(); return path;
    }
};

TEST(GeometryAttributes, CompactUploadPreservesGeometryUvAndDiskPayload)
{
    for (bool byteRle : {false, true}) {
        AttributeFixture fixture;
        const auto source = fixture.save();
        const auto cache = fixture.directory / "compact.bin";
        std::string reason;
        ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath=source,.outputPath=cache,
            .compressionMode=byteRle ? MeshletStreamPayloadCompression::ByteRle : MeshletStreamPayloadCompression::None},reason)) << reason;
        MeshletStreamAsset original, compact;
        ASSERT_TRUE(original.open(cache,reason)) << reason;
        ASSERT_TRUE(compact.open(cache,reason)) << reason;
        ASSERT_TRUE(compact.compactShadingForDevice(reason)) << reason;
        uint64_t saved = 0;
        for (uint32_t page=0; page<original.pageCount(); ++page) {
            std::vector<uint8_t> aStorage,bStorage;
            std::span<const uint8_t> a,b;
            ASSERT_TRUE(decodeMeshletStreamPayloadForDevice(original.pages()[page],original.pagePayload(page),aStorage,a,reason)) << reason;
            ASSERT_TRUE(decodeMeshletStreamPayloadForDevice(compact.pages()[page],compact.pagePayload(page),bStorage,b,reason)) << reason;
            MeshletStreamPayloadHeader ah,bh;
            std::memcpy(&ah,a.data(),sizeof(ah)); std::memcpy(&bh,b.data(),sizeof(bh));
            ASSERT_EQ(b.size(),meshletStreamDevicePayloadSize(compact.pages()[page]));
            EXPECT_TRUE(validateMeshletStreamDeviceHeader(bh,compact.pages()[page],reason)) << reason;
            ASSERT_LT(b.size(),a.size()); saved += a.size()-b.size();
            EXPECT_EQ(bh.normalFormat,uint32_t(MeshletStreamPayloadFormat::OctahedralNormal));
            EXPECT_EQ(bh.tangentFormat,uint32_t(MeshletStreamPayloadFormat::OctahedralTangent));
            EXPECT_EQ(std::memcmp(a.data()+ah.positionOffsetBytes,b.data()+bh.positionOffsetBytes,ah.vertexCount*12u),0);
            EXPECT_EQ(std::memcmp(a.data()+ah.texcoord0OffsetBytes,b.data()+bh.texcoord0OffsetBytes,ah.vertexCount*8u),0);
            EXPECT_EQ(std::memcmp(a.data()+ah.clusterOffsetBytes,b.data()+bh.clusterOffsetBytes,ah.clusterCount*sizeof(MeshletStreamPayloadCluster)),0);
            EXPECT_EQ(std::memcmp(a.data()+ah.triangleOffsetBytes,b.data()+bh.triangleOffsetBytes,ah.triangleIndexCount),0);
            for (uint32_t v=0; v<ah.vertexCount; ++v) {
                float w; uint32_t packed;
                std::memcpy(&w,a.data()+ah.tangentOffsetBytes+v*16u+12u,4);
                std::memcpy(&packed,b.data()+bh.tangentOffsetBytes+v*4u,4);
                EXPECT_EQ((packed & 0x40000000u)!=0,w<0);
            }
            MeshletStreamPayloadHeader disk;
            std::memcpy(&disk,compact.pagePayload(page).data(),sizeof(disk));
            EXPECT_EQ(disk.normalFormat,uint32_t(MeshletStreamPayloadFormat::Float32x4));
            if (!byteRle) {
                std::vector<uint8_t> damaged(compact.pagePayload(page).begin(),compact.pagePayload(page).end());
                disk.normalOffsetBytes=disk.positionOffsetBytes; std::memcpy(damaged.data(),&disk,sizeof(disk));
                EXPECT_FALSE(decodeMeshletStreamPayloadForDevice(compact.pages()[page],damaged,bStorage,b,reason));
                EXPECT_NE(reason.find("overlap"),std::string::npos) << reason;
            }
        }
        EXPECT_GT(saved,0u);
    }
}

TEST(GeometryAttributes, CookAttributesReuseAndResume)
{
    AttributeFixture fixture;
    auto source = fixture.save();
    const auto output = fixture.directory / "test.meshstream.bin";
    std::string reason;
    MeshletStreamAssetOfflineBuildDesc desc{.sourcePath = source, .outputPath = output, .maxNewGeometriesPerInvocation = 1};
    ASSERT_FALSE(buildMeshletStreamAssetOffline(desc, reason));
    EXPECT_NE(reason.find("paused"), std::string::npos) << reason;
    desc.maxNewGeometriesPerInvocation = 0;
    ASSERT_TRUE(buildMeshletStreamAssetOffline(desc, reason)) << reason;
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    EXPECT_TRUE(asset.isCurrentForSource(source));
    EXPECT_TRUE(asset.isRuntimeCompatibleForSource(source, reason)) << reason;
    ASSERT_EQ(asset.instances().size(), 4u);
    EXPECT_EQ(asset.primitiveCount(), 3u); // Same material/accessors share; different material or UV identity does not.
    EXPECT_EQ(asset.instances()[0].primitiveIndex, asset.instances()[1].primitiveIndex);
    EXPECT_NE(asset.instances()[0].primitiveIndex, asset.instances()[2].primitiveIndex);
    EXPECT_NE(asset.instances()[0].primitiveIndex, asset.instances()[3].primitiveIndex);
    std::vector<MeshletStreamAttributeValidation> validation;
    ASSERT_TRUE(validateMeshletStreamAttributes(asset, source, validation, reason)) << reason;
    for (const auto& result : validation) {
        EXPECT_EQ(result.sourceVertices, 4u); EXPECT_EQ(result.preparedVertices, 6u);
        EXPECT_EQ(result.lod0Triangles, 2u); EXPECT_GT(result.negativeTangentVertices, 0u);
    }
    // Validator must detect attribute corruption even when sizes, topology and
    // payload decoding remain valid. Mutate source UV without changing its shape.
    std::fstream binary(fixture.directory / "mesh.bin", std::ios::in | std::ios::out | std::ios::binary);
    binary.seekp(96); const float changed = 0.125f; binary.write(reinterpret_cast<const char*>(&changed), 4); binary.close();
    EXPECT_FALSE(validateMeshletStreamAttributes(asset, source, validation, reason));
}

void setCookRevision(const std::filesystem::path& path, uint32_t revision)
{
    // Container v9 retains the cook revision in its final reserved header word.
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(file.is_open());
    file.seekp(212);
    file.write(reinterpret_cast<const char*>(&revision), sizeof(revision));
    ASSERT_TRUE(file.good());
}

void usePositionOnly(AttributeFixture& fixture)
{
    for (auto& mesh : fixture.root["meshes"]) {
        mesh["primitives"][0]["attributes"] = {{"POSITION", 0}};
    }
}

TEST(GeometryAttributes, LegacyPositionOnlyCookIsRuntimeCompatibleButNotCurrent)
{
    AttributeFixture fixture;
    usePositionOnly(fixture);
    const auto source = fixture.save();
    const auto output = fixture.directory / "legacy.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = output}, reason)) << reason;
    setCookRevision(output, 0);
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    EXPECT_FALSE(asset.isCurrentForSource(source));
    EXPECT_TRUE(asset.isRuntimeCompatibleForSource(source, reason)) << reason;
    EXPECT_TRUE(reason.empty());
    asset.close();

    setCookRevision(output, kGeometryCookRevision + 1);
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    EXPECT_FALSE(asset.isRuntimeCompatibleForSource(source, reason));
    EXPECT_NE(reason.find("cook revision"), std::string::npos) << reason;
    asset.close();

    setCookRevision(output, 0);
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    const auto binary = fixture.directory / "mesh.bin";
    const auto oldWriteTime = std::filesystem::last_write_time(binary);
    std::filesystem::last_write_time(binary, oldWriteTime + std::chrono::seconds(2));
    EXPECT_FALSE(asset.isRuntimeCompatibleForSource(source, reason));
    EXPECT_NE(reason.find("dependencies"), std::string::npos) << reason;
    std::filesystem::last_write_time(binary, oldWriteTime);
    EXPECT_TRUE(asset.isRuntimeCompatibleForSource(source, reason)) << reason;
    std::filesystem::last_write_time(source, std::filesystem::last_write_time(source) + std::chrono::seconds(2));
    EXPECT_FALSE(asset.isRuntimeCompatibleForSource(source, reason));
    EXPECT_NE(reason.find("dependencies"), std::string::npos) << reason;
}

TEST(GeometryAttributes, LegacyAttributedCookStillRequiresRecooking)
{
    AttributeFixture fixture;
    const auto source = fixture.save();
    const auto output = fixture.directory / "legacy.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = output}, reason)) << reason;
    setCookRevision(output, 0);
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    EXPECT_FALSE(asset.isCurrentForSource(source));
    EXPECT_FALSE(asset.isRuntimeCompatibleForSource(source, reason));
    EXPECT_NE(reason.find("legacy pages contain vertex attributes"), std::string::npos) << reason;
}

TEST(GeometryAttributes, LegacyGpuInstancingRequiresRecookingEvenWithoutAttributes)
{
    AttributeFixture fixture;
    usePositionOnly(fixture);
    fixture.root["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]["attributes"] = {{"TRANSLATION", 0}};
    const auto source = fixture.save();
    const auto output = fixture.directory / "legacy.meshstream.bin";
    std::string reason;
    ASSERT_TRUE(buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = output}, reason)) << reason;
    setCookRevision(output, 0);
    MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(output, reason)) << reason;
    EXPECT_FALSE(asset.isRuntimeCompatibleForSource(source, reason));
    EXPECT_NE(reason.find("GPU instancing"), std::string::npos) << reason;
}

TEST(GeometryAttributes, RejectsBrokenAttributesInsteadOfDroppingThem)
{
    AttributeFixture fixture;
    fixture.root["accessors"][1]["count"] = 3;
    std::string reason;
    EXPECT_FALSE(buildMeshletStreamAssetOffline({.sourcePath = fixture.save(), .outputPath = fixture.directory / "broken.bin"}, reason));
    EXPECT_NE(reason.find("NORMAL accessor"), std::string::npos) << reason;
    auto p = mirroredQuad(); p.texcoords0[0].x = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(validateGeometryAttributes(p, reason));
    AttributeFixture morph;
    morph.root["meshes"][1]["primitives"][0]["targets"] = {{{"POSITION", 0}}};
    EXPECT_FALSE(buildMeshletStreamAssetOffline({.sourcePath = morph.save(), .outputPath = morph.directory / "morph.bin"}, reason));
    EXPECT_NE(reason.find("static attributes"), std::string::npos) << reason;
}

TEST(GeometryAttributes, SourceUvEditsInvalidateResidentCache)
{
    AttributeFixture fixture;
    const auto path = fixture.save();
    Scene first;
    ASSERT_TRUE(first.load(path)) << first.lastLoadResult().error;
    ASSERT_TRUE(first.lastLoadResult().meshletCacheSaved);
    Scene repeat;
    ASSERT_TRUE(repeat.load(path));
    EXPECT_TRUE(repeat.lastLoadResult().meshletCacheLoaded);
    std::fstream binary(fixture.directory / "mesh.bin", std::ios::in | std::ios::out | std::ios::binary);
    binary.seekp(96); const float uv = 0.05f; binary.write(reinterpret_cast<const char*>(&uv), 4); binary.close();
    Scene changed;
    ASSERT_TRUE(changed.load(path));
    EXPECT_FALSE(changed.lastLoadResult().meshletCacheLoaded);
}

TEST(GeometryAttributes, ZorahProbesMatchIndependentResidentImport)
{
    const char* manifestPath = std::getenv("METALLIC_ZORAH_Z2_PROBES");
    if (!manifestPath) { GTEST_SKIP() << "Set METALLIC_ZORAH_Z2_PROBES to cooked probes.json"; }
    std::ifstream manifestFile(manifestPath);
    const auto manifest = Json::parse(manifestFile);
    AttributeFixture fixture;
    for (const auto& probe : manifest["probes"]) {
        if (!probe["cookRecommended"].get<bool>()) { continue; }
        SCOPED_TRACE(probe["name"].get<std::string>());
        const std::filesystem::path sourcePath(probe["path"].get<std::string>());
        std::ifstream input(sourcePath); auto root = Json::parse(input);
        std::vector<uint8_t> decoded;
        // Independent reference path: decode raw views, then let the ordinary
        // TinyGLTF resident reader interpret accessors and normalized values.
        for (auto& view : root["bufferViews"]) {
            const auto ext = view.value("extensions", Json::object()).value("EXT_meshopt_compression", Json());
            const auto& stored = ext.is_null() ? view : ext;
            const auto bufferPath = sourcePath.parent_path() / root["buffers"][stored["buffer"].get<size_t>()]["uri"].get<std::string>();
            std::ifstream file(bufferPath, std::ios::binary);
            ASSERT_TRUE(file.good());
            std::vector<uint8_t> bytes(stored["byteLength"].get<size_t>());
            file.seekg(stored.value("byteOffset", uint64_t(0)));
            ASSERT_TRUE(file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()));
            decoded.resize((decoded.size() + 15) & ~size_t(15));
            const size_t offset = decoded.size(), size = view["byteLength"].get<size_t>();
            decoded.resize(offset + size);
            auto* destination = decoded.data() + offset;
            if (ext.is_null()) { ASSERT_EQ(bytes.size(), size); std::memcpy(destination, bytes.data(), size); }
            else {
                const size_t count = ext["count"], stride = ext["byteStride"];
                const std::string mode = ext["mode"], filter = ext.value("filter", "NONE");
                ASSERT_EQ(count * stride, size);
                int result = -1;
                if (mode == "ATTRIBUTES") { result = meshopt_decodeVertexBuffer(destination, count, stride, bytes.data(), bytes.size()); }
                if (mode == "TRIANGLES") { result = meshopt_decodeIndexBuffer(destination, count, stride, bytes.data(), bytes.size()); }
                if (mode == "INDICES") { result = meshopt_decodeIndexSequence(destination, count, stride, bytes.data(), bytes.size()); }
                ASSERT_EQ(result, 0);
                if (filter == "OCTAHEDRAL") { meshopt_decodeFilterOct(destination, count, stride); }
                else if (filter == "QUATERNION") { meshopt_decodeFilterQuat(destination, count, stride); }
                else if (filter == "EXPONENTIAL") { meshopt_decodeFilterExp(destination, count, stride); }
                else { ASSERT_EQ(filter, "NONE"); }
            }
            view["buffer"] = 0; view["byteOffset"] = offset; view.erase("extensions");
        }
        const auto name = probe["name"].get<std::string>();
        root["buffers"] = {{{"uri", name + ".bin"}, {"byteLength", decoded.size()}}};
        for (const char* field : {"extensionsUsed", "extensionsRequired"}) {
            auto& list = root[field];
            list.erase(std::remove(list.begin(), list.end(), Json("EXT_meshopt_compression")), list.end());
        }
        { std::ofstream file(fixture.directory / (name + ".bin"), std::ios::binary); file.write(reinterpret_cast<const char*>(decoded.data()), decoded.size()); }
        const auto referencePath = fixture.directory / (name + ".gltf");
        { std::ofstream file(referencePath); file << root.dump(); }
        Scene resident;
        ASSERT_TRUE(resident.load(referencePath)) << resident.lastLoadResult().error;
        auto assetPath = sourcePath; assetPath.replace_extension("meshstream.bin");
        MeshletStreamAsset asset; std::string reason;
        ASSERT_TRUE(asset.open(assetPath, reason)) << reason;
        for (uint32_t geometry = 0; geometry < asset.primitiveCount(); ++geometry) {
            const auto instance = std::find_if(asset.instances().begin(), asset.instances().end(),
                [&](const auto& i) { return i.primitiveIndex == geometry; });
            ASSERT_NE(instance, asset.instances().end());
            const auto& source = resident.renderPrimitives()[resident.renderNodes()[instance->renderNodeIndex].renderPrimitiveIndex];
            using Key = std::array<uint32_t, 12>;
            std::set<Key> vertices;
            for (size_t v = 0; v < source.positions.size(); ++v) {
                Key key{};
                std::memcpy(key.data(), &source.positions[v].x, 12);
                if (!source.normals.empty()) { std::memcpy(key.data() + 3, &source.normals[v].x, 12); }
                if (!source.texcoords0.empty()) { std::memcpy(key.data() + 6, &source.texcoords0[v].x, 8); }
                if (!source.tangents.empty()) { std::memcpy(key.data() + 8, &source.tangents[v].x, 16); }
                vertices.insert(key);
            }
            const auto& primitive = asset.primitives()[geometry];
            for (uint32_t page = primitive.pageOffset; page < primitive.pageOffset + primitive.pageCount; ++page) {
                std::vector<uint8_t> scratch; std::span<const uint8_t> bytes;
                ASSERT_TRUE(decodeMeshletStreamPayloadForDevice(asset.pages()[page], asset.pagePayload(page), scratch, bytes, reason)) << reason;
                MeshletStreamPayloadHeader h; std::memcpy(&h, bytes.data(), sizeof(h));
                for (uint32_t v = 0; v < h.vertexCount; ++v) {
                    Key key{}; std::memcpy(key.data(), bytes.data() + h.positionOffsetBytes + v * 12, 12);
                    if (h.attributeFlags & kMeshletStreamPayloadAttributeNormal) { std::memcpy(key.data()+3, bytes.data()+h.normalOffsetBytes+v*16, 12); }
                    if (h.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0) { std::memcpy(key.data()+6, bytes.data()+h.texcoord0OffsetBytes+v*8, 8); }
                    if (h.attributeFlags & kMeshletStreamPayloadAttributeTangent) { std::memcpy(key.data()+8, bytes.data()+h.tangentOffsetBytes+v*16, 16); }
                    ASSERT_TRUE(vertices.contains(key)) << "Resident and stream attribute tuple differ, page " << page << " vertex " << v;
                }
            }
        }
    }
}
} // namespace
