#include "Runtime/Scene/Scene.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Scene/SceneLoader.h"
#include "Runtime/Scene/ScenePicker.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Task/TaskSystem.h"
#include "meshoptimizer.h"

#include "json.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <winioctl.h>
#endif

namespace {

bool resizeSparseFile(const std::filesystem::path& path, uint64_t size)
{
#ifdef _WIN32
    HANDLE file = CreateFileW(
        path.wstring().c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD returnedBytes = 0;
    const BOOL sparse = DeviceIoControl(
        file,
        FSCTL_SET_SPARSE,
        nullptr,
        0,
        nullptr,
        0,
        &returnedBytes,
        nullptr);
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    const BOOL resized = sparse && SetFilePointerEx(file, end, nullptr, FILE_BEGIN) && SetEndOfFile(file);
    CloseHandle(file);
    return resized != FALSE;
#else
    std::error_code error;
    std::filesystem::resize_file(path, size, error);
    return !error;
#endif
}

void expect(bool condition, const std::string& message)
{
    EXPECT_TRUE(condition) << message;
}
bool nearlyEqual(float lhs, float rhs, float epsilon = 0.0001f)
{
    return std::fabs(lhs - rhs) <= epsilon;
}

void expectVec3(
    const float3& actual,
    const float3& expected,
    const std::string& label)
{
    expect(
        nearlyEqual(actual.x, expected.x) &&
            nearlyEqual(actual.y, expected.y) &&
            nearlyEqual(actual.z, expected.z),
        label + " expected [" + metallic::scene::formatVec3(expected) + "] got [" +
            metallic::scene::formatVec3(actual) + "]");
}

void expectVec4(
    const float4& actual,
    const float4& expected,
    const std::string& label)
{
    expect(
        nearlyEqual(actual.x, expected.x) &&
            nearlyEqual(actual.y, expected.y) &&
            nearlyEqual(actual.z, expected.z) &&
            nearlyEqual(actual.w, expected.w),
        label);
}

void expectCameraProperties(
    const metallic::scene::CameraProperties& actual,
    const metallic::scene::CameraProperties& expected,
    const std::string& label)
{
    EXPECT_EQ(actual.type, expected.type) << label << " type";
    EXPECT_NEAR(actual.yfov, expected.yfov, 0.000000001) << label << " yfov";
    EXPECT_NEAR(actual.aspectRatio, expected.aspectRatio, 0.000000001)
        << label << " aspectRatio";
    EXPECT_NEAR(actual.xmag, expected.xmag, 0.000000001) << label << " xmag";
    EXPECT_NEAR(actual.ymag, expected.ymag, 0.000000001) << label << " ymag";
    EXPECT_NEAR(actual.znear, expected.znear, 0.000000001) << label << " znear";
    EXPECT_NEAR(actual.zfar, expected.zfar, 0.000000001) << label << " zfar";
}

void expectLightProperties(
    const metallic::scene::LightProperties& actual,
    const metallic::scene::LightProperties& expected,
    const std::string& label)
{
    EXPECT_EQ(actual.type, expected.type) << label << " type";
    expectVec3(actual.color, expected.color, label + " color");
    EXPECT_NEAR(actual.intensity, expected.intensity, 0.000000001)
        << label << " intensity";
    EXPECT_NEAR(actual.range, expected.range, 0.000000001) << label << " range";
    EXPECT_NEAR(actual.innerConeAngle, expected.innerConeAngle, 0.000000001)
        << label << " innerConeAngle";
    EXPECT_NEAR(actual.outerConeAngle, expected.outerConeAngle, 0.000000001)
        << label << " outerConeAngle";
}

void expectCameraSnapshot(
    const metallic::scene::RenderCamera& actual,
    const metallic::scene::CameraProperties& expected,
    const std::string& label)
{
    EXPECT_EQ(actual.type, expected.type) << label << " type";
    EXPECT_NEAR(actual.yfov, expected.yfov, 0.000000001) << label << " yfov";
    EXPECT_NEAR(actual.aspectRatio, expected.aspectRatio, 0.000000001)
        << label << " aspectRatio";
    EXPECT_NEAR(actual.xmag, expected.xmag, 0.000000001) << label << " xmag";
    EXPECT_NEAR(actual.ymag, expected.ymag, 0.000000001) << label << " ymag";
    EXPECT_NEAR(actual.znear, expected.znear, 0.000000001) << label << " znear";
    EXPECT_NEAR(actual.zfar, expected.zfar, 0.000000001) << label << " zfar";
}

void expectLightSnapshot(
    const metallic::scene::RenderLight& actual,
    const metallic::scene::LightProperties& expected,
    const std::string& label)
{
    EXPECT_EQ(actual.type, expected.type) << label << " type";
    expectVec3(actual.color, expected.color, label + " color");
    EXPECT_NEAR(actual.intensity, expected.intensity, 0.000000001)
        << label << " intensity";
    EXPECT_NEAR(actual.range, expected.range, 0.000000001) << label << " range";
    EXPECT_NEAR(actual.innerConeAngle, expected.innerConeAngle, 0.000000001)
        << label << " innerConeAngle";
    EXPECT_NEAR(actual.outerConeAngle, expected.outerConeAngle, 0.000000001)
        << label << " outerConeAngle";
}

void expectTextureInfo(
    const metallic::scene::RenderTextureInfo& actual,
    int32_t expectedTextureIndex,
    int32_t expectedTexCoord,
    const std::array<float, 6>& expectedUvTransform,
    const std::string& label)
{
    expect(actual.textureIndex == expectedTextureIndex, label + " texture index");
    expect(actual.texCoord == expectedTexCoord, label + " texcoord");
    for (size_t index = 0; index < expectedUvTransform.size(); ++index) {
        expect(
            nearlyEqual(actual.uvTransform[index], expectedUvTransform[index]),
            label + " uvTransform[" + std::to_string(index) + "]");
    }
}

void writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream file(path, std::ios::binary);
    file << text;
}

void writeSceneBinary(const std::filesystem::path& path)
{
    constexpr std::array<float, 9> kPositions{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };
    constexpr std::array<uint16_t, 3> kIndices{0, 1, 2};

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(kPositions.data()), sizeof(float) * kPositions.size());
    file.write(reinterpret_cast<const char*>(kIndices.data()), sizeof(uint16_t) * kIndices.size());
}

void writeGeneratedTangentSceneBinary(const std::filesystem::path& path)
{
    constexpr std::array<float, 9> kPositions{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    constexpr std::array<float, 9> kNormals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    constexpr std::array<float, 6> kTexcoords{0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    constexpr std::array<uint16_t, 3> kIndices{0, 1, 2};

    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(kPositions.data()), sizeof(float) * kPositions.size());
    file.write(reinterpret_cast<const char*>(kNormals.data()), sizeof(float) * kNormals.size());
    file.write(reinterpret_cast<const char*>(kTexcoords.data()), sizeof(float) * kTexcoords.size());
    file.write(reinterpret_cast<const char*>(kIndices.data()), sizeof(uint16_t) * kIndices.size());
}

std::filesystem::path writeMeshoptCompressedScene(
    const std::filesystem::path& directory,
    size_t positionCompressionStride = sizeof(uint16_t) * 4u,
    uint32_t meshCount = 1u)
{
    constexpr std::array<uint16_t, 12> kPositions{
        0, 0, 0, 0,
        65535, 0, 0, 0,
        0, 65535, 0, 0,
    };
    constexpr std::array<float, 12> kNormalSources{
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
    };
    std::array<int8_t, 12> normals{};
    meshopt_encodeFilterOct(
        normals.data(),
        3,
        sizeof(int8_t) * 4u,
        8,
        kNormalSources.data());
    constexpr std::array<uint16_t, 6> kTexcoords{
        0, 0,
        65535, 0,
        0, 65535,
    };
    constexpr std::array<uint32_t, 3> kIndices{0, 1, 2};

    const auto encodeAttributes = [](const void* data, size_t count, size_t stride) {
        std::vector<uint8_t> encoded(meshopt_encodeVertexBufferBound(count, stride));
        const size_t encodedSize = meshopt_encodeVertexBufferLevel(
            encoded.data(),
            encoded.size(),
            data,
            count,
            stride,
            2,
            0);
        EXPECT_GT(encodedSize, 0u);
        encoded.resize(encodedSize);
        EXPECT_EQ(meshopt_decodeVertexVersion(encoded.data(), encoded.size()), 0);
        return encoded;
    };
    const auto encodeTriangles = [](const uint32_t* indices, size_t count, size_t vertexCount) {
        std::vector<uint8_t> encoded(meshopt_encodeIndexBufferBound(count, vertexCount));
        const size_t encodedSize = meshopt_encodeIndexBuffer(
            encoded.data(),
            encoded.size(),
            indices,
            count);
        EXPECT_GT(encodedSize, 0u);
        encoded.resize(encodedSize);
        EXPECT_EQ(meshopt_decodeIndexVersion(encoded.data(), encoded.size()), 1);
        return encoded;
    };

    const std::array<std::vector<uint8_t>, 4> encodedViews{
        encodeAttributes(kPositions.data(), 3, sizeof(uint16_t) * 4u),
        encodeAttributes(normals.data(), 3, sizeof(int8_t) * 4u),
        encodeAttributes(kTexcoords.data(), 3, sizeof(uint16_t) * 2u),
        encodeTriangles(kIndices.data(), kIndices.size(), 3),
    };
    std::array<size_t, 4> encodedOffsets{};
    std::vector<uint8_t> binary;
    for (size_t viewIndex = 0; viewIndex < encodedViews.size(); ++viewIndex) {
        encodedOffsets[viewIndex] = binary.size();
        binary.insert(binary.end(), encodedViews[viewIndex].begin(), encodedViews[viewIndex].end());
        binary.resize((binary.size() + 3u) & ~size_t{3});
    }

    const std::filesystem::path binaryPath = directory / "meshopt_compressed.bin";
    std::ofstream binaryFile(binaryPath, std::ios::binary);
    binaryFile.write(
        reinterpret_cast<const char*>(binary.data()),
        static_cast<std::streamsize>(binary.size()));

    constexpr std::array<size_t, 4> kDecodedByteSizes{
        sizeof(kPositions),
        sizeof(normals),
        sizeof(kTexcoords),
        sizeof(uint16_t) * kIndices.size(),
    };
    constexpr std::array<size_t, 4> kStrides{
        sizeof(uint16_t) * 4u,
        sizeof(int8_t) * 4u,
        sizeof(uint16_t) * 2u,
        sizeof(uint16_t),
    };
    std::array<size_t, 4> fallbackOffsets{};
    for (size_t viewIndex = 1; viewIndex < fallbackOffsets.size(); ++viewIndex) {
        fallbackOffsets[viewIndex] =
            fallbackOffsets[viewIndex - 1] + kDecodedByteSizes[viewIndex - 1];
    }
    const size_t decodedBufferSize =
        fallbackOffsets.back() + kDecodedByteSizes.back();

    const std::filesystem::path gltfPath = directory / "meshopt_compressed.gltf";
    std::ostringstream gltf;
    gltf << R"json({
  "asset": { "version": "2.0" },
  "extensionsUsed": ["EXT_meshopt_compression", "KHR_mesh_quantization"],
  "extensionsRequired": ["EXT_meshopt_compression", "KHR_mesh_quantization"],
  "buffers": [
    { "uri": "meshopt_compressed.bin", "byteLength": )json"
         << binary.size() << R"json( },
    { "byteLength": )json" << decodedBufferSize << R"json( }
  ],
  "bufferViews": [
)json";
    constexpr std::array<const char*, 4> kModes{
        "ATTRIBUTES",
        "ATTRIBUTES",
        "ATTRIBUTES",
        "TRIANGLES",
    };
    constexpr std::array<const char*, 4> kFilters{
        "NONE",
        "OCTAHEDRAL",
        "NONE",
        "NONE",
    };
    for (size_t viewIndex = 0; viewIndex < encodedViews.size(); ++viewIndex) {
        gltf << R"json(    { "buffer": 1, "byteOffset": )json"
             << fallbackOffsets[viewIndex] << R"json(, "byteLength": )json"
             << kDecodedByteSizes[viewIndex];
        if (viewIndex != 3) {
            gltf << R"json(, "byteStride": )json" << kStrides[viewIndex];
        }
        const size_t compressionStride = viewIndex == 0
            ? positionCompressionStride
            : kStrides[viewIndex];
        gltf << R"json(, "target": )json" << (viewIndex == 3 ? 34963 : 34962)
             << R"json(, "extensions": { "EXT_meshopt_compression": {
      "buffer": 0, "byteOffset": )json"
             << encodedOffsets[viewIndex] << R"json(, "byteLength": )json"
             << encodedViews[viewIndex].size() << R"json(, "byteStride": )json"
             << compressionStride << R"json(, "count": 3, "mode": ")json"
             << kModes[viewIndex] << R"json(", "filter": ")json"
             << kFilters[viewIndex] << R"json("
    } } })json";
        if (viewIndex + 1 != encodedViews.size()) {
            gltf << ',';
        }
        gltf << '\n';
    }
    gltf << R"json(  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5123, "normalized": true, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [65535, 65535, 0] },
    { "bufferView": 1, "componentType": 5120, "normalized": true, "count": 3, "type": "VEC3" },
    { "bufferView": 2, "componentType": 5123, "normalized": true, "count": 3, "type": "VEC2" },
    { "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
  ],
  "meshes": [
)json";
    meshCount = std::max(meshCount, 1u);
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ",\n";
        }
        gltf << R"json(    { "name": "Meshopt Triangle )json" << meshIndex << R"json(", "primitives": [
      { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3, "mode": 4 }
    ] })json";
    }
    gltf << R"json(
  ],
  "nodes": [
)json";
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ",\n";
        }
        gltf << R"json(    { "name": "Meshopt Triangle )json" << meshIndex
             << R"json(", "mesh": )json" << meshIndex << " }";
    }
    gltf << R"json(
  ],
  "scenes": [ { "name": "Meshopt Scene", "nodes": [)json";
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ", ";
        }
        gltf << meshIndex;
    }
    gltf << R"json(] } ],
  "scene": 0
})json";
    writeTextFile(gltfPath, gltf.str());
    return gltfPath;
}

std::filesystem::path writeMeshletLodGridScene(
    const std::filesystem::path& directory,
    uint32_t meshCount = 1,
    bool percentEncodedBufferUri = false)
{
    constexpr uint32_t kGridCells = 32;
    constexpr uint32_t kGridVertices = (kGridCells + 1) * (kGridCells + 1);
    constexpr uint32_t kGridTriangles = kGridCells * kGridCells * 2;

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> texcoords;
    std::vector<uint32_t> indices;
    positions.reserve(static_cast<size_t>(kGridVertices) * 3u);
    normals.reserve(static_cast<size_t>(kGridVertices) * 3u);
    texcoords.reserve(static_cast<size_t>(kGridVertices) * 2u);
    indices.reserve(static_cast<size_t>(kGridTriangles) * 3u);

    float3 minBounds(std::numeric_limits<float>::max());
    float3 maxBounds(-std::numeric_limits<float>::max());
    for (uint32_t y = 0; y <= kGridCells; ++y) {
        for (uint32_t x = 0; x <= kGridCells; ++x) {
            const float fx = static_cast<float>(x);
            const float fy = static_cast<float>(y);
            const float fz = std::sin(fx * 0.35f) * std::cos(fy * 0.27f) * 0.08f;
            positions.insert(positions.end(), {fx, fy, fz});
            normals.insert(normals.end(), {0.0f, 0.0f, 1.0f});
            texcoords.insert(
                texcoords.end(),
                {fx / static_cast<float>(kGridCells), fy / static_cast<float>(kGridCells)});
            minBounds.x = std::min(minBounds.x, fx);
            minBounds.y = std::min(minBounds.y, fy);
            minBounds.z = std::min(minBounds.z, fz);
            maxBounds.x = std::max(maxBounds.x, fx);
            maxBounds.y = std::max(maxBounds.y, fy);
            maxBounds.z = std::max(maxBounds.z, fz);
        }
    }

    const auto vertexIndex = [](uint32_t x, uint32_t y) {
        return y * (kGridCells + 1) + x;
    };
    for (uint32_t y = 0; y < kGridCells; ++y) {
        for (uint32_t x = 0; x < kGridCells; ++x) {
            const uint32_t i0 = vertexIndex(x, y);
            const uint32_t i1 = vertexIndex(x + 1, y);
            const uint32_t i2 = vertexIndex(x, y + 1);
            const uint32_t i3 = vertexIndex(x + 1, y + 1);
            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }

    const std::filesystem::path binaryPath = directory /
        (percentEncodedBufferUri ? "meshlet lod grid.bin" : "meshlet_lod_grid.bin");
    std::ofstream binary(binaryPath, std::ios::binary);
    binary.write(
        reinterpret_cast<const char*>(positions.data()),
        static_cast<std::streamsize>(positions.size() * sizeof(float)));
    binary.write(
        reinterpret_cast<const char*>(normals.data()),
        static_cast<std::streamsize>(normals.size() * sizeof(float)));
    binary.write(
        reinterpret_cast<const char*>(texcoords.data()),
        static_cast<std::streamsize>(texcoords.size() * sizeof(float)));
    binary.write(
        reinterpret_cast<const char*>(indices.data()),
        static_cast<std::streamsize>(indices.size() * sizeof(uint32_t)));

    const size_t positionBytes = positions.size() * sizeof(float);
    const size_t normalBytes = normals.size() * sizeof(float);
    const size_t texcoordBytes = texcoords.size() * sizeof(float);
    const size_t indexBytes = indices.size() * sizeof(uint32_t);
    const std::filesystem::path gltfPath = directory / "meshlet_lod_grid.gltf";
    std::ostringstream gltf;
    gltf << R"json({
  "asset": { "version": "2.0" },
  "buffers": [
    { "uri": ")json"
         << (percentEncodedBufferUri ? "meshlet%20lod%20grid.bin" : "meshlet_lod_grid.bin")
         << R"json(", "byteLength": )json"
         << (positionBytes + normalBytes + texcoordBytes + indexBytes) << R"json( }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": )json"
         << positionBytes << R"json(, "target": 34962 },
    { "buffer": 0, "byteOffset": )json"
         << positionBytes << R"json(, "byteLength": )json" << normalBytes << R"json(, "target": 34962 },
    { "buffer": 0, "byteOffset": )json"
         << (positionBytes + normalBytes) << R"json(, "byteLength": )json" << texcoordBytes << R"json(, "target": 34962 },
    { "buffer": 0, "byteOffset": )json"
         << (positionBytes + normalBytes + texcoordBytes) << R"json(, "byteLength": )json" << indexBytes << R"json(, "target": 34963 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": )json"
         << kGridVertices << R"json(, "type": "VEC3", "min": [)json"
         << minBounds.x << ", " << minBounds.y << ", " << minBounds.z << R"json(], "max": [)json"
         << maxBounds.x << ", " << maxBounds.y << ", " << maxBounds.z << R"json(] },
    { "bufferView": 1, "componentType": 5126, "count": )json"
         << kGridVertices << R"json(, "type": "VEC3" },
    { "bufferView": 2, "componentType": 5126, "count": )json"
         << kGridVertices << R"json(, "type": "VEC2" },
    { "bufferView": 3, "componentType": 5125, "count": )json"
         << indices.size() << R"json(, "type": "SCALAR" }
  ],
)json";
    meshCount = std::max(meshCount, 1u);
    gltf << R"json(  "meshes": [
)json";
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ",\n";
        }
        gltf << R"json(    { "name": "Meshlet LOD Grid )json" << meshIndex << R"json(", "primitives": [
      { "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 }, "indices": 3, "mode": 4 }
    ] })json";
    }
    gltf << R"json(
  ],
  "nodes": [
)json";
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ",\n";
        }
        gltf << R"json(    { "name": "Grid )json" << meshIndex << R"json(", "mesh": )json" << meshIndex;
        if (meshIndex != 0) {
            gltf << R"json(, "translation": [)json" << (static_cast<float>(meshIndex) * 40.0f) << R"json(, 0.0, 0.0])json";
        }
        gltf << " }";
    }
    gltf << R"json(
  ],
  "scenes": [ { "name": "Grid Scene", "nodes": [)json";
    for (uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        if (meshIndex != 0) {
            gltf << ", ";
        }
        gltf << meshIndex;
    }
    gltf << R"json(] } ],
  "scene": 0
})json";
    writeTextFile(gltfPath, gltf.str());
    return gltfPath;
}

void writeUint32(std::ofstream& file, uint32_t value)
{
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::filesystem::path prepareOutputDirectory()
{
    const std::filesystem::path outputDirectory = std::filesystem::current_path() / "scene-test-output";
    std::filesystem::create_directories(outputDirectory);
    return outputDirectory;
}

std::filesystem::path writeFullScene(const std::filesystem::path& directory)
{
    writeSceneBinary(directory / "scene.bin");
    const std::filesystem::path gltfPath = directory / "scene.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_diffuse_transmission"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "name": "Key Light",
          "type": "directional",
          "color": [1.0, 0.8, 0.6],
          "intensity": 3.0
        }
      ]
    }
  },
  "buffers": [
    { "uri": "scene.bin", "byteLength": 42 }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0.0, 0.0, 0.0],
      "max": [1.0, 1.0, 0.0]
    },
    {
      "bufferView": 1,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }
  ],
  "materials": [
    { "name": "Test Material" },
    {
      "name": "Tinted Material",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.25, 0.5, 0.75, 0.8],
        "metallicFactor": 0.35,
        "roughnessFactor": 0.6
      },
      "emissiveFactor": [0.05, 0.1, 0.2],
      "doubleSided": true,
      "extensions": {
        "KHR_materials_diffuse_transmission": {
          "diffuseTransmissionFactor": 0.45,
          "diffuseTransmissionColor": [0.4, 0.9, 0.55]
        }
      }
    }
  ],
  "meshes": [
    {
      "name": "Triangle Mesh",
      "primitives": [
        {
          "attributes": { "POSITION": 0 },
          "indices": 1,
          "material": 0,
          "mode": 4
        }
      ]
    }
  ],
  "cameras": [
    {
      "name": "Main Camera",
      "type": "perspective",
      "perspective": {
        "aspectRatio": 1.7777778,
        "yfov": 0.75,
        "znear": 0.1,
        "zfar": 100.0
      }
    },
    {
      "name": "Ortho Camera",
      "type": "orthographic",
      "orthographic": {
        "xmag": 4.0,
        "ymag": 3.0,
        "znear": 0.01,
        "zfar": 50.0
      }
    }
  ],
  "nodes": [
    {
      "name": "Root",
      "translation": [1.0, 2.0, 3.0],
      "children": [1, 2, 3, 4]
    },
    {
      "name": "Mesh Node",
      "matrix": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        4.0, 0.0, 0.0, 1.0
      ],
      "mesh": 0
    },
    {
      "name": "Perspective Node",
      "translation": [0.0, 0.0, 10.0],
      "camera": 0
    },
    {
      "name": "Orthographic Node",
      "translation": [2.0, 0.0, 0.0],
      "camera": 1
    },
    {
      "name": "Light Node",
      "extensions": {
        "KHR_lights_punctual": {
          "light": 0
        }
      }
    }
  ],
  "scenes": [
    { "name": "Default Scene", "nodes": [0] }
  ]
}
)json");
    return gltfPath;
}

std::filesystem::path writeEditablePropertiesScene(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = directory / "editable_properties.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "name": "Editable Spot Light",
          "type": "spot",
          "color": [1.0, 0.8, 0.6],
          "intensity": 3.0,
          "range": 12.0,
          "spot": {
            "innerConeAngle": 0.1,
            "outerConeAngle": 0.5
          }
        }
      ]
    }
  },
  "cameras": [
    {
      "name": "Editable Perspective Camera",
      "type": "perspective",
      "perspective": {
        "aspectRatio": 1.7777778,
        "yfov": 0.75,
        "znear": 0.1,
        "zfar": 100.0
      }
    },
    {
      "name": "Editable Orthographic Camera",
      "type": "orthographic",
      "orthographic": {
        "xmag": 4.0,
        "ymag": 3.0,
        "znear": 0.01,
        "zfar": 50.0
      }
    }
  ],
  "nodes": [
    { "name": "Perspective Properties Node", "camera": 0 },
    { "name": "Orthographic Properties Node", "camera": 1 },
    {
      "name": "Spot Light Properties Node",
      "extensions": {
        "KHR_lights_punctual": { "light": 0 }
      }
    }
  ],
  "scenes": [
    { "name": "Editable Properties Scene", "nodes": [0, 1, 2] }
  ]
}
)json");
    return gltfPath;
}

std::filesystem::path writeGeneratedTangentScene(const std::filesystem::path& directory)
{
    writeGeneratedTangentSceneBinary(directory / "generated_tangent.bin");
    const std::filesystem::path gltfPath = directory / "generated_tangent.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "buffers": [
    { "uri": "generated_tangent.bin", "byteLength": 102 }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 72, "byteLength": 24, "target": 34962 },
    { "buffer": 0, "byteOffset": 96, "byteLength": 6, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0.0, 0.0, 0.0],
      "max": [1.0, 1.0, 0.0]
    },
    {
      "bufferView": 1,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3"
    },
    {
      "bufferView": 2,
      "componentType": 5126,
      "count": 3,
      "type": "VEC2"
    },
    {
      "bufferView": 3,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }
  ],
  "meshes": [
    {
      "name": "Generated Tangent Mesh",
      "primitives": [
        {
          "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
          "indices": 3,
          "mode": 4
        }
      ]
    }
  ],
  "nodes": [
    { "name": "Mesh Node", "mesh": 0 }
  ],
  "scenes": [
    { "name": "Generated Tangent Scene", "nodes": [0] }
  ]
}
)json");
    return gltfPath;
}

std::filesystem::path writeMinimalGlbScene(const std::filesystem::path& directory)
{
    std::string json = R"json({"asset":{"version":"2.0"},"scene":0,"nodes":[{"name":"GLB Node"}],"scenes":[{"name":"GLB Scene","nodes":[0]}]})json";
    while ((json.size() % 4) != 0) {
        json.push_back(' ');
    }

    const std::filesystem::path glbPath = directory / "minimal.glb";
    std::ofstream file(glbPath, std::ios::binary);

    constexpr uint32_t kGlbMagic = 0x46546C67;
    constexpr uint32_t kGlbVersion = 2;
    constexpr uint32_t kJsonChunkType = 0x4E4F534A;
    const uint32_t jsonByteLength = static_cast<uint32_t>(json.size());
    const uint32_t glbLength = 12 + 8 + jsonByteLength;

    writeUint32(file, kGlbMagic);
    writeUint32(file, kGlbVersion);
    writeUint32(file, glbLength);
    writeUint32(file, jsonByteLength);
    writeUint32(file, kJsonChunkType);
    file.write(json.data(), json.size());
    return glbPath;
}

std::filesystem::path writeFallbackScene(const std::filesystem::path& directory)
{
    writeSceneBinary(directory / "fallback.bin");
    const std::filesystem::path gltfPath = directory / "fallback.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "buffers": [
    { "uri": "fallback.bin", "byteLength": 42 }
  ],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0.0, 0.0, 0.0],
      "max": [1.0, 1.0, 0.0]
    },
    {
      "bufferView": 1,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }
  ],
  "meshes": [
    {
      "name": "Fallback Mesh",
      "primitives": [
        {
          "attributes": { "POSITION": 0 },
          "indices": 1,
          "mode": 4
        }
      ]
    }
  ],
  "nodes": [
    { "name": "Fallback Mesh Node", "mesh": 0 }
  ],
  "scenes": [
    { "name": "Implicit Default Scene", "nodes": [0] }
  ]
}
)json");
    return gltfPath;
}

std::filesystem::path writeUnsupportedRequiredExtensionScene(
    const std::filesystem::path& directory,
    const std::string& extension)
{
    const std::filesystem::path gltfPath = directory / "unsupported_required.gltf";
    writeTextFile(
        gltfPath,
        std::string(R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsRequired": [")json") + extension + R"json("],
  "extensionsUsed": [")json" + extension + R"json("],
  "nodes": [
    { "name": "Node" }
  ],
  "scenes": [
    { "name": "Scene", "nodes": [0] }
  ]
}
)json");
    return gltfPath;
}

std::filesystem::path writeMaterialFeatureScene(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = directory / "materials.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsRequired": [
    "KHR_materials_emissive_strength"
  ],
  "extensionsUsed": [
    "KHR_texture_transform",
    "KHR_materials_emissive_strength",
    "KHR_materials_transmission",
    "KHR_materials_ior",
    "KHR_materials_volume",
    "KHR_materials_diffuse_transmission"
  ],
  "samplers": [
    { "magFilter": 9729, "minFilter": 9729 },
    { "magFilter": 9728, "minFilter": 9728 }
  ],
  "images": [
    { "name": "Base Color Image", "uri": "base_color.png" },
    { "name": "Metallic Roughness Image", "uri": "metallic_roughness.png" },
    { "name": "Normal Image", "uri": "normal.png" },
    { "name": "Occlusion Image", "uri": "occlusion.png" },
    { "name": "Emissive Image", "uri": "emissive.png" },
    { "name": "Transmission Image", "uri": "transmission.png" },
    { "name": "Thickness Image", "uri": "thickness.png" },
    { "name": "Diffuse Transmission Image", "uri": "diffuse_transmission.png" },
    { "name": "Diffuse Transmission Color Image", "uri": "diffuse_transmission_color.png" }
  ],
  "textures": [
    { "name": "Base Color Texture", "sampler": 0, "source": 0 },
    { "name": "Metallic Roughness Texture", "sampler": 0, "source": 1 },
    { "name": "Normal Texture", "sampler": 1, "source": 2 },
    { "name": "Occlusion Texture", "sampler": 1, "source": 3 },
    { "name": "Emissive Texture", "sampler": 0, "source": 4 },
    { "name": "Transmission Texture", "sampler": 0, "source": 5 },
    { "name": "Thickness Texture", "sampler": 1, "source": 6 },
    { "name": "Diffuse Transmission Texture", "sampler": 0, "source": 7 },
    { "name": "Diffuse Transmission Color Texture", "sampler": 1, "source": 8 }
  ],
  "materials": [
    {
      "name": "Default Material"
    },
    {
      "name": "Full Material",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.2, 0.3, 0.4, 0.5],
        "metallicFactor": 0.25,
        "roughnessFactor": 0.75,
        "baseColorTexture": {
          "index": 0,
          "texCoord": 1,
          "extensions": {
            "KHR_texture_transform": {
              "offset": [0.1, 0.2],
              "scale": [2.0, 3.0],
              "rotation": 1.5707963267948966,
              "texCoord": 2
            }
          }
        },
        "metallicRoughnessTexture": { "index": 1, "texCoord": 0 }
      },
      "normalTexture": {
        "index": 2,
        "texCoord": 0,
        "scale": 0.42,
        "extensions": {
          "KHR_texture_transform": {
            "offset": [0.05, 0.15],
            "scale": [0.5, 0.25]
          }
        }
      },
      "occlusionTexture": {
        "index": 3,
        "texCoord": 1,
        "strength": 0.66,
        "extensions": {
          "KHR_texture_transform": {
            "offset": [0.3, 0.4],
            "scale": [1.25, 0.75],
            "texCoord": 3
          }
        }
      },
      "emissiveTexture": { "index": 4, "texCoord": 0 },
      "emissiveFactor": [0.1, 0.2, 0.3],
      "alphaMode": "MASK",
      "alphaCutoff": 0.37,
      "doubleSided": true,
      "extensions": {
        "KHR_materials_emissive_strength": { "emissiveStrength": 4.0 },
        "KHR_materials_transmission": {
          "transmissionFactor": 0.8,
          "transmissionTexture": {
            "index": 5,
            "texCoord": 1,
            "extensions": {
              "KHR_texture_transform": {
                "offset": [0.25, 0.75],
                "scale": [0.5, 0.25],
                "texCoord": 0
              }
            }
          }
        },
        "KHR_materials_ior": { "ior": 2.2 },
        "KHR_materials_volume": {
          "thicknessFactor": 0.6,
          "attenuationDistance": 12.5,
          "attenuationColor": [0.7, 0.8, 0.9],
          "thicknessTexture": { "index": 6, "texCoord": 0 }
        },
        "KHR_materials_diffuse_transmission": {
          "diffuseTransmissionFactor": 0.55,
          "diffuseTransmissionColor": [0.4, 0.5, 0.6],
          "diffuseTransmissionTexture": { "index": 7, "texCoord": 0 },
          "diffuseTransmissionColorTexture": { "index": 8, "texCoord": 1 }
        }
      }
    },
    {
      "name": "Low Clamp Material",
      "extensions": {
        "KHR_materials_transmission": { "transmissionFactor": -0.5 },
        "KHR_materials_ior": { "ior": 0.5 },
        "KHR_materials_volume": {
          "thicknessFactor": -1.0,
          "attenuationDistance": -2.0
        },
        "KHR_materials_diffuse_transmission": {
          "diffuseTransmissionFactor": -0.25
        }
      }
    },
    {
      "name": "High Clamp Material",
      "extensions": {
        "KHR_materials_transmission": { "transmissionFactor": 1.5 },
        "KHR_materials_ior": { "ior": 4.5 },
        "KHR_materials_diffuse_transmission": {
          "diffuseTransmissionFactor": 1.25
        }
      }
    }
  ],
  "nodes": [
    { "name": "Material Test Node" }
  ],
  "scenes": [
    { "name": "Material Test Scene", "nodes": [0] }
  ]
}
)json");
    return gltfPath;
}

void testFullSceneImport(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeFullScene(directory);
    expect(scene.load(gltfPath), scene.lastLoadResult().error);

    expect(scene.valid(), "loaded scene should be valid");
    ASSERT_EQ(scene.sources().size(), 1u);
    EXPECT_EQ(scene.sources().front().id, "main");
    EXPECT_EQ(scene.sources().front().path, gltfPath);
    EXPECT_TRUE(scene.sources().front().enabled);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.sources().front().mountMatrix,
        float4x4::Identity()));
    expect(scene.sceneIndex() == 0, "default scene index should be 0");
    expect(scene.sceneName() == "Default Scene", "scene name should come from glTF scene");
    expect(scene.assetInfo().version == "2.0", "asset version should be preserved");
    expect(scene.rootNodeIndices().size() == 1, "scene should expose one root node");
    expect(scene.rootNodeIndices().front() == 0, "root node index should be 0");
    expect(scene.nodes().size() == 5, "scene should load all nodes");
    expect(scene.nodes()[1].parent == 0, "child parent index should be assigned");
    expect(scene.nodes()[0].children.size() == 4, "root children should be preserved");

    const metallic::scene::SceneGraph& sceneGraph = scene.sceneGraph();
    expect(sceneGraph.size() == 5, "scene graph should own one entity per source node");
    expect(sceneGraph.sourceNodeCount() == 5, "scene graph source mapping count");
    expect(sceneGraph.roots().size() == 1, "scene graph active root count");
    const metallic::scene::ConstSceneObject rootObject = scene.objectForNode(0);
    const metallic::scene::ConstSceneObject meshObject = scene.objectForNode(1);
    EXPECT_EQ(scene.objectForSourceNode("main", 0), rootObject);
    EXPECT_EQ(scene.objectForSourceNode("main", 1), meshObject);
    EXPECT_FALSE(scene.objectForSourceNode("unknown", 0));
    expect(static_cast<bool>(rootObject), "root scene object should resolve from its stable node index");
    expect(static_cast<bool>(meshObject), "mesh scene object should resolve from its stable node index");
    expect(rootObject.hasComponent<metallic::scene::RootComponent>(), "root component");
    expect(rootObject.hasComponent<metallic::scene::ActiveSceneComponent>(), "active root component");
    const metallic::scene::SourceNodeComponent& rootSource =
        rootObject.getComponent<metallic::scene::SourceNodeComponent>();
    EXPECT_EQ(rootSource.nodeIndex, 0);
    EXPECT_EQ(rootSource.sourceId, "main");
    EXPECT_EQ(rootSource.sourceNodeIndex, 0);
    expect(
        rootObject.getComponent<metallic::scene::RelationshipComponent>().children.size() == 4,
        "relationship component children");
    expect(meshObject.hasComponent<metallic::scene::MeshComponent>(), "mesh component");
    expect(
        meshObject.getComponent<metallic::scene::MeshComponent>().renderNodeIndices ==
            std::vector<int32_t>{0},
        "mesh component stable render-node mapping");
    expect(
        scene.renderNodes().front().object == meshObject.entity(),
        "render node should retain its source scene object");

    const float3 meshWorldTranslation(
        scene.nodes()[1].worldMatrix.a03,
        scene.nodes()[1].worldMatrix.a13,
        scene.nodes()[1].worldMatrix.a23);
    expectVec3(meshWorldTranslation, float3(5.0f, 2.0f, 3.0f), "mesh node world translation");

    const metallic::scene::SceneStats& stats = scene.stats();
    expect(stats.meshCount == 1, "mesh count");
    expect(stats.materialCount == 2, "material count");
    expect(stats.textureCount == 0, "texture count");
    expect(stats.imageCount == 0, "image count");
    expect(stats.primitiveCount == 1, "primitive count");
    expect(stats.renderNodeCount == 1, "render node count");
    expect(stats.triangleCount == 1, "triangle count");
    expect(stats.meshletClusterCount == 1, "meshlet cluster count");
    expect(stats.meshletVertexReferenceCount == 3, "meshlet vertex reference count");
    expect(stats.meshletTriangleIndexCount == 3, "meshlet triangle index count");
    expect(stats.meshletLodLevelCount == 1, "meshlet lod level count");
    expect(stats.meshletLodGroupCount == 1, "meshlet lod group count");
    expect(stats.meshletLodClusterCount == 1, "meshlet lod cluster count");
    expect(stats.meshletLodVertexReferenceCount == 3, "meshlet lod vertex reference count");
    expect(stats.meshletLodTriangleIndexCount == 3, "meshlet lod triangle index count");

    expect(scene.meshes().size() == 1, "mesh info vector size");
    expect(scene.meshes()[0].name == "Triangle Mesh", "mesh info name");
    expect(scene.meshes()[0].primitiveCount == 1, "mesh info primitive count");

    expect(scene.renderPrimitives().size() == 1, "primitive vector size");
    const metallic::scene::RenderPrimitive& primitive = scene.renderPrimitives().front();
    expect(primitive.vertexCount == 3, "primitive vertex count");
    expect(primitive.indexCount == 3, "primitive index count");
    expect(primitive.materialIndex == 0, "primitive material index");
    expect(primitive.positions.size() == 3, "primitive position data count");
    expect(primitive.indices.size() == 3, "primitive index data count");
    expectVec3(primitive.positions[2], float3(1.0f, 1.0f, 0.0f), "primitive position data");
    expect(primitive.indices[2] == 2, "primitive index data");
    expect(primitive.meshletClusters.size() == 1, "primitive meshlet cluster count");
    expect(primitive.meshletVertices.size() == 3, "primitive meshlet vertex reference count");
    expect(primitive.meshletTriangles.size() == 3, "primitive meshlet triangle index count");
    const metallic::scene::MeshletCluster& meshletCluster = primitive.meshletClusters.front();
    expect(meshletCluster.vertexOffset == 0, "meshlet vertex offset");
    expect(meshletCluster.vertexCount == 3, "meshlet vertex count");
    expect(meshletCluster.triangleOffset == 0, "meshlet triangle offset");
    expect(meshletCluster.triangleCount == 1, "meshlet triangle count");
    expect(meshletCluster.bounds.valid, "meshlet bounds valid");
    expectVec3(meshletCluster.bounds.min, float3(0.0f, 0.0f, 0.0f), "meshlet bounds min");
    expectVec3(meshletCluster.bounds.max, float3(1.0f, 1.0f, 0.0f), "meshlet bounds max");
    expect(meshletCluster.boundingSphereRadius > 0.0f, "meshlet sphere radius");
    for (const uint32_t vertexIndex : primitive.meshletVertices) {
        expect(vertexIndex < primitive.positions.size(), "meshlet vertex reference range");
    }
    for (const uint8_t localVertexIndex : primitive.meshletTriangles) {
        expect(localVertexIndex < meshletCluster.vertexCount, "meshlet triangle local index range");
    }
    expect(primitive.meshletLodLevels.size() == 1, "primitive meshlet lod level count");
    expect(primitive.meshletLodGroups.size() == 1, "primitive meshlet lod group count");
    expect(primitive.meshletLodClusters.size() == 1, "primitive meshlet lod cluster count");
    expect(primitive.meshletLodVertices.size() == 3, "primitive meshlet lod vertex reference count");
    expect(primitive.meshletLodTriangles.size() == 3, "primitive meshlet lod triangle index count");
    const metallic::scene::MeshletLodLevel& meshletLodLevel = primitive.meshletLodLevels.front();
    expect(meshletLodLevel.groupOffset == 0, "meshlet lod level group offset");
    expect(meshletLodLevel.groupCount == 1, "meshlet lod level group count");
    expect(meshletLodLevel.clusterOffset == 0, "meshlet lod level cluster offset");
    expect(meshletLodLevel.clusterCount == 1, "meshlet lod level cluster count");
    const metallic::scene::MeshletLodGroup& meshletLodGroup = primitive.meshletLodGroups.front();
    expect(meshletLodGroup.clusterOffset == 0, "meshlet lod group cluster offset");
    expect(meshletLodGroup.clusterCount == 1, "meshlet lod group cluster count");
    expect(meshletLodGroup.lodLevel == 0, "meshlet lod group level");
    expect(meshletLodGroup.bounds.valid, "meshlet lod group bounds valid");
    const metallic::scene::MeshletCluster& meshletLodCluster = primitive.meshletLodClusters.front();
    expect(meshletLodCluster.lodLevel == 0, "meshlet lod cluster level");
    expect(meshletLodCluster.lodGroupIndex == 0, "meshlet lod cluster group index");
    expect(meshletLodCluster.lodGroupChildIndex == 0, "meshlet lod cluster group child index");
    expect(meshletLodCluster.refinedGroupIndex == metallic::scene::kInvalidSceneIndex, "meshlet lod cluster refined group");
    expect(meshletLodCluster.vertexCount == 3, "meshlet lod cluster vertex count");
    expect(meshletLodCluster.triangleCount == 1, "meshlet lod cluster triangle count");
    for (const uint32_t vertexIndex : primitive.meshletLodVertices) {
        expect(vertexIndex < primitive.positions.size(), "meshlet lod vertex reference range");
    }
    for (const uint8_t localVertexIndex : primitive.meshletLodTriangles) {
        expect(localVertexIndex < meshletLodCluster.vertexCount, "meshlet lod triangle local index range");
    }

    expect(scene.materials().size() == 2, "material vector size");
    expect(scene.materials()[0].name == "Test Material", "default material name");
    expectVec4(
        scene.materials()[0].baseColorFactor,
        float4(1.0f, 1.0f, 1.0f, 1.0f),
        "default baseColorFactor");
    expect(scene.materials()[1].name == "Tinted Material", "tinted material name");
    expectVec4(
        scene.materials()[1].baseColorFactor,
        float4(0.25f, 0.5f, 0.75f, 0.8f),
        "explicit baseColorFactor");
    expect(std::abs(scene.materials()[1].metallicFactor - 0.35f) < 0.0001f, "explicit metallicFactor");
    expect(std::abs(scene.materials()[1].roughnessFactor - 0.6f) < 0.0001f, "explicit roughnessFactor");
    expectVec3(
        scene.materials()[1].emissiveFactor,
        float3(0.05f, 0.1f, 0.2f),
        "explicit emissiveFactor");
    expect(scene.materials()[1].doubleSided, "explicit doubleSided");
    expect(
        std::abs(scene.materials()[1].diffuseTransmissionFactor - 0.45f) < 0.0001f,
        "explicit diffuseTransmissionFactor");
    expectVec3(
        scene.materials()[1].diffuseTransmissionColor,
        float3(0.4f, 0.9f, 0.55f),
        "explicit diffuseTransmissionColor");

    expect(scene.bounds().valid, "bounds should be valid");
    expectVec3(scene.bounds().min, float3(5.0f, 2.0f, 3.0f), "scene bounds min");
    expectVec3(scene.bounds().max, float3(6.0f, 3.0f, 3.0f), "scene bounds max");

    expect(scene.cameras().size() == 2, "camera count");
    const metallic::scene::RenderCamera& perspectiveCamera = scene.cameras()[0];
    expect(
        perspectiveCamera.object == scene.objectForNode(2).entity(),
        "perspective camera object mapping");
    expect(
        perspectiveCamera.type == metallic::scene::CameraType::Perspective,
        "first camera should be perspective");
    expect(nearlyEqual(static_cast<float>(perspectiveCamera.yfov), 0.75f), "perspective yfov");
    expect(nearlyEqual(static_cast<float>(perspectiveCamera.aspectRatio), 1.7777778f), "perspective aspect");
    expectVec3(perspectiveCamera.eye, float3(1.0f, 2.0f, 13.0f), "perspective camera eye");
    expectVec3(perspectiveCamera.center, float3(1.0f, 2.0f, 12.0f), "perspective camera center");

    const metallic::scene::RenderCamera& orthoCamera = scene.cameras()[1];
    expect(
        orthoCamera.object == scene.objectForNode(3).entity(),
        "orthographic camera object mapping");
    expect(
        orthoCamera.type == metallic::scene::CameraType::Orthographic,
        "second camera should be orthographic");
    expect(nearlyEqual(static_cast<float>(orthoCamera.xmag), 4.0f), "orthographic xmag");
    expect(nearlyEqual(static_cast<float>(orthoCamera.ymag), 3.0f), "orthographic ymag");

    expect(scene.lights().size() == 1, "punctual light count");
    expect(scene.lights().front().type == "directional", "punctual light type");
    expect(
        scene.lights().front().object == scene.objectForNode(4).entity(),
        "light object mapping");
}

void testCompositeSceneImport(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "composition";
    const std::filesystem::path sourceADirectory = directory / "source_a";
    const std::filesystem::path sourceBDirectory = directory / "source_b";
    std::filesystem::create_directories(sourceADirectory);
    std::filesystem::create_directories(sourceBDirectory);

    const std::filesystem::path sourceAPath = writeFullScene(sourceADirectory);
    const std::filesystem::path sourceBPath = writeFullScene(sourceBDirectory);
    nlohmann::json sourceBJson;
    {
        std::ifstream stream(sourceBPath, std::ios::binary);
        ASSERT_TRUE(stream.good());
        stream >> sourceBJson;
    }
    sourceBJson["scenes"][0]["name"] = "Source B Scene";
    sourceBJson["nodes"][0]["name"] = "Source B Root";
    sourceBJson["nodes"][1]["name"] = "Source B Mesh Node";
    sourceBJson["meshes"][0]["name"] = "Source B Triangle Mesh";
    sourceBJson["materials"][0]["name"] = "Source B Material";
    writeTextFile(sourceBPath, sourceBJson.dump(2));

    float4x4 sourceBMount = float4x4::Identity();
    sourceBMount.SetupByTranslation(float3(20.0f, 0.0f, 0.0f));
    std::vector<metallic::scene::SceneSourceDesc> sources{
        metallic::scene::SceneSourceDesc{
            .id = "source-a",
            .path = sourceAPath,
            .mountMatrix = float4x4::Identity(),
            .enabled = true,
        },
        metallic::scene::SceneSourceDesc{
            .id = "source-b",
            .path = sourceBPath,
            .mountMatrix = sourceBMount,
            .enabled = true,
        },
    };

    metallic::scene::Scene scene;
    const uint64_t lifetimeBeforeCompose = scene.sceneGraph().lifetimeRevision();
    std::string error;
    ASSERT_TRUE(scene.compose(sources, error)) << error;
    EXPECT_TRUE(scene.valid());
    ASSERT_EQ(scene.sources().size(), 2u);
    EXPECT_EQ(scene.sources()[0].id, "source-a");
    EXPECT_EQ(scene.sources()[1].id, "source-b");
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.sources()[1].mountMatrix,
        sourceBMount));
    EXPECT_TRUE(scene.sources()[0].enabled);
    EXPECT_TRUE(scene.sources()[1].enabled);
    EXPECT_NE(scene.sceneGraph().lifetimeRevision(), lifetimeBeforeCompose);

    const metallic::scene::ConstSceneObject sourceARoot =
        scene.objectForSourceNode("source-a", 0);
    const metallic::scene::ConstSceneObject sourceBRoot =
        scene.objectForSourceNode("source-b", 0);
    const metallic::scene::ConstSceneObject sourceAMesh =
        scene.objectForSourceNode("source-a", 1);
    const metallic::scene::ConstSceneObject sourceBMesh =
        scene.objectForSourceNode("source-b", 1);
    ASSERT_TRUE(sourceARoot);
    ASSERT_TRUE(sourceBRoot);
    ASSERT_TRUE(sourceAMesh);
    ASSERT_TRUE(sourceBMesh);
    EXPECT_NE(sourceARoot.entity(), sourceBRoot.entity());
    EXPECT_NE(sourceAMesh.entity(), sourceBMesh.entity());
    const metallic::scene::SourceNodeComponent& sourceARootIdentity =
        sourceARoot.getComponent<metallic::scene::SourceNodeComponent>();
    const metallic::scene::SourceNodeComponent& sourceBRootIdentity =
        sourceBRoot.getComponent<metallic::scene::SourceNodeComponent>();
    EXPECT_EQ(sourceARootIdentity.nodeIndex, 0);
    EXPECT_EQ(sourceARootIdentity.sourceId, "source-a");
    EXPECT_EQ(sourceARootIdentity.sourceNodeIndex, 0);
    EXPECT_EQ(sourceBRootIdentity.nodeIndex, 5);
    EXPECT_EQ(sourceBRootIdentity.sourceId, "source-b");
    EXPECT_EQ(sourceBRootIdentity.sourceNodeIndex, 0);
    EXPECT_EQ(
        sourceARoot.getComponent<metallic::scene::TagComponent>().name,
        "Root");
    EXPECT_EQ(
        sourceBRoot.getComponent<metallic::scene::TagComponent>().name,
        "Source B Root");

    ASSERT_EQ(scene.nodes().size(), 10u);
    ASSERT_EQ(scene.meshes().size(), 2u);
    ASSERT_EQ(scene.renderPrimitives().size(), 2u);
    ASSERT_EQ(scene.renderNodes().size(), 2u);
    ASSERT_EQ(scene.materials().size(), 4u);
    ASSERT_EQ(scene.cameras().size(), 4u);
    ASSERT_EQ(scene.lights().size(), 2u);
    ASSERT_EQ(scene.rootNodeIndices(), std::vector<int32_t>({0, 5}));
    EXPECT_EQ(scene.nodes()[0].name, "Root");
    EXPECT_EQ(scene.nodes()[5].name, "Source B Root");
    EXPECT_EQ(scene.meshes()[0].name, "Triangle Mesh");
    EXPECT_EQ(scene.meshes()[1].name, "Source B Triangle Mesh");
    EXPECT_EQ(scene.materials()[0].name, "Test Material");
    EXPECT_EQ(scene.materials()[2].name, "Source B Material");

    const metallic::scene::MeshComponent& sourceAMeshComponent =
        sourceAMesh.getComponent<metallic::scene::MeshComponent>();
    const metallic::scene::MeshComponent& sourceBMeshComponent =
        sourceBMesh.getComponent<metallic::scene::MeshComponent>();
    EXPECT_EQ(sourceAMeshComponent.meshIndex, 0);
    EXPECT_EQ(sourceBMeshComponent.meshIndex, 1);
    ASSERT_EQ(sourceAMeshComponent.renderNodeIndices, std::vector<int32_t>{0});
    ASSERT_EQ(sourceBMeshComponent.renderNodeIndices, std::vector<int32_t>{1});
    const metallic::scene::RenderNode& sourceARenderNode = scene.renderNodes()[0];
    const metallic::scene::RenderNode& sourceBRenderNode = scene.renderNodes()[1];
    EXPECT_EQ(sourceARenderNode.object, sourceAMesh.entity());
    EXPECT_EQ(sourceBRenderNode.object, sourceBMesh.entity());
    EXPECT_EQ(sourceARenderNode.nodeIndex, 1);
    EXPECT_EQ(sourceBRenderNode.nodeIndex, 6);
    EXPECT_EQ(sourceARenderNode.renderPrimitiveIndex, 0);
    EXPECT_EQ(sourceBRenderNode.renderPrimitiveIndex, 1);
    EXPECT_EQ(sourceARenderNode.materialIndex, 0);
    EXPECT_EQ(sourceBRenderNode.materialIndex, 2);
    EXPECT_EQ(scene.renderPrimitives()[0].meshIndex, 0);
    EXPECT_EQ(scene.renderPrimitives()[1].meshIndex, 1);
    EXPECT_EQ(scene.renderPrimitives()[0].materialIndex, 0);
    EXPECT_EQ(scene.renderPrimitives()[1].materialIndex, 2);

    const metallic::scene::CameraComponent& sourceACamera =
        scene.objectForSourceNode("source-a", 2)
            .getComponent<metallic::scene::CameraComponent>();
    const metallic::scene::CameraComponent& sourceBCamera =
        scene.objectForSourceNode("source-b", 2)
            .getComponent<metallic::scene::CameraComponent>();
    const metallic::scene::LightComponent& sourceALight =
        scene.objectForSourceNode("source-a", 4)
            .getComponent<metallic::scene::LightComponent>();
    const metallic::scene::LightComponent& sourceBLight =
        scene.objectForSourceNode("source-b", 4)
            .getComponent<metallic::scene::LightComponent>();
    EXPECT_EQ(sourceACamera.cameraIndex, 0);
    EXPECT_EQ(sourceACamera.renderCameraIndex, 0);
    EXPECT_EQ(sourceBCamera.cameraIndex, 2);
    EXPECT_EQ(sourceBCamera.renderCameraIndex, 2);
    EXPECT_EQ(sourceALight.lightIndex, 0);
    EXPECT_EQ(sourceALight.renderLightIndex, 0);
    EXPECT_EQ(sourceBLight.lightIndex, 1);
    EXPECT_EQ(sourceBLight.renderLightIndex, 1);

    EXPECT_EQ(scene.stats().meshCount, 2u);
    EXPECT_EQ(scene.stats().materialCount, 4u);
    EXPECT_EQ(scene.stats().primitiveCount, 2u);
    EXPECT_EQ(scene.stats().renderNodeCount, 2u);
    EXPECT_EQ(scene.stats().triangleCount, 2u);
    ASSERT_TRUE(scene.bounds().valid);
    expectVec3(scene.bounds().min, float3(5.0f, 2.0f, 3.0f), "composite bounds min");
    expectVec3(scene.bounds().max, float3(26.0f, 3.0f, 3.0f), "composite bounds max");
    expectVec3(
        float3(
            sourceARenderNode.worldMatrix.a03,
            sourceARenderNode.worldMatrix.a13,
            sourceARenderNode.worldMatrix.a23),
        float3(5.0f, 2.0f, 3.0f),
        "source A render transform");
    expectVec3(
        float3(
            sourceBRenderNode.worldMatrix.a03,
            sourceBRenderNode.worldMatrix.a13,
            sourceBRenderNode.worldMatrix.a23),
        float3(25.0f, 2.0f, 3.0f),
        "source B mounted render transform");

    const uint64_t structuralRevision = scene.sceneGraph().structuralRevision();
    const uint64_t lifetimeRevision = scene.sceneGraph().lifetimeRevision();
    const uint64_t contentRevision = scene.contentRevision();
    const uint64_t transformRevision = scene.transformRevision();
    const uint64_t visibilityRevision = scene.visibilityRevision();
    const float4x4 sourceAWorldBeforeEdit = scene.renderNodes()[0].worldMatrix;
    const float4x4 sourceBWorldBeforeEdit = scene.renderNodes()[1].worldMatrix;
    float4x4 editedSourceBMount = sourceBMount;
    editedSourceBMount.a13 += 4.0f;
    ASSERT_TRUE(scene.setSourceMountMatrix("source-b", editedSourceBMount));
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.sources()[1].mountMatrix,
        editedSourceBMount));
    EXPECT_GT(scene.transformRevision(), transformRevision);
    EXPECT_EQ(scene.visibilityRevision(), visibilityRevision);
    EXPECT_EQ(scene.contentRevision(), contentRevision);
    EXPECT_EQ(scene.sceneGraph().structuralRevision(), structuralRevision);
    EXPECT_EQ(scene.sceneGraph().lifetimeRevision(), lifetimeRevision);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.renderNodes()[0].worldMatrix,
        sourceAWorldBeforeEdit));
    EXPECT_NEAR(
        scene.renderNodes()[1].worldMatrix.a13,
        sourceBWorldBeforeEdit.a13 + 4.0f,
        0.0001f);

    const uint64_t transformAfterEdit = scene.transformRevision();
    ASSERT_TRUE(scene.setSourceEnabled("source-b", false));
    EXPECT_FALSE(scene.sources()[1].enabled);
    EXPECT_EQ(scene.transformRevision(), transformAfterEdit);
    EXPECT_GT(scene.visibilityRevision(), visibilityRevision);
    EXPECT_EQ(scene.contentRevision(), contentRevision);
    EXPECT_EQ(scene.sceneGraph().structuralRevision(), structuralRevision);
    EXPECT_EQ(scene.sceneGraph().lifetimeRevision(), lifetimeRevision);
    EXPECT_TRUE(scene.renderNodes()[0].visible);
    EXPECT_FALSE(scene.renderNodes()[1].visible);
    EXPECT_TRUE(scene.nodes()[0].visible);
    EXPECT_FALSE(scene.nodes()[5].visible);
    EXPECT_TRUE(scene.cameras()[0].visible);
    EXPECT_TRUE(scene.cameras()[1].visible);
    EXPECT_FALSE(scene.cameras()[2].visible);
    EXPECT_FALSE(scene.cameras()[3].visible);
    EXPECT_TRUE(scene.lights()[0].visible);
    EXPECT_FALSE(scene.lights()[1].visible);

    const uint64_t visibilityAfterDisable = scene.visibilityRevision();
    EXPECT_FALSE(scene.setSourceEnabled("source-b", false));
    EXPECT_FALSE(scene.setSourceEnabled("unknown", true));
    EXPECT_EQ(scene.visibilityRevision(), visibilityAfterDisable);
    float4x4 invalidMount = sourceBMount;
    invalidMount.a00 = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(scene.setSourceMountMatrix("source-b", invalidMount));
    EXPECT_FALSE(scene.setSourceMountMatrix("unknown", sourceBMount));
    EXPECT_EQ(scene.transformRevision(), transformAfterEdit);

    const uint64_t lifetimeBeforeFailedCompose =
        scene.sceneGraph().lifetimeRevision();
    const metallic::scene::SceneEntity sourceBRootEntity = sourceBRoot.entity();
    const size_t renderNodeCountBeforeFailedCompose = scene.renderNodes().size();
    std::vector<metallic::scene::SceneSourceDesc> invalidSources{
        sources.front(),
        metallic::scene::SceneSourceDesc{
            .id = "missing-source",
            .path = directory / "missing.gltf",
            .mountMatrix = float4x4::Identity(),
            .enabled = true,
        },
    };
    error.clear();
    EXPECT_FALSE(scene.compose(std::move(invalidSources), error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(scene.sceneGraph().lifetimeRevision(), lifetimeBeforeFailedCompose);
    EXPECT_EQ(scene.renderNodes().size(), renderNodeCountBeforeFailedCompose);
    ASSERT_EQ(scene.sources().size(), 2u);
    EXPECT_EQ(scene.sources()[1].id, "source-b");
    EXPECT_EQ(
        scene.objectForSourceNode("source-b", 0).entity(),
        sourceBRootEntity);
}

void testCompositeFallbackCamera(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "composite_fallback";
    const std::filesystem::path sourceADirectory = directory / "source_a";
    const std::filesystem::path sourceBDirectory = directory / "source_b";
    std::filesystem::create_directories(sourceADirectory);
    std::filesystem::create_directories(sourceBDirectory);

    const std::filesystem::path sourceAPath = writeFallbackScene(sourceADirectory);
    const std::filesystem::path sourceBPath = writeFallbackScene(sourceBDirectory);
    float4x4 sourceBMount = float4x4::Identity();
    sourceBMount.SetupByTranslation(float3(10.0f, 0.0f, 0.0f));
    metallic::scene::Scene scene;
    std::string error;
    ASSERT_TRUE(scene.compose(
        {
            metallic::scene::SceneSourceDesc{
                .id = "source-a",
                .path = sourceAPath,
            },
            metallic::scene::SceneSourceDesc{
                .id = "source-b",
                .path = sourceBPath,
                .mountMatrix = sourceBMount,
                .enabled = false,
            },
        },
        error)) << error;

    ASSERT_EQ(scene.sources().size(), 2u);
    EXPECT_TRUE(scene.sources()[0].enabled);
    EXPECT_FALSE(scene.sources()[1].enabled);
    ASSERT_EQ(scene.nodes().size(), 2u);
    ASSERT_EQ(scene.renderNodes().size(), 2u);
    EXPECT_TRUE(scene.renderNodes()[0].visible);
    EXPECT_FALSE(scene.renderNodes()[1].visible);
    ASSERT_EQ(scene.cameras().size(), 1u);
    const metallic::scene::RenderCamera& fallback = scene.cameras().front();
    EXPECT_TRUE(fallback.fallback);
    EXPECT_EQ(fallback.nodeIndex, metallic::scene::kInvalidSceneIndex);
    EXPECT_EQ(fallback.cameraIndex, metallic::scene::kInvalidSceneIndex);
    const metallic::scene::ConstSceneObject fallbackObject =
        scene.sceneGraph().object(fallback.object);
    ASSERT_TRUE(fallbackObject);
    EXPECT_TRUE(fallbackObject.hasComponent<metallic::scene::GeneratedComponent>());
    EXPECT_FALSE(fallbackObject.hasComponent<metallic::scene::SourceNodeComponent>());
    EXPECT_FALSE(fallbackObject.hasComponent<metallic::scene::SceneSourceComponent>());
    EXPECT_EQ(
        scene.objectForSourceNode("source-a", 0)
            .getComponent<metallic::scene::TagComponent>().name,
        "Fallback Mesh Node");
    EXPECT_EQ(
        scene.objectForSourceNode("source-b", 0)
            .getComponent<metallic::scene::TagComponent>().name,
        "Fallback Mesh Node");

    const uint64_t visibilityRevision = scene.visibilityRevision();
    ASSERT_TRUE(scene.setSourceEnabled("source-b", true));
    EXPECT_GT(scene.visibilityRevision(), visibilityRevision);
    EXPECT_TRUE(scene.renderNodes()[1].visible);
    EXPECT_EQ(scene.cameras().size(), 1u);
    EXPECT_EQ(scene.cameras().front().object, fallback.object);
}

void testCompositeSceneDocumentValidation(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory =
        baseDirectory / "composite_scene_document_validation";
    const std::filesystem::path sourceADirectory = directory / "source_a";
    const std::filesystem::path sourceBDirectory = directory / "source_b";
    std::filesystem::create_directories(sourceADirectory);
    std::filesystem::create_directories(sourceBDirectory);
    writeFullScene(sourceADirectory);
    writeFullScene(sourceBDirectory);
    const std::filesystem::path documentPath =
        directory / "invalid.metallic_scene.json";

    const auto serializeMatrix = [](const float4x4& matrix) {
        nlohmann::json value = nlohmann::json::array();
        for (const float component : matrix.a) {
            value.push_back(component);
        }
        return value;
    };
    const auto source = [&](std::string id, std::string path) {
        return nlohmann::json{
            {"id", std::move(id)},
            {"path", std::move(path)},
            {"mountMatrix", serializeMatrix(float4x4::Identity())},
            {"enabled", true},
        };
    };
    const auto manifest = [&](nlohmann::json sources, nlohmann::json nodes) {
        return nlohmann::json{
            {"version", 4},
            {"sources", std::move(sources)},
            {"nodes", std::move(nodes)},
        };
    };
    const auto expectRejected =
        [&](nlohmann::json document,
            const std::string& warningFragment) {
            writeTextFile(documentPath, document.dump(2));
            metallic::scene::SceneDocument rejected;
            EXPECT_FALSE(rejected.load(documentPath));
            EXPECT_NE(
                rejected.documentWarning().find(warningFragment),
                std::string::npos) << rejected.documentWarning();
            EXPECT_FALSE(rejected.valid());
            EXPECT_TRUE(rejected.sources().empty());
        };

    expectRejected(
        manifest(
            nlohmann::json::array({
                source("duplicate", "source_a/scene.gltf"),
                source("duplicate", "source_b/scene.gltf"),
            }),
            nlohmann::json::array()),
        "duplicate source id");

    nlohmann::json invalidMount = source("source-a", "source_a/scene.gltf");
    invalidMount["mountMatrix"] = nlohmann::json::array({1.0f, 0.0f});
    expectRejected(
        manifest(
            nlohmann::json::array({std::move(invalidMount)}),
            nlohmann::json::array()),
        "mountMatrix is invalid");

    nlohmann::json invalidEnabled = source("source-a", "source_a/scene.gltf");
    invalidEnabled["enabled"] = "yes";
    expectRejected(
        manifest(
            nlohmann::json::array({std::move(invalidEnabled)}),
            nlohmann::json::array()),
        "enabled must be a boolean");

    nlohmann::json invalidEnvironment = manifest(
        nlohmann::json::array({source("source-a", "source_a/scene.gltf")}),
        nlohmann::json::array());
    invalidEnvironment["world"] = {
        {"environment", {
            {"enabled", "yes"},
            {"intensity", nlohmann::json::object()},
        }},
    };
    expectRejected(
        std::move(invalidEnvironment),
        "world.environment fields have invalid types");

    expectRejected(
        manifest(
            nlohmann::json::array({source("missing", "missing.gltf")}),
            nlohmann::json::array()),
        "Failed to load composed scene source 'missing'");

    float4x4 ignoredOverride = float4x4::Identity();
    ignoredOverride.a03 = 42.0f;
    writeTextFile(
        documentPath,
        manifest(
            nlohmann::json::array({
                source("source-a", "source_a/scene.gltf"),
                source("source-b", "source_b/scene.gltf"),
            }),
            nlohmann::json::array({
                {
                    {"sourceId", "unknown"},
                    {"nodeIndex", 1},
                    {"sourceName", "Mesh Node"},
                    {"localMatrix", serializeMatrix(ignoredOverride)},
                },
            })).dump(2));
    metallic::scene::SceneDocument partiallyApplied;
    ASSERT_TRUE(partiallyApplied.load(documentPath))
        << partiallyApplied.documentWarning();
    EXPECT_NE(
        partiallyApplied.documentWarning().find(
            "out-of-range source 'unknown' node 1 override"),
        std::string::npos) << partiallyApplied.documentWarning();
    ASSERT_EQ(partiallyApplied.sources().size(), 2u);
    const metallic::scene::ConstSceneObject sourceAMesh =
        partiallyApplied.objectForSourceNode("source-a", 1);
    const metallic::scene::ConstSceneObject sourceBMesh =
        partiallyApplied.objectForSourceNode("source-b", 1);
    ASSERT_TRUE(sourceAMesh);
    ASSERT_TRUE(sourceBMesh);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        sourceAMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        sourceAMesh.getComponent<metallic::scene::TransformComponent>().authoredLocalMatrix));
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        sourceBMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        sourceBMesh.getComponent<metallic::scene::TransformComponent>().authoredLocalMatrix));
}

void testInvalidSceneHierarchy(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "invalid_hierarchy";
    std::filesystem::create_directories(directory);
    const std::filesystem::path gltfPath = writeFullScene(directory);

    nlohmann::json gltf;
    {
        std::ifstream stream(gltfPath, std::ios::binary);
        stream >> gltf;
    }
    gltf["nodes"][1]["children"] = nlohmann::json::array({0});
    writeTextFile(gltfPath, gltf.dump(2));

    metallic::scene::Scene cyclic;
    EXPECT_FALSE(cyclic.load(gltfPath));
    EXPECT_NE(cyclic.lastLoadResult().error.find("hierarchy"), std::string::npos);

    writeFullScene(directory);
    {
        std::ifstream stream(gltfPath, std::ios::binary);
        stream >> gltf;
    }
    gltf["nodes"][2]["children"] = nlohmann::json::array({1});
    writeTextFile(gltfPath, gltf.dump(2));

    metallic::scene::Scene multiplyParented;
    EXPECT_FALSE(multiplyParented.load(gltfPath));
    EXPECT_NE(
        multiplyParented.lastLoadResult().error.find("multiply-parented"),
        std::string::npos);
}

void testMeshletLodPartition(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeMeshletLodGridScene(directory);
    ASSERT_TRUE(scene.load(gltfPath)) << scene.lastLoadResult().error;
    ASSERT_EQ(scene.renderPrimitives().size(), 1u);

    const metallic::scene::RenderPrimitive& primitive = scene.renderPrimitives().front();
    EXPECT_GT(primitive.meshletClusters.size(), 1u);
    EXPECT_GT(primitive.meshletLodLevels.size(), 1u);
    EXPECT_GT(primitive.meshletLodGroups.size(), 1u);
    EXPECT_GT(primitive.meshletLodClusters.size(), primitive.meshletClusters.size());
    EXPECT_EQ(scene.stats().meshletLodLevelCount, primitive.meshletLodLevels.size());
    EXPECT_EQ(scene.stats().meshletLodGroupCount, primitive.meshletLodGroups.size());
    EXPECT_EQ(scene.stats().meshletLodClusterCount, primitive.meshletLodClusters.size());
    EXPECT_EQ(scene.stats().meshletLodVertexReferenceCount, primitive.meshletLodVertices.size());
    EXPECT_EQ(scene.stats().meshletLodTriangleIndexCount, primitive.meshletLodTriangles.size());

    uint32_t expectedGroupOffset = 0;
    uint32_t expectedClusterOffset = 0;
    for (size_t levelIndex = 0; levelIndex < primitive.meshletLodLevels.size(); ++levelIndex) {
        const metallic::scene::MeshletLodLevel& level = primitive.meshletLodLevels[levelIndex];
        EXPECT_EQ(level.groupOffset, expectedGroupOffset);
        EXPECT_EQ(level.clusterOffset, expectedClusterOffset);
        EXPECT_GT(level.groupCount, 0u);
        EXPECT_GT(level.clusterCount, 0u);
        EXPECT_GT(level.minBoundingSphereRadius, 0.0f);
        expectedGroupOffset += level.groupCount;
        expectedClusterOffset += level.clusterCount;
    }
    EXPECT_EQ(expectedGroupOffset, primitive.meshletLodGroups.size());
    EXPECT_EQ(expectedClusterOffset, primitive.meshletLodClusters.size());

    for (size_t groupIndex = 0; groupIndex < primitive.meshletLodGroups.size(); ++groupIndex) {
        const metallic::scene::MeshletLodGroup& group = primitive.meshletLodGroups[groupIndex];
        ASSERT_LT(group.lodLevel, primitive.meshletLodLevels.size());
        EXPECT_TRUE(group.bounds.valid);
        EXPECT_GT(group.boundingSphereRadius, 0.0f);
        ASSERT_LE(
            static_cast<size_t>(group.clusterOffset) + group.clusterCount,
            primitive.meshletLodClusters.size());

        const metallic::scene::MeshletLodLevel& level = primitive.meshletLodLevels[group.lodLevel];
        EXPECT_GE(groupIndex, level.groupOffset);
        EXPECT_LT(groupIndex, static_cast<size_t>(level.groupOffset) + level.groupCount);

        for (uint32_t childIndex = 0; childIndex < group.clusterCount; ++childIndex) {
            const metallic::scene::MeshletCluster& cluster =
                primitive.meshletLodClusters[static_cast<size_t>(group.clusterOffset) + childIndex];
            EXPECT_EQ(cluster.lodLevel, group.lodLevel);
            EXPECT_EQ(cluster.lodGroupIndex, static_cast<int32_t>(groupIndex));
            EXPECT_EQ(cluster.lodGroupChildIndex, childIndex);
            EXPECT_GT(cluster.vertexCount, 0u);
            EXPECT_GT(cluster.triangleCount, 0u);
            EXPECT_LE(cluster.vertexCount, 128u);
            EXPECT_LE(cluster.triangleCount, 128u);
            EXPECT_TRUE(cluster.bounds.valid);
            EXPECT_TRUE(
                cluster.refinedGroupIndex == metallic::scene::kInvalidSceneIndex ||
                static_cast<size_t>(cluster.refinedGroupIndex) < primitive.meshletLodGroups.size());
            ASSERT_LE(
                static_cast<size_t>(cluster.vertexOffset) + cluster.vertexCount,
                primitive.meshletLodVertices.size());
            ASSERT_LE(
                static_cast<size_t>(cluster.triangleOffset) + static_cast<size_t>(cluster.triangleCount) * 3u,
                primitive.meshletLodTriangles.size());
            for (uint32_t vertex = 0; vertex < cluster.vertexCount; ++vertex) {
                EXPECT_LT(
                    primitive.meshletLodVertices[static_cast<size_t>(cluster.vertexOffset) + vertex],
                    primitive.positions.size());
            }
            for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
                EXPECT_LT(
                    primitive.meshletLodTriangles[static_cast<size_t>(cluster.triangleOffset) + index],
                    cluster.vertexCount);
            }
        }
    }
}

void expectClusterMetadataEqual(
    const metallic::scene::MeshletCluster& actual,
    const metallic::scene::MeshletCluster& expected,
    const std::string& label)
{
    EXPECT_EQ(actual.vertexOffset, expected.vertexOffset) << label;
    EXPECT_EQ(actual.vertexCount, expected.vertexCount) << label;
    EXPECT_EQ(actual.triangleOffset, expected.triangleOffset) << label;
    EXPECT_EQ(actual.triangleCount, expected.triangleCount) << label;
    EXPECT_EQ(actual.lodLevel, expected.lodLevel) << label;
    EXPECT_EQ(actual.lodGroupChildIndex, expected.lodGroupChildIndex) << label;
    EXPECT_EQ(actual.lodGroupIndex, expected.lodGroupIndex) << label;
    EXPECT_EQ(actual.refinedGroupIndex, expected.refinedGroupIndex) << label;
    EXPECT_TRUE(actual.bounds.valid == expected.bounds.valid) << label;
    expectVec3(actual.boundingSphereCenter, expected.boundingSphereCenter, label + " center");
    EXPECT_TRUE(nearlyEqual(actual.boundingSphereRadius, expected.boundingSphereRadius)) << label;
}

void testMeshletPersistence(const std::filesystem::path& directory)
{
    const std::filesystem::path persistenceDirectory = directory / "meshlet_persistence";
    std::filesystem::create_directories(persistenceDirectory);
    const std::filesystem::path gltfPath = writeMeshletLodGridScene(persistenceDirectory);
    std::filesystem::path cachePath = gltfPath;
    cachePath += ".meshlets.bin";
    std::filesystem::remove(cachePath);

    metallic::scene::Scene generatedScene;
    ASSERT_TRUE(generatedScene.load(gltfPath)) << generatedScene.lastLoadResult().error;
    EXPECT_FALSE(generatedScene.lastLoadResult().meshletCacheLoaded);
    EXPECT_TRUE(generatedScene.lastLoadResult().meshletCacheSaved);
    EXPECT_EQ(generatedScene.lastLoadResult().meshletCachePath, cachePath);
    ASSERT_TRUE(std::filesystem::exists(cachePath));
    EXPECT_GT(std::filesystem::file_size(cachePath), 0u);

    metallic::scene::Scene cachedScene;
    ASSERT_TRUE(cachedScene.load(gltfPath)) << cachedScene.lastLoadResult().error;
    EXPECT_TRUE(cachedScene.lastLoadResult().meshletCacheLoaded);
    EXPECT_FALSE(cachedScene.lastLoadResult().meshletCacheSaved);
    EXPECT_EQ(cachedScene.lastLoadResult().meshletCachePath, cachePath);

    EXPECT_EQ(cachedScene.stats().meshletClusterCount, generatedScene.stats().meshletClusterCount);
    EXPECT_EQ(cachedScene.stats().meshletVertexReferenceCount, generatedScene.stats().meshletVertexReferenceCount);
    EXPECT_EQ(cachedScene.stats().meshletTriangleIndexCount, generatedScene.stats().meshletTriangleIndexCount);
    EXPECT_EQ(cachedScene.stats().meshletLodLevelCount, generatedScene.stats().meshletLodLevelCount);
    EXPECT_EQ(cachedScene.stats().meshletLodGroupCount, generatedScene.stats().meshletLodGroupCount);
    EXPECT_EQ(cachedScene.stats().meshletLodClusterCount, generatedScene.stats().meshletLodClusterCount);
    ASSERT_EQ(cachedScene.renderPrimitives().size(), generatedScene.renderPrimitives().size());

    const metallic::scene::RenderPrimitive& generated = generatedScene.renderPrimitives().front();
    const metallic::scene::RenderPrimitive& cached = cachedScene.renderPrimitives().front();
    EXPECT_EQ(cached.meshletVertices, generated.meshletVertices);
    EXPECT_EQ(cached.meshletTriangles, generated.meshletTriangles);
    EXPECT_EQ(cached.meshletLodVertices, generated.meshletLodVertices);
    EXPECT_EQ(cached.meshletLodTriangles, generated.meshletLodTriangles);
    ASSERT_EQ(cached.meshletClusters.size(), generated.meshletClusters.size());
    ASSERT_EQ(cached.meshletLodClusters.size(), generated.meshletLodClusters.size());
    expectClusterMetadataEqual(cached.meshletClusters.front(), generated.meshletClusters.front(), "first meshlet");
    expectClusterMetadataEqual(cached.meshletLodClusters.back(), generated.meshletLodClusters.back(), "last LOD meshlet");
}

void testMeshletStreamAsset(const std::filesystem::path& directory)
{
    const std::filesystem::path streamDirectory = directory / "meshlet_streamasset";
    std::filesystem::create_directories(streamDirectory);
    const std::filesystem::path gltfPath = writeMeshletLodGridScene(streamDirectory);
    const std::filesystem::path streamAssetPath = metallic::scene::meshletStreamAssetPathFor(gltfPath);
    std::filesystem::remove(streamAssetPath);

    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(gltfPath)) << scene.lastLoadResult().error;
    ASSERT_GT(scene.stats().meshletLodGroupCount, 0u);

    std::string reason;
    ASSERT_TRUE(metallic::scene::buildMeshletStreamAsset(
        metallic::scene::MeshletStreamAssetBuildDesc{
            .scene = &scene,
            .sourcePath = gltfPath,
            .outputPath = streamAssetPath,
        },
        reason)) << reason;
    ASSERT_TRUE(std::filesystem::exists(streamAssetPath));
    EXPECT_GT(std::filesystem::file_size(streamAssetPath), 0u);

    metallic::scene::MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(streamAssetPath, reason)) << reason;
    EXPECT_TRUE(asset.isCurrentForSource(gltfPath));
    ASSERT_EQ(asset.primitiveCount(), 1u);
    ASSERT_GE(asset.instanceCount(), 1u);
    ASSERT_EQ(asset.geometryCount(), 1u);
    ASSERT_GT(asset.lodLevelCount(), 1u);
    ASSERT_GT(asset.pageCount(), 1u);
    ASSERT_EQ(asset.groupCount(), asset.pageCount());
    ASSERT_GT(asset.nodeCount(), asset.groupCount());
    ASSERT_GT(asset.maxPagePayloadBytes(), 0u);
    ASSERT_EQ(asset.pagePayloadOffsets().size(), asset.pageCount());

    const metallic::scene::MeshletStreamPrimitiveInfo& primitive = asset.primitives().front();
    EXPECT_EQ(primitive.renderPrimitiveIndex, 0u);
    EXPECT_GT(primitive.fallbackPageCount, 0u);
    EXPECT_EQ(primitive.groupCount, primitive.pageCount);
    EXPECT_EQ(primitive.fallbackGroupCount, primitive.fallbackPageCount);
    EXPECT_EQ(primitive.groupOffset, primitive.pageOffset);
    EXPECT_EQ(primitive.fallbackGroupOffset, primitive.fallbackPageOffset);
    EXPECT_EQ(primitive.nodeOffset, 0u);
    EXPECT_EQ(primitive.nodeCount, asset.nodeCount());
    const metallic::scene::MeshletStreamGeometryInfo& geometry = asset.geometries().front();
    EXPECT_EQ(geometry.primitiveIndex, 0u);
    EXPECT_EQ(geometry.renderPrimitiveIndex, primitive.renderPrimitiveIndex);
    EXPECT_EQ(geometry.pageOffset, primitive.pageOffset);
    EXPECT_EQ(geometry.pageCount, primitive.pageCount);
    EXPECT_EQ(geometry.pagePayloadOffsetTableOffset, primitive.pageOffset);
    EXPECT_EQ(geometry.pagePayloadOffsetTableCount, primitive.pageCount);
    ASSERT_EQ(asset.geometryPagePayloadOffsets(0).size(), primitive.pageCount);
    uint32_t minLodPageCount = std::numeric_limits<uint32_t>::max();
    for (uint32_t lod = 0; lod < primitive.lodLevelCount; ++lod) {
        const metallic::scene::MeshletStreamLodLevelInfo& level =
            asset.lodLevels()[primitive.lodLevelOffset + lod];
        minLodPageCount = std::min(minLodPageCount, level.pageCount);
    }
    EXPECT_EQ(primitive.fallbackPageCount, minLodPageCount);

    bool foundOriginalCluster = false;
    bool foundRefinedCluster = false;
    bool foundTerminalGroup = false;
    for (uint32_t groupIndex = 0; groupIndex < asset.groupCount(); ++groupIndex) {
        const metallic::scene::MeshletStreamGroupInfo& group = asset.groups()[groupIndex];
        foundTerminalGroup = foundTerminalGroup ||
            group.maxQuadricError == metallic::scene::kMeshletStreamTerminalGroupError;
        ASSERT_LT(group.pageIndex, asset.pageCount());
        const metallic::scene::MeshletStreamPageInfo& groupPage = asset.pages()[group.pageIndex];
        EXPECT_EQ(group.primitiveIndex, groupPage.primitiveIndex);
        EXPECT_EQ(group.lodLevel, groupPage.lodLevel);
        EXPECT_EQ(group.clusterCount, groupPage.clusterCount);
        EXPECT_EQ(groupPage.lodGroupIndex, groupIndex);
        EXPECT_EQ(groupPage.primitiveGroupOffset, primitive.groupOffset);
        std::vector<uint8_t> groupDecodeStorage;
        std::span<const uint8_t> groupPayload;
        ASSERT_TRUE(metallic::scene::decodeMeshletStreamPayloadForDevice(
            groupPage,
            asset.pagePayload(group.pageIndex),
            groupDecodeStorage,
            groupPayload,
            reason)) << reason;
        metallic::scene::MeshletStreamPayloadHeader groupHeader;
        std::memcpy(&groupHeader, groupPayload.data(), sizeof(groupHeader));
        ASSERT_EQ(groupHeader.clusterCount, group.clusterCount);
        const auto* groupClusters =
            reinterpret_cast<const metallic::scene::MeshletStreamPayloadCluster*>(
                groupPayload.data() + groupHeader.clusterOffsetBytes);
        for (uint32_t clusterIndex = 0; clusterIndex < group.clusterCount; ++clusterIndex) {
            const uint32_t refinedGroupIndex = groupClusters[clusterIndex].refinedGroupIndex;
            if (refinedGroupIndex == metallic::scene::kMeshletStreamInvalidGroupIndex) {
                foundOriginalCluster = true;
                continue;
            }
            foundRefinedCluster = true;
            EXPECT_GE(refinedGroupIndex, primitive.groupOffset);
            EXPECT_LT(refinedGroupIndex, groupIndex);
        }
    }
    EXPECT_TRUE(foundOriginalCluster);
    EXPECT_TRUE(foundRefinedCluster);
    EXPECT_TRUE(foundTerminalGroup);

    ASSERT_LT(primitive.nodeOffset, asset.nodeCount());
    const metallic::scene::MeshletStreamNodeInfo& hierarchyRoot = asset.nodes()[primitive.nodeOffset];
    EXPECT_EQ(hierarchyRoot.primitiveIndex, 0u);
    EXPECT_EQ(hierarchyRoot.groupIndex, metallic::scene::kMeshletStreamInvalidGroupIndex);
    EXPECT_EQ(hierarchyRoot.childCount, primitive.lodLevelCount);
    EXPECT_EQ(hierarchyRoot.lodLevel, metallic::scene::kMeshletStreamInvalidNodeIndex);
    std::vector<uint8_t> visitedNodes(asset.nodeCount(), 0);
    std::vector<uint32_t> pendingNodes{primitive.nodeOffset};
    uint32_t hierarchyLeafCount = 0;
    while (!pendingNodes.empty()) {
        const uint32_t nodeIndex = pendingNodes.back();
        pendingNodes.pop_back();
        ASSERT_GE(nodeIndex, primitive.nodeOffset);
        ASSERT_LT(nodeIndex, primitive.nodeOffset + primitive.nodeCount);
        ASSERT_EQ(visitedNodes[nodeIndex], 0u);
        visitedNodes[nodeIndex] = 1;
        const metallic::scene::MeshletStreamNodeInfo& node = asset.nodes()[nodeIndex];
        EXPECT_EQ(node.primitiveIndex, 0u);
        EXPECT_GE(node.boundsCenterRadius[3], 0.0f);
        EXPECT_GE(node.maxQuadricError, 0.0f);
        if (node.childCount == 0) {
            ++hierarchyLeafCount;
            ASSERT_LT(node.groupIndex, asset.groupCount());
            EXPECT_EQ(asset.groups()[node.groupIndex].lodLevel, node.lodLevel);
            continue;
        }
        EXPECT_EQ(node.groupIndex, metallic::scene::kMeshletStreamInvalidGroupIndex);
        ASSERT_LE(node.childCount, 32u);
        ASSERT_GE(node.childOffset, primitive.nodeOffset);
        ASSERT_LE(node.childOffset + node.childCount, primitive.nodeOffset + primitive.nodeCount);
        for (uint32_t child = 0; child < node.childCount; ++child) {
            pendingNodes.push_back(node.childOffset + child);
        }
    }
    EXPECT_EQ(
        std::count(visitedNodes.begin(), visitedNodes.end(), static_cast<uint8_t>(1)),
        primitive.nodeCount);
    EXPECT_EQ(hierarchyLeafCount, primitive.groupCount);

    for (uint32_t pageIndex = 0; pageIndex < asset.pageCount(); ++pageIndex) {
        const metallic::scene::MeshletStreamPageInfo& page = asset.pages()[pageIndex];
        EXPECT_EQ(page.payloadOffset, asset.pagePayloadOffsets()[pageIndex]);
        EXPECT_EQ(page.payloadOffset % 16u, 0u);
        EXPECT_GT(page.payloadSize, sizeof(metallic::scene::MeshletStreamPayloadHeader));
        EXPECT_EQ(page.payloadSize, page.uncompressedSize);
        EXPECT_LE(page.payloadSize, asset.maxPagePayloadBytes());
        EXPECT_EQ(
            page.compressionMode,
            static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadCompression::None));
        EXPECT_TRUE((page.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributePosition) != 0u);
        EXPECT_TRUE((page.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributeNormal) != 0u);
        EXPECT_TRUE((page.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributeTexcoord0) != 0u);
        EXPECT_TRUE((page.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributeMaterial) != 0u);
    }

    const metallic::scene::MeshletStreamPageInfo& page = asset.pages().front();
    const std::span<const uint8_t> payload = asset.pagePayload(0);
    ASSERT_EQ(payload.size(), page.payloadSize);
    metallic::scene::MeshletStreamPayloadHeader header;
    std::memcpy(&header, payload.data(), sizeof(header));
    EXPECT_EQ(header.version, 3u);
    EXPECT_EQ(header.clusterCount, page.clusterCount);
    EXPECT_EQ(header.vertexCount, page.vertexCount);
    EXPECT_EQ(header.triangleIndexCount, page.triangleIndexCount);
    EXPECT_EQ(header.payloadByteSize, page.payloadSize);
    EXPECT_EQ(header.uncompressedPayloadByteSize, page.uncompressedSize);
    EXPECT_EQ(header.attributeFlags, page.attributeFlags);
    EXPECT_EQ(header.compressionMode, page.compressionMode);
    EXPECT_EQ(
        header.positionFormat,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadFormat::Float32x4));
    EXPECT_EQ(
        header.normalFormat,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadFormat::Float32x4));
    EXPECT_EQ(
        header.texcoord0Format,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadFormat::Float32x2));
    EXPECT_EQ(
        header.materialFormat,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadFormat::Uint32));
    EXPECT_EQ(header.materialCount, header.clusterCount);

    ASSERT_LE(
        header.clusterOffsetBytes + header.clusterCount * sizeof(metallic::scene::MeshletStreamPayloadCluster),
        payload.size());
    ASSERT_LE(header.positionOffsetBytes + header.vertexCount * sizeof(float) * 4u, payload.size());
    ASSERT_LE(header.normalOffsetBytes + header.vertexCount * sizeof(float) * 4u, payload.size());
    ASSERT_LE(header.texcoord0OffsetBytes + header.vertexCount * sizeof(float) * 2u, payload.size());
    ASSERT_LE(header.materialOffsetBytes + header.materialCount * sizeof(uint32_t), payload.size());
    const auto* clusters = reinterpret_cast<const metallic::scene::MeshletStreamPayloadCluster*>(
        payload.data() + header.clusterOffsetBytes);
    const auto* materialIds = reinterpret_cast<const uint32_t*>(payload.data() + header.materialOffsetBytes);
    for (uint32_t clusterIndex = 0; clusterIndex < header.clusterCount; ++clusterIndex) {
        const metallic::scene::MeshletStreamPayloadCluster& cluster = clusters[clusterIndex];
        ASSERT_LE(cluster.vertexOffset + cluster.vertexCount, header.vertexCount);
        ASSERT_LE(cluster.triangleOffset + cluster.triangleCount * 3u, header.triangleIndexCount);
        EXPECT_EQ(materialIds[clusterIndex], page.materialIndex);
        if (cluster.refinedGroupIndex != metallic::scene::kMeshletStreamInvalidGroupIndex) {
            EXPECT_GE(cluster.refinedGroupIndex, page.primitiveGroupOffset);
            EXPECT_LT(cluster.refinedGroupIndex, page.lodGroupIndex);
        }
        for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
            const uint8_t localIndex = payload[header.triangleOffsetBytes + cluster.triangleOffset + index];
            EXPECT_LT(localIndex, cluster.vertexCount);
        }
    }

    const auto* normals = reinterpret_cast<const float*>(payload.data() + header.normalOffsetBytes);
    const auto* texcoords = reinterpret_cast<const float*>(payload.data() + header.texcoord0OffsetBytes);
    EXPECT_TRUE(nearlyEqual(normals[2], 1.0f));
    EXPECT_TRUE(nearlyEqual(normals[3], 0.0f));
    EXPECT_GE(texcoords[0], 0.0f);
    EXPECT_LE(texcoords[0], 1.0f);
    EXPECT_GE(texcoords[1], 0.0f);
    EXPECT_LE(texcoords[1], 1.0f);

    const std::filesystem::path compressedStreamAssetPath =
        streamDirectory / "meshlet_lod_grid.byte_rle.meshstream.bin";
    std::filesystem::remove(compressedStreamAssetPath);
    ASSERT_TRUE(metallic::scene::buildMeshletStreamAsset(
        metallic::scene::MeshletStreamAssetBuildDesc{
            .scene = &scene,
            .sourcePath = gltfPath,
            .outputPath = compressedStreamAssetPath,
            .compressionMode = metallic::scene::MeshletStreamPayloadCompression::ByteRle,
        },
        reason)) << reason;

    metallic::scene::MeshletStreamAsset compressedAsset;
    ASSERT_TRUE(compressedAsset.open(compressedStreamAssetPath, reason)) << reason;
    ASSERT_EQ(compressedAsset.pageCount(), asset.pageCount());
    ASSERT_EQ(compressedAsset.groupCount(), asset.groupCount());
    ASSERT_EQ(compressedAsset.nodeCount(), asset.nodeCount());
    const metallic::scene::MeshletStreamPageInfo& compressedPage = compressedAsset.pages().front();
    EXPECT_EQ(
        compressedPage.compressionMode,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadCompression::ByteRle));
    EXPECT_NE(compressedPage.payloadSize, compressedPage.uncompressedSize);
    EXPECT_EQ(compressedPage.uncompressedSize, page.uncompressedSize);

    const std::span<const uint8_t> storedCompressedPayload = compressedAsset.pagePayload(0);
    ASSERT_EQ(storedCompressedPayload.size(), compressedPage.payloadSize);
    metallic::scene::MeshletStreamPayloadHeader storedCompressedHeader;
    std::memcpy(&storedCompressedHeader, storedCompressedPayload.data(), sizeof(storedCompressedHeader));
    EXPECT_EQ(storedCompressedHeader.payloadByteSize, compressedPage.payloadSize);
    EXPECT_EQ(storedCompressedHeader.uncompressedPayloadByteSize, compressedPage.uncompressedSize);
    EXPECT_EQ(storedCompressedHeader.compressionMode, compressedPage.compressionMode);

    std::vector<uint8_t> decodedPayloadStorage;
    std::span<const uint8_t> decodedPayload;
    ASSERT_TRUE(metallic::scene::decodeMeshletStreamPayloadForDevice(
        compressedPage,
        storedCompressedPayload,
        decodedPayloadStorage,
        decodedPayload,
        reason)) << reason;
    ASSERT_EQ(decodedPayload.size(), compressedPage.uncompressedSize);
    metallic::scene::MeshletStreamPayloadHeader decodedHeader;
    std::memcpy(&decodedHeader, decodedPayload.data(), sizeof(decodedHeader));
    EXPECT_EQ(decodedHeader.payloadByteSize, compressedPage.uncompressedSize);
    EXPECT_EQ(decodedHeader.uncompressedPayloadByteSize, compressedPage.uncompressedSize);
    EXPECT_EQ(
        decodedHeader.compressionMode,
        static_cast<uint32_t>(metallic::scene::MeshletStreamPayloadCompression::None));
    ASSERT_LE(
        decodedHeader.clusterOffsetBytes +
            decodedHeader.clusterCount * sizeof(metallic::scene::MeshletStreamPayloadCluster),
        decodedPayload.size());
    ASSERT_LE(decodedHeader.positionOffsetBytes + decodedHeader.vertexCount * sizeof(float) * 4u, decodedPayload.size());
    ASSERT_LE(decodedHeader.normalOffsetBytes + decodedHeader.vertexCount * sizeof(float) * 4u, decodedPayload.size());
    ASSERT_LE(decodedHeader.texcoord0OffsetBytes + decodedHeader.vertexCount * sizeof(float) * 2u, decodedPayload.size());

    const std::filesystem::path lazyValidationPath =
        streamDirectory / "meshlet_lod_grid.lazy_validation.meshstream.bin";
    std::filesystem::copy_file(
        streamAssetPath,
        lazyValidationPath,
        std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream corrupted(lazyValidationPath, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(corrupted);
        corrupted.seekp(static_cast<std::streamoff>(page.payloadOffset));
        const uint32_t invalidMagic = 0;
        corrupted.write(reinterpret_cast<const char*>(&invalidMagic), sizeof(invalidMagic));
        ASSERT_TRUE(corrupted);
    }
    metallic::scene::MeshletStreamAsset lazyValidationAsset;
    ASSERT_TRUE(lazyValidationAsset.open(lazyValidationPath, reason)) << reason;
    std::vector<uint8_t> lazyDecodeStorage;
    std::span<const uint8_t> lazyDecodedPayload;
    EXPECT_FALSE(metallic::scene::decodeMeshletStreamPayloadForDevice(
        lazyValidationAsset.pages().front(),
        lazyValidationAsset.pagePayload(0),
        lazyDecodeStorage,
        lazyDecodedPayload,
        reason));
    EXPECT_NE(reason.find("header"), std::string::npos) << reason;
    lazyValidationAsset.close();
    std::filesystem::remove(lazyValidationPath);

    if constexpr (sizeof(void*) >= 8) {
        const std::filesystem::path largeSparsePath =
            streamDirectory / "meshlet_lod_grid.large_sparse.meshstream.bin";
        std::filesystem::copy_file(
            streamAssetPath,
            largeSparsePath,
            std::filesystem::copy_options::overwrite_existing);
        constexpr uint64_t kZorahScaleSparseSize = uint64_t{26} << 30u;
        ASSERT_TRUE(resizeSparseFile(largeSparsePath, kZorahScaleSparseSize));
        {
            std::fstream largeSparse(
                largeSparsePath,
                std::ios::binary | std::ios::in | std::ios::out);
            ASSERT_TRUE(largeSparse);
            constexpr std::streamoff kFileSizeHeaderOffset = 16;
            largeSparse.seekp(kFileSizeHeaderOffset);
            largeSparse.write(
                reinterpret_cast<const char*>(&kZorahScaleSparseSize),
                sizeof(kZorahScaleSparseSize));
            ASSERT_TRUE(largeSparse);
        }
        metallic::scene::MeshletStreamAsset largeSparseAsset;
        ASSERT_TRUE(largeSparseAsset.open(largeSparsePath, reason)) << reason;
        EXPECT_EQ(std::filesystem::file_size(largeSparsePath), kZorahScaleSparseSize);
        largeSparseAsset.close();
        std::filesystem::remove(largeSparsePath);
    }

    const std::filesystem::path offlineDirectory = directory / "meshlet_streamasset_offline";
    std::filesystem::create_directories(offlineDirectory);
    const std::filesystem::path offlineGltfPath = writeMeshletLodGridScene(offlineDirectory, 1, true);
    const std::filesystem::path offlineStreamAssetPath =
        offlineDirectory / "meshlet_lod_grid.offline.meshstream.bin";
    std::filesystem::path offlineMeshletCachePath = offlineGltfPath;
    offlineMeshletCachePath += ".meshlets.bin";
    std::filesystem::remove(offlineStreamAssetPath);
    std::filesystem::remove(offlineMeshletCachePath);
    metallic::scene::MeshletStreamAssetOfflineBuildStats offlineBuildStats;
    ASSERT_TRUE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = offlineGltfPath,
            .outputPath = offlineStreamAssetPath,
            .stats = &offlineBuildStats,
        },
        reason)) << reason;
    EXPECT_FALSE(std::filesystem::exists(offlineMeshletCachePath));
    EXPECT_EQ(offlineBuildStats.usedExternalBufferRangeReads, 1u);
    EXPECT_GT(offlineBuildStats.externalBufferDeclaredBytes, 0u);
    EXPECT_GT(offlineBuildStats.accessorRangeReadCount, 0u);
    EXPECT_GT(offlineBuildStats.accessorRangeReadBytes, 0u);
    EXPECT_GT(offlineBuildStats.maxAccessorRangeReadBytes, 0u);
    EXPECT_EQ(offlineBuildStats.partialCheckpointCount, 1u);
    EXPECT_LT(
        offlineBuildStats.maxAccessorRangeReadBytes,
        offlineBuildStats.externalBufferDeclaredBytes);

    metallic::scene::MeshletStreamAsset offlineAsset;
    ASSERT_TRUE(offlineAsset.open(offlineStreamAssetPath, reason)) << reason;
    EXPECT_TRUE(offlineAsset.isCurrentForSource(offlineGltfPath));
    EXPECT_EQ(offlineAsset.primitiveCount(), asset.primitiveCount());
    EXPECT_EQ(offlineAsset.geometryCount(), asset.geometryCount());
    EXPECT_EQ(offlineAsset.instanceCount(), asset.instanceCount());
    EXPECT_EQ(offlineAsset.lodLevelCount(), asset.lodLevelCount());
    EXPECT_EQ(offlineAsset.groupCount(), asset.groupCount());
    EXPECT_EQ(offlineAsset.nodeCount(), asset.nodeCount());
    EXPECT_EQ(offlineAsset.pageCount(), asset.pageCount());
    EXPECT_EQ(offlineAsset.pages().front().attributeFlags, asset.pages().front().attributeFlags);
    {
        std::ofstream externalBuffer(
            offlineDirectory / "meshlet lod grid.bin",
            std::ios::binary | std::ios::app);
        externalBuffer.put('\0');
    }
    EXPECT_FALSE(offlineAsset.isCurrentForSource(offlineGltfPath));
    offlineAsset.close();
    ASSERT_TRUE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = offlineGltfPath,
            .outputPath = offlineStreamAssetPath,
        },
        reason)) << reason;
    ASSERT_TRUE(offlineAsset.open(offlineStreamAssetPath, reason)) << reason;
    EXPECT_TRUE(offlineAsset.isCurrentForSource(offlineGltfPath));

    const std::filesystem::path partialDirectory = directory / "meshlet_streamasset_partial";
    std::filesystem::create_directories(partialDirectory);
    const std::filesystem::path partialGltfPath = writeMeshletLodGridScene(partialDirectory, 2);
    const std::filesystem::path partialStreamAssetPath =
        partialDirectory / "meshlet_lod_grid.partial.meshstream.bin";
    std::filesystem::path partialCachePath = partialStreamAssetPath;
    partialCachePath += ".partial";
    std::filesystem::remove(partialStreamAssetPath);
    std::filesystem::remove(partialCachePath);
    ASSERT_FALSE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = partialGltfPath,
            .outputPath = partialStreamAssetPath,
            .maxNewGeometriesPerInvocation = 1,
        },
        reason));
    EXPECT_NE(reason.find("paused"), std::string::npos) << reason;
    EXPECT_TRUE(std::filesystem::exists(partialStreamAssetPath));
    EXPECT_TRUE(std::filesystem::exists(partialCachePath));

    {
        std::ofstream externalBuffer(
            partialDirectory / "meshlet_lod_grid.bin",
            std::ios::binary | std::ios::app);
        externalBuffer.put('\0');
    }

    metallic::scene::MeshletStreamAssetOfflineBuildStats restartedBuildStats;
    ASSERT_TRUE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = partialGltfPath,
            .outputPath = partialStreamAssetPath,
            .stats = &restartedBuildStats,
        },
        reason)) << reason;
    EXPECT_FALSE(std::filesystem::exists(partialCachePath));
    EXPECT_EQ(
        restartedBuildStats.accessorRangeReadCount,
        offlineBuildStats.accessorRangeReadCount * 2u);
    EXPECT_EQ(restartedBuildStats.partialCheckpointCount, 1u);

    metallic::scene::MeshletStreamAsset resumedAsset;
    ASSERT_TRUE(resumedAsset.open(partialStreamAssetPath, reason)) << reason;
    EXPECT_TRUE(resumedAsset.isCurrentForSource(partialGltfPath));
    EXPECT_EQ(resumedAsset.primitiveCount(), 2u);
    EXPECT_EQ(resumedAsset.geometryCount(), 2u);
    EXPECT_EQ(resumedAsset.instanceCount(), 2u);
    EXPECT_EQ(resumedAsset.lodLevelCount(), offlineAsset.lodLevelCount() * 2u);
    EXPECT_EQ(resumedAsset.groupCount(), offlineAsset.groupCount() * 2u);
    EXPECT_EQ(resumedAsset.nodeCount(), offlineAsset.nodeCount() * 2u);
    EXPECT_EQ(resumedAsset.pageCount(), offlineAsset.pageCount() * 2u);

    asset.close();
    {
        std::ofstream source(gltfPath, std::ios::app);
        source << "\n";
    }
    ASSERT_TRUE(asset.open(streamAssetPath, reason)) << reason;
    EXPECT_FALSE(asset.isCurrentForSource(gltfPath));
}

void testMeshoptCompressedMeshletStreamAsset(const std::filesystem::path& directory)
{
    const std::filesystem::path compressedDirectory = directory / "meshopt_compressed_streamasset";
    std::filesystem::create_directories(compressedDirectory);
    const std::filesystem::path gltfPath = writeMeshoptCompressedScene(
        compressedDirectory,
        sizeof(uint16_t) * 4u,
        2u);
    const std::filesystem::path streamAssetPath =
        compressedDirectory / "meshopt_compressed.meshstream.bin";
    std::filesystem::path partialPath = streamAssetPath;
    partialPath += ".partial";
    std::filesystem::path meshoptCachePath = streamAssetPath;
    meshoptCachePath += ".meshopt-cache";
    std::filesystem::remove(streamAssetPath);
    std::filesystem::remove(partialPath);
    std::filesystem::remove_all(meshoptCachePath);

    std::string reason;
    metallic::scene::MeshletStreamAssetOfflineBuildStats pausedBuildStats;
    ASSERT_FALSE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = gltfPath,
            .outputPath = streamAssetPath,
            .maxNewGeometriesPerInvocation = 1,
            .stats = &pausedBuildStats,
        },
        reason));
    EXPECT_NE(reason.find("paused"), std::string::npos) << reason;
    EXPECT_EQ(pausedBuildStats.usedExternalBufferRangeReads, 1u);
    EXPECT_EQ(pausedBuildStats.accessorRangeReadCount, 4u);
    EXPECT_GT(pausedBuildStats.accessorRangeReadBytes, 0u);
    EXPECT_LE(
        pausedBuildStats.accessorRangeReadBytes,
        pausedBuildStats.externalBufferDeclaredBytes);
    EXPECT_GT(pausedBuildStats.maxAccessorRangeReadBytes, 0u);
    EXPECT_LT(
        pausedBuildStats.maxAccessorRangeReadBytes,
        pausedBuildStats.externalBufferDeclaredBytes);
    EXPECT_EQ(pausedBuildStats.partialCheckpointCount, 1u);
    EXPECT_TRUE(std::filesystem::exists(partialPath));
    EXPECT_TRUE(std::filesystem::exists(meshoptCachePath));

#ifdef _WIN32
    HANDLE lockedPartial = CreateFileW(
        partialPath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    ASSERT_NE(lockedPartial, INVALID_HANDLE_VALUE);
    std::thread unlockPartial([lockedPartial]() {
        Sleep(500);
        CloseHandle(lockedPartial);
    });
#endif
    metallic::scene::MeshletStreamAssetOfflineBuildStats resumedBuildStats;
    const bool resumed = metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = gltfPath,
            .outputPath = streamAssetPath,
            .stats = &resumedBuildStats,
        },
        reason);
#ifdef _WIN32
    unlockPartial.join();
#endif
    ASSERT_TRUE(resumed) << reason;
    EXPECT_EQ(resumedBuildStats.usedExternalBufferRangeReads, 1u);
    EXPECT_EQ(resumedBuildStats.accessorRangeReadCount, 0u);
    EXPECT_EQ(resumedBuildStats.accessorRangeReadBytes, 0u);
    EXPECT_EQ(resumedBuildStats.maxAccessorRangeReadBytes, 0u);
    EXPECT_EQ(resumedBuildStats.partialCheckpointCount, 1u);
    EXPECT_FALSE(std::filesystem::exists(partialPath));
    EXPECT_FALSE(std::filesystem::exists(meshoptCachePath));

    metallic::scene::MeshletStreamAsset asset;
    ASSERT_TRUE(asset.open(streamAssetPath, reason)) << reason;
    EXPECT_TRUE(asset.isCurrentForSource(gltfPath));
    ASSERT_EQ(asset.primitiveCount(), 2u);
    ASSERT_EQ(asset.geometryCount(), 2u);
    ASSERT_EQ(asset.instanceCount(), 2u);
    ASSERT_GT(asset.pageCount(), 0u);
    const metallic::scene::MeshletStreamBounds& primitiveBounds = asset.primitives().front().bounds;
    ASSERT_EQ(primitiveBounds.valid, 1u);
    EXPECT_TRUE(nearlyEqual(primitiveBounds.min[0], 0.0f));
    EXPECT_TRUE(nearlyEqual(primitiveBounds.min[1], 0.0f));
    EXPECT_TRUE(nearlyEqual(primitiveBounds.min[2], 0.0f));
    EXPECT_TRUE(nearlyEqual(primitiveBounds.max[0], 1.0f));
    EXPECT_TRUE(nearlyEqual(primitiveBounds.max[1], 1.0f));
    EXPECT_TRUE(nearlyEqual(primitiveBounds.max[2], 0.0f));

    constexpr std::array<std::array<float, 3>, 3> kExpectedPositions{
        std::array<float, 3>{0.0f, 0.0f, 0.0f},
        std::array<float, 3>{1.0f, 0.0f, 0.0f},
        std::array<float, 3>{0.0f, 1.0f, 0.0f},
    };
    constexpr std::array<std::array<float, 2>, 3> kExpectedTexcoords{
        std::array<float, 2>{0.0f, 0.0f},
        std::array<float, 2>{1.0f, 0.0f},
        std::array<float, 2>{0.0f, 1.0f},
    };
    std::array<bool, 3> foundPositions{};
    uint32_t decodedTriangleCount = 0;
    for (uint32_t pageIndex = 0; pageIndex < asset.pageCount(); ++pageIndex) {
        const metallic::scene::MeshletStreamPageInfo& page = asset.pages()[pageIndex];
        const std::span<const uint8_t> payload = asset.pagePayload(pageIndex);
        ASSERT_GE(payload.size(), sizeof(metallic::scene::MeshletStreamPayloadHeader));
        metallic::scene::MeshletStreamPayloadHeader header;
        std::memcpy(&header, payload.data(), sizeof(header));
        ASSERT_TRUE(
            (header.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributeNormal) != 0u);
        ASSERT_TRUE(
            (header.attributeFlags & metallic::scene::kMeshletStreamPayloadAttributeTexcoord0) != 0u);
        ASSERT_LE(header.positionOffsetBytes + header.vertexCount * sizeof(float) * 4u, payload.size());
        ASSERT_LE(header.normalOffsetBytes + header.vertexCount * sizeof(float) * 4u, payload.size());
        ASSERT_LE(header.texcoord0OffsetBytes + header.vertexCount * sizeof(float) * 2u, payload.size());
        ASSERT_LE(
            header.clusterOffsetBytes +
                header.clusterCount * sizeof(metallic::scene::MeshletStreamPayloadCluster),
            payload.size());

        const auto* positions = reinterpret_cast<const float*>(payload.data() + header.positionOffsetBytes);
        const auto* normals = reinterpret_cast<const float*>(payload.data() + header.normalOffsetBytes);
        const auto* texcoords = reinterpret_cast<const float*>(payload.data() + header.texcoord0OffsetBytes);
        for (uint32_t vertexIndex = 0; vertexIndex < header.vertexCount; ++vertexIndex) {
            size_t expectedIndex = kExpectedPositions.size();
            for (size_t candidate = 0; candidate < kExpectedPositions.size(); ++candidate) {
                if (nearlyEqual(positions[vertexIndex * 4u], kExpectedPositions[candidate][0]) &&
                    nearlyEqual(positions[vertexIndex * 4u + 1u], kExpectedPositions[candidate][1]) &&
                    nearlyEqual(positions[vertexIndex * 4u + 2u], kExpectedPositions[candidate][2])) {
                    expectedIndex = candidate;
                    break;
                }
            }
            ASSERT_LT(expectedIndex, kExpectedPositions.size());
            foundPositions[expectedIndex] = true;
            EXPECT_TRUE(nearlyEqual(positions[vertexIndex * 4u + 3u], 1.0f));
            EXPECT_TRUE(nearlyEqual(normals[vertexIndex * 4u], 0.0f));
            EXPECT_TRUE(nearlyEqual(normals[vertexIndex * 4u + 1u], 0.0f));
            EXPECT_TRUE(nearlyEqual(normals[vertexIndex * 4u + 2u], 1.0f));
            EXPECT_TRUE(nearlyEqual(normals[vertexIndex * 4u + 3u], 0.0f));
            EXPECT_TRUE(nearlyEqual(
                texcoords[vertexIndex * 2u],
                kExpectedTexcoords[expectedIndex][0]));
            EXPECT_TRUE(nearlyEqual(
                texcoords[vertexIndex * 2u + 1u],
                kExpectedTexcoords[expectedIndex][1]));
        }

        const auto* clusters = reinterpret_cast<const metallic::scene::MeshletStreamPayloadCluster*>(
            payload.data() + header.clusterOffsetBytes);
        for (uint32_t clusterIndex = 0; clusterIndex < header.clusterCount; ++clusterIndex) {
            const metallic::scene::MeshletStreamPayloadCluster& cluster = clusters[clusterIndex];
            decodedTriangleCount += cluster.triangleCount;
            ASSERT_LE(cluster.vertexOffset + cluster.vertexCount, header.vertexCount);
            ASSERT_LE(cluster.triangleOffset + cluster.triangleCount * 3u, header.triangleIndexCount);
            for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
                EXPECT_LT(
                    payload[header.triangleOffsetBytes + cluster.triangleOffset + index],
                    cluster.vertexCount);
            }
        }
    }
    EXPECT_TRUE(std::ranges::all_of(foundPositions, [](bool found) { return found; }));
    EXPECT_GE(decodedTriangleCount, 1u);

    asset.close();
    const std::filesystem::path invalidDirectory =
        directory / "meshopt_invalid_stride_streamasset";
    std::filesystem::create_directories(invalidDirectory);
    const std::filesystem::path invalidGltfPath = writeMeshoptCompressedScene(invalidDirectory, 10u);
    const std::filesystem::path invalidStreamAssetPath =
        invalidDirectory / "meshopt_invalid_stride.meshstream.bin";
    std::filesystem::remove(invalidStreamAssetPath);
    ASSERT_FALSE(metallic::scene::buildMeshletStreamAssetOffline(
        metallic::scene::MeshletStreamAssetOfflineBuildDesc{
            .sourcePath = invalidGltfPath,
            .outputPath = invalidStreamAssetPath,
        },
        reason));
    EXPECT_NE(reason.find("meshopt bufferView stride is invalid"), std::string::npos) << reason;
}

void testMaterialImport(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeMaterialFeatureScene(directory);
    expect(scene.load(gltfPath), scene.lastLoadResult().error);

    expect(scene.stats().materialCount == 4, "material feature scene material count");
    expect(scene.images().size() == 9, "material feature scene image count");
    expect(scene.images()[0].name == "Base Color Image", "base color image name");
    expect(scene.images()[0].uri == "base_color.png", "base color image uri");
    expect(scene.textures().size() == 9, "material feature scene texture count");
    expect(scene.textures()[2].imageIndex == 2, "normal texture image index");
    expect(scene.textures()[2].samplerIndex == 1, "normal texture sampler index");

    const std::array<float, 6> identityUv{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const metallic::scene::RenderMaterial& defaultMaterial = scene.materials()[0];
    expectVec4(
        defaultMaterial.baseColorFactor,
        float4(1.0f, 1.0f, 1.0f, 1.0f),
        "material default baseColorFactor");
    expect(nearlyEqual(defaultMaterial.metallicFactor, 1.0f), "material default metallicFactor");
    expect(nearlyEqual(defaultMaterial.roughnessFactor, 1.0f), "material default roughnessFactor");
    expectVec3(
        defaultMaterial.emissiveFactor,
        float3(0.0f, 0.0f, 0.0f),
        "material default emissiveFactor");
    expect(defaultMaterial.alphaMode == "OPAQUE", "material default alphaMode");
    expect(nearlyEqual(defaultMaterial.alphaCutoff, 0.5f), "material default alphaCutoff");
    expect(!defaultMaterial.doubleSided, "material default doubleSided");
    expect(nearlyEqual(defaultMaterial.normalTextureScale, 1.0f), "material default normal scale");
    expect(nearlyEqual(defaultMaterial.occlusionTextureStrength, 1.0f), "material default occlusion strength");
    expect(nearlyEqual(defaultMaterial.transmissionFactor, 0.0f), "material default transmissionFactor");
    expect(nearlyEqual(defaultMaterial.ior, 1.5f), "material default ior");
    expect(nearlyEqual(defaultMaterial.thicknessFactor, 0.0f), "material default thicknessFactor");
    expect(nearlyEqual(defaultMaterial.attenuationDistance, 0.0f), "material default attenuationDistance");
    expectVec3(
        defaultMaterial.attenuationColor,
        float3(1.0f, 1.0f, 1.0f),
        "material default attenuationColor");
    expect(
        nearlyEqual(defaultMaterial.diffuseTransmissionFactor, 0.0f),
        "material default diffuseTransmissionFactor");
    expectVec3(
        defaultMaterial.diffuseTransmissionColor,
        float3(1.0f, 1.0f, 1.0f),
        "material default diffuseTransmissionColor");
    expectTextureInfo(
        defaultMaterial.baseColorTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "material default baseColorTexture");
    expectTextureInfo(
        defaultMaterial.diffuseTransmissionColorTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "material default diffuseTransmissionColorTexture");

    const metallic::scene::RenderMaterial& fullMaterial = scene.materials()[1];
    expectVec4(
        fullMaterial.baseColorFactor,
        float4(0.2f, 0.3f, 0.4f, 0.5f),
        "full material baseColorFactor");
    expect(nearlyEqual(fullMaterial.metallicFactor, 0.25f), "full material metallicFactor");
    expect(nearlyEqual(fullMaterial.roughnessFactor, 0.75f), "full material roughnessFactor");
    expectVec3(fullMaterial.emissiveFactor, float3(0.4f, 0.8f, 1.2f), "full material emissiveFactor");
    expect(fullMaterial.alphaMode == "MASK", "full material alphaMode");
    expect(nearlyEqual(fullMaterial.alphaCutoff, 0.37f), "full material alphaCutoff");
    expect(fullMaterial.doubleSided, "full material doubleSided");
    expect(nearlyEqual(fullMaterial.normalTextureScale, 0.42f), "full material normal scale");
    expect(nearlyEqual(fullMaterial.occlusionTextureStrength, 0.66f), "full material occlusion strength");
    expectTextureInfo(
        fullMaterial.baseColorTexture,
        0,
        2,
        {0.0f, -3.0f, 0.1f, 2.0f, 0.0f, 0.2f},
        "full material baseColorTexture");
    expectTextureInfo(fullMaterial.metallicRoughnessTexture, 1, 0, identityUv, "full material metallicRoughnessTexture");
    expectTextureInfo(
        fullMaterial.normalTexture,
        2,
        0,
        {0.5f, 0.0f, 0.05f, 0.0f, 0.25f, 0.15f},
        "full material normalTexture");
    expectTextureInfo(
        fullMaterial.occlusionTexture,
        3,
        3,
        {1.25f, 0.0f, 0.3f, 0.0f, 0.75f, 0.4f},
        "full material occlusionTexture");
    expectTextureInfo(fullMaterial.emissiveTexture, 4, 0, identityUv, "full material emissiveTexture");

    expect(nearlyEqual(fullMaterial.transmissionFactor, 0.8f), "full material transmissionFactor");
    expect(nearlyEqual(fullMaterial.ior, 2.2f), "full material ior");
    expect(nearlyEqual(fullMaterial.thicknessFactor, 0.6f), "full material thicknessFactor");
    expect(nearlyEqual(fullMaterial.attenuationDistance, 12.5f), "full material attenuationDistance");
    expectVec3(fullMaterial.attenuationColor, float3(0.7f, 0.8f, 0.9f), "full material attenuationColor");
    expect(
        nearlyEqual(fullMaterial.diffuseTransmissionFactor, 0.55f),
        "full material diffuseTransmissionFactor");
    expectVec3(
        fullMaterial.diffuseTransmissionColor,
        float3(0.4f, 0.5f, 0.6f),
        "full material diffuseTransmissionColor");
    expectTextureInfo(
        fullMaterial.transmissionTexture,
        5,
        0,
        {0.5f, 0.0f, 0.25f, 0.0f, 0.25f, 0.75f},
        "full material transmissionTexture");
    expectTextureInfo(fullMaterial.thicknessTexture, 6, 0, identityUv, "full material thicknessTexture");
    expectTextureInfo(
        fullMaterial.diffuseTransmissionTexture,
        7,
        0,
        identityUv,
        "full material diffuseTransmissionTexture");
    expectTextureInfo(
        fullMaterial.diffuseTransmissionColorTexture,
        8,
        1,
        identityUv,
        "full material diffuseTransmissionColorTexture");

    const metallic::scene::RenderMaterial& lowClampMaterial = scene.materials()[2];
    expect(nearlyEqual(lowClampMaterial.transmissionFactor, 0.0f), "low clamp transmissionFactor");
    expect(nearlyEqual(lowClampMaterial.ior, 1.0f), "low clamp ior");
    expect(nearlyEqual(lowClampMaterial.thicknessFactor, 0.0f), "low clamp thicknessFactor");
    expect(nearlyEqual(lowClampMaterial.attenuationDistance, 0.0f), "low clamp attenuationDistance");
    expect(
        nearlyEqual(lowClampMaterial.diffuseTransmissionFactor, 0.0f),
        "low clamp diffuseTransmissionFactor");

    const metallic::scene::RenderMaterial& highClampMaterial = scene.materials()[3];
    expect(nearlyEqual(highClampMaterial.transmissionFactor, 1.0f), "high clamp transmissionFactor");
    expect(nearlyEqual(highClampMaterial.ior, 3.0f), "high clamp ior");
    expect(
        nearlyEqual(highClampMaterial.diffuseTransmissionFactor, 1.0f),
        "high clamp diffuseTransmissionFactor");
}

void testABeautifulGameMaterialImport()
{
    const std::array<float, 6> identityUv{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const std::filesystem::path scenePath =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf";

    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(scenePath)) << scene.lastLoadResult().error;

    EXPECT_EQ(scene.sceneName(), "Scene");
    EXPECT_EQ(scene.stats().meshCount, 15u);
    EXPECT_EQ(scene.stats().materialCount, 15u);
    EXPECT_EQ(scene.materials().size(), 15u);
    EXPECT_EQ(scene.textures().size(), 43u);
    EXPECT_EQ(scene.images().size(), 33u);
    EXPECT_FALSE(scene.renderPrimitives().empty());

    for (const metallic::scene::RenderPrimitive& primitive : scene.renderPrimitives()) {
        EXPECT_FALSE(primitive.hasAuthoredTangents) << primitive.name;
        EXPECT_EQ(primitive.tangents.size(), primitive.positions.size()) << primitive.name;
        EXPECT_GE(primitive.materialIndex, 0) << primitive.name;
        EXPECT_LT(static_cast<size_t>(primitive.materialIndex), scene.materials().size()) << primitive.name;
    }

    const metallic::scene::RenderMaterial& kingBlack = scene.materials()[0];
    EXPECT_EQ(kingBlack.name, "King_Black");
    expectTextureInfo(kingBlack.normalTexture, 0, 0, identityUv, "King_Black normalTexture");
    expectTextureInfo(kingBlack.occlusionTexture, 1, 0, identityUv, "King_Black occlusionTexture");
    expectTextureInfo(kingBlack.baseColorTexture, 2, 0, identityUv, "King_Black baseColorTexture");
    expectTextureInfo(
        kingBlack.metallicRoughnessTexture,
        1,
        0,
        identityUv,
        "King_Black metallicRoughnessTexture");
    EXPECT_EQ(scene.textures()[1].imageIndex, 1);
    EXPECT_EQ(scene.textures()[2].imageIndex, 2);
    EXPECT_EQ(scene.images()[1].name, "King_black_ORM");
    EXPECT_EQ(scene.images()[1].uri, "King_black_ORM.jpg");
    EXPECT_EQ(scene.images()[2].name, "king_black_base_color");
    EXPECT_EQ(scene.images()[2].uri, "king_black_base_color.jpg");

    const metallic::scene::RenderMaterial& pawnTopWhite = scene.materials()[5];
    EXPECT_EQ(pawnTopWhite.name, "Pawn_Top_White");
    expectVec4(
        pawnTopWhite.baseColorFactor,
        float4(1.0f, 1.0f, 0.828000009f, 1.0f),
        "Pawn_Top_White baseColorFactor");
    expectTextureInfo(
        pawnTopWhite.baseColorTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "Pawn_Top_White baseColorTexture");
    expectTextureInfo(pawnTopWhite.normalTexture, 15, 0, identityUv, "Pawn_Top_White normalTexture");
    expectTextureInfo(
        pawnTopWhite.metallicRoughnessTexture,
        16,
        0,
        identityUv,
        "Pawn_Top_White metallicRoughnessTexture");
    expectTextureInfo(
        pawnTopWhite.occlusionTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "Pawn_Top_White occlusionTexture");
    EXPECT_TRUE(nearlyEqual(pawnTopWhite.transmissionFactor, 1.0f));
    EXPECT_TRUE(nearlyEqual(pawnTopWhite.thicknessFactor, 0.219999999f));
    EXPECT_TRUE(nearlyEqual(pawnTopWhite.attenuationDistance, 0.0f));
    expectVec3(
        pawnTopWhite.attenuationColor,
        float3(0.800000012f, 0.800000012f, 0.800000012f),
        "Pawn_Top_White attenuationColor");
    EXPECT_EQ(scene.textures()[15].imageIndex, 15);
    EXPECT_EQ(scene.images()[15].uri, "Pawn_normal.jpg");
    EXPECT_EQ(scene.textures()[16].imageIndex, 16);
    EXPECT_EQ(scene.images()[16].uri, "Pawn_ORM.jpg");

    const metallic::scene::RenderMaterial& bishopWhite = scene.materials()[14];
    EXPECT_EQ(bishopWhite.name, "Bishop_White");
    expectTextureInfo(bishopWhite.normalTexture, 40, 0, identityUv, "Bishop_White normalTexture");
    expectTextureInfo(bishopWhite.occlusionTexture, 41, 0, identityUv, "Bishop_White occlusionTexture");
    expectTextureInfo(
        bishopWhite.metallicRoughnessTexture,
        41,
        0,
        identityUv,
        "Bishop_White metallicRoughnessTexture");
    expectTextureInfo(bishopWhite.baseColorTexture, 42, 0, identityUv, "Bishop_White baseColorTexture");
    EXPECT_EQ(scene.textures()[41].imageIndex, 31);
    EXPECT_EQ(scene.images()[31].uri, "Bishop_white_ORM.jpg");
    EXPECT_EQ(scene.textures()[42].imageIndex, 32);
    EXPECT_EQ(scene.images()[32].uri, "bishop_white_base_color.jpg");
}

float vectorLength(const float3& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

float normalAngleDegrees(const float3& lhs, const float3& rhs)
{
    const float lhsLength = std::max(vectorLength(lhs), 0.000001f);
    const float rhsLength = std::max(vectorLength(rhs), 0.000001f);
    const float normalDot = std::clamp(
        (lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z) / (lhsLength * rhsLength),
        -1.0f,
        1.0f);
    constexpr float kRadiansToDegrees = 57.29577951308232f;
    return std::acos(normalDot) * kRadiansToDegrees;
}

std::string quantizedPositionKey(const float3& position)
{
    constexpr double kScale = 1000000.0;
    return std::to_string(static_cast<int64_t>(std::llround(static_cast<double>(position.x) * kScale))) + "," +
        std::to_string(static_cast<int64_t>(std::llround(static_cast<double>(position.y) * kScale))) + "," +
        std::to_string(static_cast<int64_t>(std::llround(static_cast<double>(position.z) * kScale)));
}

float maxDuplicatePositionNormalAngle(const metallic::scene::RenderPrimitive& primitive)
{
    std::unordered_map<std::string, std::vector<size_t>> positionGroups;
    positionGroups.reserve(primitive.positions.size());
    for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
        positionGroups[quantizedPositionKey(primitive.positions[vertexIndex])].push_back(vertexIndex);
    }

    float maxAngle = 0.0f;
    for (const auto& [_, indices] : positionGroups) {
        for (size_t first = 0; first < indices.size(); ++first) {
            for (size_t second = first + 1; second < indices.size(); ++second) {
                maxAngle = std::max(
                    maxAngle,
                    normalAngleDegrees(
                        primitive.normals[indices[first]],
                        primitive.normals[indices[second]]));
            }
        }
    }
    return maxAngle;
}

void testABeautifulGameNormalData()
{
    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"))
        << scene.lastLoadResult().error;

    ASSERT_FALSE(scene.renderPrimitives().empty());
    size_t authoredNormalPrimitiveCount = 0;
    size_t pawnBodyPrimitiveCount = 0;
    float pawnBodyMaxDuplicateNormalAngle = 0.0f;
    for (const metallic::scene::RenderPrimitive& primitive : scene.renderPrimitives()) {
        EXPECT_TRUE(primitive.hasAuthoredNormals) << primitive.name;
        ASSERT_EQ(primitive.normals.size(), primitive.positions.size()) << primitive.name;
        for (const float3& normal : primitive.normals) {
            EXPECT_TRUE(nearlyEqual(vectorLength(normal), 1.0f, 0.001f)) << primitive.name;
        }
        if (primitive.hasAuthoredNormals) {
            ++authoredNormalPrimitiveCount;
        }
        if (primitive.name == "Pawn_Body_Shared") {
            ++pawnBodyPrimitiveCount;
            pawnBodyMaxDuplicateNormalAngle = std::max(
                pawnBodyMaxDuplicateNormalAngle,
                maxDuplicatePositionNormalAngle(primitive));
        }
    }

    EXPECT_EQ(authoredNormalPrimitiveCount, scene.renderPrimitives().size());
    EXPECT_GT(pawnBodyPrimitiveCount, 0u);
    EXPECT_LT(pawnBodyMaxDuplicateNormalAngle, 0.25f)
        << "Pawn body duplicate-position normals should be continuous across authored smoothing splits";
}

void testGlbImport(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path glbPath = writeMinimalGlbScene(directory);
    expect(scene.load(glbPath), scene.lastLoadResult().error);

    expect(scene.sceneName() == "GLB Scene", "GLB scene name");
    expect(scene.nodes().size() == 1, "GLB node count");
    expect(scene.nodes().front().name == "GLB Node", "GLB node name");
    expect(scene.cameras().size() == 1, "GLB scene should get fallback camera");
}

void testFallbackCamera(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeFallbackScene(directory);
    expect(scene.load(gltfPath), scene.lastLoadResult().error);

    expect(scene.sceneIndex() == 0, "missing default scene should fall back to scene 0");
    expect(scene.cameras().size() == 1, "fallback scene should expose one camera");
    expect(scene.cameras().front().fallback, "camera should be marked as fallback");
    const metallic::scene::ConstSceneObject fallbackObject =
        scene.sceneGraph().object(scene.cameras().front().object);
    expect(static_cast<bool>(fallbackObject), "fallback camera should be a scene object");
    expect(
        fallbackObject.hasComponent<metallic::scene::GeneratedComponent>(),
        "fallback camera generated component");
    expect(
        fallbackObject.hasComponent<metallic::scene::CameraComponent>(),
        "fallback camera component");
    expect(
        !fallbackObject.hasComponent<metallic::scene::SourceNodeComponent>(),
        "fallback camera should not consume a stable source-node slot");
    expect(
        scene.cameras().front().type == metallic::scene::CameraType::Perspective,
        "fallback camera should be perspective");
}

void testUnsupportedRequiredExtension(const std::filesystem::path& directory)
{
    auto expectUnsupportedRequired = [&](const std::string& extension) {
        metallic::scene::Scene scene;
        const std::filesystem::path gltfPath = writeUnsupportedRequiredExtensionScene(directory, extension);
        expect(!scene.load(gltfPath), "unsupported required extension should fail: " + extension);
        expect(!scene.valid(), "failed scene should not be valid: " + extension);
        expect(
            scene.lastLoadResult().error.find(extension) != std::string::npos,
            "unsupported extension error should mention extension name: " + extension);
    };

    expectUnsupportedRequired("EXT_meshopt_compression");
    const std::vector<std::string> unsupportedMaterialExtensions{
        "KHR_materials_anisotropy",
        "KHR_materials_clearcoat",
        "KHR_materials_dispersion",
        "KHR_materials_iridescence",
        "KHR_materials_pbrSpecularGlossiness",
        "KHR_materials_sheen",
        "KHR_materials_specular",
        "KHR_materials_unlit",
        "KHR_materials_volume_scatter",
    };
    for (const std::string& extension : unsupportedMaterialExtensions) {
        expectUnsupportedRequired(extension);
    }
}

void testGeneratedTangentHandedness(const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeGeneratedTangentScene(directory);
    ASSERT_TRUE(scene.load(gltfPath)) << scene.lastLoadResult().error;

    ASSERT_EQ(scene.renderPrimitives().size(), 1u);
    const metallic::scene::RenderPrimitive& primitive = scene.renderPrimitives().front();
    EXPECT_FALSE(primitive.hasAuthoredTangents);
    ASSERT_EQ(primitive.tangents.size(), 3u);

    for (const float4& tangent : primitive.tangents) {
        expectVec3(
            float3(tangent.x, tangent.y, tangent.z),
            float3(0.0f, 1.0f, 0.0f),
            "generated tangent direction");
        EXPECT_LT(tangent.w, 0.0f) << "generated tangent handedness";
    }
}

void testMutableSceneTransforms(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = writeFullScene(directory);
    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(gltfPath)) << scene.lastLoadResult().error;
    ASSERT_EQ(scene.nodes().size(), 5u);
    ASSERT_EQ(scene.renderNodes().size(), 1u);
    ASSERT_EQ(scene.cameras().size(), 2u);
    ASSERT_EQ(scene.lights().size(), 1u);

    const float4x4 authoredRoot = scene.nodes()[0].authoredLocalMatrix;
    float4x4 editedRoot = float4x4::Identity();
    editedRoot.SetupByTranslation(float3(2.0f, 3.0f, 4.0f));
    EXPECT_TRUE(scene.setNodeLocalMatrix(0, editedRoot));
    EXPECT_EQ(scene.transformRevision(), 1u);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(scene.nodes()[0].authoredLocalMatrix, authoredRoot));
    EXPECT_EQ(scene.nodes()[0].transformRevision, 1u);
    EXPECT_EQ(scene.nodes()[1].transformRevision, 1u);
    EXPECT_EQ(scene.renderNodes()[0].transformRevision, 1u);
    const metallic::scene::TransformComponent& rootTransform =
        scene.objectForNode(0).getComponent<metallic::scene::TransformComponent>();
    const metallic::scene::TransformComponent& meshTransform =
        scene.objectForNode(1).getComponent<metallic::scene::TransformComponent>();
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(rootTransform.localMatrix, editedRoot));
    EXPECT_EQ(rootTransform.transformRevision, 1u);
    EXPECT_EQ(meshTransform.transformRevision, 1u);
    expectVec3(
        float3(
            scene.nodes()[1].worldMatrix.a03,
            scene.nodes()[1].worldMatrix.a13,
            scene.nodes()[1].worldMatrix.a23),
        float3(6.0f, 3.0f, 4.0f),
        "child world transform propagated");
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.renderNodes()[0].worldMatrix,
        scene.nodes()[1].worldMatrix));
    expectVec3(scene.cameras()[0].eye, float3(2.0f, 3.0f, 14.0f), "camera transform propagated");
    expectVec3(
        float3(
            scene.lights()[0].worldMatrix.a03,
            scene.lights()[0].worldMatrix.a13,
            scene.lights()[0].worldMatrix.a23),
        float3(2.0f, 3.0f, 4.0f),
        "light transform propagated");
    ASSERT_TRUE(scene.bounds().valid);
    expectVec3(scene.bounds().min, float3(6.0f, 3.0f, 4.0f), "scene bounds min updated");
    expectVec3(scene.bounds().max, float3(7.0f, 4.0f, 4.0f), "scene bounds max updated");

    float4x4 editedMeshWorld = scene.nodes()[1].worldMatrix;
    editedMeshWorld.a03 += 2.0f;
    editedMeshWorld.a13 += 1.0f;
    const metallic::scene::SceneEntity meshObject = scene.objectForNode(1).entity();
    ASSERT_TRUE(scene.setObjectWorldMatrix(meshObject, editedMeshWorld));
    EXPECT_EQ(scene.transformRevision(), 2u);
    expectVec3(
        float3(
            scene.objectForNode(1).getComponent<metallic::scene::TransformComponent>().localMatrix.a03,
            scene.objectForNode(1).getComponent<metallic::scene::TransformComponent>().localMatrix.a13,
            scene.objectForNode(1).getComponent<metallic::scene::TransformComponent>().localMatrix.a23),
        float3(6.0f, 1.0f, 0.0f),
        "object world edit converted to parent-relative local transform");
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        scene.renderNodes()[0].worldMatrix,
        editedMeshWorld));

    EXPECT_FALSE(scene.setNodeLocalMatrix(0, editedRoot));
    EXPECT_EQ(scene.transformRevision(), 2u);
    EXPECT_FALSE(scene.setNodeLocalMatrix(-1, editedRoot));
}

void testMutableSceneVisibility(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = writeFullScene(directory);
    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(gltfPath)) << scene.lastLoadResult().error;
    ASSERT_EQ(scene.nodes().size(), 5u);
    ASSERT_EQ(scene.renderNodes().size(), 1u);

    const metallic::scene::SceneEntity rootObject = scene.objectForNode(0).entity();
    const metallic::scene::SceneEntity meshObject = scene.objectForNode(1).entity();
    const uint64_t initialVisibilityRevision = scene.visibilityRevision();
    const uint64_t initialStructuralRevision = scene.sceneGraph().structuralRevision();
    const uint64_t initialTransformRevision = scene.transformRevision();

    ASSERT_TRUE(scene.setObjectVisible(rootObject, false));
    EXPECT_EQ(scene.visibilityRevision(), initialVisibilityRevision + 1u);
    EXPECT_EQ(scene.sceneGraph().structuralRevision(), initialStructuralRevision);
    EXPECT_EQ(scene.transformRevision(), initialTransformRevision);
    EXPECT_FALSE(scene.nodes()[0].visible);
    EXPECT_FALSE(scene.nodes()[1].visible);
    EXPECT_FALSE(scene.renderNodes()[0].visible);
    EXPECT_FALSE(
        scene.objectForNode(0).getComponent<metallic::scene::VisibilityComponent>().worldVisible);
    EXPECT_FALSE(
        scene.objectForNode(1).getComponent<metallic::scene::VisibilityComponent>().worldVisible);

    EXPECT_FALSE(scene.setObjectVisible(rootObject, false));
    EXPECT_EQ(scene.visibilityRevision(), initialVisibilityRevision + 1u);

    ASSERT_TRUE(scene.setObjectVisible(rootObject, true));
    EXPECT_EQ(scene.visibilityRevision(), initialVisibilityRevision + 2u);
    EXPECT_TRUE(scene.nodes()[0].visible);
    EXPECT_TRUE(scene.nodes()[1].visible);
    EXPECT_TRUE(scene.renderNodes()[0].visible);

    ASSERT_TRUE(scene.setObjectVisible(meshObject, false));
    EXPECT_EQ(scene.visibilityRevision(), initialVisibilityRevision + 3u);
    EXPECT_TRUE(scene.nodes()[0].visible);
    EXPECT_FALSE(scene.nodes()[1].visible);
    EXPECT_FALSE(scene.renderNodes()[0].visible);
    EXPECT_TRUE(
        scene.objectForNode(0).getComponent<metallic::scene::VisibilityComponent>().localVisible);
    EXPECT_FALSE(
        scene.objectForNode(1).getComponent<metallic::scene::VisibilityComponent>().localVisible);
    EXPECT_EQ(scene.sceneGraph().structuralRevision(), initialStructuralRevision);
    EXPECT_EQ(scene.transformRevision(), initialTransformRevision);

    EXPECT_FALSE(scene.setObjectVisible(metallic::scene::kNullSceneEntity, true));
    EXPECT_EQ(scene.visibilityRevision(), initialVisibilityRevision + 3u);
}

void testSceneDocumentRoundTrip(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "scene_document";
    std::filesystem::create_directories(directory);
    const std::filesystem::path gltfPath = writeFullScene(directory);
    const std::filesystem::path sidecarPath =
        metallic::scene::SceneDocument::sidecarPathForSource(gltfPath);
    std::error_code cleanupError;
    std::filesystem::remove(sidecarPath, cleanupError);
    std::filesystem::remove(sidecarPath.string() + ".tmp", cleanupError);

    metallic::scene::SceneDocument unsavedDocument;
    ASSERT_TRUE(unsavedDocument.load(gltfPath)) << unsavedDocument.lastLoadResult().error;
    float4x4 discarded = unsavedDocument.nodes()[1].localMatrix;
    discarded.a03 += 25.0f;
    ASSERT_TRUE(unsavedDocument.setObjectLocalMatrix(
        unsavedDocument.objectForNode(1).entity(),
        discarded));
    ASSERT_TRUE(unsavedDocument.dirty());
    std::string revertMessage;
    ASSERT_TRUE(unsavedDocument.revert(revertMessage)) << revertMessage;
    EXPECT_FALSE(unsavedDocument.dirty());
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        unsavedDocument.nodes()[1].localMatrix,
        unsavedDocument.nodes()[1].authoredLocalMatrix));

    const std::filesystem::path failedRevertDirectory = baseDirectory / "failed_scene_revert";
    std::filesystem::create_directories(failedRevertDirectory);
    const std::filesystem::path failedRevertSource = writeFullScene(failedRevertDirectory);
    metallic::scene::SceneDocument preservedDocument;
    ASSERT_TRUE(preservedDocument.load(failedRevertSource))
        << preservedDocument.lastLoadResult().error;
    float4x4 preservedEdit = preservedDocument.nodes()[1].localMatrix;
    preservedEdit.a13 += 17.0f;
    ASSERT_TRUE(preservedDocument.setObjectLocalMatrix(
        preservedDocument.objectForNode(1).entity(),
        preservedEdit));
    ASSERT_TRUE(std::filesystem::remove(failedRevertSource));
    std::string failedRevertMessage;
    EXPECT_FALSE(preservedDocument.revert(failedRevertMessage));
    EXPECT_TRUE(preservedDocument.valid());
    EXPECT_TRUE(preservedDocument.dirty());
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        preservedDocument.nodes()[1].localMatrix,
        preservedEdit));

    metallic::scene::SceneDocument document;
    ASSERT_TRUE(document.load(gltfPath)) << document.lastLoadResult().error;
    EXPECT_FALSE(document.dirty());
    float4x4 edited = document.nodes()[1].localMatrix;
    edited.a03 = 9.0f;
    const metallic::scene::ConstSceneObject editedObject = document.objectForNode(1);
    const metallic::scene::RelationshipComponent& relationship =
        editedObject.getComponent<metallic::scene::RelationshipComponent>();
    ASSERT_NE(relationship.parent, metallic::scene::kNullSceneEntity);
    const float4x4 editedWorld =
        document.sceneGraph().object(relationship.parent)
            .getComponent<metallic::scene::TransformComponent>().worldMatrix * edited;
    EXPECT_TRUE(document.setObjectWorldMatrix(editedObject.entity(), editedWorld));
    EXPECT_TRUE(document.dirty());
    const std::filesystem::path environmentPath = directory / "lighting" / "studio.hdr";
    std::filesystem::create_directories(environmentPath.parent_path());
    writeTextFile(environmentPath, "test environment placeholder");
    EXPECT_TRUE(document.setEnvironment(metallic::scene::EnvironmentSettings{
        .enabled = true,
        .path = environmentPath,
        .intensity = 2.5f,
        .rotationDegrees = 37.0f,
        .visible = false,
    }));
    EXPECT_TRUE(document.dirty());
    std::string message;
    ASSERT_TRUE(document.save(message)) << message;
    EXPECT_FALSE(document.dirty());
    EXPECT_TRUE(std::filesystem::exists(sidecarPath));

    nlohmann::json saved;
    {
        std::ifstream stream(sidecarPath, std::ios::binary);
        stream >> saved;
    }
    ASSERT_EQ(saved.value("version", 0), 3);
    ASSERT_TRUE(saved.contains("nodes"));
    ASSERT_EQ(saved["nodes"].size(), 1u);
    EXPECT_EQ(saved["nodes"][0].value("nodeIndex", -1), 1);
    EXPECT_EQ(saved["nodes"][0].value("sourceName", std::string{}), "Mesh Node");
    ASSERT_TRUE(saved.contains("world"));
    ASSERT_TRUE(saved["world"].contains("environment"));
    const nlohmann::json& savedEnvironment = saved["world"]["environment"];
    EXPECT_EQ(savedEnvironment.value("path", std::string{}), "lighting/studio.hdr");
    EXPECT_FLOAT_EQ(savedEnvironment.value("intensity", 0.0f), 2.5f);
    EXPECT_FLOAT_EQ(savedEnvironment.value("rotationDegrees", 0.0f), 37.0f);
    EXPECT_FALSE(savedEnvironment.value("visible", true));

    metallic::scene::SceneDocument autoDiscovered;
    ASSERT_TRUE(autoDiscovered.load(gltfPath)) << autoDiscovered.lastLoadResult().error;
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(autoDiscovered.nodes()[1].localMatrix, edited));
    EXPECT_FALSE(autoDiscovered.dirty());
    EXPECT_TRUE(autoDiscovered.hasEnvironmentSettings());
    EXPECT_EQ(
        std::filesystem::weakly_canonical(autoDiscovered.environment().path),
        std::filesystem::weakly_canonical(environmentPath));
    EXPECT_FLOAT_EQ(autoDiscovered.environment().intensity, 2.5f);
    EXPECT_FLOAT_EQ(autoDiscovered.environment().rotationDegrees, 37.0f);
    EXPECT_FALSE(autoDiscovered.environment().visible);

    metallic::scene::SceneDocument directlyOpened;
    ASSERT_TRUE(directlyOpened.load(sidecarPath)) << directlyOpened.lastLoadResult().error;
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(directlyOpened.nodes()[1].localMatrix, edited));
    EXPECT_EQ(
        std::filesystem::weakly_canonical(directlyOpened.sourcePath()),
        std::filesystem::weakly_canonical(gltfPath));

    const nlohmann::json validSaved = saved;
    nlohmann::json versionTwo = validSaved;
    versionTwo["version"] = 2;
    writeTextFile(sidecarPath, versionTwo.dump(2));
    metallic::scene::SceneDocument versionTwoDocument;
    ASSERT_TRUE(versionTwoDocument.load(gltfPath)) << versionTwoDocument.lastLoadResult().error;
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        versionTwoDocument.nodes()[1].localMatrix,
        edited));
    EXPECT_TRUE(versionTwoDocument.hasEnvironmentSettings());

    nlohmann::json versionOne = validSaved;
    versionOne["version"] = 1;
    versionOne.erase("world");
    writeTextFile(sidecarPath, versionOne.dump(2));
    metallic::scene::SceneDocument versionOneDocument;
    ASSERT_TRUE(versionOneDocument.load(gltfPath)) << versionOneDocument.lastLoadResult().error;
    EXPECT_FALSE(versionOneDocument.hasEnvironmentSettings());
    EXPECT_TRUE(versionOneDocument.environment().path.empty());

    saved = validSaved;
    nlohmann::json outOfRangeOverride = saved["nodes"][0];
    outOfRangeOverride["nodeIndex"] = 9999;
    saved["nodes"].push_back(std::move(outOfRangeOverride));
    writeTextFile(sidecarPath, saved.dump(2));
    metallic::scene::SceneDocument partiallyApplied;
    ASSERT_TRUE(partiallyApplied.load(gltfPath)) << partiallyApplied.lastLoadResult().error;
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(partiallyApplied.nodes()[1].localMatrix, edited));
    EXPECT_NE(partiallyApplied.documentWarning().find("out-of-range"), std::string::npos);

    saved = validSaved;
    saved["source"] = "other.gltf";
    writeTextFile(sidecarPath, saved.dump(2));
    metallic::scene::SceneDocument wrongSource;
    EXPECT_FALSE(wrongSource.load(gltfPath));
    EXPECT_NE(wrongSource.documentWarning().find("does not match"), std::string::npos);

    saved = validSaved;
    saved["nodes"][0]["sourceName"] = "Renamed In Source";
    writeTextFile(sidecarPath, saved.dump(2));
    metallic::scene::SceneDocument mismatched;
    ASSERT_TRUE(mismatched.load(gltfPath)) << mismatched.lastLoadResult().error;
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        mismatched.nodes()[1].localMatrix,
        mismatched.nodes()[1].authoredLocalMatrix));
    EXPECT_NE(mismatched.documentWarning().find("sourceName"), std::string::npos);
}

void testSceneDocumentPropertyRoundTrip(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "scene_property_document";
    std::filesystem::create_directories(directory);
    const std::filesystem::path gltfPath = writeEditablePropertiesScene(directory);
    const std::filesystem::path sidecarPath =
        metallic::scene::SceneDocument::sidecarPathForSource(gltfPath);
    std::error_code cleanupError;
    std::filesystem::remove(sidecarPath, cleanupError);
    std::filesystem::remove(sidecarPath.string() + ".tmp", cleanupError);

    metallic::scene::SceneDocument document;
    ASSERT_TRUE(document.load(gltfPath)) << document.lastLoadResult().error;
    ASSERT_EQ(document.cameras().size(), 2u);
    ASSERT_EQ(document.lights().size(), 1u);

    constexpr int32_t kPerspectiveNodeIndex = 0;
    constexpr int32_t kOrthographicNodeIndex = 1;
    constexpr int32_t kLightNodeIndex = 2;
    const metallic::scene::ConstSceneObject perspectiveObject =
        document.objectForNode(kPerspectiveNodeIndex);
    const metallic::scene::ConstSceneObject orthographicObject =
        document.objectForNode(kOrthographicNodeIndex);
    const metallic::scene::ConstSceneObject lightObject =
        document.objectForNode(kLightNodeIndex);
    ASSERT_TRUE(perspectiveObject);
    ASSERT_TRUE(orthographicObject);
    ASSERT_TRUE(lightObject);

    const metallic::scene::CameraComponent& perspectiveComponent =
        perspectiveObject.getComponent<metallic::scene::CameraComponent>();
    const metallic::scene::CameraComponent& orthographicComponent =
        orthographicObject.getComponent<metallic::scene::CameraComponent>();
    const metallic::scene::LightComponent& lightComponent =
        lightObject.getComponent<metallic::scene::LightComponent>();
    const metallic::scene::CameraProperties authoredPerspective =
        perspectiveComponent.authoredProperties;
    const metallic::scene::CameraProperties authoredOrthographic =
        orthographicComponent.authoredProperties;
    const metallic::scene::LightProperties authoredLight = lightComponent.authoredProperties;
    expectCameraProperties(
        perspectiveComponent.properties,
        authoredPerspective,
        "initial perspective component");
    expectCameraProperties(
        orthographicComponent.properties,
        authoredOrthographic,
        "initial orthographic component");
    expectLightProperties(lightComponent.properties, authoredLight, "initial light component");

    const uint64_t initialContentRevision = document.sceneGraph().contentRevision();
    const uint64_t initialTransformRevision = document.transformRevision();

    metallic::scene::CameraProperties invalidPerspective = authoredPerspective;
    invalidPerspective.znear = 0.0;
    EXPECT_FALSE(document.setObjectCameraProperties(
        perspectiveObject.entity(),
        invalidPerspective));
    metallic::scene::CameraProperties unsupportedPerspective = authoredPerspective;
    unsupportedPerspective.yfov += 0.1;
    unsupportedPerspective.xmag += 0.0000005;
    EXPECT_FALSE(document.setObjectCameraProperties(
        perspectiveObject.entity(),
        unsupportedPerspective));
    metallic::scene::LightProperties invalidLight = authoredLight;
    invalidLight.innerConeAngle = invalidLight.outerConeAngle;
    EXPECT_FALSE(document.setObjectLightProperties(lightObject.entity(), invalidLight));
    EXPECT_FALSE(document.setObjectCameraProperties(lightObject.entity(), authoredPerspective));
    EXPECT_FALSE(document.setObjectLightProperties(
        perspectiveObject.entity(),
        authoredLight));
    EXPECT_FALSE(document.dirty());
    EXPECT_EQ(document.sceneGraph().contentRevision(), initialContentRevision);
    EXPECT_EQ(document.transformRevision(), initialTransformRevision);
    expectCameraProperties(
        perspectiveComponent.properties,
        authoredPerspective,
        "rejected perspective component");
    expectLightProperties(lightComponent.properties, authoredLight, "rejected light component");
    expectCameraSnapshot(
        document.cameras()[static_cast<size_t>(perspectiveComponent.renderCameraIndex)],
        authoredPerspective,
        "rejected perspective snapshot");
    expectLightSnapshot(
        document.lights()[static_cast<size_t>(lightComponent.renderLightIndex)],
        authoredLight,
        "rejected light snapshot");

    metallic::scene::CameraProperties editedPerspective = authoredPerspective;
    editedPerspective.yfov = 1.1;
    editedPerspective.aspectRatio = 1.5;
    editedPerspective.znear = 0.25;
    editedPerspective.zfar = 500.0;
    metallic::scene::CameraProperties editedOrthographic = authoredOrthographic;
    editedOrthographic.xmag = 8.0;
    editedOrthographic.ymag = 6.0;
    editedOrthographic.znear = 0.05;
    editedOrthographic.zfar = 90.0;
    metallic::scene::LightProperties editedLight = authoredLight;
    editedLight.color = float3(0.2f, 0.4f, 0.8f);
    editedLight.intensity = 12.5;
    editedLight.range = 25.0;
    editedLight.innerConeAngle = 0.2;
    editedLight.outerConeAngle = 1.57079632679489661923;

    ASSERT_TRUE(document.setObjectCameraProperties(
        perspectiveObject.entity(),
        editedPerspective));
    EXPECT_EQ(
        document.sceneGraph().contentRevision(),
        initialContentRevision + 1u);
    expectCameraProperties(
        perspectiveComponent.properties,
        editedPerspective,
        "immediate perspective component");
    expectCameraSnapshot(
        document.cameras()[static_cast<size_t>(perspectiveComponent.renderCameraIndex)],
        editedPerspective,
        "immediate perspective snapshot");
    ASSERT_TRUE(document.setObjectCameraProperties(
        orthographicObject.entity(),
        editedOrthographic));
    EXPECT_EQ(
        document.sceneGraph().contentRevision(),
        initialContentRevision + 2u);
    expectCameraProperties(
        orthographicComponent.properties,
        editedOrthographic,
        "immediate orthographic component");
    expectCameraSnapshot(
        document.cameras()[static_cast<size_t>(orthographicComponent.renderCameraIndex)],
        editedOrthographic,
        "immediate orthographic snapshot");
    ASSERT_TRUE(document.setObjectLightProperties(lightObject.entity(), editedLight));
    EXPECT_EQ(
        document.sceneGraph().contentRevision(),
        initialContentRevision + 3u);
    expectLightProperties(
        lightComponent.properties,
        editedLight,
        "immediate light component");
    expectLightSnapshot(
        document.lights()[static_cast<size_t>(lightComponent.renderLightIndex)],
        editedLight,
        "immediate light snapshot");
    EXPECT_TRUE(document.dirty());
    EXPECT_EQ(document.transformRevision(), initialTransformRevision);
    EXPECT_EQ(
        document.sceneGraph().contentRevision(),
        initialContentRevision + 3u);
    expectCameraProperties(
        perspectiveComponent.properties,
        editedPerspective,
        "edited perspective component");
    expectCameraProperties(
        orthographicComponent.properties,
        editedOrthographic,
        "edited orthographic component");
    expectLightProperties(lightComponent.properties, editedLight, "edited light component");
    expectCameraSnapshot(
        document.cameras()[static_cast<size_t>(perspectiveComponent.renderCameraIndex)],
        editedPerspective,
        "edited perspective snapshot");
    expectCameraSnapshot(
        document.cameras()[static_cast<size_t>(orthographicComponent.renderCameraIndex)],
        editedOrthographic,
        "edited orthographic snapshot");
    expectLightSnapshot(
        document.lights()[static_cast<size_t>(lightComponent.renderLightIndex)],
        editedLight,
        "edited light snapshot");
    expectCameraProperties(
        perspectiveComponent.authoredProperties,
        authoredPerspective,
        "preserved authored perspective");
    expectCameraProperties(
        orthographicComponent.authoredProperties,
        authoredOrthographic,
        "preserved authored orthographic");
    expectLightProperties(
        lightComponent.authoredProperties,
        authoredLight,
        "preserved authored light");

    std::string saveMessage;
    ASSERT_TRUE(document.save(saveMessage)) << saveMessage;
    EXPECT_FALSE(document.dirty());
    ASSERT_TRUE(std::filesystem::exists(sidecarPath));

    nlohmann::json saved;
    {
        std::ifstream stream(sidecarPath, std::ios::binary);
        stream >> saved;
    }
    ASSERT_EQ(saved.value("version", 0), 3);
    ASSERT_TRUE(saved.contains("nodes"));
    ASSERT_TRUE(saved["nodes"].is_array());
    ASSERT_EQ(saved["nodes"].size(), 3u);
    const auto findOverride = [&](int32_t nodeIndex) -> const nlohmann::json* {
        for (const nlohmann::json& value : saved["nodes"]) {
            if (value.is_object() && value.value("nodeIndex", -1) == nodeIndex) {
                return &value;
            }
        }
        return nullptr;
    };

    const nlohmann::json* savedPerspective = findOverride(kPerspectiveNodeIndex);
    const nlohmann::json* savedOrthographic = findOverride(kOrthographicNodeIndex);
    const nlohmann::json* savedLight = findOverride(kLightNodeIndex);
    ASSERT_NE(savedPerspective, nullptr);
    ASSERT_NE(savedOrthographic, nullptr);
    ASSERT_NE(savedLight, nullptr);
    EXPECT_FALSE(savedPerspective->contains("localMatrix"));
    EXPECT_FALSE(savedOrthographic->contains("localMatrix"));
    EXPECT_FALSE(savedLight->contains("localMatrix"));
    ASSERT_TRUE(savedPerspective->contains("camera"));
    ASSERT_TRUE(savedOrthographic->contains("camera"));
    ASSERT_TRUE(savedLight->contains("light"));
    const nlohmann::json& savedPerspectiveCamera = (*savedPerspective)["camera"];
    EXPECT_EQ(savedPerspectiveCamera.value("type", std::string{}), "perspective");
    EXPECT_DOUBLE_EQ(savedPerspectiveCamera.value("yfov", 0.0), editedPerspective.yfov);
    EXPECT_DOUBLE_EQ(
        savedPerspectiveCamera.value("aspectRatio", 0.0),
        editedPerspective.aspectRatio);
    EXPECT_DOUBLE_EQ(savedPerspectiveCamera.value("znear", 0.0), editedPerspective.znear);
    EXPECT_DOUBLE_EQ(savedPerspectiveCamera.value("zfar", 0.0), editedPerspective.zfar);
    EXPECT_FALSE(savedPerspectiveCamera.contains("xmag"));
    EXPECT_FALSE(savedPerspectiveCamera.contains("ymag"));
    const nlohmann::json& savedOrthographicCamera = (*savedOrthographic)["camera"];
    EXPECT_EQ(savedOrthographicCamera.value("type", std::string{}), "orthographic");
    EXPECT_DOUBLE_EQ(savedOrthographicCamera.value("xmag", 0.0), editedOrthographic.xmag);
    EXPECT_DOUBLE_EQ(savedOrthographicCamera.value("ymag", 0.0), editedOrthographic.ymag);
    EXPECT_DOUBLE_EQ(
        savedOrthographicCamera.value("znear", 0.0),
        editedOrthographic.znear);
    EXPECT_DOUBLE_EQ(savedOrthographicCamera.value("zfar", 0.0), editedOrthographic.zfar);
    EXPECT_FALSE(savedOrthographicCamera.contains("yfov"));
    EXPECT_FALSE(savedOrthographicCamera.contains("aspectRatio"));
    const nlohmann::json& savedLightProperties = (*savedLight)["light"];
    EXPECT_EQ(savedLightProperties.value("type", std::string{}), "spot");
    ASSERT_TRUE(savedLightProperties.contains("color"));
    ASSERT_EQ(savedLightProperties["color"].size(), 3u);
    EXPECT_FLOAT_EQ(savedLightProperties["color"][0].get<float>(), editedLight.color.x);
    EXPECT_FLOAT_EQ(savedLightProperties["color"][1].get<float>(), editedLight.color.y);
    EXPECT_FLOAT_EQ(savedLightProperties["color"][2].get<float>(), editedLight.color.z);
    EXPECT_DOUBLE_EQ(savedLightProperties.value("intensity", 0.0), editedLight.intensity);
    EXPECT_DOUBLE_EQ(savedLightProperties.value("range", 0.0), editedLight.range);
    EXPECT_DOUBLE_EQ(
        savedLightProperties.value("innerConeAngle", 0.0),
        editedLight.innerConeAngle);
    EXPECT_DOUBLE_EQ(
        savedLightProperties.value("outerConeAngle", 0.0),
        editedLight.outerConeAngle);

    const auto expectPersistedProperties = [&](const metallic::scene::SceneDocument& loaded) {
        const metallic::scene::ConstSceneObject loadedPerspective =
            loaded.objectForNode(kPerspectiveNodeIndex);
        const metallic::scene::ConstSceneObject loadedOrthographic =
            loaded.objectForNode(kOrthographicNodeIndex);
        const metallic::scene::ConstSceneObject loadedLight =
            loaded.objectForNode(kLightNodeIndex);
        ASSERT_TRUE(loadedPerspective);
        ASSERT_TRUE(loadedOrthographic);
        ASSERT_TRUE(loadedLight);
        const metallic::scene::CameraComponent& loadedPerspectiveComponent =
            loadedPerspective.getComponent<metallic::scene::CameraComponent>();
        const metallic::scene::CameraComponent& loadedOrthographicComponent =
            loadedOrthographic.getComponent<metallic::scene::CameraComponent>();
        const metallic::scene::LightComponent& loadedLightComponent =
            loadedLight.getComponent<metallic::scene::LightComponent>();
        expectCameraProperties(
            loadedPerspectiveComponent.properties,
            editedPerspective,
            "reloaded perspective component");
        expectCameraProperties(
            loadedOrthographicComponent.properties,
            editedOrthographic,
            "reloaded orthographic component");
        expectLightProperties(
            loadedLightComponent.properties,
            editedLight,
            "reloaded light component");
        expectCameraSnapshot(
            loaded.cameras()[static_cast<size_t>(
                loadedPerspectiveComponent.renderCameraIndex)],
            editedPerspective,
            "reloaded perspective snapshot");
        expectCameraSnapshot(
            loaded.cameras()[static_cast<size_t>(
                loadedOrthographicComponent.renderCameraIndex)],
            editedOrthographic,
            "reloaded orthographic snapshot");
        expectLightSnapshot(
            loaded.lights()[static_cast<size_t>(loadedLightComponent.renderLightIndex)],
            editedLight,
            "reloaded light snapshot");
        EXPECT_FALSE(loaded.dirty());
    };

    metallic::scene::SceneDocument autoDiscovered;
    ASSERT_TRUE(autoDiscovered.load(gltfPath)) << autoDiscovered.lastLoadResult().error;
    expectPersistedProperties(autoDiscovered);

    metallic::scene::SceneDocument directlyOpened;
    ASSERT_TRUE(directlyOpened.load(sidecarPath)) << directlyOpened.lastLoadResult().error;
    expectPersistedProperties(directlyOpened);

    metallic::scene::CameraProperties unsavedPerspective = editedPerspective;
    unsavedPerspective.yfov = 0.9;
    const metallic::scene::SceneEntity autoPerspectiveObject =
        autoDiscovered.objectForNode(kPerspectiveNodeIndex).entity();
    ASSERT_TRUE(autoDiscovered.setObjectCameraProperties(
        autoPerspectiveObject,
        unsavedPerspective));
    EXPECT_TRUE(autoDiscovered.dirty());
    std::string revertMessage;
    ASSERT_TRUE(autoDiscovered.revert(revertMessage)) << revertMessage;
    expectPersistedProperties(autoDiscovered);

    const std::filesystem::path fallbackDirectory = directory / "fallback";
    std::filesystem::create_directories(fallbackDirectory);
    const std::filesystem::path fallbackPath = writeFallbackScene(fallbackDirectory);
    const std::filesystem::path fallbackSidecar =
        metallic::scene::SceneDocument::sidecarPathForSource(fallbackPath);
    std::filesystem::remove(fallbackSidecar, cleanupError);
    std::filesystem::remove(fallbackSidecar.string() + ".tmp", cleanupError);
    metallic::scene::SceneDocument fallbackDocument;
    ASSERT_TRUE(fallbackDocument.load(fallbackPath))
        << fallbackDocument.lastLoadResult().error;
    ASSERT_EQ(fallbackDocument.cameras().size(), 1u);
    const metallic::scene::RenderCamera fallbackSnapshot = fallbackDocument.cameras().front();
    const metallic::scene::ConstSceneObject fallbackObject =
        fallbackDocument.sceneGraph().object(fallbackSnapshot.object);
    ASSERT_TRUE(fallbackObject);
    EXPECT_TRUE(fallbackObject.hasComponent<metallic::scene::GeneratedComponent>());
    EXPECT_FALSE(fallbackObject.hasComponent<metallic::scene::SourceNodeComponent>());
    const metallic::scene::CameraComponent& fallbackComponent =
        fallbackObject.getComponent<metallic::scene::CameraComponent>();
    const metallic::scene::CameraProperties fallbackOriginal = fallbackComponent.properties;
    metallic::scene::CameraProperties fallbackEdit = fallbackComponent.properties;
    fallbackEdit.yfov *= 0.8;
    const uint64_t fallbackContentRevision =
        fallbackDocument.sceneGraph().contentRevision();
    const uint64_t fallbackTransformRevision = fallbackDocument.transformRevision();
    EXPECT_FALSE(fallbackDocument.setObjectCameraProperties(
        fallbackObject.entity(),
        fallbackEdit));
    EXPECT_FALSE(fallbackDocument.dirty());
    EXPECT_EQ(
        fallbackDocument.sceneGraph().contentRevision(),
        fallbackContentRevision);
    EXPECT_EQ(fallbackDocument.transformRevision(), fallbackTransformRevision);
    expectCameraProperties(
        fallbackComponent.properties,
        fallbackOriginal,
        "generated fallback component");
    expectCameraSnapshot(
        fallbackDocument.cameras().front(),
        fallbackComponent.properties,
        "generated fallback snapshot");
}

void testCompositeSceneDocumentRoundTrip(const std::filesystem::path& baseDirectory)
{
    const std::filesystem::path directory = baseDirectory / "composite_scene_document";
    const std::filesystem::path sourceADirectory = directory / "source_a";
    const std::filesystem::path sourceBDirectory = directory / "source_b";
    std::filesystem::create_directories(sourceADirectory);
    std::filesystem::create_directories(sourceBDirectory);
    const std::filesystem::path sourceAPath = writeFullScene(sourceADirectory);
    const std::filesystem::path sourceBPath = writeFullScene(sourceBDirectory);
    const std::filesystem::path documentPath = directory / "world.metallic_scene.json";

    const auto serializeMatrix = [](const float4x4& matrix) {
        nlohmann::json value = nlohmann::json::array();
        for (const float component : matrix.a) {
            value.push_back(component);
        }
        return value;
    };
    float4x4 sourceBMount = float4x4::Identity();
    sourceBMount.SetupByTranslation(float3(20.0f, 0.0f, 0.0f));
    float4x4 sourceBMeshOverride = float4x4::Identity();
    sourceBMeshOverride.SetupByTranslation(float3(8.0f, 4.0f, 0.0f));
    nlohmann::json manifest{
        {"version", 4},
        {"sources", {
            {
                {"id", "source-a"},
                {"path", "source_a/scene.gltf"},
                {"mountMatrix", serializeMatrix(float4x4::Identity())},
                {"enabled", true},
            },
            {
                {"id", "source-b"},
                {"path", "source_b/scene.gltf"},
                {"mountMatrix", serializeMatrix(sourceBMount)},
                {"enabled", true},
            },
        }},
        {"nodes", {
            {
                {"sourceId", "source-b"},
                {"nodeIndex", 1},
                {"sourceName", "Mesh Node"},
                {"localMatrix", serializeMatrix(sourceBMeshOverride)},
            },
        }},
        {"world", {
            {"environment", {
                {"enabled", true},
                {"path", ""},
                {"intensity", 1.5f},
                {"rotationDegrees", 15.0f},
                {"visible", true},
            }},
        }},
    };
    writeTextFile(documentPath, manifest.dump(2));

    metallic::scene::SceneDocument document;
    ASSERT_TRUE(document.load(documentPath))
        << (document.documentWarning().empty()
                ? document.lastLoadResult().error
                : document.documentWarning());
    EXPECT_TRUE(document.sidecarLoaded());
    EXPECT_FALSE(document.dirty());
    ASSERT_EQ(document.sources().size(), 2u);
    EXPECT_EQ(document.sources()[0].id, "source-a");
    EXPECT_EQ(document.sources()[1].id, "source-b");
    EXPECT_EQ(
        std::filesystem::weakly_canonical(document.sourcePath()),
        std::filesystem::weakly_canonical(documentPath));
    EXPECT_EQ(
        std::filesystem::weakly_canonical(document.filename()),
        std::filesystem::weakly_canonical(documentPath));

    const metallic::scene::ConstSceneObject sourceAMesh =
        document.objectForSourceNode("source-a", 1);
    const metallic::scene::ConstSceneObject sourceBMesh =
        document.objectForSourceNode("source-b", 1);
    ASSERT_TRUE(sourceAMesh);
    ASSERT_TRUE(sourceBMesh);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        sourceAMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        sourceAMesh.getComponent<metallic::scene::TransformComponent>().authoredLocalMatrix));
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        sourceBMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        sourceBMeshOverride));
    ASSERT_EQ(document.renderNodes().size(), 2u);
    EXPECT_NEAR(document.renderNodes()[0].worldMatrix.a03, 5.0f, 0.0001f);
    EXPECT_NEAR(document.renderNodes()[1].worldMatrix.a03, 29.0f, 0.0001f);
    EXPECT_NEAR(document.renderNodes()[1].worldMatrix.a13, 6.0f, 0.0001f);
    const uint64_t visibilityRevision = document.visibilityRevision();
    ASSERT_TRUE(document.setSourceEnabled("source-b", false));
    EXPECT_TRUE(document.dirty());
    EXPECT_GT(document.visibilityRevision(), visibilityRevision);
    EXPECT_FALSE(document.renderNodes()[1].visible);

    std::string saveMessage;
    ASSERT_TRUE(document.save(saveMessage)) << saveMessage;
    nlohmann::json saved;
    {
        std::ifstream stream(documentPath, std::ios::binary);
        ASSERT_TRUE(stream.good());
        stream >> saved;
    }
    EXPECT_EQ(saved.value("version", 0), 4);
    EXPECT_FALSE(saved.contains("source"));
    ASSERT_TRUE(saved.contains("sources"));
    ASSERT_EQ(saved["sources"].size(), 2u);
    EXPECT_EQ(saved["sources"][0].value("path", std::string{}), "source_a/scene.gltf");
    EXPECT_EQ(saved["sources"][1].value("path", std::string{}), "source_b/scene.gltf");
    EXPECT_FALSE(saved["sources"][1].value("enabled", true));
    ASSERT_TRUE(saved.contains("nodes"));
    ASSERT_EQ(saved["nodes"].size(), 1u);
    EXPECT_EQ(saved["nodes"][0].value("sourceId", std::string{}), "source-b");
    EXPECT_EQ(saved["nodes"][0].value("nodeIndex", -1), 1);
    EXPECT_EQ(saved["nodes"][0].value("sourceName", std::string{}), "Mesh Node");

    metallic::scene::SceneDocument reloaded;
    ASSERT_TRUE(reloaded.load(documentPath))
        << (reloaded.documentWarning().empty()
                ? reloaded.lastLoadResult().error
                : reloaded.documentWarning());
    const metallic::scene::ConstSceneObject reloadedSourceAMesh =
        reloaded.objectForSourceNode("source-a", 1);
    const metallic::scene::ConstSceneObject reloadedSourceBMesh =
        reloaded.objectForSourceNode("source-b", 1);
    ASSERT_TRUE(reloadedSourceAMesh);
    ASSERT_TRUE(reloadedSourceBMesh);
    ASSERT_EQ(reloaded.sources().size(), 2u);
    EXPECT_FALSE(reloaded.sources()[1].enabled);
    ASSERT_EQ(reloaded.renderNodes().size(), 2u);
    EXPECT_FALSE(reloaded.renderNodes()[1].visible);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        reloadedSourceAMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        reloadedSourceAMesh.getComponent<metallic::scene::TransformComponent>().authoredLocalMatrix));
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        reloadedSourceBMesh.getComponent<metallic::scene::TransformComponent>().localMatrix,
        sourceBMeshOverride));

    float4x4 unsavedSourceAMesh =
        reloadedSourceAMesh.getComponent<metallic::scene::TransformComponent>().localMatrix;
    unsavedSourceAMesh.a23 += 3.0f;
    ASSERT_TRUE(reloaded.setObjectLocalMatrix(
        reloadedSourceAMesh.entity(),
        unsavedSourceAMesh));
    ASSERT_TRUE(reloaded.dirty());
    const uint64_t preservedLifetime = reloaded.sceneGraph().lifetimeRevision();
    const metallic::scene::SceneEntity preservedSourceBEntity =
        reloadedSourceBMesh.entity();
    const std::filesystem::path invalidDocumentPath =
        directory / "invalid_world.metallic_scene.json";
    nlohmann::json invalidManifest = saved;
    invalidManifest["sources"][1]["path"] = "missing.gltf";
    writeTextFile(invalidDocumentPath, invalidManifest.dump(2));
    EXPECT_FALSE(reloaded.load(invalidDocumentPath));
    EXPECT_TRUE(reloaded.valid());
    EXPECT_TRUE(reloaded.dirty());
    EXPECT_FALSE(reloaded.documentWarning().empty());
    EXPECT_EQ(reloaded.sceneGraph().lifetimeRevision(), preservedLifetime);
    EXPECT_EQ(
        reloaded.objectForSourceNode("source-b", 1).entity(),
        preservedSourceBEntity);
    EXPECT_TRUE(metallic::scene::matrixNearlyEqual(
        reloaded.objectForSourceNode("source-a", 1)
            .getComponent<metallic::scene::TransformComponent>().localMatrix,
        unsavedSourceAMesh));
}

void testScenePickerBvh(const std::filesystem::path& directory)
{
    const std::filesystem::path scenePath = writeFullScene(directory);
    nlohmann::json gltf;
    {
        std::ifstream stream(scenePath, std::ios::binary);
        stream >> gltf;
    }
    gltf["nodes"].push_back({
        {"name", "Far Mesh Node"},
        {"translation", {4.0f, 0.0f, -5.0f}},
        {"mesh", 0},
    });
    gltf["nodes"][0]["children"].push_back(5);
    writeTextFile(scenePath, gltf.dump(2));

    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(scenePath)) << scene.lastLoadResult().error;
    metallic::scene::ScenePicker picker;
    metallic::scene::ScenePickResult hit = picker.pick(
        scene,
        metallic::scene::ScenePickRay{
            .origin = float3(5.75f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
        });
    ASSERT_TRUE(hit.hit());
    EXPECT_EQ(hit.nodeIndex, 1);
    EXPECT_EQ(hit.object, scene.objectForNode(1).entity());
    EXPECT_EQ(hit.renderPrimitiveIndex, 0);
    EXPECT_EQ(hit.triangleIndex, 0u);
    EXPECT_TRUE(nearlyEqual(hit.distance, 7.0f));

    const metallic::scene::ScenePickResult farHit = picker.pick(
        scene,
        metallic::scene::ScenePickRay{
            .origin = float3(5.75f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
            .minimumDistance = 8.0f,
        });
    ASSERT_TRUE(farHit.hit());
    EXPECT_EQ(farHit.nodeIndex, 5);
    EXPECT_EQ(farHit.object, scene.objectForNode(5).entity());
    EXPECT_TRUE(nearlyEqual(farHit.distance, 12.0f));

    EXPECT_FALSE(picker.pick(
        scene,
        metallic::scene::ScenePickRay{
            .origin = float3(5.75f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
            .maximumDistance = 6.0f,
        }).hit());

    float4x4 movedRoot = scene.nodes()[0].localMatrix;
    movedRoot.a03 += 10.0f;
    ASSERT_TRUE(scene.setNodeLocalMatrix(0, movedRoot));
    EXPECT_FALSE(picker.pick(
        scene,
        metallic::scene::ScenePickRay{
            .origin = float3(5.75f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
        }).hit());
    hit = picker.pick(
        scene,
        metallic::scene::ScenePickRay{
            .origin = float3(15.75f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
        });
    ASSERT_TRUE(hit.hit());
    EXPECT_EQ(hit.nodeIndex, 1);

    metallic::scene::Scene otherScene;
    ASSERT_TRUE(otherScene.load(scene.filename())) << otherScene.lastLoadResult().error;
    float4x4 otherRoot = otherScene.nodes()[0].localMatrix;
    otherRoot.a03 -= 10.0f;
    ASSERT_TRUE(otherScene.setNodeLocalMatrix(0, otherRoot));
    EXPECT_EQ(otherScene.transformRevision(), scene.transformRevision());
    EXPECT_NE(
        otherScene.sceneGraph().lifetimeRevision(),
        scene.sceneGraph().lifetimeRevision());
    hit = picker.pick(
        otherScene,
        metallic::scene::ScenePickRay{
            .origin = float3(-4.25f, 2.25f, 10.0f),
            .direction = float3(0.0f, 0.0f, -1.0f),
        });
    ASSERT_TRUE(hit.hit());
    EXPECT_EQ(hit.nodeIndex, 1);
}

void waitForSceneLoad(metallic::scene::SceneLoadHandle& handle)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    float previousFraction = 0.0f;
    while (!handle.complete() && std::chrono::steady_clock::now() < deadline) {
        const metallic::scene::SceneLoadProgress progress = handle.progress();
        EXPECT_GE(progress.fraction, previousFraction);
        previousFraction = progress.fraction;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(handle.complete());
}

void testAsyncSceneLoad(const std::filesystem::path& directory)
{
    const auto initialization = metallic::task::initializeTaskSystem({.workerCount = 2});
    ASSERT_TRUE(initialization.has_value()) << initialization.error().message;
    struct ShutdownGuard {
        ~ShutdownGuard() { metallic::task::shutdownTaskSystem(); }
    } shutdownGuard;

    const std::filesystem::path path = writeFullScene(directory);
    metallic::scene::SceneDocument synchronous;
    ASSERT_TRUE(synchronous.load(path)) << synchronous.lastLoadResult().error;
    std::error_code cacheRemoveError;
    std::filesystem::remove(
        std::filesystem::path(path.string() + ".meshlets.bin"),
        cacheRemoveError);

    metallic::scene::SceneLoader loader;
    metallic::scene::SceneLoadHandle handle = loader.request(path);
    ASSERT_TRUE(handle.valid());
    waitForSceneLoad(handle);

    const metallic::scene::SceneLoadProgress progress = handle.progress();
    EXPECT_EQ(progress.status, metallic::scene::SceneLoadStatus::Succeeded);
    EXPECT_EQ(progress.phase, metallic::scene::SceneLoadPhase::Completed);
    EXPECT_FLOAT_EQ(progress.fraction, 1.0f);

    std::unique_ptr<metallic::scene::SceneDocument> loaded = handle.takeResult();
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->nodes().size(), synchronous.nodes().size());
    EXPECT_EQ(loaded->sceneGraph().size(), synchronous.sceneGraph().size());
    const metallic::scene::ConstSceneObject loadedMeshObject = loaded->objectForNode(1);
    ASSERT_TRUE(loadedMeshObject);
    EXPECT_EQ(
        loadedMeshObject.getComponent<metallic::scene::TagComponent>().name,
        "Mesh Node");
    EXPECT_EQ(loaded->renderPrimitives().size(), synchronous.renderPrimitives().size());
    EXPECT_EQ(loaded->stats().triangleCount, synchronous.stats().triangleCount);
    EXPECT_EQ(
        loaded->stats().meshletClusterCount,
        synchronous.stats().meshletClusterCount);
    EXPECT_TRUE(loaded->lastLoadResult().meshletCacheSaved);
    EXPECT_EQ(handle.takeResult(), nullptr);

    const std::filesystem::path compositeDirectory = directory / "async_composite";
    const std::filesystem::path sourceADirectory = compositeDirectory / "source_a";
    const std::filesystem::path sourceBDirectory = compositeDirectory / "source_b";
    std::filesystem::create_directories(sourceADirectory);
    std::filesystem::create_directories(sourceBDirectory);
    writeFullScene(sourceADirectory);
    writeFullScene(sourceBDirectory);
    const std::filesystem::path manifestPath =
        compositeDirectory / "world.metallic_scene.json";
    const auto serializeMatrix = [](const float4x4& matrix) {
        nlohmann::json value = nlohmann::json::array();
        for (const float component : matrix.a) {
            value.push_back(component);
        }
        return value;
    };
    writeTextFile(
        manifestPath,
        nlohmann::json{
            {"version", 4},
            {"sources", {
                {
                    {"id", "source-a"},
                    {"path", "source_a/scene.gltf"},
                    {"mountMatrix", serializeMatrix(float4x4::Identity())},
                    {"enabled", true},
                },
                {
                    {"id", "source-b"},
                    {"path", "source_b/scene.gltf"},
                    {"mountMatrix", serializeMatrix(float4x4::Identity())},
                    {"enabled", true},
                },
            }},
            {"nodes", nlohmann::json::array()},
        }.dump(2));
    std::filesystem::remove(
        std::filesystem::path((sourceADirectory / "scene.gltf").string() +
            ".meshlets.bin"),
        cacheRemoveError);
    std::filesystem::remove(
        std::filesystem::path((sourceBDirectory / "scene.gltf").string() +
            ".meshlets.bin"),
        cacheRemoveError);
    metallic::scene::SceneLoadHandle composite = loader.request(manifestPath);
    waitForSceneLoad(composite);
    std::unique_ptr<metallic::scene::SceneDocument> compositeScene =
        composite.takeResult();
    ASSERT_NE(compositeScene, nullptr);
    EXPECT_EQ(compositeScene->sources().size(), 2u);
    EXPECT_EQ(compositeScene->renderPrimitives().size(), 2u);
    EXPECT_FALSE(compositeScene->hasDeferredMeshlets());
    EXPECT_GT(compositeScene->stats().meshletClusterCount, 0u);
    EXPECT_TRUE(compositeScene->lastLoadResult().meshletCacheSaved);
    const std::filesystem::path sourceACache =
        std::filesystem::path((sourceADirectory / "scene.gltf").string() +
            ".meshlets.bin");
    const std::filesystem::path sourceBCache =
        std::filesystem::path((sourceBDirectory / "scene.gltf").string() +
            ".meshlets.bin");
    EXPECT_TRUE(std::filesystem::exists(sourceACache));
    EXPECT_TRUE(std::filesystem::exists(sourceBCache));

    metallic::scene::Scene cachedSourceA;
    ASSERT_TRUE(cachedSourceA.load(sourceADirectory / "scene.gltf"))
        << cachedSourceA.lastLoadResult().error;
    EXPECT_TRUE(cachedSourceA.lastLoadResult().meshletCacheLoaded);
    metallic::scene::Scene cachedSourceB;
    ASSERT_TRUE(cachedSourceB.load(sourceBDirectory / "scene.gltf"))
        << cachedSourceB.lastLoadResult().error;
    EXPECT_TRUE(cachedSourceB.lastLoadResult().meshletCacheLoaded);
    metallic::scene::SceneDocument cachedComposite;
    ASSERT_TRUE(cachedComposite.load(manifestPath))
        << cachedComposite.documentWarning();
    EXPECT_TRUE(cachedComposite.lastLoadResult().meshletCacheLoaded);

    const std::filesystem::path materialPath = writeMaterialFeatureScene(directory);
    metallic::scene::SceneLoadHandle images = loader.request(
        materialPath,
        metallic::scene::SceneLoadOptions{
            .decodeConcurrency = 2,
            .maxDecodedBytesInFlight = 1,
        });
    waitForSceneLoad(images);
    std::unique_ptr<metallic::scene::SceneDocument> imageScene = images.takeResult();
    ASSERT_NE(imageScene, nullptr);
    ASSERT_EQ(imageScene->images().size(), 9u);
    for (const metallic::scene::RenderImage& image : imageScene->images()) {
        EXPECT_TRUE(image.decodeAttempted);
    }

    metallic::scene::SceneLoadHandle cancelled = loader.request(path);
    ASSERT_TRUE(cancelled.cancel());
    waitForSceneLoad(cancelled);
    EXPECT_EQ(cancelled.progress().status, metallic::scene::SceneLoadStatus::Cancelled);
    EXPECT_EQ(cancelled.takeResult(), nullptr);
}

} // namespace

#if defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY
TEST(SceneImport, RtxcrClairePonytailDots)
{
    const std::filesystem::path path =
        std::filesystem::path(PROJECT_SOURCE_DIR) /
        "External/RTXCR-Assets/Claire/ponyTail_15vtx.gltf";
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "RTXCR-Assets submodule is not initialized";
    }

    metallic::scene::Scene scene;
    ASSERT_TRUE(scene.load(path)) << scene.lastLoadResult().error;
    ASSERT_EQ(scene.renderPrimitives().size(), 1u);
    ASSERT_EQ(scene.materials().size(), 1u);

    const metallic::scene::RenderPrimitive& primitive = scene.renderPrimitives().front();
    EXPECT_EQ(primitive.mode, 4);
    EXPECT_EQ(primitive.vertexCount, 180648u);
    EXPECT_EQ(primitive.indexCount, 180648u);
    EXPECT_EQ(primitive.triangleCount, 60216u);
    EXPECT_TRUE(primitive.hasAuthoredNormals);
    EXPECT_TRUE(primitive.hasAuthoredTangents);
    EXPECT_TRUE(primitive.localBounds.valid);
    EXPECT_NE(
        scene.lastLoadResult().meshletCachePath.string().find(".cache"),
        std::string::npos);

    const metallic::scene::RenderMaterial& material = scene.materials().front();
    EXPECT_TRUE(material.rtxcrHair);
    EXPECT_TRUE(nearlyEqual(material.rtxcrHairMelanin, 0.98f));
    EXPECT_TRUE(nearlyEqual(material.rtxcrHairLongitudinalRoughness, 0.15f));
    EXPECT_TRUE(nearlyEqual(material.rtxcrHairAzimuthalRoughness, 0.2f));
    EXPECT_TRUE(nearlyEqual(material.rtxcrHairIor, 1.55f));
}
#endif

TEST(SceneImport, FullScene)
{
    testFullSceneImport(prepareOutputDirectory());
}

TEST(SceneComposition, LoadsMultipleSourcesAndMaintainsGpuSnapshot)
{
    testCompositeSceneImport(prepareOutputDirectory());
}

TEST(SceneComposition, UsesOneFallbackCameraAndHonorsInitialDisabledSources)
{
    testCompositeFallbackCamera(prepareOutputDirectory());
}

TEST(SceneImport, RejectsInvalidHierarchy)
{
    testInvalidSceneHierarchy(prepareOutputDirectory());
}

TEST(SceneImport, MeshletLodPartition)
{
    testMeshletLodPartition(prepareOutputDirectory());
}

TEST(SceneImport, MeshletPersistence)
{
    testMeshletPersistence(prepareOutputDirectory());
}

TEST(SceneImport, MeshletStreamAsset)
{
    testMeshletStreamAsset(prepareOutputDirectory());
}

TEST(SceneImport, MeshoptCompressedMeshletStreamAsset)
{
    testMeshoptCompressedMeshletStreamAsset(prepareOutputDirectory());
}

TEST(SceneImport, Materials)
{
    testMaterialImport(prepareOutputDirectory());
}

TEST(SceneImport, ABeautifulGameMaterials)
{
    testABeautifulGameMaterialImport();
}

TEST(SceneImport, ABeautifulGameNormalData)
{
    testABeautifulGameNormalData();
}

TEST(SceneImport, GeneratedTangentHandedness)
{
    testGeneratedTangentHandedness(prepareOutputDirectory());
}

TEST(SceneImport, Glb)
{
    testGlbImport(prepareOutputDirectory());
}

TEST(SceneImport, FallbackCamera)
{
    testFallbackCamera(prepareOutputDirectory());
}

TEST(SceneImport, UnsupportedRequiredExtension)
{
    testUnsupportedRequiredExtension(prepareOutputDirectory());
}

TEST(SceneEditing, MutableTransforms)
{
    testMutableSceneTransforms(prepareOutputDirectory());
}

TEST(SceneEditing, MutableVisibility)
{
    testMutableSceneVisibility(prepareOutputDirectory());
}

TEST(SceneEditing, DocumentRoundTrip)
{
    testSceneDocumentRoundTrip(prepareOutputDirectory());
}

TEST(SceneEditing, ComponentPropertyRoundTrip)
{
    testSceneDocumentPropertyRoundTrip(prepareOutputDirectory());
}

TEST(SceneEditing, CompositeDocumentRoundTrip)
{
    testCompositeSceneDocumentRoundTrip(prepareOutputDirectory());
}

TEST(SceneEditing, CompositeDocumentValidation)
{
    testCompositeSceneDocumentValidation(prepareOutputDirectory());
}

TEST(SceneEditing, PickerBvh)
{
    testScenePickerBvh(prepareOutputDirectory());
}

TEST(SceneLoading, AsyncProgressAndCancellation)
{
    testAsyncSceneLoad(prepareOutputDirectory());
}
