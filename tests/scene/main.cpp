#include "Runtime/Scene/scene.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct TestContext {
    int failures = 0;

    void expect(bool condition, const std::string& message)
    {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }
};

bool nearlyEqual(float lhs, float rhs, float epsilon = 0.0001f)
{
    return std::fabs(lhs - rhs) <= epsilon;
}

void expectVec3(
    TestContext& test,
    const float3& actual,
    const float3& expected,
    const std::string& label)
{
    test.expect(
        nearlyEqual(actual.x, expected.x) &&
            nearlyEqual(actual.y, expected.y) &&
            nearlyEqual(actual.z, expected.z),
        label + " expected [" + metallic::scene::formatVec3(expected) + "] got [" +
            metallic::scene::formatVec3(actual) + "]");
}

void expectVec4(
    TestContext& test,
    const float4& actual,
    const float4& expected,
    const std::string& label)
{
    test.expect(
        nearlyEqual(actual.x, expected.x) &&
            nearlyEqual(actual.y, expected.y) &&
            nearlyEqual(actual.z, expected.z) &&
            nearlyEqual(actual.w, expected.w),
        label);
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
  "extensionsUsed": ["KHR_lights_punctual"],
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
      "doubleSided": true
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

std::filesystem::path writeUnsupportedRequiredExtensionScene(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = directory / "unsupported_required.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsRequired": ["EXT_meshopt_compression"],
  "extensionsUsed": ["EXT_meshopt_compression"],
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

void testFullSceneImport(TestContext& test, const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeFullScene(directory);
    test.expect(scene.load(gltfPath), scene.lastLoadResult().error);

    test.expect(scene.valid(), "loaded scene should be valid");
    test.expect(scene.sceneIndex() == 0, "default scene index should be 0");
    test.expect(scene.sceneName() == "Default Scene", "scene name should come from glTF scene");
    test.expect(scene.rootNodeIndices().size() == 1, "scene should expose one root node");
    test.expect(scene.rootNodeIndices().front() == 0, "root node index should be 0");
    test.expect(scene.nodes().size() == 5, "scene should load all nodes");
    test.expect(scene.nodes()[1].parent == 0, "child parent index should be assigned");
    test.expect(scene.nodes()[0].children.size() == 4, "root children should be preserved");

    const float3 meshWorldTranslation(
        scene.nodes()[1].worldMatrix.a03,
        scene.nodes()[1].worldMatrix.a13,
        scene.nodes()[1].worldMatrix.a23);
    expectVec3(test, meshWorldTranslation, float3(5.0f, 2.0f, 3.0f), "mesh node world translation");

    const metallic::scene::SceneStats& stats = scene.stats();
    test.expect(stats.meshCount == 1, "mesh count");
    test.expect(stats.materialCount == 2, "material count");
    test.expect(stats.primitiveCount == 1, "primitive count");
    test.expect(stats.renderNodeCount == 1, "render node count");
    test.expect(stats.triangleCount == 1, "triangle count");

    test.expect(scene.renderPrimitives().size() == 1, "primitive vector size");
    const metallic::scene::RenderPrimitive& primitive = scene.renderPrimitives().front();
    test.expect(primitive.vertexCount == 3, "primitive vertex count");
    test.expect(primitive.indexCount == 3, "primitive index count");
    test.expect(primitive.materialIndex == 0, "primitive material index");
    test.expect(primitive.positions.size() == 3, "primitive position data count");
    test.expect(primitive.indices.size() == 3, "primitive index data count");
    expectVec3(test, primitive.positions[2], float3(1.0f, 1.0f, 0.0f), "primitive position data");
    test.expect(primitive.indices[2] == 2, "primitive index data");

    test.expect(scene.materials().size() == 2, "material vector size");
    test.expect(scene.materials()[0].name == "Test Material", "default material name");
    expectVec4(
        test,
        scene.materials()[0].baseColorFactor,
        float4(1.0f, 1.0f, 1.0f, 1.0f),
        "default baseColorFactor");
    test.expect(scene.materials()[1].name == "Tinted Material", "tinted material name");
    expectVec4(
        test,
        scene.materials()[1].baseColorFactor,
        float4(0.25f, 0.5f, 0.75f, 0.8f),
        "explicit baseColorFactor");
    test.expect(std::abs(scene.materials()[1].metallicFactor - 0.35f) < 0.0001f, "explicit metallicFactor");
    test.expect(std::abs(scene.materials()[1].roughnessFactor - 0.6f) < 0.0001f, "explicit roughnessFactor");
    expectVec3(
        test,
        scene.materials()[1].emissiveFactor,
        float3(0.05f, 0.1f, 0.2f),
        "explicit emissiveFactor");
    test.expect(scene.materials()[1].doubleSided, "explicit doubleSided");

    test.expect(scene.bounds().valid, "bounds should be valid");
    expectVec3(test, scene.bounds().min, float3(5.0f, 2.0f, 3.0f), "scene bounds min");
    expectVec3(test, scene.bounds().max, float3(6.0f, 3.0f, 3.0f), "scene bounds max");

    test.expect(scene.cameras().size() == 2, "camera count");
    const metallic::scene::RenderCamera& perspectiveCamera = scene.cameras()[0];
    test.expect(
        perspectiveCamera.type == metallic::scene::CameraType::Perspective,
        "first camera should be perspective");
    test.expect(nearlyEqual(static_cast<float>(perspectiveCamera.yfov), 0.75f), "perspective yfov");
    test.expect(nearlyEqual(static_cast<float>(perspectiveCamera.aspectRatio), 1.7777778f), "perspective aspect");
    expectVec3(test, perspectiveCamera.eye, float3(1.0f, 2.0f, 13.0f), "perspective camera eye");
    expectVec3(test, perspectiveCamera.center, float3(1.0f, 2.0f, 12.0f), "perspective camera center");

    const metallic::scene::RenderCamera& orthoCamera = scene.cameras()[1];
    test.expect(
        orthoCamera.type == metallic::scene::CameraType::Orthographic,
        "second camera should be orthographic");
    test.expect(nearlyEqual(static_cast<float>(orthoCamera.xmag), 4.0f), "orthographic xmag");
    test.expect(nearlyEqual(static_cast<float>(orthoCamera.ymag), 3.0f), "orthographic ymag");

    test.expect(scene.lights().size() == 1, "punctual light count");
    test.expect(scene.lights().front().type == "directional", "punctual light type");
}

void testGlbImport(TestContext& test, const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path glbPath = writeMinimalGlbScene(directory);
    test.expect(scene.load(glbPath), scene.lastLoadResult().error);

    test.expect(scene.sceneName() == "GLB Scene", "GLB scene name");
    test.expect(scene.nodes().size() == 1, "GLB node count");
    test.expect(scene.nodes().front().name == "GLB Node", "GLB node name");
    test.expect(scene.cameras().size() == 1, "GLB scene should get fallback camera");
}

void testFallbackCamera(TestContext& test, const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeFallbackScene(directory);
    test.expect(scene.load(gltfPath), scene.lastLoadResult().error);

    test.expect(scene.sceneIndex() == 0, "missing default scene should fall back to scene 0");
    test.expect(scene.cameras().size() == 1, "fallback scene should expose one camera");
    test.expect(scene.cameras().front().fallback, "camera should be marked as fallback");
    test.expect(
        scene.cameras().front().type == metallic::scene::CameraType::Perspective,
        "fallback camera should be perspective");
}

void testUnsupportedRequiredExtension(TestContext& test, const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeUnsupportedRequiredExtensionScene(directory);
    test.expect(!scene.load(gltfPath), "unsupported required extension should fail");
    test.expect(!scene.valid(), "failed scene should not be valid");
    test.expect(
        scene.lastLoadResult().error.find("EXT_meshopt_compression") != std::string::npos,
        "unsupported extension error should mention extension name");
}

} // namespace

int main()
{
    const std::filesystem::path outputDirectory = prepareOutputDirectory();

    TestContext test;
    testFullSceneImport(test, outputDirectory);
    testGlbImport(test, outputDirectory);
    testFallbackCamera(test, outputDirectory);
    testUnsupportedRequiredExtension(test, outputDirectory);

    if (test.failures != 0) {
        std::cerr << test.failures << " scene test failure(s)\n";
        return 1;
    }

    std::cout << "MetallicSceneTests passed\n";
    return 0;
}
