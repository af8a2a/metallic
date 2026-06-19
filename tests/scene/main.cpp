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

void expectTextureInfo(
    TestContext& test,
    const metallic::scene::RenderTextureInfo& actual,
    int32_t expectedTextureIndex,
    int32_t expectedTexCoord,
    const std::array<float, 6>& expectedUvTransform,
    const std::string& label)
{
    test.expect(actual.textureIndex == expectedTextureIndex, label + " texture index");
    test.expect(actual.texCoord == expectedTexCoord, label + " texcoord");
    for (size_t index = 0; index < expectedUvTransform.size(); ++index) {
        test.expect(
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

std::filesystem::path writeMaterialFeatureScene(const std::filesystem::path& directory)
{
    const std::filesystem::path gltfPath = directory / "materials.gltf";
    writeTextFile(gltfPath, R"json(
{
  "asset": { "version": "2.0" },
  "scene": 0,
  "extensionsUsed": [
    "KHR_texture_transform",
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
      "normalTexture": { "index": 2, "texCoord": 0, "scale": 0.42 },
      "occlusionTexture": { "index": 3, "texCoord": 1, "strength": 0.66 },
      "emissiveTexture": { "index": 4, "texCoord": 0 },
      "emissiveFactor": [0.1, 0.2, 0.3],
      "alphaMode": "MASK",
      "alphaCutoff": 0.37,
      "doubleSided": true,
      "extensions": {
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
    test.expect(
        std::abs(scene.materials()[1].diffuseTransmissionFactor - 0.45f) < 0.0001f,
        "explicit diffuseTransmissionFactor");
    expectVec3(
        test,
        scene.materials()[1].diffuseTransmissionColor,
        float3(0.4f, 0.9f, 0.55f),
        "explicit diffuseTransmissionColor");

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

void testMaterialImport(TestContext& test, const std::filesystem::path& directory)
{
    metallic::scene::Scene scene;
    const std::filesystem::path gltfPath = writeMaterialFeatureScene(directory);
    test.expect(scene.load(gltfPath), scene.lastLoadResult().error);

    test.expect(scene.stats().materialCount == 4, "material feature scene material count");
    test.expect(scene.images().size() == 9, "material feature scene image count");
    test.expect(scene.images()[0].name == "Base Color Image", "base color image name");
    test.expect(scene.images()[0].uri == "base_color.png", "base color image uri");
    test.expect(scene.textures().size() == 9, "material feature scene texture count");
    test.expect(scene.textures()[2].imageIndex == 2, "normal texture image index");
    test.expect(scene.textures()[2].samplerIndex == 1, "normal texture sampler index");

    const std::array<float, 6> identityUv{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const metallic::scene::RenderMaterial& defaultMaterial = scene.materials()[0];
    expectVec4(
        test,
        defaultMaterial.baseColorFactor,
        float4(1.0f, 1.0f, 1.0f, 1.0f),
        "material default baseColorFactor");
    test.expect(nearlyEqual(defaultMaterial.metallicFactor, 1.0f), "material default metallicFactor");
    test.expect(nearlyEqual(defaultMaterial.roughnessFactor, 1.0f), "material default roughnessFactor");
    expectVec3(
        test,
        defaultMaterial.emissiveFactor,
        float3(0.0f, 0.0f, 0.0f),
        "material default emissiveFactor");
    test.expect(defaultMaterial.alphaMode == "OPAQUE", "material default alphaMode");
    test.expect(nearlyEqual(defaultMaterial.alphaCutoff, 0.5f), "material default alphaCutoff");
    test.expect(!defaultMaterial.doubleSided, "material default doubleSided");
    test.expect(nearlyEqual(defaultMaterial.normalTextureScale, 1.0f), "material default normal scale");
    test.expect(nearlyEqual(defaultMaterial.occlusionTextureStrength, 1.0f), "material default occlusion strength");
    test.expect(nearlyEqual(defaultMaterial.transmissionFactor, 0.0f), "material default transmissionFactor");
    test.expect(nearlyEqual(defaultMaterial.ior, 1.5f), "material default ior");
    test.expect(nearlyEqual(defaultMaterial.thicknessFactor, 0.0f), "material default thicknessFactor");
    test.expect(nearlyEqual(defaultMaterial.attenuationDistance, 0.0f), "material default attenuationDistance");
    expectVec3(
        test,
        defaultMaterial.attenuationColor,
        float3(1.0f, 1.0f, 1.0f),
        "material default attenuationColor");
    test.expect(
        nearlyEqual(defaultMaterial.diffuseTransmissionFactor, 0.0f),
        "material default diffuseTransmissionFactor");
    expectVec3(
        test,
        defaultMaterial.diffuseTransmissionColor,
        float3(1.0f, 1.0f, 1.0f),
        "material default diffuseTransmissionColor");
    expectTextureInfo(
        test,
        defaultMaterial.baseColorTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "material default baseColorTexture");
    expectTextureInfo(
        test,
        defaultMaterial.diffuseTransmissionColorTexture,
        metallic::scene::kInvalidSceneIndex,
        0,
        identityUv,
        "material default diffuseTransmissionColorTexture");

    const metallic::scene::RenderMaterial& fullMaterial = scene.materials()[1];
    expectVec4(
        test,
        fullMaterial.baseColorFactor,
        float4(0.2f, 0.3f, 0.4f, 0.5f),
        "full material baseColorFactor");
    test.expect(nearlyEqual(fullMaterial.metallicFactor, 0.25f), "full material metallicFactor");
    test.expect(nearlyEqual(fullMaterial.roughnessFactor, 0.75f), "full material roughnessFactor");
    expectVec3(test, fullMaterial.emissiveFactor, float3(0.1f, 0.2f, 0.3f), "full material emissiveFactor");
    test.expect(fullMaterial.alphaMode == "MASK", "full material alphaMode");
    test.expect(nearlyEqual(fullMaterial.alphaCutoff, 0.37f), "full material alphaCutoff");
    test.expect(fullMaterial.doubleSided, "full material doubleSided");
    test.expect(nearlyEqual(fullMaterial.normalTextureScale, 0.42f), "full material normal scale");
    test.expect(nearlyEqual(fullMaterial.occlusionTextureStrength, 0.66f), "full material occlusion strength");
    expectTextureInfo(
        test,
        fullMaterial.baseColorTexture,
        0,
        2,
        {0.0f, -3.0f, 0.1f, 2.0f, 0.0f, 0.2f},
        "full material baseColorTexture");
    expectTextureInfo(test, fullMaterial.metallicRoughnessTexture, 1, 0, identityUv, "full material metallicRoughnessTexture");
    expectTextureInfo(test, fullMaterial.normalTexture, 2, 0, identityUv, "full material normalTexture");
    expectTextureInfo(test, fullMaterial.occlusionTexture, 3, 1, identityUv, "full material occlusionTexture");
    expectTextureInfo(test, fullMaterial.emissiveTexture, 4, 0, identityUv, "full material emissiveTexture");

    test.expect(nearlyEqual(fullMaterial.transmissionFactor, 0.8f), "full material transmissionFactor");
    test.expect(nearlyEqual(fullMaterial.ior, 2.2f), "full material ior");
    test.expect(nearlyEqual(fullMaterial.thicknessFactor, 0.6f), "full material thicknessFactor");
    test.expect(nearlyEqual(fullMaterial.attenuationDistance, 12.5f), "full material attenuationDistance");
    expectVec3(test, fullMaterial.attenuationColor, float3(0.7f, 0.8f, 0.9f), "full material attenuationColor");
    test.expect(
        nearlyEqual(fullMaterial.diffuseTransmissionFactor, 0.55f),
        "full material diffuseTransmissionFactor");
    expectVec3(
        test,
        fullMaterial.diffuseTransmissionColor,
        float3(0.4f, 0.5f, 0.6f),
        "full material diffuseTransmissionColor");
    expectTextureInfo(
        test,
        fullMaterial.transmissionTexture,
        5,
        0,
        {0.5f, 0.0f, 0.25f, 0.0f, 0.25f, 0.75f},
        "full material transmissionTexture");
    expectTextureInfo(test, fullMaterial.thicknessTexture, 6, 0, identityUv, "full material thicknessTexture");
    expectTextureInfo(
        test,
        fullMaterial.diffuseTransmissionTexture,
        7,
        0,
        identityUv,
        "full material diffuseTransmissionTexture");
    expectTextureInfo(
        test,
        fullMaterial.diffuseTransmissionColorTexture,
        8,
        1,
        identityUv,
        "full material diffuseTransmissionColorTexture");

    const metallic::scene::RenderMaterial& lowClampMaterial = scene.materials()[2];
    test.expect(nearlyEqual(lowClampMaterial.transmissionFactor, 0.0f), "low clamp transmissionFactor");
    test.expect(nearlyEqual(lowClampMaterial.ior, 1.0f), "low clamp ior");
    test.expect(nearlyEqual(lowClampMaterial.thicknessFactor, 0.0f), "low clamp thicknessFactor");
    test.expect(nearlyEqual(lowClampMaterial.attenuationDistance, 0.0f), "low clamp attenuationDistance");
    test.expect(
        nearlyEqual(lowClampMaterial.diffuseTransmissionFactor, 0.0f),
        "low clamp diffuseTransmissionFactor");

    const metallic::scene::RenderMaterial& highClampMaterial = scene.materials()[3];
    test.expect(nearlyEqual(highClampMaterial.transmissionFactor, 1.0f), "high clamp transmissionFactor");
    test.expect(nearlyEqual(highClampMaterial.ior, 3.0f), "high clamp ior");
    test.expect(
        nearlyEqual(highClampMaterial.diffuseTransmissionFactor, 1.0f),
        "high clamp diffuseTransmissionFactor");
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
    testMaterialImport(test, outputDirectory);
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
