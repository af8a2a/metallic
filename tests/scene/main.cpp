#include "Runtime/Scene/Scene.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

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
    expect(scene.sceneIndex() == 0, "default scene index should be 0");
    expect(scene.sceneName() == "Default Scene", "scene name should come from glTF scene");
    expect(scene.assetInfo().version == "2.0", "asset version should be preserved");
    expect(scene.rootNodeIndices().size() == 1, "scene should expose one root node");
    expect(scene.rootNodeIndices().front() == 0, "root node index should be 0");
    expect(scene.nodes().size() == 5, "scene should load all nodes");
    expect(scene.nodes()[1].parent == 0, "child parent index should be assigned");
    expect(scene.nodes()[0].children.size() == 4, "root children should be preserved");

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
        perspectiveCamera.type == metallic::scene::CameraType::Perspective,
        "first camera should be perspective");
    expect(nearlyEqual(static_cast<float>(perspectiveCamera.yfov), 0.75f), "perspective yfov");
    expect(nearlyEqual(static_cast<float>(perspectiveCamera.aspectRatio), 1.7777778f), "perspective aspect");
    expectVec3(perspectiveCamera.eye, float3(1.0f, 2.0f, 13.0f), "perspective camera eye");
    expectVec3(perspectiveCamera.center, float3(1.0f, 2.0f, 12.0f), "perspective camera center");

    const metallic::scene::RenderCamera& orthoCamera = scene.cameras()[1];
    expect(
        orthoCamera.type == metallic::scene::CameraType::Orthographic,
        "second camera should be orthographic");
    expect(nearlyEqual(static_cast<float>(orthoCamera.xmag), 4.0f), "orthographic xmag");
    expect(nearlyEqual(static_cast<float>(orthoCamera.ymag), 3.0f), "orthographic ymag");

    expect(scene.lights().size() == 1, "punctual light count");
    expect(scene.lights().front().type == "directional", "punctual light type");
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

} // namespace

TEST(SceneImport, FullScene)
{
    testFullSceneImport(prepareOutputDirectory());
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
