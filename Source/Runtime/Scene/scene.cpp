/*
 * CPU glTF scene import and traversal code.
 *
 * Portions are adapted from NVIDIA nvpro_core2 nvvkgltf scene traversal ideas.
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime/Scene/Scene.h"

#include "meshoptimizer.h"
#define CLUSTERLOD_IMPLEMENTATION
#include "clusterlod.h"
#undef CLUSTERLOD_IMPLEMENTATION
#include "json.hpp"
#include "tiny_gltf.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace metallic::scene {
namespace {

constexpr const char* kExtensionLightsPunctual = "KHR_lights_punctual";
constexpr const char* kExtensionMaterialsDiffuseTransmission = "KHR_materials_diffuse_transmission";
constexpr const char* kExtensionMaterialsEmissiveStrength = "KHR_materials_emissive_strength";
constexpr const char* kExtensionMaterialsIor = "KHR_materials_ior";
constexpr const char* kExtensionMaterialsTransmission = "KHR_materials_transmission";
constexpr const char* kExtensionMaterialsVolume = "KHR_materials_volume";
constexpr const char* kExtensionNodeVisibility = "KHR_node_visibility";
constexpr const char* kExtensionTextureTransform = "KHR_texture_transform";
constexpr double kFallbackCameraYFov = 0.7853981633974483;
constexpr size_t kMeshletClusterMaxVertices = 128;
constexpr size_t kMeshletClusterMinTriangles = 32;
constexpr size_t kMeshletClusterMaxTriangles = 128;
constexpr size_t kMeshletLodGroupSize = 32;
constexpr float kMeshletClusterFillWeight = 0.5f;
constexpr float kMeshletLodErrorMergePrevious = 1.5f;
constexpr float kMeshletLodErrorMergeAdditive = 0.0f;
constexpr std::array<char, 8> kMeshletCacheMagic{'M', 'T', 'L', 'M', 'S', 'H', 'L', 'T'};
constexpr uint32_t kMeshletCacheVersion = 1;
constexpr uint32_t kMeshletCacheEndian = 0x01020304;
constexpr const char* kMeshletCacheSuffix = ".meshlets.bin";
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

using SceneLoadClock = std::chrono::steady_clock;

double sceneLoadElapsedMilliseconds(SceneLoadClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(SceneLoadClock::now() - begin).count();
}

void logSceneLoadStep(std::string_view label, SceneLoadClock::time_point begin)
{
    spdlog::info("[SceneLoad] Step {} completed in {:.2f} ms", label, sceneLoadElapsedMilliseconds(begin));
}

class SceneLoadScope {
public:
    explicit SceneLoadScope(std::filesystem::path filename)
        : filename_(std::move(filename))
    {
        spdlog::info("[SceneLoad] Begin load '{}'", filename_.string());
    }

    ~SceneLoadScope()
    {
        spdlog::info(
            "[SceneLoad] End load '{}' status={} in {:.2f} ms",
            filename_.string(),
            success_ ? "success" : "failed",
            sceneLoadElapsedMilliseconds(begin_));
    }

    void markSuccess()
    {
        success_ = true;
    }

private:
    std::filesystem::path filename_;
    SceneLoadClock::time_point begin_ = SceneLoadClock::now();
    bool success_ = false;
};

struct MeshletCacheHeader {
    char magic[8]{};
    uint32_t version = 0;
    uint32_t endian = 0;
    uint64_t sourceFileSize = 0;
    int64_t sourceWriteTime = 0;
    uint32_t primitiveCount = 0;
    uint32_t maxVertices = 0;
    uint32_t minTriangles = 0;
    uint32_t maxTriangles = 0;
    uint32_t lodGroupSize = 0;
    float fillWeight = 0.0f;
    float lodErrorMergePrevious = 0.0f;
    float lodErrorMergeAdditive = 0.0f;
    uint32_t reserved = 0;
};

struct MeshletCachePrimitiveHeader {
    int32_t meshIndex = kInvalidSceneIndex;
    int32_t primitiveIndex = kInvalidSceneIndex;
    int32_t materialIndex = kInvalidSceneIndex;
    int32_t mode = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t triangleCount = 0;
    uint64_t geometryHash = 0;
    uint64_t meshletClusterCount = 0;
    uint64_t meshletVertexCount = 0;
    uint64_t meshletTriangleCount = 0;
    uint64_t meshletLodLevelCount = 0;
    uint64_t meshletLodGroupCount = 0;
    uint64_t meshletLodClusterCount = 0;
    uint64_t meshletLodVertexCount = 0;
    uint64_t meshletLodTriangleCount = 0;
};

struct CachedBounds {
    float min[3]{};
    float max[3]{};
    uint32_t valid = 0;
};

struct CachedMeshletCluster {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupChildIndex = 0;
    int32_t lodGroupIndex = kInvalidSceneIndex;
    int32_t refinedGroupIndex = kInvalidSceneIndex;
    float lodError = 0.0f;
    CachedBounds bounds;
    float boundingSphereCenter[3]{};
    float boundingSphereRadius = 0.0f;
    float coneApex[3]{};
    float coneAxis[3]{};
    float coneCutoff = 1.0f;
    int8_t packedCone[4]{0, 0, 127, 127};
};

struct CachedMeshletLodGroup {
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
    uint32_t lodLevel = 0;
    CachedBounds bounds;
    float boundingSphereCenter[3]{};
    float boundingSphereRadius = 0.0f;
    float maxQuadricError = 0.0f;
};

struct CachedMeshletLodLevel {
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
    float minBoundingSphereRadius = 0.0f;
    float minMaxQuadricError = 0.0f;
};

static_assert(std::is_trivially_copyable_v<MeshletCacheHeader>);
static_assert(std::is_trivially_copyable_v<MeshletCachePrimitiveHeader>);
static_assert(std::is_trivially_copyable_v<CachedBounds>);
static_assert(std::is_trivially_copyable_v<CachedMeshletCluster>);
static_assert(std::is_trivially_copyable_v<CachedMeshletLodGroup>);
static_assert(std::is_trivially_copyable_v<CachedMeshletLodLevel>);

const std::unordered_set<std::string>& supportedRequiredExtensions()
{
    static const std::unordered_set<std::string> kExtensions{
        kExtensionLightsPunctual,
        kExtensionNodeVisibility,
        kExtensionMaterialsDiffuseTransmission,
        kExtensionMaterialsEmissiveStrength,
        kExtensionMaterialsIor,
        kExtensionMaterialsTransmission,
        kExtensionMaterialsVolume,
        "KHR_mesh_quantization",
        kExtensionTextureTransform,
    };
    return kExtensions;
}

std::string lowerExtension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

std::string defaultName(std::string name, std::string_view prefix, int32_t index)
{
    if (!name.empty()) {
        return name;
    }
    std::string result(prefix);
    result += ' ';
    result += std::to_string(index);
    return result;
}

float3 makeFloat3(const std::vector<double>& values, const float3& fallback)
{
    if (values.size() < 3) {
        return fallback;
    }
    return float3(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]));
}

float4 makeFloat4(const std::vector<double>& values, const float4& fallback)
{
    if (values.size() < 4) {
        return fallback;
    }
    return float4(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]),
        static_cast<float>(values[3]));
}

float4 makeQuaternion(const std::vector<double>& values)
{
    if (values.size() < 4) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return float4(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]),
        static_cast<float>(values[3]));
}

float4x4 makeMatrixFromGltf(const std::vector<double>& values)
{
    if (values.size() != 16) {
        return float4x4::Identity();
    }

    float4x4 matrix;
    for (size_t index = 0; index < values.size(); ++index) {
        matrix.a[index] = static_cast<float>(values[index]);
    }
    return matrix;
}

float4x4 makeNodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16) {
        return makeMatrixFromGltf(node.matrix);
    }

    float4x4 translation;
    translation.SetupByTranslation(makeFloat3(node.translation, float3(0.0f, 0.0f, 0.0f)));

    float4x4 rotation;
    rotation.SetupByQuaternion(makeQuaternion(node.rotation));

    float4x4 scale;
    scale.SetupByScale(makeFloat3(node.scale, float3(1.0f, 1.0f, 1.0f)));

    return translation * rotation * scale;
}

float3 transformPoint(const float4x4& matrix, const float3& point)
{
    return matrix * point;
}

float3 transformVector(const float4x4& matrix, const float3& vector)
{
    const float4 transformed = matrix * float4(vector, 0.0f);
    return float3(transformed.x, transformed.y, transformed.z);
}

float3 normalizedOr(const float3& value, const float3& fallback)
{
    const float len = length(value);
    if (len <= 0.000001f || !std::isfinite(len)) {
        return fallback;
    }
    return value / len;
}

float3 matrixTranslation(const float4x4& matrix)
{
    return float3(matrix.a03, matrix.a13, matrix.a23);
}

Bounds transformBounds(const Bounds& bounds, const float4x4& matrix)
{
    Bounds result;
    if (!bounds.valid) {
        return result;
    }

    const std::array<float3, 8> corners{
        float3(bounds.min.x, bounds.min.y, bounds.min.z),
        float3(bounds.max.x, bounds.min.y, bounds.min.z),
        float3(bounds.min.x, bounds.max.y, bounds.min.z),
        float3(bounds.max.x, bounds.max.y, bounds.min.z),
        float3(bounds.min.x, bounds.min.y, bounds.max.z),
        float3(bounds.max.x, bounds.min.y, bounds.max.z),
        float3(bounds.min.x, bounds.max.y, bounds.max.z),
        float3(bounds.max.x, bounds.max.y, bounds.max.z),
    };

    for (const float3& corner : corners) {
        result.include(transformPoint(matrix, corner));
    }
    return result;
}

uint64_t triangleCountForPrimitive(int32_t mode, uint64_t elementCount)
{
    switch (mode) {
    case TINYGLTF_MODE_TRIANGLES:
        return elementCount / 3;
    case TINYGLTF_MODE_TRIANGLE_STRIP:
    case TINYGLTF_MODE_TRIANGLE_FAN:
        return elementCount >= 3 ? elementCount - 2 : 0;
    default:
        return 0;
    }
}

bool readVec3Value(const tinygltf::Value& object, const std::string& key, float3& value)
{
    if (!object.IsObject() || !object.Has(key)) {
        return false;
    }

    const tinygltf::Value& array = object.Get(key);
    if (!array.IsArray() || array.ArrayLen() < 3) {
        return false;
    }

    for (size_t index = 0; index < 3; ++index) {
        if (!array.Get(index).IsNumber()) {
            return false;
        }
    }

    value = float3(
        static_cast<float>(array.Get(0).GetNumberAsDouble()),
        static_cast<float>(array.Get(1).GetNumberAsDouble()),
        static_cast<float>(array.Get(2).GetNumberAsDouble()));
    return true;
}

float readFloatValue(const tinygltf::Value& object, const char* key, float fallback)
{
    if (!object.IsObject() || !object.Has(key)) {
        return fallback;
    }

    const tinygltf::Value& value = object.Get(key);
    if (!value.IsNumber()) {
        return fallback;
    }
    return static_cast<float>(value.GetNumberAsDouble());
}

int32_t readIntValue(const tinygltf::Value& object, const char* key, int32_t fallback)
{
    if (!object.IsObject() || !object.Has(key)) {
        return fallback;
    }

    const tinygltf::Value& value = object.Get(key);
    if (value.IsInt()) {
        return value.Get<int>();
    }
    if (value.IsNumber()) {
        return static_cast<int32_t>(value.GetNumberAsDouble());
    }
    return fallback;
}

bool readNodeVisibility(const tinygltf::Node& node)
{
    const auto extension = node.extensions.find(kExtensionNodeVisibility);
    if (extension == node.extensions.end()) {
        return true;
    }

    const tinygltf::Value& value = extension->second;
    if (!value.IsObject() || !value.Has("visible") || !value.Get("visible").IsBool()) {
        return true;
    }
    return value.Get("visible").Get<bool>();
}

bool validIndex(int32_t index, size_t size);

Bounds accessorBounds(const tinygltf::Accessor& accessor)
{
    Bounds bounds;
    if (accessor.minValues.size() < 3 || accessor.maxValues.size() < 3) {
        return bounds;
    }

    bounds.include(makeFloat3(accessor.minValues, float3(0.0f, 0.0f, 0.0f)));
    bounds.include(makeFloat3(accessor.maxValues, float3(0.0f, 0.0f, 0.0f)));
    return bounds;
}

const uint8_t* accessorData(
    const tinygltf::Model& model,
    const tinygltf::Accessor& accessor,
    const tinygltf::BufferView*& outBufferView,
    int& outStride)
{
    outBufferView = nullptr;
    outStride = 0;
    if (accessor.sparse.isSparse || !validIndex(accessor.bufferView, model.bufferViews.size())) {
        return nullptr;
    }

    const tinygltf::BufferView& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    if (!validIndex(bufferView.buffer, model.buffers.size())) {
        return nullptr;
    }

    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];
    const size_t byteOffset = bufferView.byteOffset + accessor.byteOffset;
    if (byteOffset >= buffer.data.size()) {
        return nullptr;
    }

    const int stride = accessor.ByteStride(bufferView);
    if (stride <= 0) {
        return nullptr;
    }

    outBufferView = &bufferView;
    outStride = stride;
    return buffer.data.data() + byteOffset;
}

std::vector<float3> readPositionAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    std::vector<float3> positions;
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return positions;
    }

    const tinygltf::BufferView* bufferView = nullptr;
    int stride = 0;
    const uint8_t* data = accessorData(model, accessor, bufferView, stride);
    if (data == nullptr || bufferView == nullptr) {
        return positions;
    }

    positions.reserve(accessor.count);
    for (size_t index = 0; index < accessor.count; ++index) {
        const size_t elementOffset = static_cast<size_t>(stride) * index;
        if (bufferView->byteOffset + accessor.byteOffset + elementOffset + sizeof(float) * 3 >
            model.buffers[static_cast<size_t>(bufferView->buffer)].data.size()) {
            positions.clear();
            return positions;
        }

        float values[3] = {};
        std::memcpy(values, data + elementOffset, sizeof(values));
        positions.emplace_back(values[0], values[1], values[2]);
    }
    return positions;
}

std::vector<float2> readFloat2Accessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    std::vector<float2> values;
    if (accessor.type != TINYGLTF_TYPE_VEC2 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return values;
    }

    const tinygltf::BufferView* bufferView = nullptr;
    int stride = 0;
    const uint8_t* data = accessorData(model, accessor, bufferView, stride);
    if (data == nullptr || bufferView == nullptr) {
        return values;
    }

    values.reserve(accessor.count);
    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView->buffer)];
    for (size_t index = 0; index < accessor.count; ++index) {
        const size_t elementOffset = static_cast<size_t>(stride) * index;
        if (bufferView->byteOffset + accessor.byteOffset + elementOffset + sizeof(float) * 2 > buffer.data.size()) {
            values.clear();
            return values;
        }

        float components[2] = {};
        std::memcpy(components, data + elementOffset, sizeof(components));
        values.emplace_back(components[0], components[1]);
    }
    return values;
}

std::vector<float4> readFloat4Accessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    std::vector<float4> values;
    if (accessor.type != TINYGLTF_TYPE_VEC4 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return values;
    }

    const tinygltf::BufferView* bufferView = nullptr;
    int stride = 0;
    const uint8_t* data = accessorData(model, accessor, bufferView, stride);
    if (data == nullptr || bufferView == nullptr) {
        return values;
    }

    values.reserve(accessor.count);
    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView->buffer)];
    for (size_t index = 0; index < accessor.count; ++index) {
        const size_t elementOffset = static_cast<size_t>(stride) * index;
        if (bufferView->byteOffset + accessor.byteOffset + elementOffset + sizeof(float) * 4 > buffer.data.size()) {
            values.clear();
            return values;
        }

        float components[4] = {};
        std::memcpy(components, data + elementOffset, sizeof(components));
        values.emplace_back(components[0], components[1], components[2], components[3]);
    }
    return values;
}

std::vector<uint32_t> readIndexAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor)
{
    std::vector<uint32_t> indices;
    if (accessor.type != TINYGLTF_TYPE_SCALAR) {
        return indices;
    }

    const tinygltf::BufferView* bufferView = nullptr;
    int stride = 0;
    const uint8_t* data = accessorData(model, accessor, bufferView, stride);
    if (data == nullptr || bufferView == nullptr) {
        return indices;
    }

    size_t componentSize = 0;
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        componentSize = sizeof(uint8_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        componentSize = sizeof(uint16_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        componentSize = sizeof(uint32_t);
        break;
    default:
        return indices;
    }

    indices.reserve(accessor.count);
    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView->buffer)];
    for (size_t index = 0; index < accessor.count; ++index) {
        const size_t elementOffset = static_cast<size_t>(stride) * index;
        if (bufferView->byteOffset + accessor.byteOffset + elementOffset + componentSize > buffer.data.size()) {
            indices.clear();
            return indices;
        }

        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            indices.push_back(*(data + elementOffset));
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t value = 0;
            std::memcpy(&value, data + elementOffset, sizeof(value));
            indices.push_back(value);
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t value = 0;
            std::memcpy(&value, data + elementOffset, sizeof(value));
            indices.push_back(value);
            break;
        }
        default:
            break;
        }
    }
    return indices;
}

bool readFloatArray2(const tinygltf::Value& value, const char* key, float out[2])
{
    if (!value.IsObject() || !value.Has(key)) {
        return false;
    }
    const tinygltf::Value& array = value.Get(key);
    if (!array.IsArray() || array.ArrayLen() < 2) {
        return false;
    }
    for (size_t index = 0; index < 2; ++index) {
        if (!array.Get(index).IsNumber()) {
            return false;
        }
        out[index] = static_cast<float>(array.Get(index).GetNumberAsDouble());
    }
    return true;
}

void setUvTransform(
    RenderTextureInfo& textureInfo,
    const float offset[2],
    const float scale[2],
    float rotation)
{
    const float cosRotation = std::cos(rotation);
    const float sinRotation = std::sin(rotation);
    textureInfo.uvTransform = {
        cosRotation * scale[0],
        -sinRotation * scale[1],
        offset[0],
        sinRotation * scale[0],
        cosRotation * scale[1],
        offset[1],
    };
}

template <typename TextureInfoT>
RenderTextureInfo makeRenderTextureInfo(const TextureInfoT& gltfTextureInfo)
{
    RenderTextureInfo textureInfo;
    textureInfo.textureIndex = gltfTextureInfo.index >= 0 ? gltfTextureInfo.index : kInvalidSceneIndex;
    textureInfo.texCoord = std::max(gltfTextureInfo.texCoord, 0);

    const auto extension = gltfTextureInfo.extensions.find(kExtensionTextureTransform);
    if (extension == gltfTextureInfo.extensions.end()) {
        return textureInfo;
    }

    const tinygltf::Value& transform = extension->second;
    float offset[2] = {0.0f, 0.0f};
    float scale[2] = {1.0f, 1.0f};
    float rotation = 0.0f;
    readFloatArray2(transform, "offset", offset);
    readFloatArray2(transform, "scale", scale);
    if (transform.IsObject() && transform.Has("rotation") && transform.Get("rotation").IsNumber()) {
        rotation = static_cast<float>(transform.Get("rotation").GetNumberAsDouble());
    }
    if (transform.IsObject() && transform.Has("texCoord") && transform.Get("texCoord").IsInt()) {
        textureInfo.texCoord = std::max(transform.Get("texCoord").Get<int>(), 0);
    }
    setUvTransform(textureInfo, offset, scale, rotation);
    return textureInfo;
}

RenderTextureInfo makeRenderTextureInfo(const tinygltf::Value& gltfTextureInfo)
{
    RenderTextureInfo textureInfo;
    if (!gltfTextureInfo.IsObject()) {
        return textureInfo;
    }

    textureInfo.textureIndex = readIntValue(gltfTextureInfo, "index", kInvalidSceneIndex);
    textureInfo.texCoord = std::max(readIntValue(gltfTextureInfo, "texCoord", 0), 0);

    if (!gltfTextureInfo.Has("extensions")) {
        return textureInfo;
    }

    const tinygltf::Value& extensions = gltfTextureInfo.Get("extensions");
    if (!extensions.IsObject() || !extensions.Has(kExtensionTextureTransform)) {
        return textureInfo;
    }

    const tinygltf::Value& transform = extensions.Get(kExtensionTextureTransform);
    float offset[2] = {0.0f, 0.0f};
    float scale[2] = {1.0f, 1.0f};
    float rotation = 0.0f;
    readFloatArray2(transform, "offset", offset);
    readFloatArray2(transform, "scale", scale);
    rotation = readFloatValue(transform, "rotation", 0.0f);
    textureInfo.texCoord = std::max(readIntValue(transform, "texCoord", textureInfo.texCoord), 0);
    setUvTransform(textureInfo, offset, scale, rotation);
    return textureInfo;
}

void readExtensionTextureInfo(
    const tinygltf::Value& object,
    const char* key,
    RenderTextureInfo& textureInfo)
{
    if (!object.IsObject() || !object.Has(key)) {
        return;
    }
    textureInfo = makeRenderTextureInfo(object.Get(key));
}

float3 fallbackTangentForNormal(const float3& normal)
{
    const float3 axis = std::abs(normal.z) < 0.999f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(0.0f, 1.0f, 0.0f);
    return normalizedOr(cross(axis, normal), float3(1.0f, 0.0f, 0.0f));
}

void generateTangents(RenderPrimitive& primitive)
{
    if (primitive.positions.empty() ||
        primitive.normals.size() != primitive.positions.size() ||
        primitive.texcoords0.size() != primitive.positions.size()) {
        return;
    }

    std::vector<float3> accumulatedTangents(primitive.positions.size(), float3(0.0f, 0.0f, 0.0f));
    std::vector<float3> accumulatedBitangents(primitive.positions.size(), float3(0.0f, 0.0f, 0.0f));
    const size_t indexCount = primitive.indices.empty()
        ? (primitive.positions.size() / 3) * 3
        : (primitive.indices.size() / 3) * 3;
    for (size_t index = 0; index + 2 < indexCount; index += 3) {
        const uint32_t i0 = primitive.indices.empty() ? static_cast<uint32_t>(index + 0) : primitive.indices[index + 0];
        const uint32_t i1 = primitive.indices.empty() ? static_cast<uint32_t>(index + 1) : primitive.indices[index + 1];
        const uint32_t i2 = primitive.indices.empty() ? static_cast<uint32_t>(index + 2) : primitive.indices[index + 2];
        if (i0 >= primitive.positions.size() || i1 >= primitive.positions.size() || i2 >= primitive.positions.size()) {
            continue;
        }

        const float3 edge1 = primitive.positions[i1] - primitive.positions[i0];
        const float3 edge2 = primitive.positions[i2] - primitive.positions[i0];
        const float2 uv1 = primitive.texcoords0[i1] - primitive.texcoords0[i0];
        const float2 uv2 = primitive.texcoords0[i2] - primitive.texcoords0[i0];
        const float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        if (std::abs(determinant) <= 0.0000001f) {
            continue;
        }

        const float invDeterminant = 1.0f / determinant;
        const float3 tangent = (edge1 * uv2.y - edge2 * uv1.y) * invDeterminant;
        const float3 bitangent = (edge2 * uv1.x - edge1 * uv2.x) * invDeterminant;
        accumulatedTangents[i0] = accumulatedTangents[i0] + tangent;
        accumulatedTangents[i1] = accumulatedTangents[i1] + tangent;
        accumulatedTangents[i2] = accumulatedTangents[i2] + tangent;
        accumulatedBitangents[i0] = accumulatedBitangents[i0] + bitangent;
        accumulatedBitangents[i1] = accumulatedBitangents[i1] + bitangent;
        accumulatedBitangents[i2] = accumulatedBitangents[i2] + bitangent;
    }

    primitive.tangents.clear();
    primitive.tangents.reserve(primitive.positions.size());
    for (size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
        const float3 normal = normalizedOr(primitive.normals[vertexIndex], float3(0.0f, 1.0f, 0.0f));
        float3 tangent = accumulatedTangents[vertexIndex] - normal * dot(normal, accumulatedTangents[vertexIndex]);
        tangent = normalizedOr(tangent, fallbackTangentForNormal(normal));
        const float3 bitangent = accumulatedBitangents[vertexIndex];
        const float handedness = dot(cross(normal, tangent), bitangent) < 0.0f ? -1.0f : 1.0f;
        primitive.tangents.emplace_back(tangent.x, tangent.y, tangent.z, handedness);
    }
}

void clearMeshletClusters(RenderPrimitive& primitive)
{
    primitive.meshletClusters.clear();
    primitive.meshletVertices.clear();
    primitive.meshletTriangles.clear();
}

void clearMeshletLods(RenderPrimitive& primitive)
{
    primitive.meshletLodLevels.clear();
    primitive.meshletLodGroups.clear();
    primitive.meshletLodClusters.clear();
    primitive.meshletLodVertices.clear();
    primitive.meshletLodTriangles.clear();
}

bool buildTriangleIndexBuffer(const RenderPrimitive& primitive, std::vector<uint32_t>& outIndices)
{
    outIndices.clear();
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES ||
        primitive.positions.size() < 3 ||
        primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const size_t indexCount = primitive.indices.empty()
        ? (primitive.positions.size() / 3) * 3
        : (primitive.indices.size() / 3) * 3;
    if (indexCount < 3) {
        return false;
    }

    outIndices.reserve(indexCount);
    if (primitive.indices.empty()) {
        for (uint32_t index = 0; index < static_cast<uint32_t>(indexCount); ++index) {
            outIndices.push_back(index);
        }
    } else {
        for (size_t index = 0; index < indexCount; ++index) {
            const uint32_t vertexIndex = primitive.indices[index];
            if (vertexIndex >= primitive.positions.size()) {
                outIndices.clear();
                return false;
            }
            outIndices.push_back(vertexIndex);
        }
    }

    return true;
}

bool appendMeshletCluster(
    const RenderPrimitive& primitive,
    std::vector<MeshletCluster>& outClusters,
    std::vector<uint32_t>& outVertices,
    std::vector<uint8_t>& outTriangles,
    const uint32_t* vertices,
    uint32_t vertexCount,
    const uint8_t* triangles,
    uint32_t triangleCount,
    uint32_t lodLevel,
    int32_t lodGroupIndex,
    uint32_t lodGroupChildIndex,
    int32_t refinedGroupIndex,
    float lodError)
{
    if (vertexCount == 0 ||
        triangleCount == 0 ||
        vertexCount > kMeshletClusterMaxVertices ||
        triangleCount > kMeshletClusterMaxTriangles ||
        outVertices.size() + vertexCount > std::numeric_limits<uint32_t>::max() ||
        outTriangles.size() + static_cast<size_t>(triangleCount) * 3u > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    for (uint32_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
        if (vertices[vertexIndex] >= primitive.positions.size()) {
            return false;
        }
    }
    for (uint32_t index = 0; index < triangleCount * 3u; ++index) {
        if (triangles[index] >= vertexCount) {
            return false;
        }
    }

    const uint32_t vertexOffset = static_cast<uint32_t>(outVertices.size());
    const uint32_t triangleOffset = static_cast<uint32_t>(outTriangles.size());
    outVertices.insert(outVertices.end(), vertices, vertices + vertexCount);
    outTriangles.insert(outTriangles.end(), triangles, triangles + static_cast<size_t>(triangleCount) * 3u);

    MeshletCluster cluster;
    cluster.vertexOffset = vertexOffset;
    cluster.vertexCount = vertexCount;
    cluster.triangleOffset = triangleOffset;
    cluster.triangleCount = triangleCount;
    cluster.lodLevel = lodLevel;
    cluster.lodGroupChildIndex = lodGroupChildIndex;
    cluster.lodGroupIndex = lodGroupIndex;
    cluster.refinedGroupIndex = refinedGroupIndex;
    cluster.lodError = lodError;

    const float* positions = reinterpret_cast<const float*>(primitive.positions.data());
    const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
        vertices,
        triangles,
        triangleCount,
        positions,
        primitive.positions.size(),
        sizeof(float3));
    cluster.boundingSphereCenter = float3(bounds.center[0], bounds.center[1], bounds.center[2]);
    cluster.boundingSphereRadius = bounds.radius;
    cluster.coneApex = float3(bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2]);
    cluster.coneAxis = float3(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]);
    cluster.coneCutoff = bounds.cone_cutoff;
    cluster.packedCone = {
        bounds.cone_axis_s8[0],
        bounds.cone_axis_s8[1],
        bounds.cone_axis_s8[2],
        bounds.cone_cutoff_s8,
    };

    for (uint32_t localVertexIndex = 0; localVertexIndex < vertexCount; ++localVertexIndex) {
        cluster.bounds.include(primitive.positions[vertices[localVertexIndex]]);
    }

    outClusters.push_back(cluster);
    return true;
}

clodConfig makeMeshletLodConfig()
{
    clodConfig config = clodDefaultConfigRT(kMeshletClusterMaxTriangles);
    config.max_vertices = kMeshletClusterMaxVertices;
    config.partition_size = kMeshletLodGroupSize;
    config.partition_spatial = true;
    config.partition_sort = true;
    config.cluster_fill_weight = kMeshletClusterFillWeight;
    config.optimize_clusters = true;
    config.optimize_clusters_level = 1;
    config.simplify_error_merge_previous = kMeshletLodErrorMergePrevious;
    config.simplify_error_merge_additive = kMeshletLodErrorMergeAdditive;

    while (config.partition_size > 1 &&
        config.partition_size + config.partition_size / 3u > kMeshletLodGroupSize) {
        --config.partition_size;
    }
    return config;
}

// The reference vk_lod_clusters builder parallelizes the independent groups in
// each CLOD level. Compute those groups concurrently here, then emit them in
// task order so group/refined indices remain deterministic and checkpointable.
template <typename Output>
int emitClusterLodGroup(
    const clodConfig& config,
    const clodMesh& mesh,
    const std::vector<clod::Cluster>& clusters,
    const std::vector<int>& group,
    const clodBounds& simplified,
    int depth,
    Output& output)
{
    std::vector<clodCluster> outputClusters(group.size());
    for (size_t groupIndex = 0; groupIndex < group.size(); ++groupIndex) {
        const clod::Cluster& cluster = clusters[group[groupIndex]];
        clodCluster& outputCluster = outputClusters[groupIndex];
        outputCluster.refined = cluster.refined;
        outputCluster.bounds = config.optimize_bounds && cluster.refined != -1
            ? clod::boundsCompute(mesh, cluster.indices, cluster.bounds.error)
            : cluster.bounds;
        outputCluster.indices = cluster.indices.data();
        outputCluster.index_count = cluster.indices.size();
        outputCluster.vertex_count = cluster.vertices;
    }
    return output(
        clodGroup{depth, simplified},
        outputClusters.data(),
        outputClusters.size());
}

template <typename Output>
size_t buildClusterLodParallel(clodConfig config, clodMesh mesh, Output& output)
{
    assert(mesh.vertex_attributes_stride % sizeof(float) == 0);
    assert(mesh.attribute_count * sizeof(float) <= mesh.vertex_attributes_stride);
    assert(mesh.attribute_protect_mask <
        (1u << (mesh.vertex_attributes_stride / sizeof(float))));

    std::vector<unsigned char> locks(mesh.vertex_count);
    std::vector<unsigned int> remap(mesh.vertex_count);
    meshopt_generatePositionRemap(
        remap.data(),
        mesh.vertex_positions,
        mesh.vertex_count,
        mesh.vertex_positions_stride);

    if (mesh.attribute_protect_mask != 0) {
        const size_t maxAttributes = mesh.vertex_attributes_stride / sizeof(float);
        for (size_t vertexIndex = 0; vertexIndex < mesh.vertex_count; ++vertexIndex) {
            const unsigned int remappedVertex = remap[vertexIndex];
            for (size_t attributeIndex = 0; attributeIndex < maxAttributes; ++attributeIndex) {
                if (remappedVertex != vertexIndex &&
                    (mesh.attribute_protect_mask & (1u << attributeIndex)) != 0 &&
                    mesh.vertex_attributes[vertexIndex * maxAttributes + attributeIndex] !=
                        mesh.vertex_attributes[remappedVertex * maxAttributes + attributeIndex]) {
                    locks[vertexIndex] |= meshopt_SimplifyVertex_Protect;
                }
            }
        }
    }

    std::vector<clod::Cluster> clusters =
        clod::clusterize(config, mesh, mesh.indices, mesh.index_count);
    for (clod::Cluster& cluster : clusters) {
        cluster.bounds = clod::boundsCompute(mesh, cluster.indices, 0.0f);
    }

    std::vector<int> pending(clusters.size());
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        pending[clusterIndex] = static_cast<int>(clusterIndex);
    }

    int depth = 0;
    while (pending.size() > 1) {
        const std::vector<std::vector<int>> groups =
            clod::partition(config, mesh, clusters, pending, remap);
        clod::lockBoundary(locks, groups, clusters, remap, mesh.vertex_lock);

        struct TaskResult {
            clodBounds bounds{};
            std::vector<clod::Cluster> split;
            bool terminal = false;
        };
        std::vector<TaskResult> results(groups.size());
        std::atomic_size_t nextTask{0};
        const auto processTasks = [&]() {
            for (;;) {
                const size_t taskIndex = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIndex >= groups.size()) {
                    return;
                }

                const std::vector<int>& group = groups[taskIndex];
                std::vector<unsigned int> merged;
                merged.reserve(group.size() * config.max_triangles * 3u);
                for (const int clusterIndex : group) {
                    merged.insert(
                        merged.end(),
                        clusters[clusterIndex].indices.begin(),
                        clusters[clusterIndex].indices.end());
                }

                const size_t targetSize =
                    static_cast<size_t>((merged.size() / 3u) * config.simplify_ratio) * 3u;
                TaskResult& result = results[taskIndex];
                result.bounds = clod::boundsMerge(clusters, group);
                float error = 0.0f;
                std::vector<unsigned int> simplified =
                    clod::simplify(config, mesh, merged, locks, targetSize, &error);
                if (simplified.size() > merged.size() * config.simplify_threshold) {
                    result.bounds.error = FLT_MAX;
                    result.terminal = true;
                    continue;
                }

                result.bounds.error = std::max(
                    result.bounds.error * config.simplify_error_merge_previous,
                    error) + error * config.simplify_error_merge_additive;
                result.split = clod::clusterize(
                    config,
                    mesh,
                    simplified.data(),
                    simplified.size());
            }
        };

        const size_t hardwareThreads =
            std::max<size_t>(1u, std::thread::hardware_concurrency());
        const size_t workerCount = std::min(
            groups.size(),
            std::max<size_t>(1u, hardwareThreads / 2u));
        std::vector<std::thread> workers;
        workers.reserve(workerCount > 0 ? workerCount - 1u : 0u);
        for (size_t workerIndex = 1; workerIndex < workerCount; ++workerIndex) {
            workers.emplace_back(processTasks);
        }
        processTasks();
        for (std::thread& worker : workers) {
            worker.join();
        }

        pending.clear();
        for (size_t taskIndex = 0; taskIndex < groups.size(); ++taskIndex) {
            TaskResult& result = results[taskIndex];
            const int refined = emitClusterLodGroup(
                config,
                mesh,
                clusters,
                groups[taskIndex],
                result.bounds,
                depth,
                output);
            if (result.terminal) {
                continue;
            }

            for (const int clusterIndex : groups[taskIndex]) {
                clusters[clusterIndex].indices.clear();
            }
            for (clod::Cluster& cluster : result.split) {
                cluster.refined = refined;
                cluster.bounds = result.bounds;
                clusters.push_back(std::move(cluster));
                pending.push_back(static_cast<int>(clusters.size() - 1u));
            }
        }
        ++depth;
    }

    if (!pending.empty()) {
        assert(pending.size() == 1);
        clodBounds bounds = clusters[pending.front()].bounds;
        bounds.error = FLT_MAX;
        emitClusterLodGroup(config, mesh, clusters, pending, bounds, depth, output);
    }
    return clusters.size();
}

bool buildMeshletClusters(RenderPrimitive& primitive)
{
    clearMeshletClusters(primitive);

    std::vector<uint32_t> clusterIndices;
    if (!buildTriangleIndexBuffer(primitive, clusterIndices)) {
        return false;
    }

    const size_t meshletBound =
        meshopt_buildMeshletsBound(clusterIndices.size(), kMeshletClusterMaxVertices, kMeshletClusterMinTriangles);
    if (meshletBound == 0) {
        return false;
    }

    std::vector<meshopt_Meshlet> meshlets(meshletBound);
    std::vector<uint32_t> meshletVertices(clusterIndices.size());
    std::vector<uint8_t> meshletTriangles(clusterIndices.size());

    const float* positions = reinterpret_cast<const float*>(primitive.positions.data());
    const size_t meshletCount = meshopt_buildMeshletsSpatial(
        meshlets.data(),
        meshletVertices.data(),
        meshletTriangles.data(),
        clusterIndices.data(),
        clusterIndices.size(),
        positions,
        primitive.positions.size(),
        sizeof(float3),
        kMeshletClusterMaxVertices,
        kMeshletClusterMinTriangles,
        kMeshletClusterMaxTriangles,
        kMeshletClusterFillWeight);

    meshlets.resize(meshletCount);
    primitive.meshletClusters.reserve(meshletCount);
    primitive.meshletVertices.reserve(clusterIndices.size());
    primitive.meshletTriangles.reserve(clusterIndices.size());

    for (const meshopt_Meshlet& meshlet : meshlets) {
        if (meshlet.vertex_count == 0 || meshlet.triangle_count == 0) {
            continue;
        }

        uint32_t* const vertices = meshletVertices.data() + meshlet.vertex_offset;
        uint8_t* const triangles = meshletTriangles.data() + meshlet.triangle_offset;
        meshopt_optimizeMeshlet(vertices, triangles, meshlet.triangle_count, meshlet.vertex_count);

        if (!appendMeshletCluster(
                primitive,
                primitive.meshletClusters,
                primitive.meshletVertices,
                primitive.meshletTriangles,
                vertices,
                meshlet.vertex_count,
                triangles,
                meshlet.triangle_count,
                0,
                kInvalidSceneIndex,
                0,
                kInvalidSceneIndex,
                0.0f)) {
            clearMeshletClusters(primitive);
            return false;
        }
    }

    if (primitive.meshletClusters.empty()) {
        clearMeshletClusters(primitive);
        return false;
    }
    return true;
}

bool buildMeshletLods(RenderPrimitive& primitive)
{
    clearMeshletLods(primitive);

    std::vector<uint32_t> clusterIndices;
    if (!buildTriangleIndexBuffer(primitive, clusterIndices)) {
        return false;
    }

    clodConfig config = makeMeshletLodConfig();
    const std::array<float, 3> normalWeights{0.5f, 0.5f, 0.5f};

    clodMesh mesh{};
    mesh.indices = clusterIndices.data();
    mesh.index_count = clusterIndices.size();
    mesh.vertex_count = primitive.positions.size();
    mesh.vertex_positions = reinterpret_cast<const float*>(primitive.positions.data());
    mesh.vertex_positions_stride = sizeof(float3);
    if (primitive.normals.size() == primitive.positions.size()) {
        mesh.vertex_attributes = reinterpret_cast<const float*>(primitive.normals.data());
        mesh.vertex_attributes_stride = sizeof(float3);
        mesh.attribute_weights = normalWeights.data();
        mesh.attribute_count = normalWeights.size();
    }

    primitive.meshletLodLevels.reserve(16);
    primitive.meshletLodGroups.reserve(std::max<size_t>(1, primitive.meshletClusters.size()));
    primitive.meshletLodClusters.reserve(std::max<size_t>(1, primitive.meshletClusters.size() * 2u));
    primitive.meshletLodVertices.reserve(std::max<size_t>(clusterIndices.size(), primitive.meshletVertices.size() * 2u));
    primitive.meshletLodTriangles.reserve(std::max<size_t>(clusterIndices.size(), primitive.meshletTriangles.size() * 2u));

    bool success = true;
    auto outputGroup = [&](clodGroup group, const clodCluster* clusters, size_t clusterCount) -> int {
        if (!success ||
            group.depth < 0 ||
            clusterCount == 0 ||
            primitive.meshletLodGroups.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            success = false;
            return kInvalidSceneIndex;
        }

        const uint32_t lodLevel = static_cast<uint32_t>(group.depth);
        while (primitive.meshletLodLevels.size() <= lodLevel) {
            MeshletLodLevel level;
            level.groupOffset = static_cast<uint32_t>(primitive.meshletLodGroups.size());
            level.clusterOffset = static_cast<uint32_t>(primitive.meshletLodClusters.size());
            level.minBoundingSphereRadius = std::numeric_limits<float>::max();
            level.minMaxQuadricError = std::numeric_limits<float>::max();
            primitive.meshletLodLevels.push_back(level);
        }

        const int32_t groupIndex = static_cast<int32_t>(primitive.meshletLodGroups.size());
        MeshletLodGroup lodGroup;
        lodGroup.clusterOffset = static_cast<uint32_t>(primitive.meshletLodClusters.size());
        lodGroup.clusterCount = static_cast<uint32_t>(clusterCount);
        lodGroup.lodLevel = lodLevel;
        lodGroup.boundingSphereCenter = float3(
            group.simplified.center[0],
            group.simplified.center[1],
            group.simplified.center[2]);
        lodGroup.boundingSphereRadius = group.simplified.radius;
        lodGroup.maxQuadricError = group.simplified.error;

        for (size_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex) {
            const clodCluster& cluster = clusters[clusterIndex];
            if (cluster.index_count == 0 ||
                cluster.index_count % 3u != 0 ||
                cluster.index_count / 3u > kMeshletClusterMaxTriangles ||
                cluster.vertex_count > kMeshletClusterMaxVertices) {
                success = false;
                return kInvalidSceneIndex;
            }

            std::vector<uint32_t> localVertices(cluster.index_count);
            std::vector<uint8_t> localTriangles(cluster.index_count);
            const size_t localVertexCount = clodLocalIndices(
                localVertices.data(),
                localTriangles.data(),
                cluster.indices,
                cluster.index_count);
            const uint32_t vertexCount = static_cast<uint32_t>(localVertexCount);
            const uint32_t triangleCount = static_cast<uint32_t>(cluster.index_count / 3u);
            if (vertexCount == 0 ||
                vertexCount > kMeshletClusterMaxVertices ||
                vertexCount != cluster.vertex_count) {
                success = false;
                return kInvalidSceneIndex;
            }

            meshopt_optimizeMeshlet(localVertices.data(), localTriangles.data(), triangleCount, vertexCount);
            if (!appendMeshletCluster(
                    primitive,
                    primitive.meshletLodClusters,
                    primitive.meshletLodVertices,
                    primitive.meshletLodTriangles,
                    localVertices.data(),
                    vertexCount,
                    localTriangles.data(),
                    triangleCount,
                    lodLevel,
                    groupIndex,
                    static_cast<uint32_t>(clusterIndex),
                    cluster.refined,
                    cluster.bounds.error)) {
                success = false;
                return kInvalidSceneIndex;
            }

            lodGroup.bounds.include(primitive.meshletLodClusters.back().bounds);
        }

        MeshletLodLevel& level = primitive.meshletLodLevels[lodLevel];
        ++level.groupCount;
        level.clusterCount += lodGroup.clusterCount;
        level.minBoundingSphereRadius = std::min(level.minBoundingSphereRadius, lodGroup.boundingSphereRadius);
        level.minMaxQuadricError = std::min(level.minMaxQuadricError, lodGroup.maxQuadricError);

        primitive.meshletLodGroups.push_back(lodGroup);
        return groupIndex;
    };

    buildClusterLodParallel(config, mesh, outputGroup);
    if (!success ||
        primitive.meshletLodLevels.empty() ||
        primitive.meshletLodGroups.empty() ||
        primitive.meshletLodClusters.empty()) {
        clearMeshletLods(primitive);
        return false;
    }

    return true;
}

std::filesystem::path meshletCachePathFor(const std::filesystem::path& sourcePath)
{
    std::filesystem::path cachePath = sourcePath;
    cachePath += kMeshletCacheSuffix;
    return cachePath;
}

uint64_t sourceFileSize(const std::filesystem::path& path)
{
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(path, error);
    return error ? 0 : size;
}

int64_t sourceWriteTime(const std::filesystem::path& path)
{
    std::error_code error;
    const auto writeTime = std::filesystem::last_write_time(path, error);
    if (error) {
        return 0;
    }
    return static_cast<int64_t>(writeTime.time_since_epoch().count());
}

uint64_t hashBytes(uint64_t hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t byteIndex = 0; byteIndex < byteSize; ++byteIndex) {
        hash ^= bytes[byteIndex];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t hashValue(uint64_t hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(T));
}

uint64_t hashFloat3Vector(uint64_t hash, const std::vector<float3>& values)
{
    hash = hashValue(hash, static_cast<uint64_t>(values.size()));
    for (const float3& value : values) {
        hash = hashValue(hash, value.x);
        hash = hashValue(hash, value.y);
        hash = hashValue(hash, value.z);
    }
    return hash;
}

uint64_t hashIndexVector(uint64_t hash, const std::vector<uint32_t>& values)
{
    hash = hashValue(hash, static_cast<uint64_t>(values.size()));
    if (!values.empty()) {
        hash = hashBytes(hash, values.data(), values.size() * sizeof(uint32_t));
    }
    return hash;
}

uint64_t hashPrimitiveGeometry(const RenderPrimitive& primitive)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, primitive.meshIndex);
    hash = hashValue(hash, primitive.primitiveIndex);
    hash = hashValue(hash, primitive.mode);
    hash = hashValue(hash, primitive.vertexCount);
    hash = hashValue(hash, primitive.indexCount);
    hash = hashValue(hash, primitive.triangleCount);
    hash = hashFloat3Vector(hash, primitive.positions);
    hash = hashFloat3Vector(hash, primitive.normals);
    hash = hashIndexVector(hash, primitive.indices);
    return hash;
}

CachedBounds makeCachedBounds(const Bounds& bounds)
{
    CachedBounds cached;
    cached.min[0] = bounds.min.x;
    cached.min[1] = bounds.min.y;
    cached.min[2] = bounds.min.z;
    cached.max[0] = bounds.max.x;
    cached.max[1] = bounds.max.y;
    cached.max[2] = bounds.max.z;
    cached.valid = bounds.valid ? 1u : 0u;
    return cached;
}

Bounds makeBounds(const CachedBounds& cached)
{
    Bounds bounds;
    bounds.min = float3(cached.min[0], cached.min[1], cached.min[2]);
    bounds.max = float3(cached.max[0], cached.max[1], cached.max[2]);
    bounds.valid = cached.valid != 0;
    return bounds;
}

CachedMeshletCluster makeCachedCluster(const MeshletCluster& cluster)
{
    CachedMeshletCluster cached;
    cached.vertexOffset = cluster.vertexOffset;
    cached.vertexCount = cluster.vertexCount;
    cached.triangleOffset = cluster.triangleOffset;
    cached.triangleCount = cluster.triangleCount;
    cached.lodLevel = cluster.lodLevel;
    cached.lodGroupChildIndex = cluster.lodGroupChildIndex;
    cached.lodGroupIndex = cluster.lodGroupIndex;
    cached.refinedGroupIndex = cluster.refinedGroupIndex;
    cached.lodError = cluster.lodError;
    cached.bounds = makeCachedBounds(cluster.bounds);
    cached.boundingSphereCenter[0] = cluster.boundingSphereCenter.x;
    cached.boundingSphereCenter[1] = cluster.boundingSphereCenter.y;
    cached.boundingSphereCenter[2] = cluster.boundingSphereCenter.z;
    cached.boundingSphereRadius = cluster.boundingSphereRadius;
    cached.coneApex[0] = cluster.coneApex.x;
    cached.coneApex[1] = cluster.coneApex.y;
    cached.coneApex[2] = cluster.coneApex.z;
    cached.coneAxis[0] = cluster.coneAxis.x;
    cached.coneAxis[1] = cluster.coneAxis.y;
    cached.coneAxis[2] = cluster.coneAxis.z;
    cached.coneCutoff = cluster.coneCutoff;
    for (size_t index = 0; index < 4; ++index) {
        cached.packedCone[index] = cluster.packedCone[index];
    }
    return cached;
}

MeshletCluster makeCluster(const CachedMeshletCluster& cached)
{
    MeshletCluster cluster;
    cluster.vertexOffset = cached.vertexOffset;
    cluster.vertexCount = cached.vertexCount;
    cluster.triangleOffset = cached.triangleOffset;
    cluster.triangleCount = cached.triangleCount;
    cluster.lodLevel = cached.lodLevel;
    cluster.lodGroupChildIndex = cached.lodGroupChildIndex;
    cluster.lodGroupIndex = cached.lodGroupIndex;
    cluster.refinedGroupIndex = cached.refinedGroupIndex;
    cluster.lodError = cached.lodError;
    cluster.bounds = makeBounds(cached.bounds);
    cluster.boundingSphereCenter = float3(
        cached.boundingSphereCenter[0],
        cached.boundingSphereCenter[1],
        cached.boundingSphereCenter[2]);
    cluster.boundingSphereRadius = cached.boundingSphereRadius;
    cluster.coneApex = float3(cached.coneApex[0], cached.coneApex[1], cached.coneApex[2]);
    cluster.coneAxis = float3(cached.coneAxis[0], cached.coneAxis[1], cached.coneAxis[2]);
    cluster.coneCutoff = cached.coneCutoff;
    for (size_t index = 0; index < cluster.packedCone.size(); ++index) {
        cluster.packedCone[index] = cached.packedCone[index];
    }
    return cluster;
}

CachedMeshletLodGroup makeCachedLodGroup(const MeshletLodGroup& group)
{
    CachedMeshletLodGroup cached;
    cached.clusterOffset = group.clusterOffset;
    cached.clusterCount = group.clusterCount;
    cached.lodLevel = group.lodLevel;
    cached.bounds = makeCachedBounds(group.bounds);
    cached.boundingSphereCenter[0] = group.boundingSphereCenter.x;
    cached.boundingSphereCenter[1] = group.boundingSphereCenter.y;
    cached.boundingSphereCenter[2] = group.boundingSphereCenter.z;
    cached.boundingSphereRadius = group.boundingSphereRadius;
    cached.maxQuadricError = group.maxQuadricError;
    return cached;
}

MeshletLodGroup makeLodGroup(const CachedMeshletLodGroup& cached)
{
    MeshletLodGroup group;
    group.clusterOffset = cached.clusterOffset;
    group.clusterCount = cached.clusterCount;
    group.lodLevel = cached.lodLevel;
    group.bounds = makeBounds(cached.bounds);
    group.boundingSphereCenter = float3(
        cached.boundingSphereCenter[0],
        cached.boundingSphereCenter[1],
        cached.boundingSphereCenter[2]);
    group.boundingSphereRadius = cached.boundingSphereRadius;
    group.maxQuadricError = cached.maxQuadricError;
    return group;
}

CachedMeshletLodLevel makeCachedLodLevel(const MeshletLodLevel& level)
{
    CachedMeshletLodLevel cached;
    cached.groupOffset = level.groupOffset;
    cached.groupCount = level.groupCount;
    cached.clusterOffset = level.clusterOffset;
    cached.clusterCount = level.clusterCount;
    cached.minBoundingSphereRadius = level.minBoundingSphereRadius;
    cached.minMaxQuadricError = level.minMaxQuadricError;
    return cached;
}

MeshletLodLevel makeLodLevel(const CachedMeshletLodLevel& cached)
{
    MeshletLodLevel level;
    level.groupOffset = cached.groupOffset;
    level.groupCount = cached.groupCount;
    level.clusterOffset = cached.clusterOffset;
    level.clusterCount = cached.clusterCount;
    level.minBoundingSphereRadius = cached.minBoundingSphereRadius;
    level.minMaxQuadricError = cached.minMaxQuadricError;
    return level;
}

bool readExact(std::istream& stream, void* data, uint64_t byteSize)
{
    if (byteSize == 0) {
        return true;
    }
    if (byteSize > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(byteSize));
    return static_cast<uint64_t>(stream.gcount()) == byteSize;
}

bool writeExact(std::ostream& stream, const void* data, uint64_t byteSize)
{
    if (byteSize == 0) {
        return true;
    }
    if (byteSize > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(byteSize));
    return stream.good();
}

template <typename T>
bool readPod(std::istream& stream, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return readExact(stream, &value, sizeof(T));
}

template <typename T>
bool writePod(std::ostream& stream, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return writeExact(stream, &value, sizeof(T));
}

template <typename T>
bool readArray(std::istream& stream, uint64_t count, std::vector<T>& values)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    const size_t size = static_cast<size_t>(count);
    if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return false;
    }

    values.resize(size);
    return values.empty() || readExact(stream, values.data(), values.size() * sizeof(T));
}

template <typename T>
bool writeArray(std::ostream& stream, const std::vector<T>& values)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return values.empty() || writeExact(stream, values.data(), values.size() * sizeof(T));
}

bool addArrayByteSize(uint64_t& total, uint64_t count, uint64_t elementSize)
{
    if (count > std::numeric_limits<uint64_t>::max() / elementSize) {
        return false;
    }
    const uint64_t byteSize = count * elementSize;
    if (byteSize > std::numeric_limits<uint64_t>::max() - total) {
        return false;
    }
    total += byteSize;
    return true;
}

bool meshletCachePrimitivePayloadByteSize(const MeshletCachePrimitiveHeader& header, uint64_t& byteSize)
{
    byteSize = 0;
    return addArrayByteSize(byteSize, header.meshletClusterCount, sizeof(CachedMeshletCluster)) &&
        addArrayByteSize(byteSize, header.meshletVertexCount, sizeof(uint32_t)) &&
        addArrayByteSize(byteSize, header.meshletTriangleCount, sizeof(uint8_t)) &&
        addArrayByteSize(byteSize, header.meshletLodLevelCount, sizeof(CachedMeshletLodLevel)) &&
        addArrayByteSize(byteSize, header.meshletLodGroupCount, sizeof(CachedMeshletLodGroup)) &&
        addArrayByteSize(byteSize, header.meshletLodClusterCount, sizeof(CachedMeshletCluster)) &&
        addArrayByteSize(byteSize, header.meshletLodVertexCount, sizeof(uint32_t)) &&
        addArrayByteSize(byteSize, header.meshletLodTriangleCount, sizeof(uint8_t));
}

bool rangeWithin(uint64_t offset, uint64_t count, size_t size)
{
    return offset <= size && count <= size - static_cast<size_t>(offset);
}

bool validateClusterData(
    const RenderPrimitive& primitive,
    const std::vector<MeshletCluster>& clusters,
    const std::vector<uint32_t>& vertices,
    const std::vector<uint8_t>& triangles,
    size_t lodGroupCount)
{
    if (clusters.empty()) {
        return vertices.empty() && triangles.empty();
    }

    for (const MeshletCluster& cluster : clusters) {
        if (cluster.vertexCount == 0 ||
            cluster.triangleCount == 0 ||
            cluster.vertexCount > kMeshletClusterMaxVertices ||
            cluster.triangleCount > kMeshletClusterMaxTriangles ||
            !rangeWithin(cluster.vertexOffset, cluster.vertexCount, vertices.size()) ||
            !rangeWithin(
                cluster.triangleOffset,
                static_cast<uint64_t>(cluster.triangleCount) * 3u,
                triangles.size())) {
            return false;
        }

        if (cluster.lodGroupIndex != kInvalidSceneIndex &&
            (cluster.lodGroupIndex < 0 || static_cast<size_t>(cluster.lodGroupIndex) >= lodGroupCount)) {
            return false;
        }
        if (cluster.refinedGroupIndex != kInvalidSceneIndex &&
            (cluster.refinedGroupIndex < 0 || static_cast<size_t>(cluster.refinedGroupIndex) >= lodGroupCount)) {
            return false;
        }

        for (uint32_t vertex = 0; vertex < cluster.vertexCount; ++vertex) {
            const size_t index = static_cast<size_t>(cluster.vertexOffset) + vertex;
            if (vertices[index] >= primitive.positions.size()) {
                return false;
            }
        }

        for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
            const size_t triangleIndex = static_cast<size_t>(cluster.triangleOffset) + index;
            if (triangles[triangleIndex] >= cluster.vertexCount) {
                return false;
            }
        }
    }

    return true;
}

bool validateLodData(
    const RenderPrimitive& primitive,
    const std::vector<MeshletLodLevel>& levels,
    const std::vector<MeshletLodGroup>& groups,
    const std::vector<MeshletCluster>& clusters,
    const std::vector<uint32_t>& vertices,
    const std::vector<uint8_t>& triangles)
{
    if (levels.empty()) {
        return groups.empty() && clusters.empty() && vertices.empty() && triangles.empty();
    }
    if (groups.empty() || clusters.empty()) {
        return false;
    }

    for (const MeshletLodLevel& level : levels) {
        if (level.groupCount == 0 ||
            level.clusterCount == 0 ||
            !rangeWithin(level.groupOffset, level.groupCount, groups.size()) ||
            !rangeWithin(level.clusterOffset, level.clusterCount, clusters.size())) {
            return false;
        }
    }

    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const MeshletLodGroup& group = groups[groupIndex];
        if (group.clusterCount == 0 ||
            group.lodLevel >= levels.size() ||
            !rangeWithin(group.clusterOffset, group.clusterCount, clusters.size())) {
            return false;
        }

        const MeshletLodLevel& level = levels[group.lodLevel];
        if (groupIndex < level.groupOffset ||
            groupIndex >= static_cast<size_t>(level.groupOffset) + level.groupCount) {
            return false;
        }
    }

    return validateClusterData(primitive, clusters, vertices, triangles, groups.size());
}

bool validateMeshletData(
    const RenderPrimitive& primitive,
    const std::vector<MeshletCluster>& clusters,
    const std::vector<uint32_t>& vertices,
    const std::vector<uint8_t>& triangles,
    const std::vector<MeshletLodLevel>& lodLevels,
    const std::vector<MeshletLodGroup>& lodGroups,
    const std::vector<MeshletCluster>& lodClusters,
    const std::vector<uint32_t>& lodVertices,
    const std::vector<uint8_t>& lodTriangles)
{
    return validateClusterData(primitive, clusters, vertices, triangles, lodGroups.size()) &&
        validateLodData(primitive, lodLevels, lodGroups, lodClusters, lodVertices, lodTriangles);
}

MeshletCacheHeader makeMeshletCacheHeader(
    const std::filesystem::path& sourcePath,
    const std::vector<RenderPrimitive>& primitives)
{
    MeshletCacheHeader header;
    std::memcpy(header.magic, kMeshletCacheMagic.data(), kMeshletCacheMagic.size());
    header.version = kMeshletCacheVersion;
    header.endian = kMeshletCacheEndian;
    header.sourceFileSize = sourceFileSize(sourcePath);
    header.sourceWriteTime = sourceWriteTime(sourcePath);
    header.primitiveCount = static_cast<uint32_t>(primitives.size());
    header.maxVertices = static_cast<uint32_t>(kMeshletClusterMaxVertices);
    header.minTriangles = static_cast<uint32_t>(kMeshletClusterMinTriangles);
    header.maxTriangles = static_cast<uint32_t>(kMeshletClusterMaxTriangles);
    header.lodGroupSize = static_cast<uint32_t>(kMeshletLodGroupSize);
    header.fillWeight = kMeshletClusterFillWeight;
    header.lodErrorMergePrevious = kMeshletLodErrorMergePrevious;
    header.lodErrorMergeAdditive = kMeshletLodErrorMergeAdditive;
    return header;
}

bool meshletCacheHeaderMatches(const MeshletCacheHeader& header, const MeshletCacheHeader& expected)
{
    return std::memcmp(header.magic, kMeshletCacheMagic.data(), kMeshletCacheMagic.size()) == 0 &&
        header.version == expected.version &&
        header.endian == expected.endian &&
        header.sourceFileSize == expected.sourceFileSize &&
        header.sourceWriteTime == expected.sourceWriteTime &&
        header.primitiveCount == expected.primitiveCount &&
        header.maxVertices == expected.maxVertices &&
        header.minTriangles == expected.minTriangles &&
        header.maxTriangles == expected.maxTriangles &&
        header.lodGroupSize == expected.lodGroupSize &&
        header.fillWeight == expected.fillWeight &&
        header.lodErrorMergePrevious == expected.lodErrorMergePrevious &&
        header.lodErrorMergeAdditive == expected.lodErrorMergeAdditive;
}

MeshletCachePrimitiveHeader makeMeshletCachePrimitiveHeader(const RenderPrimitive& primitive)
{
    MeshletCachePrimitiveHeader header;
    header.meshIndex = primitive.meshIndex;
    header.primitiveIndex = primitive.primitiveIndex;
    header.materialIndex = primitive.materialIndex;
    header.mode = primitive.mode;
    header.vertexCount = primitive.vertexCount;
    header.indexCount = primitive.indexCount;
    header.triangleCount = primitive.triangleCount;
    header.geometryHash = hashPrimitiveGeometry(primitive);
    header.meshletClusterCount = primitive.meshletClusters.size();
    header.meshletVertexCount = primitive.meshletVertices.size();
    header.meshletTriangleCount = primitive.meshletTriangles.size();
    header.meshletLodLevelCount = primitive.meshletLodLevels.size();
    header.meshletLodGroupCount = primitive.meshletLodGroups.size();
    header.meshletLodClusterCount = primitive.meshletLodClusters.size();
    header.meshletLodVertexCount = primitive.meshletLodVertices.size();
    header.meshletLodTriangleCount = primitive.meshletLodTriangles.size();
    return header;
}

bool meshletCachePrimitiveHeaderMatches(
    const MeshletCachePrimitiveHeader& header,
    const RenderPrimitive& primitive)
{
    return header.meshIndex == primitive.meshIndex &&
        header.primitiveIndex == primitive.primitiveIndex &&
        header.materialIndex == primitive.materialIndex &&
        header.mode == primitive.mode &&
        header.vertexCount == primitive.vertexCount &&
        header.indexCount == primitive.indexCount &&
        header.triangleCount == primitive.triangleCount &&
        header.geometryHash == hashPrimitiveGeometry(primitive);
}

template <typename SourceT, typename CachedT, typename ConvertT>
bool writeConvertedArray(std::ostream& stream, const std::vector<SourceT>& values, ConvertT convert)
{
    std::vector<CachedT> cached;
    cached.reserve(values.size());
    for (const SourceT& value : values) {
        cached.push_back(convert(value));
    }
    return writeArray(stream, cached);
}

template <typename RuntimeT, typename CachedT, typename ConvertT>
bool readConvertedArray(
    std::istream& stream,
    uint64_t count,
    std::vector<RuntimeT>& values,
    ConvertT convert)
{
    std::vector<CachedT> cached;
    if (!readArray(stream, count, cached)) {
        return false;
    }

    values.clear();
    values.reserve(cached.size());
    for (const CachedT& value : cached) {
        values.push_back(convert(value));
    }
    return true;
}

struct MeshletCachePrimitiveData {
    std::vector<MeshletCluster> meshletClusters;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint8_t> meshletTriangles;
    std::vector<MeshletLodLevel> meshletLodLevels;
    std::vector<MeshletLodGroup> meshletLodGroups;
    std::vector<MeshletCluster> meshletLodClusters;
    std::vector<uint32_t> meshletLodVertices;
    std::vector<uint8_t> meshletLodTriangles;
};

bool readMeshletCachePrimitive(
    std::istream& stream,
    const RenderPrimitive& primitive,
    MeshletCachePrimitiveData& data,
    uint64_t cacheByteSize,
    std::string& reason)
{
    MeshletCachePrimitiveHeader header;
    if (!readPod(stream, header)) {
        reason = "primitive header is truncated";
        return false;
    }

    if (!meshletCachePrimitiveHeaderMatches(header, primitive)) {
        reason = "primitive geometry metadata changed";
        return false;
    }

    uint64_t payloadByteSize = 0;
    if (!meshletCachePrimitivePayloadByteSize(header, payloadByteSize)) {
        reason = "primitive payload size overflows";
        return false;
    }

    const std::streampos payloadStart = stream.tellg();
    if (payloadStart == std::streampos(-1)) {
        reason = "primitive payload offset is invalid";
        return false;
    }
    const uint64_t payloadOffset = static_cast<uint64_t>(payloadStart);
    if (payloadOffset > cacheByteSize || payloadByteSize > cacheByteSize - payloadOffset) {
        reason = "primitive payload exceeds cache file";
        return false;
    }

    if (!readConvertedArray<MeshletCluster, CachedMeshletCluster>(
            stream,
            header.meshletClusterCount,
            data.meshletClusters,
            makeCluster) ||
        !readArray(stream, header.meshletVertexCount, data.meshletVertices) ||
        !readArray(stream, header.meshletTriangleCount, data.meshletTriangles) ||
        !readConvertedArray<MeshletLodLevel, CachedMeshletLodLevel>(
            stream,
            header.meshletLodLevelCount,
            data.meshletLodLevels,
            makeLodLevel) ||
        !readConvertedArray<MeshletLodGroup, CachedMeshletLodGroup>(
            stream,
            header.meshletLodGroupCount,
            data.meshletLodGroups,
            makeLodGroup) ||
        !readConvertedArray<MeshletCluster, CachedMeshletCluster>(
            stream,
            header.meshletLodClusterCount,
            data.meshletLodClusters,
            makeCluster) ||
        !readArray(stream, header.meshletLodVertexCount, data.meshletLodVertices) ||
        !readArray(stream, header.meshletLodTriangleCount, data.meshletLodTriangles)) {
        reason = "primitive payload is truncated";
        return false;
    }

    if (!validateMeshletData(
            primitive,
            data.meshletClusters,
            data.meshletVertices,
            data.meshletTriangles,
            data.meshletLodLevels,
            data.meshletLodGroups,
            data.meshletLodClusters,
            data.meshletLodVertices,
            data.meshletLodTriangles)) {
        reason = "primitive payload failed validation";
        return false;
    }

    return true;
}

bool writeMeshletCachePrimitive(std::ostream& stream, const RenderPrimitive& primitive)
{
    const MeshletCachePrimitiveHeader header = makeMeshletCachePrimitiveHeader(primitive);
    return writePod(stream, header) &&
        writeConvertedArray<MeshletCluster, CachedMeshletCluster>(
            stream,
            primitive.meshletClusters,
            makeCachedCluster) &&
        writeArray(stream, primitive.meshletVertices) &&
        writeArray(stream, primitive.meshletTriangles) &&
        writeConvertedArray<MeshletLodLevel, CachedMeshletLodLevel>(
            stream,
            primitive.meshletLodLevels,
            makeCachedLodLevel) &&
        writeConvertedArray<MeshletLodGroup, CachedMeshletLodGroup>(
            stream,
            primitive.meshletLodGroups,
            makeCachedLodGroup) &&
        writeConvertedArray<MeshletCluster, CachedMeshletCluster>(
            stream,
            primitive.meshletLodClusters,
            makeCachedCluster) &&
        writeArray(stream, primitive.meshletLodVertices) &&
        writeArray(stream, primitive.meshletLodTriangles);
}

bool loadMeshletCache(
    const std::filesystem::path& cachePath,
    const std::filesystem::path& sourcePath,
    std::vector<RenderPrimitive>& primitives,
    std::string& reason)
{
    reason.clear();

    std::error_code existsError;
    if (!std::filesystem::exists(cachePath, existsError)) {
        return false;
    }

    std::ifstream stream(cachePath, std::ios::binary);
    if (!stream) {
        reason = "cache file cannot be opened";
        return false;
    }

    MeshletCacheHeader header;
    if (!readPod(stream, header)) {
        reason = "cache header is truncated";
        return false;
    }

    const MeshletCacheHeader expectedHeader = makeMeshletCacheHeader(sourcePath, primitives);
    if (!meshletCacheHeaderMatches(header, expectedHeader)) {
        reason = "cache header does not match source or meshlet settings";
        return false;
    }

    const uint64_t cacheByteSize = sourceFileSize(cachePath);
    std::vector<MeshletCachePrimitiveData> cachedPrimitives(primitives.size());
    for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
        if (!readMeshletCachePrimitive(
                stream,
                primitives[primitiveIndex],
                cachedPrimitives[primitiveIndex],
                cacheByteSize,
                reason)) {
            return false;
        }
    }

    for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
        MeshletCachePrimitiveData& cached = cachedPrimitives[primitiveIndex];
        RenderPrimitive& primitive = primitives[primitiveIndex];
        primitive.meshletClusters = std::move(cached.meshletClusters);
        primitive.meshletVertices = std::move(cached.meshletVertices);
        primitive.meshletTriangles = std::move(cached.meshletTriangles);
        primitive.meshletLodLevels = std::move(cached.meshletLodLevels);
        primitive.meshletLodGroups = std::move(cached.meshletLodGroups);
        primitive.meshletLodClusters = std::move(cached.meshletLodClusters);
        primitive.meshletLodVertices = std::move(cached.meshletLodVertices);
        primitive.meshletLodTriangles = std::move(cached.meshletLodTriangles);
    }

    return true;
}

bool saveMeshletCache(
    const std::filesystem::path& cachePath,
    const std::filesystem::path& sourcePath,
    const std::vector<RenderPrimitive>& primitives,
    std::string& reason)
{
    reason.clear();

    if (primitives.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "too many render primitives";
        return false;
    }

    std::ofstream stream(cachePath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        reason = "cache file cannot be opened for writing";
        return false;
    }

    const MeshletCacheHeader header = makeMeshletCacheHeader(sourcePath, primitives);
    if (!writePod(stream, header)) {
        reason = "cache header write failed";
        return false;
    }

    for (const RenderPrimitive& primitive : primitives) {
        if (!writeMeshletCachePrimitive(stream, primitive)) {
            reason = "primitive payload write failed";
            return false;
        }
    }

    return true;
}

void buildMeshletsForPrimitives(std::vector<RenderPrimitive>& primitives)
{
    for (RenderPrimitive& primitive : primitives) {
        buildMeshletClusters(primitive);
        buildMeshletLods(primitive);
    }
}

void accumulateMeshletStats(const std::vector<RenderPrimitive>& primitives, SceneStats& stats)
{
    for (const RenderPrimitive& primitive : primitives) {
        stats.meshletClusterCount += primitive.meshletClusters.size();
        stats.meshletVertexReferenceCount += primitive.meshletVertices.size();
        stats.meshletTriangleIndexCount += primitive.meshletTriangles.size();
        stats.meshletLodLevelCount += primitive.meshletLodLevels.size();
        stats.meshletLodGroupCount += primitive.meshletLodGroups.size();
        stats.meshletLodClusterCount += primitive.meshletLodClusters.size();
        stats.meshletLodVertexReferenceCount += primitive.meshletLodVertices.size();
        stats.meshletLodTriangleIndexCount += primitive.meshletLodTriangles.size();
    }
}

RenderCamera makeRenderCamera(
    const tinygltf::Camera& camera,
    const tinygltf::Node& node,
    int32_t nodeIndex,
    int32_t cameraIndex,
    const float4x4& worldMatrix)
{
    RenderCamera renderCamera;
    renderCamera.name = defaultName(camera.name.empty() ? node.name : camera.name, "Camera", cameraIndex);
    renderCamera.nodeIndex = nodeIndex;
    renderCamera.cameraIndex = cameraIndex;

    if (camera.type == "orthographic") {
        renderCamera.type = CameraType::Orthographic;
        renderCamera.xmag = camera.orthographic.xmag;
        renderCamera.ymag = camera.orthographic.ymag;
        renderCamera.znear = camera.orthographic.znear;
        renderCamera.zfar = camera.orthographic.zfar;
    } else {
        renderCamera.type = CameraType::Perspective;
        renderCamera.yfov = camera.perspective.yfov;
        renderCamera.aspectRatio = camera.perspective.aspectRatio;
        renderCamera.znear = camera.perspective.znear;
        renderCamera.zfar = camera.perspective.zfar;
    }

    renderCamera.eye = matrixTranslation(worldMatrix);
    const float3 forward = normalizedOr(
        transformVector(worldMatrix, float3(0.0f, 0.0f, -1.0f)),
        float3(0.0f, 0.0f, -1.0f));
    renderCamera.center = renderCamera.eye + forward;
    renderCamera.up = normalizedOr(
        transformVector(worldMatrix, float3(0.0f, 1.0f, 0.0f)),
        float3(0.0f, 1.0f, 0.0f));

    readVec3Value(node.extras, "camera::eye", renderCamera.eye);
    readVec3Value(node.extras, "camera::center", renderCamera.center);
    readVec3Value(node.extras, "camera::up", renderCamera.up);
    renderCamera.up = normalizedOr(renderCamera.up, float3(0.0f, 1.0f, 0.0f));

    return renderCamera;
}

RenderLight makeRenderLight(
    const tinygltf::Light& light,
    int32_t nodeIndex,
    int32_t lightIndex,
    const float4x4& worldMatrix)
{
    RenderLight renderLight;
    renderLight.name = defaultName(light.name, "Light", lightIndex);
    renderLight.type = light.type;
    renderLight.nodeIndex = nodeIndex;
    renderLight.lightIndex = lightIndex;
    renderLight.color = makeFloat3(light.color, float3(1.0f, 1.0f, 1.0f));
    renderLight.intensity = light.intensity;
    renderLight.range = light.range;
    renderLight.innerConeAngle = light.spot.innerConeAngle;
    renderLight.outerConeAngle = light.spot.outerConeAngle;
    renderLight.worldMatrix = worldMatrix;
    return renderLight;
}

RenderCamera makeFallbackCamera(const Bounds& bounds)
{
    const float radius = std::max(bounds.radius(), 1.0f);
    const float3 center = bounds.valid ? bounds.center() : float3(0.0f, 0.0f, 0.0f);
    const float distance = radius * 2.41421356f;

    RenderCamera camera;
    camera.name = "Fallback Camera";
    camera.type = CameraType::Perspective;
    camera.eye = center + float3(0.0f, 0.0f, distance);
    camera.center = center;
    camera.up = float3(0.0f, 1.0f, 0.0f);
    camera.yfov = kFallbackCameraYFov;
    camera.aspectRatio = 16.0 / 9.0;
    camera.znear = std::max(0.001, static_cast<double>(radius) * 0.001);
    camera.zfar = std::max(camera.znear * 2.0, static_cast<double>(radius) * 100.0);
    camera.fallback = true;
    return camera;
}

bool loadModel(
    const std::filesystem::path& filename,
    tinygltf::Model& model,
    LoadResult& loadResult)
{
    tinygltf::TinyGLTF loader;
    loader.SetImagesAsIs(true);
    loader.SetMaxExternalFileSize(static_cast<size_t>(-1));

    std::string error;
    std::string warning;
    const std::string filenameString = filename.string();
    const bool ok = lowerExtension(filename) == ".glb"
        ? loader.LoadBinaryFromFile(&model, &error, &warning, filenameString)
        : loader.LoadASCIIFromFile(&model, &error, &warning, filenameString);

    loadResult.warning = std::move(warning);
    loadResult.error = std::move(error);
    return ok;
}

void appendWarning(std::string& warning, std::string message)
{
    if (!warning.empty() && warning.back() != '\n') {
        warning += '\n';
    }
    warning += std::move(message);
}

bool validateRequiredExtensions(const tinygltf::Model& model, LoadResult& loadResult)
{
    for (const std::string& extension : model.extensionsRequired) {
        if (!supportedRequiredExtensions().contains(extension)) {
            loadResult.error = "Required extension unsupported: " + extension;
            return false;
        }
    }

    for (const std::string& extension : model.extensionsUsed) {
        if (!supportedRequiredExtensions().contains(extension)) {
            appendWarning(loadResult.warning, "Used extension ignored: " + extension);
        }
    }

    return true;
}

bool validIndex(int32_t index, size_t size)
{
    return index >= 0 && static_cast<size_t>(index) < size;
}

} // namespace

bool buildMeshletsForPrimitive(RenderPrimitive& primitive)
{
    const bool builtBaseMeshlets = buildMeshletClusters(primitive);
    const bool builtLodMeshlets = buildMeshletLods(primitive);
    return builtBaseMeshlets || builtLodMeshlets;
}

bool buildStreamMeshletsForPrimitive(RenderPrimitive& primitive)
{
    if (buildMeshletLods(primitive)) {
        return true;
    }
    return buildMeshletClusters(primitive);
}

void Bounds::reset()
{
    min = float3(0.0f, 0.0f, 0.0f);
    max = float3(0.0f, 0.0f, 0.0f);
    valid = false;
}

void Bounds::include(const float3& point)
{
    if (!valid) {
        min = point;
        max = point;
        valid = true;
        return;
    }

    min = float3(
        std::min(min.x, point.x),
        std::min(min.y, point.y),
        std::min(min.z, point.z));
    max = float3(
        std::max(max.x, point.x),
        std::max(max.y, point.y),
        std::max(max.z, point.z));
}

void Bounds::include(const Bounds& bounds)
{
    if (!bounds.valid) {
        return;
    }
    include(bounds.min);
    include(bounds.max);
}

float3 Bounds::center() const
{
    if (!valid) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return (min + max) * 0.5f;
}

float Bounds::radius() const
{
    if (!valid) {
        return 0.0f;
    }
    return length((max - min) * 0.5f);
}

void Scene::clear()
{
    clearParsedData();
    lastLoadResult_ = {};
}

bool Scene::load(const std::filesystem::path& filename)
{
    clearParsedData();
    lastLoadResult_ = {};
    lastLoadResult_.filename = filename;
    SceneLoadScope loadScope(filename);

    std::error_code existsError;
    if (!std::filesystem::exists(filename, existsError)) {
        lastLoadResult_.error = "Scene file does not exist: " + filename.string();
        return false;
    }

    tinygltf::Model model;
    const auto tinyGltfBegin = SceneLoadClock::now();
    if (!loadModel(filename, model, lastLoadResult_)) {
        logSceneLoadStep("tinygltf file import failed", tinyGltfBegin);
        if (lastLoadResult_.error.empty()) {
            lastLoadResult_.error = "tinygltf failed to load scene";
        }
        return false;
    }
    logSceneLoadStep("tinygltf file import", tinyGltfBegin);

    if (!validateRequiredExtensions(model, lastLoadResult_)) {
        return false;
    }

    if (model.scenes.empty()) {
        lastLoadResult_.error = "glTF model contains no scenes";
        return false;
    }

    const int32_t sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!validIndex(sceneIndex, model.scenes.size())) {
        lastLoadResult_.error = "glTF default scene index is out of range";
        return false;
    }

    const auto metadataBegin = SceneLoadClock::now();
    filename_ = filename;
    sceneIndex_ = sceneIndex;
    lastLoadResult_.sceneIndex = sceneIndex;
    sceneName_ = defaultName(model.scenes[static_cast<size_t>(sceneIndex)].name, filename.stem().string(), sceneIndex);
    assetInfo_ = SceneAssetInfo{
        .version = model.asset.version,
        .generator = model.asset.generator,
        .copyright = model.asset.copyright,
        .minVersion = model.asset.minVersion,
    };
    stats_.meshCount = model.meshes.size();
    stats_.materialCount = model.materials.size();
    stats_.textureCount = model.textures.size();
    stats_.imageCount = model.images.size();

    meshes_.reserve(model.meshes.size());
    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        const tinygltf::Mesh& gltfMesh = model.meshes[meshIndex];
        meshes_.push_back(SceneMesh{
            .name = defaultName(gltfMesh.name, "Mesh", static_cast<int32_t>(meshIndex)),
            .primitiveCount = gltfMesh.primitives.size(),
        });
    }

    images_.reserve(model.images.size());
    for (size_t imageIndex = 0; imageIndex < model.images.size(); ++imageIndex) {
        const tinygltf::Image& gltfImage = model.images[imageIndex];
        RenderImage image;
        image.name = defaultName(gltfImage.name, "Image", static_cast<int32_t>(imageIndex));
        image.uri = gltfImage.uri;
        image.mimeType = gltfImage.mimeType;
        image.bufferView = gltfImage.bufferView;
        if (validIndex(gltfImage.bufferView, model.bufferViews.size())) {
            const tinygltf::BufferView& bufferView = model.bufferViews[static_cast<size_t>(gltfImage.bufferView)];
            if (validIndex(bufferView.buffer, model.buffers.size())) {
                const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(bufferView.buffer)];
                if (bufferView.byteOffset <= buffer.data.size() &&
                    bufferView.byteLength <= buffer.data.size() - bufferView.byteOffset) {
                    const uint8_t* begin = buffer.data.data() + bufferView.byteOffset;
                    image.encodedData.assign(begin, begin + bufferView.byteLength);
                }
            }
        }
        images_.push_back(std::move(image));
    }

    textures_.reserve(model.textures.size());
    for (size_t textureIndex = 0; textureIndex < model.textures.size(); ++textureIndex) {
        const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
        RenderTexture texture;
        texture.name = defaultName(gltfTexture.name, "Texture", static_cast<int32_t>(textureIndex));
        texture.imageIndex = gltfTexture.source;
        texture.samplerIndex = gltfTexture.sampler;
        textures_.push_back(std::move(texture));
    }

    materials_.reserve(model.materials.size());
    for (size_t materialIndex = 0; materialIndex < model.materials.size(); ++materialIndex) {
        const tinygltf::Material& gltfMaterial = model.materials[materialIndex];
        RenderMaterial material;
        material.name = defaultName(gltfMaterial.name, "Material", static_cast<int32_t>(materialIndex));
        material.baseColorFactor = makeFloat4(
            gltfMaterial.pbrMetallicRoughness.baseColorFactor,
            float4(1.0f, 1.0f, 1.0f, 1.0f));
        material.metallicFactor = static_cast<float>(gltfMaterial.pbrMetallicRoughness.metallicFactor);
        material.roughnessFactor = static_cast<float>(gltfMaterial.pbrMetallicRoughness.roughnessFactor);
        material.emissiveFactor = makeFloat3(gltfMaterial.emissiveFactor, float3(0.0f, 0.0f, 0.0f));
        const auto emissiveStrengthExtension = gltfMaterial.extensions.find(kExtensionMaterialsEmissiveStrength);
        if (emissiveStrengthExtension != gltfMaterial.extensions.end()) {
            const float emissiveStrength = std::max(
                readFloatValue(emissiveStrengthExtension->second, "emissiveStrength", 1.0f),
                0.0f);
            material.emissiveFactor = material.emissiveFactor * emissiveStrength;
        }
        material.alphaCutoff = static_cast<float>(gltfMaterial.alphaCutoff);
        material.alphaMode = gltfMaterial.alphaMode.empty() ? "OPAQUE" : gltfMaterial.alphaMode;
        material.doubleSided = gltfMaterial.doubleSided;
        material.normalTextureScale = static_cast<float>(gltfMaterial.normalTexture.scale);
        material.occlusionTextureStrength = static_cast<float>(gltfMaterial.occlusionTexture.strength);
        material.baseColorTexture = makeRenderTextureInfo(gltfMaterial.pbrMetallicRoughness.baseColorTexture);
        material.metallicRoughnessTexture = makeRenderTextureInfo(
            gltfMaterial.pbrMetallicRoughness.metallicRoughnessTexture);
        material.normalTexture = makeRenderTextureInfo(gltfMaterial.normalTexture);
        material.occlusionTexture = makeRenderTextureInfo(gltfMaterial.occlusionTexture);
        material.emissiveTexture = makeRenderTextureInfo(gltfMaterial.emissiveTexture);

        const auto transmissionExtension = gltfMaterial.extensions.find(kExtensionMaterialsTransmission);
        if (transmissionExtension != gltfMaterial.extensions.end()) {
            const tinygltf::Value& transmission = transmissionExtension->second;
            material.transmissionFactor = std::clamp(
                readFloatValue(transmission, "transmissionFactor", material.transmissionFactor),
                0.0f,
                1.0f);
            readExtensionTextureInfo(transmission, "transmissionTexture", material.transmissionTexture);
        }

        const auto iorExtension = gltfMaterial.extensions.find(kExtensionMaterialsIor);
        if (iorExtension != gltfMaterial.extensions.end()) {
            material.ior = std::clamp(readFloatValue(iorExtension->second, "ior", material.ior), 1.0f, 3.0f);
        }

        const auto volumeExtension = gltfMaterial.extensions.find(kExtensionMaterialsVolume);
        if (volumeExtension != gltfMaterial.extensions.end()) {
            const tinygltf::Value& volume = volumeExtension->second;
            material.thicknessFactor = std::max(
                readFloatValue(volume, "thicknessFactor", material.thicknessFactor),
                0.0f);
            material.attenuationDistance = std::max(
                readFloatValue(volume, "attenuationDistance", material.attenuationDistance),
                0.0f);
            readVec3Value(volume, "attenuationColor", material.attenuationColor);
            readExtensionTextureInfo(volume, "thicknessTexture", material.thicknessTexture);
        }

        const auto diffuseTransmissionExtension =
            gltfMaterial.extensions.find(kExtensionMaterialsDiffuseTransmission);
        if (diffuseTransmissionExtension != gltfMaterial.extensions.end()) {
            const tinygltf::Value& diffuseTransmission = diffuseTransmissionExtension->second;
            material.diffuseTransmissionFactor = std::clamp(
                readFloatValue(
                    diffuseTransmission,
                    "diffuseTransmissionFactor",
                    material.diffuseTransmissionFactor),
                0.0f,
                1.0f);
            readVec3Value(
                diffuseTransmission,
                "diffuseTransmissionColor",
                material.diffuseTransmissionColor);
            readExtensionTextureInfo(
                diffuseTransmission,
                "diffuseTransmissionTexture",
                material.diffuseTransmissionTexture);
            readExtensionTextureInfo(
                diffuseTransmission,
                "diffuseTransmissionColorTexture",
                material.diffuseTransmissionColorTexture);
        }
        materials_.push_back(material);
    }
    logSceneLoadStep("asset metadata, images, textures, and materials", metadataBegin);

    const auto sceneGraphBegin = SceneLoadClock::now();
    nodes_.resize(model.nodes.size());
    for (size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex) {
        const tinygltf::Node& gltfNode = model.nodes[nodeIndex];
        SceneNode& node = nodes_[nodeIndex];
        node.name = defaultName(gltfNode.name, "Node", static_cast<int32_t>(nodeIndex));
        node.children = gltfNode.children;
        node.meshIndex = gltfNode.mesh;
        node.cameraIndex = gltfNode.camera;
        node.lightIndex = gltfNode.light;
        node.localMatrix = makeNodeLocalMatrix(gltfNode);
        node.worldMatrix = float4x4::Identity();
        node.visible = readNodeVisibility(gltfNode);
    }

    for (size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex) {
        for (const int32_t child : model.nodes[nodeIndex].children) {
            if (validIndex(child, nodes_.size())) {
                nodes_[static_cast<size_t>(child)].parent = static_cast<int32_t>(nodeIndex);
            }
        }
    }

    const tinygltf::Scene& gltfScene = model.scenes[static_cast<size_t>(sceneIndex)];
    for (const int32_t nodeIndex : gltfScene.nodes) {
        if (!validIndex(nodeIndex, nodes_.size())) {
            lastLoadResult_.error = "glTF scene references an out-of-range root node";
            clearParsedData();
            return false;
        }
        rootNodeIndices_.push_back(nodeIndex);
    }

    std::function<void(int32_t, const float4x4&, bool)> traverseNode;
    traverseNode = [&](int32_t nodeIndex, const float4x4& parentWorld, bool parentVisible) {
        SceneNode& node = nodes_[static_cast<size_t>(nodeIndex)];
        node.worldMatrix = parentWorld * node.localMatrix;
        node.visible = parentVisible && node.visible;

        const tinygltf::Node& gltfNode = model.nodes[static_cast<size_t>(nodeIndex)];
        if (validIndex(gltfNode.camera, model.cameras.size())) {
            cameras_.push_back(makeRenderCamera(
                model.cameras[static_cast<size_t>(gltfNode.camera)],
                gltfNode,
                nodeIndex,
                gltfNode.camera,
                node.worldMatrix));
        }
        if (validIndex(gltfNode.light, model.lights.size())) {
            lights_.push_back(makeRenderLight(
                model.lights[static_cast<size_t>(gltfNode.light)],
                nodeIndex,
                gltfNode.light,
                node.worldMatrix));
        }
        if (validIndex(gltfNode.mesh, model.meshes.size())) {
            const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(gltfNode.mesh)];
            for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex) {
                const tinygltf::Primitive& gltfPrimitive = mesh.primitives[primitiveIndex];
                RenderPrimitive primitive;
                primitive.name = defaultName(mesh.name, "Primitive", static_cast<int32_t>(renderPrimitives_.size()));
                primitive.meshIndex = gltfNode.mesh;
                primitive.primitiveIndex = static_cast<int32_t>(primitiveIndex);
                primitive.materialIndex = gltfPrimitive.material;
                primitive.mode = gltfPrimitive.mode;

                const auto positionAccessorIter = gltfPrimitive.attributes.find("POSITION");
                if (positionAccessorIter != gltfPrimitive.attributes.end() &&
                    validIndex(positionAccessorIter->second, model.accessors.size())) {
                    const tinygltf::Accessor& positionAccessor =
                        model.accessors[static_cast<size_t>(positionAccessorIter->second)];
                    primitive.vertexCount = positionAccessor.count;
                    primitive.localBounds = accessorBounds(positionAccessor);
                    primitive.positions = readPositionAccessor(model, positionAccessor);
                }

                const auto normalAccessorIter = gltfPrimitive.attributes.find("NORMAL");
                if (normalAccessorIter != gltfPrimitive.attributes.end() &&
                    validIndex(normalAccessorIter->second, model.accessors.size())) {
                    primitive.normals = readPositionAccessor(
                        model,
                        model.accessors[static_cast<size_t>(normalAccessorIter->second)]);
                    if (primitive.normals.size() != primitive.positions.size()) {
                        primitive.normals.clear();
                    } else {
                        primitive.hasAuthoredNormals = true;
                    }
                }

                const auto tangentAccessorIter = gltfPrimitive.attributes.find("TANGENT");
                if (tangentAccessorIter != gltfPrimitive.attributes.end() &&
                    validIndex(tangentAccessorIter->second, model.accessors.size())) {
                    primitive.tangents = readFloat4Accessor(
                        model,
                        model.accessors[static_cast<size_t>(tangentAccessorIter->second)]);
                    if (primitive.tangents.size() != primitive.positions.size()) {
                        primitive.tangents.clear();
                    } else {
                        primitive.hasAuthoredTangents = true;
                    }
                }

                const auto texcoordAccessorIter = gltfPrimitive.attributes.find("TEXCOORD_0");
                if (texcoordAccessorIter != gltfPrimitive.attributes.end() &&
                    validIndex(texcoordAccessorIter->second, model.accessors.size())) {
                    primitive.texcoords0 = readFloat2Accessor(
                        model,
                        model.accessors[static_cast<size_t>(texcoordAccessorIter->second)]);
                    if (primitive.texcoords0.size() != primitive.positions.size()) {
                        primitive.texcoords0.clear();
                    }
                }

                if (validIndex(gltfPrimitive.indices, model.accessors.size())) {
                    const tinygltf::Accessor& indexAccessor =
                        model.accessors[static_cast<size_t>(gltfPrimitive.indices)];
                    primitive.indexCount = indexAccessor.count;
                    primitive.indices = readIndexAccessor(model, indexAccessor);
                } else {
                    primitive.indexCount = primitive.vertexCount;
                    primitive.indices.reserve(static_cast<size_t>(primitive.vertexCount));
                    for (uint64_t index = 0; index < primitive.vertexCount; ++index) {
                        primitive.indices.push_back(static_cast<uint32_t>(index));
                    }
                }
                primitive.triangleCount = triangleCountForPrimitive(primitive.mode, primitive.indexCount);
                if (primitive.tangents.empty()) {
                    generateTangents(primitive);
                }

                RenderNode renderNode;
                renderNode.nodeIndex = nodeIndex;
                renderNode.renderPrimitiveIndex = static_cast<int32_t>(renderPrimitives_.size());
                renderNode.materialIndex = primitive.materialIndex;
                renderNode.worldMatrix = node.worldMatrix;
                renderNode.visible = node.visible;

                bounds_.include(transformBounds(primitive.localBounds, node.worldMatrix));
                stats_.triangleCount += primitive.triangleCount;
                renderPrimitives_.push_back(primitive);
                renderNodes_.push_back(renderNode);
            }
        }

        for (const int32_t child : node.children) {
            if (validIndex(child, nodes_.size())) {
                traverseNode(child, node.worldMatrix, node.visible);
            }
        }
    };

    for (const int32_t rootNodeIndex : rootNodeIndices_) {
        traverseNode(rootNodeIndex, float4x4::Identity(), true);
    }
    logSceneLoadStep("scene graph traversal and primitive extraction", sceneGraphBegin);

    const std::filesystem::path meshletCachePath = meshletCachePathFor(filename_);
    lastLoadResult_.meshletCachePath = meshletCachePath;

    std::string meshletCacheReason;
    const auto meshletCacheBegin = SceneLoadClock::now();
    if (loadMeshletCache(meshletCachePath, filename_, renderPrimitives_, meshletCacheReason)) {
        lastLoadResult_.meshletCacheLoaded = true;
        spdlog::info(
            "[SceneLoad] Meshlet cache loaded '{}' in {:.2f} ms",
            meshletCachePath.string(),
            sceneLoadElapsedMilliseconds(meshletCacheBegin));
    } else {
        spdlog::info(
            "[SceneLoad] Meshlet cache unavailable '{}' reason='{}' checked in {:.2f} ms",
            meshletCachePath.string(),
            meshletCacheReason,
            sceneLoadElapsedMilliseconds(meshletCacheBegin));
        if (!meshletCacheReason.empty()) {
            appendWarning(lastLoadResult_.warning, "Meshlet cache ignored: " + meshletCacheReason);
        }

        const auto meshletBuildBegin = SceneLoadClock::now();
        buildMeshletsForPrimitives(renderPrimitives_);
        logSceneLoadStep("meshlet build", meshletBuildBegin);

        const auto meshletSaveBegin = SceneLoadClock::now();
        if (saveMeshletCache(meshletCachePath, filename_, renderPrimitives_, meshletCacheReason)) {
            lastLoadResult_.meshletCacheSaved = true;
            spdlog::info(
                "[SceneLoad] Meshlet cache saved '{}' in {:.2f} ms",
                meshletCachePath.string(),
                sceneLoadElapsedMilliseconds(meshletSaveBegin));
        } else if (!meshletCacheReason.empty()) {
            spdlog::warn(
                "[SceneLoad] Meshlet cache save failed '{}' reason='{}' in {:.2f} ms",
                meshletCachePath.string(),
                meshletCacheReason,
                sceneLoadElapsedMilliseconds(meshletSaveBegin));
            appendWarning(lastLoadResult_.warning, "Meshlet cache save failed: " + meshletCacheReason);
        }
    }
    accumulateMeshletStats(renderPrimitives_, stats_);

    if (cameras_.empty()) {
        cameras_.push_back(makeFallbackCamera(bounds_));
    }

    stats_.primitiveCount = renderPrimitives_.size();
    stats_.renderNodeCount = renderNodes_.size();
    lastLoadResult_.success = true;
    loadScope.markSuccess();
    spdlog::info(
        "[SceneLoad] Summary scene='{}' nodes={} renderNodes={} primitives={} triangles={} meshes={} materials={} textures={} images={} meshletCacheLoaded={} meshletCacheSaved={}",
        sceneName_,
        nodes_.size(),
        stats_.renderNodeCount,
        stats_.primitiveCount,
        stats_.triangleCount,
        stats_.meshCount,
        stats_.materialCount,
        stats_.textureCount,
        stats_.imageCount,
        lastLoadResult_.meshletCacheLoaded,
        lastLoadResult_.meshletCacheSaved);
    return true;
}

void Scene::clearParsedData()
{
    filename_.clear();
    sceneName_.clear();
    sceneIndex_ = kInvalidSceneIndex;
    assetInfo_ = {};
    stats_ = {};
    bounds_.reset();
    rootNodeIndices_.clear();
    nodes_.clear();
    meshes_.clear();
    renderPrimitives_.clear();
    renderNodes_.clear();
    images_.clear();
    textures_.clear();
    materials_.clear();
    cameras_.clear();
    lights_.clear();
}

const char* cameraTypeName(CameraType type)
{
    switch (type) {
    case CameraType::Perspective:
        return "Perspective";
    case CameraType::Orthographic:
        return "Orthographic";
    }

    return "Unknown";
}

std::string formatVec3(const float3& value)
{
    std::ostringstream stream;
    stream << value.x << ", " << value.y << ", " << value.z;
    return stream.str();
}

} // namespace metallic::scene
