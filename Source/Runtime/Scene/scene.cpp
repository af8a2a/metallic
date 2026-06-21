/*
 * CPU glTF scene import and traversal code.
 *
 * Portions are adapted from NVIDIA nvpro_core2 nvvkgltf scene traversal ideas.
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime/Scene/Scene.h"

#include "json.hpp"
#include "tiny_gltf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <sstream>
#include <string_view>
#include <unordered_set>

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

    std::error_code existsError;
    if (!std::filesystem::exists(filename, existsError)) {
        lastLoadResult_.error = "Scene file does not exist: " + filename.string();
        return false;
    }

    tinygltf::Model model;
    if (!loadModel(filename, model, lastLoadResult_)) {
        if (lastLoadResult_.error.empty()) {
            lastLoadResult_.error = "tinygltf failed to load scene";
        }
        return false;
    }

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

    if (cameras_.empty()) {
        cameras_.push_back(makeFallbackCamera(bounds_));
    }

    stats_.primitiveCount = renderPrimitives_.size();
    stats_.renderNodeCount = renderNodes_.size();
    lastLoadResult_.success = true;
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
