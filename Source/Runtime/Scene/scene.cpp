/*
 * CPU glTF scene import and traversal code.
 *
 * Portions are adapted from NVIDIA nvpro_core2 nvvkgltf scene traversal ideas.
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime/Scene/scene.h"

#include "json.hpp"
#include "tiny_gltf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace metallic::scene {
namespace {

constexpr const char* kExtensionLightsPunctual = "KHR_lights_punctual";
constexpr const char* kExtensionNodeVisibility = "KHR_node_visibility";
constexpr double kFallbackCameraYFov = 0.7853981633974483;

const std::unordered_set<std::string>& supportedRequiredExtensions()
{
    static const std::unordered_set<std::string> kExtensions{
        kExtensionLightsPunctual,
        kExtensionNodeVisibility,
        "KHR_materials_anisotropy",
        "KHR_materials_clearcoat",
        "KHR_materials_diffuse_transmission",
        "KHR_materials_dispersion",
        "KHR_materials_emissive_strength",
        "KHR_materials_ior",
        "KHR_materials_iridescence",
        "KHR_materials_pbrSpecularGlossiness",
        "KHR_materials_sheen",
        "KHR_materials_specular",
        "KHR_materials_transmission",
        "KHR_materials_unlit",
        "KHR_materials_volume",
        "KHR_materials_volume_scatter",
        "KHR_mesh_quantization",
        "KHR_texture_transform",
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
    stats_.meshCount = model.meshes.size();
    stats_.materialCount = model.materials.size();

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
                }

                if (validIndex(gltfPrimitive.indices, model.accessors.size())) {
                    primitive.indexCount = model.accessors[static_cast<size_t>(gltfPrimitive.indices)].count;
                } else {
                    primitive.indexCount = primitive.vertexCount;
                }
                primitive.triangleCount = triangleCountForPrimitive(primitive.mode, primitive.indexCount);

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
    stats_ = {};
    bounds_.reset();
    rootNodeIndices_.clear();
    nodes_.clear();
    renderPrimitives_.clear();
    renderNodes_.clear();
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
