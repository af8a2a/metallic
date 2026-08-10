#include "Runtime/Scene/SceneDocument.h"

#include <cmath>
#include <fstream>
#include <string_view>
#include <system_error>

#include <json.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic::scene {
namespace {

constexpr int kSceneDocumentVersion = 3;
constexpr int kOldestSceneDocumentVersion = 1;
constexpr std::string_view kSceneDocumentSuffix = ".metallic_scene.json";

bool nearlyEqual(double lhs, double rhs)
{
    constexpr double kEpsilon = 0.000000001;
    return std::isfinite(lhs) && std::isfinite(rhs) && std::abs(lhs - rhs) <= kEpsilon;
}

bool cameraPropertiesNearlyEqualForDocument(
    const CameraProperties& lhs,
    const CameraProperties& rhs)
{
    return lhs.type == rhs.type &&
        nearlyEqual(lhs.yfov, rhs.yfov) &&
        nearlyEqual(lhs.aspectRatio, rhs.aspectRatio) &&
        nearlyEqual(lhs.xmag, rhs.xmag) &&
        nearlyEqual(lhs.ymag, rhs.ymag) &&
        nearlyEqual(lhs.znear, rhs.znear) &&
        nearlyEqual(lhs.zfar, rhs.zfar);
}

bool lightPropertiesNearlyEqualForDocument(
    const LightProperties& lhs,
    const LightProperties& rhs)
{
    constexpr float kColorEpsilon = 0.000001f;
    return lhs.type == rhs.type &&
        std::abs(lhs.color.x - rhs.color.x) <= kColorEpsilon &&
        std::abs(lhs.color.y - rhs.color.y) <= kColorEpsilon &&
        std::abs(lhs.color.z - rhs.color.z) <= kColorEpsilon &&
        nearlyEqual(lhs.intensity, rhs.intensity) &&
        nearlyEqual(lhs.range, rhs.range) &&
        nearlyEqual(lhs.innerConeAngle, rhs.innerConeAngle) &&
        nearlyEqual(lhs.outerConeAngle, rhs.outerConeAngle);
}

const char* cameraTypeDocumentName(CameraType type)
{
    return type == CameraType::Orthographic ? "orthographic" : "perspective";
}

bool readOptionalFiniteNumber(
    const nlohmann::json& object,
    const char* key,
    double& value,
    std::string& reason)
{
    if (!object.contains(key)) {
        return true;
    }
    if (!object[key].is_number()) {
        reason = std::string(key) + " must be a number";
        return false;
    }
    value = object[key].get<double>();
    if (!std::isfinite(value)) {
        reason = std::string(key) + " must be finite";
        return false;
    }
    return true;
}

bool readOptionalColor(
    const nlohmann::json& object,
    float3& color,
    std::string& reason)
{
    if (!object.contains("color")) {
        return true;
    }
    const nlohmann::json& value = object["color"];
    if (!value.is_array() || value.size() != 3u) {
        reason = "color must be a three-number array";
        return false;
    }
    for (size_t index = 0; index < 3; ++index) {
        if (!value[index].is_number()) {
            reason = "color must be a three-number array";
            return false;
        }
    }
    color = float3(
        value[0].get<float>(),
        value[1].get<float>(),
        value[2].get<float>());
    if (!std::isfinite(color.x) || !std::isfinite(color.y) || !std::isfinite(color.z)) {
        reason = "color must be finite";
        return false;
    }
    return true;
}

bool parseCameraProperties(
    const nlohmann::json& value,
    const CameraProperties& current,
    CameraProperties& properties,
    std::string& reason)
{
    if (!value.is_object()) {
        reason = "camera override must be an object";
        return false;
    }
    if (!value.contains("type") || !value["type"].is_string() ||
        value["type"].get<std::string>() != cameraTypeDocumentName(current.type)) {
        reason = "camera type no longer matches the source";
        return false;
    }
    properties = current;
    if (!readOptionalFiniteNumber(value, "znear", properties.znear, reason) ||
        !readOptionalFiniteNumber(value, "zfar", properties.zfar, reason)) {
        return false;
    }
    if (current.type == CameraType::Perspective) {
        if (value.contains("xmag") || value.contains("ymag")) {
            reason = "perspective cameras do not support orthographic magnification";
            return false;
        }
        return readOptionalFiniteNumber(value, "yfov", properties.yfov, reason) &&
            readOptionalFiniteNumber(value, "aspectRatio", properties.aspectRatio, reason);
    }
    if (value.contains("yfov") || value.contains("aspectRatio")) {
        reason = "orthographic cameras do not support perspective fields";
        return false;
    }
    return readOptionalFiniteNumber(value, "xmag", properties.xmag, reason) &&
        readOptionalFiniteNumber(value, "ymag", properties.ymag, reason);
}

bool parseLightProperties(
    const nlohmann::json& value,
    const LightProperties& current,
    LightProperties& properties,
    std::string& reason)
{
    if (!value.is_object()) {
        reason = "light override must be an object";
        return false;
    }
    if (!value.contains("type") || !value["type"].is_string() ||
        value["type"].get<std::string>() != current.type) {
        reason = "light type no longer matches the source";
        return false;
    }
    properties = current;
    if (!readOptionalColor(value, properties.color, reason) ||
        !readOptionalFiniteNumber(value, "intensity", properties.intensity, reason)) {
        return false;
    }
    if (current.type == "directional") {
        if (value.contains("range") || value.contains("innerConeAngle") ||
            value.contains("outerConeAngle")) {
            reason = "directional lights do not support range or cone angles";
            return false;
        }
        return true;
    }
    if (!readOptionalFiniteNumber(value, "range", properties.range, reason)) {
        return false;
    }
    if (current.type == "spot") {
        return readOptionalFiniteNumber(
                   value,
                   "innerConeAngle",
                   properties.innerConeAngle,
                   reason) &&
            readOptionalFiniteNumber(
                   value,
                   "outerConeAngle",
                   properties.outerConeAngle,
                   reason);
    }
    if (value.contains("innerConeAngle") || value.contains("outerConeAngle")) {
        reason = "non-spot lights do not support cone angles";
        return false;
    }
    return true;
}

nlohmann::json serializeCameraProperties(const CameraProperties& properties)
{
    nlohmann::json value{
        {"type", cameraTypeDocumentName(properties.type)},
        {"znear", properties.znear},
        {"zfar", properties.zfar},
    };
    if (properties.type == CameraType::Perspective) {
        value["yfov"] = properties.yfov;
        value["aspectRatio"] = properties.aspectRatio;
    } else {
        value["xmag"] = properties.xmag;
        value["ymag"] = properties.ymag;
    }
    return value;
}

nlohmann::json serializeLightProperties(const LightProperties& properties)
{
    nlohmann::json value{
        {"type", properties.type},
        {"color", {properties.color.x, properties.color.y, properties.color.z}},
        {"intensity", properties.intensity},
    };
    if (properties.type != "directional") {
        value["range"] = properties.range;
    }
    if (properties.type == "spot") {
        value["innerConeAngle"] = properties.innerConeAngle;
        value["outerConeAngle"] = properties.outerConeAngle;
    }
    return value;
}

bool isSceneDocumentPath(const std::filesystem::path& path)
{
    const std::string filename = path.filename().string();
    return filename.size() >= kSceneDocumentSuffix.size() &&
        filename.compare(
            filename.size() - kSceneDocumentSuffix.size(),
            kSceneDocumentSuffix.size(),
            kSceneDocumentSuffix) == 0;
}

void appendWarning(std::string& destination, std::string message)
{
    if (message.empty()) {
        return;
    }
    if (!destination.empty()) {
        destination += '\n';
    }
    destination += std::move(message);
}

std::filesystem::path normalizedPath(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(path, error);
    if (!error) {
        return normalized;
    }
    normalized = std::filesystem::absolute(path, error);
    return error ? path : normalized;
}

bool writeAtomically(
    const std::filesystem::path& path,
    const std::string& contents,
    std::string& message)
{
    const std::filesystem::path temporaryPath = path.string() + ".tmp";
    {
        std::ofstream stream(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!stream) {
            message = "Failed to open temporary scene document: " + temporaryPath.string();
            return false;
        }
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.flush();
        if (!stream) {
            message = "Failed to write temporary scene document: " + temporaryPath.string();
            return false;
        }
    }

#if defined(_WIN32)
    if (MoveFileExW(
            temporaryPath.wstring().c_str(),
            path.wstring().c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == FALSE) {
        message = "Failed to replace scene document: " + path.string();
        std::error_code cleanupError;
        std::filesystem::remove(temporaryPath, cleanupError);
        return false;
    }
#else
    std::error_code renameError;
    std::filesystem::rename(temporaryPath, path, renameError);
    if (renameError) {
        message = "Failed to replace scene document: " + renameError.message();
        std::error_code cleanupError;
        std::filesystem::remove(temporaryPath, cleanupError);
        return false;
    }
#endif
    return true;
}

bool readJsonFile(
    const std::filesystem::path& path,
    nlohmann::json& json,
    std::string& error)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "Failed to open scene document: " + path.string();
        return false;
    }
    try {
        stream >> json;
    } catch (const std::exception& exception) {
        error = "Failed to parse scene document: " + std::string(exception.what());
        return false;
    }
    return true;
}

} // namespace

std::filesystem::path SceneDocument::sidecarPathForSource(
    const std::filesystem::path& sourcePath)
{
    return sourcePath.parent_path() /
        (sourcePath.stem().string() + std::string(kSceneDocumentSuffix));
}

bool SceneDocument::load(const std::filesystem::path& path)
{
    return loadInternal(path, {}, false);
}

bool SceneDocument::load(
    const std::filesystem::path& path,
    const SceneLoadProgressCallback& progressCallback)
{
    return loadInternal(path, progressCallback, false);
}

bool SceneDocument::loadDeferredMeshlets(
    const std::filesystem::path& path,
    const SceneLoadProgressCallback& progressCallback)
{
    return loadInternal(path, progressCallback, true);
}

bool SceneDocument::loadInternal(
    const std::filesystem::path& path,
    const SceneLoadProgressCallback& progressCallback,
    bool deferMeshletBuild)
{
    clear();
    documentWarning_.clear();

    std::filesystem::path sourcePath = path;
    std::filesystem::path documentPath;
    if (isSceneDocumentPath(path)) {
        nlohmann::json document;
        std::string error;
        if (!readJsonFile(path, document, error)) {
            documentWarning_ = std::move(error);
            return false;
        }
        if (!document.contains("source") || !document["source"].is_string()) {
            documentWarning_ = "Scene document has no string source field.";
            return false;
        }
        sourcePath = document["source"].get<std::string>();
        if (sourcePath.is_relative()) {
            sourcePath = path.parent_path() / sourcePath;
        }
        documentPath = path;
    }

    sourcePath = normalizedPath(sourcePath);
    const SceneLoadProgressCallback sceneProgress = progressCallback
        ? SceneLoadProgressCallback([&progressCallback](const SceneLoadProgress& sourceProgress) {
            SceneLoadProgress progress = sourceProgress;
            progress.fraction *= 0.95f;
            if (progress.phase == SceneLoadPhase::Completed) {
                progress.phase = SceneLoadPhase::Finalizing;
            }
            return progressCallback(progress);
        })
        : SceneLoadProgressCallback{};
    const bool loaded = deferMeshletBuild
        ? Scene::loadDeferredMeshlets(sourcePath, sceneProgress)
        : Scene::load(sourcePath, sceneProgress);
    if (!loaded) {
        return false;
    }

    sourcePath_ = sourcePath;
    documentPath_ = documentPath.empty()
        ? sidecarPathForSource(sourcePath_)
        : normalizedPath(documentPath);

    std::error_code existsError;
    if (std::filesystem::exists(documentPath_, existsError) &&
        !applySidecar(documentPath_)) {
        Scene::clear();
        sourcePath_.clear();
        documentPath_.clear();
        return false;
    }
    if (progressCallback && !progressCallback(SceneLoadProgress{
            .status = SceneLoadStatus::Running,
            .phase = SceneLoadPhase::Finalizing,
            .fraction = 0.99f,
            .completedUnits = 1,
            .totalUnits = 1,
            .currentItem = documentPath_.string(),
        })) {
        Scene::clear();
        sourcePath_.clear();
        documentPath_.clear();
        documentWarning_ = "Scene load cancelled.";
        return false;
    }
    dirty_ = false;
    if (progressCallback) {
        (void)progressCallback(SceneLoadProgress{
            .status = SceneLoadStatus::Succeeded,
            .phase = SceneLoadPhase::Completed,
            .fraction = 1.0f,
            .completedUnits = 1,
            .totalUnits = 1,
            .currentItem = sourcePath_.string(),
        });
    }
    return true;
}

void SceneDocument::clear()
{
    Scene::clear();
    sourcePath_.clear();
    documentPath_.clear();
    documentWarning_.clear();
    environment_ = EnvironmentSettings{};
    sidecarLoaded_ = false;
    hasEnvironmentSettings_ = false;
    dirty_ = false;
}

bool SceneDocument::setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix)
{
    if (!Scene::setNodeLocalMatrix(nodeIndex, localMatrix)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setObjectLocalMatrix(SceneEntity object, const float4x4& localMatrix)
{
    const ConstSceneObject sceneObject = sceneGraph().object(object);
    if (!sceneObject || !sceneObject.hasComponent<SourceNodeComponent>() ||
        !Scene::setObjectLocalMatrix(object, localMatrix)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setObjectWorldMatrix(SceneEntity object, const float4x4& worldMatrix)
{
    const ConstSceneObject sceneObject = sceneGraph().object(object);
    if (!sceneObject || !sceneObject.hasComponent<SourceNodeComponent>() ||
        !Scene::setObjectWorldMatrix(object, worldMatrix)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setObjectCameraProperties(
    SceneEntity object,
    const CameraProperties& properties)
{
    const ConstSceneObject sceneObject = sceneGraph().object(object);
    if (!sceneObject || !sceneObject.hasComponent<SourceNodeComponent>() ||
        !Scene::setObjectCameraProperties(object, properties)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setObjectLightProperties(
    SceneEntity object,
    const LightProperties& properties)
{
    const ConstSceneObject sceneObject = sceneGraph().object(object);
    if (!sceneObject || !sceneObject.hasComponent<SourceNodeComponent>() ||
        !Scene::setObjectLightProperties(object, properties)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setEnvironment(EnvironmentSettings environment)
{
    environment.intensity = std::isfinite(environment.intensity)
        ? std::max(environment.intensity, 0.0f)
        : 1.0f;
    environment.rotationDegrees = std::isfinite(environment.rotationDegrees)
        ? environment.rotationDegrees
        : 0.0f;
    if (environment_ == environment) {
        return false;
    }
    environment_ = std::move(environment);
    hasEnvironmentSettings_ = true;
    dirty_ = true;
    return true;
}

bool SceneDocument::applySidecar(const std::filesystem::path& path)
{
    nlohmann::json document;
    std::string error;
    if (!readJsonFile(path, document, error)) {
        documentWarning_ = std::move(error);
        return false;
    }
    const int version = document.value("version", 0);
    if (version < kOldestSceneDocumentVersion || version > kSceneDocumentVersion) {
        documentWarning_ = "Unsupported scene document version in " + path.string();
        return false;
    }
    if (!document.contains("source") || !document["source"].is_string()) {
        documentWarning_ = "Scene document has no string source field.";
        return false;
    }

    std::filesystem::path declaredSource = document["source"].get<std::string>();
    if (declaredSource.is_relative()) {
        declaredSource = path.parent_path() / declaredSource;
    }
    if (normalizedPath(declaredSource) != normalizedPath(sourcePath_)) {
        documentWarning_ = "Scene document source does not match the loaded glTF.";
        return false;
    }
    if (document.value("sceneIndex", kInvalidSceneIndex) != sceneIndex()) {
        documentWarning_ = "Scene document sceneIndex does not match the loaded glTF scene.";
        return false;
    }

    environment_ = EnvironmentSettings{};
    if (version >= 2 && document.contains("world")) {
        if (!document["world"].is_object()) {
            appendWarning(documentWarning_, "Ignored a non-object world setting.");
        } else {
            const nlohmann::json& world = document["world"];
            if (world.contains("environment")) {
                if (!world["environment"].is_object()) {
                    appendWarning(documentWarning_, "Ignored a non-object world.environment setting.");
                } else {
                    const nlohmann::json& environment = world["environment"];
                    hasEnvironmentSettings_ = true;
                    environment_.enabled = environment.value("enabled", true);
                    environment_.visible = environment.value("visible", true);
                    environment_.intensity = environment.value("intensity", 1.0f);
                    environment_.rotationDegrees = environment.value("rotationDegrees", 0.0f);
                    if (!std::isfinite(environment_.intensity)) {
                        environment_.intensity = 1.0f;
                    }
                    environment_.intensity = std::max(environment_.intensity, 0.0f);
                    if (!std::isfinite(environment_.rotationDegrees)) {
                        environment_.rotationDegrees = 0.0f;
                    }
                    if (environment.contains("path") && environment["path"].is_string()) {
                        environment_.path = environment["path"].get<std::string>();
                        if (!environment_.path.empty() && environment_.path.is_relative()) {
                            environment_.path = path.parent_path() / environment_.path;
                        }
                        if (!environment_.path.empty()) {
                            environment_.path = normalizedPath(environment_.path);
                        }
                    }
                }
            }
        }
    }

    const nlohmann::json& overrides = document.contains("nodes")
        ? document["nodes"]
        : nlohmann::json::array();
    if (!overrides.is_array()) {
        documentWarning_ = "Scene document nodes field must be an array.";
        return false;
    }

    for (const nlohmann::json& overrideValue : overrides) {
        if (!overrideValue.is_object()) {
            appendWarning(documentWarning_, "Skipped a non-object node override.");
            continue;
        }
        const int32_t nodeIndex = overrideValue.value("nodeIndex", kInvalidSceneIndex);
        if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes().size()) {
            appendWarning(documentWarning_, "Skipped an out-of-range node override.");
            continue;
        }
        const SceneNode& node = nodes()[static_cast<size_t>(nodeIndex)];
        if (!overrideValue.contains("sourceName") ||
            !overrideValue["sourceName"].is_string() ||
            overrideValue["sourceName"].get<std::string>() != node.name) {
            appendWarning(
                documentWarning_,
                "Skipped node " + std::to_string(nodeIndex) + " because its sourceName changed.");
            continue;
        }
        bool recognizedOverride = false;
        if (overrideValue.contains("localMatrix")) {
            recognizedOverride = true;
            const nlohmann::json& matrixValue = overrideValue["localMatrix"];
            if (!matrixValue.is_array() || matrixValue.size() != 16) {
                appendWarning(
                    documentWarning_,
                    "Skipped node " + std::to_string(nodeIndex) +
                        " localMatrix because it is invalid.");
            } else {
                float4x4 matrix;
                bool validMatrix = true;
                for (size_t index = 0; index < 16; ++index) {
                    if (!matrixValue[index].is_number()) {
                        validMatrix = false;
                        break;
                    }
                    matrix.a[index] = matrixValue[index].get<float>();
                    validMatrix = validMatrix && std::isfinite(matrix.a[index]);
                }
                if (!validMatrix) {
                    appendWarning(
                        documentWarning_,
                        "Skipped node " + std::to_string(nodeIndex) +
                            " localMatrix because it is not finite.");
                } else {
                    Scene::setNodeLocalMatrix(nodeIndex, matrix);
                }
            }
        }

        const ConstSceneObject object = objectForNode(nodeIndex);
        if (version >= 3 && overrideValue.contains("camera")) {
            recognizedOverride = true;
            const CameraComponent* camera = object.tryGetComponent<CameraComponent>();
            if (camera == nullptr) {
                appendWarning(
                    documentWarning_,
                    "Skipped node " + std::to_string(nodeIndex) +
                        " camera override because the source node has no camera.");
            } else {
                CameraProperties properties;
                std::string reason;
                if (!parseCameraProperties(
                        overrideValue["camera"],
                        camera->properties,
                        properties,
                        reason)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped node " + std::to_string(nodeIndex) +
                            " camera override: " + reason + '.');
                } else if (!cameraPropertiesNearlyEqualForDocument(
                               camera->properties,
                               properties) &&
                    !Scene::setObjectCameraProperties(object.entity(), properties)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped node " + std::to_string(nodeIndex) +
                            " camera override because its values are unsupported.");
                }
            }
        }
        if (version >= 3 && overrideValue.contains("light")) {
            recognizedOverride = true;
            const LightComponent* light = object.tryGetComponent<LightComponent>();
            if (light == nullptr) {
                appendWarning(
                    documentWarning_,
                    "Skipped node " + std::to_string(nodeIndex) +
                        " light override because the source node has no light.");
            } else {
                LightProperties properties;
                std::string reason;
                if (!parseLightProperties(
                        overrideValue["light"],
                        light->properties,
                        properties,
                        reason)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped node " + std::to_string(nodeIndex) +
                            " light override: " + reason + '.');
                } else if (!lightPropertiesNearlyEqualForDocument(
                               light->properties,
                               properties) &&
                    !Scene::setObjectLightProperties(object.entity(), properties)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped node " + std::to_string(nodeIndex) +
                            " light override because its values are unsupported.");
                }
            }
        }
        if (!recognizedOverride) {
            appendWarning(
                documentWarning_,
                "Ignored node " + std::to_string(nodeIndex) +
                    " override because it has no supported properties.");
        }
    }
    sidecarLoaded_ = true;
    return true;
}

bool SceneDocument::save(std::string& message)
{
    message.clear();
    if (!valid() || sourcePath_.empty() || documentPath_.empty()) {
        message = "No scene document is loaded.";
        return false;
    }

    std::error_code relativeError;
    std::filesystem::path relativeSource = std::filesystem::relative(
        sourcePath_,
        documentPath_.parent_path(),
        relativeError);
    if (relativeError || relativeSource.empty()) {
        relativeSource = sourcePath_;
    }

    nlohmann::json nodeOverrides = nlohmann::json::array();
    for (size_t nodeIndex = 0; nodeIndex < nodes().size(); ++nodeIndex) {
        const SceneNode& node = nodes()[nodeIndex];
        nlohmann::json nodeOverride{
            {"nodeIndex", nodeIndex},
            {"sourceName", node.name},
        };
        bool hasOverride = false;
        if (!matrixNearlyEqual(node.localMatrix, node.authoredLocalMatrix)) {
            nlohmann::json matrix = nlohmann::json::array();
            for (const float value : node.localMatrix.a) {
                matrix.push_back(value);
            }
            nodeOverride["localMatrix"] = std::move(matrix);
            hasOverride = true;
        }

        const ConstSceneObject object = objectForNode(static_cast<int32_t>(nodeIndex));
        if (const CameraComponent* camera = object.tryGetComponent<CameraComponent>();
            camera != nullptr && !cameraPropertiesNearlyEqualForDocument(
                camera->properties,
                camera->authoredProperties)) {
            nodeOverride["camera"] = serializeCameraProperties(camera->properties);
            hasOverride = true;
        }
        if (const LightComponent* light = object.tryGetComponent<LightComponent>();
            light != nullptr && !lightPropertiesNearlyEqualForDocument(
                light->properties,
                light->authoredProperties)) {
            nodeOverride["light"] = serializeLightProperties(light->properties);
            hasOverride = true;
        }
        if (hasOverride) {
            nodeOverrides.push_back(std::move(nodeOverride));
        }
    }

    std::filesystem::path serializedEnvironmentPath = environment_.path;
    if (!serializedEnvironmentPath.empty()) {
        std::error_code environmentRelativeError;
        const std::filesystem::path relativeEnvironment = std::filesystem::relative(
            serializedEnvironmentPath,
            documentPath_.parent_path(),
            environmentRelativeError);
        if (!environmentRelativeError && !relativeEnvironment.empty()) {
            serializedEnvironmentPath = relativeEnvironment;
        }
    }

    const nlohmann::json document{
        {"version", kSceneDocumentVersion},
        {"source", relativeSource.generic_string()},
        {"sceneIndex", sceneIndex()},
        {"nodes", std::move(nodeOverrides)},
        {"world", {
            {"environment", {
                {"enabled", environment_.enabled},
                {"path", serializedEnvironmentPath.generic_string()},
                {"intensity", environment_.intensity},
                {"rotationDegrees", environment_.rotationDegrees},
                {"visible", environment_.visible},
            }},
        }},
    };
    if (!writeAtomically(documentPath_, document.dump(2) + '\n', message)) {
        return false;
    }
    sidecarLoaded_ = true;
    hasEnvironmentSettings_ = true;
    dirty_ = false;
    message = "Saved scene document: " + documentPath_.string();
    return true;
}

bool SceneDocument::revert(std::string& message)
{
    std::error_code existsError;
    const bool documentExists = !documentPath_.empty() &&
        std::filesystem::exists(documentPath_, existsError) && !existsError;
    const std::filesystem::path path = documentExists ? documentPath_ : sourcePath_;
    if (path.empty()) {
        message = "Scene document has no source to reload.";
        return false;
    }

    SceneDocument reverted;
    if (!reverted.load(path)) {
        message = reverted.documentWarning().empty()
            ? reverted.lastLoadResult().error
            : reverted.documentWarning();
        return false;
    }
    *this = std::move(reverted);
    message = "Reloaded scene document: " + path.string();
    return true;
}

} // namespace metallic::scene
