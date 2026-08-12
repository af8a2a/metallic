#include "Runtime/Scene/SceneDocument.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <string_view>
#include <system_error>
#include <unordered_set>

#include <json.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic::scene {
namespace {

constexpr int kSceneDocumentVersion = 4;
constexpr int kSingleSourceSceneDocumentVersion = 3;
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

bool readMatrix(
    const nlohmann::json& value,
    float4x4& matrix,
    std::string& reason)
{
    if (!value.is_array() || value.size() != 16u) {
        reason = "matrix must be a 16-number array";
        return false;
    }
    for (size_t index = 0; index < 16u; ++index) {
        if (!value[index].is_number()) {
            reason = "matrix must be a 16-number array";
            return false;
        }
        matrix.a[index] = value[index].get<float>();
        if (!std::isfinite(matrix.a[index])) {
            reason = "matrix must be finite";
            return false;
        }
    }
    return true;
}

nlohmann::json serializeMatrix(const float4x4& matrix)
{
    nlohmann::json value = nlohmann::json::array();
    for (const float component : matrix.a) {
        value.push_back(component);
    }
    return value;
}

bool parseCompositeSources(
    const nlohmann::json& document,
    const std::filesystem::path& documentPath,
    std::vector<SceneSourceDesc>& sources,
    std::string& error)
{
    if (!document.contains("sources") || !document["sources"].is_array() ||
        document["sources"].empty()) {
        error = "Scene document sources must be a non-empty array.";
        return false;
    }

    std::unordered_set<std::string> sourceIds;
    sources.clear();
    sources.reserve(document["sources"].size());
    for (size_t sourceIndex = 0; sourceIndex < document["sources"].size(); ++sourceIndex) {
        const nlohmann::json& value = document["sources"][sourceIndex];
        const std::string prefix = "Scene source " + std::to_string(sourceIndex) + ' ';
        if (!value.is_object()) {
            error = prefix + "must be an object.";
            return false;
        }
        if (!value.contains("id") || !value["id"].is_string() ||
            value["id"].get_ref<const std::string&>().empty()) {
            error = prefix + "has no non-empty string id.";
            return false;
        }
        if (!value.contains("path") || !value["path"].is_string() ||
            value["path"].get_ref<const std::string&>().empty()) {
            error = prefix + "has no non-empty string path.";
            return false;
        }
        if (value.contains("enabled") && !value["enabled"].is_boolean()) {
            error = prefix + "enabled must be a boolean.";
            return false;
        }

        SceneSourceDesc source;
        source.id = value["id"].get<std::string>();
        if (!sourceIds.emplace(source.id).second) {
            error = "Scene document contains duplicate source id '" + source.id + "'.";
            return false;
        }
        source.path = value["path"].get<std::string>();
        if (source.path.is_relative()) {
            source.path = documentPath.parent_path() / source.path;
        }
        source.path = normalizedPath(source.path);
        source.mountMatrix = float4x4::Identity();
        if (value.contains("mountMatrix")) {
            std::string reason;
            if (!readMatrix(value["mountMatrix"], source.mountMatrix, reason)) {
                error = prefix + "mountMatrix is invalid: " + reason + '.';
                return false;
            }
        }
        source.enabled = value.value("enabled", true);
        sources.push_back(std::move(source));
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
    SceneDocument candidate;
    if (!candidate.loadInternalInPlace(
            path,
            progressCallback,
            deferMeshletBuild)) {
        if (!valid()) {
            *this = std::move(candidate);
        } else {
            documentWarning_ = candidate.documentWarning_;
            if (documentWarning_.empty()) {
                documentWarning_ = candidate.lastLoadResult().error;
            }
        }
        return false;
    }
    *this = std::move(candidate);
    return true;
}

bool SceneDocument::loadInternalInPlace(
    const std::filesystem::path& path,
    const SceneLoadProgressCallback& progressCallback,
    bool deferMeshletBuild)
{
    clear();
    documentWarning_.clear();

    std::filesystem::path sourcePath = path;
    std::filesystem::path documentPath;
    bool compositionDocument = false;
    std::vector<SceneSourceDesc> compositionSources;
    if (isSceneDocumentPath(path)) {
        nlohmann::json document;
        std::string error;
        if (!readJsonFile(path, document, error)) {
            documentWarning_ = std::move(error);
            return false;
        }
        if (!document.contains("version") || !document["version"].is_number_integer()) {
            documentWarning_ = "Scene document has no integer version field.";
            return false;
        }
        const int version = document.value("version", 0);
        if (version < kOldestSceneDocumentVersion || version > kSceneDocumentVersion) {
            documentWarning_ = "Unsupported scene document version in " + path.string();
            return false;
        }
        if (version == kSceneDocumentVersion) {
            if (!parseCompositeSources(document, path, compositionSources, error)) {
                documentWarning_ = std::move(error);
                return false;
            }
            compositionDocument = true;
            documentPath = normalizedPath(path);
            sourcePath = documentPath;
        } else {
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
    }

    sourcePath = normalizedPath(sourcePath);
    const SceneLoadProgressCallback sceneProgress = progressCallback
        ? SceneLoadProgressCallback([&progressCallback](const SceneLoadProgress& sourceProgress) {
            SceneLoadProgress progress = sourceProgress;
            progress.status = SceneLoadStatus::Running;
            progress.fraction *= 0.95f;
            if (progress.phase == SceneLoadPhase::Completed) {
                progress.phase = SceneLoadPhase::Finalizing;
            }
            return progressCallback(progress);
        })
        : SceneLoadProgressCallback{};
    bool loaded = false;
    if (compositionDocument) {
        std::string error;
        loaded = Scene::compose(
            std::move(compositionSources),
            error,
            documentPath,
            sceneProgress,
            deferMeshletBuild);
        if (!loaded) {
            documentWarning_ = error.empty()
                ? "Failed to compose scene document."
                : std::move(error);
        }
    } else {
        loaded = deferMeshletBuild
            ? Scene::loadDeferredMeshlets(sourcePath, sceneProgress)
            : Scene::load(sourcePath, sceneProgress);
    }
    if (!loaded) {
        return false;
    }

    sourcePath_ = sourcePath;
    documentPath_ = documentPath.empty()
        ? sidecarPathForSource(sourcePath_)
        : normalizedPath(documentPath);
    compositionDocument_ = compositionDocument;

    std::error_code existsError;
    if (std::filesystem::exists(documentPath_, existsError) &&
        !applySidecar(documentPath_)) {
        Scene::clear();
        sourcePath_.clear();
        documentPath_.clear();
        compositionDocument_ = false;
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
        compositionDocument_ = false;
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
    compositionDocument_ = false;
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

bool SceneDocument::setSourceMountMatrix(
    std::string_view sourceId,
    const float4x4& mountMatrix)
{
    if (!Scene::setSourceMountMatrix(sourceId, mountMatrix)) {
        return false;
    }
    dirty_ = true;
    return true;
}

bool SceneDocument::setSourceEnabled(std::string_view sourceId, bool enabled)
{
    if (!Scene::setSourceEnabled(sourceId, enabled)) {
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
    if (!document.contains("version") || !document["version"].is_number_integer()) {
        documentWarning_ = "Scene document has no integer version field.";
        return false;
    }
    const int version = document.value("version", 0);
    if (version < kOldestSceneDocumentVersion || version > kSceneDocumentVersion) {
        documentWarning_ = "Unsupported scene document version in " + path.string();
        return false;
    }
    if (version == kSceneDocumentVersion) {
        std::vector<SceneSourceDesc> declaredSources;
        if (!parseCompositeSources(document, path, declaredSources, documentWarning_)) {
            return false;
        }
        if (!compositionDocument_ || normalizedPath(path) != normalizedPath(sourcePath_) ||
            declaredSources.size() != sources().size()) {
            documentWarning_ = "Scene document sources do not match the loaded composition.";
            return false;
        }
        for (size_t index = 0; index < declaredSources.size(); ++index) {
            const SceneSourceDesc& declared = declaredSources[index];
            const SceneSourceDesc& loaded = sources()[index];
            if (declared.id != loaded.id ||
                normalizedPath(declared.path) != normalizedPath(loaded.path) ||
                declared.enabled != loaded.enabled ||
                !matrixNearlyEqual(declared.mountMatrix, loaded.mountMatrix)) {
                documentWarning_ = "Scene document source '" + declared.id +
                    "' does not match the loaded composition.";
                return false;
            }
        }
    } else {
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
                    if ((environment.contains("enabled") &&
                         !environment["enabled"].is_boolean()) ||
                        (environment.contains("visible") &&
                         !environment["visible"].is_boolean()) ||
                        (environment.contains("intensity") &&
                         !environment["intensity"].is_number()) ||
                        (environment.contains("rotationDegrees") &&
                         !environment["rotationDegrees"].is_number()) ||
                        (environment.contains("path") &&
                         !environment["path"].is_string())) {
                        documentWarning_ =
                            "Scene document world.environment fields have invalid types.";
                        return false;
                    }
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
                    if (environment.contains("path")) {
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
        if (!overrideValue.contains("nodeIndex") ||
            !overrideValue["nodeIndex"].is_number_integer()) {
            appendWarning(documentWarning_, "Skipped a node override with no integer nodeIndex.");
            continue;
        }
        int32_t serializedNodeIndex = kInvalidSceneIndex;
        if (overrideValue["nodeIndex"].is_number_unsigned()) {
            const uint64_t value = overrideValue["nodeIndex"].get<uint64_t>();
            if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                appendWarning(documentWarning_, "Skipped an out-of-range node override.");
                continue;
            }
            serializedNodeIndex = static_cast<int32_t>(value);
        } else {
            const int64_t value = overrideValue["nodeIndex"].get<int64_t>();
            if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
                appendWarning(documentWarning_, "Skipped an out-of-range node override.");
                continue;
            }
            serializedNodeIndex = static_cast<int32_t>(value);
        }
        int32_t nodeIndex = serializedNodeIndex;
        ConstSceneObject object;
        std::string nodeDescription = "node " + std::to_string(serializedNodeIndex);
        if (version == kSceneDocumentVersion) {
            if (!overrideValue.contains("sourceId") ||
                !overrideValue["sourceId"].is_string() ||
                overrideValue["sourceId"].get_ref<const std::string&>().empty()) {
                appendWarning(
                    documentWarning_,
                    "Skipped node " + std::to_string(serializedNodeIndex) +
                        " because it has no non-empty sourceId.");
                continue;
            }
            const std::string sourceId = overrideValue["sourceId"].get<std::string>();
            nodeDescription = "source '" + sourceId + "' node " +
                std::to_string(serializedNodeIndex);
            object = objectForSourceNode(sourceId, serializedNodeIndex);
            const SourceNodeComponent* sourceNode =
                object.tryGetComponent<SourceNodeComponent>();
            if (sourceNode == nullptr || sourceNode->nodeIndex < 0 ||
                static_cast<size_t>(sourceNode->nodeIndex) >= nodes().size()) {
                appendWarning(
                    documentWarning_,
                    "Skipped an out-of-range " + nodeDescription + " override.");
                continue;
            }
            nodeIndex = sourceNode->nodeIndex;
        } else {
            if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= nodes().size()) {
                appendWarning(documentWarning_, "Skipped an out-of-range node override.");
                continue;
            }
            object = objectForNode(nodeIndex);
        }

        const SceneNode& node = nodes()[static_cast<size_t>(nodeIndex)];
        if (!overrideValue.contains("sourceName") ||
            !overrideValue["sourceName"].is_string() ||
            overrideValue["sourceName"].get<std::string>() != node.name) {
            appendWarning(
                documentWarning_,
                "Skipped " + nodeDescription + " because its sourceName changed.");
            continue;
        }
        bool recognizedOverride = false;
        if (overrideValue.contains("localMatrix")) {
            recognizedOverride = true;
            float4x4 matrix;
            std::string reason;
            if (!readMatrix(overrideValue["localMatrix"], matrix, reason)) {
                appendWarning(
                    documentWarning_,
                    "Skipped " + nodeDescription + " localMatrix because it is invalid: " +
                        reason + '.');
            } else if (!Scene::setObjectLocalMatrix(object.entity(), matrix)) {
                appendWarning(
                    documentWarning_,
                    "Skipped " + nodeDescription +
                        " localMatrix because the source node is not editable.");
            }
        }

        if (version >= 3 && overrideValue.contains("camera")) {
            recognizedOverride = true;
            const CameraComponent* camera = object.tryGetComponent<CameraComponent>();
            if (camera == nullptr) {
                appendWarning(
                    documentWarning_,
                    "Skipped " + nodeDescription +
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
                        "Skipped " + nodeDescription +
                            " camera override: " + reason + '.');
                } else if (!cameraPropertiesNearlyEqualForDocument(
                               camera->properties,
                               properties) &&
                    !Scene::setObjectCameraProperties(object.entity(), properties)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped " + nodeDescription +
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
                    "Skipped " + nodeDescription +
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
                        "Skipped " + nodeDescription +
                            " light override: " + reason + '.');
                } else if (!lightPropertiesNearlyEqualForDocument(
                               light->properties,
                               properties) &&
                    !Scene::setObjectLightProperties(object.entity(), properties)) {
                    appendWarning(
                        documentWarning_,
                        "Skipped " + nodeDescription +
                            " light override because its values are unsupported.");
                }
            }
        }
        if (!recognizedOverride) {
            appendWarning(
                documentWarning_,
                "Ignored " + nodeDescription +
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

    const bool compositionDocument = compositionDocument_ || sources().size() > 1u;
    struct SourceNodeIdentity {
        std::string sourceId;
        int32_t nodeIndex = kInvalidSceneIndex;
    };
    std::vector<SourceNodeIdentity> sourceNodeIdentities(nodes().size());
    if (compositionDocument) {
        for (size_t nodeIndex = 0; nodeIndex < nodes().size(); ++nodeIndex) {
            const SourceNodeComponent* component = objectForNode(
                static_cast<int32_t>(nodeIndex)).tryGetComponent<SourceNodeComponent>();
            if (component == nullptr || component->nodeIndex != static_cast<int32_t>(nodeIndex) ||
                component->sourceId.empty() || component->sourceNodeIndex < 0) {
                message = "Composite scene node mapping is invalid for node " +
                    std::to_string(nodeIndex) + '.';
                return false;
            }
            sourceNodeIdentities[nodeIndex] = {
                .sourceId = component->sourceId,
                .nodeIndex = component->sourceNodeIndex,
            };
        }
    }

    nlohmann::json nodeOverrides = nlohmann::json::array();
    for (size_t nodeIndex = 0; nodeIndex < nodes().size(); ++nodeIndex) {
        const SceneNode& node = nodes()[nodeIndex];
        nlohmann::json nodeOverride;
        bool hasOverride = false;
        if (!matrixNearlyEqual(node.localMatrix, node.authoredLocalMatrix)) {
            nodeOverride["localMatrix"] = serializeMatrix(node.localMatrix);
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
            if (compositionDocument) {
                const SourceNodeIdentity& identity = sourceNodeIdentities[nodeIndex];
                if (identity.sourceId.empty() || identity.nodeIndex < 0) {
                    message = "Composite scene node has no stable source identity: " + node.name;
                    return false;
                }
                nodeOverride["sourceId"] = identity.sourceId;
                nodeOverride["nodeIndex"] = identity.nodeIndex;
            } else {
                nodeOverride["nodeIndex"] = nodeIndex;
            }
            nodeOverride["sourceName"] = node.name;
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

    nlohmann::json document;
    if (compositionDocument) {
        nlohmann::json serializedSources = nlohmann::json::array();
        for (const SceneSourceDesc& source : sources()) {
            std::error_code relativeError;
            std::filesystem::path serializedPath = std::filesystem::relative(
                source.path,
                documentPath_.parent_path(),
                relativeError);
            if (relativeError || serializedPath.empty()) {
                serializedPath = source.path;
            }
            serializedSources.push_back(nlohmann::json{
                {"id", source.id},
                {"path", serializedPath.generic_string()},
                {"mountMatrix", serializeMatrix(source.mountMatrix)},
                {"enabled", source.enabled},
            });
        }
        document["version"] = kSceneDocumentVersion;
        document["sources"] = std::move(serializedSources);
    } else {
        std::error_code relativeError;
        std::filesystem::path relativeSource = std::filesystem::relative(
            sourcePath_,
            documentPath_.parent_path(),
            relativeError);
        if (relativeError || relativeSource.empty()) {
            relativeSource = sourcePath_;
        }
        document["version"] = kSingleSourceSceneDocumentVersion;
        document["source"] = relativeSource.generic_string();
        document["sceneIndex"] = sceneIndex();
    }
    document["nodes"] = std::move(nodeOverrides);
    document["world"] = {
        {"environment", {
            {"enabled", environment_.enabled},
            {"path", serializedEnvironmentPath.generic_string()},
            {"intensity", environment_.intensity},
            {"rotationDegrees", environment_.rotationDegrees},
            {"visible", environment_.visible},
        }},
    };
    if (!writeAtomically(documentPath_, document.dump(2) + '\n', message)) {
        return false;
    }
    sidecarLoaded_ = true;
    hasEnvironmentSettings_ = true;
    compositionDocument_ = compositionDocument;
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
