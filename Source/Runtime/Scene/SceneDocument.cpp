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

constexpr int kSceneDocumentVersion = 1;
constexpr std::string_view kSceneDocumentSuffix = ".metallic_scene.json";

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

bool SceneDocument::applySidecar(const std::filesystem::path& path)
{
    nlohmann::json document;
    std::string error;
    if (!readJsonFile(path, document, error)) {
        documentWarning_ = std::move(error);
        return false;
    }
    if (document.value("version", 0) != kSceneDocumentVersion) {
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
        if (!overrideValue.contains("localMatrix") ||
            !overrideValue["localMatrix"].is_array() ||
            overrideValue["localMatrix"].size() != 16) {
            appendWarning(
                documentWarning_,
                "Skipped node " + std::to_string(nodeIndex) + " because localMatrix is invalid.");
            continue;
        }

        float4x4 matrix;
        bool validMatrix = true;
        for (size_t index = 0; index < 16; ++index) {
            if (!overrideValue["localMatrix"][index].is_number()) {
                validMatrix = false;
                break;
            }
            matrix.a[index] = overrideValue["localMatrix"][index].get<float>();
            validMatrix = validMatrix && std::isfinite(matrix.a[index]);
        }
        if (!validMatrix) {
            appendWarning(
                documentWarning_,
                "Skipped node " + std::to_string(nodeIndex) + " because localMatrix is not finite.");
            continue;
        }
        Scene::setNodeLocalMatrix(nodeIndex, matrix);
    }
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
        if (matrixNearlyEqual(node.localMatrix, node.authoredLocalMatrix)) {
            continue;
        }
        nlohmann::json matrix = nlohmann::json::array();
        for (const float value : node.localMatrix.a) {
            matrix.push_back(value);
        }
        nodeOverrides.push_back({
            {"nodeIndex", nodeIndex},
            {"sourceName", node.name},
            {"localMatrix", std::move(matrix)},
        });
    }

    const nlohmann::json document{
        {"version", kSceneDocumentVersion},
        {"source", relativeSource.generic_string()},
        {"sceneIndex", sceneIndex()},
        {"nodes", std::move(nodeOverrides)},
    };
    if (!writeAtomically(documentPath_, document.dump(2) + '\n', message)) {
        return false;
    }
    dirty_ = false;
    message = "Saved scene document: " + documentPath_.string();
    return true;
}

bool SceneDocument::revert(std::string& message)
{
    const std::filesystem::path path = documentPath_.empty() ? sourcePath_ : documentPath_;
    if (path.empty() || !load(path)) {
        message = documentWarning_.empty() ? lastLoadResult().error : documentWarning_;
        return false;
    }
    message = "Reloaded scene document: " + path.string();
    return true;
}

} // namespace metallic::scene
