#pragma once

#include "Runtime/Scene/SceneEnvironment.h"
#include "Runtime/Scene/Scene.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace metallic::scene {

class SceneDocument : public Scene {
public:
    bool load(const std::filesystem::path& path);
    bool load(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback);
    bool loadDeferredMeshlets(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback);
    void clear();
    bool save(std::string& message);
    bool revert(std::string& message);
    bool setObjectLocalMatrix(SceneEntity object, const float4x4& localMatrix);
    bool setObjectWorldMatrix(SceneEntity object, const float4x4& worldMatrix);
    bool setObjectCameraProperties(SceneEntity object, const CameraProperties& properties);
    bool setObjectLightProperties(SceneEntity object, const LightProperties& properties);
    bool setSourceMountMatrix(std::string_view sourceId, const float4x4& mountMatrix);
    bool setSourceEnabled(std::string_view sourceId, bool enabled);
    bool setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix);
    bool setEnvironment(EnvironmentSettings environment);

    bool dirty() const { return dirty_; }
    void setDirty(bool dirty) { dirty_ = dirty; }
    const std::filesystem::path& sourcePath() const { return sourcePath_; }
    const std::filesystem::path& documentPath() const { return documentPath_; }
    const std::string& documentWarning() const { return documentWarning_; }
    const EnvironmentSettings& environment() const { return environment_; }
    bool sidecarLoaded() const { return sidecarLoaded_; }
    bool hasEnvironmentSettings() const { return hasEnvironmentSettings_; }

    static std::filesystem::path sidecarPathForSource(const std::filesystem::path& sourcePath);

private:
    bool loadInternal(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback,
        bool deferMeshletBuild);
    bool loadInternalInPlace(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback,
        bool deferMeshletBuild);
    bool applySidecar(const std::filesystem::path& path);

    std::filesystem::path sourcePath_;
    std::filesystem::path documentPath_;
    std::string documentWarning_;
    EnvironmentSettings environment_;
    bool sidecarLoaded_ = false;
    bool hasEnvironmentSettings_ = false;
    bool compositionDocument_ = false;
    bool dirty_ = false;
};

} // namespace metallic::scene
