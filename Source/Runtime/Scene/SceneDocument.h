#pragma once

#include "Runtime/Scene/SceneEnvironment.h"
#include "Runtime/Scene/SceneLighting.h"
#include "Runtime/Scene/Scene.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <map>

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
    bool loadStreamMetadata(const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback = {});
    void clear();
    bool save(std::string& message);
    bool revert(std::string& message);
    bool setObjectLocalMatrix(SceneEntity object, const float4x4& localMatrix);
    bool setObjectWorldMatrix(SceneEntity object, const float4x4& worldMatrix);
    bool setObjectCameraProperties(SceneEntity object, const CameraProperties& properties);
    bool setObjectLightProperties(SceneEntity object, const LightProperties& properties);
    bool setMaterialProperties(int32_t materialIndex, const RenderMaterial& properties);
    bool setMaterialAsset(int32_t materialIndex, std::string_view uri,
        const std::filesystem::path& assetRoot, std::string& error);
    bool reloadMaterialAsset(int32_t materialIndex, std::string& error);
    bool setSourceMountMatrix(std::string_view sourceId, const float4x4& mountMatrix);
    bool setSourceEnabled(std::string_view sourceId, bool enabled);
    bool setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix);
    bool setEnvironment(EnvironmentSettings environment);
    bool setLighting(LightingSettings lighting);
    const LightingSettings& lighting() const;

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
        bool deferMeshletBuild, bool streamMetadata = false);
    bool loadInternalInPlace(
        const std::filesystem::path& path,
        const SceneLoadProgressCallback& progressCallback,
        bool deferMeshletBuild, bool streamMetadata);
    bool applySidecar(const std::filesystem::path& path);
    void importVirtualLights();

    std::filesystem::path sourcePath_;
    std::filesystem::path documentPath_;
    std::string documentWarning_;
    EnvironmentSettings environment_;
    // Remembers imported nodes even after their native light is deleted, so a
    // reload does not recreate it. New source nodes can still be imported.
    std::vector<ImportedLightBinding> importedLightSources_;
    std::vector<RenderMaterial> importedMaterials_;
    struct MaterialAssetBinding
    {
        std::string uri;
        std::filesystem::path root;
        RenderMaterial resolved;
    };
    std::map<int32_t, MaterialAssetBinding> materialAssets_;
    bool sidecarLoaded_ = false;
    bool hasEnvironmentSettings_ = false;
    bool compositionDocument_ = false;
    bool dirty_ = false;
};

} // namespace metallic::scene
