#pragma once

#include "Runtime/Scene/SceneEnvironment.h"
#include "Runtime/Scene/SceneLighting.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Scene/TerrainHeightField.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <map>
#include <utility>

namespace metallic::scene {

enum class WorldItemType : uint8_t { LocalFogVolume, PunctualLight, RectLight };

struct WorldItemId {
    WorldItemType type;
    uint64_t id;
    bool operator==(const WorldItemId&) const = default;
};

enum class TerrainPlacementKind : uint8_t { Rock, Vegetation };

struct TerrainPlacement {
    uint64_t id = 0;
    TerrainPlacementKind kind = TerrainPlacementKind::Rock;
    float worldX = 0.0f;
    float worldZ = 0.0f;
    bool enabled = true;
    SceneEntity object = kNullSceneEntity;
};

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
    bool setRenderNodeLocalTransform(int32_t renderNodeIndex, const float4x4& localTransform);
    bool setRenderNodeWorldTransform(int32_t renderNodeIndex, const float4x4& worldTransform);
    bool setRenderNodeLocalOffset(int32_t renderNodeIndex, const float3& localOffset);
    bool setRenderNodeWorldPosition(int32_t renderNodeIndex, const float3& worldPosition);
    bool setObjectCameraProperties(SceneEntity object, const CameraProperties& properties);
    bool setObjectLightProperties(SceneEntity object, const LightProperties& properties);
    bool setMaterialProperties(int32_t materialIndex, const RenderMaterial& properties);
    bool setDecal(RenderDecal decal, const std::filesystem::path& baseColorFile = {},
        const std::filesystem::path& roughnessFile = {});
    bool removeDecal(uint32_t id);
    SceneEntity decalSceneGraphParent(uint32_t id) const;
    std::string_view decalName(uint32_t id) const;
    SceneEntity worldItemSceneGraphParent(WorldItemId item) const;
    std::string_view worldItemName(WorldItemId item) const;
    bool setMaterialAsset(int32_t materialIndex, std::string_view uri,
        const std::filesystem::path& assetRoot, std::string& error);
    bool reloadMaterialAsset(int32_t materialIndex, std::string& error);
    bool setSourceMountMatrix(std::string_view sourceId, const float4x4& mountMatrix);
    bool setSourceEnabled(std::string_view sourceId, bool enabled);
    bool setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix);
    bool setEnvironment(EnvironmentSettings environment);
    bool setWorldEnvironment(environment::WorldEnvironment environment);
    bool setLighting(LightingSettings lighting);
    const LightingSettings& lighting() const;

    const TerrainHeightField* terrainHeightField() const { return heightField_ ? &*heightField_ : nullptr; }
    float terrainCameraClearance() const { return terrainCameraClearance_; }
    bool supportsTerrainPlacement() const { return terrainPrototypes_.front() != kNullSceneEntity; }
    const std::vector<TerrainPlacement>& terrainPlacements() const { return terrainPlacements_; }
    const TerrainPlacement* terrainPlacementForObject(SceneEntity object) const;
    SceneEntity terrainPlacementObject(uint64_t id) const;
    std::optional<uint64_t> addTerrainPlacement(TerrainPlacementKind kind, float worldX, float worldZ,
        std::string& error);
    bool setTerrainPlacementEnabled(uint64_t id, bool enabled, std::string& error);

    bool dirty() const { return dirty_; }
    void setDirty(bool dirty) { dirty_ = dirty; }
    const std::filesystem::path& sourcePath() const { return sourcePath_; }
    const std::filesystem::path& documentPath() const { return documentPath_; }
    const std::string& documentWarning() const { return documentWarning_; }
    const EnvironmentSettings& environment() const { return environment_; }
    bool sidecarLoaded() const { return sidecarLoaded_; }
    bool hasEnvironmentSettings() const { return hasEnvironmentSettings_; }

    struct StreamGeometrySource {
        std::filesystem::path path;
        bool requiresSourceTriangles = false;
    };

    static std::filesystem::path sidecarPathForSource(const std::filesystem::path& sourcePath);
    static StreamGeometrySource streamGeometrySource(const std::filesystem::path& path);

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
    bool loadTerrain(const std::filesystem::path& heightField, std::array<std::string, 2> prototypes,
        std::vector<TerrainPlacement> placements);
    SceneEntity instantiateTerrainPlacement(const TerrainPlacement& placement, const TerrainSurfaceSample& surface);
    TerrainPlacement* findTerrainPlacement(SceneEntity object);
    void importVirtualLights();
    void pruneWorldItemPresentations();

    std::filesystem::path sourcePath_;
    std::filesystem::path documentPath_;
    std::filesystem::path heightFieldPath_;
    std::optional<TerrainHeightField> heightField_;
    float terrainCameraClearance_ = 0.0f;
    std::array<std::string, 2> terrainPrototypeNames_;
    std::array<SceneEntity, 2> terrainPrototypes_{kNullSceneEntity, kNullSceneEntity};
    std::vector<TerrainPlacement> terrainPlacements_;
    size_t terrainSourceNodeCount_ = 0;
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
        bool ownsValueProgram = false;
    };
    std::map<int32_t, MaterialAssetBinding> materialAssets_;
    struct SceneGraphPresentation
    {
        std::string name;
        std::string parentSourceId;
        int32_t parentNodeIndex = kInvalidSceneIndex;
        SceneEntity parent = kNullSceneEntity;
    };
    std::map<uint32_t, SceneGraphPresentation> decalPresentations_;
    std::map<std::pair<WorldItemType, uint64_t>, SceneGraphPresentation> worldItemPresentations_;
    bool sidecarLoaded_ = false;
    bool hasEnvironmentSettings_ = false;
    bool compositionDocument_ = false;
    bool dirty_ = false;
};

} // namespace metallic::scene
