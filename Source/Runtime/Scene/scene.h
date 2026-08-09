#pragma once


#include <cstdint>
#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include "ml.h"
#include "Runtime/Scene/SceneLoad.h"
#include "Runtime/Scene/SceneGraph.h"

namespace metallic::scene {

inline constexpr int32_t kInvalidSceneIndex = -1;

struct Bounds {
    float3 min{0.0f};
    float3 max{0.0f};
    bool valid = false;

    void reset();
    void include(const float3& point);
    void include(const Bounds& bounds);
    float3 center() const;
    float radius() const;
};

enum class CameraType : uint8_t {
    Perspective,
    Orthographic,
};

struct LoadResult {
    bool success = false;
    std::filesystem::path filename;
    std::string warning;
    std::string error;
    int32_t sceneIndex = kInvalidSceneIndex;
    bool meshletCacheLoaded = false;
    bool meshletCacheSaved = false;
    std::filesystem::path meshletCachePath;
};

struct SceneStats {
    uint64_t meshCount = 0;
    uint64_t materialCount = 0;
    uint64_t textureCount = 0;
    uint64_t imageCount = 0;
    uint64_t primitiveCount = 0;
    uint64_t renderNodeCount = 0;
    uint64_t triangleCount = 0;
    uint64_t meshletClusterCount = 0;
    uint64_t meshletVertexReferenceCount = 0;
    uint64_t meshletTriangleIndexCount = 0;
    uint64_t meshletLodLevelCount = 0;
    uint64_t meshletLodGroupCount = 0;
    uint64_t meshletLodClusterCount = 0;
    uint64_t meshletLodVertexReferenceCount = 0;
    uint64_t meshletLodTriangleIndexCount = 0;
};

struct SceneAssetInfo {
    std::string version;
    std::string generator;
    std::string copyright;
    std::string minVersion;
};

struct SceneMesh {
    std::string name;
    uint64_t primitiveCount = 0;
};

struct MeshletCluster {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupChildIndex = 0;
    int32_t lodGroupIndex = kInvalidSceneIndex;
    int32_t refinedGroupIndex = kInvalidSceneIndex;
    float lodError = 0.0f;
    Bounds bounds;
    float3 boundingSphereCenter{0.0f, 0.0f, 0.0f};
    float boundingSphereRadius = 0.0f;
    float3 coneApex{0.0f, 0.0f, 0.0f};
    float3 coneAxis{0.0f, 0.0f, 1.0f};
    float coneCutoff = 1.0f;
    std::array<int8_t, 4> packedCone{0, 0, 127, 127};
};

struct MeshletLodGroup {
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
    uint32_t lodLevel = 0;
    Bounds bounds;
    float3 boundingSphereCenter{0.0f, 0.0f, 0.0f};
    float boundingSphereRadius = 0.0f;
    float maxQuadricError = 0.0f;
};

struct MeshletLodLevel {
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
    float minBoundingSphereRadius = 0.0f;
    float minMaxQuadricError = 0.0f;
};

// Read-only compatibility projection of the ECS-owned scene object data.
struct SceneNode {
    std::string name;
    int32_t parent = kInvalidSceneIndex;
    std::vector<int32_t> children;
    int32_t meshIndex = kInvalidSceneIndex;
    int32_t cameraIndex = kInvalidSceneIndex;
    int32_t lightIndex = kInvalidSceneIndex;
    float4x4 authoredLocalMatrix = float4x4::Identity();
    float4x4 localMatrix = float4x4::Identity();
    float4x4 worldMatrix = float4x4::Identity();
    uint64_t transformRevision = 0;
    bool visible = true;
};

struct RenderPrimitive {
    std::string name;
    int32_t meshIndex = kInvalidSceneIndex;
    int32_t primitiveIndex = kInvalidSceneIndex;
    int32_t materialIndex = kInvalidSceneIndex;
    int32_t mode = 4;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t triangleCount = 0;
    Bounds localBounds;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float4> tangents;
    std::vector<float2> texcoords0;
    bool hasAuthoredNormals = false;
    bool hasAuthoredTangents = false;
    std::vector<uint32_t> indices;
    std::vector<MeshletCluster> meshletClusters;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint8_t> meshletTriangles;
    std::vector<MeshletLodLevel> meshletLodLevels;
    std::vector<MeshletLodGroup> meshletLodGroups;
    std::vector<MeshletCluster> meshletLodClusters;
    std::vector<uint32_t> meshletLodVertices;
    std::vector<uint8_t> meshletLodTriangles;
};

struct RenderImage {
    std::string name;
    std::string uri;
    std::string mimeType;
    int32_t bufferView = kInvalidSceneIndex;
    std::vector<uint8_t> encodedData;
    struct Mip {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<uint8_t> pixels;
    };
    std::vector<Mip> decodedMips;
    std::string decodeWarning;
    bool decodeAttempted = false;
};

struct RenderTexture {
    std::string name;
    int32_t imageIndex = kInvalidSceneIndex;
    int32_t samplerIndex = kInvalidSceneIndex;
};

struct RenderTextureInfo {
    int32_t textureIndex = kInvalidSceneIndex;
    int32_t texCoord = 0;
    std::array<float, 6> uvTransform{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
};

struct RenderNode {
    SceneEntity object = kNullSceneEntity;
    int32_t nodeIndex = kInvalidSceneIndex;
    int32_t renderPrimitiveIndex = kInvalidSceneIndex;
    int32_t materialIndex = kInvalidSceneIndex;
    float4x4 worldMatrix = float4x4::Identity();
    uint64_t transformRevision = 0;
    bool visible = true;
};

struct RenderMaterial {
    std::string name;
    float4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float3 emissiveFactor{0.0f, 0.0f, 0.0f};
    float alphaCutoff = 0.5f;
    std::string alphaMode = "OPAQUE";
    bool doubleSided = false;
    float normalTextureScale = 1.0f;
    float occlusionTextureStrength = 1.0f;
    float transmissionFactor = 0.0f;
    float ior = 1.5f;
    float thicknessFactor = 0.0f;
    float attenuationDistance = 0.0f;
    float3 attenuationColor{1.0f, 1.0f, 1.0f};
    float diffuseTransmissionFactor = 0.0f;
    float3 diffuseTransmissionColor{1.0f, 1.0f, 1.0f};
    bool rtxcrHair = false;
    float3 rtxcrHairBaseColor{0.2f, 0.2f, 0.2f};
    float rtxcrHairMelanin = 1.0f;
    float rtxcrHairMelaninRedness = 0.0f;
    float rtxcrHairLongitudinalRoughness = 0.3f;
    float rtxcrHairAzimuthalRoughness = 0.3f;
    float rtxcrHairIor = 1.55f;
    float rtxcrHairCuticleAngleDegrees = 3.0f;
    float rtxcrHairDiffuseReflectionWeight = 0.0f;
    float3 rtxcrHairDiffuseReflectionTint{0.0f, 0.0f, 0.0f};
    RenderTextureInfo baseColorTexture;
    RenderTextureInfo metallicRoughnessTexture;
    RenderTextureInfo normalTexture;
    RenderTextureInfo occlusionTexture;
    RenderTextureInfo emissiveTexture;
    RenderTextureInfo transmissionTexture;
    RenderTextureInfo thicknessTexture;
    RenderTextureInfo diffuseTransmissionTexture;
    RenderTextureInfo diffuseTransmissionColorTexture;
};

bool buildMeshletsForPrimitive(RenderPrimitive& primitive);
bool buildStreamMeshletsForPrimitive(RenderPrimitive& primitive);

struct RenderCamera {
    SceneEntity object = kNullSceneEntity;
    std::string name;
    int32_t nodeIndex = kInvalidSceneIndex;
    int32_t cameraIndex = kInvalidSceneIndex;
    CameraType type = CameraType::Perspective;
    float3 eye{0.0f, 0.0f, 0.0f};
    float3 center{0.0f, 0.0f, -1.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    double yfov = 0.0;
    double aspectRatio = 0.0;
    double xmag = 0.0;
    double ymag = 0.0;
    double znear = 0.0;
    double zfar = 0.0;
    bool fallback = false;
};

struct RenderLight {
    SceneEntity object = kNullSceneEntity;
    std::string name;
    std::string type;
    int32_t nodeIndex = kInvalidSceneIndex;
    int32_t lightIndex = kInvalidSceneIndex;
    float3 color{1.0f, 1.0f, 1.0f};
    double intensity = 1.0;
    double range = 0.0;
    double innerConeAngle = 0.0;
    double outerConeAngle = 0.7853981633974483;
    float4x4 worldMatrix = float4x4::Identity();
};

class Scene {
public:
    Scene() = default;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&) noexcept = default;
    Scene& operator=(Scene&&) noexcept = default;

    bool load(const std::filesystem::path& filename);
    bool load(
        const std::filesystem::path& filename,
        const SceneLoadProgressCallback& progressCallback);
    bool loadDeferredMeshlets(
        const std::filesystem::path& filename,
        const SceneLoadProgressCallback& progressCallback);
    bool hasDeferredMeshlets() const { return deferredMeshletBuild_; }
    bool buildDeferredMeshlet(size_t primitiveIndex);
    bool finalizeDeferredMeshlets();
    void clear();
    bool setObjectLocalMatrix(SceneEntity object, const float4x4& localMatrix);
    bool setObjectWorldMatrix(SceneEntity object, const float4x4& worldMatrix);
    bool setNodeLocalMatrix(int32_t nodeIndex, const float4x4& localMatrix);
    bool setImageDecodeResult(
        size_t imageIndex,
        std::vector<RenderImage::Mip> mips,
        std::string warning);

    bool valid() const { return lastLoadResult_.success; }
    const LoadResult& lastLoadResult() const { return lastLoadResult_; }
    const std::filesystem::path& filename() const { return filename_; }
    const std::string& sceneName() const { return sceneName_; }
    int32_t sceneIndex() const { return sceneIndex_; }
    const SceneAssetInfo& assetInfo() const { return assetInfo_; }
    const SceneStats& stats() const { return stats_; }
    const Bounds& bounds() const { return bounds_; }
    const SceneGraph& sceneGraph() const { return sceneGraph_; }
    ConstSceneObject objectForNode(int32_t nodeIndex) const;
    const std::vector<int32_t>& rootNodeIndices() const { return rootNodeIndices_; }
    const std::vector<SceneNode>& nodes() const { return nodes_; }
    const std::vector<SceneMesh>& meshes() const { return meshes_; }
    const std::vector<RenderPrimitive>& renderPrimitives() const { return renderPrimitives_; }
    const std::vector<RenderNode>& renderNodes() const { return renderNodes_; }
    const std::vector<RenderImage>& images() const { return images_; }
    const std::vector<RenderTexture>& textures() const { return textures_; }
    const std::vector<RenderMaterial>& materials() const { return materials_; }
    const std::vector<RenderCamera>& cameras() const { return cameras_; }
    const std::vector<RenderLight>& lights() const { return lights_; }
    uint64_t transformRevision() const { return sceneGraph_.transformRevision(); }

private:
    bool loadInternal(
        const std::filesystem::path& filename,
        const SceneLoadProgressCallback& progressCallback,
        bool deferMeshletBuild);
    void clearParsedData();
    void refreshTransforms();
    void syncSceneNodeProjection();

    LoadResult lastLoadResult_;
    std::filesystem::path filename_;
    std::string sceneName_;
    int32_t sceneIndex_ = kInvalidSceneIndex;
    SceneAssetInfo assetInfo_;
    SceneStats stats_;
    Bounds bounds_;
    SceneGraph sceneGraph_;
    std::vector<int32_t> rootNodeIndices_;
    std::vector<SceneNode> nodes_;
    std::vector<SceneMesh> meshes_;
    std::vector<RenderPrimitive> renderPrimitives_;
    std::vector<RenderNode> renderNodes_;
    std::vector<RenderImage> images_;
    std::vector<RenderTexture> textures_;
    std::vector<RenderMaterial> materials_;
    std::vector<RenderCamera> cameras_;
    std::vector<RenderLight> lights_;
    bool deferredMeshletBuild_ = false;
};

bool matrixNearlyEqual(const float4x4& lhs, const float4x4& rhs, float epsilon = 0.000001f);

const char* cameraTypeName(CameraType type);
std::string formatVec3(const float3& value);

} // namespace metallic::scene
