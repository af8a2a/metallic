#pragma once


#include <cstdint>
#include <array>
#include <filesystem>
#include <string>
#include <vector>
#include "ml.h"

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
};

struct SceneStats {
    uint64_t meshCount = 0;
    uint64_t materialCount = 0;
    uint64_t primitiveCount = 0;
    uint64_t renderNodeCount = 0;
    uint64_t triangleCount = 0;
};

struct SceneNode {
    std::string name;
    int32_t parent = kInvalidSceneIndex;
    std::vector<int32_t> children;
    int32_t meshIndex = kInvalidSceneIndex;
    int32_t cameraIndex = kInvalidSceneIndex;
    int32_t lightIndex = kInvalidSceneIndex;
    float4x4 localMatrix = float4x4::Identity();
    float4x4 worldMatrix = float4x4::Identity();
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
    std::vector<uint32_t> indices;
};

struct RenderImage {
    std::string name;
    std::string uri;
    std::string mimeType;
    int32_t bufferView = kInvalidSceneIndex;
    std::vector<uint8_t> encodedData;
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
    int32_t nodeIndex = kInvalidSceneIndex;
    int32_t renderPrimitiveIndex = kInvalidSceneIndex;
    int32_t materialIndex = kInvalidSceneIndex;
    float4x4 worldMatrix = float4x4::Identity();
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

struct RenderCamera {
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
    bool load(const std::filesystem::path& filename);
    void clear();

    bool valid() const { return lastLoadResult_.success; }
    const LoadResult& lastLoadResult() const { return lastLoadResult_; }
    const std::filesystem::path& filename() const { return filename_; }
    const std::string& sceneName() const { return sceneName_; }
    int32_t sceneIndex() const { return sceneIndex_; }
    const SceneStats& stats() const { return stats_; }
    const Bounds& bounds() const { return bounds_; }
    const std::vector<int32_t>& rootNodeIndices() const { return rootNodeIndices_; }
    const std::vector<SceneNode>& nodes() const { return nodes_; }
    const std::vector<RenderPrimitive>& renderPrimitives() const { return renderPrimitives_; }
    const std::vector<RenderNode>& renderNodes() const { return renderNodes_; }
    const std::vector<RenderImage>& images() const { return images_; }
    const std::vector<RenderTexture>& textures() const { return textures_; }
    const std::vector<RenderMaterial>& materials() const { return materials_; }
    const std::vector<RenderCamera>& cameras() const { return cameras_; }
    const std::vector<RenderLight>& lights() const { return lights_; }

private:
    void clearParsedData();

    LoadResult lastLoadResult_;
    std::filesystem::path filename_;
    std::string sceneName_;
    int32_t sceneIndex_ = kInvalidSceneIndex;
    SceneStats stats_;
    Bounds bounds_;
    std::vector<int32_t> rootNodeIndices_;
    std::vector<SceneNode> nodes_;
    std::vector<RenderPrimitive> renderPrimitives_;
    std::vector<RenderNode> renderNodes_;
    std::vector<RenderImage> images_;
    std::vector<RenderTexture> textures_;
    std::vector<RenderMaterial> materials_;
    std::vector<RenderCamera> cameras_;
    std::vector<RenderLight> lights_;
};

const char* cameraTypeName(CameraType type);
std::string formatVec3(const float3& value);

} // namespace metallic::scene
