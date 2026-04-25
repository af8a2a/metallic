#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct ScenePrimitive {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = 0;
    int meshID = 0;
};

struct SceneMeshInfo {
    uint32_t firstPrimitive = 0;
    uint32_t primitiveCount = 0;
};

struct SceneMaterial {
    float baseColorFactor[4] = {1, 1, 1, 1};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float alphaCutoff = 0.5f;
    uint32_t alphaMode = 0;
    int baseColorTexture = -1;
    int normalTexture = -1;
    int metallicRoughnessTexture = -1;
};

struct SceneImage {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int channels = 4;
    std::string uri;
};

struct SceneNodeData {
    std::string name;
    int parent = -1;
    std::vector<int> children;
    float translation[3] = {0, 0, 0};
    float rotation[4] = {0, 0, 0, 1};
    float scale[3] = {1, 1, 1};
    bool hasMatrix = false;
    float localMatrix[16] = {};
    int mesh = -1;
    int light = -1;
    int camera = -1;
};

struct SceneCamera {
    enum Type { Perspective, Orthographic };
    Type type = Perspective;
    float yfov = 0.0f;
    float znear = 0.01f;
    float zfar = 1000.0f;
    float aspectRatio = 0.0f;
};

struct SceneLight {
    enum Type { Directional, Point, Spot };
    Type type = Directional;
    float color[3] = {1, 1, 1};
    float intensity = 1.0f;
};

class Scene {
public:
    bool load(const std::string& gltfPath);
    void clear();
    bool isLoaded() const { return m_loaded; }
    const std::string& filePath() const { return m_filePath; }

    std::vector<float> positions;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<uint32_t> indices;

    std::vector<ScenePrimitive> primitives;
    std::vector<SceneMeshInfo> meshInfos;
    std::vector<SceneMaterial> materials;
    std::vector<SceneImage> images;
    std::vector<SceneNodeData> nodes;
    std::vector<int> rootNodes;
    std::vector<SceneCamera> cameras;
    std::vector<SceneLight> lights;

    float bboxMin[3] = {};
    float bboxMax[3] = {};

    bool hasBakedRootScale = false;
    float bakedRootScale = 1.0f;

private:
    bool m_loaded = false;
    std::string m_filePath;
};
