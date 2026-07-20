#pragma once

#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::render {

struct RenderSampleEnvironmentDesc {
    bool enabled = true;
    std::string path;
    float intensity = 1.0f;
    float rotationDegrees = 0.0f;
    bool visible = true;
};

struct RenderSampleDesc {
    std::string id;
    std::string name;
    std::string category;
    std::string description;
    std::string scenePath;
    bool loadSceneInEditor = true;
    std::string graphPath;
    std::vector<std::string> scenePathTargets;
    RenderSampleEnvironmentDesc environment;
    std::vector<std::string> environmentTargets;
    std::string previewOutput;
    bool requiresStreamline = false;
};

struct RenderSampleLoadResult {
    RenderSampleDesc desc;
    RenderGraph graph;
    std::filesystem::path graphFilePath;
};

class RenderSample {
public:
    virtual ~RenderSample() = default;

    virtual std::string_view id() const = 0;
    virtual std::string_view name() const = 0;
    virtual std::string_view category() const = 0;
    virtual std::string_view description() const { return {}; }
    virtual std::string scenePath() const = 0;
    virtual bool loadSceneInEditor() const { return true; }
    virtual std::string graphPath() const = 0;
    virtual std::vector<std::string> scenePathTargets() const = 0;
    virtual RenderSampleEnvironmentDesc environment() const { return {}; }
    virtual std::vector<std::string> environmentTargets() const { return {}; }
    virtual std::string previewOutput() const { return {}; }
    virtual bool requiresStreamline() const { return false; }

    RenderSampleDesc desc() const;
};

bool loadRenderSample(
    const RenderSample& sample,
    RenderSampleLoadResult& outResult,
    std::string& outMessage);
std::vector<RenderSampleDesc> listBuiltInRenderSamples();
bool loadBuiltInRenderSample(
    std::string_view id,
    RenderSampleLoadResult& outResult,
    std::string& outMessage);
bool queryBuiltInRenderSampleStreamlineRequirement(
    std::string_view id,
    bool& outRequiresStreamline);
bool setRenderSampleScenePath(
    RenderSampleLoadResult& sample,
    std::string scenePath,
    std::string& outMessage);

} // namespace metallic::render
