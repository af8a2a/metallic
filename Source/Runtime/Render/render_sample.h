#pragma once

#include "Runtime/Render/RenderGraph/render_graph.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::render {

struct RenderSampleDesc {
    std::string id;
    std::string name;
    std::string category;
    std::string description;
    std::string scenePath;
    std::string graphPath;
    std::vector<std::string> scenePathTargets;
    std::string previewOutput;
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
    virtual std::string graphPath() const = 0;
    virtual std::vector<std::string> scenePathTargets() const = 0;
    virtual std::string previewOutput() const { return {}; }

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

} // namespace metallic::render