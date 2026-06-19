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
    std::filesystem::path samplePath;
    std::filesystem::path graphFilePath;
};

bool loadRenderSampleFromFile(
    const std::filesystem::path& path,
    RenderSampleLoadResult& outResult,
    std::string& outMessage);
std::vector<RenderSampleDesc> listBuiltInRenderSamples();
bool loadBuiltInRenderSample(
    std::string_view id,
    RenderSampleLoadResult& outResult,
    std::string& outMessage);

} // namespace metallic::render
