#pragma once
#include "Runtime/Material/MaterialGraph.h"
#include <array>
#include <vector>

struct ImNodesContext;

namespace metallic {

// Authoring state belongs to the editor. Only compiled IR reaches the scene.
class EditorMaterialGraph
{
public:
    bool open = false;
    bool draw(const char* targetName);
    void shutdown();
    bool compile();
    bool load(const std::filesystem::path& path);
    bool save(const std::filesystem::path& path);
    bool exportAssets(const std::filesystem::path& definitionPath);
    void replaceGraph(nlohmann::json graph);
    const nlohmann::json& graph() const { return graph_; }
    const material::CompiledMaterialFrontend& compiled() const { return compiled_; }
    const std::string& status() const { return status_; }
    void setStatus(std::string status) { status_ = std::move(status); }
private:
    nlohmann::json graph_ = material::defaultMaterialGraph();
    material::CompiledMaterialFrontend compiled_;
    ImNodesContext* context_ = nullptr;
    std::vector<nlohmann::json> undo_, redo_;
    std::optional<nlohmann::json> editStart_;
    std::array<char, 1024> path_{"build/MaterialGraph/Draft.materialgraph"};
    std::array<char, 16385> source_{"import MaterialAuthoringSDK;\nMaterial evaluate() {\n    let color = parameter(0);\n    return openPBR(color, 0.0, 0.35);\n}\n"};
    std::string status_;
    bool textMode_ = false;
    bool restorePositions_ = true;
    bool dirty_ = false;
    int selected_ = 2;
    void canvas();
    void properties();
};
} // namespace metallic
