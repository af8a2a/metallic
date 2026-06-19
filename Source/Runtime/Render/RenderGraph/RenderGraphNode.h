#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <filesystem>

namespace metallic::render {
struct RenderGraphNode {
    uint32_t id = 0;
    std::string name;
    std::string type;
    RenderGraphProperties properties = RenderGraphProperties::object();
    float uiX = 0.0f;
    float uiY = 0.0f;
};

struct RenderGraphEdge {
    uint32_t id = 0;
    std::string srcPass;
    std::string srcField;
    std::string dstPass;
    std::string dstField;
};

struct RenderGraphOutput {
    std::string passName;
    std::string fieldName;
};

class RenderGraph {
public:
    RenderGraph();

    const std::string& name() const { return name_; }
    void setName(std::string name);

    const std::vector<RenderGraphNode>& nodes() const { return nodes_; }
    const std::vector<RenderGraphEdge>& edges() const { return edges_; }
    const std::vector<RenderGraphOutput>& outputs() const { return outputs_; }

    const RenderGraphNode* findNode(std::string_view name) const;
    RenderGraphNode* findNode(std::string_view name);
    const RenderGraphNode* findNode(uint32_t id) const;
    RenderGraphNode* findNode(uint32_t id);
    const RenderGraphEdge* findEdge(uint32_t id) const;

    RenderGraphNode* addNode(
        std::string type,
        std::string name,
        RenderGraphProperties properties = RenderGraphProperties::object(),
        float uiX = 0.0f,
        float uiY = 0.0f);
    bool removeNode(uint32_t id);
    bool renameNode(uint32_t id, std::string newName);
    bool setNodeProperties(uint32_t id, RenderGraphProperties properties);
    bool setNodePosition(uint32_t id, float uiX, float uiY);

    RenderGraphEdge* addEdge(std::string src, std::string dst);
    bool removeEdge(uint32_t id);
    bool markOutput(std::string output);
    bool unmarkOutput(std::string output);
    void clearOutputs();

    bool validate(std::string& log) const;

    bool dirty() const { return dirty_; }
    void clearDirty() { dirty_ = false; }
    void markDirty() { dirty_ = true; }
    void clear();

    std::string firstOutputName() const;

    static RenderGraph createDefaultTriangleGraph();
    static RenderGraph createDefaultBunnyGraph();

private:
    std::string name_ = "RenderGraph";
    std::vector<RenderGraphNode> nodes_;
    std::vector<RenderGraphEdge> edges_;
    std::vector<RenderGraphOutput> outputs_;
    uint32_t nextNodeId_ = 1;
    uint32_t nextEdgeId_ = 1;
    bool dirty_ = true;

    friend bool deserializeRenderGraphFromString(
        const std::string& text,
        RenderGraph& outGraph,
        std::string& outMessage);
};

bool splitRenderGraphFieldName(
    std::string_view fullName,
    std::string& outPassName,
    std::string& outFieldName);
std::string makeRenderGraphFieldName(std::string_view passName, std::string_view fieldName);

std::string serializeRenderGraphToString(const RenderGraph& graph);
bool deserializeRenderGraphFromString(
    const std::string& text,
    RenderGraph& outGraph,
    std::string& outMessage);
bool saveRenderGraphToFile(
    const RenderGraph& graph,
    const std::filesystem::path& path,
    std::string& outMessage);
bool loadRenderGraphFromFile(
    const std::filesystem::path& path,
    RenderGraph& outGraph,
    std::string& outMessage);

} // namespace metallic::render
