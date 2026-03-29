#pragma once

#include "pipeline_asset.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "imgui.h"

class PipelineEditor {
public:
    PipelineEditor() = default;
    ~PipelineEditor() = default;

    void render(PipelineAsset& asset);

    bool isDirty() const { return m_dirty; }
    void markClean() { m_dirty = false; }

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    void resetLayout();

    void collectNodePositions(PipelineAsset& asset);
    void autoReorderNodes(PipelineAsset& asset);

    void undo(PipelineAsset& asset);
    void redo(PipelineAsset& asset);
    bool canUndo() const { return !m_undoStack.empty(); }
    bool canRedo() const { return !m_redoStack.empty(); }

    bool m_visible = false;

private:
    enum class GraphViewMode {
        PassFlow,
        ResourceGraph
    };

    enum class PinKind {
        ResourceInput,
        ResourceOutput,
        PassInput,
        PassOutput
    };

    struct PinInfo {
        PinKind kind;
        std::string ownerId;
        std::string slotKey;
    };

    void renderGraphToolbar(PipelineAsset& asset);
    void renderNodeGraph(PipelineAsset& asset);
    void renderPropertyPanel(PipelineAsset& asset);
    void renderCompilationPreview(const PipelineAsset& asset);
    void handleNewLinks(PipelineAsset& asset);
    void handleDeletedLinks(PipelineAsset& asset);
    void pushUndo(const PipelineAsset& asset);

    int ensureUiId(const std::string& key);

    bool setSlotBinding(PipelineAsset& asset,
                        const std::string& passId,
                        const std::string& direction,
                        const std::string& slotKey,
                        const std::string& resourceId);

    bool m_dirty = false;
    GraphViewMode m_graphViewMode = GraphViewMode::PassFlow;
    std::string m_selectedPassId;
    std::string m_selectedResourceId;

    std::unordered_map<int, bool> m_nodePositioned;
    std::unordered_map<std::string, int> m_uiIds;
    int m_nextUiId = 1;

    std::unordered_map<int, std::string> m_nodeIdToPassId;
    std::unordered_map<int, std::string> m_nodeIdToResourceId;
    std::unordered_map<int, PinInfo> m_pinInfos;
    std::unordered_map<int, std::string> m_linkIdToEdgeId;
    std::unordered_set<int> m_renderedNodeIds;

    std::vector<PipelineAsset> m_undoStack;
    std::vector<PipelineAsset> m_redoStack;
    static constexpr int kMaxUndoLevels = 50;

    bool m_hasTextEditSnapshot = false;
    ImVec2 m_contextMenuSpawnPos = {0, 0};
};
