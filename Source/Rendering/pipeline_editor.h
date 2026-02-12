#pragma once

#include "pipeline_asset.h"
#include <string>
#include <unordered_map>

// Node graph pipeline editor using imnodes

class PipelineEditor {
public:
    PipelineEditor();
    ~PipelineEditor();

    // Render the editor UI
    void render(PipelineAsset& asset);

    // Check if pipeline was modified
    bool isDirty() const { return m_dirty; }
    void markClean() { m_dirty = false; }

    // Show/hide the editor
    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

private:
    void renderNodeGraph(PipelineAsset& asset);
    void renderPropertyPanel(PipelineAsset& asset);
    void handleNewLinks(PipelineAsset& asset);
    void handleDeletedLinks(PipelineAsset& asset);

    // Node/pin ID encoding
    int getPassNodeId(int passIndex) const { return 1000 + passIndex; }
    int getResourceNodeId(int resIndex) const { return 2000 + resIndex; }
    int getPassInputPinId(int passIndex, int inputIndex) const { return 10000 + passIndex * 100 + inputIndex; }
    int getPassOutputPinId(int passIndex, int outputIndex) const { return 20000 + passIndex * 100 + outputIndex; }
    int getResourcePinId(int resIndex) const { return 30000 + resIndex; }

    // Reverse lookups
    int getPassIndexFromNodeId(int id) const;
    int getResourceIndexFromNodeId(int id) const;
    std::pair<int, int> getPassInputFromPinId(int pinId) const;
    std::pair<int, int> getPassOutputFromPinId(int pinId) const;
    int getResourceIndexFromPinId(int pinId) const;

    bool m_dirty = false;
    bool m_visible = false;
    int m_selectedPassIndex = -1;
    int m_selectedResourceIndex = -1;
    bool m_firstFrame = true;

    // Track node positions for auto-layout
    std::unordered_map<int, bool> m_nodePositioned;
};
