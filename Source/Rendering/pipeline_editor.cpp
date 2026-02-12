#include "pipeline_editor.h"
#include "pass_registry.h"
#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>
#include <algorithm>

PipelineEditor::PipelineEditor() {
    ImNodes::CreateContext();
    ImNodes::StyleColorsDark();

    // Customize style
    ImNodesStyle& style = ImNodes::GetStyle();
    style.NodeCornerRounding = 4.0f;
    style.NodePadding = ImVec2(8.0f, 8.0f);
    style.NodeBorderThickness = 1.0f;
}

PipelineEditor::~PipelineEditor() {
    ImNodes::DestroyContext();
}

int PipelineEditor::getPassIndexFromNodeId(int id) const {
    if (id >= 1000 && id < 2000) return id - 1000;
    return -1;
}

int PipelineEditor::getResourceIndexFromNodeId(int id) const {
    if (id >= 2000 && id < 3000) return id - 2000;
    return -1;
}

std::pair<int, int> PipelineEditor::getPassInputFromPinId(int pinId) const {
    if (pinId >= 10000 && pinId < 20000) {
        int passIndex = (pinId - 10000) / 100;
        int inputIndex = (pinId - 10000) % 100;
        return {passIndex, inputIndex};
    }
    return {-1, -1};
}

std::pair<int, int> PipelineEditor::getPassOutputFromPinId(int pinId) const {
    if (pinId >= 20000 && pinId < 30000) {
        int passIndex = (pinId - 20000) / 100;
        int outputIndex = (pinId - 20000) % 100;
        return {passIndex, outputIndex};
    }
    return {-1, -1};
}

int PipelineEditor::getResourceIndexFromPinId(int pinId) const {
    if (pinId >= 30000 && pinId < 40000) return pinId - 30000;
    return -1;
}

void PipelineEditor::render(PipelineAsset& asset) {
    if (!m_visible) return;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Pipeline Editor", &m_visible, ImGuiWindowFlags_MenuBar)) {
        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Save")) {
                    m_dirty = true;
                }
                if (ImGui::MenuItem("Reset Layout")) {
                    m_nodePositioned.clear();
                    m_firstFrame = true;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Add")) {
                if (ImGui::BeginMenu("Pass")) {
                    for (const auto& type : PassRegistry::instance().registeredTypes()) {
                        if (ImGui::MenuItem(type.c_str())) {
                            PassDecl newPass;
                            newPass.name = type + "_" + std::to_string(asset.passes.size());
                            newPass.type = type;
                            newPass.enabled = true;
                            asset.passes.push_back(newPass);
                            m_dirty = true;
                        }
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::MenuItem("Resource")) {
                    ResourceDecl newRes;
                    newRes.name = "resource_" + std::to_string(asset.resources.size());
                    newRes.type = "texture";
                    newRes.format = "RGBA16Float";
                    newRes.size = "screen";
                    asset.resources.push_back(newRes);
                    m_dirty = true;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Split: node graph on left, properties on right
        float panelWidth = 280.0f;
        ImVec2 contentSize = ImGui::GetContentRegionAvail();

        // Node graph
        ImGui::BeginChild("NodeGraph", ImVec2(contentSize.x - panelWidth - 10, 0), true);
        renderNodeGraph(asset);
        ImGui::EndChild();

        ImGui::SameLine();

        // Property panel
        ImGui::BeginChild("Properties", ImVec2(panelWidth, 0), true);
        renderPropertyPanel(asset);
        ImGui::EndChild();
    }
    ImGui::End();
}

void PipelineEditor::renderNodeGraph(PipelineAsset& asset) {
    ImNodes::BeginNodeEditor();

    // Auto-layout on first frame
    float nodeX = 50.0f;
    float nodeY = 50.0f;
    float nodeSpacingX = 220.0f;
    float nodeSpacingY = 120.0f;

    // Draw resource nodes (blue, on the left)
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(50, 80, 140, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(70, 100, 160, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(90, 120, 180, 255));

    for (size_t i = 0; i < asset.resources.size(); i++) {
        const auto& res = asset.resources[i];
        int nodeId = getResourceNodeId(static_cast<int>(i));

        // Set position on first frame
        if (m_nodePositioned.find(nodeId) == m_nodePositioned.end()) {
            ImNodes::SetNodeScreenSpacePos(nodeId, ImVec2(nodeX, nodeY + i * nodeSpacingY));
            m_nodePositioned[nodeId] = true;
        }

        ImNodes::BeginNode(nodeId);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(res.name.c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::TextDisabled("%s", res.format.c_str());

        // Output pin
        ImNodes::BeginOutputAttribute(getResourcePinId(static_cast<int>(i)));
        ImGui::Text("out");
        ImNodes::EndOutputAttribute();

        ImNodes::EndNode();
    }
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();

    // Draw pass nodes (orange)
    ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(140, 80, 40, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(160, 100, 60, 255));
    ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(180, 120, 80, 255));

    auto sortedOrder = asset.topologicalSort();
    int col = 0;
    int row = 0;
    for (size_t si = 0; si < sortedOrder.size(); si++) {
        size_t i = sortedOrder[si];
        auto& pass = asset.passes[i];
        int nodeId = getPassNodeId(static_cast<int>(i));

        // Set position on first frame
        if (m_nodePositioned.find(nodeId) == m_nodePositioned.end()) {
            float x = 300.0f + col * nodeSpacingX;
            float y = nodeY + row * nodeSpacingY;
            ImNodes::SetNodeScreenSpacePos(nodeId, ImVec2(x, y));
            m_nodePositioned[nodeId] = true;
            row++;
            if (row > 4) {
                row = 0;
                col++;
            }
        }

        // Dim disabled passes
        if (!pass.enabled) {
            ImNodes::PushColorStyle(ImNodesCol_NodeBackground, IM_COL32(60, 60, 60, 200));
        }

        ImNodes::BeginNode(nodeId);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(pass.name.c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::TextDisabled("(%s)", pass.type.c_str());

        // Input pins
        for (size_t j = 0; j < pass.inputs.size(); j++) {
            ImNodes::BeginInputAttribute(getPassInputPinId(static_cast<int>(i), static_cast<int>(j)));
            ImGui::Text("-> %s", pass.inputs[j].c_str());
            ImNodes::EndInputAttribute();
        }

        // Output pins
        for (size_t j = 0; j < pass.outputs.size(); j++) {
            ImNodes::BeginOutputAttribute(getPassOutputPinId(static_cast<int>(i), static_cast<int>(j)));
            ImGui::Text("%s ->", pass.outputs[j].c_str());
            ImNodes::EndOutputAttribute();
        }

        ImNodes::EndNode();

        if (!pass.enabled) {
            ImNodes::PopColorStyle();
        }
    }
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();
    ImNodes::PopColorStyle();

    // Draw links
    int linkId = 0;

    // Build resource name -> index map
    std::unordered_map<std::string, int> resourceIndex;
    for (size_t i = 0; i < asset.resources.size(); i++) {
        resourceIndex[asset.resources[i].name] = static_cast<int>(i);
    }

    // Build output resource -> pass/output index map
    std::unordered_map<std::string, std::pair<int, int>> outputProducer;
    for (size_t pi = 0; pi < asset.passes.size(); pi++) {
        for (size_t oi = 0; oi < asset.passes[pi].outputs.size(); oi++) {
            const auto& outName = asset.passes[pi].outputs[oi];
            if (!outName.empty() && outName[0] != '$') {
                outputProducer[outName] = {static_cast<int>(pi), static_cast<int>(oi)};
            }
        }
    }

    // Draw links from outputs to inputs
    for (size_t pi = 0; pi < asset.passes.size(); pi++) {
        for (size_t ii = 0; ii < asset.passes[pi].inputs.size(); ii++) {
            const std::string& inputName = asset.passes[pi].inputs[ii];

            // Check if input comes from another pass's output
            auto it = outputProducer.find(inputName);
            if (it != outputProducer.end()) {
                ImNodes::Link(linkId++,
                    getPassOutputPinId(it->second.first, it->second.second),
                    getPassInputPinId(static_cast<int>(pi), static_cast<int>(ii)));
            }
            // Check if input comes from a declared resource
            else {
                auto resIt = resourceIndex.find(inputName);
                if (resIt != resourceIndex.end()) {
                    ImNodes::Link(linkId++,
                        getResourcePinId(resIt->second),
                        getPassInputPinId(static_cast<int>(pi), static_cast<int>(ii)));
                }
            }
        }
    }

    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    // Handle selection
    int numSelected = ImNodes::NumSelectedNodes();
    if (numSelected > 0) {
        std::vector<int> selectedNodes(numSelected);
        ImNodes::GetSelectedNodes(selectedNodes.data());
        int nodeId = selectedNodes[0];

        int passIdx = getPassIndexFromNodeId(nodeId);
        if (passIdx >= 0) {
            m_selectedPassIndex = passIdx;
            m_selectedResourceIndex = -1;
        } else {
            int resIdx = getResourceIndexFromNodeId(nodeId);
            if (resIdx >= 0) {
                m_selectedResourceIndex = resIdx;
                m_selectedPassIndex = -1;
            }
        }
    }

    // Handle new links
    handleNewLinks(asset);

    // Handle deleted links/nodes
    handleDeletedLinks(asset);

    m_firstFrame = false;
}

void PipelineEditor::handleNewLinks(PipelineAsset& asset) {
    int startPin, endPin;
    if (ImNodes::IsLinkCreated(&startPin, &endPin)) {
        // Determine what was connected
        auto [srcPassIdx, srcOutputIdx] = getPassOutputFromPinId(startPin);
        auto [dstPassIdx, dstInputIdx] = getPassInputFromPinId(endPin);
        int srcResIdx = getResourceIndexFromPinId(startPin);

        if (dstPassIdx >= 0 && dstInputIdx >= 0) {
            std::string resourceName;

            if (srcPassIdx >= 0 && srcOutputIdx >= 0) {
                // Link from pass output to pass input
                resourceName = asset.passes[srcPassIdx].outputs[srcOutputIdx];
            } else if (srcResIdx >= 0) {
                // Link from resource to pass input
                resourceName = asset.resources[srcResIdx].name;
            }

            if (!resourceName.empty()) {
                // Add or update the input
                if (dstInputIdx < static_cast<int>(asset.passes[dstPassIdx].inputs.size())) {
                    asset.passes[dstPassIdx].inputs[dstInputIdx] = resourceName;
                } else {
                    asset.passes[dstPassIdx].inputs.push_back(resourceName);
                }
                m_dirty = true;
            }
        }
    }
}

void PipelineEditor::handleDeletedLinks(PipelineAsset& asset) {
    int linkId;
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        // Links are derived from pass inputs, removing a link means clearing an input
        // This is complex to track, so for now we just mark dirty
        m_dirty = true;
    }

    // Handle node deletion via keyboard
    if (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
        int numSelected = ImNodes::NumSelectedNodes();
        if (numSelected > 0) {
            std::vector<int> selectedNodes(numSelected);
            ImNodes::GetSelectedNodes(selectedNodes.data());

            // Sort in reverse order to delete from end first
            std::sort(selectedNodes.begin(), selectedNodes.end(), std::greater<int>());

            for (int nodeId : selectedNodes) {
                int passIdx = getPassIndexFromNodeId(nodeId);
                if (passIdx >= 0 && passIdx < static_cast<int>(asset.passes.size())) {
                    asset.passes.erase(asset.passes.begin() + passIdx);
                    m_nodePositioned.erase(nodeId);
                    m_dirty = true;
                }

                int resIdx = getResourceIndexFromNodeId(nodeId);
                if (resIdx >= 0 && resIdx < static_cast<int>(asset.resources.size())) {
                    asset.resources.erase(asset.resources.begin() + resIdx);
                    m_nodePositioned.erase(nodeId);
                    m_dirty = true;
                }
            }

            m_selectedPassIndex = -1;
            m_selectedResourceIndex = -1;
        }
    }
}

void PipelineEditor::renderPropertyPanel(PipelineAsset& asset) {
    ImGui::Text("Properties");
    ImGui::Separator();

    if (m_selectedPassIndex >= 0 && m_selectedPassIndex < static_cast<int>(asset.passes.size())) {
        auto& pass = asset.passes[m_selectedPassIndex];

        ImGui::Text("Pass");
        ImGui::Separator();

        // Name
        char nameBuf[128];
        strncpy(nameBuf, pass.name.c_str(), sizeof(nameBuf) - 1);
        nameBuf[sizeof(nameBuf) - 1] = '\0';
        if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
            pass.name = nameBuf;
            m_dirty = true;
        }

        ImGui::Text("Type: %s", pass.type.c_str());

        if (ImGui::Checkbox("Enabled", &pass.enabled)) {
            m_dirty = true;
        }

        if (ImGui::Checkbox("Side Effect", &pass.sideEffect)) {
            m_dirty = true;
        }

        ImGui::Separator();
        ImGui::Text("Inputs (%zu):", pass.inputs.size());
        for (size_t i = 0; i < pass.inputs.size(); i++) {
            ImGui::BulletText("%s", pass.inputs[i].c_str());
        }
        if (ImGui::Button("+ Input")) {
            pass.inputs.push_back("new_input");
            m_dirty = true;
        }

        ImGui::Separator();
        ImGui::Text("Outputs (%zu):", pass.outputs.size());
        for (size_t i = 0; i < pass.outputs.size(); i++) {
            ImGui::BulletText("%s", pass.outputs[i].c_str());
        }
        if (ImGui::Button("+ Output")) {
            pass.outputs.push_back("new_output");
            m_dirty = true;
        }

    } else if (m_selectedResourceIndex >= 0 && m_selectedResourceIndex < static_cast<int>(asset.resources.size())) {
        auto& res = asset.resources[m_selectedResourceIndex];

        ImGui::Text("Resource");
        ImGui::Separator();

        char nameBuf[128];
        strncpy(nameBuf, res.name.c_str(), sizeof(nameBuf) - 1);
        nameBuf[sizeof(nameBuf) - 1] = '\0';
        if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
            res.name = nameBuf;
            m_dirty = true;
        }

        const char* formats[] = {
            "R8Unorm", "R16Float", "R32Float", "R32Uint",
            "RGBA8Unorm", "BGRA8Unorm", "RGBA16Float", "RGBA32Float",
            "Depth32Float", "Depth16Unorm"
        };
        int currentFormat = 0;
        for (int i = 0; i < IM_ARRAYSIZE(formats); i++) {
            if (res.format == formats[i]) {
                currentFormat = i;
                break;
            }
        }
        if (ImGui::Combo("Format", &currentFormat, formats, IM_ARRAYSIZE(formats))) {
            res.format = formats[currentFormat];
            m_dirty = true;
        }

        char sizeBuf[64];
        strncpy(sizeBuf, res.size.c_str(), sizeof(sizeBuf) - 1);
        sizeBuf[sizeof(sizeBuf) - 1] = '\0';
        if (ImGui::InputText("Size", sizeBuf, sizeof(sizeBuf))) {
            res.size = sizeBuf;
            m_dirty = true;
        }

    } else {
        ImGui::TextDisabled("Select a node");
    }

    ImGui::Separator();

    // Validation
    std::string errorMsg;
    bool valid = asset.validate(errorMsg);
    if (valid) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Valid");
    } else {
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "Error:");
        ImGui::TextWrapped("%s", errorMsg.c_str());
    }

    ImGui::Text("Passes: %zu", asset.passes.size());
    ImGui::Text("Resources: %zu", asset.resources.size());
}
