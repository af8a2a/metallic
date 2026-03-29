#include "pipeline_editor.h"

#include "pass_registry.h"

#include <imgui.h>
#include <imnodes.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <numeric>

namespace {

struct PassBindingInfo {
    const PassDecl* pass = nullptr;
    const EdgeDecl* edge = nullptr;
};

struct ResourceFlowInfo {
    const ResourceDecl* resource = nullptr;
    PassBindingInfo producer;
    std::vector<PassBindingInfo> writers;
    std::vector<PassBindingInfo> readers;
};

bool hasStoredPosition(const std::array<float, 2>& pos) {
    return pos[0] != 0.0f || pos[1] != 0.0f;
}

ImU32 resourceTitleColor(const std::string& kind) {
    if (kind == "imported") {
        return IM_COL32(50, 120, 70, 255);
    }
    if (kind == "backbuffer") {
        return IM_COL32(140, 70, 50, 255);
    }
    return IM_COL32(50, 80, 140, 255);
}

ImU32 resourceTitleHoveredColor(const std::string& kind) {
    if (kind == "imported") {
        return IM_COL32(70, 140, 90, 255);
    }
    if (kind == "backbuffer") {
        return IM_COL32(160, 90, 70, 255);
    }
    return IM_COL32(70, 100, 160, 255);
}

ImU32 resourceTitleSelectedColor(const std::string& kind) {
    if (kind == "imported") {
        return IM_COL32(90, 160, 110, 255);
    }
    if (kind == "backbuffer") {
        return IM_COL32(180, 110, 90, 255);
    }
    return IM_COL32(90, 120, 180, 255);
}

std::string makeDefaultPassName(const PipelineAsset& asset, const PassTypeInfo& info) {
    int count = 0;
    for (const auto& pass : asset.passes) {
        if (pass.type == info.typeName) {
            ++count;
        }
    }
    return info.displayName + " " + std::to_string(count + 1);
}

std::string makeDefaultResourceName(const PipelineAsset& asset, const std::string& kind) {
    int count = 0;
    for (const auto& resource : asset.resources) {
        if (resource.kind == kind) {
            ++count;
        }
    }
    if (kind == "backbuffer") {
        return "Backbuffer";
    }
    if (kind == "imported") {
        return "Imported " + std::to_string(count + 1);
    }
    return "Resource " + std::to_string(count + 1);
}

std::string resourceLabel(const ResourceDecl& resource) {
    return resource.name + " (" + resource.kind + ")";
}

template <size_t N>
void copyTextToBuffer(char (&buffer)[N], const std::string& value) {
    std::snprintf(buffer, N, "%s", value.c_str());
}

std::string truncateLabel(const std::string& value, size_t maxChars = 26) {
    if (value.size() <= maxChars) {
        return value;
    }
    if (maxChars <= 3) {
        return value.substr(0, maxChars);
    }
    return value.substr(0, maxChars - 3) + "...";
}

std::string findResourceName(const PipelineAsset& asset, const std::string& resourceId) {
    const ResourceDecl* resource = asset.findResourceById(resourceId);
    return resource ? resource->name : std::string{"<missing>"};
}

const ResourceDecl* findBoundResource(const PipelineAsset& asset,
                                      const std::string& passId,
                                      const std::string& direction,
                                      const std::string& slotKey) {
    const EdgeDecl* edge = asset.findEdge(passId, direction, slotKey);
    return edge ? asset.findResourceById(edge->resourceId) : nullptr;
}

bool slotAllowsKind(const PassSlotInfo& slot, const std::string& kind) {
    if (slot.allowedResourceKinds.empty()) {
        return true;
    }
    return std::find(slot.allowedResourceKinds.begin(),
                     slot.allowedResourceKinds.end(),
                     kind) != slot.allowedResourceKinds.end();
}

const PassSlotInfo* findSlotInfo(const std::vector<PassSlotInfo>& slots, const std::string& key) {
    for (const auto& slot : slots) {
        if (slot.key == key) {
            return &slot;
        }
    }
    return nullptr;
}

std::string slotBindingName(const PipelineAsset& asset,
                            const std::string& passId,
                            const std::string& direction,
                            const PassSlotInfo& slot) {
    const EdgeDecl* edge = asset.findEdge(passId, direction, slot.key);
    if (!edge) {
        return "<unbound>";
    }
    return findResourceName(asset, edge->resourceId);
}

std::vector<const ResourceDecl*> collectCandidates(const PipelineAsset& asset, const PassSlotInfo& slot) {
    std::vector<const ResourceDecl*> candidates;
    for (const auto& resource : asset.resources) {
        if (slotAllowsKind(slot, resource.kind)) {
            candidates.push_back(&resource);
        }
    }
    return candidates;
}

std::unordered_map<std::string, ResourceFlowInfo> buildResourceFlowMap(const PipelineAsset& asset) {
    std::unordered_map<std::string, ResourceFlowInfo> flow;
    for (const auto& resource : asset.resources) {
        flow.emplace(resource.id, ResourceFlowInfo{&resource});
    }

    for (const auto& edge : asset.edges) {
        auto flowIt = flow.find(edge.resourceId);
        if (flowIt == flow.end()) {
            continue;
        }

        const PassDecl* pass = asset.findPassById(edge.passId);
        if (!pass) {
            continue;
        }

        PassBindingInfo binding{pass, &edge};
        if (edge.direction == "output") {
            flowIt->second.writers.push_back(binding);
            if (flowIt->second.resource &&
                flowIt->second.resource->kind == "transient" &&
                flowIt->second.producer.pass == nullptr) {
                flowIt->second.producer = binding;
            }
        } else if (edge.direction == "input") {
            flowIt->second.readers.push_back(binding);
        }
    }

    return flow;
}

std::vector<int> computePassLayers(const PipelineAsset& asset) {
    const auto sortedOrder = asset.topologicalSort();
    std::vector<int> passLayer(asset.passes.size(), 0);
    std::unordered_map<std::string, int> resourceProducerLayer;

    for (const size_t passIndex : sortedOrder) {
        const PassDecl& pass = asset.passes[passIndex];
        int maxLayer = 0;

        const PassTypeInfo* info = PassRegistry::instance().getTypeInfo(pass.type);
        if (!info) {
            continue;
        }

        for (const auto& slot : info->inputSlots) {
            const EdgeDecl* edge = asset.findEdge(pass.id, "input", slot.key);
            if (!edge) {
                continue;
            }
            auto it = resourceProducerLayer.find(edge->resourceId);
            if (it != resourceProducerLayer.end()) {
                maxLayer = std::max(maxLayer, it->second + 1);
            }
        }

        passLayer[passIndex] = maxLayer;

        for (const auto& slot : info->outputSlots) {
            const EdgeDecl* edge = asset.findEdge(pass.id, "output", slot.key);
            if (!edge) {
                continue;
            }
            const ResourceDecl* resource = asset.findResourceById(edge->resourceId);
            if (resource && resource->kind == "transient") {
                resourceProducerLayer[resource->id] = maxLayer;
            }
        }
    }

    return passLayer;
}

ImU32 passCategoryTitleColor(const PassTypeInfo& info) {
    if (info.category == "Geometry") {
        return IM_COL32(120, 78, 32, 255);
    }
    if (info.category == "Lighting") {
        return IM_COL32(118, 92, 28, 255);
    }
    if (info.category == "Post-Process") {
        return IM_COL32(88, 78, 126, 255);
    }
    if (info.category == "Environment") {
        return IM_COL32(44, 108, 120, 255);
    }
    if (info.category == "UI") {
        return IM_COL32(100, 56, 120, 255);
    }
    return IM_COL32(110, 90, 54, 255);
}

ImU32 brightenColor(ImU32 color, int delta) {
    ImVec4 value = ImGui::ColorConvertU32ToFloat4(color);
    auto brighten = [delta](float channel) {
        return std::clamp(channel + static_cast<float>(delta) / 255.0f, 0.0f, 1.0f);
    };
    return ImGui::ColorConvertFloat4ToU32(ImVec4{
        brighten(value.x),
        brighten(value.y),
        brighten(value.z),
        value.w
    });
}

ImU32 linkColorForResource(const ResourceDecl& resource) {
    if (resource.kind == "imported") {
        return IM_COL32(84, 188, 132, 255);
    }
    if (resource.kind == "backbuffer") {
        return IM_COL32(220, 154, 90, 255);
    }
    if (resource.type == "token") {
        return IM_COL32(170, 130, 230, 255);
    }
    if (resource.type == "buffer") {
        return IM_COL32(104, 168, 224, 255);
    }
    return IM_COL32(82, 146, 232, 255);
}

bool isTerminalResource(const ResourceDecl& resource) {
    return resource.kind == "imported" || resource.kind == "backbuffer";
}

} // namespace

void PipelineEditor::pushUndo(const PipelineAsset& asset) {
    m_undoStack.push_back(asset);
    if (static_cast<int>(m_undoStack.size()) > kMaxUndoLevels) {
        m_undoStack.erase(m_undoStack.begin());
    }
    m_redoStack.clear();
}

void PipelineEditor::undo(PipelineAsset& asset) {
    if (m_undoStack.empty()) {
        return;
    }
    m_redoStack.push_back(asset);
    asset = m_undoStack.back();
    m_undoStack.pop_back();
    resetLayout();
    m_dirty = true;
}

void PipelineEditor::redo(PipelineAsset& asset) {
    if (m_redoStack.empty()) {
        return;
    }
    m_undoStack.push_back(asset);
    asset = m_redoStack.back();
    m_redoStack.pop_back();
    resetLayout();
    m_dirty = true;
}

void PipelineEditor::resetLayout() {
    m_nodePositioned.clear();
    m_uiIds.clear();
    m_nextUiId = 1;
    m_selectedPassId.clear();
    m_selectedResourceId.clear();
    m_nodeIdToPassId.clear();
    m_nodeIdToResourceId.clear();
    m_pinInfos.clear();
    m_linkIdToEdgeId.clear();
    m_renderedNodeIds.clear();
}

int PipelineEditor::ensureUiId(const std::string& key) {
    auto it = m_uiIds.find(key);
    if (it != m_uiIds.end()) {
        return it->second;
    }
    const int id = m_nextUiId++;
    m_uiIds.emplace(key, id);
    return id;
}

bool PipelineEditor::setSlotBinding(PipelineAsset& asset,
                                    const std::string& passId,
                                    const std::string& direction,
                                    const std::string& slotKey,
                                    const std::string& resourceId) {
    EdgeDecl* edge = asset.findEdge(passId, direction, slotKey);
    if (resourceId.empty()) {
        if (!edge) {
            return false;
        }
        asset.edges.erase(
            std::remove_if(asset.edges.begin(),
                           asset.edges.end(),
                           [&](const EdgeDecl& candidate) { return candidate.id == edge->id; }),
            asset.edges.end());
        return true;
    }

    if (edge) {
        if (edge->resourceId == resourceId) {
            return false;
        }
        edge->resourceId = resourceId;
        return true;
    }

    asset.edges.push_back(EdgeDecl{
        generatePipelineAssetGuid(),
        passId,
        slotKey,
        direction,
        resourceId
    });
    return true;
}

void PipelineEditor::render(PipelineAsset& asset) {
    if (!m_visible) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(1200, 720), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Pipeline Editor", &m_visible, ImGuiWindowFlags_MenuBar)) {
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Save")) {
                    collectNodePositions(asset);
                    m_dirty = true;
                }
                if (ImGui::MenuItem("Reset Layout")) {
                    m_nodePositioned.clear();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Add")) {
                if (ImGui::BeginMenu("Pass")) {
                    auto byCategory = PassRegistry::instance().getTypesByCategory();
                    for (const auto& [category, infos] : byCategory) {
                        if (ImGui::BeginMenu(category.c_str())) {
                            for (const auto* info : infos) {
                                if (ImGui::MenuItem(info->displayName.c_str())) {
                                    pushUndo(asset);
                                    PassDecl pass;
                                    pass.id = generatePipelineAssetGuid();
                                    pass.name = makeDefaultPassName(asset, *info);
                                    pass.type = info->typeName;
                                    pass.enabled = true;
                                    asset.passes.push_back(pass);
                                    m_selectedPassId = pass.id;
                                    m_selectedResourceId.clear();
                                    m_dirty = true;
                                }
                            }
                            ImGui::EndMenu();
                        }
                    }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Resource")) {
                    const char* kinds[] = {"transient", "imported", "backbuffer"};
                    for (const char* kind : kinds) {
                        if (ImGui::MenuItem(kind)) {
                            pushUndo(asset);
                            ResourceDecl resource;
                            resource.id = generatePipelineAssetGuid();
                            resource.name = makeDefaultResourceName(asset, kind);
                            resource.kind = kind;
                            resource.type = "texture";
                            resource.format = kind == std::string("backbuffer") ? "BGRA8Unorm" : "RGBA16Float";
                            resource.size = "screen";
                            if (resource.kind == "imported") {
                                resource.importKey = resource.name;
                            }
                            asset.resources.push_back(resource);
                            if (m_graphViewMode == GraphViewMode::PassFlow && resource.kind == "transient") {
                                m_graphViewMode = GraphViewMode::ResourceGraph;
                                m_nodePositioned.clear();
                            }
                            m_selectedResourceId = resource.id;
                            m_selectedPassId.clear();
                            m_dirty = true;
                        }
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Auto Reorder")) {
                    pushUndo(asset);
                    autoReorderNodes(asset);
                    m_dirty = true;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        const float panelWidth = 360.0f;
        const ImVec2 contentSize = ImGui::GetContentRegionAvail();

        ImGui::BeginChild("NodeGraph", ImVec2(contentSize.x - panelWidth - 10.0f, 0), true);
        renderGraphToolbar(asset);
        ImGui::Separator();
        renderNodeGraph(asset);
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("Properties", ImVec2(panelWidth, 0), true);
        renderPropertyPanel(asset);
        ImGui::EndChild();
    }
    ImGui::End();
}

void PipelineEditor::renderGraphToolbar(PipelineAsset& asset) {
    if (ImGui::RadioButton("Pass Flow", m_graphViewMode == GraphViewMode::PassFlow)) {
        m_graphViewMode = GraphViewMode::PassFlow;
        m_nodePositioned.clear();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Resource Graph", m_graphViewMode == GraphViewMode::ResourceGraph)) {
        m_graphViewMode = GraphViewMode::ResourceGraph;
        m_nodePositioned.clear();
    }
    ImGui::SameLine();
    if (ImGui::Button("Auto Layout")) {
        pushUndo(asset);
        autoReorderNodes(asset);
        m_dirty = true;
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextDisabled("Falcor-style tip: inspect flow in Pass Flow, edit exact resources in Resource Graph.");
}

void PipelineEditor::renderNodeGraph(PipelineAsset& asset) {
    m_nodeIdToPassId.clear();
    m_nodeIdToResourceId.clear();
    m_pinInfos.clear();
    m_linkIdToEdgeId.clear();
    m_renderedNodeIds.clear();

    const auto flowMap = buildResourceFlowMap(asset);
    const auto passLayers = computePassLayers(asset);
    std::unordered_map<std::string, int> passLayerById;
    int maxPassLayer = 0;
    for (size_t i = 0; i < asset.passes.size() && i < passLayers.size(); ++i) {
        passLayerById[asset.passes[i].id] = passLayers[i];
        maxPassLayer = std::max(maxPassLayer, passLayers[i]);
    }

    auto sortedOrder = asset.topologicalSort();
    if (sortedOrder.empty() && !asset.passes.empty()) {
        sortedOrder.resize(asset.passes.size());
        for (size_t i = 0; i < asset.passes.size(); ++i) {
            sortedOrder[i] = i;
        }
    }

    ImNodes::BeginNodeEditor();

    const float resourceX = 40.0f;
    const float resourceY = 50.0f;
    const float passX = 320.0f;
    const float nodeSpacingX = 320.0f;
    const float nodeSpacingY = 170.0f;
    const float terminalRightX = passX + static_cast<float>(maxPassLayer + 1) * nodeSpacingX;

    auto placeNode = [&](int nodeId,
                         std::array<float, 2>& editorPos,
                         const ImVec2& fallbackPos,
                         bool useStoredPosition) {
        if (m_nodePositioned.count(nodeId)) {
            return;
        }

        if (useStoredPosition && hasStoredPosition(editorPos)) {
            ImNodes::SetNodeGridSpacePos(nodeId, ImVec2(editorPos[0], editorPos[1]));
        } else {
            ImNodes::SetNodeScreenSpacePos(nodeId, fallbackPos);
        }
        m_nodePositioned[nodeId] = true;
    };

    auto drawResourceNode = [&](ResourceDecl& resource, const ImVec2& fallbackPos) {
        const int nodeId = ensureUiId("node:resource:" + resource.id);
        m_nodeIdToResourceId[nodeId] = resource.id;
        m_renderedNodeIds.insert(nodeId);
        placeNode(nodeId,
                  resource.editorPos,
                  fallbackPos,
                  m_graphViewMode == GraphViewMode::ResourceGraph);

        ImNodes::PushColorStyle(ImNodesCol_TitleBar, resourceTitleColor(resource.kind));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, resourceTitleHoveredColor(resource.kind));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, resourceTitleSelectedColor(resource.kind));

        ImNodes::BeginNode(nodeId);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(resource.name.c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::TextDisabled("%s / %s", resource.kind.c_str(), resource.type.c_str());
        if (resource.kind == "imported" && !resource.importKey.empty()) {
            ImGui::TextDisabled("key: %s", resource.importKey.c_str());
        }

        auto flowIt = flowMap.find(resource.id);
        if (flowIt != flowMap.end()) {
            const ResourceFlowInfo& flow = flowIt->second;
            if (!flow.writers.empty()) {
                ImGui::TextDisabled("writers: %zu", flow.writers.size());
            }
            if (!flow.readers.empty()) {
                ImGui::TextDisabled("readers: %zu", flow.readers.size());
            }
        }

        const int inputPinId = ensureUiId("pin:resource:in:" + resource.id);
        m_pinInfos[inputPinId] = PinInfo{PinKind::ResourceInput, resource.id, {}};
        ImNodes::BeginInputAttribute(inputPinId);
        ImGui::TextUnformatted("write");
        ImNodes::EndInputAttribute();

        const int outputPinId = ensureUiId("pin:resource:out:" + resource.id);
        m_pinInfos[outputPinId] = PinInfo{PinKind::ResourceOutput, resource.id, {}};
        ImNodes::BeginOutputAttribute(outputPinId);
        ImGui::TextUnformatted("read");
        ImNodes::EndOutputAttribute();

        ImNodes::EndNode();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    };

    auto drawPassNode = [&](PassDecl& pass, const PassTypeInfo& passTypeInfo, const ImVec2& fallbackPos) {
        const int nodeId = ensureUiId("node:pass:" + pass.id);
        m_nodeIdToPassId[nodeId] = pass.id;
        m_renderedNodeIds.insert(nodeId);
        placeNode(nodeId,
                  pass.editorPos,
                  fallbackPos,
                  m_graphViewMode == GraphViewMode::ResourceGraph);

        const ImU32 titleColor = passCategoryTitleColor(passTypeInfo);
        ImNodes::PushColorStyle(ImNodesCol_TitleBar, titleColor);
        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, brightenColor(titleColor, 18));
        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, brightenColor(titleColor, 36));
        if (!pass.enabled) {
            ImNodes::PushColorStyle(ImNodesCol_NodeBackground, IM_COL32(55, 55, 55, 200));
        }

        ImNodes::BeginNode(nodeId);
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(pass.name.c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::TextDisabled("(%s | %s)", pass.type.c_str(), passTypeInfo.category.c_str());

        for (const auto& slot : passTypeInfo.inputSlots) {
            const int pinId = ensureUiId("pin:pass:in:" + pass.id + ":" + slot.key);
            m_pinInfos[pinId] = PinInfo{PinKind::PassInput, pass.id, slot.key};
            ImNodes::BeginInputAttribute(pinId);
            const ResourceDecl* resource = findBoundResource(asset, pass.id, "input", slot.key);
            ImGui::Text("-> %s", slot.displayName.c_str());
            ImGui::TextDisabled("%s%s",
                                resource ? truncateLabel(resource->name).c_str() : "<unbound>",
                                slot.optional ? " (optional)" : "");
            ImNodes::EndInputAttribute();
        }

        for (const auto& slot : passTypeInfo.outputSlots) {
            const int pinId = ensureUiId("pin:pass:out:" + pass.id + ":" + slot.key);
            m_pinInfos[pinId] = PinInfo{PinKind::PassOutput, pass.id, slot.key};
            ImNodes::BeginOutputAttribute(pinId);
            const ResourceDecl* resource = findBoundResource(asset, pass.id, "output", slot.key);
            ImGui::Text("%s ->", slot.displayName.c_str());
            ImGui::TextDisabled("%s%s",
                                resource ? truncateLabel(resource->name).c_str() : "<unbound>",
                                slot.optional ? " (optional)" : "");
            ImNodes::EndOutputAttribute();
        }

        ImNodes::EndNode();
        if (!pass.enabled) {
            ImNodes::PopColorStyle();
        }
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    };

    if (m_graphViewMode == GraphViewMode::ResourceGraph) {
        for (size_t i = 0; i < asset.resources.size(); ++i) {
            drawResourceNode(asset.resources[i],
                             ImVec2(resourceX, resourceY + static_cast<float>(i) * nodeSpacingY));
        }
    } else {
        size_t importedIndex = 0;
        size_t backbufferIndex = 0;
        for (auto& resource : asset.resources) {
            if (!isTerminalResource(resource)) {
                continue;
            }

            float fallbackY = resourceY;
            const auto flowIt = flowMap.find(resource.id);
            if (flowIt != flowMap.end()) {
                float totalY = 0.0f;
                int count = 0;
                for (const auto& writer : flowIt->second.writers) {
                    auto layerIt = passLayerById.find(writer.pass->id);
                    const int layer = layerIt != passLayerById.end() ? layerIt->second : 0;
                    totalY += resourceY + static_cast<float>(layer) * 0.25f * nodeSpacingY;
                    ++count;
                }
                for (const auto& reader : flowIt->second.readers) {
                    auto layerIt = passLayerById.find(reader.pass->id);
                    const int layer = layerIt != passLayerById.end() ? layerIt->second : 0;
                    totalY += resourceY + static_cast<float>(layer) * 0.25f * nodeSpacingY;
                    ++count;
                }
                if (count > 0) {
                    fallbackY = totalY / static_cast<float>(count);
                }
            }

            const bool backbuffer = resource.kind == "backbuffer";
            const float fallbackX = backbuffer ? terminalRightX : resourceX;
            const size_t visibleIndex = backbuffer ? backbufferIndex++ : importedIndex++;
            drawResourceNode(resource,
                             ImVec2(fallbackX,
                                    std::max(resourceY + static_cast<float>(visibleIndex) * 120.0f, fallbackY)));
        }
    }

    std::unordered_map<int, size_t> layerRowCount;
    size_t defaultRow = 0;
    size_t defaultCol = 0;
    for (const size_t passIndex : sortedOrder) {
        PassDecl& pass = asset.passes[passIndex];
        const PassTypeInfo* typeInfo = PassRegistry::instance().getTypeInfo(pass.type);
        if (!typeInfo) {
            continue;
        }

        const int layer = passIndex < passLayers.size() ? passLayers[passIndex] : 0;
        const size_t row = layerRowCount[layer]++;
        ImVec2 fallbackPos;
        if (m_graphViewMode == GraphViewMode::PassFlow) {
            fallbackPos = ImVec2(passX + static_cast<float>(layer) * nodeSpacingX,
                                 resourceY + static_cast<float>(row) * nodeSpacingY);
        } else {
            fallbackPos = ImVec2(passX + static_cast<float>(defaultCol) * nodeSpacingX,
                                 resourceY + static_cast<float>(defaultRow) * nodeSpacingY);
            ++defaultRow;
            if (defaultRow > 4) {
                defaultRow = 0;
                ++defaultCol;
            }
        }

        drawPassNode(pass, *typeInfo, fallbackPos);
    }

    auto drawLink = [&](const std::string& linkKey,
                        const std::string& edgeId,
                        int srcPinId,
                        int dstPinId,
                        ImU32 color) {
        const int linkId = ensureUiId(linkKey);
        m_linkIdToEdgeId[linkId] = edgeId;
        ImNodes::PushColorStyle(ImNodesCol_Link, color);
        ImNodes::PushColorStyle(ImNodesCol_LinkHovered, brightenColor(color, 24));
        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, brightenColor(color, 40));
        ImNodes::Link(linkId, srcPinId, dstPinId);
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
        ImNodes::PopColorStyle();
    };

    if (m_graphViewMode == GraphViewMode::ResourceGraph) {
        for (const auto& edge : asset.edges) {
            const PassDecl* pass = asset.findPassById(edge.passId);
            const ResourceDecl* resource = asset.findResourceById(edge.resourceId);
            if (!pass || !resource) {
                continue;
            }

            int srcPinId = 0;
            int dstPinId = 0;
            if (edge.direction == "input") {
                srcPinId = ensureUiId("pin:resource:out:" + resource->id);
                dstPinId = ensureUiId("pin:pass:in:" + pass->id + ":" + edge.slotKey);
            } else {
                srcPinId = ensureUiId("pin:pass:out:" + pass->id + ":" + edge.slotKey);
                dstPinId = ensureUiId("pin:resource:in:" + resource->id);
            }

            drawLink("link:" + edge.id, edge.id, srcPinId, dstPinId, linkColorForResource(*resource));
        }
    } else {
        for (const auto& edge : asset.edges) {
            const PassDecl* consumerPass = asset.findPassById(edge.passId);
            const ResourceDecl* resource = asset.findResourceById(edge.resourceId);
            if (!consumerPass || !resource) {
                continue;
            }

            if (edge.direction == "input") {
                int srcPinId = 0;
                int dstPinId = ensureUiId("pin:pass:in:" + consumerPass->id + ":" + edge.slotKey);

                const auto flowIt = flowMap.find(resource->id);
                const bool hasProducer = flowIt != flowMap.end() && flowIt->second.producer.pass && flowIt->second.producer.edge;
                if (resource->kind == "transient" && hasProducer) {
                    srcPinId = ensureUiId("pin:pass:out:" +
                                          flowIt->second.producer.pass->id + ":" +
                                          flowIt->second.producer.edge->slotKey);
                    drawLink("link:flow:" + edge.id,
                             edge.id,
                             srcPinId,
                             dstPinId,
                             linkColorForResource(*resource));
                } else if (isTerminalResource(*resource)) {
                    srcPinId = ensureUiId("pin:resource:out:" + resource->id);
                    drawLink("link:flow:" + edge.id,
                             edge.id,
                             srcPinId,
                             dstPinId,
                             linkColorForResource(*resource));
                }
            } else if (isTerminalResource(*resource)) {
                const int srcPinId = ensureUiId("pin:pass:out:" + consumerPass->id + ":" + edge.slotKey);
                const int dstPinId = ensureUiId("pin:resource:in:" + resource->id);
                drawLink("link:flow:" + edge.id,
                         edge.id,
                         srcPinId,
                         dstPinId,
                         linkColorForResource(*resource));
            }
        }
    }

    ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        m_contextMenuSpawnPos = ImGui::GetMousePos();
        ImGui::OpenPopup("##NodeEditorContextMenu");
    }

    if (ImGui::BeginPopup("##NodeEditorContextMenu")) {
        auto byCategory = PassRegistry::instance().getTypesByCategory();
        if (ImGui::BeginMenu("Add Pass")) {
            for (const auto& [category, infos] : byCategory) {
                if (ImGui::BeginMenu(category.c_str())) {
                    for (const auto* info : infos) {
                        if (ImGui::MenuItem(info->displayName.c_str())) {
                            pushUndo(asset);
                            PassDecl pass;
                            pass.id = generatePipelineAssetGuid();
                            pass.name = makeDefaultPassName(asset, *info);
                            pass.type = info->typeName;
                            pass.enabled = true;
                            asset.passes.push_back(pass);
                            const int nodeId = ensureUiId("node:pass:" + pass.id);
                            ImNodes::SetNodeScreenSpacePos(nodeId, m_contextMenuSpawnPos);
                            m_nodePositioned[nodeId] = true;
                            m_selectedPassId = pass.id;
                            m_selectedResourceId.clear();
                            m_dirty = true;
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Add Resource")) {
            const char* kinds[] = {"transient", "imported", "backbuffer"};
            for (const char* kind : kinds) {
                if (ImGui::MenuItem(kind)) {
                    pushUndo(asset);
                    ResourceDecl resource;
                    resource.id = generatePipelineAssetGuid();
                    resource.name = makeDefaultResourceName(asset, kind);
                    resource.kind = kind;
                    resource.type = "texture";
                    resource.format = kind == std::string("backbuffer") ? "BGRA8Unorm" : "RGBA16Float";
                    resource.size = "screen";
                    if (resource.kind == "imported") {
                        resource.importKey = resource.name;
                    }
                    asset.resources.push_back(resource);
                    if (m_graphViewMode == GraphViewMode::PassFlow && resource.kind == "transient") {
                        m_graphViewMode = GraphViewMode::ResourceGraph;
                        m_nodePositioned.clear();
                    }
                    const int nodeId = ensureUiId("node:resource:" + resource.id);
                    ImNodes::SetNodeScreenSpacePos(nodeId, m_contextMenuSpawnPos);
                    m_nodePositioned[nodeId] = true;
                    m_selectedResourceId = resource.id;
                    m_selectedPassId.clear();
                    m_dirty = true;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }

    const int numSelected = ImNodes::NumSelectedNodes();
    if (numSelected > 0) {
        std::vector<int> selectedNodes(numSelected);
        ImNodes::GetSelectedNodes(selectedNodes.data());
        const int nodeId = selectedNodes[0];

        auto passIt = m_nodeIdToPassId.find(nodeId);
        if (passIt != m_nodeIdToPassId.end()) {
            m_selectedPassId = passIt->second;
            m_selectedResourceId.clear();
        } else {
            auto resourceIt = m_nodeIdToResourceId.find(nodeId);
            if (resourceIt != m_nodeIdToResourceId.end()) {
                m_selectedResourceId = resourceIt->second;
                m_selectedPassId.clear();
            }
        }
    }

    handleNewLinks(asset);
    handleDeletedLinks(asset);
}

void PipelineEditor::handleNewLinks(PipelineAsset& asset) {
    int startPin = 0;
    int endPin = 0;
    if (!ImNodes::IsLinkCreated(&startPin, &endPin)) {
        return;
    }

    const auto startIt = m_pinInfos.find(startPin);
    const auto endIt = m_pinInfos.find(endPin);
    if (startIt == m_pinInfos.end() || endIt == m_pinInfos.end()) {
        return;
    }

    auto tryCreateBinding = [&](const PinInfo& src, const PinInfo& dst) -> bool {
        if (src.kind == PinKind::ResourceOutput && dst.kind == PinKind::PassInput) {
            const PassDecl* pass = asset.findPassById(dst.ownerId);
            const ResourceDecl* resource = asset.findResourceById(src.ownerId);
            const PassTypeInfo* typeInfo = pass ? PassRegistry::instance().getTypeInfo(pass->type) : nullptr;
            const PassSlotInfo* slot = typeInfo ? findSlotInfo(typeInfo->inputSlots, dst.slotKey) : nullptr;
            if (!pass || !resource || !slot || !slotAllowsKind(*slot, resource->kind)) {
                return false;
            }
            pushUndo(asset);
            if (setSlotBinding(asset, pass->id, "input", dst.slotKey, resource->id)) {
                m_dirty = true;
            }
            return true;
        }

        if (src.kind == PinKind::PassOutput && dst.kind == PinKind::ResourceInput) {
            const PassDecl* pass = asset.findPassById(src.ownerId);
            const ResourceDecl* resource = asset.findResourceById(dst.ownerId);
            const PassTypeInfo* typeInfo = pass ? PassRegistry::instance().getTypeInfo(pass->type) : nullptr;
            const PassSlotInfo* slot = typeInfo ? findSlotInfo(typeInfo->outputSlots, src.slotKey) : nullptr;
            if (!pass || !resource || !slot || !slotAllowsKind(*slot, resource->kind)) {
                return false;
            }
            pushUndo(asset);
            if (setSlotBinding(asset, pass->id, "output", src.slotKey, resource->id)) {
                m_dirty = true;
            }
            return true;
        }

        if (src.kind == PinKind::PassOutput && dst.kind == PinKind::PassInput) {
            const PassDecl* srcPass = asset.findPassById(src.ownerId);
            const PassDecl* dstPass = asset.findPassById(dst.ownerId);
            const PassTypeInfo* srcTypeInfo = srcPass ? PassRegistry::instance().getTypeInfo(srcPass->type) : nullptr;
            const PassTypeInfo* dstTypeInfo = dstPass ? PassRegistry::instance().getTypeInfo(dstPass->type) : nullptr;
            const PassSlotInfo* srcSlot = srcTypeInfo ? findSlotInfo(srcTypeInfo->outputSlots, src.slotKey) : nullptr;
            const PassSlotInfo* dstSlot = dstTypeInfo ? findSlotInfo(dstTypeInfo->inputSlots, dst.slotKey) : nullptr;
            if (!srcPass || !dstPass || !srcSlot || !dstSlot) {
                return false;
            }

            const EdgeDecl* outputEdge = asset.findEdge(srcPass->id, "output", src.slotKey);
            const EdgeDecl* inputEdge = asset.findEdge(dstPass->id, "input", dst.slotKey);
            const ResourceDecl* outputResource = outputEdge ? asset.findResourceById(outputEdge->resourceId) : nullptr;
            const ResourceDecl* inputResource = inputEdge ? asset.findResourceById(inputEdge->resourceId) : nullptr;

            const ResourceDecl* bindingResource = nullptr;
            if (outputResource && outputResource->kind == "transient" && slotAllowsKind(*dstSlot, "transient")) {
                bindingResource = outputResource;
            } else if (!outputResource &&
                       inputResource &&
                       inputResource->kind == "transient" &&
                       slotAllowsKind(*srcSlot, "transient")) {
                bindingResource = inputResource;
            } else {
                return false;
            }

            pushUndo(asset);
            bool changed = false;
            if ((!outputResource || outputResource->id != bindingResource->id) &&
                slotAllowsKind(*srcSlot, bindingResource->kind)) {
                changed |= setSlotBinding(asset, srcPass->id, "output", src.slotKey, bindingResource->id);
            }
            if (!inputResource || inputResource->id != bindingResource->id) {
                changed |= setSlotBinding(asset, dstPass->id, "input", dst.slotKey, bindingResource->id);
            }
            if (changed) {
                m_dirty = true;
            }
            return changed;
        }

        return false;
    };

    if (!tryCreateBinding(startIt->second, endIt->second)) {
        tryCreateBinding(endIt->second, startIt->second);
    }
}

void PipelineEditor::handleDeletedLinks(PipelineAsset& asset) {
    int linkId = 0;
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        auto it = m_linkIdToEdgeId.find(linkId);
        if (it != m_linkIdToEdgeId.end()) {
            pushUndo(asset);
            asset.edges.erase(
                std::remove_if(asset.edges.begin(),
                               asset.edges.end(),
                               [&](const EdgeDecl& edge) { return edge.id == it->second; }),
                asset.edges.end());
            m_dirty = true;
        }
    }

    if (!(ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_Backspace))) {
        return;
    }

    const int selectedCount = ImNodes::NumSelectedNodes();
    if (selectedCount <= 0) {
        return;
    }

    std::vector<int> selectedNodes(selectedCount);
    ImNodes::GetSelectedNodes(selectedNodes.data());

    std::vector<std::string> passIds;
    std::vector<std::string> resourceIds;
    for (const int nodeId : selectedNodes) {
        auto passIt = m_nodeIdToPassId.find(nodeId);
        if (passIt != m_nodeIdToPassId.end()) {
            passIds.push_back(passIt->second);
        }
        auto resourceIt = m_nodeIdToResourceId.find(nodeId);
        if (resourceIt != m_nodeIdToResourceId.end()) {
            resourceIds.push_back(resourceIt->second);
        }
    }

    if (passIds.empty() && resourceIds.empty()) {
        return;
    }

    pushUndo(asset);
    for (const auto& passId : passIds) {
        asset.removeEdgesForPass(passId);
    }
    for (const auto& resourceId : resourceIds) {
        asset.removeEdgesForResource(resourceId);
    }

    asset.passes.erase(
        std::remove_if(asset.passes.begin(),
                       asset.passes.end(),
                       [&](const PassDecl& pass) {
                           return std::find(passIds.begin(), passIds.end(), pass.id) != passIds.end();
                       }),
        asset.passes.end());

    asset.resources.erase(
        std::remove_if(asset.resources.begin(),
                       asset.resources.end(),
                       [&](const ResourceDecl& resource) {
                           return std::find(resourceIds.begin(), resourceIds.end(), resource.id) != resourceIds.end();
                       }),
        asset.resources.end());

    m_selectedPassId.clear();
    m_selectedResourceId.clear();
    m_dirty = true;
}

void PipelineEditor::renderPropertyPanel(PipelineAsset& asset) {
    ImGui::Text("Properties");
    ImGui::Separator();

    const auto flowMap = buildResourceFlowMap(asset);
    auto findFlowInfo = [&](const std::string& resourceId) -> const ResourceFlowInfo* {
        auto it = flowMap.find(resourceId);
        return it != flowMap.end() ? &it->second : nullptr;
    };

    auto renderPassRef = [&](const PassBindingInfo& binding, const char* prefix) {
        if (!binding.pass || !binding.edge) {
            return;
        }
        const ResourceDecl* resource = asset.findResourceById(binding.edge->resourceId);
        ImGui::BulletText("%s%s.%s", prefix, binding.pass->name.c_str(), binding.edge->slotKey.c_str());
        if (resource) {
            ImGui::SameLine();
            ImGui::TextDisabled("[%s]", resource->name.c_str());
        }
    };

    if (!m_selectedPassId.empty()) {
        PassDecl* pass = asset.findPassById(m_selectedPassId);
        if (pass) {
            const PassTypeInfo* typeInfo = PassRegistry::instance().getTypeInfo(pass->type);

            ImGui::Text("Pass");
            ImGui::Separator();

            char nameBuf[128];
            copyTextToBuffer(nameBuf, pass->name);
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
                pass->name = nameBuf;
                m_dirty = true;
            }
            if (ImGui::IsItemActivated() && !m_hasTextEditSnapshot) {
                pushUndo(asset);
                m_hasTextEditSnapshot = true;
            }
            if (ImGui::IsItemDeactivated()) {
                m_hasTextEditSnapshot = false;
            }

            ImGui::Text("Type: %s", pass->type.c_str());
            ImGui::TextDisabled("ID: %.8s", pass->id.c_str());

            {
                const bool prev = pass->enabled;
                if (ImGui::Checkbox("Enabled", &pass->enabled)) {
                    pass->enabled = prev;
                    pushUndo(asset);
                    pass->enabled = !prev;
                    m_dirty = true;
                }
            }

            {
                const bool prev = pass->sideEffect;
                if (ImGui::Checkbox("Side Effect", &pass->sideEffect)) {
                    pass->sideEffect = prev;
                    pushUndo(asset);
                    pass->sideEffect = !prev;
                    m_dirty = true;
                }
            }

            if (typeInfo) {
                auto renderSlotEditor = [&](const char* title,
                                            const std::vector<PassSlotInfo>& slots,
                                            const char* direction) {
                    if (slots.empty()) {
                        return;
                    }

                    ImGui::Separator();
                    ImGui::Text("%s", title);
                    for (const auto& slot : slots) {
                        const EdgeDecl* edge = asset.findEdge(pass->id, direction, slot.key);
                        const ResourceDecl* boundResource = edge ? asset.findResourceById(edge->resourceId) : nullptr;
                        const std::string preview = boundResource ? resourceLabel(*boundResource) : std::string{"<unbound>"};
                        const std::string comboLabel = slot.displayName + "##" + direction + slot.key;

                        if (ImGui::BeginCombo(comboLabel.c_str(), preview.c_str())) {
                            if (ImGui::Selectable("<unbound>", boundResource == nullptr)) {
                                pushUndo(asset);
                                if (setSlotBinding(asset, pass->id, direction, slot.key, {})) {
                                    m_dirty = true;
                                }
                            }

                            for (const ResourceDecl* candidate : collectCandidates(asset, slot)) {
                                const bool selected = boundResource && candidate->id == boundResource->id;
                                const std::string label = resourceLabel(*candidate);
                                if (ImGui::Selectable(label.c_str(), selected)) {
                                    pushUndo(asset);
                                    if (setSlotBinding(asset, pass->id, direction, slot.key, candidate->id)) {
                                        m_dirty = true;
                                    }
                                }
                            }
                            ImGui::EndCombo();
                        }
                        if (slot.optional) {
                            ImGui::SameLine();
                            ImGui::TextDisabled("(optional)");
                        }
                    }
                };

                renderSlotEditor("Inputs", typeInfo->inputSlots, "input");
                renderSlotEditor("Outputs", typeInfo->outputSlots, "output");

                if (ImGui::CollapsingHeader("Flow", ImGuiTreeNodeFlags_DefaultOpen)) {
                    std::vector<PassBindingInfo> upstream;
                    std::vector<PassBindingInfo> downstream;

                    for (const auto& slot : typeInfo->inputSlots) {
                        const EdgeDecl* edge = asset.findEdge(pass->id, "input", slot.key);
                        if (!edge) {
                            continue;
                        }
                        const ResourceFlowInfo* flow = findFlowInfo(edge->resourceId);
                        if (flow && flow->producer.pass && flow->producer.pass->id != pass->id) {
                            upstream.push_back(flow->producer);
                        }
                    }

                    for (const auto& slot : typeInfo->outputSlots) {
                        const EdgeDecl* edge = asset.findEdge(pass->id, "output", slot.key);
                        if (!edge) {
                            continue;
                        }
                        const ResourceFlowInfo* flow = findFlowInfo(edge->resourceId);
                        if (!flow) {
                            continue;
                        }
                        for (const auto& reader : flow->readers) {
                            if (reader.pass && reader.pass->id != pass->id) {
                                downstream.push_back(reader);
                            }
                        }
                    }

                    ImGui::Text("Upstream");
                    if (upstream.empty()) {
                        ImGui::TextDisabled("none");
                    } else {
                        for (const auto& binding : upstream) {
                            renderPassRef(binding, "");
                        }
                    }

                    ImGui::Separator();
                    ImGui::Text("Downstream");
                    if (downstream.empty()) {
                        ImGui::TextDisabled("none");
                    } else {
                        for (const auto& binding : downstream) {
                            renderPassRef(binding, "");
                        }
                    }
                }
            }
        }
    } else if (!m_selectedResourceId.empty()) {
        ResourceDecl* resource = asset.findResourceById(m_selectedResourceId);
        if (resource) {
            ImGui::Text("Resource");
            ImGui::Separator();

            char nameBuf[128];
            copyTextToBuffer(nameBuf, resource->name);
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
                resource->name = nameBuf;
                m_dirty = true;
            }
            if (ImGui::IsItemActivated() && !m_hasTextEditSnapshot) {
                pushUndo(asset);
                m_hasTextEditSnapshot = true;
            }
            if (ImGui::IsItemDeactivated()) {
                m_hasTextEditSnapshot = false;
            }

            ImGui::TextDisabled("ID: %.8s", resource->id.c_str());

            {
                const char* kinds[] = {"transient", "imported", "backbuffer"};
                int currentKind = 0;
                for (int i = 0; i < IM_ARRAYSIZE(kinds); ++i) {
                    if (resource->kind == kinds[i]) {
                        currentKind = i;
                        break;
                    }
                }
                const int prevKind = currentKind;
                if (ImGui::Combo("Kind", &currentKind, kinds, IM_ARRAYSIZE(kinds))) {
                    pushUndo(asset);
                    resource->kind = kinds[currentKind];
                    if (resource->kind == "imported" && resource->importKey.empty()) {
                        resource->importKey = resource->name;
                    }
                    if (resource->kind != "imported") {
                        resource->importKey.clear();
                    }
                    if (resource->kind == "backbuffer") {
                        resource->type = "texture";
                    }
                    m_dirty = currentKind != prevKind;
                }
            }

            {
                const char* types[] = {"texture", "buffer", "token"};
                int currentType = 0;
                for (int i = 0; i < IM_ARRAYSIZE(types); ++i) {
                    if (resource->type == types[i]) {
                        currentType = i;
                        break;
                    }
                }
                const int prevType = currentType;
                if (ImGui::Combo("Type", &currentType, types, IM_ARRAYSIZE(types))) {
                    pushUndo(asset);
                    resource->type = types[currentType];
                    m_dirty = currentType != prevType;
                }
            }

            const char* formats[] = {
                "R8Unorm", "R16Float", "R32Float", "R32Uint",
                "RGBA8Unorm", "RGBA8Srgb", "BGRA8Unorm", "RGBA16Float", "RGBA32Float",
                "Depth32Float", "Depth16Unorm"
            };
            int currentFormat = 0;
            for (int i = 0; i < IM_ARRAYSIZE(formats); ++i) {
                if (resource->format == formats[i]) {
                    currentFormat = i;
                    break;
                }
            }
            if (ImGui::Combo("Format", &currentFormat, formats, IM_ARRAYSIZE(formats))) {
                pushUndo(asset);
                resource->format = formats[currentFormat];
                m_dirty = true;
            }

            char sizeBuf[64];
            copyTextToBuffer(sizeBuf, resource->size);
            if (ImGui::InputText("Size", sizeBuf, sizeof(sizeBuf))) {
                resource->size = sizeBuf;
                m_dirty = true;
            }
            if (ImGui::IsItemActivated() && !m_hasTextEditSnapshot) {
                pushUndo(asset);
                m_hasTextEditSnapshot = true;
            }
            if (ImGui::IsItemDeactivated()) {
                m_hasTextEditSnapshot = false;
            }

            if (resource->kind == "imported") {
                char importBuf[128];
                copyTextToBuffer(importBuf, resource->importKey);
                if (ImGui::InputText("Import Key", importBuf, sizeof(importBuf))) {
                    resource->importKey = importBuf;
                    m_dirty = true;
                }
                if (ImGui::IsItemActivated() && !m_hasTextEditSnapshot) {
                    pushUndo(asset);
                    m_hasTextEditSnapshot = true;
                }
                if (ImGui::IsItemDeactivated()) {
                    m_hasTextEditSnapshot = false;
                }
            }

            if (ImGui::CollapsingHeader("Flow", ImGuiTreeNodeFlags_DefaultOpen)) {
                const ResourceFlowInfo* flow = findFlowInfo(resource->id);
                if (!flow) {
                    ImGui::TextDisabled("No flow data");
                } else {
                    ImGui::Text("Writers");
                    if (flow->writers.empty()) {
                        ImGui::TextDisabled("none");
                    } else {
                        for (const auto& writer : flow->writers) {
                            renderPassRef(writer, "");
                        }
                    }

                    ImGui::Separator();
                    ImGui::Text("Readers");
                    if (flow->readers.empty()) {
                        ImGui::TextDisabled("none");
                    } else {
                        for (const auto& reader : flow->readers) {
                            renderPassRef(reader, "");
                        }
                    }
                }
            }
        }
    } else {
        ImGui::TextDisabled("Select a node");
    }

    ImGui::Separator();

    std::string errorMsg;
    const bool valid = asset.validate(errorMsg);
    if (valid) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Valid");
    } else {
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f), "Error:");
        ImGui::TextWrapped("%s", errorMsg.c_str());
    }

    ImGui::Text("Passes: %zu", asset.passes.size());
    ImGui::Text("Resources: %zu", asset.resources.size());
    ImGui::Text("Edges: %zu", asset.edges.size());

    ImGui::Separator();
    renderCompilationPreview(asset);
}

void PipelineEditor::renderCompilationPreview(const PipelineAsset& asset) {
    if (!ImGui::CollapsingHeader("Compilation Preview", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    const auto sortedIndices = asset.topologicalSort(false);
    size_t enabledCount = 0;
    for (const auto& pass : asset.passes) {
        if (pass.enabled) {
            ++enabledCount;
        }
    }

    if (sortedIndices.size() != enabledCount) {
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.2f, 1.0f),
                           "Cycle detected! Cannot determine execution order.");
        return;
    }

    ImGui::Text("Execution Order:");
    if (ImGui::BeginTable("##exec_order", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 24.0f);
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Category");
        ImGui::TableSetupColumn("Bindings");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            const PassDecl& pass = asset.passes[sortedIndices[i]];
            const PassTypeInfo* info = PassRegistry::instance().getTypeInfo(pass.type);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i);
            ImGui::TableNextColumn();
            ImGui::Text("%s", pass.name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", info ? info->category.c_str() : "?");
            ImGui::TableNextColumn();

            std::string summary;
            if (info) {
                for (const auto& slot : info->inputSlots) {
                    if (!summary.empty()) {
                        summary += ", ";
                    }
                    summary += slot.displayName + "=" + slotBindingName(asset, pass.id, "input", slot);
                }
                summary += " -> ";
                for (size_t slotIndex = 0; slotIndex < info->outputSlots.size(); ++slotIndex) {
                    if (slotIndex > 0) {
                        summary += ", ";
                    }
                    const auto& slot = info->outputSlots[slotIndex];
                    summary += slot.displayName + "=" + slotBindingName(asset, pass.id, "output", slot);
                }
            }
            ImGui::TextDisabled("%s", summary.c_str());
        }
        ImGui::EndTable();
    }

    bool hasDisabled = false;
    for (const auto& pass : asset.passes) {
        if (!pass.enabled) {
            hasDisabled = true;
            break;
        }
    }
    if (hasDisabled) {
        ImGui::Spacing();
        ImGui::TextDisabled("Disabled:");
        for (const auto& pass : asset.passes) {
            if (!pass.enabled) {
                ImGui::TextDisabled("  - %s (%s)", pass.name.c_str(), pass.type.c_str());
            }
        }
    }

    if (!asset.resources.empty() && ImGui::TreeNode("Resource Lifetimes")) {
        std::unordered_map<std::string, std::string> producer;
        std::unordered_map<std::string, std::string> lastConsumer;

        for (const size_t idx : sortedIndices) {
            const PassDecl& pass = asset.passes[idx];
            const PassTypeInfo* info = PassRegistry::instance().getTypeInfo(pass.type);
            if (!info) {
                continue;
            }

            for (const auto& slot : info->outputSlots) {
                const EdgeDecl* edge = asset.findEdge(pass.id, "output", slot.key);
                if (!edge) {
                    continue;
                }
                const ResourceDecl* resource = asset.findResourceById(edge->resourceId);
                if (!resource) {
                    continue;
                }
                if (resource->kind == "transient" && producer.find(resource->id) == producer.end()) {
                    producer[resource->id] = pass.name;
                }
            }

            for (const auto& slot : info->inputSlots) {
                const EdgeDecl* edge = asset.findEdge(pass.id, "input", slot.key);
                if (!edge) {
                    continue;
                }
                lastConsumer[edge->resourceId] = pass.name;
            }
        }

        if (ImGui::BeginTable("##res_lifetime", 4,
                ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Resource");
            ImGui::TableSetupColumn("Kind");
            ImGui::TableSetupColumn("Producer");
            ImGui::TableSetupColumn("Last Consumer");
            ImGui::TableHeadersRow();

            for (const auto& resource : asset.resources) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", resource.name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%s", resource.kind.c_str());
                ImGui::TableNextColumn();
                auto producerIt = producer.find(resource.id);
                if (producerIt != producer.end()) {
                    ImGui::Text("%s", producerIt->second.c_str());
                } else {
                    ImGui::TextDisabled("none");
                }
                ImGui::TableNextColumn();
                auto consumerIt = lastConsumer.find(resource.id);
                if (consumerIt != lastConsumer.end()) {
                    ImGui::Text("%s", consumerIt->second.c_str());
                } else {
                    ImGui::TextDisabled("none");
                }
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }
}

void PipelineEditor::collectNodePositions(PipelineAsset& asset) {
    for (auto& resource : asset.resources) {
        const int nodeId = ensureUiId("node:resource:" + resource.id);
        if (!m_renderedNodeIds.count(nodeId)) {
            continue;
        }
        const ImVec2 pos = ImNodes::GetNodeGridSpacePos(nodeId);
        resource.editorPos = {pos.x, pos.y};
    }
    for (auto& pass : asset.passes) {
        const int nodeId = ensureUiId("node:pass:" + pass.id);
        if (!m_renderedNodeIds.count(nodeId)) {
            continue;
        }
        const ImVec2 pos = ImNodes::GetNodeGridSpacePos(nodeId);
        pass.editorPos = {pos.x, pos.y};
    }
}

void PipelineEditor::autoReorderNodes(PipelineAsset& asset) {
    const auto passLayers = computePassLayers(asset);
    const auto flowMap = buildResourceFlowMap(asset);

    std::map<int, std::vector<size_t>> layerPasses;
    int maxLayer = 0;
    for (size_t i = 0; i < asset.passes.size(); ++i) {
        const int layer = i < passLayers.size() ? passLayers[i] : 0;
        layerPasses[layer].push_back(i);
        maxLayer = std::max(maxLayer, layer);
    }

    const float resourceX = 40.0f;
    const float resourceY = 50.0f;
    const float passX = 320.0f;
    const float nodeSpacingX = 320.0f;
    const float nodeSpacingY = 170.0f;
    const float terminalRightX = passX + static_cast<float>(maxLayer + 1) * nodeSpacingX;

    std::unordered_map<std::string, ImVec2> passPositions;
    for (const auto& [layer, indices] : layerPasses) {
        for (size_t row = 0; row < indices.size(); ++row) {
            PassDecl& pass = asset.passes[indices[row]];
            pass.editorPos = {
                passX + static_cast<float>(layer) * nodeSpacingX,
                resourceY + static_cast<float>(row) * nodeSpacingY
            };
            passPositions[pass.id] = ImVec2(pass.editorPos[0], pass.editorPos[1]);
        }
    }

    size_t importedRow = 0;
    size_t backbufferRow = 0;
    size_t fallbackRow = 0;
    for (auto& resource : asset.resources) {
        const ResourceFlowInfo* flow = nullptr;
        auto flowIt = flowMap.find(resource.id);
        if (flowIt != flowMap.end()) {
            flow = &flowIt->second;
        }

        float posX = resourceX;
        float posY = resourceY + static_cast<float>(fallbackRow++) * 120.0f;

        if (resource.kind == "imported") {
            posX = resourceX;
            posY = resourceY + static_cast<float>(importedRow++) * 140.0f;
            if (flow && !flow->readers.empty()) {
                float totalY = 0.0f;
                for (const auto& reader : flow->readers) {
                    totalY += passPositions[reader.pass->id].y;
                }
                posY = totalY / static_cast<float>(flow->readers.size());
            }
        } else if (resource.kind == "backbuffer") {
            posX = terminalRightX;
            posY = resourceY + static_cast<float>(backbufferRow++) * 140.0f;
            if (flow && !flow->writers.empty()) {
                float totalY = 0.0f;
                for (const auto& writer : flow->writers) {
                    totalY += passPositions[writer.pass->id].y;
                }
                posY = totalY / static_cast<float>(flow->writers.size());
            }
        } else if (flow && flow->producer.pass) {
            const ImVec2 producerPos = passPositions[flow->producer.pass->id];
            posX = producerPos.x + nodeSpacingX * 0.5f;
            posY = producerPos.y;
            if (!flow->readers.empty()) {
                float totalY = producerPos.y;
                for (const auto& reader : flow->readers) {
                    totalY += passPositions[reader.pass->id].y;
                }
                posY = totalY / static_cast<float>(flow->readers.size() + 1);
            }
        }

        resource.editorPos = {posX, posY};
    }

    m_nodePositioned.clear();
}
