#include "Editor/EditorMaterialGraph.h"
#include "Editor/EditorApplication.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "imgui.h"
#include "imnodes.h"
#include <algorithm>
#include <fstream>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif

namespace metallic {
namespace {
using Json = nlohmann::json;
const material::MaterialGraphNodeDesc& descriptor(const Json& node)
{
    const auto& kinds = material::materialGraphNodes();
    return *std::find_if(kinds.begin(), kinds.end(), [&](const auto& d) { return d.kind == node["kind"].get<std::string>(); });
}
bool editValue(const char* label, Json& value)
{
    if (!value.is_number() && !value.is_array()) { ImGui::TextDisabled("%s: input expression", label); return false; }
    if (value.is_array() && value.size() != 4) { return false; }
    float values[4];
    for (int i = 0; i < 4; ++i) { values[i] = value.is_array() ? value[i].get<float>() : value.get<float>(); }
    if (!ImGui::DragFloat4(label, values, .005f, 0, 0, "%.3f")) { return false; }
    value = Json::array({values[0],values[1],values[2],values[3]}); return true;
}
void combo(const char* label, Json& value, std::initializer_list<const char*> options)
{
    const auto current = value.get<std::string>();
    if (ImGui::BeginCombo(label, current.c_str())) {
        for (const auto* option : options) { if (ImGui::Selectable(option, current == option)) { value = option; } }
        ImGui::EndCombo();
    }
}
void writeFile(const std::filesystem::path& path, const std::string& contents)
{
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    auto temporary = path; temporary += ".tmp";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file << contents; file.close();
    if (!file) { throw std::runtime_error("Cannot write " + path.string()); }
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("Cannot replace " + path.string());
    }
#else
    std::filesystem::rename(temporary, path);
#endif
}
} // namespace

bool EditorApplication::applyMaterialGraph(int32_t materialIndex, const material::CompiledMaterialFrontend& compiled)
{
    if (materialIndex < 0 || static_cast<size_t>(materialIndex) >= scene_.materials().size()) { return false; }
    finishActiveInspectorPropertyTransaction();
    const auto before = scene_.materials()[materialIndex];
    auto after = before;
    after.valueProgram = compiled.definition.surfaceProgram;
    for (const auto& feature : compiled.definition.schema.features) {
        if (feature.name == "alphaMode") { after.alphaMode = feature.defaultValue == "mask" ? "MASK" : "OPAQUE"; }
    }
    after.valueParameters.fill(0);
    for (const auto& [slot,value] : compiled.definition.valueParameters.items()) {
        const auto offset = std::stoul(slot) * 4;
        for (size_t i = 0; i < 4; ++i) { after.valueParameters.at(offset+i) = value[i].get<float>(); }
    }
    std::string diagnostics;
    std::vector<scene::RenderMaterial> candidates(scene_.materials().begin(),scene_.materials().end());
    candidates[materialIndex]=after;
    if (!render::MaterialValueProgramSet::create(candidates,diagnostics)) {
        sceneStatus_ = "Material Graph: " + diagnostics;
        return false;
    }
    if (!applySceneEditValue(scene::kNullSceneEntity,MaterialEditValue{materialIndex,after})) {
        sceneStatus_ = "Material Graph rejected: use a lit opaque/MASK material without transmission. " + scene_.documentWarning();
        return false;
    }
    pushSceneEditCommand(scene::kNullSceneEntity,scene_.sceneGraph().lifetimeRevision(),
        MaterialEditValue{materialIndex,before},MaterialEditValue{materialIndex,after});
    sceneStatus_ = "Applied Material Graph to " + before.name + ". Ctrl+Z to undo; Ctrl+S to save scene.";
    return true;
}

void EditorApplication::drawMaterialGraphEditor()
{
    const auto selected = selectedMaterialIndices();
    const char* name = selected.size() == 1 ? scene_.materials()[selected.front()].name.c_str() : nullptr;
    if (materialGraphEditor_.draw(name)) {
        applyMaterialGraph(selected.front(),materialGraphEditor_.compiled());
        materialGraphEditor_.setStatus(sceneStatus_);
    }
}

void EditorMaterialGraph::shutdown()
{
    if (context_) { ImNodes::DestroyContext(context_); context_ = nullptr; }
}

void EditorMaterialGraph::replaceGraph(Json graph)
{
    graph_ = std::move(graph); restorePositions_ = true; selected_ = 0; editStart_.reset();
}

bool EditorMaterialGraph::compile()
{
    std::string error;
    Json defaults = Json::object();
    for (const auto& node : graph_["nodes"]) {
        if (node["kind"] == "Parameter") { defaults[std::to_string(node["slot"].get<int>())] = node["default"]; }
    }
    const bool valid = textMode_
        ? material::compileSlangMaterial(source_.data(), defaults, compiled_, error)
        : material::compileMaterialGraph(graph_, compiled_, error);
    status_ = valid ? "Compiled successfully. Apply to preview or export assets." : error;
    return valid;
}

bool EditorMaterialGraph::load(const std::filesystem::path& path)
{
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file || std::filesystem::file_size(path) > 262144) { throw std::runtime_error("Cannot read graph (256 KiB limit)"); }
        auto graph = Json::parse(file);
        // Allow incomplete links during authoring, but validate the editor's structural data.
        if (graph.at("type") != "Metallic.MaterialGraph" || graph.at("version") != 1 ||
            !graph.at("nodes").is_array() || graph["nodes"].size() > 128 || !graph.at("output").is_number_integer()) { throw std::runtime_error("Invalid graph document"); }
        const auto validateValue = [](const Json& value) {
            const auto number = [](const Json& v) { return v.is_number() && std::isfinite(v.get<float>()) && std::abs(v.get<float>()) <= 1e6f; };
            if (number(value)) { return; }
            if (value.is_array() && value.size() == 4 && std::all_of(value.begin(),value.end(),number)) { return; }
            throw std::runtime_error("Expected a finite scalar or float4 literal");
        };
        std::vector<int> ids;
        for (const auto& node : graph["nodes"]) {
            const int id = node.at("id").get<int>();
            if (id <= 0 || id > 100000 || std::find(ids.begin(),ids.end(),id) != ids.end()) { throw std::runtime_error("Invalid node ID"); }
            ids.push_back(id);
            auto prototype = material::makeMaterialGraphNode(node.at("kind").get<std::string>(), id);
            if (!node.at("inputs").is_object() || node.at("position").size() != 2) { throw std::runtime_error("Invalid node properties"); }
            for (const auto& coordinate : node["position"]) {
                if (!coordinate.is_number() || !std::isfinite(coordinate.get<float>())) { throw std::runtime_error("Invalid position"); }
            }
            // Every property used by the inspector must have the prototype's shape.
            for (const auto& [key,value] : prototype.items()) {
                if (!node.contains(key) || (value.is_string() && !node[key].is_string())) { throw std::runtime_error("Missing node property " + key); }
            }
            for (const auto& input : descriptor(prototype).inputs) {
                if (!node["inputs"].contains(input)) { throw std::runtime_error("Missing input " + input); }
                const auto& value=node["inputs"][input];
                if (!value.is_object()) { validateValue(value); }
                else if (value.contains("node") && (!value["node"].is_number_integer() || value["node"] < 1 || value["node"] > 100000)) {
                    throw std::runtime_error("Invalid link ID");
                }
            }
            if (node["kind"] == "Parameter" && (!node["slot"].is_number_integer() || node["slot"] < 0 || node["slot"] > 3)) { throw std::runtime_error("Invalid parameter slot"); }
            if (node["kind"] == "Constant") { validateValue(node["value"]); }
            if (node["kind"] == "Parameter") { validateValue(node["default"]); }
        }
        undo_.push_back(graph_); redo_.clear(); replaceGraph(std::move(graph)); dirty_ = false;
        status_ = "Loaded " + path.string(); return true;
    } catch (const std::exception& error) { status_ = error.what(); return false; }
}

bool EditorMaterialGraph::save(const std::filesystem::path& path)
{
    try { writeFile(path, graph_.dump(2)); dirty_ = false; status_ = "Saved " + path.string(); return true; }
    catch (const std::exception& error) { status_ = error.what(); return false; }
}

bool EditorMaterialGraph::exportAssets(const std::filesystem::path& definitionPath)
{
    if (!compile()) { return false; }
    try {
        material::MaterialInstance instance;
        instance.definition = "asset://" + definitionPath.filename().generic_string();
        auto instancePath = definitionPath; instancePath.replace_extension(".material");
        auto reflectionPath = definitionPath; reflectionPath.replace_extension(".reflection.json");
        writeFile(definitionPath, material::serializeMaterialDefinition(compiled_.definition));
        writeFile(instancePath, material::serializeMaterialInstance(instance));
        writeFile(reflectionPath, compiled_.reflection.dump(2));
        if (textMode_) { auto sourcePath = definitionPath; sourcePath.replace_extension(".material.slang"); writeFile(sourcePath,source_.data()); }
        status_ = "Exported definition, instance and reflection: " + definitionPath.string(); return true;
    } catch (const std::exception& error) { status_ = error.what(); return false; }
}

void EditorMaterialGraph::properties()
{
    auto node = std::find_if(graph_["nodes"].begin(),graph_["nodes"].end(),[&](const auto& n) { return n["id"] == selected_; });
    if (node == graph_["nodes"].end()) { ImGui::TextWrapped("Select a node to edit its inputs. Drag pins to connect nodes."); return; }
    auto& n = *node;
    const auto kind = n["kind"].get<std::string>();
    ImGui::Text("%s #%d", kind.c_str(), selected_);
    if (kind == "Output" && ImGui::Button("Use as graph output")) { graph_["output"] = selected_; }
    if (kind == "Constant") { editValue("Value", n["value"]); }
    if (kind == "Parameter") {
        int slot = n["slot"].get<int>();
        if (ImGui::SliderInt("Slot", &slot, 0, 3)) { n["slot"] = slot; }
        editValue("Default", n["default"]);
        ImGui::TextWrapped("float4 instance input. Defaults do not change program identity.");
    }
    if (kind == "Texture") {
        combo("Resource", n["texture"], {"baseColor","metallicRoughness","normal","occlusion","emissive","specular"});
        combo("Footprint", n["footprint"], {"RayCone","ExplicitLOD"});
    }
    if (kind == "Swizzle") { combo("Components", n["components"], {"xxxx","yyyy","zzzz","wwww","xyzw","xyzx","xyxy"}); }
    if (kind == "Math") { combo("Operation", n["operation"], {"add","multiply","dot","lerp","clamp","select","sin","fract","abs","saturate","normalize"}); }
    for (const auto& input : descriptor(n).inputs) {
        ImGui::PushID(input.c_str());
        auto& value = n["inputs"][input];
        if (value.is_object()) {
            ImGui::TextWrapped("%s: %s", input.c_str(), value.dump().c_str());
            if (ImGui::SmallButton("Disconnect / reset")) { value = material::makeMaterialGraphNode(kind,selected_)["inputs"][input]; }
        } else { editValue(input.c_str(), value); }
        ImGui::PopID();
    }
    if (ImGui::Button("Delete node")) {
        graph_["nodes"].erase(node);
        for (auto& other : graph_["nodes"]) {
            for (auto& [input,value] : other["inputs"].items()) {
                if (value.is_object() && value.value("node",0) == selected_) { value = material::makeMaterialGraphNode(other["kind"].get<std::string>(),other["id"])["inputs"][input]; }
            }
        }
        selected_ = 0;
    }
}

void EditorMaterialGraph::canvas()
{
    auto* previous = ImNodes::GetCurrentContext();
    if (!context_) { context_ = ImNodes::CreateContext(); }
    ImNodes::SetCurrentContext(context_);
    ImNodes::BeginNodeEditor();
    for (const auto& node : graph_["nodes"]) {
        const int id = node["id"];
        if (restorePositions_) { ImNodes::SetNodeGridSpacePos(id, {node["position"][0].get<float>(),node["position"][1].get<float>()}); }
        const auto& desc = descriptor(node);
        ImNodes::BeginNode(id);
        ImNodes::BeginNodeTitleBar(); ImGui::Text("%s #%d%s",desc.kind.c_str(), id, graph_["output"] == id ? " [root]" : ""); ImNodes::EndNodeTitleBar();
        for (size_t i = 0; i < desc.inputs.size(); ++i) {
            ImNodes::BeginInputAttribute(id * 16 + static_cast<int>(i) + 1);
            ImGui::TextUnformatted(desc.inputs[i].c_str()); ImNodes::EndInputAttribute();
        }
        if (desc.kind != "Output") {
            ImNodes::BeginOutputAttribute(id * 16); ImGui::TextUnformatted(desc.surface ? "Surface" : "Value (float4)"); ImNodes::EndOutputAttribute();
        }
        ImNodes::EndNode();
    }
    for (const auto& node : graph_["nodes"]) {
        const auto& desc = descriptor(node);
        for (size_t i = 0; i < desc.inputs.size(); ++i) {
            const auto& value = node["inputs"][desc.inputs[i]];
            if (value.is_object() && value.contains("node") && value["node"].is_number_integer()) {
                const int source = value["node"];
                const bool exists = std::any_of(graph_["nodes"].begin(),graph_["nodes"].end(),[&](const auto& n) { return n["id"] == source && n["kind"] != "Output"; });
                const int pin = node["id"].get<int>() * 16 + static_cast<int>(i) + 1;
                if (exists) { ImNodes::Link(pin, source * 16, pin); }
            }
        }
    }
    ImNodes::MiniMap(.15f);
    ImNodes::EndNodeEditor();
    int start = 0, end = 0;
    if (ImNodes::IsLinkCreated(&start, &end)) {
        if (start % 16 != 0) { std::swap(start,end); }
        for (auto& node : graph_["nodes"]) {
            const auto& inputs = descriptor(node).inputs;
            if (node["id"] == end / 16 && end % 16 > 0 && size_t(end % 16) <= inputs.size() && start % 16 == 0) {
                node["inputs"][inputs[end % 16 - 1]] = {{"node",start / 16}};
            }
        }
    }
    const int count = ImNodes::NumSelectedNodes();
    if (count > 0) { std::vector<int> selection(count); ImNodes::GetSelectedNodes(selection.data()); selected_ = selection.front(); }
    for (auto& node : graph_["nodes"]) {
        const auto position = ImNodes::GetNodeGridSpacePos(node["id"].get<int>());
        node["position"] = {position.x,position.y};
    }
    restorePositions_ = false;
    ImNodes::SetCurrentContext(previous);
}

bool EditorMaterialGraph::draw(const char* targetName)
{
    if (!open) { return false; }
    ImGui::SetNextWindowSize({1120,700}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Material Graph", &open)) { ImGui::End(); return false; }
    bool apply = false;
    ImGui::InputText("Graph path", path_.data(),path_.size());
    if (ImGui::Button("Save graph")) { save(path_.data()); }
    ImGui::SameLine();
    if (ImGui::Button("Load graph")) { if (dirty_) { ImGui::OpenPopup("Replace unsaved graph?"); } else { load(path_.data()); } }
    if (ImGui::BeginPopupModal("Replace unsaved graph?",nullptr,ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("The current graph has unsaved changes.");
        if (ImGui::Button("Load and replace")) { load(path_.data()); ImGui::CloseCurrentPopup(); }
        ImGui::SameLine(); if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Export assets")) { auto path=std::filesystem::path(path_.data()); path.replace_extension(".materialdef"); exportAssets(path); }
    ImGui::SameLine(); if (ImGui::Button("Compile")) { compile(); }
    ImGui::SameLine(); ImGui::BeginDisabled(targetName == nullptr);
    if (ImGui::Button("Apply to selected material") && compile()) { apply = true; }
    ImGui::EndDisabled();
    ImGui::Text("Target: %s%s", targetName ? targetName : "select one material in the Scene Browser", dirty_ ? " | graph modified" : "");
    ImGui::Checkbox("Slang SDK expression profile", &textMode_);
    if (textMode_) {
        ImGui::TextWrapped("Read-only subset: let, float4, Material and SDK functions. No arbitrary Slang modules. Parameter defaults come from graph Parameter nodes.");
        auto sourcePath=std::filesystem::path(path_.data()); sourcePath.replace_extension(".material.slang");
        if (ImGui::Button("Save SDK source")) {
            try { writeFile(sourcePath,source_.data()); status_="Saved "+sourcePath.string(); }
            catch (const std::exception& error) { status_=error.what(); }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load SDK source")) { ImGui::OpenPopup("Replace SDK source?"); }
        if (ImGui::BeginPopupModal("Replace SDK source?",nullptr,ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("Replace the text buffer with %s?",sourcePath.string().c_str());
            if (ImGui::Button("Load source")) {
                try {
                    std::ifstream file(sourcePath,std::ios::binary);
                    if (!file || std::filesystem::file_size(sourcePath)>source_.size()-1) { throw std::runtime_error("Cannot read SDK source (16 KiB limit)"); }
                    const std::string source{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
                    source_.fill(0); std::copy(source.begin(),source.end(),source_.begin()); status_="Loaded "+sourcePath.string();
                } catch (const std::exception& error) { status_=error.what(); }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine(); if (ImGui::Button("Cancel")) { ImGui::CloseCurrentPopup(); }
            ImGui::EndPopup();
        }
        ImGui::InputTextMultiline("##SDK",source_.data(),source_.size(),{-1,ImGui::GetContentRegionAvail().y * .65f});
    } else {
        ImGui::BeginDisabled(undo_.empty());
        if (ImGui::Button("Undo graph")) { redo_.push_back(graph_); replaceGraph(undo_.back()); undo_.pop_back(); dirty_=true; }
        ImGui::EndDisabled(); ImGui::SameLine(); ImGui::BeginDisabled(redo_.empty());
        if (ImGui::Button("Redo graph")) { undo_.push_back(graph_); replaceGraph(redo_.back()); redo_.pop_back(); dirty_=true; }
        ImGui::EndDisabled(); ImGui::SameLine();
        const auto before = graph_;
        ImGui::BeginDisabled(graph_["nodes"].size() >= 128);
        if (ImGui::BeginCombo("Add node", "Choose node")) {
            for (const auto& kind : material::materialGraphNodes()) {
                if (ImGui::Selectable(kind.kind.c_str())) {
                    int id = 1; for (const auto& node : graph_["nodes"]) { id = std::max(id,node["id"].get<int>()+1); }
                    if (id <= 100000) { auto node=material::makeMaterialGraphNode(kind.kind,id); node["position"]={120,120}; graph_["nodes"].push_back(node); selected_=id; restorePositions_=true; }
                }
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        const float height = std::max(180.0f,ImGui::GetContentRegionAvail().y - 115.0f);
        ImGui::BeginChild("GraphCanvas",{ImGui::GetContentRegionAvail().x * .64f,height},true); canvas(); ImGui::EndChild();
        ImGui::SameLine(); ImGui::BeginChild("NodeProperties",{0,height},true); properties(); ImGui::EndChild();
        if (before != graph_) {
            if (!editStart_) { editStart_=before; }
            redo_.clear(); dirty_=true;
        }
        if (editStart_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsAnyItemActive()) {
            undo_.push_back(std::move(*editStart_)); editStart_.reset();
            if (undo_.size()>128) { undo_.erase(undo_.begin()); }
        }
    }
    ImGui::TextWrapped("%s", status_.c_str());
    if (ImGui::CollapsingHeader("Last successful compile: resource / feature reflection")) { ImGui::TextWrapped("%s", compiled_.reflection.dump(2).c_str()); }
    ImGui::End(); return apply;
}
} // namespace metallic
