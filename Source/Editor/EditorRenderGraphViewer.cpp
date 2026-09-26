#include "Editor/EditorRenderGraphViewer.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

namespace metallic::editor {
namespace {
using namespace render;
using Snapshot = RenderGraphExecutionSnapshot;
using Resource = RenderGraphExecutionResourceSnapshot;
using Pass = RenderGraphExecutionPassSnapshot;
using Barrier = RenderGraphExecutionBarrierSnapshot;

constexpr ImU32 kRead = IM_COL32(157, 203, 73, 255);
constexpr ImU32 kWrite = IM_COL32(244, 104, 83, 255);
constexpr ImU32 kBarrier = IM_COL32(255, 202, 84, 255);
constexpr ImU32 kLayout = IM_COL32(104, 207, 230, 255);
constexpr ImU32 kGraphics = IM_COL32(110, 170, 235, 255);
constexpr ImU32 kCompute = IM_COL32(177, 137, 233, 255);
constexpr ImU32 kCopy = IM_COL32(90, 192, 160, 255);

const char* queueName(QueueType type)
{
    switch (type) {
    case QueueType::Graphics: return "Graphics";
    case QueueType::Compute: return "Compute";
    case QueueType::Copy: return "Copy";
    }
    return "Unknown";
}

ImU32 queueColor(QueueType type)
{
    return type == QueueType::Compute ? kCompute : type == QueueType::Copy ? kCopy : kGraphics;
}

const char* stateName(ResourceState state)
{
    switch (state) {
    case ResourceState::Undefined: return "Undefined";
    case ResourceState::Present: return "Present";
    case ResourceState::General: return "General";
    case ResourceState::ShaderRead: return "Shader read";
    case ResourceState::ColorAttachment: return "Color attachment";
    case ResourceState::DepthStencilAttachment: return "Depth/stencil";
    case ResourceState::TransferSource: return "Transfer source";
    case ResourceState::TransferDestination: return "Transfer destination";
    case ResourceState::IndirectArgument: return "Indirect argument";
    case ResourceState::DecompressionSource: return "Decompression source";
    case ResourceState::DecompressionDestination: return "Decompression destination";
    default: return "Other";
    }
}

const char* statusName(RenderGraphExecutionSnapshotStatus status)
{
    switch (status) {
    case RenderGraphExecutionSnapshotStatus::Planned: return "Planned";
    case RenderGraphExecutionSnapshotStatus::Recording: return "Recording";
    case RenderGraphExecutionSnapshotStatus::Recorded: return "Recorded / caller submits";
    case RenderGraphExecutionSnapshotStatus::Submitted: return "Submitted";
    case RenderGraphExecutionSnapshotStatus::Failed: return "Failed / partial capture";
    }
    return "Unknown";
}

const char* roleName(RenderGraphSegmentRole role)
{
    switch (role) {
    case RenderGraphSegmentRole::Pass: return "Pass";
    case RenderGraphSegmentRole::Prologue: return "Prologue";
    case RenderGraphSegmentRole::ComputeBranch: return "Compute branch";
    case RenderGraphSegmentRole::GraphicsBranch: return "Graphics branch";
    case RenderGraphSegmentRole::Join: return "Join";
    case RenderGraphSegmentRole::Epilogue: return "Epilogue";
    }
    return "Segment";
}

bool matches(std::string_view text, std::string_view filter)
{
    return filter.empty() || std::search(text.begin(), text.end(), filter.begin(), filter.end(),
        [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) ==
            std::tolower(static_cast<unsigned char>(b)); }) != text.end();
}

std::string bytesText(uint64_t bytes)
{
    char text[64];
    if (bytes >= 1024ull * 1024) { std::snprintf(text, sizeof(text), "%.2f MiB", double(bytes) / (1024 * 1024)); }
    else if (bytes >= 1024) { std::snprintf(text, sizeof(text), "%.1f KiB", double(bytes) / 1024); }
    else { std::snprintf(text, sizeof(text), "%llu B", static_cast<unsigned long long>(bytes)); }
    return text;
}

const Resource* findResource(const Snapshot& snapshot, uint64_t id)
{
    const auto found = std::find_if(snapshot.resources.begin(), snapshot.resources.end(),
        [&](const auto& value) { return value.id == id; });
    return found == snapshot.resources.end() ? nullptr : &*found;
}

const Pass* findPass(const Snapshot& snapshot, uint32_t id)
{
    const auto found = std::find_if(snapshot.passes.begin(), snapshot.passes.end(),
        [&](const auto& value) { return value.id == id; });
    return found == snapshot.passes.end() ? nullptr : &*found;
}

struct Cell {
    bool read = false, write = false, layout = false;
    uint32_t barriers = 0;
};

Cell cellFor(const Pass& pass, uint64_t resource)
{
    Cell cell;
    const auto accumulate = [&](const auto& uses, const auto& barriers) {
        for (const auto& use : uses) {
            if (use.resourceId == resource) { cell.read |= use.reads; cell.write |= use.writes; }
        }
        for (const auto& barrier : barriers) {
            if (barrier.resourceId != resource) { continue; }
            ++cell.barriers;
            cell.layout |= barrier.before != barrier.after;
        }
    };
    accumulate(pass.uses, pass.barriers);
    for (const auto& stage : pass.stages) { accumulate(stage.uses, stage.barriers); }
    return cell;
}

template<class Bits>
std::string bitNames(Bits value, std::initializer_list<std::pair<Bits, const char*>> names)
{
    std::string text;
    auto remaining = uint64_t(value);
    for (const auto& [bit, name] : names) {
        if ((remaining & uint64_t(bit)) == 0) { continue; }
        if (!text.empty()) { text += " | "; }
        text += name; remaining &= ~uint64_t(bit);
    }
    if (remaining) { text += " | Other bits"; }
    return text.empty() ? "None" : text;
}

void scopeText(const char* label, SyncScope scope)
{
    using S = PipelineStageBits;
    using A = AccessBits;
    const auto stages = bitNames<S>(scope.stages, {{S::TopOfPipe, "Top"}, {S::DrawIndirect, "Indirect"},
        {S::VertexShader, "Vertex"}, {S::FragmentShader, "Fragment"}, {S::ComputeShader, "Compute"},
        {S::ColorAttachment, "Color"}, {S::Transfer, "Transfer"}, {S::BottomOfPipe, "Bottom"},
        {S::AllCommands, "All commands"}, {S::DepthStencil, "Depth/stencil"}, {S::PreRasterization, "Pre-raster"},
        {S::AccelerationStructureBuild, "AS build"}, {S::RayTracingShader, "Ray tracing"},
        {S::MemoryDecompression, "Decompression"}, {S::Host, "Host"}});
    const auto access = bitNames<A>(scope.access, {{A::ShaderRead, "Shader R"}, {A::ShaderWrite, "Shader W"},
        {A::UniformRead, "Uniform R"}, {A::IndirectRead, "Indirect R"}, {A::TransferRead, "Transfer R"},
        {A::TransferWrite, "Transfer W"}, {A::ColorRead, "Color R"}, {A::ColorWrite, "Color W"},
        {A::DepthStencilRead, "Depth R"}, {A::DepthStencilWrite, "Depth W"},
        {A::AccelerationStructureRead, "AS R"}, {A::AccelerationStructureWrite, "AS W"},
        {A::DecompressionRead, "Decompression R"}, {A::DecompressionWrite, "Decompression W"},
        {A::DescriptorRead, "Descriptor R"}, {A::HostRead, "Host R"}, {A::HostWrite, "Host W"},
        {A::MemoryRead, "Memory R"}, {A::MemoryWrite, "Memory W"}});
    ImGui::TextWrapped("%s: %s / %s", label, stages.c_str(), access.c_str());
}

void synchronizationText(const SynchronizationStats& stats)
{
    ImGui::TextWrapped("Encoded boundary: %llu memory barriers / %llu image transitions (%llu calls)",
        static_cast<unsigned long long>(stats.memoryBarriers), static_cast<unsigned long long>(stats.imageTransitions),
        static_cast<unsigned long long>(stats.calls));
}

void barrierDetails(const Barrier& barrier)
{
    ImGui::Text("%s -> %s%s", stateName(barrier.before), stateName(barrier.after),
        barrier.executionOnly ? " (execution only)" : "");
    scopeText("From", barrier.beforeScope);
    scopeText("To  ", barrier.afterScope);
}

void swatch(ImU32 color, const char* label)
{
    const auto p = ImGui::GetCursorScreenPos();
    const float size = ImGui::GetTextLineHeight();
    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + size, p.y + size), color, 2);
    ImGui::Dummy(ImVec2(size, size)); ImGui::SameLine(0, 5); ImGui::TextUnformatted(label);
}

// Build text inside the current visible area before rotating its vertices. The
// final draw command retains the header clip, including the frozen name column.
void angledText(ImDrawList& draw, ImVec2 origin, const char* text, ImVec2 clipMin, ImVec2 clipMax)
{
    draw.PushClipRect(clipMin, clipMax, false);
    const ImVec2 base(clipMin.x + 1, clipMin.y + 1);
    const int first = draw.VtxBuffer.Size;
    draw.AddText(base, ImGui::GetColorU32(ImGuiCol_Text), text);
    constexpr float c = 0.5f, s = -0.8660254f;
    for (int i = first; i < draw.VtxBuffer.Size; ++i) {
        const auto p = draw.VtxBuffer[i].pos;
        const float x = p.x - base.x, y = p.y - base.y;
        draw.VtxBuffer[i].pos = ImVec2(origin.x + c * x - s * y, origin.y + s * x + c * y);
    }
    draw.PopClipRect();
}

bool overlaps(const Resource& a, const Resource& b)
{
    if (!a.memory.known || !b.memory.known || a.memory.allocationId == b.memory.allocationId ||
        a.memory.memoryBlockId != b.memory.memoryBlockId) { return false; }
    if (!a.memory.sizeBytes || !b.memory.sizeBytes) { return false; }
    return a.memory.offsetBytes <= b.memory.offsetBytes ?
        b.memory.offsetBytes - a.memory.offsetBytes < a.memory.sizeBytes :
        a.memory.offsetBytes - b.memory.offsetBytes < b.memory.sizeBytes;
}

std::string captureJson(const Snapshot& snapshot)
{
    using Json = RenderGraphProperties;
    Json value{{"graph", snapshot.graphName}, {"generation", snapshot.graphGeneration},
        {"execution", snapshot.executionId}, {"status", statusName(snapshot.status)},
        {"success", snapshot.success}, {"pipelinedSubmission", snapshot.pipelinedSubmission},
        {"externalRecording", snapshot.externalRecording}, {"scope", "Declared graph and captured internal stages; opaque internals excluded"},
        {"resources", Json::array()}, {"passes", Json::array()}, {"queues", Json::array()},
        {"segments", Json::array()}, {"batches", Json::array()}};
    for (const auto& resource : snapshot.resources) {
        const auto& m = resource.memory;
        value["resources"].push_back({{"id", resource.id}, {"name", resource.name}, {"aliases", resource.aliases},
            {"private", resource.privateResource}, {"buffer", resource.type == RenderGraphResourceType::Buffer},
            {"allocation", m.allocationId}, {"memoryBlock", std::to_string(m.memoryBlockId)}, {"memoryKnown", m.known},
            {"offsetBytes", m.offsetBytes}, {"sizeBytes", m.sizeBytes}, {"memoryType", m.memoryTypeIndex}, {"heap", m.heapIndex}});
        auto& exported = value["resources"].back();
        if (resource.type == RenderGraphResourceType::Buffer) { exported["logicalBytes"] = resource.bufferDesc.size; }
        else {
            const auto& desc = resource.textureDesc;
            exported["texture"] = {{"width", desc.width}, {"height", desc.height}, {"depth", desc.depth},
                {"format", uint32_t(desc.format)}, {"mips", desc.mipCount}, {"layers", desc.layerCount}};
        }
    }
    const auto usesJson = [](const auto& uses) {
        auto result = Json::array();
        for (const auto& use : uses) { result.push_back({{"resource", use.resourceId}, {"state", stateName(use.state)},
            {"reads", use.reads}, {"writes", use.writes}, {"exclusive", use.exclusive},
            {"stages", uint64_t(use.scope.stages)}, {"access", uint64_t(use.scope.access)}}); }
        return result;
    };
    const auto barriersJson = [](const auto& barriers) {
        auto result = Json::array();
        for (const auto& b : barriers) { result.push_back({{"resource", b.resourceId}, {"before", stateName(b.before)},
            {"after", stateName(b.after)}, {"executionOnly", b.executionOnly},
            {"sourceStages", uint64_t(b.beforeScope.stages)}, {"sourceAccess", uint64_t(b.beforeScope.access)},
            {"destinationStages", uint64_t(b.afterScope.stages)}, {"destinationAccess", uint64_t(b.afterScope.access)}}); }
        return result;
    };
    const auto synchronizationJson = [](const SynchronizationStats& stats) {
        return Json{{"calls", stats.calls}, {"memoryBarriers", stats.memoryBarriers},
            {"imageTransitions", stats.imageTransitions}, {"coalescedResources", stats.coalescedResources}};
    };
    for (const auto& pass : snapshot.passes) {
        Json p{{"id", pass.id}, {"name", pass.name}, {"type", pass.type}, {"logicalQueue", queueName(pass.logicalQueue)},
            {"actualQueue", pass.actualQueueId}, {"recorded", pass.recorded}, {"predecessors", pass.predecessors},
            {"uses", usesJson(pass.uses)}, {"barriers", barriersJson(pass.barriers)},
            {"encodedBoundary", synchronizationJson(pass.synchronization)}, {"stages", Json::array()}};
        for (const auto& stage : pass.stages) { p["stages"].push_back({{"name", stage.name}, {"recorded", stage.recorded},
            {"restoreBoundary", stage.restoreBoundary}, {"allowsFork", stage.allowParallelCompute},
            {"uses", usesJson(stage.uses)}, {"barriers", barriersJson(stage.barriers)},
            {"encodedBoundary", synchronizationJson(stage.synchronization)}}); }
        value["passes"].push_back(std::move(p));
    }
    for (const auto& q : snapshot.queues) { value["queues"].push_back({{"id", q.id}, {"type", queueName(q.type)}}); }
    for (const auto& s : snapshot.segments) { value["segments"].push_back({{"id", s.id}, {"pass", s.passId},
        {"queue", s.queueId}, {"role", roleName(s.role)}, {"predecessors", s.predecessors},
        {"recorded", s.recorded}, {"accepted", s.accepted}, {"completionKnown", s.completionKnown}, {"completed", s.completed}}); }
    for (const auto& b : snapshot.batches) { value["batches"].push_back({{"id", b.id}, {"queue", b.queueId},
        {"segments", b.segmentIds}, {"waitPredecessors", b.waitPredecessors}, {"externalWaitCount", b.externalWaitCount},
        {"semaphoreWaitCount", b.semaphoreWaitCount}, {"waitDetailsComplete", b.waitDetailsComplete}, {"accepted", b.accepted}}); }
    return value.dump(2);
}

} // namespace

void RenderGraphExecutionViewer::update(std::shared_ptr<const Snapshot> snapshot)
{
    if (!snapshot || (!live_ && !captureNext_)) { return; }
    if (captureNext_ && snapshot_ && snapshot->executionId == snapshot_->executionId &&
        snapshot->graphGeneration == snapshot_->graphGeneration) { return; }
    const bool changed = !snapshot_ || snapshot_->graphGeneration != snapshot->graphGeneration;
    const auto* previousResource = snapshot_ ? findResource(*snapshot_, selectedResourceId_) : nullptr;
    const auto selectedName = previousResource ? previousResource->name : std::string();
    snapshot_ = std::move(snapshot);
    captureNext_ = false;
    if (changed || !findPass(*snapshot_, selectedPassId_)) { selectedPassId_ = UINT32_MAX; }
    if (changed || !findResource(*snapshot_, selectedResourceId_)) {
        selectedResourceId_ = 0;
        if (!changed && !selectedName.empty()) {
            for (const auto& resource : snapshot_->resources) {
                if (resource.name == selectedName) { selectedResourceId_ = resource.id; break; }
            }
        }
    }
}

void RenderGraphExecutionViewer::draw(float scale)
{
    ImGui::Checkbox("Live", &live_);
    ImGui::SameLine();
    if (ImGui::Button("Capture next")) { requestCapture(); }
    ImGui::SameLine();
    if (snapshot_ && ImGui::Button("Copy capture JSON")) { ImGui::SetClipboardText(captureJson(*snapshot_).c_str()); }
    if (captureNext_) { ImGui::SameLine(); ImGui::TextDisabled("Waiting for next execution..."); }
    if (!snapshot_) {
        ImGui::Spacing();
        ImGui::TextWrapped("No execution captured. Render the viewport with this tab open to inspect its actual resource plan and submissions.");
        return;
    }
    ImGui::Text("%s  |  generation %llu / execution %llu  |  %s%s", snapshot_->graphName.c_str(),
        static_cast<unsigned long long>(snapshot_->graphGeneration), static_cast<unsigned long long>(snapshot_->executionId),
        statusName(snapshot_->status), live_ ? "" : "  (frozen)");
    ImGui::SetNextItemWidth(190 * scale);
    ImGui::InputTextWithHint("##ResourceSearch", "Search resource / alias", resourceFilter_, sizeof(resourceFilter_));
    ImGui::SameLine(); ImGui::SetNextItemWidth(170 * scale);
    ImGui::InputTextWithHint("##PassSearch", "Search pass", passFilter_, sizeof(passFilter_));
    ImGui::SameLine(); ImGui::SetNextItemWidth(135 * scale);
    ImGui::Combo("##ResourceType", &resourceType_, "All resources\0Textures\0Buffers\0Private imports\0");
    ImGui::Separator();

    const auto available = ImGui::GetContentRegionAvail();
    const bool beside = available.x >= 850 * scale;
    const float inspector = std::clamp(available.x * 0.28f, 260 * scale, 340 * scale);
    ImGui::BeginChild("ExecutionViews", ImVec2(beside ? available.x - inspector - ImGui::GetStyle().ItemSpacing.x : 0,
        beside ? 0 : std::max(180 * scale, available.y * 0.65f)), false);
    if (ImGui::BeginTabBar("ExecutionViewTabs")) {
        const Tab requestedTab = tab_;
        const auto item = [&](const char* label, Tab tab) {
            const auto flags = selectTab_ && requestedTab == tab ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            if (!ImGui::BeginTabItem(label, nullptr, flags)) { return false; }
            tab_ = tab; return true;
        };
        if (item("Resources", Tab::Resources)) { drawResources(scale); ImGui::EndTabItem(); }
        if (item("Queues", Tab::Queues)) { drawQueues(scale); ImGui::EndTabItem(); }
        if (item("Memory", Tab::Memory)) { drawMemory(scale); ImGui::EndTabItem(); }
        ImGui::EndTabBar();
        selectTab_ = false;
    }
    ImGui::EndChild();
    if (beside) { ImGui::SameLine(); }
    ImGui::BeginChild("ExecutionInspector", ImVec2(0, 0), true);
    drawInspector();
    ImGui::EndChild();
}

void RenderGraphExecutionViewer::drawResources(float scale)
{
    std::vector<const Pass*> passes;
    for (const auto& pass : snapshot_->passes) {
        if (matches(pass.name, passFilter_) || matches(pass.type, passFilter_)) { passes.push_back(&pass); }
    }
    swatch(kRead, "Read"); ImGui::SameLine(); swatch(kWrite, "Write");
    ImGui::SameLine(); swatch(kBarrier, "Barrier"); ImGui::SameLine(); swatch(kLayout, "Layout");
    ImGui::Checkbox("Show barriers", &showBarriers_); ImGui::SameLine(); ImGui::Checkbox("Used only", &onlyUsed_);
    ImGui::SameLine(); ImGui::SetNextItemWidth(100 * scale); ImGui::SliderFloat("Columns", &columnWidth_, 32, 120, "%.0f");
    ImGui::TextDisabled("Pass order, not GPU time. Gray span = declared use interval, not allocation lifetime.");
    if (passes.empty()) { ImGui::TextDisabled("No passes match the filter."); return; }
    // ImGui tables support up to 512 columns; make truncation explicit.
    if (passes.size() > 500) { passes.resize(500); ImGui::TextDisabled("Showing first 500 matching passes. Narrow the search."); }
    const float nameWidth = 238 * scale, cellWidth = columnWidth_ * scale;
    const auto min = ImGui::GetCursorScreenPos();
    const auto available = ImGui::GetContentRegionAvail();
    if (!ImGui::BeginTable("ResourcePassMatrix", int(passes.size() + 1), ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit, available)) { return; }
    ImGui::TableSetupColumn("Resource", ImGuiTableColumnFlags_WidthFixed, nameWidth);
    for (const auto* pass : passes) { ImGui::TableSetupColumn(pass->name.c_str(), ImGuiTableColumnFlags_WidthFixed, cellWidth); }
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers, 140 * scale);
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%zu resources", snapshot_->resources.size());
    ImGui::Text("%zu passes", passes.size());
    ImGui::TextDisabled("Click a cell to inspect");
    for (size_t i = 0; i < passes.size(); ++i) {
        if (!ImGui::TableSetColumnIndex(int(i + 1))) { continue; }
        const auto& pass = *passes[i];
        ImGui::PushID(int(pass.id));
        const auto p = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("Pass", ImVec2(cellWidth, 132 * scale));
        auto* draw = ImGui::GetWindowDrawList();
        if (selectedPassId_ == pass.id) {
            draw->AddRectFilled(p, ImVec2(p.x + cellWidth, p.y + 132 * scale), IM_COL32(90, 105, 120, 90));
        }
        std::string label = pass.name.size() > 25 ? pass.name.substr(0, 23) + ".." : pass.name;
        angledText(*draw, ImVec2(p.x + 4 * scale, p.y + 119 * scale), label.c_str(),
            ImVec2(min.x + nameWidth + 8 * scale, min.y), ImVec2(min.x + available.x, min.y + 140 * scale));
        draw->AddRectFilled(ImVec2(p.x, p.y + 128 * scale), ImVec2(p.x + cellWidth, p.y + 133 * scale), queueColor(pass.logicalQueue));
        if (ImGui::IsItemClicked()) { selectedPassId_ = pass.id; }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip(); ImGui::TextUnformatted(pass.name.c_str()); ImGui::TextUnformatted(pass.type.c_str());
            ImGui::Text("%s; %s", queueName(pass.logicalQueue), pass.recorded ? "recorded" : "planned");
            if (pass.actualQueueId != UINT32_MAX) { ImGui::Text("Actual queue Q%u", pass.actualQueueId); }
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }
    for (const auto& resource : snapshot_->resources) {
        bool match = matches(resource.name, resourceFilter_);
        for (const auto& alias : resource.aliases) { match |= matches(alias, resourceFilter_); }
        if (!match || (resourceType_ == 1 && resource.type != RenderGraphResourceType::Texture2D) ||
            (resourceType_ == 2 && resource.type != RenderGraphResourceType::Buffer) ||
            (resourceType_ == 3 && !resource.privateResource)) { continue; }
        std::vector<Cell> cells;
        size_t first = passes.size(), last = 0;
        for (size_t i = 0; i < passes.size(); ++i) {
            const auto cell = cellFor(*passes[i], resource.id); cells.push_back(cell);
            if (cell.read || cell.write || cell.barriers) { first = std::min(first, i); last = i; }
        }
        if (onlyUsed_ && first == passes.size()) { continue; }
        ImGui::PushID(resource.name.c_str());
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 25 * scale);
        ImGui::TableSetColumnIndex(0);
        const auto label = std::string(resource.type == RenderGraphResourceType::Buffer ? "B  " : "T  ") + resource.name;
        if (ImGui::Selectable(label.c_str(), selectedResourceId_ == resource.id, ImGuiSelectableFlags_None,
            ImVec2(nameWidth - 10 * scale, 21 * scale))) { selectedResourceId_ = resource.id; }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip(); ImGui::TextUnformatted(resource.name.c_str());
            ImGui::Text("%s | %s", resource.privateResource ? "Private import" : "Graph resource",
                resource.memory.known ? bytesText(resource.memory.sizeBytes).c_str() : "Memory unknown");
            for (const auto& alias : resource.aliases) { ImGui::TextUnformatted(alias.c_str()); }
            ImGui::EndTooltip();
        }
        for (size_t i = 0; i < passes.size(); ++i) {
            if (!ImGui::TableSetColumnIndex(int(i + 1))) { continue; }
            ImGui::PushID(int(i));
            const auto p = ImGui::GetCursorScreenPos();
            const ImVec2 end(p.x + cellWidth, p.y + 21 * scale);
            ImGui::InvisibleButton("Use", ImVec2(cellWidth, 21 * scale));
            auto* draw = ImGui::GetWindowDrawList();
            const auto cell = cells[i];
            if (i >= first && i <= last) { draw->AddRectFilled(p, end, IM_COL32(86, 89, 94, 160)); }
            if (cell.read || cell.write) {
                if (cell.read && cell.write) {
                    draw->AddTriangleFilled(p, ImVec2(end.x, p.y), ImVec2(p.x, end.y), kRead);
                    draw->AddTriangleFilled(ImVec2(end.x, p.y), end, ImVec2(p.x, end.y), kWrite);
                } else { draw->AddRectFilled(p, end, cell.write ? kWrite : kRead); }
                draw->AddText(ImVec2(p.x + cellWidth * 0.5f - (cell.read && cell.write ? 9 : 4), p.y + 2 * scale),
                    IM_COL32(20, 25, 30, 255), cell.read && cell.write ? "RW" : cell.write ? "W" : "R");
            }
            if (showBarriers_ && cell.barriers) {
                const float x = p.x + 4 * scale, y = p.y + 10 * scale, r = 4 * scale;
                draw->AddQuadFilled(ImVec2(x, y-r), ImVec2(x+r, y), ImVec2(x, y+r), ImVec2(x-r, y), cell.layout ? kLayout : kBarrier);
            }
            if (selectedPassId_ == passes[i]->id && selectedResourceId_ == resource.id) {
                draw->AddRect(p, end, IM_COL32(255, 255, 255, 255), 0.0f, 2.0f, ImDrawFlags_None);
            }
            if (ImGui::IsItemClicked()) { selectedResourceId_ = resource.id; selectedPassId_ = passes[i]->id; }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip(); ImGui::Text("%s / %s", resource.name.c_str(), passes[i]->name.c_str());
                ImGui::Text("%s | %u planned barrier boundaries", cell.read && cell.write ? "Read + write" :
                    cell.write ? "Write" : cell.read ? "Read" : "No declared data access", cell.barriers);
                ImGui::TextDisabled("Includes boundary declarations and captured internal stages.");
                ImGui::EndTooltip();
            }
            ImGui::PopID();
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void RenderGraphExecutionViewer::drawQueues(float scale)
{
    ImGui::TextWrapped("Submission topology. Horizontal distance is dependency depth, not elapsed GPU time.");
    ImGui::TextDisabled("Solid gold arrows: submitted cross-queue waits. Dashed gray: same-queue / planned dependencies.");
    ImGui::TextDisabled("Wait counts cover explicit executor waits; opaque command dependencies are not expanded.");
    if (snapshot_->externalRecording || snapshot_->segments.empty()) {
        ImGui::TextWrapped("Submission details are unavailable for caller-owned external recording. The Resources view still contains its access plan.");
        return;
    }
    std::unordered_map<uint32_t, size_t> lanes;
    for (size_t i = 0; i < snapshot_->queues.size(); ++i) { lanes[snapshot_->queues[i].id] = i; }
    std::unordered_map<uint32_t, uint32_t> depth, lastInQueue;
    uint32_t maximum = 0;
    for (const auto& segment : snapshot_->segments) {
        uint32_t value = 0;
        for (const auto predecessor : segment.predecessors) {
            if (depth.contains(predecessor)) { value = std::max(value, depth[predecessor] + 1); }
        }
        if (lastInQueue.contains(segment.queueId)) { value = std::max(value, lastInQueue[segment.queueId] + 1); }
        depth[segment.id] = value; lastInQueue[segment.queueId] = value; maximum = std::max(maximum, value);
    }
    ImGui::Checkbox("Fit topology", &fitQueues_); ImGui::SameLine();
    ImGui::BeginDisabled(fitQueues_);
    ImGui::SetNextItemWidth(130 * scale); ImGui::SliderFloat("Zoom", &queueZoom_, 0.35f, 1.5f, "%.2fx");
    ImGui::EndDisabled();
    const float zoom = fitQueues_ ? std::clamp((ImGui::GetContentRegionAvail().x - 20 * scale) /
        ((115 + float(maximum + 1) * 152) * scale), 0.35f, 1.0f) : queueZoom_;
    const float diagramScale = scale * zoom;
    const float step = 152 * diagramScale, laneHeight = 108 * diagramScale, left = 115 * diagramScale;
    ImGui::BeginChild("QueueTopology", ImVec2(0, std::min(ImGui::GetContentRegionAvail().y * 0.60f,
        std::max(180 * diagramScale, float(lanes.size()) * laneHeight + 30 * diagramScale))), true, ImGuiWindowFlags_HorizontalScrollbar);
    const auto origin = ImGui::GetCursorScreenPos();
    auto* draw = ImGui::GetWindowDrawList();
    const float width = left + float(maximum + 1) * step;
    for (const auto& queue : snapshot_->queues) {
        const float y = origin.y + float(lanes[queue.id]) * laneHeight;
        char text[64]; std::snprintf(text, sizeof(text), "Q%u  %s", queue.id, queueName(queue.type));
        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, ImVec2(origin.x + 4 * diagramScale, y + 12 * diagramScale), queueColor(queue.type), text);
        draw->AddLine(ImVec2(origin.x + left, y + laneHeight - 15 * diagramScale), ImVec2(origin.x + width, y + laneHeight - 15 * diagramScale), IM_COL32(80, 83, 92, 130));
    }
    const auto position = [&](const auto& segment) {
        return ImVec2(origin.x + left + float(depth[segment.id]) * step,
            origin.y + float(lanes[segment.queueId]) * laneHeight + 10 * diagramScale);
    };
    std::unordered_map<uint32_t, const RenderGraphExecutionSegmentSnapshot*> segmentMap;
    for (const auto& segment : snapshot_->segments) { segmentMap[segment.id] = &segment; }
    std::set<std::pair<uint32_t, uint32_t>> actualWaits;
    for (const auto& batch : snapshot_->batches) {
        if (!batch.accepted || batch.segmentIds.empty()) { continue; }
        for (const auto source : batch.waitPredecessors) { actualWaits.emplace(source, batch.segmentIds.front()); }
    }
    const auto arrow = [&](uint32_t source, uint32_t target, bool actual) {
        if (!segmentMap.contains(source) || !segmentMap.contains(target)) { return; }
        const auto a = position(*segmentMap[source]), b = position(*segmentMap[target]);
        const ImVec2 from(a.x + 130 * diagramScale, a.y + 31 * diagramScale), to(b.x - 2 * diagramScale, b.y + 31 * diagramScale);
        const ImU32 color = actual ? kBarrier : IM_COL32(143, 151, 161, 130);
        if (actual) { draw->AddLine(from, to, color, 2 * diagramScale); }
        else {
            for (int i = 0; i < 12; i += 2) {
                const float a = float(i) / 12, b = float(i + 1) / 12;
                draw->AddLine(ImVec2(from.x + (to.x-from.x)*a, from.y + (to.y-from.y)*a),
                    ImVec2(from.x + (to.x-from.x)*b, from.y + (to.y-from.y)*b), color);
            }
        }
        draw->AddTriangleFilled(to, ImVec2(to.x - 6*diagramScale, to.y - 4*diagramScale), ImVec2(to.x - 6*diagramScale, to.y + 4*diagramScale), color);
    };
    for (const auto& segment : snapshot_->segments) {
        for (const auto source : segment.predecessors) {
            if (!actualWaits.contains({source, segment.id})) { arrow(source, segment.id, false); }
        }
    }
    for (const auto [source, target] : actualWaits) { arrow(source, target, true); }
    for (const auto& segment : snapshot_->segments) {
        const auto p = position(segment);
        const auto* pass = findPass(*snapshot_, segment.passId);
        const bool faded = pass && !matches(pass->name, passFilter_);
        const auto queue = std::find_if(snapshot_->queues.begin(), snapshot_->queues.end(), [&](const auto& q) { return q.id == segment.queueId; });
        const ImU32 color = queueColor(queue == snapshot_->queues.end() ? QueueType::Graphics : queue->type);
        draw->AddRectFilled(p, ImVec2(p.x + 132 * diagramScale, p.y + 70 * diagramScale), faded ? IM_COL32(50, 52, 57, 100) : IM_COL32(44, 48, 57, 255), 4 * diagramScale);
        draw->AddRect(p, ImVec2(p.x + 132 * diagramScale, p.y + 70 * diagramScale), selectedPassId_ == segment.passId ? IM_COL32(245, 245, 245, 255) : color, 4 * diagramScale);
        char label[80]; std::snprintf(label, sizeof(label), "S%u  %s", segment.id, roleName(segment.role));
        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, ImVec2(p.x + 6 * diagramScale, p.y + 5 * diagramScale), color, label);
        const auto name = pass ? pass->name.substr(0, 17) : std::string("Frame boundary");
        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, ImVec2(p.x + 6 * diagramScale, p.y + 26 * diagramScale), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, ImVec2(p.x + 6 * diagramScale, p.y + 47 * diagramScale), ImGui::GetColorU32(ImGuiCol_TextDisabled),
            segment.completed ? "GPU complete" : segment.accepted ? "Accepted" : segment.recorded ? "Recorded" : "Planned");
        ImGui::SetCursorScreenPos(p); ImGui::PushID(int(segment.id));
        ImGui::InvisibleButton("Segment", ImVec2(132 * diagramScale, 70 * diagramScale));
        if (ImGui::IsItemClicked() && pass) { selectedPassId_ = pass->id; }
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip(); ImGui::Text("S%u  %s", segment.id, roleName(segment.role));
            if (pass) { ImGui::TextUnformatted(pass->name.c_str()); }
            ImGui::Text("Q%u | %zu predecessors", segment.queueId, segment.predecessors.size());
            if (!segment.completionKnown) { ImGui::TextDisabled("GPU completion unavailable in this capture."); }
            ImGui::EndTooltip();
        }
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(origin); ImGui::Dummy(ImVec2(width, float(lanes.size()) * laneHeight));
    ImGui::EndChild();
    if (ImGui::BeginTable("SubmissionBatches", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
        ImGuiTableFlags_ScrollY, ImVec2(0, 0))) {
        for (const char* label : {"Batch / queue", "Segments", "Cross-queue waits", "External waits", "Receipt"}) { ImGui::TableSetupColumn(label); }
        ImGui::TableHeadersRow();
        for (const auto& batch : snapshot_->batches) {
            ImGui::TableNextRow(); ImGui::TableNextColumn(); ImGui::Text("B%u / Q%u", batch.id, batch.queueId);
            ImGui::TableNextColumn();
            for (const auto id : batch.segmentIds) { ImGui::Text("S%u", id); ImGui::SameLine(); }
            ImGui::TableNextColumn();
            for (const auto id : batch.waitPredecessors) { ImGui::Text("S%u", id); ImGui::SameLine(); }
            ImGui::TableNextColumn(); ImGui::Text("%u (%u explicit waits)", batch.externalWaitCount, batch.semaphoreWaitCount);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(batch.accepted ? "Accepted" : "Not accepted");
        }
        ImGui::EndTable();
    }
}

void RenderGraphExecutionViewer::drawMemory(float scale)
{
    std::map<uint64_t, std::vector<const Resource*>> blocks;
    std::set<uint64_t> allocations;
    uint64_t bytes = 0; size_t unknown = 0, aliasPairs = 0;
    for (const auto& resource : snapshot_->resources) {
        if (!resource.memory.known) { ++unknown; continue; }
        if (!allocations.insert(resource.memory.allocationId).second) { continue; }
        bytes += resource.memory.sizeBytes;
        blocks[resource.memory.memoryBlockId].push_back(&resource);
    }
    for (const auto& [block, resources] : blocks) {
        for (size_t i = 0; i < resources.size(); ++i) {
            for (size_t j = i + 1; j < resources.size(); ++j) { aliasPairs += overlaps(*resources[i], *resources[j]); }
        }
    }
    ImGui::Text("%zu unique allocations | %s | %zu overlapping pairs", allocations.size(), bytesText(bytes).c_str(), aliasPairs);
    ImGui::TextWrapped("Actual memory block + byte ranges. Shared block does not imply aliasing; only overlapping ranges do.");
    ImGui::TextDisabled("Capture allocations only; excludes opaque subsystem memory and other in-flight slots. Unknown: %zu", unknown);
    if (aliasPairs == 0) { ImGui::TextDisabled("No physical memory aliasing observed. Graph field aliases share one resource."); }
    ImGui::BeginChild("MemoryBlocks", ImVec2(0, 0), true);
    size_t blockIndex = 0;
    for (auto& [block, resources] : blocks) {
        ++blockIndex;
        std::sort(resources.begin(), resources.end(), [](auto* a, auto* b) { return a->memory.offsetBytes < b->memory.offsetBytes; });
        uint64_t extent = 1;
        for (const auto* r : resources) { extent = std::max(extent, r->memory.offsetBytes + r->memory.sizeBytes); }
        bool visible = false;
        for (const auto* r : resources) {
            bool match = matches(r->name, resourceFilter_);
            for (const auto& alias : r->aliases) { match |= matches(alias, resourceFilter_); }
            visible |= match && (resourceType_ != 1 || r->type == RenderGraphResourceType::Texture2D) &&
                (resourceType_ != 2 || r->type == RenderGraphResourceType::Buffer) && (resourceType_ != 3 || r->privateResource);
        }
        if (!visible) { continue; }
        ImGui::PushID(int(blockIndex));
        ImGui::Text("Block %zu | heap %u / type %u | %zu allocations", blockIndex, resources.front()->memory.heapIndex,
            resources.front()->memory.memoryTypeIndex, resources.size());
        ImGui::TextDisabled("0 .. %s observed span (not total block capacity)", bytesText(extent).c_str());
        const float width = std::max(50.0f, ImGui::GetContentRegionAvail().x - 6 * scale);
        for (const auto* resource : resources) {
            const auto p = ImGui::GetCursorScreenPos();
            const float x0 = float(double(resource->memory.offsetBytes) / double(extent)) * width;
            const float x1 = float(double(resource->memory.offsetBytes + resource->memory.sizeBytes) / double(extent)) * width;
            bool alias = false;
            for (const auto* other : resources) { alias |= overlaps(*resource, *other); }
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(p, ImVec2(p.x + width, p.y + 26 * scale), IM_COL32(42, 45, 50, 255), 2);
            draw->AddRectFilled(ImVec2(p.x + x0, p.y), ImVec2(p.x + std::max(x1, x0 + 2), p.y + 26 * scale),
                alias ? IM_COL32(180, 71, 60, 255) : resource->type == RenderGraphResourceType::Buffer ? IM_COL32(73, 111, 151, 255) : IM_COL32(66, 126, 119, 255), 2);
            draw->AddText(ImVec2(p.x + 5 * scale, p.y + 5 * scale), IM_COL32(243, 244, 246, 255), resource->name.c_str());
            if (selectedResourceId_ == resource->id) { draw->AddRect(p, ImVec2(p.x + width, p.y + 26 * scale), kBarrier, 2.0f, 2.0f, ImDrawFlags_None); }
            ImGui::PushID(resource->name.c_str());
            ImGui::InvisibleButton("Allocation", ImVec2(width, 28 * scale));
            if (ImGui::IsItemClicked()) { selectedResourceId_ = resource->id; }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip(); ImGui::TextUnformatted(resource->name.c_str());
                ImGui::Text("Allocation #%llu | offset %llu | size %s", static_cast<unsigned long long>(resource->memory.allocationId),
                    static_cast<unsigned long long>(resource->memory.offsetBytes), bytesText(resource->memory.sizeBytes).c_str());
                ImGui::TextUnformatted(alias ? "Physical range overlaps another allocation." : "No overlapping allocation range.");
                ImGui::EndTooltip();
            }
            ImGui::PopID();
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::PopID();
    }
    ImGui::EndChild();
}

void RenderGraphExecutionViewer::drawInspector()
{
    ImGui::TextUnformatted("Selection"); ImGui::Separator();
    const auto* resource = findResource(*snapshot_, selectedResourceId_);
    const auto* pass = findPass(*snapshot_, selectedPassId_);
    if (!resource && !pass) { ImGui::TextWrapped("Select a resource, pass, matrix cell or queue segment to inspect its declarations and synchronization."); }
    if (resource) {
        ImGui::TextWrapped("%s", resource->name.c_str());
        ImGui::TextDisabled("%s / %s", resource->type == RenderGraphResourceType::Buffer ? "Buffer" : "Texture",
            resource->privateResource ? "Private import" : "Graph resource");
        if (resource->type == RenderGraphResourceType::Buffer) {
            ImGui::Text("Logical size: %s", bytesText(resource->bufferDesc.size).c_str());
        } else {
            ImGui::Text("%u x %u x %u | format %u", resource->textureDesc.width, resource->textureDesc.height,
                resource->textureDesc.depth, uint32_t(resource->textureDesc.format));
            ImGui::Text("%u mip(s), %u layer(s)", resource->textureDesc.mipCount, resource->textureDesc.layerCount);
        }
        if (resource->memory.known) {
            ImGui::Text("Allocation #%llu", static_cast<unsigned long long>(resource->memory.allocationId));
            ImGui::Text("%s at byte %llu", bytesText(resource->memory.sizeBytes).c_str(), static_cast<unsigned long long>(resource->memory.offsetBytes));
            ImGui::Text("Heap %u / memory type %u", resource->memory.heapIndex, resource->memory.memoryTypeIndex);
        } else { ImGui::TextDisabled("Native allocation information unavailable."); }
        if (ImGui::TreeNodeEx("Field aliases", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (const auto& alias : resource->aliases) { ImGui::TextWrapped("%s", alias.c_str()); }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }
    if (pass) {
        ImGui::TextWrapped("%s", pass->name.c_str()); ImGui::TextWrapped("%s", pass->type.c_str());
        ImGui::Text("Requested queue: %s", queueName(pass->logicalQueue));
        if (pass->actualQueueId != UINT32_MAX) { ImGui::Text("Actual queue: Q%u", pass->actualQueueId); }
        ImGui::Text("%s | %zu dependency edges", pass->recorded ? "Recorded" : "Planned", pass->predecessors.size());
        if (pass->recorded) { synchronizationText(pass->synchronization); }
        const auto showUses = [&](const auto& uses) {
            for (const auto& use : uses) {
                if (resource && use.resourceId != resource->id) { continue; }
                const auto* used = findResource(*snapshot_, use.resourceId);
                ImGui::TextWrapped("%s  %s", use.reads && use.writes ? "RW" : use.writes ? "W" : use.reads ? "R" : "Layout",
                    used ? used->name.c_str() : "Unknown resource");
                ImGui::TextDisabled("%s", stateName(use.state)); scopeText("Use", use.scope);
            }
        };
        const auto showBarriers = [&](const auto& barriers) {
            for (const auto& barrier : barriers) {
                if (resource && barrier.resourceId != resource->id) { continue; }
                const auto* used = findResource(*snapshot_, barrier.resourceId);
                if (used) { ImGui::TextWrapped("%s", used->name.c_str()); }
                barrierDetails(barrier); ImGui::Spacing();
            }
        };
        if (ImGui::TreeNodeEx("Boundary accesses", ImGuiTreeNodeFlags_DefaultOpen)) { showUses(pass->uses); ImGui::TreePop(); }
        if (ImGui::TreeNodeEx("Barriers before pass", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (pass->barriers.empty()) { ImGui::TextDisabled("None (visibility may already be covered)."); }
            showBarriers(pass->barriers); ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("Internal stages", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t index = 0; index < pass->stages.size(); ++index) {
                const auto& stage = pass->stages[index];
                ImGui::PushID(int(index));
                if (ImGui::TreeNode(stage.name.c_str())) {
                    ImGui::TextDisabled("%s%s", stage.recorded ? "Recorded" : "Planned", stage.restoreBoundary ? " / layout restore" : "");
                    if (stage.recorded) { synchronizationText(stage.synchronization); }
                    if (stage.allowParallelCompute) { ImGui::TextWrapped("Opaque fork/join boundary; inspect Queues for actual branches."); }
                    showUses(stage.uses); showBarriers(stage.barriers);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (pass->stages.empty()) { ImGui::TextDisabled("No captured declarative stages."); }
            ImGui::TreePop();
        }
    }
    ImGui::Spacing(); ImGui::Separator();
    ImGui::TextWrapped("Barriers are planner boundaries. The backend can merge or elide native operations. Opaque SDK/subsystem internals are not expanded.");
    ImGui::TextWrapped("Memory is observed at capture. A resource's first/last use is not its allocation lifetime, and a queue receipt is not GPU completion.");
}

} // namespace metallic::editor
