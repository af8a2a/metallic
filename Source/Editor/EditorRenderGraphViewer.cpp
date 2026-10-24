#include "Runtime/Render/Core/ResourceState.h"
#include "Editor/EditorRenderGraphViewer.h"

#include <imgui.h>
#include <imgui_internal.h>

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

const char* resourceTypeName(RenderGraphResourceType type)
{
    switch (type) {
    case RenderGraphResourceType::Texture2D: return "Texture";
    case RenderGraphResourceType::Buffer: return "Buffer";
    case RenderGraphResourceType::AccelerationStructure: return "Acceleration structure";
    }
    return "Unknown";
}

const char* accelerationStructureTypeName(RayTracingAccelerationStructureType type)
{
    switch (type) {
    case RayTracingAccelerationStructureType::TopLevel: return "TLAS";
    }
    return "Unknown";
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

template<class Stats>
void drawAliasMemoryStats(
    const Stats& stats, const char* kind, const char* resources,
    uint32_t count, uint32_t transientCount, uint32_t pinnedCount,
    uint32_t eligibleCount, uint32_t aliasedCount, uint32_t unknownCount,
    const char* filter, float scale)
{
    const double savedPercent = stats.complete && stats.logicalBytes
        ? 100.0 * double(stats.savedBytes) / double(stats.logicalBytes) : 0.0;
    ImGui::Text("Graph %s aliasing: %s", kind, stats.aliasingEnabled ? "Enabled" : "Disabled");
    ImGui::Text("Without aliasing: %s | Backing: %s%s", bytesText(stats.logicalBytes).c_str(),
        bytesText(stats.backingBytes).c_str(), stats.complete ? "" : " (known capacity only)");
    if (stats.complete) {
        if (stats.overheadBytes) {
            ImGui::Text("Backing overhead: %s", bytesText(stats.overheadBytes).c_str());
        } else {
            ImGui::Text("Saved: %s (%.2f%%)", bytesText(stats.savedBytes).c_str(), savedPercent);
        }
    } else {
        ImGui::TextDisabled("Partial comparison: %u %s have unknown capacity; savings percentage unavailable.",
            unknownCount, resources);
    }
    ImGui::Text("%u %s | %u transient | %u pinned | %u eligible", count, resources,
        transientCount, pinnedCount, eligibleCount);
    ImGui::Text("%u aliased %s in %u shared slots | %u backing allocations", aliasedCount, resources,
        stats.aliasSlotCount, stats.backingAllocationCount);
    ImGui::TextDisabled("Graph-owned %s capacity; excludes scene, private imports and opaque subsystem memory.", kind);
    ImGui::PushID(kind);
    const std::string slotLabel = std::string("Shared ") + kind + " slots";
    if (!stats.slots.empty() && ImGui::TreeNodeEx(slotLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("AliasSlots", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Resources", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("Backing ID", ImGuiTableColumnFlags_WidthFixed, 88 * scale);
            ImGui::TableSetupColumn("Without alias", ImGuiTableColumnFlags_WidthFixed, 98 * scale);
            ImGui::TableSetupColumn("Backing", ImGuiTableColumnFlags_WidthFixed, 88 * scale);
            ImGui::TableSetupColumn("Net saved", ImGuiTableColumnFlags_WidthFixed, 88 * scale);
            ImGui::TableHeadersRow();
            for (const auto& slot : stats.slots) {
                bool visible = filter[0] == '\0';
                for (const auto& name : slot.resources) { visible |= matches(name, filter); }
                if (!visible) { continue; }
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                for (const auto& name : slot.resources) { ImGui::TextWrapped("%s", name.c_str()); }
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(slot.backingAllocationId));
                ImGui::TableNextColumn(); ImGui::TextUnformatted(bytesText(slot.logicalBytes).c_str());
                ImGui::TableNextColumn(); ImGui::TextUnformatted(bytesText(slot.backingBytes).c_str());
                ImGui::TableNextColumn();
                if (slot.complete) {
                    ImGui::Text("%s%s", slot.overheadBytes ? "-" : "",
                        bytesText(slot.overheadBytes ? slot.overheadBytes : slot.savedBytes).c_str());
                } else { ImGui::TextDisabled("N/A"); }
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
    ImGui::Separator();
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
    if (barrier.memoryAliasing) { ImGui::TextUnformatted("Physical memory alias handoff"); }
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

struct CanvasView {
    ImVec2 origin;
    float zoom;
    float diagramScale;
    bool canHover;

    ImVec2 point(float x, float y) const
    {
        return ImVec2(origin.x + x * diagramScale, origin.y + y * diagramScale);
    }

    bool visible(ImVec2 min, ImVec2 max) const
    {
        return ImGui::IsRectVisible(min, max);
    }

    bool hovered(ImVec2 min, ImVec2 max) const
    {
        return canHover && ImGui::IsMouseHoveringRect(min, max);
    }

    void text(ImDrawList& draw, ImVec2 pos, ImU32 color, const char* value) const
    {
        draw.AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, pos, color, value);
    }
};

// All diagram views use one input transform for drawing and hit testing. The
// returned child remains open until the caller has drawn its diagram.
CanvasView beginCanvas(const char* child, const char* item, const char* fitLabel,
    auto& state, float scale, ImVec2 contentSize, float heightFraction = 1.0f)
{
    constexpr float minimumZoom = 0.05f, maximumZoom = 8.0f;
    if (ImGui::Button(fitLabel)) { state.fit = true; }
    ImGui::SameLine();
    const float previousZoom = state.zoom;
    bool zoomEdited = false;
    if (ImGui::Button("1:1")) { state.zoom = 1.0f; zoomEdited = true; }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(130 * scale);
    zoomEdited |= ImGui::DragFloat("Zoom", &state.zoom, 0.01f, minimumZoom, maximumZoom,
        "%.2fx", ImGuiSliderFlags_AlwaysClamp);
    ImGui::TextDisabled("Wheel: zoom at cursor | Middle-drag: pan | %s: frame all", fitLabel);
    const float height = std::max(1.0f, ImGui::GetContentRegionAvail().y * heightFraction);
    ImGui::BeginChild(child, ImVec2(0, height), true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const auto topLeft = ImGui::GetCursorScreenPos();
    const ImVec2 size(std::max(ImGui::GetContentRegionAvail().x, 1.0f),
        std::max(ImGui::GetContentRegionAvail().y, 1.0f));
    ImGui::InvisibleButton(item, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);
    auto& io = ImGui::GetIO();
    const auto zoomAt = [&](float oldZoom, ImVec2 anchor) {
        const float ratio = state.zoom / oldZoom;
        state.panX = anchor.x - (anchor.x - state.panX) * ratio;
        state.panY = anchor.y - (anchor.y - state.panY) * ratio;
        state.fit = false;
    };
    const ImVec2 extent(std::max(contentSize.x * scale, 1.0f), std::max(contentSize.y * scale, 1.0f));
    if (zoomEdited) { zoomAt(previousZoom, ImVec2(size.x * 0.5f, size.y * 0.5f)); }
    if (state.fit) {
        state.zoom = std::clamp(std::min(std::max(size.x - 24 * scale, 1.0f) / extent.x,
            std::max(size.y - 24 * scale, 1.0f) / extent.y), minimumZoom, 1.0f);
        state.panX = (size.x - extent.x * state.zoom) * 0.5f;
        state.panY = (size.y - extent.y * state.zoom) * 0.5f;
        state.fit = false;
    }
    if (hovered && io.MouseWheel != 0.0f) {
        const float oldZoom = state.zoom;
        state.zoom = std::clamp(state.zoom * std::pow(1.15f, io.MouseWheel), minimumZoom, maximumZoom);
        zoomAt(oldZoom, ImVec2(io.MousePos.x - topLeft.x, io.MousePos.y - topLeft.y));
    }
    const bool panning = active && ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    if (panning) {
        if (!ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            state.panX += io.MouseDelta.x;
            state.panY += io.MouseDelta.y;
        }
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
    }
    return {ImVec2(topLeft.x + state.panX, topLeft.y + state.panY), state.zoom, scale * state.zoom, hovered && !panning};
}

// Build text inside the current visible area before rotating its vertices. The
// final draw command retains the header clip, including the frozen name column.
void angledText(ImDrawList& draw, ImVec2 origin, const char* text, ImVec2 clipMin, ImVec2 clipMax, float zoom)
{
    draw.PushClipRect(clipMin, clipMax, true);
    const auto clip = draw.GetClipRectMin();
    const ImVec2 base(clip.x + 1, clip.y + 1);
    const int first = draw.VtxBuffer.Size;
    draw.AddText(ImGui::GetFont(), ImGui::GetFontSize() * zoom, base, ImGui::GetColorU32(ImGuiCol_Text), text);
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

template<class Stats>
RenderGraphProperties memoryStatsJson(const Stats& stats, const char* scope)
{
    using Json = RenderGraphProperties;
    Json value{{"aliasingEnabled", stats.aliasingEnabled}, {"complete", stats.complete},
        {"aliasSlotCount", stats.aliasSlotCount}, {"backingAllocationCount", stats.backingAllocationCount},
        {"logicalBytes", stats.logicalBytes}, {"backingBytes", stats.backingBytes},
        {"savedBytes", stats.savedBytes}, {"overheadBytes", stats.overheadBytes},
        {"scope", scope}, {"slots", Json::array()}};
    for (const auto& slot : stats.slots) {
        value["slots"].push_back({{"backingAllocationId", slot.backingAllocationId}, {"complete", slot.complete},
            {"logicalBytes", slot.logicalBytes}, {"backingBytes", slot.backingBytes},
            {"savedBytes", slot.savedBytes}, {"overheadBytes", slot.overheadBytes}, {"resources", slot.resources}});
    }
    return value;
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
    const auto& textureMemory = snapshot.textureMemory;
    auto& textureValue = value["textureMemory"] = memoryStatsJson(textureMemory,
        "Graph-owned texture backing capacity; excludes scene, buffers and opaque subsystem memory");
    textureValue["textureCount"] = textureMemory.textureCount;
    textureValue["transientTextureCount"] = textureMemory.transientTextureCount;
    textureValue["pinnedTextureCount"] = textureMemory.pinnedTextureCount;
    textureValue["eligibleTextureCount"] = textureMemory.eligibleTextureCount;
    textureValue["aliasedTextureCount"] = textureMemory.aliasedTextureCount;
    textureValue["unknownTextureCount"] = textureMemory.unknownTextureCount;
    const auto& bufferMemory = snapshot.bufferMemory;
    auto& bufferValue = value["bufferMemory"] = memoryStatsJson(bufferMemory,
        "Graph-owned buffer backing capacity; excludes scene, textures and opaque subsystem memory");
    bufferValue["bufferCount"] = bufferMemory.bufferCount;
    bufferValue["transientBufferCount"] = bufferMemory.transientBufferCount;
    bufferValue["pinnedBufferCount"] = bufferMemory.pinnedBufferCount;
    bufferValue["eligibleBufferCount"] = bufferMemory.eligibleBufferCount;
    bufferValue["aliasedBufferCount"] = bufferMemory.aliasedBufferCount;
    bufferValue["unknownBufferCount"] = bufferMemory.unknownBufferCount;
    for (const auto& resource : snapshot.resources) {
        const auto& m = resource.memory;
        value["resources"].push_back({{"id", resource.id}, {"name", resource.name}, {"aliases", resource.aliases},
            {"private", resource.privateResource}, {"buffer", resource.type == RenderGraphResourceType::Buffer},
            {"type", resourceTypeName(resource.type)},
            {"allocation", m.allocationId}, {"memoryBlock", std::to_string(m.memoryBlockId)}, {"memoryKnown", m.known},
            {"backingAllocationId", m.backingAllocationId}, {"backingSizeBytes", m.backingSizeBytes},
            {"offsetBytes", m.offsetBytes}, {"sizeBytes", m.sizeBytes}, {"memoryType", m.memoryTypeIndex}, {"heap", m.heapIndex}});
        auto& exported = value["resources"].back();
        if (resource.type == RenderGraphResourceType::Buffer) { exported["logicalBytes"] = resource.bufferDesc.size; }
        else if (resource.type == RenderGraphResourceType::AccelerationStructure) {
            const auto& desc = resource.accelerationStructureDesc;
            exported["logicalBytes"] = desc.size;
            exported["accelerationStructure"] = {{"type", accelerationStructureTypeName(desc.type)},
                {"topLevelBackend", desc.topLevelBackend == RayTracingTopLevelBackend::Partitioned ? "Partitioned" : "Standard"},
                {"buildFlags", uint32_t(desc.buildFlags)}};
        } else {
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
            {"after", stateName(b.after)}, {"executionOnly", b.executionOnly}, {"memoryAliasing", b.memoryAliasing},
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
    if (changed) { resourceCanvas_.fit = queueCanvas_.fit = memoryCanvas_.fit = true; }
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
    ImGui::Combo("##ResourceType", &resourceType_, "All resources\0Textures\0Buffers\0Private imports\0Acceleration structures\0");
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
    if (passes.size() > 500) { passes.resize(500); ImGui::TextDisabled("Showing first 500 matching passes. Narrow the search."); }
    struct Row {
        const Resource* resource;
        std::vector<Cell> cells;
        size_t first, last;
    };
    std::vector<Row> rows;
    for (const auto& resource : snapshot_->resources) {
        bool match = matches(resource.name, resourceFilter_);
        for (const auto& alias : resource.aliases) { match |= matches(alias, resourceFilter_); }
        if (!match || (resourceType_ == 1 && resource.type != RenderGraphResourceType::Texture2D) ||
            (resourceType_ == 2 && resource.type != RenderGraphResourceType::Buffer) ||
            (resourceType_ == 3 && !resource.privateResource) ||
            (resourceType_ == 4 && resource.type != RenderGraphResourceType::AccelerationStructure)) { continue; }
        Row row{&resource, {}, passes.size(), 0};
        row.cells.reserve(passes.size());
        for (size_t i = 0; i < passes.size(); ++i) {
            const auto cell = cellFor(*passes[i], resource.id); row.cells.push_back(cell);
            if (cell.read || cell.write || cell.barriers) { row.first = std::min(row.first, i); row.last = i; }
        }
        if (!onlyUsed_ || row.first != passes.size()) { rows.push_back(std::move(row)); }
    }
    constexpr float nameWidth = 238, headerHeight = 140, rowHeight = 25;
    const float cellWidth = columnWidth_;
    const float totalWidth = nameWidth + float(passes.size()) * cellWidth;
    const auto canvas = beginCanvas("ResourcePassMatrix", "ResourceCanvas", "Fit matrix", resourceCanvas_, scale,
        ImVec2(totalWidth, headerHeight + float(rows.size()) * rowHeight));
    const float unit = canvas.diagramScale;
    auto* draw = ImGui::GetWindowDrawList();
    char counts[128];
    std::snprintf(counts, sizeof(counts), "%zu resources / %zu passes", rows.size(), passes.size());
    canvas.text(*draw, canvas.point(4, 8), ImGui::GetColorU32(ImGuiCol_Text), counts);
    canvas.text(*draw, canvas.point(4, 28), ImGui::GetColorU32(ImGuiCol_TextDisabled), "Click a cell to inspect");
    for (size_t i = 0; i < passes.size(); ++i) {
        const auto& pass = *passes[i];
        const float x = nameWidth + float(i) * cellWidth;
        const auto p = canvas.point(x, 0), end = canvas.point(x + cellWidth, headerHeight);
        if (!canvas.visible(p, end)) { continue; }
        if (selectedPassId_ == pass.id) { draw->AddRectFilled(p, end, IM_COL32(90, 105, 120, 90)); }
        const std::string label = pass.name.size() > 25 ? pass.name.substr(0, 23) + ".." : pass.name;
        angledText(*draw, canvas.point(x + 4, 119), label.c_str(),
            canvas.point(nameWidth, 0), canvas.point(totalWidth, headerHeight), canvas.zoom);
        draw->AddRectFilled(canvas.point(x, 128), canvas.point(x + cellWidth, 133), queueColor(pass.logicalQueue));
        if (canvas.hovered(p, end)) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { selectedPassId_ = pass.id; }
            ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
            if (ImGui::BeginTooltip()) {
                ImGui::TextUnformatted(pass.name.c_str()); ImGui::TextUnformatted(pass.type.c_str());
                ImGui::Text("%s; %s", queueName(pass.logicalQueue), pass.recorded ? "recorded" : "planned");
                if (pass.actualQueueId != UINT32_MAX) { ImGui::Text("Actual queue Q%u", pass.actualQueueId); }
                ImGui::EndTooltip();
            }
        }
    }
    for (size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex) {
        const auto& row = rows[rowIndex];
        const auto& resource = *row.resource;
        const float y = headerHeight + float(rowIndex) * rowHeight;
        if (!canvas.visible(canvas.point(0, y), canvas.point(totalWidth, y + rowHeight))) { continue; }
        draw->AddRectFilled(canvas.point(0, y), canvas.point(totalWidth, y + rowHeight),
            rowIndex % 2 == 0 ? IM_COL32(50, 52, 57, 100) : IM_COL32(42, 45, 50, 100));
        const auto nameMin = canvas.point(0, y), nameMax = canvas.point(nameWidth, y + 21);
        if (canvas.visible(nameMin, nameMax)) {
            if (selectedResourceId_ == resource.id) { draw->AddRectFilled(nameMin, nameMax, IM_COL32(90, 105, 120, 120)); }
            const auto label = std::string(resource.type == RenderGraphResourceType::Buffer ? "B  " :
                resource.type == RenderGraphResourceType::AccelerationStructure ? "AS  " : "T  ") + resource.name;
            draw->PushClipRect(nameMin, nameMax, true);
            canvas.text(*draw, canvas.point(4, y + 2), ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
            draw->PopClipRect();
            if (canvas.hovered(nameMin, nameMax)) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { selectedResourceId_ = resource.id; }
                ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
                if (ImGui::BeginTooltip()) {
                    ImGui::TextUnformatted(resource.name.c_str());
                    ImGui::Text("%s | %s", resource.privateResource ? "Private import" : "Graph resource",
                        resource.memory.known ? bytesText(resource.memory.sizeBytes).c_str() : "Memory unknown");
                    for (const auto& alias : resource.aliases) { ImGui::TextUnformatted(alias.c_str()); }
                    ImGui::EndTooltip();
                }
            }
        }
        for (size_t i = 0; i < passes.size(); ++i) {
            const float x = nameWidth + float(i) * cellWidth;
            const auto p = canvas.point(x, y), end = canvas.point(x + cellWidth, y + 21);
            if (!canvas.visible(p, end)) { continue; }
            const auto cell = row.cells[i];
            if (i >= row.first && i <= row.last) { draw->AddRectFilled(p, end, IM_COL32(86, 89, 94, 160)); }
            if (cell.read || cell.write) {
                if (cell.read && cell.write) {
                    draw->AddTriangleFilled(p, ImVec2(end.x, p.y), ImVec2(p.x, end.y), kRead);
                    draw->AddTriangleFilled(ImVec2(end.x, p.y), end, ImVec2(p.x, end.y), kWrite);
                } else { draw->AddRectFilled(p, end, cell.write ? kWrite : kRead); }
                canvas.text(*draw, canvas.point(x + cellWidth * 0.5f - (cell.read && cell.write ? 9 : 4), y + 2),
                    IM_COL32(20, 25, 30, 255), cell.read && cell.write ? "RW" : cell.write ? "W" : "R");
            }
            if (showBarriers_ && cell.barriers) {
                const float bx = p.x + 4 * unit, by = p.y + 10 * unit, r = 4 * unit;
                draw->AddQuadFilled(ImVec2(bx, by-r), ImVec2(bx+r, by), ImVec2(bx, by+r), ImVec2(bx-r, by), cell.layout ? kLayout : kBarrier);
            }
            if (selectedPassId_ == passes[i]->id && selectedResourceId_ == resource.id) {
                draw->AddRect(p, end, IM_COL32(255, 255, 255, 255), 0.0f, 2.0f, ImDrawFlags_None);
            }
            if (canvas.hovered(p, end)) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { selectedResourceId_ = resource.id; selectedPassId_ = passes[i]->id; }
                ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
                if (ImGui::BeginTooltip()) {
                    ImGui::Text("%s / %s", resource.name.c_str(), passes[i]->name.c_str());
                    ImGui::Text("%s | %u planned barrier boundaries", cell.read && cell.write ? "Read + write" :
                        cell.write ? "Write" : cell.read ? "Read" : "No declared data access", cell.barriers);
                    ImGui::TextDisabled("Includes boundary declarations and captured internal stages.");
                    ImGui::EndTooltip();
                }
            }
        }
    }
    ImGui::EndChild();
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
    const auto canvas = beginCanvas("QueueTopology", "QueueCanvas", "Fit topology", queueCanvas_, scale,
        ImVec2(115 + float(maximum + 1) * 152, std::max(float(lanes.size()) * 108, 1.0f)), 0.60f);
    const float zoom = canvas.zoom;
    const float diagramScale = canvas.diagramScale;
    const float step = 152 * diagramScale, laneHeight = 108 * diagramScale, left = 115 * diagramScale;
    const ImVec2 origin = canvas.origin;
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
        const bool hovered = canvas.hovered(p, ImVec2(p.x + 132 * diagramScale, p.y + 70 * diagramScale));
        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && pass) { selectedPassId_ = pass->id; }
        if (hovered) {
            ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
            if (ImGui::BeginTooltip()) {
                ImGui::Text("S%u  %s", segment.id, roleName(segment.role));
                if (pass) { ImGui::TextUnformatted(pass->name.c_str()); }
                ImGui::Text("Q%u | %zu predecessors", segment.queueId, segment.predecessors.size());
                if (!segment.completionKnown) { ImGui::TextDisabled("GPU completion unavailable in this capture."); }
                ImGui::EndTooltip();
            }
        }
    }
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
    // Keep allocation statistics scrollable without crowding out the diagram.
    ImGui::BeginChild("MemorySummary", ImVec2(0, std::max(1.0f,
        std::min(300 * scale, ImGui::GetContentRegionAvail().y * 0.40f))), false);
    const auto& textureMemory = snapshot_->textureMemory;
    drawAliasMemoryStats(textureMemory, "texture", "textures", textureMemory.textureCount,
        textureMemory.transientTextureCount, textureMemory.pinnedTextureCount, textureMemory.eligibleTextureCount,
        textureMemory.aliasedTextureCount, textureMemory.unknownTextureCount, resourceFilter_, scale);
    const auto& bufferMemory = snapshot_->bufferMemory;
    drawAliasMemoryStats(bufferMemory, "buffer", "buffers", bufferMemory.bufferCount,
        bufferMemory.transientBufferCount, bufferMemory.pinnedBufferCount, bufferMemory.eligibleBufferCount,
        bufferMemory.aliasedBufferCount, bufferMemory.unknownBufferCount, resourceFilter_, scale);
    std::map<uint64_t, std::vector<const Resource*>> blocks;
    std::set<uint64_t> allocations;
    std::map<uint64_t, uint64_t> backingSizes;
    uint64_t bytes = 0; size_t unknown = 0, aliasPairs = 0;
    for (const auto& resource : snapshot_->resources) {
        if (!resource.memory.known) { ++unknown; continue; }
        if (!allocations.insert(resource.memory.allocationId).second) { continue; }
        // Old captures may lack backing metadata. Keep each native resource row,
        // while shared physical allocation capacity contributes only once.
        const auto backingId = resource.memory.backingAllocationId
            ? resource.memory.backingAllocationId : resource.memory.allocationId;
        const auto backingSize = resource.memory.backingSizeBytes
            ? resource.memory.backingSizeBytes : resource.memory.sizeBytes;
        backingSizes[backingId] = std::max(backingSizes[backingId], backingSize);
        blocks[resource.memory.memoryBlockId].push_back(&resource);
    }
    for (const auto& [backingId, size] : backingSizes) { bytes += size; }
    for (const auto& [block, resources] : blocks) {
        for (size_t i = 0; i < resources.size(); ++i) {
            for (size_t j = i + 1; j < resources.size(); ++j) { aliasPairs += overlaps(*resources[i], *resources[j]); }
        }
    }
    ImGui::Text("%zu native resources | %zu backing allocations | %s | %zu overlapping pairs",
        allocations.size(), backingSizes.size(), bytesText(bytes).c_str(), aliasPairs);
    ImGui::TextWrapped("Actual memory block + byte ranges. Shared block does not imply aliasing; only overlapping ranges do.");
    ImGui::TextDisabled("Capture allocations only; excludes opaque subsystem memory and other in-flight slots. Unknown: %zu", unknown);
    if (aliasPairs == 0) { ImGui::TextDisabled("No physical memory aliasing observed. Graph field aliases share one resource."); }
    ImGui::EndChild();
    struct Block {
        size_t index;
        std::vector<const Resource*> resources;
        uint64_t extent;
    };
    std::vector<Block> visibleBlocks;
    size_t blockIndex = 0;
    float height = 0;
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
                (resourceType_ != 2 || r->type == RenderGraphResourceType::Buffer) && (resourceType_ != 3 || r->privateResource) &&
                (resourceType_ != 4 || r->type == RenderGraphResourceType::AccelerationStructure);
        }
        if (visible) {
            height += 64 + float(resources.size()) * 32;
            visibleBlocks.push_back({blockIndex, resources, extent});
        }
    }
    constexpr float width = 900;
    const auto canvas = beginCanvas("MemoryBlocks", "MemoryCanvas", "Fit memory", memoryCanvas_, scale,
        ImVec2(width, std::max(height, 48.0f)));
    auto* draw = ImGui::GetWindowDrawList();
    float y = 0;
    if (visibleBlocks.empty()) {
        canvas.text(*draw, canvas.point(0, 0), ImGui::GetColorU32(ImGuiCol_TextDisabled), "No memory blocks match the filter.");
    }
    for (const auto& block : visibleBlocks) {
        char label[160];
        std::snprintf(label, sizeof(label), "Block %zu | heap %u / type %u | %zu native resources", block.index,
            block.resources.front()->memory.heapIndex, block.resources.front()->memory.memoryTypeIndex, block.resources.size());
        canvas.text(*draw, canvas.point(0, y), ImGui::GetColorU32(ImGuiCol_Text), label);
        const auto span = "0 .. " + bytesText(block.extent) + " observed span (not total block capacity)";
        canvas.text(*draw, canvas.point(0, y + 20), ImGui::GetColorU32(ImGuiCol_TextDisabled), span.c_str());
        y += 48;
        for (const auto* resource : block.resources) {
            const auto p = canvas.point(0, y), end = canvas.point(width, y + 26);
            y += 32;
            if (!canvas.visible(p, end)) { continue; }
            const float x0 = float(double(resource->memory.offsetBytes) / double(block.extent)) * width;
            const float x1 = float(double(resource->memory.offsetBytes + resource->memory.sizeBytes) / double(block.extent)) * width;
            bool alias = false;
            for (const auto* other : block.resources) { alias |= overlaps(*resource, *other); }
            draw->AddRectFilled(p, end, IM_COL32(42, 45, 50, 255), 2 * canvas.diagramScale);
            draw->AddRectFilled(ImVec2(p.x + x0 * canvas.diagramScale, p.y),
                ImVec2(p.x + std::max(x1, x0 + 2) * canvas.diagramScale, end.y),
                alias ? IM_COL32(180, 71, 60, 255) :
                resource->type == RenderGraphResourceType::AccelerationStructure ? IM_COL32(129, 91, 162, 255) :
                resource->type == RenderGraphResourceType::Buffer ? IM_COL32(73, 111, 151, 255) : IM_COL32(66, 126, 119, 255),
                2 * canvas.diagramScale);
            draw->PushClipRect(p, end, true);
            canvas.text(*draw, ImVec2(p.x + 5 * canvas.diagramScale, p.y + 5 * canvas.diagramScale),
                IM_COL32(243, 244, 246, 255), resource->name.c_str());
            draw->PopClipRect();
            if (selectedResourceId_ == resource->id) { draw->AddRect(p, end, kBarrier, 2.0f, 2.0f, ImDrawFlags_None); }
            if (canvas.hovered(p, end)) {
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { selectedResourceId_ = resource->id; }
                ImGui::SetNextWindowViewport(ImGui::GetWindowViewport()->ID);
                if (ImGui::BeginTooltip()) {
                    ImGui::TextUnformatted(resource->name.c_str());
                    ImGui::Text("Allocation #%llu | offset %llu | size %s", static_cast<unsigned long long>(resource->memory.allocationId),
                        static_cast<unsigned long long>(resource->memory.offsetBytes), bytesText(resource->memory.sizeBytes).c_str());
                    ImGui::Text("Backing #%llu | capacity %s", static_cast<unsigned long long>(resource->memory.backingAllocationId),
                        bytesText(resource->memory.backingSizeBytes).c_str());
                    ImGui::TextUnformatted(alias ? "Physical range overlaps another resource." : "No overlapping resource range.");
                    ImGui::EndTooltip();
                }
            }
        }
        draw->AddLine(canvas.point(0, y + 8), canvas.point(width, y + 8), ImGui::GetColorU32(ImGuiCol_Separator));
        y += 16;
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
        ImGui::TextDisabled("%s / %s", resourceTypeName(resource->type),
            resource->privateResource ? "Private import" : "Graph resource");
        if (resource->type == RenderGraphResourceType::Buffer) {
            ImGui::Text("Logical size: %s", bytesText(resource->bufferDesc.size).c_str());
        } else if (resource->type == RenderGraphResourceType::AccelerationStructure) {
            const auto& desc = resource->accelerationStructureDesc;
            ImGui::Text("%s | logical size: %s", accelerationStructureTypeName(desc.type), bytesText(desc.size).c_str());
            if (desc.type == RayTracingAccelerationStructureType::TopLevel) {
                ImGui::Text("Backend: %s", desc.topLevelBackend == RayTracingTopLevelBackend::Partitioned ? "Partitioned" : "Standard");
            }
        } else {
            ImGui::Text("%u x %u x %u | format %u", resource->textureDesc.width, resource->textureDesc.height,
                resource->textureDesc.depth, uint32_t(resource->textureDesc.format));
            ImGui::Text("%u mip(s), %u layer(s)", resource->textureDesc.mipCount, resource->textureDesc.layerCount);
        }
        if (resource->memory.known) {
            ImGui::Text("Allocation #%llu", static_cast<unsigned long long>(resource->memory.allocationId));
            ImGui::Text("Backing #%llu | capacity %s", static_cast<unsigned long long>(resource->memory.backingAllocationId),
                bytesText(resource->memory.backingSizeBytes).c_str());
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
