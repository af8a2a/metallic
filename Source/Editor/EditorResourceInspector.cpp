#include "Editor/EditorResourceInspector.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic {
namespace {
using debug::DebugValue;

DebugValue call(debug::DebugCore& core, const char* method, DebugValue params = DebugValue::object())
{
    return core.dispatch({{"method", method}, {"params", std::move(params)}});
}

template<typename T>
T read(const uint8_t* data)
{
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

float halfFloat(const uint8_t* data)
{
    const auto h = read<uint16_t>(data);
    const int exponent = (h >> 10) & 31, mantissa = h & 1023;
    const float magnitude = exponent == 31
        ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
        : std::ldexp(exponent ? 1.0f + mantissa / 1024.0f : mantissa / 1024.0f, exponent ? exponent - 15 : -14);
    return h & 0x8000 ? -magnitude : magnitude;
}

std::array<float, 4> pixel(const debug::DebugArtifact& artifact, size_t index)
{
    const auto* p = artifact.bytes.data() + index * artifact.layout.stride;
    const auto& name = artifact.layout.name;
    if (name == "RGBA8") { return {p[0] / 255.f, p[1] / 255.f, p[2] / 255.f, p[3] / 255.f}; }
    if (name == "BGRA8") { return {p[2] / 255.f, p[1] / 255.f, p[0] / 255.f, p[3] / 255.f}; }
    if (name == "RGBA16F") { return {halfFloat(p), halfFloat(p + 2), halfFloat(p + 4), halfFloat(p + 6)}; }
    if (name == "RGBA32F") { return {read<float>(p), read<float>(p + 4), read<float>(p + 8), read<float>(p + 12)}; }
    std::array<float, 4> result{0, 0, 0, 1};
    for (size_t c = 0; c < std::min(size_t(4), artifact.layout.fields.size()); ++c) {
        const auto& field = artifact.layout.fields[c];
        const auto* data = p + field.offset;
        result[c] = field.type == "f16" ? halfFloat(data) : field.type == "u32" ? float(read<uint32_t>(data)) :
            field.type == "i32" ? float(read<int32_t>(data)) : read<float>(data);
    }
    if (artifact.layout.fields.size() == 1) { result[1] = result[2] = result[0]; }
    return result;
}

std::string displayValue(const DebugValue& value)
{
    if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        const double number = value.get<double>();
        return std::isnan(number) ? "NaN" : number < 0 ? "-Inf" : "+Inf";
    }
    if (value.is_array()) {
        std::string text = "[";
        for (const auto& item : value) {
            if (text.size() > 1) { text += ", "; }
            text += displayValue(item);
        }
        return text + "]";
    }
    // JSON integer formatting preserves all 64 bits without wire-protocol tags.
    return value.dump();
}
} // namespace

void EditorResourceInspector::select(std::string resource)
{
    if (selected_ == resource) { return; }
    selected_ = std::move(resource);
    capture_.reset();
    refresh_ = true;
    offset_ = 0;
    roi_ = {0, 0, 0, 0};
    status_.clear();
}

debug::DebugValue EditorResourceInspector::diagnostics() const
{
    DebugValue result{{"selected", selected_}, {"job", job_}, {"status", status_}, {"live", live_},
        {"graph", graph_}, {"generation", generation_}, {"previewReady", descriptor_ != 0 && capture_ &&
            !capture_->artifacts.empty() && capture_->artifacts[0].metadata.value("kind", "") == "texture" && !imageDirty_ && !upload_}};
    if (capture_ && !capture_->artifacts.empty()) {
        result["capture"] = capture_->artifacts[0].metadata;
        result["capture"]["byteCount"] = capture_->artifacts[0].bytes.size();
        result["capture"]["evidence"] = capture_->snapshot.evidence.value();
    }
    return result;
}

void EditorResourceInspector::cancel(debug::DebugCore& core)
{
    if (!job_.empty()) { call(core, "jobs.cancel", {{"job", job_}}); job_.clear(); }
}

void EditorResourceInspector::request(render::RenderDebugRuntime& runtime,
    const debug::DebugSnapshot& snapshot, const DebugValue& resource)
{
    refresh_ = false;
    nextRefresh_ = ImGui::GetTime() + 0.5;
    const auto layoutName = resource.value("layout", "");
    const bool texture = resource.value("kind", "") == "texture";
    const std::string layout = layoutName == "raw" ? "u32" : layoutName;
    if (!runtime.layouts().contains(layout) || !resource.value("captureSupported", true)) {
        status_ = resource.value("reason", "This resource format cannot be captured.");
        live_ = false;
        return;
    }
    DebugValue spec{{"id", selected_}, {"layout", layout}, {"allocation", resource.at("allocation")}};
    const uint64_t stride = runtime.layouts().at(layout).stride;
    // Leave room for the snapshot metadata charged by the capture runtime.
    const uint64_t budget = std::min(runtime.core().limits().jobBytes, runtime.core().limits().frameBytes);
    const uint64_t availableBytes = budget > (1u << 20) ? budget - (1u << 20) : 0;
    if (texture) {
        const int width = resource.at("width"), height = resource.at("height");
        roi_[0] = std::clamp(roi_[0], 0, width - 1);
        roi_[1] = std::clamp(roi_[1], 0, height - 1);
        const int w = roi_[2] > 0 ? std::min(roi_[2], width - roi_[0]) : width - roi_[0];
        const int h = roi_[3] > 0 ? std::min(roi_[3], height - roi_[1]) : height - roi_[1];
        if (uint64_t(w) * h * stride > availableBytes) {
            status_ = "Image exceeds the capture budget. Select a smaller ROI and Refresh.";
            live_ = false;
            return;
        }
        spec["roi"] = {{"x", roi_[0]}, {"y", roi_[1]}, {"width", w}, {"height", h}};
    } else {
        const uint64_t capacity = resource.value("size", uint64_t(0)) / stride;
        if (offset_ >= capacity || !availableBytes) { status_ = "Element offset is outside the buffer or capture budget."; live_ = false; return; }
        uint64_t count = std::min({uint64_t(std::clamp(count_, 1, 65536)), capacity - offset_, availableBytes / stride});
        // The RHI requires four-byte aligned copies (all registered buffer layouts here satisfy this).
        spec["offset"] = offset_;
        spec["count"] = count;
    }
    const auto response = call(runtime.core(), "capture.batch", {{"generation", snapshot.evidence.generation},
        {"pass", resource.at("pass")}, {"checkpoint", resource.at("checkpoint")}, {"resources", DebugValue::array({spec})}});
    if (response.at("status") == "ok") { job_ = response.at("result").at("job"); status_ = "Queued for next graph execution"; }
    else { status_ = response.at("error").at("message"); live_ = false; }
}

void EditorResourceInspector::draw(render::RenderDebugRuntime& runtime, render::Device& device,
    render::vulkan::VulkanImGuiBackend& backend, float scale)
{
    runtime.poll();
    auto& core = runtime.core();
    const auto snapshot = core.latestSnapshot();
    const auto graphResponse = call(core, "rg.describe");
    const auto& graph = graphResponse.at("result");
    const auto generation = graph.value("generation", uint64_t(0));
    const auto graphId = graph.value("id", "");
    if (generation != generation_ || graphId != graph_) {
        cancel(core);
        capture_.reset();
        generation_ = generation;
        graph_ = graphId;
        refresh_ = true;
        status_ = "Graph changed; waiting for current resource descriptors";
    }
    if (!snapshot || snapshot->evidence.generation != generation_ || snapshot->evidence.graph != graph_) {
        ImGui::TextUnformatted("Waiting for a completed graph execution...");
        return;
    }
    const auto& resources = snapshot->values.at("resources");
    ImGui::TextDisabled("Pass-boundary snapshots | async GPU readback | live refresh: 2 Hz");
    ImGui::BeginChild("ResourceList", ImVec2(280 * scale, 0), true);
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##ResourceFilter", "Filter resource / pass", filter_, sizeof(filter_));
    for (const auto* kind : {"texture", "buffer"}) {
        if (!ImGui::CollapsingHeader(kind[0] == 't' ? "Textures" : "Buffers", ImGuiTreeNodeFlags_DefaultOpen)) { continue; }
        for (auto it = resources.begin(); it != resources.end(); ++it) {
            if (it.value().value("kind", "") != kind || (filter_[0] && it.key().find(filter_) == std::string::npos)) { continue; }
            if (ImGui::Selectable(it.key().c_str(), selected_ == it.key())) { cancel(core); select(it.key()); }
            if (ImGui::IsItemHovered()) { ImGui::SetTooltip("%s / %s", it.value().value("pass", "").c_str(), it.value().value("checkpoint", "").c_str()); }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("ResourceDetail", ImVec2(0, 0));
    if (!resources.contains(selected_)) {
        ImGui::TextWrapped("Select a Texture or Buffer. Only resources from executed passes are listed.");
        ImGui::EndChild();
        return;
    }
    if (refresh_) { cancel(core); }
    const auto& resource = resources.at(selected_);
    ImGui::TextUnformatted(selected_.c_str());
    ImGui::TextDisabled("%s / %s | layout: %s", resource.value("pass", "").c_str(),
        resource.value("checkpoint", "").c_str(), resource.value("layout", "").c_str());
    const bool texture = resource.value("kind", "") == "texture";
    if (texture) {
        ImGui::Text("%u x %u | mip 0, layer 0", resource.value("width", 0u), resource.value("height", 0u));
        ImGui::SetNextItemWidth(320 * scale);
        ImGui::InputInt4("ROI x/y/w/h", roi_.data());
        ImGui::TextDisabled("Width/height 0: remaining extent. Refresh applies the range.");
    } else {
        ImGui::Text("%llu bytes | declared stride: %u", static_cast<unsigned long long>(resource.value("size", uint64_t(0))), resource.value("structureStride", 0u));
        ImGui::SetNextItemWidth(160 * scale);
        ImGui::InputScalar("First element", ImGuiDataType_U64, &offset_);
        ImGui::SameLine(); ImGui::SetNextItemWidth(110 * scale); ImGui::InputInt("Count", &count_);
    }
    if (ImGui::Button("Refresh")) { cancel(core); refresh_ = true; }
    ImGui::SameLine();
    if (ImGui::Checkbox("Live", &live_) && !live_) { cancel(core); refresh_ = false; }
    ImGui::SameLine(); ImGui::TextDisabled(live_ ? "Updating" : "Frozen snapshot");
    if (!job_.empty()) {
        const auto jobResponse = call(core, "jobs.get", {{"job", job_}});
        if (jobResponse.at("status") != "ok") { status_ = jobResponse.at("error").at("message"); job_.clear(); live_ = false; }
        else {
            const auto& job = jobResponse.at("result");
            status_ = job.value("state", "");
            if (status_ == "Ready") {
                const auto completed = core.completedCapture(job_);
                if (completed && !completed->artifacts.empty() && completed->artifacts[0].metadata.at("id") == selected_) {
                    if (!capture_) {
                        const auto& layout = completed->artifacts[0].layout.name;
                        scalar_ = layout == "f32" ? 2 : layout == "i32" ? 1 : 0;
                    }
                    capture_ = completed;
                    imageDirty_ = texture;
                }
                job_.clear();
            } else if (status_ == "Failed" || status_ == "Cancelled") {
                if (job.contains("error")) { status_ = job.at("error").at("message"); }
                job_.clear(); live_ = false;
            }
        }
    }
    if (job_.empty() && (refresh_ || (live_ && ImGui::GetTime() >= nextRefresh_))) { request(runtime, *snapshot, resource); }
    ImGui::TextWrapped("%s", status_.c_str());
    if (capture_) {
        ImGui::TextDisabled("Captured execution %llu | generation %llu | %llu bytes",
            static_cast<unsigned long long>(capture_->snapshot.evidence.execution),
            static_cast<unsigned long long>(capture_->snapshot.evidence.generation),
            static_cast<unsigned long long>(capture_->artifacts[0].bytes.size()));
        if (texture) { drawTexture(device, backend); } else { drawBuffer(); }
    }
    ImGui::EndChild();
}

bool EditorResourceInspector::rebuildImage(render::Device& device, render::vulkan::VulkanImGuiBackend& backend)
{
    using namespace render;
    const auto& artifact = capture_->artifacts[0];
    const uint32_t width = artifact.metadata.at("roi").at("width"), height = artifact.metadata.at("roi").at("height");
    auto texture = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
        .format = Format::RGBA8Unorm, .width = width, .height = height});
    if (!texture) { status_ = resultToString(texture); return false; }
    auto view = device.createTextureView(**texture, {});
    if (!view) { status_ = resultToString(view); return false; }
    auto staging = device.createBuffer({.size = uint64_t(width) * height * 4,
        .usage = BufferUsageBits::TransferSource, .memoryLocation = MemoryLocation::HostUpload});
    if (!staging) { status_ = resultToString(staging); return false; }
    auto* bytes = static_cast<uint8_t*>((*staging)->map());
    if (!bytes) { status_ = "Could not map preview upload"; return false; }
    const float gain = std::exp2(exposure_), span = std::max(rangeMax_ - rangeMin_, 1e-20f);
    for (size_t i = 0; i < size_t(width) * height; ++i) {
        const auto p = pixel(artifact, i);
        bool finite = true;
        for (size_t c = 0; c < 3; ++c) {
            const float value = (p[channel_ ? size_t(channel_ - 1) : c] * gain - rangeMin_) / span;
            finite &= std::isfinite(value);
            bytes[i * 4 + c] = std::isfinite(value) ? uint8_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f) : 0;
        }
        if (!finite) { bytes[i * 4] = 255; bytes[i * 4 + 1] = 0; bytes[i * 4 + 2] = 255; }
        bytes[i * 4 + 3] = 255;
    }
    (*staging)->flush(); (*staging)->unmap();
    auto descriptor = backend.addTexture(**view);
    if (!descriptor) { status_ = resultToString(descriptor); return false; }
    if (descriptor_) { backend.removeTexture(descriptor_); }
    image_ = std::move(*texture); view_ = std::move(*view); upload_ = std::move(*staging); descriptor_ = *descriptor;
    imageDirty_ = false;
    return true;
}

render::Result<> EditorResourceInspector::upload(render::CommandBuffer& commands)
{
    using namespace render;
    if (!upload_) { return {}; }
    TextureBarrierDesc barrier{.texture = image_.get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    auto result = commands.synchronize({.textures = {&barrier, 1}});
    if (!result) { return result; }
    result = upload_->slice().and_then([&](const auto& slice) {
        return commands.copyBufferToTexture({.texture = image_.get(), .buffer = slice,
            .bufferRowPitch = image_->desc().width * 4, .bufferSlicePitch = image_->desc().width * image_->desc().height * 4,
            .width = image_->desc().width, .height = image_->desc().height});
    });
    if (!result) { return result; }
    barrier.oldLayout = TextureLayout::TransferDestination; barrier.newLayout = TextureLayout::ShaderRead;
    barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
    barrier.after = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead};
    result = commands.synchronize({.textures = {&barrier, 1}});
    // The command buffer retains the upload allocation and image until completion.
    if (result) { upload_.reset(); }
    return result;
}

void EditorResourceInspector::drawTexture(render::Device& device, render::vulkan::VulkanImGuiBackend& backend)
{
    ImGui::SetNextItemWidth(100); imageDirty_ |= ImGui::Combo("Channel", &channel_, "RGB\0R\0G\0B\0A\0");
    ImGui::SameLine(); ImGui::SetNextItemWidth(130); imageDirty_ |= ImGui::SliderFloat("Exposure", &exposure_, -16, 16);
    ImGui::SetNextItemWidth(110); imageDirty_ |= ImGui::DragFloat("Min", &rangeMin_, 0.01f);
    ImGui::SameLine(); ImGui::SetNextItemWidth(110); imageDirty_ |= ImGui::DragFloat("Max", &rangeMax_, 0.01f);
    ImGui::Checkbox("Fit", &fit_); ImGui::SameLine(); ImGui::SetNextItemWidth(130); ImGui::SliderFloat("Zoom", &zoom_, 0.1f, 16.f);
    ImGui::TextDisabled("Display: range/exposure + clamp; magenta = non-finite. Hover for raw values.");
    if (imageDirty_ && !rebuildImage(device, backend)) { return; }
    if (!descriptor_) { return; }
    const auto& artifact = capture_->artifacts[0];
    const auto& roi = artifact.metadata.at("roi");
    const int width = roi.at("width"), height = roi.at("height");
    ImGui::BeginChild("TexturePixels", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    const auto available = ImGui::GetContentRegionAvail();
    const float factor = fit_ ? std::max(0.01f, std::min(available.x / width, available.y / height)) : zoom_;
    const auto origin = ImGui::GetCursorScreenPos();
    ImGui::Image(static_cast<ImTextureID>(descriptor_), ImVec2(width * factor, height * factor));
    if (ImGui::IsItemHovered()) {
        const auto mouse = ImGui::GetMousePos();
        const int x = std::clamp(int((mouse.x - origin.x) / factor), 0, width - 1);
        const int y = std::clamp(int((mouse.y - origin.y) / factor), 0, height - 1);
        const auto index = size_t(y) * width + x;
        const auto values = debug::decodeBuffer(std::span(artifact.bytes).subspan(index * artifact.layout.stride, artifact.layout.stride), artifact.layout);
        ImGui::BeginTooltip();
        ImGui::Text("Pixel (%d, %d)", x + roi.at("x").get<int>(), y + roi.at("y").get<int>());
        if (values) {
            for (const auto& field : artifact.layout.fields) {
                ImGui::Text("%s: %s", field.name.c_str(), displayValue(values->at(0).at(field.name)).c_str());
            }
        }
        ImGui::Text("Bytes: %s", debug::hexEncode(std::span(artifact.bytes).subspan(index * artifact.layout.stride, artifact.layout.stride)).c_str());
        ImGui::EndTooltip();
    }
    ImGui::EndChild();
}

void EditorResourceInspector::drawBuffer()
{
    const auto& artifact = capture_->artifacts[0];
    const bool raw = artifact.layout.name == "u32" || artifact.layout.name == "i32" || artifact.layout.name == "f32";
    if (raw) {
        ImGui::SetNextItemWidth(120); ImGui::Combo("Interpret as", &scalar_, "uint32\0int32\0float32\0");
        ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::SliderInt("Words / row", &columns_, 1, 16);
        ImGui::TextDisabled("Raw 32-bit words; this does not infer the shader's struct layout.");
    }
    ImGui::Checkbox("Hex bytes", &hex_);
    const int columns = raw ? columns_ : std::min(int(artifact.layout.fields.size()), 32);
    const size_t elements = artifact.bytes.size() / artifact.layout.stride;
    const int rows = int(raw ? (elements + columns - 1) / columns : elements);
    if (!ImGui::BeginTable("BufferValues", columns + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, ImVec2(0, 0))) { return; }
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Element / byte", ImGuiTableColumnFlags_WidthFixed, 140);
    for (int c = 0; c < columns; ++c) {
        const auto label = raw ? "+" + std::to_string(c) : artifact.layout.fields[c].name;
        ImGui::TableSetupColumn(label.c_str(), ImGuiTableColumnFlags_WidthFixed, 130);
    }
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper; clipper.Begin(rows);
    while (clipper.Step()) {
        for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
            const size_t first = raw ? size_t(row) * columns : size_t(row);
            const uint64_t absolute = artifact.metadata.value("elementOffset", uint64_t(0)) + first;
            ImGui::TableNextRow(); ImGui::TableNextColumn();
            ImGui::Text("%llu / 0x%llX", static_cast<unsigned long long>(absolute), static_cast<unsigned long long>(absolute * artifact.layout.stride));
            DebugValue decoded;
            if (!raw) {
                auto result = debug::decodeBuffer(std::span(artifact.bytes).subspan(first * artifact.layout.stride, artifact.layout.stride), artifact.layout);
                if (result) { decoded = result->at(0); }
            }
            for (int c = 0; c < columns; ++c) {
                ImGui::TableNextColumn();
                if (raw) {
                    const size_t index = first + c;
                    if (index >= elements) { continue; }
                    const auto* p = artifact.bytes.data() + index * 4;
                    if (hex_) { ImGui::Text("0x%08X", read<uint32_t>(p)); }
                    else if (scalar_ == 0) { ImGui::Text("%u", read<uint32_t>(p)); }
                    else if (scalar_ == 1) { ImGui::Text("%d", read<int32_t>(p)); }
                    else { ImGui::Text("%.9g", read<float>(p)); }
                } else {
                    const auto& field = artifact.layout.fields[c];
                    if (hex_) {
                        const size_t size = field.type == "u8" ? 1 : field.type == "f16" || field.type == "u16" ? 2 : field.type == "u64" || field.type == "i64" || field.type == "f64" ? 8 : 4;
                        ImGui::TextUnformatted(debug::hexEncode(std::span(artifact.bytes).subspan(first * artifact.layout.stride + field.offset, size * field.count)).c_str());
                    } else if (decoded.contains(field.name)) { ImGui::TextUnformatted(displayValue(decoded.at(field.name)).c_str()); }
                }
            }
        }
    }
    ImGui::EndTable();
}

void EditorResourceInspector::shutdown(render::vulkan::VulkanImGuiBackend& backend)
{
    runtime_.drain();
    if (descriptor_) { backend.removeTexture(descriptor_); descriptor_ = 0; }
    upload_.reset(); view_.reset(); image_.reset(); capture_.reset();
}

} // namespace metallic
