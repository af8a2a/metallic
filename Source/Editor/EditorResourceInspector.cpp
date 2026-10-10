#include "Editor/EditorResourceInspector.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

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
    count_ = 256;
    bufferLayout_.clear();
    rawBuffer_ = false;
    fieldPage_ = 0;
    roi_ = {0, 0, 0, 0};
    slice_ = 0;
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
        result["capture"]["schema"] = capture_->artifacts[0].layout.schema();
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
    const std::string layout = layoutName == "raw" ? (bufferLayout_.empty() ? "u32" : bufferLayout_) : layoutName;
    if (!runtime.layouts().contains(layout) || !resource.value("captureSupported", true)) {
        status_ = resource.value("reason", "This resource format cannot be captured.");
        live_ = false;
        return;
    }
    DebugValue spec{{"id", selected_}, {"layout", layout}, {"allocation", resource.at("allocation")},
        {"layoutHash", runtime.layouts().at(layout).layoutHash()}};
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
        const bool volume = resource.value("textureType", 0u) == static_cast<uint32_t>(render::TextureType::Texture3D);
        const int depth = std::max(1, resource.value("depth", 1));
        const int slices = volume && cube_ ? depth : 1;
        if (uint64_t(w) * h > availableBytes / stride / slices) {
            status_ = "Image exceeds the capture budget. Select a smaller ROI or switch to Z slice, then Refresh.";
            live_ = false;
            return;
        }
        spec["roi"] = {{"x", roi_[0]}, {"y", roi_[1]}, {"width", w}, {"height", h}};
        slice_ = std::clamp(slice_, 0, depth - 1);
        spec["slice"] = volume && cube_ ? 0 : slice_;
        spec["sliceCount"] = slices;
    } else {
        const uint64_t capacity = resource.value("size", uint64_t(0)) / stride;
        if (offset_ >= capacity || !availableBytes) { status_ = "Element offset is outside the buffer or capture budget."; live_ = false; return; }
        uint64_t count = std::min({uint64_t(std::clamp(count_, 1, 65536)), capacity - offset_, availableBytes / stride});
        count_ = static_cast<int>(count);
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
    ImGui::TextDisabled("Pass / subsystem snapshots | async GPU readback | live refresh: 2 Hz");
    ImGui::BeginChild("ResourceList", ImVec2(280 * scale, 0), true);
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##ResourceFilter", "Filter resource / subsystem", filter_, sizeof(filter_));
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##ResourceSource", &sourceFilter_, "All resources\0Render graph\0Subsystems\0");
    std::set<std::string> owners;
    for (const auto& resource : resources) { owners.insert(resource.value("subsystem", "")); }
    for (const auto& owner : owners) {
        if ((sourceFilter_ == 1 && !owner.empty()) || (sourceFilter_ == 2 && owner.empty())) { continue; }
        if (!ImGui::TreeNodeEx(owner.empty() ? "Render graph" : owner.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { continue; }
        for (const auto* kind : {"texture", "buffer"}) {
            if (std::none_of(resources.begin(), resources.end(), [&](const auto& resource) {
                return resource.value("subsystem", "") == owner && resource.value("kind", "") == kind;
            })) { continue; }
            if (!ImGui::TreeNodeEx(kind[0] == 't' ? "Textures" : "Buffers", ImGuiTreeNodeFlags_DefaultOpen)) { continue; }
            for (auto it = resources.begin(); it != resources.end(); ++it) {
                const auto& resource = it.value();
                if (resource.value("subsystem", "") != owner || resource.value("kind", "") != kind ||
                    (filter_[0] && it.key().find(filter_) == std::string::npos && owner.find(filter_) == std::string::npos)) { continue; }
                ImGui::PushID(it.key().c_str());
                const auto label = resource.value("displayName", resource.value("resourceName", it.key()));
                if (ImGui::Selectable(label.c_str(), selected_ == it.key())) { cancel(core); select(it.key()); }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s\n%s / %s", it.key().c_str(), resource.value("pass", "").c_str(), resource.value("checkpoint", "").c_str());
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        ImGui::TreePop();
    }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("ResourceDetail", ImVec2(0, 0));
    if (!resources.contains(selected_)) {
        cancel(core); capture_.reset(); refresh_ = true;
        ImGui::TextWrapped("Select a Texture or Buffer from an executed pass or subsystem. Resources unavailable in this execution are omitted.");
        ImGui::EndChild();
        return;
    }
    if (refresh_) { cancel(core); }
    const auto& resource = resources.at(selected_);
    if (capture_ && !capture_->artifacts.empty() &&
        capture_->artifacts[0].metadata.value("allocation", uint64_t(0)) != resource.value("allocation", uint64_t(0))) {
        cancel(core); capture_.reset(); refresh_ = true;
        status_ = "Resource allocation changed; refreshing snapshot";
    }
    ImGui::TextWrapped("%s", selected_.c_str());
    ImGui::TextDisabled("%s / %s | layout: %s", resource.value("pass", "").c_str(),
        resource.value("checkpoint", "").c_str(), resource.value("layout", "").c_str());
    if (resource.contains("subsystem")) { ImGui::TextDisabled("Subsystem: %s", resource.at("subsystem").get_ref<const std::string&>().c_str()); }
    if (resource.contains("validity")) { ImGui::TextWrapped("%s", resource.at("validity").get_ref<const std::string&>().c_str()); }
    if (resource.value("placeholder", false)) { ImGui::TextDisabled("Fallback environment: no HDRI map loaded"); }
    if (!resource.value("captureSupported", true)) { ImGui::TextWrapped("%s", resource.value("reason", "Capture unavailable").c_str()); }
    const bool texture = resource.value("kind", "") == "texture";
    if (texture) {
        const bool volume = resource.value("textureType", 0u) == static_cast<uint32_t>(render::TextureType::Texture3D);
        if (volume) {
            const int depth = std::max(1, resource.value("depth", 1));
            ImGui::Text("Texture3D | %u x %u x %d | mip 0", resource.value("width", 0u), resource.value("height", 0u), depth);
            if (ImGui::RadioButton("Cube", cube_)) {
                cube_ = true; cancel(core); capture_.reset(); refresh_ = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Z slice", !cube_)) {
                cube_ = false; cancel(core); capture_.reset(); refresh_ = true;
            }
            slice_ = std::clamp(slice_, 0, depth - 1);
            ImGui::SetNextItemWidth(320 * scale);
            if (!cube_ && ImGui::SliderInt("Slice index", &slice_, 0, depth - 1)) {
                cancel(core);
                capture_.reset();
                refresh_ = true;
            }
        } else {
            ImGui::Text("%u x %u | mip 0, layer 0", resource.value("width", 0u), resource.value("height", 0u));
        }
        ImGui::SetNextItemWidth(320 * scale);
        ImGui::InputInt4("ROI x/y/w/h", roi_.data());
        ImGui::TextDisabled("Width/height 0: remaining extent. Refresh applies the range.");
    } else {
        ImGui::Text("%llu bytes | declared stride: %u", static_cast<unsigned long long>(resource.value("size", uint64_t(0))), resource.value("structureStride", 0u));
        if (resource.value("layout", "") == "raw") {
            const auto current = bufferLayout_.empty() ? std::string("u32") : bufferLayout_;
            ImGui::SetNextItemWidth(260 * scale);
            if (ImGui::BeginCombo("Capture type", current.c_str())) {
                std::vector<std::string> types;
                const auto stride = resource.value("structureStride", 0u);
                for (const auto& [name, type] : runtime.layouts()) {
                    // Manual interpretation only. Offer whole-element layouts for a
                    // structured allocation, plus raw 32-bit scalar views.
                    if (type.stride && type.stride % 4 == 0 &&
                        ((!stride || type.stride == stride) || name == "u32" || name == "i32" || name == "f32")) {
                        types.push_back(name);
                    }
                }
                std::sort(types.begin(), types.end());
                for (const auto& name : types) {
                    if (ImGui::Selectable(name.c_str(), current == name)) {
                        cancel(core); bufferLayout_ = name; offset_ = 0; fieldPage_ = 0;
                        capture_.reset(); refresh_ = true; rawBuffer_ = false;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::TextDisabled("Manual interpretation; matching stride does not prove the shader type.");
        } else {
            ImGui::TextDisabled("Type declared by the producer; ranges are measured in typed elements.");
        }
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
                if (completed && !completed->artifacts.empty() && completed->artifacts[0].metadata.at("id") == selected_ &&
                    completed->artifacts[0].metadata.value("allocation", uint64_t(0)) == resource.value("allocation", uint64_t(0))) {
                    if (!capture_) {
                        const auto& layout = completed->artifacts[0].layout.name;
                        scalar_ = layout == "f32" ? 2 : layout == "i32" ? 1 : 0;
                    }
                    capture_ = completed;
                    imageDirty_ = texture;
                } else {
                    refresh_ = true;
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
    const uint32_t w = artifact.metadata.at("roi").at("width"), h = artifact.metadata.at("roi").at("height");
    const uint32_t d = artifact.metadata.value("sliceCount", 1u);
    const bool cube = cube_ && artifact.metadata.value("textureType", 0u) == static_cast<uint32_t>(TextureType::Texture3D);
    const uint32_t width = cube ? d + 2 * w : w, height = cube ? 2 * std::max(h, d) : h;
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
    std::memset(bytes, 0, size_t(width) * height * 4);
    const float gain = std::exp2(exposure_), span = std::max(rangeMax_ - rangeMin_, 1e-20f);
    const auto writePixel = [&](size_t destination, size_t source) {
        const auto p = pixel(artifact, source);
        bool finite = true;
        for (size_t c = 0; c < 3; ++c) {
            const float value = (p[channel_ ? size_t(channel_ - 1) : c] * gain - rangeMin_) / span;
            finite &= std::isfinite(value);
            bytes[destination * 4 + c] = std::isfinite(value) ? uint8_t(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f) : 0;
        }
        if (!finite) { bytes[destination * 4] = 255; bytes[destination * 4 + 1] = 0; bytes[destination * 4 + 2] = 255; }
        bytes[destination * 4 + 3] = 255;
    };
    if (cube) {
        // Six boundary planes packed as X/Y/Z columns, negative/positive rows.
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint32_t faceWidth = axis == 0 ? d : w, faceHeight = axis == 1 ? d : h;
            const uint32_t left = axis == 0 ? 0 : d + (axis - 1) * w;
            for (uint32_t side = 0; side < 2; ++side) {
                for (uint32_t v = 0; v < faceHeight; ++v) {
                    for (uint32_t u = 0; u < faceWidth; ++u) {
                        const uint32_t x = axis == 0 ? side * (w - 1) : u;
                        const uint32_t y = axis == 1 ? side * (h - 1) : v;
                        const uint32_t z = axis == 2 ? side * (d - 1) : axis == 0 ? u : v;
                        writePixel(size_t(side * std::max(h, d) + v) * width + left + u, (size_t(z) * h + y) * w + x);
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < size_t(w) * h; ++i) { writePixel(i, i); }
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
    const bool cube = cube_ && capture_->artifacts[0].metadata.value("textureType", 0u) == static_cast<uint32_t>(render::TextureType::Texture3D);
    ImGui::SetNextItemWidth(100); imageDirty_ |= ImGui::Combo("Channel", &channel_, "RGB\0R\0G\0B\0A\0");
    ImGui::SameLine(); ImGui::SetNextItemWidth(130); imageDirty_ |= ImGui::SliderFloat("Exposure", &exposure_, -16, 16);
    ImGui::SetNextItemWidth(110); imageDirty_ |= ImGui::DragFloat("Min", &rangeMin_, 0.01f);
    ImGui::SameLine(); ImGui::SetNextItemWidth(110); imageDirty_ |= ImGui::DragFloat("Max", &rangeMax_, 0.01f);
    if (!cube) {
        ImGui::Checkbox("Fit", &fit_); ImGui::SameLine(); ImGui::SetNextItemWidth(130); ImGui::SliderFloat("Zoom", &zoom_, 0.1f, 16.f);
    }
    ImGui::TextDisabled("Display: range/exposure + clamp; magenta = non-finite. Hover for raw values.");
    if (imageDirty_ && !rebuildImage(device, backend)) { return; }
    if (!descriptor_) { return; }
    if (cube) { drawCube(); return; }
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
        if (artifact.metadata.value("textureType", 0u) == static_cast<uint32_t>(render::TextureType::Texture3D)) {
            ImGui::Text("Voxel (%d, %d, %u)", x + roi.at("x").get<int>(), y + roi.at("y").get<int>(),
                artifact.metadata.value("slice", 0u));
        } else {
            ImGui::Text("Pixel (%d, %d)", x + roi.at("x").get<int>(), y + roi.at("y").get<int>());
        }
        if (values) {
            for (const auto& field : artifact.layout.fields) {
                const auto& value = values->at(0);
                ImGui::Text("%s: %s", field.name.c_str(), displayValue(value.is_object() ? value.at(field.name) : value).c_str());
            }
        }
        ImGui::Text("Bytes: %s", debug::hexEncode(std::span(artifact.bytes).subspan(index * artifact.layout.stride, artifact.layout.stride)).c_str());
        ImGui::EndTooltip();
    }
    ImGui::EndChild();
}

void EditorResourceInspector::drawCube()
{
    ImGui::SetNextItemWidth(130); ImGui::SliderFloat("Zoom", &cubeZoom_, 0.25f, 2.f);
    ImGui::SameLine();
    if (ImGui::Button("Reset view")) { cubeYaw_ = 0.65f; cubePitch_ = 0.45f; cubeZoom_ = 1.f; }
    ImGui::TextDisabled("Drag to rotate | boundary slices of the captured ROI | hover for voxel values");
    ImGui::BeginChild("TextureCube", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const auto origin = ImGui::GetCursorScreenPos();
    const auto available = ImGui::GetContentRegionAvail();
    const ImVec2 size(std::max(1.f, available.x), std::max(1.f, available.y));
    ImGui::InvisibleButton("Cube rotation", size);
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        cubeYaw_ = std::remainder(cubeYaw_ + ImGui::GetIO().MouseDelta.x * 0.01f, 6.2831853f);
        cubePitch_ = std::remainder(cubePitch_ + ImGui::GetIO().MouseDelta.y * 0.01f, 6.2831853f);
    }
    const bool hovered = ImGui::IsItemHovered();
    const float cy = std::cos(cubeYaw_), sy = std::sin(cubeYaw_), cp = std::cos(cubePitch_), sp = std::sin(cubePitch_);
    const auto rotate = [&](std::array<float, 3> p) {
        const float x = cy * p[0] + sy * p[2], z = -sy * p[0] + cy * p[2];
        return std::array<float, 3>{x, cp * p[1] - sp * z, sp * p[1] + cp * z};
    };
    const float scale = std::min(size.x, size.y) * 0.58f * cubeZoom_;
    const auto project = [&](std::array<float, 3> p) {
        const auto r = rotate(p);
        return ImVec2(origin.x + size.x * 0.5f + r[0] * scale, origin.y + size.y * 0.5f - r[1] * scale);
    };
    const auto& artifact = capture_->artifacts[0];
    const auto& roi = artifact.metadata.at("roi");
    const uint32_t w = roi.at("width"), h = roi.at("height"), d = artifact.metadata.value("sliceCount", 1u);
    const float atlasWidth = float(d + 2 * w), atlasHeight = float(2 * std::max(h, d));
    auto* draw = ImGui::GetWindowDrawList();
    draw->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);
    for (uint32_t axis = 0; axis < 3; ++axis) {
        for (uint32_t side = 0; side < 2; ++side) {
            std::array<float, 3> normal{};
            normal[axis] = side ? 1.f : -1.f;
            if (rotate(normal)[2] <= 0.0001f) { continue; }
            const auto position = [&](float u, float v) {
                return std::array<float, 3>{axis == 0 ? float(side) - 0.5f : u - 0.5f,
                    axis == 1 ? float(side) - 0.5f : v - 0.5f,
                    axis == 2 ? float(side) - 0.5f : axis == 0 ? u - 0.5f : v - 0.5f};
            };
            const ImVec2 corners[] = {project(position(0, 0)), project(position(1, 0)), project(position(1, 1)), project(position(0, 1))};
            const uint32_t fw = axis == 0 ? d : w, fh = axis == 1 ? d : h;
            const float left = float(axis == 0 ? 0 : d + (axis - 1) * w), top = float(side * std::max(h, d));
            const ImVec2 uv0((left + 0.5f) / atlasWidth, (top + 0.5f) / atlasHeight);
            const ImVec2 uv1((left + fw - 0.5f) / atlasWidth, (top + fh - 0.5f) / atlasHeight);
            draw->AddImageQuad(static_cast<ImTextureID>(descriptor_), corners[0], corners[1], corners[2], corners[3],
                uv0, ImVec2(uv1.x, uv0.y), uv1, ImVec2(uv0.x, uv1.y));
            draw->AddPolyline(corners, 4, IM_COL32(210, 220, 235, 255), ImDrawFlags_Closed, 1.5f);
            const char* labels[] = {"X-", "X+", "Y-", "Y+", "Z-", "Z+"};
            const auto center = project(position(0.5f, 0.5f));
            const auto textSize = ImGui::CalcTextSize(labels[axis * 2 + side]);
            const ImVec2 label(center.x - textSize.x * 0.5f, center.y - textSize.y * 0.5f);
            draw->AddRectFilled(ImVec2(label.x - 3, label.y - 2), ImVec2(label.x + textSize.x + 3, label.y + textSize.y + 2), IM_COL32(0, 0, 0, 140), 3);
            draw->AddText(label, IM_COL32_WHITE, labels[axis * 2 + side]);
            const ImVec2 a(corners[1].x - corners[0].x, corners[1].y - corners[0].y);
            const ImVec2 b(corners[3].x - corners[0].x, corners[3].y - corners[0].y);
            const ImVec2 mouse(ImGui::GetMousePos().x - corners[0].x, ImGui::GetMousePos().y - corners[0].y);
            const float determinant = a.x * b.y - a.y * b.x;
            if (!hovered || std::abs(determinant) < 1e-5f) { continue; }
            const float u = (mouse.x * b.y - mouse.y * b.x) / determinant, v = (a.x * mouse.y - a.y * mouse.x) / determinant;
            if (u < 0 || u > 1 || v < 0 || v > 1) { continue; }
            const uint32_t iu = uint32_t(u * (fw - 1) + 0.5f), iv = uint32_t(v * (fh - 1) + 0.5f);
            const uint32_t x = axis == 0 ? side * (w - 1) : iu, y = axis == 1 ? side * (h - 1) : iv;
            const uint32_t z = axis == 2 ? side * (d - 1) : axis == 0 ? iu : iv;
            const auto bytes = std::span(artifact.bytes).subspan(((size_t(z) * h + y) * w + x) * artifact.layout.stride, artifact.layout.stride);
            const auto values = debug::decodeBuffer(bytes, artifact.layout);
            ImGui::BeginTooltip();
            ImGui::Text("%s | Voxel (%u, %u, %u)", labels[axis * 2 + side], x + roi.at("x").get<uint32_t>(),
                y + roi.at("y").get<uint32_t>(), z + artifact.metadata.value("slice", 0u));
            if (values) {
                for (const auto& field : artifact.layout.fields) {
                    const auto& value = values->at(0);
                    ImGui::Text("%s: %s", field.name.c_str(), displayValue(value.is_object() ? value.at(field.name) : value).c_str());
                }
            }
            ImGui::Text("Bytes: %s", debug::hexEncode(bytes).c_str());
            ImGui::EndTooltip();
        }
    }
    draw->PopClipRect();
    ImGui::EndChild();
}

void EditorResourceInspector::drawBuffer()
{
    const auto& artifact = capture_->artifacts[0];
    const bool scalar = artifact.layout.name == "u32" || artifact.layout.name == "i32" || artifact.layout.name == "f32";
    if (!scalar) { ImGui::Checkbox("Raw 32-bit words", &rawBuffer_); }
    const bool raw = scalar || rawBuffer_;
    if (raw) {
        ImGui::SetNextItemWidth(120); ImGui::Combo("Interpret as", &scalar_, "uint32\0int32\0float32\0");
        ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::SliderInt("Words / row", &columns_, 1, 16);
    }
    ImGui::Checkbox("Hex bytes", &hex_);
    struct Column {
        const debug::DebugFieldDesc* field;
        uint32_t component;
        uint32_t size;
        std::string label;
    };
    std::vector<Column> fields;
    if (!raw) {
        for (const auto& field : artifact.layout.fields) {
            const uint32_t size = field.type == "u8" ? 1 : field.type == "f16" || field.type == "u16" ? 2 :
                field.type == "u64" || field.type == "i64" || field.type == "f64" ? 8 : 4;
            for (uint32_t i = 0; i < field.count; ++i) {
                const auto suffix = field.count == 1 ? std::string{} : "[" + std::to_string(i) + "]";
                fields.push_back({&field, i, size, field.name + suffix + " (" + field.type + ")"});
            }
        }
        if (fields.empty()) { ImGui::TextUnformatted("This layout has no displayable fields."); return; }
        const int pages = int((fields.size() + 31) / 32);
        fieldPage_ = std::clamp(fieldPage_, 0, pages - 1);
        if (pages > 1) {
            ImGui::SetNextItemWidth(160); ImGui::SliderInt("Field page", &fieldPage_, 0, pages - 1);
        }
        ImGui::TextDisabled("%s | element stride: %u bytes | hover a column for its byte offset",
            artifact.layout.name.c_str(), artifact.layout.stride);
    }
    const size_t fieldStart = size_t(fieldPage_) * 32;
    const int columns = raw ? columns_ : int(std::min(size_t(32), fields.size() - fieldStart));
    const size_t stride = raw ? 4 : artifact.layout.stride;
    const size_t elements = artifact.bytes.size() / stride;
    const int rows = int(raw ? (elements + columns - 1) / columns : elements);
    // Distinct table IDs prevent persisted raw-column widths/order from masking typed fields.
    ImGui::PushID(raw ? "raw" : artifact.layout.name.c_str());
    ImGui::PushID(fieldPage_);
    if (ImGui::BeginTable("BufferValues", columns + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, ImVec2(0, 0))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        const char* indexLabel = raw ? "Word / byte" : "Element / byte";
        ImGui::TableSetupColumn(indexLabel, ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize(indexLabel).x + 24);
        for (int c = 0; c < columns; ++c) {
            const auto label = raw ? "+" + std::to_string(c) : fields[fieldStart + c].label;
            ImGui::TableSetupColumn(label.c_str(), ImGuiTableColumnFlags_WidthFixed,
                std::max(130.f, ImGui::CalcTextSize(label.c_str()).x + 24));
        }
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (int c = 0; c <= columns; ++c) {
            ImGui::TableSetColumnIndex(c); ImGui::TableHeader(ImGui::TableGetColumnName(c));
            if (!raw && c && ImGui::IsItemHovered()) {
                const auto& column = fields[fieldStart + c - 1];
                ImGui::SetTooltip("%s | byte offset: %u | scalar size: %u | bit offset/width: %u/%u",
                    column.field->type.c_str(), column.field->offset + column.component * column.size,
                    column.size, column.field->bitOffset, column.field->bitWidth);
            }
        }
        ImGuiListClipper clipper; clipper.Begin(rows);
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                const size_t first = raw ? size_t(row) * columns : size_t(row);
                const uint64_t byteOffset = artifact.metadata.value("elementOffset", uint64_t(0)) * artifact.layout.stride + first * stride;
                ImGui::TableNextRow(); ImGui::TableNextColumn();
                ImGui::Text("%llu / 0x%llX", static_cast<unsigned long long>(byteOffset / stride), static_cast<unsigned long long>(byteOffset));
                DebugValue decoded;
                if (!raw) {
                    auto result = debug::decodeBuffer(std::span(artifact.bytes).subspan(first * stride, stride), artifact.layout);
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
                        const auto& column = fields[fieldStart + c];
                        const auto& field = *column.field;
                        if (hex_) {
                            ImGui::TextUnformatted(debug::hexEncode(std::span(artifact.bytes).subspan(
                                first * stride + field.offset + column.component * column.size, column.size)).c_str());
                        } else if (decoded.contains(field.name) || (artifact.layout.fields.size() == 1 && field.name == "value")) {
                            const auto& value = decoded.contains(field.name) ? decoded.at(field.name) : decoded;
                            const auto& component = field.count == 1 ? value : value.at(column.component);
                            if (component.is_number_float() && (field.type == "f32" || field.type == "f16")) {
                                ImGui::Text("%.9g", component.get<double>());
                            } else { ImGui::TextUnformatted(displayValue(component).c_str()); }
                        }
                    }
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopID(); ImGui::PopID();
}

void EditorResourceInspector::shutdown(render::vulkan::VulkanImGuiBackend& backend)
{
    runtime_.drain();
    if (descriptor_) { backend.removeTexture(descriptor_); descriptor_ = 0; }
    upload_.reset(); view_.reset(); image_.reset(); capture_.reset();
}

} // namespace metallic
