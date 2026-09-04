#include "Runtime/Render/Debug/RenderDebug.h"

#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
#include <cstring>

namespace metallic::render {
using debug::DebugValue;

namespace {
class GPUSceneProvider final : public debug::IDebugProvider {
public:
    explicit GPUSceneProvider(const GPUSceneSubsystem* scene) : scene_(scene) {}
    std::string_view name() const override { return "gpuScene"; }
    DebugValue schema() const override
    {
        return {{"fields", {"stats", "instances", "geometries", "drawSetGeneration", "drawSetRevision"}},
            {"recordLimit", 4096}, {"note", "CPU authored records; GPU worklists require capture"}};
    }
    void publish(DebugValue& destination) const override
    {
        if (!scene_) { destination = {{"available", false}}; return; }
        const auto& stats = scene_->stats();
        destination = {{"available", true}, {"drawSetGeneration", stats.drawSetGeneration}, {"drawSetRevision", stats.drawSetRevision},
            {"stats", {{"geometryCount", stats.geometryCount}, {"materialCount", stats.materialCount}, {"instanceCount", stats.instanceCount},
                {"viewCount", stats.viewCount}, {"invalidPrimitiveCount", stats.invalidPrimitiveCount}, {"skippedRenderNodeCount", stats.skippedRenderNodeCount}}},
            {"instances", DebugValue::array()}, {"geometries", DebugValue::array()}};
        for (const auto& instance : scene_->instances().first(std::min(scene_->instances().size(), size_t(4096)))) {
            destination["instances"].push_back({{"index", instance.id.index}, {"generation", instance.id.generation},
                {"geometryIndex", instance.geometry.index}, {"geometryGeneration", instance.geometry.generation},
                {"materialIndex", instance.material.index}, {"sourceRenderNodeIndex", instance.sourceRenderNodeIndex},
                {"visible", instance.visible}, {"bounds", {instance.localBoundingSphere.x, instance.localBoundingSphere.y, instance.localBoundingSphere.z, instance.localBoundingSphere.w}},
                {"transformRevision", instance.transformRevision}});
        }
        for (const auto& geometry : scene_->geometries().first(std::min(scene_->geometries().size(), size_t(4096)))) {
            destination["geometries"].push_back({{"index", geometry.id.index}, {"generation", geometry.id.generation},
                {"sourceRenderPrimitiveIndex", geometry.sourceRenderPrimitiveIndex}, {"vertexCount", geometry.vertexCount},
                {"indexCount", geometry.indexCount}, {"triangleCount", geometry.triangleCount}});
        }
        destination["instancesTruncated"] = scene_->instances().size() > 4096;
        destination["geometriesTruncated"] = scene_->geometries().size() > 4096;
    }
private:
    const GPUSceneSubsystem* scene_;
};

std::string textureLayout(Format format)
{
    switch (format) {
    case Format::Rgba8Unorm: case Format::Rgba8Srgb: return "RGBA8";
    case Format::Bgra8Unorm: case Format::Bgra8Srgb: return "BGRA8";
    case Format::Rgba16Sfloat: return "RGBA16F";
    case Format::Rgba32Sfloat: return "RGBA32F";
    case Format::R32Uint: return "u32";
    case Format::R32Sfloat: case Format::D32Sfloat: return "f32";
    default: return {};
    }
}

DebugValue resourceMetadata(const DebugResourceBinding& binding)
{
    DebugValue value = binding.metadata;
    value["id"] = binding.id;
    value["allocation"] = binding.allocation;
    value["captured"] = false;
    value["state"] = static_cast<uint32_t>(binding.state);
    value["offset"] = binding.offset;
    if (binding.texture) {
        const auto& desc = binding.texture->desc();
        value["kind"] = "texture";
        value["width"] = desc.width; value["height"] = desc.height;
        value["format"] = static_cast<uint32_t>(desc.format);
        value["layout"] = textureLayout(desc.format);
    } else if (binding.buffer) {
        value["kind"] = "buffer";
        value["size"] = binding.size ? binding.size : binding.buffer->desc().size - binding.offset;
        value["layout"] = binding.layout;
        value["structureStride"] = binding.buffer->desc().structureStride;
    }
    return value;
}

} // namespace

struct RenderDebugRuntime::Execution {
    debug::DebugSnapshot snapshot;
    GpuCompletionPoint completion;
    std::vector<debug::DebugCaptureRequest> requests;
    uint64_t capturedBytes = 0;
    uint64_t recordingNs = 0;
    bool ended = false;
    bool success = false;
};

struct RenderDebugRuntime::Readback {
    std::string job;
    std::shared_ptr<Execution> execution;
    std::shared_ptr<debug::DebugCapture> capture;
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::atomic<int> submission{0}; // 0 recorded, 1 submitted, -1 cancelled
};

struct RenderDebugRuntime::TaskSink final : task::ITaskEventSink {
    std::mutex mutex;
    debug::DebugCore* core = nullptr;
    void event(DebugValue value)
    {
        std::lock_guard lock(mutex);
        if (core) { core->pushEvent("tasks", std::move(value)); }
    }
    void onGraphSubmitted(const task::TaskGraphSnapshot& snapshot) override
    {
        DebugValue nodes = DebugValue::array(), edges = DebugValue::array();
        for (size_t i = 0; i < std::min(snapshot.nodes.size(), size_t(64)); ++i) {
            const auto& node = snapshot.nodes[i];
            nodes.push_back({{"id", node.handle.nodeIndex}, {"name", node.desc.name.substr(0, 128)}, {"category", node.desc.category.substr(0, 128)}});
        }
        for (size_t i = 0; i < std::min(snapshot.edges.size(), size_t(256)); ++i) {
            edges.push_back({snapshot.edges[i].prerequisite.nodeIndex, snapshot.edges[i].dependent.nodeIndex});
        }
        event({{"event", "submitted"}, {"execution", snapshot.executionId}, {"graph", snapshot.graphId}, {"name", snapshot.name.substr(0, 128)},
            {"nodes", std::move(nodes)}, {"edges", std::move(edges)}, {"nodeCount", snapshot.nodes.size()}, {"edgeCount", snapshot.edges.size()},
            {"truncated", snapshot.nodes.size() > 64 || snapshot.edges.size() > 256}});
    }
    void onTaskStateChanged(const task::TaskNodeEvent& changed) override
    {
        event({{"event", "task"}, {"execution", changed.executionId}, {"graph", changed.graphId},
            {"node", changed.node.handle.nodeIndex}, {"name", changed.node.desc.name.substr(0, 128)}, {"state", static_cast<uint32_t>(changed.node.state)},
            {"worker", changed.node.workerThreadId}, {"error", changed.node.error.substr(0, 4096)}});
    }
    void onGraphCompleted(const task::TaskGraphSnapshot& snapshot) override
    {
        event({{"event", "completed"}, {"execution", snapshot.executionId}, {"graph", snapshot.graphId}, {"status", static_cast<uint32_t>(snapshot.status)}});
    }
};

RenderDebugRuntime::RenderDebugRuntime(debug::DebugLimits limits)
    : core_(limits), server_(core_), layouts_(renderDebugLayouts())
{
    DebugValue layouts = DebugValue::object();
    for (const auto& [name, type] : layouts_) { layouts[name] = type.schema(); }
    core_.setSchema("layouts", std::move(layouts));
    core_.setSchema("gpuScene", GPUSceneProvider(nullptr).schema());
    core_.setSchema("streaming", {{"fields", {"instances"}}, {"requestSourceFrame", "Explicitly tracked; null before any consumed GPU request"}});
    core_.setSchema("engine", {{"fields", {"frame", "device"}}});
    core_.setSchema("rg", {{"fields", {"passes", "edges", "resources"}}});
    core_.setSchema("resources", {{"access", "Descriptor only; capture.batch reads contents"}});
    core_.setSchema("tasks", {{"fields", {"events", "dropped"}}});
    core_.setSchema("validation", {{"fields", {"events", "dropped"}}, {"attribution", "No inferred pass association"}});
}

RenderDebugRuntime::~RenderDebugRuntime()
{
    stop();
    drain();
}

debug::DebugResult<void> RenderDebugRuntime::start()
{
    if (auto* tasks = task::tryGetTaskSystem()) {
        taskSink_ = std::make_shared<TaskSink>(); taskSink_->core = &core_;
        taskSubscription_ = tasks->subscribe(taskSink_);
    }
    return server_.start();
}

void RenderDebugRuntime::stop()
{
    server_.stop();
    if (taskSink_) {
        { std::lock_guard lock(taskSink_->mutex); taskSink_->core = nullptr; }
        if (auto* tasks = task::tryGetTaskSystem()) { tasks->unsubscribe(taskSubscription_); }
        taskSink_.reset(); taskSubscription_ = 0;
    }
}

ValidationSink RenderDebugRuntime::validationSink()
{
    return {[](void* context, const ValidationMessage& message) noexcept {
        try {
            auto& runtime = *static_cast<RenderDebugRuntime*>(context);
            const auto copy = [](const char* text) { return text ? std::string(text).substr(0, 4096) : std::string(); };
            DebugValue objects = DebugValue::array();
            for (const auto& object : message.objects) {
                objects.push_back({{"handle", object.handle}, {"type", object.type}, {"name", copy(object.name)}});
            }
            runtime.core_.pushEvent("validation", {{"severity", message.severity}, {"type", message.type},
                {"messageId", message.messageId}, {"messageIdName", copy(message.messageIdName)}, {"message", copy(message.message)},
                {"objects", objects}, {"execution", nullptr}, {"pass", nullptr},
                {"timestampNs", uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count())}});
        } catch (...) {} // Never throw through a Vulkan validation callback.
    }, this};
}

void RenderDebugRuntime::compiled(DebugValue graph)
{
    graph_ = std::move(graph);
    core_.setGraph(graph_);
}

void RenderDebugRuntime::beginExecution(Device& device, debug::DebugEvidenceStamp evidence, RenderSubsystemHost* subsystems)
{
    poll();
    device_ = &device; subsystems_ = subsystems;
    current_ = std::make_shared<Execution>();
    current_->snapshot.evidence = std::move(evidence);
    current_->snapshot.evidence.session = core_.session();
    current_->snapshot.evidence.sample = nextSample_++;
    auto& values = current_->snapshot.values;
    values = {{"frame", {{"id", current_->snapshot.evidence.execution}, {"frameSlot", current_->snapshot.evidence.frameSlot}}},
        {"rg", graph_}, {"resources", DebugValue::object()}, {"streaming", {{"instances", DebugValue::array()}}}};
    values["engine"] = {{"frame", values["frame"]}, {"device", {{"independentCopyQueue", device.capabilities().independentCopyQueue},
        {"meshShader", device.capabilities().meshShader}, {"rayQuery", device.capabilities().rayQuery}}}};
    GPUSceneProvider(subsystems ? subsystems->get<GPUSceneSubsystem>() : nullptr).publish(values["gpuScene"]);
    current_->requests = core_.takeRequests(current_->snapshot.evidence.graph, current_->snapshot.evidence.generation);
    executions_.push_back(current_);
}

void RenderDebugRuntime::boundary(CommandBuffer& commands, std::string_view checkpoint,
    uint32_t passId, std::string_view pass, std::span<const DebugResourceBinding> resources, const DebugValue& values)
{
    if (!current_) { return; }
    auto& snapshot = current_->snapshot;
    snapshot.evidence.passId = passId; snapshot.evidence.pass = pass;
    snapshot.evidence.checkpoint = checkpoint;
    snapshot.evidence.sample = nextSample_++;
    snapshot.evidence.provenance["recordedTimeNs"] = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    if (values.contains("gpuSceneView")) { snapshot.evidence.provenance["gpuSceneView"] = values.at("gpuSceneView"); }
    if (values.contains("streaming")) {
        snapshot.evidence.provenance["streaming"] = DebugValue::array();
        for (const auto& instance : values.at("streaming").at("instances")) {
            snapshot.evidence.provenance["streaming"].push_back({{"generation", instance.at("generation")}, {"frame", instance.at("frame")}, {"requestSourceFrame", instance.at("requestSourceFrame")}});
        }
    }
    if (auto* frame = commands.frameContext()) {
        current_->completion = frame->completion();
        snapshot.evidence.provenance["submissionFrame"] = frame->frameIndex();
        snapshot.evidence.frameSlot = frame->slotIndex();
    }
    for (auto it = values.begin(); it != values.end(); ++it) { snapshot.values[it.key()] = it.value(); }
    for (const auto& binding : resources) {
        auto meta = resourceMetadata(binding);
        if (!binding.allocation) { meta["allocation"] = snapshot.evidence.generation; }
        meta["checkpoint"] = checkpoint;
        meta["pass"] = pass;
        meta["execution"] = snapshot.evidence.execution;
        meta["sample"] = snapshot.evidence.sample;
        snapshot.values["resources"][binding.id] = std::move(meta);
    }
    for (auto it = current_->requests.begin(); it != current_->requests.end();) {
        if (it->specification.at("pass").get<std::string>() == pass && it->specification.value("checkpoint", "AfterPass") == checkpoint) {
            const auto started = std::chrono::steady_clock::now();
            capture(commands, *it, resources, snapshot);
            current_->recordingNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
            it = current_->requests.erase(it);
        } else { ++it; }
    }
}

void RenderDebugRuntime::capture(CommandBuffer& commands, const debug::DebugCaptureRequest& request,
    std::span<const DebugResourceBinding> resources, const debug::DebugSnapshot& snapshot)
{
    if (core_.cancelled(request.id)) { return; }
    auto reject = [&](std::string code, std::string message) { core_.fail(request.id, {std::move(code), std::move(message)}); };
    if (!commands.frameContext()) { reject("Unsupported", "GPU capture requires a tracked RenderFrameContext"); return; }
    struct Copy {
        const DebugResourceBinding* source;
        uint64_t offset = 0;
        uint64_t bytes = 0;
        uint32_t x = 0, y = 0, width = 0, height = 0;
    };
    auto readback = std::make_shared<Readback>();
    readback->job = request.id; readback->execution = current_;
    readback->capture = std::make_shared<debug::DebugCapture>();
    readback->capture->snapshot = snapshot;
    std::vector<Copy> copies;
    uint64_t total = 0;
    try {
        for (const auto& spec : request.specification.at("resources")) {
            const std::string id = spec.at("id");
            const auto found = std::find_if(resources.begin(), resources.end(), [&](const auto& b) { return b.id == id; });
            if (found == resources.end()) { reject("NotFound", "Resource unavailable at checkpoint: " + id); return; }
            const auto& source = *found;
            if (!source.metadata.value("captureSupported", true)) {
                reject("Unsupported", source.metadata.value("reason", "No producer contract at this checkpoint")); return;
            }
            if (source.state == ResourceState::Undefined || (!source.buffer && !source.texture)) {
                reject("Unsupported", "Resource has no initialized state at checkpoint: " + id); return;
            }
            const uint64_t allocation = source.allocation ? source.allocation : snapshot.evidence.generation;
            if (spec.contains("allocation") && debug::debugUnsigned(spec.at("allocation")) != allocation) {
                reject("StaleHandle", "Resource allocation changed: " + id); return;
            }
            const std::string registered = source.texture ? textureLayout(source.texture->desc().format) : source.layout;
            const std::string layoutName = spec.value("layout", registered);
            if (!layouts_.contains(layoutName) || (layoutName != registered && registered != "raw")) {
                reject("LayoutMismatch", "Use the registered layout for " + id); return;
            }
            const auto& layout = layouts_.at(layoutName);
            if (spec.contains("layoutHash") && spec.at("layoutHash") != layout.layoutHash()) {
                reject("LayoutMismatch", "Layout hash mismatch for " + id); return;
            }
            Copy copy{&source};
            DebugValue metadata = resourceMetadata(source);
            metadata["allocation"] = allocation;
            metadata["evidence"] = snapshot.evidence.value();
            metadata["pass"] = snapshot.evidence.pass;
            metadata["checkpoint"] = snapshot.evidence.checkpoint;
            metadata["captured"] = true;
            metadata["contentVersion"] = snapshot.evidence.sample;
            metadata["layout"] = layoutName;
            metadata["layoutHash"] = layout.layoutHash();
            if (source.texture) {
                const auto& desc = source.texture->desc();
                const auto queues = static_cast<uint32_t>(commands.queueCapabilities());
                const bool graphics = (queues & static_cast<uint32_t>(QueueAccessBits::Graphics)) != 0;
                const bool compute = (queues & static_cast<uint32_t>(QueueAccessBits::Compute)) != 0;
                if ((!graphics && !compute) || (desc.format == Format::D32Sfloat && !graphics)) {
                    reject("Unsupported", "Texture ROI requires graphics/compute; depth capture requires a graphics-capable queue"); return;
                }
                if ((uint32_t(desc.usage) & uint32_t(TextureUsageBits::TransferSource)) == 0 || desc.type != TextureType::Texture2D) {
                    reject("Unsupported", "Texture does not support transfer-source capture"); return;
                }
                const auto roi = spec.value("roi", DebugValue{{"x", 0u}, {"y", 0u}, {"width", 1u}, {"height", 1u}});
                copy.x = static_cast<uint32_t>(debug::debugUnsigned(roi.at("x"), INT32_MAX));
                copy.y = static_cast<uint32_t>(debug::debugUnsigned(roi.at("y"), INT32_MAX));
                copy.width = static_cast<uint32_t>(debug::debugUnsigned(roi.at("width"), INT32_MAX));
                copy.height = static_cast<uint32_t>(debug::debugUnsigned(roi.at("height"), INT32_MAX));
                if (!copy.width || !copy.height || copy.x >= desc.width || copy.y >= desc.height ||
                    copy.width > desc.width - copy.x || copy.height > desc.height - copy.y) {
                    reject("OutOfRange", "ROI outside texture"); return;
                }
                const uint64_t pixels = uint64_t(copy.width) * copy.height;
                if (pixels > core_.limits().jobBytes / layout.stride) { reject("BudgetExceeded", "Texture ROI exceeds job budget"); return; }
                copy.bytes = pixels * layout.stride;
                metadata["roi"] = roi;
                metadata["completeCoverage"] = copy.width == desc.width && copy.height == desc.height;
                metadata["encoding"] = (layoutName == "RGBA8" || layoutName == "BGRA8") ? "unorm8-storage-values" : "native";
            } else {
                const auto& desc = source.buffer->desc();
                if ((uint32_t(desc.usage) & uint32_t(BufferUsageBits::TransferSource)) == 0) {
                    reject("Unsupported", "Buffer was not allocated for debug readback"); return;
                }
                const uint64_t available = source.size ? source.size : desc.size - source.offset;
                const uint64_t offset = debug::debugUnsigned(spec.value("offset", DebugValue(0)));
                const uint64_t count = debug::debugUnsigned(spec.value("count", DebugValue(1)));
                if (source.offset > desc.size || available > desc.size - source.offset ||
                    offset > available / layout.stride || !count || count > available / layout.stride - offset) {
                    reject("OutOfRange", "Element range exceeds buffer view"); return;
                }
                copy.offset = source.offset + offset * layout.stride;
                copy.bytes = count * layout.stride;
                if (copy.offset % 4 || copy.bytes % 4) { reject("Unsupported", "Buffer copies require four-byte aligned ranges"); return; }
                metadata["elementOffset"] = offset; metadata["elementCount"] = count;
                metadata["capacity"] = available / layout.stride;
                metadata["completeCoverage"] = offset == 0 && copy.bytes == available;
            }
            if (copy.bytes > core_.limits().jobBytes - total) { reject("BudgetExceeded", "Batch exceeds job budget"); return; }
            total += copy.bytes;
            readback->capture->artifacts.push_back({std::move(metadata), layout, {}});
            copies.push_back(copy);
        }
    } catch (const std::exception& error) { reject("InvalidArgument", error.what()); return; }
    const uint64_t metadataBytes = debug::encodeLossless(readback->capture->manifest()).dump().size();
    if (metadataBytes > core_.limits().jobBytes - total ||
        total > core_.limits().frameBytes - current_->capturedBytes || !core_.reserve(request.id, total + metadataBytes)) {
        reject("BudgetExceeded", "Capture pool or execution byte budget exceeded"); return;
    }
    // Allocate every destination before recording anything: batch preflight is atomic.
    for (const auto& copy : copies) {
        std::unique_ptr<Buffer> buffer;
        const Result result = device_->createBuffer(BufferDesc{.size = copy.bytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback, .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy}, buffer);
        if (!result) { reject("ReadbackAllocationFailed", resultToString(result)); return; }
        readback->buffers.push_back(std::move(buffer));
    }
    auto transaction = std::make_shared<SubmissionTransaction>([readback] { readback->submission = 1; }, [readback] { readback->submission = -1; });
    if (!commands.addSubmissionTransaction(transaction)) { reject("InvalidState", "Could not track capture submission"); return; }
    commands.frameContext()->retain(readback);
    for (size_t i = 0; i < copies.size(); ++i) {
        const auto& copy = copies[i];
        if (copy.source->texture) {
            TextureBarrierDesc barrier{.texture = copy.source->texture, .before = copy.source->state, .after = ResourceState::TransferSource};
            commands.barrier({.textures = &barrier, .textureCount = 1});
            commands.copyTextureToBuffer({.texture = copy.source->texture, .buffer = readback->buffers[i].get(),
                .bufferRowPitch = copy.width * readback->capture->artifacts[i].layout.stride,
                .bufferSlicePitch = static_cast<uint32_t>(copy.bytes), .textureOffsetX = static_cast<int32_t>(copy.x), .textureOffsetY = static_cast<int32_t>(copy.y),
                .width = copy.width, .height = copy.height});
            std::swap(barrier.before, barrier.after);
            commands.barrier({.textures = &barrier, .textureCount = 1});
        } else {
            BufferBarrierDesc barrier{.buffer = copy.source->buffer, .before = copy.source->state, .after = ResourceState::TransferSource, .offset = copy.offset, .size = copy.bytes};
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
            commands.copyBuffer({.source = copy.source->buffer, .destination = readback->buffers[i].get(), .sourceOffset = copy.offset, .size = copy.bytes});
            std::swap(barrier.before, barrier.after);
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
        }
    }
    current_->capturedBytes += total;
    core_.transition(request.id, "Recorded");
    readbacks_.push_back(std::move(readback));
}

void RenderDebugRuntime::endExecution(bool success)
{
    if (!current_) { return; }
    current_->ended = true; current_->success = success;
    current_->snapshot.evidence.provenance["executionComplete"] = success;
    for (const auto& request : current_->requests) { core_.fail(request.id, {"NotCaptured", "Execution did not reach requested checkpoint"}); }
    current_->snapshot.values["tasks"] = core_.events("tasks");
    current_->snapshot.values["validation"] = core_.events("validation");
    current_->snapshot.values["debugControl"] = {{"recordedBytes", current_->capturedBytes},
        {"captureRecordingNs", current_->recordingNs}, {"gpuCaptureTimeNs", nullptr}, {"gpuProbesEnabled", false}};
    if (!current_->completion.valid()) {
        current_->snapshot.evidence.provenance["completion"] = "Untracked";
        core_.publish(current_->snapshot);
    }
    current_.reset();
}

void RenderDebugRuntime::poll()
{
    core_.expire();
    for (auto it = readbacks_.begin(); it != readbacks_.end();) {
        const auto readback = *it;
        const auto& completion = readback->execution->completion;
        if (readback->submission == 1) { core_.transition(readback->job, "Submitted"); }
        if (readback->submission == -1 || completion.isCancelled()) {
            core_.fail(readback->job, {"Cancelled", "Recording was not submitted"});
        } else if (readback->submission != 1 || !completion.isComplete()) { ++it; continue; }
        else if (core_.cancelled(readback->job)) { core_.complete(readback->job, nullptr); }
        else {
            bool valid = true;
            for (size_t i = 0; i < readback->buffers.size(); ++i) {
                auto& buffer = *readback->buffers[i];
                buffer.invalidate();
                void* mapped = buffer.map();
                if (!mapped) { valid = false; break; }
                auto& bytes = readback->capture->artifacts[i].bytes;
                bytes.resize(buffer.desc().size);
                std::memcpy(bytes.data(), mapped, bytes.size());
                buffer.unmap();
            }
            if (valid) {
                readback->capture->snapshot.evidence.provenance["executionComplete"] = readback->execution->success;
                readback->capture->snapshot.evidence.provenance["completion"] = "Ready";
                core_.complete(readback->job, readback->capture);
            }
            else { core_.fail(readback->job, {"ReadbackFailed", "Could not map completed readback"}); }
        }
        readback->buffers.clear();
        readback->capture.reset();
        readback->execution.reset();
        it = readbacks_.erase(it);
    }
    for (auto it = executions_.begin(); it != executions_.end();) {
        auto& execution = **it;
        if (!execution.ended) { ++it; continue; }
        if (execution.completion.valid() && !execution.completion.isCancelled() && !execution.completion.isComplete()) { ++it; continue; }
        if (execution.success && execution.completion.valid() && execution.completion.isComplete()) {
            execution.snapshot.evidence.provenance["completion"] = "Ready";
            core_.publish(std::move(execution.snapshot));
        }
        it = executions_.erase(it);
    }
}

void RenderDebugRuntime::drain()
{
    for (const auto& execution : executions_) {
        if (execution->completion.isSubmitted()) {
            const auto result = execution->completion.wait();
            if (!result) {
                for (const auto& readback : readbacks_) {
                    core_.fail(readback->job, {"DeviceLost", resultToString(result)});
                    readback->buffers.clear(); readback->capture.reset(); readback->execution.reset();
                }
                readbacks_.clear(); executions_.clear(); return;
            }
        }
    }
    poll();
    for (const auto& readback : readbacks_) { core_.fail(readback->job, {"Cancelled", "Runtime shutdown"}); }
    readbacks_.clear(); executions_.clear(); current_.reset();
}

} // namespace metallic::render
