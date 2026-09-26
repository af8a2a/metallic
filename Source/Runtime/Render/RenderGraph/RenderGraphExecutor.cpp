#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Profiling/CpuProfile.h"
#include "Runtime/Render/Profiling/CpuPhaseTrace.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/RenderGraph/RenderGraphGpuLabels.h"
#include "Runtime/Render/Streamer/StreamingUploads.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/Profiling/NsightEvents.h"
#include "Runtime/Render/Profiling/TracyProfiler.h"
#include "Runtime/Render/Streamer/SceneResourceManager.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Render/Subsystem/BuiltinRenderSubsystems.h"
#include "Runtime/Task/TaskSystem.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metallic::render {

using namespace detail;

namespace {
std::atomic<uint64_t> nextDebugGraphId{1};
struct DebugExecutionScope {
    IRenderDebugObserver* observer = nullptr;
    bool success = false;
    ~DebugExecutionScope() { if (observer) { observer->endExecution(success); } }
};
}

namespace {

using RenderGraphLogClock = std::chrono::steady_clock;

double renderGraphElapsedMilliseconds(RenderGraphLogClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(RenderGraphLogClock::now() - begin).count();
}

class RenderGraphLogScope {
public:
    explicit RenderGraphLogScope(std::string label)
        : label_(std::move(label))
    {
        spdlog::info("[RenderGraph] Begin {}", label_);
    }

    ~RenderGraphLogScope()
    {
        spdlog::info("[RenderGraph] End {} in {:.2f} ms", label_, renderGraphElapsedMilliseconds(begin_));
    }

private:
    std::string label_;
    RenderGraphLogClock::time_point begin_ = RenderGraphLogClock::now();
};

class RenderSubsystemFrameEndScope {
public:
    explicit RenderSubsystemFrameEndScope(RenderSubsystemHost& host)
        : host_(&host)
    {
    }

    ~RenderSubsystemFrameEndScope()
    {
        if (host_ != nullptr) {
            host_->endFrame();
        }
    }

private:
    RenderSubsystemHost* host_ = nullptr;
};

uint32_t previewReadbackTexelByteSize(Format format)
{
    switch (format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::R8Uint:
    case Format::R8Sint:
        return 1;
    case Format::Rg8Unorm:
    case Format::Rg8Snorm:
    case Format::Rg8Uint:
    case Format::Rg8Sint:
    case Format::Bgra4Unorm:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Uint:
    case Format::R16Sint:
    case Format::R16Sfloat:
        return 2;
    case Format::Bgra8Unorm:
    case Format::Bgra8Srgb:
    case Format::Rgba8Unorm:
    case Format::Rgba8Snorm:
    case Format::Rgba8Srgb:
    case Format::Rgba8Uint:
    case Format::Rgba8Sint:
    case Format::Rg16Unorm:
    case Format::Rg16Snorm:
    case Format::Rg16Uint:
    case Format::Rg16Sint:
    case Format::Rg16Sfloat:
    case Format::R32Uint:
    case Format::R32Sint:
    case Format::R32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::A2R10G10B10UintPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::D32Sfloat:
        return 4;
    case Format::Rgba16Sfloat:
        return 8;
    default:
        return 0;
    }
}

float halfToFloat(uint16_t value)
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113u;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            bits = sign | (exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    return std::bit_cast<float>(bits);
}

uint8_t floatToUnorm8(float value)
{
    if (!(value > 0.0f)) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

bool convertPreviewReadback(
    Format format,
    const void* source,
    size_t pixelCount,
    std::vector<uint32_t>& destination)
{
    if (source == nullptr || destination.size() != pixelCount) {
        return false;
    }
    const uint32_t texelByteSize = previewReadbackTexelByteSize(format);
    if (texelByteSize > 0 && texelByteSize <= 4) {
        std::fill(destination.begin(), destination.end(), 0u);
        std::memcpy(destination.data(), source, pixelCount * texelByteSize);
        return true;
    }
    if (format != Format::Rgba16Sfloat) {
        return false;
    }

    const auto* sourceBytes = static_cast<const uint8_t*>(source);
    auto* destinationBytes = reinterpret_cast<uint8_t*>(destination.data());
    for (size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
        for (size_t component = 0; component < 4; ++component) {
            uint16_t half = 0;
            std::memcpy(
                &half,
                sourceBytes + (pixelIndex * 4u + component) * sizeof(uint16_t),
                sizeof(half));
            destinationBytes[pixelIndex * 4u + component] = floatToUnorm8(halfToFloat(half));
        }
    }
    return true;
}

} // namespace

RenderGraphProperties mergeRenderGraphProperties(
    const RenderGraphProperties& staticProperties,
    const RenderGraphProperties& runtimeProperties)
{
    RenderGraphProperties merged = staticProperties.is_object()
        ? staticProperties
        : RenderGraphProperties::object();
    if (!runtimeProperties.is_object()) {
        return merged;
    }

    std::function<void(RenderGraphProperties&, const RenderGraphProperties&)> mergeObject =
        [&](RenderGraphProperties& destination, const RenderGraphProperties& source) {
            for (auto iter = source.begin(); iter != source.end(); ++iter) {
                if (iter.value().is_object() &&
                    destination.contains(iter.key()) &&
                    destination[iter.key()].is_object()) {
                    mergeObject(destination[iter.key()], iter.value());
                    continue;
                }
                destination[iter.key()] = iter.value();
            }
        };
    mergeObject(merged, runtimeProperties);
    return merged;
}

struct RenderGraphExecutor::Impl {
    struct ResourceSlot {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> textureView;
        std::unique_ptr<Buffer> buffer;
        std::unique_ptr<BufferView> bufferView;
        RenderGraphResource resource;
    };

    struct SceneBinding {
        const scene::Scene* source = nullptr;
        // Capture values, not a pointer dereference during execute: a SceneDocument
        // can be replaced in-place while retaining its C++ address.
        std::array<uint64_t, 7> version{};
        std::string path;
        bool localView = false;
        bool operator==(const SceneBinding&) const = default;
    };

    static SceneBinding captureSceneBinding(const scene::Scene* source)
    {
        if (source == nullptr) { return {}; }
        return {source, {source->resourceIdentity(), source->sceneGraph().lifetimeRevision(),
            source->sceneGraph().structuralRevision(), source->contentRevision(),
            source->transformRevision(), source->visibilityRevision(), source->materialRevision()}, source->filename().generic_string()};
    }

    struct CompiledNode {
        uint32_t id = 0;
        std::string name;
        std::string type;
        RenderGraphPassKind kind = RenderGraphPassKind::Unsafe;
        QueueType queueType = QueueType::Graphics;
        RenderGraphProperties staticProperties = RenderGraphProperties::object();
        RenderGraphProperties runtimeProperties = RenderGraphProperties::object();
        RenderGraphProperties effectiveProperties = RenderGraphProperties::object();
        std::unique_ptr<RenderGraphPass> pass;
        RenderPassReflection reflection;
        RenderGraphSceneDependency sceneDependency;
        std::shared_ptr<PreparedSceneResources> preparedScene;
        SceneStreamingRequirements sceneRequirements;
        SceneBinding sceneBinding;
        uint32_t executionWidth = 0;
        uint32_t executionHeight = 0;
    };

    struct QueueCommandContext {
        Queue* queue = nullptr;
        std::unique_ptr<CommandPool> commandPool;
    };

    struct SubmissionSlot {
        RenderFrameContext frame;
        std::array<QueueCommandContext, 3> queues;
        std::vector<std::unique_ptr<CommandBuffer>> commandBuffers;
        std::array<std::vector<std::unique_ptr<CommandRecordingContext>>, 3> recordingContexts;
        explicit SubmissionSlot(uint32_t index) : frame(index) {}
    };

    struct SubmissionSegment {
        Queue* queue = nullptr;
        CommandBuffer* commandBuffer = nullptr;
        std::vector<size_t> predecessors;
        GpuCompletionPoint completion;
        bool passWork = false;
    };

    struct BindlessResourcePlan {
        std::vector<std::string> sampledImageResources;
        std::vector<std::string> bufferResources;
        std::unordered_set<std::string> sampledImageResourceSet;
        std::unordered_set<std::string> bufferResourceSet;
    };

    struct ResolvedTextureExtent {
        uint32_t width = 0;
        uint32_t height = 0;
        std::string widthSource;
        std::string heightSource;
    };

    using ResolvedTextureExtentMap = std::unordered_map<std::string, ResolvedTextureExtent>;

    struct TimerRef {
        uint32_t queue = UINT32_MAX;
        uint32_t begin = 0;
    };
    struct NodeRecording {
        CompiledNode* node = nullptr;
        RenderGraphProperties properties;
        std::vector<RenderGraphResource> resources;
        std::unique_ptr<RenderGraphExecutionContext> context;
        RenderGraphNodeExecutionStat stats;
        std::vector<SceneStreamingProfile> streaming;
        profiling::GpuProfileFrame profile;
        TimerRef passTimer;
        std::vector<TimerRef> sectionTimers;
        TimestampQueryPool* queryPool = nullptr;
        uint32_t queueIndex = 0;
        uint32_t firstQuery = 0;
        uint32_t nextQuery = 0;
        uint32_t endQuery = 0;
        bool timingValid = true;
        bool profilingOverflow = false;
        Result<> result = makeError(Error::Failure);

        TimerRef beginInterval(CommandBuffer& commands)
        {
            if (!queryPool || !timingValid) { return {}; }
            if (nextQuery + 2 > endQuery) { profilingOverflow = true; return {}; }
            TimerRef timer{queueIndex, nextQuery};
            nextQuery += 2;
            timingValid = commands.writeTimestamp(*queryPool, firstQuery + timer.begin, PipelineStageBits::BottomOfPipe).has_value();
            return timer;
        }
        void endInterval(CommandBuffer& commands, TimerRef timer)
        {
            if (!timingValid || timer.queue == UINT32_MAX) { return; }
            timingValid = commands.writeTimestamp(*queryPool, firstQuery + timer.begin + 1, PipelineStageBits::BottomOfPipe).has_value();
        }
    };
    struct GpuTimingSlot {
        uint32_t firstQuery = 0;
        uint32_t queryCount = 0;
        bool pending = false;
        GpuCompletionPoint completion;
        RenderGraphExecutionStats stats;
        profiling::GpuProfileFrame profile;
        std::array<uint32_t, 3> used{};
        TimerRef frameTimer;
        std::vector<TimerRef> nodeTimers;
        std::vector<std::vector<TimerRef>> sectionTimers;
    };

    static constexpr uint32_t kGpuTimingSlotCount = 3;

    Device* device = nullptr;
    std::unordered_map<std::string, MemoryBudgetReservation> firstFeatureReservations;
    uint32_t width = 0;
    uint32_t height = 0;
    Format defaultFormat = Format::Rgba8Unorm;
    DisplayOutputParameters displayOutput;
    HistoryResourceManager* historyResources = nullptr;
    const scene::Scene* runtimeScene = nullptr;
    RenderSubsystemHost ownedSubsystemHost;
    RenderWorld ownedWorld;
    RenderSubsystemHost* subsystemHost = &ownedSubsystemHost;
    RenderWorld* world = &ownedWorld;
    RenderView ownedView;
    RenderView* externalView = nullptr;
    bool hasOwnedView = false;
    RenderGraphProperties ownedViewProperties;
    ViewConstants frameView;
    ViewConstants previousView;
    bool hasPreviousView = false;
    GpuCompletionPoint previousViewCompletion;
    RenderGraphProperties frameCameraProperties;
    std::vector<std::shared_ptr<Buffer>> viewBuffers;
    Buffer* frameViewBuffer = nullptr;
    std::vector<CompiledNode> executionList;
    std::unordered_map<std::string, ResourceSlot> resources;
    std::unordered_map<std::string, std::string> inputAliases;
    std::unique_ptr<BindlessHeap> bindlessHeap;
    std::shared_ptr<SceneResourceSnapshot> pendingSceneResourceSnapshot;
    std::vector<std::string> requiredSubsystemIds;
    std::array<std::unique_ptr<SubmissionSlot>, 2> submissionSlots{
        std::make_unique<SubmissionSlot>(0), std::make_unique<SubmissionSlot>(1)};
    std::unordered_map<Queue*, std::unique_ptr<QueueSubmissionTracker>> submissionTrackers;
    GpuCompletionPoint lastSubmittedCompletion;
    Queue* recordingQueue = nullptr;
    GraphAccessPlan accessPlan;
    std::vector<RenderGraphResource*> accessResources;
    std::vector<GraphAccessBinding> accessBindings;
    std::array<std::unique_ptr<TimestampQueryPool>, 3> gpuTimestampQueryPools;
    std::array<GpuTimingSlot, kGpuTimingSlotCount> gpuTimingSlots;
    std::vector<RenderGraphExecutionStats> completedGpuExecutionStats;
    GpuTimingSlot* activeGpuTimingSlot = nullptr;
    uint32_t nextGpuTimingSlot = 0;
    bool activeGpuTimingValid = false;
    profiling::TracyGpuProfiler tracyGpuProfiler;
    RenderGraphExecutionStats lastExecutionStats;
    uint32_t preparationWorkerLimit = 1, preparationBatchWorkload = 1;
    uint64_t executionFrameIndex = 0;
    uint64_t profilingGeneration = 0;
    inline static std::atomic_uint64_t nextProfilingGeneration{1};
    std::vector<GpuCompletionPoint> externalCompletions;
    std::array<uint64_t, 5> recordedSceneStamp{};
    bool hasSubmittedWork = false;
    bool isCompiled = false;
    bool sceneBindingsReady = true;
    IRenderDebugObserver* debugObserver = nullptr;
    uint64_t debugGraphId = nextDebugGraphId++;
    uint64_t debugGeneration = 0;
    uint64_t debugExecutionIndex = 0;
    debug::DebugValue debugGraph;
    std::array<uint64_t, 2> debugSceneIdentity{};

    void publishDebugGraph(const RenderGraph& graph)
    {
        if (!debugObserver) { return; }
        ++debugGeneration;
        debugGraph = {{"id", std::to_string(debugGraphId)}, {"generation", debugGeneration}, {"name", graph.name()}, {"state", "Ready"},
            {"passes", debug::DebugValue::array()}, {"edges", debug::DebugValue::array()},
            {"executionOrder", debug::DebugValue::array()}, {"resources", debug::DebugValue::array()}};
        for (const auto& node : graph.nodes()) {
            const auto found = std::find_if(executionList.begin(), executionList.end(), [&](const auto& item) { return item.id == node.id; });
            debug::DebugValue value{{"id", node.id}, {"name", node.name}, {"type", node.type}, {"active", found != executionList.end()},
                {"properties", node.properties}, {"runtimeProperties", node.runtimeProperties}};
            if (found != executionList.end()) {
                value["sceneBinding"] = {{"source", static_cast<uint32_t>(found->sceneDependency.source)},
                    {"inputs", found->sceneDependency.inputs}, {"path", found->sceneBinding.path},
                    {"identity", found->sceneBinding.version[0]}, {"version", found->sceneBinding.version}};
                value["checkpoints"] = found->pass->debugCheckpoints();
                value["declaredQueue"] = static_cast<uint32_t>(found->queueType);
                value["runtimeSettings"] = debug::DebugValue::array();
                for (const auto& setting : found->pass->runtimeSettings()) {
                    value["runtimeSettings"].push_back({{"key", setting.key}, {"type", static_cast<uint32_t>(setting.type)},
                        {"default", setting.defaultValue}, {"min", setting.minValue}, {"max", setting.maxValue},
                        {"invalidateHistory", setting.invalidateHistory}, {"rebuildGraph", setting.rebuildGraph}});
                }
            }
            debugGraph["passes"].push_back(std::move(value));
        }
        for (const auto& node : executionList) { debugGraph["executionOrder"].push_back(node.name); }
        for (const auto& edge : graph.edges()) {
            debugGraph["edges"].push_back({{"srcPass", edge.srcPass}, {"srcField", edge.srcField}, {"dstPass", edge.dstPass}, {"dstField", edge.dstField}});
        }
        for (const auto& [name, slot] : resources) {
            const auto& resource = slot.resource;
            debugGraph["resources"].push_back({{"id", name}, {"allocation", debugGeneration},
                {"kind", resource.buffer ? "buffer" : "texture"}, {"size", resource.bufferDesc.size}, {"stride", resource.bufferDesc.structureStride},
                {"width", resource.desc.width}, {"height", resource.desc.height}, {"format", static_cast<uint32_t>(resource.desc.format)}});
        }
        debugSceneIdentity = runtimeScene ? std::array<uint64_t, 2>{runtimeScene->resourceIdentity(), runtimeScene->contentRevision()} : std::array<uint64_t, 2>{};
        debugObserver->compiled(debugGraph);
    }

    void beginDebugExecution(uint64_t frameIndex, uint32_t slotIndex)
    {
        if (!debugObserver) { return; }
        for (auto& description : debugGraph["passes"]) {
            const auto found = std::find_if(executionList.begin(), executionList.end(),
                [&](const auto& node) { return description.at("id") == node.id; });
            if (found != executionList.end()) {
                description["sceneBinding"] = {{"source", static_cast<uint32_t>(found->sceneDependency.source)},
                    {"inputs", found->sceneDependency.inputs}, {"path", found->sceneBinding.path},
                    {"identity", found->sceneBinding.version[0]}, {"version", found->sceneBinding.version}};
            }
        }
        const auto identity = runtimeScene ? std::array<uint64_t, 2>{runtimeScene->resourceIdentity(), runtimeScene->contentRevision()} : std::array<uint64_t, 2>{};
        if (identity != debugSceneIdentity) {
            debugSceneIdentity = identity;
            debugGraph["generation"] = ++debugGeneration;
            for (auto& resource : debugGraph["resources"]) { resource["allocation"] = debugGeneration; }
            debugObserver->compiled(debugGraph);
        }
        debugObserver->beginExecution(*device, {.graph = std::to_string(debugGraphId), .generation = debugGeneration,
            .execution = debugExecutionIndex++, .frameSlot = slotIndex,
            .provenance = {{"sceneIdentity", identity[0]}, {"sceneContentRevision", identity[1]}, {"renderFrame", frameIndex},
                {"runtimeRevision", debugGraph.value("runtimeRevision", uint64_t(0))},
                {"historyFrame", historyResources ? debug::DebugValue(historyResources->frameIndex()) : debug::DebugValue(nullptr)}}}, subsystemHost);
    }

    ~Impl() { (void)waitForSubmittedWork(UINT64_MAX); }

    RenderView* renderView() { return externalView != nullptr ? externalView : hasOwnedView ? &ownedView : nullptr; }

    Result<> prepareView(CommandBuffer& commands, uint64_t frameIndex)
    {
        auto* view = renderView();
        frameViewBuffer = nullptr;
        if (view == nullptr) { return {}; }
        uint32_t renderWidth = width, renderHeight = height;
        // Resolution negotiation already resolved scene producers before recording.
        // The view uses the scene's render extent, not an upscaler's display extent.
        for (const auto& node : executionList) {
            if (node.sceneDependency.source != RenderGraphSceneSource::None &&
                !node.sceneBinding.localView) {
                renderWidth = node.executionWidth;
                renderHeight = node.executionHeight;
                break;
            }
        }
        frameView = view->constants(frameIndex, renderWidth, renderHeight, width, height,
            hasPreviousView && previousViewCompletion.isSubmitted() ? &previousView : nullptr);
        frameCameraProperties = view->cameraProperties();
        RenderFrameContext* frame = commands.frameContext();
        const uint32_t slot = frame != nullptr ? frame->slotIndex() : 0;
        if (viewBuffers.size() <= slot) { viewBuffers.resize(slot + 1); }
        if (viewBuffers[slot] == nullptr) {
            std::unique_ptr<Buffer> buffer;
            Result<> result = device->createBuffer({.size = sizeof(ViewConstants), .structureStride = sizeof(ViewConstants),
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
            if (!result) { return result; }
            viewBuffers[slot] = std::move(buffer);
        }
        frameViewBuffer = viewBuffers[slot].get();
        void* mapped = frameViewBuffer->map();
        if (mapped == nullptr) { return makeError(Error::Failure); }
        std::memcpy(mapped, &frameView, sizeof(frameView));
        frameViewBuffer->unmap();
        frameViewBuffer->flush();
        const BufferBarrierDesc barrier{.buffer = frameViewBuffer, .before = ResourceState::Undefined, .after = ResourceState::ShaderRead};
        commands.barrier({.buffers = &barrier, .bufferCount = 1});
        if (frame != nullptr) { frame->retain(viewBuffers[slot]); }
        previousView = frameView;
        hasPreviousView = frame != nullptr;
        previousViewCompletion = frame != nullptr ? frame->completion() : GpuCompletionPoint{};
        return {};
    }

    StreamerSubsystem* streamerSubsystem() const
    {
        return subsystemHost != nullptr ? subsystemHost->get<StreamerSubsystem>() : nullptr;
    }

    std::vector<RenderSubsystemId> requiredSubsystemViews() const
    {
        std::vector<RenderSubsystemId> result;
        result.reserve(requiredSubsystemIds.size());
        for (const std::string& id : requiredSubsystemIds) {
            result.push_back(id);
        }
        return result;
    }

    RenderGraphResource* resource(std::string_view fullName)
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const RenderGraphResource* resource(std::string_view fullName) const
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const CompiledNode* compiledNode(std::string_view name) const
    {
        const auto iter = std::find_if(
            executionList.begin(),
            executionList.end(),
            [name](const CompiledNode& node) {
                return node.name == name;
            });
        return iter == executionList.end() ? nullptr : &(*iter);
    }

    const RenderGraphField* reflectedField(
        std::string_view passName,
        std::string_view fieldName,
        RenderGraphFieldVisibility visibility) const
    {
        const CompiledNode* node = compiledNode(passName);
        if (node == nullptr) {
            return nullptr;
        }
        return node->reflection.findField(fieldName, visibility);
    }

    static bool usesBindlessResource(const CompiledNode& node)
    {
        return std::any_of(
            node.reflection.fields().begin(),
            node.reflection.fields().end(),
            [](const RenderGraphField& field) {
                return isBindlessField(field);
            });
    }

    bool canReuseCompiledPasses(
        Device& newDevice,
        const RenderGraph& graph,
        const ActiveGraph& activeGraph) const
    {
        if (!isCompiled ||
            device != &newDevice ||
            executionList.size() != activeGraph.executionOrder.size()) {
            return false;
        }

        for (size_t index = 0; index < activeGraph.executionOrder.size(); ++index) {
            const std::string& passName = activeGraph.executionOrder[index];
            const RenderGraphNode* graphNode = graph.findNode(passName);
            if (graphNode == nullptr) {
                return false;
            }

            const CompiledNode& compiledNode = executionList[index];
            if (compiledNode.pass == nullptr ||
                compiledNode.id != graphNode->id ||
                compiledNode.name != graphNode->name ||
                compiledNode.type != graphNode->type ||
                compiledNode.staticProperties != graphNode->properties) {
                return false;
            }
        }

        return true;
    }

    void rebuildInputAliases(const RenderGraph& graph, const ActiveGraph& activeGraph)
    {
        inputAliases.clear();
        for (const RenderGraphEdge& edge : graph.edges()) {
            if (!activeGraph.activePasses.contains(edge.srcPass) ||
                !activeGraph.activePasses.contains(edge.dstPass)) {
                continue;
            }
            inputAliases.emplace(
                makeRenderGraphFieldName(edge.dstPass, edge.dstField),
                makeRenderGraphFieldName(edge.srcPass, edge.srcField));
        }
    }

    RenderGraphCompileContext contextForScene(
        const RenderGraphCompileContext& context, const SceneBinding& binding,
        const std::shared_ptr<PreparedSceneResources>& prepared = {}) const
    {
        auto result = context;
        result.preparedScene = prepared;
        if (binding.source != nullptr) { result.runtimeScene = binding.source; }
        if (binding.localView) { result.renderView = nullptr; }
        return result;
    }

    Result<> prepareNodeScene(CompiledNode& node, RenderGraphCompileContext& context, std::string& log)
    {
        node.sceneRequirements = node.pass->sceneResourcesRequired(context);
        auto result = streamerSubsystem()->prepareScene(node.sceneRequirements, node.effectiveProperties,
            context.runtimeScene, node.preparedScene, log, context.debugReadback);
        context.preparedScene = node.preparedScene;
        return result;
    }

    void applySceneProperties(CompiledNode& node, const SceneBinding& binding)
    {
        node.effectiveProperties = mergeRenderGraphProperties(node.staticProperties, node.runtimeProperties);
        if (binding.source != nullptr) {
            node.effectiveProperties["path"] = binding.path;
        }
        node.pass->setProperties(node.effectiveProperties);
    }

    Result<> resolveSceneBindings(std::vector<SceneBinding>& bindings, std::string& log)
    {
        bindings.assign(executionList.size(), {});
        auto* resources = streamerSubsystem();
        if (resources == nullptr) { return makeError(Error::InvalidArgument); }
        for (size_t index = 0; index < executionList.size(); ++index) {
            const auto& node = executionList[index];
            const auto& dependency = node.sceneDependency;
            auto properties = mergeRenderGraphProperties(node.staticProperties, node.runtimeProperties);
            if (dependency.source == RenderGraphSceneSource::None) {
                bindings[index].localView = renderGraphUsesLocalView(properties);
                continue;
            }
            const std::string mode = properties.value("sceneBinding", "world");
            if (mode != "world" && mode != "asset") {
                log = "Pass '" + node.name + "' has invalid sceneBinding '" + mode + "'";
                return makeError(Error::InvalidArgument);
            }
            if (dependency.source == RenderGraphSceneSource::Input) {
                const CompiledNode* producer = nullptr;
                for (const auto& input : dependency.inputs) {
                    const auto alias = inputAliases.find(makeRenderGraphFieldName(node.name, input));
                    std::string sourceName, sourceField;
                    if (alias == inputAliases.end() ||
                        !splitRenderGraphFieldName(alias->second, sourceName, sourceField)) {
                        log = "Pass '" + node.name + "' scene input '" + input + "' is not connected";
                        return makeError(Error::InvalidArgument);
                    }
                    const auto source = std::find_if(executionList.begin(), executionList.begin() + index,
                        [&](const auto& candidate) { return candidate.name == sourceName; });
                    if (source == executionList.begin() + index ||
                        (producer != nullptr && producer != &*source) ||
                        bindings[static_cast<size_t>(source - executionList.begin())].source == nullptr) {
                        log = "Pass '" + node.name + "' scene inputs must come from the same scene producer";
                        return makeError(Error::InvalidArgument);
                    }
                    producer = &*source;
                    bindings[index] = bindings[static_cast<size_t>(source - executionList.begin())];
                }
                if (producer == nullptr) {
                    log = "Pass '" + node.name + "' declares no scene inputs";
                    return makeError(Error::InvalidArgument);
                }
            } else if (mode == "world" && runtimeScene != nullptr && runtimeScene->valid()) {
                bindings[index] = captureSceneBinding(runtimeScene);
            } else {
                const scene::Scene* source = nullptr;
                // Only an explicit asset binding or an absent world may resolve path.
                Result<> result = resources->manager().resolveScene(properties, nullptr, source, log);
                if (!result) { log = "Pass '" + node.name + "' scene resolution failed: " + log; return result; }
                bindings[index] = captureSceneBinding(source);
            }
            bindings[index].localView = bindings[index].localView || renderGraphUsesLocalView(properties);
        }
        return {};
    }

    Result<> refreshFrameSceneBindings(HistoryResourceManager* history, std::string& log)
    {
        std::vector<SceneBinding> bindings;
        Result<> result = resolveSceneBindings(bindings, log);
        if (!result) { return result; }
        const bool forceRefresh = !sceneBindingsReady;
        bool changed = forceRefresh;
        for (size_t index = 0; index < executionList.size(); ++index) {
            changed |= bindings[index] != executionList[index].sceneBinding;
        }
        if (!changed) { return {}; }
        // Prepare the entire scene dependency set before recording any pass. Old
        // GPU resources remain valid until their submitted work has completed.
        result = waitForSubmittedWork(UINT64_MAX);
        if (!result) { return result; }
        sceneBindingsReady = false;
        if (history != nullptr) { history->invalidateAll(); }
        const RenderGraphCompileContext context{
            .device = device, .graphicsQueue = device->getQueue(QueueType::Graphics),
            .runtimeScene = runtimeScene,
            .renderWorld = world, .subsystemHost = subsystemHost, .width = width, .height = height,
            .defaultFormat = defaultFormat, .debugReadback = debugObserver != nullptr,
            .renderView = renderView(),
            .displayOutput = displayOutput,
        };
        for (size_t index = 0; index < executionList.size(); ++index) {
            auto& node = executionList[index];
            if (node.sceneDependency.source == RenderGraphSceneSource::None || (!forceRefresh && bindings[index] == node.sceneBinding)) { continue; }
            applySceneProperties(node, bindings[index]);
            auto nodeContext = contextForScene(context, bindings[index]);
            result = prepareNodeScene(node, nodeContext, log);
            if (!result) { return result; }
            result = node.pass->prepare(nodeContext, log);
            if (result) { result = node.pass->compile(nodeContext, log); }
            if (!result) { log = "Scene refresh failed for pass '" + node.name + "': " + log; return result; }
            // Scene refresh keeps exported graph handles stable for external users.
            if (node.pass->reflect(nodeContext) != node.reflection) {
                log = "Scene refresh changed the resource contract of pass '" + node.name + "'; rebuild the graph";
                return makeError(Error::InvalidArgument);
            }
        }
        for (size_t index = 0; index < executionList.size(); ++index) {
            executionList[index].sceneBinding = bindings[index];
        }
        sceneBindingsReady = true;
        return {};
    }

    Result<> refreshReusablePasses(
        const RenderGraph& graph,
        const ActiveGraph& activeGraph,
        const RenderGraphCompileContext& compileContext,
        std::string& log)
    {
        if (executionList.size() != activeGraph.executionOrder.size()) {
            log = validationPrefix("compiled pass list does not match active graph");
            return makeError(Error::InvalidArgument);
        }

        rebuildInputAliases(graph, activeGraph);
        for (auto& node : executionList) {
            const auto* graphNode = graph.findNode(node.id);
            if (graphNode == nullptr) { return makeError(Error::InvalidArgument); }
            node.runtimeProperties = graphNode->runtimeProperties;
        }
        std::vector<SceneBinding> sceneBindings;
        Result<> bindingResult = resolveSceneBindings(sceneBindings, log);
        if (!bindingResult) { return bindingResult; }
        for (size_t index = 0; index < activeGraph.executionOrder.size(); ++index) {
            const std::string& passName = activeGraph.executionOrder[index];
            const RenderGraphNode* graphNode = graph.findNode(passName);
            if (graphNode == nullptr) {
                log = validationPrefix(std::string("active pass is missing '") + passName + "'");
                return makeError(Error::InvalidArgument);
            }

            CompiledNode& compiledNode = executionList[index];
            if (compiledNode.pass == nullptr ||
                compiledNode.id != graphNode->id ||
                compiledNode.name != graphNode->name ||
                compiledNode.type != graphNode->type ||
                compiledNode.staticProperties != graphNode->properties) {
                log = validationPrefix("compiled pass reuse rejected by graph mismatch");
                return makeError(Error::InvalidArgument);
            }

            applySceneProperties(compiledNode, sceneBindings[index]);
            auto nodeContext = contextForScene(compileContext, sceneBindings[index]);
            std::string prepareLog;
            Result<> prepareResult = prepareNodeScene(compiledNode, nodeContext, prepareLog);
            if (!prepareResult) { log = prepareLog; return prepareResult; }
            prepareResult = compiledNode.pass->prepare(nodeContext, prepareLog);
            if (prepareResult && sceneBindings[index] != compiledNode.sceneBinding) {
                prepareResult = compiledNode.pass->compile(nodeContext, prepareLog);
            }
            if (!prepareResult) {
                log = "RenderGraph prepare failed for pass '" + compiledNode.name + "' (" +
                    compiledNode.type + ")";
                if (!prepareLog.empty()) {
                    log += ": " + prepareLog;
                }
                return prepareResult;
            }
            compiledNode.kind = compiledNode.pass->kind();
            compiledNode.queueType = compiledNode.pass->queueType();
            compiledNode.reflection = compiledNode.pass->reflect(nodeContext);
            compiledNode.sceneBinding = sceneBindings[index];
        }

        return {};
    }

    BindlessResourcePlan collectBindlessResourcePlan() const
    {
        BindlessResourcePlan plan;
        for (const CompiledNode& node : executionList) {
            for (const RenderGraphField& field : node.reflection.fields()) {
                if (!isBindlessField(field)) {
                    continue;
                }

                const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
                std::string resourceName = fullName;
                if (field.visibility == RenderGraphFieldVisibility::Input) {
                    const auto alias = inputAliases.find(fullName);
                    if (alias == inputAliases.end()) {
                        continue;
                    }
                    resourceName = alias->second;
                }

                if (isBindlessSampledImageField(field) &&
                    plan.sampledImageResourceSet.insert(resourceName).second) {
                    plan.sampledImageResources.push_back(std::move(resourceName));
                    continue;
                }
                if (isBindlessBufferField(field) &&
                    plan.bufferResourceSet.insert(resourceName).second) {
                    plan.bufferResources.push_back(std::move(resourceName));
                }
            }
        }
        return plan;
    }

    Result<> resolveTextureOutputExtents(
        const RenderGraph& graph,
        const ActiveGraph& activeGraph,
        ResolvedTextureExtentMap& resolvedExtents,
        std::string& log) const
    {
        resolvedExtents.clear();
        std::unordered_map<std::string, std::vector<std::string>> nodeTextureOutputs;
        for (const CompiledNode& node : executionList) {
            std::vector<std::string>& outputs = nodeTextureOutputs[node.name];
            for (const RenderGraphField& field : node.reflection.fields()) {
                if (field.visibility != RenderGraphFieldVisibility::Output ||
                    field.resourceType != RenderGraphResourceType::Texture2D) {
                    continue;
                }
                const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
                outputs.push_back(fullName);
                resolvedExtents.emplace(
                    fullName,
                    ResolvedTextureExtent{
                        .width = field.width,
                        .height = field.height,
                        .widthSource = field.width != 0 ? std::string("output '") + fullName + "'" : std::string{},
                        .heightSource = field.height != 0 ? std::string("output '") + fullName + "'" : std::string{},
                    });
            }
        }

        const auto constrainDimension = [&log](
                                            uint32_t& value,
                                            std::string& source,
                                            uint32_t constraint,
                                            const std::string& constraintSource,
                                            const char* dimension,
                                            bool& changed) {
            if (constraint == 0) {
                return true;
            }
            if (value == 0) {
                value = constraint;
                source = constraintSource;
                changed = true;
                return true;
            }
            if (value == constraint) {
                return true;
            }
            log = validationPrefix(
                std::string("texture ") + dimension + " extent conflict: " +
                source + " requests " + std::to_string(value) + ", " +
                constraintSource + " requests " + std::to_string(constraint));
            return false;
        };
        const auto linkDimension = [&constrainDimension](
                                       uint32_t& left,
                                       std::string& leftSource,
                                       uint32_t& right,
                                       std::string& rightSource,
                                       const std::string& relation,
                                       const char* dimension,
                                       bool& changed) {
            if (left != 0) {
                const std::string source = leftSource.empty() ? relation : leftSource;
                return constrainDimension(
                    right,
                    rightSource,
                    left,
                    source,
                    dimension,
                    changed);
            }
            if (right != 0) {
                const std::string source = rightSource.empty() ? relation : rightSource;
                return constrainDimension(
                    left,
                    leftSource,
                    right,
                    source,
                    dimension,
                    changed);
            }
            return true;
        };

        const size_t maximumIterations = resolvedExtents.size() * 2u + 2u;
        for (size_t iteration = 0; iteration < maximumIterations; ++iteration) {
            const bool forceRefresh = !sceneBindingsReady;
        bool changed = forceRefresh;

            for (const auto& [nodeName, outputNames] : nodeTextureOutputs) {
                if (outputNames.size() < 2) {
                    continue;
                }
                ResolvedTextureExtent& anchor = resolvedExtents.at(outputNames.front());
                for (size_t index = 1; index < outputNames.size(); ++index) {
                    ResolvedTextureExtent& output = resolvedExtents.at(outputNames[index]);
                    const std::string relation = std::string("pass '") + nodeName + "' output extent";
                    if (!linkDimension(
                            anchor.width,
                            anchor.widthSource,
                            output.width,
                            output.widthSource,
                            relation,
                            "width",
                            changed) ||
                        !linkDimension(
                            anchor.height,
                            anchor.heightSource,
                            output.height,
                            output.heightSource,
                            relation,
                            "height",
                            changed)) {
                        return makeError(Error::InvalidArgument);
                    }
                }
            }

            for (const RenderGraphEdge& edge : graph.edges()) {
                if (!activeGraph.activePasses.contains(edge.srcPass) ||
                    !activeGraph.activePasses.contains(edge.dstPass)) {
                    continue;
                }
                const std::string sourceName = makeRenderGraphFieldName(edge.srcPass, edge.srcField);
                auto sourceIter = resolvedExtents.find(sourceName);
                if (sourceIter == resolvedExtents.end()) {
                    continue;
                }
                const RenderGraphField* destinationField = reflectedField(
                    edge.dstPass,
                    edge.dstField,
                    RenderGraphFieldVisibility::Input);
                if (destinationField == nullptr ||
                    destinationField->resourceType != RenderGraphResourceType::Texture2D) {
                    continue;
                }

                ResolvedTextureExtent& source = sourceIter->second;
                const std::string destinationName = makeRenderGraphFieldName(
                    edge.dstPass,
                    edge.dstField);
                const auto destinationOutputs = nodeTextureOutputs.find(edge.dstPass);
                const bool destinationHasTextureOutputs =
                    destinationOutputs != nodeTextureOutputs.end() &&
                    !destinationOutputs->second.empty();

                if (destinationField->width != 0) {
                    if (!constrainDimension(
                            source.width,
                            source.widthSource,
                            destinationField->width,
                            std::string("input '") + destinationName + "'",
                            "width",
                            changed)) {
                        return makeError(Error::InvalidArgument);
                    }
                } else if (!destinationField->matchOutputExtent) {
                    // Resampling inputs keep their producer's independent extent.
                } else if (!destinationHasTextureOutputs) {
                    if (!constrainDimension(
                            source.width,
                            source.widthSource,
                            width,
                            std::string("graph width for input '") + destinationName + "'",
                            "width",
                            changed)) {
                        return makeError(Error::InvalidArgument);
                    }
                } else {
                    for (const std::string& outputName : destinationOutputs->second) {
                        ResolvedTextureExtent& destination = resolvedExtents.at(outputName);
                        if (!linkDimension(
                                source.width,
                                source.widthSource,
                                destination.width,
                                destination.widthSource,
                                std::string("implicit input '") + destinationName + "'",
                                "width",
                                changed)) {
                            return makeError(Error::InvalidArgument);
                        }
                    }
                }

                if (destinationField->height != 0) {
                    if (!constrainDimension(
                            source.height,
                            source.heightSource,
                            destinationField->height,
                            std::string("input '") + destinationName + "'",
                            "height",
                            changed)) {
                        return makeError(Error::InvalidArgument);
                    }
                } else if (!destinationField->matchOutputExtent) {
                    // Resampling inputs keep their producer's independent extent.
                } else if (!destinationHasTextureOutputs) {
                    if (!constrainDimension(
                            source.height,
                            source.heightSource,
                            height,
                            std::string("graph height for input '") + destinationName + "'",
                            "height",
                            changed)) {
                        return makeError(Error::InvalidArgument);
                    }
                } else {
                    for (const std::string& outputName : destinationOutputs->second) {
                        ResolvedTextureExtent& destination = resolvedExtents.at(outputName);
                        if (!linkDimension(
                                source.height,
                                source.heightSource,
                                destination.height,
                                destination.heightSource,
                                std::string("implicit input '") + destinationName + "'",
                                "height",
                                changed)) {
                            return makeError(Error::InvalidArgument);
                        }
                    }
                }
            }

            if (!changed) {
                break;
            }
            if (iteration + 1 == maximumIterations) {
                log = validationPrefix("texture extent constraint resolution did not converge");
                return makeError(Error::Failure);
            }
        }

        for (auto& [fullName, extent] : resolvedExtents) {
            (void)fullName;
            if (extent.width == 0) {
                extent.width = width;
                extent.widthSource = "graph default width";
            }
            if (extent.height == 0) {
                extent.height = height;
                extent.heightSource = "graph default height";
            }
        }
        return {};
    }

    Result<> resolveNodeExecutionExtents(std::string& log)
    {
        for (CompiledNode& node : executionList) {
            node.executionWidth = width;
            node.executionHeight = height;
            bool foundTextureOutput = false;
            for (const RenderGraphField& field : node.reflection.fields()) {
                if (field.visibility != RenderGraphFieldVisibility::Output ||
                    field.resourceType != RenderGraphResourceType::Texture2D) {
                    continue;
                }

                const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
                const RenderGraphResource* output = resource(fullName);
                if (output == nullptr || output->texture == nullptr) {
                    log = validationPrefix(std::string("texture output resource is missing '") + fullName + "'");
                    return makeError(Error::InvalidArgument);
                }
                if (!foundTextureOutput) {
                    node.executionWidth = output->desc.width;
                    node.executionHeight = output->desc.height;
                    foundTextureOutput = true;
                    continue;
                }
                if (node.executionWidth != output->desc.width ||
                    node.executionHeight != output->desc.height) {
                    log = validationPrefix(
                        std::string("pass '") + node.name +
                        "' has texture outputs with different extents; split the pass or use matching output extents");
                    return makeError(Error::InvalidArgument);
                }
            }
        }
        return {};
    }

    Result<> allocateGraphResources(
        Device& graphDevice,
        const RenderGraph& graph,
        const ActiveGraph& activeGraph,
        const RenderGraphCompileOptions& options,
        const BindlessResourcePlan& bindlessPlan,
        std::string& log)
    {
        resources.clear();
        bindlessHeap.reset();

        ResolvedTextureExtentMap resolvedTextureExtents;
        Result<> extentConstraintResult = resolveTextureOutputExtents(
            graph,
            activeGraph,
            resolvedTextureExtents,
            log);
        if (!extentConstraintResult) {
            return extentConstraintResult;
        }

        for (const CompiledNode& node : executionList) {
            for (const RenderGraphField& field : node.reflection.fields()) {
                if (field.visibility != RenderGraphFieldVisibility::Output) {
                    continue;
                }

                const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
                ResourceSlot slot;

                if (field.resourceType == RenderGraphResourceType::Texture2D) {
                    const auto resolvedExtent = resolvedTextureExtents.find(fullName);
                    if (resolvedExtent == resolvedTextureExtents.end()) {
                        log = validationPrefix(
                            std::string("resolved texture extent is missing '") + fullName + "'");
                        return makeError(Error::InvalidArgument);
                    }

                    TextureUsageBits usage = textureUsageForField(field);
                    if (usage == TextureUsageBits::None) {
                        usage = TextureUsageBits::ColorAttachment;
                    }
                    if (field.presentationOutput || isOutputMarked(graph, fullName) ||
                        options.enablePreviewOutputAccess) {
                        usage = addTextureUsage(usage, TextureUsageBits::TransferSource);
                        usage = addTextureUsage(usage, TextureUsageBits::Sampled);
                    }
                    if (debugObserver) { usage = addTextureUsage(usage, TextureUsageBits::TransferSource); }
                    for (const RenderGraphEdge& edge : graph.edges()) {
                        if (edge.srcPass != node.name ||
                            edge.srcField != field.name ||
                            !activeGraph.activePasses.contains(edge.dstPass)) {
                            continue;
                        }

                        const RenderGraphField* dstField = reflectedField(
                            edge.dstPass,
                            edge.dstField,
                            RenderGraphFieldVisibility::Input);
                        if (dstField != nullptr) {
                            usage = addTextureUsage(usage, textureUsageForField(*dstField));
                        }
                    }

                    TextureDesc desc{
                        .type = TextureType::Texture2D,
                        .usage = usage,
                        .format = resolveFormat(field.format, defaultFormat),
                        .width = resolvedExtent->second.width,
                        .height = resolvedExtent->second.height,
                        .depth = 1,
                        .mipCount = 1,
                        .layerCount = 1,
                        .memoryLocation = MemoryLocation::Device,
                        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
                    };

                    Result<> result = graphDevice.createTexture(desc).transform([&](auto rhiValue) { slot.texture = std::move(rhiValue); });
                    if (!result || slot.texture == nullptr) {
                        log += resultMessage(std::string("createTexture(") + fullName + ")", result);
                        log += '\n';
                        return result ? makeError(Error::Failure) : result;
                    }
                    result = graphDevice.createTextureView(*slot.texture,
                        TextureViewDesc{
                            .format = desc.format,
                            .baseMip = 0,
                            .mipCount = 1,
                            .baseLayer = 0,
                            .layerCount = 1,
                        }).transform([&](auto rhiValue) { slot.textureView = std::move(rhiValue); });
                    if (!result || slot.textureView == nullptr) {
                        log += resultMessage(std::string("createTextureView(") + fullName + ")", result);
                        log += '\n';
                        return result ? makeError(Error::Failure) : result;
                    }
                    slot.resource = RenderGraphResource{
                        .type = RenderGraphResourceType::Texture2D,
                        .texture = slot.texture.get(),
                        .view = slot.textureView.get(),
                        .desc = desc,
                        .colorEncoding = field.colorEncoding,
                        .state = ResourceState::Undefined,
                    };
                } else {
                    BufferUsageBits usage = bufferUsageForField(field);
                    if (usage == BufferUsageBits::None) {
                        usage = BufferUsageBits::Storage;
                    }
                    BufferViewType viewType = bufferViewTypeForField(field);
                    for (const RenderGraphEdge& edge : graph.edges()) {
                        if (edge.srcPass != node.name ||
                            edge.srcField != field.name ||
                            !activeGraph.activePasses.contains(edge.dstPass)) {
                            continue;
                        }

                        const RenderGraphField* dstField = reflectedField(
                            edge.dstPass,
                            edge.dstField,
                            RenderGraphFieldVisibility::Input);
                        if (dstField == nullptr) {
                            continue;
                        }
                        usage = addBufferUsage(usage, bufferUsageForField(*dstField));
                        if (dstField->access == RenderGraphResourceAccess::BufferStorageReadWrite ||
                            dstField->access == RenderGraphResourceAccess::BufferStorageWrite) {
                            viewType = dstField->structureStride == 0
                                ? BufferViewType::ReadWriteRaw
                                : BufferViewType::ReadWriteStructured;
                        }
                    }

                    const bool markedBufferOutput = isOutputMarked(graph, fullName);
                    if (debugObserver) { usage = addBufferUsage(usage, BufferUsageBits::TransferSource); }
                    if (markedBufferOutput || options.enablePreviewOutputAccess) {
                        usage = addBufferUsage(usage, BufferUsageBits::TransferSource);
                    }
                    BufferDesc desc{
                        .size = field.size,
                        .structureStride = field.structureStride,
                        .usage = usage,
                        .memoryLocation = markedBufferOutput
                            ? MemoryLocation::HostReadback
                            : field.memoryLocation,
                        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
                    };

                    Result<> result = graphDevice.createBuffer(desc).transform([&](auto rhiValue) { slot.buffer = std::move(rhiValue); });
                    if (!result || slot.buffer == nullptr) {
                        log += resultMessage(std::string("createBuffer(") + fullName + ")", result);
                        log += '\n';
                        return result ? makeError(Error::Failure) : result;
                    }

                    BufferViewDesc viewDesc{
                        .type = viewType,
                        .offset = 0,
                        .size = desc.size,
                        .structureStride = desc.structureStride,
                    };
                    const bool needsBindlessBuffer = bindlessPlan.bufferResourceSet.contains(fullName);
                    if (needsBindlessBuffer) {
                        result = graphDevice.createBufferView(*slot.buffer, viewDesc).transform([&](auto rhiValue) { slot.bufferView = std::move(rhiValue); });
                        if (!result || slot.bufferView == nullptr) {
                            log += resultMessage(std::string("createBufferView(") + fullName + ")", result);
                            log += '\n';
                            return result ? makeError(Error::Failure) : result;
                        }
                        viewDesc = slot.bufferView->desc();
                    }

                    slot.resource = RenderGraphResource{
                        .type = RenderGraphResourceType::Buffer,
                        .buffer = slot.buffer.get(),
                        .bufferView = slot.bufferView.get(),
                        .bufferDesc = desc,
                        .bufferViewDesc = viewDesc,
                        .state = ResourceState::Undefined,
                    };
                }

                resources.emplace(fullName, std::move(slot));
            }
        }

        Result<> extentResult = resolveNodeExecutionExtents(log);
        if (!extentResult) {
            return extentResult;
        }

        if (!bindlessPlan.sampledImageResources.empty() || !bindlessPlan.bufferResources.empty()) {
            Result<> result = graphDevice.createBindlessHeap(BindlessHeapDesc{
                    .maxSampledImages = static_cast<uint32_t>(bindlessPlan.sampledImageResources.size()),
                    .maxBuffers = static_cast<uint32_t>(bindlessPlan.bufferResources.size()),
                }).transform([&](auto rhiValue) { bindlessHeap = std::move(rhiValue); });
            if (!result || bindlessHeap == nullptr) {
                log += resultMessage("createBindlessHeap(RenderGraph)", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }

            for (const std::string& fullName : bindlessPlan.sampledImageResources) {
                RenderGraphResource* graphResource = resource(fullName);
                if (graphResource == nullptr || graphResource->view == nullptr) {
                    log = validationPrefix(std::string("bindless sampled image resource is missing '") + fullName + "'");
                    return makeError(Error::InvalidArgument);
                }

                BindlessHandle handle;
                result = bindlessHeap->allocateSampledImage().transform([&](auto rhiValue) { handle = std::move(rhiValue); });
                if (!result) {
                    log += resultMessage(std::string("allocateSampledImage(") + fullName + ")", result);
                    log += '\n';
                    return result;
                }

                result = bindlessHeap->writeSampledImage(
                    handle,
                    *graphResource->view,
                    ResourceState::ShaderRead);
                if (!result) {
                    log += resultMessage(std::string("writeSampledImage(") + fullName + ")", result);
                    log += '\n';
                    return result;
                }
                graphResource->bindlessHandle = handle;
                graphResource->sampledImageBindlessHandle = handle;
            }

            for (const std::string& fullName : bindlessPlan.bufferResources) {
                RenderGraphResource* graphResource = resource(fullName);
                if (graphResource == nullptr || graphResource->bufferView == nullptr) {
                    log = validationPrefix(std::string("bindless buffer resource is missing '") + fullName + "'");
                    return makeError(Error::InvalidArgument);
                }

                BindlessHandle handle;
                result = bindlessHeap->allocateBuffer().transform([&](auto rhiValue) { handle = std::move(rhiValue); });
                if (!result) {
                    log += resultMessage(std::string("allocateBuffer(") + fullName + ")", result);
                    log += '\n';
                    return result;
                }

                result = bindlessHeap->writeBufferView(handle, *graphResource->bufferView);
                if (!result) {
                    log += resultMessage(std::string("writeBufferView(") + fullName + ")", result);
                    log += '\n';
                    return result;
                }
                graphResource->bindlessHandle = handle;
            }
        }

        return {};
    }

    Result<> rebuildGraphResources(
        Device& graphDevice,
        const RenderGraph& graph,
        const ActiveGraph& activeGraph,
        const RenderGraphCompileOptions& options,
        std::string& log)
    {
        rebuildInputAliases(graph, activeGraph);
        BindlessResourcePlan bindlessPlan = collectBindlessResourcePlan();
        if ((!bindlessPlan.sampledImageResources.empty() || !bindlessPlan.bufferResources.empty()) &&
            !graphDevice.capabilities().bindlessDescriptorHeap) {
            log = "RenderGraph compile failed: bindless resources require "
                "DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }

        return allocateGraphResources(graphDevice, graph, activeGraph, options, bindlessPlan, log);
    }

    static size_t queueContextIndex(QueueType type)
    {
        switch (type) {
        case QueueType::Graphics:
            return 0;
        case QueueType::Compute:
            return 1;
        case QueueType::Copy:
            return 2;
        }

        return 0;
    }

    Result<> waitForSubmittedWork(uint64_t timeoutNanoseconds)
    {
        profiling::CpuPhase phase("drain.externalWait");
        const auto begin = std::chrono::steady_clock::now();
        auto remaining = [&]() {
            if (timeoutNanoseconds == UINT64_MAX) { return UINT64_MAX; }
            const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count());
            return timeoutNanoseconds - std::min(timeoutNanoseconds, elapsed);
        };
        for (const GpuCompletionPoint& completion : externalCompletions) {
            Result<> result = completion.wait(remaining());
            if (!result) { return result; }
        }
        externalCompletions.clear();
        phase.next("drain.slotWait");
        for (const auto& slot : submissionSlots) {
            Result<> result = slot->frame.wait(remaining());
            if (!result) { return result; }
        }
        // Descriptor heap reserved ranges remain associated with executable
        // command buffers even after GPU completion. Drop completed recordings
        // before recompile/scene refresh can recycle their heap addresses.
        phase.next("drain.releaseCommands");
        for (const auto& slot : submissionSlots) {
            slot->commandBuffers.clear();
            for (auto& contexts : slot->recordingContexts) {
                for (auto& context : contexts) {
                    auto result = context->reset();
                    if (!result) { return result; }
                }
            }
        }
        hasSubmittedWork = false;
        // Also flush the last frames at shutdown/recompile; no additional wait
        // is introduced beyond the caller's existing completion wait above.
        phase.next("drain.resolveTimings");
        (void)resolveGpuTimings();
        // Completion alone does not release retained scene/stream owners. A
        // graph switch must retire them before allocating the replacement graph,
        // rather than waiting for begin() on a slot the new graph may never reach.
        phase.next("drain.releaseResources");
        for (const auto& slot : submissionSlots) {
            Result<> result = slot->frame.reset();
            if (!result) { return result; }
        }
        return {};
    }

    uint32_t timingQueueIndex() const
    {
        const auto type = recordingQueue ? recordingQueue->type() : QueueType::Graphics;
        return type == QueueType::Compute ? 1u : type == QueueType::Copy ? 2u : 0u;
    }

    void initializeGpuTiming(Device& graphDevice)
    {
        profilingGeneration = nextProfilingGeneration.fetch_add(1, std::memory_order_relaxed);
        for (auto& pool : gpuTimestampQueryPools) { pool.reset(); }
        gpuTimingSlots = {};
        completedGpuExecutionStats.clear();
        activeGpuTimingSlot = nullptr;
        nextGpuTimingSlot = 0;
        activeGpuTimingValid = false;
        if (executionList.empty() || !graphDevice.capabilities().timestampQueries) { return; }
        // Bounded dynamic scope space per queue, in addition to every pass and frame.
        const auto parallelNodes = std::count_if(executionList.begin(), executionList.end(), [](const auto& node) {
            return node.pass->cpuRecordingPolicy() == CpuRecordingPolicy::ParallelJoined;
        });
        // Parallel nodes own a bounded range for their 256 nested scopes. Only
        // actual intervals are resolved; unused reserved queries stay unwritten.
        const uint64_t perSlot = (executionList.size() + 1ull) * 2ull + 512ull * (parallelNodes + 1);
        if (perSlot * kGpuTimingSlotCount > UINT32_MAX) { return; }
        constexpr std::array types{QueueType::Graphics, QueueType::Compute, QueueType::Copy};
        for (uint32_t i = 0; i < types.size(); ++i) {
            auto* queue = graphDevice.getQueue(types[i]);
            if (!queue || queue->timestampValidBits() == 0) { continue; }
            const auto result = graphDevice.createTimestampQueryPool(*queue,
                {.queryCount = uint32_t(perSlot * kGpuTimingSlotCount)}).transform([&](auto rhiValue) { gpuTimestampQueryPools[i] = std::move(rhiValue); });
            if (!result) { gpuTimestampQueryPools[i].reset(); }
        }
        for (uint32_t i = 0; i < kGpuTimingSlotCount; ++i) {
            gpuTimingSlots[i].firstQuery = i * uint32_t(perSlot);
            gpuTimingSlots[i].queryCount = uint32_t(perSlot);
        }
    }

    TimerRef beginInterval(CommandBuffer& commands)
    {
        if (!activeGpuTimingSlot || !activeGpuTimingValid) { return {}; }
        auto& slot = *activeGpuTimingSlot;
        const uint32_t queue = timingQueueIndex();
        auto* pool = gpuTimestampQueryPools[queue].get();
        if (!pool) { return {}; }
        auto& used = slot.used[queue];
        if (used + 2 > slot.queryCount) { lastExecutionStats.profilingOverflow = true; return {}; }
        TimerRef timer{queue, used};
        used += 2;
        if (!commands.writeTimestamp(*pool, slot.firstQuery + timer.begin, PipelineStageBits::BottomOfPipe)) {
            activeGpuTimingValid = false; return {};
        }
        return timer;
    }

    void endInterval(CommandBuffer& commands, TimerRef timer)
    {
        if (!activeGpuTimingSlot || !activeGpuTimingValid || timer.queue == UINT32_MAX) { return; }
        if (!commands.writeTimestamp(*gpuTimestampQueryPools[timer.queue],
            activeGpuTimingSlot->firstQuery + timer.begin + 1, PipelineStageBits::BottomOfPipe)) {
            activeGpuTimingValid = false;
        }
    }

    Result<> resolveGpuTimings()
    {
        std::array<GpuTimingSlot*, kGpuTimingSlotCount> ordered;
        for (size_t i = 0; i < ordered.size(); ++i) { ordered[i] = &gpuTimingSlots[i]; }
        std::sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) { return a->stats.executionId < b->stats.executionId; });
        for (auto* entry : ordered) {
            auto& slot = *entry;
            if (!slot.pending) { continue; }
            if (slot.completion.isCancelled()) {
                slot.pending = false; slot.stats = {}; slot.profile = {}; continue;
            }
            if (!slot.completion.valid() || !slot.completion.isComplete()) { break; }
            std::array<std::vector<TimestampQueryResult>, 3> values;
            bool ready = true;
            for (uint32_t q = 0; q < values.size(); ++q) {
                if (!slot.used[q]) { continue; }
                values[q].resize(slot.used[q]);
                const auto result = gpuTimestampQueryPools[q]->readResults(slot.firstQuery, slot.used[q], values[q].data());
                if (!result) { return result; }
            }
            const auto intervalReady = [&](TimerRef timer) {
                return timer.queue == UINT32_MAX ||
                    (values[timer.queue][timer.begin].available && values[timer.queue][timer.begin + 1].available);
            };
            ready &= intervalReady(slot.frameTimer);
            for (auto timer : slot.nodeTimers) { ready &= intervalReady(timer); }
            for (const auto& sections : slot.sectionTimers) {
                for (auto timer : sections) { ready &= intervalReady(timer); }
            }
            if (!ready) { break; }
            const auto resolve = [&](TimerRef timer, double& ms, bool& available) {
                if (timer.queue == UINT32_MAX) { return; }
                const auto& data = values[timer.queue];
                ms = gpuTimestampQueryPools[timer.queue]->durationMilliseconds(data[timer.begin].value, data[timer.begin + 1].value);
                available = true;
            };
            resolve(slot.frameTimer, slot.stats.gpuMilliseconds, slot.stats.gpuTimingAvailable);
            for (size_t n = 0; n < slot.stats.nodes.size(); ++n) {
                auto& node = slot.stats.nodes[n];
                resolve(slot.nodeTimers[n], node.gpuMilliseconds, node.gpuTimingAvailable);
                for (size_t i = 0; i < node.sections.size(); ++i) {
                    resolve(slot.sectionTimers[n][i], node.sections[i].gpuMilliseconds, node.sections[i].gpuTimingAvailable);
                }
            }
            // Preserve the existing Tracy graphics envelope; never mix queue clocks
            // or misrepresent concurrent compute intervals as serialized graphics zones.
            if (slot.frameTimer.queue == 0 && std::all_of(slot.nodeTimers.begin(), slot.nodeTimers.end(),
                [](auto timer) { return timer.queue == 0; })) {
                std::vector<TimestampQueryResult> tracy{values[0][slot.frameTimer.begin], values[0][slot.frameTimer.begin + 1]};
                for (auto timer : slot.nodeTimers) {
                    tracy.push_back(values[0][timer.begin]); tracy.push_back(values[0][timer.begin + 1]);
                }
                tracyGpuProfiler.publish(slot.profile, tracy, device->capabilities().timestampPeriodNanoseconds);
            }
            if (auto* trace = profiling::CpuPhaseTrace::active;
                trace && trace->gpuSpans.size() < profiling::CpuPhaseTrace::kMaxFrames && slot.frameTimer.queue == 0) {
                GpuClockCalibration calibration;
                const auto before = profiling::CpuPhaseTrace::Clock::now();
                const auto* queue = device->getQueue(QueueType::Graphics);
                const auto calibrated = queue ? queue->calibrateTimestamps().transform([&](auto rhiValue) { calibration = std::move(rhiValue); }) : makeError(Error::Unsupported);
                const auto after = profiling::CpuPhaseTrace::Clock::now();
                if (calibrated) {
                    trace->gpuSpans.push_back({slot.stats.executionId, values[0][slot.frameTimer.begin].value, values[0][slot.frameTimer.begin + 1].value,
                        calibration.gpuTimestamp, device->capabilities().timestampPeriodNanoseconds,
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(before + (after - before) / 2 - trace->origin).count()),
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count()), calibration.maxDeviationNanoseconds});
                }
            }
            slot.profile = {};
            completedGpuExecutionStats.push_back(std::move(slot.stats));
            slot.stats = {}; slot.pending = false;
        }
        return {};
    }

    void beginGpuTiming(CommandBuffer& commands)
    {
        activeGpuTimingSlot = nullptr; activeGpuTimingValid = false;
        // Raw external recording exposes no GPU completion. Queue acceptance
        // and query availability cannot distinguish an unexecuted reset from
        // a previous query generation, so retain CPU scopes without timestamps.
        if (!gpuTimestampQueryPools[0] || !commands.frameContext() ||
            !commands.frameContext()->completion().valid()) { return; }
        const auto resolved = resolveGpuTimings();
        if (!resolved) { return; }
        for (uint32_t offset = 0; offset < kGpuTimingSlotCount; ++offset) {
            const uint32_t index = (nextGpuTimingSlot + offset) % kGpuTimingSlotCount;
            auto& slot = gpuTimingSlots[index];
            if (slot.pending) { continue; }
            slot.stats = {}; slot.profile = {}; slot.used = {};
            slot.nodeTimers.clear(); slot.sectionTimers.clear();
            activeGpuTimingSlot = &slot; activeGpuTimingValid = true;
            slot.completion = commands.frameContext()->completion();
            // vkCmdResetQueryPool cannot run on a transfer-only queue. Reset every
            // queue's range in this graphics prologue; all branches depend on it.
            for (const auto& pool : gpuTimestampQueryPools) {
                if (pool && !commands.resetTimestampQueries(*pool, slot.firstQuery, slot.queryCount)) {
                    activeGpuTimingValid = false;
                }
            }
            slot.frameTimer = beginInterval(commands);
            if (auto* queue = device->getQueue(QueueType::Graphics)) { tracyGpuProfiler.beginFrame(*queue, slot.profile); }
            nextGpuTimingSlot = (index + 1) % kGpuTimingSlotCount;
            return;
        }
    }

    void finishGpuTiming(CommandBuffer& commands, bool completed)
    {
        if (!activeGpuTimingSlot) { return; }
        endInterval(commands, activeGpuTimingSlot->frameTimer);
        tracyGpuProfiler.endFrame(activeGpuTimingSlot->profile);
        if (completed && activeGpuTimingValid) {
            activeGpuTimingSlot->stats = lastExecutionStats;
            activeGpuTimingSlot->pending = true;
        } else {
            activeGpuTimingSlot->stats = {}; activeGpuTimingSlot->profile = {}; activeGpuTimingSlot->pending = false;
        }
        activeGpuTimingSlot = nullptr; activeGpuTimingValid = false;
    }

    Result<> prepareCommandPool(SubmissionSlot& slot, QueueType type, Queue& queue, CommandPool*& out)
    {
        QueueCommandContext& context = slot.queues[queueContextIndex(type)];
        if (context.queue != &queue || context.commandPool == nullptr) {
            context.commandPool.reset();
            Result<> result = device->createCommandPool(queue).transform([&](auto rhiValue) { context.commandPool = std::move(rhiValue); });
            if (!result) { return result; }
            context.queue = &queue;
        }
        out = context.commandPool.get();
        return out != nullptr ? Result<>{} : makeError(Error::Failure);
    }

    RenderGraphResource* fieldResource(const CompiledNode& node, const RenderGraphField& field)
    {
        const std::string name = makeRenderGraphFieldName(node.name, field.name);
        if (field.visibility == RenderGraphFieldVisibility::Output) { return resource(name); }
        const auto alias = inputAliases.find(name);
        return alias == inputAliases.end() ? nullptr : resource(alias->second);
    }

    static GraphAccessResource accessInitialState(const RenderGraphResource& resource)
    {
        // Incoming completion waits cover prior queues. Keep a conservative local
        // scope as well: the external-command API does not expose queue identity,
        // and the last reader alone does not describe visibility of earlier writes.
        return {resource.type, resource.state,
            {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
    }

    Result<> buildAccessPlan(std::span<const uint32_t> queues)
    {
        if (queues.size() != executionList.size()) { return makeError(Error::InvalidArgument); }
        std::vector<GraphAccessResource> initial;
        std::vector<RenderGraphResource*> resolved;
        std::vector<GraphAccessBinding> bindings;
        std::unordered_map<RenderGraphResource*, size_t> identities;
        std::vector<GraphAccessPass> passes;
        passes.reserve(executionList.size());
        for (size_t i = 0; i < executionList.size(); ++i) {
            const auto& node = executionList[i];
            auto& pass = passes.emplace_back();
            pass.queue = queues[i];
            for (const auto& field : node.reflection.fields()) {
                auto* allocation = fieldResource(node, field);
                if (!allocation) { continue; }
                // Graph-owned slots are canonical allocation identities for this
                // compile generation. Every input alias resolves to the same slot.
                const auto [entry, inserted] = identities.try_emplace(allocation, resolved.size());
                if (inserted) {
                    auto binding = bindGraphAccessResource(*allocation);
                    if (!binding) { return makeError(binding.error()); }
                    resolved.push_back(allocation);
                    bindings.push_back(std::move(*binding));
                    initial.push_back(accessInitialState(*allocation));
                }
                pass.uses.push_back({entry->second, stateForAccess(field.access),
                    fieldAccessScope(field, node.kind), fieldAccessWrites(field)});
            }
        }
        auto planned = buildGraphAccessPlan(initial, passes);
        if (!planned) {
            spdlog::error("[RenderGraph] Conflicting or invalid pass resource access declarations");
            return makeError(planned.error());
        }
        accessPlan = std::move(*planned);
        accessResources = std::move(resolved);
        accessBindings = std::move(bindings);
        return {};
    }

    static Result<> applyAccessPlan(CommandBuffer& commands, const GraphAccessPassPlan& pass,
        std::span<RenderGraphResource* const> resolved, std::span<const GraphAccessBinding> bindings)
    {
        for (const auto& use : pass.uses) {
            if (use.resource >= resolved.size() || !resolved[use.resource]) {
                return makeError(Error::InvalidArgument);
            }
        }
        auto result = recordGraphAccessBarriers(commands, pass, bindings);
        if (!result) { return result; }
        for (const auto& use : pass.uses) {
            auto& resource = *resolved[use.resource];
            resource.state = use.state;
        }
        return {};
    }

    Result<> transition(CommandBuffer& commands, RenderGraphResource& resource,
        ResourceState state, RenderGraphResourceAccess access)
    {
        SyncScope scope = scopeForGraphAccess(access, RenderGraphPassKind::Unsafe);
        bool writes = accessWrites(access);
        // External consumers also use states that have no graph field spelling.
        // Preserve their destination access instead of treating None as a read.
        switch (state) {
        case ResourceState::IndirectArgument:
            scope = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead};
            break;
        case ResourceState::DecompressionSource:
            scope = {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionRead};
            break;
        case ResourceState::DecompressionDestination:
            scope = {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionWrite};
            writes = true;
            break;
        default:
            break;
        }
        const std::array initial{accessInitialState(resource)};
        const std::array passes{GraphAccessPass{.uses = {{0, state, scope, writes}}}};
        auto planned = buildGraphAccessPlan(initial, passes);
        if (!planned) { return makeError(planned.error()); }
        auto binding = bindGraphAccessResource(resource);
        if (!binding) { return makeError(binding.error()); }
        const std::array resolved{&resource};
        const std::array bindings{std::move(*binding)};
        return applyAccessPlan(commands, planned->passes.front(), resolved, bindings);
    }

    Result<> prepareNode(CommandBuffer& commandBuffer, CompiledNode& node, uint64_t frameIndex,
        RenderGraphProperties& executionProperties, std::unique_ptr<RenderGraphExecutionContext>& prepared,
        std::vector<RenderGraphResource>* snapshots = nullptr)
    {
        profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::prepareNs);
        std::vector<RenderGraphExecutionContext::Binding> bindings;
        const size_t passIndex = size_t(&node - executionList.data());
        auto synchronized = applyAccessPlan(commandBuffer, accessPlan.passes[passIndex], accessResources, accessBindings);
        if (!synchronized) { return synchronized; }

        for (const RenderGraphField& field : node.reflection.fields()) {
            const std::string localName = field.name;
            RenderGraphResource* resource = fieldResource(node, field);

            bindings.push_back(RenderGraphExecutionContext::Binding{
                .fieldName = localName,
                .resource = resource,
                .visibility = field.visibility,
                .bindlessAccess = field.bindlessAccess,
                .scope = fieldAccessScope(field, node.kind),
                .internalLayouts = fieldChangesLayout(field),
                .bindlessHandle = resource != nullptr
                    ? resource->bindlessHandle
                    : BindlessHandle{},
                .sampledImageBindlessHandle = resource != nullptr
                    ? resource->sampledImageBindlessHandle
                    : BindlessHandle{},
            });
        }

        if (bindlessHeap != nullptr && usesBindlessResource(node)) {
            commandBuffer.bindBindlessHeap(*bindlessHeap);
        }

        StreamerSubsystem* upload = streamerSubsystem();
        const bool usesView = frameViewBuffer != nullptr && !node.sceneBinding.localView;
        executionProperties = node.effectiveProperties;
        if (usesView) {
            // Compatibility adapter for passes still using the packed camera ABI.
            // Authored node properties remain untouched; RenderView is authoritative.
            executionProperties["camera"] = frameCameraProperties;
            executionProperties["temporalJitter"] = frameView.frame[2] != 0;
        }
        if (snapshots) {
            snapshots->reserve(bindings.size());
            std::unordered_map<RenderGraphResource*, RenderGraphResource*> copies;
            for (auto& binding : bindings) {
                if (!binding.resource) { continue; }
                auto [entry, inserted] = copies.try_emplace(binding.resource);
                if (inserted) {
                    snapshots->push_back(*binding.resource);
                    entry->second = &snapshots->back();
                }
                binding.resource = entry->second;
            }
        }
        prepared.reset(new RenderGraphExecutionContext(
            commandBuffer,
            frameIndex,
            node.executionWidth,
            node.executionHeight,
            node.name,
            executionProperties,
            std::move(bindings),
            historyResources,
            upload != nullptr ? upload->streamer() : nullptr,
            node.sceneBinding.source != nullptr ? node.sceneBinding.source : runtimeScene,
            world,
            subsystemHost));
        auto& context = *prepared;
        context.preparedScene_ = node.preparedScene;
        context.preparationWorkerLimit_ = preparationWorkerLimit;
        context.preparationBatchWorkload_ = preparationBatchWorkload;
        context.viewConstants_ = usesView ? &frameView : nullptr;
        context.viewConstantsBuffer_ = usesView ? frameViewBuffer : nullptr;
        context.debugObserver_ = debugObserver;
        context.debugPassId_ = node.id;
        return {};
    }

    Result<> prepareRecording(NodeRecording& recording, CommandBuffer& commands, CompiledNode& node, uint64_t frameIndex)
    {
        recording.node = &node;
        recording.stats = {.id = node.id, .name = node.name, .type = node.type, .queue = recordingQueue->type()};
        auto result = prepareNode(commands, node, frameIndex, recording.properties, recording.context, &recording.resources);
        if (!result) { return result; }
        if (activeGpuTimingSlot && activeGpuTimingValid) {
            auto& slot = *activeGpuTimingSlot;
            const uint32_t queue = timingQueueIndex();
            constexpr uint32_t queries = 2 * (256 + 1);
            if (gpuTimestampQueryPools[queue] && slot.used[queue] + queries <= slot.queryCount) {
                recording.queryPool = gpuTimestampQueryPools[queue].get();
                recording.queueIndex = queue;
                recording.firstQuery = slot.firstQuery;
                recording.nextQuery = slot.used[queue];
                slot.used[queue] += queries;
                recording.endQuery = slot.used[queue];
            } else if (gpuTimestampQueryPools[queue]) { recording.profilingOverflow = true; }
            recording.profile.active = slot.profile.active;
        }
        recording.passTimer = recording.beginInterval(commands);
        const auto begin = std::chrono::steady_clock::now();
        {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::prepareNs);
            result = node.pass->prepareExecution(*recording.context);
        }
        recording.stats.cpuMilliseconds = renderGraphElapsedMilliseconds(begin);
        return result;
    }

    // Only the batch worker touches this result. Merge statistics, query metadata
    // and CPU publication order on the coordinator after every task has joined.
    Result<> recordNode(NodeRecording& recording)
    {
        auto& context = *recording.context;
        auto& commands = context.commandBuffer();
        auto& stats = recording.stats;
        const std::string marker = passProfileMarkerName(stats.name, stats.type);
        METALLIC_TRACY_CPU_SCOPE(marker.c_str());
        const profiling::NsightProfileRange passMarker(profiling::NsightDomain::Render, marker.c_str(),
            profiling::NsightCategory::RenderPass, stats.id, profiling::nsightColorFromName(stats.type));
        RenderGraphGpuLabels<CommandBuffer> labels(marker, debugLabelColorFromArgb(profiling::nsightColorFromName(stats.type)));
        labels.resume(commands);
        tracyGpuProfiler.beginZone(recording.profile, marker);
        context.beginProfile_ = [&](CommandBuffer& buffer, std::string_view name, uint32_t parent) {
            if (stats.sections.size() >= 256) { recording.profilingOverflow = true; return UINT32_MAX; }
            const auto index = uint32_t(stats.sections.size());
            stats.sections.push_back({.name = std::string(name), .parent = parent, .queue = stats.queue});
            labels.begin(name, debugLabelColorFromArgb(profiling::nsightColorFromName(name)));
            recording.sectionTimers.push_back(recording.beginInterval(buffer));
            return index;
        };
        context.endProfile_ = [&](CommandBuffer& buffer, uint32_t index, double cpuMs) {
            stats.sections[index].cpuMilliseconds = cpuMs;
            recording.endInterval(buffer, recording.sectionTimers[index]);
            labels.end();
        };
        context.cpuProfile_ = [&](std::span<const RenderGraphProfileSection> samples, uint32_t parent) {
            if (stats.sections.size() + samples.size() > 256) { recording.profilingOverflow = true; return; }
            const uint32_t base = uint32_t(stats.sections.size());
            for (uint32_t i = 0; i < samples.size(); ++i) {
                auto section = samples[i];
                section.parent = section.parent < i ? base + section.parent : parent;
                section.cpuOnly = true;
                section.gpuTimingAvailable = false;
                section.gpuMilliseconds = 0;
                stats.sections.push_back(std::move(section));
                recording.sectionTimers.push_back({});
            }
        };
        context.streamingProfile_ = [&](SceneStreamingProfile sample) { recording.streaming.push_back(std::move(sample)); };
        const auto begin = std::chrono::steady_clock::now();
        {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::workerExecuteNs);
            recording.result = recording.node->pass->execute(context);
        }
        stats.cpuMilliseconds += renderGraphElapsedMilliseconds(begin);
        tracyGpuProfiler.endZone(recording.profile);
        if (recording.result) { recording.endInterval(commands, recording.passTimer); }
        labels.suspend();
        if (recording.result) { recording.result = commands.end(); }
        return recording.result;
    }

    void mergeRecording(NodeRecording& recording)
    {
        lastExecutionStats.profilingOverflow |= recording.profilingOverflow;
        if (recording.context) { lastExecutionStats.preparationTaskCount += recording.context->preparationTaskCount_; }
        activeGpuTimingValid &= recording.timingValid;
        if (activeGpuTimingSlot) {
            activeGpuTimingSlot->nodeTimers.push_back(recording.passTimer);
            activeGpuTimingSlot->sectionTimers.push_back(std::move(recording.sectionTimers));
            for (auto& zone : recording.profile.zones) { activeGpuTimingSlot->profile.zones.push_back(std::move(zone)); }
        }
        for (auto& sample : recording.streaming) { lastExecutionStats.streaming.push_back(std::move(sample)); }
        if (!recording.result) {
            spdlog::error("[RenderGraph] Parallel recording of '{}' failed: {}", recording.stats.name, resultToString(recording.result));
        }
        lastExecutionStats.nodes.push_back(std::move(recording.stats));
    }

    Result<> executeNode(CommandBuffer& commandBuffer, CompiledNode& node, uint64_t frameIndex,
        const RenderGraphExecutionContext::ParallelRecorder& parallel = {})
    {
        RenderGraphProperties executionProperties;
        std::unique_ptr<RenderGraphExecutionContext> prepared;
        auto preparation = prepareNode(commandBuffer, node, frameIndex, executionProperties, prepared);
        if (!preparation) { return preparation; }
        auto& context = *prepared;
        StreamerSubsystem* upload = streamerSubsystem();
        const std::string markerName = passProfileMarkerName(node.name, node.type);
        METALLIC_TRACY_CPU_SCOPE(markerName.c_str());
        const uint32_t markerColor = profiling::nsightColorFromName(node.type);
        const profiling::NsightProfileRange passMarker(
            profiling::NsightDomain::Render,
            markerName.c_str(),
            profiling::NsightCategory::RenderPass,
            node.id,
            markerColor);
        const size_t nodeIndex = lastExecutionStats.nodes.size();
        lastExecutionStats.nodes.push_back({.id = node.id, .name = node.name, .type = node.type,
            .queue = recordingQueue ? recordingQueue->type() : QueueType::Graphics});
        // Timestamp scopes may span the graphics producer/join. Recreate balanced
        // label ranges on each branch so Nsight sees the work and its ancestry
        // on the queue that executes it.
        RenderGraphGpuLabels<CommandBuffer> labels(markerName, debugLabelColorFromArgb(markerColor));
        labels.resume(commandBuffer);
        if (parallel) {
            context.parallelRecorder_ = [&](RenderGraphExecutionContext& current,
                const RenderGraphExecutionContext::CommandRecorder& compute,
                const RenderGraphExecutionContext::CommandRecorder& graphics) -> Result<> {
                labels.suspend();
                const auto branch = [&](CommandBuffer& commands,
                    const RenderGraphExecutionContext::CommandRecorder& record) {
                    labels.resume(commands);
                    const Result<> result = record(commands);
                    labels.suspend();
                    return result;
                };
                const Result<> result = parallel(current,
                    [&](CommandBuffer& commands) { return branch(commands, compute); },
                    [&](CommandBuffer& commands) { return branch(commands, graphics); });
                // A failed fork may leave current pointing at an ended producer.
                // Outer RAII scopes must not emit timestamps into that recording.
                if (!result) { activeGpuTimingValid = false; }
                if (result) {
                    labels.resume(current.commandBuffer());
                }
                return result;
            };
        }
        const TimerRef passTimer = beginInterval(commandBuffer);
        if (activeGpuTimingSlot) {
            activeGpuTimingSlot->nodeTimers.push_back(passTimer);
            activeGpuTimingSlot->sectionTimers.emplace_back();
            tracyGpuProfiler.beginZone(activeGpuTimingSlot->profile, markerName);
        }
        context.beginProfile_ = [&, nodeIndex](CommandBuffer& commands, std::string_view name, uint32_t parent) {
            auto& sections = lastExecutionStats.nodes[nodeIndex].sections;
            if (sections.size() >= 256) { lastExecutionStats.profilingOverflow = true; return UINT32_MAX; }
            const uint32_t index = uint32_t(sections.size());
            sections.push_back({.name = std::string(name), .parent = parent,
                .queue = recordingQueue ? recordingQueue->type() : QueueType::Graphics});
            labels.begin(name, debugLabelColorFromArgb(profiling::nsightColorFromName(name)));
            const TimerRef timer = beginInterval(commands);
            if (activeGpuTimingSlot) { activeGpuTimingSlot->sectionTimers[nodeIndex].push_back(timer); }
            return index;
        };
        context.endProfile_ = [&, nodeIndex](CommandBuffer& commands, uint32_t index, double cpuMs) {
            lastExecutionStats.nodes[nodeIndex].sections[index].cpuMilliseconds = cpuMs;
            if (activeGpuTimingSlot) { endInterval(commands, activeGpuTimingSlot->sectionTimers[nodeIndex][index]); }
            // On fork failure the producer has ended and its labels are already
            // closed; only unwind logical scopes while the pass returns its error.
            labels.end();
        };
        context.cpuProfile_ = [&, nodeIndex](std::span<const RenderGraphProfileSection> samples, uint32_t parent) {
            auto& sections = lastExecutionStats.nodes[nodeIndex].sections;
            if (sections.size() + samples.size() > 256) { lastExecutionStats.profilingOverflow = true; return; }
            const uint32_t base = uint32_t(sections.size());
            for (uint32_t i = 0; i < samples.size(); ++i) {
                auto section = samples[i];
                section.parent = section.parent < i ? base + section.parent : parent;
                section.cpuOnly = true;
                section.gpuTimingAvailable = false;
                section.gpuMilliseconds = 0;
                sections.push_back(std::move(section));
                // Keep indices aligned without allocating a GPU timestamp pair.
                if (activeGpuTimingSlot) { activeGpuTimingSlot->sectionTimers[nodeIndex].push_back({}); }
            }
        };
        context.streamingProfile_ = [&](SceneStreamingProfile sample) { lastExecutionStats.streaming.push_back(std::move(sample)); };
        const auto cpuBegin = std::chrono::steady_clock::now();
        const auto featureReservation = firstFeatureReservations.find(node.name);
        const bool firstFeature = featureReservation != firstFeatureReservations.end() && bool(featureReservation->second);
        if (firstFeature) { device->logMemoryBudget("before first external feature"); }
        std::string streamingLog;
        Result<> result;
        if (upload && node.preparedScene) {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::scenePrepareNs);
            auto scope = context.profileScope("Streamer prepare");
            MeshletStreamFrameDesc view;
            view.freezeRasterSnapshot = context.properties().value("benchmarkFreezeStreaming", false);
            result = upload->recordSceneBegin(*node.preparedScene, node.sceneRequirements, context, view, streamingLog);
        }
        const bool sceneReady = !node.preparedScene || node.preparedScene->ready;
        if (result && sceneReady) {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::prepareNs);
            result = node.pass->prepareExecution(context);
        }
        if (result && sceneReady && upload && node.preparedScene && node.preparedScene->geometry) {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::scenePrepareNs);
            MeshletStreamFrameDesc view;
            node.pass->describeSceneView(context, view);
            result = upload->recordSceneTraversal(*node.preparedScene, context, view,
                [&](std::string_view point) { node.pass->sceneTraversalCheckpoint(context, point); });
        }
        if (result && sceneReady) {
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::serialExecuteNs);
            result = node.pass->execute(context);
        }
        if (result && sceneReady && upload && node.preparedScene) {
            result = upload->recordSceneEnd(*node.preparedScene, context);
        }
        if (firstFeature) {
            firstFeatureReservations.erase(featureReservation);
            device->logMemoryBudget("after first external feature");
        }
        if (!result) {
            const auto& sections = lastExecutionStats.nodes[nodeIndex].sections;
            spdlog::error("[RenderGraph] Pass '{}' ({}) failed in '{}': {}", node.name, node.type,
                sections.empty() ? std::string_view("execute") : std::string_view(sections.back().name), resultToString(result));
            if (!streamingLog.empty()) { spdlog::error("[Streamer] {}", streamingLog); }
        }
        if (result && upload != nullptr) {
            auto scope = context.profileScope("Upload flush");
            upload->flush(context.commandBuffer());
        }
        lastExecutionStats.preparationTaskCount += context.preparationTaskCount_;
        lastExecutionStats.nodes[nodeIndex].cpuMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpuBegin).count();
        if (activeGpuTimingSlot) { tracyGpuProfiler.endZone(activeGpuTimingSlot->profile); }
        if (result) { endInterval(context.commandBuffer(), passTimer); }
        labels.suspend();
        if (result && debugObserver && !context.debugAfterPassPublished_) { context.debugCheckpoint("AfterPass"); }
        return result;
    }
};

RenderGraphExecutor::RenderGraphExecutor()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphExecutor::RenderGraphExecutor(RenderSubsystemHost& subsystemHost, RenderWorld& world)
    : impl_(std::make_unique<Impl>())
{
    impl_->subsystemHost = &subsystemHost;
    impl_->world = &world;
}

RenderGraphExecutor::~RenderGraphExecutor() = default;
RenderGraphExecutor::RenderGraphExecutor(RenderGraphExecutor&&) noexcept = default;
RenderGraphExecutor& RenderGraphExecutor::operator=(RenderGraphExecutor&&) noexcept = default;

Result<> RenderGraphExecutor::compile(
    Device& device,
    const RenderGraph& graph,
    uint32_t width,
    uint32_t height,
    std::string& log)
{
    return compile(device, graph, width, height, RenderGraphCompileOptions{}, log);
}

Result<> RenderGraphExecutor::compile(
    Device& device,
    const RenderGraph& graph,
    uint32_t width,
    uint32_t height,
    const RenderGraphCompileOptions& options,
    std::string& log)
{
    RenderGraphLogScope compileScope(
        "compile graph '" + graph.name() + "' " + std::to_string(width) + "x" + std::to_string(height));
    spdlog::info(
        "[RenderGraph] Compile inputs graphNodes={} graphEdges={} markedOutputs={} extraOutputs={}",
        graph.nodes().size(),
        graph.edges().size(),
        graph.outputs().size(),
        options.extraOutputs.size());

    if (impl_->debugObserver) {
        impl_->debugGraph = {{"id", std::to_string(impl_->debugGraphId)}, {"generation", ++impl_->debugGeneration},
            {"state", "Unavailable"}, {"reason", "Graph recompilation invalidates pending handles until success"}};
        impl_->debugObserver->compiled(impl_->debugGraph);
    }

    if (width == 0 || height == 0) {
        log = validationPrefix("invalid default dimensions");
        return makeError(Error::InvalidArgument);
    }

    std::string validationLog;
    if (!graph.validate(validationLog)) {
        log = validationLog;
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    ActiveGraph activeGraph;
    if (!buildActiveGraph(graph, options.extraOutputs, activeGraph, log)) {
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }
    spdlog::info(
        "[RenderGraph] Active graph passCount={} requestedExtraOutputs={}",
        activeGraph.executionOrder.size(),
        options.extraOutputs.size());

    Result<> pendingResult;
    {
        RenderGraphLogScope scope("wait for previous submitted RenderGraph work");
        pendingResult = impl_->waitForSubmittedWork(UINT64_MAX);
    }
    if (!pendingResult) {
        log = resultMessage("RenderGraph waitForSubmittedWork", pendingResult);
        impl_->isCompiled = false;
        return pendingResult;
    }

    impl_->firstFeatureReservations.clear();
    device.logMemoryBudget("graph load / previous work complete");
    MemoryBudgetReservation graphReservation;
    std::unordered_map<std::string, MemoryBudgetReservation> featureReservations;
    const auto budgetPolicy = device.memoryBudget().policy;
    Result<> budgetResult = device.reserveMemoryBudget(budgetPolicy.graphReserveBytes).transform([&](auto rhiValue) { graphReservation = std::move(rhiValue); });
    for (const auto& passName : activeGraph.executionOrder) {
        const auto* node = graph.findNode(passName);
        if (budgetResult && node && mergeRenderGraphProperties(node->properties, node->runtimeProperties).value("enabled", true) &&
                (node->type == "StreamlineDlssSrPass" || node->type == "StreamlineDlssRrPass" || node->type == "DlssNrPass")) {
            budgetResult = device.reserveMemoryBudget(budgetPolicy.externalFeatureReserveBytes).transform([&](auto rhiValue) { featureReservations[node->name] = std::move(rhiValue); });
        }
    }
    if (!budgetResult) {
        log = "Unified GPU budget cannot reserve headroom for graph resources and external features";
        impl_->isCompiled = false;
        return budgetResult;
    }
    device.logMemoryBudget("graph load / headroom reserved");

    if (impl_->device != nullptr && impl_->device != &device) {
        impl_->tracyGpuProfiler = {};
        impl_->viewBuffers.clear();
        impl_->hasPreviousView = false;
        for (auto& slot : impl_->submissionSlots) {
            slot->commandBuffers.clear();
            (void)slot->frame.reset();
            slot->queues = {};
        }
        impl_->lastSubmittedCompletion = {};
        impl_->submissionTrackers.clear();
        impl_->pendingSceneResourceSnapshot.reset();
    }

    if (impl_->ownedViewProperties != graph.viewProperties()) {
        const bool hadView = impl_->renderView() != nullptr;
        if (!graph.viewProperties().is_object()) {
            log = "RenderGraph view must be an object";
            impl_->isCompiled = false;
            return makeError(Error::InvalidArgument);
        }
        impl_->ownedView = RenderView{};
        impl_->hasOwnedView = !graph.viewProperties().empty();
        if (impl_->hasOwnedView && (!impl_->ownedView.setCameraProperties(
                graph.viewProperties().value("camera", RenderGraphProperties::object())) ||
                !graph.viewProperties().value("temporalJitter", RenderGraphProperties(false)).is_boolean())) {
            log = "RenderGraph contains invalid view properties";
            impl_->isCompiled = false;
            return makeError(Error::InvalidArgument);
        }
        impl_->ownedView.setTemporalJitter(graph.viewProperties().value("temporalJitter", false));
        impl_->ownedViewProperties = graph.viewProperties();
        impl_->hasPreviousView = false;
        if (hadView != (impl_->renderView() != nullptr)) { impl_->isCompiled = false; }
    }
    impl_->accessPlan = {};
    impl_->accessResources.clear();
    impl_->accessBindings.clear();
    if (!registerBuiltInRenderSubsystems(*impl_->subsystemHost, log)) {
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }
    if (impl_->subsystemHost->device() != nullptr && impl_->subsystemHost->device() != &device) {
        if (impl_->subsystemHost != &impl_->ownedSubsystemHost) {
            log = "RenderGraphExecutor external RenderSubsystemHost belongs to another Device";
            impl_->isCompiled = false;
            return makeError(Error::InvalidArgument);
        }
        impl_->subsystemHost->shutdown();
    }
    Result<> subsystemResult = impl_->subsystemHost->initialize(device,
        impl_->subsystemHost->frameSlotCount() != 0 ? impl_->subsystemHost->frameSlotCount() : 2, log);
    if (!subsystemResult) {
        impl_->isCompiled = false;
        return subsystemResult;
    }
    impl_->subsystemHost->setWorld(impl_->world);

    impl_->requiredSubsystemIds.clear();
    impl_->requiredSubsystemIds.emplace_back(StreamerSubsystem::kSubsystemId);
    std::unordered_set<std::string> requiredSubsystemSet(impl_->requiredSubsystemIds.begin(), impl_->requiredSubsystemIds.end());
    std::vector<std::pair<std::string, std::string>> passSubsystemRequirements;
    for (const std::string& passName : activeGraph.executionOrder) {
        const RenderGraphNode* node = graph.findNode(passName);
        if (node == nullptr) {
            continue;
        }
        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node->type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node->type + "'");
            impl_->isCompiled = false;
            return makeError(Error::InvalidArgument);
        }
        for (RenderSubsystemId id : pass->requiredSubsystems()) {
            if (requiredSubsystemSet.emplace(id).second) {
                impl_->requiredSubsystemIds.emplace_back(id);
                passSubsystemRequirements.emplace_back(std::string(id), passName);
            }
        }
    }
    subsystemResult = impl_->subsystemHost->activate(StreamerSubsystem::kSubsystemId, log);
    if (!subsystemResult) {
        impl_->isCompiled = false;
        return subsystemResult;
    }
    for (const auto& [subsystemId, passName] : passSubsystemRequirements) {
        std::string activationLog;
        subsystemResult = impl_->subsystemHost->activate(subsystemId, activationLog);
        if (!subsystemResult) {
            log = "RenderGraph pass '" + passName + "' requires subsystem '" +
                subsystemId + "': " + activationLog;
            impl_->isCompiled = false;
            return subsystemResult;
        }
    }

    const bool dimensionsChanged = impl_->width != width || impl_->height != height;
    if (!options.displayOutput.valid()) {
        log = "Invalid display output parameters";
        return makeError(Error::InvalidArgument);
    }
    const bool canReuseCompiledPasses = impl_->displayOutput == options.displayOutput &&
        impl_->canReuseCompiledPasses(device, graph, activeGraph);
    impl_->displayOutput = options.displayOutput;

    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;

    StreamerSubsystem* sceneResources = impl_->streamerSubsystem();
    if (sceneResources == nullptr) {
        log = "RenderGraph compile failed: render.streamer was not activated";
        impl_->isCompiled = false;
        return makeError(Error::Failure);
    }

    const RenderGraphCompileContext compileContext{
        .device = &device,
        .graphicsQueue = device.getQueue(QueueType::Graphics),
        .runtimeScene = impl_->runtimeScene,
        .renderWorld = impl_->world,
        .subsystemHost = impl_->subsystemHost,
        .width = width,
        .height = height,
        .defaultFormat = impl_->defaultFormat,
        .debugReadback = impl_->debugObserver != nullptr,
        .renderView = impl_->renderView(),
        .displayOutput = impl_->displayOutput,
    };

    if (auto* gpuScene = impl_->subsystemHost->get<GPUSceneSubsystem>()) {
        gpuScene->setDebugReadbackEnabled(impl_->debugObserver != nullptr);
    }

    if (canReuseCompiledPasses) {
        impl_->isCompiled = false;
        Result<> refreshResult;
        {
            RenderGraphLogScope scope("refresh reusable passes");
            refreshResult = impl_->refreshReusablePasses(graph, activeGraph, compileContext, log);
        }
        if (!refreshResult) {
            return refreshResult;
        }

        Result<> resourceResult;
        graphReservation.reset();
        {
            RenderGraphLogScope scope("rebuild graph resources");
            resourceResult = impl_->rebuildGraphResources(device, graph, activeGraph, options, log);
        }
        if (!resourceResult) {
            return resourceResult;
        }

        impl_->sceneBindingsReady = true;
        impl_->isCompiled = true;
        log = dimensionsChanged ? "RenderGraph resized" : "RenderGraph resources rebuilt";
        impl_->firstFeatureReservations = std::move(featureReservations);
        device.logMemoryBudget("graph rebuilt / external feature pending");
        impl_->publishDebugGraph(graph);
        return {};
    }

    impl_->executionList.clear();
    impl_->resources.clear();
    impl_->inputAliases.clear();
    impl_->bindlessHeap.reset();
    impl_->isCompiled = false;
    device.logMemoryBudget("graph load / previous graph resources released");

    for (const std::string& passName : activeGraph.executionOrder) {
        const RenderGraphNode* node = graph.findNode(passName);
        if (node == nullptr) {
            log = validationPrefix(std::string("active pass is missing '") + passName + "'");
            return makeError(Error::InvalidArgument);
        }

        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node->type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node->type + "'");
            return makeError(Error::InvalidArgument);
        }
        const RenderGraphProperties effectiveProperties =
            mergeRenderGraphProperties(node->properties, node->runtimeProperties);
        pass->setProperties(effectiveProperties);
        impl_->executionList.push_back(Impl::CompiledNode{
            .id = node->id, .name = node->name, .type = node->type,
            .staticProperties = node->properties, .runtimeProperties = node->runtimeProperties,
            .effectiveProperties = effectiveProperties, .pass = std::move(pass),
        });
        impl_->executionList.back().sceneDependency = impl_->executionList.back().pass->sceneDependency();
    }
    spdlog::info("[RenderGraph] Created {} compiled pass objects", impl_->executionList.size());
    impl_->rebuildInputAliases(graph, activeGraph);
    std::vector<Impl::SceneBinding> sceneBindings;
    Result<> bindingResult = impl_->resolveSceneBindings(sceneBindings, log);
    if (!bindingResult) { return bindingResult; }
    for (size_t index = 0; index < impl_->executionList.size(); ++index) {
        auto& node = impl_->executionList[index];
        node.sceneBinding = sceneBindings[index];
        impl_->applySceneProperties(node, node.sceneBinding);
        auto nodeContext = impl_->contextForScene(compileContext, node.sceneBinding);
        Result<> result = impl_->prepareNodeScene(node, nodeContext, log);
        if (!result) { return result; }
        result = node.pass->prepare(nodeContext, log);
        if (!result) { log = "RenderGraph prepare failed for pass '" + node.name + "': " + log; return result; }
        node.kind = node.pass->kind();
        node.queueType = node.pass->queueType();
        node.reflection = node.pass->reflect(nodeContext);
    }
    Impl::BindlessResourcePlan bindlessPlan = impl_->collectBindlessResourcePlan();
    if ((!bindlessPlan.sampledImageResources.empty() || !bindlessPlan.bufferResources.empty()) &&
        !device.capabilities().bindlessDescriptorHeap) {
        log = "RenderGraph compile failed: bindless resources require "
            "DeviceCapabilities::bindlessDescriptorHeap";
        return makeError(Error::Unsupported);
    }

    for (Impl::CompiledNode& node : impl_->executionList) {
        Result<> result;
        {
            RenderGraphLogScope scope(
                "compile pass '" + node.name + "' (" + node.type + ")");
            result = node.pass->compile(impl_->contextForScene(compileContext, node.sceneBinding, node.preparedScene), log);
        }
        if (!result) {
            impl_->isCompiled = false;
            return result;
        }
        node.pass->setProperties(node.effectiveProperties);
    }

    Result<> resourceResult;
    graphReservation.reset(); // Spend the graph promise; newly created resources enter heap usage.
    {
        RenderGraphLogScope scope("allocate graph resources");
        resourceResult = impl_->allocateGraphResources(
            device,
            graph,
            activeGraph,
            options,
            bindlessPlan,
            log);
    }
    if (!resourceResult) {
        impl_->isCompiled = false;
        return resourceResult;
    }

    impl_->initializeGpuTiming(device);
    impl_->sceneBindingsReady = true;
    impl_->isCompiled = true;
    impl_->firstFeatureReservations = std::move(featureReservations);
    device.logMemoryBudget("graph compiled / external feature pending");
    log = "RenderGraph compiled";
    impl_->publishDebugGraph(graph);
    return {};
}

Result<> RenderGraphExecutor::reloadShaders(std::string& log)
{
    RenderGraphLogScope reloadScope("transactional shader reload");
    log.clear();
    if (!impl_->isCompiled || impl_->device == nullptr || impl_->executionList.empty()) {
        log = "RenderGraph shader reload requires a compiled graph";
        return makeError(Error::InvalidArgument);
    }

    Result<> result = impl_->waitForSubmittedWork(UINT64_MAX);
    if (!result) {
        log = resultMessage("RenderGraph waitForSubmittedWork before shader reload", result);
        return result;
    }
    result = impl_->device->waitIdle();
    if (!result) {
        log = resultMessage("Device waitIdle before shader reload", result);
        return result;
    }

    StreamerSubsystem* sceneResources = impl_->streamerSubsystem();
    if (sceneResources == nullptr) {
        log = "RenderGraph shader reload requires render.streamer";
        return makeError(Error::Failure);
    }
    const RenderGraphCompileContext compileContext{
        .device = impl_->device,
        .graphicsQueue = impl_->device->getQueue(QueueType::Graphics),
        .runtimeScene = impl_->runtimeScene,
        .renderWorld = impl_->world,
        .subsystemHost = impl_->subsystemHost,
        .width = impl_->width,
        .height = impl_->height,
        .defaultFormat = impl_->defaultFormat,
        .debugReadback = impl_->debugObserver != nullptr,
        .renderView = impl_->renderView(),
        .displayOutput = impl_->displayOutput,
    };

    result = impl_->refreshFrameSceneBindings(nullptr, log);
    if (!result) { return result; }

    std::vector<Impl::CompiledNode> replacements;
    replacements.reserve(impl_->executionList.size());
    std::string reloadDetails;
    for (const Impl::CompiledNode& compiledNode : impl_->executionList) {
        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(compiledNode.type);
        if (pass == nullptr) {
            log = "Shader reload could not recreate pass '" + compiledNode.name +
                "' of type '" + compiledNode.type + "'";
            return makeError(Error::InvalidArgument);
        }
        auto preparedScene = compiledNode.preparedScene;
        if (preparedScene && preparedScene->geometry) {
            // Reload internal traversal/streaming shaders transactionally as well.
            // A failed replacement must leave the active session and its heap intact.
            preparedScene = std::make_shared<PreparedSceneResources>(*preparedScene);
            preparedScene->state.reset();
            preparedScene->geometry.reset();
            result = impl_->streamerSubsystem()->prepareScene(compiledNode.sceneRequirements,
                compiledNode.effectiveProperties, compiledNode.sceneBinding.source,
                preparedScene, log, compileContext.debugReadback);
            if (!result) { return result; }
        }
        const auto nodeContext = impl_->contextForScene(compileContext, compiledNode.sceneBinding, preparedScene);
        pass->setProperties(compiledNode.effectiveProperties);
        std::string prepareLog;
        result = pass->prepare(nodeContext, prepareLog);
        if (!result) {
            log = "Shader reload could not prepare pass '" + compiledNode.name + "' (" +
                compiledNode.type + ")";
            if (!prepareLog.empty()) {
                log += ": " + prepareLog;
            }
            return result;
        }
        const RenderGraphPassKind kind = pass->kind();
        const QueueType queueType = pass->queueType();
        RenderPassReflection reflection = pass->reflect(nodeContext);
        const std::span<const RenderSubsystemId> oldSubsystems =
            compiledNode.pass->requiredSubsystems();
        const std::span<const RenderSubsystemId> newSubsystems = pass->requiredSubsystems();
        const bool subsystemRequirementsMatch =
            oldSubsystems.size() == newSubsystems.size() &&
            std::equal(oldSubsystems.begin(), oldSubsystems.end(), newSubsystems.begin());
        if (kind != compiledNode.kind ||
            queueType != compiledNode.queueType ||
            reflection != compiledNode.reflection ||
            !subsystemRequirementsMatch || pass->sceneDependency() != compiledNode.sceneDependency ||
            pass->sceneResourcesRequired(nodeContext) != compiledNode.sceneRequirements) {
            log = "Shader reload rejected pass '" + compiledNode.name +
                "' because its render-graph contract changed; rebuild the graph instead";
            return makeError(Error::InvalidArgument);
        }

        std::string passLog;
        {
            RenderGraphLogScope scope(
                "reload shaders for pass '" + compiledNode.name + "' (" +
                compiledNode.type + ")");
            result = pass->compile(nodeContext, passLog);
        }
        if (!result) {
            log = "Shader reload failed for pass '" + compiledNode.name + "' (" +
                compiledNode.type + ")";
            if (!passLog.empty()) {
                log += ": " + passLog;
            }
            return result;
        }
        if (!passLog.empty()) {
            if (!reloadDetails.empty()) {
                reloadDetails += '\n';
            }
            reloadDetails += compiledNode.name + ": " + passLog;
        }
        pass->setProperties(compiledNode.effectiveProperties);
        replacements.push_back(Impl::CompiledNode{
            .id = compiledNode.id,
            .name = compiledNode.name,
            .type = compiledNode.type,
            .kind = kind,
            .queueType = queueType,
            .staticProperties = compiledNode.staticProperties,
            .runtimeProperties = compiledNode.runtimeProperties,
            .effectiveProperties = compiledNode.effectiveProperties,
            .pass = std::move(pass),
            .reflection = std::move(reflection),
            .sceneDependency = compiledNode.sceneDependency,
            .preparedScene = std::move(preparedScene),
            .sceneRequirements = compiledNode.sceneRequirements,
            .sceneBinding = compiledNode.sceneBinding,
            .executionWidth = compiledNode.executionWidth,
            .executionHeight = compiledNode.executionHeight,
        });
    }

    std::string subsystemLog;
    result = impl_->subsystemHost->reloadShaders(subsystemLog);
    if (!result) {
        log = subsystemLog.empty()
            ? "Render subsystem shader reload failed"
            : std::move(subsystemLog);
        return result;
    }

    impl_->executionList = std::move(replacements);
    if (impl_->debugObserver) {
        impl_->debugGraph["generation"] = ++impl_->debugGeneration;
        for (auto& resource : impl_->debugGraph["resources"]) { resource["allocation"] = impl_->debugGeneration; }
        impl_->debugObserver->compiled(impl_->debugGraph);
    }
    log = "Reloaded shaders for " + std::to_string(impl_->executionList.size()) +
        " render pass(es)";
    if (!subsystemLog.empty()) {
        log += '\n';
        log += subsystemLog;
    }
    if (!reloadDetails.empty()) {
        log += '\n';
        log += reloadDetails;
    }
    return {};
}

Result<> RenderGraphExecutor::execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources)
{
    impl_->preparationWorkerLimit = 1;
    METALLIC_TRACY_CPU_SCOPE("RenderGraph Record");
    DebugExecutionScope debugScope;
    if (!impl_->isCompiled) {
        return makeError(Error::InvalidArgument);
    }

    std::string sceneLog;
    Result<> sceneResult = impl_->refreshFrameSceneBindings(historyResources, sceneLog);
    if (!sceneResult) { spdlog::error("[RenderGraph] {}", sceneLog); return sceneResult; }

    // External command buffers now outlive execute(). Guard destructive graph
    // changes and legacy passes using the same completion points as the caller.
    const scene::Scene* currentScene = impl_->runtimeScene;
    const std::array<uint64_t, 5> sceneStamp = currentScene != nullptr
        ? std::array<uint64_t, 5>{currentScene->resourceIdentity(), currentScene->contentRevision(),
            currentScene->sceneGraph().structuralRevision(), currentScene->transformRevision(),
            currentScene->visibilityRevision()}
        : std::array<uint64_t, 5>{};
    const bool requiresCompletedFrame = std::any_of(impl_->executionList.begin(), impl_->executionList.end(),
        [](const Impl::CompiledNode& node) { return !node.pass->supportsFrameOverlap(); });
    if (requiresCompletedFrame || sceneStamp != impl_->recordedSceneStamp || impl_->hasSubmittedWork) {
        const profiling::NsightProfileRange waitMarker(profiling::NsightDomain::Render,
            "Wait Graph Resources", profiling::NsightCategory::RenderGraph);
        Result<> result = impl_->waitForSubmittedWork(UINT64_MAX);
        if (!result) {
            return result;
        }
    }
    impl_->recordedSceneStamp = sceneStamp;
    std::erase_if(impl_->externalCompletions, [](const auto& point) { return point.isComplete(); });
    if (RenderFrameContext* frame = commandBuffer.frameContext()) {
        if (!frame->recording()) {
            return makeError(Error::InvalidArgument);
        }
        if (std::none_of(impl_->externalCompletions.begin(), impl_->externalCompletions.end(),
                [&](const auto& point) { return point.sameSubmission(frame->completion()); })) {
            impl_->externalCompletions.push_back(frame->completion());
        }
    }
    Result<> dependencyResult = commandBuffer.addDependency(impl_->lastSubmittedCompletion);
    if (!dependencyResult) { return dependencyResult; }
    const std::vector<uint32_t> queues(impl_->executionList.size(), 0);
    Result<> planned = impl_->buildAccessPlan(queues);
    if (!planned) { return planned; }
    impl_->historyResources = historyResources;
    std::string subsystemLog;
    const uint64_t frameIndex = impl_->executionFrameIndex++;
    const profiling::NsightProfileRange executeMarker(
        profiling::NsightDomain::Render,
        "Render Graph Execute",
        profiling::NsightCategory::RenderGraph,
        frameIndex);
    RenderFrameContext* frameResources = commandBuffer.frameContext();
    Result<> result = impl_->subsystemHost->beginFrame(
        frameResources != nullptr ? frameResources->frameIndex() : frameIndex,
        frameResources != nullptr ? frameResources->slotIndex()
            : static_cast<uint32_t>(frameIndex % impl_->subsystemHost->frameSlotCount()),
        historyResources,
        subsystemLog,
        frameResources);
    if (!result) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
        impl_->historyResources = nullptr;
        return result;
    }
    result = impl_->prepareView(commandBuffer, frameIndex);
    if (!result) {
        impl_->subsystemHost->endFrame();
        impl_->historyResources = nullptr;
        return result;
    }
    StreamerSubsystem* upload = impl_->streamerSubsystem();
    const std::vector<RenderSubsystemId> requiredSubsystems = impl_->requiredSubsystemViews();
    impl_->beginDebugExecution(frameIndex, frameResources ? frameResources->slotIndex() : 0);
    debugScope.observer = impl_->debugObserver;
    result = impl_->subsystemHost->recordPreGraph(
        commandBuffer,
        upload != nullptr ? upload->streamer() : nullptr,
        requiredSubsystems,
        subsystemLog);
    if (result && upload != nullptr) {
        upload->flush(commandBuffer);
    }
    if (!result) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
        std::string cleanupLog;
        (void)impl_->subsystemHost->recordPostGraph(
            commandBuffer,
            upload != nullptr ? upload->streamer() : nullptr,
            requiredSubsystems,
            cleanupLog);
        impl_->subsystemHost->endFrame();
        impl_->historyResources = nullptr;
        return result;
    }
    impl_->lastExecutionStats = RenderGraphExecutionStats{.executionId = frameIndex, .graphGeneration = impl_->profilingGeneration};
    impl_->beginGpuTiming(commandBuffer);
    const auto cpuBegin = std::chrono::steady_clock::now();
    for (Impl::CompiledNode& node : impl_->executionList) {
        result = impl_->executeNode(commandBuffer, node, frameIndex);
        if (!result) {
            break;
        }
    }
    const auto cpuEnd = std::chrono::steady_clock::now();
    impl_->lastExecutionStats.cpuMilliseconds =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();
    impl_->finishGpuTiming(commandBuffer, result.has_value());

    const Result<> graphResult = result;
    Result<> postResult = impl_->subsystemHost->recordPostGraph(
        commandBuffer,
        upload != nullptr ? upload->streamer() : nullptr,
        requiredSubsystems,
        subsystemLog);
    impl_->subsystemHost->endFrame();
    if (!postResult) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
    }

    debugScope.success = graphResult.has_value() && postResult.has_value();
    if (!debugScope.success) { impl_->isCompiled = false; }
    impl_->historyResources = nullptr;
    return graphResult ? postResult : graphResult;
}

void RenderGraphExecutor::bindRuntimeScene(const scene::Scene* scene)
{
    impl_->runtimeScene = scene;
    if (impl_->world != nullptr) {
        impl_->world->setScene(scene);
    }
}

void RenderGraphExecutor::bindRenderWorld(RenderWorld* world)
{
    impl_->world = world != nullptr ? world : &impl_->ownedWorld;
    impl_->runtimeScene = impl_->world->scene();
    impl_->subsystemHost->setWorld(impl_->world);
}

void RenderGraphExecutor::bindRenderView(RenderView* view)
{
    if (impl_->externalView != view) {
        impl_->externalView = view;
        impl_->hasPreviousView = false;
        impl_->isCompiled = false;
    }
}

RenderView* RenderGraphExecutor::renderView()
{
    return impl_->renderView();
}

RenderSubsystemHost* RenderGraphExecutor::subsystemHost()
{
    return impl_->subsystemHost;
}

const RenderSubsystemHost* RenderGraphExecutor::subsystemHost() const
{
    return impl_->subsystemHost;
}

Result<> RenderGraphExecutor::beginSceneResourcePreparation(
    Device& device,
    const RenderGraphProperties& properties,
    const scene::Scene& scene,
    std::string& log)
{
    Queue* graphicsQueue = device.getQueue(QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (!registerBuiltInRenderSubsystems(*impl_->subsystemHost, log)) {
        return makeError(Error::InvalidArgument);
    }
    Result<> result = impl_->subsystemHost->initialize(device,
        impl_->subsystemHost->frameSlotCount() != 0 ? impl_->subsystemHost->frameSlotCount() : 2, log);
    if (!result) {
        return result;
    }
    result = impl_->subsystemHost->activate(StreamerSubsystem::kSubsystemId, log);
    if (!result) {
        return result;
    }
    StreamerSubsystem* sceneResources = impl_->streamerSubsystem();
    if (sceneResources == nullptr) {
        return makeError(Error::Failure);
    }
    cancelSceneResourcePreparation();
    return sceneResources->manager().beginAcquireAsync(
        device,
        *graphicsQueue,
        properties,
        scene,
        SceneResourceFeatureBits::Geometry |
            SceneResourceFeatureBits::Materials |
            SceneResourceFeatureBits::MaterialTextures |
            SceneResourceFeatureBits::Meshlets |
            SceneResourceFeatureBits::StandardAccelerationStructure,
        impl_->pendingSceneResourceSnapshot,
        log);
}

Result<> RenderGraphExecutor::pumpSceneResourcePreparation(
    const scene::Scene& scene,
    double budgetMilliseconds,
    bool& complete,
    scene::SceneLoadProgress& progress,
    std::string& log)
{
    if (impl_->pendingSceneResourceSnapshot == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    StreamerSubsystem* sceneResources = impl_->streamerSubsystem();
    if (sceneResources == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    Result<> result = sceneResources->manager().pumpAsync(
        impl_->pendingSceneResourceSnapshot,
        scene,
        budgetMilliseconds,
        complete,
        progress,
        log);
    return result;
}

void RenderGraphExecutor::cancelSceneResourcePreparation()
{
    StreamerSubsystem* sceneResources = impl_->streamerSubsystem();
    if (sceneResources != nullptr) {
        sceneResources->manager().discard(impl_->pendingSceneResourceSnapshot);
    }
    impl_->pendingSceneResourceSnapshot.reset();
}

void RenderGraphExecutor::acceptSceneResourcePreparation()
{
    impl_->pendingSceneResourceSnapshot.reset();
}

Result<> RenderGraphExecutor::execute(const RenderGraphSubmitDesc& desc)
{
    profiling::SchedulingMetrics scheduling;
    profiling::SchedulingCapture schedulingCapture(desc.schedulingDiagnostics || profiling::SchedulingCapture::requested() ? &scheduling : nullptr);
    CpuProfileRecorder preparation;
    CpuProfileScope preparationPhase(&preparation, "Refresh scene bindings");
    profiling::CpuPhase phase("graph.refreshSceneBindings");
    DebugExecutionScope debugScope;
    if (!impl_->isCompiled || impl_->device == nullptr || impl_->executionList.empty()) {
        return makeError(Error::InvalidArgument);
    }

    std::string sceneLog;
    Result<> sceneResult = impl_->refreshFrameSceneBindings(desc.historyResources, sceneLog);
    if (!sceneResult) { spdlog::error("[RenderGraph] {}", sceneLog); return sceneResult; }

    phase.next("graph.preflight");
    preparationPhase.next("Preflight");
    CpuProfileScope preflightDetail(&preparation, "Validate queue contracts");
    // Preflight before beginning a slot or mutating subsystem/resource state.
    // Unreviewed passes retain the universal-queue execution contract.
    const auto selectedType = [](const Impl::CompiledNode& node) {
        return node.pass->supportsAsyncQueue() ? node.queueType : QueueType::Graphics;
    };
    const auto selectedQueue = [&](QueueType type) {
        Queue* queue = queueForSubmitDesc(desc, type);
        return queue != nullptr ? queue : desc.graphicsQueue;
    };
    const bool subsystemCommands = impl_->requiredSubsystemIds.size() > 1;
    std::vector<std::string> submissionBlockingPasses;
    for (const auto& node : impl_->executionList) {
        if (!node.pass->supportsPipelinedSubmission() || node.preparedScene ||
            node.sceneDependency.source != RenderGraphSceneSource::None || !node.pass->requiredSubsystems().empty()) {
            submissionBlockingPasses.push_back(node.name);
        }
    }
    const bool pipelined = desc.submissionMode == FrameSubmissionMode::Pipelined &&
        submissionBlockingPasses.empty() && !subsystemCommands && !impl_->debugObserver && !desc.historyResources;
    if (desc.graphicsQueue != nullptr && desc.graphicsQueue->type() != QueueType::Graphics) {
        return makeError(Error::InvalidArgument);
    }
    if (subsystemCommands && desc.graphicsQueue == nullptr) { return makeError(Error::InvalidArgument); }
    for (const auto& node : impl_->executionList) {
        const QueueType type = selectedType(node);
        Queue* queue = selectedQueue(type);
        if (queue == nullptr || (type == QueueType::Graphics && queue->type() != QueueType::Graphics) ||
            (type == QueueType::Compute && queue->type() == QueueType::Copy)) {
            return makeError(Error::InvalidArgument);
        }
    }
    preflightDetail.next("Compile resource access plan");
    std::vector<Queue*> nativeQueues;
    std::vector<uint32_t> accessQueues;
    for (const auto& node : impl_->executionList) {
        Queue* queue = selectedQueue(selectedType(node));
        const auto found = std::find_if(nativeQueues.begin(), nativeQueues.end(),
            [&](Queue* previous) { return previous->sameQueue(*queue); });
        const uint32_t identity = uint32_t(found - nativeQueues.begin());
        if (found == nativeQueues.end()) { nativeQueues.push_back(queue); }
        accessQueues.push_back(identity);
    }
    Result<> planned = impl_->buildAccessPlan(accessQueues);
    if (!planned) { return planned; }
    preflightDetail.next("Append incoming waits");
    std::vector<SemaphoreSubmitDesc> initialWaits;
    for (const auto& point : desc.waitCompletions) {
        Result<> result = point.appendWaits(initialWaits);
        if (!result) { return result; }
    }

    // Output consumers are GPU dependencies, not a reason to drain the CPU.
    // Keep unfinished points for rebuild/shutdown and prune completed generations
    // so a continuously presented viewport does not accumulate old frame states.
    preflightDetail.next("Poll / retire external completions");
    std::erase_if(impl_->externalCompletions, [](const auto& point) { return point.isComplete(); });
    preflightDetail.next("Copy external dependencies");
    const auto externalDependencies = impl_->externalCompletions;
    preflightDetail.next("Read scene revisions");
    const scene::Scene* scene = impl_->runtimeScene;
    const std::array<uint64_t, 5> sceneStamp = scene != nullptr
        ? std::array<uint64_t, 5>{scene->resourceIdentity(), scene->contentRevision(),
            scene->sceneGraph().structuralRevision(), scene->transformRevision(), scene->visibilityRevision()}
        : std::array<uint64_t, 5>{};
    preflightDetail.next("Check frame overlap contracts");
    // Evaluate each dynamic contract once; diagnostics use the same result.
    std::vector<std::string> overlapBlockingPasses;
    for (const auto& node : impl_->executionList) {
        if (!node.pass->supportsFrameOverlap()) { overlapBlockingPasses.push_back(node.name); }
    }
    const uint32_t drainReasonMask = (!overlapBlockingPasses.empty() ? 1u : 0u) |
        (sceneStamp != impl_->recordedSceneStamp ? 2u : 0u);
    preflightDetail.end();
    if (drainReasonMask != 0) {
        profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::frameWaitNs);
        phase.next("graph.priorFrameDrain");
        preparationPhase.next("Prior frame drain");
        Result<> result = impl_->waitForSubmittedWork(desc.slotWaitTimeoutNanoseconds);
        if (!result) { return result; }
    }

    const uint64_t frameIndex = impl_->executionFrameIndex;
    const uint32_t slotCount = std::min(2u, impl_->subsystemHost->frameSlotCount());
    if (slotCount == 0) { return makeError(Error::InvalidArgument); }
    Impl::SubmissionSlot& slot = *impl_->submissionSlots[frameIndex % slotCount];
    phase.next("graph.slotWait", frameIndex);
    preparationPhase.next("Submission slot wait");
    Result<> result;
    {
        profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::frameWaitNs);
        result = slot.frame.wait(desc.slotWaitTimeoutNanoseconds);
    }
    if (!result) { return result; }
    phase.next("graph.poolReset");
    preparationPhase.next("Command pool reset");
    slot.commandBuffers.clear();
    for (auto& contexts : slot.recordingContexts) {
        for (auto& context : contexts) {
            result = context->reset();
            if (!result) { return result; }
        }
    }
    for (auto& context : slot.queues) {
        if (context.commandPool != nullptr) {
            result = context.commandPool->reset();
            if (!result) { return result; }
        }
    }
    phase.next("graph.frameBegin");
    preparationPhase.next("Frame begin");
    result = slot.frame.begin(frameIndex, 0, pipelined ? FrameSubmissionMode::Pipelined : FrameSubmissionMode::Joined);
    if (!result) { return result; }
    phase.next("graph.frameSetup");
    preparationPhase.next("Frame setup");
    // Shared graph targets/history remain ordered across frames on the GPU.
    // This bounds CPU recording to two slots without cloning persistent targets.
    result = impl_->lastSubmittedCompletion.appendWaits(initialWaits);
    if (!result) { slot.frame.cancel(); return result; }
    slot.frame.retain(std::make_shared<GpuCompletionPoint>(impl_->lastSubmittedCompletion));
    for (const auto& point : desc.waitCompletions) {
        slot.frame.retain(std::make_shared<GpuCompletionPoint>(point));
    }
    // Every queue segment waits for output readers before reusing shared targets.
    // Frame dependencies retain their timeline semaphores until GPU completion.
    // Do not retire the external points here: a failed recording or destructive
    // graph change must still wait for the consumer that owns those resources.
    for (const auto& point : externalDependencies) {
        result = slot.frame.addDependency(point);
        if (!result) { slot.frame.cancel(); return result; }
    }
    impl_->recordedSceneStamp = sceneStamp;
    impl_->historyResources = desc.historyResources;
    preparationPhase.end();
    const auto cpuBegin = std::chrono::steady_clock::now();
    impl_->lastExecutionStats = RenderGraphExecutionStats{.executionId = frameIndex, .graphGeneration = impl_->profilingGeneration};
    impl_->lastExecutionStats.preparation = std::move(preparation.sections);
    impl_->lastExecutionStats.drainReasonMask = drainReasonMask;
    impl_->lastExecutionStats.externalCompletionCount = uint32_t(externalDependencies.size());
    impl_->lastExecutionStats.overlapBlockingPasses = std::move(overlapBlockingPasses);
    impl_->lastExecutionStats.pipelinedSubmission = pipelined;
    impl_->lastExecutionStats.submissionBlockingPasses = std::move(submissionBlockingPasses);
    const auto updateCpuTime = [&]() {
        scheduling.executeNs = schedulingCapture.elapsed();
        impl_->lastExecutionStats.scheduling = scheduling;
        impl_->lastExecutionStats.cpuMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpuBegin).count();
        for (auto& timing : impl_->gpuTimingSlots) {
            if (timing.pending && timing.stats.executionId == frameIndex) {
                timing.stats.cpuMilliseconds = impl_->lastExecutionStats.cpuMilliseconds;
                timing.stats.submittedBatchCount = impl_->lastExecutionStats.submittedBatchCount;
                timing.stats.batchesSubmittedWhileRecording = impl_->lastExecutionStats.batchesSubmittedWhileRecording;
                timing.stats.scheduling = scheduling;
            }
        }
    };
    const auto abort = [&](Result<> failure) {
        impl_->recordingQueue = nullptr;
        impl_->historyResources = nullptr;
        impl_->activeGpuTimingSlot = nullptr;
        impl_->activeGpuTimingValid = false;
        const bool discardAll = !slot.frame.hasAcceptedWork();
        // Roll back in reverse recording order before destroying individual
        // command buffers; container destruction order is not transaction order.
        slot.frame.cancel(); // Preserves resources for any accepted prefix.
        if (discardAll) {
            slot.commandBuffers.clear();
            for (auto& contexts : slot.recordingContexts) {
                for (auto& context : contexts) { (void)context->reset(); }
            }
            for (auto& context : slot.queues) {
                if (context.commandPool != nullptr) { (void)context.commandPool->reset(); }
            }
        }
        if (slot.frame.completion().isSubmitted()) {
            impl_->lastSubmittedCompletion = slot.frame.completion();
            impl_->hasSubmittedWork = true;
        }
        // Resource states were advanced while recording. Recompile before retrying.
        impl_->isCompiled = false;
        if (desc.historyResources != nullptr) {
            // A partially accepted legacy graph can still reference history.
            (void)slot.frame.wait();
            desc.historyResources->reset();
            (void)desc.historyResources->initialize(*impl_->device);
        }
        updateCpuTime();
        return failure;
    };

    std::string log;
    if (desc.historyResources != nullptr) { desc.historyResources->beginFrame(frameIndex); }
    phase.next("graph.subsystemBegin");
    result = impl_->subsystemHost->beginFrame(frameIndex, slot.frame.slotIndex(),
        desc.historyResources, log, &slot.frame);
    if (!result) { return abort(result); }
    RenderSubsystemFrameEndScope subsystemFrameScope(*impl_->subsystemHost);
    phase.next("graph.record");
    impl_->beginDebugExecution(frameIndex, slot.frame.slotIndex());
    debugScope.observer = impl_->debugObserver;
    StreamerSubsystem* upload = impl_->streamerSubsystem();
    const auto requiredSubsystems = impl_->requiredSubsystemViews();
    std::vector<Impl::SubmissionSegment> segments;
    const bool graphicsTimings = desc.graphicsQueue && impl_->gpuTimestampQueryPools[0];
    const auto beginSegment = [&](QueueType type, CommandRecordingContext* recording = nullptr) -> Result<> {
        Queue* queue = selectedQueue(type);
        Result<> created;
        auto& tracker = impl_->submissionTrackers[queue];
        if (tracker == nullptr) {
            tracker = std::make_unique<QueueSubmissionTracker>();
            created = tracker->initialize(*impl_->device, *queue);
            if (!created) { tracker.reset(); return created; }
        }
        CommandBuffer* commands = nullptr;
        if (recording) {
            created = recording->prepare(slot.frame).transform([&](auto value) { commands = value; });
        } else {
            CommandPool* pool = nullptr;
            created = impl_->prepareCommandPool(slot, type, *queue, pool);
            if (!created) { return created; }
            std::unique_ptr<CommandBuffer> buffer;
            created = pool->createCommandBuffer().transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
            if (!created) { return created; }
            slot.commandBuffers.push_back(std::move(buffer));
            commands = slot.commandBuffers.back().get();
            created = commands->begin(&slot.frame);
        }
        if (!created) { return created; }
        segments.push_back({.queue = queue, .commandBuffer = commands});
        impl_->recordingQueue = queue;
        if (segments.size() == 1) {
            created = impl_->prepareView(*commands, frameIndex);
            if (created && graphicsTimings) { impl_->beginGpuTiming(*commands); }
            return created;
        }
        return {};
    };
    const auto addDependency = [&](size_t destination, size_t source) {
        auto& predecessors = segments[destination].predecessors;
        if (source != destination && std::find(predecessors.begin(), predecessors.end(), source) == predecessors.end()) {
            predecessors.push_back(source);
        }
    };
    struct PendingBatch {
        size_t end = 0;
        uint64_t readyNs = 0;
        RecordedBatch commands;
    };
    std::map<size_t, PendingBatch> sealedBatches;
    size_t nextSubmission = 0;
    std::unordered_set<Queue*> startedQueues;
    // Seal contiguous commands only. Merging across another queue can introduce
    // a cycle (graphics A -> compute B -> graphics C).
    const auto sealRange = [&](size_t first, size_t end, bool coalesce = false, uint64_t readyNs = 0) -> Result<> {
        profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::sealNs);
        while (first < end) {
            size_t last = first + 1;
            while (coalesce && last < end && segments[last].queue == segments[first].queue) { ++last; }
            std::vector<CommandBuffer*> commands;
            for (size_t i = first; i < last; ++i) { commands.push_back(segments[i].commandBuffer); }
            PendingBatch batch{.end = last, .readyNs = readyNs ? readyNs : schedulingCapture.elapsed()};
            auto sealed = batch.commands.seal(slot.frame, commands);
            if (!sealed) { return sealed; }
            sealedBatches.emplace(first, std::move(batch));
            first = last;
        }
        return {};
    };
    // Stable graph order preserves actual-queue order even when queue wrappers
    // alias. A consumer is never submitted against a merely reserved signal.
    const auto submitReady = [&](bool whileRecording = false) -> Result<> {
        for (;;) {
            auto ready = sealedBatches.find(nextSubmission);
            if (ready == sealedBatches.end()) { return {}; }
            const size_t end = ready->second.end;
            Queue* queue = segments[nextSubmission].queue;
            std::vector<SemaphoreSubmitDesc> waits;
            if (!startedQueues.contains(queue)) { waits = initialWaits; }
            for (size_t i = nextSubmission; i < end; ++i) {
                for (size_t predecessor : segments[i].predecessors) {
                    if (predecessor >= nextSubmission && predecessor < end) { continue; }
                    const auto& producer = segments[predecessor];
                    if (!producer.completion.isSubmitted()) { return makeError(Error::InvalidArgument); }
                    if (!producer.queue->sameQueue(*queue)) {
                        auto appended = producer.completion.appendWaits(waits);
                        if (!appended) { return appended; }
                    }
                }
            }
            SubmissionReceipt receipt;
            const uint64_t submitBegin = schedulingCapture.elapsed();
            auto accepted = impl_->submissionTrackers.at(queue)->submitBatch(ready->second.commands, {
                .waitSemaphores = waits.data(), .waitSemaphoreCount = uint32_t(waits.size()),
            }, slot.frame, receipt);
            if (!accepted) { return accepted; }
            if (scheduling.enabled) {
                const uint64_t delay = submitBegin - ready->second.readyNs;
                scheduling.readyDelayNs += delay;
                scheduling.maxReadyDelayNs = std::max(scheduling.maxReadyDelayNs, delay);
                if (!scheduling.firstSubmitNs) { scheduling.firstSubmitNs = submitBegin; }
                if (!scheduling.firstPassSubmitNs && std::any_of(segments.begin() + nextSubmission, segments.begin() + end,
                        [](const auto& segment) { return segment.passWork; })) { scheduling.firstPassSubmitNs = submitBegin; }
            }
            startedQueues.insert(queue);
            for (size_t i = nextSubmission; i < end; ++i) { segments[i].completion = receipt.completion(); }
            ++impl_->lastExecutionStats.submittedBatchCount;
            if (whileRecording) { ++impl_->lastExecutionStats.batchesSubmittedWhileRecording; }
            nextSubmission = end;
            sealedBatches.erase(ready);
        }
    };
    RenderGraphExecutionContext::ParallelRecorder parallel;
    if (desc.computeQueue && desc.graphicsQueue && desc.computeQueue->type() != QueueType::Copy &&
        !desc.computeQueue->sameQueue(*desc.graphicsQueue)) {
        parallel = [&](RenderGraphExecutionContext& context,
            const RenderGraphExecutionContext::CommandRecorder& compute,
            const RenderGraphExecutionContext::CommandRecorder& graphics) -> Result<> {
            const size_t producer = segments.size() - 1;
            if (segments[producer].queue != desc.graphicsQueue ||
                segments[producer].commandBuffer != &context.commandBuffer()) { return makeError(Error::InvalidArgument); }
            Result<> result = segments[producer].commandBuffer->end();
            if (!result) { return result; }
            result = beginSegment(QueueType::Compute);
            if (!result) { return result; }
            const size_t software = segments.size() - 1;
            segments[software].passWork = true;
            addDependency(software, producer);
            result = compute(*segments[software].commandBuffer);
            if (result) { result = segments[software].commandBuffer->end(); }
            if (!result) { return result; }
            result = beginSegment(QueueType::Graphics);
            if (!result) { return result; }
            const size_t hardware = segments.size() - 1;
            segments[hardware].passWork = true;
            addDependency(hardware, producer); // Deliberately independent of software.
            result = graphics(*segments[hardware].commandBuffer);
            if (result) { result = segments[hardware].commandBuffer->end(); }
            if (!result) { return result; }
            result = beginSegment(QueueType::Graphics);
            if (!result) { return result; }
            const size_t join = segments.size() - 1;
            segments[join].passWork = true;
            addDependency(join, hardware);
            addDependency(join, software);
            context.commandBuffer_ = segments[join].commandBuffer;
            ++impl_->lastExecutionStats.asyncComputeBranches;
            return {};
        };
    }
    // The always-present scene resource registry/upload subsystem has no GPU
    // hooks. Explicit subsystem requirements get graphics prologue/epilogue
    // boundaries because their private resource accesses are not graph fields.
    if (subsystemCommands) {
        result = beginSegment(QueueType::Graphics);
        if (!result) { return abort(result); }
        auto& commands = *segments.back().commandBuffer;
        result = impl_->subsystemHost->recordPreGraph(commands,
            upload != nullptr ? upload->streamer() : nullptr, requiredSubsystems, log);
        if (!result) {
            std::string cleanupLog;
            (void)impl_->subsystemHost->recordPostGraph(commands,
                upload != nullptr ? upload->streamer() : nullptr, requiredSubsystems, cleanupLog);
            return abort(result);
        }
        if (upload != nullptr) { upload->flush(commands); }
        result = commands.end();
        if (!result) { return abort(result); }
    }

    // A graphics start/join pair measures elapsed graph time even with independent
    // compute/copy passes. Each pass and inner scope uses its own queue's pool.
    if (!subsystemCommands && graphicsTimings) {
        result = beginSegment(QueueType::Graphics);
        if (result) { result = segments.back().commandBuffer->end(); }
        if (!result) { return abort(result); }
    }
    std::vector<size_t> passCompletions(impl_->executionList.size(), SIZE_MAX);
    result = sealRange(0, segments.size());
    if (result && pipelined) { result = submitReady(); }
    if (!result) { return abort(result); }
    size_t orderingBoundary = segments.empty() ? SIZE_MAX : 0;
    const auto taskSystem = task::detail::tryAcquireTaskSystem();
    const uint32_t workerLimit = taskSystem && !task::isInsideTaskCallback() && !impl_->debugObserver
        ? std::max(1u, std::min(taskSystem->workerCount(), desc.recordingWorkerLimit ? desc.recordingWorkerLimit : 8u)) : 1u;
    impl_->preparationWorkerLimit = workerLimit;
    impl_->preparationBatchWorkload = desc.preparationBatchWorkload;
    struct RecordingBatch {
        CommandRecordingContext* context = nullptr;
        size_t first = 0;
        uint64_t workload = 0;
        std::vector<std::unique_ptr<Impl::NodeRecording>> nodes;
        Result<> result = makeError(Error::Failure);
        std::atomic<bool> done = false;
        bool sealed = false;
        uint64_t readyNs = 0;
        profiling::SchedulingMetrics scheduling;
    };
    std::vector<std::unique_ptr<RecordingBatch>> batches;
    const auto flushRecordings = [&]() -> Result<> {
        if (batches.empty()) { return {}; }
        phase.next("graph.recordBatches", batches.size());
        std::mutex completionMutex;
        std::condition_variable completionCv;
        size_t completedBatches = 0;
        const auto record = [&](RecordingBatch& batch) {
            profiling::SchedulingCapture workerCapture(scheduling.enabled ? &batch.scheduling : nullptr, schedulingCapture.origin());
            try {
                batch.result = batch.context->record([&]() -> Result<> {
                    for (auto& node : batch.nodes) {
                        auto recorded = impl_->recordNode(*node);
                        if (!recorded) { return recorded; }
                    }
                    return {};
                });
            } catch (const std::exception& error) {
                spdlog::error("[RenderGraph] Recording task failed: {}", error.what());
                batch.result = makeError(Error::Failure);
            } catch (...) { batch.result = makeError(Error::Failure); }
            batch.readyNs = workerCapture.elapsed();
            {
                std::lock_guard lock(completionMutex);
                batch.done.store(true, std::memory_order_release);
                ++completedBatches;
            }
            completionCv.notify_one();
        };
        Result<> submitted;
        const auto collectBatches = [&]() {
            // Observe every completed failure before accepting another batch.
            for (const auto& batch : batches) {
                if (batch->done.load(std::memory_order_acquire) && !batch->result) {
                    submitted = batch->result;
                    return;
                }
            }
            if (!submitted) { return; }
            for (auto& batch : batches) {
                if (!batch->sealed && batch->done.load(std::memory_order_acquire)) {
                    submitted = sealRange(batch->first, batch->first + batch->nodes.size(), true, batch->readyNs);
                    if (!submitted) { return; }
                    batch->sealed = true;
                }
            }
            if (pipelined) {
                const bool recording = std::any_of(batches.begin(), batches.end(),
                    [](const auto& batch) { return !batch->done.load(std::memory_order_acquire); });
                submitted = submitReady(recording);
            }
        };
        bool completed = true;
        if (batches.size() > 1) {
            task::TaskGraph graph("Render graph recording");
            for (auto& batch : batches) {
                auto* work = batch.get();
                graph.addTask({.name = "Record " + work->nodes.front()->node->name, .category = "Render recording",
                    .userTag = work->workload}, [&, work] { record(*work); });
            }
            auto run = taskSystem->submit(std::move(graph));
            if (!run) { return makeError(Error::Failure); }
            impl_->lastExecutionStats.recordingTaskCount += uint32_t(batches.size());
            for (const auto& batch : batches) { impl_->lastExecutionStats.parallelRecordedPassCount += uint32_t(batch->nodes.size()); }
            try {
                if (pipelined) {
                    while (!run->isComplete()) {
                        size_t observedCompletions;
                        {
                            std::lock_guard lock(completionMutex);
                            observedCompletions = completedBatches;
                        }
                        collectBatches();
                        if (!submitted || observedCompletions == batches.size()) { break; }
                        std::unique_lock lock(completionMutex);
                        // Bounded polling also handles TaskSystem shutdown cancelling
                        // a job before it can publish done. The predicate preserves
                        // notifications arriving during collection/submission.
                        profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::waitNs);
                        completionCv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                            return completedBatches != observedCompletions || run->isComplete();
                        });
                    }
                }
            } catch (...) { submitted = makeError(Error::Failure); }
            // Always join, including a queue error or exception after acceptance.
            profiling::SchedulingPhase diagnostic(&profiling::SchedulingMetrics::waitNs);
            auto joined = run->wait();
            completed = joined && joined->status == task::TaskGraphStatus::Succeeded;
        } else { record(*batches.front()); }
        if (completed && submitted) { collectBatches(); }
        // Task completion order never changes submission/cancellation order.
        Result<> result = completed ? submitted : makeError(Error::Failure);
        for (auto& batch : batches) {
            if (scheduling.enabled) { scheduling.mergeRecording(batch->scheduling); }
            ++impl_->lastExecutionStats.recordingBatchCount;
            if (result && !batch->result) { result = batch->result; }
            for (auto& node : batch->nodes) { impl_->mergeRecording(*node); }
        }
        batches.clear();
        return result;
    };
    const auto invokePass = [](auto&& callback) -> Result<> {
        try { return callback(); }
        catch (const std::exception& error) { spdlog::error("[RenderGraph] Pass callback failed: {}", error.what()); }
        catch (...) { spdlog::error("[RenderGraph] Pass callback failed with an unknown exception"); }
        return makeError(Error::Failure);
    };
    for (auto& node : impl_->executionList) {
        const size_t passIndex = size_t(&node - impl_->executionList.data());
        const QueueType type = selectedType(node);
        const bool recordOnWorker = workerLimit > 1 &&
            node.pass->cpuRecordingPolicy() == CpuRecordingPolicy::ParallelJoined &&
            !node.preparedScene && node.sceneDependency.source == RenderGraphSceneSource::None &&
            node.pass->requiredSubsystems().empty() && !impl_->firstFeatureReservations.contains(node.name);
        RecordingBatch* batch = nullptr;
        if (recordOnWorker) {
            const uint32_t workload = std::max(1u, node.pass->recordingWorkload());
            const uint32_t target = std::max(1u, desc.recordingBatchWorkload);
            if (batches.empty() || batches.back()->context->queue() != selectedQueue(type) ||
                batches.back()->workload + workload > target) {
                if (batches.size() == workerLimit) {
                    result = flushRecordings();
                    if (!result) { return abort(result); }
                }
                auto& contexts = slot.recordingContexts[Impl::queueContextIndex(type)];
                const size_t lane = batches.size();
                while (contexts.size() <= lane) { contexts.push_back(std::make_unique<CommandRecordingContext>()); }
                auto& context = contexts[lane];
                if (context->queue() != selectedQueue(type)) {
                    context = std::make_unique<CommandRecordingContext>();
                    result = context->initialize(*impl_->device, *selectedQueue(type));
                    if (!result) { return abort(result); }
                }
                auto created = std::make_unique<RecordingBatch>();
                created->context = context.get();
                created->first = segments.size();
                batches.push_back(std::move(created));
            }
            batch = batches.back().get();
            batch->workload += workload;
        } else {
            result = flushRecordings();
            if (!result) { return abort(result); }
        }
        result = beginSegment(type, batch ? batch->context : nullptr);
        if (!result) { return abort(result); }
        const size_t index = segments.size() - 1;
        segments[index].passWork = true;
        if (orderingBoundary != SIZE_MAX) { addDependency(index, orderingBoundary); }
        const bool opaque = !node.pass->supportsAsyncQueue();
        if (opaque) {
            for (size_t previous = 0; previous < index; ++previous) { addDependency(index, previous); }
            orderingBoundary = index;
        }
        // The same access plan supplies both barriers and queue dependencies.
        // A pass with GPU branches completes at its join, not its producer.
        for (const size_t predecessor : impl_->accessPlan.passes[passIndex].predecessors) {
            addDependency(index, passCompletions[predecessor]);
        }
        passCompletions[passIndex] = index;
        if (batch) {
            auto recording = std::make_unique<Impl::NodeRecording>();
            result = invokePass([&] { return impl_->prepareRecording(*recording, *segments[index].commandBuffer, node, frameIndex); });
            if (!result) { return abort(result); }
            batch->nodes.push_back(std::move(recording));
            continue;
        }
        result = invokePass([&] { return impl_->executeNode(*segments[index].commandBuffer, node, frameIndex,
            segments[index].queue == desc.graphicsQueue ? parallel : RenderGraphExecutionContext::ParallelRecorder{}); });
        if (!result) {
            if (subsystemCommands && beginSegment(QueueType::Graphics)) {
                std::string cleanupLog;
                (void)impl_->subsystemHost->recordPostGraph(*segments.back().commandBuffer,
                    upload != nullptr ? upload->streamer() : nullptr, requiredSubsystems, cleanupLog);
            }
            return abort(result);
        }
        const size_t completedIndex = segments.size() - 1;
        if (opaque) { orderingBoundary = completedIndex; }
        passCompletions[passIndex] = completedIndex;
        result = segments[completedIndex].commandBuffer->end();
        if (!result) { return abort(result); }
        result = sealRange(index, completedIndex + 1);
        if (result && pipelined) { result = submitReady(); }
        if (!result) { return abort(result); }
    }
    result = flushRecordings();
    if (!result) { return abort(result); }
    const size_t epilogueBegin = segments.size();
    if (subsystemCommands) {
        result = beginSegment(QueueType::Graphics);
        if (!result) { return abort(result); }
        const size_t index = segments.size() - 1;
        for (size_t previous = 0; previous < index; ++previous) { addDependency(index, previous); }
        result = impl_->subsystemHost->recordPostGraph(*segments[index].commandBuffer,
            upload != nullptr ? upload->streamer() : nullptr, requiredSubsystems, log);
        if (!result) { return abort(result); }
        if (graphicsTimings) { impl_->finishGpuTiming(*segments[index].commandBuffer, true); }
        result = segments[index].commandBuffer->end();
        if (!result) { return abort(result); }
    }
    if (!subsystemCommands && graphicsTimings) {
        result = beginSegment(QueueType::Graphics);
        if (!result) { return abort(result); }
        const size_t index = segments.size() - 1;
        for (size_t previous = 0; previous < index; ++previous) { addDependency(index, previous); }
        impl_->finishGpuTiming(*segments[index].commandBuffer, true);
        result = segments[index].commandBuffer->end();
        if (!result) { return abort(result); }
    }
    impl_->recordingQueue = nullptr;
    impl_->historyResources = nullptr;

    phase.next("graph.submit", segments.size());
    result = sealRange(epilogueBegin, segments.size());
    scheduling.recordingEndNs = schedulingCapture.elapsed();
    if (result) { result = slot.frame.sealRecording(); }
    if (result) { result = submitReady(); }
    if (!result) { return abort(result); }
    if (nextSubmission != segments.size()) { return abort(makeError(Error::Failure)); }
    phase.next("graph.sealAndFinish");
    result = slot.frame.finishSubmission();
    if (!result) { return abort(result); }
    impl_->lastSubmittedCompletion = slot.frame.completion();
    impl_->hasSubmittedWork = true;
    ++impl_->executionFrameIndex;
    updateCpuTime();
    debugScope.success = true;
    return {};
}

GpuCompletionPoint RenderGraphExecutor::lastSubmittedCompletion() const
{
    return impl_->lastSubmittedCompletion;
}

void RenderGraphExecutor::setDebugObserver(IRenderDebugObserver* observer)
{
    impl_->debugObserver = observer;
}

Result<> RenderGraphExecutor::waitForSubmittedWork(uint64_t timeoutNanoseconds)
{
    return impl_->waitForSubmittedWork(timeoutNanoseconds);
}

bool RenderGraphExecutor::syncProperties(const RenderGraph& graph)
{
    return syncRuntimeProperties(graph);
}

bool RenderGraphExecutor::syncRuntimeProperties(const RenderGraph& graph)
{
    if (!impl_->isCompiled) {
        return false;
    }

    bool synced = false;
    for (Impl::CompiledNode& compiledNode : impl_->executionList) {
        const RenderGraphNode* graphNode = graph.findNode(compiledNode.id);
        if (graphNode == nullptr ||
            graphNode->name != compiledNode.name ||
            graphNode->type != compiledNode.type ||
            graphNode->properties != compiledNode.staticProperties) {
            return false;
        }

        if (compiledNode.runtimeProperties != graphNode->runtimeProperties) {
            compiledNode.runtimeProperties = graphNode->runtimeProperties;
            impl_->applySceneProperties(compiledNode, compiledNode.sceneBinding);
            synced = true;
        }
    }
    if (synced && impl_->debugObserver) {
        for (auto& description : impl_->debugGraph["passes"]) {
            const auto node = std::find_if(impl_->executionList.begin(), impl_->executionList.end(),
                [&](const auto& value) { return description.at("id") == value.id; });
            if (node != impl_->executionList.end()) { description["runtimeProperties"] = node->runtimeProperties; }
        }
        impl_->debugGraph["runtimeRevision"] = impl_->debugGraph.value("runtimeRevision", uint64_t(0)) + 1;
        impl_->debugObserver->compiled(impl_->debugGraph);
    }
    return synced;
}

Result<> RenderGraphExecutor::transitionOutput(
    CommandBuffer& commandBuffer,
    std::string_view fullName,
    ResourceState state)
{
    RenderGraphResource* resource = outputResource(fullName);
    if (resource == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (impl_->lastSubmittedCompletion.valid()) {
        Result<> result = commandBuffer.addDependency(impl_->lastSubmittedCompletion);
        if (!result) { return result; }
        if (auto* frame = commandBuffer.frameContext()) {
            if (std::none_of(impl_->externalCompletions.begin(), impl_->externalCompletions.end(),
                    [&](const auto& point) { return point.sameSubmission(frame->completion()); })) {
                impl_->externalCompletions.push_back(frame->completion());
            }
        }
    }
    return impl_->transition(
        commandBuffer,
        *resource,
        state,
        explicitAccessForState(resource->type, state));
}

RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName)
{
    return impl_->resource(fullName);
}

const RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName) const
{
    return impl_->resource(fullName);
}

const RenderGraphExecutionStats& RenderGraphExecutor::executionStats() const
{
    return impl_->lastExecutionStats;
}

Result<> RenderGraphExecutor::collectCompletedGpuExecutionStats(
    std::vector<RenderGraphExecutionStats>& outStats)
{
    outStats.clear();
    Result<> result = impl_->resolveGpuTimings();
    if (!result) {
        return result;
    }
    outStats = std::move(impl_->completedGpuExecutionStats);
    impl_->completedGpuExecutionStats.clear();
    return {};
}

const RenderGraphStreamingStats& RenderGraphExecutor::streamingStats() const
{
    static const RenderGraphStreamingStats kEmptyStats;
    const StreamerSubsystem* upload = impl_->subsystemHost != nullptr
        ? impl_->subsystemHost->get<StreamerSubsystem>()
        : nullptr;
    return upload != nullptr ? upload->stats() : kEmptyStats;
}

bool RenderGraphExecutor::compiled() const
{
    return impl_->isCompiled;
}

uint32_t RenderGraphExecutor::width() const
{
    return impl_->width;
}

uint32_t RenderGraphExecutor::height() const
{
    return impl_->height;
}

struct RenderGraphPreviewRenderer::Impl {
    Impl() : executor(subsystemHost, world) {}

    ~Impl()
    {
        if (device != nullptr) {
            (void)device->waitIdle();
        }
        if (commandPool != nullptr) {
            (void)commandPool->reset();
        }
        (void)frameContext.reset();
        (void)submissions.reset();
    }

    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    QueueSubmissionTracker submissions;
    RenderFrameContext frameContext;
    std::unique_ptr<Buffer> readbackBuffer;
    RenderSubsystemHost subsystemHost;
    RenderWorld world;
    RenderGraphExecutor executor;
    HistoryResourceManager historyResources;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readbackWidth = 0;
    uint32_t readbackHeight = 0;
    uint32_t readbackTexelByteSize = 0;
    uint64_t historyFrameIndex = 0;
    uint32_t recordingWorkerLimit = 0;
    std::string lastLog;

    Result<> ensureReadback(uint32_t newWidth, uint32_t newHeight, uint32_t texelByteSize)
    {
        if (device == nullptr || newWidth == 0 || newHeight == 0 || texelByteSize == 0) {
            return makeError(Error::InvalidArgument);
        }
        const uint32_t allocationTexelByteSize = std::max(texelByteSize, 4u);
        if (readbackBuffer != nullptr &&
            readbackWidth == newWidth &&
            readbackHeight == newHeight &&
            readbackTexelByteSize == allocationTexelByteSize) {
            return {};
        }
        readbackBuffer.reset();
        const uint64_t byteSize =
            static_cast<uint64_t>(newWidth) *
            static_cast<uint64_t>(newHeight) *
            allocationTexelByteSize;
        Result<> result = device->createBuffer(BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
        if (!result) {
            return result;
        }
        readbackWidth = newWidth;
        readbackHeight = newHeight;
        readbackTexelByteSize = allocationTexelByteSize;
        pixels.resize(static_cast<size_t>(newWidth) * static_cast<size_t>(newHeight));
        return {};
    }
};

RenderGraphPreviewRenderer::RenderGraphPreviewRenderer()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphPreviewRenderer::~RenderGraphPreviewRenderer() = default;
RenderGraphPreviewRenderer::RenderGraphPreviewRenderer(RenderGraphPreviewRenderer&&) noexcept = default;
RenderGraphPreviewRenderer& RenderGraphPreviewRenderer::operator=(RenderGraphPreviewRenderer&&) noexcept = default;

void RenderGraphPreviewRenderer::setRecordingWorkerLimit(uint32_t limit)
{
    impl_->recordingWorkerLimit = limit;
}

void RenderGraphPreviewRenderer::setEnvironment(EnvironmentSettings environment)
{
    impl_->world.setEnvironment(std::move(environment));
}

bool RenderGraphPreviewRenderer::setLighting(scene::LightingSettings lighting)
{
    return impl_->world.setLighting(std::move(lighting));
}

void RenderGraphPreviewRenderer::setDebugObserver(IRenderDebugObserver* observer)
{
    impl_->executor.setDebugObserver(observer);
}

void RenderGraphPreviewRenderer::bindRuntimeScene(const scene::Scene* scene)
{
    impl_->executor.bindRuntimeScene(scene);
}

void RenderGraphPreviewRenderer::bindRenderView(RenderView* view)
{
    impl_->executor.bindRenderView(view);
}

RenderSubsystemHost* RenderGraphPreviewRenderer::subsystemHost()
{
    return &impl_->subsystemHost;
}

const RenderSubsystemHost* RenderGraphPreviewRenderer::subsystemHost() const
{
    return &impl_->subsystemHost;
}

const RenderGraphExecutionStats& RenderGraphPreviewRenderer::executionStats() const
{
    return impl_->executor.executionStats();
}

Result<> RenderGraphPreviewRenderer::initialize(bool enableValidation, bool enableRayQuery, bool enableAftermath)
{
    Result<> result = createDevice(DeviceDesc{
            .applicationName = "Metallic RenderGraph Preview",
            .enableValidation = enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableMeshShader = true,
            .enableTaskShader = true,
            .enableTaskShaderSubgroupBallot = true,
            .enableGeometryShader = true,
            .enableSubgroupSizeControl = true,
            .enableComputeFullSubgroups = true,
            .preferredTaskSubgroupSize = 32,
            .enableRayTracingAccelerationStructure = enableRayQuery,
            .enableRayQuery = enableRayQuery,
            .enablePushDescriptor = enableRayQuery,
            .enableClusterAccelerationStructure = enableRayQuery,
            .enableAftermath = enableAftermath,
            .enableAsyncCompute = true,
        }).transform([&](auto rhiValue) { impl_->device = std::move(rhiValue); });
    if (!result) {
        return result;
    }

    impl_->graphicsQueue = impl_->device->getQueue(QueueType::Graphics);
    if (impl_->graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = impl_->device->createCommandPool(*impl_->graphicsQueue).transform([&](auto rhiValue) { impl_->commandPool = std::move(rhiValue); });
    if (!result) {
        return result;
    }
    result = impl_->commandPool->createCommandBuffer().transform([&](auto rhiValue) { impl_->commandBuffer = std::move(rhiValue); });
    if (!result) {
        return result;
    }
    result = impl_->historyResources.initialize(*impl_->device);
    if (!result) {
        return result;
    }
    return impl_->submissions.initialize(*impl_->device, *impl_->graphicsQueue);
}

Result<> RenderGraphPreviewRenderer::render(RenderGraph& graph, uint32_t newWidth, uint32_t newHeight)
{
    return render(graph, newWidth, newHeight, graph.firstOutputName());
}

Result<> RenderGraphPreviewRenderer::render(
    RenderGraph& graph,
    uint32_t newWidth,
    uint32_t newHeight,
    std::string_view outputName, bool readback)
{
    profiling::CpuPhase phase("preview.preflight");
    if (impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->commandPool == nullptr ||
        impl_->commandBuffer == nullptr ||
        newWidth == 0 ||
        newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    const std::string resolvedOutputName = outputName.empty()
        ? graph.firstOutputName()
        : std::string(outputName);
    if (resolvedOutputName.empty()) {
        impl_->lastLog = "RenderGraph preview output resource is missing";
        return makeError(Error::InvalidArgument);
    }

    phase.next("preview.previousReadbackWait");
    Result<> result = impl_->frameContext.wait();
    if (!result) {
        return result;
    }

    phase.next("preview.compileCheck");
    const bool outputCompiled = impl_->executor.compiled() &&
        impl_->executor.outputResource(resolvedOutputName) != nullptr;
    const bool needsCompile =
        graph.dirty() ||
        !impl_->executor.compiled() ||
        impl_->executor.width() != newWidth ||
        impl_->executor.height() != newHeight ||
        !outputCompiled;
    if (needsCompile) {
        phase.next("preview.compile");
        result = impl_->device->waitIdle();
        if (!result) {
            return result;
        }
        impl_->historyResources.invalidateAll();
        impl_->historyFrameIndex = 0;
        RenderGraphCompileOptions options;
        options.extraOutputs.push_back(resolvedOutputName);
        options.enablePreviewOutputAccess = true;
        result = impl_->executor.compile(
            *impl_->device,
            graph,
            newWidth,
            newHeight,
            options,
            impl_->lastLog);
        if (!result) {
            return result;
        }
        graph.clearDirty();
    } else {
        phase.next("preview.syncProperties");
        impl_->executor.syncRuntimeProperties(graph);
    }

    phase.next("preview.outputCheck");
    RenderGraphResource* output = impl_->executor.outputResource(resolvedOutputName);
    if (output == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output resource is missing '") + resolvedOutputName + "'";
        return makeError(Error::InvalidArgument);
    }
    if (output->type != RenderGraphResourceType::Texture2D || output->texture == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output is not a Texture2D '") + resolvedOutputName + "'";
        return makeError(Error::InvalidArgument);
    }
    const uint32_t outputWidth = output->desc.width;
    const uint32_t outputHeight = output->desc.height;
    const uint32_t outputTexelByteSize = previewReadbackTexelByteSize(output->desc.format);
    if (outputWidth == 0 || outputHeight == 0 || outputTexelByteSize == 0) {
        impl_->lastLog =
            std::string("RenderGraph preview output has an unsupported readback format '") +
            resolvedOutputName + "'";
        return makeError(Error::Unsupported);
    }

    if (!readback) {
        ++impl_->historyFrameIndex;
        phase.next("preview.execute");
        result = impl_->executor.execute(RenderGraphSubmitDesc{.graphicsQueue = impl_->graphicsQueue,
            .computeQueue = impl_->device->getQueue(QueueType::Compute), .historyResources = &impl_->historyResources,
            .recordingWorkerLimit = impl_->recordingWorkerLimit});
        phase.next("preview.waitAndCollect");
        if (result) { result = impl_->executor.waitForSubmittedWork(); }
        phase.next("preview.finish");
        impl_->pixels.clear();
        impl_->width = outputWidth;
        impl_->height = outputHeight;
        return result;
    }
    phase.next("preview.readbackSetup");
    result = impl_->ensureReadback(outputWidth, outputHeight, outputTexelByteSize);
    if (!result) {
        return result;
    }

    impl_->pixels.resize(static_cast<size_t>(outputWidth) * outputHeight);
    result = impl_->frameContext.begin(impl_->historyFrameIndex);
    if (!result) {
        return result;
    }
    struct RecordingScope {
        Impl& impl;
        RenderGraph& graph;
        ~RecordingScope()
        {
            if (impl.frameContext.recording()) {
                (void)impl.commandPool->reset();
                impl.frameContext.cancel();
                impl.historyResources.invalidateAll();
                graph.markDirty();
            }
        }
    } recordingScope{*impl_, graph};
    result = impl_->commandPool->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandBuffer->begin(&impl_->frameContext);
    if (!result) {
        return result;
    }

    ++impl_->historyFrameIndex;
    phase.next("preview.execute");
    result = impl_->executor.execute(RenderGraphSubmitDesc{.graphicsQueue = impl_->graphicsQueue,
        .computeQueue = impl_->device->getQueue(QueueType::Compute), .historyResources = &impl_->historyResources,
        .recordingWorkerLimit = impl_->recordingWorkerLimit});
    if (!result) {
        return result;
    }

    phase.next("preview.readbackRecord");
    if (impl_->readbackBuffer == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output resource is missing '") + resolvedOutputName + "'";
        return makeError(Error::InvalidArgument);
    }
    result = impl_->executor.transitionOutput(
        *impl_->commandBuffer,
        resolvedOutputName,
        ResourceState::TransferSource);
    if (!result) {
        return result;
    }
    impl_->commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
        .texture = output->texture,
        .buffer = impl_->readbackBuffer.get(),
        .width = outputWidth,
        .height = outputHeight,
        .depth = 1,
        .mipLevel = 0,
        .baseLayer = 0,
    });

    result = impl_->commandBuffer->end();
    if (!result) {
        return result;
    }

    CommandBuffer* commandBuffers[] = {impl_->commandBuffer.get()};
    phase.next("preview.readbackSubmit");
    result = impl_->submissions.submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
    }, impl_->frameContext);
    if (!result) {
        return result;
    }
    phase.next("preview.readbackWait");
    result = impl_->frameContext.wait();
    if (!result) {
        return result;
    }

    phase.next("preview.readbackConvert");
    impl_->readbackBuffer->invalidate();
    void* mapped = impl_->readbackBuffer->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    const size_t pixelCount = static_cast<size_t>(outputWidth) * static_cast<size_t>(outputHeight);
    if (!convertPreviewReadback(output->desc.format, mapped, pixelCount, impl_->pixels)) {
        impl_->readbackBuffer->unmap();
        impl_->lastLog =
            std::string("RenderGraph preview could not convert output readback '") +
            resolvedOutputName + "'";
        return makeError(Error::Unsupported);
    }
    impl_->readbackBuffer->unmap();

    impl_->width = outputWidth;
    impl_->height = outputHeight;
    return {};
}
Result<> RenderGraphPreviewRenderer::collectCompletedGpuExecutionStats(std::vector<RenderGraphExecutionStats>& outStats)
{
    return impl_->executor.collectCompletedGpuExecutionStats(outStats);
}

const std::vector<uint32_t>& RenderGraphPreviewRenderer::pixels() const
{
    return impl_->pixels;
}

uint32_t RenderGraphPreviewRenderer::width() const
{
    return impl_->width;
}

uint32_t RenderGraphPreviewRenderer::height() const
{
    return impl_->height;
}

const std::string& RenderGraphPreviewRenderer::lastLog() const
{
    return impl_->lastLog;
}

} // namespace metallic::render
