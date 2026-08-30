#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"
#include "Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/Profiling/NsightEvents.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/BuiltinRenderSubsystems.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metallic::render {

using namespace detail;

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
        uint32_t executionWidth = 0;
        uint32_t executionHeight = 0;
    };

    struct QueueCommandContext {
        Queue* queue = nullptr;
        std::unique_ptr<CommandPool> commandPool;
        bool resetForCurrentSubmit = false;
    };

    struct SubmissionSegment {
        QueueType queueType = QueueType::Graphics;
        Queue* queue = nullptr;
        CommandBuffer* commandBuffer = nullptr;
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

    struct GpuTimingSlot {
        uint32_t firstQuery = 0;
        uint32_t queryCount = 0;
        bool pending = false;
        RenderGraphExecutionStats stats;
    };

    static constexpr uint32_t kGpuTimingSlotCount = 3;

    Device* device = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    Format defaultFormat = Format::Rgba8Unorm;
    HistoryResourceManager* historyResources = nullptr;
    const scene::Scene* runtimeScene = nullptr;
    RenderSubsystemHost ownedSubsystemHost;
    RenderWorld ownedWorld;
    RenderSubsystemHost* subsystemHost = &ownedSubsystemHost;
    RenderWorld* world = &ownedWorld;
    std::vector<CompiledNode> executionList;
    std::unordered_map<std::string, ResourceSlot> resources;
    std::unordered_map<std::string, std::string> inputAliases;
    std::unique_ptr<BindlessHeap> bindlessHeap;
    std::shared_ptr<SceneResourceSnapshot> pendingSceneResourceSnapshot;
    std::vector<std::string> requiredSubsystemIds;
    std::array<QueueCommandContext, 3> queueCommandContexts;
    std::vector<std::unique_ptr<CommandBuffer>> submittedCommandBuffers;
    std::unique_ptr<Semaphore> submittedTimelineSemaphore;
    uint64_t submittedTimelineValue = 0;
    std::unique_ptr<TimestampQueryPool> gpuTimestampQueryPool;
    std::array<GpuTimingSlot, kGpuTimingSlotCount> gpuTimingSlots;
    std::vector<RenderGraphExecutionStats> completedGpuExecutionStats;
    GpuTimingSlot* activeGpuTimingSlot = nullptr;
    uint32_t nextGpuTimingSlot = 0;
    bool activeGpuTimingValid = false;
    RenderGraphExecutionStats lastExecutionStats;
    uint64_t executionFrameIndex = 0;
    bool hasSubmittedWork = false;
    bool isCompiled = false;

    RenderUploadSubsystem* uploadSubsystem() const
    {
        return subsystemHost != nullptr ? subsystemHost->get<RenderUploadSubsystem>() : nullptr;
    }

    SceneResourcesSubsystem* sceneResourcesSubsystem() const
    {
        return subsystemHost != nullptr ? subsystemHost->get<SceneResourcesSubsystem>() : nullptr;
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

    Result refreshReusablePasses(
        const RenderGraph& graph,
        const ActiveGraph& activeGraph,
        const RenderGraphCompileContext& compileContext,
        std::string& log)
    {
        if (executionList.size() != activeGraph.executionOrder.size()) {
            log = validationPrefix("compiled pass list does not match active graph");
            return makeError(Error::InvalidArgument);
        }

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

            compiledNode.runtimeProperties = graphNode->runtimeProperties;
            compiledNode.effectiveProperties = mergeRenderGraphProperties(
                compiledNode.staticProperties,
                compiledNode.runtimeProperties);

            compiledNode.pass->setProperties(compiledNode.effectiveProperties);
            std::string prepareLog;
            Result prepareResult = compiledNode.pass->prepare(compileContext, prepareLog);
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
            compiledNode.reflection = compiledNode.pass->reflect(compileContext);
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

    Result resolveTextureOutputExtents(
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
            bool changed = false;

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

    Result resolveNodeExecutionExtents(std::string& log)
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

    Result allocateGraphResources(
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
        Result extentConstraintResult = resolveTextureOutputExtents(
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
                    if (isOutputMarked(graph, fullName) || options.enablePreviewOutputAccess) {
                        usage = addTextureUsage(usage, TextureUsageBits::TransferSource);
                        usage = addTextureUsage(usage, TextureUsageBits::Sampled);
                    }
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
                    };

                    Result result = graphDevice.createTexture(desc, slot.texture);
                    if (!result || slot.texture == nullptr) {
                        log += resultMessage(std::string("createTexture(") + fullName + ")", result);
                        log += '\n';
                        return result ? makeError(Error::Failure) : result;
                    }
                    result = graphDevice.createTextureView(
                        *slot.texture,
                        TextureViewDesc{
                            .format = desc.format,
                            .baseMip = 0,
                            .mipCount = 1,
                            .baseLayer = 0,
                            .layerCount = 1,
                        },
                        slot.textureView);
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
                        if (dstField->access == RenderGraphResourceAccess::BufferStorageReadWrite) {
                            viewType = dstField->structureStride == 0
                                ? BufferViewType::ReadWriteRaw
                                : BufferViewType::ReadWriteStructured;
                        }
                    }

                    const bool markedBufferOutput = isOutputMarked(graph, fullName);
                    BufferDesc desc{
                        .size = field.size,
                        .structureStride = field.structureStride,
                        .usage = usage,
                        .memoryLocation = markedBufferOutput
                            ? MemoryLocation::HostReadback
                            : field.memoryLocation,
                    };

                    Result result = graphDevice.createBuffer(desc, slot.buffer);
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
                        result = graphDevice.createBufferView(*slot.buffer, viewDesc, slot.bufferView);
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

        Result extentResult = resolveNodeExecutionExtents(log);
        if (!extentResult) {
            return extentResult;
        }

        if (!bindlessPlan.sampledImageResources.empty() || !bindlessPlan.bufferResources.empty()) {
            Result result = graphDevice.createBindlessHeap(
                BindlessHeapDesc{
                    .maxSampledImages = static_cast<uint32_t>(bindlessPlan.sampledImageResources.size()),
                    .maxBuffers = static_cast<uint32_t>(bindlessPlan.bufferResources.size()),
                },
                bindlessHeap);
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
                result = bindlessHeap->allocateSampledImage(handle);
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
                result = bindlessHeap->allocateBuffer(handle);
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

    Result rebuildGraphResources(
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

    QueueCommandContext& queueCommandContext(QueueType type)
    {
        return queueCommandContexts[queueContextIndex(type)];
    }

    Result waitForSubmittedWork(uint64_t timeoutNanoseconds)
    {
        if (!hasSubmittedWork) {
            return {};
        }

        if (submittedTimelineSemaphore != nullptr) {
            Result result = submittedTimelineSemaphore->wait(submittedTimelineValue, timeoutNanoseconds);
            if (!result) {
                return result;
            }
        }

        hasSubmittedWork = false;
        submittedTimelineValue = 0;
        submittedCommandBuffers.clear();
        submittedTimelineSemaphore.reset();
        return {};
    }

    void initializeGpuTiming(Device& graphDevice)
    {
        gpuTimestampQueryPool.reset();
        gpuTimingSlots = {};
        completedGpuExecutionStats.clear();
        activeGpuTimingSlot = nullptr;
        nextGpuTimingSlot = 0;
        activeGpuTimingValid = false;

        Queue* graphicsQueue = graphDevice.getQueue(QueueType::Graphics);
        if (executionList.empty() ||
            graphicsQueue == nullptr ||
            !graphDevice.capabilities().timestampQueries ||
            graphicsQueue->timestampValidBits() == 0) {
            return;
        }

        const uint64_t queriesPerSlot64 = static_cast<uint64_t>(executionList.size()) * 2ull;
        const uint64_t totalQueryCount64 = queriesPerSlot64 * kGpuTimingSlotCount;
        if (queriesPerSlot64 > std::numeric_limits<uint32_t>::max() ||
            totalQueryCount64 > std::numeric_limits<uint32_t>::max()) {
            spdlog::warn("[RenderGraph] GPU timing disabled because the timestamp query count is too large");
            return;
        }

        const uint32_t queriesPerSlot = static_cast<uint32_t>(queriesPerSlot64);
        Result result = graphDevice.createTimestampQueryPool(
            *graphicsQueue,
            TimestampQueryPoolDesc{.queryCount = static_cast<uint32_t>(totalQueryCount64)},
            gpuTimestampQueryPool);
        if (!result || gpuTimestampQueryPool == nullptr) {
            spdlog::warn(
                "[RenderGraph] GPU timestamp queries are unavailable: {}",
                resultToString(result));
            gpuTimestampQueryPool.reset();
            return;
        }

        for (uint32_t slotIndex = 0; slotIndex < kGpuTimingSlotCount; ++slotIndex) {
            gpuTimingSlots[slotIndex].firstQuery = slotIndex * queriesPerSlot;
            gpuTimingSlots[slotIndex].queryCount = queriesPerSlot;
        }
    }

    Result resolveGpuTimings()
    {
        if (gpuTimestampQueryPool == nullptr) {
            return {};
        }

        for (GpuTimingSlot& slot : gpuTimingSlots) {
            if (!slot.pending || slot.queryCount == 0) {
                continue;
            }

            std::vector<TimestampQueryResult> queryResults(slot.queryCount);
            Result result = gpuTimestampQueryPool->readResults(
                slot.firstQuery,
                slot.queryCount,
                queryResults.data());
            if (!result) {
                return result;
            }
            if (!std::all_of(
                    queryResults.begin(),
                    queryResults.end(),
                    [](const TimestampQueryResult& value) { return value.available; })) {
                continue;
            }

            const size_t timedNodeCount = std::min(
                slot.stats.nodes.size(),
                queryResults.size() / 2);
            for (size_t nodeIndex = 0; nodeIndex < timedNodeCount; ++nodeIndex) {
                RenderGraphNodeExecutionStat& node = slot.stats.nodes[nodeIndex];
                const TimestampQueryResult& begin = queryResults[nodeIndex * 2];
                const TimestampQueryResult& end = queryResults[nodeIndex * 2 + 1];
                node.gpuMilliseconds = gpuTimestampQueryPool->durationMilliseconds(
                    begin.value,
                    end.value);
                node.gpuTimingAvailable = true;
            }
            if (timedNodeCount > 0) {
                slot.stats.gpuMilliseconds = gpuTimestampQueryPool->durationMilliseconds(
                    queryResults.front().value,
                    queryResults[timedNodeCount * 2 - 1].value);
                slot.stats.gpuTimingAvailable = true;
            }

            completedGpuExecutionStats.push_back(std::move(slot.stats));
            slot.stats = {};
            slot.pending = false;
        }
        return {};
    }

    void beginGpuTiming(CommandBuffer& commandBuffer)
    {
        activeGpuTimingSlot = nullptr;
        activeGpuTimingValid = false;
        if (gpuTimestampQueryPool == nullptr) {
            return;
        }

        Result resolveResult = resolveGpuTimings();
        if (!resolveResult) {
            spdlog::warn(
                "[RenderGraph] Failed to resolve GPU timestamp queries: {}",
                resultToString(resolveResult));
            return;
        }

        for (uint32_t offset = 0; offset < kGpuTimingSlotCount; ++offset) {
            const uint32_t slotIndex = (nextGpuTimingSlot + offset) % kGpuTimingSlotCount;
            GpuTimingSlot& slot = gpuTimingSlots[slotIndex];
            if (slot.pending) {
                continue;
            }

            Result resetResult = commandBuffer.resetTimestampQueries(
                *gpuTimestampQueryPool,
                slot.firstQuery,
                slot.queryCount);
            if (!resetResult) {
                return;
            }
            slot.stats = {};
            activeGpuTimingSlot = &slot;
            activeGpuTimingValid = true;
            nextGpuTimingSlot = (slotIndex + 1) % kGpuTimingSlotCount;
            return;
        }
    }

    void finishGpuTiming(bool completed)
    {
        if (activeGpuTimingSlot == nullptr) {
            return;
        }

        if (completed &&
            activeGpuTimingValid &&
            lastExecutionStats.nodes.size() * 2 == activeGpuTimingSlot->queryCount) {
            activeGpuTimingSlot->stats = lastExecutionStats;
            activeGpuTimingSlot->pending = true;
        } else {
            activeGpuTimingSlot->stats = {};
            activeGpuTimingSlot->pending = false;
        }
        activeGpuTimingSlot = nullptr;
        activeGpuTimingValid = false;
    }

    Result prepareCommandPool(QueueType type, Queue& queue, CommandPool*& outCommandPool)
    {
        outCommandPool = nullptr;
        QueueCommandContext& context = queueCommandContext(type);
        if (context.queue != &queue || context.commandPool == nullptr) {
            context.commandPool.reset();
            Result result = device->createCommandPool(queue, context.commandPool);
            if (!result || context.commandPool == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
            context.queue = &queue;
        }

        if (!context.resetForCurrentSubmit) {
            Result result = context.commandPool->reset();
            if (!result) {
                return result;
            }
            context.resetForCurrentSubmit = true;
        }

        outCommandPool = context.commandPool.get();
        return {};
    }

    bool hasCrossQueueResourceEdges(std::string& log) const
    {
        for (const auto& [inputName, outputName] : inputAliases) {
            std::string inputPass;
            std::string inputField;
            std::string outputPass;
            std::string outputField;
            if (!splitRenderGraphFieldName(inputName, inputPass, inputField) ||
                !splitRenderGraphFieldName(outputName, outputPass, outputField)) {
                continue;
            }

            const CompiledNode* inputNode = compiledNode(inputPass);
            const CompiledNode* outputNode = compiledNode(outputPass);
            if (inputNode == nullptr || outputNode == nullptr) {
                continue;
            }

            if (inputNode->queueType != outputNode->queueType) {
                log = std::string("RenderGraph multi-queue submission does not yet support "
                    "cross-queue resource edges: ") +
                    outputName +
                    " (" +
                    queueTypeName(outputNode->queueType) +
                    ") -> " +
                    inputName +
                    " (" +
                    queueTypeName(inputNode->queueType) +
                    ")";
                return true;
            }
        }

        return false;
    }

    Result transition(
        CommandBuffer& commandBuffer,
        RenderGraphResource& resource,
        ResourceState state,
        RenderGraphResourceAccess access)
    {
        const bool needsSameStateStorageBarrier =
            resource.state == state &&
            state == ResourceState::General &&
            (accessWrites(resource.lastAccess) || accessWrites(access));
        if (resource.state == state && !needsSameStateStorageBarrier) {
            resource.lastAccess = access;
            return {};
        }

        if (resource.type == RenderGraphResourceType::Texture2D) {
            if (resource.texture == nullptr) {
                return {};
            }
            TextureBarrierDesc barrier{
                .texture = resource.texture,
                .before = resource.state,
                .after = state,
                .baseMip = 0,
                .mipCount = resource.desc.mipCount,
                .baseLayer = 0,
                .layerCount = resource.desc.layerCount,
            };
            commandBuffer.barrier(BarrierDesc{
                .textures = &barrier,
                .textureCount = 1,
            });
        } else {
            if (resource.buffer == nullptr) {
                return {};
            }
            BufferBarrierDesc barrier{
                .buffer = resource.buffer,
                .before = resource.state,
                .after = state,
                .offset = 0,
                .size = resource.bufferDesc.size,
            };
            commandBuffer.barrier(BarrierDesc{
                .buffers = &barrier,
                .bufferCount = 1,
            });
        }

        resource.state = state;
        resource.lastAccess = access;
        return {};
    }

    Result executeNode(CommandBuffer& commandBuffer, CompiledNode& node)
    {
        std::vector<RenderGraphExecutionContext::Binding> bindings;

        for (const RenderGraphField& field : node.reflection.fields()) {
            const std::string localName = field.name;
            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            RenderGraphResource* resource = nullptr;

            if (field.visibility == RenderGraphFieldVisibility::Output) {
                resource = this->resource(fullName);
                if (resource != nullptr) {
                    Result result = transition(
                        commandBuffer,
                        *resource,
                        stateForAccess(field.access),
                        field.access);
                    if (!result) {
                        return result;
                    }
                }
            } else {
                const auto alias = inputAliases.find(fullName);
                if (alias != inputAliases.end()) {
                    resource = this->resource(alias->second);
                    if (resource != nullptr) {
                        Result result = transition(
                            commandBuffer,
                            *resource,
                            stateForAccess(field.access),
                            field.access);
                        if (!result) {
                            return result;
                        }
                    }
                }
            }

            bindings.push_back(RenderGraphExecutionContext::Binding{
                .fieldName = localName,
                .resource = resource,
                .visibility = field.visibility,
                .bindlessAccess = field.bindlessAccess,
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

        RenderUploadSubsystem* upload = uploadSubsystem();
        RenderGraphExecutionContext context(
            commandBuffer,
            node.executionWidth,
            node.executionHeight,
            node.name,
            node.effectiveProperties,
            std::move(bindings),
            historyResources,
            upload != nullptr ? upload->streamer() : nullptr,
            runtimeScene,
            world,
            subsystemHost);
        const std::string markerName = passProfileMarkerName(node.name, node.type);
        const uint32_t markerColor = profiling::nsightColorFromName(node.type);
        const profiling::NsightProfileRange passMarker(
            profiling::NsightDomain::Render,
            markerName.c_str(),
            profiling::NsightCategory::RenderPass,
            node.id,
            markerColor);
        commandBuffer.beginDebugLabel(DebugLabelDesc{
            .name = markerName.c_str(),
            .color = debugLabelColorFromArgb(markerColor),
        });
        bool gpuTimingRecorded = false;
        uint32_t gpuEndQuery = 0;
        if (activeGpuTimingSlot != nullptr && activeGpuTimingValid) {
            const uint32_t gpuBeginQuery = activeGpuTimingSlot->firstQuery +
                static_cast<uint32_t>(lastExecutionStats.nodes.size()) * 2u;
            gpuEndQuery = gpuBeginQuery + 1u;
            Result timestampResult = commandBuffer.writeTimestamp(
                *gpuTimestampQueryPool,
                gpuBeginQuery,
                PipelineStageBits::TopOfPipe);
            gpuTimingRecorded = timestampResult.has_value();
            activeGpuTimingValid = gpuTimingRecorded;
        }
        const auto cpuBegin = std::chrono::steady_clock::now();
        Result result = node.pass->execute(context);
        if (result && upload != nullptr) {
            upload->flush(commandBuffer);
        }
        const auto cpuEnd = std::chrono::steady_clock::now();
        if (gpuTimingRecorded) {
            Result timestampResult = commandBuffer.writeTimestamp(
                *gpuTimestampQueryPool,
                gpuEndQuery,
                PipelineStageBits::BottomOfPipe);
            activeGpuTimingValid = timestampResult.has_value();
        }
        commandBuffer.endDebugLabel();
        lastExecutionStats.nodes.push_back(RenderGraphNodeExecutionStat{
            .id = node.id,
            .name = node.name,
            .type = node.type,
            .cpuMilliseconds = std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count(),
        });
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

RenderGraphExecutor::~RenderGraphExecutor()
{
    if (impl_ != nullptr) {
        (void)impl_->waitForSubmittedWork(UINT64_MAX);
    }
}
RenderGraphExecutor::RenderGraphExecutor(RenderGraphExecutor&&) noexcept = default;
RenderGraphExecutor& RenderGraphExecutor::operator=(RenderGraphExecutor&&) noexcept = default;

Result RenderGraphExecutor::compile(
    Device& device,
    const RenderGraph& graph,
    uint32_t width,
    uint32_t height,
    std::string& log)
{
    return compile(device, graph, width, height, RenderGraphCompileOptions{}, log);
}

Result RenderGraphExecutor::compile(
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

    Result pendingResult;
    {
        RenderGraphLogScope scope("wait for previous submitted RenderGraph work");
        pendingResult = impl_->waitForSubmittedWork(UINT64_MAX);
    }
    if (!pendingResult) {
        log = resultMessage("RenderGraph waitForSubmittedWork", pendingResult);
        impl_->isCompiled = false;
        return pendingResult;
    }

    if (impl_->device != nullptr && impl_->device != &device) {
        for (Impl::QueueCommandContext& queueContext : impl_->queueCommandContexts) {
            queueContext.queue = nullptr;
            queueContext.commandPool.reset();
            queueContext.resetForCurrentSubmit = false;
        }
        impl_->pendingSceneResourceSnapshot.reset();
    }

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
    Result subsystemResult = impl_->subsystemHost->initialize(device, 3, log);
    if (!subsystemResult) {
        impl_->isCompiled = false;
        return subsystemResult;
    }
    impl_->subsystemHost->setWorld(impl_->world);

    impl_->requiredSubsystemIds.clear();
    impl_->requiredSubsystemIds.emplace_back(SceneResourcesSubsystem::kSubsystemId);
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
    subsystemResult = impl_->subsystemHost->activate(SceneResourcesSubsystem::kSubsystemId, log);
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
    const bool canReuseCompiledPasses = impl_->canReuseCompiledPasses(device, graph, activeGraph);

    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;

    SceneResourcesSubsystem* sceneResources = impl_->sceneResourcesSubsystem();
    if (sceneResources == nullptr) {
        log = "RenderGraph compile failed: render.scene-resources was not activated";
        impl_->isCompiled = false;
        return makeError(Error::Failure);
    }

    const RenderGraphCompileContext compileContext{
        .device = &device,
        .graphicsQueue = device.getQueue(QueueType::Graphics),
        .runtimeScene = impl_->runtimeScene,
        .sceneResourceManager = &sceneResources->manager(),
        .renderWorld = impl_->world,
        .subsystemHost = impl_->subsystemHost,
        .width = width,
        .height = height,
        .defaultFormat = impl_->defaultFormat,
    };

    if (canReuseCompiledPasses) {
        impl_->isCompiled = false;
        Result refreshResult;
        {
            RenderGraphLogScope scope("refresh reusable passes");
            refreshResult = impl_->refreshReusablePasses(graph, activeGraph, compileContext, log);
        }
        if (!refreshResult) {
            return refreshResult;
        }

        Result resourceResult;
        {
            RenderGraphLogScope scope("rebuild graph resources");
            resourceResult = impl_->rebuildGraphResources(device, graph, activeGraph, options, log);
        }
        if (!resourceResult) {
            return resourceResult;
        }

        impl_->isCompiled = true;
        log = dimensionsChanged ? "RenderGraph resized" : "RenderGraph resources rebuilt";
        return {};
    }

    impl_->executionList.clear();
    impl_->resources.clear();
    impl_->inputAliases.clear();
    impl_->bindlessHeap.reset();
    impl_->isCompiled = false;

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
        std::string prepareLog;
        Result prepareResult = pass->prepare(compileContext, prepareLog);
        if (!prepareResult) {
            log = "RenderGraph prepare failed for pass '" + node->name + "' (" + node->type + ")";
            if (!prepareLog.empty()) {
                log += ": " + prepareLog;
            }
            return prepareResult;
        }
        const RenderGraphPassKind kind = pass->kind();
        const QueueType queueType = pass->queueType();
        RenderPassReflection reflection = pass->reflect(compileContext);
        impl_->executionList.push_back(Impl::CompiledNode{
            .id = node->id,
            .name = node->name,
            .type = node->type,
            .kind = kind,
            .queueType = queueType,
            .staticProperties = node->properties,
            .runtimeProperties = node->runtimeProperties,
            .effectiveProperties = effectiveProperties,
            .pass = std::move(pass),
            .reflection = std::move(reflection),
        });
    }
    spdlog::info("[RenderGraph] Created {} compiled pass objects", impl_->executionList.size());

    impl_->rebuildInputAliases(graph, activeGraph);
    Impl::BindlessResourcePlan bindlessPlan = impl_->collectBindlessResourcePlan();
    if ((!bindlessPlan.sampledImageResources.empty() || !bindlessPlan.bufferResources.empty()) &&
        !device.capabilities().bindlessDescriptorHeap) {
        log = "RenderGraph compile failed: bindless resources require "
            "DeviceCapabilities::bindlessDescriptorHeap";
        return makeError(Error::Unsupported);
    }

    for (Impl::CompiledNode& node : impl_->executionList) {
        Result result;
        {
            RenderGraphLogScope scope(
                "compile pass '" + node.name + "' (" + node.type + ")");
            result = node.pass->compile(compileContext, log);
        }
        if (!result) {
            impl_->isCompiled = false;
            return result;
        }
        node.pass->setProperties(node.effectiveProperties);
    }

    Result resourceResult;
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
    impl_->isCompiled = true;
    log = "RenderGraph compiled";
    return {};
}

Result RenderGraphExecutor::reloadShaders(std::string& log)
{
    RenderGraphLogScope reloadScope("transactional shader reload");
    log.clear();
    if (!impl_->isCompiled || impl_->device == nullptr || impl_->executionList.empty()) {
        log = "RenderGraph shader reload requires a compiled graph";
        return makeError(Error::InvalidArgument);
    }

    Result result = impl_->waitForSubmittedWork(UINT64_MAX);
    if (!result) {
        log = resultMessage("RenderGraph waitForSubmittedWork before shader reload", result);
        return result;
    }
    result = impl_->device->waitIdle();
    if (!result) {
        log = resultMessage("Device waitIdle before shader reload", result);
        return result;
    }

    SceneResourcesSubsystem* sceneResources = impl_->sceneResourcesSubsystem();
    if (sceneResources == nullptr) {
        log = "RenderGraph shader reload requires render.scene-resources";
        return makeError(Error::Failure);
    }
    const RenderGraphCompileContext compileContext{
        .device = impl_->device,
        .graphicsQueue = impl_->device->getQueue(QueueType::Graphics),
        .runtimeScene = impl_->runtimeScene,
        .sceneResourceManager = &sceneResources->manager(),
        .renderWorld = impl_->world,
        .subsystemHost = impl_->subsystemHost,
        .width = impl_->width,
        .height = impl_->height,
        .defaultFormat = impl_->defaultFormat,
    };

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
        pass->setProperties(compiledNode.effectiveProperties);
        std::string prepareLog;
        result = pass->prepare(compileContext, prepareLog);
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
        RenderPassReflection reflection = pass->reflect(compileContext);
        const std::span<const RenderSubsystemId> oldSubsystems =
            compiledNode.pass->requiredSubsystems();
        const std::span<const RenderSubsystemId> newSubsystems = pass->requiredSubsystems();
        const bool subsystemRequirementsMatch =
            oldSubsystems.size() == newSubsystems.size() &&
            std::equal(oldSubsystems.begin(), oldSubsystems.end(), newSubsystems.begin());
        if (kind != compiledNode.kind ||
            queueType != compiledNode.queueType ||
            reflection != compiledNode.reflection ||
            !subsystemRequirementsMatch) {
            log = "Shader reload rejected pass '" + compiledNode.name +
                "' because its render-graph contract changed; rebuild the graph instead";
            return makeError(Error::InvalidArgument);
        }

        std::string passLog;
        {
            RenderGraphLogScope scope(
                "reload shaders for pass '" + compiledNode.name + "' (" +
                compiledNode.type + ")");
            result = pass->compile(compileContext, passLog);
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

Result RenderGraphExecutor::execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources)
{
    if (!impl_->isCompiled) {
        return makeError(Error::InvalidArgument);
    }

    impl_->historyResources = historyResources;
    std::string subsystemLog;
    const uint64_t frameIndex = impl_->executionFrameIndex++;
    const profiling::NsightProfileRange executeMarker(
        profiling::NsightDomain::Render,
        "Render Graph Execute",
        profiling::NsightCategory::RenderGraph,
        frameIndex);
    Result result = impl_->subsystemHost->beginFrame(
        frameIndex,
        static_cast<uint32_t>(frameIndex % impl_->subsystemHost->frameSlotCount()),
        historyResources,
        subsystemLog);
    if (!result) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
        impl_->historyResources = nullptr;
        return result;
    }
    RenderUploadSubsystem* upload = impl_->uploadSubsystem();
    const std::vector<RenderSubsystemId> requiredSubsystems = impl_->requiredSubsystemViews();
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
    impl_->lastExecutionStats = RenderGraphExecutionStats{.executionId = frameIndex};
    impl_->beginGpuTiming(commandBuffer);
    const auto cpuBegin = std::chrono::steady_clock::now();
    for (Impl::CompiledNode& node : impl_->executionList) {
        result = impl_->executeNode(commandBuffer, node);
        if (!result) {
            break;
        }
    }
    const auto cpuEnd = std::chrono::steady_clock::now();
    impl_->lastExecutionStats.cpuMilliseconds =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();

    const Result graphResult = result;
    Result postResult = impl_->subsystemHost->recordPostGraph(
        commandBuffer,
        upload != nullptr ? upload->streamer() : nullptr,
        requiredSubsystems,
        subsystemLog);
    impl_->subsystemHost->endFrame();
    if (!postResult) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
    }

    impl_->finishGpuTiming(graphResult.has_value() && postResult.has_value());
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

RenderSubsystemHost* RenderGraphExecutor::subsystemHost()
{
    return impl_->subsystemHost;
}

const RenderSubsystemHost* RenderGraphExecutor::subsystemHost() const
{
    return impl_->subsystemHost;
}

Result RenderGraphExecutor::beginSceneResourcePreparation(
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
    Result result = impl_->subsystemHost->initialize(device, 3, log);
    if (!result) {
        return result;
    }
    result = impl_->subsystemHost->activate(SceneResourcesSubsystem::kSubsystemId, log);
    if (!result) {
        return result;
    }
    SceneResourcesSubsystem* sceneResources = impl_->sceneResourcesSubsystem();
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

Result RenderGraphExecutor::pumpSceneResourcePreparation(
    const scene::Scene& scene,
    double budgetMilliseconds,
    bool& complete,
    scene::SceneLoadProgress& progress,
    std::string& log)
{
    if (impl_->pendingSceneResourceSnapshot == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    SceneResourcesSubsystem* sceneResources = impl_->sceneResourcesSubsystem();
    if (sceneResources == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    Result result = sceneResources->manager().pumpAsync(
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
    SceneResourcesSubsystem* sceneResources = impl_->sceneResourcesSubsystem();
    if (sceneResources != nullptr) {
        sceneResources->manager().discard(impl_->pendingSceneResourceSnapshot);
    }
    impl_->pendingSceneResourceSnapshot.reset();
}

void RenderGraphExecutor::acceptSceneResourcePreparation()
{
    impl_->pendingSceneResourceSnapshot.reset();
}

Result RenderGraphExecutor::execute(const RenderGraphSubmitDesc& desc)
{
    if (!impl_->isCompiled || impl_->device == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    impl_->historyResources = nullptr;
    Result result = impl_->waitForSubmittedWork(UINT64_MAX);
    if (!result) {
        return result;
    }

    std::string crossQueueLog;
    if (impl_->hasCrossQueueResourceEdges(crossQueueLog)) {
        return makeError(Error::Unsupported);
    }

    std::string subsystemLog;
    const uint64_t frameIndex = impl_->executionFrameIndex++;
    result = impl_->subsystemHost->beginFrame(
        frameIndex,
        static_cast<uint32_t>(frameIndex % impl_->subsystemHost->frameSlotCount()),
        nullptr,
        subsystemLog);
    if (!result) {
        spdlog::error("[RenderGraph] {}", subsystemLog);
        return result;
    }
    RenderSubsystemFrameEndScope subsystemFrameScope(*impl_->subsystemHost);
    RenderUploadSubsystem* upload = impl_->uploadSubsystem();
    const std::vector<RenderSubsystemId> requiredSubsystems = impl_->requiredSubsystemViews();
    impl_->submittedCommandBuffers.clear();
    impl_->submittedTimelineSemaphore.reset();
    impl_->submittedTimelineValue = 0;

    for (Impl::QueueCommandContext& queueContext : impl_->queueCommandContexts) {
        queueContext.resetForCurrentSubmit = false;
    }

    impl_->lastExecutionStats = RenderGraphExecutionStats{.executionId = frameIndex};
    const auto cpuBegin = std::chrono::steady_clock::now();
    std::vector<Impl::SubmissionSegment> segments;
    CommandBuffer* currentCommandBuffer = nullptr;
    QueueType currentQueueType = QueueType::Graphics;
    bool hasCurrentSegment = false;
    bool preGraphRecorded = false;

    auto endCurrentSegment = [&]() -> Result {
        if (currentCommandBuffer == nullptr) {
            return {};
        }
        Result endResult = currentCommandBuffer->end();
        if (!endResult) {
            return endResult;
        }
        currentCommandBuffer = nullptr;
        hasCurrentSegment = false;
        return {};
    };

    auto beginSegment = [&](QueueType queueType, Queue& queue) -> Result {
        CommandPool* commandPool = nullptr;
        Result prepareResult = impl_->prepareCommandPool(queueType, queue, commandPool);
        if (!prepareResult) {
            return prepareResult;
        }
        if (commandPool == nullptr) {
            return makeError(Error::Failure);
        }

        std::unique_ptr<CommandBuffer> commandBuffer;
        Result createResult = commandPool->createCommandBuffer(commandBuffer);
        if (!createResult || commandBuffer == nullptr) {
            return createResult ? makeError(Error::Failure) : createResult;
        }
        Result beginResult = commandBuffer->begin();
        if (!beginResult) {
            return beginResult;
        }

        currentCommandBuffer = commandBuffer.get();
        currentQueueType = queueType;
        hasCurrentSegment = true;
        segments.push_back(Impl::SubmissionSegment{
            .queueType = queueType,
            .queue = &queue,
            .commandBuffer = currentCommandBuffer,
        });
        impl_->submittedCommandBuffers.push_back(std::move(commandBuffer));
        return {};
    };

    for (Impl::CompiledNode& node : impl_->executionList) {
        Queue* queue = queueForSubmitDesc(desc, node.queueType);
        if (queue == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!hasCurrentSegment || node.queueType != currentQueueType) {
            result = endCurrentSegment();
            if (!result) {
                return result;
            }
            result = beginSegment(node.queueType, *queue);
            if (!result) {
                return result;
            }
        }

        if (!preGraphRecorded) {
            result = impl_->subsystemHost->recordPreGraph(
                *currentCommandBuffer,
                upload != nullptr ? upload->streamer() : nullptr,
                requiredSubsystems,
                subsystemLog);
            if (!result) {
                spdlog::error("[RenderGraph] {}", subsystemLog);
                std::string cleanupLog;
                (void)impl_->subsystemHost->recordPostGraph(
                    *currentCommandBuffer,
                    upload != nullptr ? upload->streamer() : nullptr,
                    requiredSubsystems,
                    cleanupLog);
                return result;
            }
            if (upload != nullptr) {
                upload->flush(*currentCommandBuffer);
            }
            preGraphRecorded = true;
        }

        result = impl_->executeNode(*currentCommandBuffer, node);
        if (!result) {
            std::string cleanupLog;
            (void)impl_->subsystemHost->recordPostGraph(
                *currentCommandBuffer,
                upload != nullptr ? upload->streamer() : nullptr,
                requiredSubsystems,
                cleanupLog);
            const auto cpuEnd = std::chrono::steady_clock::now();
            impl_->lastExecutionStats.cpuMilliseconds =
                std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();
            return result;
        }
    }

    if (currentCommandBuffer != nullptr) {
        result = impl_->subsystemHost->recordPostGraph(
            *currentCommandBuffer,
            upload != nullptr ? upload->streamer() : nullptr,
            requiredSubsystems,
            subsystemLog);
        if (!result) {
            spdlog::error("[RenderGraph] {}", subsystemLog);
            return result;
        }
    }

    result = endCurrentSegment();
    if (!result) {
        const auto cpuEnd = std::chrono::steady_clock::now();
        impl_->lastExecutionStats.cpuMilliseconds =
            std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();
        return result;
    }

    if (segments.empty()) {
        return makeError(Error::InvalidArgument);
    }

    result = impl_->device->createSemaphore(impl_->submittedTimelineSemaphore);
    if (!result || impl_->submittedTimelineSemaphore == nullptr) {
        return result ? makeError(Error::Failure) : result;
    }

    for (size_t index = 0; index < segments.size(); ++index) {
        Impl::SubmissionSegment& segment = segments[index];
        CommandBuffer* commandBuffers[] = {segment.commandBuffer};

        SemaphoreSubmitDesc waitSemaphore{};
        const bool waitsOnPrevious = index > 0;
        if (waitsOnPrevious) {
            waitSemaphore = SemaphoreSubmitDesc{
                .semaphore = impl_->submittedTimelineSemaphore.get(),
                .value = static_cast<uint64_t>(index),
                .stages = PipelineStageBits::AllCommands,
            };
        }

        SemaphoreSubmitDesc signalSemaphore{
            .semaphore = impl_->submittedTimelineSemaphore.get(),
            .value = static_cast<uint64_t>(index + 1),
            .stages = PipelineStageBits::AllCommands,
        };

        result = segment.queue->submit(QueueSubmitDesc{
            .waitSemaphores = waitsOnPrevious ? &waitSemaphore : nullptr,
            .waitSemaphoreCount = waitsOnPrevious ? 1u : 0u,
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSemaphores = &signalSemaphore,
            .signalSemaphoreCount = 1,
        });
        if (!result) {
            impl_->hasSubmittedWork = index > 0;
            impl_->submittedTimelineValue = static_cast<uint64_t>(index);
            if (!impl_->hasSubmittedWork) {
                impl_->submittedCommandBuffers.clear();
                impl_->submittedTimelineSemaphore.reset();
                impl_->submittedTimelineValue = 0;
            }
            const auto cpuEnd = std::chrono::steady_clock::now();
            impl_->lastExecutionStats.cpuMilliseconds =
                std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();
            return result;
        }
        impl_->hasSubmittedWork = true;
        impl_->submittedTimelineValue = static_cast<uint64_t>(index + 1);
    }

    const auto cpuEnd = std::chrono::steady_clock::now();
    impl_->lastExecutionStats.cpuMilliseconds =
        std::chrono::duration<double, std::milli>(cpuEnd - cpuBegin).count();
    return {};
}

Result RenderGraphExecutor::waitForSubmittedWork(uint64_t timeoutNanoseconds)
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
            compiledNode.effectiveProperties = mergeRenderGraphProperties(
                compiledNode.staticProperties,
                compiledNode.runtimeProperties);
            compiledNode.pass->setProperties(compiledNode.effectiveProperties);
            synced = true;
        }
    }
    return synced;
}

Result RenderGraphExecutor::transitionOutput(
    CommandBuffer& commandBuffer,
    std::string_view fullName,
    ResourceState state)
{
    RenderGraphResource* resource = outputResource(fullName);
    if (resource == nullptr) {
        return makeError(Error::InvalidArgument);
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

Result RenderGraphExecutor::collectCompletedGpuExecutionStats(
    std::vector<RenderGraphExecutionStats>& outStats)
{
    outStats.clear();
    Result result = impl_->resolveGpuTimings();
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
    const RenderUploadSubsystem* upload = impl_->subsystemHost != nullptr
        ? impl_->subsystemHost->get<RenderUploadSubsystem>()
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

    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
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
    std::string lastLog;

    Result ensureReadback(uint32_t newWidth, uint32_t newHeight, uint32_t texelByteSize)
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
        Result result = device->createBuffer(
            BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            },
            readbackBuffer);
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

void RenderGraphPreviewRenderer::setEnvironment(EnvironmentSettings environment)
{
    impl_->world.setEnvironment(std::move(environment));
}

RenderSubsystemHost* RenderGraphPreviewRenderer::subsystemHost()
{
    return &impl_->subsystemHost;
}

const RenderSubsystemHost* RenderGraphPreviewRenderer::subsystemHost() const
{
    return &impl_->subsystemHost;
}

Result RenderGraphPreviewRenderer::initialize(bool enableValidation, bool enableRayQuery)
{
    Result result = createDevice(
        DeviceDesc{
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
            .enableAftermath = true,
        },
        impl_->device);
    if (!result) {
        return result;
    }

    impl_->graphicsQueue = impl_->device->getQueue(QueueType::Graphics);
    if (impl_->graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = impl_->device->createCommandPool(*impl_->graphicsQueue, impl_->commandPool);
    if (!result) {
        return result;
    }
    result = impl_->commandPool->createCommandBuffer(impl_->commandBuffer);
    if (!result) {
        return result;
    }
    result = impl_->historyResources.initialize(*impl_->device);
    if (!result) {
        return result;
    }
    return impl_->device->createFence(true, impl_->fence);
}

Result RenderGraphPreviewRenderer::render(RenderGraph& graph, uint32_t newWidth, uint32_t newHeight)
{
    return render(graph, newWidth, newHeight, graph.firstOutputName());
}

Result RenderGraphPreviewRenderer::render(
    RenderGraph& graph,
    uint32_t newWidth,
    uint32_t newHeight,
    std::string_view outputName)
{
    if (impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->commandPool == nullptr ||
        impl_->commandBuffer == nullptr ||
        impl_->fence == nullptr ||
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

    Result result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    const bool outputCompiled = impl_->executor.compiled() &&
        impl_->executor.outputResource(resolvedOutputName) != nullptr;
    const bool needsCompile =
        graph.dirty() ||
        !impl_->executor.compiled() ||
        impl_->executor.width() != newWidth ||
        impl_->executor.height() != newHeight ||
        !outputCompiled;
    if (needsCompile) {
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
        impl_->executor.syncRuntimeProperties(graph);
    }

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

    result = impl_->ensureReadback(outputWidth, outputHeight, outputTexelByteSize);
    if (!result) {
        return result;
    }

    result = impl_->fence->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandPool->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandBuffer->begin();
    if (!result) {
        return result;
    }

    impl_->historyResources.beginFrame(impl_->historyFrameIndex++);
    result = impl_->executor.execute(*impl_->commandBuffer, &impl_->historyResources);
    if (!result) {
        return result;
    }

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
    result = impl_->graphicsQueue->submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = impl_->fence.get(),
    });
    if (!result) {
        return result;
    }
    result = impl_->fence->wait();
    if (!result) {
        return result;
    }

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
