#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderGraph/RenderGraphInternal.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/Profiling/NsightEvents.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metallic::render {

using namespace detail;
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
        RenderGraphProperties properties = RenderGraphProperties::object();
        std::unique_ptr<RenderGraphPass> pass;
        RenderPassReflection reflection;
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

    Device* device = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    Format defaultFormat = Format::Rgba8Unorm;
    HistoryResourceManager* historyResources = nullptr;
    std::vector<CompiledNode> executionList;
    std::unordered_map<std::string, ResourceSlot> resources;
    std::unordered_map<std::string, std::string> inputAliases;
    std::unique_ptr<BindlessHeap> bindlessHeap;
    std::array<QueueCommandContext, 3> queueCommandContexts;
    std::vector<std::unique_ptr<CommandBuffer>> submittedCommandBuffers;
    std::vector<std::unique_ptr<Semaphore>> submittedSemaphores;
    std::vector<std::unique_ptr<Fence>> submittedFences;
    bool hasSubmittedWork = false;
    bool isCompiled = false;

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

        for (const std::unique_ptr<Fence>& fence : submittedFences) {
            if (fence == nullptr) {
                continue;
            }
            Result result = fence->wait(timeoutNanoseconds);
            if (!result) {
                return result;
            }
        }

        hasSubmittedWork = false;
        submittedCommandBuffers.clear();
        submittedSemaphores.clear();
        submittedFences.clear();
        return {};
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

        RenderGraphExecutionContext context(
            commandBuffer,
            width,
            height,
            node.name,
            node.properties,
            std::move(bindings),
            historyResources);
        const std::string markerName = passProfileMarkerName(node.name, node.type);
        const uint32_t markerColor = profiling::nsightColorFromName(node.type);
        const profiling::NsightProfileRange passMarker(
            markerName.c_str(),
            markerColor,
            node.id);
        commandBuffer.beginDebugLabel(DebugLabelDesc{
            .name = markerName.c_str(),
            .color = debugLabelColorFromArgb(markerColor),
        });
        Result result = node.pass->execute(context);
        commandBuffer.endDebugLabel();
        return result;
    }
};

RenderGraphExecutor::RenderGraphExecutor()
    : impl_(std::make_unique<Impl>())
{
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
    if (!buildActiveGraph(graph, activeGraph, log)) {
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    Result pendingResult = impl_->waitForSubmittedWork(UINT64_MAX);
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
    }

    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;
    impl_->executionList.clear();
    impl_->resources.clear();
    impl_->inputAliases.clear();
    impl_->bindlessHeap.reset();
    impl_->isCompiled = false;

    const RenderGraphCompileContext compileContext{
        .device = &device,
        .graphicsQueue = device.getQueue(QueueType::Graphics),
        .width = width,
        .height = height,
        .defaultFormat = impl_->defaultFormat,
    };

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
        pass->setProperties(node->properties);
        const RenderGraphPassKind kind = pass->kind();
        const QueueType queueType = pass->queueType();
        RenderPassReflection reflection = pass->reflect(compileContext);
        impl_->executionList.push_back(Impl::CompiledNode{
            .id = node->id,
            .name = node->name,
            .type = node->type,
            .kind = kind,
            .queueType = queueType,
            .properties = node->properties,
            .pass = std::move(pass),
            .reflection = std::move(reflection),
        });
    }

    for (const RenderGraphEdge& edge : graph.edges()) {
        if (!activeGraph.activePasses.contains(edge.srcPass) ||
            !activeGraph.activePasses.contains(edge.dstPass)) {
            continue;
        }
        impl_->inputAliases.emplace(
            makeRenderGraphFieldName(edge.dstPass, edge.dstField),
            makeRenderGraphFieldName(edge.srcPass, edge.srcField));
    }

    std::vector<std::string> bindlessSampledImageResources;
    std::vector<std::string> bindlessBufferResources;
    std::unordered_set<std::string> bindlessSampledImageResourceSet;
    std::unordered_set<std::string> bindlessBufferResourceSet;
    for (const Impl::CompiledNode& node : impl_->executionList) {
        for (const RenderGraphField& field : node.reflection.fields()) {
            if (!isBindlessField(field)) {
                continue;
            }

            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            std::string resourceName = fullName;
            if (field.visibility == RenderGraphFieldVisibility::Input) {
                const auto alias = impl_->inputAliases.find(fullName);
                if (alias == impl_->inputAliases.end()) {
                    continue;
                }
                resourceName = alias->second;
            }

            if (isBindlessSampledImageField(field) &&
                bindlessSampledImageResourceSet.insert(resourceName).second) {
                bindlessSampledImageResources.push_back(std::move(resourceName));
                continue;
            }
            if (isBindlessBufferField(field) &&
                bindlessBufferResourceSet.insert(resourceName).second) {
                bindlessBufferResources.push_back(std::move(resourceName));
            }
        }
    }

    if ((!bindlessSampledImageResources.empty() || !bindlessBufferResources.empty()) &&
        !device.capabilities().bindlessDescriptorHeap) {
        log = "RenderGraph compile failed: bindless resources require "
            "DeviceCapabilities::bindlessDescriptorHeap";
        return makeError(Error::Unsupported);
    }

    for (Impl::CompiledNode& node : impl_->executionList) {
        Result result = node.pass->compile(compileContext, log);
        if (!result) {
            impl_->isCompiled = false;
            return result;
        }
    }

    for (const Impl::CompiledNode& node : impl_->executionList) {
        for (const RenderGraphField& field : node.reflection.fields()) {
            if (field.visibility != RenderGraphFieldVisibility::Output) {
                continue;
            }

            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            Impl::ResourceSlot slot;

            if (field.resourceType == RenderGraphResourceType::Texture2D) {
                TextureUsageBits usage = textureUsageForField(field);
                if (usage == TextureUsageBits::None) {
                    usage = TextureUsageBits::ColorAttachment;
                }
                if (isOutputMarked(graph, fullName)) {
                    usage = addTextureUsage(usage, TextureUsageBits::TransferSource);
                    usage = addTextureUsage(usage, TextureUsageBits::Sampled);
                }
                for (const RenderGraphEdge& edge : graph.edges()) {
                    if (edge.srcPass != node.name ||
                        edge.srcField != field.name ||
                        !activeGraph.activePasses.contains(edge.dstPass)) {
                        continue;
                    }

                    const RenderGraphField* dstField = impl_->reflectedField(
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
                    .format = resolveFormat(field.format, impl_->defaultFormat),
                    .width = field.width == 0 ? width : field.width,
                    .height = field.height == 0 ? height : field.height,
                    .depth = 1,
                    .mipCount = 1,
                    .layerCount = 1,
                    .memoryLocation = MemoryLocation::Device,
                };

                Result result = device.createTexture(desc, slot.texture);
                if (!result || slot.texture == nullptr) {
                    log += resultMessage(std::string("createTexture(") + fullName + ")", result);
                    log += '\n';
                    return result ? makeError(Error::Failure) : result;
                }
                result = device.createTextureView(
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

                    const RenderGraphField* dstField = impl_->reflectedField(
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

                Result result = device.createBuffer(desc, slot.buffer);
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
                const bool needsBindlessBuffer = bindlessBufferResourceSet.contains(fullName);
                if (needsBindlessBuffer) {
                    result = device.createBufferView(*slot.buffer, viewDesc, slot.bufferView);
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

            impl_->resources.emplace(fullName, std::move(slot));
        }
    }

    if (!bindlessSampledImageResources.empty() || !bindlessBufferResources.empty()) {
        Result result = device.createBindlessHeap(
            BindlessHeapDesc{
                .maxSampledImages = static_cast<uint32_t>(bindlessSampledImageResources.size()),
                .maxBuffers = static_cast<uint32_t>(bindlessBufferResources.size()),
            },
            impl_->bindlessHeap);
        if (!result || impl_->bindlessHeap == nullptr) {
            log += resultMessage("createBindlessHeap(RenderGraph)", result);
            log += '\n';
            return result ? makeError(Error::Failure) : result;
        }

        for (const std::string& fullName : bindlessSampledImageResources) {
            RenderGraphResource* resource = impl_->resource(fullName);
            if (resource == nullptr || resource->view == nullptr) {
                log = validationPrefix(std::string("bindless sampled image resource is missing '") + fullName + "'");
                return makeError(Error::InvalidArgument);
            }

            BindlessHandle handle;
            result = impl_->bindlessHeap->allocateSampledImage(handle);
            if (!result) {
                log += resultMessage(std::string("allocateSampledImage(") + fullName + ")", result);
                log += '\n';
                return result;
            }

            result = impl_->bindlessHeap->writeSampledImage(
                handle,
                *resource->view,
                ResourceState::ShaderRead);
            if (!result) {
                log += resultMessage(std::string("writeSampledImage(") + fullName + ")", result);
                log += '\n';
                return result;
            }
            resource->bindlessHandle = handle;
            resource->sampledImageBindlessHandle = handle;
        }

        for (const std::string& fullName : bindlessBufferResources) {
            RenderGraphResource* resource = impl_->resource(fullName);
            if (resource == nullptr || resource->bufferView == nullptr) {
                log = validationPrefix(std::string("bindless buffer resource is missing '") + fullName + "'");
                return makeError(Error::InvalidArgument);
            }

            BindlessHandle handle;
            result = impl_->bindlessHeap->allocateBuffer(handle);
            if (!result) {
                log += resultMessage(std::string("allocateBuffer(") + fullName + ")", result);
                log += '\n';
                return result;
            }

            result = impl_->bindlessHeap->writeBufferView(handle, *resource->bufferView);
            if (!result) {
                log += resultMessage(std::string("writeBufferView(") + fullName + ")", result);
                log += '\n';
                return result;
            }
            resource->bindlessHandle = handle;
        }
    }

    impl_->isCompiled = true;
    log = "RenderGraph compiled";
    return {};
}

Result RenderGraphExecutor::execute(CommandBuffer& commandBuffer, HistoryResourceManager* historyResources)
{
    if (!impl_->isCompiled) {
        return makeError(Error::InvalidArgument);
    }

    impl_->historyResources = historyResources;
    for (Impl::CompiledNode& node : impl_->executionList) {
        Result result = impl_->executeNode(commandBuffer, node);
        if (!result) {
            impl_->historyResources = nullptr;
            return result;
        }
    }

    impl_->historyResources = nullptr;
    return {};
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
    impl_->submittedCommandBuffers.clear();
    impl_->submittedSemaphores.clear();
    impl_->submittedFences.clear();

    std::string crossQueueLog;
    if (impl_->hasCrossQueueResourceEdges(crossQueueLog)) {
        return makeError(Error::Unsupported);
    }

    for (Impl::QueueCommandContext& queueContext : impl_->queueCommandContexts) {
        queueContext.resetForCurrentSubmit = false;
    }

    std::vector<Impl::SubmissionSegment> segments;
    CommandBuffer* currentCommandBuffer = nullptr;
    QueueType currentQueueType = QueueType::Graphics;
    bool hasCurrentSegment = false;

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

        result = impl_->executeNode(*currentCommandBuffer, node);
        if (!result) {
            return result;
        }
    }

    result = endCurrentSegment();
    if (!result) {
        return result;
    }

    if (segments.empty()) {
        return makeError(Error::InvalidArgument);
    }

    if (segments.size() > 1) {
        impl_->submittedSemaphores.reserve(segments.size() - 1);
        for (size_t index = 0; index + 1 < segments.size(); ++index) {
            std::unique_ptr<Semaphore> semaphore;
            result = impl_->device->createSemaphore(semaphore);
            if (!result || semaphore == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
            impl_->submittedSemaphores.push_back(std::move(semaphore));
        }
    }

    impl_->submittedFences.reserve(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
        std::unique_ptr<Fence> fence;
        result = impl_->device->createFence(false, fence);
        if (!result || fence == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }
        impl_->submittedFences.push_back(std::move(fence));
    }

    for (size_t index = 0; index < segments.size(); ++index) {
        Impl::SubmissionSegment& segment = segments[index];
        CommandBuffer* commandBuffers[] = {segment.commandBuffer};

        SemaphoreSubmitDesc waitSemaphore{};
        const bool waitsOnPrevious = index > 0;
        if (waitsOnPrevious) {
            waitSemaphore = SemaphoreSubmitDesc{
                .semaphore = impl_->submittedSemaphores[index - 1].get(),
                .stages = PipelineStageBits::AllCommands,
            };
        }

        SemaphoreSubmitDesc signalSemaphore{};
        const bool signalsNext = index + 1 < segments.size();
        if (signalsNext) {
            signalSemaphore = SemaphoreSubmitDesc{
                .semaphore = impl_->submittedSemaphores[index].get(),
                .stages = PipelineStageBits::AllCommands,
            };
        }

        result = segment.queue->submit(QueueSubmitDesc{
            .waitSemaphores = waitsOnPrevious ? &waitSemaphore : nullptr,
            .waitSemaphoreCount = waitsOnPrevious ? 1u : 0u,
            .commandBuffers = commandBuffers,
            .commandBufferCount = 1,
            .signalSemaphores = signalsNext ? &signalSemaphore : nullptr,
            .signalSemaphoreCount = signalsNext ? 1u : 0u,
            .signalFence = impl_->submittedFences[index].get(),
        });
        if (!result) {
            impl_->submittedFences.resize(index);
            impl_->hasSubmittedWork = index > 0;
            if (!impl_->hasSubmittedWork) {
                impl_->submittedCommandBuffers.clear();
                impl_->submittedSemaphores.clear();
                impl_->submittedFences.clear();
            }
            return result;
        }
        impl_->hasSubmittedWork = true;
    }

    return {};
}

Result RenderGraphExecutor::waitForSubmittedWork(uint64_t timeoutNanoseconds)
{
    return impl_->waitForSubmittedWork(timeoutNanoseconds);
}

bool RenderGraphExecutor::syncProperties(const RenderGraph& graph)
{
    if (!impl_->isCompiled) {
        return false;
    }

    bool synced = false;
    for (Impl::CompiledNode& compiledNode : impl_->executionList) {
        const RenderGraphNode* graphNode = graph.findNode(compiledNode.id);
        if (graphNode == nullptr ||
            graphNode->name != compiledNode.name ||
            graphNode->type != compiledNode.type) {
            return false;
        }

        if (compiledNode.properties != graphNode->properties) {
            compiledNode.properties = graphNode->properties;
            compiledNode.pass->setProperties(compiledNode.properties);
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
    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    std::unique_ptr<Buffer> readbackBuffer;
    RenderGraphExecutor executor;
    HistoryResourceManager historyResources;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readbackWidth = 0;
    uint32_t readbackHeight = 0;
    uint64_t historyFrameIndex = 0;
    std::string lastLog;

    Result ensureReadback(uint32_t newWidth, uint32_t newHeight)
    {
        if (device == nullptr || newWidth == 0 || newHeight == 0) {
            return makeError(Error::InvalidArgument);
        }
        if (readbackBuffer != nullptr && readbackWidth == newWidth && readbackHeight == newHeight) {
            return {};
        }
        readbackBuffer.reset();
        const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
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

Result RenderGraphPreviewRenderer::initialize(bool enableValidation, bool enableRayQuery)
{
    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic RenderGraph Preview",
            .enableValidation = enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableRayTracingAccelerationStructure = enableRayQuery,
            .enableRayQuery = enableRayQuery,
            .enablePushDescriptor = enableRayQuery,
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
    if (impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->commandPool == nullptr ||
        impl_->commandBuffer == nullptr ||
        impl_->fence == nullptr ||
        newWidth == 0 ||
        newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    Result result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    const bool needsCompile =
        graph.dirty() ||
        !impl_->executor.compiled() ||
        impl_->executor.width() != newWidth ||
        impl_->executor.height() != newHeight;
    if (needsCompile) {
        result = impl_->device->waitIdle();
        if (!result) {
            return result;
        }
        impl_->historyResources.invalidateAll();
        impl_->historyFrameIndex = 0;
        result = impl_->executor.compile(
            *impl_->device,
            graph,
            newWidth,
            newHeight,
            impl_->lastLog);
        if (!result) {
            return result;
        }
        graph.clearDirty();
    } else {
        impl_->executor.syncProperties(graph);
    }

    result = impl_->ensureReadback(newWidth, newHeight);
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

    const std::string outputName = graph.firstOutputName();
    RenderGraphResource* output = impl_->executor.outputResource(outputName);
    if (output == nullptr || impl_->readbackBuffer == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output resource is missing '") + outputName + "'";
        return makeError(Error::InvalidArgument);
    }
    if (output->type != RenderGraphResourceType::Texture2D || output->texture == nullptr) {
        impl_->lastLog = std::string("RenderGraph preview output is not a Texture2D '") + outputName + "'";
        return makeError(Error::InvalidArgument);
    }
    result = impl_->executor.transitionOutput(
        *impl_->commandBuffer,
        outputName,
        ResourceState::TransferSource);
    if (!result) {
        return result;
    }
    impl_->commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
        .texture = output->texture,
        .buffer = impl_->readbackBuffer.get(),
        .width = newWidth,
        .height = newHeight,
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
    const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
    std::memcpy(impl_->pixels.data(), mapped, static_cast<size_t>(byteSize));
    impl_->readbackBuffer->unmap();

    impl_->width = newWidth;
    impl_->height = newHeight;
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
