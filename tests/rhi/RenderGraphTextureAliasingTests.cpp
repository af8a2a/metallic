#include "Runtime/Render/Core/ResourceState.h"
#include "RHITest.h"
#include "RenderGraphViewerTestUI.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <array>
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>

namespace metallic::tests {
namespace {

using namespace render;

constexpr uint32_t kAliasFirstExtent = 32;
constexpr uint32_t kAliasSecondExtent = 16;
constexpr uint64_t kAliasFirstBytes = uint64_t(kAliasFirstExtent) * kAliasFirstExtent * 4;
constexpr std::string_view kAliasClearType = "TestTextureAliasClear";
constexpr std::string_view kAliasReadType = "TestTextureAliasRead";

class AliasClearPass final : public RasterPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const uint32_t extent = properties().value("extent", kAliasFirstExtent);
        const uint32_t width = properties().value("width", extent);
        const uint32_t height = properties().value("height", extent);
        RenderPassReflection reflection;
        reflection.addBufferInput("token").buffer(kAliasFirstBytes).transferRead().setOptional();
        auto& source = reflection.addTextureInput("source").texture2D(kAliasFirstExtent, kAliasFirstExtent)
            .transferRead().setOptional();
        source.matchOutputExtent = false;
        auto& output = reflection.addTextureOutput("color").texture2D(width, height).colorWrite();
        if (properties().value("transient", true)) {
            const auto initialization = properties().value("unknown", false)
                ? RenderGraphInitialization::Unknown : RenderGraphInitialization::Clear;
            output.transient(initialization);
        }
        output.presentationOutput = properties().value("presentation", false);
        return reflection;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto output = context.outputTexture("color");
        if (!output.valid()) { return makeError(Error::InvalidArgument); }
        const uint32_t phase = properties().value("phase", 0u);
        const bool red = ((context.frameIndex() + phase) % 2) == 0;
        const RenderingAttachmentDesc attachment{.view = output.view(), .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = red ? ColorValue{1, 0, 0, 1} : ColorValue{0, 1, 0, 1}};
        auto result = context.commandBuffer().beginRendering({
            .renderArea = {.width = output.desc().width, .height = output.desc().height},
            .colorAttachments = {&attachment, 1}});
        if (result) { context.commandBuffer().endRendering(); }
        return result;
    }
};

class AliasReadPass final : public UnsafePass {
public:
    QueueType queueType() const override
    {
        const std::string queue = properties().value("queue", std::string("graphics"));
        return queue == "copy" ? QueueType::Copy : queue == "compute" ? QueueType::Compute : QueueType::Graphics;
    }

    bool supportsAsyncQueue() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const uint32_t extent = properties().value("extent", kAliasFirstExtent);
        const uint32_t width = properties().value("width", extent);
        const uint32_t height = properties().value("height", extent);
        RenderPassReflection reflection;
        reflection.addTextureInput("color").texture2D(width, height).transferRead();
        reflection.addBufferOutput("data").buffer(uint64_t(width) * height * 4).transferWrite().hostReadback();
        return reflection;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto input = context.inputTexture("color");
        auto output = context.outputBuffer("data");
        if (!input.valid() || !output.valid()) { return makeError(Error::InvalidArgument); }
        context.commandBuffer().copyTextureToBuffer({.texture = input.texture(), .buffer = output.buffer(),
            .width = input.desc().width, .height = input.desc().height});
        const BufferBarrierDesc host{.buffer = output.buffer(),
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead}};
        return context.commandBuffer().synchronize({.buffers = {&host, 1}});
    }
};

void registerTextureAliasPasses()
{
    registerRenderGraphPassType(std::string(kAliasClearType), "Shaderless transient clear",
        [] { return std::make_unique<AliasClearPass>(); });
    registerRenderGraphPassType(std::string(kAliasReadType), "Shaderless transient readback",
        [] { return std::make_unique<AliasReadPass>(); });
}

RenderGraph aliasGraph(bool dependency = true, bool overlappingUse = false, std::string_view queue = "graphics")
{
    RenderGraph graph;
    graph.addNode(std::string(kAliasClearType), "A", {{"extent", kAliasFirstExtent}, {"phase", 0}});
    graph.addNode(std::string(kAliasReadType), "ReadA", {{"extent", kAliasFirstExtent}, {"queue", std::string(queue)}});
    graph.addNode(std::string(kAliasClearType), "B", {{"extent", kAliasSecondExtent}, {"phase", 1}});
    graph.addNode(std::string(kAliasReadType), "ReadB", {{"extent", kAliasSecondExtent}, {"queue", std::string(queue)}});
    graph.addEdge("A.color", "ReadA.color");
    if (dependency) { graph.addEdge("ReadA.data", "B.token"); }
    if (overlappingUse) { graph.addEdge("A.color", "B.source"); }
    graph.addEdge("B.color", "ReadB.color");
    graph.markOutput("ReadA.data");
    graph.markOutput("ReadB.data");
    return graph;
}

RenderGraph aliasGraphWithDimensions(uint32_t width, uint32_t height)
{
    auto graph = aliasGraph();
    for (const auto name : {"A", "ReadA", "B", "ReadB"}) {
        auto* node = graph.findNode(name);
        auto properties = node->properties;
        const bool second = std::string_view(name) == "B" || std::string_view(name) == "ReadB";
        properties["width"] = second ? std::max(1u, width / 2) : width;
        properties["height"] = second ? std::max(1u, height / 2) : height;
        graph.setNodeProperties(node->id, std::move(properties));
    }
    return graph;
}

bench::Json textureMemoryEvidence(const auto& stats)
{
    auto slots = bench::Json::array();
    for (const auto& slot : stats.slots) {
        slots.push_back({{"backingAllocationId", slot.backingAllocationId}, {"complete", slot.complete},
            {"logicalBytes", slot.logicalBytes}, {"backingBytes", slot.backingBytes},
            {"savedBytes", slot.savedBytes}, {"overheadBytes", slot.overheadBytes},
            {"resources", slot.resources}});
    }
    return {{"aliasingEnabled", stats.aliasingEnabled}, {"complete", stats.complete},
        {"textureCount", stats.textureCount}, {"transientTextureCount", stats.transientTextureCount},
        {"pinnedTextureCount", stats.pinnedTextureCount}, {"eligibleTextureCount", stats.eligibleTextureCount},
        {"aliasedTextureCount", stats.aliasedTextureCount}, {"aliasSlotCount", stats.aliasSlotCount},
        {"backingAllocationCount", stats.backingAllocationCount}, {"unknownTextureCount", stats.unknownTextureCount},
        {"logicalBytes", stats.logicalBytes}, {"backingBytes", stats.backingBytes},
        {"savedBytes", stats.savedBytes}, {"overheadBytes", stats.overheadBytes}, {"slots", std::move(slots)}};
}

std::string validateTextureMemoryStatistics(Device& device, RenderGraphExecutor& executor, bool aliases)
{
    const auto& stats = executor.textureMemoryStats();
    if (!stats.complete || stats.unknownTextureCount || stats.aliasingEnabled != aliases ||
        stats.textureCount != 2 || stats.transientTextureCount != 2 || stats.pinnedTextureCount) {
        return "Texture statistics include buffers, miss a texture, or have incorrect policy/completeness counts";
    }
    uint64_t logicalBytes = 0;
    uint64_t backingBytes = 0;
    uint64_t backingId = 0;
    for (const auto name : {"A.color", "B.color"}) {
        const auto* resource = executor.outputResource(name);
        if (!resource || !resource->texture) { return "Texture statistics fixture omitted an image"; }
        const auto allocation = resource->texture->memoryInfo();
        if (!allocation.known || !allocation.backingAllocationId || !allocation.backingSizeBytes) {
            return "Cannot verify statistics against actual native backing metadata";
        }
        if (aliases) {
            const auto standalone = device.textureAllocationSize(resource->texture->desc());
            if (!standalone) { return "Cannot query the standalone texture requirement for the savings oracle"; }
            logicalBytes += *standalone;
            if (backingId && allocation.backingAllocationId != backingId) {
                return "The alias statistics fixture did not share actual backing";
            }
            backingId = allocation.backingAllocationId;
            backingBytes = allocation.backingSizeBytes;
        } else {
            logicalBytes += allocation.backingSizeBytes;
            backingBytes += allocation.backingSizeBytes;
        }
    }
    if (stats.logicalBytes != logicalBytes || stats.backingBytes != backingBytes || stats.overheadBytes ||
        stats.savedBytes != logicalBytes - backingBytes || stats.backingAllocationCount != (aliases ? 1u : 2u) ||
        stats.eligibleTextureCount != (aliases ? 2u : 0u) || stats.aliasedTextureCount != (aliases ? 2u : 0u) ||
        stats.aliasSlotCount != (aliases ? 1u : 0u) || stats.slots.size() != (aliases ? 1u : 0u)) {
        return "Texture savings/counts do not agree with independent native allocation requirements and unique backing";
    }
    if (aliases) {
        const auto& slot = stats.slots.front();
        auto names = slot.resources;
        std::sort(names.begin(), names.end());
        if (!stats.savedBytes || !slot.complete || slot.backingAllocationId != backingId || slot.logicalBytes != logicalBytes ||
            slot.backingBytes != backingBytes || slot.savedBytes != stats.savedBytes || slot.overheadBytes ||
            names != std::vector<std::string>{"A.color", "B.color"}) {
            return "Alias slot statistics double-count shared backing or lose their resource membership";
        }
    }
    return {};
}

void saveAliasMemoryEvidence(RHITestContext& context, const std::string& filename, const bench::Json& value)
{
    if (context.evidence) {
        context.evidence->json(filename, value);
        return;
    }
    const auto directory = context.outputDirectory / "texture-alias-memory";
    std::filesystem::create_directories(directory);
    std::ofstream output(directory / filename, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << value.dump(2) << '\n';
}

bool graphTexturesShareBacking(RenderGraphExecutor& executor)
{
    const auto* first = executor.outputResource("A.color");
    const auto* second = executor.outputResource("B.color");
    if (!first || !second || !first->texture || !second->texture) { return false; }
    const auto a = first->texture->memoryInfo();
    const auto b = second->texture->memoryInfo();
    return a.known && b.known && a.backingAllocationId && a.backingAllocationId == b.backingAllocationId &&
        a.memoryBlockId == b.memoryBlockId && a.offsetBytes == b.offsetBytes;
}

bool readGraphAliasPixels(RenderGraphExecutor& executor, std::string_view name,
    std::vector<uint8_t>& pixels)
{
    const auto* resource = executor.outputResource(name);
    if (!resource || !resource->buffer) { return false; }
    resource->buffer->invalidate();
    const void* mapped = resource->buffer->map();
    if (!mapped) { return false; }
    pixels.resize(size_t(resource->buffer->desc().size));
    std::memcpy(pixels.data(), mapped, pixels.size());
    resource->buffer->unmap();
    return true;
}

bool graphAliasPixelsMatch(const std::vector<uint8_t>& pixels, bool red)
{
    const std::array<uint8_t, 4> expected = red ? std::array<uint8_t, 4>{255, 0, 0, 255}
        : std::array<uint8_t, 4>{0, 255, 0, 255};
    if (pixels.empty() || pixels.size() % 4) { return false; }
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        if (std::memcmp(pixels.data() + offset, expected.data(), 4)) { return false; }
    }
    return true;
}

std::string validateAliasCapture(const RenderGraphExecutionSnapshot& snapshot, QueueType readQueue)
{
    if (!snapshot.success || snapshot.externalRecording || snapshot.status != RenderGraphExecutionSnapshotStatus::Submitted) {
        return "Alias execution capture does not describe successful managed submission";
    }
    const auto findPass = [&](std::string_view name) -> const RenderGraphExecutionPassSnapshot* {
        const auto found = std::find_if(snapshot.passes.begin(), snapshot.passes.end(),
            [name](const auto& pass) { return pass.name == name; });
        return found == snapshot.passes.end() ? nullptr : &*found;
    };
    const auto* first = findPass("A");
    const auto* read = findPass("ReadA");
    const auto* second = findPass("B");
    const auto* finalRead = findPass("ReadB");
    if (!first || !read || !second || !finalRead) { return "Alias capture omitted a chain pass"; }
    for (const auto* pass : {first, second}) {
        const std::string resourceName = pass->name + ".color";
        const auto resource = std::find_if(snapshot.resources.begin(), snapshot.resources.end(), [&](const auto& candidate) {
            return candidate.name == resourceName ||
                std::find(candidate.aliases.begin(), candidate.aliases.end(), resourceName) != candidate.aliases.end();
        });
        if (resource == snapshot.resources.end()) { return "Alias capture omitted a transient resource"; }
        const auto activation = std::find_if(pass->barriers.begin(), pass->barriers.end(), [&](const auto& barrier) {
            return barrier.memoryAliasing && barrier.resourceId == resource->id &&
                barrier.beforeScope.stages == PipelineStageBits::AllCommands &&
                barrier.afterScope.stages == PipelineStageBits::AllCommands &&
                barrier.beforeScope.access == (AccessBits::MemoryRead | AccessBits::MemoryWrite) &&
                barrier.afterScope.access == (AccessBits::MemoryRead | AccessBits::MemoryWrite);
        });
        const auto discard = std::find_if(pass->barriers.begin(), pass->barriers.end(), [&](const auto& barrier) {
            return !barrier.memoryAliasing && barrier.resourceId == resource->id &&
                barrier.before == ResourceState::Undefined && barrier.after == ResourceState::ColorAttachment &&
                barrier.beforeScope.stages == PipelineStageBits::AllCommands;
        });
        if (activation == pass->barriers.end() || discard == pass->barriers.end()) {
            return "An alias activation lacked its global memory dependency or ordered discard transition";
        }
    }
    if (std::find(second->predecessors.begin(), second->predecessors.end(), read->id) == second->predecessors.end()) {
        return "Alias activation lost its previous occupant's terminal read predecessor";
    }
    if (read->logicalQueue != readQueue || finalRead->logicalQueue != readQueue) {
        return "Alias readback capture changed the declared queue";
    }
    if (readQueue == QueueType::Graphics) { return {}; }
    if (read->actualQueueId == second->actualQueueId || finalRead->actualQueueId == second->actualQueueId) {
        return "Alias readback did not execute on the available independent queue";
    }
    const auto readSegment = std::find_if(snapshot.segments.begin(), snapshot.segments.end(),
        [&](const auto& segment) { return segment.passId == read->id; });
    const auto activationSegment = std::find_if(snapshot.segments.begin(), snapshot.segments.end(),
        [&](const auto& segment) { return segment.passId == second->id; });
    if (readSegment == snapshot.segments.end() || activationSegment == snapshot.segments.end() ||
        std::find(activationSegment->predecessors.begin(), activationSegment->predecessors.end(), readSegment->id)
            == activationSegment->predecessors.end()) {
        return "Cross-queue alias activation lost its terminal read segment join";
    }
    const auto activationBatch = std::find_if(snapshot.batches.begin(), snapshot.batches.end(), [&](const auto& batch) {
        return std::find(batch.segmentIds.begin(), batch.segmentIds.end(), activationSegment->id) != batch.segmentIds.end();
    });
    if (activationBatch == snapshot.batches.end() || !activationBatch->accepted ||
        std::find(activationBatch->waitPredecessors.begin(), activationBatch->waitPredecessors.end(), readSegment->id)
            == activationBatch->waitPredecessors.end() || !activationBatch->semaphoreWaitCount) {
        return "Cross-queue alias handover submitted without its previous occupant's semaphore wait";
    }
    return {};
}

class RenderGraphTextureAliasingReadbackTest final : public RHITest {
public:
    RenderGraphTextureAliasingReadbackTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_texture_aliasing_frame_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.textureAliasing.frameReadback", "graph.textureAliasing.activationBarrier",
                "graph.textureAliasing.crossQueueJoin"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTextureAliasPasses();
        struct QueueMode {
            const char* name;
            QueueType type;
            Queue* queue;
        };
        std::vector<QueueMode> modes{{"graphics", QueueType::Graphics, &context.graphicsQueue}};
        for (const auto type : {QueueType::Copy, QueueType::Compute}) {
            auto* queue = context.device.getQueue(type);
            if (queue && vulkan::nativeQueue(*queue).queue != vulkan::nativeQueue(context.graphicsQueue).queue) {
                modes.push_back({type == QueueType::Copy ? "copy" : "compute", type, queue});
            }
        }
        for (const auto& mode : modes) {
            auto graph = aliasGraph(true, false, mode.name);
            std::string log;
            RenderGraphExecutor baseline;
            RenderGraphExecutor aliased;
            aliased.setExecutionCaptureEnabled(true);
            if (!baseline.compile(context.device, graph, 32, 32, log)) {
                return RHITestResult::fail("Alias-off graph compilation failed: " + log);
            }
            const RenderGraphCompileOptions options{.enableTextureAliasing = true};
            if (!aliased.compile(context.device, graph, 32, 32, options, log)) {
                return RHITestResult::fail("Alias-on graph compilation failed: " + log);
            }
            if (graphTexturesShareBacking(baseline)) {
                return RHITestResult::fail("Default compilation aliased textures without opt-in");
            }
            if (!graphTexturesShareBacking(aliased)) {
                return RHITestResult::fail("Ordered nonoverlapping transient textures did not share backing");
            }
            if (const auto error = validateTextureMemoryStatistics(context.device, baseline, false); !error.empty()) {
                return RHITestResult::fail(std::string(mode.name) + " alias-off: " + error);
            }
            if (const auto error = validateTextureMemoryStatistics(context.device, aliased, true); !error.empty()) {
                return RHITestResult::fail(std::string(mode.name) + " alias-on: " + error);
            }
            const auto compiledMemory = textureMemoryEvidence(aliased.textureMemoryStats());
            const auto* first = aliased.outputResource("A.color");
            const auto* second = aliased.outputResource("B.color");
            if (first->texture->memoryInfo().allocationId == second->texture->memoryInfo().allocationId ||
                vulkan::nativeTexture(*first->texture).image == vulkan::nativeTexture(*second->texture).image) {
                return RHITestResult::fail("Graph aliasing replaced distinct logical images with one native object");
            }
            if (aliased.isExportedOutput("A.color") || aliased.isExportedOutput("B.color") ||
                !aliased.isExportedOutput("ReadA.data") || !aliased.isExportedOutput("ReadB.data")) {
                return RHITestResult::fail("Graph exports do not distinguish pinned readbacks from alias transients");
            }
            RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue};
            if (mode.type == QueueType::Copy) { submit.copyQueue = mode.queue; }
            if (mode.type == QueueType::Compute) { submit.computeQueue = mode.queue; }
            for (uint32_t frame = 0; frame < 4; ++frame) {
                for (auto* executor : {&baseline, &aliased}) {
                    if (!executor->execute(submit) || !executor->waitForSubmittedWork(5'000'000'000ull)) {
                        (void)context.device.waitIdle();
                        return RHITestResult::fail(std::string(mode.name) + " alias comparison frame did not complete");
                    }
                }
                const auto snapshot = aliased.executionSnapshot();
                if (!snapshot) { return RHITestResult::fail("Alias execution snapshot is missing"); }
                if (textureMemoryEvidence(snapshot->textureMemory) != compiledMemory ||
                    textureMemoryEvidence(aliased.executionStats().textureMemory) != compiledMemory) {
                    return RHITestResult::fail("Execution statistics or capture lost the compiled texture memory totals");
                }
                if (const auto error = validateAliasCapture(*snapshot, mode.type); !error.empty()) {
                    return RHITestResult::fail(std::string(mode.name) + ": " + error);
                }
                for (const auto name : {std::string_view("ReadA.data"), std::string_view("ReadB.data")}) {
                    std::vector<uint8_t> expected, actual;
                    if (!readGraphAliasPixels(baseline, name, expected) || !readGraphAliasPixels(aliased, name, actual)) {
                        return RHITestResult::fail("Cannot read graph alias comparison pixels");
                    }
                    const bool red = ((frame + (name == "ReadB.data" ? 1u : 0u)) % 2) == 0;
                    if (actual != expected || !graphAliasPixelsMatch(actual, red)) {
                        return RHITestResult::fail("Alias-on output differed from alias-off or expected full-image clear");
                    }
                }
            }
        }
        return RHITestResult::pass("Alias on/off match every pixel over four frames in " + std::to_string(modes.size()) +
            " available queue modes; activation barriers and cross-queue joins are captured");
    }
};

class RenderGraphTextureAliasingExternalGuardTest final : public RHITest {
public:
    RenderGraphTextureAliasingExternalGuardTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_texture_aliasing_external_recording_guards";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.textureAliasing.externalRecordingGuard", "graph.textureAliasing.cancelRebuild"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTextureAliasPasses();
        const auto graph = aliasGraph();
        const RenderGraphCompileOptions options{.enableTextureAliasing = true};
        RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        if (!executor.compile(context.device, graph, 32, 32, options, log) || !graphTexturesShareBacking(executor)) {
            return RHITestResult::fail("Cannot compile external alias guard graph: " + log);
        }
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands, otherCommands;
        RenderFrameContext frame;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) || !commands->begin()) {
            return RHITestResult::fail("Cannot record untracked alias guard probe");
        }
        if (!hasError(executor.execute(*commands), Error::Unsupported) || executor.executionSnapshot()) {
            return RHITestResult::fail("Untracked alias execution was accepted or mutated execution capture");
        }
        if (!hasError(executor.transitionOutput(*commands, "A.color", ResourceState::TransferSource), Error::InvalidArgument)) {
            return RHITestResult::fail("An alias transient was exposed to an external consumer");
        }
        if (!commands->end()) { return RHITestResult::fail("Cannot finish untracked guard probe"); }
        commands.reset();
        if (!pool->reset() || !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !frame.begin(0) || !commands->begin(frame.submissionContext()) || !executor.execute(*commands) || !commands->end()) {
            return RHITestResult::fail("Cannot record tracked unsubmitted alias graph");
        }
        const auto recorded = executor.executionSnapshot();
        const uint64_t frameIndex = executor.streamingStats().frameIndex;
        if (!recorded || !recorded->success || !recorded->externalRecording ||
            recorded->status != RenderGraphExecutionSnapshotStatus::Recorded || frame.completion().isSubmitted()) {
            return RHITestResult::fail("External alias execution did not remain recorded and unsubmitted");
        }
        if (!pool->createCommandBuffer().transform([&](auto value) { otherCommands = std::move(value); }) ||
            !otherCommands->begin(frame.submissionContext())) {
            return RHITestResult::fail("Cannot record second external guard probe");
        }
        if (!hasError(executor.execute(*otherCommands), Error::InvalidArgument) ||
            !hasError(executor.execute({.graphicsQueue = &context.graphicsQueue}), Error::InvalidArgument) ||
            executor.streamingStats().frameIndex != frameIndex || executor.executionSnapshot() != recorded) {
            return RHITestResult::fail("Pending unsubmitted alias recording admitted a second execution or mutated state");
        }
        if (!otherCommands->end()) { return RHITestResult::fail("Cannot finish second external guard probe"); }
        const auto cancelled = frame.completion();
        frame.cancel();
        commands.reset();
        otherCommands.reset();
        if (!pool->reset() || !frame.reset() || !cancelled.isCancelled()) {
            return RHITestResult::fail("Cannot cancel and reset unsubmitted alias recordings");
        }
        if (!hasError(executor.execute({.graphicsQueue = &context.graphicsQueue}), Error::InvalidArgument) || executor.compiled()) {
            return RHITestResult::fail("Cancelled alias recording allowed stale resource states to execute");
        }
        if (!executor.compile(context.device, graph, 32, 32, options, log) ||
            !executor.execute({.graphicsQueue = &context.graphicsQueue}) ||
            !executor.waitForSubmittedWork(5'000'000'000ull)) {
            (void)context.device.waitIdle();
            return RHITestResult::fail("Alias graph could not rebuild and safely retry after cancellation: " + log);
        }
        const auto retry = executor.executionSnapshot();
        if (!retry) { return RHITestResult::fail("Rebuilt alias graph retry has no execution snapshot"); }
        for (const auto name : {std::string_view("ReadA.data"), std::string_view("ReadB.data")}) {
            std::vector<uint8_t> actual;
            const bool red = ((retry->executionId + (name == "ReadB.data" ? 1u : 0u)) % 2) == 0;
            if (!readGraphAliasPixels(executor, name, actual) || !graphAliasPixelsMatch(actual, red)) {
                return RHITestResult::fail("Rebuilt alias graph retry produced stale or incorrect pixels");
            }
        }
        return RHITestResult::pass("Untracked and pending executions are rejected without mutation; cancelled recordings require rebuild before a pixel-correct retry");
    }
};

class RenderGraphTextureAliasingEligibilityTest final : public RHITest {
public:
    RenderGraphTextureAliasingEligibilityTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_texture_aliasing_lifetime_and_exports";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.textureAliasing.lifetimeExclusions", "graph.textureAliasing.exports"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTextureAliasPasses();
        struct Case {
            const char* name;
            bool independent = false;
            bool overlapping = false;
            bool persistent = false;
            bool unknown = false;
            bool marked = false;
            bool extra = false;
            bool preview = false;
            bool presentation = false;
        };
        constexpr Case cases[] = {
            {.name = "Independent branches", .independent = true},
            {.name = "Overlapping pass use", .overlapping = true},
            {.name = "Persistent output", .persistent = true},
            {.name = "Unknown initialization", .unknown = true},
            {.name = "Marked output", .marked = true},
            {.name = "Extra output", .extra = true},
            {.name = "Preview outputs", .preview = true},
            {.name = "Presentation output", .presentation = true},
        };
        for (const auto& test : cases) {
            auto graph = aliasGraph(!test.independent, test.overlapping);
            auto properties = graph.findNode("A")->properties;
            properties["transient"] = !test.persistent;
            properties["unknown"] = test.unknown;
            properties["presentation"] = test.presentation;
            graph.setNodeProperties(graph.findNode("A")->id, std::move(properties));
            if (test.marked) { graph.markOutput("A.color"); }
            RenderGraphCompileOptions options{.enableTextureAliasing = true};
            options.enablePreviewOutputAccess = test.preview;
            if (test.extra || test.preview) { options.extraOutputs.push_back("A.color"); }
            RenderGraphExecutor executor;
            std::string log;
            if (!executor.compile(context.device, graph, 32, 32, options, log)) {
                return RHITestResult::fail(std::string(test.name) + " compilation failed: " + log);
            }
            if (graphTexturesShareBacking(executor)) {
                return RHITestResult::fail(std::string(test.name) + " incorrectly aliased an ineligible resource");
            }
            if (!executor.isExportedOutput("A.color")) {
                return RHITestResult::fail(std::string(test.name) + " has incorrect external-output eligibility");
            }
            const auto& memory = executor.textureMemoryStats();
            if (!memory.aliasingEnabled || !memory.complete || memory.unknownTextureCount || memory.textureCount != 2 ||
                memory.aliasedTextureCount || memory.aliasSlotCount || memory.backingAllocationCount != 2 ||
                memory.savedBytes || memory.overheadBytes || memory.logicalBytes != memory.backingBytes ||
                !memory.slots.empty()) {
                return RHITestResult::fail(std::string(test.name) + " incorrectly reports savings for excluded textures");
            }
            if ((test.marked || test.extra || test.preview || test.presentation) && !memory.pinnedTextureCount) {
                return RHITestResult::fail(std::string(test.name) + " lost the pinned texture count");
            }
        }
        return RHITestResult::pass("Unordered branches, overlapping use, persistent/unknown outputs, and all export paths exclude aliasing");
    }
};

class RenderGraphTextureAliasingMemoryStatisticsTest final : public RHITest {
public:
    RenderGraphTextureAliasingMemoryStatisticsTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_texture_aliasing_vram_statistics";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.textureAliasing.memoryStatistics", "graph.textureAliasing.actualBudgetSavings",
                "graph.textureAliasing.statisticsRecompile"},
            .artifacts = {"alias-memory.json", "readback.bin", "alias-memory.png"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        uint32_t width = 64;
        uint32_t height = 64;
        for (const auto& [variable, dimension] : {std::pair{"METALLIC_TEST_ALIAS_WIDTH", &width},
                std::pair{"METALLIC_TEST_ALIAS_HEIGHT", &height}}) {
            if (const char* text = std::getenv(variable); text && *text) {
                const auto* end = text + std::strlen(text);
                const auto parsed = std::from_chars(text, end, *dimension);
                if (parsed.ec != std::errc{} || parsed.ptr != end || !*dimension || *dimension > 16384) {
                    return RHITestResult::fail(std::string(variable) + " must be an integer in [1, 16384]");
                }
            }
        }
        registerTextureAliasPasses();
        auto graph = aliasGraphWithDimensions(width, height);
        const RenderGraphCompileOptions options{.enableTextureAliasing = true};
        constexpr size_t domain = size_t(MemoryBudgetDomain::FrameResources);
        const auto initialBudget = context.device.memoryBudget().domains[domain];
        RenderGraphExecutor baseline;
        RenderGraphExecutor aliased;
        baseline.setExecutionCaptureEnabled(true);
        aliased.setExecutionCaptureEnabled(true);
        std::string log;
        if (!baseline.compile(context.device, graph, width, height, log)) {
            return RHITestResult::fail("Statistics alias-off graph compilation failed: " + log);
        }
        const auto baselineBudget = context.device.memoryBudget().domains[domain];
        if (!aliased.compile(context.device, graph, width, height, options, log)) {
            return RHITestResult::fail("Statistics alias-on graph compilation failed: " + log);
        }
        const auto liveBudget = context.device.memoryBudget();
        const auto aliasBudget = liveBudget.domains[domain];
        for (const auto& [executor, aliases] : {std::pair{&baseline, false}, std::pair{&aliased, true}}) {
            if (const auto error = validateTextureMemoryStatistics(context.device, *executor, aliases); !error.empty()) {
                return RHITestResult::fail(error);
            }
            for (const auto name : {"A.color", "B.color"}) {
                const auto allocation = executor->outputResource(name)->texture->memoryInfo();
                if (allocation.heapIndex >= liveBudget.heaps.size()) {
                    return RHITestResult::fail("Cannot verify the measured image's device-local heap");
                }
                if (!liveBudget.heaps[allocation.heapIndex].deviceLocal) {
                    return RHITestResult::skip("Texture fixture used a non-device-local fallback heap; allocation savings cannot certify VRAM savings");
                }
            }
        }
        const auto baselineMemory = textureMemoryEvidence(baseline.textureMemoryStats());
        const auto aliasMemory = textureMemoryEvidence(aliased.textureMemoryStats());
        const auto& memory = aliased.textureMemoryStats();
        const uint64_t baselineAllocationBytes = baselineBudget.allocationBytes - initialBudget.allocationBytes;
        const uint64_t aliasAllocationBytes = aliasBudget.allocationBytes - baselineBudget.allocationBytes;
        const uint64_t baselineAllocationCount = baselineBudget.allocationCount - initialBudget.allocationCount;
        const uint64_t aliasAllocationCount = aliasBudget.allocationCount - baselineBudget.allocationCount;
        const uint64_t baselineDeviceLocalBytes = baselineBudget.deviceLocalBytes - initialBudget.deviceLocalBytes;
        const uint64_t aliasDeviceLocalBytes = aliasBudget.deviceLocalBytes - baselineBudget.deviceLocalBytes;
        if (baseline.textureMemoryStats().logicalBytes != memory.logicalBytes ||
            baselineAllocationBytes < aliasAllocationBytes ||
            baselineAllocationBytes - aliasAllocationBytes != memory.savedBytes ||
            baselineAllocationCount != aliasAllocationCount + 1 ||
            baselineDeviceLocalBytes < aliasDeviceLocalBytes ||
            baselineDeviceLocalBytes - aliasDeviceLocalBytes != memory.savedBytes) {
            return RHITestResult::fail("Reported savings do not equal actual FrameResources allocation and device-local byte deltas for identical graphs");
        }
        const uint64_t initialLogicalBytes = memory.logicalBytes;
        const uint64_t savedBytes = memory.savedBytes;
        const uint64_t initialBackingId = memory.slots.front().backingAllocationId;
        const double savedPercent = 100.0 * double(savedBytes) / double(initialLogicalBytes);
        bench::Json evidence{{"width", width}, {"height", height},
            {"secondWidth", std::max(1u, width / 2)}, {"secondHeight", std::max(1u, height / 2)},
            {"format", "RGBA8Unorm"},
            {"scope", "Current compiled graph texture allocations; excludes buffers, scene assets, driver heaps and process VRAM"},
            {"oracle", "Standalone Vulkan image requirements and unique native backing sizes; identical buffers cancel in FrameResources deltas"},
            {"aliasOff", baselineMemory}, {"aliasOn", aliasMemory}, {"savedPercent", savedPercent},
            {"frameResourcesAliasOff", {{"allocationBytes", baselineAllocationBytes}, {"allocationCount", baselineAllocationCount},
                {"deviceLocalBytes", baselineDeviceLocalBytes}}},
            {"frameResourcesAliasOn", {{"allocationBytes", aliasAllocationBytes}, {"allocationCount", aliasAllocationCount},
                {"deviceLocalBytes", aliasDeviceLocalBytes}}},
            {"frameResourcesSavedBytes", baselineAllocationBytes - aliasAllocationBytes},
            {"frameResourcesSavedDeviceLocalBytes", baselineDeviceLocalBytes - aliasDeviceLocalBytes},
            {"texturesUseDeviceLocalHeaps", true}};
        for (auto* executor : {&baseline, &aliased}) {
            if (!executor->execute({.graphicsQueue = &context.graphicsQueue}) ||
                !executor->waitForSubmittedWork(5'000'000'000ull)) {
                (void)context.device.waitIdle();
                return RHITestResult::fail("Statistics comparison graph did not complete");
            }
            const auto expectedMemory = executor == &baseline ? baselineMemory : aliasMemory;
            const auto snapshot = executor->executionSnapshot();
            if (!snapshot || textureMemoryEvidence(snapshot->textureMemory) != expectedMemory ||
                textureMemoryEvidence(executor->executionStats().textureMemory) != expectedMemory) {
                return RHITestResult::fail("Compiled, execution and captured texture statistics disagree");
            }
        }
        for (const auto output : {"ReadA.data", "ReadB.data"}) {
            std::vector<uint8_t> expected, actual;
            if (!readGraphAliasPixels(baseline, output, expected) || !readGraphAliasPixels(aliased, output, actual) ||
                expected != actual || !graphAliasPixelsMatch(actual, std::string_view(output) == "ReadA.data")) {
                return RHITestResult::fail("Statistics workload produced different alias-on/off pixels");
            }
            if (std::string_view(output) == "ReadA.data") {
                if (context.evidence) {
                    bench::readbackEvidence(context, "readback.bin", std::span<const uint8_t>(actual));
                } else {
                    const auto directory = context.outputDirectory / "texture-alias-memory";
                    std::filesystem::create_directories(directory);
                    std::ofstream file(directory / ("TextureAliasReadback-" + std::to_string(width) + "x" +
                        std::to_string(height) + ".bin"), std::ios::binary | std::ios::trunc);
                    file.exceptions(std::ios::badbit | std::ios::failbit);
                    file.write(reinterpret_cast<const char*>(actual.data()), std::streamsize(actual.size()));
                }
            }
        }
        evidence["pixelReadbackMatches"] = true;
        evidence["executionStatisticsAgree"] = true;
        const auto screenshotDirectory = context.evidence ? context.evidence->root()
            : context.outputDirectory / "texture-alias-memory";
        std::filesystem::create_directories(screenshotDirectory);
        const auto screenshotName = context.evidence ? std::string("alias-memory.png")
            : "TextureAliasMemory-" + std::to_string(width) + "x" + std::to_string(height) + ".png";
        {
            editor::RenderGraphExecutionViewer viewer;
            viewer.update(aliased.executionSnapshot());
            ViewerUIContext ui;
            if (const auto error = ui.save(viewer, editor::RenderGraphExecutionViewer::Tab::Memory,
                    screenshotDirectory / screenshotName); !error.empty()) {
                return RHITestResult::fail("Alias memory viewer screenshot: " + error);
            }
        }
        evidence["memoryViewerScreenshot"] = screenshotName;
        saveAliasMemoryEvidence(context, "alias-memory.json", evidence);
        saveAliasMemoryEvidence(context, "TextureAliasMemory-" + std::to_string(width) + "x" +
            std::to_string(height) + ".json", evidence);

        const uint32_t resizedWidth = width == 512 && height == 256 ? 64 : 512;
        const uint32_t resizedHeight = width == 512 && height == 256 ? 64 : 256;
        auto resized = aliasGraphWithDimensions(resizedWidth, resizedHeight);
        if (!aliased.compile(context.device, resized, resizedWidth, resizedHeight, options, log)) {
            return RHITestResult::fail("Resized statistics graph compilation failed: " + log);
        }
        if (const auto error = validateTextureMemoryStatistics(context.device, aliased, true); !error.empty()) {
            return RHITestResult::fail("Resized graph: " + error);
        }
        if (aliased.executionSnapshot() || aliased.textureMemoryStats().logicalBytes == initialLogicalBytes ||
            aliased.textureMemoryStats().slots.front().backingAllocationId == initialBackingId) {
            return RHITestResult::fail("Recompilation retained stale dimensions, backing identities, or execution capture");
        }
        if (!aliased.compile(context.device, resized, resizedWidth, resizedHeight, log)) {
            return RHITestResult::fail("Alias-off statistics recompile failed: " + log);
        }
        if (const auto error = validateTextureMemoryStatistics(context.device, aliased, false); !error.empty()) {
            return RHITestResult::fail("Alias-off recompile: " + error);
        }
        return RHITestResult::pass(std::to_string(width) + "x" + std::to_string(height) + ": logical=" +
            std::to_string(initialLogicalBytes) + " bytes, backing=" +
            std::to_string(initialLogicalBytes - savedBytes) + " bytes, saved=" + std::to_string(savedBytes) +
            " bytes (" + std::to_string(savedPercent) + "%); actual device-local FrameResources deltas and all readback pixels agree; resized/off recompiles refresh statistics");
    }
};

class RenderGraphBuiltinTextureAliasingTest final : public RHITest {
public:
    RenderGraphBuiltinTextureAliasingTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_texture_aliasing_builtin_clear_raster_copy";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.textureAliasing.builtinClear", "graph.textureAliasing.builtinRaster",
                "graph.textureAliasing.builtinCopy", "graph.textureAliasing.runtimePropertiesAndResize"},
            .artifacts = {"builtin-aliasing.json"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerTextureAliasPasses();
        constexpr std::array<std::pair<uint32_t, uint32_t>, 2> extents{{{64, 48}, {97, 65}}};
        const RenderGraphCompileOptions options{.enableTextureAliasing = true};
        const RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
            .computeQueue = context.device.getQueue(QueueType::Compute),
            .copyQueue = context.device.getQueue(QueueType::Copy)};
        auto cases = bench::Json::array();
        for (const auto sourceType : {"ClearColorPass", "TriangleRasterPass"}) {
            RenderGraph graph;
            graph.addNode(sourceType, "Source");
            graph.addNode("CopyColorPass", "Copy1");
            graph.addNode("CopyColorPass", "Copy2");
            graph.addNode("CopyColorPass", "Copy3");
            graph.addNode(std::string(kAliasReadType), "Read", {{"queue", "copy"}});
            graph.addEdge("Source.color", "Copy1.source");
            graph.addEdge("Copy1.color", "Copy2.source");
            graph.addEdge("Copy2.color", "Copy3.source");
            graph.addEdge("Copy3.color", "Read.color");
            graph.markOutput("Read.data");
            RenderGraphExecutor baseline;
            RenderGraphExecutor aliased;
            aliased.setExecutionCaptureEnabled(true);
            std::string log;
            for (const auto [width, height] : extents) {
                graph.setNodeProperties(graph.findNode("Read")->id,
                    {{"width", width}, {"height", height}, {"queue", "copy"}});
                if (!baseline.compile(context.device, graph, width, height, log) ||
                    !aliased.compile(context.device, graph, width, height, options, log)) {
                    return RHITestResult::fail(std::string(sourceType) + " chain compilation failed: " + log);
                }
                for (const auto& [executor, aliases] : {std::pair{&baseline, false}, std::pair{&aliased, true}}) {
                    const auto& memory = executor->textureMemoryStats();
                    if (!memory.complete || memory.unknownTextureCount || memory.textureCount != 4 ||
                        memory.transientTextureCount != 4 || memory.pinnedTextureCount ||
                        memory.backingAllocationCount != (aliases ? 2u : 4u) ||
                        memory.eligibleTextureCount != (aliases ? 4u : 0u) ||
                        memory.aliasedTextureCount != (aliases ? 4u : 0u) ||
                        memory.aliasSlotCount != (aliases ? 2u : 0u) || memory.overheadBytes ||
                        (aliases ? !memory.savedBytes : memory.savedBytes != 0)) {
                        return RHITestResult::fail(std::string(sourceType) + " chain has incorrect transient/backing/savings counts");
                    }
                    std::vector<uint64_t> backings;
                    std::vector<VkImage> images;
                    for (const auto output : {"Source.color", "Copy1.color", "Copy2.color", "Copy3.color"}) {
                        const auto* resource = executor->outputResource(output);
                        if (!resource || !resource->texture || resource->desc.width != width || resource->desc.height != height) {
                            return RHITestResult::fail("Builtin alias recompile retained a missing or stale texture extent");
                        }
                        const auto allocation = resource->texture->memoryInfo();
                        const auto image = vulkan::nativeTexture(*resource->texture).image;
                        if (!allocation.known || !allocation.backingAllocationId ||
                            std::find(images.begin(), images.end(), image) != images.end()) {
                            return RHITestResult::fail("Builtin alias chain lost distinct VkImages or native backing metadata");
                        }
                        images.push_back(image);
                        if (std::find(backings.begin(), backings.end(), allocation.backingAllocationId) == backings.end()) {
                            backings.push_back(allocation.backingAllocationId);
                        }
                    }
                    if (backings.size() != (aliases ? 2u : 4u)) {
                        return RHITestResult::fail("Builtin alias statistics do not match the actual native backing identities");
                    }
                }
                if (baseline.textureMemoryStats().logicalBytes != aliased.textureMemoryStats().logicalBytes ||
                    baseline.textureMemoryStats().backingBytes - aliased.textureMemoryStats().backingBytes !=
                        aliased.textureMemoryStats().savedBytes) {
                    return RHITestResult::fail("Builtin alias savings do not match the same graph's independent allocation capacity");
                }
                const auto compiledMemory = textureMemoryEvidence(aliased.textureMemoryStats());
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    const bool red = (frame % 2) == 0;
                    if (std::string_view(sourceType) == "ClearColorPass") {
                        graph.findNode("Source")->runtimeProperties = {{"color", red
                            ? RenderGraphProperties::array({1.0f, 0.0f, 0.0f, 1.0f})
                            : RenderGraphProperties::array({0.0f, 1.0f, 0.0f, 1.0f})}};
                        baseline.syncRuntimeProperties(graph);
                        aliased.syncRuntimeProperties(graph);
                    }
                    for (auto* executor : {&baseline, &aliased}) {
                        if (!executor->execute(submit) || !executor->waitForSubmittedWork(5'000'000'000ull)) {
                            (void)context.device.waitIdle();
                            return RHITestResult::fail(std::string(sourceType) + " builtin alias frame did not complete");
                        }
                    }
                    std::vector<uint8_t> expected, actual;
                    if (!readGraphAliasPixels(baseline, "Read.data", expected) ||
                        !readGraphAliasPixels(aliased, "Read.data", actual) || expected != actual ||
                        actual.size() != size_t(width) * height * 4 ||
                        (std::string_view(sourceType) == "ClearColorPass" && !graphAliasPixelsMatch(actual, red))) {
                        return RHITestResult::fail(std::string(sourceType) + " alias-on/off pixels differ or a runtime clear is stale");
                    }
                    if (std::string_view(sourceType) == "TriangleRasterPass") {
                        bool rasterized = false;
                        for (size_t pixel = 4; pixel < actual.size(); pixel += 4) {
                            if (std::memcmp(actual.data(), actual.data() + pixel, 3) != 0) {
                                rasterized = true;
                                break;
                            }
                        }
                        if (!rasterized) { return RHITestResult::fail("Triangle alias regression has no rasterized color coverage"); }
                    }
                    const auto snapshot = aliased.executionSnapshot();
                    if (!snapshot || !snapshot->success || textureMemoryEvidence(snapshot->textureMemory) != compiledMemory ||
                        textureMemoryEvidence(aliased.executionStats().textureMemory) != compiledMemory) {
                        return RHITestResult::fail("Builtin alias capture or execution lost its compiled memory statistics");
                    }
                }
                cases.push_back({{"source", sourceType}, {"width", width}, {"height", height}, {"frames", 4},
                    {"allPixelsMatch", true}, {"aliasOff", textureMemoryEvidence(baseline.textureMemoryStats())},
                    {"aliasOn", compiledMemory}});
            }
        }
        saveAliasMemoryEvidence(context, "builtin-aliasing.json", {{"cases", std::move(cases)}});
        return RHITestResult::pass("Real clear and triangle raster outputs survive three copy passes in two shared slots; all alias-off/on pixels match over four frames at each extent, including runtime clear changes and resize");
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphTextureAliasingReadbackTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureAliasingExternalGuardTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureAliasingEligibilityTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphTextureAliasingMemoryStatisticsTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBuiltinTextureAliasingTest);

} // namespace
} // namespace metallic::tests
