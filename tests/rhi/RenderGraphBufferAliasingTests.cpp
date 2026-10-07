#include "Runtime/Render/Core/ResourceState.h"
#include "RHITest.h"
#include "RenderGraphViewerTestUI.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace metallic::tests {
namespace {

using namespace render;

constexpr uint64_t kBufferAliasBytes = 64 * 1024;
constexpr std::string_view kBufferAliasFillType = "TestBufferAliasFill";
constexpr std::string_view kBufferAliasCopyType = "TestBufferAliasCopy";

uint32_t bufferAliasPattern(uint64_t frame)
{
    return 0x73010000u + uint32_t(frame);
}

QueueType bufferAliasQueue(const RenderGraphProperties& properties)
{
    const std::string queue = properties.value("queue", std::string("graphics"));
    return queue == "copy" ? QueueType::Copy : queue == "compute" ? QueueType::Compute : QueueType::Graphics;
}

class BufferAliasFillPass final : public UnsafePass {
public:
    QueueType queueType() const override { return bufferAliasQueue(properties()); }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return properties().value("sceneDependent", false)
            ? RenderGraphSceneDependency{.source = RenderGraphSceneSource::World} : RenderGraphSceneDependency{};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const auto bytes = properties().value("bytes", kBufferAliasBytes);
        RenderPassReflection reflection;
        reflection.addBufferInput("token").buffer(bytes).transferRead().setOptional();
        reflection.addBufferInput("source").buffer(bytes).transferRead().setOptional();
        auto& output = reflection.addBufferOutput("data").buffer(bytes).transferWrite();
        if (properties().value("transient", true)) {
            output.transient(properties().value("unknown", false)
                ? RenderGraphInitialization::Unknown : RenderGraphInitialization::FullOverwrite);
        }
        if (properties().value("host", false)) { output.hostReadback(); }
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (properties().value("sceneDependent", false) &&
            (!context.runtimeScene || !context.runtimeScene->valid())) {
            log = "The scene-dependent alias fixture must resolve a valid scene before compilation";
            return makeError(Error::InvalidArgument);
        }
        return {};
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto output = context.outputBuffer("data");
        if (!output.valid()) { return makeError(Error::InvalidArgument); }
        if (auto retained = context.commandBuffer().retainResource(output.buffer()->retainAllocation()); !retained) {
            return retained;
        }
        vulkan::ExternalCommandScope scope(context.commandBuffer());
        scope.functions().vkCmdFillBuffer(scope.commandBuffer(),
            vulkan::nativeBuffer(*output.buffer()).buffer, 0, output.buffer()->desc().size,
            bufferAliasPattern(context.frameIndex()));
        return {};
    }
};

class BufferAliasCopyPass final : public UnsafePass {
public:
    QueueType queueType() const override { return bufferAliasQueue(properties()); }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const auto bytes = properties().value("bytes", kBufferAliasBytes);
        const auto inputBytes = properties().value("inputBytes", bytes);
        RenderPassReflection reflection;
        reflection.addBufferInput("source").buffer(inputBytes).transferRead();
        auto& output = reflection.addBufferOutput("data").buffer(bytes).transferWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        if (properties().value("readback", false)) { output.hostReadback(); }
        return reflection;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto input = context.inputBuffer("source");
        auto output = context.outputBuffer("data");
        if (!input.valid() || !output.valid()) { return makeError(Error::InvalidArgument); }
        const uint64_t bytes = output.buffer()->desc().size;
        const auto source = input.buffer()->slice({.size = bytes});
        const auto destination = output.buffer()->slice();
        if (!source || !destination) { return makeError(Error::InvalidArgument); }
        if (auto copied = context.commandBuffer().copyBuffer(*source, *destination); !copied) { return copied; }
        if (properties().value("readback", false)) {
            const BufferBarrierDesc host{.buffer = output.buffer(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
            return context.commandBuffer().synchronize({.buffers = {&host, 1}});
        }
        return {};
    }
};

void registerBufferAliasPasses()
{
    registerRenderGraphPassType(std::string(kBufferAliasFillType), "Shaderless full buffer initialization",
        [] { return std::make_unique<BufferAliasFillPass>(); });
    registerRenderGraphPassType(std::string(kBufferAliasCopyType), "Full buffer copy and optional host readback",
        [] { return std::make_unique<BufferAliasCopyPass>(); });
}

RenderGraph bufferAliasChain(uint64_t largeBytes, std::string_view firstQueue, std::string_view secondQueue)
{
    const uint64_t smallBytes = largeBytes / 2;
    RenderGraph graph;
    graph.addNode(std::string(kBufferAliasFillType), "Write", {{"bytes", largeBytes}});
    graph.addNode(std::string(kBufferAliasCopyType), "Copy1", {{"bytes", largeBytes}, {"queue", firstQueue}});
    graph.addNode(std::string(kBufferAliasCopyType), "Copy2", {{"bytes", smallBytes}, {"inputBytes", largeBytes}});
    graph.addNode(std::string(kBufferAliasCopyType), "Copy3", {{"bytes", smallBytes}, {"queue", secondQueue}});
    graph.addNode(std::string(kBufferAliasCopyType), "Readback", {{"bytes", smallBytes}, {"readback", true}});
    graph.addEdge("Write.data", "Copy1.source");
    graph.addEdge("Copy1.data", "Copy2.source");
    graph.addEdge("Copy2.data", "Copy3.source");
    graph.addEdge("Copy3.data", "Readback.source");
    graph.markOutput("Readback.data");
    return graph;
}

RenderGraph bufferAliasBranches(bool dependency = true, bool overlapping = false)
{
    RenderGraph graph;
    graph.addNode(std::string(kBufferAliasFillType), "A", {{"bytes", kBufferAliasBytes}});
    graph.addNode(std::string(kBufferAliasCopyType), "ReadA", {{"bytes", kBufferAliasBytes}, {"readback", true}});
    graph.addNode(std::string(kBufferAliasFillType), "B", {{"bytes", kBufferAliasBytes}});
    graph.addNode(std::string(kBufferAliasCopyType), "ReadB", {{"bytes", kBufferAliasBytes}, {"readback", true}});
    graph.addEdge("A.data", "ReadA.source");
    if (dependency) { graph.addEdge("ReadA.data", "B.token"); }
    if (overlapping) { graph.addEdge("A.data", "B.source"); }
    graph.addEdge("B.data", "ReadB.source");
    graph.markOutput("ReadA.data");
    graph.markOutput("ReadB.data");
    return graph;
}

bool bufferAliasSharesBacking(RenderGraphExecutor& executor, std::string_view first, std::string_view second)
{
    const auto* a = executor.outputResource(first);
    const auto* b = executor.outputResource(second);
    if (!a || !b || !a->buffer || !b->buffer) { return false; }
    const auto firstInfo = a->buffer->memoryInfo();
    const auto secondInfo = b->buffer->memoryInfo();
    return firstInfo.known && secondInfo.known && firstInfo.backingAllocationId &&
        firstInfo.backingAllocationId == secondInfo.backingAllocationId &&
        firstInfo.memoryBlockId == secondInfo.memoryBlockId && firstInfo.offsetBytes == secondInfo.offsetBytes;
}

bool readBufferAliasWords(RenderGraphExecutor& executor, std::string_view name, std::vector<uint32_t>& words)
{
    const auto* resource = executor.outputResource(name);
    if (!resource || !resource->buffer) { return false; }
    resource->buffer->invalidate();
    const void* mapped = resource->buffer->map();
    if (!mapped) { return false; }
    words.resize(size_t(resource->buffer->desc().size / 4));
    std::memcpy(words.data(), mapped, words.size() * sizeof(uint32_t));
    resource->buffer->unmap();
    return true;
}

bool bufferAliasWordsMatch(const std::vector<uint32_t>& words, uint32_t expected)
{
    return !words.empty() && std::all_of(words.begin(), words.end(),
        [expected](uint32_t word) { return word == expected; });
}

bench::Json bufferMemoryEvidence(const auto& stats)
{
    auto slots = bench::Json::array();
    for (const auto& slot : stats.slots) {
        slots.push_back({{"complete", slot.complete}, {"backingAllocationId", slot.backingAllocationId},
            {"logicalBytes", slot.logicalBytes}, {"backingBytes", slot.backingBytes},
            {"savedBytes", slot.savedBytes}, {"overheadBytes", slot.overheadBytes}, {"resources", slot.resources}});
    }
    return {{"aliasingEnabled", stats.aliasingEnabled}, {"complete", stats.complete},
        {"bufferCount", stats.bufferCount}, {"transientBufferCount", stats.transientBufferCount},
        {"pinnedBufferCount", stats.pinnedBufferCount}, {"eligibleBufferCount", stats.eligibleBufferCount},
        {"aliasedBufferCount", stats.aliasedBufferCount}, {"aliasSlotCount", stats.aliasSlotCount},
        {"backingAllocationCount", stats.backingAllocationCount}, {"unknownBufferCount", stats.unknownBufferCount},
        {"logicalBytes", stats.logicalBytes}, {"backingBytes", stats.backingBytes},
        {"savedBytes", stats.savedBytes}, {"overheadBytes", stats.overheadBytes}, {"slots", std::move(slots)}};
}

void saveBufferMemoryEvidence(RHITestContext& context, const bench::Json& value)
{
    if (context.evidence) { context.evidence->json("buffer-memory.json", value); return; }
    const auto directory = context.outputDirectory / "buffer-alias-memory";
    std::filesystem::create_directories(directory);
    std::ofstream output(directory / "BufferAliasMemory.json", std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << value.dump(2) << '\n';
}

std::string validateBufferChainMemory(Device& device, RenderGraphExecutor& executor, bool aliases)
{
    const auto& stats = executor.bufferMemoryStats();
    if (!stats.complete || stats.unknownBufferCount || stats.aliasingEnabled != aliases ||
        stats.bufferCount != 5 || stats.transientBufferCount != 5 || stats.pinnedBufferCount != 1 ||
        stats.eligibleBufferCount != (aliases ? 4u : 0u) || stats.aliasedBufferCount != (aliases ? 4u : 0u) ||
        stats.aliasSlotCount != (aliases ? 2u : 0u) || stats.backingAllocationCount != (aliases ? 3u : 5u)) {
        return "Buffer statistics do not count four device outputs, one pinned host readback and two actual alias slots";
    }
    uint64_t logicalBytes = 0;
    uint64_t backingBytes = 0;
    std::unordered_map<uint64_t, uint64_t> backingSizes;
    for (const auto name : {"Write.data", "Copy1.data", "Copy2.data", "Copy3.data", "Readback.data"}) {
        const auto* resource = executor.outputResource(name);
        if (!resource || !resource->buffer) { return "Buffer alias chain omitted a resource"; }
        const auto info = resource->buffer->memoryInfo();
        if (!info.known || !info.backingAllocationId || !info.backingSizeBytes) {
            return "Cannot verify buffer savings against native backing metadata";
        }
        backingSizes[info.backingAllocationId] = info.backingSizeBytes;
        if (aliases && std::string_view(name) != "Readback.data") {
            const auto standalone = device.bufferAllocationSize(resource->buffer->desc());
            if (!standalone) { return "Cannot query independent native buffer allocation requirements"; }
            logicalBytes += *standalone;
        } else {
            logicalBytes += info.backingSizeBytes;
        }
    }
    for (const auto& [id, size] : backingSizes) { backingBytes += size; }
    if (stats.logicalBytes != logicalBytes || stats.backingBytes != backingBytes || stats.overheadBytes ||
        stats.savedBytes != logicalBytes - backingBytes || stats.slots.size() != (aliases ? 2u : 0u)) {
        return "Buffer memory totals disagree with native requirement sizes and unique backing owners";
    }
    if (aliases) {
        uint64_t slotSavings = 0;
        for (const auto& slot : stats.slots) {
            if (!slot.complete || slot.resources.size() != 2 || !backingSizes.contains(slot.backingAllocationId) ||
                slot.backingBytes != backingSizes.at(slot.backingAllocationId) || slot.overheadBytes ||
                slot.savedBytes != slot.logicalBytes - slot.backingBytes) {
                return "An actual buffer alias slot lost native backing capacity, membership or savings";
            }
            slotSavings += slot.savedBytes;
        }
        if (!stats.savedBytes || slotSavings != stats.savedBytes ||
            !bufferAliasSharesBacking(executor, "Write.data", "Copy2.data") ||
            !bufferAliasSharesBacking(executor, "Copy1.data", "Copy3.data") ||
            bufferAliasSharesBacking(executor, "Write.data", "Copy1.data")) {
            return "Buffer alias allocation paired overlapping copies or did not reuse the two disjoint lifetimes";
        }
    }
    return {};
}

std::string validateBufferAliasCapture(const RenderGraphExecutionSnapshot& snapshot)
{
    if (!snapshot.success || snapshot.externalRecording || snapshot.status != RenderGraphExecutionSnapshotStatus::Submitted) {
        return "Buffer alias capture does not describe successful managed submission";
    }
    for (const auto name : {"Write", "Copy1", "Copy2", "Copy3"}) {
        const auto pass = std::find_if(snapshot.passes.begin(), snapshot.passes.end(),
            [name](const auto& value) { return value.name == name; });
        const std::string resourceName = std::string(name) + ".data";
        const auto resource = std::find_if(snapshot.resources.begin(), snapshot.resources.end(),
            [&](const auto& value) { return value.name == resourceName ||
                std::find(value.aliases.begin(), value.aliases.end(), resourceName) != value.aliases.end(); });
        if (pass == snapshot.passes.end() || resource == snapshot.resources.end()) {
            return "Buffer alias capture omitted an occupant or its native allocation";
        }
        const auto handover = std::find_if(pass->barriers.begin(), pass->barriers.end(), [&](const auto& value) {
            return value.memoryAliasing && value.resourceId == resource->id &&
                value.beforeScope.stages == PipelineStageBits::AllCommands &&
                value.afterScope.stages == PipelineStageBits::AllCommands &&
                value.beforeScope.access == (AccessBits::MemoryRead | AccessBits::MemoryWrite) &&
                value.afterScope.access == (AccessBits::MemoryRead | AccessBits::MemoryWrite);
        });
        if (handover == pass->barriers.end()) { return "Buffer activation has no global physical-memory handover barrier"; }
    }
    for (const auto pair : {std::pair{"Copy1", "Copy2"}, std::pair{"Copy2", "Copy3"}}) {
        const auto before = std::find_if(snapshot.passes.begin(), snapshot.passes.end(),
            [&](const auto& value) { return value.name == pair.first; });
        const auto after = std::find_if(snapshot.passes.begin(), snapshot.passes.end(),
            [&](const auto& value) { return value.name == pair.second; });
        if (std::find(after->predecessors.begin(), after->predecessors.end(), before->id) == after->predecessors.end()) {
            return "Buffer activation lost its previous occupant's terminal use predecessor";
        }
        if (before->actualQueueId == after->actualQueueId) { continue; }
        const auto source = std::find_if(snapshot.segments.begin(), snapshot.segments.end(),
            [&](const auto& value) { return value.passId == before->id; });
        const auto activation = std::find_if(snapshot.segments.begin(), snapshot.segments.end(),
            [&](const auto& value) { return value.passId == after->id; });
        if (source == snapshot.segments.end() || activation == snapshot.segments.end() ||
            std::find(activation->predecessors.begin(), activation->predecessors.end(), source->id) == activation->predecessors.end()) {
            return "Cross-queue buffer activation has no previous occupant segment join";
        }
        const auto batch = std::find_if(snapshot.batches.begin(), snapshot.batches.end(), [&](const auto& value) {
            return std::find(value.segmentIds.begin(), value.segmentIds.end(), activation->id) != value.segmentIds.end();
        });
        if (batch == snapshot.batches.end() || !batch->accepted || !batch->semaphoreWaitCount ||
            std::find(batch->waitPredecessors.begin(), batch->waitPredecessors.end(), source->id) == batch->waitPredecessors.end()) {
            return "Cross-queue buffer alias handover submitted without the terminal use semaphore wait";
        }
    }
    return {};
}

class RenderGraphBufferAliasingReadbackTest final : public RHITest {
public:
    RenderGraphBufferAliasingReadbackTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_buffer_aliasing_frame_readback_and_statistics";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.bufferAliasing.frameReadback", "graph.bufferAliasing.activationBarrier",
                "graph.bufferAliasing.crossQueueJoin", "graph.bufferAliasing.nativeSavingsStatistics"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerBufferAliasPasses();
        auto* compute = context.device.getQueue(QueueType::Compute);
        auto* copy = context.device.getQueue(QueueType::Copy);
        const auto graphicsNative = vulkan::nativeQueue(context.graphicsQueue).queue;
        if (compute && vulkan::nativeQueue(*compute).queue == graphicsNative) { compute = nullptr; }
        if (copy && vulkan::nativeQueue(*copy).queue == graphicsNative) { copy = nullptr; }
        struct Mode { const char* first; const char* second; };
        std::vector<Mode> modes{{"graphics", "graphics"}};
        if (compute) { modes.push_back({"compute", "graphics"}); }
        if (copy) { modes.push_back({"graphics", "copy"}); }
        if (compute && copy) { modes.push_back({"compute", "copy"}); }
        bench::Json evidence{{"queueModes", bench::Json::array()}};
        for (size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
            const auto mode = modes[modeIndex];
            const uint64_t largeBytes = modeIndex == 0 ? 8 * 1024 * 1024 : kBufferAliasBytes;
            const auto graph = bufferAliasChain(largeBytes, mode.first, mode.second);
            RenderGraphExecutor baseline, aliased;
            baseline.setExecutionCaptureEnabled(true);
            aliased.setExecutionCaptureEnabled(true);
            std::string log;
            if (!baseline.compile(context.device, graph, 32, 32, log) ||
                !aliased.compile(context.device, graph, 32, 32, {.enableBufferAliasing = true}, log)) {
                return RHITestResult::fail("Buffer alias chain compilation failed: " + log);
            }
            if (const auto error = validateBufferChainMemory(context.device, baseline, false); !error.empty()) {
                return RHITestResult::fail("Alias-off buffer statistics: " + error);
            }
            if (const auto error = validateBufferChainMemory(context.device, aliased, true); !error.empty()) {
                return RHITestResult::fail("Alias-on buffer statistics: " + error);
            }
            const auto compiledMemory = aliased.bufferMemoryStats();
            const auto* first = aliased.outputResource("Write.data");
            const auto* reused = aliased.outputResource("Copy2.data");
            if (first->buffer->memoryInfo().allocationId == reused->buffer->memoryInfo().allocationId ||
                vulkan::nativeBuffer(*first->buffer).buffer == vulkan::nativeBuffer(*reused->buffer).buffer ||
                !first->buffer->deviceAddress() || !reused->buffer->deviceAddress() ||
                aliased.isExportedOutput("Write.data") || !aliased.isExportedOutput("Readback.data")) {
                return RHITestResult::fail("Graph buffer aliases lost independent native identity/BDA or exposed a transient export");
            }
            RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
                .computeQueue = compute, .copyQueue = copy};
            for (uint32_t frame = 0; frame < 4; ++frame) {
                for (auto* executor : {&baseline, &aliased}) {
                    if (!executor->execute(submit) || !executor->waitForSubmittedWork(5'000'000'000ull)) {
                        (void)context.device.waitIdle();
                        return RHITestResult::fail("Buffer alias on/off frame did not complete");
                    }
                }
                const auto snapshot = aliased.executionSnapshot();
                if (!snapshot || snapshot->bufferMemory != compiledMemory ||
                    aliased.executionStats().bufferMemory != compiledMemory) {
                    return RHITestResult::fail("Execution statistics/capture lost compiled buffer allocation totals");
                }
                if (const auto error = validateBufferAliasCapture(*snapshot); !error.empty()) {
                    return RHITestResult::fail(error);
                }
                std::vector<uint32_t> expected, actual;
                if (!readBufferAliasWords(baseline, "Readback.data", expected) ||
                    !readBufferAliasWords(aliased, "Readback.data", actual) || actual != expected ||
                    !bufferAliasWordsMatch(actual, bufferAliasPattern(frame))) {
                    return RHITestResult::fail("Buffer alias chain readback differs from baseline or a fully initialized frame pattern");
                }
                if (modeIndex == 0 && frame == 0) {
                    if (context.evidence) {
                        bench::readbackEvidence(context, "buffer-readback.bin",
                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(actual.data()), actual.size() * 4));
                    }
                    const auto directory = context.evidence ? context.evidence->root()
                        : context.outputDirectory / "buffer-alias-memory";
                    std::filesystem::create_directories(directory);
                    editor::RenderGraphExecutionViewer viewer;
                    viewer.update(snapshot);
                    ViewerUIContext ui;
                    if (const auto error = ui.save(viewer, editor::RenderGraphExecutionViewer::Tab::Memory,
                            directory / "buffer-alias-memory.png"); !error.empty()) {
                        return RHITestResult::fail("Buffer alias memory viewer screenshot: " + error);
                    }
                    evidence["memoryViewerScreenshot"] = "buffer-alias-memory.png";
                }
            }
            evidence["queueModes"].push_back({{"firstQueue", mode.first}, {"secondQueue", mode.second},
                {"largeBufferBytes", largeBytes}, {"smallBufferBytes", largeBytes / 2},
                {"baseline", bufferMemoryEvidence(baseline.bufferMemoryStats())},
                {"aliased", bufferMemoryEvidence(compiledMemory)}, {"readbackWordsMatch", true}, {"frames", 4}});
            const auto resized = bufferAliasChain(kBufferAliasBytes / 2, mode.first, mode.second);
            if (!aliased.compile(context.device, resized, 32, 32, {.enableBufferAliasing = true}, log) ||
                !validateBufferChainMemory(context.device, aliased, true).empty() || aliased.executionSnapshot() ||
                aliased.bufferMemoryStats().logicalBytes == compiledMemory.logicalBytes ||
                aliased.bufferMemoryStats().slots.front().backingAllocationId == compiledMemory.slots.front().backingAllocationId) {
                return RHITestResult::fail("Buffer size recompile retained stale capacities, capture or backing identities");
            }
            if (!aliased.compile(context.device, resized, 32, 32, log) ||
                !validateBufferChainMemory(context.device, aliased, false).empty()) {
                return RHITestResult::fail("Alias-off buffer recompile retained shared backing or savings");
            }
        }
        saveBufferMemoryEvidence(context, evidence);
        return RHITestResult::pass("Four device buffers reuse two backing slots; all native byte/statistics and four-frame word oracles agree in " +
            std::to_string(modes.size()) + " available graphics/compute/copy modes; resizing/off recompiles refresh statistics");
    }
};

class RenderGraphBuiltinBufferAliasingTest final : public RHITest {
public:
    RenderGraphBuiltinBufferAliasingTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_buffer_aliasing_builtin_bindless_shader_chain";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization,
                .capabilities = {bench::Capability::Bindless}},
            .coverage = {"graph.bufferAliasing.builtinShaderChain", "graph.bufferAliasing.bindlessViews",
                "graph.bufferAliasing.deviceAddressAndNativeIdentity", "graph.bufferAliasing.pipelinedFrameReuse"},
            .artifacts = {"buffer-builtin-memory.json", "buffer-builtin-readback.bin"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr std::array<uint32_t, 4> expectedWords{0x11223344u, 0xaabbccddu, 0xdeadbeefu, 0xcafebabeu};
        constexpr uint64_t bytes = sizeof(expectedWords);
        if (!context.device.capabilities().bindlessDescriptorHeap) {
            return RHITestResult::skip("Builtin buffer shaders require the bindless descriptor heap");
        }
        registerBufferAliasPasses();
        // These builtins dispatch one Store4/Load4, so their FullOverwrite
        // contract is valid specifically for their fixed 16-byte descriptors.
        for (const auto type : {"RenderGraphBufferWritePass", "RenderGraphBufferCopyPass"}) {
            auto pass = createRenderGraphPass(type);
            if (!pass) { return RHITestResult::fail("Builtin buffer pass registration is missing"); }
            const auto reflection = pass->reflect({.device = &context.device});
            const auto* output = reflection.findField("data", RenderGraphFieldVisibility::Output);
            if (!output || output->size != bytes || output->memoryLocation != MemoryLocation::Device ||
                output->lifetime != RenderGraphResourceLifetime::Transient ||
                output->initialization != RenderGraphInitialization::FullOverwrite ||
                output->access != RenderGraphResourceAccess::BufferStorageWrite ||
                output->bindlessAccess != RenderGraphBindlessAccess::Buffer) {
                return RHITestResult::fail("Builtin buffer descriptor no longer matches its full 16-byte shader initialization contract");
            }
        }
        RenderGraph graph;
        graph.setName("BuiltinBufferAliasShaderChain");
        graph.addNode("RenderGraphBufferWritePass", "Write");
        for (const auto name : {"Copy1", "Copy2", "Copy3"}) {
            graph.addNode("RenderGraphBufferCopyPass", name);
        }
        graph.addNode(std::string(kBufferAliasCopyType), "Readback", {{"bytes", bytes}, {"readback", true}});
        graph.addEdge("Write.data", "Copy1.source");
        graph.addEdge("Copy1.data", "Copy2.source");
        graph.addEdge("Copy2.data", "Copy3.source");
        graph.addEdge("Copy3.data", "Readback.source");
        graph.markOutput("Readback.data");
        RenderGraphExecutor baseline, aliased;
        aliased.setExecutionCaptureEnabled(true);
        std::string log;
        if (!baseline.compile(context.device, graph, 1, 1, log) ||
            !aliased.compile(context.device, graph, 1, 1, {.enableBufferAliasing = true}, log)) {
            return RHITestResult::fail("Builtin buffer alias shader chain compilation failed: " + log);
        }
        for (auto* executor : {&baseline, &aliased}) {
            if (const auto error = validateBufferChainMemory(context.device, *executor, executor == &aliased); !error.empty()) {
                return RHITestResult::fail("Builtin buffer alias memory oracle: " + error);
            }
        }
        const auto compiledMemory = aliased.bufferMemoryStats();
        auto resources = bench::Json::array();
        std::vector<uint32_t> descriptorIndices;
        std::vector<VkBuffer> nativeBuffers;
        for (const auto name : {"Write.data", "Copy1.data", "Copy2.data", "Copy3.data"}) {
            const auto* resource = aliased.outputResource(name);
            if (!resource || !resource->buffer || !resource->bufferView ||
                resource->buffer->desc().size != bytes || resource->buffer->desc().memoryLocation != MemoryLocation::Device ||
                resource->bufferView->desc().range.offset || resource->bufferView->desc().range.size != bytes ||
                !resource->bindlessHandle.valid() || !resource->buffer->deviceAddress()) {
                return RHITestResult::fail("Deferred alias buffer view creation omitted an exact-size bindless descriptor or valid BDA");
            }
            const auto native = vulkan::nativeBuffer(*resource->buffer);
            if (native.buffer == VK_NULL_HANDLE || native.address != resource->buffer->deviceAddress() ||
                std::find(descriptorIndices.begin(), descriptorIndices.end(), resource->bindlessHandle.shaderIndex) != descriptorIndices.end() ||
                std::find(nativeBuffers.begin(), nativeBuffers.end(), native.buffer) != nativeBuffers.end()) {
                return RHITestResult::fail("Builtin shader aliases conflated native buffer objects or bindless descriptor indices");
            }
            descriptorIndices.push_back(resource->bindlessHandle.shaderIndex);
            nativeBuffers.push_back(native.buffer);
            const auto memory = resource->buffer->memoryInfo();
            // Device addresses may coincide for overlapping aliases. Native
            // objects and descriptor indices still identify separate resources.
            resources.push_back({{"name", name}, {"deviceAddress", native.address},
                {"descriptorIndex", resource->bindlessHandle.shaderIndex}, {"allocationId", memory.allocationId},
                {"backingAllocationId", memory.backingAllocationId}, {"viewBytes", resource->bufferView->desc().range.size}});
        }
        auto* compute = context.device.getQueue(QueueType::Compute);
        std::vector<Queue*> queues{&context.graphicsQueue};
        if (compute && !compute->sameQueue(context.graphicsQueue)) { queues.push_back(compute); }
        auto configurations = bench::Json::array();
        uint32_t frameCount = 0;
        std::vector<uint32_t> actualWords;
        for (auto* shaderQueue : queues) {
            for (const uint32_t workers : {1u, 4u}) {
                for (const auto mode : {FrameSubmissionMode::Joined, FrameSubmissionMode::Pipelined}) {
                    const RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
                        .computeQueue = shaderQueue, .recordingWorkerLimit = workers,
                        .recordingBatchWorkload = 1, .submissionMode = mode};
                    for (auto* executor : {&baseline, &aliased}) {
                        // Consecutive submissions exercise alias handovers across
                        // both frame slots before the final host mapping.
                        for (uint32_t frame = 0; frame < 4; ++frame) {
                            if (!executor->execute(submit)) {
                                (void)context.device.waitIdle();
                                return RHITestResult::fail("Builtin buffer alias continuous-frame submission failed");
                            }
                        }
                        if (!executor->waitForSubmittedWork(5'000'000'000ull)) {
                            (void)context.device.waitIdle();
                            return RHITestResult::fail("Builtin buffer alias shader chain did not complete");
                        }
                    }
                    const auto snapshot = aliased.executionSnapshot();
                    if (!snapshot || snapshot->bufferMemory != compiledMemory || aliased.executionStats().bufferMemory != compiledMemory) {
                        return RHITestResult::fail("Builtin shader capture lost buffer memory statistics");
                    }
                    if (const auto error = validateBufferAliasCapture(*snapshot); !error.empty()) {
                        return RHITestResult::fail("Builtin shader alias capture: " + error);
                    }
                    std::vector<uint32_t> baselineWords;
                    if (!readBufferAliasWords(baseline, "Readback.data", baselineWords) ||
                        !readBufferAliasWords(aliased, "Readback.data", actualWords) || actualWords != baselineWords ||
                        actualWords.size() != expectedWords.size() ||
                        !std::equal(actualWords.begin(), actualWords.end(), expectedWords.begin())) {
                        return RHITestResult::fail("Builtin bindless Store4/Load4 chain changed or corrupted the four-word readback pattern");
                    }
                    frameCount += 4;
                    configurations.push_back({{"shaderQueue", shaderQueue->sameQueue(context.graphicsQueue) ? "graphics" : "compute"},
                        {"recordingWorkers", workers}, {"submissionMode", mode == FrameSubmissionMode::Joined ? "joined" : "pipelined"},
                        {"continuousFrames", 4}, {"readbackWordsMatch", true}});
                }
            }
        }
        const bench::Json evidence{{"expectedWords", expectedWords}, {"actualWords", actualWords},
            {"builtinDescriptorBytes", bytes}, {"resources", std::move(resources)},
            {"baseline", bufferMemoryEvidence(baseline.bufferMemoryStats())},
            {"aliased", bufferMemoryEvidence(compiledMemory)}, {"configurations", std::move(configurations)},
            {"aliasFrames", frameCount}, {"readbackWordsMatch", true}};
        if (context.evidence) {
            context.evidence->json("buffer-builtin-memory.json", evidence);
            bench::readbackEvidence(context, "buffer-builtin-readback.bin", std::span<const uint32_t>(actualWords));
        } else {
            const auto directory = context.outputDirectory / "buffer-alias-memory";
            std::filesystem::create_directories(directory);
            std::ofstream output(directory / "BuiltinBufferAliasMemory.json", std::ios::binary | std::ios::trunc);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << evidence.dump(2) << '\n';
        }
        return RHITestResult::pass("Builtin Slang writer and three bindless shader copies preserve all four words over " +
            std::to_string(frameCount) + " alias frames; four independent VkBuffers/descriptors reuse two compatible backing slots");
    }
};

class BufferAliasDebugObserver final : public IRenderDebugObserver {
public:
    void compiled(debug::DebugValue) override {}
    void beginExecution(Device&, debug::DebugEvidenceStamp, RenderSubsystemHost*) override {}
    void boundary(CommandBuffer&, std::string_view, uint32_t, std::string_view,
        std::span<const DebugResourceBinding>, const debug::DebugValue&) override {}
    void endExecution(bool) override {}
};

class RenderGraphBufferAliasingEligibilityTest final : public RHITest {
public:
    RenderGraphBufferAliasingEligibilityTest()
    {
        type = RHITestType::Resource;
        name = "render_graph_buffer_aliasing_lifetime_exports_and_debug";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.bufferAliasing.lifetimeExclusions", "graph.bufferAliasing.exports",
                "graph.bufferAliasing.debugAndHostExclusions"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerBufferAliasPasses();
        struct Case {
            const char* name;
            bool independent = false;
            bool overlapping = false;
            bool persistent = false;
            bool unknown = false;
            bool marked = false;
            bool extra = false;
            bool preview = false;
            bool debug = false;
            bool host = false;
            bool sceneDependent = false;
        };
        constexpr Case cases[] = {
            {.name = "Independent branches", .independent = true},
            {.name = "Same-pass overlapping use", .overlapping = true},
            {.name = "Persistent output", .persistent = true},
            {.name = "Unknown initialization", .unknown = true},
            {.name = "Marked output", .marked = true},
            {.name = "Extra output", .extra = true},
            {.name = "Preview output", .preview = true},
            {.name = "Debug observer", .debug = true},
            {.name = "Host buffer", .host = true},
            {.name = "Scene-dependent producer", .sceneDependent = true},
        };
        for (const auto& test : cases) {
            auto graph = bufferAliasBranches(!test.independent, test.overlapping);
            auto properties = graph.findNode("A")->properties;
            properties["transient"] = !test.persistent;
            properties["unknown"] = test.unknown;
            properties["host"] = test.host;
            properties["sceneDependent"] = test.sceneDependent;
            if (test.sceneDependent) {
                const auto directory = std::filesystem::absolute(
                    (context.evidence ? context.evidence->root() : context.outputDirectory) / "buffer-alias-scene");
                std::filesystem::create_directories(directory);
                const auto scenePath = directory / "EmptyScene.gltf";
                {
                    std::ofstream output(scenePath, std::ios::binary | std::ios::trunc);
                    output.exceptions(std::ios::badbit | std::ios::failbit);
                    output << R"({"asset":{"version":"2.0","generator":"MetallicRHITests"},"scene":0,"scenes":[{"nodes":[]}],"nodes":[]})";
                }
                // Resolve an actual scene through the asset/local path, without
                // requiring an editor world or external scene geometry/assets.
                properties["path"] = scenePath.generic_string();
                properties["sceneBinding"] = "asset";
                properties["viewBinding"] = "local";
            }
            graph.setNodeProperties(graph.findNode("A")->id, std::move(properties));
            if (test.marked) { graph.markOutput("A.data"); }
            RenderGraphCompileOptions options{.enableBufferAliasing = true};
            options.enablePreviewOutputAccess = test.preview;
            if (test.extra || test.preview) { options.extraOutputs.push_back("A.data"); }
            BufferAliasDebugObserver observer;
            RenderGraphExecutor executor;
            if (test.debug) { executor.setDebugObserver(&observer); }
            std::string log;
            if (!executor.compile(context.device, graph, 32, 32, options, log)) {
                return RHITestResult::fail(std::string(test.name) + " buffer exclusion compilation failed: " + log);
            }
            const auto& memory = executor.bufferMemoryStats();
            if (bufferAliasSharesBacking(executor, "A.data", "B.data") || !memory.aliasingEnabled || !memory.complete ||
                memory.unknownBufferCount || memory.bufferCount != 4 || memory.aliasedBufferCount ||
                memory.aliasSlotCount || memory.backingAllocationCount != 4 || memory.savedBytes ||
                memory.overheadBytes || memory.logicalBytes != memory.backingBytes || !memory.slots.empty()) {
                return RHITestResult::fail(std::string(test.name) + " incorrectly reused backing or reported savings");
            }
            if (test.sceneDependent && memory.eligibleBufferCount != 1) {
                return RHITestResult::fail("A successfully resolved scene-dependent producer entered the native alias candidate set");
            }
            if (!executor.isExportedOutput("A.data") || !executor.isExportedOutput("ReadA.data") ||
                ((test.marked || test.extra || test.preview) && memory.pinnedBufferCount < 3)) {
                return RHITestResult::fail(std::string(test.name) + " lost the pinned export contract");
            }
        }
        return RHITestResult::pass("Independent/same-pass lifetimes, persistent/unknown/host/scene outputs and export/debug access do not alias");
    }
};

class RenderGraphBufferAliasingExternalGuardTest final : public RHITest {
public:
    RenderGraphBufferAliasingExternalGuardTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_buffer_aliasing_external_recording_guards";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization},
            .coverage = {"graph.bufferAliasing.externalRecordingGuard", "graph.bufferAliasing.cancelRebuild"}};
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerBufferAliasPasses();
        const auto graph = bufferAliasChain(kBufferAliasBytes, "graphics", "graphics");
        const RenderGraphCompileOptions options{.enableBufferAliasing = true};
        RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        if (!executor.compile(context.device, graph, 32, 32, options, log) ||
            !bufferAliasSharesBacking(executor, "Write.data", "Copy2.data")) {
            return RHITestResult::fail("Cannot compile buffer external guard fixture: " + log);
        }
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands, otherCommands;
        RenderFrameContext frame;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) || !commands->begin()) {
            return RHITestResult::fail("Cannot begin untracked buffer alias guard commands");
        }
        if (!hasError(executor.execute(*commands), Error::Unsupported) || executor.executionSnapshot() ||
            !hasError(executor.transitionOutput(*commands, "Write.data", ResourceState::TransferSource), Error::InvalidArgument)) {
            return RHITestResult::fail("An untracked buffer alias recording or shared transient export was accepted");
        }
        if (!commands->end()) { return RHITestResult::fail("Cannot finish untracked buffer guard commands"); }
        commands.reset();
        if (!pool->reset() || !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !frame.begin(0) || !commands->begin(frame.submissionContext()) || !executor.execute(*commands) || !commands->end()) {
            return RHITestResult::fail("Cannot record tracked but unsubmitted buffer alias graph");
        }
        const auto recorded = executor.executionSnapshot();
        const auto frameIndex = executor.streamingStats().frameIndex;
        if (!recorded || recorded->status != RenderGraphExecutionSnapshotStatus::Recorded ||
            !recorded->success || !recorded->externalRecording || frame.completion().isSubmitted()) {
            return RHITestResult::fail("External buffer alias capture did not remain recorded and unsubmitted");
        }
        if (!pool->createCommandBuffer().transform([&](auto value) { otherCommands = std::move(value); }) ||
            !otherCommands->begin(frame.submissionContext())) {
            return RHITestResult::fail("Cannot begin a second buffer external guard probe");
        }
        if (!hasError(executor.execute(*otherCommands), Error::InvalidArgument) ||
            !hasError(executor.execute({.graphicsQueue = &context.graphicsQueue}), Error::InvalidArgument) ||
            executor.executionSnapshot() != recorded || executor.streamingStats().frameIndex != frameIndex) {
            return RHITestResult::fail("Pending buffer alias recording admitted another execution or mutated state");
        }
        if (!otherCommands->end()) { return RHITestResult::fail("Cannot end second buffer guard probe"); }
        const auto cancelled = frame.completion();
        frame.cancel();
        commands.reset();
        otherCommands.reset();
        if (!pool->reset() || !frame.reset() || !cancelled.isCancelled() ||
            !hasError(executor.execute({.graphicsQueue = &context.graphicsQueue}), Error::InvalidArgument) || executor.compiled()) {
            return RHITestResult::fail("Cancelled buffer alias recording reused stale native resource states");
        }
        if (!executor.compile(context.device, graph, 32, 32, options, log) ||
            !executor.execute({.graphicsQueue = &context.graphicsQueue}) ||
            !executor.waitForSubmittedWork(5'000'000'000ull)) {
            (void)context.device.waitIdle();
            return RHITestResult::fail("Buffer alias graph failed a rebuilt retry after cancellation: " + log);
        }
        std::vector<uint32_t> words;
        const auto retry = executor.executionSnapshot();
        if (!retry || !readBufferAliasWords(executor, "Readback.data", words) ||
            !bufferAliasWordsMatch(words, bufferAliasPattern(retry->executionId))) {
            return RHITestResult::fail("Buffer alias graph rebuilt after cancellation produced stale readback words");
        }
        return RHITestResult::pass("Untracked/pending buffer alias executions and transient exports are rejected; cancellation requires rebuild before a word-correct retry");
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphBufferAliasingReadbackTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBuiltinBufferAliasingTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBufferAliasingEligibilityTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphBufferAliasingExternalGuardTest);

} // namespace
} // namespace metallic::tests
