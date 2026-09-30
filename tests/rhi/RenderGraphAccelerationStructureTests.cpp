#include "RhiTest.h"
#include "harness/RayQueryFixture.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

using namespace render;

#define GRAPH_AS_REQUIRE(expression) do { \
    const auto& checked = (expression); \
    if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + \
        resultToString(Result<>{std::unexpected(checked.error())}) + " " + log); } \
} while (false)
#define GRAPH_AS_CHECK(expression) do { if (!(expression)) { return RhiTestResult::fail(#expression); } } while (false)

class GraphAccelerationStructureBuildPass final : public RenderGraphPass {
public:
    RenderGraphPassKind kind() const override { return RenderGraphPassKind::Unsafe; }
    QueueType queueType() const override { return properties().value("fork", false) ? QueueType::Graphics : QueueType::Compute; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    CpuRecordingPolicy cpuRecordingPolicy() const override
    {
        return properties().value("fork", false) ? CpuRecordingPolicy::Serial : CpuRecordingPolicy::ParallelJoined;
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureOutput("structure");
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string&) override
    {
        device_ = context.device;
        if (!device_) { return makeError(Error::InvalidArgument); }
        auto vertex = device_->createBuffer({.size = sizeof(bench::kRayVertices),
            .usage = BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
        if (!vertex) { return makeError(vertex.error()); }
        vertex_ = std::move(*vertex);
        auto* mapped = vertex_->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, bench::kRayVertices.data(), sizeof(bench::kRayVertices));
        vertex_->flush();
        vertex_->unmap();
        geometry_ = {.vertexBuffer = vertex_.get(), .vertexStride = 12,
            .vertexCount = 3, .indexType = RayTracingIndexType::None, .primitiveCount = 1};
        auto bottomSizes = device_->queryRayTracingAccelerationStructureBuildSizes({.geometries = {&geometry_, 1}});
        if (!bottomSizes) { return makeError(bottomSizes.error()); }
        auto bottom = device_->createRayTracingAccelerationStructure({.size = bottomSizes->accelerationStructureSize});
        if (!bottom) { return makeError(bottom.error()); }
        bottom_ = std::move(*bottom);
        auto topSizes = device_->queryRayTracingAccelerationStructureBuildSizes({
            .type = RayTracingAccelerationStructureType::TopLevel, .instanceCount = 1});
        if (!topSizes) { return makeError(topSizes.error()); }
        topSize_ = topSizes->accelerationStructureSize;
        const RayTracingInstanceDesc instance{.bottomLevel = bottom_.get(), .customIndex = 37, .mask = 1};
        auto instances = device_->createRayTracingInstanceBuffer({&instance, 1});
        if (!instances) { return makeError(instances.error()); }
        instances_ = std::move(*instances);
        auto properties = device_->queryRayTracingAccelerationStructureProperties();
        if (!properties) { return makeError(properties.error()); }
        auto scratch = device_->createBuffer({
            .size = std::max(bottomSizes->buildScratchSize, topSizes->buildScratchSize) + properties->scratchAlignment,
            .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
        if (!scratch) { return makeError(scratch.error()); }
        scratch_ = std::move(*scratch);
        return {};
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        // Replace the wrapper every frame. Old recorded/submitted consumers must
        // retain their native allocation, and the graph must publish the new AS.
        auto top = device_->createRayTracingAccelerationStructure({
            .type = RayTracingAccelerationStructureType::TopLevel, .size = topSize_});
        if (!top) { return makeError(top.error()); }
        top_ = std::move(*top);
        Result<> built;
        if (properties().value("fork", false)) {
            built = context.parallelCompute([&](CommandBuffer& commands) { return build(commands); },
                [&](CommandBuffer& commands) -> Result<> {
                    if (&context.commandBuffer() != &commands || context.supportsParallelCompute()) {
                        return makeError(Error::InvalidArgument);
                    }
                    const std::array stages{RenderGraphStage{.name = "Graphics branch context", .uses = {},
                        .record = [&](CommandBuffer& active) -> Result<> {
                            if (&active != &commands) { return makeError(Error::InvalidArgument); }
                            auto scope = context.profileScope("Graphics branch profile");
                            return {};
                        }}};
                    return context.executeStages(stages);
                });
        } else {
            built = build(context.commandBuffer());
        }
        if (!built) { return built; }
        return context.publishAccelerationStructure("structure", top_.get());
    }
private:
    Result<> build(CommandBuffer& commands)
    {
        using namespace detail;
        const std::array resources{
            GraphAccessResource{.type = RenderGraphResourceType::Buffer},
            GraphAccessResource{.type = RenderGraphResourceType::Buffer},
            GraphAccessResource{.type = RenderGraphResourceType::Buffer},
            GraphAccessResource{.type = RenderGraphResourceType::AccelerationStructure},
            GraphAccessResource{.type = RenderGraphResourceType::AccelerationStructure},
        };
        auto vertex = vertex_->slice({0, vertex_->desc().size});
        auto instances = instances_->slice({0, instances_->desc().size});
        auto scratch = scratch_->slice({0, scratch_->desc().size});
        if (!vertex || !instances || !scratch) { return makeError(Error::Failure); }
        const std::array bindings{
            GraphAccessBinding{.buffer = *vertex}, GraphAccessBinding{.buffer = *instances},
            GraphAccessBinding{.buffer = *scratch}, GraphAccessBinding{.accelerationStructure = bottom_.get()},
            GraphAccessBinding{.accelerationStructure = top_.get()},
        };
        using Access = RenderGraphResourceAccess;
        const std::array passes{
            GraphAccessPass{.uses = {
                declaredGraphAccess(0, Access::BufferAccelerationStructureBuildRead, RenderGraphPassKind::Compute),
                declaredGraphAccess(2, Access::BufferAccelerationStructureScratchReadWrite, RenderGraphPassKind::Compute),
                declaredGraphAccess(3, Access::AccelerationStructureBuildWrite, RenderGraphPassKind::Compute)}},
            GraphAccessPass{.uses = {
                declaredGraphAccess(1, Access::BufferAccelerationStructureBuildRead, RenderGraphPassKind::Compute),
                declaredGraphAccess(2, Access::BufferAccelerationStructureScratchReadWrite, RenderGraphPassKind::Compute),
                declaredGraphAccess(3, Access::AccelerationStructureBuildRead, RenderGraphPassKind::Compute),
                declaredGraphAccess(4, Access::AccelerationStructureBuildWrite, RenderGraphPassKind::Compute)}},
        };
        auto plan = buildGraphAccessPlan(resources, passes);
        if (!plan) { return makeError(plan.error()); }
        auto result = recordGraphAccessBarriers(commands, plan->passes[0], bindings);
        if (!result) { return result; }
        result = commands.buildRayTracingAccelerationStructure({.destination = bottom_.get(),
            .geometries = {&geometry_, 1}, .scratchBuffer = scratch_.get(), .graphManagedSynchronization = true});
        if (!result) { return result; }
        result = recordGraphAccessBarriers(commands, plan->passes[1], bindings);
        if (!result) { return result; }
        return commands.buildRayTracingAccelerationStructure({.destination = top_.get(),
            .instanceBuffer = instances_.get(), .instanceCount = 1, .scratchBuffer = scratch_.get(), .graphManagedSynchronization = true});
    }
    Device* device_ = nullptr;
    uint64_t topSize_ = 0;
    std::unique_ptr<Buffer> vertex_, instances_, scratch_;
    std::unique_ptr<RayTracingAccelerationStructure> bottom_, top_;
    RayTracingTriangleGeometryDesc geometry_;
};

class GraphAccelerationStructureReadPass final : public ComputePass {
public:
    QueueType queueType() const override { return QueueType::Graphics; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    CpuRecordingPolicy cpuRecordingPolicy() const override { return CpuRecordingPolicy::ParallelJoined; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("structure");
        reflection.addBufferOutput("observations").buffer(sizeof(bench::RayObservations), sizeof(bench::RayObservation))
            .storageWrite().hostReadback();
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const char* capabilities[]{"spvRayQueryKHR"};
        auto shader = compileSlangShaderToSpirv({.moduleName = "UnifiedTopLevelProbe",
            .entryPointName = "unifiedTopLevelMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .capabilities = capabilities, .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log);
        if (!shader) { return makeError(shader.error()); }
        const ComputeProgramBindingDesc bindings[]{{0, ComputeResourceBindingKind::AccelerationStructure}, {1}};
        return program_.initialize(*context.device, {.spirv = shader->spirv, .bindings = bindings}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto* accelerationStructure = context.inputAccelerationStructure("structure");
        auto* output = context.outputBuffer("observations").buffer();
        if (!accelerationStructure || !output) { return makeError(Error::InvalidArgument); }
        const ComputeDispatchBinding bindings[]{{.binding = 0, .accelerationStructure = accelerationStructure},
            {.binding = 1, .buffer = output}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings});
    }
private:
    ComputeProgram program_;
};

class RenderGraphAccelerationStructureTest final : public RhiTest {
public:
    RenderGraphAccelerationStructureTest() { type = RhiTestType::Rendering; name = "render_graph_acceleration_structure_publication"; }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"graph.accelerationStructure.publication.rayQuery", "graph.accelerationStructure.cancel.lifetime",
            "graph.accelerationStructure.queueDependencies", "graph.parallel.contextStages"}, bench::Layer::RenderGraph,
            "ray-query", "sync", {"graph-as-rays.bin"});
        result.requirements.capabilities.push_back(bench::Capability::RayQuery);
        return result;
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::atomic_uint validationErrors = 0;
        auto created = bench::createTestDevice(context, {.applicationName = "RenderGraph AS publication test",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}, .enableAsyncCompute = true});
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("ray query/descriptor heap unavailable"); }
        GRAPH_AS_REQUIRE(created);
        auto& device = **created;
        auto* graphics = device.getQueue(QueueType::Graphics);
        auto* compute = device.getQueue(QueueType::Compute);
        GRAPH_AS_CHECK(graphics);
        if (!compute) { compute = graphics; }
        static const bool registeredBuild = registerRenderGraphPassType("GraphASBuildTest", "AS publication test builder",
            [] { return std::make_unique<GraphAccelerationStructureBuildPass>(); });
        static const bool registeredRead = registerRenderGraphPassType("GraphASReadTest", "AS publication ray-query reader",
            [] { return std::make_unique<GraphAccelerationStructureReadPass>(); });
        (void)registeredBuild;
        (void)registeredRead;
        const auto makeGraph = [](bool fork) {
            RenderGraph graph;
            graph.addNode("GraphASBuildTest", "Build", {{"fork", fork}});
            graph.addNode("GraphASReadTest", "Read");
            graph.addEdge("Build.structure", "Read.structure");
            graph.markOutput("Read.observations");
            return graph;
        };
        for (const bool fork : {false, true}) {
            auto graph = makeGraph(fork);
            RenderGraphExecutor executor;
            executor.setExecutionCaptureEnabled(true);
            GRAPH_AS_REQUIRE(executor.compile(device, graph, 1, 1, log));
            for (const bool aliased : {false, true}) {
                for (const auto mode : {FrameSubmissionMode::Joined, FrameSubmissionMode::Pipelined}) {
                    for (uint32_t frame = 0; frame < 2; ++frame) {
                        GRAPH_AS_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = aliased ? graphics : compute,
                            .recordingWorkerLimit = 4, .recordingBatchWorkload = 1, .submissionMode = mode}));
                        GRAPH_AS_REQUIRE(executor.waitForSubmittedWork(5'000'000'000ull));
                        const auto* output = executor.outputResource("Read.observations");
                        const auto* published = executor.outputResource("Build.structure");
                        GRAPH_AS_CHECK(output && output->buffer && published && published->accelerationStructure);
                        auto* data = output->buffer->map();
                        GRAPH_AS_CHECK(data);
                        output->buffer->invalidate();
                        bench::RayObservations actual;
                        std::memcpy(actual.data(), data, sizeof(actual));
                        output->buffer->unmap();
                        (void)bench::rayOracle(actual);
                        bench::readbackEvidence(context, "graph-as-rays.bin", std::span<const bench::RayObservation>(actual));
                        const auto snapshot = executor.executionSnapshot();
                        GRAPH_AS_CHECK(snapshot && snapshot->success && snapshot->passes.size() == 2);
                        GRAPH_AS_CHECK(snapshot->passes[1].predecessors == std::vector<uint32_t>{snapshot->passes[0].id});
                        const auto asResource = std::find_if(snapshot->resources.begin(), snapshot->resources.end(),
                            [](const auto& resource) { return resource.type == RenderGraphResourceType::AccelerationStructure; });
                        GRAPH_AS_CHECK(asResource != snapshot->resources.end() && asResource->memory.allocationId != 0);
                        GRAPH_AS_CHECK(asResource->aliases.size() == 2 && asResource->accelerationStructureDesc.size != 0);
                        GRAPH_AS_CHECK(snapshot->passes[0].uses.front().resourceId == asResource->id);
                        if (fork && !aliased && !graphics->sameQueue(*compute)) {
                            GRAPH_AS_CHECK(executor.executionStats().asyncComputeBranches == 1);
                        }
                    }
                }
            }
        }
        // Destroy public graph wrappers after an external recording, before its
        // commands are ever submitted. Native retention lasts until cancellation.
        auto pool = device.createCommandPool(*graphics);
        GRAPH_AS_REQUIRE(pool);
        auto commands = (*pool)->createCommandBuffer();
        GRAPH_AS_REQUIRE(commands);
        GRAPH_AS_REQUIRE((*commands)->begin());
        std::weak_ptr<void> allocation;
        {
            auto graph = makeGraph(false);
            RenderGraphExecutor executor;
            GRAPH_AS_REQUIRE(executor.compile(device, graph, 1, 1, log));
            GRAPH_AS_REQUIRE(executor.execute(**commands));
            const auto* published = executor.outputResource("Build.structure");
            GRAPH_AS_CHECK(published && published->accelerationStructure);
            allocation = published->accelerationStructure->retainAllocation();
            GRAPH_AS_REQUIRE((*commands)->end());
        }
        GRAPH_AS_CHECK(!allocation.expired());
        commands->reset();
        pool->reset();
        GRAPH_AS_CHECK(allocation.expired());
        GRAPH_AS_CHECK(validationErrors.load() == 0);
        return RhiTestResult::pass("16 analytic ray-query frames, queue aliases, AS replacement, fork stage context and cancelled lifetime");
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphAccelerationStructureTest);

#undef GRAPH_AS_CHECK
#undef GRAPH_AS_REQUIRE

} // namespace
} // namespace metallic::tests
