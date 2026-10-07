#include "TestResourceLayouts.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"
#include "harness/RayQueryFixture.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

using namespace render;

#define SCENE_AS_REQUIRE(expression) do { \
    const auto& checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + \
        resultToString(Result<>{std::unexpected(checked.error())}) + " " + log); } \
} while (false)
#define SCENE_AS_CHECK(expression) do { if (!(expression)) { return RHITestResult::fail(#expression); } } while (false)

class SceneAccelerationStructureProbePass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::StandardAccelerationStructure};
    }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"accelerationStructure"}};
    }
    QueueType queueType() const override { return QueueType::Graphics; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("accelerationStructure").accelerationStructureRead();
        reflection.addBufferOutput("observations").buffer(sizeof(bench::RayObservations), sizeof(bench::RayObservation))
            .storageWrite().hostReadback();
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (!context.preparedScene || !context.preparedScene->snapshot ||
            !context.preparedScene->snapshot->pathTraceResources) {
            return makeError(Error::InvalidArgument);
        }
        sceneResources_ = context.preparedScene->snapshot->pathTraceResources;
        const char* capabilities[]{"spvRayQueryKHR"};
        auto shader = compileSlangShaderToSpirv({.moduleName = "UnifiedTopLevelProbe",
            .entryPointName = "unifiedTopLevelMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .capabilities = capabilities, .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log);
        if (!shader) { return makeError(shader.error()); }
        const ComputeResourceBindingDesc bindings[]{{0, ComputeResourceBindingKind::AccelerationStructure}, {1}};
        return program_.initialize(*context.device, {.spirv = shader->spirv, .bindings = bindings, .resourceParameters = metallic::tests::kUnifiedTopLevelProbeLayout}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto* structure = context.inputAccelerationStructure("accelerationStructure");
        // The connected producer's mode/backend/preference must carry through
        // resource preparation, avoiding an unused second RTAS on the consumer.
        if (!sceneResources_ || sceneResources_->accelerationStructure().accelerationStructure() != structure) {
            return makeError(Error::InvalidArgument);
        }
        const ComputeDispatchBinding bindings[]{{.binding = 0,
            .accelerationStructure = structure},
            {.binding = 1, .buffer = context.outputBuffer("observations").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings});
    }
private:
    ComputeProgram program_;
    std::shared_ptr<ScenePathTraceResources> sceneResources_;
};

class SceneAccelerationStructurePassTest : public RHITest {
public:
    explicit SceneAccelerationStructurePassTest(bool partitioned = false) : partitioned_(partitioned)
    {
        type = RHITestType::Rendering;
        name = partitioned ? "scene_acceleration_structure_pass_partitioned" : "scene_acceleration_structure_pass_standard";
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        std::atomic_uint validationErrors = 0;
        auto created = bench::createTestDevice(context, {.applicationName = "Scene AS graph refit test",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enablePartitionedAccelerationStructure = partitioned_,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}, .enableAsyncCompute = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("scene RTAS ray-query capabilities unavailable"); }
        SCENE_AS_REQUIRE(created);
        auto& device = **created;
        if (partitioned_ && !device.capabilities().partitionedAccelerationStructure) {
            return RHITestResult::skip("PTLAS unavailable");
        }
        auto* graphics = device.getQueue(QueueType::Graphics);
        auto* compute = device.getQueue(QueueType::Compute);
        SCENE_AS_CHECK(graphics);
        if (!compute) { compute = graphics; }
        const auto directory = context.outputDirectory / (partitioned_ ? "scene-as-partitioned" : "scene-as-standard");
        std::filesystem::create_directories(directory);
        const auto path = directory / "scene.gltf";
        {
            std::ofstream binary(directory / "triangle.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(bench::kRayVertices.data()), sizeof(bench::kRayVertices));
            std::ofstream gltf(path);
            gltf << R"json({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
                "nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
                "buffers":[{"uri":"triangle.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,2],"max":[1,1,2]}]})json";
        }
        scene::Scene scene;
        SCENE_AS_CHECK(scene.load(path));
        const auto backend = partitioned_ ? RayTracingTopLevelBackend::Partitioned : RayTracingTopLevelBackend::Standard;
        // A cancelled recording must not acknowledge the geometry revision, and
        // preparing a later update must not overwrite the earlier upload.
        {
            SceneAccelerationStructureBuilder builder;
            SCENE_AS_REQUIRE(builder.build(device, *graphics, scene, log, {.topLevelBackend = backend}));
            auto transform = scene.nodes()[0].localMatrix;
            transform.a03 = 0.1f;
            SCENE_AS_CHECK(scene.setNodeLocalMatrix(0, transform));
            SCENE_AS_REQUIRE(builder.prepareInstanceTransformUpdate(device, scene, log));
            SCENE_AS_CHECK(builder.hasPendingInstanceTransformUpdate());
            const auto oldUpload = builder.instanceTransformUpdateBuffer()->retainAllocation();
            auto pool = device.createCommandPool(*compute);
            SCENE_AS_REQUIRE(pool);
            auto commands = (*pool)->createCommandBuffer();
            SCENE_AS_REQUIRE(commands);
            SCENE_AS_REQUIRE((*commands)->begin());
            SCENE_AS_REQUIRE(builder.recordInstanceTransformUpdate(**commands, log, false));
            SCENE_AS_REQUIRE((*commands)->end());
            commands->reset();
            pool->reset();
            SCENE_AS_REQUIRE(builder.prepareInstanceTransformUpdate(device, scene, log));
            SCENE_AS_CHECK(builder.hasPendingInstanceTransformUpdate());
            transform.a03 = 0.15f;
            SCENE_AS_CHECK(scene.setNodeLocalMatrix(0, transform));
            SCENE_AS_REQUIRE(builder.prepareInstanceTransformUpdate(device, scene, log));
            SCENE_AS_CHECK(oldUpload != builder.instanceTransformUpdateBuffer()->retainAllocation());
        }
        static const bool registered = registerRenderGraphPassType("SceneASProbeTest", "Scene AS analytic ray probe",
            [] { return std::make_unique<SceneAccelerationStructureProbePass>(); });
        (void)registered;
        registerBuiltInRenderGraphPasses();
        for (const bool asyncComputePreferred : {true, false}) {
            auto initialTransform = scene.nodes()[0].localMatrix;
            initialTransform.a03 = 0;
            if (scene.nodes()[0].localMatrix.a03 != initialTransform.a03) {
                SCENE_AS_CHECK(scene.setNodeLocalMatrix(0, initialTransform));
            }
            RenderGraph graph;
            RenderGraphProperties properties{{"path", path.string()},
                {"topLevelBackend", partitioned_ ? "partitioned" : "standard"}};
            if (!asyncComputePreferred) { properties["AsyncComputePreferred"] = false; }
            graph.addNode("SceneAccelerationStructurePass", "Build", properties);
            graph.addNode("SceneASProbeTest", "Read");
            SCENE_AS_CHECK(graph.addEdge("Build.accelerationStructure", "Read.accelerationStructure"));
            SCENE_AS_CHECK(graph.markOutput("Read.observations"));
            RenderGraphExecutor executor;
            executor.bindRuntimeScene(&scene);
            // The editor prepares before compilation. Preparation must use the
            // graph owner's cache key rather than create a legacy AS bundle
            // that remains resident alongside another bundle during compile.
            SCENE_AS_REQUIRE(executor.beginSceneResourcePreparation(device, {{"path", path.string()}}, scene, log, &graph));
            bool prepared = false;
            scene::SceneLoadProgress progress;
            const auto prepareDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!prepared && std::chrono::steady_clock::now() < prepareDeadline) {
                auto completion = executor.pumpSceneResourcePreparation(scene, 10.0, progress, log);
                SCENE_AS_REQUIRE(completion);
                prepared = *completion;
            }
            SCENE_AS_CHECK(prepared);
            const auto preparedMemory = device.memoryBudget().domains[size_t(MemoryBudgetDomain::RayTracing)];
            SCENE_AS_CHECK(preparedMemory.allocationCount >= 2 && preparedMemory.allocationBytes != 0);
            executor.acceptSceneResourcePreparation();
            SCENE_AS_REQUIRE(executor.compile(device, graph, 1, 1, log));
            const auto compiledMemory = device.memoryBudget().domains[size_t(MemoryBudgetDomain::RayTracing)];
            SCENE_AS_CHECK(compiledMemory.allocationCount == preparedMemory.allocationCount);
            SCENE_AS_CHECK(compiledMemory.allocationBytes == preparedMemory.allocationBytes);
            for (const float translation : {0.0f, 0.4f, 2.0f, 0.0f}) {
                auto transform = scene.nodes()[0].localMatrix;
                transform.a03 = translation;
                if (scene.nodes()[0].localMatrix.a03 != translation) {
                    SCENE_AS_CHECK(scene.setNodeLocalMatrix(0, transform));
                }
                // Drop a recorded refit before acceptance, then rerun it through
                // graph submission. The consumer must see the new transform.
                if (translation == 0.4f) {
                    auto pool = device.createCommandPool(*graphics);
                    SCENE_AS_REQUIRE(pool);
                    auto commands = (*pool)->createCommandBuffer();
                    SCENE_AS_REQUIRE(commands);
                    SCENE_AS_REQUIRE((*commands)->begin());
                    SCENE_AS_REQUIRE(executor.execute(**commands));
                    SCENE_AS_REQUIRE((*commands)->end());
                    commands->reset();
                    pool->reset();
                }
                SCENE_AS_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute,
                    .recordingWorkerLimit = 2, .recordingBatchWorkload = 1,
                    .submissionMode = FrameSubmissionMode::Pipelined}));
                SCENE_AS_REQUIRE(executor.waitForSubmittedWork(5'000'000'000ull));
                const auto* output = executor.outputResource("Read.observations");
                SCENE_AS_CHECK(output && output->buffer);
                auto* mapped = output->buffer->map();
                SCENE_AS_CHECK(mapped);
                output->buffer->invalidate();
                bench::RayObservations actual;
                std::memcpy(actual.data(), mapped, sizeof(actual));
                output->buffer->unmap();
                const auto rays = bench::rayFixture().at("rays");
                for (size_t i = 0; i < actual.size(); ++i) {
                    const float x = rays[i][0].get<float>() - translation, y = rays[i][1].get<float>();
                    const bool hit = x > 0 && y > 0 && x + y < 1 && i != 4;
                    SCENE_AS_CHECK(actual[i].hit == uint32_t(hit));
                    SCENE_AS_CHECK(actual[i].instance == (hit ? 0u : UINT32_MAX));
                    if (hit) {
                        SCENE_AS_CHECK(std::abs(actual[i].u - x) < 0.00001f && std::abs(actual[i].v - y) < 0.00001f);
                        SCENE_AS_CHECK(std::abs(actual[i].distance - (i == 5 ? 3.0f : 2.0f)) < 0.00001f);
                    }
                }
                const auto& stats = executor.executionStats();
                const auto built = std::find_if(stats.nodes.begin(), stats.nodes.end(),
                    [](const auto& node) { return node.name == "Build"; });
                SCENE_AS_CHECK(built != stats.nodes.end());
                const auto expectedQueue = asyncComputePreferred ? compute->type() : graphics->type();
                SCENE_AS_CHECK(built->queue == expectedQueue);
            }
        }
        // Two views can share one subsystem host and its scene-resource manager.
        // Their mutable top levels/scratch must still belong to their graph so
        // one view's refit cannot change the other view's queued ray queries.
        {
            RenderSubsystemHost host;
            RenderWorld world;
            RenderGraphExecutor first(host, world), second(host, world);
            first.bindRuntimeScene(&scene);
            second.bindRuntimeScene(&scene);
            RenderGraph graph;
            graph.addNode("SceneAccelerationStructurePass", "Build", {{"path", path.string()},
                {"topLevelBackend", partitioned_ ? "partitioned" : "standard"}});
            graph.addNode("SceneASProbeTest", "Read");
            SCENE_AS_CHECK(graph.addEdge("Build.accelerationStructure", "Read.accelerationStructure"));
            SCENE_AS_CHECK(graph.markOutput("Read.observations"));
            SCENE_AS_REQUIRE(first.compile(device, graph, 1, 1, log));
            SCENE_AS_REQUIRE(second.compile(device, graph, 1, 1, log));
            const std::array<std::array<float, 2>, 3> translations{{{{0.0f, 2.0f}}, {{2.0f, 0.0f}}, {{0.0f, 0.4f}}}};
            for (const auto& pair : translations) {
                const std::array executors{&first, &second};
                for (size_t view = 0; view < executors.size(); ++view) {
                    auto transform = scene.nodes()[0].localMatrix;
                    transform.a03 = pair[view];
                    if (scene.nodes()[0].localMatrix.a03 != pair[view]) {
                        SCENE_AS_CHECK(scene.setNodeLocalMatrix(0, transform));
                    }
                    SCENE_AS_REQUIRE(executors[view]->execute({.graphicsQueue = graphics, .computeQueue = compute,
                        .recordingWorkerLimit = 2, .recordingBatchWorkload = 1,
                        .submissionMode = FrameSubmissionMode::Pipelined}));
                }
                const auto* firstStructure = first.outputResource("Build.accelerationStructure");
                const auto* secondStructure = second.outputResource("Build.accelerationStructure");
                SCENE_AS_CHECK(firstStructure && firstStructure->accelerationStructure &&
                    secondStructure && secondStructure->accelerationStructure);
                SCENE_AS_CHECK(firstStructure->accelerationStructure->memoryInfo().allocationId !=
                    secondStructure->accelerationStructure->memoryInfo().allocationId);
                for (size_t view = 0; view < executors.size(); ++view) {
                    SCENE_AS_REQUIRE(executors[view]->waitForSubmittedWork(5'000'000'000ull));
                    const auto* output = executors[view]->outputResource("Read.observations");
                    SCENE_AS_CHECK(output && output->buffer);
                    auto* mapped = output->buffer->map();
                    SCENE_AS_CHECK(mapped);
                    output->buffer->invalidate();
                    bench::RayObservation observation;
                    std::memcpy(&observation, mapped, sizeof(observation));
                    output->buffer->unmap();
                    SCENE_AS_CHECK(observation.hit == uint32_t(pair[view] == 0.0f));
                }
            }
            // Independent producers in one graph have distinct logical AS
            // resources, so they must also own distinct mutable native storage.
            graph.addNode("SceneAccelerationStructurePass", "BuildOther", {{"path", path.string()},
                {"topLevelBackend", partitioned_ ? "partitioned" : "standard"}});
            graph.addNode("SceneASProbeTest", "ReadOther");
            SCENE_AS_CHECK(graph.addEdge("BuildOther.accelerationStructure", "ReadOther.accelerationStructure"));
            SCENE_AS_CHECK(graph.markOutput("ReadOther.observations"));
            RenderGraphExecutor multipleProducers(host, world);
            multipleProducers.bindRuntimeScene(&scene);
            SCENE_AS_REQUIRE(multipleProducers.compile(device, graph, 1, 1, log));
            SCENE_AS_REQUIRE(multipleProducers.execute({.graphicsQueue = graphics, .computeQueue = compute,
                .recordingWorkerLimit = 2, .recordingBatchWorkload = 1,
                .submissionMode = FrameSubmissionMode::Pipelined}));
            SCENE_AS_REQUIRE(multipleProducers.waitForSubmittedWork(5'000'000'000ull));
            const auto* firstProducer = multipleProducers.outputResource("Build.accelerationStructure");
            const auto* otherProducer = multipleProducers.outputResource("BuildOther.accelerationStructure");
            SCENE_AS_CHECK(firstProducer && firstProducer->accelerationStructure &&
                otherProducer && otherProducer->accelerationStructure);
            SCENE_AS_CHECK(firstProducer->accelerationStructure->memoryInfo().allocationId !=
                otherProducer->accelerationStructure->memoryInfo().allocationId);
        }
        SCENE_AS_CHECK(validationErrors.load() == 0);
        return RHITestResult::pass("15 analytic scene frames verify TLAS/PTLAS refits, async preference, cancellation, connected scene sharing and AS producer isolation");
    }
private:
    bool partitioned_;
};

class ScenePartitionedAccelerationStructurePassTest final : public SceneAccelerationStructurePassTest {
public:
    ScenePartitionedAccelerationStructurePassTest() : SceneAccelerationStructurePassTest(true) {}
};

METALLIC_REGISTER_RHI_TEST(SceneAccelerationStructurePassTest);
METALLIC_REGISTER_RHI_TEST(ScenePartitionedAccelerationStructurePassTest);

#undef SCENE_AS_CHECK
#undef SCENE_AS_REQUIRE

} // namespace
} // namespace metallic::tests
