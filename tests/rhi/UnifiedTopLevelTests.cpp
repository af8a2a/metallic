#include "TestResourceLayouts.h"
#include "RHITest.h"
#include "harness/RayQueryFixture.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

#define TLAS_REQUIRE(expression) do { \
    const auto& checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + \
        toString(render::Result<>{std::unexpected(checked.error())}) + " " + log); } \
} while (false)
#define TLAS_CHECK(expression) do { if (!(expression)) { return RHITestResult::fail(#expression); } } while (false)

class UnifiedTopLevelTest : public RHITest {
public:
    UnifiedTopLevelTest(bool partitioned = false, bool native = false)
        : partitioned_(partitioned), native_(native)
    {
        type = RHITestType::Rendering;
        name = partitioned ? (native ? "unified_top_level_partitioned_native" : "unified_top_level_partitioned")
            : (native ? "unified_top_level_standard_native" : "unified_top_level_standard");
    }

    std::optional<bench::Metadata> metadata() const override
    {
        if (!partitioned_) { return std::nullopt; }
        return bench::comparisonMetadata({"rayQuery.analytic.hit.miss.mask.distance.barycentric.frontFace", "topLevel.typedBackend.contract.lifetime", "topLevel.transform.standardRefit.partitionedRebuild"},
            bench::Layer::RHI, "ray-query", {"ray-query-ptlas", "partitionedAS", bench::Capability::PartitionedAS, 0.00001}, native_);
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const bool usePartitioned = context.deviceDesc ? context.deviceDesc->enablePartitionedAccelerationStructure : partitioned_;
        std::string log;
        std::atomic_uint validationErrors = 0;
        auto created = bench::createTestDevice(context, {.applicationName = "Unified TLAS test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enablePartitionedAccelerationStructure = usePartitioned,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("ray query/descriptor heap unavailable"); }
        TLAS_REQUIRE(created);
        auto& device = **created;
        TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure(RayTracingAccelerationStructureDesc{}), Error::InvalidArgument));
        TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure(PartitionedAccelerationStructureDesc{}), Error::InvalidArgument));
        if (usePartitioned && !device.capabilities().partitionedAccelerationStructure) {
            return RHITestResult::skip("PTLAS unavailable");
        }
        auto& queue = *device.getQueue(QueueType::Graphics);
        const auto& vertices = bench::kRayVertices;
        auto vertex = device.createBuffer({.size = sizeof(vertices) + 32,
            .usage = BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload});
        TLAS_REQUIRE(vertex);
        void* mapped = (*vertex)->map();
        TLAS_CHECK(mapped);
        std::memcpy(static_cast<uint8_t*>(mapped) + 32, vertices.data(), sizeof(vertices));
        (*vertex)->flush();
        (*vertex)->unmap();
        auto vertexSlice = (*vertex)->slice({32, sizeof(vertices)});
        TLAS_REQUIRE(vertexSlice);
        const RayTracingTriangleGeometryDesc geometry{.vertexBuffer = *vertexSlice, .vertexStride = 12,
            .vertexCount = 3, .indexType = RayTracingIndexType::None, .primitiveCount = 1};
        auto blasSizes = device.queryRayTracingAccelerationStructureBuildSizes({.geometries = {&geometry, 1}});
        TLAS_REQUIRE(blasSizes);
        auto blas = device.createRayTracingAccelerationStructure({.size = blasSizes->accelerationStructureSize});
        TLAS_REQUIRE(blas);
        const auto flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace |
            RayTracingAccelerationStructureBuildFlags::AllowUpdate | RayTracingAccelerationStructureBuildFlags::AllowCompaction;
        auto standardSizes = device.queryRayTracingAccelerationStructureBuildSizes({
            .type = RayTracingAccelerationStructureType::TopLevel, .flags = flags, .instanceCount = 1});
        TLAS_REQUIRE(standardSizes);
        auto standard = device.createRayTracingAccelerationStructure({.type = RayTracingAccelerationStructureType::TopLevel,
            .buildFlags = flags, .size = standardSizes->accelerationStructureSize});
        TLAS_REQUIRE(standard);
        TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure({.type = RayTracingAccelerationStructureType::TopLevel,
            .size = standardSizes->accelerationStructureSize, .topLevelBackend = RayTracingTopLevelBackend::Partitioned}), Error::InvalidArgument));
        const RayTracingInstanceDesc instance{.bottomLevel = blas->get(), .customIndex = 37, .mask = 1};
        auto instances = device.createRayTracingInstanceBuffer({&instance, 1});
        TLAS_REQUIRE(instances);
        auto instanceSlice = (*instances)->slice();
        TLAS_REQUIRE(instanceSlice);
        Result<BufferSlice> partitionedSlice = BufferSlice{};
        std::unique_ptr<RayTracingAccelerationStructure> partitioned;
        std::unique_ptr<Buffer> partitionedInstances;
        uint64_t scratchSize = std::max({blasSizes->buildScratchSize, standardSizes->buildScratchSize, standardSizes->updateScratchSize});
        if (usePartitioned) {
            const PartitionedAccelerationStructureBuildInputs inputs{.instanceCount = 1,
                .partitionCount = 1, .maxInstancePerPartitionCount = 1};
            auto sizes = device.queryPartitionedAccelerationStructureBuildSizes(inputs);
            TLAS_REQUIRE(sizes);
            auto invalidSizes = *sizes;
            --invalidSizes.accelerationStructureSize;
            TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure(PartitionedAccelerationStructureDesc{
                .inputs = inputs, .sizes = invalidSizes}), Error::InvalidArgument));
            auto resource = device.createRayTracingAccelerationStructure(PartitionedAccelerationStructureDesc{.inputs = inputs, .sizes = *sizes});
            TLAS_REQUIRE(resource);
            partitioned = std::move(*resource);
            const PartitionedAccelerationStructureInstanceDesc partitionedInstance{
                .bottomLevel = blas->get(), .customIndex = 37, .mask = 1};
            auto encoded = device.createPartitionedAccelerationStructureInstanceBuffer({&partitionedInstance, 1});
            TLAS_REQUIRE(encoded);
            partitionedInstances = std::move(*encoded);
            partitionedSlice = partitionedInstances->slice();
            TLAS_REQUIRE(partitionedSlice);
            scratchSize = std::max(scratchSize, sizes->buildScratchSize);
        }
        auto properties = device.queryRayTracingAccelerationStructureProperties();
        TLAS_REQUIRE(properties);
        auto scratch = device.createBuffer({.size = scratchSize + std::max<uint64_t>(256, properties->scratchAlignment),
            .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress});
        TLAS_REQUIRE(scratch);
        auto scratchSlice = (*scratch)->slice();
        TLAS_REQUIRE(scratchSlice);
        const char* capabilities[] = {"spvRayQueryKHR"};
        ShaderCompileResult shader;
        const auto compiled = compileSlangShaderToSpirv({
            .moduleName = "UnifiedTopLevelProbe",
            .entryPointName = "unifiedTopLevelMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .capabilities = {capabilities, 1},
            .descriptorHeapMode = native_ ? SlangDescriptorHeapMode::Native : SlangDescriptorHeapMode::Mapped,
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        log = shader.diagnostics;
        TLAS_REQUIRE(compiled);
        const ComputeProgramBindingDesc layout[] = {{0, ComputeResourceBindingKind::AccelerationStructure}, {1}};
        ComputeProgram program;
        const auto initialized = program.initialize(device, {
            .spirv = shader.spirv,
            .bindings = {layout, 2},
            .resourceParameters = metallic::tests::kUnifiedTopLevelProbeLayout,
        }, log);
        if (native_ && hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("native descriptor heap unavailable"); }
        TLAS_REQUIRE(initialized);
        using Probe = bench::RayObservations;
        auto output = device.createBuffer({.size = sizeof(Probe), .structureStride = sizeof(bench::RayObservation), .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostReadback});
        TLAS_REQUIRE(output);
        ResourceRegistry registry;
        TLAS_REQUIRE(registry.initialize(device, {.maxBuffers = 1}));
        auto heap = device.createBindlessHeap({.maxBuffers = 1});
        TLAS_REQUIRE(heap);
        auto handle = (*heap)->allocate(metallic::render::BindlessHandleKind::AccelerationStructure);
        TLAS_REQUIRE(handle);
        TLAS_CHECK(handle->kind == BindlessHandleKind::AccelerationStructure);
        ResourceLease lease;
        TLAS_CHECK(hasError(registry.accelerationStructure(**blas).transform([&](auto value) { lease = std::move(value); }), Error::InvalidArgument));
        TLAS_CHECK(hasError((*heap)->writeAccelerationStructure(*handle, **blas), Error::InvalidArgument));
        auto queries = device.createRayTracingAccelerationStructureCompactionQueryPool({.queryCount = 1});
        TLAS_REQUIRE(queries);
        QueueSubmissionTracker tracker;
        TLAS_REQUIRE(tracker.initialize(device, queue));
        auto pool = device.createCommandPool(queue);
        TLAS_REQUIRE(pool);
        auto commands = (*pool)->createCommandBuffer();
        TLAS_REQUIRE(commands);
        RenderFrameContext frame;
        struct Drain {
            RenderFrameContext& frame;
            CommandPool& pool;
            ~Drain()
            {
                if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                (void)pool.reset();
                (void)frame.reset();
            }
        } drain{frame, **pool};
        TLAS_REQUIRE(frame.begin(0));
        TLAS_REQUIRE((*commands)->begin(frame.submissionContext()));
        auto shortVertices = vertexSlice->subslice({0, sizeof(vertices) - 1});
        auto shortInstances = instanceSlice->subslice({0, sizeof(RayTracingGPUInstance) - 1});
        auto shortScratch = scratchSlice->subslice({0, 1});
        TLAS_REQUIRE(shortVertices); TLAS_REQUIRE(shortInstances); TLAS_REQUIRE(shortScratch);
        auto invalidGeometry = geometry;
        invalidGeometry.vertexBuffer = *shortVertices;
        TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = blas->get(),
            .geometries = {&invalidGeometry, 1}, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
        TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = blas->get(),
            .geometries = {&geometry, 1}, .scratchBuffer = *shortScratch}), Error::InvalidArgument));
        TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = standard->get(),
            .instanceBuffer = *shortInstances, .instanceCount = 1, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
        TLAS_REQUIRE((*commands)->buildRayTracingAccelerationStructure({
            .destination = blas->get(),
            .geometries = {&geometry, 1},
            .scratchBuffer = *scratchSlice,
        }));
        TLAS_REQUIRE((*commands)->buildRayTracingAccelerationStructure({.destination = standard->get(),
            .instanceBuffer = *instanceSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}));
        if (usePartitioned) {
            TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = partitioned.get(),
                .instanceBuffer = *instanceSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = standard->get(), .source = partitioned.get(),
                .mode = RayTracingAccelerationStructureBuildMode::Update, .instanceBuffer = *instanceSlice,
                .instanceCount = 1, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->buildPartitionedAccelerationStructure({.destination = standard->get(),
                .instanceBuffer = *partitionedSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->compactRayTracingAccelerationStructure(*partitioned, **standard), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->compactRayTracingAccelerationStructure(**standard, *partitioned), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->writeRayTracingAccelerationStructureCompactedSize(**queries, 0, *partitioned), Error::InvalidArgument));
            auto shortPartitioned = partitionedSlice->subslice({0, partitionedSlice->size() - 1});
            TLAS_REQUIRE(shortPartitioned);
            TLAS_CHECK(hasError((*commands)->buildPartitionedAccelerationStructure({.destination = partitioned.get(),
                .instanceBuffer = *shortPartitioned, .instanceCount = 1, .scratchBuffer = *scratchSlice}), Error::InvalidArgument));
            TLAS_REQUIRE((*commands)->buildPartitionedAccelerationStructure({.destination = partitioned.get(),
                .instanceBuffer = *partitionedSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}));
        }
        // Reuse the same shader, program, layout, binding slot and heap handle for both backends.
        std::array<RayTracingAccelerationStructure*, 2> structures{standard->get(), partitioned.get()};
        for (uint32_t backend = context.evidence && usePartitioned ? 1u : 0u; backend < (usePartitioned ? 2u : 1u); ++backend) {
            auto& structure = *structures[backend];
            const auto address = structure.deviceAddress();
            TLAS_CHECK(structure.valid() && address && structure.desc().type == RayTracingAccelerationStructureType::TopLevel);
            TLAS_CHECK(structure.desc().topLevelBackend == (backend == 0 ? RayTracingTopLevelBackend::Standard : RayTracingTopLevelBackend::Partitioned));
            RayTracingAccelerationStructure moved = std::move(structure);
            TLAS_CHECK(!structure.valid() && structure.deviceAddress() == 0 && moved.deviceAddress() == address);
            structure = std::move(moved);
            TLAS_REQUIRE((*heap)->writeAccelerationStructure(*handle, structure));
            TLAS_REQUIRE(registry.accelerationStructure(structure).transform([&](auto value) { lease = std::move(value); }));
            TLAS_CHECK(lease.kind() == ShaderResourceKind::AccelerationStructure && lease.shaderValue() == address);
            bench::Json observations = bench::Json::array();
            for (uint32_t step = 0; step < 2; ++step) {
                const float translationX = step ? 0.4f : 0.0f;
                if (step) {
                    TLAS_REQUIRE(frame.begin(step));
                    TLAS_REQUIRE((*commands)->begin(frame.submissionContext()));
                    if (backend == 0) {
                        auto changed = instance; changed.transform[0][3] = translationX;
                        TLAS_REQUIRE(device.createRayTracingInstanceBuffer({&changed, 1}).transform([&](auto value) { *instances = std::move(value); }));
                        instanceSlice = (*instances)->slice(); TLAS_REQUIRE(instanceSlice);
                        TLAS_REQUIRE((*commands)->buildRayTracingAccelerationStructure({.destination = &structure, .source = &structure,
                            .mode = RayTracingAccelerationStructureBuildMode::Update, .instanceBuffer = *instanceSlice,
                            .instanceCount = 1, .scratchBuffer = *scratchSlice}));
                    } else {
                        PartitionedAccelerationStructureInstanceDesc changed{.bottomLevel = blas->get(), .customIndex = 37, .mask = 1};
                        changed.transform[0][3] = translationX;
                        TLAS_REQUIRE(device.createPartitionedAccelerationStructureInstanceBuffer({&changed, 1}).transform([&](auto value) { partitionedInstances = std::move(value); }));
                        partitionedSlice = partitionedInstances->slice(); TLAS_REQUIRE(partitionedSlice);
                        // The public PTLAS API currently rewrites all instances into the same allocation.
                        TLAS_REQUIRE((*commands)->buildPartitionedAccelerationStructure({.destination = &structure,
                            .instanceBuffer = *partitionedSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}));
                    }
                    TLAS_CHECK(structure.deviceAddress() == address);
                }
                const ComputeDispatchBinding bindings[] = {{.binding = 0, .accelerationStructure = &structure}, {.binding = 1, .buffer = output->get()}};
                TLAS_REQUIRE(program.dispatch({.commandBuffer = commands->get(), .bindings = {bindings, 2}}));
                TLAS_REQUIRE((*commands)->end());
                CommandBuffer* submitted[] = {commands->get()};
                TLAS_REQUIRE(tracker.submit({.commandBuffers = {submitted, 1}}, frame));
                TLAS_REQUIRE(frame.wait(10'000'000'000ull));
                Probe actual{};
                const void* readback = (*output)->map();
                TLAS_CHECK(readback);
                (*output)->invalidate();
                std::memcpy(actual.data(), readback, sizeof(actual));
                (*output)->unmap();
                bench::readbackEvidence(context, "readback.bin", std::span<const bench::RayObservation>(actual));
                observations.push_back(bench::rayOracle(actual, translationX));
                TLAS_REQUIRE((*pool)->reset());
                TLAS_REQUIRE(frame.reset());
            }
            auto fixture = bench::rayFixture(); fixture["native"] = native_; fixture["translationX"] = {0.0f, 0.4f};
            bench::comparisonEvidence(context, fixture, observations, backend == 1);
            std::weak_ptr<void> allocation = structure.retainAllocation();
            structure = {};
            TLAS_CHECK(!allocation.expired());
            lease = {};
            registry.collect();
            TLAS_CHECK(allocation.expired());
            if (usePartitioned && backend == 0) {
                TLAS_REQUIRE(frame.begin(1));
                TLAS_REQUIRE((*commands)->begin(frame.submissionContext()));
            }
        }
        (*heap)->release(*handle);
        TLAS_CHECK(validationErrors.load() == 0);
        return RHITestResult::pass("unified binding, ray hits/miss/mask, backend guards, move and allocation lifetime");
    }
private:
    bool partitioned_;
    bool native_;
};

class UnifiedPartitionedTopLevelTest final : public UnifiedTopLevelTest {
public:
    UnifiedPartitionedTopLevelTest() : UnifiedTopLevelTest(true) {}
};
class UnifiedNativeTopLevelTest final : public UnifiedTopLevelTest {
public:
    UnifiedNativeTopLevelTest() : UnifiedTopLevelTest(false, true) {}
};
class UnifiedNativePartitionedTopLevelTest final : public UnifiedTopLevelTest {
public:
    UnifiedNativePartitionedTopLevelTest() : UnifiedTopLevelTest(true, true) {}
};
METALLIC_REGISTER_RHI_TEST(UnifiedTopLevelTest);
METALLIC_REGISTER_RHI_TEST(UnifiedPartitionedTopLevelTest);
METALLIC_REGISTER_RHI_TEST(UnifiedNativeTopLevelTest);
METALLIC_REGISTER_RHI_TEST(UnifiedNativePartitionedTopLevelTest);

#undef TLAS_REQUIRE
#undef TLAS_CHECK
} // namespace
} // namespace metallic::tests
