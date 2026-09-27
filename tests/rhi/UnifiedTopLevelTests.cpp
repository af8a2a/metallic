#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/ResourceRegistry.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

#define TLAS_REQUIRE(expression) do { \
    const auto& checked = (expression); \
    if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + \
        toString(render::Result<>{std::unexpected(checked.error())}) + " " + log); } \
} while (false)
#define TLAS_CHECK(expression) do { if (!(expression)) { return RhiTestResult::fail(#expression); } } while (false)

class UnifiedTopLevelTest : public RhiTest {
public:
    UnifiedTopLevelTest(bool partitioned = false, bool native = false)
        : partitioned_(partitioned), native_(native)
    {
        type = RhiTestType::Rendering;
        name = partitioned ? (native ? "unified_top_level_partitioned_native" : "unified_top_level_partitioned")
            : (native ? "unified_top_level_standard_native" : "unified_top_level_standard");
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::string log;
        std::atomic_uint validationErrors = 0;
        auto created = createDevice({.applicationName = "Unified TLAS test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enablePartitionedAccelerationStructure = partitioned_,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}});
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("ray query/descriptor heap unavailable"); }
        TLAS_REQUIRE(created);
        auto& device = **created;
        TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure(RayTracingAccelerationStructureDesc{}), Error::InvalidArgument));
        TLAS_CHECK(hasError(device.createRayTracingAccelerationStructure(PartitionedAccelerationStructureDesc{}), Error::InvalidArgument));
        if (partitioned_ && !device.capabilities().partitionedAccelerationStructure) {
            return RhiTestResult::skip("PTLAS unavailable");
        }
        auto& queue = *device.getQueue(QueueType::Graphics);
        constexpr float vertices[] = {0, 0, 2, 1, 0, 2, 0, 1, 2};
        auto vertex = device.createBuffer({.size = sizeof(vertices),
            .usage = BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload});
        TLAS_REQUIRE(vertex);
        void* mapped = (*vertex)->map();
        TLAS_CHECK(mapped);
        std::memcpy(mapped, vertices, sizeof(vertices));
        (*vertex)->flush();
        (*vertex)->unmap();
        const RayTracingTriangleGeometryDesc geometry{.vertexBuffer = vertex->get(), .vertexStride = 12,
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
        std::unique_ptr<RayTracingAccelerationStructure> partitioned;
        std::unique_ptr<Buffer> partitionedInstances;
        uint64_t scratchSize = std::max(blasSizes->buildScratchSize, standardSizes->buildScratchSize);
        if (partitioned_) {
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
            scratchSize = std::max(scratchSize, sizes->buildScratchSize);
        }
        auto properties = device.queryRayTracingAccelerationStructureProperties();
        TLAS_REQUIRE(properties);
        auto scratch = device.createBuffer({.size = scratchSize + std::max<uint64_t>(256, properties->scratchAlignment),
            .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress});
        TLAS_REQUIRE(scratch);
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
        }, log);
        if (native_ && hasError(initialized, Error::Unsupported)) { return RhiTestResult::skip("native descriptor heap unavailable"); }
        TLAS_REQUIRE(initialized);
        using Probe = std::array<std::array<uint32_t, 2>, 3>;
        auto output = device.createBuffer({.size = sizeof(Probe), .structureStride = 8, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostReadback});
        TLAS_REQUIRE(output);
        ResourceRegistry registry;
        TLAS_REQUIRE(registry.initialize(device, {.maxBuffers = 1}));
        auto heap = device.createBindlessHeap({.maxBuffers = 1});
        TLAS_REQUIRE(heap);
        auto handle = (*heap)->allocateAccelerationStructure();
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
        TLAS_REQUIRE((*commands)->begin(&frame));
        TLAS_REQUIRE((*commands)->buildRayTracingAccelerationStructure({
            .destination = blas->get(),
            .geometries = {&geometry, 1},
            .scratchBuffer = scratch->get(),
        }));
        TLAS_REQUIRE((*commands)->buildRayTracingAccelerationStructure({.destination = standard->get(),
            .instanceBuffer = instances->get(), .instanceCount = 1, .scratchBuffer = scratch->get()}));
        if (partitioned_) {
            TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = partitioned.get(),
                .instanceBuffer = instances->get(), .instanceCount = 1, .scratchBuffer = scratch->get()}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->buildRayTracingAccelerationStructure({.destination = standard->get(), .source = partitioned.get(),
                .mode = RayTracingAccelerationStructureBuildMode::Update, .instanceBuffer = instances->get(),
                .instanceCount = 1, .scratchBuffer = scratch->get()}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->buildPartitionedAccelerationStructure({.destination = standard->get(),
                .instanceBuffer = partitionedInstances.get(), .instanceCount = 1, .scratchBuffer = scratch->get()}), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->compactRayTracingAccelerationStructure(*partitioned, **standard), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->compactRayTracingAccelerationStructure(**standard, *partitioned), Error::InvalidArgument));
            TLAS_CHECK(hasError((*commands)->writeRayTracingAccelerationStructureCompactedSize(**queries, 0, *partitioned), Error::InvalidArgument));
            TLAS_REQUIRE((*commands)->buildPartitionedAccelerationStructure({.destination = partitioned.get(),
                .instanceBuffer = partitionedInstances.get(), .instanceCount = 1, .scratchBuffer = scratch->get()}));
        }
        // Reuse the same shader, program, layout, binding slot and heap handle for both backends.
        std::array<RayTracingAccelerationStructure*, 2> structures{standard->get(), partitioned.get()};
        for (uint32_t backend = 0; backend < (partitioned_ ? 2u : 1u); ++backend) {
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
            const Probe expected{{{1, 37}, {0, UINT32_MAX}, {0, UINT32_MAX}}};
            TLAS_CHECK(actual == expected);
            TLAS_REQUIRE((*pool)->reset());
            TLAS_REQUIRE(frame.reset());
            std::weak_ptr<void> allocation = structure.retainAllocation();
            structure = {};
            TLAS_CHECK(!allocation.expired());
            lease = {};
            registry.collect();
            TLAS_CHECK(allocation.expired());
            if (partitioned_ && backend == 0) {
                TLAS_REQUIRE(frame.begin(1));
                TLAS_REQUIRE((*commands)->begin(&frame));
            }
        }
        (*heap)->release(*handle);
        TLAS_CHECK(validationErrors.load() == 0);
        return RhiTestResult::pass("unified binding, ray hits/miss/mask, backend guards, move and allocation lifetime");
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
