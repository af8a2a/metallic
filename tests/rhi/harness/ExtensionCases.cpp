#include "RayQueryFixture.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "../UnifiedTopLevelProbeParameters.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <cstring>

namespace metallic::tests::bench {
namespace {
using namespace render;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

std::unique_ptr<Buffer> buffer(Device& device, uint64_t size, MemoryLocation location = MemoryLocation::Device)
{
    return checked(device.createBuffer({.size = size,
        .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage,
        .memoryLocation = location}));
}

template<typename T>
void upload(Buffer& buffer, const T& value)
{
    auto* data = buffer.map();
    if (!data) { throw std::runtime_error("upload map failed"); }
    std::memcpy(data, &value, sizeof(value)); buffer.flush(); buffer.unmap();
}

template<typename T>
T read(Buffer& buffer)
{
    auto* data = buffer.map();
    if (!data) { throw std::runtime_error("readback map failed"); }
    buffer.invalidate();
    T value{}; std::memcpy(&value, data, sizeof(value)); buffer.unmap();
    return value;
}

uint64_t alignedOffset(const Buffer& buffer, uint64_t alignment)
{
    return (alignment - buffer.deviceAddress() % alignment) % alignment;
}

Json trace(RHITestContext& context, uint64_t blasAddress)
{
    auto& device = context.device;
    auto& queue = context.graphicsQueue;
    RayTracingGPUInstance instance;
    instance.customIndexAndMask = 37 | (1u << 24);
    instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
    instance.accelerationStructureReference = blasAddress;
    auto instances = buffer(device, sizeof(instance), MemoryLocation::HostUpload);
    upload(*instances, instance);
    const auto sizes = checked(device.queryRayTracingAccelerationStructureBuildSizes({
        .type = RayTracingAccelerationStructureType::TopLevel, .instanceCount = 1}));
    auto tlas = checked(device.createRayTracingAccelerationStructure({.type = RayTracingAccelerationStructureType::TopLevel,
        .size = sizes.accelerationStructureSize}));
    const auto properties = checked(device.queryRayTracingAccelerationStructureProperties());
    auto scratch = buffer(device, sizes.buildScratchSize + properties.scratchAlignment);
    {
        GPUCommands build(queue); checked(build.initialize(device));
        checked(build.commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
            .instanceBuffer = instances.get(), .instanceCount = 1, .scratchBuffer = scratch.get()}));
        checked(build.submitAndWait());
    }
    const char* capabilities[]{"spvRayQueryKHR"};
    std::string log;
    const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "UnifiedTopLevelProbe",
        .entryPointName = "unifiedTopLevelMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .capabilities = capabilities}, log));
    ComputeKernel program;
    checked(program.initialize(device, {.spirv = shader.spirv,
        .parameters = parameterAbi<UnifiedTopLevelProbeParameters>(kUnifiedRayProbeABI, ParameterTransport::InlinePush)}, log));
    auto output = buffer(device, sizeof(RayObservations), MemoryLocation::HostReadback);
    auto pool = checked(device.createCommandPool(queue));
    auto commands = checked(pool->createCommandBuffer());
    QueueSubmissionTracker tracker; checked(tracker.initialize(device, queue));
    RenderFrameContext frame;
    struct Drain {
        RenderFrameContext& frame; CommandPool& pool;
        ~Drain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
    } drain{frame, *pool};
    checked(frame.begin(0)); checked(commands->begin(&frame));
    auto registry = checked(device.resourceRegistry());
    ParameterWriter writer(device, *registry, &frame);
    const UnifiedTopLevelProbeParameters params{writer.accelerationStructure(tlas.get()),
        writer.dataBuffer(output.get(), sizeof(RayObservation), 4)};
    auto encoded = checked(writer.encode(params, kUnifiedRayProbeABI, ParameterTransport::InlinePush));
    checked(program.dispatch(*commands, encoded, 1));
    checked(commands->end());
    CommandBuffer* submitted[]{commands.get()};
    checked(tracker.submit({.commandBuffers = submitted}, frame)); checked(frame.wait(5'000'000'000ull));
    const auto actual = read<RayObservations>(*output);
    readbackEvidence(context, "readback.bin", std::span<const RayObservation>(actual));
    return rayOracle(actual);
}

class ClusterRayQueryTest final : public RHITest {
public:
    ClusterRayQueryTest() { type = RHITestType::Rendering; name = "clas_relocated_analytic_ray_query"; }
    std::optional<Metadata> metadata() const override
    {
        return comparisonMetadata({"clas.move.retire.traversal", "rayQuery.analytic.hit.miss.mask.distance.barycentric.frontFace"},
            Layer::RHI, "ray-query", {"ray-query-clas", "clusterAS", Capability::ClusterAS, 0.00001});
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto created = createTestDevice(context, {.applicationName = "CLAS analytic probe", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enableRayTracingPositionFetch = false, .enableOpacityMicromap = false, .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("ray query/CLAS profile unavailable"); }
        auto device = checked(std::move(created));
        if (!device->capabilities().rayQuery || !device->capabilities().bindlessDescriptorHeap ||
            (!context.evidence && !device->capabilities().clusterAccelerationStructure)) {
            return RHITestResult::skip("ray query/CLAS unavailable");
        }
        RHITestContext active{*device, *device->getQueue(QueueType::Graphics), context.outputDirectory,
            context.enableValidation, context.validationMessageCount, context.nsightCapture, context.evidence, context.deviceDesc};
        return runDevice(active);
    }
private:
    RHITestResult runDevice(RHITestContext& context)
    {
        auto& device = context.device;
        const bool clusters = context.deviceDesc ? context.deviceDesc->enableClusterAccelerationStructure : device.capabilities().clusterAccelerationStructure;
        auto vertices = buffer(device, sizeof(kRayVertices), MemoryLocation::HostUpload);
        upload(*vertices, kRayVertices);
        Json observations = Json::array();
        if (!clusters) {
            const RayTracingTriangleGeometryDesc geometry{.vertexBuffer = vertices.get(), .vertexStride = 12,
                .vertexCount = 3, .indexType = RayTracingIndexType::None, .primitiveCount = 1};
            const auto sizes = checked(device.queryRayTracingAccelerationStructureBuildSizes({.geometries = {&geometry, 1}}));
            auto blas = checked(device.createRayTracingAccelerationStructure({.size = sizes.accelerationStructureSize}));
            auto scratch = buffer(device, sizes.buildScratchSize + checked(device.queryRayTracingAccelerationStructureProperties()).scratchAlignment);
            {
                GPUCommands build(context.graphicsQueue); checked(build.initialize(device));
                checked(build.commands->buildRayTracingAccelerationStructure({.destination = blas.get(),
                    .geometries = {&geometry, 1}, .scratchBuffer = scratch.get()}));
                checked(build.submitAndWait());
            }
            for (unsigned i = 0; i < 2; ++i) { observations.push_back(trace(context, blas->deviceAddress())); }
        } else {
            const auto properties = checked(device.queryClusterAccelerationStructureProperties());
            const auto sizes = checked(device.queryClusterAccelerationStructureTriangleBuildSizes({.maxClusterTriangleCount = 128,
                .maxClusterVertexCount = 128, .maxTotalTriangleCount = 128, .maxTotalVertexCount = 128}));
            auto storage = buffer(device, sizes.accelerationStructureSize + properties.clusterStorageAlignment);
            uint64_t offset = alignedOffset(*storage, properties.clusterStorageAlignment);
            uint32_t actualSize = 0;
            {
                const std::array<uint8_t, 3> indexData{0, 1, 2};
                auto indices = buffer(device, sizeof(indexData), MemoryLocation::HostUpload); upload(*indices, indexData);
                auto infos = buffer(device, properties.triangleBuildInfoSize, MemoryLocation::HostUpload);
                auto addresses = buffer(device, 8, MemoryLocation::HostUpload);
                auto encodedSize = buffer(device, 4, MemoryLocation::HostReadback);
                auto scratch = buffer(device, sizes.buildScratchSize + properties.scratchAlignment);
                const ClusterAccelerationStructureTriangleBuildInfo input{.triangleCount = 1, .vertexCount = 3,
                    .vertexBufferStride = 12, .indexBuffer = indices.get(), .vertexBuffer = vertices.get(),
                    .destinationBuffer = storage.get(), .destinationBufferOffset = offset, .destinationSize = sizes.accelerationStructureSize};
                GPUCommands build(context.graphicsQueue); checked(build.initialize(device));
                checked(build.commands->buildClusterAccelerationStructureTriangles({.clusters = {&input, 1},
                    .maxClusterTriangleCount = 128, .maxClusterVertexCount = 128,
                    .scratchBuffer = scratch.get(), .scratchBufferOffset = alignedOffset(*scratch, properties.scratchAlignment),
                    .buildInfoBuffer = infos.get(), .destinationAddressBuffer = addresses.get(), .destinationSizeBuffer = encodedSize.get()}));
                checked(build.submitAndWait());
                actualSize = read<uint32_t>(*encodedSize);
                if (!actualSize || actualSize >= sizes.accelerationStructureSize || actualSize % properties.clusterStorageAlignment) {
                    return RHITestResult::fail("CLAS actual encoded size was not compact and aligned");
                }
            }
            Json relocations = Json::array();
            for (unsigned iteration = 0; iteration < 2; ++iteration) {
                auto destination = buffer(device, actualSize + properties.clusterStorageAlignment);
                const auto destinationOffset = alignedOffset(*destination, properties.clusterStorageAlignment);
                {
                    const auto moveSizes = checked(device.queryClusterAccelerationStructureMoveSizes(1, actualSize));
                    auto scratch = buffer(device, moveSizes.updateScratchSize + properties.scratchAlignment);
                    auto sources = buffer(device, 8, MemoryLocation::HostUpload), destinations = buffer(device, 8, MemoryLocation::HostUpload);
                    const ClusterAccelerationStructureMoveInfo move{.sourceBuffer = storage.get(), .sourceOffset = offset,
                        .destinationBuffer = destination.get(), .destinationOffset = destinationOffset, .size = actualSize};
                    GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
                    checked(commands.commands->moveClusterAccelerationStructures({.objects = {&move, 1},
                        .sourceAddressBuffer = sources.get(), .destinationAddressBuffer = destinations.get(),
                        .scratchBuffer = scratch.get(), .scratchBufferOffset = alignedOffset(*scratch, properties.scratchAlignment)}));
                    checked(commands.submitAndWait());
                }
                std::weak_ptr<void> retired = storage->retainAllocation();
                storage.reset();
                if (!retired.expired()) { return RHITestResult::fail("old CLAS allocation survived retirement"); }
                storage = std::move(destination); offset = destinationOffset;
                const auto blasSizes = checked(device.queryClusterAccelerationStructureBottomLevelBuildSizes({
                    .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1}));
                auto blas = buffer(device, blasSizes.accelerationStructureSize + properties.bottomLevelStorageAlignment);
                const uint64_t blasAddress = blas->deviceAddress() + alignedOffset(*blas, properties.bottomLevelStorageAlignment);
                {
                    auto references = buffer(device, 8, MemoryLocation::HostUpload);
                    upload(*references, storage->deviceAddress() + offset);
                    const ClusterAccelerationStructureBottomLevelBuildInfo info{.clusterReferencesCount = 1,
                        .clusterReferencesAddress = references->deviceAddress()};
                    auto infos = buffer(device, sizeof(info), MemoryLocation::HostUpload); upload(*infos, info);
                    auto addresses = buffer(device, 8, MemoryLocation::HostUpload); upload(*addresses, blasAddress);
                    auto scratch = buffer(device, blasSizes.buildScratchSize + properties.scratchAlignment);
                    GPUCommands commands(context.graphicsQueue); checked(commands.initialize(device));
                    checked(commands.commands->buildClusterAccelerationStructureBottomLevels({
                        .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
                        .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1,
                        .buildInfoBuffer = infos.get(), .destinationAddressBuffer = addresses.get(), .scratchBuffer = scratch.get()}));
                    checked(commands.submitAndWait());
                }
                observations.push_back(trace(context, blasAddress));
                relocations.push_back({{"iteration", iteration}, {"oldAllocationRetired", true}, {"traceCompleted", true}, {"encodedBytes", actualSize}});
            }
            if (context.evidence) { context.evidence->json("relocations.json", relocations); }
        }
        auto fixture = rayFixture(); fixture["relocations"] = 2;
        comparisonEvidence(context, fixture, observations, clusters);
        return RHITestResult::pass("analytic rays passed after each completed relocation and old allocation retirement");
    }
};
METALLIC_REGISTER_RHI_TEST(ClusterRayQueryTest);

} // namespace
} // namespace metallic::tests::bench
