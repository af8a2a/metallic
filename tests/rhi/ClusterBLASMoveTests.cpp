#include "RHITest.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "harness/RayQueryFixture.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void require(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}

template<typename T>
void upload(Buffer& buffer, const T& value)
{
    void* data = buffer.map();
    require(data != nullptr, "Upload map failed");
    std::memcpy(data, &value, sizeof(value));
    buffer.flush(); buffer.unmap();
}

template<typename T>
T read(Buffer& buffer)
{
    buffer.invalidate();
    const void* data = buffer.map();
    require(data != nullptr, "Readback map failed");
    T value;
    std::memcpy(&value, data, sizeof(value)); buffer.unmap();
    return value;
}

class ClusterBLASMoveTest final : public RHITest {
public:
    ClusterBLASMoveTest()
    {
        type = RHITestType::Resource;
        name = "cluster_blas_indirect_move";
    }

    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint validationErrors = 0;
            auto created = createDevice({.applicationName = "Cluster BLAS move regression",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
                .enableClusterAccelerationStructure = true,
                .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                    if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                        ++*static_cast<std::atomic_uint*>(data);
                    }
                }, .context = &validationErrors}});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Cluster BLAS unavailable"); }
            auto device = checked(std::move(created));
            auto* queue = device->getQueue(QueueType::Graphics);
            auto buffer = [&](uint64_t bytes, MemoryLocation location = MemoryLocation::Device) {
                return checked(device->createBuffer({.size = bytes,
                    .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage |
                        BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
                    .memoryLocation = location}));
            };
            auto slice = [](Buffer& buffer, BufferRange range = {}) { return checked(buffer.slice(range)); };
            const auto properties = checked(device->queryClusterAccelerationStructureProperties());
            auto alignedSlice = [&](Buffer& buffer, uint64_t alignment, uint64_t bytes) {
                const uint64_t offset = (alignment - buffer.deviceAddress() % alignment) % alignment;
                return slice(buffer, {offset, bytes});
            };
            const auto clusterSizes = checked(device->queryClusterAccelerationStructureTriangleBuildSizes({
                .maxClusterTriangleCount = 1, .maxClusterVertexCount = 3,
                .maxTotalTriangleCount = 1, .maxTotalVertexCount = 3}));
            const auto blasSizes = checked(device->queryClusterAccelerationStructureBottomLevelBuildSizes({
                .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1}));
            const auto moveSizes = checked(device->queryClusterAccelerationStructureMoveSizes(
                1, blasSizes.accelerationStructureSize, ClusterAccelerationStructureObjectType::BottomLevel));
            const auto tlasSizes = checked(device->queryRayTracingAccelerationStructureBuildSizes({
                .type = RayTracingAccelerationStructureType::TopLevel, .instanceCount = 1}));
            const auto rtProperties = checked(device->queryRayTracingAccelerationStructureProperties());
            const uint64_t stride = (blasSizes.accelerationStructureSize + properties.bottomLevelStorageAlignment - 1) /
                properties.bottomLevelStorageAlignment * properties.bottomLevelStorageAlignment;
            auto cluster = buffer(clusterSizes.accelerationStructureSize + properties.clusterStorageAlignment);
            auto implicit = buffer(stride + properties.bottomLevelStorageAlignment);
            auto persistent = buffer(stride * 2 + properties.bottomLevelStorageAlignment);
            const auto clusterStorage = alignedSlice(*cluster, properties.clusterStorageAlignment, clusterSizes.accelerationStructureSize);
            const auto implicitStorage = alignedSlice(*implicit, properties.bottomLevelStorageAlignment, stride);
            const auto persistentStorage = alignedSlice(*persistent, properties.bottomLevelStorageAlignment, stride * 2);
            const uint64_t persistentOffset = persistentStorage.deviceAddress() - persistent->deviceAddress();
            auto scratch = buffer(std::max({clusterSizes.buildScratchSize, blasSizes.buildScratchSize,
                moveSizes.updateScratchSize, tlasSizes.buildScratchSize}) +
                std::max(properties.scratchAlignment, rtProperties.scratchAlignment));
            auto vertices = buffer(sizeof(bench::kRayVertices), MemoryLocation::HostUpload);
            auto indices = buffer(3, MemoryLocation::HostUpload);
            upload(*vertices, bench::kRayVertices); upload(*indices, std::array<uint8_t, 3>{0, 1, 2});
            auto clusterInfo = buffer(properties.triangleBuildInfoSize, MemoryLocation::HostUpload);
            auto clusterAddress = buffer(8, MemoryLocation::HostUpload);
            auto blasInfo = buffer(sizeof(ClusterAccelerationStructureBottomLevelBuildInfo), MemoryLocation::HostUpload);
            upload(*blasInfo, ClusterAccelerationStructureBottomLevelBuildInfo{.clusterReferencesCount = 1,
                .clusterReferencesAddress = clusterAddress->deviceAddress()});
            auto implicitAddress = buffer(8, MemoryLocation::HostReadback);
            auto actualSize = buffer(4, MemoryLocation::HostReadback);
            auto destinationAddress = buffer(8, MemoryLocation::HostUpload);
            auto moveCount = buffer(4, MemoryLocation::HostUpload);
            upload(*destinationAddress, persistentStorage.deviceAddress()); upload(*moveCount, uint32_t(1));
            auto instances = buffer(sizeof(RayTracingGPUInstance), MemoryLocation::HostUpload);
            auto tlas = checked(device->createRayTracingAccelerationStructure({
                .type = RayTracingAccelerationStructureType::TopLevel, .size = tlasSizes.accelerationStructureSize}));
            auto rays = buffer(sizeof(bench::RayObservations), MemoryLocation::HostReadback);
            std::string log;
            const char* capabilities[]{"spvRayQueryKHR"};
            const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "UnifiedTopLevelProbe",
                .entryPointName = "unifiedTopLevelMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = capabilities}, log));
            const ComputeResourceBindingDesc layout[]{
                {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
                {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output)}};
            ComputeProgram probe;
            checked(probe.initialize(*device, {.spirv = shader.spirv, .bindings = layout,
                .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));
            auto commandPool = checked(device->createCommandPool(*queue));
            auto commands = checked(commandPool->createCommandBuffer());
            auto fence = checked(device->createFence(false));
            struct Drain {
                Queue& queue;
                ~Drain() { (void)queue.waitIdle(); }
            } drain{*queue};
            auto begin = [&]() {
                checked(commandPool->reset()); checked(fence->reset()); checked(commands->begin());
            };
            auto trace = [&](uint64_t address) {
                RayTracingGPUInstance instance;
                instance.customIndexAndMask = 37 | (1u << 24);
                instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
                instance.accelerationStructureReference = address;
                upload(*instances, instance);
                checked(commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
                    .instanceBuffer = slice(*instances), .instanceCount = 1, .scratchBuffer = slice(*scratch)}));
                const ComputeDispatchBinding bindings[]{
                    {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), .accelerationStructure = tlas.get()},
                    {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output), .buffer = rays.get()}};
                checked(probe.dispatch({.commandBuffer = commands.get(), .bindings = bindings}));
            };
            auto submit = [&]() {
                checked(commands->end()); CommandBuffer* submitted[]{commands.get()};
                checked(queue->submit({.commandBuffers = submitted, .signalFence = fence.get()}));
                checked(fence->wait(5'000'000'000ull));
                return bench::rayOracle(read<bench::RayObservations>(*rays));
            };
            begin();
            const ClusterAccelerationStructureTriangleBuildInfo input{.triangleCount = 1, .vertexCount = 3,
                .indexBufferStride = 1, .vertexBufferStride = 12,
                .indexBuffer = slice(*indices), .vertexBuffer = slice(*vertices), .destinationBuffer = clusterStorage};
            checked(commands->buildClusterAccelerationStructureTriangles({.clusters = {&input, 1},
                .maxClusterTriangleCount = 1, .maxClusterVertexCount = 3, .scratchBuffer = slice(*scratch),
                .buildInfoBuffer = slice(*clusterInfo), .destinationAddressBuffer = slice(*clusterAddress)}));
            checked(commands->buildClusterAccelerationStructureBottomLevels({
                .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1,
                .buildInfoBuffer = slice(*blasInfo), .destinationStorageBuffer = implicitStorage,
                .destinationAddressBuffer = slice(*implicitAddress), .destinationSizeBuffer = slice(*actualSize),
                .scratchBuffer = slice(*scratch)}));
            ClusterAccelerationStructureIndirectMoveDesc move{.maxObjectCount = 1, .maxMovedBytes = stride,
                .sourceAddressBuffer = slice(*implicitAddress), .destinationAddressBuffer = slice(*destinationAddress),
                .objectCountBuffer = slice(*moveCount), .scratchBuffer = slice(*scratch),
                .sourceStorageBuffer = implicitStorage, .destinationStorageBuffer = persistentStorage,
                .objectType = ClusterAccelerationStructureObjectType::BottomLevel};
            auto invalid = move;
            invalid.destinationStorageBuffer = implicitStorage;
            require(hasError(commands->moveClusterAccelerationStructuresIndirect(invalid), Error::InvalidArgument),
                "Overlapping retained storage accepted");
            invalid = move;
            invalid.destinationStorageBuffer = slice(*persistent, {persistentOffset + 1, stride});
            require(hasError(commands->moveClusterAccelerationStructuresIndirect(invalid), Error::InvalidArgument),
                "Unaligned BLAS storage accepted");
            checked(commands->moveClusterAccelerationStructuresIndirect(move));
            trace(persistentStorage.deviceAddress());
            const auto firstRays = submit();
            const uint32_t encodedBytes = read<uint32_t>(*actualSize);
            const uint64_t sourceAddress = read<uint64_t>(*implicitAddress);
            require(encodedBytes > 0 && encodedBytes <= stride && encodedBytes % properties.bottomLevelStorageAlignment == 0,
                "Invalid GPU BLAS encoded size");

            // Relocate to a disjoint slice in the same allocation. Every old
            // source remains a valid BLAS because noMoveOverlap is true.
            auto secondSourceAddress = buffer(8, MemoryLocation::HostUpload);
            upload(*secondSourceAddress, persistentStorage.deviceAddress());
            upload(*destinationAddress, persistentStorage.deviceAddress() + stride);
            move.sourceAddressBuffer = slice(*secondSourceAddress);
            move.sourceStorageBuffer = persistentStorage;
            move.destinationStorageBuffer = persistentStorage;
            move.maxMovedBytes = encodedBytes;
            begin();
            require(hasError(commands->moveClusterAccelerationStructuresIndirect(move), Error::InvalidArgument),
                "Aliased storage accepted without GPU producer contract");
            move.allowAliasedStorage = true;
            checked(commands->moveClusterAccelerationStructuresIndirect(move));
            trace(persistentStorage.deviceAddress() + stride);
            (void)submit();
            for (uint32_t frame = 0; frame < 32; ++frame) {
                begin();
                trace(frame % 3 == 0 ? sourceAddress : persistentStorage.deviceAddress() + (frame % 3 == 2 ? stride : 0));
                (void)submit();
            }

            // A lower hole smaller than the BLAS cannot receive a direct Move.
            // Stage the source, retire its original interval, then copy back to
            // a lower range that overlaps that retired interval. Distinct
            // immutable address tables keep both copies in one submission.
            require(encodedBytes > properties.bottomLevelStorageAlignment,
                "BLAS fixture must span multiple alignment units to exercise overlapping return ranges");
            const uint64_t stagedSource = persistentStorage.deviceAddress() + stride;
            const uint64_t returnedAddress = stagedSource - properties.bottomLevelStorageAlignment;
            require(returnedAddress < stagedSource && returnedAddress + encodedBytes > stagedSource,
                "Relocation return does not overlap the retired original source");
            auto relocationStaging = buffer(encodedBytes + properties.bottomLevelStorageAlignment);
            const auto relocationStorage = alignedSlice(*relocationStaging, properties.bottomLevelStorageAlignment, encodedBytes);
            auto oldAddressTable = buffer(8, MemoryLocation::HostUpload);
            auto relocationAddressTable = buffer(8, MemoryLocation::HostUpload);
            auto returnedAddressTable = buffer(8, MemoryLocation::HostUpload);
            upload(*oldAddressTable, stagedSource);
            upload(*relocationAddressTable, relocationStorage.deviceAddress());
            upload(*returnedAddressTable, returnedAddress);
            ClusterAccelerationStructureIndirectMoveDesc toStaging{
                .maxObjectCount = 1, .maxMovedBytes = encodedBytes,
                .sourceAddressBuffer = slice(*oldAddressTable), .destinationAddressBuffer = slice(*relocationAddressTable),
                .objectCountBuffer = slice(*moveCount), .scratchBuffer = slice(*scratch),
                .sourceStorageBuffer = persistentStorage, .destinationStorageBuffer = relocationStorage,
                .objectType = ClusterAccelerationStructureObjectType::BottomLevel};
            ClusterAccelerationStructureIndirectMoveDesc fromStaging{
                .maxObjectCount = 1, .maxMovedBytes = encodedBytes,
                .sourceAddressBuffer = slice(*relocationAddressTable), .destinationAddressBuffer = slice(*returnedAddressTable),
                .objectCountBuffer = slice(*moveCount), .scratchBuffer = slice(*scratch),
                .sourceStorageBuffer = relocationStorage, .destinationStorageBuffer = persistentStorage,
                .objectType = ClusterAccelerationStructureObjectType::BottomLevel};
            begin();
            checked(commands->moveClusterAccelerationStructuresIndirect(toStaging));
            checked(commands->moveClusterAccelerationStructuresIndirect(fromStaging));
            trace(returnedAddress);
            const auto returnedRays = submit();
            // The original persistent copies are intentionally retired here:
            // the return overwrote their ranges. Only the independent implicit
            // source, relocation staging and newly published BLAS remain live.
            for (uint32_t frame = 0; frame < 32; ++frame) {
                begin();
                trace(frame % 3 == 0 ? sourceAddress : frame % 3 == 1 ? relocationStorage.deviceAddress() : returnedAddress);
                (void)submit();
            }
            std::filesystem::create_directories(context.outputDirectory);
            require(validationErrors == 0, "Vulkan validation reported errors");
            std::ofstream(context.outputDirectory / "ClusterBLASMove.json") << nlohmann::json{
                {"encodedBytes", encodedBytes}, {"reservedBytes", stride}, {"firstSubmissionRay", firstRays},
                {"cpuAddressReadbackBeforeFirstMove", false}, {"stableFrames", 32},
                {"oldSourcesValidBeforeScratchReturn", true}, {"sameAllocationDisjointMove", true},
                {"scratchAssistedRelocation", {{"originalOffset", stride},
                    {"returnedOffset", returnedAddress - persistentStorage.deviceAddress()},
                    {"overlapsRetiredSource", true}, {"copiesPerSubmission", 2},
                    {"allowAliasedStorage", false}, {"stableFrames", 32}, {"returnedRays", returnedRays}}},
                {"validationErrors", validationErrors.load()}}.dump(2);
            return RHITestResult::pass("Implicit BLAS and indirect Move; same-buffer copy; scratch-assisted overlapping return; 32 stable frames per relocation mode with analytic rays");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(ClusterBLASMoveTest);
} // namespace
} // namespace metallic::tests
