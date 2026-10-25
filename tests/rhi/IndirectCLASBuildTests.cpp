#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace metallic::tests {
namespace {

class IndirectCLASBuildTest final : public RHITest {
public:
    IndirectCLASBuildTest()
    {
        type = RHITestType::Resource;
        name = "clas_indirect_triangle_build";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"clas.indirect.gpu.triangles"}, bench::Layer::RHI, "ray-query", "core");
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::atomic_uint validationErrors{0};
        auto deviceResult = bench::createTestDevice(context, {
            .applicationName = "Indirect CLAS build regression",
            .enableValidation = context.enableValidation,
            .enableClusterAccelerationStructure = true,
            .validationSink = {
                .callback = [](void* opaque, const ValidationMessage& message) noexcept {
                    if (message.severity == ValidationSeverity::Error) {
                        static_cast<std::atomic_uint*>(opaque)->fetch_add(1);
                    }
                },
                .context = &validationErrors,
            },
        });
        if (hasError(deviceResult, Error::Unsupported)) { return RHITestResult::skip("CLAS unavailable"); }
        if (!deviceResult) { return RHITestResult::fail("CLAS device creation failed"); }
        auto device = std::move(*deviceResult);
        const auto require = [](bool condition, const char* message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        try {
#ifdef VK_NV_cluster_acceleration_structure
            // Compare every packed field against the native bitfield ABI before
            // passing any GPU-produced record to the driver.
            static_assert(METALLIC_CLAS_INDEX_UINT8 == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV);
            static_assert(METALLIC_CLAS_INDEX_UINT16 == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV);
            static_assert(METALLIC_CLAS_INDEX_UINT32 == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV);
            static_assert(METALLIC_CLAS_GEOMETRY_OPAQUE ==
                (uint32_t(VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV) << 29u));
            for (const uint32_t indexType : {METALLIC_CLAS_INDEX_UINT8, METALLIC_CLAS_INDEX_UINT16,
                                            METALLIC_CLAS_INDEX_UINT32}) {
                ClusterAccelerationStructureTriangleIndirectBuildInfo record{};
                record.clusterId = 0x12345678;
                record.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(96u, 64u, 7u, indexType);
                record.geometryIndexAndFlags = 0x123456u | METALLIC_CLAS_GEOMETRY_OPAQUE;
                record.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(4u, 32u);
                record.indexBufferAddress = 0x0123456789abcdefull;
                record.vertexBufferAddress = 0xfedcba9876543210ull;
                VkClusterAccelerationStructureBuildTriangleClusterInfoNV native{};
                native.clusterID = record.clusterId;
                native.triangleCount = 96; native.vertexCount = 64;
                native.positionTruncateBitCount = 7; native.indexType = indexType;
                native.baseGeometryIndexAndGeometryFlags.geometryIndex = 0x123456;
                native.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
                native.indexBufferStride = 4; native.vertexBufferStride = 32;
                native.indexBuffer = record.indexBufferAddress; native.vertexBuffer = record.vertexBufferAddress;
                require(std::memcmp(&record, &native, sizeof(record)) == 0, "Shared CLAS packed fields differ from native ABI");
            }
#endif
            constexpr auto usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage |
                BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination;
            const auto buffer = [&](uint64_t size, MemoryLocation location) {
                auto result = device->createBuffer({.size = size, .usage = usage, .memoryLocation = location});
                require(bool(result), "Buffer allocation failed");
                return std::move(*result);
            };
            const auto slice = [&](Buffer& value, BufferRange range = {}) {
                auto result = value.slice(range);
                require(bool(result), "Buffer slice failed");
                return *result;
            };
            const auto write = [&](Buffer& value, const void* bytes, size_t size) {
                void* mapped = value.map();
                require(mapped != nullptr, "Upload map failed");
                std::memcpy(mapped, bytes, size);
                value.flush(); value.unmap();
            };
            auto propertiesResult = device->queryClusterAccelerationStructureProperties();
            require(bool(propertiesResult), "CLAS properties unavailable");
            const auto properties = *propertiesResult;
            const ClusterAccelerationStructureTriangleBuildSizesDesc limits{
                .maxClusterTriangleCount = 1, .maxClusterVertexCount = 3,
                .maxTotalTriangleCount = 2, .maxTotalVertexCount = 6,
                .maxAccelerationStructureCount = 2,
            };
            const auto nativeDevice = vulkan::nativeDevice(*device);
            VkFormatProperties depthFormatProperties{};
            nativeDevice.instanceFunctions->vkGetPhysicalDeviceFormatProperties(nativeDevice.physicalDevice,
                VK_FORMAT_D32_SFLOAT, &depthFormatProperties);
            const bool depthVertexFormatUnsupported = !(depthFormatProperties.bufferFeatures &
                VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR);
            if (depthVertexFormatUnsupported) {
                auto invalidLimits = limits;
                invalidLimits.vertexFormat = Format::D32Sfloat;
                require(hasError(device->queryClusterAccelerationStructureTriangleBuildSizes(invalidLimits,
                    ClusterAccelerationStructureDestinationMode::Implicit), Error::InvalidArgument),
                    "Unsupported CLAS vertex format reached the size query");
            }
            auto sizesResult = device->queryClusterAccelerationStructureTriangleBuildSizes(
                limits, ClusterAccelerationStructureDestinationMode::Implicit);
            require(bool(sizesResult), "Implicit CLAS size query failed");
            auto storage = buffer(sizesResult->accelerationStructureSize, MemoryLocation::Device);
            auto scratch = buffer(sizesResult->buildScratchSize + properties.scratchAlignment, MemoryLocation::Device);
            auto vertices = buffer(36, MemoryLocation::HostUpload);
            auto indices = buffer(3, MemoryLocation::HostUpload);
            const float points[]{-1, -1, 0, 1, -1, 0, 0, 1, 0};
            const uint8_t triangles[]{0, 1, 2};
            write(*vertices, points, sizeof(points)); write(*indices, triangles, sizeof(triangles));

            // Nondefault strides and offset slices catch accidentally assuming
            // tightly packed arrays in either the command or native record ABI.
            constexpr uint64_t infoStride = 80;
            std::array<uint8_t, infoStride * 2> records{};
            for (uint32_t index = 0; index < 2; ++index) {
                ClusterAccelerationStructureTriangleIndirectBuildInfo record{};
                record.clusterId = 23 + index;
                record.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(1u, 3u, 0u, METALLIC_CLAS_INDEX_UINT8);
                record.geometryIndexAndFlags = METALLIC_CLAS_GEOMETRY_OPAQUE;
                record.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(1u, 12u);
                record.indexBufferAddress = indices->deviceAddress();
                record.vertexBufferAddress = vertices->deviceAddress();
#ifdef VK_NV_cluster_acceleration_structure
                VkClusterAccelerationStructureBuildTriangleClusterInfoNV native{};
                native.clusterID = record.clusterId;
                native.triangleCount = 1; native.vertexCount = 3;
                native.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
                native.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
                native.indexBufferStride = 1; native.vertexBufferStride = 12;
                native.indexBuffer = record.indexBufferAddress; native.vertexBuffer = record.vertexBufferAddress;
                require(std::memcmp(&record, &native, sizeof(record)) == 0, "Shared CLAS record differs from native ABI");
#endif
                std::memcpy(records.data() + infoStride * index, &record, sizeof(record));
            }
            auto infosUpload = buffer(records.size(), MemoryLocation::HostUpload);
            write(*infosUpload, records.data(), records.size());
            auto infos = buffer(16 + records.size(), MemoryLocation::Device);
            auto countUpload = buffer(4, MemoryLocation::HostUpload);
            auto count = buffer(8, MemoryLocation::Device);
            auto addresses = buffer(48, MemoryLocation::HostReadback);
            auto sizes = buffer(32, MemoryLocation::HostReadback);
            std::array<BufferSlice, 2> geometry{slice(*vertices), slice(*indices)};
            ClusterAccelerationStructureTriangleIndirectBuildDesc build{
                .limits = limits,
                .buildInfoBuffer = slice(*infos, {16, records.size()}), .buildInfoStride = infoStride,
                .buildInfoCountBuffer = slice(*count, {4, 4}),
                .destinationStorageBuffer = slice(*storage),
                .destinationAddressBuffer = slice(*addresses, {8, 32}), .destinationAddressStride = 16,
                .destinationSizeBuffer = slice(*sizes, {8, 16}), .destinationSizeStride = 8,
                .scratchBuffer = slice(*scratch), .geometryBuffers = geometry,
            };
            auto* queue = device->getQueue(QueueType::Graphics);
            auto poolResult = device->createCommandPool(*queue);
            require(bool(poolResult), "Command pool creation failed");
            auto pool = std::move(*poolResult);
            auto commandsResult = pool->createCommandBuffer();
            require(bool(commandsResult), "Command buffer creation failed");
            auto commands = std::move(*commandsResult);
            auto fenceResult = device->createFence(false);
            require(bool(fenceResult), "Fence creation failed");
            auto fence = std::move(*fenceResult);
            for (uint32_t activeCount = 0; activeCount <= 2; ++activeCount) {
                std::array<uint64_t, 6> addressWords;
                std::array<uint32_t, 8> sizeWords;
                addressWords.fill(UINT64_MAX); sizeWords.fill(UINT32_MAX);
                write(*addresses, addressWords.data(), sizeof(addressWords));
                write(*sizes, sizeWords.data(), sizeof(sizeWords));
                write(*countUpload, &activeCount, sizeof(activeCount));
                require(bool(commands->begin()), "Begin failed");
                if (activeCount == 0) {
                    auto invalid = build;
                    if (depthVertexFormatUnsupported) {
                        invalid.limits.vertexFormat = Format::D32Sfloat;
                        require(hasError(commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                            "Unsupported indirect CLAS vertex format accepted");
                        invalid = build;
                    }
                    invalid.buildInfoStride = 63;
                    require(hasError(commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                        "Invalid indirect record stride accepted");
                    invalid = build; invalid.buildInfoCountBuffer = slice(*count, {4, 3});
                    require(hasError(commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                        "Undersized count slice accepted");
                    invalid = build; invalid.destinationStorageBuffer = slice(*storage, {0, 1});
                    require(hasError(commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                        "Undersized implicit arena accepted");
                    invalid = build; invalid.scratchBuffer = slice(*scratch, {0, 1});
                    require(hasError(commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                        "Undersized indirect scratch accepted");
                }
                require(bool(commands->copyBuffer(slice(*infosUpload), build.buildInfoBuffer)) &&
                    bool(commands->copyBuffer(slice(*countUpload), build.buildInfoCountBuffer)), "GPU input/count transfer failed");
                require(bool(commands->buildClusterAccelerationStructureTrianglesIndirect(build)), "Indirect CLAS recording failed");
                const MemoryBarrierDesc hostRead{
                    .before = {PipelineStageBits::AccelerationStructureBuild, AccessBits::AccelerationStructureWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead},
                };
                require(bool(commands->synchronize({.memory = {&hostRead, 1}})), "Readback barrier failed");
                require(bool(commands->end()), "End failed");
                CommandBuffer* submitted[]{commands.get()};
                require(bool(queue->submit({.commandBuffers = submitted, .signalFence = fence.get()})) &&
                    bool(fence->wait(5'000'000'000ull)), "CLAS submission failed");
                addresses->invalidate(); sizes->invalidate();
                std::memcpy(addressWords.data(), addresses->map(), sizeof(addressWords)); addresses->unmap();
                std::memcpy(sizeWords.data(), sizes->map(), sizeof(sizeWords)); sizes->unmap();
                for (uint32_t index = 0; index < activeCount; ++index) {
                    const uint64_t address = addressWords[1 + index * 2];
                    const uint32_t size = sizeWords[2 + index * 2];
                    require(address >= storage->deviceAddress() && address < storage->deviceAddress() + storage->desc().size &&
                        address % properties.clusterStorageAlignment == 0 && size > 0 &&
                        size <= storage->deviceAddress() + storage->desc().size - address,
                        "Indirect CLAS output exceeds its implicit arena");
                }
                for (uint32_t word = 0; word < addressWords.size(); ++word) {
                    const bool output = (word == 1 && activeCount >= 1) || (word == 3 && activeCount >= 2);
                    require(output || addressWords[word] == UINT64_MAX, "Indirect address output overwrote a guard/unused record");
                }
                for (uint32_t word = 0; word < sizeWords.size(); ++word) {
                    const bool output = (word == 2 && activeCount >= 1) || (word == 4 && activeCount >= 2);
                    require(output || sizeWords[word] == UINT32_MAX, "Indirect size output overwrote a guard/unused record");
                }
                require(bool(pool->reset()) && bool(fence->reset()), "Command retirement failed");
            }
            require(bool(commands->begin()) && bool(commands->buildClusterAccelerationStructureTrianglesIndirect(build)),
                "Cancellation recording failed");
            const std::weak_ptr<void> vertexLifetime = vertices->retainAllocation();
            geometry = {}; build.geometryBuffers = {}; vertices.reset();
            require(!vertexLifetime.expired(), "Indirect build did not retain GPU-addressed geometry");
            require(bool(commands->end()) && bool(pool->reset()), "Cancellation retirement failed");
            require(vertexLifetime.expired(), "Cancelled indirect build leaked geometry");
            require(validationErrors.load() == 0, "Vulkan validation reported an indirect CLAS error");
            return RHITestResult::pass("Native ABI, GPU-written counts 0/1/2, strided implicit outputs, bounds rejection and cancellation retention");
        } catch (const std::exception& error) {
            (void)device->waitIdle();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(IndirectCLASBuildTest);

} // namespace
} // namespace metallic::tests
