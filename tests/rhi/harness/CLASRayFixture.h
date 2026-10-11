#pragma once

#include "RayQueryFixture.h"
#include <algorithm>
#include <cstring>

namespace metallic::tests::bench {

// The analytic ray fixture follows the production geometry representation:
// triangle cluster -> cluster BLAS -> standard or partitioned TLAS.
class CLASRayFixture {
public:
    render::Result<> initialize(render::Device& device)
    {
        using namespace render;
        auto properties = device.queryClusterAccelerationStructureProperties();
        if (!properties) { return makeError(properties.error()); }
        properties_ = *properties;
        auto clusterSizes = device.queryClusterAccelerationStructureTriangleBuildSizes({
            .maxClusterTriangleCount = 128, .maxClusterVertexCount = 128,
            .maxTotalTriangleCount = 128, .maxTotalVertexCount = 128});
        if (!clusterSizes) { return makeError(clusterSizes.error()); }
        auto bottomSizes = device.queryClusterAccelerationStructureBottomLevelBuildSizes({
            .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1});
        if (!bottomSizes) { return makeError(bottomSizes.error()); }
        const auto allocate = [&](std::unique_ptr<Buffer>& buffer, uint64_t size, MemoryLocation location) -> Result<> {
            auto result = device.createBuffer({.size = size,
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                    BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage,
                .memoryLocation = location, .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
            if (!result) { return makeError(result.error()); }
            buffer = std::move(*result);
            return {};
        };
        const auto upload = [&](std::unique_ptr<Buffer>& buffer, const auto& data) -> Result<> {
            auto result = allocate(buffer, sizeof(data), MemoryLocation::HostUpload);
            if (!result) { return result; }
            void* mapped = buffer->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, &data, sizeof(data));
            buffer->flush(); buffer->unmap();
            return {};
        };
#define CLAS_FIXTURE_REQUIRE(expression) do { auto result = (expression); if (!result) { return makeError(result.error()); } } while (false)
        CLAS_FIXTURE_REQUIRE(upload(vertices_, kRayVertices));
        CLAS_FIXTURE_REQUIRE(upload(indices_, (std::array<uint8_t, 3>{0, 1, 2})));
        CLAS_FIXTURE_REQUIRE(allocate(cluster_, clusterSizes->accelerationStructureSize + properties_.clusterStorageAlignment, MemoryLocation::Device));
        CLAS_FIXTURE_REQUIRE(allocate(bottom_, bottomSizes->accelerationStructureSize + properties_.bottomLevelStorageAlignment, MemoryLocation::Device));
        CLAS_FIXTURE_REQUIRE(allocate(scratch_, std::max(clusterSizes->buildScratchSize, bottomSizes->buildScratchSize) + properties_.scratchAlignment, MemoryLocation::Device));
        CLAS_FIXTURE_REQUIRE(allocate(clusterInfo_, properties_.triangleBuildInfoSize, MemoryLocation::HostUpload));
        CLAS_FIXTURE_REQUIRE(upload(clusterAddress_, cluster_->deviceAddress() + offset(*cluster_, properties_.clusterStorageAlignment)));
        CLAS_FIXTURE_REQUIRE(upload(bottomAddress_, bottom_->deviceAddress() + offset(*bottom_, properties_.bottomLevelStorageAlignment)));
        const ClusterAccelerationStructureBottomLevelBuildInfo info{.clusterReferencesCount = 1,
            .clusterReferencesAddress = clusterAddress_->deviceAddress()};
        CLAS_FIXTURE_REQUIRE(upload(bottomInfo_, info));
#undef CLAS_FIXTURE_REQUIRE
        return {};
    }

    render::Result<render::BufferSlice> bottomLevel() const
    {
        if (!bottom_) { return render::makeError(render::Error::InvalidArgument); }
        return bottom_->slice({offset(*bottom_, properties_.bottomLevelStorageAlignment)});
    }

    render::Result<> record(render::CommandBuffer& commands)
    {
        using namespace render;
        auto vertices = vertices_->slice(), indices = indices_->slice();
        auto cluster = cluster_->slice({offset(*cluster_, properties_.clusterStorageAlignment)});
        auto scratch = scratch_->slice({offset(*scratch_, properties_.scratchAlignment)});
        auto clusterInfo = clusterInfo_->slice(), clusterAddress = clusterAddress_->slice();
        auto bottomInfo = bottomInfo_->slice(), bottomAddress = bottomAddress_->slice();
        if (!vertices || !indices || !cluster || !scratch || !clusterInfo || !clusterAddress || !bottomInfo || !bottomAddress) {
            return makeError(Error::InvalidArgument);
        }
        const ClusterAccelerationStructureTriangleBuildInfo input{.triangleCount = 1, .vertexCount = 3,
            .vertexBufferStride = 12, .indexBuffer = *indices, .vertexBuffer = *vertices, .destinationBuffer = *cluster};
        auto result = commands.buildClusterAccelerationStructureTriangles({.clusters = {&input, 1},
            .maxClusterTriangleCount = 128, .maxClusterVertexCount = 128, .scratchBuffer = *scratch,
            .buildInfoBuffer = *clusterInfo, .destinationAddressBuffer = *clusterAddress});
        if (!result) { return result; }
        // The CLAS address list and explicit BLAS destination are owned by this fixture.
        result = commands.retainResource(bottom_->retainAllocation());
        if (!result) { return result; }
        return commands.buildClusterAccelerationStructureBottomLevels({
            .destinationMode = ClusterAccelerationStructureDestinationMode::Explicit,
            .maxClusterCountPerAccelerationStructure = 1, .maxTotalClusterCount = 1,
            .buildInfoBuffer = *bottomInfo, .destinationAddressBuffer = *bottomAddress, .scratchBuffer = *scratch});
    }

private:
    static uint64_t offset(const render::Buffer& buffer, uint64_t alignment)
    {
        return (0 - buffer.deviceAddress()) & (alignment - 1);
    }
    render::ClusterAccelerationStructureProperties properties_;
    std::unique_ptr<render::Buffer> vertices_, indices_, cluster_, bottom_, scratch_;
    std::unique_ptr<render::Buffer> clusterInfo_, clusterAddress_, bottomInfo_, bottomAddress_;
};

} // namespace metallic::tests::bench
