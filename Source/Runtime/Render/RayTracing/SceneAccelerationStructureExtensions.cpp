#include "Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render {

namespace {

struct RayTracingVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct ClusterPrimitiveRange {
    uint32_t firstCluster = 0;
    uint32_t clusterCount = 0;
};

struct ClusterBuildInput {
    uint32_t renderPrimitiveIndex = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t triangleCount = 0;
    bool opaque = true;
};

struct ClusterBlasInstanceInput {
    uint32_t renderPrimitiveIndex = 0;
    ClusterPrimitiveRange clusters;
    float4x4 worldMatrix = float4x4::Identity();
};

struct ClusterSceneInputs {
    std::vector<RayTracingVertex> vertices;
    std::vector<uint8_t> indices;
    std::vector<ClusterBuildInput> clusters;
    std::vector<ClusterBlasInstanceInput> instances;
    std::vector<ClusterPrimitiveRange> primitiveSelectedRanges;
    uint64_t triangleCount = 0;
};

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t checkedByteSize(uint64_t count, uint64_t stride)
{
    if (count == 0 || stride == 0 || count > std::numeric_limits<uint64_t>::max() / stride) {
        return 0;
    }
    return count * stride;
}

std::string resultMessage(const char* action, Result<> result)
{
    return std::string(action) + " returned " + resultToString(result);
}

void copyTransform(float (&destination)[3][4], const float4x4& source)
{
    destination[0][0] = source.a00;
    destination[0][1] = source.a01;
    destination[0][2] = source.a02;
    destination[0][3] = source.a03;
    destination[1][0] = source.a10;
    destination[1][1] = source.a11;
    destination[1][2] = source.a12;
    destination[1][3] = source.a13;
    destination[2][0] = source.a20;
    destination[2][1] = source.a21;
    destination[2][2] = source.a22;
    destination[2][3] = source.a23;
}

Result<> createBuffer(
    Device& device,
    const char* label,
    uint64_t size,
    BufferUsageBits usage,
    MemoryLocation location,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log)
{
    Result<> result = device.createBuffer(BufferDesc{
            .size = size,
            .usage = usage,
            .memoryLocation = location,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
        }).transform([&](auto rhiValue) { outBuffer = std::move(rhiValue); });
    if (!result) {
        log = resultMessage(label, result);
    }
    return result;
}

template <typename T>
Result<> uploadVector(Buffer& buffer, const std::vector<T>& values, const char* label, std::string& log)
{
    if (values.empty()) {
        return {};
    }

    void* mapped = buffer.map();
    if (mapped == nullptr) {
        log = std::string(label) + " map failed";
        return makeError(Error::Failure);
    }

    const uint64_t byteSize = static_cast<uint64_t>(values.size() * sizeof(T));
    std::memcpy(mapped, values.data(), static_cast<size_t>(byteSize));
    buffer.flush(0, byteSize);
    buffer.unmap();
    return {};
}

bool primitiveUsesAlphaMask(
    const scene::Scene& scene,
    const scene::RenderPrimitive& primitive)
{
    if (primitive.materialIndex < 0 ||
        static_cast<size_t>(primitive.materialIndex) >= scene.materials().size()) {
        return false;
    }

    return scene.materials()[static_cast<size_t>(primitive.materialIndex)].alphaMode == "MASK";
}

Result<> recordSubmitWait(
    Device& device,
    Queue& queue,
    CommandPool& commandPool,
    const char* label,
    const std::function<Result<>(CommandBuffer&)>& record,
    std::string& log)
{
    std::unique_ptr<CommandBuffer> commandBuffer;
    Result<> result = commandPool.createCommandBuffer().transform([&](auto rhiValue) { commandBuffer = std::move(rhiValue); });
    if (!result) {
        const std::string action = std::string("createCommandBuffer(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    std::unique_ptr<Fence> fence;
    result = device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); });
    if (!result) {
        const std::string action = std::string("createFence(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    result = commandBuffer->begin();
    if (result) {
        result = record(*commandBuffer);
    }
    if (result) {
        result = commandBuffer->end();
    }
    if (!result) {
        const std::string action = std::string("record(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    CommandBuffer* submittedCommandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = submittedCommandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (result) {
        result = fence->wait();
    }
    if (!result) {
        const std::string action = std::string("submit(") + label + ")";
        log = resultMessage(action.c_str(), result);
    }
    return result;
}

bool appendClusterInput(
    const scene::RenderPrimitive& primitive,
    uint32_t renderPrimitiveIndex,
    const scene::MeshletCluster& cluster,
    const std::vector<uint32_t>& meshletVertices,
    const std::vector<uint8_t>& meshletTriangles,
    bool opaque,
    ClusterSceneInputs& outInputs)
{
    if (cluster.vertexCount == 0 ||
        cluster.triangleCount == 0 ||
        cluster.vertexOffset > meshletVertices.size() ||
        cluster.triangleOffset > meshletTriangles.size() ||
        static_cast<size_t>(cluster.vertexOffset) + cluster.vertexCount > meshletVertices.size() ||
        static_cast<size_t>(cluster.triangleOffset) + static_cast<size_t>(cluster.triangleCount) * 3u >
            meshletTriangles.size() ||
        outInputs.vertices.size() + cluster.vertexCount > std::numeric_limits<uint32_t>::max() ||
        outInputs.indices.size() + static_cast<size_t>(cluster.triangleCount) * 3u >
            std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const uint32_t firstVertex = static_cast<uint32_t>(outInputs.vertices.size());
    const uint32_t firstIndex = static_cast<uint32_t>(outInputs.indices.size());
    for (uint32_t vertexIndex = 0; vertexIndex < cluster.vertexCount; ++vertexIndex) {
        const uint32_t sourceVertexIndex = meshletVertices[static_cast<size_t>(cluster.vertexOffset) + vertexIndex];
        if (sourceVertexIndex >= primitive.positions.size()) {
            return false;
        }
        const float3& position = primitive.positions[sourceVertexIndex];
        outInputs.vertices.push_back(RayTracingVertex{position.x, position.y, position.z});
    }

    for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
        const uint8_t localVertexIndex = meshletTriangles[static_cast<size_t>(cluster.triangleOffset) + index];
        if (localVertexIndex >= cluster.vertexCount) {
            return false;
        }
        outInputs.indices.push_back(localVertexIndex);
    }

    outInputs.clusters.push_back(ClusterBuildInput{
        .renderPrimitiveIndex = renderPrimitiveIndex,
        .firstVertex = firstVertex,
        .vertexCount = cluster.vertexCount,
        .firstIndex = firstIndex,
        .triangleCount = cluster.triangleCount,
        .opaque = opaque,
    });
    outInputs.triangleCount += cluster.triangleCount;
    return true;
}

ClusterPrimitiveRange selectLowestLodRange(
    uint32_t firstPrimitiveCluster,
    const scene::RenderPrimitive& primitive,
    bool usingLodClusters)
{
    if (!usingLodClusters) {
        return ClusterPrimitiveRange{
            .firstCluster = firstPrimitiveCluster,
            .clusterCount = static_cast<uint32_t>(primitive.meshletClusters.size()),
        };
    }

    for (size_t reverseIndex = primitive.meshletLodLevels.size(); reverseIndex > 0; --reverseIndex) {
        const scene::MeshletLodLevel& level = primitive.meshletLodLevels[reverseIndex - 1u];
        if (level.clusterCount != 0) {
            return ClusterPrimitiveRange{
                .firstCluster = firstPrimitiveCluster + level.clusterOffset,
                .clusterCount = level.clusterCount,
            };
        }
    }
    return ClusterPrimitiveRange{
        .firstCluster = firstPrimitiveCluster,
        .clusterCount = static_cast<uint32_t>(primitive.meshletLodClusters.size()),
    };
}

bool buildClusterSceneInputs(const scene::Scene& scene, ClusterSceneInputs& outInputs, std::string& log)
{
    outInputs = {};
    const std::vector<scene::RenderPrimitive>& renderPrimitives = scene.renderPrimitives();
    outInputs.primitiveSelectedRanges.resize(renderPrimitives.size());

    for (uint32_t primitiveIndex = 0; primitiveIndex < renderPrimitives.size(); ++primitiveIndex) {
        const scene::RenderPrimitive& primitive = renderPrimitives[primitiveIndex];
        if (primitive.mode != 4 || primitive.positions.empty()) {
            continue;
        }

        const bool usingLodClusters =
            !primitive.meshletLodClusters.empty() &&
            !primitive.meshletLodVertices.empty() &&
            !primitive.meshletLodTriangles.empty();
        const std::vector<scene::MeshletCluster>& clusters = usingLodClusters
            ? primitive.meshletLodClusters
            : primitive.meshletClusters;
        const std::vector<uint32_t>& vertices = usingLodClusters
            ? primitive.meshletLodVertices
            : primitive.meshletVertices;
        const std::vector<uint8_t>& triangles = usingLodClusters
            ? primitive.meshletLodTriangles
            : primitive.meshletTriangles;
        if (clusters.empty() || vertices.empty() || triangles.empty()) {
            continue;
        }

        const uint32_t firstPrimitiveCluster = static_cast<uint32_t>(outInputs.clusters.size());
        const bool opaque = !primitiveUsesAlphaMask(scene, primitive);
        for (const scene::MeshletCluster& cluster : clusters) {
            if (!appendClusterInput(
                    primitive,
                    primitiveIndex,
                    cluster,
                    vertices,
                    triangles,
                    opaque,
                    outInputs)) {
                log = "Scene meshlet cluster data is invalid for CLAS build.";
                return false;
            }
        }

        const uint32_t primitiveClusterCount =
            static_cast<uint32_t>(outInputs.clusters.size()) - firstPrimitiveCluster;
        if (primitiveClusterCount != 0) {
            outInputs.primitiveSelectedRanges[primitiveIndex] =
                selectLowestLodRange(firstPrimitiveCluster, primitive, usingLodClusters);
        }
    }

    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible ||
            renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= outInputs.primitiveSelectedRanges.size()) {
            continue;
        }

        const ClusterPrimitiveRange& selectedRange =
            outInputs.primitiveSelectedRanges[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (selectedRange.clusterCount == 0) {
            continue;
        }
        outInputs.instances.push_back(ClusterBlasInstanceInput{
            .renderPrimitiveIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex),
            .clusters = selectedRange,
            .worldMatrix = renderNode.worldMatrix,
        });
    }

    if (outInputs.clusters.empty() || outInputs.vertices.empty() || outInputs.indices.empty()) {
        log = "Scene contains no meshlet clusters suitable for CLAS build.";
        return false;
    }
    if (outInputs.instances.empty()) {
        log = "Scene contains no visible meshlet cluster instances.";
        return false;
    }
    return true;
}

} // namespace

struct SceneClusterAccelerationStructureBuilder::Impl {
    SceneClusterAccelerationStructureStats stats;
    std::unique_ptr<Buffer> clusterVertexBuffer;
    std::unique_ptr<Buffer> clusterIndexBuffer;
    std::unique_ptr<Buffer> clasStorageBuffer;
    std::unique_ptr<Buffer> clasBuildInfoBuffer;
    std::unique_ptr<Buffer> clasAddressBuffer;
    std::unique_ptr<Buffer> clusterReferenceBuffer;
    std::unique_ptr<Buffer> clusterBlasStorageBuffer;
    std::unique_ptr<Buffer> clusterBlasBuildInfoBuffer;
    std::unique_ptr<Buffer> clusterBlasAddressBuffer;
    std::unique_ptr<Buffer> tlasInstanceBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<RayTracingAccelerationStructure> tlas;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        stats = {};
        tlas.reset();
        scratchBuffer.reset();
        tlasInstanceBuffer.reset();
        clusterBlasAddressBuffer.reset();
        clusterBlasBuildInfoBuffer.reset();
        clusterBlasStorageBuffer.reset();
        clusterReferenceBuffer.reset();
        clasAddressBuffer.reset();
        clasBuildInfoBuffer.reset();
        clasStorageBuffer.reset();
        clusterIndexBuffer.reset();
        clusterVertexBuffer.reset();
    }
};

SceneClusterAccelerationStructureBuilder::SceneClusterAccelerationStructureBuilder()
    : impl_(std::make_unique<Impl>())
{
}

SceneClusterAccelerationStructureBuilder::~SceneClusterAccelerationStructureBuilder() = default;
SceneClusterAccelerationStructureBuilder::SceneClusterAccelerationStructureBuilder(
    SceneClusterAccelerationStructureBuilder&&) noexcept = default;
SceneClusterAccelerationStructureBuilder& SceneClusterAccelerationStructureBuilder::operator=(
    SceneClusterAccelerationStructureBuilder&&) noexcept = default;

Result<> SceneClusterAccelerationStructureBuilder::build(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    std::string& log)
{
    log.clear();
    if (!scene.valid()) {
        log = "Scene is not loaded.";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().clusterAccelerationStructure ||
        !device.capabilities().rayTracingAccelerationStructure) {
        log = "Cluster and ray tracing acceleration structure capabilities are required.";
        return makeError(Error::Unsupported);
    }

    clear();

    ClusterSceneInputs inputs;
    if (!buildClusterSceneInputs(scene, inputs, log)) {
        clear();
        return makeError(Error::Unsupported);
    }
    if (inputs.clusters.size() > std::numeric_limits<uint32_t>::max() ||
        inputs.instances.size() > std::numeric_limits<uint32_t>::max() ||
        inputs.vertices.size() > std::numeric_limits<uint32_t>::max() ||
        inputs.triangleCount > std::numeric_limits<uint32_t>::max()) {
        log = "Scene cluster acceleration-structure inputs exceed 32-bit limits.";
        clear();
        return makeError(Error::Unsupported);
    }

    ClusterAccelerationStructureProperties clusterProperties;
    Result<> result = device.queryClusterAccelerationStructureProperties().transform([&](auto rhiValue) { clusterProperties = std::move(rhiValue); });
    if (!result ||
        clusterProperties.clusterStorageAlignment == 0 ||
        clusterProperties.bottomLevelStorageAlignment == 0 ||
        clusterProperties.scratchAlignment == 0 ||
        clusterProperties.triangleBuildInfoSize == 0 ||
        clusterProperties.bottomLevelBuildInfoSize == 0) {
        log = resultMessage("queryClusterAccelerationStructureProperties", result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    uint32_t maxClusterTriangleCount = 0;
    uint32_t maxClusterVertexCount = 0;
    for (const ClusterBuildInput& cluster : inputs.clusters) {
        maxClusterTriangleCount = std::max(maxClusterTriangleCount, cluster.triangleCount);
        maxClusterVertexCount = std::max(maxClusterVertexCount, cluster.vertexCount);
    }
    const uint32_t clusterCount = static_cast<uint32_t>(inputs.clusters.size());
    const uint32_t instanceCount = static_cast<uint32_t>(inputs.instances.size());
    const uint32_t totalClusterTriangleCount = static_cast<uint32_t>(inputs.triangleCount);
    const uint32_t totalClusterVertexCount = static_cast<uint32_t>(inputs.vertices.size());

    ClusterAccelerationStructureBuildSizes singleClasSizes;
    result = device.queryClusterAccelerationStructureTriangleBuildSizes(ClusterAccelerationStructureTriangleBuildSizesDesc{
            .maxClusterTriangleCount = maxClusterTriangleCount,
            .maxClusterVertexCount = maxClusterVertexCount,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = maxClusterTriangleCount,
            .maxTotalVertexCount = maxClusterVertexCount,
            .vertexFormat = Format::Rgb32Sfloat,
            .maxAccelerationStructureCount = 1,
        }).transform([&](auto rhiValue) { singleClasSizes = std::move(rhiValue); });
    if (!result || singleClasSizes.accelerationStructureSize == 0) {
        log = resultMessage(
            "queryClusterAccelerationStructureTriangleBuildSizes(single CLAS)",
            result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    ClusterAccelerationStructureBuildSizes clasBatchSizes;
    result = device.queryClusterAccelerationStructureTriangleBuildSizes(ClusterAccelerationStructureTriangleBuildSizesDesc{
            .maxClusterTriangleCount = maxClusterTriangleCount,
            .maxClusterVertexCount = maxClusterVertexCount,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = totalClusterTriangleCount,
            .maxTotalVertexCount = totalClusterVertexCount,
            .vertexFormat = Format::Rgb32Sfloat,
            .maxAccelerationStructureCount = clusterCount,
        }).transform([&](auto rhiValue) { clasBatchSizes = std::move(rhiValue); });
    if (!result || clasBatchSizes.buildScratchSize == 0) {
        log = resultMessage(
            "queryClusterAccelerationStructureTriangleBuildSizes(CLAS batch)",
            result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    uint32_t maxClustersPerBlas = 0;
    uint64_t selectedClusterReferenceCount = 0;
    for (const ClusterBlasInstanceInput& instance : inputs.instances) {
        maxClustersPerBlas = std::max(
            maxClustersPerBlas,
            instance.clusters.clusterCount);
        selectedClusterReferenceCount += instance.clusters.clusterCount;
    }
    if (maxClustersPerBlas == 0 ||
        selectedClusterReferenceCount > std::numeric_limits<uint32_t>::max()) {
        log = "Scene cluster BLAS inputs exceed supported limits.";
        clear();
        return makeError(Error::Unsupported);
    }

    ClusterAccelerationStructureBuildSizes singleClusterBlasSizes;
    result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(ClusterAccelerationStructureBottomLevelBuildSizesDesc{
            .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .maxClusterCountPerAccelerationStructure = maxClustersPerBlas,
            .maxTotalClusterCount = maxClustersPerBlas,
            .maxAccelerationStructureCount = 1,
        }).transform([&](auto rhiValue) { singleClusterBlasSizes = std::move(rhiValue); });
    if (!result || singleClusterBlasSizes.accelerationStructureSize == 0) {
        log = resultMessage(
            "queryClusterAccelerationStructureBottomLevelBuildSizes(single BLAS)",
            result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    ClusterAccelerationStructureBuildSizes clusterBlasBatchSizes;
    result = device.queryClusterAccelerationStructureBottomLevelBuildSizes(ClusterAccelerationStructureBottomLevelBuildSizesDesc{
            .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .maxClusterCountPerAccelerationStructure = maxClustersPerBlas,
            .maxTotalClusterCount = static_cast<uint32_t>(selectedClusterReferenceCount),
            .maxAccelerationStructureCount = instanceCount,
        }).transform([&](auto rhiValue) { clusterBlasBatchSizes = std::move(rhiValue); });
    if (!result || clusterBlasBatchSizes.buildScratchSize == 0) {
        log = resultMessage(
            "queryClusterAccelerationStructureBottomLevelBuildSizes(BLAS batch)",
            result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    RayTracingAccelerationStructureBuildSizes tlasSizes;
    result = device.queryRayTracingAccelerationStructureBuildSizes(RayTracingAccelerationStructureBuildInputs{
            .type = RayTracingAccelerationStructureType::TopLevel,
            .flags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .instanceCount = instanceCount,
        }).transform([&](auto rhiValue) { tlasSizes = std::move(rhiValue); });
    if (!result ||
        tlasSizes.accelerationStructureSize == 0 ||
        tlasSizes.buildScratchSize == 0) {
        log = resultMessage(
            "queryRayTracingAccelerationStructureBuildSizes(cluster TLAS)",
            result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    RayTracingAccelerationStructureProperties rtasProperties;
    result = device.queryRayTracingAccelerationStructureProperties().transform([&](auto rhiValue) { rtasProperties = std::move(rhiValue); });
    if (!result || rtasProperties.scratchAlignment == 0) {
        log = resultMessage("queryRayTracingAccelerationStructureProperties", result);
        clear();
        return result ? makeError(Error::Failure) : result;
    }

    const uint64_t clasStride = alignUp(
        singleClasSizes.accelerationStructureSize,
        clusterProperties.clusterStorageAlignment);
    const uint64_t clusterBlasStride = alignUp(
        singleClusterBlasSizes.accelerationStructureSize,
        clusterProperties.bottomLevelStorageAlignment);
    const uint64_t clasStorageBytes = checkedByteSize(clusterCount, clasStride);
    const uint64_t clusterBlasStorageBytes =
        checkedByteSize(instanceCount, clusterBlasStride);
    if (clasStride == 0 ||
        clusterBlasStride == 0 ||
        clasStorageBytes == 0 ||
        clusterBlasStorageBytes == 0) {
        log = "Scene cluster acceleration-structure storage sizing overflowed.";
        clear();
        return makeError(Error::OutOfMemory);
    }

    result = createBuffer(
        device,
        "createBuffer(cluster vertices)",
        checkedByteSize(inputs.vertices.size(), sizeof(RayTracingVertex)),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterVertexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->clusterVertexBuffer,
        inputs.vertices,
        "cluster vertices",
        log);
    if (!result) {
        clear();
        return result;
    }

    result = createBuffer(
        device,
        "createBuffer(cluster indices)",
        inputs.indices.size(),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterIndexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->clusterIndexBuffer,
        inputs.indices,
        "cluster indices",
        log);
    if (!result) {
        clear();
        return result;
    }

    result = createBuffer(
        device,
        "createBuffer(CLAS storage)",
        clasStorageBytes,
        BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clasStorageBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(CLAS build infos)",
        checkedByteSize(clusterCount, clusterProperties.triangleBuildInfoSize),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clasBuildInfoBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(CLAS addresses)",
        checkedByteSize(clusterCount, sizeof(uint64_t)),
        BufferUsageBits::Storage |
            BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clasAddressBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }

    const uint64_t clasStorageAddress = impl_->clasStorageBuffer->deviceAddress();
    if (clasStorageAddress == 0) {
        log = "CLAS storage buffer has no device address.";
        clear();
        return makeError(Error::Failure);
    }
    std::vector<ClusterAccelerationStructureTriangleBuildInfo> clasBuildInfos;
    clasBuildInfos.reserve(clusterCount);
    for (uint32_t clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex) {
        const ClusterBuildInput& cluster = inputs.clusters[clusterIndex];
        clasBuildInfos.push_back(ClusterAccelerationStructureTriangleBuildInfo{
            .clusterId = clusterIndex,
            .triangleCount = cluster.triangleCount,
            .vertexCount = cluster.vertexCount,
            .positionTruncateBitCount = 0,
            .geometryIndex = 0,
            .indexFormat = ClusterAccelerationStructureIndexFormat::Uint8,
            .indexBufferStride = 1,
            .vertexBufferStride = sizeof(RayTracingVertex),
            .indexBuffer = impl_->clusterIndexBuffer.get(),
            .indexBufferOffset = cluster.firstIndex,
            .vertexBuffer = impl_->clusterVertexBuffer.get(),
            .vertexBufferOffset =
                static_cast<uint64_t>(cluster.firstVertex) * sizeof(RayTracingVertex),
            .destinationBuffer = impl_->clasStorageBuffer.get(),
            .destinationBufferOffset =
                static_cast<uint64_t>(clusterIndex) * clasStride,
            .destinationSize = clasStride,
            .opaque = cluster.opaque,
        });
    }

    std::vector<uint64_t> clusterReferences;
    clusterReferences.reserve(static_cast<size_t>(selectedClusterReferenceCount));
    std::vector<ClusterAccelerationStructureBottomLevelBuildInfo> clusterBlasBuildInfos;
    clusterBlasBuildInfos.reserve(instanceCount);
    for (const ClusterBlasInstanceInput& instance : inputs.instances) {
        const uint64_t firstReference = clusterReferences.size();
        for (uint32_t clusterOffset = 0;
             clusterOffset < instance.clusters.clusterCount;
             ++clusterOffset) {
            const uint32_t clusterIndex =
                instance.clusters.firstCluster + clusterOffset;
            if (clusterIndex >= clusterCount) {
                log = "Cluster BLAS references an invalid CLAS index.";
                clear();
                return makeError(Error::Failure);
            }
            clusterReferences.push_back(
                clasStorageAddress + static_cast<uint64_t>(clusterIndex) * clasStride);
        }
        clusterBlasBuildInfos.push_back(
            ClusterAccelerationStructureBottomLevelBuildInfo{
                .clusterReferencesCount = instance.clusters.clusterCount,
                .clusterReferencesStride = sizeof(uint64_t),
                .clusterReferencesAddress = firstReference,
            });
    }

    result = createBuffer(
        device,
        "createBuffer(cluster references)",
        checkedByteSize(clusterReferences.size(), sizeof(uint64_t)),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterReferenceBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->clusterReferenceBuffer,
        clusterReferences,
        "cluster references",
        log);
    if (!result) {
        clear();
        return result;
    }
    const uint64_t clusterReferenceAddress =
        impl_->clusterReferenceBuffer->deviceAddress();
    if (clusterReferenceAddress == 0) {
        log = "Cluster reference buffer has no device address.";
        clear();
        return makeError(Error::Failure);
    }
    uint64_t referenceOffset = 0;
    for (ClusterAccelerationStructureBottomLevelBuildInfo& buildInfo :
         clusterBlasBuildInfos) {
        buildInfo.clusterReferencesAddress =
            clusterReferenceAddress + referenceOffset * sizeof(uint64_t);
        referenceOffset += buildInfo.clusterReferencesCount;
    }

    result = createBuffer(
        device,
        "createBuffer(cluster BLAS storage)",
        clusterBlasStorageBytes,
        BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clusterBlasStorageBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS build infos)",
        checkedByteSize(instanceCount, clusterProperties.bottomLevelBuildInfoSize),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterBlasBuildInfoBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->clusterBlasBuildInfoBuffer,
        clusterBlasBuildInfos,
        "cluster BLAS build infos",
        log);
    if (!result) {
        clear();
        return result;
    }

    const uint64_t clusterBlasStorageAddress =
        impl_->clusterBlasStorageBuffer->deviceAddress();
    if (clusterBlasStorageAddress == 0) {
        log = "Cluster BLAS storage buffer has no device address.";
        clear();
        return makeError(Error::Failure);
    }
    std::vector<uint64_t> clusterBlasAddresses(instanceCount);
    for (uint32_t instanceIndex = 0; instanceIndex < instanceCount; ++instanceIndex) {
        clusterBlasAddresses[instanceIndex] =
            clusterBlasStorageAddress +
            static_cast<uint64_t>(instanceIndex) * clusterBlasStride;
    }
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS addresses)",
        checkedByteSize(instanceCount, sizeof(uint64_t)),
        BufferUsageBits::Storage |
            BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterBlasAddressBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->clusterBlasAddressBuffer,
        clusterBlasAddresses,
        "cluster BLAS addresses",
        log);
    if (!result) {
        clear();
        return result;
    }

    std::vector<RayTracingGpuInstance> tlasInstances(instanceCount);
    for (uint32_t instanceIndex = 0; instanceIndex < instanceCount; ++instanceIndex) {
        const ClusterBlasInstanceInput& source = inputs.instances[instanceIndex];
        RayTracingGpuInstance& destination = tlasInstances[instanceIndex];
        copyTransform(destination.transform, source.worldMatrix);
        destination.customIndexAndMask =
            (source.renderPrimitiveIndex & 0x00ffffffu) | (0xffu << 24u);
        destination.shaderBindingTableRecordOffsetAndFlags =
            static_cast<uint32_t>(
                RayTracingInstanceFlags::TriangleFacingCullDisable) << 24u;
        destination.accelerationStructureReference =
            clusterBlasAddresses[instanceIndex];
    }
    result = createBuffer(
        device,
        "createBuffer(cluster TLAS instances)",
        checkedByteSize(instanceCount, sizeof(RayTracingGpuInstance)),
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->tlasInstanceBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(
        *impl_->tlasInstanceBuffer,
        tlasInstances,
        "cluster TLAS instances",
        log);
    if (!result) {
        clear();
        return result;
    }

    result = device.createRayTracingAccelerationStructure(RayTracingAccelerationStructureDesc{
            .type = RayTracingAccelerationStructureType::TopLevel,
            .buildFlags = RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
            .size = tlasSizes.accelerationStructureSize,
        }).transform([&](auto rhiValue) { impl_->tlas = std::move(rhiValue); });
    if (!result) {
        log = resultMessage(
            "createRayTracingAccelerationStructure(cluster TLAS)",
            result);
        clear();
        return result;
    }

    const uint64_t maxScratchSize = std::max(
        clasBatchSizes.buildScratchSize,
        std::max(
            clusterBlasBatchSizes.buildScratchSize,
            tlasSizes.buildScratchSize));
    const uint64_t maxScratchAlignment = std::max(
        clusterProperties.scratchAlignment,
        rtasProperties.scratchAlignment);
    const uint64_t scratchBytes =
        maxScratchSize + maxScratchAlignment - 1u;
    result = createBuffer(
        device,
        "createBuffer(cluster acceleration-structure scratch)",
        scratchBytes,
        BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    const uint64_t scratchAddress = impl_->scratchBuffer->deviceAddress();
    const uint64_t alignedScratchAddress = alignUp(
        scratchAddress,
        clusterProperties.scratchAlignment);
    const uint64_t scratchOffset = alignedScratchAddress - scratchAddress;
    if (scratchAddress == 0 ||
        scratchOffset >= scratchBytes ||
        maxScratchSize > scratchBytes - scratchOffset) {
        log = "Cluster acceleration-structure scratch alignment is invalid.";
        clear();
        return makeError(Error::Failure);
    }

    std::unique_ptr<CommandPool> commandPool;
    result = device.createCommandPool(queue).transform([&](auto rhiValue) { commandPool = std::move(rhiValue); });
    if (!result) {
        log = resultMessage(
            "createCommandPool(cluster acceleration-structure build)",
            result);
        clear();
        return result;
    }

    result = recordSubmitWait(
        device,
        queue,
        *commandPool,
        "cluster acceleration-structure build",
        [&](CommandBuffer& commandBuffer) -> Result<> {
            Result<> buildResult =
                commandBuffer.buildClusterAccelerationStructureTriangles(
                    ClusterAccelerationStructureTriangleBuildDesc{
                        .clusters = clasBuildInfos.data(),
                        .clusterCount = clusterCount,
                        .maxClusterTriangleCount = maxClusterTriangleCount,
                        .maxClusterVertexCount = maxClusterVertexCount,
                        .maxClusterUniqueGeometryCount = 1,
                        .maxGeometryIndexValue = 0,
                        .minPositionTruncateBitCount = 0,
                        .vertexFormat = Format::Rgb32Sfloat,
                        .scratchBuffer = impl_->scratchBuffer.get(),
                        .scratchBufferOffset = scratchOffset,
                        .buildInfoBuffer = impl_->clasBuildInfoBuffer.get(),
                        .destinationAddressBuffer = impl_->clasAddressBuffer.get(),
                    });
            if (!buildResult) {
                return buildResult;
            }

            buildResult =
                commandBuffer.buildClusterAccelerationStructureBottomLevels(
                    ClusterAccelerationStructureBottomLevelBuildDesc{
                        .flags =
                            RayTracingAccelerationStructureBuildFlags::PreferFastTrace,
                        .destinationMode =
                            ClusterAccelerationStructureDestinationMode::Explicit,
                        .maxClusterCountPerAccelerationStructure =
                            maxClustersPerBlas,
                        .maxTotalClusterCount =
                            static_cast<uint32_t>(selectedClusterReferenceCount),
                        .maxAccelerationStructureCount = instanceCount,
                        .buildInfoBuffer =
                            impl_->clusterBlasBuildInfoBuffer.get(),
                        .buildInfoStride =
                            clusterProperties.bottomLevelBuildInfoSize,
                        .buildInfoSize = checkedByteSize(
                            instanceCount,
                            clusterProperties.bottomLevelBuildInfoSize),
                        .destinationAddressBuffer =
                            impl_->clusterBlasAddressBuffer.get(),
                        .destinationAddressStride = sizeof(uint64_t),
                        .destinationAddressSize =
                            checkedByteSize(instanceCount, sizeof(uint64_t)),
                        .scratchBuffer = impl_->scratchBuffer.get(),
                        .scratchBufferOffset = scratchOffset,
                    });
            if (!buildResult) {
                return buildResult;
            }

            return commandBuffer.buildRayTracingAccelerationStructure(
                RayTracingAccelerationStructureBuildDesc{
                    .destination = impl_->tlas.get(),
                    .instanceBuffer = impl_->tlasInstanceBuffer.get(),
                    .instanceCount = instanceCount,
                    .scratchBuffer = impl_->scratchBuffer.get(),
                });
        },
        log);
    if (!result) {
        clear();
        return result;
    }

    const uint64_t geometryBytes =
        checkedByteSize(inputs.vertices.size(), sizeof(RayTracingVertex)) +
        inputs.indices.size() +
        checkedByteSize(clusterReferences.size(), sizeof(uint64_t)) +
        checkedByteSize(instanceCount, sizeof(RayTracingGpuInstance));
    impl_->stats = SceneClusterAccelerationStructureStats{
        .clasCount = clusterCount,
        .clusterBlasCount = instanceCount,
        .instanceCount = instanceCount,
        .clusterTriangleCount = inputs.triangleCount,
        .clusterVertexCount = inputs.vertices.size(),
        .clusterIndexBytes = inputs.indices.size(),
        .selectedClusterReferenceCount = selectedClusterReferenceCount,
        .geometryBytes = geometryBytes,
        .clasBytes = clasStorageBytes,
        .clusterBlasBytes = clusterBlasStorageBytes,
        .tlasBytes = tlasSizes.accelerationStructureSize,
        .accelerationStructureBytes =
            clasStorageBytes +
            clusterBlasStorageBytes +
            tlasSizes.accelerationStructureSize,
        .scratchBytes = scratchBytes,
    };

    log = "Built cluster acceleration structures: " +
        std::to_string(impl_->stats.clasCount) +
        " CLAS, " +
        std::to_string(impl_->stats.clusterBlasCount) +
        " cluster BLAS, " +
        std::to_string(impl_->stats.instanceCount) +
        " TLAS instances.";
    return {};
}

void SceneClusterAccelerationStructureBuilder::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool SceneClusterAccelerationStructureBuilder::valid() const
{
    return impl_ != nullptr &&
        impl_->tlas != nullptr &&
        impl_->tlas->valid();
}

RayTracingAccelerationStructure*
SceneClusterAccelerationStructureBuilder::accelerationStructure() const
{
    return valid() ? impl_->tlas.get() : nullptr;
}

const SceneClusterAccelerationStructureStats&
SceneClusterAccelerationStructureBuilder::stats() const
{
    static const SceneClusterAccelerationStructureStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

} // namespace metallic::render
