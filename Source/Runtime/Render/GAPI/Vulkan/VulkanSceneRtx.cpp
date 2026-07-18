/*
 * Vulkan acceleration-structure build flow adapted from NVIDIA nvpro_core2 nvvk
 * acceleration_structures.cpp/.hpp.
 *
 * Copyright (c) 2014-2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Profiling/NsightAftermath.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render::vulkan {
namespace {

using RtxLogClock = std::chrono::steady_clock;

double rtxElapsedMilliseconds(RtxLogClock::time_point begin)
{
    return std::chrono::duration<double, std::milli>(RtxLogClock::now() - begin).count();
}

class RtxLogScope {
public:
    explicit RtxLogScope(std::string label)
        : label_(std::move(label))
    {
        spdlog::info("[RTX] Begin {}", label_);
    }

    ~RtxLogScope()
    {
        spdlog::info("[RTX] End {} in {:.2f} ms", label_, rtxElapsedMilliseconds(begin_));
    }

private:
    std::string label_;
    RtxLogClock::time_point begin_ = RtxLogClock::now();
};

constexpr VkBuildAccelerationStructureFlagsKHR kBuildFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
constexpr VkDeviceSize kDefaultScratchAlignment = 256;

struct RtxVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct PrimitiveInput {
    uint32_t renderPrimitiveIndex = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    uint32_t triangleCount = 0;
    bool opaque = true;
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
    std::vector<RtxVertex> vertices;
    std::vector<uint8_t> indices;
    std::vector<ClusterBuildInput> clusters;
    std::vector<ClusterBlasInstanceInput> instances;
    std::vector<ClusterPrimitiveRange> primitiveSelectedRanges;
    uint64_t triangleCount = 0;
};

struct BuiltBlas {
    std::unique_ptr<Buffer> storage;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
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

std::string resultMessage(const char* action, Result result)
{
    return std::string(action) + " returned " + resultToString(result);
}

Result resultFromVk(VkResult result)
{
    switch (result) {
    case VK_SUCCESS:
        return {};
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return makeError(Error::OutOfMemory);
    case VK_ERROR_DEVICE_LOST:
        profiling::handleNsightAftermathDeviceLost();
        return makeError(Error::DeviceLost);
    default:
        return makeError(Error::Failure);
    }
}

VkTransformMatrixKHR toVkTransform(const float4x4& matrix)
{
    VkTransformMatrixKHR transform{};
    transform.matrix[0][0] = matrix.a00;
    transform.matrix[0][1] = matrix.a01;
    transform.matrix[0][2] = matrix.a02;
    transform.matrix[0][3] = matrix.a03;
    transform.matrix[1][0] = matrix.a10;
    transform.matrix[1][1] = matrix.a11;
    transform.matrix[1][2] = matrix.a12;
    transform.matrix[1][3] = matrix.a13;
    transform.matrix[2][0] = matrix.a20;
    transform.matrix[2][1] = matrix.a21;
    transform.matrix[2][2] = matrix.a22;
    transform.matrix[2][3] = matrix.a23;
    return transform;
}

void accelerationStructureBarrier(VkCommandBuffer commandBuffer)
{
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    };
    VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void memoryBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags2 srcStage,
    VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 dstAccess)
{
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
    };
    VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

Result createBuffer(
    Device& device,
    const char* label,
    uint64_t size,
    BufferUsageBits usage,
    MemoryLocation location,
    std::unique_ptr<Buffer>& outBuffer,
    std::string& log)
{
    Result result = device.createBuffer(
        BufferDesc{
            .size = size,
            .usage = usage,
            .memoryLocation = location,
        },
        outBuffer);
    if (!result) {
        log = resultMessage(label, result);
    }
    return result;
}

template <typename T>
Result uploadVector(Buffer& buffer, const std::vector<T>& values, const char* label, std::string& log)
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

template <typename T>
Result readbackVector(Buffer& buffer, size_t count, const char* label, std::vector<T>& outValues, std::string& log)
{
    outValues.clear();
    if (count == 0) {
        return {};
    }

    const uint64_t byteSize = checkedByteSize(count, sizeof(T));
    if (byteSize == 0 || byteSize > buffer.desc().size) {
        log = std::string(label) + " readback size is invalid";
        return makeError(Error::InvalidArgument);
    }

    buffer.invalidate(0, byteSize);
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        log = std::string(label) + " map failed";
        return makeError(Error::Failure);
    }

    outValues.resize(count);
    std::memcpy(outValues.data(), mapped, static_cast<size_t>(byteSize));
    buffer.unmap();
    return {};
}

VkAccelerationStructureGeometryKHR makeBlasGeometry(
    VkDeviceAddress vertexAddress,
    VkDeviceAddress indexAddress,
    const PrimitiveInput& input)
{
    const VkGeometryFlagsKHR geometryFlags = input.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .vertexData = VkDeviceOrHostAddressConstKHR{
            .deviceAddress = vertexAddress + static_cast<VkDeviceAddress>(input.firstVertex) * sizeof(RtxVertex),
        },
        .vertexStride = sizeof(RtxVertex),
        .maxVertex = input.vertexCount - 1,
        .indexType = VK_INDEX_TYPE_UINT32,
        .indexData = VkDeviceOrHostAddressConstKHR{
            .deviceAddress = indexAddress + static_cast<VkDeviceAddress>(input.firstIndex) * sizeof(uint32_t),
        },
    };

    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .flags = geometryFlags,
    };
    geometry.geometry.triangles = triangles;
    return geometry;
}

VkAccelerationStructureBuildRangeInfoKHR makeBuildRange(uint32_t primitiveCount)
{
    return VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
}

VkAccelerationStructureBuildGeometryInfoKHR makeBuildInfo(
    VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry)
{
    return VkAccelerationStructureBuildGeometryInfoKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = type,
        .flags = kBuildFlags |
            (type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                    ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                    : 0),
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1,
        .pGeometries = &geometry,
    };
}

VkAccelerationStructureBuildSizesInfoKHR queryBuildSize(
    VkDevice device,
    VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
    uint32_t primitiveCount)
{
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &primitiveCount,
        &sizeInfo);
    return sizeInfo;
}

VkAccelerationStructureCreateInfoKHR makeAccelerationStructureCreateInfo(
    VkBuffer buffer,
    VkAccelerationStructureTypeKHR type,
    VkDeviceSize size)
{
    return VkAccelerationStructureCreateInfoKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = buffer,
        .size = size,
        .type = type,
    };
}

VkDeviceAddress accelerationStructureAddress(VkDevice device, VkAccelerationStructureKHR accelerationStructure)
{
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = accelerationStructure,
    };
    return vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);
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

VkDeviceSize scratchAlignment(VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    };
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &accelerationProperties,
    };
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    return accelerationProperties.minAccelerationStructureScratchOffsetAlignment != 0
        ? accelerationProperties.minAccelerationStructureScratchOffsetAlignment
        : kDefaultScratchAlignment;
}

VkDeviceSize clusterScratchAlignment(VkPhysicalDevice physicalDevice)
{
#ifdef VK_NV_cluster_acceleration_structure
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusterProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    };
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &clusterProperties,
    };
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    return clusterProperties.clusterScratchByteAlignment != 0
        ? clusterProperties.clusterScratchByteAlignment
        : kDefaultScratchAlignment;
#else
    (void)physicalDevice;
    return kDefaultScratchAlignment;
#endif
}

VkDeviceSize clusterStorageAlignment(VkPhysicalDevice physicalDevice)
{
#ifdef VK_NV_cluster_acceleration_structure
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clusterProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    };
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &clusterProperties,
    };
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
    return clusterProperties.clusterByteAlignment != 0
        ? clusterProperties.clusterByteAlignment
        : 1;
#else
    (void)physicalDevice;
    return 1;
#endif
}

Result recordSubmitWait(
    Device& device,
    Queue& queue,
    CommandPool& commandPool,
    const char* label,
    const std::function<void(VkCommandBuffer)>& record,
    std::string& log)
{
    std::unique_ptr<CommandBuffer> commandBuffer;
    Result result = commandPool.createCommandBuffer(commandBuffer);
    if (!result) {
        const std::string action = std::string("createCommandBuffer(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    std::unique_ptr<Fence> fence;
    result = device.createFence(false, fence);
    if (!result) {
        const std::string action = std::string("createFence(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    result = commandBuffer->begin();
    if (!result) {
        const std::string action = std::string("CommandBuffer::begin(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    VkCommandBuffer vkCommandBuffer = nativeCommandBuffer(*commandBuffer);
    if (vkCommandBuffer == VK_NULL_HANDLE) {
        log = std::string(label) + " command buffer is unavailable";
        return makeError(Error::Failure);
    }
    record(vkCommandBuffer);

    result = commandBuffer->end();
    if (!result) {
        const std::string action = std::string("CommandBuffer::end(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    CommandBuffer* submittedCommandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = submittedCommandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        const std::string action = std::string("Queue::submit(") + label + ")";
        log = resultMessage(action.c_str(), result);
        return result;
    }

    result = fence->wait();
    if (!result) {
        const std::string action = std::string("Fence::wait(") + label + ")";
        log = resultMessage(action.c_str(), result);
    }
    return result;
}

Result createAccelerationStructure(
    Device& device,
    VkDevice vkDevice,
    VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureBuildSizesInfoKHR& sizeInfo,
    std::unique_ptr<Buffer>& outStorage,
    VkAccelerationStructureKHR& outHandle,
    VkDeviceAddress& outAddress,
    std::string& log)
{
    Result result = createBuffer(
        device,
        "createBuffer(acceleration structure storage)",
        sizeInfo.accelerationStructureSize,
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        outStorage,
        log);
    if (!result) {
        return result;
    }

    const NativeBuffer nativeStorage = nativeBuffer(*outStorage);
    if (nativeStorage.buffer == VK_NULL_HANDLE) {
        log = "acceleration structure storage buffer is unavailable";
        return makeError(Error::Failure);
    }

    VkAccelerationStructureCreateInfoKHR createInfo = makeAccelerationStructureCreateInfo(
        nativeStorage.buffer,
        type,
        sizeInfo.accelerationStructureSize);
    const VkResult vkResult = vkCreateAccelerationStructureKHR(vkDevice, &createInfo, nullptr, &outHandle);
    if (vkResult != VK_SUCCESS) {
        log = std::string("vkCreateAccelerationStructureKHR returned ") + std::to_string(static_cast<int>(vkResult));
        return resultFromVk(vkResult);
    }

    outAddress = accelerationStructureAddress(vkDevice, outHandle);
    if (outAddress == 0) {
        log = "vkGetAccelerationStructureDeviceAddressKHR returned 0";
        return makeError(Error::Failure);
    }

    return {};
}

struct PtlasPartitioning {
    uint32_t partitionCount = 1;
    uint32_t maxInstancesPerPartition = 0;
};

uint32_t ptlasPartitionAxis(size_t instanceCount)
{
    if (instanceCount >= 16) {
        return 4;
    }
    if (instanceCount >= 4) {
        return 2;
    }
    return 1;
}

PtlasPartitioning assignPtlasSpatialPartitions(
    std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV>& instances,
    const std::vector<float4x4>& instanceMatrices)
{
    if (instances.empty()) {
        return {};
    }
    if (instances.size() != instanceMatrices.size()) {
        for (VkPartitionedAccelerationStructureWriteInstanceDataNV& instance : instances) {
            instance.partitionIndex = 0;
        }
        return PtlasPartitioning{
            .partitionCount = 1,
            .maxInstancesPerPartition = static_cast<uint32_t>(instances.size()),
        };
    }

    float minX = instanceMatrices.front().a03;
    float maxX = minX;
    float minZ = instanceMatrices.front().a23;
    float maxZ = minZ;
    for (const float4x4& matrix : instanceMatrices) {
        minX = std::min(minX, matrix.a03);
        maxX = std::max(maxX, matrix.a03);
        minZ = std::min(minZ, matrix.a23);
        maxZ = std::max(maxZ, matrix.a23);
    }

    constexpr float kPartitionEpsilon = 1.0e-5f;
    const float spanX = maxX - minX;
    const float spanZ = maxZ - minZ;
    uint32_t axis = ptlasPartitionAxis(instances.size());
    if (spanX <= kPartitionEpsilon && spanZ <= kPartitionEpsilon) {
        axis = 1;
    }

    const uint32_t partitionCount = axis * axis;
    std::vector<uint32_t> counts(partitionCount, 0);
    for (size_t index = 0; index < instances.size(); ++index) {
        const float4x4& matrix = instanceMatrices[index];
        const uint32_t x = spanX > kPartitionEpsilon
            ? std::min(
                axis - 1u,
                static_cast<uint32_t>(((matrix.a03 - minX) / spanX) * static_cast<float>(axis)))
            : 0;
        const uint32_t z = spanZ > kPartitionEpsilon
            ? std::min(
                axis - 1u,
                static_cast<uint32_t>(((matrix.a23 - minZ) / spanZ) * static_cast<float>(axis)))
            : 0;
        const uint32_t partitionIndex = z * axis + x;
        instances[index].partitionIndex = partitionIndex;
        ++counts[partitionIndex];
    }

    return PtlasPartitioning{
        .partitionCount = partitionCount,
        .maxInstancesPerPartition = *std::max_element(counts.begin(), counts.end()),
    };
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
        outInputs.vertices.push_back(RtxVertex{position.x, position.y, position.z});
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

VkDescriptorType descriptorTypeFor(SceneRayQueryBindingKind kind)
{
    switch (kind) {
    case SceneRayQueryBindingKind::AccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case SceneRayQueryBindingKind::PartitionedAccelerationStructure:
#ifdef VK_NV_partitioned_acceleration_structure
        return VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV;
#else
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
#endif
    case SceneRayQueryBindingKind::StorageImage:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case SceneRayQueryBindingKind::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case SceneRayQueryBindingKind::SampledImage:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

const SceneRayQueryDispatchBinding* findDispatchBinding(
    const SceneRayQueryDispatchDesc& desc,
    uint32_t binding)
{
    if (desc.bindings == nullptr) {
        return nullptr;
    }
    for (uint32_t index = 0; index < desc.bindingCount; ++index) {
        if (desc.bindings[index].binding == binding) {
            return &desc.bindings[index];
        }
    }
    return nullptr;
}

bool hasDuplicateBindings(const SceneRayQueryBindingDesc* bindings, uint32_t bindingCount)
{
    if (bindings == nullptr) {
        return bindingCount != 0;
    }
    for (uint32_t lhs = 0; lhs < bindingCount; ++lhs) {
        for (uint32_t rhs = lhs + 1; rhs < bindingCount; ++rhs) {
            if (bindings[lhs].binding == bindings[rhs].binding) {
                return true;
            }
        }
    }
    return false;
}

void addPoolSize(
    std::array<VkDescriptorPoolSize, 5>& poolSizes,
    uint32_t& poolSizeCount,
    VkDescriptorType type,
    uint32_t descriptorCount)
{
    for (uint32_t index = 0; index < poolSizeCount; ++index) {
        if (poolSizes[index].type == type) {
            poolSizes[index].descriptorCount += descriptorCount;
            return;
        }
    }
    poolSizes[poolSizeCount++] = VkDescriptorPoolSize{
        .type = type,
        .descriptorCount = descriptorCount,
    };
}

} // namespace

struct SceneRtxBuilder::Impl {
    VkDevice device = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkDeviceAddress tlasAddress = 0;
    SceneRtxStats stats;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<Buffer> tlasStorage;
    std::vector<BuiltBlas> blases;
    std::vector<int32_t> primitiveToBlas;
    uint64_t sourceTransformRevision = 0;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        if (device != VK_NULL_HANDLE) {
            if (tlas != VK_NULL_HANDLE) {
                vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
                tlas = VK_NULL_HANDLE;
            }
            for (BuiltBlas& blas : blases) {
                if (blas.handle != VK_NULL_HANDLE) {
                    vkDestroyAccelerationStructureKHR(device, blas.handle, nullptr);
                    blas.handle = VK_NULL_HANDLE;
                }
            }
        }

        tlasAddress = 0;
        stats = {};
        blases.clear();
        primitiveToBlas.clear();
        sourceTransformRevision = 0;
        tlasStorage.reset();
        scratchBuffer.reset();
        instanceBuffer.reset();
        indexBuffer.reset();
        vertexBuffer.reset();
        device = VK_NULL_HANDLE;
    }
};

SceneRtxBuilder::SceneRtxBuilder()
    : impl_(std::make_unique<Impl>())
{
}

SceneRtxBuilder::~SceneRtxBuilder() = default;
SceneRtxBuilder::SceneRtxBuilder(SceneRtxBuilder&&) noexcept = default;
SceneRtxBuilder& SceneRtxBuilder::operator=(SceneRtxBuilder&&) noexcept = default;

Result SceneRtxBuilder::build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log)
{
    log.clear();
    RtxLogScope buildScope("SceneRtxBuilder build");
    if (!scene.valid()) {
        log = "Scene is not loaded.";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayTracingAccelerationStructure) {
        log = "Vulkan ray tracing acceleration structure capability is unavailable.";
        return makeError(Error::Unsupported);
    }

    NativeDevice nativeDeviceInfo = nativeDevice(device);
    NativeQueue nativeQueueInfo = nativeQueue(queue);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE || nativeQueueInfo.queue == VK_NULL_HANDLE) {
        log = "Vulkan native device or queue is unavailable.";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(nativeDeviceInfo.device);
    spdlog::info(
        "[RTX] Scene input renderPrimitives={} renderNodes={}",
        scene.renderPrimitives().size(),
        scene.renderNodes().size());

    clear();
    impl_->device = nativeDeviceInfo.device;

    const std::vector<scene::RenderPrimitive>& renderPrimitives = scene.renderPrimitives();
    std::vector<int32_t> primitiveToBlas(renderPrimitives.size(), -1);
    std::vector<PrimitiveInput> primitiveInputs;
    std::vector<RtxVertex> vertices;
    std::vector<uint32_t> indices;

    const auto collectInputsBegin = RtxLogClock::now();
    for (uint32_t primitiveIndex = 0; primitiveIndex < renderPrimitives.size(); ++primitiveIndex) {
        const scene::RenderPrimitive& primitive = renderPrimitives[primitiveIndex];
        if (primitive.mode != 4 || primitive.positions.size() < 3) {
            continue;
        }

        const uint64_t sourceIndexCount = primitive.indices.empty()
            ? (primitive.positions.size() / 3) * 3
            : (primitive.indices.size() / 3) * 3;
        if (sourceIndexCount < 3 ||
            sourceIndexCount > std::numeric_limits<uint32_t>::max() ||
            primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        PrimitiveInput input{
            .renderPrimitiveIndex = primitiveIndex,
            .firstVertex = static_cast<uint32_t>(vertices.size()),
            .vertexCount = static_cast<uint32_t>(primitive.positions.size()),
            .firstIndex = static_cast<uint32_t>(indices.size()),
            .indexCount = static_cast<uint32_t>(sourceIndexCount),
            .triangleCount = static_cast<uint32_t>(sourceIndexCount / 3),
            .opaque = !primitiveUsesAlphaMask(scene, primitive),
        };

        for (const float3& position : primitive.positions) {
            vertices.push_back(RtxVertex{position.x, position.y, position.z});
        }

        if (primitive.indices.empty()) {
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                indices.push_back(index);
            }
        } else {
            bool indicesValid = true;
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                const uint32_t sourceIndex = primitive.indices[index];
                if (sourceIndex >= input.vertexCount) {
                    indicesValid = false;
                    break;
                }
                indices.push_back(sourceIndex);
            }
            if (!indicesValid) {
                vertices.resize(input.firstVertex);
                indices.resize(input.firstIndex);
                continue;
            }
        }

        primitiveToBlas[primitiveIndex] = static_cast<int32_t>(primitiveInputs.size());
        primitiveInputs.push_back(input);
    }
    spdlog::info(
        "[RTX] Collected BLAS inputs primitives={} vertices={} indices={} in {:.2f} ms",
        primitiveInputs.size(),
        vertices.size(),
        indices.size(),
        rtxElapsedMilliseconds(collectInputsBegin));

    if (primitiveInputs.empty() || vertices.empty() || indices.empty()) {
        log = "Scene contains no triangle primitives suitable for RTX acceleration structures.";
        clear();
        return makeError(Error::Unsupported);
    }

    const auto uploadGeometryBegin = RtxLogClock::now();
    Result result = createBuffer(
        device,
        "createBuffer(RTX vertices)",
        vertices.size() * sizeof(RtxVertex),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->vertexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->vertexBuffer, vertices, "RTX vertices", log);
    if (!result) {
        clear();
        return result;
    }

    result = createBuffer(
        device,
        "createBuffer(RTX indices)",
        indices.size() * sizeof(uint32_t),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->indexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->indexBuffer, indices, "RTX indices", log);
    if (!result) {
        clear();
        return result;
    }
    spdlog::info(
        "[RTX] Uploaded RTX geometry buffers vertexBytes={} indexBytes={} in {:.2f} ms",
        vertices.size() * sizeof(RtxVertex),
        indices.size() * sizeof(uint32_t),
        rtxElapsedMilliseconds(uploadGeometryBegin));

    const NativeBuffer nativeVertexBuffer = nativeBuffer(*impl_->vertexBuffer);
    const NativeBuffer nativeIndexBuffer = nativeBuffer(*impl_->indexBuffer);
    if (nativeVertexBuffer.address == 0 || nativeIndexBuffer.address == 0) {
        log = "RTX geometry buffers do not have device addresses.";
        clear();
        return makeError(Error::Failure);
    }

    const auto createBlasBegin = RtxLogClock::now();
    impl_->blases.resize(primitiveInputs.size());
    VkDeviceSize maxScratchSize = 0;
    for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
        const PrimitiveInput& input = primitiveInputs[blasIndex];
        VkAccelerationStructureGeometryKHR geometry = makeBlasGeometry(
            nativeVertexBuffer.address,
            nativeIndexBuffer.address,
            input);
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometry);
        BuiltBlas& blas = impl_->blases[blasIndex];
        blas.sizeInfo = queryBuildSize(nativeDeviceInfo.device, buildInfo, input.triangleCount);
        maxScratchSize = std::max(maxScratchSize, blas.sizeInfo.buildScratchSize);

        result = createAccelerationStructure(
            device,
            nativeDeviceInfo.device,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            blas.sizeInfo,
            blas.storage,
            blas.handle,
            blas.address,
            log);
        if (!result) {
            clear();
            return result;
        }
    }
    spdlog::info(
        "[RTX] Created {} BLAS objects in {:.2f} ms",
        impl_->blases.size(),
        rtxElapsedMilliseconds(createBlasBegin));

    const auto collectInstancesBegin = RtxLogClock::now();
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(scene.renderNodes().size());
    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible ||
            renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveToBlas.size()) {
            continue;
        }

        const int32_t blasIndex = primitiveToBlas[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (blasIndex < 0 || static_cast<size_t>(blasIndex) >= impl_->blases.size()) {
            continue;
        }

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = toVkTransform(renderNode.worldMatrix);
        instance.instanceCustomIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex) & 0x00ffffffu;
        instance.mask = 0xff;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = impl_->blases[static_cast<size_t>(blasIndex)].address;
        instances.push_back(instance);
    }
    spdlog::info(
        "[RTX] Collected {} TLAS instances in {:.2f} ms",
        instances.size(),
        rtxElapsedMilliseconds(collectInstancesBegin));

    if (instances.empty()) {
        log = "Scene contains no visible RTX instances.";
        clear();
        return makeError(Error::Unsupported);
    }

    const auto createTlasInputsBegin = RtxLogClock::now();
    result = createBuffer(
        device,
        "createBuffer(RTX instances)",
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->instanceBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->instanceBuffer, instances, "RTX instances", log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeInstanceBuffer = nativeBuffer(*impl_->instanceBuffer);
    if (nativeInstanceBuffer.address == 0) {
        log = "RTX instance buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = VkDeviceOrHostAddressConstKHR{
            .deviceAddress = nativeInstanceBuffer.address,
        },
    };
    VkAccelerationStructureGeometryKHR tlasGeometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    };
    tlasGeometry.geometry.instances = instancesData;
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo = makeBuildInfo(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlasGeometry);
    VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo = queryBuildSize(
        nativeDeviceInfo.device,
        tlasBuildInfo,
        static_cast<uint32_t>(instances.size()));
    maxScratchSize = std::max(
        maxScratchSize,
        std::max(tlasSizeInfo.buildScratchSize, tlasSizeInfo.updateScratchSize));

    result = createAccelerationStructure(
        device,
        nativeDeviceInfo.device,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlasSizeInfo,
        impl_->tlasStorage,
        impl_->tlas,
        impl_->tlasAddress,
        log);
    if (!result) {
        clear();
        return result;
    }

    const VkDeviceSize alignment = scratchAlignment(nativeDeviceInfo.physicalDevice);
    const VkDeviceSize scratchSize = maxScratchSize + alignment - 1;
    result = createBuffer(
        device,
        "createBuffer(RTX scratch)",
        scratchSize,
        BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeScratchBuffer = nativeBuffer(*impl_->scratchBuffer);
    const VkDeviceAddress scratchAddress = alignUp(nativeScratchBuffer.address, alignment);
    if (nativeScratchBuffer.address == 0 ||
        scratchAddress == 0 ||
        scratchAddress + maxScratchSize > nativeScratchBuffer.address + nativeScratchBuffer.size) {
        log = "RTX scratch buffer does not provide a valid aligned device address.";
        clear();
        return makeError(Error::Failure);
    }
    spdlog::info(
        "[RTX] Created TLAS, instance, and scratch resources scratchBytes={} in {:.2f} ms",
        scratchSize,
        rtxElapsedMilliseconds(createTlasInputsBegin));

    const auto recordBuildBegin = RtxLogClock::now();
    std::unique_ptr<CommandPool> commandPool;
    result = device.createCommandPool(queue, commandPool);
    if (!result) {
        log = resultMessage("createCommandPool(RTX AS build)", result);
        clear();
        return result;
    }

    std::unique_ptr<CommandBuffer> commandBuffer;
    result = commandPool->createCommandBuffer(commandBuffer);
    if (!result) {
        log = resultMessage("createCommandBuffer(RTX AS build)", result);
        clear();
        return result;
    }

    std::unique_ptr<Fence> fence;
    result = device.createFence(false, fence);
    if (!result) {
        log = resultMessage("createFence(RTX AS build)", result);
        clear();
        return result;
    }

    result = commandBuffer->begin();
    if (!result) {
        log = resultMessage("CommandBuffer::begin(RTX AS build)", result);
        clear();
        return result;
    }

    VkCommandBuffer vkCommandBuffer = nativeCommandBuffer(*commandBuffer);
    for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
        const PrimitiveInput& input = primitiveInputs[blasIndex];
        BuiltBlas& blas = impl_->blases[blasIndex];
        VkAccelerationStructureGeometryKHR geometry = makeBlasGeometry(
            nativeVertexBuffer.address,
            nativeIndexBuffer.address,
            input);
        VkAccelerationStructureBuildRangeInfoKHR range = makeBuildRange(input.triangleCount);
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometry);
        buildInfo.dstAccelerationStructure = blas.handle;
        buildInfo.scratchData.deviceAddress = scratchAddress;

        const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&range};
        vkCmdBuildAccelerationStructuresKHR(vkCommandBuffer, 1, &buildInfo, rangeInfos);
        accelerationStructureBarrier(vkCommandBuffer);
    }

    VkAccelerationStructureBuildRangeInfoKHR tlasRange = makeBuildRange(static_cast<uint32_t>(instances.size()));
    tlasBuildInfo.dstAccelerationStructure = impl_->tlas;
    tlasBuildInfo.scratchData.deviceAddress = scratchAddress;
    const VkAccelerationStructureBuildRangeInfoKHR* tlasRangeInfos[] = {&tlasRange};
    vkCmdBuildAccelerationStructuresKHR(vkCommandBuffer, 1, &tlasBuildInfo, tlasRangeInfos);
    accelerationStructureBarrier(vkCommandBuffer);

    result = commandBuffer->end();
    if (!result) {
        log = resultMessage("CommandBuffer::end(RTX AS build)", result);
        clear();
        return result;
    }
    spdlog::info(
        "[RTX] Recorded RTX AS build commands blasCount={} instanceCount={} in {:.2f} ms",
        primitiveInputs.size(),
        instances.size(),
        rtxElapsedMilliseconds(recordBuildBegin));

    CommandBuffer* submittedCommandBuffers[] = {commandBuffer.get()};
    const auto submitBegin = RtxLogClock::now();
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = submittedCommandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        log = resultMessage("Queue::submit(RTX AS build)", result);
        clear();
        return result;
    }
    spdlog::info("[RTX] Submitted RTX AS build in {:.2f} ms", rtxElapsedMilliseconds(submitBegin));

    const auto waitBegin = RtxLogClock::now();
    result = fence->wait();
    if (!result) {
        log = resultMessage("Fence::wait(RTX AS build)", result);
        clear();
        return result;
    }
    spdlog::info("[RTX] RTX AS build fence wait completed in {:.2f} ms", rtxElapsedMilliseconds(waitBegin));

    uint64_t accelerationBytes = tlasSizeInfo.accelerationStructureSize;
    uint64_t triangleCount = 0;
    for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
        accelerationBytes += impl_->blases[blasIndex].sizeInfo.accelerationStructureSize;
        triangleCount += primitiveInputs[blasIndex].triangleCount;
    }

    impl_->stats = SceneRtxStats{
        .blasCount = static_cast<uint32_t>(impl_->blases.size()),
        .instanceCount = static_cast<uint32_t>(instances.size()),
        .triangleCount = triangleCount,
        .vertexCount = vertices.size(),
        .indexCount = indices.size(),
        .geometryBytes =
            vertices.size() * sizeof(RtxVertex) +
            indices.size() * sizeof(uint32_t) +
            instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        .accelerationStructureBytes = accelerationBytes,
        .scratchBytes = scratchSize,
    };
    impl_->primitiveToBlas = std::move(primitiveToBlas);
    impl_->sourceTransformRevision = scene.transformRevision();

    log = "Built Vulkan RTX acceleration structures: " +
        std::to_string(impl_->stats.blasCount) +
        " BLAS, " +
        std::to_string(impl_->stats.instanceCount) +
        " TLAS instances.";
    spdlog::info(
        "[RTX] Built acceleration structures blas={} instances={} triangles={} geometryBytes={} asBytes={} scratchBytes={}",
        impl_->stats.blasCount,
        impl_->stats.instanceCount,
        impl_->stats.triangleCount,
        impl_->stats.geometryBytes,
        impl_->stats.accelerationStructureBytes,
        impl_->stats.scratchBytes);
    return {};
}

Result SceneRtxBuilder::updateInstanceTransforms(
    Device& device,
    Queue& queue,
    const scene::Scene& scene,
    std::string& log)
{
    log.clear();
    if (!valid() || !scene.valid() || impl_->instanceBuffer == nullptr ||
        impl_->scratchBuffer == nullptr || impl_->primitiveToBlas.empty()) {
        log = "Scene RTX acceleration structures are not ready for an instance update.";
        return makeError(Error::InvalidArgument);
    }
    if (impl_->sourceTransformRevision == scene.transformRevision()) {
        return {};
    }

    const NativeDevice nativeDeviceInfo = nativeDevice(device);
    const NativeQueue nativeQueueInfo = nativeQueue(queue);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE || nativeQueueInfo.queue == VK_NULL_HANDLE ||
        nativeDeviceInfo.device != impl_->device) {
        log = "Scene RTX instance update device or queue does not match the original build.";
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(scene.renderNodes().size());
    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible || renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= impl_->primitiveToBlas.size()) {
            continue;
        }
        const int32_t blasIndex =
            impl_->primitiveToBlas[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (blasIndex < 0 || static_cast<size_t>(blasIndex) >= impl_->blases.size()) {
            continue;
        }
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = toVkTransform(renderNode.worldMatrix);
        instance.instanceCustomIndex = static_cast<uint32_t>(renderNode.renderPrimitiveIndex) & 0x00ffffffu;
        instance.mask = 0xff;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = impl_->blases[static_cast<size_t>(blasIndex)].address;
        instances.push_back(instance);
    }
    if (instances.size() != impl_->stats.instanceCount) {
        log = "Scene RTX instance layout changed; a full acceleration-structure rebuild is required.";
        return makeError(Error::InvalidArgument);
    }

    Result result = uploadVector(*impl_->instanceBuffer, instances, "RTX instance transform update", log);
    if (!result) {
        return result;
    }
    const NativeBuffer nativeInstanceBuffer = nativeBuffer(*impl_->instanceBuffer);
    const NativeBuffer nativeScratchBuffer = nativeBuffer(*impl_->scratchBuffer);
    const VkDeviceSize alignment = scratchAlignment(nativeDeviceInfo.physicalDevice);
    const VkDeviceAddress scratchAddress = alignUp(nativeScratchBuffer.address, alignment);
    if (nativeInstanceBuffer.address == 0 || scratchAddress == 0) {
        log = "Scene RTX instance update buffers do not have valid device addresses.";
        return makeError(Error::Failure);
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = VkDeviceOrHostAddressConstKHR{.deviceAddress = nativeInstanceBuffer.address},
    };
    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    };
    geometry.geometry.instances = instancesData;
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        geometry);
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = impl_->tlas;
    buildInfo.dstAccelerationStructure = impl_->tlas;
    buildInfo.scratchData.deviceAddress = scratchAddress;
    VkAccelerationStructureBuildRangeInfoKHR range = makeBuildRange(
        static_cast<uint32_t>(instances.size()));
    const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&range};

    std::unique_ptr<CommandPool> commandPool;
    result = device.createCommandPool(queue, commandPool);
    if (!result) {
        log = resultMessage("createCommandPool(RTX instance update)", result);
        return result;
    }
    std::unique_ptr<CommandBuffer> commandBuffer;
    result = commandPool->createCommandBuffer(commandBuffer);
    if (!result) {
        log = resultMessage("createCommandBuffer(RTX instance update)", result);
        return result;
    }
    std::unique_ptr<Fence> fence;
    result = device.createFence(false, fence);
    if (!result) {
        log = resultMessage("createFence(RTX instance update)", result);
        return result;
    }
    result = commandBuffer->begin();
    if (!result) {
        return result;
    }
    const VkCommandBuffer vkCommandBuffer = nativeCommandBuffer(*commandBuffer);
    vkCmdBuildAccelerationStructuresKHR(vkCommandBuffer, 1, &buildInfo, ranges);
    accelerationStructureBarrier(vkCommandBuffer);
    result = commandBuffer->end();
    if (!result) {
        return result;
    }
    CommandBuffer* commandBuffers[] = {commandBuffer.get()};
    result = queue.submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = fence.get(),
    });
    if (!result) {
        return result;
    }
    result = fence->wait();
    if (!result) {
        return result;
    }

    impl_->sourceTransformRevision = scene.transformRevision();
    log = "Updated Vulkan RTX instance transforms and refit the TLAS.";
    return {};
}

void SceneRtxBuilder::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool SceneRtxBuilder::valid() const
{
    return impl_ != nullptr && impl_->tlas != VK_NULL_HANDLE && impl_->tlasAddress != 0;
}

const SceneRtxStats& SceneRtxBuilder::stats() const
{
    static const SceneRtxStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

struct ScenePartitionedRtxBuilder::Impl {
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceAddress ptlasAddress = 0;
    ScenePartitionedRtxStats stats;
    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
    std::unique_ptr<Buffer> instanceWriteBuffer;
    std::unique_ptr<Buffer> operationBuffer;
    std::unique_ptr<Buffer> operationCountBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<Buffer> ptlasStorage;
    std::vector<BuiltBlas> blases;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        if (device != VK_NULL_HANDLE) {
            volkLoadDevice(device);
            for (BuiltBlas& blas : blases) {
                if (blas.handle != VK_NULL_HANDLE) {
                    vkDestroyAccelerationStructureKHR(device, blas.handle, nullptr);
                    blas.handle = VK_NULL_HANDLE;
                }
            }
        }

        ptlasAddress = 0;
        stats = {};
        blases.clear();
        ptlasStorage.reset();
        scratchBuffer.reset();
        operationCountBuffer.reset();
        operationBuffer.reset();
        instanceWriteBuffer.reset();
        indexBuffer.reset();
        vertexBuffer.reset();
        device = VK_NULL_HANDLE;
    }
};

ScenePartitionedRtxBuilder::ScenePartitionedRtxBuilder()
    : impl_(std::make_unique<Impl>())
{
}

ScenePartitionedRtxBuilder::~ScenePartitionedRtxBuilder() = default;
ScenePartitionedRtxBuilder::ScenePartitionedRtxBuilder(ScenePartitionedRtxBuilder&&) noexcept = default;
ScenePartitionedRtxBuilder& ScenePartitionedRtxBuilder::operator=(ScenePartitionedRtxBuilder&&) noexcept = default;

Result ScenePartitionedRtxBuilder::build(
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
    if (!device.capabilities().partitionedAccelerationStructure) {
        log = "Vulkan partitioned acceleration structure capability is unavailable.";
        return makeError(Error::Unsupported);
    }

    NativeDevice nativeDeviceInfo = nativeDevice(device);
    NativeQueue nativeQueueInfo = nativeQueue(queue);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE || nativeQueueInfo.queue == VK_NULL_HANDLE) {
        log = "Vulkan native device or queue is unavailable.";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(nativeDeviceInfo.device);
    if (vkCmdBuildPartitionedAccelerationStructuresNV == nullptr ||
        vkGetPartitionedAccelerationStructuresBuildSizesNV == nullptr) {
        log = "VK_NV_partitioned_acceleration_structure commands are unavailable.";
        return makeError(Error::Unsupported);
    }

    clear();
    impl_->device = nativeDeviceInfo.device;

    const std::vector<scene::RenderPrimitive>& renderPrimitives = scene.renderPrimitives();
    std::vector<int32_t> primitiveToBlas(renderPrimitives.size(), -1);
    std::vector<PrimitiveInput> primitiveInputs;
    std::vector<RtxVertex> vertices;
    std::vector<uint32_t> indices;

    for (uint32_t primitiveIndex = 0; primitiveIndex < renderPrimitives.size(); ++primitiveIndex) {
        const scene::RenderPrimitive& primitive = renderPrimitives[primitiveIndex];
        if (primitive.mode != 4 || primitive.positions.size() < 3) {
            continue;
        }

        const uint64_t sourceIndexCount = primitive.indices.empty()
            ? (primitive.positions.size() / 3) * 3
            : (primitive.indices.size() / 3) * 3;
        if (sourceIndexCount < 3 ||
            sourceIndexCount > std::numeric_limits<uint32_t>::max() ||
            primitive.positions.size() > std::numeric_limits<uint32_t>::max()) {
            continue;
        }

        PrimitiveInput input{
            .renderPrimitiveIndex = primitiveIndex,
            .firstVertex = static_cast<uint32_t>(vertices.size()),
            .vertexCount = static_cast<uint32_t>(primitive.positions.size()),
            .firstIndex = static_cast<uint32_t>(indices.size()),
            .indexCount = static_cast<uint32_t>(sourceIndexCount),
            .triangleCount = static_cast<uint32_t>(sourceIndexCount / 3),
            .opaque = !primitiveUsesAlphaMask(scene, primitive),
        };

        for (const float3& position : primitive.positions) {
            vertices.push_back(RtxVertex{position.x, position.y, position.z});
        }

        if (primitive.indices.empty()) {
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                indices.push_back(index);
            }
        } else {
            bool indicesValid = true;
            for (uint32_t index = 0; index < input.indexCount; ++index) {
                const uint32_t sourceIndex = primitive.indices[index];
                if (sourceIndex >= input.vertexCount) {
                    indicesValid = false;
                    break;
                }
                indices.push_back(sourceIndex);
            }
            if (!indicesValid) {
                vertices.resize(input.firstVertex);
                indices.resize(input.firstIndex);
                continue;
            }
        }

        primitiveToBlas[primitiveIndex] = static_cast<int32_t>(primitiveInputs.size());
        primitiveInputs.push_back(input);
    }

    if (primitiveInputs.empty() || vertices.empty() || indices.empty()) {
        log = "Scene contains no triangle primitives suitable for PTLAS acceleration structures.";
        clear();
        return makeError(Error::Unsupported);
    }

    Result result = createBuffer(
        device,
        "createBuffer(PTLAS vertices)",
        vertices.size() * sizeof(RtxVertex),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->vertexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->vertexBuffer, vertices, "PTLAS vertices", log);
    if (!result) {
        clear();
        return result;
    }

    result = createBuffer(
        device,
        "createBuffer(PTLAS indices)",
        indices.size() * sizeof(uint32_t),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->indexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->indexBuffer, indices, "PTLAS indices", log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeVertexBuffer = nativeBuffer(*impl_->vertexBuffer);
    const NativeBuffer nativeIndexBuffer = nativeBuffer(*impl_->indexBuffer);
    if (nativeVertexBuffer.address == 0 || nativeIndexBuffer.address == 0) {
        log = "PTLAS geometry buffers do not have device addresses.";
        clear();
        return makeError(Error::Failure);
    }

    impl_->blases.resize(primitiveInputs.size());
    VkDeviceSize maxScratchSize = 0;
    for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
        const PrimitiveInput& input = primitiveInputs[blasIndex];
        VkAccelerationStructureGeometryKHR geometry = makeBlasGeometry(
            nativeVertexBuffer.address,
            nativeIndexBuffer.address,
            input);
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometry);
        BuiltBlas& blas = impl_->blases[blasIndex];
        blas.sizeInfo = queryBuildSize(nativeDeviceInfo.device, buildInfo, input.triangleCount);
        maxScratchSize = std::max(maxScratchSize, blas.sizeInfo.buildScratchSize);

        result = createAccelerationStructure(
            device,
            nativeDeviceInfo.device,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            blas.sizeInfo,
            blas.storage,
            blas.handle,
            blas.address,
            log);
        if (!result) {
            clear();
            return result;
        }
    }

    std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> instances;
    std::vector<float4x4> instanceMatrices;
    instances.reserve(scene.renderNodes().size());
    instanceMatrices.reserve(scene.renderNodes().size());
    for (const scene::RenderNode& renderNode : scene.renderNodes()) {
        if (!renderNode.visible ||
            renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveToBlas.size()) {
            continue;
        }

        const int32_t blasIndex = primitiveToBlas[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (blasIndex < 0 || static_cast<size_t>(blasIndex) >= impl_->blases.size()) {
            continue;
        }

        const NativeBuffer nativeBlasStorage =
            nativeBuffer(*impl_->blases[static_cast<size_t>(blasIndex)].storage);
        if (nativeBlasStorage.address == 0) {
            log = "PTLAS BLAS storage buffer does not have a device address.";
            clear();
            return makeError(Error::Failure);
        }

        const uint32_t instanceIndex = static_cast<uint32_t>(instances.size());
        VkPartitionedAccelerationStructureWriteInstanceDataNV instance{};
        instance.transform = toVkTransform(renderNode.worldMatrix);
        instance.instanceID = static_cast<uint32_t>(renderNode.renderPrimitiveIndex) & 0x00ffffffu;
        instance.instanceMask = 0xff;
        instance.instanceContributionToHitGroupIndex = 0;
        instance.instanceFlags =
            VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE_BIT_NV;
        instance.instanceIndex = instanceIndex;
        instance.partitionIndex = 0;
        instance.accelerationStructure = nativeBlasStorage.address;
        instances.push_back(instance);
        instanceMatrices.push_back(renderNode.worldMatrix);
    }

    if (instances.empty()) {
        log = "Scene contains no visible PTLAS instances.";
        clear();
        return makeError(Error::Unsupported);
    }
    if (instances.size() > std::numeric_limits<uint32_t>::max()) {
        log = "Scene contains too many PTLAS instances.";
        clear();
        return makeError(Error::Unsupported);
    }

    const PtlasPartitioning partitioning = assignPtlasSpatialPartitions(instances, instanceMatrices);
    if (partitioning.partitionCount == 0 || partitioning.maxInstancesPerPartition == 0) {
        log = "PTLAS partitioning produced no usable partitions.";
        clear();
        return makeError(Error::Failure);
    }

    PartitionedAccelerationStructureBuildSizes ptlasSizes;
    result = queryPartitionedAccelerationStructureBuildSizes(
        device,
        PartitionedAccelerationStructureBuildSizesDesc{
            .flags = kBuildFlags,
            .instanceCount = static_cast<uint32_t>(instances.size()),
            .partitionCount = partitioning.partitionCount,
            .maxInstancePerPartitionCount = partitioning.maxInstancesPerPartition,
            .maxInstanceInGlobalPartitionCount = 0,
            .maxOperationCount = 1,
        },
        ptlasSizes);
    if (!result) {
        log = resultMessage("queryPartitionedAccelerationStructureBuildSizes(PTLAS)", result);
        clear();
        return result;
    }
    if (ptlasSizes.accelerationStructureSize == 0 ||
        ptlasSizes.buildScratchSize == 0 ||
        ptlasSizes.operationInfoSize == 0 ||
        ptlasSizes.operationCountSize == 0 ||
        ptlasSizes.instanceWriteInfoSize == 0) {
        log = "PTLAS size query returned zero build size.";
        clear();
        return makeError(Error::Failure);
    }
    maxScratchSize = std::max(maxScratchSize, static_cast<VkDeviceSize>(ptlasSizes.buildScratchSize));

    result = createBuffer(
        device,
        "createBuffer(PTLAS storage)",
        ptlasSizes.accelerationStructureSize,
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->ptlasStorage,
        log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativePtlasStorage = nativeBuffer(*impl_->ptlasStorage);
    if (nativePtlasStorage.address == 0) {
        log = "PTLAS storage buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }
    impl_->ptlasAddress = nativePtlasStorage.address;

    result = createBuffer(
        device,
        "createBuffer(PTLAS instance write info)",
        ptlasSizes.instanceWriteInfoSize,
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::Storage,
        MemoryLocation::HostUpload,
        impl_->instanceWriteBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->instanceWriteBuffer, instances, "PTLAS instance write info", log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeInstanceWriteBuffer = nativeBuffer(*impl_->instanceWriteBuffer);
    if (nativeInstanceWriteBuffer.address == 0) {
        log = "PTLAS instance write buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    std::vector<VkBuildPartitionedAccelerationStructureIndirectCommandNV> operations(1);
    operations[0].opType = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV;
    operations[0].argCount = static_cast<uint32_t>(instances.size());
    operations[0].argData.startAddress = nativeInstanceWriteBuffer.address;
    operations[0].argData.strideInBytes =
        sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV);

    result = createBuffer(
        device,
        "createBuffer(PTLAS operations)",
        ptlasSizes.operationInfoSize,
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::Storage,
        MemoryLocation::HostUpload,
        impl_->operationBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->operationBuffer, operations, "PTLAS operations", log);
    if (!result) {
        clear();
        return result;
    }

    const std::vector<uint32_t> operationCounts = {static_cast<uint32_t>(operations.size())};
    result = createBuffer(
        device,
        "createBuffer(PTLAS operation count)",
        ptlasSizes.operationCountSize,
        BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::Storage,
        MemoryLocation::HostUpload,
        impl_->operationCountBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->operationCountBuffer, operationCounts, "PTLAS operation count", log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeOperationBuffer = nativeBuffer(*impl_->operationBuffer);
    const NativeBuffer nativeOperationCountBuffer = nativeBuffer(*impl_->operationCountBuffer);
    if (nativeOperationBuffer.address == 0 || nativeOperationCountBuffer.address == 0) {
        log = "PTLAS operation buffers do not have device addresses.";
        clear();
        return makeError(Error::Failure);
    }

    const VkDeviceSize alignment = scratchAlignment(nativeDeviceInfo.physicalDevice);
    const VkDeviceSize scratchSize = maxScratchSize + alignment - 1;
    result = createBuffer(
        device,
        "createBuffer(PTLAS scratch)",
        scratchSize,
        BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeScratchBuffer = nativeBuffer(*impl_->scratchBuffer);
    const VkDeviceAddress scratchAddress = alignUp(nativeScratchBuffer.address, alignment);
    if (nativeScratchBuffer.address == 0 ||
        scratchAddress == 0 ||
        scratchAddress + maxScratchSize > nativeScratchBuffer.address + nativeScratchBuffer.size) {
        log = "PTLAS scratch buffer does not provide a valid aligned device address.";
        clear();
        return makeError(Error::Failure);
    }

    std::unique_ptr<CommandPool> commandPool;
    result = device.createCommandPool(queue, commandPool);
    if (!result) {
        log = resultMessage("createCommandPool(PTLAS AS build)", result);
        clear();
        return result;
    }

    result = recordSubmitWait(
        device,
        queue,
        *commandPool,
        "PTLAS AS build",
        [&](VkCommandBuffer vkCommandBuffer) {
            for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
                const PrimitiveInput& input = primitiveInputs[blasIndex];
                BuiltBlas& blas = impl_->blases[blasIndex];
                VkAccelerationStructureGeometryKHR geometry = makeBlasGeometry(
                    nativeVertexBuffer.address,
                    nativeIndexBuffer.address,
                    input);
                VkAccelerationStructureBuildRangeInfoKHR range = makeBuildRange(input.triangleCount);
                VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
                    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    geometry);
                buildInfo.dstAccelerationStructure = blas.handle;
                buildInfo.scratchData.deviceAddress = scratchAddress;

                const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&range};
                vkCmdBuildAccelerationStructuresKHR(vkCommandBuffer, 1, &buildInfo, rangeInfos);
                accelerationStructureBarrier(vkCommandBuffer);
            }

            VkPartitionedAccelerationStructureFlagsNV partitionedFlags{
                .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV,
                .enablePartitionTranslation = VK_FALSE,
            };
            VkPartitionedAccelerationStructureInstancesInputNV inputInfo{
                .sType = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV,
                .pNext = &partitionedFlags,
                .flags = kBuildFlags,
                .instanceCount = static_cast<uint32_t>(instances.size()),
                .maxInstancePerPartitionCount = partitioning.maxInstancesPerPartition,
                .partitionCount = partitioning.partitionCount,
                .maxInstanceInGlobalPartitionCount = 0,
            };
            VkBuildPartitionedAccelerationStructureInfoNV buildInfo{
                .sType = VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV,
                .input = inputInfo,
                .srcAccelerationStructureData = 0,
                .dstAccelerationStructureData = impl_->ptlasAddress,
                .scratchData = scratchAddress,
                .srcInfos = nativeOperationBuffer.address,
                .srcInfosCount = nativeOperationCountBuffer.address,
            };
            vkCmdBuildPartitionedAccelerationStructuresNV(vkCommandBuffer, &buildInfo);
            accelerationStructureBarrier(vkCommandBuffer);
        },
        log);
    if (!result) {
        clear();
        return result;
    }

    uint64_t blasBytes = 0;
    uint64_t triangleCount = 0;
    for (size_t blasIndex = 0; blasIndex < primitiveInputs.size(); ++blasIndex) {
        blasBytes += impl_->blases[blasIndex].sizeInfo.accelerationStructureSize;
        triangleCount += primitiveInputs[blasIndex].triangleCount;
    }
    const uint64_t operationBytes =
        ptlasSizes.operationInfoSize +
        ptlasSizes.operationCountSize +
        ptlasSizes.instanceWriteInfoSize;

    impl_->stats = ScenePartitionedRtxStats{
        .blasCount = static_cast<uint32_t>(impl_->blases.size()),
        .instanceCount = static_cast<uint32_t>(instances.size()),
        .partitionCount = partitioning.partitionCount,
        .maxInstancesPerPartition = partitioning.maxInstancesPerPartition,
        .triangleCount = triangleCount,
        .vertexCount = vertices.size(),
        .indexCount = indices.size(),
        .geometryBytes =
            vertices.size() * sizeof(RtxVertex) +
            indices.size() * sizeof(uint32_t) +
            instances.size() * sizeof(VkPartitionedAccelerationStructureWriteInstanceDataNV),
        .blasBytes = blasBytes,
        .ptlasBytes = ptlasSizes.accelerationStructureSize,
        .accelerationStructureBytes = blasBytes + ptlasSizes.accelerationStructureSize,
        .scratchBytes = scratchSize,
        .operationBytes = operationBytes,
    };

    log = "Built Vulkan PTLAS acceleration structures: " +
        std::to_string(impl_->stats.blasCount) +
        " BLAS, " +
        std::to_string(impl_->stats.instanceCount) +
        " PTLAS instances.";
    return {};
}

void ScenePartitionedRtxBuilder::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool ScenePartitionedRtxBuilder::valid() const
{
    return impl_ != nullptr && impl_->ptlasAddress != 0;
}

const ScenePartitionedRtxStats& ScenePartitionedRtxBuilder::stats() const
{
    static const ScenePartitionedRtxStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

struct SceneClusterRtxBuilder::Impl {
    VkDevice device = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkDeviceAddress tlasAddress = 0;
    SceneClusterRtxStats stats;
    std::unique_ptr<Buffer> clusterVertexBuffer;
    std::unique_ptr<Buffer> clusterIndexBuffer;
    std::unique_ptr<Buffer> clasBuildInfoBuffer;
    std::unique_ptr<Buffer> clasSizeBuffer;
    std::unique_ptr<Buffer> clasSizeReadbackBuffer;
    std::unique_ptr<Buffer> clasAddressBuffer;
    std::unique_ptr<Buffer> clasStorageBuffer;
    std::unique_ptr<Buffer> clusterBlasBuildInfoBuffer;
    std::unique_ptr<Buffer> clusterBlasStorageBuffer;
    std::unique_ptr<Buffer> clusterBlasAddressBuffer;
    std::unique_ptr<Buffer> clusterBlasAddressReadbackBuffer;
    std::unique_ptr<Buffer> clusterBlasSizeBuffer;
    std::unique_ptr<Buffer> instanceBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<Buffer> tlasStorage;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        if (device != VK_NULL_HANDLE && tlas != VK_NULL_HANDLE) {
            volkLoadDevice(device);
            vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
            tlas = VK_NULL_HANDLE;
        }

        tlasAddress = 0;
        stats = {};
        tlasStorage.reset();
        scratchBuffer.reset();
        instanceBuffer.reset();
        clusterBlasSizeBuffer.reset();
        clusterBlasAddressReadbackBuffer.reset();
        clusterBlasAddressBuffer.reset();
        clusterBlasStorageBuffer.reset();
        clusterBlasBuildInfoBuffer.reset();
        clasStorageBuffer.reset();
        clasAddressBuffer.reset();
        clasSizeReadbackBuffer.reset();
        clasSizeBuffer.reset();
        clasBuildInfoBuffer.reset();
        clusterIndexBuffer.reset();
        clusterVertexBuffer.reset();
        device = VK_NULL_HANDLE;
    }
};

SceneClusterRtxBuilder::SceneClusterRtxBuilder()
    : impl_(std::make_unique<Impl>())
{
}

SceneClusterRtxBuilder::~SceneClusterRtxBuilder() = default;
SceneClusterRtxBuilder::SceneClusterRtxBuilder(SceneClusterRtxBuilder&&) noexcept = default;
SceneClusterRtxBuilder& SceneClusterRtxBuilder::operator=(SceneClusterRtxBuilder&&) noexcept = default;

Result SceneClusterRtxBuilder::build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log)
{
    log.clear();
#ifndef VK_NV_cluster_acceleration_structure
    (void)device;
    (void)queue;
    (void)scene;
    log = "VK_NV_cluster_acceleration_structure is unavailable in this Vulkan header.";
    return makeError(Error::Unsupported);
#else
    if (!scene.valid()) {
        log = "Scene is not loaded.";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().clusterAccelerationStructure) {
        log = "Vulkan cluster acceleration structure capability is unavailable.";
        return makeError(Error::Unsupported);
    }

    NativeDevice nativeDeviceInfo = nativeDevice(device);
    NativeQueue nativeQueueInfo = nativeQueue(queue);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE || nativeQueueInfo.queue == VK_NULL_HANDLE) {
        log = "Vulkan native device or queue is unavailable.";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(nativeDeviceInfo.device);
    if (vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr ||
        vkGetClusterAccelerationStructureBuildSizesNV == nullptr) {
        log = "VK_NV_cluster_acceleration_structure entry points are unavailable.";
        return makeError(Error::Unsupported);
    }

    clear();
    impl_->device = nativeDeviceInfo.device;

    ClusterSceneInputs sceneInputs;
    if (!buildClusterSceneInputs(scene, sceneInputs, log)) {
        clear();
        return makeError(Error::Unsupported);
    }

    if (sceneInputs.clusters.size() > std::numeric_limits<uint32_t>::max() ||
        sceneInputs.vertices.size() > std::numeric_limits<uint32_t>::max() ||
        sceneInputs.triangleCount > std::numeric_limits<uint32_t>::max() ||
        sceneInputs.instances.size() > std::numeric_limits<uint32_t>::max()) {
        log = "Scene meshlet CLAS build inputs exceed Vulkan 32-bit count limits.";
        clear();
        return makeError(Error::Unsupported);
    }

    uint32_t maxClusterTriangleCount = 0;
    uint32_t maxClusterVertexCount = 0;
    uint32_t maxClusterCountPerBlas = 0;
    uint64_t selectedClusterReferenceCount = 0;
    for (const ClusterBuildInput& cluster : sceneInputs.clusters) {
        maxClusterTriangleCount = std::max(maxClusterTriangleCount, cluster.triangleCount);
        maxClusterVertexCount = std::max(maxClusterVertexCount, cluster.vertexCount);
    }
    for (const ClusterBlasInstanceInput& instance : sceneInputs.instances) {
        maxClusterCountPerBlas = std::max(maxClusterCountPerBlas, instance.clusters.clusterCount);
        selectedClusterReferenceCount += instance.clusters.clusterCount;
    }
    if (maxClusterTriangleCount == 0 ||
        maxClusterVertexCount == 0 ||
        maxClusterCountPerBlas == 0 ||
        selectedClusterReferenceCount == 0 ||
        selectedClusterReferenceCount > std::numeric_limits<uint32_t>::max()) {
        log = "Scene meshlet CLAS build produced empty cluster ranges.";
        clear();
        return makeError(Error::Unsupported);
    }

    Result result = createBuffer(
        device,
        "createBuffer(CLAS vertices)",
        checkedByteSize(sceneInputs.vertices.size(), sizeof(RtxVertex)),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterVertexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->clusterVertexBuffer, sceneInputs.vertices, "CLAS vertices", log);
    if (!result) {
        clear();
        return result;
    }

    result = createBuffer(
        device,
        "createBuffer(CLAS indices)",
        checkedByteSize(sceneInputs.indices.size(), sizeof(uint8_t)),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterIndexBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->clusterIndexBuffer, sceneInputs.indices, "CLAS indices", log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeVertexBuffer = nativeBuffer(*impl_->clusterVertexBuffer);
    const NativeBuffer nativeIndexBuffer = nativeBuffer(*impl_->clusterIndexBuffer);
    if (nativeVertexBuffer.address == 0 || nativeIndexBuffer.address == 0) {
        log = "CLAS geometry buffers do not have device addresses.";
        clear();
        return makeError(Error::Failure);
    }

    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clasBuildInfos;
    clasBuildInfos.reserve(sceneInputs.clusters.size());
    for (uint32_t clusterIndex = 0; clusterIndex < sceneInputs.clusters.size(); ++clusterIndex) {
        const ClusterBuildInput& cluster = sceneInputs.clusters[clusterIndex];
        VkClusterAccelerationStructureBuildTriangleClusterInfoNV buildInfo{};
        buildInfo.clusterID = clusterIndex;
        buildInfo.triangleCount = cluster.triangleCount;
        buildInfo.vertexCount = cluster.vertexCount;
        buildInfo.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
        buildInfo.indexBufferStride = 1;
        buildInfo.vertexBufferStride = sizeof(RtxVertex);
        buildInfo.indexBuffer =
            nativeIndexBuffer.address + static_cast<VkDeviceAddress>(cluster.firstIndex) * sizeof(uint8_t);
        buildInfo.vertexBuffer =
            nativeVertexBuffer.address + static_cast<VkDeviceAddress>(cluster.firstVertex) * sizeof(RtxVertex);
        buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags =
            cluster.opaque ? VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV : 0;
        clasBuildInfos.push_back(buildInfo);
    }

    result = createBuffer(
        device,
        "createBuffer(CLAS build infos)",
        checkedByteSize(clasBuildInfos.size(), sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV)),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clasBuildInfoBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->clasBuildInfoBuffer, clasBuildInfos, "CLAS build infos", log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeClasBuildInfoBuffer = nativeBuffer(*impl_->clasBuildInfoBuffer);
    if (nativeClasBuildInfoBuffer.address == 0) {
        log = "CLAS build info buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    ClusterAccelerationStructureBuildSizes clasSizes;
    result = queryClusterAccelerationStructureTriangleBuildSizes(
        device,
        ClusterAccelerationStructureTriangleBuildSizesDesc{
            .flags = kBuildFlags,
            .maxClusterTriangleCount = maxClusterTriangleCount,
            .maxClusterVertexCount = maxClusterVertexCount,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = static_cast<uint32_t>(sceneInputs.triangleCount),
            .maxTotalVertexCount = static_cast<uint32_t>(sceneInputs.vertices.size()),
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxAccelerationStructureCount = static_cast<uint32_t>(sceneInputs.clusters.size()),
        },
        clasSizes);
    if (!result) {
        log = resultMessage("queryClusterAccelerationStructureTriangleBuildSizes(CLAS)", result);
        clear();
        return result;
    }

    ClusterAccelerationStructureBuildSizes clusterBlasSizes;
    result = queryClusterAccelerationStructureBottomLevelBuildSizes(
        device,
        ClusterAccelerationStructureBottomLevelBuildSizesDesc{
            .flags = kBuildFlags,
            .maxClusterCountPerAccelerationStructure = maxClusterCountPerBlas,
            .maxTotalClusterCount = static_cast<uint32_t>(selectedClusterReferenceCount),
            .maxAccelerationStructureCount = static_cast<uint32_t>(sceneInputs.instances.size()),
        },
        clusterBlasSizes);
    if (!result) {
        log = resultMessage("queryClusterAccelerationStructureBottomLevelBuildSizes(cluster BLAS)", result);
        clear();
        return result;
    }

    VkAccelerationStructureGeometryInstancesDataKHR dummyInstancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
    };
    VkAccelerationStructureGeometryKHR tlasGeometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    };
    tlasGeometry.geometry.instances = dummyInstancesData;
    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo = makeBuildInfo(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlasGeometry);
    VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo = queryBuildSize(
        nativeDeviceInfo.device,
        tlasBuildInfo,
        static_cast<uint32_t>(sceneInputs.instances.size()));

    const VkDeviceSize scratchAlign = std::max(
        scratchAlignment(nativeDeviceInfo.physicalDevice),
        clusterScratchAlignment(nativeDeviceInfo.physicalDevice));
    const VkDeviceSize maxScratchSize = std::max<VkDeviceSize>(
        static_cast<VkDeviceSize>(clasSizes.buildScratchSize),
        std::max<VkDeviceSize>(
            static_cast<VkDeviceSize>(clusterBlasSizes.buildScratchSize),
            tlasSizeInfo.buildScratchSize));
    const VkDeviceSize scratchSize = maxScratchSize + scratchAlign - 1;
    result = createBuffer(
        device,
        "createBuffer(CLAS scratch)",
        scratchSize,
        BufferUsageBits::Storage |
            BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeScratchBuffer = nativeBuffer(*impl_->scratchBuffer);
    const VkDeviceAddress scratchAddress = alignUp(nativeScratchBuffer.address, scratchAlign);
    if (nativeScratchBuffer.address == 0 ||
        scratchAddress == 0 ||
        scratchAddress + maxScratchSize > nativeScratchBuffer.address + nativeScratchBuffer.size) {
        log = "CLAS scratch buffer does not provide a valid aligned device address.";
        clear();
        return makeError(Error::Failure);
    }

    const uint64_t clasCount = sceneInputs.clusters.size();
    const uint64_t instanceCount = sceneInputs.instances.size();
    result = createBuffer(
        device,
        "createBuffer(CLAS size device)",
        checkedByteSize(clasCount, sizeof(uint32_t)),
        BufferUsageBits::Storage |
            BufferUsageBits::TransferSource |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clasSizeBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(CLAS size readback)",
        checkedByteSize(clasCount, sizeof(uint32_t)),
        BufferUsageBits::TransferDestination,
        MemoryLocation::HostReadback,
        impl_->clasSizeReadbackBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }

    std::unique_ptr<CommandPool> commandPool;
    result = device.createCommandPool(queue, commandPool);
    if (!result) {
        log = resultMessage("createCommandPool(CLAS build)", result);
        clear();
        return result;
    }

    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .maxGeometryIndexValue = 0,
        .maxClusterUniqueGeometryCount = 1,
        .maxClusterTriangleCount = maxClusterTriangleCount,
        .maxClusterVertexCount = maxClusterVertexCount,
        .maxTotalTriangleCount = static_cast<uint32_t>(sceneInputs.triangleCount),
        .maxTotalVertexCount = static_cast<uint32_t>(sceneInputs.vertices.size()),
        .minPositionTruncateBitCount = 0,
    };
    VkClusterAccelerationStructureInputInfoNV clasInputInfo{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = static_cast<uint32_t>(sceneInputs.clusters.size()),
        .flags = kBuildFlags,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV,
        .opInput = {.pTriangleClusters = &triangleInput},
    };

    const NativeBuffer nativeClasSizeBuffer = nativeBuffer(*impl_->clasSizeBuffer);
    const NativeBuffer nativeClasSizeReadbackBuffer = nativeBuffer(*impl_->clasSizeReadbackBuffer);
    if (nativeClasSizeBuffer.address == 0 ||
        nativeClasSizeBuffer.buffer == VK_NULL_HANDLE ||
        nativeClasSizeReadbackBuffer.buffer == VK_NULL_HANDLE) {
        log = "CLAS size buffers are unavailable.";
        clear();
        return makeError(Error::Failure);
    }

    result = recordSubmitWait(
        device,
        queue,
        *commandPool,
        "CLAS size build",
        [&](VkCommandBuffer commandBuffer) {
            VkClusterAccelerationStructureCommandsInfoNV cmdInfo{
                .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
                .input = clasInputInfo,
                .scratchData = scratchAddress,
                .dstSizesArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClasSizeBuffer.address,
                    .stride = sizeof(uint32_t),
                    .size = nativeClasSizeBuffer.size,
                },
                .srcInfosArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClasBuildInfoBuffer.address,
                    .stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
                    .size = nativeClasBuildInfoBuffer.size,
                },
            };
            vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, &cmdInfo);
            memoryBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferCopy copy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = nativeClasSizeBuffer.size,
            };
            vkCmdCopyBuffer(
                commandBuffer,
                nativeClasSizeBuffer.buffer,
                nativeClasSizeReadbackBuffer.buffer,
                1,
                &copy);
        },
        log);
    if (!result) {
        clear();
        return result;
    }

    std::vector<uint32_t> clasSizeValues;
    result = readbackVector(
        *impl_->clasSizeReadbackBuffer,
        sceneInputs.clusters.size(),
        "CLAS sizes",
        clasSizeValues,
        log);
    if (!result) {
        clear();
        return result;
    }

    const VkDeviceSize clasAlignment = clusterStorageAlignment(nativeDeviceInfo.physicalDevice);
    uint64_t clasStorageSize = 0;
    std::vector<uint64_t> clasAddresses(sceneInputs.clusters.size());
    for (size_t clusterIndex = 0; clusterIndex < clasSizeValues.size(); ++clusterIndex) {
        if (clasSizeValues[clusterIndex] == 0) {
            log = "VK_NV_cluster_acceleration_structure reported a zero CLAS size.";
            clear();
            return makeError(Error::Failure);
        }
        clasStorageSize = alignUp(clasStorageSize, clasAlignment);
        clasAddresses[clusterIndex] = clasStorageSize;
        clasStorageSize += clasSizeValues[clusterIndex];
    }
    if (clasStorageSize == 0) {
        log = "CLAS storage size is zero.";
        clear();
        return makeError(Error::Failure);
    }

    result = createBuffer(
        device,
        "createBuffer(CLAS storage)",
        clasStorageSize,
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clasStorageBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeClasStorageBuffer = nativeBuffer(*impl_->clasStorageBuffer);
    if (nativeClasStorageBuffer.address == 0) {
        log = "CLAS storage buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }
    for (uint64_t& address : clasAddresses) {
        address += nativeClasStorageBuffer.address;
    }

    result = createBuffer(
        device,
        "createBuffer(CLAS addresses)",
        checkedByteSize(clasAddresses.size(), sizeof(uint64_t)),
        BufferUsageBits::Storage |
            BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clasAddressBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->clasAddressBuffer, clasAddresses, "CLAS addresses", log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeClasAddressBuffer = nativeBuffer(*impl_->clasAddressBuffer);
    if (nativeClasAddressBuffer.address == 0) {
        log = "CLAS address buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    std::vector<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> clusterBlasBuildInfos;
    clusterBlasBuildInfos.reserve(sceneInputs.instances.size());
    for (const ClusterBlasInstanceInput& instance : sceneInputs.instances) {
        clusterBlasBuildInfos.push_back(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV{
            .clusterReferencesCount = instance.clusters.clusterCount,
            .clusterReferencesStride = sizeof(uint64_t),
            .clusterReferences =
                nativeClasAddressBuffer.address +
                static_cast<VkDeviceAddress>(instance.clusters.firstCluster) * sizeof(uint64_t),
        });
    }
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS build infos)",
        checkedByteSize(
            clusterBlasBuildInfos.size(),
            sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV)),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->clusterBlasBuildInfoBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->clusterBlasBuildInfoBuffer, clusterBlasBuildInfos, "cluster BLAS build infos", log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeClusterBlasBuildInfoBuffer = nativeBuffer(*impl_->clusterBlasBuildInfoBuffer);
    if (nativeClusterBlasBuildInfoBuffer.address == 0) {
        log = "cluster BLAS build info buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    result = createBuffer(
        device,
        "createBuffer(cluster BLAS storage)",
        clusterBlasSizes.accelerationStructureSize,
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clusterBlasStorageBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeClusterBlasStorageBuffer = nativeBuffer(*impl_->clusterBlasStorageBuffer);
    if (nativeClusterBlasStorageBuffer.address == 0) {
        log = "cluster BLAS storage buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    const uint64_t clusterBlasArrayByteSize = checkedByteSize(instanceCount, sizeof(uint64_t));
    const uint64_t clusterBlasSizeArrayByteSize = checkedByteSize(instanceCount, sizeof(uint32_t));
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS addresses device)",
        clusterBlasArrayByteSize,
        BufferUsageBits::Storage |
            BufferUsageBits::TransferSource |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clusterBlasAddressBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS addresses readback)",
        clusterBlasArrayByteSize,
        BufferUsageBits::TransferDestination,
        MemoryLocation::HostReadback,
        impl_->clusterBlasAddressReadbackBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = createBuffer(
        device,
        "createBuffer(cluster BLAS sizes device)",
        clusterBlasSizeArrayByteSize,
        BufferUsageBits::Storage |
            BufferUsageBits::TransferSource |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->clusterBlasSizeBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }

    const NativeBuffer nativeClusterBlasAddressBuffer = nativeBuffer(*impl_->clusterBlasAddressBuffer);
    const NativeBuffer nativeClusterBlasAddressReadbackBuffer = nativeBuffer(*impl_->clusterBlasAddressReadbackBuffer);
    const NativeBuffer nativeClusterBlasSizeBuffer = nativeBuffer(*impl_->clusterBlasSizeBuffer);
    if (nativeClusterBlasAddressBuffer.address == 0 ||
        nativeClusterBlasSizeBuffer.address == 0 ||
        nativeClusterBlasAddressBuffer.buffer == VK_NULL_HANDLE ||
        nativeClusterBlasAddressReadbackBuffer.buffer == VK_NULL_HANDLE) {
        log = "cluster BLAS output buffers are unavailable.";
        clear();
        return makeError(Error::Failure);
    }

    VkClusterAccelerationStructureClustersBottomLevelInputNV clusterBlasInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
        .maxTotalClusterCount = static_cast<uint32_t>(selectedClusterReferenceCount),
        .maxClusterCountPerAccelerationStructure = maxClusterCountPerBlas,
    };
    VkClusterAccelerationStructureInputInfoNV clusterBlasInputInfo{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = static_cast<uint32_t>(sceneInputs.instances.size()),
        .flags = kBuildFlags,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
        .opInput = {.pClustersBottomLevel = &clusterBlasInput},
    };

    result = recordSubmitWait(
        device,
        queue,
        *commandPool,
        "CLAS and cluster BLAS build",
        [&](VkCommandBuffer commandBuffer) {
            VkClusterAccelerationStructureInputInfoNV explicitClasInput = clasInputInfo;
            explicitClasInput.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
            VkClusterAccelerationStructureCommandsInfoNV clasCmdInfo{
                .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
                .input = explicitClasInput,
                .scratchData = scratchAddress,
                .dstAddressesArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClasAddressBuffer.address,
                    .stride = sizeof(uint64_t),
                    .size = nativeClasAddressBuffer.size,
                },
                .srcInfosArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClasBuildInfoBuffer.address,
                    .stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
                    .size = nativeClasBuildInfoBuffer.size,
                },
            };
            vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, &clasCmdInfo);
            memoryBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

            VkClusterAccelerationStructureCommandsInfoNV blasCmdInfo{
                .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
                .input = clusterBlasInputInfo,
                .dstImplicitData = nativeClusterBlasStorageBuffer.address,
                .scratchData = scratchAddress,
                .dstAddressesArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClusterBlasAddressBuffer.address,
                    .stride = sizeof(uint64_t),
                    .size = nativeClusterBlasAddressBuffer.size,
                },
                .dstSizesArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClusterBlasSizeBuffer.address,
                    .stride = sizeof(uint32_t),
                    .size = nativeClusterBlasSizeBuffer.size,
                },
                .srcInfosArray = VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = nativeClusterBlasBuildInfoBuffer.address,
                    .stride = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                    .size = nativeClusterBlasBuildInfoBuffer.size,
                },
            };
            vkCmdBuildClusterAccelerationStructureIndirectNV(commandBuffer, &blasCmdInfo);
            memoryBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferCopy copy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = nativeClusterBlasAddressBuffer.size,
            };
            vkCmdCopyBuffer(
                commandBuffer,
                nativeClusterBlasAddressBuffer.buffer,
                nativeClusterBlasAddressReadbackBuffer.buffer,
                1,
                &copy);
        },
        log);
    if (!result) {
        clear();
        return result;
    }

    std::vector<uint64_t> clusterBlasAddresses;
    result = readbackVector(
        *impl_->clusterBlasAddressReadbackBuffer,
        sceneInputs.instances.size(),
        "cluster BLAS addresses",
        clusterBlasAddresses,
        log);
    if (!result) {
        clear();
        return result;
    }
    for (uint64_t address : clusterBlasAddresses) {
        if (address == 0) {
            log = "VK_NV_cluster_acceleration_structure reported a zero cluster BLAS address.";
            clear();
            return makeError(Error::Failure);
        }
    }

    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(sceneInputs.instances.size());
    for (size_t instanceIndex = 0; instanceIndex < sceneInputs.instances.size(); ++instanceIndex) {
        const ClusterBlasInstanceInput& sourceInstance = sceneInputs.instances[instanceIndex];
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = toVkTransform(sourceInstance.worldMatrix);
        instance.instanceCustomIndex = sourceInstance.renderPrimitiveIndex & 0x00ffffffu;
        instance.mask = 0xff;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = clusterBlasAddresses[instanceIndex];
        tlasInstances.push_back(instance);
    }

    result = createBuffer(
        device,
        "createBuffer(cluster TLAS instances)",
        checkedByteSize(tlasInstances.size(), sizeof(VkAccelerationStructureInstanceKHR)),
        BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->instanceBuffer,
        log);
    if (!result) {
        clear();
        return result;
    }
    result = uploadVector(*impl_->instanceBuffer, tlasInstances, "cluster TLAS instances", log);
    if (!result) {
        clear();
        return result;
    }
    const NativeBuffer nativeInstanceBuffer = nativeBuffer(*impl_->instanceBuffer);
    if (nativeInstanceBuffer.address == 0) {
        log = "cluster TLAS instance buffer does not have a device address.";
        clear();
        return makeError(Error::Failure);
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = VkDeviceOrHostAddressConstKHR{
            .deviceAddress = nativeInstanceBuffer.address,
        },
    };
    tlasGeometry.geometry.instances = instancesData;
    tlasBuildInfo = makeBuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlasGeometry);
    tlasSizeInfo = queryBuildSize(
        nativeDeviceInfo.device,
        tlasBuildInfo,
        static_cast<uint32_t>(tlasInstances.size()));
    if (tlasSizeInfo.buildScratchSize > maxScratchSize) {
        log = "cluster TLAS scratch size exceeded the precomputed CLAS scratch budget.";
        clear();
        return makeError(Error::Failure);
    }

    result = createAccelerationStructure(
        device,
        nativeDeviceInfo.device,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlasSizeInfo,
        impl_->tlasStorage,
        impl_->tlas,
        impl_->tlasAddress,
        log);
    if (!result) {
        clear();
        return result;
    }

    result = recordSubmitWait(
        device,
        queue,
        *commandPool,
        "cluster TLAS build",
        [&](VkCommandBuffer commandBuffer) {
            VkAccelerationStructureBuildRangeInfoKHR tlasRange =
                makeBuildRange(static_cast<uint32_t>(tlasInstances.size()));
            VkAccelerationStructureBuildGeometryInfoKHR buildInfo = makeBuildInfo(
                VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                tlasGeometry);
            buildInfo.dstAccelerationStructure = impl_->tlas;
            buildInfo.scratchData.deviceAddress = scratchAddress;
            const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&tlasRange};
            vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, rangeInfos);
            accelerationStructureBarrier(commandBuffer);
        },
        log);
    if (!result) {
        clear();
        return result;
    }

    impl_->stats = SceneClusterRtxStats{
        .clasCount = static_cast<uint32_t>(sceneInputs.clusters.size()),
        .clusterBlasCount = static_cast<uint32_t>(sceneInputs.instances.size()),
        .instanceCount = static_cast<uint32_t>(tlasInstances.size()),
        .clusterTriangleCount = sceneInputs.triangleCount,
        .clusterVertexCount = sceneInputs.vertices.size(),
        .clusterIndexBytes = sceneInputs.indices.size(),
        .selectedClusterReferenceCount = selectedClusterReferenceCount,
        .geometryBytes =
            sceneInputs.vertices.size() * sizeof(RtxVertex) +
            sceneInputs.indices.size() * sizeof(uint8_t) +
            clasBuildInfos.size() * sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) +
            clusterBlasBuildInfos.size() * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) +
            tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        .clasBytes = clasStorageSize,
        .clusterBlasBytes = clusterBlasSizes.accelerationStructureSize,
        .tlasBytes = tlasSizeInfo.accelerationStructureSize,
        .accelerationStructureBytes =
            clasStorageSize +
            clusterBlasSizes.accelerationStructureSize +
            tlasSizeInfo.accelerationStructureSize,
        .scratchBytes = scratchSize,
    };
    log = "Built Vulkan cluster acceleration structures: " +
        std::to_string(impl_->stats.clasCount) +
        " CLAS, " +
        std::to_string(impl_->stats.clusterBlasCount) +
        " cluster BLAS, " +
        std::to_string(impl_->stats.instanceCount) +
        " TLAS instances.";
    return {};
#endif
}

void SceneClusterRtxBuilder::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool SceneClusterRtxBuilder::valid() const
{
    return impl_ != nullptr &&
        impl_->tlas != VK_NULL_HANDLE &&
        impl_->tlasAddress != 0 &&
        impl_->stats.clasCount != 0;
}

const SceneClusterRtxStats& SceneClusterRtxBuilder::stats() const
{
    static const SceneClusterRtxStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

struct SceneRayQueryProgram::Impl {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    uint32_t pushConstantSize = 0;
    std::string debugName = "SceneRayQueryProgram";
    std::vector<SceneRayQueryBindingDesc> bindings;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        if (device != VK_NULL_HANDLE) {
            volkLoadDevice(device);
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
                pipeline = VK_NULL_HANDLE;
            }
            if (pipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
                pipelineLayout = VK_NULL_HANDLE;
            }
            if (shaderModule != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shaderModule, nullptr);
                shaderModule = VK_NULL_HANDLE;
            }
            if (descriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
                descriptorPool = VK_NULL_HANDLE;
            }
            if (descriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = VK_NULL_HANDLE;
            }
        }

        device = VK_NULL_HANDLE;
        descriptorSets.clear();
        pushConstantSize = 0;
        bindings.clear();
        debugName = "SceneRayQueryProgram";
    }
};

SceneRayQueryProgram::SceneRayQueryProgram()
    : impl_(std::make_unique<Impl>())
{
}

SceneRayQueryProgram::~SceneRayQueryProgram() = default;
SceneRayQueryProgram::SceneRayQueryProgram(SceneRayQueryProgram&&) noexcept = default;
SceneRayQueryProgram& SceneRayQueryProgram::operator=(SceneRayQueryProgram&&) noexcept = default;

Result SceneRayQueryProgram::initialize(
    Device& device,
    const SceneRayQueryProgramDesc& desc,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    log.clear();

    if (desc.spirv == nullptr ||
        desc.byteSize == 0 ||
        (desc.byteSize % sizeof(uint32_t)) != 0 ||
        desc.bindings == nullptr ||
        desc.bindingCount == 0 ||
        desc.descriptorSetCount == 0 ||
        hasDuplicateBindings(desc.bindings, desc.bindingCount)) {
        log = "SceneRayQueryProgramDesc is invalid";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayTracingAccelerationStructure || !device.capabilities().rayQuery) {
        log = "SceneRayQueryProgram requires rayTracingAccelerationStructure and rayQuery capabilities";
        return makeError(Error::Unsupported);
    }
    for (uint32_t bindingIndex = 0; bindingIndex < desc.bindingCount; ++bindingIndex) {
        if (desc.bindings[bindingIndex].kind == SceneRayQueryBindingKind::PartitionedAccelerationStructure &&
            !device.capabilities().partitionedAccelerationStructure) {
            log = "SceneRayQueryProgram requires partitionedAccelerationStructure capability";
            return makeError(Error::Unsupported);
        }
    }

    NativeDevice nativeDeviceInfo = nativeDevice(device);
    if (nativeDeviceInfo.device == VK_NULL_HANDLE) {
        log = "SceneRayQueryProgram Vulkan device is unavailable";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(nativeDeviceInfo.device);

    impl_->destroy();
    impl_->device = nativeDeviceInfo.device;
    impl_->pushConstantSize = desc.pushConstantSize;
    impl_->debugName = desc.debugName != nullptr ? desc.debugName : "SceneRayQueryProgram";
    impl_->bindings.assign(desc.bindings, desc.bindings + desc.bindingCount);

    std::vector<VkDescriptorSetLayoutBinding> vkBindings;
    vkBindings.reserve(desc.bindingCount);
    std::array<VkDescriptorPoolSize, 5> poolSizes{};
    uint32_t poolSizeCount = 0;
    for (const SceneRayQueryBindingDesc& binding : impl_->bindings) {
        const VkDescriptorType descriptorType = descriptorTypeFor(binding.kind);
        const uint32_t descriptorCount = std::max(binding.descriptorCount, 1u);
        vkBindings.push_back(VkDescriptorSetLayoutBinding{
            .binding = binding.binding,
            .descriptorType = descriptorType,
            .descriptorCount = descriptorCount,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        });
        if (descriptorCount > UINT32_MAX / desc.descriptorSetCount) {
            log = "SceneRayQueryProgram descriptor pool size overflows uint32_t";
            clear();
            return makeError(Error::InvalidArgument);
        }
        addPoolSize(
            poolSizes,
            poolSizeCount,
            descriptorType,
            descriptorCount * desc.descriptorSetCount);
    }

    VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(vkBindings.size()),
        .pBindings = vkBindings.data(),
    };
    VkResult vkResult = vkCreateDescriptorSetLayout(
        impl_->device,
        &setLayoutInfo,
        nullptr,
        &impl_->descriptorSetLayout);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreateDescriptorSetLayout(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = desc.descriptorSetCount,
        .poolSizeCount = poolSizeCount,
        .pPoolSizes = poolSizes.data(),
    };
    vkResult = vkCreateDescriptorPool(impl_->device, &poolInfo, nullptr, &impl_->descriptorPool);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreateDescriptorPool(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }

    std::vector<VkDescriptorSetLayout> setLayouts(
        desc.descriptorSetCount,
        impl_->descriptorSetLayout);
    impl_->descriptorSets.resize(desc.descriptorSetCount, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = impl_->descriptorPool,
        .descriptorSetCount = desc.descriptorSetCount,
        .pSetLayouts = setLayouts.data(),
    };
    vkResult = vkAllocateDescriptorSets(
        impl_->device,
        &allocateInfo,
        impl_->descriptorSets.data());
    if (vkResult != VK_SUCCESS) {
        log = "vkAllocateDescriptorSets(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }

    VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = desc.pushConstantSize,
    };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &impl_->descriptorSetLayout,
        .pushConstantRangeCount = desc.pushConstantSize > 0 ? 1u : 0u,
        .pPushConstantRanges = desc.pushConstantSize > 0 ? &pushRange : nullptr,
    };
    vkResult = vkCreatePipelineLayout(impl_->device, &pipelineLayoutInfo, nullptr, &impl_->pipelineLayout);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreatePipelineLayout(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }

    VkShaderModuleCreateInfo shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = desc.byteSize,
        .pCode = desc.spirv,
    };
    vkResult = vkCreateShaderModule(impl_->device, &shaderInfo, nullptr, &impl_->shaderModule);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreateShaderModule(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }
    profiling::registerNsightAftermathShaderBinary(desc.spirv, desc.byteSize);

    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = impl_->shaderModule,
        .pName = "main",
    };
    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stageInfo,
        .layout = impl_->pipelineLayout,
    };
    vkResult = vkCreateComputePipelines(
        impl_->device,
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &impl_->pipeline);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreateComputePipelines(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return resultFromVk(vkResult);
    }

    return {};
}

void SceneRayQueryProgram::clear()
{
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool SceneRayQueryProgram::valid() const
{
    return impl_ != nullptr &&
        impl_->device != VK_NULL_HANDLE &&
        !impl_->descriptorSets.empty() &&
        impl_->pipelineLayout != VK_NULL_HANDLE &&
        impl_->pipeline != VK_NULL_HANDLE;
}

Result SceneRayQueryProgram::dispatch(const SceneRayQueryDispatchDesc& desc)
{
    if (!valid() ||
        desc.commandBuffer == nullptr ||
        desc.groupCountX == 0 ||
        desc.groupCountY == 0 ||
        desc.groupCountZ == 0 ||
        desc.descriptorSetIndex >= impl_->descriptorSets.size() ||
        (impl_->pushConstantSize > 0 && (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        return makeError(Error::InvalidArgument);
    }

    const VkDescriptorSet descriptorSet = impl_->descriptorSets[desc.descriptorSetIndex];
    if (descriptorSet == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accelerationInfos;
    std::vector<VkAccelerationStructureKHR> accelerationStructures;
    std::vector<VkWriteDescriptorSetPartitionedAccelerationStructureNV> partitionedAccelerationInfos;
    std::vector<VkDeviceAddress> partitionedAccelerationStructures;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    writes.reserve(impl_->bindings.size());
    accelerationInfos.reserve(impl_->bindings.size());
    accelerationStructures.reserve(impl_->bindings.size());
    partitionedAccelerationInfos.reserve(impl_->bindings.size());
    partitionedAccelerationStructures.reserve(impl_->bindings.size());
    size_t imageInfoCapacity = 0;
    for (const SceneRayQueryBindingDesc& binding : impl_->bindings) {
        if (binding.kind == SceneRayQueryBindingKind::StorageImage ||
            binding.kind == SceneRayQueryBindingKind::SampledImage) {
            imageInfoCapacity += std::max(binding.descriptorCount, 1u);
        }
    }
    imageInfos.reserve(imageInfoCapacity);
    bufferInfos.reserve(impl_->bindings.size());

    for (const SceneRayQueryBindingDesc& expectedBinding : impl_->bindings) {
        const SceneRayQueryDispatchBinding* binding = findDispatchBinding(desc, expectedBinding.binding);
        if (binding == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = expectedBinding.binding,
            .descriptorCount = std::max(expectedBinding.descriptorCount, 1u),
            .descriptorType = descriptorTypeFor(expectedBinding.kind),
        };
        switch (expectedBinding.kind) {
        case SceneRayQueryBindingKind::AccelerationStructure: {
            VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
            if (binding->accelerationStructureHandle != 0) {
                static_assert(sizeof(binding->accelerationStructureHandle) == sizeof(accelerationStructure));
                accelerationStructure = std::bit_cast<VkAccelerationStructureKHR>(
                    binding->accelerationStructureHandle);
            } else if (binding->accelerationStructure != nullptr &&
                binding->accelerationStructure->valid() &&
                binding->accelerationStructure->impl_ != nullptr &&
                binding->accelerationStructure->impl_->tlas != VK_NULL_HANDLE) {
                accelerationStructure = binding->accelerationStructure->impl_->tlas;
            } else if (binding->clusterAccelerationStructure != nullptr &&
                binding->clusterAccelerationStructure->valid() &&
                binding->clusterAccelerationStructure->impl_ != nullptr &&
                binding->clusterAccelerationStructure->impl_->tlas != VK_NULL_HANDLE) {
                accelerationStructure = binding->clusterAccelerationStructure->impl_->tlas;
            }
            if (accelerationStructure == VK_NULL_HANDLE) {
                return makeError(Error::InvalidArgument);
            }
            write.descriptorCount = 1;
            accelerationStructures.push_back(accelerationStructure);
            accelerationInfos.push_back(VkWriteDescriptorSetAccelerationStructureKHR{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &accelerationStructures.back(),
            });
            write.pNext = &accelerationInfos.back();
            break;
        }
        case SceneRayQueryBindingKind::PartitionedAccelerationStructure:
            if (binding->partitionedAccelerationStructure == nullptr ||
                !binding->partitionedAccelerationStructure->valid() ||
                binding->partitionedAccelerationStructure->impl_ == nullptr ||
                binding->partitionedAccelerationStructure->impl_->ptlasAddress == 0) {
                return makeError(Error::InvalidArgument);
            }
            write.descriptorCount = 1;
            partitionedAccelerationStructures.push_back(
                binding->partitionedAccelerationStructure->impl_->ptlasAddress);
            partitionedAccelerationInfos.push_back(VkWriteDescriptorSetPartitionedAccelerationStructureNV{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &partitionedAccelerationStructures.back(),
            });
            write.pNext = &partitionedAccelerationInfos.back();
            break;
        case SceneRayQueryBindingKind::StorageImage: {
            const uint32_t descriptorCount = std::max(expectedBinding.descriptorCount, 1u);
            const bool useTextureArray =
                binding->textureViews != nullptr && binding->textureViewCount >= descriptorCount;
            if (!useTextureArray && (descriptorCount != 1u || binding->textureView == nullptr)) {
                return makeError(Error::InvalidArgument);
            }
            const size_t firstImageInfo = imageInfos.size();
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                TextureView* textureView = useTextureArray
                    ? binding->textureViews[index]
                    : binding->textureView;
                if (textureView == nullptr) {
                    return makeError(Error::InvalidArgument);
                }
                VkImageView imageView = nativeImageView(*textureView);
                if (imageView == VK_NULL_HANDLE) {
                    return makeError(Error::InvalidArgument);
                }
                imageInfos.push_back(VkDescriptorImageInfo{
                    .imageView = imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                });
            }
            write.descriptorCount = descriptorCount;
            write.pImageInfo = imageInfos.data() + firstImageInfo;
            break;
        }
        case SceneRayQueryBindingKind::SampledImage: {
            const uint32_t descriptorCount = std::max(expectedBinding.descriptorCount, 1u);
            if (binding->textureViews == nullptr || binding->textureViewCount < descriptorCount) {
                return makeError(Error::InvalidArgument);
            }
            const size_t firstImageInfo = imageInfos.size();
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                TextureView* textureView = binding->textureViews[index];
                if (textureView == nullptr) {
                    return makeError(Error::InvalidArgument);
                }
                VkImageView imageView = nativeImageView(*textureView);
                if (imageView == VK_NULL_HANDLE) {
                    return makeError(Error::InvalidArgument);
                }
                imageInfos.push_back(VkDescriptorImageInfo{
                    .imageView = imageView,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
            }
            write.descriptorCount = descriptorCount;
            write.pImageInfo = imageInfos.data() + firstImageInfo;
            break;
        }
        case SceneRayQueryBindingKind::StorageBuffer: {
            if (binding->buffer == nullptr) {
                return makeError(Error::InvalidArgument);
            }
            write.descriptorCount = 1;
            const NativeBuffer native = nativeBuffer(*binding->buffer);
            if (native.buffer == VK_NULL_HANDLE ||
                binding->offset > native.size ||
                (binding->size != UINT64_MAX && binding->size > native.size - binding->offset)) {
                return makeError(Error::InvalidArgument);
            }
            bufferInfos.push_back(VkDescriptorBufferInfo{
                .buffer = native.buffer,
                .offset = binding->offset,
                .range = binding->size == UINT64_MAX ? VK_WHOLE_SIZE : binding->size,
            });
            write.pBufferInfo = &bufferInfos.back();
            break;
        }
        }
        writes.push_back(write);
    }

    vkUpdateDescriptorSets(
        impl_->device,
        static_cast<uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    VkCommandBuffer commandBuffer = nativeCommandBuffer(*desc.commandBuffer);
    if (commandBuffer == VK_NULL_HANDLE) {
        return makeError(Error::InvalidArgument);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        impl_->pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    if (impl_->pushConstantSize > 0) {
        vkCmdPushConstants(
            commandBuffer,
            impl_->pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            desc.pushDataSize,
            desc.pushData);
    }
    vkCmdDispatch(commandBuffer, desc.groupCountX, desc.groupCountY, desc.groupCountZ);
    return {};
}

} // namespace metallic::render::vulkan
