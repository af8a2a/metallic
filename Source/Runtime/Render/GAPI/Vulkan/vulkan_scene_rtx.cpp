/*
 * Vulkan acceleration-structure build flow adapted from NVIDIA nvpro_core2 nvvk
 * acceleration_structures.cpp/.hpp.
 *
 * Copyright (c) 2014-2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Runtime/Render/GAPI/Vulkan/vulkan_scene_rtx.h"

#include "Runtime/Render/GAPI/Vulkan/vulkan_native.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render::vulkan {
namespace {

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

std::string resultMessage(const char* action, Result result)
{
    return std::string(action) + " returned " + resultToString(result);
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

VkAccelerationStructureGeometryKHR makeBlasGeometry(
    VkDeviceAddress vertexAddress,
    VkDeviceAddress indexAddress,
    const PrimitiveInput& input)
{
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
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
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
        .flags = kBuildFlags,
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
        return makeError(Error::Failure);
    }

    outAddress = accelerationStructureAddress(vkDevice, outHandle);
    if (outAddress == 0) {
        log = "vkGetAccelerationStructureDeviceAddressKHR returned 0";
        return makeError(Error::Failure);
    }

    return {};
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
        log = "Scene contains no triangle primitives suitable for RTX acceleration structures.";
        clear();
        return makeError(Error::Unsupported);
    }

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

    const NativeBuffer nativeVertexBuffer = nativeBuffer(*impl_->vertexBuffer);
    const NativeBuffer nativeIndexBuffer = nativeBuffer(*impl_->indexBuffer);
    if (nativeVertexBuffer.address == 0 || nativeIndexBuffer.address == 0) {
        log = "RTX geometry buffers do not have device addresses.";
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

    if (instances.empty()) {
        log = "Scene contains no visible RTX instances.";
        clear();
        return makeError(Error::Unsupported);
    }

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
    maxScratchSize = std::max(maxScratchSize, tlasSizeInfo.buildScratchSize);

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

    CommandBuffer* submittedCommandBuffers[] = {commandBuffer.get()};
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

    result = fence->wait();
    if (!result) {
        log = resultMessage("Fence::wait(RTX AS build)", result);
        clear();
        return result;
    }

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

    log = "Built Vulkan RTX acceleration structures: " +
        std::to_string(impl_->stats.blasCount) +
        " BLAS, " +
        std::to_string(impl_->stats.instanceCount) +
        " TLAS instances.";
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

VkAccelerationStructureKHR SceneRtxBuilder::tlas() const
{
    return impl_ != nullptr ? impl_->tlas : VK_NULL_HANDLE;
}

VkDeviceAddress SceneRtxBuilder::tlasDeviceAddress() const
{
    return impl_ != nullptr ? impl_->tlasAddress : 0;
}

const SceneRtxStats& SceneRtxBuilder::stats() const
{
    static const SceneRtxStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

} // namespace metallic::render::vulkan
