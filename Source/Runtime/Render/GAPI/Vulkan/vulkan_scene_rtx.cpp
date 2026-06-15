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

VkDescriptorType descriptorTypeFor(SceneRayQueryBindingKind kind)
{
    switch (kind) {
    case SceneRayQueryBindingKind::AccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    case SceneRayQueryBindingKind::StorageImage:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case SceneRayQueryBindingKind::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
    std::array<VkDescriptorPoolSize, 3>& poolSizes,
    uint32_t& poolSizeCount,
    VkDescriptorType type)
{
    for (uint32_t index = 0; index < poolSizeCount; ++index) {
        if (poolSizes[index].type == type) {
            ++poolSizes[index].descriptorCount;
            return;
        }
    }
    poolSizes[poolSizeCount++] = VkDescriptorPoolSize{
        .type = type,
        .descriptorCount = 1,
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

const SceneRtxStats& SceneRtxBuilder::stats() const
{
    static const SceneRtxStats kEmptyStats;
    return impl_ != nullptr ? impl_->stats : kEmptyStats;
}

struct SceneRayQueryProgram::Impl {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
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
                descriptorSet = VK_NULL_HANDLE;
            }
            if (descriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
                descriptorSetLayout = VK_NULL_HANDLE;
            }
        }

        device = VK_NULL_HANDLE;
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
        hasDuplicateBindings(desc.bindings, desc.bindingCount)) {
        log = "SceneRayQueryProgramDesc is invalid";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayTracingAccelerationStructure || !device.capabilities().rayQuery) {
        log = "SceneRayQueryProgram requires rayTracingAccelerationStructure and rayQuery capabilities";
        return makeError(Error::Unsupported);
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
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    uint32_t poolSizeCount = 0;
    for (const SceneRayQueryBindingDesc& binding : impl_->bindings) {
        const VkDescriptorType descriptorType = descriptorTypeFor(binding.kind);
        vkBindings.push_back(VkDescriptorSetLayoutBinding{
            .binding = binding.binding,
            .descriptorType = descriptorType,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        });
        addPoolSize(poolSizes, poolSizeCount, descriptorType);
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
        return makeError(Error::Failure);
    }

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = poolSizeCount,
        .pPoolSizes = poolSizes.data(),
    };
    vkResult = vkCreateDescriptorPool(impl_->device, &poolInfo, nullptr, &impl_->descriptorPool);
    if (vkResult != VK_SUCCESS) {
        log = "vkCreateDescriptorPool(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return makeError(Error::Failure);
    }

    VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = impl_->descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &impl_->descriptorSetLayout,
    };
    vkResult = vkAllocateDescriptorSets(impl_->device, &allocateInfo, &impl_->descriptorSet);
    if (vkResult != VK_SUCCESS) {
        log = "vkAllocateDescriptorSets(" + impl_->debugName + ") returned " +
            std::to_string(static_cast<int>(vkResult));
        clear();
        return makeError(Error::Failure);
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
        return makeError(Error::Failure);
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
        return makeError(Error::Failure);
    }

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
        return makeError(Error::Failure);
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
        impl_->descriptorSet != VK_NULL_HANDLE &&
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
        (impl_->pushConstantSize > 0 && (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        return makeError(Error::InvalidArgument);
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accelerationInfos;
    std::vector<VkAccelerationStructureKHR> accelerationStructures;
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    writes.reserve(impl_->bindings.size());
    accelerationInfos.reserve(impl_->bindings.size());
    accelerationStructures.reserve(impl_->bindings.size());
    imageInfos.reserve(impl_->bindings.size());
    bufferInfos.reserve(impl_->bindings.size());

    for (const SceneRayQueryBindingDesc& expectedBinding : impl_->bindings) {
        const SceneRayQueryDispatchBinding* binding = findDispatchBinding(desc, expectedBinding.binding);
        if (binding == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = impl_->descriptorSet,
            .dstBinding = expectedBinding.binding,
            .descriptorCount = 1,
            .descriptorType = descriptorTypeFor(expectedBinding.kind),
        };
        switch (expectedBinding.kind) {
        case SceneRayQueryBindingKind::AccelerationStructure:
            if (binding->accelerationStructure == nullptr ||
                !binding->accelerationStructure->valid() ||
                binding->accelerationStructure->impl_ == nullptr ||
                binding->accelerationStructure->impl_->tlas == VK_NULL_HANDLE) {
                return makeError(Error::InvalidArgument);
            }
            accelerationStructures.push_back(binding->accelerationStructure->impl_->tlas);
            accelerationInfos.push_back(VkWriteDescriptorSetAccelerationStructureKHR{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &accelerationStructures.back(),
            });
            write.pNext = &accelerationInfos.back();
            break;
        case SceneRayQueryBindingKind::StorageImage: {
            if (binding->textureView == nullptr) {
                return makeError(Error::InvalidArgument);
            }
            VkImageView imageView = nativeImageView(*binding->textureView);
            if (imageView == VK_NULL_HANDLE) {
                return makeError(Error::InvalidArgument);
            }
            imageInfos.push_back(VkDescriptorImageInfo{
                .imageView = imageView,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            });
            write.pImageInfo = &imageInfos.back();
            break;
        }
        case SceneRayQueryBindingKind::StorageBuffer: {
            if (binding->buffer == nullptr) {
                return makeError(Error::InvalidArgument);
            }
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
        &impl_->descriptorSet,
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
