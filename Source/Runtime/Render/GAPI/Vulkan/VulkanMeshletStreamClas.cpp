#include "Runtime/Render/GAPI/Vulkan/VulkanMeshletStreamClas.h"

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/MeshletStreamClas.h"
#include "Runtime/Render/MeshletStreamResidency.h"

#include <volk.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::render::vulkan {
namespace {

constexpr VkBuildAccelerationStructureFlagsKHR kClasBuildFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
constexpr uint64_t kDefaultClasAlignment = 256;

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1u) / alignment) * alignment;
}

uint64_t clusterStorageAlignment(VkPhysicalDevice physicalDevice)
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
        : kDefaultClasAlignment;
#else
    (void)physicalDevice;
    return kDefaultClasAlignment;
#endif
}

uint64_t clusterScratchAlignment(VkPhysicalDevice physicalDevice)
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
        : kDefaultClasAlignment;
#else
    (void)physicalDevice;
    return kDefaultClasAlignment;
#endif
}

Result createBuffer(
    Device& device,
    uint64_t size,
    BufferUsageBits usage,
    MemoryLocation location,
    std::unique_ptr<Buffer>& outBuffer,
    const char* label,
    std::string& log)
{
    Result result = device.createBuffer(
        BufferDesc{
            .size = size,
            .usage = usage,
            .memoryLocation = location,
        },
        outBuffer);
    if (!result || outBuffer == nullptr) {
        log = std::string(label) + " returned " + resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

void clusterBuildInputBarrier(VkCommandBuffer commandBuffer)
{
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void clusterBuildOutputBarrier(VkCommandBuffer commandBuffer)
{
    VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_SHADER_READ_BIT,
    };
    VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

struct MeshletStreamClasPool::Impl {
    enum class PageState : uint8_t {
        Empty,
        Built,
        Retiring,
    };

    struct PageEntry {
        MeshletStreamStorageAllocation allocation;
        uint32_t addressOffset = UINT32_MAX;
        uint32_t clusterCount = 0;
        uint32_t generation = 0;
        PageState state = PageState::Empty;
    };

    struct RetiredPage {
        uint32_t pageIndex = UINT32_MAX;
        uint32_t generation = 0;
        uint64_t completionFrame = 0;
    };

    struct FrameResources {
        std::unique_ptr<Buffer> buildInfoBuffer;
        std::unique_ptr<Buffer> destinationAddressBuffer;
    };

    const scene::MeshletStreamAsset* asset = nullptr;
    MeshletStreamStorage storage;
    std::unique_ptr<Buffer> storageBuffer;
    std::unique_ptr<Buffer> scratchBuffer;
    std::unique_ptr<Buffer> addressBuffer;
    std::unique_ptr<Buffer> pageTableBuffer;
    std::vector<FrameResources> frames;
    std::unordered_map<uint32_t, PageEntry> pages;
    std::vector<RetiredPage> retiredPages;
    MeshletStreamClasPoolStats stats;
    uint64_t frameIndex = 0;
    uint64_t storageAddress = 0;
    uint64_t scratchAddress = 0;
    uint64_t clusterStride = 0;
    uint32_t pageCount = 0;
    uint32_t clusterIdStride = 0;
    uint32_t maxBuildClusters = 0;
    uint32_t queuedFrameCount = 0;

    void writePageEntry(uint32_t pageIndex)
    {
        if (pageTableBuffer == nullptr || pageIndex >= pageCount) {
            return;
        }
        MeshletStreamClasPageEntry entry;
        auto pageIter = pages.find(pageIndex);
        if (pageIter != pages.end()) {
            const PageEntry& page = pageIter->second;
            if (page.state != PageState::Empty) {
                entry.addressOffset = page.addressOffset;
                entry.metadata = packMeshletStreamClasPageMetadata(
                    page.clusterCount,
                    page.state == PageState::Built
                        ? MeshletStreamClasPageState::Active
                        : MeshletStreamClasPageState::Retiring);
            }
        }

        const uint64_t offset = static_cast<uint64_t>(pageIndex) * sizeof(entry);
        void* mapped = pageTableBuffer->map();
        if (mapped == nullptr) {
            return;
        }
        std::memcpy(static_cast<uint8_t*>(mapped) + offset, &entry, sizeof(entry));
        pageTableBuffer->flush(offset, sizeof(entry));
        pageTableBuffer->unmap();
    }

    void releasePage(PageEntry& page)
    {
        const uint32_t generation = page.generation;
        if (!page.allocation.valid()) {
            page = {};
            page.generation = generation;
            return;
        }
        if (addressBuffer != nullptr && page.addressOffset != UINT32_MAX) {
            void* mapped = addressBuffer->map();
            if (mapped != nullptr) {
                std::memset(
                    static_cast<uint8_t*>(mapped) +
                        static_cast<uint64_t>(page.addressOffset) * sizeof(uint64_t),
                    0,
                    static_cast<size_t>(page.clusterCount) * sizeof(uint64_t));
                addressBuffer->flush(
                    static_cast<uint64_t>(page.addressOffset) * sizeof(uint64_t),
                    static_cast<uint64_t>(page.clusterCount) * sizeof(uint64_t));
                addressBuffer->unmap();
            }
        }
        storage.release(page.allocation);
        page = {};
        page.generation = generation;
    }
};

MeshletStreamClasPool::MeshletStreamClasPool()
    : impl_(std::make_unique<Impl>())
{
}

MeshletStreamClasPool::~MeshletStreamClasPool() = default;
MeshletStreamClasPool::MeshletStreamClasPool(MeshletStreamClasPool&&) noexcept = default;
MeshletStreamClasPool& MeshletStreamClasPool::operator=(MeshletStreamClasPool&&) noexcept = default;

Result MeshletStreamClasPool::initialize(
    Device& device,
    const MeshletStreamClasPoolDesc& desc,
    std::string& log)
{
    clear();
    log.clear();
#ifndef VK_NV_cluster_acceleration_structure
    (void)device;
    (void)desc;
    log = "VK_NV_cluster_acceleration_structure is unavailable in this Vulkan header";
    return makeError(Error::Unsupported);
#else
    if (desc.asset == nullptr ||
        !desc.asset->valid() ||
        desc.maxStorageBytes == 0 ||
        desc.maxBuildClusters == 0 ||
        desc.queuedFrameCount == 0) {
        log = "MeshletStreamClasPool requires a valid asset and non-zero bounded capacities";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().clusterAccelerationStructure) {
        log = "MeshletStreamClasPool requires cluster acceleration structure support";
        return makeError(Error::Unsupported);
    }

    const NativeDevice native = nativeDevice(device);
    if (native.device == VK_NULL_HANDLE || native.physicalDevice == VK_NULL_HANDLE) {
        log = "MeshletStreamClasPool native Vulkan device is unavailable";
        return makeError(Error::InvalidArgument);
    }
    volkLoadDevice(native.device);
    if (vkCmdBuildClusterAccelerationStructureIndirectNV == nullptr) {
        log = "vkCmdBuildClusterAccelerationStructureIndirectNV is unavailable";
        return makeError(Error::Unsupported);
    }

    const uint32_t maxClusterVertices = desc.asset->maxClusterVertices();
    const uint32_t maxClusterTriangles = desc.asset->maxClusterTriangles();
    if (maxClusterVertices == 0 ||
        maxClusterTriangles == 0 ||
        desc.maxBuildClusters > std::numeric_limits<uint32_t>::max() / maxClusterVertices ||
        desc.maxBuildClusters > std::numeric_limits<uint32_t>::max() / maxClusterTriangles) {
        log = "MeshletStreamClasPool CLAS build bounds overflowed";
        return makeError(Error::InvalidArgument);
    }

    ClusterAccelerationStructureBuildSizes singleClusterSizes;
    Result result = queryClusterAccelerationStructureTriangleBuildSizes(
        device,
        ClusterAccelerationStructureTriangleBuildSizesDesc{
            .flags = kClasBuildFlags,
            .maxClusterTriangleCount = maxClusterTriangles,
            .maxClusterVertexCount = maxClusterVertices,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = maxClusterTriangles,
            .maxTotalVertexCount = maxClusterVertices,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxAccelerationStructureCount = 1,
        },
        singleClusterSizes);
    if (!result || singleClusterSizes.accelerationStructureSize == 0) {
        log = std::string("queryClusterAccelerationStructureTriangleBuildSizes(single CLAS) returned ") +
            resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }

    ClusterAccelerationStructureBuildSizes batchSizes;
    result = queryClusterAccelerationStructureTriangleBuildSizes(
        device,
        ClusterAccelerationStructureTriangleBuildSizesDesc{
            .flags = kClasBuildFlags,
            .maxClusterTriangleCount = maxClusterTriangles,
            .maxClusterVertexCount = maxClusterVertices,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = desc.maxBuildClusters * maxClusterTriangles,
            .maxTotalVertexCount = desc.maxBuildClusters * maxClusterVertices,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxAccelerationStructureCount = desc.maxBuildClusters,
        },
        batchSizes);
    if (!result || batchSizes.buildScratchSize == 0) {
        log = std::string("queryClusterAccelerationStructureTriangleBuildSizes(batch CLAS) returned ") +
            resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }

    const uint64_t storageAlignment = clusterStorageAlignment(native.physicalDevice);
    const uint64_t clusterStride = alignUp(singleClusterSizes.accelerationStructureSize, storageAlignment);
    const uint64_t storageBytes = (desc.maxStorageBytes / clusterStride) * clusterStride;
    if (clusterStride == 0 || storageBytes < clusterStride) {
        log = "MeshletStreamClasPool storage budget cannot hold one maximum-size CLAS";
        return makeError(Error::OutOfMemory);
    }

    std::string reason;
    if (!impl_->storage.initialize(
            storageBytes,
            clusterStride,
            reason,
            std::numeric_limits<uint64_t>::max())) {
        log = "MeshletStreamClasPool storage allocator initialization failed: " + reason;
        return makeError(Error::InvalidArgument);
    }
    result = createBuffer(
        device,
        storageBytes,
        BufferUsageBits::AccelerationStructureStorage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->storageBuffer,
        "createBuffer(stream CLAS storage)",
        log);
    if (!result) {
        clear();
        return result;
    }

    const uint64_t scratchAlignment = clusterScratchAlignment(native.physicalDevice);
    const uint64_t scratchBytes = batchSizes.buildScratchSize + scratchAlignment - 1u;
    result = createBuffer(
        device,
        scratchBytes,
        BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::Device,
        impl_->scratchBuffer,
        "createBuffer(stream CLAS scratch)",
        log);
    if (!result) {
        clear();
        return result;
    }

    const uint64_t clusterSlotCapacity = storageBytes / clusterStride;
    if (clusterSlotCapacity > std::numeric_limits<uint32_t>::max()) {
        log = "MeshletStreamClasPool cluster slot capacity exceeds 32-bit addressing";
        clear();
        return makeError(Error::InvalidArgument);
    }
    result = createBuffer(
        device,
        clusterSlotCapacity * sizeof(uint64_t),
        BufferUsageBits::Storage |
            BufferUsageBits::AccelerationStructureBuildInput |
            BufferUsageBits::ShaderDeviceAddress,
        MemoryLocation::HostUpload,
        impl_->addressBuffer,
        "createBuffer(stream CLAS addresses)",
        log);
    if (!result) {
        clear();
        return result;
    }
    if (void* mapped = impl_->addressBuffer->map(); mapped != nullptr) {
        std::memset(mapped, 0, static_cast<size_t>(impl_->addressBuffer->desc().size));
        impl_->addressBuffer->flush();
        impl_->addressBuffer->unmap();
    } else {
        log = "MeshletStreamClasPool address buffer did not map";
        clear();
        return makeError(Error::Failure);
    }

    result = createBuffer(
        device,
        static_cast<uint64_t>(desc.asset->pageCount()) * sizeof(MeshletStreamClasPageEntry),
        BufferUsageBits::Storage,
        MemoryLocation::HostUpload,
        impl_->pageTableBuffer,
        "createBuffer(stream CLAS page table)",
        log);
    if (!result) {
        clear();
        return result;
    }
    if (void* mapped = impl_->pageTableBuffer->map(); mapped != nullptr) {
        std::fill_n(
            static_cast<MeshletStreamClasPageEntry*>(mapped),
            desc.asset->pageCount(),
            MeshletStreamClasPageEntry{});
        impl_->pageTableBuffer->flush();
        impl_->pageTableBuffer->unmap();
    } else {
        log = "MeshletStreamClasPool page table buffer did not map";
        clear();
        return makeError(Error::Failure);
    }

    impl_->frames.resize(desc.queuedFrameCount);
    for (Impl::FrameResources& frame : impl_->frames) {
        result = createBuffer(
            device,
            static_cast<uint64_t>(desc.maxBuildClusters) *
                sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
            BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            MemoryLocation::HostUpload,
            frame.buildInfoBuffer,
            "createBuffer(stream CLAS build infos)",
            log);
        if (!result) {
            clear();
            return result;
        }
        result = createBuffer(
            device,
            static_cast<uint64_t>(desc.maxBuildClusters) * sizeof(uint64_t),
            BufferUsageBits::Storage |
                BufferUsageBits::AccelerationStructureBuildInput |
                BufferUsageBits::ShaderDeviceAddress,
            MemoryLocation::HostUpload,
            frame.destinationAddressBuffer,
            "createBuffer(stream CLAS destination addresses)",
            log);
        if (!result) {
            clear();
            return result;
        }
    }

    const NativeBuffer nativeStorage = nativeBuffer(*impl_->storageBuffer);
    const NativeBuffer nativeScratch = nativeBuffer(*impl_->scratchBuffer);
    if (nativeStorage.address == 0 || nativeScratch.address == 0) {
        log = "MeshletStreamClasPool storage or scratch buffer has no device address";
        clear();
        return makeError(Error::Failure);
    }

    const uint32_t clusterIdStride = desc.asset->maxPageClusters();
    const uint64_t clusterIdCapacity =
        static_cast<uint64_t>(desc.asset->pageCount()) * clusterIdStride;
    if (clusterIdStride == 0 || clusterIdStride > kMeshletStreamClasClusterCountMask) {
        log = "MeshletStreamClasPool page cluster count exceeds packed GPU metadata";
        clear();
        return makeError(Error::InvalidArgument);
    }
    if (clusterIdCapacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ull) {
        log = "MeshletStreamClasPool page-strided cluster IDs exceed the 32-bit Vulkan limit";
        clear();
        return makeError(Error::InvalidArgument);
    }

    impl_->asset = desc.asset;
    impl_->pages.reserve(std::min(desc.maxBuildClusters, desc.asset->pageCount()));
    impl_->storageAddress = nativeStorage.address;
    impl_->scratchAddress = alignUp(nativeScratch.address, scratchAlignment);
    impl_->clusterStride = clusterStride;
    impl_->pageCount = desc.asset->pageCount();
    impl_->clusterIdStride = clusterIdStride;
    impl_->maxBuildClusters = desc.maxBuildClusters;
    impl_->queuedFrameCount = desc.queuedFrameCount;
    impl_->stats.pageCapacity = desc.asset->pageCount();
    impl_->stats.clusterSlotCapacity = static_cast<uint32_t>(clusterSlotCapacity);
    impl_->stats.storageBytes = storageBytes;
    impl_->stats.clusterStrideBytes = clusterStride;
    impl_->stats.scratchBytes = scratchBytes;
    return {};
#endif
}

void MeshletStreamClasPool::clear()
{
    impl_ = std::make_unique<Impl>();
}

void MeshletStreamClasPool::beginFrame()
{
    if (!ready()) {
        return;
    }
    ++impl_->frameIndex;
    impl_->stats.frameBuiltPageCount = 0;
    impl_->stats.frameBuiltClusterCount = 0;
    impl_->stats.frameRejectedPageCount = 0;

    for (auto iter = impl_->retiredPages.begin(); iter != impl_->retiredPages.end();) {
        if (iter->completionFrame > impl_->frameIndex) {
            ++iter;
            continue;
        }
        auto pageIter = impl_->pages.find(iter->pageIndex);
        if (pageIter != impl_->pages.end()) {
            Impl::PageEntry& page = pageIter->second;
            if (page.state == Impl::PageState::Retiring && page.generation == iter->generation) {
                impl_->stats.retiringPageCount -= 1u;
                impl_->stats.retiringClusterCount -= page.clusterCount;
                impl_->releasePage(page);
                impl_->writePageEntry(iter->pageIndex);
                impl_->pages.erase(pageIter);
            }
        }
        iter = impl_->retiredPages.erase(iter);
    }
}

Result MeshletStreamClasPool::cmdBuildPages(
    CommandBuffer& commandBuffer,
    Buffer& pageBuffer,
    std::span<const MeshletStreamClasPageBuild> pages,
    std::string& log)
{
    log.clear();
    if (!ready()) {
        return makeError(Error::InvalidArgument);
    }
#ifndef VK_NV_cluster_acceleration_structure
    (void)commandBuffer;
    (void)pageBuffer;
    (void)pages;
    return makeError(Error::Unsupported);
#else
    if (!hasFlag(pageBuffer.desc().usage, BufferUsageBits::ShaderDeviceAddress) ||
        !hasFlag(pageBuffer.desc().usage, BufferUsageBits::AccelerationStructureBuildInput)) {
        log = "MeshletStreamClasPool page buffer lacks AS build-input device-address usage";
        return makeError(Error::InvalidArgument);
    }

    const NativeBuffer nativePages = nativeBuffer(pageBuffer);
    const VkCommandBuffer nativeCommands = nativeCommandBuffer(commandBuffer);
    if (nativePages.address == 0 || nativeCommands == VK_NULL_HANDLE) {
        log = "MeshletStreamClasPool native page buffer or command buffer is unavailable";
        return makeError(Error::InvalidArgument);
    }

    struct PendingPage {
        uint32_t pageIndex = UINT32_MAX;
        uint32_t addressOffset = UINT32_MAX;
        MeshletStreamStorageAllocation allocation;
        std::vector<uint64_t> addresses;
    };
    std::vector<PendingPage> pendingPages;
    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> buildInfos;
    std::vector<uint64_t> destinationAddresses;
    pendingPages.reserve(pages.size());
    buildInfos.reserve(impl_->maxBuildClusters);
    destinationAddresses.reserve(impl_->maxBuildClusters);
    uint32_t totalTriangles = 0;
    uint32_t totalVertices = 0;

    auto rollback = [this, &pendingPages]() {
        for (PendingPage& pending : pendingPages) {
            impl_->storage.release(pending.allocation);
        }
    };

    for (const MeshletStreamClasPageBuild& request : pages) {
        if (request.pageIndex >= impl_->pageCount || request.deviceOffsetBytes == UINT64_MAX) {
            rollback();
            log = "MeshletStreamClasPool page build request is invalid";
            return makeError(Error::InvalidArgument);
        }
        auto existingIter = impl_->pages.find(request.pageIndex);
        if (existingIter != impl_->pages.end()) {
            Impl::PageEntry& existing = existingIter->second;
            if (existing.state == Impl::PageState::Built) {
                continue;
            }
            existing.state = Impl::PageState::Built;
            ++existing.generation;
            --impl_->stats.retiringPageCount;
            impl_->stats.retiringClusterCount -= existing.clusterCount;
            ++impl_->stats.builtPageCount;
            impl_->stats.builtClusterCount += existing.clusterCount;
            impl_->writePageEntry(request.pageIndex);
            continue;
        }
        if (std::find_if(
                pendingPages.begin(),
                pendingPages.end(),
                [request](const PendingPage& pending) {
                    return pending.pageIndex == request.pageIndex;
                }) != pendingPages.end()) {
            continue;
        }

        const scene::MeshletStreamPageInfo& assetPage = impl_->asset->pages()[request.pageIndex];
        if (request.deviceOffsetBytes > nativePages.size ||
            assetPage.uncompressedSize > nativePages.size - request.deviceOffsetBytes) {
            rollback();
            log = "MeshletStreamClasPool page-buffer range exceeded its bound";
            return makeError(Error::InvalidArgument);
        }
        if (assetPage.clusterCount > impl_->maxBuildClusters - buildInfos.size()) {
            ++impl_->stats.frameRejectedPageCount;
            ++impl_->stats.totalRejectedPageCount;
            continue;
        }

        const uint64_t allocationBytes =
            static_cast<uint64_t>(assetPage.clusterCount) * impl_->clusterStride;
        const MeshletStreamStorageAllocation allocation = impl_->storage.allocate(allocationBytes);
        if (!allocation.valid()) {
            ++impl_->stats.frameRejectedPageCount;
            ++impl_->stats.totalRejectedPageCount;
            continue;
        }

        MeshletStreamClasPagePlan plan;
        std::string reason;
        if (!buildMeshletStreamClasPagePlan(
                *impl_->asset,
                request.pageIndex,
                request.pageIndex * impl_->clusterIdStride,
                plan,
                reason)) {
            impl_->storage.release(allocation);
            rollback();
            log = "MeshletStreamClasPool page plan failed: " + reason;
            return makeError(Error::InvalidArgument);
        }

        PendingPage pending;
        pending.pageIndex = request.pageIndex;
        pending.addressOffset = static_cast<uint32_t>(allocation.offset / impl_->clusterStride);
        pending.allocation = allocation;
        pending.addresses.reserve(plan.clusters.size());

        for (uint32_t clusterIndex = 0; clusterIndex < plan.clusters.size(); ++clusterIndex) {
            const MeshletStreamClasClusterInput& cluster = plan.clusters[clusterIndex];
            VkClusterAccelerationStructureBuildTriangleClusterInfoNV buildInfo{};
            buildInfo.clusterID = cluster.clusterId;
            buildInfo.triangleCount = cluster.triangleCount;
            buildInfo.vertexCount = cluster.vertexCount;
            buildInfo.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
            buildInfo.indexBufferStride = 1;
            buildInfo.vertexBufferStride = sizeof(float) * 4u;
            buildInfo.indexBuffer =
                nativePages.address + request.deviceOffsetBytes + cluster.triangleOffsetBytes;
            buildInfo.vertexBuffer =
                nativePages.address + request.deviceOffsetBytes + cluster.vertexOffsetBytes;
            buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags =
                VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
            buildInfos.push_back(buildInfo);

            const uint64_t address =
                impl_->storageAddress + allocation.offset +
                static_cast<uint64_t>(clusterIndex) * impl_->clusterStride;
            destinationAddresses.push_back(address);
            pending.addresses.push_back(address);
            totalTriangles += cluster.triangleCount;
            totalVertices += cluster.vertexCount;
        }
        pendingPages.push_back(std::move(pending));
    }

    if (buildInfos.empty()) {
        return {};
    }

    Impl::FrameResources& frame = impl_->frames[impl_->frameIndex % impl_->frames.size()];
    void* mappedBuildInfos = frame.buildInfoBuffer->map();
    void* mappedDestinations = frame.destinationAddressBuffer->map();
    void* mappedAddresses = impl_->addressBuffer->map();
    if (mappedBuildInfos == nullptr || mappedDestinations == nullptr || mappedAddresses == nullptr) {
        if (mappedBuildInfos != nullptr) {
            frame.buildInfoBuffer->unmap();
        }
        if (mappedDestinations != nullptr) {
            frame.destinationAddressBuffer->unmap();
        }
        if (mappedAddresses != nullptr) {
            impl_->addressBuffer->unmap();
        }
        rollback();
        log = "MeshletStreamClasPool build upload buffers did not map";
        return makeError(Error::Failure);
    }

    const uint64_t buildInfoBytes =
        static_cast<uint64_t>(buildInfos.size()) * sizeof(buildInfos.front());
    const uint64_t destinationBytes =
        static_cast<uint64_t>(destinationAddresses.size()) * sizeof(destinationAddresses.front());
    std::memcpy(mappedBuildInfos, buildInfos.data(), static_cast<size_t>(buildInfoBytes));
    std::memcpy(mappedDestinations, destinationAddresses.data(), static_cast<size_t>(destinationBytes));
    for (const PendingPage& pending : pendingPages) {
        std::memcpy(
            static_cast<uint8_t*>(mappedAddresses) +
                static_cast<uint64_t>(pending.addressOffset) * sizeof(uint64_t),
            pending.addresses.data(),
            pending.addresses.size() * sizeof(uint64_t));
    }
    frame.buildInfoBuffer->flush(0, buildInfoBytes);
    frame.destinationAddressBuffer->flush(0, destinationBytes);
    impl_->addressBuffer->flush();
    frame.buildInfoBuffer->unmap();
    frame.destinationAddressBuffer->unmap();
    impl_->addressBuffer->unmap();

    const NativeBuffer nativeBuildInfos = nativeBuffer(*frame.buildInfoBuffer);
    const NativeBuffer nativeDestinations = nativeBuffer(*frame.destinationAddressBuffer);
    if (nativeBuildInfos.address == 0 || nativeDestinations.address == 0) {
        rollback();
        log = "MeshletStreamClasPool build upload buffers have no device address";
        return makeError(Error::Failure);
    }

    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .maxGeometryIndexValue = 0,
        .maxClusterUniqueGeometryCount = 1,
        .maxClusterTriangleCount = impl_->asset->maxClusterTriangles(),
        .maxClusterVertexCount = impl_->asset->maxClusterVertices(),
        .maxTotalTriangleCount = totalTriangles,
        .maxTotalVertexCount = totalVertices,
        .minPositionTruncateBitCount = 0,
    };
    VkClusterAccelerationStructureInputInfoNV input{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = static_cast<uint32_t>(buildInfos.size()),
        .flags = kClasBuildFlags,
        .opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
        .opInput = {.pTriangleClusters = &triangleInput},
    };
    VkClusterAccelerationStructureCommandsInfoNV commands{
        .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input = input,
        .scratchData = impl_->scratchAddress,
        .dstAddressesArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = nativeDestinations.address,
            .stride = sizeof(uint64_t),
            .size = destinationBytes,
        },
        .srcInfosArray = VkStridedDeviceAddressRegionKHR{
            .deviceAddress = nativeBuildInfos.address,
            .stride = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
            .size = buildInfoBytes,
        },
    };
    clusterBuildInputBarrier(nativeCommands);
    vkCmdBuildClusterAccelerationStructureIndirectNV(nativeCommands, &commands);
    clusterBuildOutputBarrier(nativeCommands);

    for (PendingPage& pending : pendingPages) {
        Impl::PageEntry& page = impl_->pages.try_emplace(pending.pageIndex).first->second;
        page.allocation = pending.allocation;
        page.addressOffset = pending.addressOffset;
        page.clusterCount = static_cast<uint32_t>(pending.addresses.size());
        ++page.generation;
        page.state = Impl::PageState::Built;
        impl_->writePageEntry(pending.pageIndex);
        ++impl_->stats.builtPageCount;
        impl_->stats.builtClusterCount += page.clusterCount;
        ++impl_->stats.frameBuiltPageCount;
        impl_->stats.frameBuiltClusterCount += page.clusterCount;
        ++impl_->stats.totalBuiltPageCount;
        impl_->stats.totalBuiltClusterCount += page.clusterCount;
    }
    return {};
#endif
}

void MeshletStreamClasPool::retirePages(std::span<const uint32_t> pageIndices)
{
    if (!ready()) {
        return;
    }
    for (uint32_t pageIndex : pageIndices) {
        if (pageIndex >= impl_->pageCount) {
            continue;
        }
        auto pageIter = impl_->pages.find(pageIndex);
        if (pageIter == impl_->pages.end()) {
            continue;
        }
        Impl::PageEntry& page = pageIter->second;
        if (page.state != Impl::PageState::Built) {
            continue;
        }
        page.state = Impl::PageState::Retiring;
        ++page.generation;
        --impl_->stats.builtPageCount;
        impl_->stats.builtClusterCount -= page.clusterCount;
        ++impl_->stats.retiringPageCount;
        impl_->stats.retiringClusterCount += page.clusterCount;
        impl_->retiredPages.push_back(Impl::RetiredPage{
            .pageIndex = pageIndex,
            .generation = page.generation,
            .completionFrame = impl_->frameIndex + impl_->queuedFrameCount,
        });
        impl_->writePageEntry(pageIndex);
    }
}

bool MeshletStreamClasPool::ready() const
{
    return impl_ != nullptr &&
        impl_->asset != nullptr &&
        impl_->storageBuffer != nullptr &&
        impl_->scratchBuffer != nullptr &&
        impl_->addressBuffer != nullptr &&
        impl_->pageTableBuffer != nullptr &&
        !impl_->frames.empty();
}

bool MeshletStreamClasPool::pageHasClas(uint32_t pageIndex) const
{
    if (!ready() || pageIndex >= impl_->pageCount) {
        return false;
    }
    const auto pageIter = impl_->pages.find(pageIndex);
    return pageIter != impl_->pages.end() && pageIter->second.state != Impl::PageState::Empty;
}

uint32_t MeshletStreamClasPool::pageClasAddressOffset(uint32_t pageIndex) const
{
    if (!pageHasClas(pageIndex)) {
        return UINT32_MAX;
    }
    return impl_->pages.find(pageIndex)->second.addressOffset;
}

uint64_t MeshletStreamClasPool::clusterAddress(uint32_t pageIndex, uint32_t clusterIndex) const
{
    if (!pageHasClas(pageIndex)) {
        return 0;
    }
    const Impl::PageEntry& page = impl_->pages.find(pageIndex)->second;
    if (clusterIndex >= page.clusterCount) {
        return 0;
    }
    return impl_->storageAddress + page.allocation.offset +
        static_cast<uint64_t>(clusterIndex) * impl_->clusterStride;
}

Buffer* MeshletStreamClasPool::clusterAddressBuffer() const
{
    return ready() ? impl_->addressBuffer.get() : nullptr;
}

Buffer* MeshletStreamClasPool::pageTableBuffer() const
{
    return ready() ? impl_->pageTableBuffer.get() : nullptr;
}

MeshletStreamClasPoolStats MeshletStreamClasPool::stats() const
{
    if (!ready()) {
        return {};
    }
    MeshletStreamClasPoolStats result = impl_->stats;
    result.trackedPageCount = static_cast<uint32_t>(impl_->pages.size());
    result.usedStorageBytes = impl_->storage.usedBytes();
    return result;
}

} // namespace metallic::render::vulkan
