#include "Runtime/Render/MeshletStreamClas.h"

#include "Runtime/Render/MeshletStreamResidency.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1u) / alignment) * alignment;
}

Result createClasBuffer(
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
    uint64_t scratchOffset = 0;
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
                entry.addressOffsetAndState = packMeshletStreamClasPageEntry(
                    page.addressOffset,
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

    ClusterAccelerationStructureProperties properties;
    Result result = device.queryClusterAccelerationStructureProperties(properties);
    if (!result ||
        properties.clusterStorageAlignment == 0 ||
        properties.scratchAlignment == 0 ||
        properties.triangleBuildInfoSize == 0) {
        log = std::string("queryClusterAccelerationStructureProperties returned ") +
            resultToString(result);
        return result ? makeError(Error::Failure) : result;
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
    result = device.queryClusterAccelerationStructureTriangleBuildSizes(
        ClusterAccelerationStructureTriangleBuildSizesDesc{
            .maxClusterTriangleCount = maxClusterTriangles,
            .maxClusterVertexCount = maxClusterVertices,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = maxClusterTriangles,
            .maxTotalVertexCount = maxClusterVertices,
            .vertexFormat = Format::Rgb32Sfloat,
            .maxAccelerationStructureCount = 1,
        },
        singleClusterSizes);
    if (!result || singleClusterSizes.accelerationStructureSize == 0) {
        log = std::string("queryClusterAccelerationStructureTriangleBuildSizes(single CLAS) returned ") +
            resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }

    ClusterAccelerationStructureBuildSizes batchSizes;
    result = device.queryClusterAccelerationStructureTriangleBuildSizes(
        ClusterAccelerationStructureTriangleBuildSizesDesc{
            .maxClusterTriangleCount = maxClusterTriangles,
            .maxClusterVertexCount = maxClusterVertices,
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .maxTotalTriangleCount = desc.maxBuildClusters * maxClusterTriangles,
            .maxTotalVertexCount = desc.maxBuildClusters * maxClusterVertices,
            .vertexFormat = Format::Rgb32Sfloat,
            .maxAccelerationStructureCount = desc.maxBuildClusters,
        },
        batchSizes);
    if (!result || batchSizes.buildScratchSize == 0) {
        log = std::string("queryClusterAccelerationStructureTriangleBuildSizes(batch CLAS) returned ") +
            resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }

    const uint64_t clusterStride = alignUp(
        singleClusterSizes.accelerationStructureSize,
        properties.clusterStorageAlignment);
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
    result = createClasBuffer(
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

    const uint64_t scratchBytes = batchSizes.buildScratchSize + properties.scratchAlignment - 1u;
    result = createClasBuffer(
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
    if (clusterSlotCapacity >
        static_cast<uint64_t>(kMeshletStreamClasPageAddressOffsetMask) + 1u) {
        log = "MeshletStreamClasPool cluster slot capacity exceeds packed page addressing";
        clear();
        return makeError(Error::InvalidArgument);
    }
    result = createClasBuffer(
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

    result = createClasBuffer(
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

    if (desc.maxBuildClusters >
        std::numeric_limits<uint64_t>::max() / properties.triangleBuildInfoSize) {
        log = "MeshletStreamClasPool build-info capacity overflowed";
        clear();
        return makeError(Error::InvalidArgument);
    }
    impl_->frames.resize(desc.queuedFrameCount);
    for (Impl::FrameResources& frame : impl_->frames) {
        result = createClasBuffer(
            device,
            static_cast<uint64_t>(desc.maxBuildClusters) * properties.triangleBuildInfoSize,
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
        result = createClasBuffer(
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

    const uint64_t storageAddress = impl_->storageBuffer->deviceAddress();
    const uint64_t scratchAddress = impl_->scratchBuffer->deviceAddress();
    if (storageAddress == 0 || scratchAddress == 0) {
        log = "MeshletStreamClasPool storage or scratch buffer has no device address";
        clear();
        return makeError(Error::Failure);
    }
    const uint64_t alignedScratchAddress = alignUp(scratchAddress, properties.scratchAlignment);
    const uint64_t scratchOffset = alignedScratchAddress - scratchAddress;
    if (scratchOffset > scratchBytes || batchSizes.buildScratchSize > scratchBytes - scratchOffset) {
        log = "MeshletStreamClasPool aligned scratch range exceeded its buffer";
        clear();
        return makeError(Error::Failure);
    }

    const uint32_t clusterIdStride = desc.asset->maxPageClusters();
    const uint64_t clusterIdCapacity =
        static_cast<uint64_t>(desc.asset->pageCount()) * clusterIdStride;
    if (clusterIdStride == 0 || clusterIdStride > 32) {
        log = "MeshletStreamClasPool page cluster count exceeds the traversal mask capacity";
        clear();
        return makeError(Error::InvalidArgument);
    }
    if (clusterIdCapacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ull) {
        log = "MeshletStreamClasPool page-strided cluster IDs exceed the 32-bit limit";
        clear();
        return makeError(Error::InvalidArgument);
    }

    impl_->asset = desc.asset;
    impl_->pages.reserve(std::min(desc.maxBuildClusters, desc.asset->pageCount()));
    impl_->storageAddress = storageAddress;
    impl_->scratchOffset = scratchOffset;
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
    if (!hasFlag(pageBuffer.desc().usage, BufferUsageBits::ShaderDeviceAddress) ||
        !hasFlag(pageBuffer.desc().usage, BufferUsageBits::AccelerationStructureBuildInput) ||
        pageBuffer.deviceAddress() == 0) {
        log = "MeshletStreamClasPool page buffer lacks AS build-input device-address usage";
        return makeError(Error::InvalidArgument);
    }

    struct PendingPage {
        uint32_t pageIndex = UINT32_MAX;
        uint32_t addressOffset = UINT32_MAX;
        MeshletStreamStorageAllocation allocation;
        std::vector<uint64_t> addresses;
    };
    std::vector<PendingPage> pendingPages;
    std::vector<ClusterAccelerationStructureTriangleBuildInfo> buildInfos;
    pendingPages.reserve(pages.size());
    buildInfos.reserve(impl_->maxBuildClusters);

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
        if (request.deviceOffsetBytes > pageBuffer.desc().size ||
            assetPage.uncompressedSize > pageBuffer.desc().size - request.deviceOffsetBytes) {
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
            const uint64_t destinationOffset = allocation.offset +
                static_cast<uint64_t>(clusterIndex) * impl_->clusterStride;
            buildInfos.push_back(ClusterAccelerationStructureTriangleBuildInfo{
                .clusterId = cluster.clusterId,
                .triangleCount = cluster.triangleCount,
                .vertexCount = cluster.vertexCount,
                .positionTruncateBitCount = 0,
                .geometryIndex = 0,
                .indexFormat = ClusterAccelerationStructureIndexFormat::Uint8,
                .indexBufferStride = 1,
                .vertexBufferStride = sizeof(float) * 4u,
                .indexBuffer = &pageBuffer,
                .indexBufferOffset = request.deviceOffsetBytes + cluster.triangleOffsetBytes,
                .vertexBuffer = &pageBuffer,
                .vertexBufferOffset = request.deviceOffsetBytes + cluster.vertexOffsetBytes,
                .destinationBuffer = impl_->storageBuffer.get(),
                .destinationBufferOffset = destinationOffset,
                .destinationSize = impl_->clusterStride,
                .opaque = true,
            });
            pending.addresses.push_back(impl_->storageAddress + destinationOffset);
        }
        pendingPages.push_back(std::move(pending));
    }

    if (buildInfos.empty()) {
        return {};
    }

    void* mappedAddresses = impl_->addressBuffer->map();
    if (mappedAddresses == nullptr) {
        rollback();
        log = "MeshletStreamClasPool address buffer did not map";
        return makeError(Error::Failure);
    }
    for (const PendingPage& pending : pendingPages) {
        std::memcpy(
            static_cast<uint8_t*>(mappedAddresses) +
                static_cast<uint64_t>(pending.addressOffset) * sizeof(uint64_t),
            pending.addresses.data(),
            pending.addresses.size() * sizeof(uint64_t));
    }
    impl_->addressBuffer->flush();
    impl_->addressBuffer->unmap();

    Impl::FrameResources& frame = impl_->frames[impl_->frameIndex % impl_->frames.size()];
    Result result = commandBuffer.buildClusterAccelerationStructureTriangles(
        ClusterAccelerationStructureTriangleBuildDesc{
            .clusters = buildInfos.data(),
            .clusterCount = static_cast<uint32_t>(buildInfos.size()),
            .maxClusterTriangleCount = impl_->asset->maxClusterTriangles(),
            .maxClusterVertexCount = impl_->asset->maxClusterVertices(),
            .maxClusterUniqueGeometryCount = 1,
            .maxGeometryIndexValue = 0,
            .minPositionTruncateBitCount = 0,
            .vertexFormat = Format::Rgb32Sfloat,
            .scratchBuffer = impl_->scratchBuffer.get(),
            .scratchBufferOffset = impl_->scratchOffset,
            .buildInfoBuffer = frame.buildInfoBuffer.get(),
            .destinationAddressBuffer = frame.destinationAddressBuffer.get(),
        });
    if (!result) {
        rollback();
        log = std::string("buildClusterAccelerationStructureTriangles returned ") +
            resultToString(result);
        return result;
    }

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

} // namespace metallic::render
