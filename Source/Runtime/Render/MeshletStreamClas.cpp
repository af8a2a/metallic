#include "Runtime/Render/MeshletStreamClas.h"

#include <cstring>
#include <limits>

namespace metallic::render {
namespace {

bool byteRangeWithin(uint64_t byteSize, uint64_t offset, uint64_t rangeSize)
{
    return offset <= byteSize && rangeSize <= byteSize - offset;
}

} // namespace

bool buildMeshletStreamPageClusterOffsets(
    const scene::MeshletStreamAsset& asset,
    std::vector<uint32_t>& outOffsets,
    uint32_t& outClusterCount,
    std::string& reason)
{
    outOffsets.clear();
    outClusterCount = 0;
    reason.clear();
    if (!asset.valid()) {
        reason = "meshlet stream CLAS page offsets require a valid streamasset";
        return false;
    }

    outOffsets.reserve(static_cast<size_t>(asset.pageCount()) + 1u);
    uint64_t clusterCount = 0;
    for (const scene::MeshletStreamPageInfo& page : asset.pages()) {
        outOffsets.push_back(static_cast<uint32_t>(clusterCount));
        clusterCount += page.clusterCount;
        if (clusterCount > std::numeric_limits<uint32_t>::max()) {
            outOffsets.clear();
            reason = "meshlet stream CLAS cluster IDs exceed the 32-bit limit";
            return false;
        }
    }
    outOffsets.push_back(static_cast<uint32_t>(clusterCount));
    outClusterCount = static_cast<uint32_t>(clusterCount);
    return true;
}

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamPageInfo& page,
    std::span<const uint8_t> devicePayload,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason)
{
    outPlan = {};
    reason.clear();
    if (devicePayload.size() != page.uncompressedSize ||
        devicePayload.size() < sizeof(scene::MeshletStreamPayloadHeader) ||
        devicePayload.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "meshlet stream CLAS payload does not match its page directory";
        return false;
    }

    scene::MeshletStreamPayloadHeader header;
    std::memcpy(&header, devicePayload.data(), sizeof(header));
    const uint64_t clusterBytes =
        static_cast<uint64_t>(header.clusterCount) * sizeof(scene::MeshletStreamPayloadCluster);
    const uint64_t positionBytes = static_cast<uint64_t>(header.vertexCount) * sizeof(float) * 4u;
    const uint64_t triangleBytes = header.triangleIndexCount;
    if (header.clusterCount == 0 ||
        header.clusterCount != page.clusterCount ||
        header.vertexCount != page.vertexCount ||
        header.triangleIndexCount != page.triangleIndexCount ||
        header.primitiveIndex != page.primitiveIndex ||
        header.materialIndex != page.materialIndex ||
        header.lodLevel != page.lodLevel ||
        header.lodGroupIndex != page.lodGroupIndex ||
        header.payloadByteSize != devicePayload.size() ||
        header.positionFormat != static_cast<uint32_t>(scene::MeshletStreamPayloadFormat::Float32x4) ||
        (header.attributeFlags & scene::kMeshletStreamPayloadAttributePosition) == 0u ||
        !byteRangeWithin(devicePayload.size(), header.clusterOffsetBytes, clusterBytes) ||
        !byteRangeWithin(devicePayload.size(), header.positionOffsetBytes, positionBytes) ||
        !byteRangeWithin(devicePayload.size(), header.triangleOffsetBytes, triangleBytes) ||
        firstClusterId > std::numeric_limits<uint32_t>::max() - (header.clusterCount - 1u)) {
        reason = "meshlet stream CLAS payload header is invalid";
        return false;
    }

    outPlan.pageIndex = pageIndex;
    outPlan.firstClusterId = firstClusterId;
    outPlan.primitiveIndex = page.primitiveIndex;
    outPlan.lodLevel = page.lodLevel;
    outPlan.payloadByteSize = static_cast<uint32_t>(devicePayload.size());
    outPlan.clusters.reserve(header.clusterCount);

    for (uint32_t clusterIndex = 0; clusterIndex < header.clusterCount; ++clusterIndex) {
        scene::MeshletStreamPayloadCluster cluster;
        const uint64_t clusterOffset =
            static_cast<uint64_t>(header.clusterOffsetBytes) +
            static_cast<uint64_t>(clusterIndex) * sizeof(cluster);
        std::memcpy(&cluster, devicePayload.data() + clusterOffset, sizeof(cluster));

        const uint64_t indexCount = static_cast<uint64_t>(cluster.triangleCount) * 3u;
        if (cluster.vertexCount == 0 ||
            cluster.triangleCount == 0 ||
            cluster.primitiveIndex != header.primitiveIndex ||
            cluster.lodLevel != header.lodLevel ||
            cluster.lodGroupIndex != header.lodGroupIndex ||
            cluster.vertexOffset > header.vertexCount ||
            cluster.vertexCount > header.vertexCount - cluster.vertexOffset ||
            cluster.triangleOffset > header.triangleIndexCount ||
            indexCount > header.triangleIndexCount - cluster.triangleOffset) {
            outPlan = {};
            reason = "meshlet stream CLAS cluster range is invalid";
            return false;
        }

        const uint64_t vertexOffsetBytes =
            static_cast<uint64_t>(header.positionOffsetBytes) +
            static_cast<uint64_t>(cluster.vertexOffset) * sizeof(float) * 4u;
        const uint64_t triangleOffsetBytes =
            static_cast<uint64_t>(header.triangleOffsetBytes) + cluster.triangleOffset;
        if (vertexOffsetBytes > std::numeric_limits<uint32_t>::max() ||
            triangleOffsetBytes > std::numeric_limits<uint32_t>::max()) {
            outPlan = {};
            reason = "meshlet stream CLAS cluster offsets exceed the 32-bit page range";
            return false;
        }

        for (uint64_t index = 0; index < indexCount; ++index) {
            if (devicePayload[triangleOffsetBytes + index] >= cluster.vertexCount) {
                outPlan = {};
                reason = "meshlet stream CLAS cluster contains an out-of-range local vertex index";
                return false;
            }
        }

        outPlan.clusters.push_back(MeshletStreamClasClusterInput{
            .clusterId = firstClusterId + clusterIndex,
            .pageIndex = pageIndex,
            .clusterIndex = clusterIndex,
            .primitiveIndex = cluster.primitiveIndex,
            .materialIndex = cluster.materialIndex,
            .vertexOffsetBytes = static_cast<uint32_t>(vertexOffsetBytes),
            .vertexCount = cluster.vertexCount,
            .triangleOffsetBytes = static_cast<uint32_t>(triangleOffsetBytes),
            .triangleCount = cluster.triangleCount,
        });
    }
    return true;
}

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamAsset& asset,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason)
{
    outPlan = {};
    reason.clear();
    if (!asset.valid() || pageIndex >= asset.pages().size()) {
        reason = "meshlet stream CLAS page index is invalid";
        return false;
    }

    std::vector<uint8_t> scratchPayload;
    std::span<const uint8_t> devicePayload;
    const scene::MeshletStreamPageInfo& page = asset.pages()[pageIndex];
    if (!scene::decodeMeshletStreamPayloadForDevice(
            page,
            asset.pagePayload(pageIndex),
            scratchPayload,
            devicePayload,
            reason)) {
        return false;
    }
    return buildMeshletStreamClasPagePlan(
        page,
        devicePayload,
        pageIndex,
        firstClusterId,
        outPlan,
        reason);
}

} // namespace metallic::render
