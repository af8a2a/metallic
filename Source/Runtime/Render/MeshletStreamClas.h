#pragma once

#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

struct MeshletStreamClasClusterInput {
    uint32_t clusterId = 0;
    uint32_t pageIndex = 0;
    uint32_t clusterIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffsetBytes = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffsetBytes = 0;
    uint32_t triangleCount = 0;
};

struct MeshletStreamClasPagePlan {
    uint32_t pageIndex = 0;
    uint32_t firstClusterId = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t payloadByteSize = 0;
    std::vector<MeshletStreamClasClusterInput> clusters;
};

bool buildMeshletStreamPageClusterOffsets(
    const scene::MeshletStreamAsset& asset,
    std::vector<uint32_t>& outOffsets,
    uint32_t& outClusterCount,
    std::string& reason);

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamPageInfo& page,
    std::span<const uint8_t> devicePayload,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason);

bool buildMeshletStreamClasPagePlan(
    const scene::MeshletStreamAsset& asset,
    uint32_t pageIndex,
    uint32_t firstClusterId,
    MeshletStreamClasPagePlan& outPlan,
    std::string& reason);

} // namespace metallic::render
