#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::scene {

struct MeshCardBuildRecord {
    std::array<float, 3> origin, axisX, axisY, axisZ, extent;
    uint32_t axisAlignedDirectionIndex;
};

struct MeshCards {
    uint64_t sourceFingerprint = 0;
    int32_t sourceMeshIndex = -1;
    std::vector<MeshCardBuildRecord> cards;
};

bool loadMeshCards(const std::filesystem::path& path, MeshCards& result, std::string& error);
bool saveMeshCards(const std::filesystem::path& path, const MeshCards& cards, std::string& error);

} // namespace metallic::scene
