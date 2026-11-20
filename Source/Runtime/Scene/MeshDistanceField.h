#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::scene {

class Scene;

inline constexpr uint32_t kMeshDistanceFieldMipCount = 3;
inline constexpr uint32_t kMeshDistanceFieldBrickSize = 8;
inline constexpr uint32_t kMeshDistanceFieldUniqueBrickSize = 7;
inline constexpr uint32_t kMeshDistanceFieldInvalidBrick = UINT32_MAX;

struct MeshDistanceFieldMip {
    std::array<uint32_t, 3> indirectionDimensions{};
    std::array<float, 3> volumeToVirtualUVScale{};
    std::array<float, 3> volumeToVirtualUVAdd{};
    std::array<float, 2> distanceFieldToVolumeScaleBias{};
    std::vector<uint32_t> indirection;
    std::vector<uint8_t> bricks;
};

struct MeshDistanceField {
    uint64_t sourceFingerprint = 0;
    int32_t sourceMeshIndex = -1;
    std::array<float, 3> localBoundsMin{};
    std::array<float, 3> localBoundsMax{};
    std::array<MeshDistanceFieldMip, kMeshDistanceFieldMipCount> mips;
};

struct MeshDistanceFieldGeometry {
    std::vector<std::array<float, 3>> positions;
    std::vector<uint32_t> indices;
    uint64_t fingerprint = 0;
};

bool getMeshDistanceFieldGeometry(const Scene& scene, int32_t meshIndex,
    MeshDistanceFieldGeometry& result, std::string& error);
bool loadMeshDistanceField(const std::filesystem::path& path, MeshDistanceField& result, std::string& error);
bool saveMeshDistanceField(const std::filesystem::path& path, const MeshDistanceField& field, std::string& error);

} // namespace metallic::scene
