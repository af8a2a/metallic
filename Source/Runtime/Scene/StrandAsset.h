#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::scene {
struct StrandPoint
{
    std::array<float, 3> position{}, previousPosition{};
    float radius = 0;
};
struct NativeStrand
{
    uint32_t id = 0, material = 0;
    float opacity = 1;
    std::array<float, 3> normal{0, 0, 1};
    std::vector<StrandPoint> points;
};
struct StrandAsset
{
    std::filesystem::path materialRoot;
    std::vector<std::string> materials;
    std::vector<NativeStrand> strands;
    uint32_t segmentCount = 0;
};
// Transactional load. Stable authored IDs survive file ordering, deformation and LOD.
bool loadStrandAsset(const std::filesystem::path& path, StrandAsset& output, std::string& error);
} // namespace metallic::scene
