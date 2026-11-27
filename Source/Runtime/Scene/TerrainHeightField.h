#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ml.h"

namespace metallic::scene {

struct ScenePickRay;

struct TerrainSurfaceSample {
    float3 position;
    float3 normal;
};

struct TerrainRayHit {
    float3 position;
    float3 normal;
    float distance;
};

// Source heightfield in world coordinates. The analytic surface follows the two
// triangles per cell of the source mesh, including the authored smooth normals.
class TerrainHeightField {
public:
    // MTHF v1 stores the magic, version, columns, rows and (minX, minZ, maxX,
    // maxZ), followed by row-major float32 world heights. All fields are little-endian.
    static std::optional<TerrainHeightField> load(const std::filesystem::path& path, std::string& error);
    static std::optional<TerrainHeightField> create(uint32_t columns, uint32_t rows,
        float2 minimumXZ, float2 maximumXZ, std::vector<float> worldHeights, std::string& error);

    std::optional<TerrainSurfaceSample> sample(float worldX, float worldZ) const;
    std::optional<TerrainRayHit> raycast(const ScenePickRay& ray) const;

private:
    TerrainHeightField() = default;

    struct Triangle {
        size_t center;
        size_t x;
        size_t z;
        double weightX;
        double weightZ;
    };

    Triangle triangle(size_t column, size_t row, double u, double v) const;
    double height(const Triangle& weights) const;

    std::vector<float> xCoordinates_;
    std::vector<float> zCoordinates_;
    std::vector<float> heights_;
    std::vector<float3> normals_;
    float minimumHeight_ = 0.0f;
    float maximumHeight_ = 0.0f;
};

} // namespace metallic::scene
