#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <array>
#include <vector>

namespace metallic::render::detail {

// Backend-private packed device format. No scene or material objects enter this API.
enum class OpacityMicromapFormat : uint16_t {
    TwoState = 1,
    FourState = 2,
};

struct OpacityMicromapTriangle {
    uint32_t dataOffset = 0;
    uint16_t subdivisionLevel = 0;
    OpacityMicromapFormat format = OpacityMicromapFormat::FourState;
};
static_assert(sizeof(OpacityMicromapTriangle) == 8);

struct OpacityMicromapUsage {
    uint32_t count = 0;
    uint32_t subdivisionLevel = 0;
    OpacityMicromapFormat format = OpacityMicromapFormat::FourState;
};

struct BakedOpacityMicromap {
    std::vector<uint8_t> data;
    std::vector<OpacityMicromapTriangle> triangles;
    std::vector<OpacityMicromapUsage> usages;
    // Transparent, opaque, unknown-transparent, unknown-opaque microtriangles.
    std::array<uint64_t, 4> stateCounts{};
};

// Conservative coverage of bilinear/repeat sampling. Input triangles already
// contain the resolved texture-space UVs in original BLAS triangle order.
// The caller keeps the immutable input snapshot alive through bake().
class OpacityMicromapBaker {
public:
    explicit OpacityMicromapBaker(const RayTracingCoverageDesc& coverage);
    bool bake(uint32_t subdivisionLevel, BakedOpacityMicromap& output) const;

private:
    using UV = std::array<float, 2>;
    uint8_t classify(const std::array<UV, 3>& uv) const;
    RayTracingCoverageDesc coverage_;
    uint32_t width_ = 1, height_ = 1;
    bool valid_ = false;
    std::vector<uint32_t> opaquePrefix_, unknownPrefix_;
};

} // namespace metallic::render::detail
