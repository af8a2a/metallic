#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapBake.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace metallic::render::detail {
namespace {

// Adapted from Khronos Vulkan's VK_KHR_opacity_micromap Reference Code,
// Copyright 2022-2026 The Khronos Group Inc., CC-BY-4.0.
// https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html
// Used only for strictly interior microtriangle centroids, so no edge clamping is needed.
uint32_t microtriangleIndex(float u, float v, uint32_t level)
{
    const uint32_t scale = 1u << level;
    const float fu = u * float(scale), fv = v * float(scale);
    const uint32_t iu = uint32_t(fu), iv = uint32_t(fv);
    uint32_t iw = ~(iu + iv);
    if ((fu - float(iu)) + (fv - float(iv)) >= 1.0f && iu + iv < scale - 1u) {
        --iw;
    }
    uint32_t b0 = ~(iu ^ iw) & (scale - 1u);
    const uint32_t t = (iu ^ iv) & b0;
    uint32_t f = t;
    f ^= f >> 1u;
    f ^= f >> 2u;
    f ^= f >> 4u;
    f ^= f >> 8u;
    uint32_t b1 = ((f ^ iu) & ~b0) | t;
    const auto interleave = [](uint32_t bits) {
        bits = (bits | (bits << 8u)) & 0x00ff00ffu;
        bits = (bits | (bits << 4u)) & 0x0f0f0f0fu;
        bits = (bits | (bits << 2u)) & 0x33333333u;
        return (bits | (bits << 1u)) & 0x55555555u;
    };
    return interleave(b0) | (interleave(b1) << 1u);
}

} // namespace

OpacityMicromapBaker::OpacityMicromapBaker(const RayTracingCoverageDesc& coverage)
    : coverage_(coverage)
{
    if ((coverage.mode != RayTracingCoverageMode::Mask && coverage.mode != RayTracingCoverageMode::Blend) ||
        !std::isfinite(coverage.alphaFactor) || !std::isfinite(coverage.alphaCutoff)) {
        return;
    }
    if (!coverage.pixelsRGBA8.empty()) {
        width_ = coverage.width;
        height_ = coverage.height;
        // Bound CPU bake memory to 128 MiB per coverage source (two tables).
        if (width_ == 0 || height_ == 0 ||
            uint64_t(width_) + 1 > (16ull * 1024 * 1024) / (uint64_t(height_) + 1) ||
            coverage.pixelsRGBA8.size() < uint64_t(width_) * height_ * 4) {
            return;
        }
    }
    const size_t stride = size_t(width_) + 1;
    opaquePrefix_.resize(stride * (size_t(height_) + 1), 0);
    unknownPrefix_.resize(opaquePrefix_.size(), 0);
    for (uint32_t y = 0; y < height_; ++y) {
        uint32_t rowOpaque = 0, rowUnknown = 0;
        for (uint32_t x = 0; x < width_; ++x) {
            const float textureAlpha = coverage.pixelsRGBA8.empty() ? 1.0f : float(coverage.pixelsRGBA8[(size_t(y) * width_ + x) * 4 + 3]) / 255.0f;
            const float alpha = std::clamp(coverage.alphaFactor * textureAlpha, 0.0f, 1.0f);
            const bool opaque = coverage.mode == RayTracingCoverageMode::Mask ? alpha >= std::clamp(coverage.alphaCutoff, 0.0f, 1.0f) : alpha >= 1.0f;
            rowOpaque += opaque;
            rowUnknown += coverage.mode == RayTracingCoverageMode::Blend && alpha > 0.0f && alpha < 1.0f;
            const size_t index = (size_t(y) + 1) * stride + x + 1;
            opaquePrefix_[index] = opaquePrefix_[index - stride] + rowOpaque;
            unknownPrefix_[index] = unknownPrefix_[index - stride] + rowUnknown;
        }
    }
    valid_ = true;
}

uint8_t OpacityMicromapBaker::classify(const std::array<UV, 3>& uv) const
{
    if (std::any_of(uv.begin(), uv.end(), [](const UV& value) {
            return !std::isfinite(value[0]) || !std::isfinite(value[1]);
        })) {
        return 3;
    }
    double minX = std::min({double(uv[0][0]), double(uv[1][0]), double(uv[2][0])});
    double maxX = std::max({double(uv[0][0]), double(uv[1][0]), double(uv[2][0])});
    double minY = std::min({double(uv[0][1]), double(uv[1][1]), double(uv[2][1])});
    double maxY = std::max({double(uv[0][1]), double(uv[1][1]), double(uv[2][1])});
    if (!std::isfinite(minX + maxX + minY + maxY) ||
        std::max({std::abs(minX), std::abs(maxX), std::abs(minY), std::abs(maxY)}) > 1048576.0) {
        return 3;
    }
    // Include round-off at microtriangle/texel boundaries, including wrap seams.
    const double pad = 0.000004 * (1.0 + std::max({std::abs(minX), std::abs(maxX), std::abs(minY), std::abs(maxY)}));
    const auto intervals = [pad](double low, double high, uint32_t dimension) {
        std::array<std::array<uint32_t, 2>, 2> ranges{};
        // sampleAlphaCoverage uses texel = frac(uv) * dimension - 0.5
        // and blends floor(texel) with floor(texel) + 1, including wrap seams.
        int64_t begin = int64_t(std::floor((low - pad) * dimension - 0.5));
        int64_t end = int64_t(std::floor((high + pad) * dimension - 0.5)) + 2;
        if (end - begin >= dimension) {
            ranges[0] = {0, dimension};
        } else {
            const uint32_t first = uint32_t((begin % dimension + dimension) % dimension);
            const uint32_t last = first + uint32_t(end - begin);
            ranges[0] = {first, std::min(last, dimension)};
            if (last > dimension) {
                ranges[1] = {0, last - dimension};
            }
        }
        return ranges;
    };
    const auto xs = intervals(minX, maxX, width_), ys = intervals(minY, maxY, height_);
    uint64_t area = 0, opaque = 0, unknown = 0;
    const size_t stride = size_t(width_) + 1;
    for (const auto& x : xs) {
        for (const auto& y : ys) {
            if (x[0] == x[1] || y[0] == y[1]) {
                continue;
            }
            const auto sum = [&](const std::vector<uint32_t>& prefix) -> uint32_t {
                return prefix[size_t(y[1]) * stride + x[1]] - prefix[size_t(y[0]) * stride + x[1]] -
                    prefix[size_t(y[1]) * stride + x[0]] + prefix[size_t(y[0]) * stride + x[0]];
            };
            area += uint64_t(x[1] - x[0]) * (y[1] - y[0]);
            opaque += sum(opaquePrefix_);
            unknown += sum(unknownPrefix_);
        }
    }
    return unknown != 0 ? 3 : opaque == 0 ? 0 : opaque == area ? 1 : 3;
}

bool OpacityMicromapBaker::bake(uint32_t level, BakedOpacityMicromap& output) const
{
    output = {};
    const size_t count = coverage_.triangles.size();
    if (!valid_ || level > 8 || count == 0 || count > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t microCount = 1u << (2 * level), scale = 1u << level;
    // Avoid an unbounded scene bake; callers retain normal alpha traversal on failure.
    constexpr size_t kMaxBakeBytes = 256ull * 1024 * 1024;
    if (count > kMaxBakeBytes / (sizeof(OpacityMicromapTriangle) + std::max(1u, microCount / 4))) {
        return false;
    }
    std::vector<uint32_t> usageCounts(level + 1, 0);
    output.triangles.reserve(count);
    for (size_t triangle = 0; triangle < count; ++triangle) {
        const auto& uv = coverage_.triangles[triangle].uv;
        const uint8_t wholeState = classify(uv);
        const uint32_t actualLevel = wholeState < 2 ? 0 : level;
        const uint32_t offset = uint32_t(output.data.size());
        const uint32_t byteCount = std::max(1u, (1u << (actualLevel * 2)) / 4);
        output.triangles.push_back({offset, uint16_t(actualLevel), OpacityMicromapFormat::FourState});
        ++usageCounts[actualLevel];
        output.data.resize(offset + byteCount, 0);
        if (actualLevel == 0) {
            output.data[offset] = wholeState;
            ++output.stateCounts[wholeState];
            continue;
        }
        const auto store = [&](std::array<UV, 3> bary) {
            std::array<UV, 3> microUv;
            UV centroid{0.0f, 0.0f};
            for (size_t i = 0; i < 3; ++i) {
                const float u = bary[i][0] / float(scale), v = bary[i][1] / float(scale);
                centroid[0] += u / 3.0f;
                centroid[1] += v / 3.0f;
                microUv[i] = {uv[0][0] * (1.0f - u - v) + uv[1][0] * u + uv[2][0] * v,
                    uv[0][1] * (1.0f - u - v) + uv[1][1] * u + uv[2][1] * v};
            }
            const uint32_t index = microtriangleIndex(centroid[0], centroid[1], level);
            const uint8_t state = classify(microUv);
            output.data[offset + index / 4] |= uint8_t(state << ((index % 4) * 2));
            ++output.stateCounts[state];
        };
        for (uint32_t y = 0; y < scale; ++y) {
            for (uint32_t x = 0; x < scale - y; ++x) {
                store({UV{float(x), float(y)}, UV{float(x + 1), float(y)}, UV{float(x), float(y + 1)}});
                if (x + y + 1 < scale) {
                    store({UV{float(x + 1), float(y)}, UV{float(x + 1), float(y + 1)}, UV{float(x), float(y + 1)}});
                }
            }
        }
    }
    for (uint32_t i = 0; i <= level; ++i) {
        if (usageCounts[i] != 0) {
            output.usages.push_back({usageCounts[i], i, OpacityMicromapFormat::FourState});
        }
    }
    return true;
}

} // namespace metallic::render::detail
