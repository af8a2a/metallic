#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kTessellationMaxFactor = 8;

struct TessellationPattern {
    // Integer barycentrics sum to 65535, including the centre (21845 each).
    std::vector<std::array<uint32_t, 3>> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
};

// Independent edge rates, joined by concentric triangular rings. Each shared
// edge depends only on its own rate; neither the opposite vertex nor interior
// budget changes its samples. This is an original pattern generator, not UE's LUT.
inline TessellationPattern makeTessellationPattern(std::array<uint32_t, 3> factors)
{
    TessellationPattern pattern;
    for (auto& rate : factors) { rate = std::clamp(rate, 1u, kTessellationMaxFactor); }
    if (factors == std::array<uint32_t, 3>{1, 1, 1}) {
        pattern.vertices = {{{65535, 0, 0}}, {{0, 65535, 0}}, {{0, 0, 65535}}};
        pattern.triangles = {{{0, 1, 2}}};
        return pattern;
    }
    struct Ring {
        std::array<std::vector<uint32_t>, 3> edges;
    };
    const auto ring = [&](std::array<uint32_t, 3> rates, double scale) {
        Ring result;
        const uint32_t first = static_cast<uint32_t>(pattern.vertices.size());
        for (uint32_t edge = 0; edge < 3; ++edge) {
            for (uint32_t i = 0; i < rates[edge]; ++i) {
                std::array<double, 3> bary{(1.0 - scale) / 3.0, (1.0 - scale) / 3.0, (1.0 - scale) / 3.0};
                bary[edge] += scale * (1.0 - double(i) / rates[edge]);
                bary[(edge + 1) % 3] += scale * double(i) / rates[edge];
                // Round the smallest components first so quantization never
                // moves a boundary vertex off its edge.
                std::array<uint32_t, 3> q{};
                const auto largest = uint32_t(std::max_element(bary.begin(), bary.end()) - bary.begin());
                q[(largest + 1) % 3] = uint32_t(std::lround(bary[(largest + 1) % 3] * 65535.0));
                q[(largest + 2) % 3] = uint32_t(std::lround(bary[(largest + 2) % 3] * 65535.0));
                q[largest] = 65535 - q[(largest + 1) % 3] - q[(largest + 2) % 3];
                result.edges[edge].push_back(static_cast<uint32_t>(pattern.vertices.size()));
                pattern.vertices.push_back(q);
            }
        }
        for (uint32_t edge = 0; edge < 3; ++edge) {
            result.edges[edge].push_back(edge == 2 ? first : result.edges[edge + 1].front());
        }
        return result;
    };
    Ring outer = ring(factors, 1.0);
    const uint32_t maximum = *std::max_element(factors.begin(), factors.end());
    for (int count = int(maximum) - 2; count > 0; count -= 2) {
        Ring inner = ring({uint32_t(count), uint32_t(count), uint32_t(count)}, double(count) / maximum);
        for (uint32_t edge = 0; edge < 3; ++edge) {
            const auto& a = outer.edges[edge];
            const auto& b = inner.edges[edge];
            uint32_t i = 0, j = 0;
            while (i + 1 < a.size() || j + 1 < b.size()) {
                if (j + 1 == b.size() || (i + 1 < a.size() &&
                    (i + 1) * (b.size() - 1) <= (j + 1) * (a.size() - 1))) {
                    pattern.triangles.push_back({a[i], a[i + 1], b[j]});
                    ++i;
                } else {
                    pattern.triangles.push_back({a[i], b[j + 1], b[j]});
                    ++j;
                }
            }
        }
        outer = std::move(inner);
    }
    const uint32_t centre = static_cast<uint32_t>(pattern.vertices.size());
    pattern.vertices.push_back({21845, 21845, 21845});
    for (const auto& edge : outer.edges) {
        for (size_t i = 1; i < edge.size(); ++i) { pattern.triangles.push_back({edge[i - 1], edge[i], centre}); }
    }
    return pattern;
}

// 512 uint4 headers followed by packed barycentric and triangle words.
inline std::vector<uint32_t> buildTessellationPatterns()
{
    std::vector<uint32_t> words(512 * 4);
    for (uint32_t c = 1; c <= 8; ++c) {
        for (uint32_t b = 1; b <= 8; ++b) {
            for (uint32_t a = 1; a <= 8; ++a) {
                const auto pattern = makeTessellationPattern({a, b, c});
                const uint32_t header = ((a - 1) + 8 * (b - 1) + 64 * (c - 1)) * 4;
                words[header] = static_cast<uint32_t>(words.size());
                words[header + 1] = static_cast<uint32_t>(pattern.vertices.size());
                for (const auto& v : pattern.vertices) { words.push_back(v[1] | (v[2] << 16)); }
                words[header + 2] = static_cast<uint32_t>(words.size());
                words[header + 3] = static_cast<uint32_t>(pattern.triangles.size());
                for (const auto& t : pattern.triangles) { words.push_back(t[0] | (t[1] << 8) | (t[2] << 16)); }
            }
        }
    }
    return words;
}

} // namespace metallic::render
