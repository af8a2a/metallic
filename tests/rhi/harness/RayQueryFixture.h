#pragma once
#include "Fixtures.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace metallic::tests::bench {

// A single authored triangle; every hit is strictly inside it, away from shared edges.
inline constexpr std::array<float, 9> kRayVertices{0, 0, 2, 1, 0, 2, 0, 1, 2};
struct RayObservation {
    uint32_t hit, instance, primitive, front;
    float distance, u, v, padding;
};
static_assert(sizeof(RayObservation) == 32);
using RayObservations = std::array<RayObservation, 6>;

inline Json rayFixture()
{
    return {{"vertices", kRayVertices}, {"instanceID", 37}, {"instanceMask", 1},
        {"rays", {{0.2, 0.3, 0.0, 1.0, 10.0, 1}, {0.65, 0.15, 0.0, 1.0, 10.0, 1},
            {3.0, 0.2, 0.0, 1.0, 10.0, 1}, {0.2, 0.3, 0.0, 1.0, 10.0, 2},
            {0.2, 0.3, 0.0, 1.0, 1.0, 1}, {0.2, 0.3, 5.0, -1.0, 10.0, 1}}}};
}

inline Json rayOracle(const RayObservations& actual, float translationX = 0.0f)
{
    Json observations = Json::array();
    const auto rays = rayFixture().at("rays");
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& ray = rays[i];
        const float x = ray[0].get<float>() - translationX, y = ray[1].get<float>();
        const float distance = (2.0f - ray[2].get<float>()) / ray[3].get<float>();
        const bool hit = x > 0 && y > 0 && x + y < 1 && distance >= 0.001f &&
            distance <= ray[4].get<float>() && (ray[5].get<uint32_t>() & 1);
        const auto near = [](float x, float y) { return std::isfinite(x) && std::abs(x - y) <= 0.00001f; };
        // The triangle's signed ray-space area is positive along +Z (back face), negative along -Z.
        // https://docs.vulkan.org/spec/latest/chapters/raytraversal.html#ray-traversal-culling-face
        if (a.hit != uint32_t(hit) || a.instance != (hit ? 37u : UINT32_MAX) ||
            a.primitive != (hit ? 0u : UINT32_MAX) || a.front != uint32_t(hit && ray[3].get<float>() < 0) ||
            !near(a.distance, hit ? distance : 0) || !near(a.u, hit ? x : 0) || !near(a.v, hit ? y : 0)) {
            throw std::runtime_error("analytic ray mismatch at ray " + std::to_string(i) + ": " +
                Json::array({a.hit, a.instance, a.primitive, a.front, a.distance, a.u, a.v}).dump());
        }
        observations.push_back(Json::array({a.hit, a.instance, a.primitive, a.front, a.distance, a.u, a.v}));
    }
    return observations;
}

} // namespace metallic::tests::bench
