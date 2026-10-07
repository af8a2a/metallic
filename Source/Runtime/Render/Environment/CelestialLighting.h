#pragma once

#include "Runtime/Environment/WorldEnvironment.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::scene { class Scene; }

namespace metallic::render {

class RenderWorld;

inline constexpr uint32_t kGPUCelestialLightEnabled = 1u;
inline constexpr uint32_t kGPUCelestialLightCastsShadow = 2u;
inline constexpr uint32_t kCelestialLightSourceTag = 0x80000000u;

// Shared C++/Slang ABI. Fixed Sun/Moon slots, independent of local light lists.
struct alignas(16) GPUCelestialLight {
    float direction[3]{}; // Direction of emitted light, in world space.
    float angularRadius = 0.0f; // Radians.
    float irradiance[3]{}; // Scene working color basis, consistent with lux lighting.
    uint32_t flags = 0;
    float diskRadiance[3]{};
    float shadowImportance = 0.0f;
};
static_assert(sizeof(GPUCelestialLight) == 48);
static_assert(offsetof(GPUCelestialLight, irradiance) == 16);
static_assert(offsetof(GPUCelestialLight, diskRadiance) == 32);
static_assert(std::is_trivially_copyable_v<GPUCelestialLight>);

using GPUCelestialLightRecords = std::array<GPUCelestialLight, environment::kCelestialLightCount>;

struct CelestialShadowPlan {
    uint32_t dominantIndex = 0xffffffffu;
    uint32_t activeMask = 0;
    std::array<uint32_t, environment::kCelestialLightCount> sampleCounts{};
    std::array<float, environment::kCelestialLightCount> importance{};
};

CelestialShadowPlan buildCelestialShadowPlan(const environment::EnvironmentSnapshot& snapshot,
    const std::array<double, 3>& observerWorldMetres);

GPUCelestialLightRecords buildCelestialLightRecords(const environment::EnvironmentSnapshot& snapshot);

// An override scene owns its Sun/Moon. Same-scene world edits and an explicit
// unbound world override take precedence; an unbound default inherits the scene.
environment::EnvironmentSnapshot resolveWorldEnvironment(const scene::Scene* actualScene, const RenderWorld* world);

} // namespace metallic::render
