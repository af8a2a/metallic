#pragma once

#include "Runtime/Render/Environment/VolumetricFogParameters.h"

#include <span>
#include <vector>

namespace metallic::environment { struct LocalFogVolume; }

namespace metallic::render {

inline constexpr uint32_t kLocalFogTilePixels = 128;
inline constexpr uint32_t kLocalFogMaxInstances = 32;
using PackedLocalFogVolume = std::array<FogUInt4, 3>;
std::vector<PackedLocalFogVolume> packLocalFogVolumes(std::span<const environment::LocalFogVolume> volumes,
    const ViewCameraConstants& camera);

} // namespace metallic::render
