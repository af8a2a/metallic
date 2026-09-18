#pragma once

#include "Runtime/Scene/SceneLoad.h"
#include "json.hpp"
#include <filesystem>

namespace metallic::scene::detail {

// Canonical child-node projection shared by metadata import and offline cook.
// Reads only external, uncompressed instance accessor ranges, never geometry
// or images. Unsupported instance accessor storage is rejected explicitly.
bool expandGltfGpuInstances(nlohmann::json& root, const std::filesystem::path& directory,
    GltfInstanceExpansion& expansion, std::string& reason);

} // namespace metallic::scene::detail
