#pragma once

#include <filesystem>

namespace metallic::scene {

struct EnvironmentSettings {
    bool enabled = true;
    std::filesystem::path path;
    float intensity = 1.0f;
    float rotationDegrees = 0.0f;
    bool visible = true;

    bool operator==(const EnvironmentSettings&) const = default;
};

} // namespace metallic::scene
