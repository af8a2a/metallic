#pragma once

#include <filesystem>
#include "Runtime/Render/Core/ColorSpace.h"

namespace metallic::scene {

struct EnvironmentSettings {
    bool enabled = true;
    std::filesystem::path path;
    float intensity = 1.0f;
    float rotationDegrees = 0.0f;
    bool visible = true;
    render::ColorSpaceDesc sourceColorSpace = render::kLinearRec709;
    // Untagged LDR environments retain the historical stb gamma 2.2 decode.
    // Set this for an explicitly linear Rec.709 source; other spaces imply it.
    bool sourceColorSpaceExplicit = false;

    bool hasExplicitSourceColorSpace() const
    {
        return sourceColorSpaceExplicit || sourceColorSpace != render::kLinearRec709;
    }

    bool operator==(const EnvironmentSettings&) const = default;
};

} // namespace metallic::scene
