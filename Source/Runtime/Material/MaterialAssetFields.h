#pragma once

#include "Runtime/Scene/Scene.h"
#include <array>

namespace metallic::material::detail {

using scene::RenderMaterial;
struct ScalarField
{
    const char* name;
    float RenderMaterial::* member;
    double minimum;
    double maximum;
};
inline constexpr std::array kScalars{
    ScalarField{"metalness", &RenderMaterial::metallicFactor, 0, 1},
    ScalarField{"roughness", &RenderMaterial::roughnessFactor, 0, 1},
    ScalarField{"alphaCutoff", &RenderMaterial::alphaCutoff, 0, 1},
    ScalarField{"normalScale", &RenderMaterial::normalTextureScale, -3.4028234663852886e38, 3.4028234663852886e38},
    ScalarField{"displacementMagnitude", &RenderMaterial::displacementMagnitude, -3.4028234663852886e38, 3.4028234663852886e38},
    ScalarField{"displacementCenter", &RenderMaterial::displacementCenter, 0, 1},
    ScalarField{"occlusionStrength", &RenderMaterial::occlusionTextureStrength, 0, 1},
    ScalarField{"specularWeight", &RenderMaterial::specularFactor, 0, 1},
    ScalarField{"transmission", &RenderMaterial::transmissionFactor, 0, 1},
    ScalarField{"ior", &RenderMaterial::ior, 1, 3.4028234663852886e38},
    ScalarField{"thickness", &RenderMaterial::thicknessFactor, 0, 3.4028234663852886e38},
    ScalarField{"attenuationDistance", &RenderMaterial::attenuationDistance, 0, 3.4028234663852886e38},
    ScalarField{"diffuseTransmission", &RenderMaterial::diffuseTransmissionFactor, 0, 1},
};
struct ColorField
{
    const char* name;
    float3 RenderMaterial::* member;
    double maximum;
};
inline constexpr std::array kColors{
    ColorField{"emission", &RenderMaterial::emissiveFactor, 3.4028234663852886e38},
    ColorField{"specularColor", &RenderMaterial::specularColorFactor, 1},
    ColorField{"attenuationColor", &RenderMaterial::attenuationColor, 1},
    ColorField{"diffuseTransmissionColor", &RenderMaterial::diffuseTransmissionColor, 1},
};
inline constexpr std::array kTextures{
    std::pair{"baseColorTexture", &RenderMaterial::baseColorTexture},
    std::pair{"metallicRoughnessTexture", &RenderMaterial::metallicRoughnessTexture},
    std::pair{"normalTexture", &RenderMaterial::normalTexture},
    std::pair{"displacementTexture", &RenderMaterial::displacementTexture},
    std::pair{"occlusionTexture", &RenderMaterial::occlusionTexture},
    std::pair{"emissiveTexture", &RenderMaterial::emissiveTexture},
    std::pair{"transmissionTexture", &RenderMaterial::transmissionTexture},
    std::pair{"thicknessTexture", &RenderMaterial::thicknessTexture},
    std::pair{"diffuseTransmissionTexture", &RenderMaterial::diffuseTransmissionTexture},
    std::pair{"diffuseTransmissionColorTexture", &RenderMaterial::diffuseTransmissionColorTexture},
    std::pair{"specularTexture", &RenderMaterial::specularTexture},
    std::pair{"specularColorTexture", &RenderMaterial::specularColorTexture},
};

} // namespace metallic::material::detail
