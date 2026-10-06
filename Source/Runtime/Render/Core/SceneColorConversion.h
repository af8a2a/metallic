#pragma once
#include "Runtime/Scene/scene.h"
#include "Runtime/Render/Core/ColorSpace.h"

namespace metallic::render {
inline scene::RenderMaterial resolveWorkingMaterial(const scene::RenderMaterial& authored)
{
    auto resolved = authored;
    const auto base = color::fromLinearRec709({authored.baseColorFactor.x, authored.baseColorFactor.y, authored.baseColorFactor.z});
    resolved.baseColorFactor = float4(base[0], base[1], base[2], authored.baseColorFactor.w);
    for (auto member : {&scene::RenderMaterial::emissiveFactor, &scene::RenderMaterial::specularColorFactor,
            &scene::RenderMaterial::attenuationColor, &scene::RenderMaterial::diffuseTransmissionColor,
            &scene::RenderMaterial::rtxcrHairBaseColor, &scene::RenderMaterial::rtxcrHairDiffuseReflectionTint}) {
        const auto& value = authored.*member;
        const auto rgb = color::fromLinearRec709({value.x, value.y, value.z});
        resolved.*member = float3(rgb[0], rgb[1], rgb[2]);
    }
    return resolved;
}
} // namespace metallic::render
