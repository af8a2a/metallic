#pragma once

#include "Runtime/Scene/Scene.h"

namespace metallic::scene {

// Increment when generated attributes or simplification policy change. The disk
// payload layout is independent. Runtime compatibility may allow older cooks
// when the changed attribute rules do not affect their position-only geometry.
inline constexpr uint32_t kGeometryCookRevision = 1;

bool validateGeometryAttributes(const RenderPrimitive& primitive, std::string& reason);
void generateMissingTangents(RenderPrimitive& primitive);

} // namespace metallic::scene
