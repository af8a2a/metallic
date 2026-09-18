#pragma once

#include "Runtime/Scene/Scene.h"

namespace metallic::scene {

// Increment when generated attributes or simplification policy change. The disk
// payload layout is independent; old pages remain readable, but need re-cooking.
inline constexpr uint32_t kGeometryCookRevision = 1;

bool validateGeometryAttributes(const RenderPrimitive& primitive, std::string& reason);
void generateMissingTangents(RenderPrimitive& primitive);

} // namespace metallic::scene
