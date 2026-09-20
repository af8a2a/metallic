#pragma once

#include "Runtime/Scene/Scene.h"

namespace metallic::scene {

// Increment when generated attributes or simplification policy change. The disk
// payload layout is independent. Runtime compatibility may allow older cooks
// when the changed attribute rules do not affect their position-only geometry.
inline constexpr uint32_t kGeometryCookRevision = 2;

bool validateGeometryAttributes(const RenderPrimitive& primitive, std::string& reason);
// Repair finite zero-length authored normals from incident triangles. Preserve
// all valid authored values; malformed/non-finite attributes still fail validation.
uint32_t repairZeroGeometryNormals(RenderPrimitive& primitive);
void generateMissingTangents(RenderPrimitive& primitive);

} // namespace metallic::scene
