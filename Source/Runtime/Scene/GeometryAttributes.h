#pragma once

#include "Runtime/Scene/Scene.h"

namespace metallic::scene {

// Increment when generated attributes or simplification policy change. The disk
// payload layout is independent. Runtime compatibility may allow older cooks
// when an older policy still produces valid attributes and complete LOD cuts.
inline constexpr uint32_t kGeometryCookRevision = 3;

// Small normalized normal-direction differences remain weighted simplification
// error instead of hard seams. UVs and tangent handedness stay protected exactly.
// This is a cook policy, not an attribute quantization or welding tolerance.
inline constexpr float kMeshletLodNormalSeamTolerance = 1e-3f;

bool validateGeometryAttributes(const RenderPrimitive& primitive, std::string& reason);
// Repair finite zero-length authored normals from incident triangles. Preserve
// all valid authored values; malformed/non-finite attributes still fail validation.
uint32_t repairZeroGeometryNormals(RenderPrimitive& primitive);
void generateMissingTangents(RenderPrimitive& primitive);

} // namespace metallic::scene
