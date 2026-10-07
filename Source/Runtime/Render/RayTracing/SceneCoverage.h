#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Scene/Scene.h"

#include <memory>
#include <vector>

namespace metallic::render {

struct ScenePrimitiveCoverage {
    // -2: no instances; -1: divergent materials cannot share static coverage.
    int32_t materialIndex = -2;
    bool usesAlpha = false;
};

// An immutable snapshot of resolved coverage inputs. Geometry UVs are canonical
// texture-space triangles; all primitives using a material share its alpha copy.
// The backend consumes these inputs while preparing its opaque build plan.
class SceneCoverageInput {
public:
    SceneCoverageInput(const scene::RenderMaterial& material,
        std::shared_ptr<const scene::RenderImage::Mip> image,
        const scene::RenderPrimitive& primitive);
    SceneCoverageInput(const SceneCoverageInput&) = delete;
    SceneCoverageInput& operator=(const SceneCoverageInput&) = delete;
    const RayTracingCoverageDesc& desc() const { return desc_; }
    bool valid() const { return !triangles_.empty(); }

private:
    std::shared_ptr<const scene::RenderImage::Mip> image_;
    std::vector<RayTracingCoverageTriangle> triangles_;
    RayTracingCoverageDesc desc_;
};

std::vector<ScenePrimitiveCoverage> scenePrimitiveCoverage(const scene::Scene& scene);
// CPU alpha copies and canonical UVs share a 256 MiB snapshot budget. Inputs
// that cannot be resolved within it retain ordinary live coverage evaluation.
std::vector<std::unique_ptr<const SceneCoverageInput>> makeSceneCoverageInputs(
    const scene::Scene& scene, std::span<const ScenePrimitiveCoverage> coverage);

} // namespace metallic::render
