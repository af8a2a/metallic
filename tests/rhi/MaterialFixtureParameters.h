#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using MaterialFixtureVisibility = render::ShaderStorageImage;
using MaterialFixtureRecords = render::ShaderDataSpan;
using MaterialFixtureInstances = render::ShaderDataSpan;
using MaterialFixtureMaterials = render::ShaderDataSpan;
using MaterialFixtureShading = render::ShaderDataSpan;
#else
import ShaderCore;
import GPUDriven;
import Material;
using Metallic;
using Metallic.GPUDriven;
using Metallic.Material;
typealias MaterialFixtureVisibility = DescriptorHandle<RWTexture2D<uint>>;
typealias MaterialFixtureRecords = DataSpan<VisibleClusterRecord>;
typealias MaterialFixtureInstances = DataSpan<GPUDrivenPreviewInstance>;
typealias MaterialFixtureMaterials = DataSpan<GPUDrivenPreviewMaterial>;
typealias MaterialFixtureShading = DataSpan<PathTraceMaterial>;
#endif
struct MaterialFixtureParameters
{
    MaterialFixtureVisibility visibility;
    MaterialFixtureRecords records;
    MaterialFixtureInstances instances;
    MaterialFixtureMaterials materials;
    MaterialFixtureShading shading;
    uint32_t width, height, binCount, bin;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialFixtureABI = 0x4d41544649580001ull;
static_assert(sizeof(MaterialFixtureParameters) == 88);
} // namespace metallic::tests
#endif
