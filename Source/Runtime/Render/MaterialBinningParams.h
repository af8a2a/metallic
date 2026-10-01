#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"

namespace metallic::render {

inline constexpr uint64_t kMaterialBinningABI = 0x4d42494e00000003ull;
struct MaterialBinningParams {
    ShaderSampledImage visibility;
    ShaderDataSpan records;
    ShaderDataSpan instances;
    ShaderDataSpan materials;
    ShaderDataSpan shadingMaterials;
    ShaderDataSpan bins;
    ShaderDataSpan tiles;
    ShaderDataSpan arguments;
    ShaderDataSpan streamRecords;
    ShaderDataSpan streamGroups;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileCount = 0;
    uint32_t residentRecordCount = 0;
};
static_assert(sizeof(MaterialBinningParams) == 168);
static_assert(offsetof(MaterialBinningParams, width) == 152);

} // namespace metallic::render
