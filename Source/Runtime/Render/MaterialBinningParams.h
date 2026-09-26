#pragma once

#include "Runtime/Render/ResourceRegistry.h"

namespace metallic::render {

inline constexpr uint64_t kMaterialBinningAbi = 0x4d42494e00000002ull;
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
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileCount = 0;
    uint32_t residentRecordCount = 0;
};
static_assert(sizeof(MaterialBinningParams) == 152);
static_assert(offsetof(MaterialBinningParams, width) == 136);

} // namespace metallic::render
