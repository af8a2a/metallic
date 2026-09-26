#pragma once

#include "Runtime/Render/ResourceRegistry.h"

namespace metallic::render {

inline constexpr uint64_t kMaterialBinningAbi = 0x4d42494e00000001ull;
struct MaterialBinningParams {
    ShaderSampledImage visibility;
    ShaderBuffer records;
    ShaderBuffer instances;
    ShaderBuffer materials;
    ShaderBuffer shadingMaterials;
    ShaderBuffer bins;
    ShaderBuffer tiles;
    ShaderBuffer arguments;
    ShaderBuffer streamRecords;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileCount = 0;
    uint32_t residentRecordCount = 0;
};
static_assert(sizeof(MaterialBinningParams) == 88);
static_assert(offsetof(MaterialBinningParams, width) == 72);

} // namespace metallic::render
