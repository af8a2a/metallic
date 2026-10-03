#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"

namespace metallic::render {

inline constexpr uint64_t kMaterialBinningABI = 0x4d42494e00000004ull;
struct MaterialBinningParams {
    ShaderSampledImage visibility;
    GPUBufferSpan records;
    GPUBufferSpan instances;
    GPUBufferSpan materials;
    GPUBufferSpan shadingMaterials;
    GPUBufferSpan bins;
    GPUBufferSpan tiles;
    GPUBufferSpan arguments;
    GPUBufferSpan streamRecords;
    GPUBufferSpan streamGroups;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileCount = 0;
    uint32_t residentRecordCount = 0;
};
static_assert(sizeof(MaterialBinningParams) == 128);
static_assert(offsetof(MaterialBinningParams, width) == 112);

} // namespace metallic::render
