#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using RasterSettingsUInt = uint32_t;
using RasterVisibleClusterBuffer = ShaderBuffer;
using RasterInstanceVisibilityBuffer = ShaderBuffer;
using RasterHzbBuffer0 = ShaderBuffer;
using RasterHzbBuffer1 = ShaderBuffer;
using RasterGpuSceneInstanceBuffer = ShaderBuffer;
using RasterTessellationBuffer = ShaderDataSpan;
using RasterMaterialBuffer = ShaderBuffer;
using RasterMaterialTextureRemapBuffer = ShaderBuffer;
#define METALLIC_RASTER_DEFAULT(value) = value
#else
import ShaderCore;
using Metallic;
typealias RasterSettingsUInt = uint;
typealias RasterVisibleClusterBuffer = DescriptorHandle<RWStructuredBuffer<CompactStreamVisibleRecord>>;
typealias RasterInstanceVisibilityBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias RasterHzbBuffer0 = DescriptorHandle<StructuredBuffer<float>>;
typealias RasterHzbBuffer1 = DescriptorHandle<StructuredBuffer<float>>;
typealias RasterGpuSceneInstanceBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>;
typealias RasterTessellationBuffer = DataSpan<uint>;
typealias RasterMaterialBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>;
typealias RasterMaterialTextureRemapBuffer = DescriptorHandle<StructuredBuffer<DescriptorHandle<Texture2D<float4>>>>;
#define METALLIC_RASTER_DEFAULT(value)
#endif
struct StreamRasterResourceSettings
{
    RasterVisibleClusterBuffer visibleClusterBuffer;
    RasterInstanceVisibilityBuffer instanceVisibilityBuffer;
    RasterHzbBuffer0 hzbBuffer0;
    RasterHzbBuffer1 hzbBuffer1;
    RasterSettingsUInt visibleRecordBase METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt visibleRecordCapacity METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt hzbMipCount METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt hzbValid METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt cullingFlags METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt width METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt height METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt resourcePadding METALLIC_RASTER_DEFAULT(0);
    RasterGpuSceneInstanceBuffer gpuSceneInstanceBuffer;
    RasterTessellationBuffer tessellationBuffer;
    float displacementBound METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt classificationFlags METALLIC_RASTER_DEFAULT(0);
    RasterMaterialBuffer materialBuffer;
    RasterMaterialTextureRemapBuffer materialTextureRemapBuffer;
    RasterSettingsUInt materialTextureCount METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt materialPadding METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt resourceFlags METALLIC_RASTER_DEFAULT(0);
    RasterSettingsUInt flagsPadding METALLIC_RASTER_DEFAULT(0);
};
#undef METALLIC_RASTER_DEFAULT
#ifndef __cplusplus
StreamRasterResourceSettings emptyStreamRasterResourceSettings()
{
    return StreamRasterResourceSettings(
        RasterVisibleClusterBuffer(uint2(0xffffffffu)),
        RasterInstanceVisibilityBuffer(uint2(0xffffffffu)),
        RasterHzbBuffer0(uint2(0xffffffffu)),
        RasterHzbBuffer1(uint2(0xffffffffu)),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        RasterGpuSceneInstanceBuffer(uint2(0xffffffffu)),
        RasterTessellationBuffer(nullptr, 0u, 4u),
        0,
        0,
        RasterMaterialBuffer(uint2(0xffffffffu)),
        RasterMaterialTextureRemapBuffer(uint2(0xffffffffu)),
        0,
        0,
        0,
        0);
}
#endif
#ifdef __cplusplus
inline void updateStreamRasterResourceFlags(StreamRasterResourceSettings& settings)
{
    settings.resourceFlags = (settings.gpuSceneInstanceBuffer.value != UINT64_MAX ? 1u : 0u) |
        (settings.tessellationBuffer.address != 0 && settings.tessellationBuffer.count != 0 && settings.tessellationBuffer.stride == 4 ? 2u : 0u) |
        (settings.materialBuffer.value != UINT64_MAX ? 4u : 0u) |
        (settings.materialTextureRemapBuffer.value != UINT64_MAX ? 8u : 0u);
}
static_assert(sizeof(StreamRasterResourceSettings) == 128);
static_assert(offsetof(StreamRasterResourceSettings, gpuSceneInstanceBuffer) == 64);
static_assert(offsetof(StreamRasterResourceSettings, materialBuffer) == 96);
static_assert(offsetof(StreamRasterResourceSettings, resourceFlags) == 120);
} // namespace metallic::render
#endif
