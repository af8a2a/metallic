#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamBLASSettings = ShaderDataSpan;
using StreamBLASActiveGroupBuffer = ShaderBuffer;
using StreamBLASActiveHeaderBuffer = ShaderBuffer;
using StreamBLASBlasBuildInfoBuffer = ShaderBuffer;
using StreamBLASBlasClusterReferenceBuffer = ShaderBuffer;
using StreamBLASBlasHeaderBuffer = ShaderBuffer;
using StreamBLASClasAddressBuffer = ShaderBuffer;
using StreamBLASClasPageTableBuffer = ShaderBuffer;
using StreamBLASDynamicBlasAddressBuffer = ShaderBuffer;
using StreamBLASInstanceBlasBuffer = ShaderBuffer;
using StreamBLASScratch = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamBLASSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamBLASActiveGroupBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>;
typealias StreamBLASActiveHeaderBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>;
typealias StreamBLASBlasBuildInfoBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBLASBuildInfo>>;
typealias StreamBLASBlasClusterReferenceBuffer = DescriptorHandle<RWStructuredBuffer<uint2>>;
typealias StreamBLASBlasHeaderBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader>>;
typealias StreamBLASClasAddressBuffer = DescriptorHandle<StructuredBuffer<uint2>>;
typealias StreamBLASClasPageTableBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry>>;
typealias StreamBLASDynamicBlasAddressBuffer = DescriptorHandle<RWStructuredBuffer<uint2>>;
typealias StreamBLASInstanceBlasBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>;
typealias StreamBLASScratch = DescriptorHandle<RWStructuredBuffer<uint4>>;
#endif
struct StreamBLASParameters
{
    StreamBLASSettings settings;
    StreamBLASActiveGroupBuffer activeGroupBuffer;
    StreamBLASActiveHeaderBuffer activeHeaderBuffer;
    StreamBLASBlasBuildInfoBuffer blasBuildInfoBuffer;
    StreamBLASBlasClusterReferenceBuffer blasClusterReferenceBuffer;
    StreamBLASBlasHeaderBuffer blasHeaderBuffer;
    StreamBLASClasAddressBuffer clasAddressBuffer;
    StreamBLASClasPageTableBuffer clasPageTableBuffer;
    StreamBLASDynamicBlasAddressBuffer dynamicBlasAddressBuffer;
    StreamBLASInstanceBlasBuffer instanceBlasBuffer;
    StreamBLASScratch scratch;
    uint32_t activeBuildPhase;
    uint32_t traversalPhase;
    uint32_t clasPublicationRevision;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamBLASABI = 0x535452424c410001ull;
static_assert(sizeof(StreamBLASParameters) == 112);
static_assert(offsetof(StreamBLASParameters, activeBuildPhase) == 96);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
