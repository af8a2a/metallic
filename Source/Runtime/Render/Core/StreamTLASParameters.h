#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamTLASSettings = ShaderDataSpan;
using StreamTLASInstances = ShaderDataSpan;
using StreamTLASBLASRecords = ShaderDataSpan;
using StreamTLASAddresses = ShaderDataSpan;
using StreamTLASOutput = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamTLASSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamTLASInstances = DataSpan<GPUDrivenStreamAssetInstance>;
typealias StreamTLASBLASRecords = DataSpan<GPUDrivenStreamAssetInstanceBLAS>;
typealias StreamTLASAddresses = DataSpan<uint2>;
typealias StreamTLASOutput = DataSpan<GPUDrivenStreamAssetTLASInstance>;
#endif
struct StreamTLASParameters
{
    StreamTLASSettings settings;
    StreamTLASInstances instances;
    StreamTLASBLASRecords blasRecords;
    StreamTLASAddresses fallbackAddresses;
    StreamTLASOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamTLASABI = 0x535452544c410001ull;
static_assert(sizeof(StreamTLASParameters) == 80);
static_assert(offsetof(StreamTLASParameters, output) == 64);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
