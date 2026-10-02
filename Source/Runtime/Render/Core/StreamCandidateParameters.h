#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamCandidateHeaders = ShaderDataSpan;
using StreamCandidateGroups = ShaderDataSpan;
using StreamCandidateArguments = ShaderDataSpan;
using StreamCandidateBins = ShaderBuffer;
using StreamCandidateVisibility = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamCandidateHeaders = DataSpan<GPUDrivenStreamAssetActiveHeader>;
typealias StreamCandidateGroups = DataSpan<GPUDrivenStreamAssetActiveGroup>;
typealias StreamCandidateArguments = DataSpan<uint>;
typealias StreamCandidateBins = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamCandidateVisibility = DescriptorHandle<StructuredBuffer<uint>>;
#endif
struct StreamCandidateParameters
{
    StreamCandidateHeaders headers;
    StreamCandidateGroups groups;
    StreamCandidateArguments arguments;
    StreamCandidateBins bins;
    StreamCandidateVisibility visibility;
    uint32_t stage;
    uint32_t late;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamCandidateABI = 0x53545243414e0001ull;
static_assert(sizeof(StreamCandidateParameters) == 72);
static_assert(offsetof(StreamCandidateParameters, stage) == 64);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
