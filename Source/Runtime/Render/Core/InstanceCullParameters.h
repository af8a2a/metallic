#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using InstanceCullSettings = ShaderDataSpan;
using InstanceCullInstances = ShaderDataSpan;
using InstanceCullWords = ShaderDataSpan;
using InstanceCullCounter = ShaderBuffer;
using InstanceCullHZB = ShaderBuffer;
#else
import ShaderCore;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
namespace Metallic {
typealias InstanceCullSettings = DataSpan<GPUDrivenPreviewParams>;
typealias InstanceCullInstances = DataSpan<GPUDrivenPreviewInstance>;
typealias InstanceCullWords = DataSpan<uint>;
typealias InstanceCullCounter = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias InstanceCullHZB = DescriptorHandle<RWStructuredBuffer<float>>;
#endif
struct InstanceCullParameters
{
    InstanceCullSettings settings;
    InstanceCullInstances instances;
    InstanceCullWords visibility;
    InstanceCullWords visibleIds;
    InstanceCullWords streamOwners;
    InstanceCullCounter counter;
    InstanceCullHZB hzb;
    uint32_t phase;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kInstanceCullABI = 0x494e535443550001ull;
static_assert(sizeof(InstanceCullParameters) == 104);
static_assert(offsetof(InstanceCullParameters, counter) == 80);
static_assert(offsetof(InstanceCullParameters, phase) == 96);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
