#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"

namespace metallic::render {

// Wire-compatible with Core.ComputeResourceParameters. The per-pass ABI ID
// identifies the resource and constant types; it is not sent to the shader.
struct NamedComputeParameters {
    GPUBufferSpan resources;
    GPUBufferSpan constants;
};
static_assert(sizeof(NamedComputeParameters) == 24);

template<typename Resources, typename Constants>
Result<EncodedParameters> encodeNamedParameters(ParameterWriter& writer,
    const Resources& resources, const Constants& constants, uint64_t abiId)
{
    static_assert(std::is_standard_layout_v<Resources> && std::is_trivially_copyable_v<Resources>);
    static_assert(std::is_standard_layout_v<Constants> && std::is_trivially_copyable_v<Constants>);
    static_assert(sizeof(Resources) % 4 == 0 && sizeof(Constants) % 4 == 0);
    NamedComputeParameters root;
    root.resources = writer.dataSpan(&resources, sizeof(resources), sizeof(Resources), alignof(Resources));
    root.resources.count = sizeof(Resources) / 4;
    root.constants = writer.dataSpan(&constants, sizeof(constants), 4, alignof(Constants));
    return writer.encode(root, abiId);
}

} // namespace metallic::render
