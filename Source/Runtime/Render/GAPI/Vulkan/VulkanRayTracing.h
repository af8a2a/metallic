#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <volk.h>
#include <algorithm>
#include <limits>

namespace metallic::render::vulkan {

struct FixedRayTracingSBTLayout {
    uint64_t alignment = 0;
    uint64_t stride = 0;
    uint64_t allocationSize = 0;

    Result<uint64_t> offset(uint64_t address) const
    {
        if (!address || !alignment || address > std::numeric_limits<uint64_t>::max() - allocationSize) {
            return makeError(Error::InvalidArgument);
        }
        return (alignment - address % alignment) % alignment;
    }
};

inline Result<FixedRayTracingSBTLayout> fixedRayTracingSBTLayout(const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& properties)
{
    const auto powerOfTwo = [](uint32_t value) { return value && !(value & (value - 1u)); };
    if (!properties.shaderGroupHandleSize || !powerOfTwo(properties.shaderGroupHandleAlignment) ||
        !powerOfTwo(properties.shaderGroupBaseAlignment)) {
        return makeError(Error::Unsupported);
    }
    const uint64_t alignment = std::max(properties.shaderGroupHandleAlignment, properties.shaderGroupBaseAlignment);
    const uint64_t stride = (uint64_t(properties.shaderGroupHandleSize) + alignment - 1) & ~(alignment - 1);
    if (stride > properties.maxShaderGroupStride) { return makeError(Error::Unsupported); }
    // Padding also handles a buffer address that is not group-base aligned.
    return FixedRayTracingSBTLayout{alignment, stride, 3 * stride + alignment - 1};
}

} // namespace metallic::render::vulkan
