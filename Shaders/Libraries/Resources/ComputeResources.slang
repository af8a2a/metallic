#pragma once

// Shared with ComputeProgram. Resource slots are application IDs, not Vulkan
// bindings. Each entry contains an absolute BindlessHandle::shaderIndex in x,
// or a 64-bit acceleration-structure address (Slang's RTAS handle convention).
struct ComputeResourcePush
{
    // Header prepended by the native bindless RHI.
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint2* resources;
    uint* constants;
};

[[vk::push_constant]] ComputeResourcePush gComputeResources;

#define METALLIC_RESOURCE(type, slot) \
    (*DescriptorHandle<type>(gComputeResources.resources[slot]))
#define METALLIC_RESOURCE_ARRAY(type, slot, index) \
    (*nonuniform(DescriptorHandle<type>(uint2(gComputeResources.resources[slot].x + index, 0))))
#define METALLIC_CONSTANTS(type) (*((type*)gComputeResources.constants))
