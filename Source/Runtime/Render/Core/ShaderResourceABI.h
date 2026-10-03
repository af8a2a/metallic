#pragma once

#include <cstdint>
#include <type_traits>

namespace metallic::render {

// Shader wire types. These are indices, never CPU allocation handles or addresses.
// UINT32_MAX is invalid during migration; slot zero is not yet a universal null.
enum class ResourceViewKind : uint8_t { RawBuffer, SampledImage, StorageImage };

template<ResourceViewKind Kind>
struct GPUResourceHandle {
    uint32_t index = UINT32_MAX;
};

struct GPUSamplerHandle {
    uint32_t index = UINT32_MAX;
};

struct GPUBufferSpan {
    GPUResourceHandle<ResourceViewKind::RawBuffer> resource;
    uint32_t byteOffset = 0;
    uint32_t count = 0;
};

// An explicit physical address capability; never converted to descriptor + offset.
template<typename T>
struct PhysicalPtr {
    uint64_t address = 0;
};

static_assert(sizeof(GPUResourceHandle<ResourceViewKind::RawBuffer>) == 4);
static_assert(sizeof(GPUSamplerHandle) == 4);
static_assert(sizeof(GPUBufferSpan) == 12 && alignof(GPUBufferSpan) == 4);
static_assert(std::is_trivially_copyable_v<GPUBufferSpan>);
static_assert(sizeof(PhysicalPtr<void>) == 8);

} // namespace metallic::render
