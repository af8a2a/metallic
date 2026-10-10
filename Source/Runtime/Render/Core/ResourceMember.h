#pragma once

#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace metallic::render {

// CPU-only member identity. It carries the information needed by a dynamic
// manifest without a second input-ID-to-offset table. Never uploaded to shaders.
inline constexpr uint32_t kResourceMemberTag = 0x80000000u;

template<typename Owner, typename Member, bool Data = false>
consteval uint32_t resourceMember(uint32_t offset)
{
    static_assert(std::is_standard_layout_v<Owner> && sizeof(Owner) <= 65536);
    using T = std::remove_cvref_t<Member>;
    constexpr bool span = std::is_same_v<T, GPUBufferSpan>;
    static_assert(!Data || span);
    static_assert(span || std::is_same_v<T, uint64_t> || std::is_same_v<T, GPUSamplerHandle> ||
        std::is_same_v<T, GPUResourceHandle<ResourceViewKind::RawBuffer>> ||
        std::is_same_v<T, GPUResourceHandle<ResourceViewKind::SampledImage>> ||
        std::is_same_v<T, GPUResourceHandle<ResourceViewKind::StorageImage>>);
    if (offset > sizeof(Owner) || sizeof(T) > sizeof(Owner) - offset) {
        throw "Resource member must lie within its parameter structure";
    }
    constexpr auto kind = Data ? ComputeResourceBindingKind::DataBuffer :
        std::is_same_v<T, uint64_t> ? ComputeResourceBindingKind::AccelerationStructure :
        std::is_same_v<T, GPUSamplerHandle> ? ComputeResourceBindingKind::Sampler :
        std::is_same_v<T, GPUResourceHandle<ResourceViewKind::RawBuffer>> ? ComputeResourceBindingKind::StorageBuffer :
        std::is_same_v<T, GPUResourceHandle<ResourceViewKind::StorageImage>> ? ComputeResourceBindingKind::StorageImage :
        ComputeResourceBindingKind::SampledImage;
    constexpr auto format = Data ? ComputeResourceFieldFormat::DataSpan :
        span ? ComputeResourceFieldFormat::IndexSpan : ComputeResourceFieldFormat::Handle;
    return kResourceMemberTag | (uint32_t(format) << 24) | (uint32_t(kind) << 16) | offset;
}

template<typename T>
constexpr ComputeResourceLayout resourceParameterLayout()
{
    static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
    return {sizeof(T), {}};
}

} // namespace metallic::render

#define METALLIC_RESOURCE_MEMBER(Type, member) \
    ::metallic::render::resourceMember<Type, decltype(std::declval<Type>().member)>(offsetof(Type, member))
#define METALLIC_DATA_MEMBER(Type, member) \
    ::metallic::render::resourceMember<Type, decltype(std::declval<Type>().member), true>(offsetof(Type, member))
