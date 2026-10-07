#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render::detail {

inline constexpr uint64_t kFnvOffset = 14695981039346656037ull;

// Raw byte FNV-1a. Callers own framing (lengths, tags and field order).
inline uint64_t hashBytes(uint64_t hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < byteSize; ++index) {
        hash = (hash ^ bytes[index]) * 1099511628211ull;
    }
    return hash;
}

// Native object representation; persistent formats must choose fixed-width fields.
template <typename T>
uint64_t hashValue(uint64_t hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(T));
}

} // namespace metallic::render::detail
