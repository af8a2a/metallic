#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace metallic::render {

enum class ShaderSpecializationType : uint32_t { Int32, Bool32 };

// A scalar payload is exactly four bytes, independently of this CPU record's
// layout. Bool32 uses 0/1 VkBool32 bits, never sizeof(C++ bool).
struct ShaderSpecializationConstant {
    uint32_t id = 0;
    ShaderSpecializationType type = ShaderSpecializationType::Int32;
    uint32_t bits = 0;
    bool operator==(const ShaderSpecializationConstant&) const = default;
};

namespace detail {

inline std::vector<ShaderSpecializationConstant> canonicalShaderSpecializationConstants(
    std::span<const ShaderSpecializationConstant> values)
{
    // Bound serialized recipe input and Vulkan map offsets. This API deliberately
    // supports scalar32 values only; it is not an arbitrary specialization blob.
    if (values.size() > 1024) { throw std::invalid_argument("too many shader specialization constants"); }
    std::vector<ShaderSpecializationConstant> result(values.begin(), values.end());
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    for (size_t i = 0; i < result.size(); ++i) {
        const auto& value = result[i];
        if ((value.type != ShaderSpecializationType::Int32 && value.type != ShaderSpecializationType::Bool32) ||
            (value.type == ShaderSpecializationType::Bool32 && value.bits > 1) ||
            (i && result[i - 1].id == value.id)) {
            throw std::invalid_argument("invalid or duplicate scalar32 shader specialization constant");
        }
    }
    return result;
}

inline void appendSpecializationUint32(std::vector<uint8_t>& bytes, uint32_t value)
{
    for (uint32_t shift = 0; shift < 32; shift += 8) { bytes.push_back(uint8_t(value >> shift)); }
}

// Explicit little-endian encoding: no host struct padding, pointers or bool ABI.
inline std::vector<uint8_t> shaderSpecializationBytes(std::span<const ShaderSpecializationConstant> values)
{
    const auto canonical = canonicalShaderSpecializationConstants(values);
    std::vector<uint8_t> bytes;
    appendSpecializationUint32(bytes, 1); // scalar32 encoding version
    appendSpecializationUint32(bytes, static_cast<uint32_t>(canonical.size()));
    for (const auto& value : canonical) {
        appendSpecializationUint32(bytes, value.id);
        appendSpecializationUint32(bytes, static_cast<uint32_t>(value.type));
        appendSpecializationUint32(bytes, 4);
        appendSpecializationUint32(bytes, value.bits);
    }
    return bytes;
}

inline std::vector<uint8_t> specializationCompatibilityKey(std::span<const uint8_t> driverKey,
    std::span<const ShaderSpecializationConstant> values)
{
    if (values.empty()) { return {driverKey.begin(), driverKey.end()}; }
    const auto encoded = shaderSpecializationBytes(values);
    // NativeBinaryCacheFile's existing bounded compatibility-key contract.
    if (driverKey.size() + encoded.size() + 16 > 256) {
        throw std::invalid_argument("specialization native compatibility key exceeds 256 bytes");
    }
    std::vector<uint8_t> key;
    appendSpecializationUint32(key, 0x4350534du); // MSPC
    appendSpecializationUint32(key, 1);
    appendSpecializationUint32(key, static_cast<uint32_t>(driverKey.size()));
    key.insert(key.end(), driverKey.begin(), driverKey.end());
    appendSpecializationUint32(key, static_cast<uint32_t>(encoded.size()));
    key.insert(key.end(), encoded.begin(), encoded.end());
    return key;
}

// Validate overrides against actual module declarations, not OpName/source
// attributes. No layout/resource parameters are derived from specialization.
inline bool validateShaderSpecializationModule(std::span<const uint32_t> words,
    std::span<const ShaderSpecializationConstant> values, std::string& error)
{
    try {
        const auto canonical = canonicalShaderSpecializationConstants(values);
        if (canonical.empty()) { return true; } // preserve all old empty-stage behavior
        if (words.size() < 5 || words[0] != 0x07230203u) { throw std::invalid_argument("invalid specialization SPIR-V header"); }
        struct Integer { uint32_t width, signedness; };
        std::unordered_map<uint32_t, Integer> integers;
        std::unordered_map<uint32_t, bool> booleans;
        std::unordered_map<uint32_t, uint32_t> resultTypes, ids;
        for (size_t offset = 5; offset < words.size();) {
            const uint32_t count = words[offset] >> 16, op = words[offset] & 0xffffu;
            if (!count || count > words.size() - offset) { throw std::invalid_argument("malformed specialization SPIR-V instruction"); }
            if (op == 20 && count == 2) { booleans.emplace(words[offset + 1], true); } // OpTypeBool
            if (op == 21 && count == 4) { integers.emplace(words[offset + 1], Integer{words[offset + 2], words[offset + 3]}); }
            if ((op == 48 || op == 49) && count == 3) { resultTypes.emplace(words[offset + 2], words[offset + 1]); }
            if (op == 50 && count == 4) { resultTypes.emplace(words[offset + 2], words[offset + 1]); }
            if (op == 71 && count >= 3 && words[offset + 2] == 1) { // OpDecorate SpecId
                if (count != 4 || !ids.emplace(words[offset + 3], words[offset + 1]).second) {
                    throw std::invalid_argument("unsupported repeated or malformed SPIR-V SpecId");
                }
            }
            offset += count;
        }
        for (const auto& value : canonical) {
            const auto id = ids.find(value.id);
            if (id == ids.end()) { throw std::invalid_argument("specialization SpecId is absent from module"); }
            const auto type = resultTypes.find(id->second);
            if (type == resultTypes.end()) { throw std::invalid_argument("unsupported specialization declaration"); }
            if (value.type == ShaderSpecializationType::Bool32) {
                if (!booleans.contains(type->second)) { throw std::invalid_argument("specialization Bool32 type mismatch"); }
            } else {
                const auto integer = integers.find(type->second);
                if (integer == integers.end() || integer->second.width != 32 || integer->second.signedness != 1) {
                    throw std::invalid_argument("specialization Int32 type mismatch");
                }
            }
        }
        error.clear();
        return true;
    } catch (const std::invalid_argument& failure) {
        error = failure.what();
        return false;
    }
}

} // namespace detail
} // namespace metallic::render
