#pragma once
#include <span>
#include <string>
#include <string_view>
#include <cstdint>
namespace metallic::debug {
std::string debugSha256(std::span<const uint8_t> bytes);
inline std::string debugSha256(std::string_view text)
{
    return debugSha256(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}
} // namespace metallic::debug
