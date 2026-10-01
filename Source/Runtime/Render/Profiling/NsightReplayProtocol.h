#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace metallic::render::profiling {

// Private, local UTF-16 file protocol between the editor and its pre-injection
// worker. Requests are published by rename, so readers never see partial data.
inline bool writeReplayString(std::ostream& stream, const std::wstring& value)
{
    if (value.size() > 32768) { return false; }
    const auto length = static_cast<uint32_t>(value.size());
    stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
    stream.write(reinterpret_cast<const char*>(value.data()), length * sizeof(wchar_t));
    return static_cast<bool>(stream);
}

inline bool readReplayString(std::istream& stream, std::wstring& value)
{
    uint32_t length = 0;
    if (!stream.read(reinterpret_cast<char*>(&length), sizeof(length)) || length > 32768) { return false; }
    value.resize(length);
    stream.read(reinterpret_cast<char*>(value.data()), length * sizeof(wchar_t));
    return static_cast<bool>(stream);
}

} // namespace metallic::render::profiling
