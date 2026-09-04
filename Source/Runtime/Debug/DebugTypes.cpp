#include "Runtime/Debug/DebugTypes.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::debug {
namespace {

uint32_t scalarBytes(std::string_view type)
{
    if (type == "u8") { return 1; }
    if (type == "u16" || type == "f16") { return 2; }
    if (type == "u32" || type == "i32" || type == "f32") { return 4; }
    if (type == "u64" || type == "i64" || type == "f64") { return 8; }
    return 0;
}

template <typename T>
T read(const uint8_t* bytes)
{
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

DebugValue scalar(const uint8_t* bytes, std::string_view type)
{
    if (type == "u8") { return *bytes; }
    if (type == "u16") { return read<uint16_t>(bytes); }
    if (type == "u32") { return read<uint32_t>(bytes); }
    if (type == "i32") { return read<int32_t>(bytes); }
    if (type == "u64") { return read<uint64_t>(bytes); }
    if (type == "i64") { return read<int64_t>(bytes); }
    if (type == "f32") { return read<float>(bytes); }
    if (type == "f64") { return read<double>(bytes); }
    const uint16_t h = read<uint16_t>(bytes);
    const int exponent = (h >> 10) & 31;
    const int mantissa = h & 1023;
    const double magnitude = exponent == 31
        ? (mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity())
        : std::ldexp(exponent ? 1.0 + mantissa / 1024.0 : mantissa / 1024.0,
            exponent ? exponent - 15 : -14);
    return (h & 0x8000) ? -magnitude : magnitude;
}

} // namespace

DebugValue encodeLossless(const DebugValue& value)
{
    if (value.is_number_unsigned()) {
        return DebugValue{{"$type", "u64"}, {"value", std::to_string(value.get<uint64_t>())}};
    }
    if (value.is_number_integer()) {
        return DebugValue{{"$type", "i64"}, {"value", std::to_string(value.get<int64_t>())}};
    }
    if (value.is_number_float() && !std::isfinite(value.get<double>())) {
        const double f = value.get<double>();
        return DebugValue{{"$type", "f64"}, {"value", std::isnan(f) ? "NaN" : f < 0 ? "-Inf" : "+Inf"}};
    }
    DebugValue result = value;
    if (result.is_structured()) {
        for (auto& item : result) { item = encodeLossless(item); }
    }
    return result;
}

DebugValue decodeLossless(const DebugValue& value)
{
    if (value.is_object() && value.size() == 2 && value.contains("$type") && value.contains("value")) {
        const std::string type = value.at("$type").get<std::string>();
        const std::string data = value.at("value").get<std::string>();
        size_t used = 0;
        if (type == "u64") {
            if (data.empty() || data[0] == '-') { throw std::invalid_argument("Invalid u64"); }
            const auto n = std::stoull(data, &used);
            if (used != data.size()) { throw std::invalid_argument("Invalid u64"); }
            return uint64_t(n);
        }
        if (type == "i64") {
            const auto n = std::stoll(data, &used);
            if (used != data.size()) { throw std::invalid_argument("Invalid i64"); }
            return int64_t(n);
        }
        if (type == "f64") {
            if (data == "NaN") { return std::numeric_limits<double>::quiet_NaN(); }
            if (data == "+Inf") { return std::numeric_limits<double>::infinity(); }
            if (data == "-Inf") { return -std::numeric_limits<double>::infinity(); }
            throw std::invalid_argument("Invalid non-finite f64");
        }
    }
    DebugValue result = value;
    if (result.is_structured()) {
        for (auto& item : result) { item = decodeLossless(item); }
    }
    return result;
}

uint64_t debugUnsigned(const DebugValue& value, uint64_t maximum)
{
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0)) {
        throw std::invalid_argument("Expected a nonnegative integer");
    }
    const uint64_t result = value.get<uint64_t>();
    if (result > maximum) { throw std::invalid_argument("Integer exceeds field range"); }
    return result;
}

std::string DebugTypeDesc::layoutHash() const
{
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&](std::string_view text) {
        for (unsigned char c : text) { hash = (hash ^ c) * 1099511628211ull; }
    };
    append(name + ":" + std::to_string(stride));
    for (const auto& field : fields) {
        append("|" + field.name + ":" + field.type + ":" + std::to_string(field.offset) + ":" + std::to_string(field.count));
        append(":" + std::to_string(field.bitOffset) + ":" + std::to_string(field.bitWidth) + ":" + std::to_string(field.scale) + ":" + field.enumNames.dump());
    }
    return std::to_string(hash);
}

DebugValue DebugTypeDesc::schema() const
{
    DebugValue result{{"name", name}, {"stride", stride}, {"layoutHash", layoutHash()}, {"fields", DebugValue::array()}};
    for (const auto& field : fields) {
        result["fields"].push_back({{"name", field.name}, {"type", field.type}, {"offset", field.offset}, {"count", field.count},
            {"bitOffset", field.bitOffset}, {"bitWidth", field.bitWidth}, {"scale", field.scale}, {"enumNames", field.enumNames}});
    }
    return result;
}

DebugValue DebugEvidenceStamp::value() const
{
    return {{"session", session}, {"graph", graph}, {"generation", generation},
        {"execution", execution}, {"frameSlot", frameSlot}, {"passId", passId},
        {"pass", pass}, {"checkpoint", checkpoint}, {"sample", sample}, {"provenance", provenance}};
}

DebugResult<DebugValue> decodeBuffer(std::span<const uint8_t> bytes, const DebugTypeDesc& type)
{
    if (std::endian::native != std::endian::little || !type.stride || bytes.size() % type.stride != 0) {
        return std::unexpected(DebugError{"LayoutMismatch", "Expected little-endian whole elements"});
    }
    if (type.fields.empty() || type.fields.size() > 128) {
        return std::unexpected(DebugError{"LayoutMismatch", "Expected 1..128 layout fields"});
    }
    uint64_t scalarCount = 0;
    for (const auto& field : type.fields) {
        const uint64_t size = scalarBytes(field.type);
        if (!size || !field.count || field.offset > type.stride || size * field.count > type.stride - field.offset) {
            return std::unexpected(DebugError{"LayoutMismatch", "Field exceeds the registered stride"});
        }
        if (field.bitWidth && (field.type != "u32" || field.bitOffset >= 32 || field.bitWidth > 32 - field.bitOffset ||
            field.scale > UINT32_MAX / ((uint64_t(1) << field.bitWidth) - 1))) {
            return std::unexpected(DebugError{"LayoutMismatch", "Invalid packed field"});
        }
        scalarCount += field.count;
    }
    if (scalarCount > 1048576 || bytes.size() / type.stride > 1048576 / scalarCount) {
        return std::unexpected(DebugError{"BudgetExceeded", "Decoding exceeds one million scalar fields; select a smaller range"});
    }
    DebugValue result = DebugValue::array();
    for (size_t offset = 0; offset < bytes.size(); offset += type.stride) {
        DebugValue element = DebugValue::object();
        for (const auto& field : type.fields) {
            DebugValue value = DebugValue::array();
            for (uint32_t i = 0; i < field.count; ++i) {
                auto decoded = scalar(bytes.data() + offset + field.offset + i * scalarBytes(field.type), field.type);
                if (field.bitWidth) {
                    decoded = ((decoded.get<uint64_t>() >> field.bitOffset) & ((uint64_t(1) << field.bitWidth) - 1)) * field.scale;
                }
                if (!field.enumNames.empty()) {
                    const std::string key = std::to_string(decoded.get<uint64_t>());
                    decoded = DebugValue{{"value", decoded}, {"name", field.enumNames.value(key, "Unknown")}};
                }
                value.push_back(std::move(decoded));
            }
            element[field.name] = field.count == 1 ? value[0] : std::move(value);
        }
        if (type.fields.size() == 1 && type.fields.front().name == "value") { element = element["value"]; }
        result.push_back(std::move(element));
    }
    return result;
}

std::string hexEncode(std::span<const uint8_t> bytes)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string text(bytes.size() * 2, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        text[i * 2] = kHex[bytes[i] >> 4];
        text[i * 2 + 1] = kHex[bytes[i] & 15];
    }
    return text;
}

DebugResult<std::vector<uint8_t>> hexDecode(std::string_view text)
{
    if (text.size() % 2) { return std::unexpected(DebugError{"InvalidArgument", "Odd hexadecimal length"}); }
    std::vector<uint8_t> bytes(text.size() / 2);
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') { return c - '0'; }
        if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
        return -1;
    };
    for (size_t i = 0; i < bytes.size(); ++i) {
        const int a = nibble(text[i * 2]), b = nibble(text[i * 2 + 1]);
        if (a < 0 || b < 0) { return std::unexpected(DebugError{"InvalidArgument", "Invalid hexadecimal data"}); }
        bytes[i] = static_cast<uint8_t>((a << 4) | b);
    }
    return bytes;
}

} // namespace metallic::debug
