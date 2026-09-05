#include "Runtime/Debug/DebugProbe.h"

#include <bit>
#include <cmath>
#include <set>

namespace metallic::debug {

DebugResult<void> validateProbeSpecification(const DebugValue& specification)
{
    try {
        const auto& probes = specification.at("probes");
        if (!probes.is_array() || probes.empty() || probes.size() > 8) {
            return std::unexpected(DebugError{"InvalidArgument", "Expected 1..8 fixed probes"});
        }
        std::set<std::string> names;
        for (const auto& probe : probes) {
            const auto id = probe.at("id").get<std::string>();
            const auto name = probe.value("name", id);
            const auto operation = probe.at("operation").get<std::string>();
            if (name.empty() || name.size() > 256 || !names.insert(name).second) {
                return std::unexpected(DebugError{"InvalidArgument", "Probe names must be unique, nonempty and at most 256 bytes"});
            }
            if (operation != "count" && operation != "outOfBounds" && operation != "nonFinite" && operation != "minMax") {
                return std::unexpected(DebugError{"Unsupported", "Unknown fixed GPU probe operation"});
            }
            if (!debugUnsigned(probe.at("count"), UINT32_MAX)) { throw std::invalid_argument("Probe count must be positive"); }
            (void)debugUnsigned(probe.value("offset", DebugValue(0)), UINT32_MAX);
            (void)debugUnsigned(probe.value("component", DebugValue(0)), 127);
            if (probe.contains("field")) { (void)probe.at("field").get<std::string>(); }
            if (operation == "outOfBounds" && !probe.contains("upper")) { throw std::invalid_argument("outOfBounds requires exclusive upper bound"); }
            if (operation == "count") {
                const auto predicate = probe.value("predicate", "all");
                const std::set<std::string> allowed{"all", "eq", "ne", "lt", "le", "gt", "ge"};
                if (!allowed.contains(predicate)) { throw std::invalid_argument("Unsupported count predicate"); }
                if (predicate != "all" && !probe.contains("value")) { throw std::invalid_argument("Count predicate requires value"); }
            }
        }
        return {};
    } catch (const std::exception& error) { return std::unexpected(DebugError{"InvalidArgument", error.what()}); }
}

DebugTypeDesc probePartialLayout()
{
    return {"GpuProbePartial", 32, {{"finiteCount", "u32", 0}, {"matchedCount", "u32", 4},
        {"nanCount", "u32", 8}, {"infCount", "u32", 12}, {"minBits", "u32", 16},
        {"maxBits", "u32", 20}, {"firstIndex", "u32", 24}, {"firstBits", "u32", 28}}};
}

DebugResult<DebugValue> summarizeProbe(std::span<const uint8_t> bytes, const DebugValue& metadata)
{
    try {
        if (bytes.empty() || bytes.size() % 32 || bytes.size() > 256 * 32) { throw std::invalid_argument("Invalid GPU partial result size"); }
        const auto type = metadata.at("scalarType").get<std::string>();
        if (type != "u32" && type != "i32" && type != "f32") { throw std::invalid_argument("Unknown probe scalar type"); }
        auto value = [&](uint32_t bits) -> DebugValue {
            if (type == "f32") { return double(std::bit_cast<float>(bits)); }
            if (type == "i32") { return int64_t(std::bit_cast<int32_t>(bits)); }
            return uint64_t(bits);
        };
        uint64_t finite = 0, matched = 0, nan = 0, inf = 0;
        uint32_t first = UINT32_MAX, firstBits = 0;
        DebugValue minimum = nullptr, maximum = nullptr;
        for (size_t base = 0; base < bytes.size(); base += 32) {
            const auto word = [&](size_t offset) {
                const auto* p = bytes.data() + base + offset;
                return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
            };
            const auto count = word(0);
            if (count) {
                const auto lo = value(word(16)), hi = value(word(20));
                if (minimum.is_null() || lo < minimum) { minimum = lo; }
                if (maximum.is_null() || hi > maximum) { maximum = hi; }
            }
            finite += count; matched += word(4); nan += word(8); inf += word(12);
            if (word(24) < first) { first = word(24); firstBits = word(28); }
        }
        const uint64_t count = debugUnsigned(metadata.at("elementCount"));
        if (finite + nan + inf != count || matched > count ||
            (first != UINT32_MAX && first >= count) || ((matched == 0) != (first == UINT32_MAX))) {
            throw std::invalid_argument("GPU probe result failed coverage/count validation");
        }
        return DebugValue{{"operation", metadata.at("operation")}, {"source", metadata.at("source")},
            {"field", metadata.at("field")}, {"scalarType", type}, {"count", count}, {"finiteCount", finite},
            {"matchedCount", matched}, {"nanCount", nan}, {"infCount", inf}, {"min", minimum}, {"max", maximum},
            {"firstIndex", first == UINT32_MAX ? DebugValue(nullptr) : DebugValue(debugUnsigned(metadata.at("elementOffset")) + first)},
            {"firstValue", first == UINT32_MAX ? DebugValue(nullptr) : value(firstBits)},
            {"coverage", metadata.at("coverage")}, {"configuration", metadata.at("configuration")},
            {"interpretation", "Finite-only extrema; findings apply only to the specified storage interval, not a live worklist or the entire resource"}};
    } catch (const std::exception& error) { return std::unexpected(DebugError{"InvalidProbeResult", error.what()}); }
}

} // namespace metallic::debug
