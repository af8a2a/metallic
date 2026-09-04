#pragma once

#include <json.hpp>

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace metallic::debug {

// Values retain native integer/float types in memory. Only the wire format tags
// 64-bit integers and non-finite floats; JSON must never round or discard them.
using DebugValue = nlohmann::json;

struct DebugError {
    std::string code;
    std::string message;
};

template <typename T>
using DebugResult = std::expected<T, DebugError>;

struct DebugFieldDesc {
    std::string name;
    std::string type;
    uint32_t offset = 0;
    uint32_t count = 1;
    uint32_t bitOffset = 0;
    uint32_t bitWidth = 0;
    uint64_t scale = 1;
    DebugValue enumNames = DebugValue::object();
};

struct DebugTypeDesc {
    std::string name;
    uint32_t stride = 0;
    std::vector<DebugFieldDesc> fields;
    std::string layoutHash() const;
    DebugValue schema() const;
};

struct DebugEvidenceStamp {
    std::string session;
    std::string graph;
    uint64_t generation = 0;
    uint64_t execution = 0;
    uint32_t frameSlot = 0;
    uint32_t passId = 0;
    std::string pass;
    std::string checkpoint;
    uint64_t sample = 0;
    DebugValue provenance = DebugValue::object();
    DebugValue value() const;
};

struct DebugSnapshot {
    DebugEvidenceStamp evidence;
    DebugValue values = DebugValue::object();
};

// Provider methods run on their owner's thread, never on the IPC thread.
class IDebugProvider {
public:
    virtual ~IDebugProvider() = default;
    virtual std::string_view name() const = 0;
    virtual DebugValue schema() const = 0;
    virtual void publish(DebugValue& destination) const = 0;
};

struct DebugLimits {
    uint32_t snapshotCount = 120;
    uint64_t snapshotBytes = 16ull << 20;
    uint64_t capturePoolBytes = 128ull << 20;
    uint64_t jobBytes = 16ull << 20;
    uint64_t frameBytes = 16ull << 20;
    uint32_t queueCount = 256;
    uint32_t commandsPerFrame = 8;
};

DebugValue encodeLossless(const DebugValue& value);
DebugValue decodeLossless(const DebugValue& value);
uint64_t debugUnsigned(const DebugValue& value, uint64_t maximum = UINT64_MAX);
DebugResult<DebugValue> evaluate(std::string_view expression, const DebugValue& root,
    uint64_t operationBudget = 1000000);
DebugResult<DebugValue> decodeBuffer(std::span<const uint8_t> bytes, const DebugTypeDesc& type);
std::string hexEncode(std::span<const uint8_t> bytes);
DebugResult<std::vector<uint8_t>> hexDecode(std::string_view text);

} // namespace metallic::debug
