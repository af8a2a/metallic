#pragma once
#include "DebugTypes.h"
#include <optional>

namespace metallic::debug {
inline constexpr uint32_t kShaderTraceParserVersion = 1;
inline constexpr uint64_t kShaderTraceArtifactBudget = 2ull << 20;

DebugValue shaderTraceSite(DebugValue description);
DebugResult<void> validateShaderWatch(const DebugValue& request, const DebugValue& site, uint64_t generation);
// Parses raw evidence again, never trusts exported decoded records/outcome.
DebugResult<DebugValue> analyzeShaderTrace(const DebugValue& bundle);
DebugResult<DebugValue> decodeShaderTraceArtifact(std::span<const uint8_t> bytes, std::string_view sha256);

// Owner-thread state. IPC cancellation is observed by the owner; Vulkan handles never enter this class.
// Only one observation may be active. Tokens are never recycled, including after cancellation.
class ShaderTraceCore {
public:
    explicit ShaderTraceCore(std::string session, uint64_t firstToken = 1);
    DebugResult<DebugValue> begin(DebugValue request, DebugValue site, DebugValue identity);
    void recorded(DebugValue identity);
    void compiledVariant(DebugValue variant);
    void submitted(DebugValue queue, DebugValue submit);
    void ingest(DebugValue raw);
    void accountLoss(uint64_t dropped, uint64_t truncated);
    void stop(std::string reason);
    // A closed device/instance is the P1 fixture's collection boundary. Persistent adapters need their own proof.
    void completion(bool gpuComplete, bool backendClosed, bool readbackValid);
    DebugResult<DebugValue> seal();
    bool active() const { return active_.has_value(); }
private:
    std::string session_;
    uint64_t sessionToken_ = 0;
    uint64_t nextToken_ = 1;
    std::optional<DebugValue> active_;
};
} // namespace metallic::debug
