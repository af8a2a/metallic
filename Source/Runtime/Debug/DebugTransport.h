#pragma once

#include "Runtime/Debug/DebugCore.h"

#include <thread>

namespace metallic::debug {

inline constexpr uint32_t kDebugProtocolMaxBytes = 1u << 20;

class DebugServer {
public:
    explicit DebugServer(DebugCore& core) : core_(core) {}
    ~DebugServer();
    DebugServer(const DebugServer&) = delete;
    DebugServer& operator=(const DebugServer&) = delete;
    DebugResult<void> start();
    void stop();
private:
    DebugCore& core_;
    std::jthread thread_;
};

DebugResult<DebugValue> debugRequest(uint32_t pid, const DebugValue& request, uint32_t timeoutMs = 5000);
std::vector<uint32_t> debugProcesses();

} // namespace metallic::debug
