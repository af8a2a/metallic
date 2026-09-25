#pragma once
#include <cstdint>

namespace metallic::render::profiling {
// Opt-in ETW diagnostics (METALLIC_PACING_TRACE=1). Queue events carry no
// application frame ID: worker submissions must be correlated by ETW time/TID.
void pacingTrace(const char* phase, uint64_t frameId = UINT64_MAX, uint32_t value = 0);
} // namespace metallic::render::profiling
