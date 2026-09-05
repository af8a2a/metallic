#pragma once

#include "Runtime/Debug/DebugTypes.h"

namespace metallic::debug {

DebugResult<void> validateProbeSpecification(const DebugValue& specification);
DebugTypeDesc probePartialLayout();
// Metadata carries the source layout/hash, field and coverage; the binary is a
// fixed array of group summaries, never a masquerading copy of the source.
DebugResult<DebugValue> summarizeProbe(std::span<const uint8_t> bytes, const DebugValue& metadata);

} // namespace metallic::debug
