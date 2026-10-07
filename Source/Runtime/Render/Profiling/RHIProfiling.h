#pragma once
namespace metallic::render::profiling {
// Install the process-lifetime CPU marker/pacing adapter. Safe to call repeatedly.
void initializeRHIProfiling();
} // namespace metallic::render::profiling
