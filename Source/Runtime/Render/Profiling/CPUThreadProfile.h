#pragma once

#include <cstdint>

namespace metallic::render {

// Raw diagnostic endpoints. Cycle counts are not a frequency or elapsed-time
// measurement; thread CPU times have platform accounting granularity.
struct CPUThreadSample {
    bool supported = false;
    bool cyclesAvailable = false;
    bool timesAvailable = false;
    uint32_t threadId = 0;
    uint32_t processor = UINT32_MAX; // Processor group in high 16 bits, number in low 16.
    uint32_t cyclesError = 0;
    uint32_t timesError = 0;
    uint64_t cycles = 0;
    uint64_t user100ns = 0;
    uint64_t kernel100ns = 0;
    uint64_t qpc = 0;
    uint64_t qpcFrequency = 0;
};

struct CPUThreadProfile {
    bool supported = false;
    bool sameThread = false;
    bool cyclesAvailable = false;
    bool timesAvailable = false;
    bool qpcAvailable = false;
    uint32_t beginThreadId = 0, endThreadId = 0;
    uint32_t beginProcessor = UINT32_MAX, endProcessor = UINT32_MAX;
    uint32_t beginCyclesError = 0, endCyclesError = 0;
    uint32_t beginTimesError = 0, endTimesError = 0;
    uint64_t cycles = 0, user100ns = 0, kernel100ns = 0;
    uint64_t beginQpc = 0, endQpc = 0, qpcFrequency = 0;
};

CPUThreadSample sampleCPUThread();
CPUThreadProfile cpuThreadProfile(const CPUThreadSample& begin, const CPUThreadSample& end);

} // namespace metallic::render
