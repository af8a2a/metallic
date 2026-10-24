#include "Runtime/Render/Profiling/CPUThreadProfile.h"

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace metallic::render {

CPUThreadSample sampleCPUThread()
{
    CPUThreadSample result;
#if defined(_WIN32)
    result.supported = true;
    result.threadId = GetCurrentThreadId();
    PROCESSOR_NUMBER processor{};
    GetCurrentProcessorNumberEx(&processor);
    result.processor = (uint32_t(processor.Group) << 16u) | processor.Number;
    ULONG64 cycles = 0;
    result.cyclesAvailable = QueryThreadCycleTime(GetCurrentThread(), &cycles) != 0;
    if (result.cyclesAvailable) { result.cycles = cycles; }
    else { result.cyclesError = GetLastError(); }
    FILETIME creation{}, exit{}, kernel{}, user{};
    result.timesAvailable = GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) != 0;
    if (result.timesAvailable) {
        result.user100ns = (uint64_t(user.dwHighDateTime) << 32u) | user.dwLowDateTime;
        result.kernel100ns = (uint64_t(kernel.dwHighDateTime) << 32u) | kernel.dwLowDateTime;
    } else { result.timesError = GetLastError(); }
    static const uint64_t frequency = [] {
        LARGE_INTEGER value{};
        return QueryPerformanceFrequency(&value) ? uint64_t(value.QuadPart) : 0u;
    }();
    LARGE_INTEGER qpc{};
    if (QueryPerformanceCounter(&qpc)) { result.qpc = uint64_t(qpc.QuadPart); }
    result.qpcFrequency = frequency;
#endif
    return result;
}

CPUThreadProfile cpuThreadProfile(const CPUThreadSample& begin, const CPUThreadSample& end)
{
    CPUThreadProfile result;
    result.supported = begin.supported && end.supported;
    result.sameThread = result.supported && begin.threadId == end.threadId;
    result.beginThreadId = begin.threadId; result.endThreadId = end.threadId;
    result.beginProcessor = begin.processor; result.endProcessor = end.processor;
    result.beginCyclesError = begin.cyclesError; result.endCyclesError = end.cyclesError;
    result.beginTimesError = begin.timesError; result.endTimesError = end.timesError;
    result.beginQpc = begin.qpc; result.endQpc = end.qpc;
    result.qpcFrequency = begin.qpcFrequency == end.qpcFrequency ? begin.qpcFrequency : 0u;
    result.qpcAvailable = result.supported && result.qpcFrequency != 0u && begin.qpc != 0u && end.qpc >= begin.qpc;
    result.cyclesAvailable = result.sameThread && begin.cyclesAvailable && end.cyclesAvailable && end.cycles >= begin.cycles;
    result.timesAvailable = result.sameThread && begin.timesAvailable && end.timesAvailable &&
        end.user100ns >= begin.user100ns && end.kernel100ns >= begin.kernel100ns;
    if (result.cyclesAvailable) { result.cycles = end.cycles - begin.cycles; }
    if (result.timesAvailable) {
        result.user100ns = end.user100ns - begin.user100ns;
        result.kernel100ns = end.kernel100ns - begin.kernel100ns;
    }
    return result;
}

} // namespace metallic::render
