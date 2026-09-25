#include "Runtime/Render/Profiling/PacingTrace.h"
#include <cstdlib>
#include <string_view>
#if defined(_WIN32)
#include <Windows.h>
#include <TraceLoggingProvider.h>
#endif

namespace metallic::render::profiling {
namespace {
#if defined(_WIN32)
TRACELOGGING_DEFINE_PROVIDER(kPacingProvider, "Metallic.Pacing",
    (0x69d58154, 0x7152, 0x4112, 0xa8, 0x1b, 0x69, 0xf8, 0x59, 0x8b, 0xb3, 0x78));
struct PacingProvider {
    PacingProvider() { TraceLoggingRegister(kPacingProvider); }
    ~PacingProvider() { TraceLoggingUnregister(kPacingProvider); }
};
#endif
} // namespace

void pacingTrace(const char* phase, uint64_t frameId, uint32_t value)
{
#if defined(_WIN32)
    static const bool enabled = [] {
        const char* setting = std::getenv("METALLIC_PACING_TRACE");
        return setting && std::string_view(setting) == "1";
    }();
    if (!enabled) { return; }
    static PacingProvider provider;
    TraceLoggingWrite(kPacingProvider, "FramePhase", TraceLoggingString(phase, "Phase"),
        TraceLoggingUInt64(frameId, "FrameId"), TraceLoggingUInt32(value, "Value"));
#else
    (void)phase; (void)frameId; (void)value;
#endif
}
} // namespace metallic::render::profiling
