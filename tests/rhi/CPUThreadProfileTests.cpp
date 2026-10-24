#include "RHITest.h"
#include "Editor/EditorProfiler.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

#include <algorithm>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

void requireThreadProfile(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

class CPUThreadProfileTest final : public RHITest {
public:
    CPUThreadProfileTest() { type = RHITestType::Command; name = "cpu_thread_profile_evidence"; }
    RHITestResult run(RHITestContext&) override
    {
        CPUThreadSample begin{.supported = true, .cyclesAvailable = true, .timesAvailable = true,
            .threadId = 7, .processor = 2, .cycles = 100, .user100ns = 20, .kernel100ns = 30,
            .qpc = 1000, .qpcFrequency = 10000};
        auto end = begin;
        end.processor = 0x10003; end.cycles = 150; end.user100ns = 21; end.kernel100ns = 33; end.qpc = 2000;
        const auto profile = cpuThreadProfile(begin, end);
        requireThreadProfile(profile.sameThread && profile.cyclesAvailable && profile.timesAvailable && profile.qpcAvailable &&
            profile.cycles == 50 && profile.user100ns == 1 && profile.kernel100ns == 3 &&
            profile.beginProcessor == 2 && profile.endProcessor == 0x10003,
            "Raw thread deltas or processor group endpoints changed");
        auto invalid = end;
        invalid.threadId = 8;
        auto unavailable = cpuThreadProfile(begin, invalid);
        requireThreadProfile(!unavailable.sameThread && !unavailable.cyclesAvailable && !unavailable.timesAvailable,
            "Different-thread endpoints produced valid deltas");
        invalid = end; invalid.cyclesAvailable = false; invalid.cyclesError = 5;
        unavailable = cpuThreadProfile(begin, invalid);
        requireThreadProfile(!unavailable.cyclesAvailable && unavailable.timesAvailable && unavailable.endCyclesError == 5,
            "Unavailable cycles were treated as zero or discarded API failure");
        invalid = end; invalid.cycles = 99; invalid.kernel100ns = 29;
        unavailable = cpuThreadProfile(begin, invalid);
        requireThreadProfile(!unavailable.cyclesAvailable && !unavailable.timesAvailable,
            "Counter reversal underflowed into a valid delta");
        requireThreadProfile(!cpuThreadProfile({}, {}).supported && !cpuThreadProfile({}, {}).cyclesAvailable,
            "Unsupported platform became valid zero data");
        const nlohmann::json unavailableJson = unavailable;
        requireThreadProfile(unavailableJson.at("cycles").is_null() && unavailableJson.at("user100ns").is_null(),
            "Unavailable native measurements serialized as valid zeros");

        CPUProfileRecorder recorder;
        {
            CPUProfileScope scope(&recorder, "Disabled native probe");
            scope.work("items", 137);
        }
        requireThreadProfile(!recorder.sections.front().cpuThreadProfile && recorder.sections.front().cpuWork.front().value == 137,
            "Disabled diagnostic collected native evidence or lost work count");
        CPUProfileScope absent(nullptr, "Absent recorder");
        absent.work("ignored", 1);
        recorder.reset(); recorder.threadDiagnostics = true;
        {
            CPUProfileScope scope(&recorder, "Thread evidence");
            scope.work("items", 137);
        }
        const auto native = recorder.sections.front().cpuThreadProfile;
        requireThreadProfile(native != nullptr, "Opt-in diagnostic produced no status record");
#if defined(_WIN32)
        requireThreadProfile(native->supported && native->sameThread && native->cyclesAvailable && native->timesAvailable &&
            native->qpcAvailable && native->beginThreadId != 0 && native->endQpc >= native->beginQpc,
            "Current-thread Windows diagnostic APIs failed");
#else
        requireThreadProfile(!native->supported && !native->cyclesAvailable && !native->timesAvailable,
            "Non-Windows native probe fabricated support");
#endif
        EditorProfiler editor;
        editor.beginCapture();
        RenderGraphExecutionStats stats;
        stats.graphGeneration = 1;
        stats.executionId = 4;
        stats.nodes.push_back({.id = 7, .name = "Pass", .cpuMilliseconds = 1});
        stats.nodes.front().sections = recorder.sections;
        stats.preparation = recorder.sections;
        {
            auto frame = editor.beginFrame();
            editor.addCpuProfile(recorder.sections);
            editor.addRenderGraphStats(stats);
        }
        editor.endCapture();
        const auto matching = std::count_if(editor.capturedFrames().front().nodes.begin(), editor.capturedFrames().front().nodes.end(),
            [&](const auto& node) { return node.cpuThreadProfile == native && node.cpuWork.size() == 1 && node.cpuWork[0].value == 137; });
        requireThreadProfile(matching == 3, "CPU, preparation or graph section lost captured native/work evidence");
        stats.nodes.front().gpuTimingAvailable = true;
        editor.updateRenderGraphGpuStats(stats);
        requireThreadProfile(editor.capturedFrames().front().nodes.back().cpuThreadProfile == native,
            "Delayed GPU result replaced CPU diagnostic attribution");
        return RHITestResult::pass("Raw thread deltas, failures, migration endpoints, opt-in scopes, work counters and capture propagation verified; cycles are not elapsed time");
    }
};
METALLIC_REGISTER_RHI_TEST(CPUThreadProfileTest);
} // namespace
} // namespace metallic::tests
