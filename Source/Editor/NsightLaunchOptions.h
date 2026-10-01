#pragma once

#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"

#include <cstdio>
#include <string_view>

namespace metallic {

// Shared by the editor and sample launchers. Conflicting selections are errors,
// so changing argument order cannot silently select a different activity.
struct NsightLaunchOptions {
    render::profiling::NsightCaptureMode mode = render::profiling::NsightCaptureMode::Default;

    static constexpr const char* kUsage =
        "  --nsight-mode <gputrace|capture> Select and enable Nsight export\n"
        "  --nsight-gputrace               Enable GPU Trace with the default preset\n"
        "  --nsight-capture                Enable one-click capture + replay GPU Trace\n"
        "Capture collection (no metrics preset required):\n"
        "  Launch with --nsight-capture, then click 'Export View Capture + GPU Trace' in Profiler.\n"
        "  Saves .ngfx-capture and automatically profiles its replay into <capture>_Collected/.\n"
        "  Editor rendering pauses during collection. Results measure replayed GPU work.\n"
        "Direct live GPU Trace usage (--nsight-gputrace):\n"
        "  Default preset: single-pass performance overview, GPU clocks unchanged.\n"
        "  Metrics: Top-Level Triage (Throughput Metrics on Turing).\n"
        "  In Profiler, click 'Export Current View GPU Trace' to export the next full View frame.\n"
        "  Output: Captures/NsightGraphics/*.ngfx-gputrace under the project directory.\n"
        "  Advanced configuration (PowerShell, before launching):\n"
        "    $env:METALLIC_NSIGHT_GPU_TRACE_METRICS = 'E:/path/MyMetrics.json'\n"
        "    .\\LookDev.exe --nsight-gputrace\n"
        "  The JSON uses ngfx per-architecture settings; it overrides the default preset.\n"
        "  Copy a generated GpuTraceMetrics-<pid>.json as a starting point.\n"
        "  Run the installed ngfx.exe --help-all for supported architectures and metric sets.\n"
        "  To restore the default (PowerShell):\n"
        "    Remove-Item Env:METALLIC_NSIGHT_GPU_TRACE_METRICS -ErrorAction SilentlyContinue\n"
        "  Run this executable with --help to display these options without starting Vulkan.";

    // Returns 0 for unrelated arguments, 1 for consumed arguments, -1 on error.
    int consume(int argc, char** argv, int& index)
    {
        using render::profiling::NsightCaptureMode;
        const std::string_view argument(argv[index]);
        NsightCaptureMode selected = NsightCaptureMode::Default;
        if (argument == "--nsight-gputrace") {
            selected = NsightCaptureMode::GPUTrace;
        } else if (argument == "--nsight-capture") {
            selected = NsightCaptureMode::GraphicsCapture;
        } else if (argument == "--nsight-mode" || argument.starts_with("--nsight-mode=")) {
            std::string_view value;
            if (argument == "--nsight-mode") {
                if (index + 1 < argc) { value = argv[++index]; }
            } else {
                value = argument.substr(std::string_view("--nsight-mode=").size());
            }
            if (value == "gputrace") {
                selected = NsightCaptureMode::GPUTrace;
            } else if (value == "capture") {
                selected = NsightCaptureMode::GraphicsCapture;
            } else {
                std::fputs("--nsight-mode requires gputrace or capture\n", stderr);
                return -1;
            }
        } else {
            return 0;
        }
        if (mode != NsightCaptureMode::Default && mode != selected) {
            std::fputs("Conflicting Nsight modes: select either gputrace or capture\n", stderr);
            return -1;
        }
        mode = selected;
        return 1;
    }
};

} // namespace metallic
