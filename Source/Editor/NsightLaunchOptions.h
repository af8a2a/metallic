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
        "  --nsight-gputrace               Alias for --nsight-mode gputrace\n"
        "  --nsight-capture                Alias for --nsight-mode capture";

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
