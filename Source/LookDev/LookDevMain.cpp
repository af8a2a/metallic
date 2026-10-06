#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Editor/EditorApplication.h"
#include "Editor/NsightLaunchOptions.h"
#include "Runtime/Render/RenderSample.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace {

constexpr const char* kDefaultLookDevSampleId = "openpbr-lookdev";

void printUsage()
{
    std::puts(metallic::render::ShaderWarmupLaunchOptions::kUsage);
    std::puts(metallic::NsightLaunchOptions::kUsage);
    spdlog::info(
        "LookDev material playground options:\n"
        "  Default: MaterialX OpenPBR shaderball with fixed exposure and sRGB display\n"
        "  --sample <id>               Load a built-in sample (default: openpbr-lookdev)\n"
        "  --list-samples              List available sample IDs and exit\n"
        "  --scene <path>              Override the selected sample's scene\n"
        "  --render-path <mode>        comparison | pathtrace | deferred (defaults sample to lookdev-vbuffer)\n"
        "  --smoke-test                Render one frame and exit\n"
        "  --debug-control             Enable local Agent debug control\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

} // namespace

int main(int argc, char** argv)
{
    metallic::NsightLaunchOptions nsightOptions;
    metallic::render::ShaderWarmupLaunchOptions warmupOptions;
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    bool debugControl = false;
    bool listSamples = false;
    std::string sampleId = kDefaultLookDevSampleId;
    std::string scenePath;
    bool explicitSample = false;
    bool explicitRenderPath = false;
    auto renderPath = metallic::render::LookDevRenderPath::Comparison;
    for (int index = 1; index < argc; ++index) {
        if (warmupOptions.consume(argv[index])) { continue; }
        const int nsightArgument = nsightOptions.consume(argc, argv, index);
        if (nsightArgument < 0) { return 1; }
        if (nsightArgument > 0) { continue; }
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        }
        if (argument == "--smoke-test") {
            smokeTest = true;
            continue;
        }
        if (argument == "--wait-for-graphics-debugger") {
            waitForGraphicsDebugger = true;
            continue;
        }
        if (argument == "--debug-control") {
            debugControl = true;
            continue;
        }
        if (argument == "--list-samples") {
            listSamples = true;
            continue;
        }
        if (argument == "--render-path") {
            if (++index >= argc) { spdlog::error("--render-path requires a mode"); return 1; }
            const std::string_view mode(argv[index]);
            if (mode == "comparison") { renderPath = metallic::render::LookDevRenderPath::Comparison; }
            else if (mode == "pathtrace") { renderPath = metallic::render::LookDevRenderPath::PathTraceOnly; }
            else if (mode == "deferred") { renderPath = metallic::render::LookDevRenderPath::DeferredOnly; }
            else { spdlog::error("Unknown render path: {}", mode); return 1; }
            explicitRenderPath = true;
            continue;
        }
        if (argument == "--sample" || argument == "--scene") {
            if (index + 1 >= argc || std::string_view(argv[index + 1]).empty() ||
                std::string_view(argv[index + 1]).starts_with("--")) {
                spdlog::error("{} requires a value", argument);
                return 1;
            }
            (argument == "--sample" ? sampleId : scenePath) = argv[++index];
            explicitSample |= argument == "--sample";
            continue;
        }
        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    if (explicitRenderPath && !explicitSample) { sampleId = "lookdev-vbuffer"; }
    const auto samples = metallic::render::listBuiltInRenderSamples();
    if (listSamples) {
        for (const auto& sample : samples) {
            spdlog::info("{} [{}] {}", sample.id, sample.category, sample.name);
        }
        return 0;
    }
    if (std::none_of(samples.begin(), samples.end(), [&](const auto& sample) { return sample.id == sampleId; })) {
        spdlog::error("Unknown sample '{}'. Use --list-samples to see available IDs.", sampleId);
        return 1;
    }

    if (explicitRenderPath) {
        metallic::render::RenderSampleLoadResult sample;
        std::string message;
        if (!metallic::render::loadBuiltInRenderSample(sampleId, sample, message) ||
            !metallic::render::supportsLookDevRenderPaths(sample.graph)) {
            spdlog::error("--render-path requires a PT/Deferred comparison sample: {} ({})", sampleId, message);
            return 1;
        }
    }
    metallic::EditorApplication app;
    return app.run(
        smokeTest,
        waitForGraphicsDebugger,
        sampleId.c_str(),
        scenePath.empty() ? nullptr : scenePath.c_str(),
        nullptr,
        nsightOptions.mode,
        false,
        debugControl, false, warmupOptions.skip, renderPath);
}
