#include "Editor/EditorApplication.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace {

constexpr const char* kRtxdiSampleId = "rtxdi-sample";

void printUsage()
{
    spdlog::info(
        "MetallicRtxdiSample options:\n"
        "  --smoke-test                 Render eight ReSTIR/RELAX history frames and exit\n"
        "  --wait-for-graphics-debugger Wait before Vulkan initialization");
}

int runSmokeTest()
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        spdlog::error("RTXDI smoke test failed to initialize SDL: {}", SDL_GetError());
        return 1;
    }
    struct SdlShutdownGuard {
        ~SdlShutdownGuard() { SDL_Quit(); }
    } sdlShutdownGuard;

    metallic::render::RenderSampleLoadResult sample;
    std::string message;
    if (!metallic::render::loadBuiltInRenderSample(kRtxdiSampleId, sample, message)) {
        spdlog::error("RTXDI smoke test failed to load Sample: {}", message);
        return 1;
    }
    if (!sample.desc.environment.has_value()) {
        spdlog::error("RTXDI smoke test Sample does not define an environment");
        return 1;
    }

    metallic::render::RenderGraphPreviewRenderer preview;
    const metallic::render::RenderSampleEnvironmentDesc& sampleEnvironment =
        *sample.desc.environment;
    metallic::render::EnvironmentSettings environment{
        .enabled = sampleEnvironment.enabled,
        .path = sampleEnvironment.path,
        .intensity = sampleEnvironment.intensity,
        .rotationDegrees = sampleEnvironment.rotationDegrees,
        .visible = sampleEnvironment.visible,
    };
    if (!environment.path.empty() && environment.path.is_relative()) {
        environment.path = std::filesystem::path(PROJECT_SOURCE_DIR) / environment.path;
    }
    preview.setEnvironment(std::move(environment));
    metallic::render::Result result = preview.initialize(false, true);
    if (!result) {
        spdlog::error(
            "RTXDI smoke test failed to initialize preview renderer: {}",
            metallic::render::resultToString(result));
        return 1;
    }

    constexpr uint32_t kSmokeWidth = 256;
    constexpr uint32_t kSmokeHeight = 256;
    constexpr uint32_t kSmokeFrameCount = 8;
    for (uint32_t frame = 0; frame < kSmokeFrameCount; ++frame) {
        result = preview.render(sample.graph, kSmokeWidth, kSmokeHeight, sample.desc.previewOutput);
        if (!result) {
            spdlog::error(
                "RTXDI smoke frame {} failed: {}: {}",
                frame,
                metallic::render::resultToString(result),
                preview.lastLog());
            return 1;
        }
    }

    const size_t visiblePixelCount = std::count_if(
        preview.pixels().begin(),
        preview.pixels().end(),
        [](uint32_t pixel) { return (pixel & 0x00ffffffu) != 0u; });
    if (visiblePixelCount < 1024) {
        spdlog::error("RTXDI smoke test produced too few visible pixels: {}", visiblePixelCount);
        return 1;
    }
    spdlog::info(
        "RTXDI/RELAX smoke test rendered {} {}x{} frames with {} visible pixels",
        kSmokeFrameCount,
        kSmokeWidth,
        kSmokeHeight,
        visiblePixelCount);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool waitForGraphicsDebugger = false;
    for (int index = 1; index < argc; ++index) {
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
        spdlog::error("Unknown argument: {}", argument);
        printUsage();
        return 1;
    }

    if (smokeTest) {
        return runSmokeTest();
    }

    metallic::EditorApplication app;
    return app.run(false, waitForGraphicsDebugger, kRtxdiSampleId);
}
