#include "rhi_test.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

namespace metallic::tests {

const char* toString(render::Result result)
{
    return render::resultToString(result);
}

const char* toString(RhiTestType type)
{
    switch (type) {
    case RhiTestType::Validation:
        return "validation";
    case RhiTestType::Resource:
        return "resource";
    case RhiTestType::Command:
        return "command";
    case RhiTestType::Rendering:
        return "rendering";
    }

    return "unknown";
}

} // namespace metallic::tests

namespace {

constexpr int kSkipExitCode = 77;

struct Options {
    bool help = false;
    bool list = false;
    bool enableValidation = true;
    std::string filter;
    std::filesystem::path outputDirectory = "rhi-test-output";
};

void printUsage(const char* executableName)
{
    std::cout << "Usage: " << executableName
              << " [--list] [--filter <text>] [--output-dir <path>] [--rhi-no-validation]\n";
}

bool parseArguments(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            return true;
        }
        if (argument == "--list") {
            options.list = true;
            continue;
        }
        if (argument == "--rhi-no-validation") {
            options.enableValidation = false;
            continue;
        }
        if (argument == "--rhi-validation") {
            options.enableValidation = true;
            continue;
        }
        if (argument == "--filter") {
            if (index + 1 >= argc) {
                std::cerr << "--filter requires a value\n";
                return false;
            }
            options.filter = argv[++index];
            continue;
        }
        if (argument.starts_with("--filter=")) {
            options.filter = std::string(argument.substr(9));
            continue;
        }
        if (argument == "--output-dir") {
            if (index + 1 >= argc) {
                std::cerr << "--output-dir requires a value\n";
                return false;
            }
            options.outputDirectory = argv[++index];
            continue;
        }
        if (argument.starts_with("--output-dir=")) {
            options.outputDirectory = std::string(argument.substr(13));
            continue;
        }

        std::cerr << "Unknown argument: " << argument << '\n';
        return false;
    }

    return true;
}

bool matchesFilter(const metallic::tests::RhiTest& test, const std::string& filter)
{
    if (filter.empty()) {
        return true;
    }

    const std::string_view name = test.name != nullptr ? test.name : "";
    const std::string_view type = metallic::tests::toString(test.type);
    const std::string_view filterView(filter);
    return name.find(filterView) != std::string_view::npos ||
        type.find(filterView) != std::string_view::npos;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace metallic;

    Options options;
    if (!parseArguments(argc, argv, options)) {
        return 1;
    }
    if (options.help) {
        printUsage(argv[0]);
        return 0;
    }

    std::vector<std::unique_ptr<tests::RhiTest>> allTests = tests::RhiTestRegistry::createAll();
    std::vector<tests::RhiTest*> selectedTests;
    selectedTests.reserve(allTests.size());
    for (const std::unique_ptr<tests::RhiTest>& test : allTests) {
        if (test != nullptr && matchesFilter(*test, options.filter)) {
            selectedTests.push_back(test.get());
        }
    }

    if (options.list) {
        for (const tests::RhiTest* test : selectedTests) {
            std::cout << tests::toString(test->type) << '\t' << test->name << '\n';
        }
        return 0;
    }

    if (selectedTests.empty()) {
        std::cerr << "No RHI tests matched";
        if (!options.filter.empty()) {
            std::cerr << " filter '" << options.filter << "'";
        }
        std::cerr << ".\n";
        return 1;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Skipping RHI tests: SDL_Init failed: " << SDL_GetError() << '\n';
        return kSkipExitCode;
    }

    std::unique_ptr<render::Device> device;
    render::Result result = render::createDevice(
        render::DeviceDesc{
            .applicationName = "Metallic RHI Tests",
            .enableValidation = options.enableValidation,
        },
        device);
    if (!result) {
        std::cerr << "Skipping RHI tests: createDevice returned " << tests::toString(result) << '\n';
        SDL_Quit();
        return render::hasError(result, render::Error::Unsupported) ? kSkipExitCode : 1;
    }

    render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
    if (graphicsQueue == nullptr) {
        std::cerr << "RHI test setup failed: no graphics queue is available.\n";
        device.reset();
        SDL_Quit();
        return 1;
    }

    tests::RhiTestContext context{
        .device = *device,
        .graphicsQueue = *graphicsQueue,
        .outputDirectory = options.outputDirectory,
        .enableValidation = options.enableValidation,
    };

    int passedCount = 0;
    int skippedCount = 0;
    int failedCount = 0;

    for (tests::RhiTest* test : selectedTests) {
        std::cout << "  [RUN ] " << test->name << " ... ";
        std::cout.flush();

        tests::RhiTestResult testResult;
        try {
            test->init(context);
            testResult = test->run(context);
            test->cleanup(context);
        } catch (const std::exception& exception) {
            testResult = tests::RhiTestResult::fail(exception.what());
            test->cleanup(context);
        } catch (...) {
            testResult = tests::RhiTestResult::fail("unknown exception");
            test->cleanup(context);
        }

        if (testResult.skipped) {
            ++skippedCount;
            std::cout << "SKIP";
        } else if (testResult.passed) {
            ++passedCount;
            std::cout << "PASS";
        } else {
            ++failedCount;
            std::cout << "FAIL";
        }

        if (!testResult.message.empty()) {
            std::cout << " - " << testResult.message;
        }
        std::cout << '\n';
    }

    device->waitIdle();
    device.reset();
    SDL_Quit();

    std::cout << "\nResults: " << passedCount << " passed, "
              << skippedCount << " skipped, "
              << failedCount << " failed";
    std::cout << " (" << selectedTests.size() << " selected)\n";
    std::cout << "Image output: " << options.outputDirectory.string() << '\n';

    return failedCount == 0 ? 0 : 1;
}
