#include "RhiTest.h"
#include "Runtime/Task/TaskSystem.h"

#include <SDL3/SDL.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

namespace render = metallic::render;

struct Options {
    bool enableValidation = true;
    std::filesystem::path outputDirectory = "rhi-test-output";
};

void printRhiUsage()
{
    spdlog::info(
        "Metallic RHI options:\n"
        "  --output-dir <path>      Write generated images to <path>\n"
        "  --rhi-no-validation      Disable Vulkan validation for RHI tests\n"
        "  --rhi-validation         Enable Vulkan validation for RHI tests\n"
        "\n"
        "GoogleTest options replace the old custom runner flags:\n"
        "  --gtest_list_tests       List registered tests\n"
        "  --gtest_filter=<filter>  Run a subset of tests");
}

bool parseArguments(int argc, char** argv, Options& options, std::vector<std::string>& gtestArguments)
{
    gtestArguments.clear();
    gtestArguments.emplace_back(argv[0]);

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--rhi-no-validation") {
            options.enableValidation = false;
            continue;
        }
        if (argument == "--rhi-validation") {
            options.enableValidation = true;
            continue;
        }
        if (argument == "--output-dir") {
            if (index + 1 >= argc) {
                spdlog::error("--output-dir requires a value");
                return false;
            }
            options.outputDirectory = argv[++index];
            continue;
        }
        if (argument.starts_with("--output-dir=")) {
            options.outputDirectory = std::string(argument.substr(13));
            continue;
        }

        if (argument == "--list") {
            gtestArguments.emplace_back("--gtest_list_tests");
            continue;
        }
        if (argument == "--filter") {
            if (index + 1 >= argc) {
                spdlog::error("--filter requires a value");
                return false;
            }
            gtestArguments.emplace_back(std::string("--gtest_filter=*") + argv[++index] + "*");
            continue;
        }
        if (argument.starts_with("--filter=")) {
            gtestArguments.emplace_back(
                std::string("--gtest_filter=*") + std::string(argument.substr(9)) + "*");
            continue;
        }
        if (argument == "--help" || argument == "-h") {
            printRhiUsage();
        }

        gtestArguments.emplace_back(argument);
    }

    return true;
}

std::vector<char*> makeMutableArgv(std::vector<std::string>& arguments)
{
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) {
        argv.push_back(argument.data());
    }
    return argv;
}

const char* suiteNameFor(metallic::tests::RhiTestType type)
{
    switch (type) {
    case metallic::tests::RhiTestType::Validation:
        return "RhiValidation";
    case metallic::tests::RhiTestType::Resource:
        return "RhiResource";
    case metallic::tests::RhiTestType::Command:
        return "RhiCommand";
    case metallic::tests::RhiTestType::Rendering:
        return "RhiRendering";
    }

    return "RhiUnknown";
}

class RhiTestEnvironment : public ::testing::Environment {
public:
    explicit RhiTestEnvironment(Options options)
        : options_(std::move(options))
    {
    }

    void SetUp() override
    {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            skipReason_ = std::string("SDL_Init failed: ") + SDL_GetError();
            return;
        }
        sdlInitialized_ = true;

        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RHI Tests",
                .enableValidation = options_.enableValidation,
            },
            device_);
        if (!result) {
            const std::string message = std::string("createDevice returned ") + metallic::tests::toString(result);
            if (render::hasError(result, render::Error::Unsupported)) {
                skipReason_ = message;
            } else {
                setupFailure_ = message;
            }
            return;
        }

        graphicsQueue_ = device_->getQueue(render::QueueType::Graphics);
        if (graphicsQueue_ == nullptr) {
            setupFailure_ = "no graphics queue is available";
            return;
        }

        context_ = std::make_unique<metallic::tests::RhiTestContext>(
            metallic::tests::RhiTestContext{
                .device = *device_,
                .graphicsQueue = *graphicsQueue_,
                .outputDirectory = options_.outputDirectory,
                .enableValidation = options_.enableValidation,
            });
    }

    void TearDown() override
    {
        context_.reset();
        graphicsQueue_ = nullptr;
        if (device_ != nullptr) {
            (void)device_->waitIdle();
            device_.reset();
        }
        if (sdlInitialized_) {
            SDL_Quit();
            sdlInitialized_ = false;
        }
    }

    metallic::tests::RhiTestContext* context() const
    {
        return context_.get();
    }

    const std::string& skipReason() const
    {
        return skipReason_;
    }

    const std::string& setupFailure() const
    {
        return setupFailure_;
    }

private:
    Options options_;
    bool sdlInitialized_ = false;
    std::unique_ptr<render::Device> device_;
    render::Queue* graphicsQueue_ = nullptr;
    std::unique_ptr<metallic::tests::RhiTestContext> context_;
    std::string skipReason_;
    std::string setupFailure_;
};

RhiTestEnvironment* gEnvironment = nullptr;

class RhiGTestAdapter : public ::testing::Test {
public:
    explicit RhiGTestAdapter(std::unique_ptr<metallic::tests::RhiTest> test)
        : test_(std::move(test))
    {
    }

protected:
    void TestBody() override
    {
        if (gEnvironment == nullptr) {
            FAIL() << "RHI test environment is missing";
            return;
        }
        if (!gEnvironment->setupFailure().empty()) {
            FAIL() << gEnvironment->setupFailure();
            return;
        }
        if (!gEnvironment->skipReason().empty()) {
            GTEST_SKIP() << gEnvironment->skipReason();
        }

        metallic::tests::RhiTestContext* context = gEnvironment->context();
        if (context == nullptr) {
            FAIL() << "RHI test context is unavailable";
            return;
        }

        metallic::tests::RhiTestResult result;
        try {
            test_->init(*context);
            result = test_->run(*context);
        } catch (const std::exception& exception) {
            result = metallic::tests::RhiTestResult::fail(exception.what());
        } catch (...) {
            result = metallic::tests::RhiTestResult::fail("unknown exception");
        }

        try {
            test_->cleanup(*context);
        } catch (const std::exception& exception) {
            if (result.passed || result.skipped) {
                result = metallic::tests::RhiTestResult::fail(
                    std::string("cleanup failed: ") + exception.what());
            }
        } catch (...) {
            if (result.passed || result.skipped) {
                result = metallic::tests::RhiTestResult::fail("cleanup failed with unknown exception");
            }
        }

        if (result.skipped) {
            GTEST_SKIP() << result.message;
        }
        EXPECT_TRUE(result.passed) << result.message;
        if (!result.message.empty()) {
            RecordProperty("message", result.message);
        }
    }

private:
    std::unique_ptr<metallic::tests::RhiTest> test_;
};

void registerRhiTests()
{
    using metallic::tests::RhiTestRegistry;

    for (const RhiTestRegistry::Factory& factory : RhiTestRegistry::factories()) {
        std::unique_ptr<metallic::tests::RhiTest> prototype = factory();
        if (prototype == nullptr || prototype->name == nullptr) {
            continue;
        }

        const char* suiteName = suiteNameFor(prototype->type);
        const std::string testName = prototype->name;
        ::testing::RegisterTest(
            suiteName,
            testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [factory]() -> RhiGTestAdapter* {
                return new RhiGTestAdapter(factory());
            });
    }
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    std::vector<std::string> gtestArguments;
    if (!parseArguments(argc, argv, options, gtestArguments)) {
        return 1;
    }

    std::vector<char*> gtestArgv = makeMutableArgv(gtestArguments);
    int gtestArgc = static_cast<int>(gtestArgv.size());
    ::testing::InitGoogleTest(&gtestArgc, gtestArgv.data());

    registerRhiTests();

    auto* environment = new RhiTestEnvironment(options);
    gEnvironment = environment;
    ::testing::AddGlobalTestEnvironment(environment);

    const auto taskInitialization = metallic::task::initializeTaskSystem();
    if (!taskInitialization) {
        spdlog::error("TaskSystem initialization failed: {}", taskInitialization.error().message);
        return 1;
    }
    const int result = RUN_ALL_TESTS();
    metallic::task::shutdownTaskSystem();
    return result;
}
