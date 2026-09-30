#include "Editor/NsightLaunchOptions.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using metallic::NsightLaunchOptions;
using metallic::render::profiling::NsightCaptureMode;

bool parse(std::initializer_list<const char*> arguments, NsightLaunchOptions& options)
{
    std::vector<std::string> storage{"Metallic"};
    for (const char* argument : arguments) { storage.emplace_back(argument); }
    std::vector<char*> argv;
    for (auto& argument : storage) { argv.push_back(argument.data()); }
    for (int index = 1; index < static_cast<int>(argv.size()); ++index) {
        if (options.consume(static_cast<int>(argv.size()), argv.data(), index) != 1) { return false; }
    }
    return true;
}

TEST(NsightLaunchOptions, DefaultDoesNotExplicitlyEnableInjection)
{
    NsightLaunchOptions options;
    EXPECT_EQ(options.mode, NsightCaptureMode::Default);
}

TEST(NsightLaunchOptions, ModesAndAliases)
{
    for (auto arguments : {std::initializer_list<const char*>{"--nsight-mode", "gputrace"},
             {"--nsight-mode=gputrace"}, {"--nsight-gputrace"}}) {
        NsightLaunchOptions options;
        ASSERT_TRUE(parse(arguments, options));
        EXPECT_EQ(options.mode, NsightCaptureMode::GPUTrace);
    }
    for (auto arguments : {std::initializer_list<const char*>{"--nsight-mode", "capture"},
             {"--nsight-mode=capture"}, {"--nsight-capture"}}) {
        NsightLaunchOptions options;
        ASSERT_TRUE(parse(arguments, options));
        EXPECT_EQ(options.mode, NsightCaptureMode::GraphicsCapture);
    }
}

TEST(NsightLaunchOptions, RejectsMissingInvalidAndConflictingModes)
{
    for (auto arguments : {std::initializer_list<const char*>{"--nsight-mode"},
             {"--nsight-mode="}, {"--nsight-mode", "trace"}, {"--nsight-mode", "--help"},
             {"--nsight-capture", "--nsight-gputrace"}, {"--nsight-gputrace", "--nsight-capture"}}) {
        NsightLaunchOptions options;
        EXPECT_FALSE(parse(arguments, options));
    }
    NsightLaunchOptions options;
    EXPECT_TRUE(parse({"--nsight-gputrace", "--nsight-mode", "gputrace"}, options));
}

TEST(NsightLaunchOptions, LeavesUnrelatedArgumentsUntouched)
{
    NsightLaunchOptions options;
    char executable[] = "Metallic";
    char scene[] = "--scene";
    char path[] = "example.gltf";
    char* argv[] = {executable, scene, path};
    int index = 1;
    EXPECT_EQ(options.consume(3, argv, index), 0);
    EXPECT_EQ(index, 1);
}

} // namespace
