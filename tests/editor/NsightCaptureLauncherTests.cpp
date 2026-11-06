#include "Editor/NsightCaptureLauncher.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

namespace {

namespace fs = std::filesystem;

class NsightCaptureLauncherTests : public testing::Test {
protected:
    void SetUp() override
    {
        const auto run = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        directory_ = fs::path(TEST_BINARY_DIR) / "NsightCaptureLauncher" / run;
        installationRoot_ = directory_ / u8"Nsight 安装 目录";
        executable_ = installationRoot_ / "host/windows-desktop-nomad-x64/ngfx-ui.exe";
        fs::create_directories(executable_.parent_path());
    }

    fs::path directory_;
    fs::path installationRoot_;
    fs::path executable_;
};

TEST_F(NsightCaptureLauncherTests, LaunchesConfiguredExecutableWithExactUnicodeCaptureArgument)
{
    fs::copy_file(NSIGHT_LAUNCH_PROBE_EXECUTABLE, executable_);
    for (const auto* fileName : {u8"捕获 trace.ngfx-gputrace", u8"捕获 capture.ngfx-capture"}) {
        const auto capturePath = directory_ / fileName;
        std::ofstream(capturePath, std::ios::binary).put('\0');
        std::string error = "previous error";
        ASSERT_TRUE(metallic::openNsightCapture(installationRoot_, capturePath, error)) << error;
        EXPECT_TRUE(error.empty()) << error;

        auto receiptPath = capturePath;
        receiptPath += L".opened";
        const auto expectedUtf8 = capturePath.u8string();
        const std::string expected(reinterpret_cast<const char*>(expectedUtf8.data()), expectedUtf8.size());
        std::string received;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        do {
            std::ifstream receipt(receiptPath, std::ios::binary);
            received.assign(std::istreambuf_iterator<char>(receipt), std::istreambuf_iterator<char>());
            if (received == expected) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (std::chrono::steady_clock::now() < deadline);
        EXPECT_EQ(received, expected) << "Configured launch probe did not receive the capture path";
    }
}

TEST_F(NsightCaptureLauncherTests, ReportsMissingExecutable)
{
    const auto capturePath = directory_ / "trace.ngfx-gputrace";
    std::ofstream(capturePath, std::ios::binary).put('\0');
    std::string error;
    EXPECT_FALSE(metallic::openNsightCapture(installationRoot_, capturePath, error));
    EXPECT_FALSE(error.empty());
}

TEST_F(NsightCaptureLauncherTests, ReportsInvalidExecutable)
{
    const auto capturePath = directory_ / "trace.ngfx-gputrace";
    std::ofstream(capturePath, std::ios::binary).put('\0');
    std::ofstream(executable_, std::ios::binary) << "Invalid executable";
    std::string error;
    EXPECT_FALSE(metallic::openNsightCapture(installationRoot_, capturePath, error));
    EXPECT_FALSE(error.empty());
}

} // namespace
