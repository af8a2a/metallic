#pragma once
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>
#include <optional>

namespace metallic::tests::bench {
struct ProcessResult {
    uint32_t exitCode = 0;
    bool timedOut = false;
};
std::filesystem::path executablePath();
std::optional<std::filesystem::path> findExecutable(const std::string& name);
std::vector<std::string> nativeArguments(int argc, char** argv);
ProcessResult runProcess(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
    const std::filesystem::path& output, std::chrono::milliseconds timeout);
} // namespace metallic::tests::bench
