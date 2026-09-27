#pragma once
#include "Evidence.h"
#include "Requirements.h"
#include "ProcessRunner.h"
#include <optional>

namespace metallic::tests::bench {
std::optional<int> runIfRequested(int argc, char** argv);
Json verifyChild(const std::filesystem::path& directory, const Json& input, ProcessResult process);
bool matchesFilter(const std::string& id, const std::string& filter);
} // namespace metallic::tests::bench
