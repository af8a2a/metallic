#pragma once
#include "Evidence.h"
#include "Requirements.h"
#include "ProcessRunner.h"
#include <optional>

namespace metallic::tests::bench {
std::optional<int> runIfRequested(int argc, char** argv);
Json verifyChild(const std::filesystem::path& directory, const Json& input, ProcessResult process);
Json compareEvidence(const std::filesystem::path& reference, const std::filesystem::path& target, const Json& spec);
bool matchesFilter(const std::string& id, const std::string& filter);
} // namespace metallic::tests::bench
