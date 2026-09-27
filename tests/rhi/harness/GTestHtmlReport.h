#pragma once
#include "Evidence.h"
#include <functional>
#include <memory>

namespace metallic::tests::bench {
struct GTestHtmlState { bool failed = false; };
std::shared_ptr<GTestHtmlState> installGTestHtmlReport(const std::filesystem::path& output,
    std::string mode, std::function<Json()> device = {});
} // namespace metallic::tests::bench
