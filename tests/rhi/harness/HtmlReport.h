#pragma once
#include "Evidence.h"
#include <chrono>

namespace metallic::tests::bench {
// Presentation only: verdicts come from the verified parent result. No GPU work.
Json htmlCase(const std::filesystem::path& directory, const Json& input, const Json& result);
void writeHtmlReport(const std::filesystem::path& root, const Json& run, const Json& cases, bool complete);
class HtmlReport {
public:
    HtmlReport(std::filesystem::path root, Json run);
    void append(const std::filesystem::path& directory, const Json& input, const Json& result);
    void finish();
    ~HtmlReport();
private:
    std::filesystem::path root_;
    Json run_, cases_ = Json::array();
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    size_t previewBytes_ = 0;
    bool complete_ = false;
};
} // namespace metallic::tests::bench
