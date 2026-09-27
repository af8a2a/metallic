#include "GTestHtmlReport.h"
#include "HtmlReport.h"
#include <gtest/gtest.h>
#include <iostream>
#include <map>

namespace metallic::tests::bench {
namespace {
class Listener final : public ::testing::EmptyTestEventListener {
public:
    Listener(std::filesystem::path output, std::string mode, std::function<Json()> device,
        std::shared_ptr<GTestHtmlState> state) : output_(std::move(output)), mode_(std::move(mode)),
        device_(std::move(device)), state_(std::move(state)) {}
    void OnTestProgramStart(const ::testing::UnitTest&) override
    {
        protect([&] {
            root_ = std::filesystem::absolute(output_ / "reports" / std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
            report_ = std::make_unique<HtmlReport>(root_, Json{{"mode", mode_},
                {"completionStatus", mode_ == "legacy" ? "Legacy GoogleTest verdicts; isolated validation audit is available in --tb-run." : "Harness self-tests"}});
        });
    }
    void OnTestIterationStart(const ::testing::UnitTest&, int iteration) override { iteration_ = iteration; }
    void OnTestStart(const ::testing::TestInfo&) override { protect([&] { before_ = files(); }); }
    void OnTestEnd(const ::testing::TestInfo& info) override
    {
        protect([&] {
            if (!report_) { return; }
            const auto directory = root_ / "cases" / std::to_string(index_++);
            Evidence evidence(directory);
            if (device_) { const auto device = device_(); if (!device.empty()) { evidence.json("capabilities.json", device); } }
            // Copy only artifacts touched by this case; never show an earlier run's images as current.
            size_t copied = 0;
            for (const auto& [path, stamp] : files()) {
                if (const auto previous = before_.find(path); previous != before_.end() && previous->second == stamp) { continue; }
                if (std::filesystem::file_size(path) > 16 * 1024 * 1024 || copied++ >= 32) { continue; }
                auto name = path.filename();
                if (std::filesystem::exists(directory / name)) { name = std::to_string(copied) + "-" + name.string(); }
                std::filesystem::copy_file(path, directory / name);
            }
            const auto* observed = info.result();
            std::string message;
            for (int i = 0; i < observed->total_part_count(); ++i) {
                const auto& part = observed->GetTestPartResult(i);
                if (part.failed() || part.skipped()) { message += std::string(part.message()) + '\n'; }
            }
            const std::string id = std::string(info.test_suite_name()) + "." + info.name();
            const Json input{{"mode", mode_}, {"metadata", {{"suite", info.test_suite_name()},
                {"layer", mode_ == "harness" ? "Harness" : "Legacy"}, {"requiresDevice", mode_ != "harness"}}}};
            const Json result{{"id", id}, {"profile", mode_}, {"iteration", iteration_},
                {"status", observed->Skipped() ? "Skipped" : observed->Passed() ? "Pass" : "Fail"},
                {"executed", !observed->Skipped()}, {"failed", observed->Failed()}, {"message", message},
                {"durationMs", observed->elapsed_time()}, {"durationScope", "GoogleTest case wall time"}};
            evidence.json("parent-result.json", result); report_->append(directory, input, result);
        });
    }
    void OnTestProgramEnd(const ::testing::UnitTest&) override
    {
        protect([&] { if (report_) { if (!state_->failed) { report_->finish(); } std::cout << "HTML report: " << (root_ / "report.html").string() << '\n'; } });
    }
private:
    using Stamps = std::map<std::filesystem::path, std::pair<std::filesystem::file_time_type, uintmax_t>>;
    Stamps files() const
    {
        Stamps found;
        if (mode_ != "legacy" || !std::filesystem::exists(output_)) { return found; }
        for (auto it = std::filesystem::recursive_directory_iterator(output_); it != std::filesystem::recursive_directory_iterator(); ++it) {
            if (it->is_symlink()) { if (it->is_directory()) { it.disable_recursion_pending(); } continue; }
            if (it->is_directory() && it->path().filename() == "reports") { it.disable_recursion_pending(); continue; }
            const auto extension = it->path().extension();
            if (it->is_regular_file() && (extension == ".png" || extension == ".bin" || extension == ".json")) {
                found[it->path()] = {it->last_write_time(), it->file_size()};
            }
        }
        return found;
    }
    template<typename F> void protect(F&& action)
    {
        try { action(); }
        catch (const std::exception& error) { state_->failed = true; std::cerr << "HTML report failed: " << error.what() << '\n'; }
    }
    std::filesystem::path output_, root_;
    std::string mode_;
    std::function<Json()> device_;
    std::shared_ptr<GTestHtmlState> state_;
    std::unique_ptr<HtmlReport> report_;
    Stamps before_;
    int iteration_ = 0;
    size_t index_ = 0;
};
} // namespace
std::shared_ptr<GTestHtmlState> installGTestHtmlReport(const std::filesystem::path& output,
    std::string mode, std::function<Json()> device)
{
    auto state = std::make_shared<GTestHtmlState>();
    ::testing::UnitTest::GetInstance()->listeners().Append(new Listener(output, std::move(mode), std::move(device), state));
    return state;
}
} // namespace metallic::tests::bench
