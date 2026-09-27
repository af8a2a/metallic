#include "HtmlReport.h"
#include <gtest/gtest.h>
#include <fstream>

namespace metallic::tests::bench {
namespace {
std::string textFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}
Json embedded(const std::filesystem::path& path)
{
    const auto html = textFile(path);
    const std::string marker = "<script id=\"report-data\" type=\"application/json\">";
    const auto begin = html.find(marker) + marker.size(), end = html.find("</script>", begin);
    return Json::parse(html.substr(begin, end - begin));
}
} // namespace
void htmlReportProtocol()
{
    const auto root = std::filesystem::absolute(std::filesystem::path(".tmp/testbench-html-selftest") /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto directory = root / "case with spaces #1";
    Evidence evidence(directory);
    const std::array<std::byte, 4> actual{std::byte{0}, std::byte{1}, std::byte{9}, std::byte{3}};
    const std::array<std::byte, 4> expected{std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
    evidence.bytes("actual.bin", actual); evidence.bytes("expected.bin", expected);
    evidence.json("visuals.json", Json::array({{{"format", "rgba8"}, {"label", "Explicit RGBA"},
        {"actual", "actual.bin"}, {"expected", "expected.bin"}, {"width", 1}, {"height", 1}}}));
    evidence.json("capabilities.json", {{"device", "CPU fixture"}, {"validationMode", "off"}});
    const std::string attack = "</script><script>window.reportInjection=true</script>&\"";
    Json input{{"seed", 42}, {"metadata", {{"suite", "contract"}, {"layer", "Harness"}, {"requiresDevice", false}}}};
    Json result{{"id", attack}, {"profile", "core"}, {"iteration", 2}, {"status", "Fail"},
        {"failed", true}, {"executed", true}, {"message", attack}, {"durationMs", 12.5}};
    const auto card = htmlCase(directory, input, result);
    ASSERT_EQ(card.at("buffers").size(), 1);
    EXPECT_EQ(card.at("buffers")[0].at("differentBytes"), 1);
    EXPECT_EQ(card.at("buffers")[0].at("firstMismatch"), 2);
    ASSERT_EQ(card.at("images").size(), 1);
    EXPECT_EQ(card.at("images")[0].at("actual").at("rgba"), "AAEJAw==");
    EXPECT_EQ(card.at("status"), "Fail");
    {
        HtmlReport report(root, {{"mode", "synthetic protocol self-test"}, {"planned", 7}});
        report.append(directory, input, result);
        for (const auto* status : {"Pass", "SkipUnsupported", "SkipNotEnabled", "Crash", "Timeout", "EnvironmentFailure"}) {
            const auto path = root / status;
            auto sample = result; sample["id"] = status; sample["status"] = status;
            sample["failed"] = std::string(status) == "Crash" || std::string(status) == "Timeout" || std::string(status) == "EnvironmentFailure";
            sample["policyFailure"] = std::string(status) == "SkipNotEnabled"; sample["executed"] = std::string(status) == "Pass";
            sample.erase("durationMs"); report.append(path, input, sample);
        }
        const auto pending = embedded(root / "report.html");
        EXPECT_FALSE(pending.at("complete").get<bool>());
        report.finish();
    }
    const auto html = textFile(root / "report.html");
    EXPECT_EQ(html.find(attack), std::string::npos);
    EXPECT_NE(html.find("\\u003c/script\\u003e"), std::string::npos);
    EXPECT_EQ(html.find("fetch("), std::string::npos);
    const auto data = embedded(root / "report.html");
    EXPECT_TRUE(data.at("complete").get<bool>());
    ASSERT_EQ(data.at("cases").size(), 7);
    EXPECT_EQ(data.at("cases")[0].at("id"), attack);
    EXPECT_EQ(data.at("cases")[0].at("casePage"), "case%20with%20spaces%20%231/report.html");
    EXPECT_EQ(data.at("cases")[1].at("durationMs"), nullptr); // Missing time is not zero.
    EXPECT_TRUE(data.at("cases")[3].at("policyFailure").get<bool>());
    EXPECT_TRUE(std::filesystem::exists(root / "Crash/report.html"));
    evidence.json("visuals.json", Json::array({{{"format", "rgba8"}, {"actual", "../outside.bin"}, {"width", 1}, {"height", 1}}}));
    const auto invalid = htmlCase(directory, input, result);
    EXPECT_TRUE(invalid.at("images").empty());
    EXPECT_FALSE(invalid.at("warnings").empty());
    evidence.json("capabilities.json", Json::array({1, 2, 3}));
    EXPECT_NO_THROW(htmlCase(directory, input, result));
    // A failed report write must be observable to the runner, not silent success.
    Evidence blocked(root / "blocked");
    std::filesystem::create_directory(root / "blocked/report.html");
    EXPECT_THROW(writeHtmlReport(root / "blocked", {}, Json::array(), true), std::filesystem::filesystem_error);
    {
        HtmlReport interrupted(root / "interrupted", {{"mode", "self-test"}});
    }
    EXPECT_FALSE(embedded(root / "interrupted/report.html").at("complete").get<bool>());
    EXPECT_TRUE(embedded(root / "interrupted/report.html").at("run").at("interrupted").get<bool>());
}
} // namespace metallic::tests::bench
