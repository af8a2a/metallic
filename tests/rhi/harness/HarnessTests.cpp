#include "Runner.h"
#include "ValidationRecorder.h"
#include <gtest/gtest.h>
#include <volk.h>
#include <thread>
#include <fstream>

namespace metallic::tests::bench {
void htmlReportProtocol();
namespace {

std::filesystem::path outputDirectory()
{
    const auto path = std::filesystem::absolute(std::filesystem::path(".tmp/testbench-harness") /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    return path;
}

std::string pathArgument(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

void requirements()
{
    auto config = profile("core", Validation::Core).value();
    render::DeviceCapabilities caps;
    caps.shaderObject = true;
    Requirements required{.capabilities = {Capability::Bindless}};
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Core).status, Status::SkipNotEnabled);
    config = profile("binding", Validation::Core).value();
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Core).status, Status::SkipUnsupported);
    caps.bindlessDescriptorHeap = true;
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Off).status, Status::EnvironmentFailure);
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Core).status, Status::Pass);
    EXPECT_EQ(evaluate(required, config, caps, {}, Validation::Core).status, Status::SkipUnsupported);
    EXPECT_FALSE(profile("typo", Validation::Core));
    required.validation = Validation::Synchronization;
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Core).status, Status::EnvironmentFailure);
    EXPECT_EQ(evaluate(required, config, caps, {render::QueueType::Graphics}, Validation::Synchronization).status, Status::Pass);
    const auto sync = profile("core", parseValidation("sync")).value();
    EXPECT_TRUE(sync.desc.enableValidation && sync.desc.enableSynchronizationValidation);
    EXPECT_EQ(std::string(name(Validation::Synchronization)), "sync");
    EXPECT_THROW(parseValidation("typo"), std::invalid_argument);
    const auto unified = profile("core-unified", Validation::Core).value();
    Requirements unifiedRequired{.capabilities = {Capability::UnifiedLayouts}};
    EXPECT_EQ(evaluate(unifiedRequired, unified, caps, {render::QueueType::Graphics}, Validation::Core).status, Status::SkipUnsupported);
    EXPECT_EQ(evaluate(unifiedRequired, unified, caps, {render::QueueType::Graphics}, Validation::Core,
        {.unifiedImageLayouts = true}).status, Status::Pass);
}

void recorderLifetime()
{
    ValidationRecorder recorder;
    std::string id = "SYNC-HAZARD-WRITE-AFTER-READ", text = "original", objectName = "buffer";
    render::ValidationObject object{1, 2, objectName.c_str()};
    const auto sink = recorder.sink();
    sink.callback(sink.context, {VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT, 1, 42, id.c_str(), text.c_str(), {&object, 1}});
    text.assign("changed"); objectName.assign("changed"); id.assign("changed");
    const auto snapshot = recorder.snapshot();
    EXPECT_EQ(snapshot["messages"][0]["text"], "original");
    EXPECT_EQ(snapshot["messages"][0]["objects"][0]["name"], "buffer");
    EXPECT_TRUE(recorder.failed());
    ValidationRecorder small(1);
    const auto bounded = small.sink();
    bounded.callback(bounded.context, {0, 0, 0, "info", "message", {}});
    bounded.callback(bounded.context, {0, 0, 0, "info", "message", {}});
    EXPECT_TRUE(small.failed());
    EXPECT_TRUE(small.snapshot()["captureFailed"]);
}

void recorderThreads()
{
    ValidationRecorder recorder;
    auto sink = recorder.sink();
    std::vector<std::jthread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([sink] { for (int n = 0; n < 50; ++n) { sink.callback(sink.context, {0, 0, n, "info", "message", {}}); } });
    }
    threads.clear();
    EXPECT_EQ(recorder.snapshot()["messages"].size(), 200);
    EXPECT_FALSE(recorder.failed());
    sink.callback(sink.context, {VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT, 0, 0, "non-VUID", "warning", {}});
    EXPECT_TRUE(recorder.failed());
}

void filtering()
{
    EXPECT_TRUE(matchesFilter("RHICommand.timestamp_query", "*timestamp*:*copy*"));
    EXPECT_FALSE(matchesFilter("RHICommand.timestamp_query", "*-*timestamp*"));
    EXPECT_TRUE(matchesFilter("abc", "a?c"));
    EXPECT_FALSE(matchesFilter("abc", "a?"));
    EXPECT_TRUE(matchesFilter("abc", "-def"));
}

void evidenceIntegrity()
{
    const auto root = outputDirectory();
    Evidence evidence(root);
    EXPECT_THROW(evidence.json("../outside.json", {}), std::runtime_error);
    evidence.json("data.json", {{"value", 1}});
    const auto before = fileHash(root / "data.json");
    std::ofstream(root / "data.json", std::ios::trunc) << "truncated";
    EXPECT_NE(before, fileHash(root / "data.json"));
    Json input{{"id", "case"}, {"profile", "core"}, {"iteration", 0}};
    EXPECT_EQ(verifyChild(root, input, {0, true})["status"], "Timeout");
    EXPECT_EQ(verifyChild(root, input, {3, false})["status"], "Crash");
    EXPECT_EQ(verifyChild(root, input, {0, false})["status"], "InfrastructureFailure");
    std::ofstream(root / "result.json") << "{";
    EXPECT_EQ(verifyChild(root, input, {0, false})["status"], "InfrastructureFailure");
}

void childFailures()
{
    const auto root = outputDirectory();
    const auto result = runProcess(executablePath(), {"--tb-run", "--tb-suite", "harness-fixtures", "--output-dir",
        pathArgument(root / "results")}, root / "process", std::chrono::seconds(60));
    ASSERT_FALSE(result.timedOut);
    ASSERT_EQ(result.exitCode, 1);
    const auto cases = readJson(root / "results/results.json").at("cases");
    ASSERT_EQ(cases.size(), 7);
    EXPECT_TRUE(std::filesystem::exists(root / "results/report.html"));
    EXPECT_EQ(cases[0].at("status"), "Timeout"); // Later cases must still execute.
    for (const auto& sample : cases) {
        const auto id = sample.at("id").get<std::string>();
        EXPECT_TRUE(std::filesystem::exists(root / "results/core" / id / "0/report.html"));
        const auto expected = id.ends_with("_pass") ? "Pass" : id.ends_with("_crash") ? "Crash" : id.ends_with("_timeout") ? "Timeout" : "Fail";
        EXPECT_EQ(sample.at("status"), expected) << sample.dump();
        if (id.ends_with("fail_cleanup")) {
            EXPECT_NE(sample.at("message").get<std::string>().find("first run failure"), std::string::npos);
            EXPECT_NE(sample.at("message").get<std::string>().find("cleanup failure"), std::string::npos);
        }
        if (expected == std::string("Pass")) {
            const auto capabilities = readJson(root / "results/core" / id / "0/capabilities.json");
            EXPECT_FALSE(capabilities.at("deviceCreated").get<bool>());
        }
    }
}

void replayAndPaths()
{
    const auto root = outputDirectory() / u8"space path 中文";
    const auto output = root / "original";
    const auto result = runProcess(executablePath(), {"--tb-run", "--tb-suite", "contract", "--tb-filter", "*buffer_range_cpu_contract", "--tb-repeat", "2",
        "--output-dir", pathArgument(output)}, root / "process", std::chrono::seconds(30));
    ASSERT_EQ(result.exitCode, 0);
    const auto cases = readJson(output / "results.json").at("cases");
    ASSERT_EQ(cases.size(), 2);
    const auto casePath = output / "core" / cases[0].at("id").get<std::string>() / "0";
    const auto replay = runProcess(executablePath(), {"--tb-replay", pathArgument(casePath), "--output-dir",
        pathArgument(root / "replay")}, root / "replay-process", std::chrono::seconds(30));
    EXPECT_EQ(replay.exitCode, 0);
    const auto rejected = runProcess(executablePath(), {"--tb-run", "--rhi-bindless"}, root / "invalid-process", std::chrono::seconds(10));
    EXPECT_EQ(rejected.exitCode, 2);
}

void resultProtocol()
{
    const auto root = outputDirectory();
    const auto output = root / "results";
    const auto process = runProcess(executablePath(), {"--tb-run", "--tb-suite", "contract", "--tb-filter", "*buffer_range_cpu_contract", "--output-dir",
        pathArgument(output)}, root / "process", std::chrono::seconds(30));
    ASSERT_EQ(process.exitCode, 0);
    const auto directory = output / "core/RHIValidation.buffer_range_cpu_contract/0";
    const auto input = readJson(directory / "input.json");
    EXPECT_EQ(verifyChild(directory, input, {0, false}).at("status"), "Pass");
    EXPECT_EQ(verifyChild(directory, input, {1, false}).at("status"), "InfrastructureFailure");
    const auto result = readJson(directory / "result.json");
    auto invalid = result;
    invalid["executed"] = false;
    std::ofstream(directory / "result.json") << invalid.dump();
    EXPECT_EQ(verifyChild(directory, input, {0, false}).at("status"), "InfrastructureFailure");
    invalid = result;
    invalid["files"] = Json::array();
    std::ofstream(directory / "result.json") << invalid.dump();
    EXPECT_EQ(verifyChild(directory, input, {0, false}).at("status"), "InfrastructureFailure");
    std::ofstream(directory / "result.json") << result.dump();
    std::ofstream(directory / "ranges.json") << "changed";
    EXPECT_EQ(verifyChild(directory, input, {0, false}).at("status"), "InfrastructureFailure");
}

void differentialProtocol()
{
    const auto root = outputDirectory();
    const auto reference = root / "reference", target = root / "target";
    const Json spec{{"toggle", "positionFetch"}, {"capability", "positionFetch"},
        {"absoluteTolerance", 0.0001}, {"relativeTolerance", 0.0}};
    const auto populate = [&](const std::filesystem::path& directory, bool enabled) {
        Evidence evidence(directory);
        evidence.json("parent-result.json", {{"status", "Pass"}, {"executed", true}, {"failed", false}});
        evidence.json("input.json", {{"id", "case"}, {"seed", 1}, {"iteration", 0}, {"binaryHash", "binary"},
            {"shaderFingerprint", "shader"}, {"validation", "core"}, {"layerPath", "layers"},
            {"variant", enabled ? "target" : "reference"}});
        evidence.json("capabilities.json", {{"uuid", "gpu"}, {"driverVersion", 1}, {"driverInfo", "driver"},
            {"apiVersion", 1}, {"validationMode", "core"}, {"capabilities", Json::array({
                {{"id", "positionFetch"}, {"requested", enabled}, {"enabled", enabled}}})}});
        evidence.json("profile.json", {{"id", enabled ? "target" : "reference"}, {"positionFetch", enabled}, {"rayQuery", true}});
        evidence.json("fixture.json", {{"vertices", {0, 1, 2}}});
        evidence.json("execution.json", {{"targetUsed", enabled}});
        evidence.json("observations.json", {37, 1.25, 0.5});
    };
    const auto check = [&] { return compareEvidence(reference, target, spec); };
    populate(reference, false); populate(target, true);
    EXPECT_EQ(check().at("status"), "Pass");
    writeJson(target / "observations.json", {37, 1.25001, 0.5});
    EXPECT_EQ(check().at("status"), "Pass");
    writeJson(target / "observations.json", {38, 1.25, 0.5});
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    writeJson(target / "execution.json", {{"targetUsed", false}});
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    auto device = readJson(target / "capabilities.json"); device["uuid"] = "another-gpu";
    writeJson(target / "capabilities.json", device);
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    auto profile = readJson(target / "profile.json"); profile["rayQuery"] = false;
    writeJson(target / "profile.json", profile);
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    auto input = readJson(target / "input.json"); input["seed"] = 2;
    writeJson(target / "input.json", input);
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    writeJson(target / "fixture.json", {{"vertices", {3, 2, 1}}});
    EXPECT_EQ(check().at("status"), "Fail");
    populate(target, true);
    writeJson(target / "observations.json", {37, nullptr, 0.5});
    EXPECT_EQ(check().at("status"), "Fail");
    writeJson(target / "parent-result.json", {{"status", "SkipUnsupported"}, {"executed", false}, {"failed", false}});
    EXPECT_EQ(check().at("status"), "SkipUnsupported");
    EXPECT_EQ(readJson(reference / "parent-result.json").at("status"), "Pass");
    writeJson(target / "parent-result.json", {{"status", "Fail"}, {"executed", true}, {"failed", true}});
    EXPECT_EQ(check().at("status"), "Fail");
}

class HarnessAdapter : public ::testing::Test {
public:
    explicit HarnessAdapter(void (*body)()) : body_(body) {}
    void TestBody() override { body_(); }
private:
    void (*body_)();
};
} // namespace

void registerHarnessTests()
{
    for (const auto& entry : std::vector<std::pair<const char*, void (*)()>>{
        {"Requirements", requirements}, {"RecorderLifetime", recorderLifetime}, {"RecorderThreads", recorderThreads},
        {"Filtering", filtering}, {"EvidenceIntegrity", evidenceIntegrity}, {"ChildFailures", childFailures},
        {"ReplayAndPaths", replayAndPaths}, {"ResultProtocol", resultProtocol}, {"DifferentialProtocol", differentialProtocol}, {"HtmlReportProtocol", htmlReportProtocol}}) {
        ::testing::RegisterTest("TestbenchHarness", entry.first, nullptr, nullptr, __FILE__, __LINE__,
            [body = entry.second]() -> HarnessAdapter* { return new HarnessAdapter(body); });
    }
}
} // namespace metallic::tests::bench
