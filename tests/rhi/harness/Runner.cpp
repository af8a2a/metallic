#include "Runner.h"
#include "RhiTest.h"
#include "ValidationRecorder.h"
#include "VulkanDiagnostics.h"
#include "Runtime/Task/TaskSystem.h"
#include <SDL3/SDL.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <set>
#include <cstdlib>

namespace metallic::tests {
std::vector<RhiTestRegistry::Factory> testbenchFaultFactories();
}

namespace metallic::tests::bench {
namespace {

struct Case {
    std::string id;
    Metadata metadata;
    RhiTestRegistry::Factory factory;
};
struct Options {
    std::string mode;
    std::string suite = "core";
    std::string profile;
    std::string filter = "*";
    Validation validation = Validation::Core;
    std::filesystem::path output;
    std::filesystem::path input;
    std::filesystem::path replay;
    std::filesystem::path layerPath;
    uint32_t repeat = 1;
    uint64_t seed = 1;
    bool requireAll = false;
    bool allowMismatch = false;
};

const char* suiteName(RhiTestType type)
{
    switch (type) {
    case RhiTestType::Validation: return "RhiValidation";
    case RhiTestType::Resource: return "RhiResource";
    case RhiTestType::Command: return "RhiCommand";
    case RhiTestType::Rendering: return "RhiRendering";
    }
    return "Unknown";
}

std::vector<Case> cases()
{
    std::vector<Case> result;
    auto factories = RhiTestRegistry::factories();
    const auto faults = testbenchFaultFactories();
    factories.insert(factories.end(), faults.begin(), faults.end());
    for (const auto& factory : factories) {
        const auto test = factory();
        if (auto metadata = test->metadata()) {
            result.push_back({std::string(suiteName(test->type)) + "." + test->name, *metadata, factory});
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    for (size_t i = 1; i < result.size(); ++i) {
        if (result[i - 1].id == result[i].id) { throw std::runtime_error("duplicate testbench case ID"); }
    }
    return result;
}

uint64_t number(const std::string& value)
{
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) { throw std::runtime_error("invalid number: " + value); }
    return result;
}

Options parse(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i], value;
        const auto separator = key.find('=');
        if (separator != std::string::npos) { value = key.substr(separator + 1); key.resize(separator); }
        const auto argument = [&]() {
            if (separator != std::string::npos) { return value; }
            if (++i >= argc) { throw std::runtime_error("missing value: " + key); }
            return std::string(argv[i]);
        };
        if (key == "--tb-plan" || key == "--tb-run" || key == "--tb-child" || key == "--tb-self-test" || key == "--tb-help") {
            if (!options.mode.empty()) { throw std::runtime_error("choose one testbench mode"); }
            options.mode = key;
        } else if (key == "--tb-replay") {
            if (!options.mode.empty()) { throw std::runtime_error("choose one testbench mode"); }
            options.mode = key; options.replay = std::filesystem::u8path(argument());
        } else if (key == "--tb-suite") { options.suite = argument(); }
        else if (key == "--tb-profile") { options.profile = argument(); }
        else if (key == "--tb-filter" || key == "--gtest_filter") { options.filter = argument(); }
        else if (key == "--output-dir") { options.output = std::filesystem::u8path(argument()); }
        else if (key == "--tb-input") { options.input = std::filesystem::u8path(argument()); }
        else if (key == "--tb-layer-path") { options.layerPath = std::filesystem::absolute(std::filesystem::u8path(argument())); }
        else if (key == "--tb-seed") { options.seed = number(argument()); }
        else if (key == "--tb-repeat" || key == "--gtest_repeat") {
            const auto count = number(argument());
            if (!count || count > 1000) { throw std::runtime_error("repeat must be 1..1000"); }
            options.repeat = uint32_t(count);
        } else if (key == "--tb-validation") {
            const auto mode = argument();
            options.validation = parseValidation(mode);
        } else if (key == "--tb-require-all") { options.requireAll = true; }
        else if (key == "--tb-allow-version-mismatch") { options.allowMismatch = true; }
        else { throw std::runtime_error("unsupported/conflicting testbench option: " + key); }
    }
    if (options.mode.empty()) { throw std::runtime_error("missing testbench mode"); }
    if (!options.profile.empty() && !profile(options.profile, options.validation)) { throw std::runtime_error("unknown profile"); }
    return options;
}

Json metadataJson(const Metadata& value)
{
    Json caps = Json::array(), queues = Json::array(), timestampQueues = Json::array();
    for (auto cap : value.requirements.capabilities) { caps.push_back(name(cap)); }
    for (auto queue : value.requirements.queues) { queues.push_back(int(queue)); }
    for (auto queue : value.requirements.timestampQueues) { timestampQueues.push_back(int(queue)); }
    return {{"suite", value.suite}, {"layer", name(value.layer)}, {"profile", value.profile},
        {"isolation", "FreshProcess"}, {"requiresDevice", value.requirements.requiresDevice},
        {"validationRequired", value.requirements.validation != Validation::Off},
        {"minimumValidation", name(value.requirements.validation)},
        {"capabilities", caps}, {"queues", queues}, {"timestampQueues", timestampQueues}, {"nativeDescriptorPointers", value.requirements.nativeDescriptorPointers}, {"coverage", value.coverage}, {"timeoutMs", value.timeout.count()},
        {"artifacts", value.artifacts}};
}

Json profileJson(const Profile& value)
{
    const auto& desc = value.desc;
    return {{"id", value.id}, {"validation", desc.enableSynchronizationValidation ? "sync" : desc.enableValidation ? "core" : "off"},
        {"shaderObject", desc.enableShaderObject}, {"bindless", desc.enableBindlessDescriptorHeap},
        {"asyncCompute", desc.enableAsyncCompute}, {"unifiedLayouts", desc.preferUnifiedImageLayouts},
        {"rayQuery", desc.enableRayQuery}, {"opacityMicromap", desc.enableOpacityMicromap},
        {"positionFetch", desc.enableRayTracingPositionFetch}, {"dgc", desc.enableDeviceGeneratedCommands},
        {"streamline", desc.enableStreamline}, {"aftermath", desc.enableAftermath}};
}

std::string utf8(const std::filesystem::path& path)
{
    const auto bytes = path.u8string();
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

void issue(Verdict& verdict, Status status, const std::string& message)
{
    if (!failed(verdict.status)) { verdict.status = status; verdict.message = message; }
    else { verdict.message += "\n" + message; }
}

Json execute(const Case& selected, const Json& input, Evidence& evidence)
{
    Verdict verdict;
    ValidationRecorder recorder;
    auto config = profile(input.at("profile").get<std::string>(), parseValidation(input.at("validation").get<std::string>())).value();
    auto test = selected.factory();
    std::unique_ptr<render::Device> device;
    std::unique_ptr<RhiTestContext> context;
    bool sdl = false, tasks = false, initialized = false;
    auto phase = [&](const char* label) { recorder.phase(label); evidence.phase(label); };
    auto step = [&](const char* label, auto&& body) {
        try { phase(label); }
        catch (const std::exception& error) { issue(verdict, Status::InfrastructureFailure, error.what()); }
        // Recording failure must never suppress cleanup or GPU resource retirement.
        try { body(); }
        catch (const std::filesystem::filesystem_error& error) { issue(verdict, Status::InfrastructureFailure, error.what()); }
        catch (const std::ios_base::failure& error) { issue(verdict, Status::InfrastructureFailure, error.what()); }
        catch (const std::exception& error) { issue(verdict, Status::Fail, std::string(label) + ": " + error.what()); }
        catch (...) { issue(verdict, Status::Fail, std::string(label) + ": unknown exception"); }
    };
    step("setup", [&] {
        evidence.json("profile.json", profileJson(config));
        if (!selected.metadata.requirements.requiresDevice) {
            evidence.json("capabilities.json", {{"deviceCreated", false}, {"validationMode", "off"}});
            return;
        }
        if (failed(verdict.status)) { return; }
        // Layer filters can silently override explicitly requested validation.
        // Core conformance uses an unmodified validation configuration.
        for (const auto* variable : {"VK_LOADER_LAYERS_DISABLE", "VK_LOADER_LAYERS_ENABLE", "VK_INSTANCE_LAYERS",
            "VK_LAYER_DISABLES", "VK_LAYER_ENABLES", "VK_LAYER_SETTINGS_PATH"}) {
            if (const auto* value = std::getenv(variable); value && *value) {
                issue(verdict, Status::EnvironmentFailure, std::string("external validation override is not supported in conformance: ") + variable);
                return;
            }
        }
        auto* variables = SDL_GetEnvironmentVariables(SDL_GetEnvironment());
        if (!variables) { issue(verdict, Status::EnvironmentFailure, "cannot inspect validation environment"); return; }
        std::string overrideName;
        for (auto** variable = variables; *variable; ++variable) {
            const std::string_view entry(*variable);
            if (entry.starts_with("VK_VALIDATION_") || entry.starts_with("VK_KHRONOS_VALIDATION_")) {
                overrideName = entry.substr(0, entry.find('='));
                break;
            }
        }
        SDL_free(variables);
        if (!overrideName.empty()) {
            issue(verdict, Status::EnvironmentFailure, "external validation override is not supported in conformance: " + overrideName);
            return;
        }
        const auto layerPath = input.value("layerPath", std::string{});
        if (!layerPath.empty()) {
            if (!std::filesystem::is_directory(std::filesystem::u8path(layerPath))) {
                issue(verdict, Status::EnvironmentFailure, "configured layer directory is missing"); return;
            }
            const auto emptyLayers = evidence.root() / "implicit-layers";
            std::filesystem::create_directory(emptyLayers);
            // SDL_setenv_unsafe affects only this isolated child, before loader initialization.
            if (SDL_setenv_unsafe("VK_LAYER_PATH", layerPath.c_str(), 1) != 0 ||
                SDL_setenv_unsafe("VK_IMPLICIT_LAYER_PATH", utf8(emptyLayers).c_str(), 1) != 0) {
                issue(verdict, Status::EnvironmentFailure, "cannot configure layer discovery"); return;
            }
        }
        if (!SDL_Init(SDL_INIT_VIDEO)) { issue(verdict, Status::EnvironmentFailure, SDL_GetError()); return; }
        sdl = true;
        const auto taskResult = task::initializeTaskSystem();
        if (!taskResult) { issue(verdict, Status::EnvironmentFailure, taskResult.error().message); return; }
        tasks = true;
        config.desc.validationSink = recorder.sink();
        auto created = render::createDevice(config.desc);
        if (!created) {
            // Device creation is a profile prerequisite, never an all-tests skip.
            issue(verdict, Status::EnvironmentFailure, std::string("createDevice: ") + render::resultToString(created));
            return;
        }
        device = std::move(*created);
        evidence.json("capabilities.json", describeDevice(*device, config));
        std::vector<render::QueueType> queues;
        for (const auto queue : {render::QueueType::Graphics, render::QueueType::Compute, render::QueueType::Copy}) {
            if (device->getQueue(queue)) { queues.push_back(queue); }
        }
        verdict = evaluate(selected.metadata.requirements, config, device->capabilities(), queues, activeValidation(*device));
        if (verdict.status != Status::Pass) { return; }
        if (selected.metadata.requirements.nativeDescriptorPointers && !nativeDescriptorPointersEnabled(*device)) {
            verdict = {Status::SkipUnsupported, "native descriptor pointers are not enabled"}; return;
        }
        for (auto type : selected.metadata.requirements.timestampQueues) {
            auto* queue = device->getQueue(type);
            if (!queue || !queue->timestampValidBits()) {
                verdict = {Status::SkipUnsupported, "required queue has no timestamp support"};
                return;
            }
        }
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        if (!graphics) { issue(verdict, Status::EnvironmentFailure, "no graphics queue"); return; }
        context = std::make_unique<RhiTestContext>(RhiTestContext{*device, *graphics, evidence.root(),
            activeValidation(*device) != Validation::Off, &recorder.messageCount, nullptr, &evidence, &config.desc});
    });
    if (recorder.failed()) { issue(verdict, Status::EnvironmentFailure, "validation reported a setup error/warning"); }
    if (verdict.status == Status::Pass) {
        initialized = true;
        step("run", [&] {
            verdict.executed = true;
            RhiTestResult result;
            if (context) { test->init(*context); result = test->run(*context); }
            else { result = test->runCpu(evidence); }
            if (result.skipped) { issue(verdict, Status::Fail, "unexpected skip after requirements passed: " + result.message); }
            else if (!result.passed) { issue(verdict, Status::Fail, result.message); }
        });
    }
    auto drain = [&] {
        if (device) {
            const auto result = device->waitIdle();
            if (!result) { issue(verdict, render::hasError(result, render::Error::DeviceLost) ? Status::DeviceLost : Status::Fail,
                std::string("GPU drain: ") + render::resultToString(result)); }
        }
    };
    step("completion", drain); // Parent watchdog bounds drivers without a timed waitIdle API.
    step("cleanup", [&] {
        if (initialized) {
            if (context) { test->cleanup(*context); }
            else { test->cleanupCpu(); }
        }
    });
    step("destroyTest", [&] { test.reset(); context.reset(); });
    step("finalDrain", drain);
    step("destroyDevice", [&] { device.reset(); });
    if (tasks) { task::shutdownTaskSystem(); }
    if (sdl) { SDL_Quit(); }
    if (recorder.failed()) { issue(verdict, Status::Fail, "validation error/warning or recorder overflow; see validation.json"); }
    evidence.json("validation.json", recorder.snapshot());
    evidence.phase("completed");
    return {{"schema", 1}, {"id", selected.id}, {"profile", config.id}, {"iteration", input.at("iteration")},
        {"status", name(verdict.status)}, {"message", verdict.message}, {"executed", verdict.executed},
        {"failed", failed(verdict.status)}, {"files", evidence.manifest()}};
}

Json childResult;
void writeReport(const std::filesystem::path& path, const Json& results);
class ChildAdapter : public ::testing::Test {
public:
    ChildAdapter(Case selected, Json input, std::filesystem::path directory)
        : selected_(std::move(selected)), input_(std::move(input)), directory_(std::move(directory)) {}
    void TestBody() override
    {
        Evidence evidence(directory_);
        childResult = execute(selected_, input_, evidence);
        if (childResult["status"] == "SkipUnsupported" || childResult["status"] == "SkipNotEnabled") {
            GTEST_SKIP() << childResult["message"].get<std::string>();
        }
        EXPECT_FALSE(childResult["failed"].get<bool>()) << childResult["message"].get<std::string>();
    }
private:
    Case selected_;
    Json input_;
    std::filesystem::path directory_;
};

int child(const Options& options)
{
    const auto input = readJson(options.input);
    if (input.at("schema") != 1) { throw std::runtime_error("unknown testbench input schema"); }
    const auto catalog = cases();
    const auto selected = std::find_if(catalog.begin(), catalog.end(), [&](const auto& value) { return value.id == input.at("id").get<std::string>(); });
    if (selected == catalog.end()) { throw std::runtime_error("case is not migrated to testbench"); }
    const auto id = selected->id;
    const auto dot = id.find('.');
    const auto directory = options.input.parent_path();
    std::string filter = "--gtest_filter=" + id;
    std::string repeat = "--gtest_repeat=1", color = "--gtest_color=no", program = "MetallicRhiTests";
    char* arguments[]{program.data(), filter.data(), repeat.data(), color.data()};
    int count = 4;
    ::testing::InitGoogleTest(&count, arguments);
    // GoogleTest's narrow fopen cannot represent arbitrary Windows Unicode paths.
    // Write its observed verdict through filesystem::path after RUN_ALL_TESTS.
    GTEST_FLAG_SET(output, "");
    ::testing::RegisterTest(id.substr(0, dot).c_str(), id.substr(dot + 1).c_str(), nullptr, nullptr,
        __FILE__, __LINE__, [value = *selected, input, directory]() -> ChildAdapter* { return new ChildAdapter(value, input, directory); });
    const int result = RUN_ALL_TESTS();
    if (childResult.is_null()) { throw std::runtime_error("child did not produce a verdict"); }
    if (result && !childResult["failed"].get<bool>()) {
        childResult["failed"] = true; childResult["status"] = "Fail"; childResult["message"] = "GoogleTest assertion failed";
    }
    writeReport(directory / "gtest.xml", Json::array({childResult}));
    childResult["files"] = Evidence(directory).manifest();
    writeJson(directory / "result.json", childResult);
    return childResult["failed"].get<bool>() ? 1 : 0;
}

std::string xmlEscape(const std::string& value)
{
    std::string output;
    for (char c : value) {
        switch (c) {
        case '&': output += "&amp;"; break;
        case '<': output += "&lt;"; break;
        case '>': output += "&gt;"; break;
        case '"': output += "&quot;"; break;
        case '\'': output += "&apos;"; break;
        default: if (static_cast<unsigned char>(c) >= 32 || c == '\n' || c == '\t') { output += c; } break;
        }
    }
    return output;
}

void writeReport(const std::filesystem::path& path, const Json& results)
{
    size_t failures = 0, skips = 0;
    for (const auto& result : results) {
        failures += result.at("failed").get<bool>() || result.value("policyFailure", false);
        skips += !result.at("failed").get<bool>() && !result.value("policyFailure", false) && result.at("status") != "Pass";
    }
    std::ofstream xml(path, std::ios::binary);
    xml.exceptions(std::ios::badbit | std::ios::failbit);
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?><testsuites><testsuite name=\"MetallicTestbench\" tests=\"" << results.size()
        << "\" failures=\"" << failures << "\" skipped=\"" << skips << "\">";
    for (const auto& result : results) {
        xml << "<testcase classname=\"" << xmlEscape(result.at("profile")) << "\" name=\"" << xmlEscape(result.at("id"))
            << "/" << result.at("iteration") << "\">";
        if (result.at("failed").get<bool>() || result.value("policyFailure", false)) {
            xml << "<failure message=\"" << xmlEscape(result.at("status")) << "\">" << xmlEscape(result.at("message")) << "</failure>";
        } else if (result.at("status") != "Pass") { xml << "<skipped message=\"" << xmlEscape(result.at("message")) << "\"/>"; }
        xml << "</testcase>";
    }
    xml << "</testsuite></testsuites>\n";
    xml.close();
}

int parent(const Options& options)
{
    const auto executable = executablePath();
    const auto executableHash = fileHash(executable);
    Json shaderInputs = Json::array();
    std::vector<std::filesystem::path> sourceFiles;
    for (const auto* relative : {"Shaders", "tests/rhi/shaders"}) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(std::filesystem::path(PROJECT_SOURCE_DIR) / relative)) {
            if (entry.is_regular_file()) { sourceFiles.push_back(entry.path()); }
        }
    }
    std::sort(sourceFiles.begin(), sourceFiles.end());
    uint64_t shaderHash = 14695981039346656037ull;
    for (const auto& path : sourceFiles) {
        shaderInputs.push_back({{"file", utf8(std::filesystem::relative(path, PROJECT_SOURCE_DIR))}, {"fnv1a64", fileHash(path)}});
    }
    if (std::filesystem::exists(executable.parent_path() / "slang.dll")) {
        shaderInputs.push_back({{"file", "slang.dll"}, {"fnv1a64", fileHash(executable.parent_path() / "slang.dll")}});
    }
    for (const unsigned char byte : shaderInputs.dump()) { shaderHash = (shaderHash ^ byte) * 1099511628211ull; }
    const auto shaderFingerprint = std::to_string(shaderHash);
    Json replay;
    if (!options.replay.empty()) {
        replay = readJson(options.replay / "input.json");
        if (replay.at("schema") != 1) { throw std::runtime_error("unknown replay schema"); }
        if ((replay.at("binaryHash") != executableHash || replay.value("shaderFingerprint", std::string{}) != shaderFingerprint) && !options.allowMismatch) {
            throw std::runtime_error("replay executable/shader inputs differ; use --tb-allow-version-mismatch explicitly");
        }
    }
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::absolute(options.output.empty() ?
        std::filesystem::path(".tmp/testbench") / std::to_string(stamp) : options.output);
    if (std::filesystem::exists(root) && !std::filesystem::is_empty(root)) { throw std::runtime_error("output directory must be empty"); }
    Evidence evidence(root);
    Json plan = Json::array();
    for (const auto& selected : cases()) {
        if (replay.is_null()) {
            if ((selected.metadata.suite != options.suite && !(options.suite == "sync" && selected.metadata.suite == "async")) || !matchesFilter(selected.id, options.filter)) { continue; }
        } else if (selected.id != replay.at("id").get<std::string>()) { continue; }
        const auto configId = replay.is_null() ? (options.profile.empty() ? selected.metadata.profile : options.profile) : replay.at("profile").get<std::string>();
        const auto validation = replay.is_null() ? options.validation : parseValidation(replay.at("validation").get<std::string>());
        if (!profile(configId, validation)) { throw std::runtime_error("invalid replay/profile"); }
        for (uint32_t iteration = 0; iteration < options.repeat; ++iteration) {
            plan.push_back({{"schema", 1}, {"id", selected.id}, {"profile", configId},
                {"validation", name(validation)},
                {"seed", replay.is_null() ? options.seed : replay.at("seed").get<uint64_t>()},
                {"iteration", iteration}, {"metadata", metadataJson(selected.metadata)},
                {"binaryHash", executableHash}, {"timeoutMs", selected.metadata.timeout.count()}});
            plan.back()["layerPath"] = replay.is_null() ? utf8(options.layerPath) : replay.value("layerPath", std::string{});
            plan.back()["shaderFingerprint"] = shaderFingerprint;
        }
    }
    if (plan.empty()) { throw std::runtime_error("no migrated cases selected"); }
    evidence.json("plan.json", plan);
    evidence.json("shader-inputs.json", shaderInputs);
    Json sourceState{{"configureRevision", METALLIC_TESTBENCH_REVISION}, {"dirty", "Unknown"}};
    if (const auto git = findExecutable("git.exe")) {
        const auto status = runProcess(*git, {"--no-optional-locks", "-C", PROJECT_SOURCE_DIR, "status", "--porcelain=v1"},
            root / "source-state", std::chrono::seconds(10));
        if (!status.timedOut && status.exitCode == 0) {
            std::ifstream file(root / "source-state/stdout.log", std::ios::binary);
            const std::string summary((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            sourceState["statusAtRun"] = summary;
            sourceState["dirty"] = !summary.empty();
        }
    }
    evidence.json("run.json", {{"schema", 1}, {"executable", utf8(executable)}, {"binaryHash", executableHash},
        {"hashAlgorithm", "fnv1a64"}, {"buildRevision", METALLIC_TESTBENCH_REVISION},
        {"compiler", METALLIC_TESTBENCH_COMPILER}, {"sourceDirectory", PROJECT_SOURCE_DIR},
        {"sourceState", sourceState}, {"shaderFingerprint", shaderFingerprint},
        {"requireAll", options.requireAll}, {"replayInput", replay}, {"mode", options.mode},
        {"workingDirectory", utf8(std::filesystem::current_path())}});
    if (options.mode == "--tb-plan") {
        std::cout << plan.dump(2) << '\n';
        return 0;
    }
    Json results = Json::array(), coverage = Json::array();
    bool failure = false;
    for (const auto& input : plan) {
        const std::string id = input.at("id"), configId = input.at("profile");
        const auto directory = root / configId / id / std::to_string(input.at("iteration").get<uint32_t>());
        Evidence sample(directory);
        sample.json("input.json", input);
        sample.phase("scheduled");
        Json result;
        try {
            const auto process = runProcess(executable, {"--tb-child", "--tb-input", utf8(directory / "input.json")},
                directory, std::chrono::milliseconds(input.at("timeoutMs").get<int64_t>()));
            result = verifyChild(directory, input, process);
        } catch (const std::exception& error) {
            result = {{"id", id}, {"profile", configId}, {"iteration", input.at("iteration")},
                {"status", "InfrastructureFailure"}, {"failed", true}, {"executed", false}, {"message", error.what()}};
        }
        const bool skipped = result.at("status") == "SkipUnsupported" || result.at("status") == "SkipNotEnabled";
        if (options.requireAll && skipped) { result["policyFailure"] = true; failure = true; }
        failure |= result.at("failed").get<bool>();
        sample.json("parent-result.json", result);
        std::cout << result.at("status").get<std::string>() << " " << id << " [" << configId << "] " << result.at("message").get<std::string>() << '\n';
        results.push_back(result);
        for (const auto& claim : input.at("metadata").at("coverage")) {
            coverage.push_back({{"claim", claim}, {"id", id}, {"profile", configId}, {"iteration", input.at("iteration")},
                {"executed", result.at("executed")}, {"status", result.at("status")}});
        }
    }
    evidence.json("results.json", {{"schema", 1}, {"failed", failure}, {"cases", results}});
    evidence.json("coverage.json", {{"schema", 1}, {"scope", "selected migrated cases only; iterations are not independent coverage"}, {"observations", coverage}});
    writeReport(root / "gtest.xml", results);
    std::cout << "Evidence: " << root.string() << '\n';
    return failure ? 1 : 0;
}

bool glob(std::string_view value, std::string_view pattern)
{
    size_t v = 0, p = 0, star = std::string_view::npos, restart = 0;
    while (v < value.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == value[v])) { ++v; ++p; }
        else if (p < pattern.size() && pattern[p] == '*') { star = p++; restart = v; }
        else if (star != std::string_view::npos) { p = star + 1; v = ++restart; }
        else { return false; }
    }
    while (p < pattern.size() && pattern[p] == '*') { ++p; }
    return p == pattern.size();
}

} // namespace

bool matchesFilter(const std::string& id, const std::string& filter)
{
    const auto dash = filter.find('-');
    auto any = [&](std::string patterns) {
        std::istringstream input(patterns);
        std::string pattern;
        while (std::getline(input, pattern, ':')) { if (glob(id, pattern)) { return true; } }
        return false;
    };
    return any(dash == 0 ? "*" : filter.substr(0, dash)) && (dash == std::string::npos || !any(filter.substr(dash + 1)));
}

Json verifyChild(const std::filesystem::path& directory, const Json& input, ProcessResult process)
{
    Json failure{{"id", input.at("id")}, {"profile", input.at("profile")}, {"iteration", input.at("iteration")},
        {"failed", true}, {"executed", false}, {"exitCode", process.exitCode}, {"status", "InfrastructureFailure"}, {"message", ""}};
    if (process.timedOut) { failure["status"] = "Timeout"; failure["message"] = "child exceeded wall-clock deadline"; return failure; }
    if (process.exitCode > 1) {
        failure["status"] = process.exitCode == 2 ? "InfrastructureFailure" : "Crash";
        failure["message"] = "child exited before finalizing evidence; see stderr.log";
        return failure;
    }
    try {
        auto result = readJson(directory / "result.json");
        if (result.at("schema") != 1 || result.at("id") != input.at("id") || result.at("profile") != input.at("profile") ||
            result.at("iteration") != input.at("iteration")) { throw std::runtime_error("child identity/schema mismatch"); }
        const auto status = result.at("status").get<std::string>();
        bool known = false, expectedFailure = false;
        for (auto value : {Status::Pass, Status::Fail, Status::SkipUnsupported, Status::SkipNotEnabled, Status::EnvironmentFailure,
            Status::InfrastructureFailure, Status::Timeout, Status::Crash, Status::DeviceLost}) {
            if (status == name(value)) { known = true; expectedFailure = failed(value); }
        }
        if (!known || result.at("failed").get<bool>() != expectedFailure || (process.exitCode != 0) != expectedFailure ||
            (status == "Pass" && !result.at("executed").get<bool>())) { throw std::runtime_error("contradictory child verdict/exit code"); }
        std::set<std::string> files;
        for (const auto& file : result.at("files")) {
            const std::filesystem::path relative(file.at("file").get<std::string>());
            if (relative.empty() || relative.has_parent_path() || relative.is_absolute()) { throw std::runtime_error("invalid artifact path"); }
            const auto path = directory / relative;
            if (!files.insert(relative.string()).second) { throw std::runtime_error("duplicate artifact entry"); }
            if (std::filesystem::file_size(path) != file.at("bytes").get<uint64_t>() || fileHash(path) != file.at("fnv1a64").get<std::string>()) {
                throw std::runtime_error("missing/truncated/changed evidence: " + relative.string());
            }
        }
        for (const auto* required : {"input.json", "profile.json", "validation.json", "journal.jsonl", "gtest.xml"}) {
            if (!files.contains(required)) { throw std::runtime_error(std::string("missing evidence manifest entry: ") + required); }
        }
        if (!std::filesystem::exists(directory / "gtest.xml")) { throw std::runtime_error("missing GoogleTest report"); }
        if (status == "Pass") {
            if (!files.contains("capabilities.json")) { throw std::runtime_error("missing actual capability evidence"); }
            for (const auto& required : input.at("metadata").at("artifacts")) {
                if (!files.contains(required.get<std::string>())) { throw std::runtime_error("missing oracle evidence"); }
            }
        }
        result["exitCode"] = process.exitCode;
        return result;
    } catch (const std::exception& error) { failure["message"] = error.what(); return failure; }
}

std::optional<int> runIfRequested(int argc, char** argv)
{
    bool requested = false;
    for (int i = 1; i < argc; ++i) { requested |= std::string_view(argv[i]).starts_with("--tb-"); }
    if (!requested) { return std::nullopt; }
    try {
        auto arguments = nativeArguments(argc, argv);
        std::vector<char*> pointers;
        for (auto& argument : arguments) { pointers.push_back(argument.data()); }
        const auto options = parse(int(pointers.size()), pointers.data());
        if (options.mode == "--tb-help") {
            std::cout << "Metallic M1/M2 testbench (Windows process isolation)\n"
                "  --tb-plan | --tb-run | --tb-replay <case-directory> | --tb-self-test\n"
                "  --tb-suite core|contract|binding|sync|async  --tb-profile core|binding|async\n"
                "  --tb-filter <GoogleTest-pattern>  --tb-repeat 1..1000  --tb-seed <uint64>\n"
                "  --tb-validation core|sync|off  --tb-require-all  --output-dir <empty-directory>\n"
                "  --tb-layer-path <explicit-layer-directory> (isolates implicit layers in child)\n"
                "  --tb-allow-version-mismatch (replay only; differences remain in evidence)\n";
            return 0;
        }
        if (options.mode == "--tb-self-test") {
            extern void registerHarnessTests();
            registerHarnessTests();
            std::string program = argv[0], filter = "--gtest_filter=TestbenchHarness.*";
            char* arguments[]{program.data(), filter.data()};
            int count = 2;
            ::testing::InitGoogleTest(&count, arguments);
            return RUN_ALL_TESTS();
        }
        if (options.mode == "--tb-child") { return child(options); }
        return parent(options);
    } catch (const std::exception& error) {
        std::cerr << "Testbench infrastructure: " << error.what() << '\n';
        return 2;
    }
}

} // namespace metallic::tests::bench
