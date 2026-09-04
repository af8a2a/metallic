#include "Runtime/Debug/DebugTransport.h"

#include <filesystem>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace {
using namespace metallic::debug;

DebugValue readJson(const std::filesystem::path& path)
{
    if (std::filesystem::file_size(path) > (16u << 20)) { throw std::runtime_error("JSON file exceeds 16 MiB"); }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("Cannot open " + path.string()); }
    return decodeLossless(DebugValue::parse(input, [](int depth, DebugValue::parse_event_t, DebugValue&) {
        if (depth > 64) { throw std::runtime_error("JSON nesting exceeds 64"); }
        return true;
    }));
}

uint64_t parseUnsigned(std::string_view text, uint64_t maximum = UINT64_MAX)
{
    uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value > maximum) { throw std::runtime_error("Invalid unsigned argument"); }
    return value;
}

struct Options {
    uint32_t pid = 0;
    bool json = false;
    bool wait = false;
    uint32_t timeoutMs = 30000;
    std::string session;
    std::filesystem::path capture;
    DebugValue params = DebugValue::object();
    std::vector<std::string> words;
};

DebugValue invoke(Options& options, std::string method, DebugValue params = DebugValue::object())
{
    auto response = debugRequest(options.pid, {{"version", 1}, {"id", "cli"},
        {"method", std::move(method)}, {"params", std::move(params)}, {"session", options.session}}, options.timeoutMs);
    if (!response) { return debugErrorResponse("cli", response.error().code, response.error().message); }
    return *response;
}

DebugCapture loadCapture(const std::filesystem::path& directory)
{
    const auto manifest = readJson(directory / "manifest.json");
    if (manifest.at("version") != 1) { throw std::runtime_error("Unsupported capture version"); }
    DebugCapture capture;
    capture.snapshot.values = manifest.at("values");
    const auto& evidence = manifest.at("evidence");
    capture.snapshot.evidence = {
        .session = evidence.at("session"), .graph = evidence.at("graph"),
        .generation = evidence.at("generation"), .execution = evidence.at("execution"),
        .frameSlot = evidence.at("frameSlot"), .passId = evidence.at("passId"),
        .pass = evidence.at("pass"), .checkpoint = evidence.at("checkpoint"),
        .sample = evidence.at("sample"), .provenance = evidence.at("provenance")};
    uint64_t total = 0;
    if (manifest.at("artifacts").size() > 64) { throw std::runtime_error("Too many artifacts"); }
    for (size_t index = 0; index < manifest.at("artifacts").size(); ++index) {
        const auto& meta = manifest.at("artifacts")[index];
        // Filenames are generated locally, never trusted from the manifest.
        const auto path = directory / (std::to_string(index) + ".bin");
        const uint64_t size = std::filesystem::file_size(path);
        total += size;
        if (total > (128ull << 20) || size != meta.at("bytes").get<uint64_t>()) { throw std::runtime_error("Artifact size mismatch or capture exceeds 128 MiB"); }
        DebugArtifact artifact;
        artifact.metadata = meta;
        artifact.layout.name = meta.at("layout").at("name");
        artifact.metadata["layout"] = artifact.layout.name;
        artifact.metadata.erase("index"); artifact.metadata.erase("file"); artifact.metadata.erase("bytes");
        artifact.layout.stride = meta.at("layout").at("stride");
        for (const auto& field : meta.at("layout").at("fields")) {
            artifact.layout.fields.push_back({field.at("name"), field.at("type"), field.at("offset"), field.at("count"),
                field.value("bitOffset", 0u), field.value("bitWidth", 0u), field.value("scale", uint64_t(1)), field.value("enumNames", DebugValue::object())});
        }
        if (artifact.layout.layoutHash() != meta.at("layout").at("layoutHash").get<std::string>()) { throw std::runtime_error("Layout hash mismatch"); }
        artifact.bytes.resize(size);
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(artifact.bytes.data()), size)) { throw std::runtime_error("Could not read artifact"); }
        capture.artifacts.push_back(std::move(artifact));
    }
    return capture;
}

DebugValue run(Options& options)
{
    const auto& words = options.words;
    if (words.empty()) { throw std::runtime_error("Expected a command; use --help"); }
    const auto word = [&](size_t index) -> const std::string& {
        if (index >= words.size()) { throw std::runtime_error("Missing command argument"); }
        return words[index];
    };
    if (word(0) == "list") {
        DebugValue processes = DebugValue::array();
        for (uint32_t pid : debugProcesses()) {
            auto response = debugRequest(pid, {{"method", "hello"}}, 1000);
            if (response && response->value("status", "") == "ok") { processes.push_back((*response)["result"]); }
        }
        return {{"status", "ok"}, {"result", processes}};
    }
    if (!options.capture.empty()) {
        const auto capture = loadCapture(options.capture);
        if (word(0) == "stats") {
            const auto stats = capture.statistics();
            if (!stats) { return debugErrorResponse("cli", stats.error().code, stats.error().message); }
            return {{"status", "ok"}, {"result", *stats}};
        }
        if (word(0) == "schema") { return {{"status", "ok"}, {"result", capture.manifest()["artifacts"]}}; }
        if (word(0) != "eval" && word(0) != "object.get") { throw std::runtime_error("Offline mode supports eval, object.get, stats and schema"); }
        const auto root = capture.evaluationRoot();
        if (!root) { return debugErrorResponse("cli", root.error().code, root.error().message); }
        auto result = evaluate(word(1), *root);
        if (!result) { return debugErrorResponse("cli", result.error().code, result.error().message); }
        return {{"status", "ok"}, {"result", {{"value", paginateDebugValue(std::move(*result), options.params)}, {"evidence", capture.snapshot.evidence.value()}, {"source", "captured"}, {"coverage", root->at("coverage")}}}};
    }
    if (!options.session.empty() && !options.pid) { throw std::runtime_error("--session requires --pid"); }
    if (options.session.empty()) {
        if (!options.pid) {
            const auto processes = debugProcesses();
            if (processes.size() != 1) { throw std::runtime_error("Specify --pid (use metallicctl list)"); }
            options.pid = processes.front();
        }
        auto hello = debugRequest(options.pid, {{"method", "hello"}}, options.timeoutMs);
        if (!hello) { return debugErrorResponse("cli", hello.error().code, hello.error().message); }
        if ((*hello)["status"] != "ok") { return *hello; }
        options.session = (*hello)["result"]["session"];
    }
    if (word(0) == "capture" && word(1) == "export") {
        auto response = invoke(options, "jobs.get", {{"job", word(2)}});
        if (response["status"] != "ok") { return response; }
        if (response["result"]["state"] != "Ready") { throw std::runtime_error("Capture is not ready"); }
        std::string manifestBytes;
        uint64_t manifestSize = 0;
        do {
            auto chunk = invoke(options, "artifact.read", {{"job", word(2)}, {"manifest", true}, {"offset", manifestBytes.size()}});
            if (chunk["status"] != "ok") { return chunk; }
            manifestSize = chunk["result"]["total"].get<uint64_t>();
            if (manifestSize > (16ull << 20)) { throw std::runtime_error("Manifest exceeds 16 MiB export limit"); }
            auto bytes = hexDecode(chunk["result"]["hex"].get<std::string>());
            if (!bytes || bytes->empty() || bytes->size() > manifestSize - manifestBytes.size()) { throw std::runtime_error("Invalid manifest chunk"); }
            manifestBytes.append(reinterpret_cast<const char*>(bytes->data()), bytes->size());
        } while (manifestBytes.size() < manifestSize);
        const auto manifest = decodeLossless(DebugValue::parse(manifestBytes));
        const std::filesystem::path directory = options.params.at("out").get<std::string>();
        if (!std::filesystem::create_directory(directory)) { throw std::runtime_error("Output directory already exists"); }
        for (size_t index = 0; index < manifest["artifacts"].size(); ++index) {
            const uint64_t total = manifest["artifacts"][index]["bytes"].get<uint64_t>();
            std::ofstream output(directory / (std::to_string(index) + ".bin"), std::ios::binary);
            for (uint64_t offset = 0; offset < total;) {
                auto chunk = invoke(options, "artifact.read", {{"job", word(2)}, {"index", index}, {"offset", offset}});
                if (chunk["status"] != "ok") { return chunk; }
                auto bytes = hexDecode(chunk["result"]["hex"].get<std::string>());
                if (!bytes || bytes->empty() || bytes->size() > total - offset) { throw std::runtime_error("Invalid artifact chunk"); }
                output.write(reinterpret_cast<const char*>(bytes->data()), bytes->size());
                if (!output) { throw std::runtime_error("Could not write artifact"); }
                offset += bytes->size();
            }
        }
        // Manifest is written last, so an interrupted export is not a valid capture.
        std::ofstream output(directory / "manifest.json", std::ios::binary);
        output << encodeLossless(manifest).dump(2);
        if (!output) { throw std::runtime_error("Could not write manifest"); }
        return {{"status", "ok"}, {"result", {{"directory", std::filesystem::absolute(directory).string()}}}};
    }
    std::string method = word(0);
    DebugValue params = options.params;
    if (method == "schema") { if (words.size() > 1) { params["name"] = word(1); } }
    else if (method == "eval") { params["expression"] = word(1); }
    else if (method == "object.get") { params["path"] = word(1); }
    else if (method == "frame") { method = "frame.latest"; }
    else if (method == "rg") {
        method = word(1) == "trace" ? "rg.trace" : "rg.describe";
        if (method == "rg.trace") { params["resource"] = word(2); }
    } else if (method == "jobs") {
        method = "jobs." + word(1); params["job"] = word(2);
    } else if (method == "capture" && word(1) == "batch") {
        method = "capture.batch"; params = readJson(params.at("spec").get<std::string>());
    } else if (method == "inspect") {
        if (word(1) != "buffer" && word(1) != "texture") { throw std::runtime_error("Expected inspect buffer or texture"); }
        method = "capture.batch";
        DebugValue resource{{"id", word(2)}};
        for (const char* key : {"offset", "count", "layout", "layoutHash", "roi"}) {
            if (params.contains(key)) { resource[key] = params[key]; params.erase(key); }
        }
        params["resources"] = DebugValue::array({resource});
        if (!params.contains("pass")) {
            const auto dot = word(2).find('.');
            if (dot == std::string::npos) { throw std::runtime_error("inspect requires --pass for private resources"); }
            if (word(2).starts_with("gpuScene.") || word(2).starts_with("streaming.")) {
                const auto end = word(2).find('.', dot + 1);
                if (end == std::string::npos) { throw std::runtime_error("Private resource requires --pass"); }
                params["pass"] = word(2).substr(dot + 1, end - dot - 1);
            } else { params["pass"] = word(2).substr(0, dot); }
        }
    } else if (method == "call") {
        method = word(1);
        if (words.size() > 2) { params = decodeLossless(DebugValue::parse(word(2))); }
    }
    auto response = invoke(options, method, params);
    if (method == "capture.batch" && options.wait && response["status"] == "ok") {
        const auto waitForJob = [&](const DebugValue& job) -> DebugValue {
            DebugValue response;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(options.timeoutMs);
            do {
                response = invoke(options, "jobs.get", {{"job", job}, {"stats", options.params.value("stats", false)}});
                if (response["status"] != "ok") { break; }
                const auto state = response["result"].value("state", "");
                if (state == "Ready") { break; }
                if (state == "Failed" || state == "Cancelled") {
                    response["status"] = "error";
                    response["error"] = response["result"].value("error", DebugValue{{"code", state}, {"message", state}});
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } while (std::chrono::steady_clock::now() < deadline);
            if (response["status"] == "ok" && response["result"].value("state", "") != "Ready") {
                invoke(options, "jobs.cancel", {{"job", job}});
                return debugErrorResponse("cli", "Timeout", "Capture timed out; job " + job.get<std::string>());
            }
            return response;
        };
        if (response["result"].contains("jobs")) {
            DebugValue results = DebugValue::array();
            bool success = true;
            for (const auto& item : response["result"]["jobs"]) {
                auto completed = waitForJob(item.at("job"));
                success = success && completed.value("status", "") == "ok";
                results.push_back(std::move(completed));
            }
            return {{"status", success ? "ok" : "error"}, {"result", {{"jobs", std::move(results)}, {"sameExecution", true}}}};
        }
        return waitForJob(response["result"]["job"]);
    }
    return response;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto next = [&]() -> std::string {
                if (++i >= argc) { throw std::runtime_error("Missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--help" || arg == "-h") {
                std::cout << "metallicctl [--pid PID] [--session SESSION] [--json] <command>\n"
                    "  list | hello | schema [provider] | frame latest | rg graph\n"
                    "  rg trace RESOURCE [--direction backward|forward]\n"
                    "  eval EXPR [--job ID] [--frame N] | object.get PATH [--offset N --count N]\n"
                    "  capture batch --spec FILE [--wait] | capture export JOB --out NEW_DIRECTORY\n"
                    "  inspect buffer ID --pass PASS --checkpoint POINT --count N [--offset N --layout TYPE]\n"
                    "  inspect texture ID --pass PASS --roi X,Y,W,H [--stats --wait]\n"
                    "  jobs get|cancel ID | call METHOD JSON | repl\n"
                    "  --capture DIRECTORY eval EXPR | stats (offline)\n";
                return 0;
            } else if (arg == "--json") { options.json = true; }
            else if (arg == "--wait") { options.wait = true; }
            else if (arg == "--stats") { options.params["stats"] = true; }
            else if (arg == "--recorded") { options.params["recorded"] = true; }
            else if (arg == "--pid") { options.pid = static_cast<uint32_t>(parseUnsigned(next(), UINT32_MAX)); }
            else if (arg == "--session") { options.session = next(); }
            else if (arg == "--timeout-ms") { options.timeoutMs = static_cast<uint32_t>(parseUnsigned(next(), 300000)); }
            else if (arg == "--capture") { options.capture = next(); }
            else if (arg == "--count" || arg == "--offset" || arg == "--frame" || arg == "--generation") { options.params[arg.substr(2)] = parseUnsigned(next()); }
            else if (arg == "--roi" || arg == "--pixel") {
                auto value = next(); std::replace(value.begin(), value.end(), ',', ' ');
                std::istringstream stream(value); uint32_t x = 0, y = 0, w = 1, h = 1;
                if (!(stream >> x >> y) || (arg == "--roi" && !(stream >> w >> h))) { throw std::runtime_error("Invalid ROI/pixel"); }
                options.params["roi"] = {{"x", x}, {"y", y}, {"width", w}, {"height", h}};
            } else if (arg.starts_with("--")) { options.params[arg.substr(2)] = next(); }
            else { options.words.push_back(arg); }
        }
        if (!options.words.empty() && options.words[0] == "repl") {
            std::string line;
            while (std::cerr << "metallic> " && std::getline(std::cin, line)) {
                if (line == "exit" || line == "quit") { break; }
                std::istringstream input(line); std::string command;
                if (!(input >> command)) { continue; }
                options.words = {command};
                if (command == "eval" || command == "object.get") {
                    std::string expression; std::getline(input >> std::ws, expression); options.words.push_back(expression);
                } else { while (input >> command) { options.words.push_back(command); } }
                try { std::cout << encodeLossless(run(options)).dump(options.json ? -1 : 2) << '\n'; }
                catch (const DebugError& error) { std::cout << debugErrorResponse("cli", error.code, error.message).dump() << '\n'; }
                catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
            }
            return 0;
        }
        const auto response = run(options);
        std::cout << encodeLossless(response).dump(options.json ? -1 : 2) << '\n';
        return response.value("status", "error") == "ok" ? 0 : 1;
    } catch (const DebugError& error) {
        std::cout << debugErrorResponse("cli", error.code, error.message).dump() << '\n';
        return 1;
    } catch (const std::exception& error) {
        if (options.json) { std::cout << debugErrorResponse("cli", "InvalidArgument", error.what()).dump() << '\n'; }
        else { std::cerr << error.what() << '\n'; }
        return 1;
    }
}
