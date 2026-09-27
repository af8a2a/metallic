#include "HtmlReport.h"
#include "HtmlReportTemplate.h"
#include <algorithm>
#include <array>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

namespace metallic::tests::bench {
namespace {
constexpr size_t kReadLimit = 16 * 1024 * 1024;
std::string utf8(const std::filesystem::path& path)
{
    const auto value = path.generic_u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::string url(const std::string& value)
{
    constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '/' || c == '-' || c == '_' || c == '.') { result += char(c); }
        else { result += '%'; result += hex[c >> 4]; result += hex[c & 15]; }
    }
    return result;
}
std::optional<std::filesystem::path> artifact(const std::filesystem::path& directory, const std::string& name)
{
    const auto relative = std::filesystem::u8path(name);
    if (relative.empty() || relative.has_parent_path() || relative.is_absolute() || name.find(':') != std::string::npos) { return {}; }
    const auto path = directory / relative;
    if (!std::filesystem::is_regular_file(path) || std::filesystem::is_symlink(path) ||
        std::filesystem::weakly_canonical(path).parent_path() != std::filesystem::weakly_canonical(directory)) { return {}; }
    return path;
}
std::vector<uint8_t> bytes(const std::filesystem::path& path, size_t limit = kReadLimit)
{
    const auto size = std::filesystem::file_size(path);
    if (size > limit) { throw std::runtime_error("preview omitted: file exceeds display budget"); }
    std::vector<uint8_t> value(size);
    std::ifstream input(path, std::ios::binary); input.exceptions(std::ios::badbit | std::ios::failbit);
    if (size) { input.read(reinterpret_cast<char*>(value.data()), std::streamsize(size)); }
    return value;
}
std::string base64(std::span<const uint8_t> input)
{
    constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result; result.reserve((input.size() + 2) / 3 * 4);
    for (size_t i = 0; i < input.size(); i += 3) {
        const uint32_t word = uint32_t(input[i]) << 16 | (i + 1 < input.size() ? uint32_t(input[i + 1]) << 8 : 0) |
            (i + 2 < input.size() ? input[i + 2] : 0);
        result += table[(word >> 18) & 63]; result += table[(word >> 12) & 63];
        result += i + 1 < input.size() ? table[(word >> 6) & 63] : '=';
        result += i + 2 < input.size() ? table[word & 63] : '=';
    }
    return result;
}
Json optionalJson(const std::filesystem::path& directory, const std::string& name, Json& warnings)
{
    try {
        if (const auto path = artifact(directory, name)) {
            if (std::filesystem::file_size(*path) > kReadLimit) { throw std::runtime_error("JSON exceeds display budget"); }
            return readJson(*path);
        }
    } catch (const std::exception& error) { warnings.push_back(name + ": " + error.what()); }
    return Json::object();
}
Json optionalObject(const std::filesystem::path& directory, const std::string& name, Json& warnings)
{
    auto value = optionalJson(directory, name, warnings);
    if (!value.is_object()) { warnings.push_back(name + ": expected a JSON object"); return Json::object(); }
    return value;
}
Json binaryComparison(const std::filesystem::path& actualPath, const std::filesystem::path& expectedPath)
{
    const auto actual = bytes(actualPath), expected = bytes(expectedPath);
    const auto size = std::max(actual.size(), expected.size());
    std::array<uint32_t, 128> bins{};
    size_t first = size, count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (i >= actual.size() || i >= expected.size() || actual[i] != expected[i]) {
            if (count == 0) { first = i; }
            ++count; ++bins[i * bins.size() / size];
        }
    }
    const size_t offset = count ? (first > 32 ? (first - 32) / 16 * 16 : 0) : 0;
    const auto sample = [offset](const auto& source) {
        Json value = Json::array();
        for (size_t i = offset; i < std::min(offset + 256, source.size()); ++i) { value.push_back(source[i]); }
        return value;
    };
    return {{"actual", utf8(actualPath.filename())}, {"expected", utf8(expectedPath.filename())},
        {"actualBytes", actual.size()}, {"expectedBytes", expected.size()}, {"differentBytes", count},
        {"firstMismatch", count ? Json(first) : Json(nullptr)}, {"bins", bins}, {"sampleOffset", offset},
        {"actualSample", sample(actual)}, {"expectedSample", sample(expected)}};
}
Json rgba(const std::filesystem::path& directory, const Json& description, const std::string& key)
{
    const auto path = artifact(directory, description.at(key).get<std::string>());
    if (!path) { throw std::runtime_error("invalid RGBA artifact path"); }
    const auto width = description.at("width").get<uint64_t>(), height = description.at("height").get<uint64_t>();
    if (!width || !height || width > 512 || height > 512) { throw std::runtime_error("RGBA preview dimensions exceed budget"); }
    const uint64_t row = description.value("rowPitch", width * 4), offset = description.value("offset", uint64_t(0));
    const auto source = bytes(*path);
    if (row < width * 4 || offset > source.size() || row > source.size() ||
        (height - 1) * row + width * 4 > source.size() - offset) { throw std::runtime_error("RGBA preview range exceeds artifact"); }
    std::vector<uint8_t> pixels(width * height * 4);
    for (uint64_t y = 0; y < height; ++y) {
        std::copy_n(source.begin() + offset + y * row, width * 4, pixels.begin() + y * width * 4);
    }
    return {{"width", width}, {"height", height}, {"rgba", base64(pixels)}};
}
std::string boundedText(const Json& value, size_t limit = 16000)
{
    auto text = value.dump(2, ' ', false, Json::error_handler_t::replace);
    if (text.size() > limit) { text.resize(limit); text += "\n[display truncated; open artifact for full data]"; }
    return text;
}
std::string timestamp()
{
    const auto time = std::time(nullptr); std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream result; result << std::put_time(&utc, "%Y-%m-%d %H:%M:%S UTC"); return result.str();
}
} // namespace

Json htmlCase(const std::filesystem::path& directory, const Json& input, const Json& result)
{
    Json entry{{"id", result.value("id", "Unknown")}, {"profile", result.value("profile", "Unknown")},
        {"iteration", result.value("iteration", 0)}, {"status", result.value("status", "InfrastructureFailure")},
        {"failed", result.value("failed", true)}, {"policyFailure", result.value("policyFailure", false)},
        {"executed", result.value("executed", false)}, {"durationMs", result.value("durationMs", Json(nullptr))},
        {"durationScope", result.value("durationScope", "isolated child wall time")},
        {"message", result.value("message", "").substr(0, 8000)}, {"warnings", Json::array()},
        {"artifacts", Json::array()}, {"images", Json::array()}, {"buffers", Json::array()}, {"details", Json::object()}};
    const auto metadata = input.value("metadata", Json::object());
    entry["suite"] = metadata.value("suite", input.value("mode", "testbench"));
    entry["layer"] = metadata.value("layer", "Unknown");
    entry["variant"] = input.value("variant", "");
    entry["seed"] = input.value("seed", Json(nullptr));
    entry["coverage"] = metadata.value("coverage", Json::array());
    auto& warnings = entry["warnings"];
    const auto device = optionalObject(directory, "capabilities.json", warnings);
    entry["device"] = device.value("device", metadata.value("requiresDevice", true) ? "Unavailable" : "CPU only");
    entry["driver"] = device.value("driverInfo", "");
    entry["validation"] = optionalObject(directory, "validation.json", warnings);
    if (entry["validation"].dump().size() > 32000) {
        entry["details"]["validation.json"] = boundedText(entry["validation"]);
        entry["validation"].erase("messages"); warnings.push_back("Validation text truncated for display; full artifact retained.");
    }
    entry["validationMode"] = device.value("validationMode", "Unknown");
    entry["details"]["input"] = boundedText(input);
    entry["details"]["result"] = boundedText(result);
    for (const auto* name : {"diff.json", "sequence-diff.json", "graph-trace-checks.json", "graph.json", "encoding.json", "capabilities.json"}) {
        const auto document = optionalJson(directory, name, warnings);
        if (!document.empty()) { entry["details"][name] = boundedText(document); }
    }
    const auto trace = optionalObject(directory, "trace.json", warnings);
    if (!trace.empty()) {
        entry["trace"] = {{"compiled", trace.value("compiled", false)}, {"captureFailed", trace.value("captureFailed", true)},
            {"events", trace.value("events", Json::array()).size()}, {"eventLimit", trace.value("eventLimit", 0)}};
        if (trace.contains("events") && trace.at("events").is_array()) {
            const auto& events = trace.at("events");
            entry["details"]["trace (first 16 events)"] = boundedText(Json(events.begin(), events.begin() + std::min(size_t(16), events.size())));
        }
    }
    std::vector<std::string> names;
    if (std::filesystem::exists(directory)) {
        for (const auto& file : std::filesystem::directory_iterator(directory)) {
            const auto name = utf8(file.path().filename());
            if (name.ends_with(".html") || name.ends_with(".tmp") || !artifact(directory, name)) { continue; }
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    size_t imageBytes = 0;
    for (const auto& name : names) {
        const auto path = artifact(directory, name).value();
        entry["artifacts"].push_back({{"name", name}, {"href", url(name)}, {"bytes", std::filesystem::file_size(path)}});
        try {
            if (name.ends_with(".png") && entry["images"].size() < 4 && imageBytes < 512 * 1024) {
                const auto data = bytes(path, 256 * 1024); imageBytes += data.size();
                entry["images"].push_back({{"label", name}, {"actual", {{"src", "data:image/png;base64," + base64(data)}}}});
            }
            std::string expected;
            if (name.ends_with("-actual.bin")) { expected = name.substr(0, name.size() - 11) + "-expected.bin"; }
            else if (name == "actual.bin" || name == "readback.bin") { expected = "expected.bin"; }
            if (!expected.empty() && entry["buffers"].size() < 4) {
                if (const auto reference = artifact(directory, expected)) { entry["buffers"].push_back(binaryComparison(path, *reference)); }
            }
            if (name == "stderr.log" || name == "stdout.log") {
                std::ifstream stream(path, std::ios::binary); std::string text(12000, '\0');
                stream.read(text.data(), std::streamsize(text.size())); text.resize(size_t(stream.gcount()));
                if (!text.empty()) { entry["details"][name] = text; }
            }
        } catch (const std::exception& error) { warnings.push_back(name + ": " + error.what()); }
    }
    const auto visuals = optionalJson(directory, "visuals.json", warnings);
    if (visuals.is_array()) {
        for (const auto& description : visuals) {
            if (entry["images"].size() >= 4) { break; }
            try {
                if (description.at("format") != "rgba8") { throw std::runtime_error("unsupported explicit preview format"); }
                auto pair = Json{{"label", description.value("label", "RGBA8")}, {"actual", rgba(directory, description, "actual")}};
                if (description.contains("expected")) { pair["expected"] = rgba(directory, description, "expected"); }
                entry["images"].push_back(pair);
            } catch (const std::exception& error) { warnings.push_back(std::string("visuals.json: ") + error.what()); }
        }
    }
    entry["kind"] = !entry["images"].empty() ? "texture" : !entry["buffers"].empty() ? "buffer" :
        result.value("profile", "") == "comparison" ? "comparison" : "diagnostic";
    return entry;
}
void writeHtmlReport(const std::filesystem::path& root, const Json& run, const Json& cases, bool complete)
{
    std::filesystem::create_directories(root);
    auto data = Json{{"schema", 1}, {"generatedUtc", timestamp()}, {"run", run}, {"complete", complete}, {"cases", cases}}
        .dump(-1, ' ', true, Json::error_handler_t::replace);
    // JSON inside a script element must not contain a literal HTML end tag.
    std::string safe;
    for (char c : data) {
        if (c == '<') { safe += "\\u003c"; }
        else if (c == '>') { safe += "\\u003e"; }
        else if (c == '&') { safe += "\\u0026"; }
        else { safe += c; }
    }
    std::string html(kHtmlReportTemplate);
    const auto position = html.find("__REPORT_DATA__");
    if (position == std::string::npos) { throw std::runtime_error("missing HTML report data marker"); }
    html.replace(position, std::string("__REPORT_DATA__").size(), safe);
    const auto temp = root / "report.html.tmp";
    std::ofstream output(temp, std::ios::binary | std::ios::trunc); output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(html.data(), std::streamsize(html.size())); output.close();
    std::filesystem::rename(temp, root / "report.html");
}
HtmlReport::HtmlReport(std::filesystem::path root, Json run) : root_(std::move(root)), run_(std::move(run))
{
    writeHtmlReport(root_, run_, cases_, false);
}
void HtmlReport::append(const std::filesystem::path& directory, const Json& input, const Json& result)
{
    auto entry = htmlCase(directory, input, result);
    writeHtmlReport(directory, {{"mode", "case"}}, Json::array({entry}), true);
    const auto prefix = url(utf8(directory.lexically_relative(root_))) + "/";
    entry["casePage"] = prefix + "report.html";
    for (auto& file : entry["artifacts"]) { file["href"] = prefix + file.at("href").get<std::string>(); }
    // Keep large repeated runs usable. The per-case report retains its previews.
    if (previewBytes_ + entry.dump().size() > 24 * 1024 * 1024) {
        entry["images"] = Json::array(); entry["buffers"] = Json::array(); entry["details"] = Json::object();
        entry["warnings"].push_back("Run preview budget reached; open the case report for details.");
    }
    previewBytes_ += entry.dump().size();
    cases_.push_back(std::move(entry));
    run_["durationMs"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_).count();
    writeHtmlReport(root_, run_, cases_, false);
}
void HtmlReport::finish()
{
    run_["durationMs"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_).count();
    writeHtmlReport(root_, run_, cases_, true); complete_ = true;
}
HtmlReport::~HtmlReport()
{
    if (!complete_) {
        try { run_["interrupted"] = true; writeHtmlReport(root_, run_, cases_, false); } catch (...) {}
    }
}
} // namespace metallic::tests::bench
