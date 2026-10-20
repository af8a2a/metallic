#include "Runtime/Render/Core/PipelineCreationHistory.h"

#include <json.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace metallic::render::detail {
namespace {

using Json = nlohmann::json;
constexpr size_t kMaxHistoryBytes = 16 * 1024 * 1024;
constexpr size_t kMaxHistoryEntries = 65536;
constexpr char kHexDigits[] = "0123456789abcdef";

std::string identityString(const std::array<uint8_t, 32>& identity)
{
    std::string result;
    result.reserve(identity.size() * 2);
    for (uint8_t byte : identity) {
        result += kHexDigits[byte >> 4];
        result += kHexDigits[byte & 15];
    }
    return result;
}

std::string hashString(uint64_t hash)
{
    std::string result(16, '0');
    for (size_t index = result.size(); index != 0; --index) {
        result[index - 1] = kHexDigits[hash & 15];
        hash >>= 4;
    }
    return result;
}

struct TemporaryFile {
    std::filesystem::path path;
    ~TemporaryFile()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

PipelineCreationHistory::PipelineCreationHistory(
    std::filesystem::path path, std::array<uint8_t, 32> identity)
    : path_(std::move(path)), identity_(identity)
{
}

bool PipelineCreationHistory::load(std::string& reason)
{
    std::lock_guard ioLock(ioMutex_);
    reason.clear();
    {
        std::lock_guard lock(mutex_);
        durations_.clear();
        revision_ = 0;
    }
    std::error_code error;
    const bool exists = std::filesystem::exists(path_, error);
    if (error) {
        reason = "could not inspect pipeline creation history: " + error.message();
        return false;
    }
    if (!exists) { return true; }

    std::ifstream input(path_, std::ios::binary | std::ios::ate);
    const auto byteCount = input.tellg();
    if (!input || byteCount <= 0 || byteCount > static_cast<std::streamoff>(kMaxHistoryBytes)) {
        reason = "pipeline creation history is unreadable, empty or exceeds its size limit";
        return false;
    }
    std::string payload(static_cast<size_t>(byteCount), '\0');
    input.seekg(0);
    input.read(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        reason = "could not read complete pipeline creation history";
        return false;
    }

    try {
        // Bound nesting and reject duplicate members before the JSON DOM can
        // erase malformed input. This file is an external cache boundary.
        std::vector<std::unordered_set<std::string>> members;
        const auto validate = [&](int depth, Json::parse_event_t event, Json& value) {
            if (depth > 8) { throw std::runtime_error("excessive JSON nesting"); }
            if (event == Json::parse_event_t::object_start) { members.emplace_back(); }
            else if (event == Json::parse_event_t::object_end) { members.pop_back(); }
            else if (event == Json::parse_event_t::key &&
                !members.back().insert(value.get<std::string>()).second) {
                throw std::runtime_error("duplicate JSON member");
            }
            return true;
        };
        const Json document = Json::parse(payload, validate);
        if (!document.is_object() || document.size() != 3 ||
            !document.contains("version") || !document["version"].is_number_unsigned() ||
            document["version"].get<uint64_t>() != 1 ||
            !document.contains("identity") || !document["identity"].is_string() ||
            document["identity"].get<std::string>() != identityString(identity_) ||
            !document.contains("entries") || !document["entries"].is_array() ||
            document["entries"].size() > kMaxHistoryEntries) {
            reason = "pipeline creation history schema, version or device identity mismatch";
            return false;
        }
        std::unordered_map<uint64_t, uint64_t> loaded;
        loaded.reserve(document["entries"].size());
        for (const auto& entry : document["entries"]) {
            if (!entry.is_object() || entry.size() != 2 ||
                !entry.contains("psoHash") || !entry["psoHash"].is_string() ||
                !entry.contains("nativeNanoseconds") || !entry["nativeNanoseconds"].is_number_unsigned()) {
                reason = "invalid pipeline creation history entry";
                return false;
            }
            const auto hashText = entry["psoHash"].get<std::string>();
            uint64_t hash = 0;
            const auto parsed = std::from_chars(hashText.data(), hashText.data() + hashText.size(), hash, 16);
            const uint64_t nanoseconds = entry["nativeNanoseconds"].get<uint64_t>();
            if (hashText.size() != 16 || parsed.ec != std::errc{} ||
                parsed.ptr != hashText.data() + hashText.size() || nanoseconds == 0 ||
                !loaded.emplace(hash, nanoseconds).second) {
                reason = "invalid or duplicate pipeline creation history entry";
                return false;
            }
        }
        std::lock_guard lock(mutex_);
        durations_ = std::move(loaded);
        return true;
    } catch (const std::exception& exception) {
        reason = "could not parse pipeline creation history: " + std::string(exception.what());
        return false;
    }
}

std::optional<uint64_t> PipelineCreationHistory::duration(uint64_t psoHash) const
{
    std::lock_guard lock(mutex_);
    const auto found = durations_.find(psoHash);
    if (found == durations_.end()) { return std::nullopt; }
    return found->second;
}

uint64_t PipelineCreationHistory::observe(uint64_t psoHash, uint64_t nativeNanoseconds,
    bool feedbackValid, bool nativeCacheHit)
{
    if (!feedbackValid || nativeCacheHit || nativeNanoseconds == 0) { return 0; }
    std::lock_guard lock(mutex_);
    auto [entry, inserted] = durations_.try_emplace(psoHash, nativeNanoseconds);
    if (!inserted) {
        if (nativeNanoseconds <= entry->second) { return 0; }
        entry->second = nativeNanoseconds;
    }
    return ++revision_;
}

bool PipelineCreationHistory::save(std::string& reason)
{
    std::lock_guard ioLock(ioMutex_);
    reason.clear();
    std::vector<std::pair<uint64_t, uint64_t>> snapshot;
    {
        std::lock_guard lock(mutex_);
        snapshot.assign(durations_.begin(), durations_.end());
    }
    if (snapshot.size() > kMaxHistoryEntries) {
        reason = "pipeline creation history exceeds its entry limit";
        return false;
    }
    std::sort(snapshot.begin(), snapshot.end());
    Json entries = Json::array();
    for (const auto& [hash, nanoseconds] : snapshot) {
        entries.push_back({{"psoHash", hashString(hash)}, {"nativeNanoseconds", nanoseconds}});
    }
    const std::string payload = Json{{"version", 1}, {"identity", identityString(identity_)},
        {"entries", std::move(entries)}}.dump();
    if (payload.size() > kMaxHistoryBytes) {
        reason = "pipeline creation history exceeds its size limit";
        return false;
    }

    std::error_code error;
    if (!path_.parent_path().empty()) { std::filesystem::create_directories(path_.parent_path(), error); }
    if (error) {
        reason = "could not create pipeline creation history directory: " + error.message();
        return false;
    }
    static std::atomic<uint64_t> temporaryIndex{0};
#if defined(_WIN32)
    const auto processId = GetCurrentProcessId();
#else
    const auto processId = getpid();
#endif
    auto temporaryPath = path_;
    temporaryPath += ".tmp." + std::to_string(processId) + "." +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
        "." + std::to_string(temporaryIndex.fetch_add(1, std::memory_order_relaxed));
    const TemporaryFile temporary{std::move(temporaryPath)};
    {
        std::ofstream output(temporary.path, std::ios::binary | std::ios::trunc);
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        output.flush();
        output.close();
        if (!output) {
            reason = "could not write pipeline creation history temporary file";
            return false;
        }
    }
#if defined(_WIN32)
    if (!MoveFileExW(temporary.path.c_str(), path_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
    }
#else
    std::filesystem::rename(temporary.path, path_, error);
#endif
    if (error) {
        reason = "could not replace pipeline creation history: " + error.message();
        return false;
    }
    return true;
}

} // namespace metallic::render::detail
