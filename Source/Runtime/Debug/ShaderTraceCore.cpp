#include "ShaderTraceCore.h"
#include "DebugHash.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <map>
#include <set>

namespace metallic::debug {
namespace {
DebugError invalid(std::string message) { return {"InvalidShaderTrace", std::move(message)}; }
uint64_t number(const DebugValue& v, uint64_t max = UINT64_MAX) { return debugUnsigned(v, max); }
uint64_t join(uint32_t low, uint32_t high) { return uint64_t(low) | (uint64_t(high) << 32); }

void knownKeys(const DebugValue& value, std::initializer_list<std::string_view> allowed)
{
    if (!value.is_object()) { throw std::invalid_argument("Expected an object"); }
    for (auto item = value.begin(); item != value.end(); ++item) {
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end()) {
            throw std::invalid_argument("Unsupported request field: " + item.key());
        }
    }
}

std::vector<uint32_t> words(std::string_view text)
{
    std::vector<uint32_t> result;
    while (!text.empty()) {
        if (text.front() == ' ' || text.front() == '\r' || text.front() == '\n') { text.remove_prefix(1); continue; }
        uint32_t n = 0;
        auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
        if (error != std::errc{} || end == text.data() || (end != text.data() + text.size() && *end != ' ' && *end != '\r' && *end != '\n')) {
            throw std::invalid_argument("Invalid protocol word");
        }
        result.push_back(n);
        if (result.size() > 30) { throw std::invalid_argument("Protocol word budget exceeded"); }
        text.remove_prefix(end - text.data());
    }
    return result;
}

DebugValue fields(const DebugValue& schema, const std::vector<uint32_t>& w)
{
    DebugValue result = DebugValue::object();
    size_t i = 14;
    for (const auto& field : schema) {
        const std::string type = field.at("type"), name = field.at("name");
        if (i >= w.size()) { throw std::invalid_argument("Missing payload word"); }
        const uint32_t bits = w[i++];
        if (type == "f32") {
            const float value = std::bit_cast<float>(bits);
            result[name] = {{"type", type}, {"bits", bits}, {"value", std::isfinite(value) ? DebugValue(value) : DebugValue(nullptr)},
                {"classification", std::isnan(value) ? "NaN" : std::isinf(value) ? "Infinity" : bits == 0x80000000 ? "NegativeZero" : "Finite"}};
        } else if (type == "u64") {
            if (i >= w.size()) { throw std::invalid_argument("Missing u64 high word"); }
            result[name] = {{"type", type}, {"low", bits}, {"high", w[i]}, {"value", join(bits, w[i])}}; ++i;
        } else if (type == "u32" || type == "i32") {
            result[name] = {{"type", type}, {"bits", bits},
                {"value", type == "u32" ? DebugValue(bits) : DebugValue(std::bit_cast<int32_t>(bits))}};
        } else { throw std::invalid_argument("Unknown field type"); }
    }
    if (i != w.size()) { throw std::invalid_argument("Unexpected payload words"); }
    return result;
}
} // namespace

DebugValue shaderTraceSite(DebugValue description)
{
    description.erase("schemaHash");
    description["schemaHash"] = debugSha256(description.dump());
    return description;
}

DebugResult<void> validateShaderWatch(const DebugValue& request, const DebugValue& site, uint64_t generation)
{
    try {
        knownKeys(request, {"version", "generation", "target", "invocation", "limits", "fixtureScenario", "predicate", "fields"});
        knownKeys(request.at("target"), {"site", "expectedSiteSchemaHash"});
        knownKeys(request.at("invocation"), {"group", "localIndex"});
        knownKeys(request.at("limits"), {"targetFrames", "maxRecords", "timeoutMs"});
        if (request.at("version") != 1 || number(request.at("generation")) != generation) {
            return std::unexpected(DebugError{"StaleHandle", "Watch version/generation is not current"});
        }
        if (request.at("target").at("site") != site.at("name") ||
            request.at("target").at("expectedSiteSchemaHash") != site.at("schemaHash")) {
            return std::unexpected(DebugError{"StaleHandle", "Site or schema changed"});
        }
        const auto& invocation = request.at("invocation");
        if (invocation.at("group") != site.at("invocation").at("group") || invocation.at("localIndex") != site.at("invocation").at("localIndex")) {
            return std::unexpected(DebugError{"Unsupported", "This site adapter supports only its declared invocation"});
        }
        const auto& limits = request.at("limits");
        if (number(limits.at("targetFrames")) != 1 || number(limits.at("maxRecords"), 16) < 2 ||
            number(limits.at("timeoutMs"), 30000) == 0) { return std::unexpected(invalid("Invalid limits")); }
        if (request.contains("predicate") || request.contains("fields")) {
            return std::unexpected(DebugError{"Unsupported", "P1 transports the complete declared schema; predicates/field selection require a site adapter"});
        }
        const std::string scenario = request.value("fixtureScenario", "matched");
        const std::set<std::string> scenarios{"matched", "no-match", "site-not-reached", "missing-end", "quota"};
        if ((request.contains("fixtureScenario") && !site.value("fixture", false)) || !scenarios.contains(scenario)) {
            return std::unexpected(invalid("Unsupported fixture scenario"));
        }
        return {};
    } catch (const std::exception& e) { return std::unexpected(invalid(e.what())); }
}

ShaderTraceCore::ShaderTraceCore(std::string session, uint64_t firstToken) : session_(std::move(session)), nextToken_(firstToken)
{
    const auto hash = debugSha256(session_);
    std::from_chars(hash.data(), hash.data() + 16, sessionToken_, 16);
}

DebugResult<DebugValue> ShaderTraceCore::begin(DebugValue request, DebugValue site, DebugValue identity)
{
    if (active_) { return std::unexpected(DebugError{"Busy", "A shader observation still owns its submission"}); }
    if (!nextToken_ || nextToken_ == UINT64_MAX) { return std::unexpected(DebugError{"TokenExhausted", "Start a new session; tokens cannot wrap"}); }
    auto valid = validateShaderWatch(request, site, number(identity.at("generation")));
    if (!valid) { return std::unexpected(valid.error()); }
    identity["session"] = session_;
    identity["sessionToken"] = sessionToken_;
    identity["runToken"] = nextToken_;
    identity["dispatchToken"] = nextToken_++;
    identity["queue"] = nullptr; identity["submit"] = nullptr;
    active_ = DebugValue{{"version", 1}, {"parserVersion", kShaderTraceParserVersion}, {"request", std::move(request)},
        {"site", std::move(site)}, {"dispatch", std::move(identity)}, {"rawMessages", DebugValue::array()},
        {"health", {{"submitted", false}, {"gpuComplete", false}, {"backendClosed", false}, {"readbackValid", false},
            {"hostDropped", 0u}, {"hostTruncated", 0u}, {"stopReason", ""}}}};
    return active_->at("dispatch");
}

void ShaderTraceCore::compiledVariant(DebugValue variant)
{
    if (!active_ || (*active_)["health"]["submitted"] == true) { throw std::logic_error("Variant identity is frozen at submission"); }
    variant["fingerprint"] = debugSha256(variant.dump());
    (*active_)["dispatch"]["variant"] = std::move(variant);
}

void ShaderTraceCore::submitted(DebugValue queue, DebugValue submit)
{
    if (!active_ || (*active_)["health"]["submitted"] == true) { throw std::logic_error("Invalid shader submission transition"); }
    (*active_)["dispatch"]["queue"] = std::move(queue);
    (*active_)["dispatch"]["submit"] = std::move(submit);
    (*active_)["health"]["submitted"] = true;
}

void ShaderTraceCore::ingest(DebugValue raw)
{
    if (!active_) { throw std::logic_error("No observation owns these raw messages"); }
    auto& list = (*active_)["rawMessages"];
    if (list.size() >= 256) { accountLoss(1, 0); return; }
    // Bound persisted data independently of backend behavior.
    for (const char* key : {"text", "idName"}) {
        auto text = raw.at(key).get<std::string>();
        const size_t bound = std::string_view(key) == "text" ? 4095 : 159;
        if (text.size() > bound) { text.resize(bound); raw[key] = text; raw["truncated"] = true; }
    }
    list.push_back(std::move(raw));
}

void ShaderTraceCore::accountLoss(uint64_t dropped, uint64_t truncated)
{
    if (!active_) { throw std::logic_error("No observation"); }
    auto& h = (*active_)["health"];
    h["hostDropped"] = number(h["hostDropped"]) + dropped;
    h["hostTruncated"] = number(h["hostTruncated"]) + truncated;
}

void ShaderTraceCore::stop(std::string reason)
{
    if (active_ && (*active_)["health"]["stopReason"] == "") { (*active_)["health"]["stopReason"] = std::move(reason); }
}

void ShaderTraceCore::completion(bool gpuComplete, bool backendClosed, bool readbackValid)
{
    if (!active_) { throw std::logic_error("No observation"); }
    auto& h = (*active_)["health"];
    h["gpuComplete"] = gpuComplete; h["backendClosed"] = backendClosed; h["readbackValid"] = readbackValid;
}

DebugResult<DebugValue> ShaderTraceCore::seal()
{
    if (!active_) { return std::unexpected(invalid("No observation")); }
    if ((*active_)["health"]["submitted"] == true && (*active_)["health"]["backendClosed"] != true) {
        return std::unexpected(DebugError{"NotReady", "Submitted resources must survive through the backend collection boundary"});
    }
    DebugValue result = std::move(*active_); active_.reset();
    return result;
}

DebugResult<DebugValue> analyzeShaderTrace(const DebugValue& bundle)
{
    try {
        if (bundle.at("version") != 1 || bundle.at("parserVersion") != kShaderTraceParserVersion || bundle.dump().size() > kShaderTraceArtifactBudget) {
            return std::unexpected(invalid("Unsupported or oversized shader trace"));
        }
        const auto& site = bundle.at("site");
        if (shaderTraceSite(site).at("schemaHash") != site.at("schemaHash")) { return std::unexpected(invalid("Site schema hash mismatch")); }
        std::set<std::string> names;
        uint32_t payloadWords = 0;
        for (const auto& field : site.at("fields")) {
            const std::string type = field.at("type"), name = field.at("name");
            if (name.empty() || name.size() > 128 || !names.insert(name).second ||
                (type != "f32" && type != "u32" && type != "i32" && type != "u64")) { return std::unexpected(invalid("Invalid field schema")); }
            payloadWords += type == "u64" ? 2 : 1;
        }
        if (payloadWords > 16) { return std::unexpected(invalid("Payload budget exceeded")); }
        const auto& identity = bundle.at("dispatch");
        const auto valid = validateShaderWatch(bundle.at("request"), site, number(identity.at("generation")));
        if (!valid) { return std::unexpected(valid.error()); }
        const auto& health = bundle.at("health");
        const uint32_t maxRecords = uint32_t(number(bundle.at("request").at("limits").at("maxRecords"), 16));
        DebugValue records = DebugValue::array(), orphans = DebugValue::array(), errors = DebugValue::array();
        std::map<uint32_t, DebugValue> ordered;
        uint64_t decodeErrors = 0, duplicates = 0, truncated = 0;
        bool gpuOverflow = false, backendError = false;
        const auto& raw = bundle.at("rawMessages");
        if (!raw.is_array() || raw.size() > 256) { return std::unexpected(invalid("Raw queue budget exceeded")); }
        for (size_t arrival = 0; arrival < raw.size(); ++arrival) {
            const auto& message = raw[arrival];
            const std::string text = message.at("text"), idName = message.at("idName");
            if (text.size() > 4095 || idName.size() > 159) { return std::unexpected(invalid("Raw slot budget exceeded")); }
            if (message.at("truncated").get<bool>()) { ++truncated; }
            backendError |= (number(message.at("severity")) & (256 | 4096)) != 0;
            if (message.at("id") != 0x4fe1fef9 || message.at("severity") != 16) { continue; }
            if (text.find("[WARNING]") != std::string::npos) { gpuOverflow = true; continue; }
            size_t start = text.starts_with("MTS1 ") ? 0 : text.find("\nMTS1 ");
            if (start == std::string::npos) { ++decodeErrors; errors.push_back({{"arrival", arrival}, {"reason", "Unknown Printf protocol"}}); continue; }
            if (start) { ++start; }
            try {
                const auto w = words(std::string_view(text).substr(start + 5));
                if (w.size() < 14 || w[13] > 16 || w.size() != 14 + w[13]) { throw std::invalid_argument("Record word count mismatch"); }
                DebugValue event{{"arrival", arrival}, {"sessionToken", join(w[0],w[1])}, {"runToken", join(w[2],w[3])},
                    {"dispatchToken", join(w[4],w[5])}, {"siteId", w[6]}, {"kind", w[7]},
                    {"group", {w[8],w[9],w[10]}}, {"localIndex", w[11]}, {"seq", w[12]}};
                if (event["sessionToken"] != identity.at("sessionToken") || event["runToken"] != identity.at("runToken") ||
                    event["dispatchToken"] != identity.at("dispatchToken")) { orphans.push_back(event); continue; }
                if (event["siteId"] != site.at("id") || event["group"] != bundle.at("request").at("invocation").at("group") ||
                    event["localIndex"] != bundle.at("request").at("invocation").at("localIndex") || w[12] >= maxRecords || w[7] > 2) {
                    throw std::invalid_argument("Scope/site/sequence mismatch");
                }
                if (w[7] == 0 && (w[12] != 0 || w[13] != 0)) { throw std::invalid_argument("Invalid BEGIN"); }
                if (w[7] == 1) { event["fields"] = fields(site.at("fields"), w); }
                if (w[7] == 2) {
                    if (w[13] != 5 || w[18] > 1) { throw std::invalid_argument("Invalid END"); }
                    event["summary"] = {{"siteEvaluationCount",w[14]}, {"matchedCount",w[15]}, {"emittedCount",w[16]},
                        {"exitReason",w[17]}, {"budgetExceeded",w[18] == 1}};
                }
                if (!ordered.emplace(w[12], event).second) { ++duplicates; }
            } catch (const std::exception& e) { ++decodeErrors; errors.push_back({{"arrival",arrival}, {"reason",e.what()}}); }
        }
        truncated = std::max(truncated, number(health.at("hostTruncated")));
        bool sequenceComplete = ordered.size() >= 2;
        DebugValue end = DebugValue::object();
        uint32_t dataCount = 0, next = 0;
        for (auto& [seq, record] : ordered) {
            sequenceComplete &= seq == next++;
            const auto kind = record.at("kind").get<uint32_t>();
            sequenceComplete &= seq == 0 ? kind == 0 : seq == ordered.rbegin()->first ? kind == 2 : kind == 1;
            if (kind == 1) { ++dataCount; records.push_back(record); }
            if (kind == 2) { end = record.at("summary"); }
        }
        bool quota = false;
        if (!end.empty()) {
            quota = end.at("budgetExceeded");
            sequenceComplete &= number(end.at("emittedCount")) == dataCount && number(end.at("matchedCount")) >= dataCount &&
                number(end.at("siteEvaluationCount")) >= number(end.at("matchedCount")) &&
                (quota || number(end.at("matchedCount")) == dataCount);
        } else { sequenceComplete = false; }
        const std::string stop = health.at("stopReason");
        const bool submitted = health.at("submitted"), gpu = health.at("gpuComplete"), closed = health.at("backendClosed"), readback = health.at("readbackValid");
        const bool complete = sequenceComplete && submitted && gpu && closed && readback && stop.empty() && !backendError && !gpuOverflow &&
            !quota && !number(health.at("hostDropped")) && !truncated && !decodeErrors && !duplicates && orphans.empty();
        std::string outcome = "Incomplete";
        if (!stop.empty()) { outcome = stop; }
        else if (!submitted) { outcome = "TargetNotExecuted"; }
        else if (complete) { outcome = number(end.at("siteEvaluationCount")) == 0 ? "SiteNotReached" : dataCount == 0 ? "NoMatch" : "Matched"; }
        return DebugValue{{"parserVersion",kShaderTraceParserVersion}, {"dispatch",identity}, {"site",site}, {"records",records},
            {"orphans",orphans}, {"decodeErrors",errors}, {"summary",end}, {"outcome",outcome}, {"selectedScopeComplete",complete},
            {"instrumentationStatus","Printf"}, {"targetExecutionStatus",submitted ? (gpu ? "Completed" : "Unknown") : "NotSubmitted"},
            {"readbackStatus",readback ? "Verified" : "Unknown"}, {"backendCollectionStatus",closed ? "Closed" : "Unknown"},
            {"receivedRecordCount",ordered.size()}, {"hostDroppedRecordCount",health.at("hostDropped")}, {"hostTruncatedCount",truncated},
            {"decodeErrorCount",decodeErrors}, {"duplicateCount",duplicates}, {"orphanCount",orphans.size()},
            {"gpuOverflowDetected",gpuOverflow}, {"gpuDroppedRecordCount",nullptr}, {"performanceEligible",false}};
    } catch (const std::exception& e) { return std::unexpected(invalid(e.what())); }
}

DebugResult<DebugValue> decodeShaderTraceArtifact(std::span<const uint8_t> bytes, std::string_view sha256)
{
    if (bytes.size() > kShaderTraceArtifactBudget || debugSha256(bytes) != sha256) { return std::unexpected(invalid("Artifact size/hash mismatch")); }
    try {
        const auto bundle = decodeLossless(DebugValue::parse(bytes.begin(), bytes.end(), [](int depth, DebugValue::parse_event_t, DebugValue&) {
            if (depth > 32) { throw std::invalid_argument("Trace nesting limit exceeded"); } return true;
        }));
        return analyzeShaderTrace(bundle);
    } catch (const std::exception& e) { return std::unexpected(invalid(e.what())); }
}
} // namespace metallic::debug
