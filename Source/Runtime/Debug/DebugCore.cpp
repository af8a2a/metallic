#include "Runtime/Debug/DebugCore.h"
#include "Runtime/Debug/DebugProbe.h"

#include <algorithm>
#include <random>
#include <set>

namespace metallic::debug {
namespace {

[[noreturn]] void reject(std::string code, std::string message)
{
    throw DebugError{std::move(code), std::move(message)};
}

bool terminal(std::string_view state)
{
    return state == "Ready" || state == "Cancelled" || state == "Failed";
}

DebugValue page(DebugValue value, const DebugValue& params)
{
    if (!value.is_array()) { return value; }
    const uint64_t offset = debugUnsigned(params.value("offset", DebugValue(0)));
    const uint64_t count = std::min(debugUnsigned(params.value("count", DebugValue(256))), uint64_t(4096));
    const uint64_t size = value.size();
    if (offset > size) { reject("OutOfRange", "Page offset exceeds array length"); }
    DebugValue items = DebugValue::array();
    for (uint64_t i = offset; i < size && i - offset < count; ++i) { items.push_back(std::move(value[i])); }
    return {{"items", std::move(items)}, {"offset", offset}, {"total", size}, {"truncated", offset + count < size}};
}

} // namespace

DebugValue paginateDebugValue(DebugValue value, const DebugValue& params)
{
    return page(std::move(value), params);
}

DebugValue DebugCapture::manifest() const
{
    DebugValue result{{"version", 1}, {"evidence", snapshot.evidence.value()},
        {"values", snapshot.values}, {"artifacts", DebugValue::array()}};
    for (size_t i = 0; i < artifacts.size(); ++i) {
        auto item = artifacts[i].metadata;
        item["index"] = i;
        item["bytes"] = artifacts[i].bytes.size();
        item["layout"] = artifacts[i].layout.schema();
        item["file"] = std::to_string(i) + ".bin";
        result["artifacts"].push_back(std::move(item));
    }
    return result;
}

DebugResult<DebugValue> DebugCapture::evaluationRoot() const
{
    DebugValue root = snapshot.values;
    root["buffers"] = DebugValue::object();
    root["coverage"] = DebugValue::object();
    uint64_t elementCount = 0;
    for (const auto& artifact : artifacts) {
        if (!artifact.layout.stride) { return std::unexpected(DebugError{"LayoutMismatch", "Missing layout"}); }
        elementCount += artifact.bytes.size() / artifact.layout.stride;
        if (elementCount > 262144) {
            return std::unexpected(DebugError{"BudgetExceeded", "CPU eval supports 262144 captured elements; capture a smaller range"});
        }
        auto decoded = decodeBuffer(artifact.bytes, artifact.layout);
        if (!decoded) { return std::unexpected(decoded.error()); }
        const std::string id = artifact.metadata.at("id").get<std::string>();
        if (artifact.metadata.value("kind", "") == "gpuProbe") {
            auto summary = summarizeProbe(artifact.bytes, artifact.metadata);
            if (!summary) { return std::unexpected(summary.error()); }
            root["probes"][artifact.metadata.at("name").get<std::string>()] = std::move(*summary);
        } else { root["buffers"][id] = std::move(*decoded); }
        root["coverage"][id] = artifact.metadata;
    }
    addCaptureRelations(root);
    return root;
}

DebugCore::DebugCore(DebugLimits limits) : limits_(limits)
{
    std::random_device random;
    std::array<uint8_t, 16> bytes;
    for (auto& b : bytes) { b = static_cast<uint8_t>(random()); }
    session_ = hexEncode(bytes);
    if (!limits_.snapshotCount || !limits_.queueCount || !limits_.commandsPerFrame ||
        !limits_.snapshotBytes || !limits_.jobBytes || !limits_.frameBytes || limits_.jobBytes > limits_.capturePoolBytes ||
        limits_.capturePoolBytes > (1ull << 30) || limits_.snapshotBytes > (1ull << 30) ||
        limits_.frameBytes > (1ull << 30) || !limits_.probeScanBytes || limits_.probeScanBytes > (1ull << 30) ||
        limits_.queueCount > 4096 || limits_.snapshotCount > 10000 || limits_.commandsPerFrame > 256) {
        throw std::invalid_argument("Invalid DebugLimits");
    }
}

void DebugCore::setProcess(uint32_t pid, std::string startTime)
{
    std::lock_guard lock(mutex_);
    process_ = {{"pid", pid}, {"startTime", std::move(startTime)}};
}

void DebugCore::setSchema(std::string name, DebugValue schema)
{
    std::lock_guard lock(mutex_);
    schemas_[std::move(name)] = std::move(schema);
}

void DebugCore::setGraph(DebugValue graph)
{
    std::lock_guard lock(mutex_);
    graph_ = std::move(graph);
    for (auto& [id, job] : jobs_) {
        if (job.state == "Queued" && (job.request.graph != graph_.value("id", "") ||
                job.request.generation != graph_.value("generation", uint64_t(0)))) {
            job.state = "Failed";
            job.error = {"StaleHandle", "Graph or scene generation changed before capture"};
        }
    }
}

void DebugCore::setEngineState(DebugValue state)
{
    std::lock_guard lock(mutex_);
    engineState_ = std::move(state);
}

void DebugCore::pushEvent(std::string provider, DebugValue event)
{
    std::lock_guard lock(mutex_);
    auto& queue = events_[provider];
    if (queue.size() == 256) { queue.pop_front(); ++droppedEvents_[provider]; }
    queue.push_back(std::move(event));
}

DebugValue DebugCore::events(std::string_view provider) const
{
    std::lock_guard lock(mutex_);
    const std::string name(provider);
    const auto found = events_.find(name);
    return {{"events", found == events_.end() ? DebugValue::array() : DebugValue(found->second)},
        {"dropped", droppedEvents_.contains(name) ? droppedEvents_.at(name) : 0}};
}

void DebugCore::publish(DebugSnapshot snapshot)
{
    snapshot.evidence.session = session_;
    const uint64_t bytes = snapshot.values.dump(-1, ' ', false, DebugValue::error_handler_t::replace).size();
    if (bytes > limits_.snapshotBytes) { return; }
    std::lock_guard lock(mutex_);
    // Completion collection may happen out of order across graphs/queues.
    auto entry = std::make_shared<const DebugSnapshot>(std::move(snapshot));
    auto position = std::upper_bound(snapshots_.begin(), snapshots_.end(), entry,
        [](const auto& a, const auto& b) { return a->evidence.sample < b.first->evidence.sample; });
    snapshots_.insert(position, {std::move(entry), bytes});
    snapshotBytes_ += bytes;
    while (snapshots_.size() > limits_.snapshotCount || snapshotBytes_ > limits_.snapshotBytes) {
        snapshotBytes_ -= snapshots_.front().second;
        snapshots_.pop_front();
    }
}

void DebugCore::expireLocked()
{
    const auto now = std::chrono::steady_clock::now();
    for (auto& [id, job] : jobs_) {
        if (!terminal(job.state) && now >= job.deadline) {
            job.state = "Cancelled";
            job.error = {"Timeout", "Capture deadline reached; submitted work remains alive until completion"};
        }
    }
    updateWatchesLocked();
}

void DebugCore::expire()
{
    std::lock_guard lock(mutex_);
    expireLocked();
}

void DebugCore::pruneLocked(uint64_t needed)
{
    for (auto it = order_.begin(); it != order_.end() &&
            (captureBytes_ + needed > limits_.capturePoolBytes || jobs_.size() >= limits_.queueCount);) {
        auto found = jobs_.find(*it);
        const bool pinned = std::any_of(watches_.begin(), watches_.end(), [&](const auto& item) {
            return item.second.job == *it && (item.second.state == "Active" || item.second.state == "Triggered");
        });
        if (!pinned && found != jobs_.end() && terminal(found->second.state) &&
            (found->second.capture || found->second.reservedBytes == 0)) {
            captureBytes_ -= found->second.reservedBytes;
            jobs_.erase(found);
            it = order_.erase(it);
        } else { ++it; }
    }
}

std::vector<DebugCaptureRequest> DebugCore::takeRequests(std::string_view graph, uint64_t generation)
{
    std::lock_guard lock(mutex_);
    expireLocked();
    std::vector<DebugCaptureRequest> requests;
    const auto begin = std::chrono::steady_clock::now();
    ++executionTick_;
    uint32_t scheduled = 0;
    for (auto& [id, watch] : watches_) {
        if (scheduled >= limits_.commandsPerFrame || std::chrono::steady_clock::now() - begin >= std::chrono::milliseconds(1)) { break; }
        if (watch.state != "Active" || !watch.job.empty() || executionTick_ < watch.nextTick) { continue; }
        try { watch.job = enqueueCaptureLocked(watch.specification, true).at("job"); ++scheduled; }
        catch (const DebugError& error) { watch.state = "Failed"; watch.result = {{"reason", error.code}}; }
    }
    for (const auto& id : order_) {
        auto& job = jobs_.at(id);
        if (job.state != "Queued") { continue; }
        if (job.request.graph != graph || job.request.generation != generation) {
            job.state = "Failed";
            job.error = {"StaleHandle", "Capture generation is no longer current"};
            continue;
        }
        if (job.request.groupRemaining > limits_.commandsPerFrame - requests.size()) { break; }
        requests.push_back(job.request);
        job.state = "Recording";
        for (auto& [watchId, watch] : watches_) {
            if (watch.job == id && watch.state == "Active") {
                ++watch.samples; watch.nextTick = executionTick_ + watch.every;
            }
        }
        if (job.request.groupRemaining == 1 && (requests.size() >= limits_.commandsPerFrame ||
            std::chrono::steady_clock::now() - begin >= std::chrono::milliseconds(1))) { break; }
    }
    return requests;
}

bool DebugCore::reserve(std::string_view id, uint64_t bytes)
{
    std::lock_guard lock(mutex_);
    pruneLocked(bytes);
    auto found = jobs_.find(std::string(id));
    if (found == jobs_.end() || terminal(found->second.state) || found->second.reservedBytes ||
        bytes > limits_.jobBytes || bytes > limits_.capturePoolBytes - captureBytes_) { return false; }
    found->second.reservedBytes = bytes;
    captureBytes_ += bytes;
    return true;
}

void DebugCore::transition(std::string_view id, std::string state)
{
    std::lock_guard lock(mutex_);
    auto found = jobs_.find(std::string(id));
    if (found != jobs_.end() && !terminal(found->second.state)) { found->second.state = std::move(state); }
}

bool DebugCore::cancelled(std::string_view id) const
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(std::string(id));
    return found == jobs_.end() || found->second.state == "Cancelled" || found->second.state == "Failed";
}

void DebugCore::fail(std::string_view id, DebugError error)
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(std::string(id));
    if (found == jobs_.end()) { return; }
    auto& job = found->second;
    captureBytes_ -= job.reservedBytes;
    job.reservedBytes = 0;
    job.capture.reset();
    if (job.state != "Cancelled") { job.state = "Failed"; job.error = std::move(error); }
}

void DebugCore::complete(std::string_view id, std::shared_ptr<const DebugCapture> capture)
{
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(std::string(id));
    if (found == jobs_.end()) { return; }
    auto& job = found->second;
    if (job.state == "Cancelled" || job.state == "Failed") {
        captureBytes_ -= job.reservedBytes; job.reservedBytes = 0;
    } else {
        job.capture = std::move(capture);
        job.state = "Ready";
    }
    updateWatchesLocked();
}

DebugValue DebugCore::enqueueCaptureLocked(const DebugValue& params, bool probe)
{
    if (graph_.empty() || graph_.value("state", "Ready") != "Ready") { reject("NotReady", "No compiled graph"); }
    const auto specifications = params.contains("batches") ? params.at("batches") : DebugValue::array({params});
    if (!specifications.is_array() || specifications.empty() || specifications.size() > limits_.commandsPerFrame) {
        reject("InvalidArgument", "A checkpoint group must fit within commandsPerFrame");
    }
    std::vector<DebugJob> pending;
    for (const auto& specification : specifications) {
        const auto resources = specification.value("resources", DebugValue::array());
        if (probe) {
            const auto valid = validateProbeSpecification(specification);
            if (!valid) { reject(valid.error().code, valid.error().message); }
        } else if (specification.contains("probes")) { reject("InvalidArgument", "GPU work requires gpu.probe"); }
        if (!resources.is_array() || (!probe && resources.empty()) || resources.size() > 64) { reject("InvalidArgument", "Expected 1..64 resources"); }
        const std::string pass = specification.at("pass").get<std::string>();
        const std::string checkpoint = specification.value("checkpoint", "AfterPass");
        bool valid = false;
        for (const auto& node : graph_.value("passes", DebugValue::array())) {
            if (node.at("name") != pass || !node.value("active", false)) { continue; }
            for (const auto& point : node.value("checkpoints", DebugValue::array())) { if (point == checkpoint) { valid = true; } }
        }
        if (!valid) { reject("Unsupported", "Pass/checkpoint is not registered in the active graph"); }
        const auto generation = debugUnsigned(specification.value("generation", graph_.value("generation", DebugValue(0))));
        if (generation != graph_.value("generation", uint64_t(0))) { reject("StaleHandle", "Graph generation changed"); }
        std::set<std::string> names;
        for (const auto& resource : resources) {
            if (!resource.is_object() || !names.insert(resource.at("id").get<std::string>()).second) { reject("InvalidArgument", "Resource IDs must be unique"); }
        }
        const auto timeout = debugUnsigned(specification.value("timeoutMs", DebugValue(30000)), 300000);
        if (!timeout) { reject("InvalidArgument", "timeoutMs must be positive"); }
        DebugJob job;
        job.request = {"", graph_.value("id", ""), generation, specification, static_cast<uint32_t>(specifications.size() - pending.size())};
        job.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
        pending.push_back(std::move(job));
    }
    pruneLocked();
    if (pending.size() > limits_.queueCount - jobs_.size()) { reject("QueueFull", "Capture job queue is full"); }
    DebugValue result = DebugValue::array();
    for (auto& job : pending) {
        const std::string id = std::to_string(nextJob_++);
        job.request.id = id;
        result.push_back({{"job", id}, {"state", "Queued"}, {"generation", job.request.generation}});
        jobs_.emplace(id, std::move(job)); order_.push_back(id);
    }
    return params.contains("batches") ? DebugValue{{"jobs", std::move(result)}, {"sameExecution", true}} : result.front();
}

DebugValue debugErrorResponse(const DebugValue& id, std::string code, std::string message)
{
    return {{"version", 1}, {"id", id}, {"status", "error"}, {"error", {{"code", std::move(code)}, {"message", std::move(message)}}}};
}

DebugValue DebugCore::dispatch(const DebugValue& request)
{
    const auto id = request.is_object() ? request.value("id", DebugValue(nullptr)) : DebugValue(nullptr);
    try {
        if (!request.is_object() || request.value("version", 1) != 1 || !request.contains("method")) { reject("ProtocolError", "Expected version 1 request"); }
        if (request.contains("session") && request.at("session") != session_) { reject("StaleSession", "Engine process session changed"); }
        auto params = request.value("params", DebugValue::object());
        if (!params.is_object()) { reject("InvalidArgument", "params must be an object"); }
        auto result = route(request.at("method").get<std::string>(), params);
        return {{"version", 1}, {"id", id}, {"status", "ok"}, {"result", std::move(result)}};
    } catch (const DebugError& error) { return debugErrorResponse(id, error.code, error.message); }
    catch (const std::exception& error) { return debugErrorResponse(id, "InvalidArgument", error.what()); }
}

DebugValue DebugCore::route(std::string_view method, const DebugValue& params)
{
    std::unique_lock lock(mutex_);
    expireLocked();
    if (method.starts_with("watch.")) { return watchRouteLocked(method, params); }
    if (method == "hello") {
        DebugValue providers = DebugValue::array();
        for (auto it = schemas_.begin(); it != schemas_.end(); ++it) { providers.push_back(it.key()); }
        return {{"protocolVersion", 1}, {"session", session_}, {"process", process_}, {"engine", engineState_},
            {"providers", std::move(providers)}, {"schemaMethod", "schema"}, {"transport", "u32le-json-client-ack"},
            {"methods", {"hello", "schema", "frame.latest", "rg.describe", "rg.trace", "object.get", "eval", "capture.batch", "gpu.probe", "watch.create", "watch.get", "watch.list", "watch.cancel", "watch.delete", "jobs.get", "jobs.cancel", "artifact.read"}},
            {"limits", {{"snapshotCount", limits_.snapshotCount}, {"snapshotBytes", limits_.snapshotBytes}, {"capturePoolBytes", limits_.capturePoolBytes},
                {"jobBytes", limits_.jobBytes}, {"frameBytes", limits_.frameBytes}, {"queueCount", limits_.queueCount},
                {"commandsPerFrame", limits_.commandsPerFrame}, {"probeScanBytes", limits_.probeScanBytes},
                {"messageBytes", 1048576}, {"cpuEvalOperations", 1000000}}}, {"graph", graph_}};
    }
    if (method == "schema") {
        const auto name = params.value("name", "");
        if (name.empty()) { return schemas_; }
        if (!schemas_.contains(name)) { reject("NotFound", "Unknown schema"); }
        return schemas_.at(name);
    }
    if (method == "rg.describe") { return graph_; }
    if (method == "rg.trace") {
        const auto resource = params.at("resource").get<std::string>();
        const bool backward = params.value("direction", "backward") == "backward";
        std::set<std::string> visited{resource.substr(0, resource.find('.'))};
        DebugValue edges = DebugValue::array();
        auto remaining = graph_.value("edges", DebugValue::array());
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = remaining.begin(); it != remaining.end();) {
                const auto src = it->at("srcPass").get<std::string>(), dst = it->at("dstPass").get<std::string>();
                if (visited.contains(backward ? dst : src)) {
                    changed = visited.insert(backward ? src : dst).second || changed;
                    edges.push_back(*it); it = remaining.erase(it);
                } else { ++it; }
            }
        }
        return {{"passes", visited}, {"edges", edges}, {"direction", backward ? "backward" : "forward"}};
    }
    if (method == "capture.batch" || method == "gpu.probe") { return enqueueCaptureLocked(params, method == "gpu.probe"); }
    if (method == "jobs.get" || method == "jobs.cancel" || method == "artifact.read") {
        const std::string id = params.at("job").get<std::string>();
        const auto found = jobs_.find(id);
        if (found == jobs_.end()) { reject("NotCaptured", "Job expired or was evicted"); }
        auto& job = found->second;
        if (method == "jobs.cancel" && !terminal(job.state)) { job.state = "Cancelled"; job.error = {"Cancelled", "Cancelled by client"}; }
        if (method == "artifact.read") {
            if (!job.capture) { reject("NotReady", "Capture is not ready"); }
            if (params.value("manifest", false)) {
                const auto capture = job.capture;
                lock.unlock();
                const auto bytes = encodeLossless(capture->manifest()).dump();
                const uint64_t offset = debugUnsigned(params.value("offset", DebugValue(0)));
                if (offset > bytes.size()) { reject("OutOfRange", "Manifest offset out of range"); }
                const uint64_t count = std::min({debugUnsigned(params.value("count", DebugValue(262144))), uint64_t(262144), uint64_t(bytes.size() - offset)});
                return {{"offset", offset}, {"total", bytes.size()}, {"hex", hexEncode({reinterpret_cast<const uint8_t*>(bytes.data()) + offset, static_cast<size_t>(count)})}};
            }
            const uint64_t index = debugUnsigned(params.at("index"));
            if (index >= job.capture->artifacts.size()) { reject("OutOfRange", "Artifact index out of range"); }
            const auto capture = job.capture;
            lock.unlock();
            const auto& bytes = capture->artifacts[index].bytes;
            const uint64_t offset = debugUnsigned(params.value("offset", DebugValue(0)));
            if (offset > bytes.size()) { reject("OutOfRange", "Byte offset out of range"); }
            const uint64_t count = std::min({debugUnsigned(params.value("count", DebugValue(262144))), uint64_t(262144), uint64_t(bytes.size() - offset)});
            return {{"offset", offset}, {"total", bytes.size()}, {"hex", hexEncode(std::span(bytes).subspan(offset, count))}};
        }
        DebugValue result{{"job", id}, {"state", job.state}, {"reservedBytes", job.reservedBytes}};
        if (!job.error.code.empty()) { result["error"] = {{"code", job.error.code}, {"message", job.error.message}}; }
        const auto capture = job.capture;
        lock.unlock();
        if (capture) {
            // Raw evidence is chunked through artifact.read, including the
            // manifest. A large metadata snapshot must not break job polling.
            result["evidence"] = capture->snapshot.evidence.value();
            result["artifactCount"] = capture->artifacts.size();
            if (capture->snapshot.values.contains("probes")) { result["probes"] = capture->snapshot.values.at("probes"); }
            if (params.value("includeCapture", false)) { result["capture"] = capture->manifest(); }
            if (params.value("stats", false)) {
                const auto stats = capture->statistics();
                if (!stats) { reject(stats.error().code, stats.error().message); }
                result["statistics"] = *stats;
            }
        }
        return result;
    }
    if (method == "frame.latest" || method == "object.get" || method == "eval") {
        std::shared_ptr<const DebugSnapshot> snapshot;
        std::shared_ptr<const DebugCapture> capture;
        if (params.contains("job")) {
            const auto found = jobs_.find(params.at("job").get<std::string>());
            if (found == jobs_.end() || !found->second.capture) { reject("NotCaptured", "Capture is unavailable"); }
            capture = found->second.capture;
        } else {
            for (auto it = snapshots_.rbegin(); it != snapshots_.rend(); ++it) {
                if (!params.value("recorded", false) && it->first->evidence.provenance.value("completion", "Ready") == "Untracked") { continue; }
                if ((!params.contains("frame") || it->first->evidence.execution == debugUnsigned(params.at("frame"))) &&
                    (!params.contains("graph") || it->first->evidence.graph == params.at("graph").get<std::string>())) { snapshot = it->first; break; }
            }
            if (!snapshot) { reject("NotCaptured", "No completed snapshot for the requested frame"); }
        }
        uint64_t latest = 0;
        for (auto it = snapshots_.rbegin(); it != snapshots_.rend(); ++it) {
            if (it->first->evidence.provenance.value("completion", "Ready") == "Ready") { latest = it->first->evidence.execution; break; }
        }
        lock.unlock();
        DebugValue root;
        DebugEvidenceStamp evidence;
        if (capture) {
            auto decoded = capture->evaluationRoot();
            if (!decoded) { reject(decoded.error().code, decoded.error().message); }
            root = std::move(*decoded); evidence = capture->snapshot.evidence;
        } else { root = snapshot->values; evidence = snapshot->evidence; }
        DebugValue result;
        if (method == "frame.latest") { result = root; }
        else {
            auto evaluated = evaluate(params.at(method == "eval" ? "expression" : "path").get<std::string>(), root);
            if (!evaluated) { reject(evaluated.error().code, evaluated.error().message); }
            result = page(std::move(*evaluated), params);
        }
        return {{"value", std::move(result)}, {"evidence", evidence.value()},
            {"source", capture ? "captured" : (evidence.provenance.value("completion", "Ready") == "Untracked" ? "recorded-untracked" : (params.contains("frame") ? "frame" : "latest-completed-frame"))},
            {"stalenessFrames", latest >= evidence.execution ? latest - evidence.execution : 0},
            {"coverage", root.value("coverage", DebugValue::object())}};
    }
    reject("Unsupported", "Unknown method: " + std::string(method));
}

} // namespace metallic::debug
