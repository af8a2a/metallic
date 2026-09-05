#include "Runtime/Debug/DebugCore.h"

namespace metallic::debug {
namespace {
bool finished(std::string_view state)
{
    return state == "Ready" || state == "Failed" || state == "Cancelled";
}
} // namespace

void DebugCore::updateWatchesLocked()
{
    for (auto& [id, watch] : watches_) {
        if (watch.state != "Active") { continue; }
        const auto stop = [&](std::string state, std::string reason) {
            watch.state = std::move(state); watch.result = {{"reason", std::move(reason)}};
            if (auto job = jobs_.find(watch.job); job != jobs_.end() && !finished(job->second.state)) {
                job->second.state = "Cancelled";
                job->second.error = {"Cancelled", "Watch stopped; submitted resources retained until completion"};
            }
        };
        if (watch.graph != graph_.value("id", "") || watch.generation != graph_.value("generation", uint64_t(0))) {
            stop("Failed", "StaleHandle"); continue;
        }
        if (std::chrono::steady_clock::now() >= watch.deadline) { stop("Expired", "Watch deadline reached"); continue; }
        if (watch.job.empty()) { continue; }
        const auto found = jobs_.find(watch.job);
        if (found == jobs_.end()) { stop("Failed", "Watch job unavailable"); continue; }
        auto& job = found->second;
        if (!finished(job.state)) { continue; }
        if (job.state != "Ready" || !job.capture) {
            stop("Failed", job.error.code.empty() ? job.state : job.error.code); continue;
        }
        const auto& probes = job.capture->snapshot.values.at("probes");
        const auto name = watch.trigger.at("probe").get<std::string>();
        const auto field = watch.trigger.value("field", "matchedCount");
        const auto op = watch.trigger.value("op", "gt");
        const auto value = debugUnsigned(probes.at(name).at(field));
        const auto threshold = debugUnsigned(watch.trigger.value("value", DebugValue(0)));
        const bool hit = op == "gt" ? value > threshold : op == "ge" ? value >= threshold : value == threshold;
        watch.result = {{"evidence", job.capture->snapshot.evidence.value()}, {"probes", probes},
            {"triggered", hit}, {"asynchronous", true}};
        if (hit) { watch.state = "Triggered"; continue; }
        // Non-triggering evidence is discarded only after GPU completion.
        captureBytes_ -= job.reservedBytes; job.reservedBytes = 0; job.capture.reset();
        watch.job.clear();
        if (watch.samples >= watch.maxSamples) { watch.state = "Completed"; }
    }
}

DebugValue DebugCore::watchRouteLocked(std::string_view method, const DebugValue& params)
{
    updateWatchesLocked();
    const auto describe = [](const std::string& id, const Watch& watch) {
        return DebugValue{{"watch", id}, {"state", watch.state}, {"job", watch.job}, {"samples", watch.samples},
            {"everyExecutions", watch.every}, {"maxSamples", watch.maxSamples}, {"generation", watch.generation},
            {"trigger", watch.trigger}, {"result", watch.result}};
    };
    if (method == "watch.list") {
        DebugValue result = DebugValue::array();
        for (const auto& [id, watch] : watches_) { result.push_back(describe(id, watch)); }
        return result;
    }
    if (method == "watch.create") {
        if (watches_.size() >= 16) { throw DebugError{"QueueFull", "At most 16 watches; delete a finished watch first"}; }
        Watch watch;
        watch.specification = params.at("probe");
        if (watch.specification.contains("batches")) { throw DebugError{"InvalidArgument", "A watch uses one checkpoint"}; }
        watch.trigger = params.at("trigger");
        const auto name = watch.trigger.at("probe").get<std::string>();
        const auto field = watch.trigger.value("field", "matchedCount"), op = watch.trigger.value("op", "gt");
        if ((field != "matchedCount" && field != "nanCount" && field != "infCount") || (op != "gt" && op != "ge" && op != "eq")) {
            throw DebugError{"InvalidArgument", "Trigger supports matchedCount/nanCount/infCount and gt/ge/eq"};
        }
        (void)debugUnsigned(watch.trigger.value("value", DebugValue(0)), UINT32_MAX);
        bool named = false;
        for (const auto& probe : watch.specification.at("probes")) {
            named |= probe.value("name", probe.at("id").get<std::string>()) == name;
        }
        if (!named) { throw DebugError{"InvalidArgument", "Trigger references an unknown probe"}; }
        watch.every = debugUnsigned(params.value("everyExecutions", DebugValue(60)), 1000000);
        watch.maxSamples = debugUnsigned(params.value("maxSamples", DebugValue(120)), 10000);
        const auto duration = debugUnsigned(params.value("timeoutMs", DebugValue(300000)), 3600000);
        if (!watch.every || !watch.maxSamples || !duration) { throw DebugError{"InvalidArgument", "Watch bounds must be positive"}; }
        watch.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration);
        const auto queued = enqueueCaptureLocked(watch.specification, true);
        watch.job = queued.at("job");
        watch.graph = graph_.at("id"); watch.generation = graph_.at("generation");
        watch.specification["generation"] = watch.generation;
        const auto id = std::to_string(nextWatch_++);
        watches_.emplace(id, std::move(watch));
        return describe(id, watches_.at(id));
    }
    const auto id = params.at("watch").get<std::string>();
    const auto found = watches_.find(id);
    if (found == watches_.end()) { throw DebugError{"NotFound", "Unknown watch"}; }
    auto& watch = found->second;
    if (method == "watch.cancel" || method == "watch.delete") {
        watch.state = "Cancelled";
        if (auto job = jobs_.find(watch.job); job != jobs_.end() && !finished(job->second.state)) {
            job->second.state = "Cancelled"; job->second.error = {"Cancelled", "Watch cancelled"};
        }
        const auto result = describe(id, watch);
        if (method == "watch.delete") { watches_.erase(found); }
        return result;
    }
    if (method != "watch.get") { throw DebugError{"MethodNotFound", "Unknown watch method"}; }
    return describe(id, watch);
}

} // namespace metallic::debug
