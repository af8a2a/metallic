#include "Runtime/Debug/DebugCore.h"
#include "Runtime/Debug/DebugProbe.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace metallic::debug {
namespace {

struct NumericSummary {
    uint64_t count = 0, finite = 0, nan = 0, infinity = 0;
    double mean = 0;
    DebugValue minimum, maximum;
    void add(const DebugValue& value)
    {
        ++count;
        const double number = value.get<double>();
        if (std::isnan(number)) { ++nan; return; }
        if (std::isinf(number)) { ++infinity; return; }
        ++finite;
        // All values for one registered field have the same scalar type.
        if (minimum.is_null() || value < minimum) { minimum = value; }
        if (maximum.is_null() || value > maximum) { maximum = value; }
        mean = mean * (double(finite - 1) / double(finite)) + number / double(finite);
    }
    DebugValue value() const
    {
        return {{"count", count}, {"finiteCount", finite}, {"nanCount", nan}, {"infCount", infinity},
            {"min", minimum}, {"max", maximum}, {"mean", finite ? DebugValue(mean) : DebugValue(nullptr)},
            {"meanType", "f64-approximate"}};
    }
};

void summarize(const DebugValue& value, const std::string& path, std::map<std::string, NumericSummary>& fields)
{
    if (value.is_number()) { fields[path.empty() ? "value" : path].add(value); }
    else if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) { summarize(it.value(), path.empty() ? it.key() : path + "." + it.key(), fields); }
    } else if (value.is_array()) {
        for (size_t i = 0; i < value.size(); ++i) { summarize(value[i], path + "[" + std::to_string(i) + "]", fields); }
    }
}

// References carry absolute element indices. An absent captured subrange is
// never filled from current engine state, even when metadata describes it.
DebugValue gpuReference(const DebugValue& root, const std::string& id, uint64_t index)
{
    DebugValue reference{{"resource", id}, {"index", index}, {"status", "MissingDependency"}};
    const auto& buffers = root.at("buffers");
    const auto& coverage = root.at("coverage");
    if (!buffers.contains(id) || !coverage.contains(id)) { return reference; }
    const auto& range = coverage.at(id);
    if (index >= range.value("capacity", UINT64_MAX)) { reference["status"] = "OutOfRange"; return reference; }
    const uint64_t offset = range.value("elementOffset", uint64_t(0));
    if (index < offset || index - offset >= buffers.at(id).size()) { return reference; }
    reference["status"] = "Captured";
    reference["value"] = buffers.at(id)[index - offset];
    return reference;
}

DebugValue cpuReference(const DebugValue& scene, const char* table, uint64_t index, const char* countName)
{
    DebugValue reference{{"provider", "gpuScene"}, {"table", table}, {"index", index}, {"status", "MissingDependency"}};
    if (!scene.value("available", false)) { return reference; }
    if (index >= scene.at("stats").value(countName, UINT64_MAX)) { reference["status"] = "OutOfRange"; return reference; }
    const auto records = scene.find(table);
    if (records == scene.end()) { return reference; }
    for (const auto& record : *records) {
        if (record.at("index") == index) { reference["status"] = "Captured"; reference["value"] = record; break; }
    }
    return reference;
}

} // namespace

DebugResult<DebugValue> DebugCapture::statistics() const
{
    DebugValue result = DebugValue::object();
    uint64_t elements = 0;
    for (const auto& artifact : artifacts) {
        if (artifact.metadata.value("kind", "") == "gpuProbe") {
            auto summary = summarizeProbe(artifact.bytes, artifact.metadata);
            if (!summary) { return std::unexpected(summary.error()); }
            result[artifact.metadata.at("id").get<std::string>()] = std::move(*summary);
            continue;
        }
        if (!artifact.layout.stride) { return std::unexpected(DebugError{"LayoutMismatch", "Missing layout"}); }
        elements += artifact.bytes.size() / artifact.layout.stride;
        if (elements > 262144) { return std::unexpected(DebugError{"BudgetExceeded", "CPU statistics support 262144 elements; select a smaller range"}); }
        auto rows = decodeBuffer(artifact.bytes, artifact.layout);
        if (!rows) { return std::unexpected(rows.error()); }
        std::map<std::string, NumericSummary> fields;
        for (const auto& row : *rows) { summarize(row, "", fields); }
        DebugValue values = DebugValue::object();
        for (const auto& [name, summary] : fields) { values[name] = summary.value(); }
        result[artifact.metadata.at("id").get<std::string>()] = {{"fields", std::move(values)},
            {"coverage", artifact.metadata}, {"interpretation", "Statistics describe captured storage only; uninitialized capacity is not a live record set"}};
    }
    return result;
}

void addCaptureRelations(DebugValue& root)
{
    root["links"] = DebugValue::object();
    root["diagnostics"] = DebugValue::array();
    const auto scene = root.value("gpuScene", DebugValue::object());
    std::map<std::string, std::set<uint64_t>> visibleRecords;
    for (auto it = root.at("buffers").begin(); it != root.at("buffers").end(); ++it) {
        const auto& metadata = root.at("coverage").at(it.key());
        if (it.key().ends_with(".visibility") && metadata.value("kind", "") == "texture" && metadata.value("layout", "") == "u32") {
            auto& indices = visibleRecords[it.key()];
            for (const auto& pixel : it.value()) {
                const auto encoded = pixel.get<uint32_t>() >> 7;
                if (encoded) { indices.insert(encoded - 1); }
            }
        }
    }
    for (auto it = root.at("buffers").begin(); it != root.at("buffers").end(); ++it) {
        const auto& id = it.key();
        const auto& metadata = root.at("coverage").at(id);
        const auto layout = metadata.value("layout", "");
        const auto prefix = id.substr(0, id.rfind('.') + 1);
        const auto offset = metadata.value("elementOffset", uint64_t(0));
        DebugValue links = DebugValue::array();
        for (size_t i = 0; i < std::min(it.value().size(), size_t(4096)); ++i) {
            const auto& record = it.value()[i];
            DebugValue link{{"index", offset + i}};
            if (layout == "VisibleClusterRecord") {
                const uint32_t source = (record.at("flags").get<uint32_t>() >> 28) & 3;
                const std::string visibility = metadata.value("pass", "") + ".visibility";
                const uint64_t globalIndex = metadata.value("visibleRecordBase", uint64_t(0)) + offset + i;
                link["recordValidity"] = visibleRecords[visibility].contains(globalIndex) ? "ReferencedByCapturedPixel" : "Unverified";
                link["source"] = source == 0 ? "Resident" : source == 1 ? "StreamPage" : "Unknown";
                link["instance"] = cpuReference(scene, "instances", record.at("instanceIndex"), "instanceCount");
                if (source == 0) {
                    link["geometry"] = cpuReference(scene, "geometries", record.at("dataIndex"), "geometryCount");
                    const std::string pass = metadata.value("pass", "");
                    link["geometryGpu"] = gpuReference(root, "gpuScene." + pass + ".geometries", record.at("dataIndex"));
                    if (link["geometryGpu"]["status"] == "Captured") {
                        const auto& payload = link["geometryGpu"]["value"]["payload"];
                        link["baseMeshletRange"] = {{"offset", payload[2]}, {"count", payload[3]}};
                    }
                    link["meshlet"] = gpuReference(root, "gpuScene." + pass + ".meshlets", record.at("clusterIndex"));
                } else if (source == 1) {
                    link["activeGroup"] = gpuReference(root, prefix + "activeGroups", record.at("dataIndex"));
                    link["activeHeader"] = gpuReference(root, prefix + "activeHeader", 0);
                    const auto& group = link["activeGroup"];
                    if (group.at("status") == "Captured") {
                        link["page"] = gpuReference(root, prefix + "pageTable", group.at("value").at("pageIndex"));
                        link["clusterInRange"] = record.at("clusterIndex").get<uint64_t>() < group.at("value").at("clusterCount").get<uint64_t>();
                    }
                }
            } else if (layout == "MeshletStreamGpuActiveGroup") {
                link["page"] = gpuReference(root, prefix + "pageTable", record.at("pageIndex"));
                link["header"] = gpuReference(root, prefix + "activeHeader", 0);
                link["instance"] = cpuReference(scene, "instances", record.at("gpuSceneInstanceIndex"), "instanceCount");
                if (record.at("gpuSceneInstanceIndex") == UINT32_MAX) { link["instance"]["status"] = "Unmapped"; }
                link["recordValidity"] = link["header"]["status"] == "Captured"
                    ? (offset + i < link["header"]["value"]["activeGroupCount"].get<uint64_t>() ? "Live" : "OutsideLiveRange") : "MissingDependency";
            } else if (id.ends_with(".visibleInstanceIds")) {
                link["instance"] = cpuReference(scene, "instances", record.get<uint64_t>(), "instanceCount");
                link["counter"] = gpuReference(root, prefix + "visibleInstanceCounter", 0);
                link["recordValidity"] = link["counter"]["status"] == "Captured"
                    ? (offset + i < link["counter"]["value"].get<uint64_t>() ? "Live" : "OutsideLiveRange") : "MissingDependency";
            } else if (layout == "StreamRequestBufferHeader" || layout == "MeshletStreamGpuActiveHeader") {
                const auto check = [&](const char* count, const char* capacity) {
                    if (record.at(count).get<uint64_t>() > record.at(capacity).get<uint64_t>()) {
                        root["diagnostics"].push_back({{"code", "CounterExceedsCapacity"}, {"resource", id}, {"index", offset + i},
                            {"counter", count}, {"value", record.at(count)}, {"capacity", record.at(capacity)}});
                    }
                };
                if (layout == "StreamRequestBufferHeader") { check("loadCounter", "maxLoadRequests"); check("unloadCounter", "maxUnloadRequests"); }
                else { check("activeGroupCount", "activeGroupCapacity"); }
                continue;
            } else { continue; }
            links.push_back(std::move(link));
        }
        if (!links.empty()) { root["links"][id] = {{"items", std::move(links)}, {"truncated", it.value().size() > 4096}}; }
    }
}

} // namespace metallic::debug
