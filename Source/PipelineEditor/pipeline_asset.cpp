#include "pipeline_asset.h"
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <spdlog/spdlog.h>

PipelineAsset PipelineAsset::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("PipelineAsset: failed to open '{}'", path);
        return {};
    }

    try {
        nlohmann::json j;
        file >> j;
        return j.get<PipelineAsset>();
    } catch (const nlohmann::json::exception& e) {
        spdlog::error("PipelineAsset: JSON parse error in '{}': {}", path, e.what());
        return {};
    }
}

void PipelineAsset::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        spdlog::error("PipelineAsset: failed to write '{}'", path);
        return;
    }

    nlohmann::json j = *this;
    file << j.dump(2);
    spdlog::info("PipelineAsset: saved '{}'", path);
}

bool PipelineAsset::validate(std::string& errorMsg) const {
    // Check for duplicate resource names
    std::unordered_set<std::string> resourceNames;
    for (const auto& res : resources) {
        if (resourceNames.count(res.name)) {
            errorMsg = "Duplicate resource name: " + res.name;
            return false;
        }
        resourceNames.insert(res.name);
    }

    // Build map of resource producers (which pass outputs each resource)
    std::unordered_map<std::string, std::string> resourceProducer;
    for (const auto& pass : passes) {
        for (const auto& output : pass.outputs) {
            if (output.empty() || output[0] == '$') continue;  // skip special resources like $backbuffer
            if (resourceProducer.count(output)) {
                errorMsg = "Resource '" + output + "' produced by multiple passes: " +
                           resourceProducer[output] + " and " + pass.name;
                return false;
            }
            resourceProducer[output] = pass.name;
        }
    }

    // Check all inputs have producers (or are imported/special)
    for (const auto& pass : passes) {
        for (const auto& input : pass.inputs) {
            if (input.empty() || input[0] == '$') continue;  // skip special resources
            if (!resourceProducer.count(input) && !resourceNames.count(input)) {
                errorMsg = "Pass '" + pass.name + "' reads undefined resource: " + input;
                return false;
            }
        }
    }

    // Check for cycles using topological sort
    auto sorted = topologicalSort();
    if (sorted.size() != passes.size()) {
        errorMsg = "Pipeline contains a cycle";
        return false;
    }

    return true;
}

std::vector<size_t> PipelineAsset::topologicalSort() const {
    // Build adjacency list and in-degree count
    std::unordered_map<std::string, size_t> passIndex;
    for (size_t i = 0; i < passes.size(); i++) {
        passIndex[passes[i].name] = i;
    }

    // Map resource -> producer pass index
    std::unordered_map<std::string, size_t> resourceProducer;
    for (size_t i = 0; i < passes.size(); i++) {
        for (const auto& output : passes[i].outputs) {
            resourceProducer[output] = i;
        }
    }

    // Build dependency graph
    std::vector<std::vector<size_t>> adj(passes.size());
    std::vector<size_t> inDegree(passes.size(), 0);

    for (size_t i = 0; i < passes.size(); i++) {
        for (const auto& input : passes[i].inputs) {
            if (input.empty()) continue;
            auto it = resourceProducer.find(input);
            if (it != resourceProducer.end() && it->second != i) {
                adj[it->second].push_back(i);
                inDegree[i]++;
            }
        }
    }

    // Chain passes that share the same output (e.g. $backbuffer).
    // When multiple passes write to the same resource, they must execute
    // in declaration order so later passes see earlier writes.
    std::unordered_map<std::string, std::vector<size_t>> outputWriters;
    for (size_t i = 0; i < passes.size(); i++) {
        for (const auto& output : passes[i].outputs) {
            outputWriters[output].push_back(i);
        }
    }
    for (const auto& [name, writers] : outputWriters) {
        for (size_t w = 1; w < writers.size(); w++) {
            adj[writers[w - 1]].push_back(writers[w]);
            inDegree[writers[w]]++;
        }
    }

    // Kahn's algorithm
    std::queue<size_t> q;
    for (size_t i = 0; i < passes.size(); i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<size_t> result;
    result.reserve(passes.size());

    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        result.push_back(u);

        for (size_t v : adj[u]) {
            if (--inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    return result;
}
