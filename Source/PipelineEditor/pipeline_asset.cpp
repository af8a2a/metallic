#include "pipeline_asset.h"

#include "pass_registry.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <spdlog/spdlog.h>

namespace {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

template <typename T>
T getOrDefault(const Json& j, const char* key, T defaultValue) {
    if (!j.contains(key) || j[key].is_null()) {
        return defaultValue;
    }
    return j[key].get<T>();
}

std::array<float, 2> parseEditorPos(const Json& j) {
    if (!j.is_array() || j.size() != 2) {
        return {0.0f, 0.0f};
    }
    return {j[0].get<float>(), j[1].get<float>()};
}

OrderedJson serializeEditorPos(const std::array<float, 2>& pos) {
    OrderedJson j = OrderedJson::array();
    j.push_back(pos[0]);
    j.push_back(pos[1]);
    return j;
}

bool isKnownResourceKind(const std::string& kind) {
    return kind == "transient" || kind == "imported" || kind == "backbuffer";
}

bool isKnownEdgeDirection(const std::string& direction) {
    return direction == "input" || direction == "output";
}

const PassSlotInfo* findSlotInfo(const std::vector<PassSlotInfo>& slots, const std::string& key) {
    for (const auto& slot : slots) {
        if (slot.key == key) {
            return &slot;
        }
    }
    return nullptr;
}

bool resourceKindAllowed(const PassSlotInfo& slot, const std::string& resourceKind) {
    if (slot.allowedResourceKinds.empty()) {
        return true;
    }
    return std::find(slot.allowedResourceKinds.begin(),
                     slot.allowedResourceKinds.end(),
                     resourceKind) != slot.allowedResourceKinds.end();
}

ResourceDecl parseResourceDecl(const Json& j) {
    ResourceDecl resource;
    resource.id = j.at("id").get<std::string>();
    resource.name = getOrDefault<std::string>(j, "name", {});
    resource.kind = getOrDefault<std::string>(j, "kind", "transient");
    resource.type = getOrDefault<std::string>(j, "type", "texture");
    resource.format = getOrDefault<std::string>(j, "format", {});
    resource.size = getOrDefault<std::string>(j, "size", {});
    resource.importKey = getOrDefault<std::string>(j, "importKey", {});
    resource.editorPos = parseEditorPos(j.value("editorPos", Json::array()));
    return resource;
}

PassDecl parsePassDecl(const Json& j) {
    PassDecl pass;
    pass.id = j.at("id").get<std::string>();
    pass.name = getOrDefault<std::string>(j, "name", {});
    pass.type = getOrDefault<std::string>(j, "type", {});
    pass.enabled = getOrDefault<bool>(j, "enabled", true);
    pass.sideEffect = getOrDefault<bool>(j, "sideEffect", false);
    if (j.contains("config")) {
        pass.config = j["config"];
    }
    pass.editorPos = parseEditorPos(j.value("editorPos", Json::array()));
    return pass;
}

EdgeDecl parseEdgeDecl(const Json& j) {
    EdgeDecl edge;
    edge.id = j.at("id").get<std::string>();
    edge.passId = j.at("passId").get<std::string>();
    edge.slotKey = j.at("slotKey").get<std::string>();
    edge.direction = j.at("direction").get<std::string>();
    edge.resourceId = j.at("resourceId").get<std::string>();
    return edge;
}

OrderedJson toJson(const ResourceDecl& resource) {
    OrderedJson j;
    j["id"] = resource.id;
    j["name"] = resource.name;
    j["kind"] = resource.kind;
    j["type"] = resource.type;
    if (!resource.format.empty()) {
        j["format"] = resource.format;
    }
    if (!resource.size.empty()) {
        j["size"] = resource.size;
    }
    if (!resource.importKey.empty()) {
        j["importKey"] = resource.importKey;
    }
    j["editorPos"] = serializeEditorPos(resource.editorPos);
    return j;
}

OrderedJson toJson(const PassDecl& pass) {
    OrderedJson j;
    j["id"] = pass.id;
    j["name"] = pass.name;
    j["type"] = pass.type;
    j["enabled"] = pass.enabled;
    j["sideEffect"] = pass.sideEffect;
    j["config"] = pass.config;
    j["editorPos"] = serializeEditorPos(pass.editorPos);
    return j;
}

OrderedJson toJson(const EdgeDecl& edge) {
    OrderedJson j;
    j["id"] = edge.id;
    j["passId"] = edge.passId;
    j["slotKey"] = edge.slotKey;
    j["direction"] = edge.direction;
    j["resourceId"] = edge.resourceId;
    return j;
}

} // namespace

std::string generatePipelineAssetGuid() {
    static constexpr char kHex[] = "0123456789abcdef";
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist(0, 15);

    std::string id;
    id.reserve(32);
    for (int i = 0; i < 32; ++i) {
        id.push_back(kHex[dist(rng)]);
    }
    return id;
}

PipelineAsset PipelineAsset::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::error("PipelineAsset: failed to open '{}'", path);
        return {};
    }

    try {
        Json j;
        file >> j;

        const int schemaVersion = j.value("schemaVersion", 0);
        if (schemaVersion != kPipelineAssetSchemaVersion) {
            spdlog::error("PipelineAsset: '{}' uses unsupported schemaVersion {} (expected {})",
                          path,
                          schemaVersion,
                          kPipelineAssetSchemaVersion);
            return {};
        }

        PipelineAsset asset;
        asset.schemaVersion = schemaVersion;
        asset.name = getOrDefault<std::string>(j, "name", {});

        if (j.contains("resources")) {
            for (const auto& resourceJson : j.at("resources")) {
                asset.resources.push_back(parseResourceDecl(resourceJson));
            }
        }

        if (j.contains("passes")) {
            for (const auto& passJson : j.at("passes")) {
                asset.passes.push_back(parsePassDecl(passJson));
            }
        }

        if (j.contains("edges")) {
            for (const auto& edgeJson : j.at("edges")) {
                asset.edges.push_back(parseEdgeDecl(edgeJson));
            }
        }

        return asset;
    } catch (const std::exception& e) {
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

    OrderedJson j;
    j["schemaVersion"] = kPipelineAssetSchemaVersion;
    j["name"] = name;

    OrderedJson resourceArray = OrderedJson::array();
    for (const auto& resource : resources) {
        resourceArray.push_back(toJson(resource));
    }
    j["resources"] = std::move(resourceArray);

    OrderedJson passArray = OrderedJson::array();
    for (const auto& pass : passes) {
        passArray.push_back(toJson(pass));
    }
    j["passes"] = std::move(passArray);

    std::unordered_map<std::string, size_t> passOrder;
    std::unordered_map<std::string, size_t> resourceOrder;
    for (size_t i = 0; i < passes.size(); ++i) {
        passOrder[passes[i].id] = i;
    }
    for (size_t i = 0; i < resources.size(); ++i) {
        resourceOrder[resources[i].id] = i;
    }

    std::vector<EdgeDecl> sortedEdges = edges;
    std::stable_sort(sortedEdges.begin(), sortedEdges.end(), [&](const EdgeDecl& lhs, const EdgeDecl& rhs) {
        const size_t lhsPassOrder = passOrder.count(lhs.passId) ? passOrder[lhs.passId] : passes.size();
        const size_t rhsPassOrder = passOrder.count(rhs.passId) ? passOrder[rhs.passId] : passes.size();
        if (lhsPassOrder != rhsPassOrder) {
            return lhsPassOrder < rhsPassOrder;
        }

        if (lhs.direction != rhs.direction) {
            return lhs.direction < rhs.direction;
        }

        const PassDecl* pass = findPassById(lhs.passId);
        const PassTypeInfo* info = pass ? PassRegistry::instance().getTypeInfo(pass->type) : nullptr;
        auto slotOrder = [&](const EdgeDecl& edge) {
            if (!info) {
                return std::pair<size_t, std::string>{std::numeric_limits<size_t>::max(), edge.slotKey};
            }
            const auto& slots = edge.direction == "input" ? info->inputSlots : info->outputSlots;
            for (size_t i = 0; i < slots.size(); ++i) {
                if (slots[i].key == edge.slotKey) {
                    return std::pair<size_t, std::string>{i, edge.slotKey};
                }
            }
            return std::pair<size_t, std::string>{std::numeric_limits<size_t>::max(), edge.slotKey};
        };

        const auto lhsSlot = slotOrder(lhs);
        const auto rhsSlot = slotOrder(rhs);
        if (lhsSlot != rhsSlot) {
            return lhsSlot < rhsSlot;
        }

        const size_t lhsResOrder = resourceOrder.count(lhs.resourceId) ? resourceOrder[lhs.resourceId] : resources.size();
        const size_t rhsResOrder = resourceOrder.count(rhs.resourceId) ? resourceOrder[rhs.resourceId] : resources.size();
        if (lhsResOrder != rhsResOrder) {
            return lhsResOrder < rhsResOrder;
        }

        return lhs.id < rhs.id;
    });

    OrderedJson edgeArray = OrderedJson::array();
    for (const auto& edge : sortedEdges) {
        edgeArray.push_back(toJson(edge));
    }
    j["edges"] = std::move(edgeArray);

    file << j.dump(2);
    spdlog::info("PipelineAsset: saved '{}'", path);
}

bool PipelineAsset::validate(std::string& errorMsg) const {
    if (schemaVersion != kPipelineAssetSchemaVersion) {
        errorMsg = "Unsupported schemaVersion: " + std::to_string(schemaVersion);
        return false;
    }
    if (name.empty()) {
        errorMsg = "Pipeline name is empty";
        return false;
    }
    if (passes.empty()) {
        errorMsg = "Pipeline has no passes";
        return false;
    }

    std::unordered_map<std::string, const ResourceDecl*> resourceById;
    std::unordered_map<std::string, const PassDecl*> passById;
    std::unordered_set<std::string> edgeIds;

    for (const auto& resource : resources) {
        if (resource.id.empty()) {
            errorMsg = "Pipeline contains a resource with an empty id";
            return false;
        }
        if (!resourceById.emplace(resource.id, &resource).second) {
            errorMsg = "Duplicate resource id: " + resource.id;
            return false;
        }
        if (resource.name.empty()) {
            errorMsg = "Resource '" + resource.id + "' has an empty name";
            return false;
        }
        if (!isKnownResourceKind(resource.kind)) {
            errorMsg = "Resource '" + resource.name + "' has invalid kind '" + resource.kind + "'";
            return false;
        }
        if (resource.kind == "imported" && resource.importKey.empty()) {
            errorMsg = "Imported resource '" + resource.name + "' is missing importKey";
            return false;
        }
        if (resource.kind == "backbuffer" && !resource.importKey.empty()) {
            errorMsg = "Backbuffer resource '" + resource.name + "' must not define importKey";
            return false;
        }
    }

    for (const auto& pass : passes) {
        if (pass.id.empty()) {
            errorMsg = "Pipeline contains a pass with an empty id";
            return false;
        }
        if (!passById.emplace(pass.id, &pass).second) {
            errorMsg = "Duplicate pass id: " + pass.id;
            return false;
        }
        if (pass.name.empty()) {
            errorMsg = "Pass '" + pass.id + "' has an empty name";
            return false;
        }
        if (pass.type.empty()) {
            errorMsg = "Pass '" + pass.name + "' has an empty type";
            return false;
        }
        if (!PassRegistry::instance().getTypeInfo(pass.type)) {
            errorMsg = "Pass '" + pass.name + "' references unknown type '" + pass.type + "'";
            return false;
        }
    }

    std::unordered_map<std::string, int> transientProducerCount;
    std::unordered_map<std::string, int> slotBindingCount;

    for (const auto& edge : edges) {
        if (edge.id.empty()) {
            errorMsg = "Pipeline contains an edge with an empty id";
            return false;
        }
        if (!edgeIds.insert(edge.id).second) {
            errorMsg = "Duplicate edge id: " + edge.id;
            return false;
        }
        if (!isKnownEdgeDirection(edge.direction)) {
            errorMsg = "Edge '" + edge.id + "' has invalid direction '" + edge.direction + "'";
            return false;
        }

        const auto* pass = findPassById(edge.passId);
        if (!pass) {
            errorMsg = "Edge '" + edge.id + "' references unknown pass id '" + edge.passId + "'";
            return false;
        }
        const auto* resource = findResourceById(edge.resourceId);
        if (!resource) {
            errorMsg = "Edge '" + edge.id + "' references unknown resource id '" + edge.resourceId + "'";
            return false;
        }

        const PassTypeInfo* typeInfo = PassRegistry::instance().getTypeInfo(pass->type);
        const auto& slots = edge.direction == "input" ? typeInfo->inputSlots : typeInfo->outputSlots;
        const PassSlotInfo* slotInfo = findSlotInfo(slots, edge.slotKey);
        if (!slotInfo) {
            errorMsg = "Edge '" + edge.id + "' references unknown " + edge.direction +
                       " slot '" + edge.slotKey + "' on pass '" + pass->name + "'";
            return false;
        }
        if (!resourceKindAllowed(*slotInfo, resource->kind)) {
            errorMsg = "Edge '" + edge.id + "' binds resource '" + resource->name +
                       "' with incompatible kind '" + resource->kind + "'";
            return false;
        }

        const std::string slotBindingKey = edge.passId + "|" + edge.direction + "|" + edge.slotKey;
        if (++slotBindingCount[slotBindingKey] > 1) {
            errorMsg = "Pass '" + pass->name + "' has multiple bindings on " + edge.direction +
                       " slot '" + edge.slotKey + "'";
            return false;
        }

        if (pass->enabled && edge.direction == "output" && resource->kind == "transient") {
            if (++transientProducerCount[resource->id] > 1) {
                errorMsg = "Transient resource '" + resource->name + "' is produced by multiple passes";
                return false;
            }
        }
    }

    for (const auto& pass : passes) {
        if (!pass.enabled) {
            continue;
        }

        const PassTypeInfo* typeInfo = PassRegistry::instance().getTypeInfo(pass.type);
        for (const auto& slot : typeInfo->inputSlots) {
            if (!slot.optional && !findEdge(pass.id, "input", slot.key)) {
                errorMsg = "Pass '" + pass.name + "' is missing input slot '" + slot.key + "'";
                return false;
            }
        }
        for (const auto& slot : typeInfo->outputSlots) {
            if (!slot.optional && !findEdge(pass.id, "output", slot.key)) {
                errorMsg = "Pass '" + pass.name + "' is missing output slot '" + slot.key + "'";
                return false;
            }
        }
    }

    const size_t enabledPassCount = std::count_if(
        passes.begin(),
        passes.end(),
        [](const PassDecl& pass) { return pass.enabled; });
    const auto sorted = topologicalSort(false);
    if (sorted.size() != enabledPassCount) {
        errorMsg = "Pipeline contains a cycle";
        return false;
    }

    return true;
}

std::vector<size_t> PipelineAsset::topologicalSort(bool includeDisabled) const {
    std::unordered_map<std::string, size_t> passIndex;
    for (size_t i = 0; i < passes.size(); ++i) {
        if (!includeDisabled && !passes[i].enabled) {
            continue;
        }
        passIndex[passes[i].id] = i;
    }

    std::unordered_map<std::string, size_t> transientProducer;
    std::unordered_map<std::string, std::vector<size_t>> outputWriters;
    for (const auto& edge : edges) {
        if (edge.direction != "output") {
            continue;
        }
        auto passIt = passIndex.find(edge.passId);
        if (passIt == passIndex.end()) {
            continue;
        }
        const ResourceDecl* resource = findResourceById(edge.resourceId);
        if (!resource) {
            continue;
        }
        if (resource->kind == "transient") {
            transientProducer[resource->id] = passIt->second;
        }
        outputWriters[resource->id].push_back(passIt->second);
    }

    std::vector<std::vector<size_t>> adj(passes.size());
    std::vector<size_t> inDegree(passes.size(), 0);

    for (const auto& edge : edges) {
        if (edge.direction != "input") {
            continue;
        }
        auto consumerIt = passIndex.find(edge.passId);
        if (consumerIt == passIndex.end()) {
            continue;
        }
        auto producerIt = transientProducer.find(edge.resourceId);
        if (producerIt != transientProducer.end() && producerIt->second != consumerIt->second) {
            adj[producerIt->second].push_back(consumerIt->second);
            ++inDegree[consumerIt->second];
        }
    }

    for (auto& [resourceId, writers] : outputWriters) {
        const ResourceDecl* resource = findResourceById(resourceId);
        if (!resource || resource->kind == "transient" || writers.size() < 2) {
            continue;
        }
        std::sort(writers.begin(), writers.end());
        writers.erase(std::unique(writers.begin(), writers.end()), writers.end());
        for (size_t i = 1; i < writers.size(); ++i) {
            adj[writers[i - 1]].push_back(writers[i]);
            ++inDegree[writers[i]];
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < passes.size(); ++i) {
        if (!includeDisabled && !passes[i].enabled) {
            continue;
        }
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<size_t> result;
    result.reserve(passIndex.size());
    while (!q.empty()) {
        const size_t passIdx = q.front();
        q.pop();
        result.push_back(passIdx);

        for (const size_t next : adj[passIdx]) {
            if (--inDegree[next] == 0) {
                q.push(next);
            }
        }
    }

    return result;
}

ResourceDecl* PipelineAsset::findResourceById(const std::string& id) {
    for (auto& resource : resources) {
        if (resource.id == id) {
            return &resource;
        }
    }
    return nullptr;
}

const ResourceDecl* PipelineAsset::findResourceById(const std::string& id) const {
    for (const auto& resource : resources) {
        if (resource.id == id) {
            return &resource;
        }
    }
    return nullptr;
}

PassDecl* PipelineAsset::findPassById(const std::string& id) {
    for (auto& pass : passes) {
        if (pass.id == id) {
            return &pass;
        }
    }
    return nullptr;
}

const PassDecl* PipelineAsset::findPassById(const std::string& id) const {
    for (const auto& pass : passes) {
        if (pass.id == id) {
            return &pass;
        }
    }
    return nullptr;
}

EdgeDecl* PipelineAsset::findEdgeById(const std::string& id) {
    for (auto& edge : edges) {
        if (edge.id == id) {
            return &edge;
        }
    }
    return nullptr;
}

const EdgeDecl* PipelineAsset::findEdgeById(const std::string& id) const {
    for (const auto& edge : edges) {
        if (edge.id == id) {
            return &edge;
        }
    }
    return nullptr;
}

EdgeDecl* PipelineAsset::findEdge(const std::string& passId,
                                  const std::string& direction,
                                  const std::string& slotKey) {
    for (auto& edge : edges) {
        if (edge.passId == passId && edge.direction == direction && edge.slotKey == slotKey) {
            return &edge;
        }
    }
    return nullptr;
}

const EdgeDecl* PipelineAsset::findEdge(const std::string& passId,
                                        const std::string& direction,
                                        const std::string& slotKey) const {
    for (const auto& edge : edges) {
        if (edge.passId == passId && edge.direction == direction && edge.slotKey == slotKey) {
            return &edge;
        }
    }
    return nullptr;
}

void PipelineAsset::removeEdgesForPass(const std::string& passId) {
    edges.erase(
        std::remove_if(edges.begin(), edges.end(), [&](const EdgeDecl& edge) { return edge.passId == passId; }),
        edges.end());
}

void PipelineAsset::removeEdgesForResource(const std::string& resourceId) {
    edges.erase(
        std::remove_if(edges.begin(), edges.end(), [&](const EdgeDecl& edge) { return edge.resourceId == resourceId; }),
        edges.end());
}
