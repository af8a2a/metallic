#include "MaterialClosureIR.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <json.hpp>

namespace metallic::render {
namespace {
void require(bool condition, const char* message)
{
    if (!condition) { throw std::invalid_argument(message); }
}
uint32_t add(uint32_t a, uint32_t b)
{
    require(b <= std::numeric_limits<uint32_t>::max() - a, "Closure IR complexity overflow");
    return a + b;
}
} // namespace

MaterialClosureIR MaterialClosureIR::create(std::span<const MaterialClosureNode> nodes, uint32_t root)
{
    require(!nodes.empty() && nodes.size() <= 4096 && root < nodes.size(), "Invalid Closure IR size/root (safety limit 4096 nodes)");
    MaterialClosureIR result;
    std::vector<uint32_t> remap(nodes.size(), ~0u);
    // Requiring children before parents rejects cycles without recursion.
    std::vector<bool> live(nodes.size());
    live[root] = true;
    for (size_t i = root + 1; i-- > 0;) {
        if (!live[i]) { continue; }
        const auto& node = nodes[i];
        require(node.op == MaterialClosureOp::Slab || node.op == MaterialClosureOp::Mix || node.op == MaterialClosureOp::Layer,
            "Unknown Closure IR operator");
        if (node.op != MaterialClosureOp::Slab) {
            for (uint32_t child : node.operands) {
                require(child < i, "Closure IR operands must precede their parent");
                live[child] = true;
            }
        }
    }
    std::vector<MaterialClosureComplexity> metrics;
    std::set<uint32_t> bases;
    nlohmann::json encoded = nlohmann::json::array();
    // Canonical child-first traversal removes incidental authoring order and
    // node IDs while retaining ordered Mix A/B and Layer top/bottom semantics.
    std::vector<std::pair<uint32_t, bool>> pending{{root, false}};
    std::vector<uint32_t> order;
    std::vector<bool> visited(nodes.size());
    while (!pending.empty()) {
        const auto [id, expanded] = pending.back(); pending.pop_back();
        if (visited[id]) { continue; }
        if (!expanded && nodes[id].op != MaterialClosureOp::Slab) {
            pending.emplace_back(id, true);
            pending.emplace_back(nodes[id].operands[1], false);
            pending.emplace_back(nodes[id].operands[0], false);
        } else {
            visited[id] = true;
            order.push_back(id);
        }
    }
    for (uint32_t i : order) {
        auto node = nodes[i];
        MaterialClosureComplexity metric;
        if (node.op == MaterialClosureOp::Slab) {
            for (size_t c = 0; c < 3; ++c) {
                require(std::isfinite(node.slab.reflectance[c]) && node.slab.reflectance[c] >= 0 && node.slab.reflectance[c] <= 1,
                    "Slab reflectance must be finite and in [0,1]");
                require(std::isfinite(node.slab.opticalDepth[c]) && node.slab.opticalDepth[c] >= 0 && node.slab.opticalDepth[c] <= 1e6f,
                    "Slab optical depth must be finite and in [0,1e6]");
                if (node.slab.reflectance[c] == 0) { node.slab.reflectance[c] = 0; }
                if (node.slab.opticalDepth[c] == 0) { node.slab.opticalDepth[c] = 0; }
            }
            node.slab.reflectance[3] = node.slab.opticalDepth[3] = 0;
            node.operands = {}; node.weight = 0;
            metric.closureRecordCount = metric.scatteringLobeCount = 1;
            bases.insert(node.normalBasis);
            encoded.push_back({0, node.slab.reflectance, node.slab.opticalDepth, node.normalBasis});
        } else {
            for (auto& child : node.operands) { child = remap[child]; }
            const auto& a = metrics[node.operands[0]];
            const auto& b = metrics[node.operands[1]];
            metric.closureRecordCount = add(a.closureRecordCount, b.closureRecordCount);
            metric.scatteringLobeCount = add(a.scatteringLobeCount, b.scatteringLobeCount);
            metric.operatorCount = add(add(a.operatorCount, b.operatorCount), 1);
            metric.layerDepth = add(std::max(a.layerDepth, b.layerDepth), node.op == MaterialClosureOp::Layer ? 1 : 0);
            if (node.op == MaterialClosureOp::Mix) {
                require(std::isfinite(node.weight) && node.weight >= 0 && node.weight <= 1, "Mix weight must be finite and in [0,1]");
                if (node.weight == 0) { node.weight = 0; }
            } else { node.weight = 0; }
            node.slab = {}; node.normalBasis = 0;
            encoded.push_back({static_cast<uint32_t>(node.op), node.operands, node.weight});
        }
        remap[i] = static_cast<uint32_t>(result.nodes_.size());
        result.nodes_.push_back(node);
        metrics.push_back(metric);
    }
    result.complexity_ = metrics.back();
    result.complexity_.normalBasisCount = static_cast<uint32_t>(bases.size());
    result.canonical_ = nlohmann::json{{"closureIR", 1}, {"nodes", encoded}}.dump();
    result.hash_ = 14695981039346656037ull;
    for (unsigned char c : result.canonical_) { result.hash_ = (result.hash_ ^ c) * 1099511628211ull; }
    return result;
}

LoweredMaterialClosure lowerMaterialClosure(const MaterialClosureIR& ir, const RealtimeBackendProfile& profile)
{
    auto metric = ir.complexity();
    require(metric.closureRecordCount <= profile.maxClosures, "RealtimeBackendProfile: maxClosures exceeded");
    require(metric.operatorCount <= profile.maxOperators, "RealtimeBackendProfile: maxOperators exceeded");
    require(metric.normalBasisCount <= profile.maxNormalBases, "RealtimeBackendProfile: maxNormalBases exceeded");
    require(metric.layerDepth <= profile.maxLayerDepth, "RealtimeBackendProfile: maxLayerDepth exceeded");
    const auto nodes = ir.nodes();
    const auto& root = nodes[ir.root()];
    LoweredMaterialClosure result{};
    if (root.op == MaterialClosureOp::Slab) {
        require(root.normalBasis == 0, "Slab prototype supports only context normal basis 0");
        result.family = MaterialClosureFamily::SingleSlabClosure;
        result.packet.first = root.slab;
        metric.payloadBytes = 48; // SlabRecord + float4 resolved normal.
    } else {
        const auto& a = nodes[root.operands[0]];
        const auto& b = nodes[root.operands[1]];
        require(a.op == MaterialClosureOp::Slab && b.op == MaterialClosureOp::Slab,
            "Slab prototype has no executable family for nested operators, even with a larger profile");
        require(a.normalBasis == 0 && b.normalBasis == 0, "Dual slab prototype supports only shared context normal basis 0");
        result.family = MaterialClosureFamily::DualSlabClosure;
        result.packet.first = a.slab;
        result.packet.second = b.slab;
        result.packet.control = {static_cast<float>(root.op), root.weight, 0, 0};
        metric.payloadBytes = 96; // Two records + float4 normal + float4 control.
    }
    require(metric.payloadBytes <= profile.maxPayloadBytes, "RealtimeBackendProfile: maxPayloadBytes exceeded");
    result.complexity = metric;
    return result;
}
} // namespace metallic::render
