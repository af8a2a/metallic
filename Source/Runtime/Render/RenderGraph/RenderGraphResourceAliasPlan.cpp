#include "Runtime/Render/RenderGraph/RenderGraphResourceAliasPlan.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_set>

namespace metallic::render::detail {

Result<GraphResourceAliasPlan> buildGraphResourceAliasPlan(size_t passCount,
    std::span<const GraphResourceAliasCandidate> candidates,
    std::span<const GraphResourceAliasDependency> dependencies)
{
    std::vector<GraphResourceAliasDependency> semanticEdges(dependencies.begin(), dependencies.end());
    std::sort(semanticEdges.begin(), semanticEdges.end(), [](const auto& left, const auto& right) {
        if (left.predecessor != right.predecessor) { return left.predecessor < right.predecessor; }
        return left.successor < right.successor;
    });
    semanticEdges.erase(std::unique(semanticEdges.begin(), semanticEdges.end()), semanticEdges.end());

    std::vector<std::vector<size_t>> successors(passCount);
    std::vector<size_t> indegrees(passCount, 0);
    for (const auto& edge : semanticEdges) {
        if (edge.predecessor >= passCount || edge.successor >= passCount || edge.predecessor == edge.successor) {
            return makeError(Error::InvalidArgument);
        }
        successors[edge.predecessor].push_back(edge.successor);
        ++indegrees[edge.successor];
    }

    // A stable topological order also makes normalization deterministic for a
    // caller whose pass IDs do not themselves follow the graph's GPU order.
    std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> ready;
    for (size_t pass = 0; pass < passCount; ++pass) {
        if (indegrees[pass] == 0) { ready.push(pass); }
    }
    std::vector<size_t> topologicalOrder;
    topologicalOrder.reserve(passCount);
    while (!ready.empty()) {
        const size_t pass = ready.top();
        ready.pop();
        topologicalOrder.push_back(pass);
        for (const size_t successor : successors[pass]) {
            if (--indegrees[successor] == 0) { ready.push(successor); }
        }
    }
    if (topologicalOrder.size() != passCount) { return makeError(Error::InvalidArgument); }

    std::vector<size_t> orderIndices(passCount);
    for (size_t index = 0; index < topologicalOrder.size(); ++index) {
        orderIndices[topologicalOrder[index]] = index;
    }
    const size_t wordCount = passCount / 64 + (passCount % 64 != 0 ? 1 : 0);
    std::vector<std::vector<uint64_t>> reachability(passCount, std::vector<uint64_t>(wordCount, 0));
    for (auto pass = topologicalOrder.rbegin(); pass != topologicalOrder.rend(); ++pass) {
        for (const size_t successor : successors[*pass]) {
            reachability[*pass][successor / 64] |= uint64_t{1} << (successor % 64);
            for (size_t word = 0; word < wordCount; ++word) {
                reachability[*pass][word] |= reachability[successor][word];
            }
        }
    }
    const auto precedes = [&](size_t before, size_t after) {
        return (reachability[before][after / 64] & (uint64_t{1} << (after % 64))) != 0;
    };

    GraphResourceAliasPlan result;
    result.resourceSlots.assign(candidates.size(), kNoGraphResourceAliasSlot);
    std::vector<std::vector<size_t>> uses(candidates.size());
    std::vector<size_t> eligible;
    std::unordered_set<size_t> resourceIds;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const auto& candidate = candidates[index];
        if (!resourceIds.insert(candidate.resource).second) { return makeError(Error::InvalidArgument); }
        if (!candidate.eligible) { continue; }
        if (!candidate.sizeBytes || !candidate.alignment ||
            (candidate.alignment & (candidate.alignment - 1)) != 0 ||
            !candidate.memoryTypeBits || candidate.uses.empty()) {
            return makeError(Error::InvalidArgument);
        }
        auto& normalized = uses[index];
        normalized = candidate.uses;
        for (const size_t pass : normalized) {
            if (pass >= passCount) { return makeError(Error::InvalidArgument); }
        }
        std::sort(normalized.begin(), normalized.end(), [&](size_t left, size_t right) {
            return orderIndices[left] < orderIndices[right];
        });
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        const size_t producer = normalized.front();
        for (size_t use = 1; use < normalized.size(); ++use) {
            if (!precedes(producer, normalized[use])) { return makeError(Error::InvalidArgument); }
        }
        eligible.push_back(index);
    }

    const auto resourcePrecedes = [&](size_t before, size_t after) {
        for (const size_t beforeUse : uses[before]) {
            for (const size_t afterUse : uses[after]) {
                if (!precedes(beforeUse, afterUse)) { return false; }
            }
        }
        return true;
    };
    std::sort(eligible.begin(), eligible.end(), [&](size_t left, size_t right) {
        const auto& before = candidates[left];
        const auto& after = candidates[right];
        if (before.sizeBytes != after.sizeBytes) { return before.sizeBytes > after.sizeBytes; }
        if (before.name != after.name) { return before.name < after.name; }
        return before.resource < after.resource;
    });

    std::vector<std::vector<size_t>> slotCandidates;
    for (const size_t index : eligible) {
        const auto& candidate = candidates[index];
        size_t bestSlot = kNoGraphResourceAliasSlot;
        uint64_t bestWaste = std::numeric_limits<uint64_t>::max();
        for (size_t slotIndex = 0; slotIndex < result.slots.size(); ++slotIndex) {
            const auto& slot = result.slots[slotIndex];
            if ((slot.memoryTypeBits & candidate.memoryTypeBits) == 0) { continue; }
            const bool compatible = std::all_of(slotCandidates[slotIndex].begin(), slotCandidates[slotIndex].end(),
                [&](size_t member) { return resourcePrecedes(member, index) || resourcePrecedes(index, member); });
            if (!compatible) { continue; }
            const uint64_t waste = std::max(slot.sizeBytes, candidate.sizeBytes) - candidate.sizeBytes;
            if (bestSlot == kNoGraphResourceAliasSlot || waste < bestWaste) {
                bestSlot = slotIndex;
                bestWaste = waste;
            }
        }
        if (bestSlot == kNoGraphResourceAliasSlot) {
            bestSlot = result.slots.size();
            result.slots.push_back({.sizeBytes = candidate.sizeBytes,
                .alignment = candidate.alignment, .memoryTypeBits = candidate.memoryTypeBits});
            slotCandidates.emplace_back();
        } else {
            auto& slot = result.slots[bestSlot];
            slot.sizeBytes = std::max(slot.sizeBytes, candidate.sizeBytes);
            slot.alignment = std::max(slot.alignment, candidate.alignment);
            slot.memoryTypeBits &= candidate.memoryTypeBits;
        }
        result.resourceSlots[index] = bestSlot;
        slotCandidates[bestSlot].push_back(index);
    }

    for (size_t slotIndex = 0; slotIndex < slotCandidates.size(); ++slotIndex) {
        auto& members = slotCandidates[slotIndex];
        // Every pair was checked above, so this is a strict total order within
        // the slot, even when separate slots contain concurrent branches.
        std::sort(members.begin(), members.end(), resourcePrecedes);
        auto& slot = result.slots[slotIndex];
        for (const size_t member : members) { slot.resources.push_back(candidates[member].resource); }
        for (size_t member = 1; member < members.size(); ++member) {
            const size_t before = members[member - 1];
            const size_t after = members[member];
            GraphResourceAliasHandoff handoff{
                .beforeResource = candidates[before].resource,
                .afterResource = candidates[after].resource,
                .activationPass = uses[after].front(),
            };
            for (const size_t beforeUse : uses[before]) {
                const bool maximal = std::none_of(uses[before].begin(), uses[before].end(),
                    [&](size_t other) { return precedes(beforeUse, other); });
                if (maximal) { handoff.predecessors.push_back(beforeUse); }
            }
            std::sort(handoff.predecessors.begin(), handoff.predecessors.end());
            result.handoffs.push_back(std::move(handoff));
        }
    }
    if (!result.handoffs.empty()) { result.safetyDependencies = std::move(semanticEdges); }
    return result;
}

} // namespace metallic::render::detail
