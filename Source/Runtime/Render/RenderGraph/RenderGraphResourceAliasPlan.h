#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace metallic::render::detail {

inline constexpr size_t kNoGraphResourceAliasSlot = std::numeric_limits<size_t>::max();

// Resource is the caller's canonical logical resource index. The uses include
// its producer and every consumer, including inputs that alias the same output.
// Eligibility is decided by the caller's lifetime, initialization and export
// contracts, and excludes requirements that mandate a dedicated allocation.
struct GraphResourceAliasCandidate {
    size_t resource = 0;
    std::string name;
    std::vector<size_t> uses;
    uint64_t sizeBytes = 0;
    uint64_t alignment = 0;
    uint32_t memoryTypeBits = 0;
    bool eligible = false;
};

struct GraphResourceAliasDependency {
    size_t predecessor = 0;
    size_t successor = 0;
    bool operator==(const GraphResourceAliasDependency&) const = default;
};

// All members bind offset zero in one allocation. Resources are ordered by
// their proven GPU lifetimes, rather than by their names or allocation sizes.
struct GraphResourceAliasSlot {
    std::vector<size_t> resources;
    uint64_t sizeBytes = 0;
    uint64_t alignment = 0;
    uint32_t memoryTypeBits = 0;
};

struct GraphResourceAliasHandoff {
    size_t beforeResource = 0;
    size_t afterResource = 0;
    // The previous occupant's maximal uses. The executor must resolve each
    // pass to its completed join segment, including any GPU fork/branches.
    std::vector<size_t> predecessors;
    size_t activationPass = 0;
};

struct GraphResourceAliasPlan {
    std::vector<GraphResourceAliasSlot> slots;
    // Indexed by the input candidate's position; excluded candidates have
    // kNoGraphResourceAliasSlot. Slot members retain the caller's resource IDs.
    std::vector<size_t> resourceSlots;
    std::vector<GraphResourceAliasHandoff> handoffs;
    // Existing semantic edges used by the ordering proof. The executor must
    // encode them as GPU predecessors even when no logical access hazard
    // would generate a barrier. Empty when no slot has multiple occupants.
    std::vector<GraphResourceAliasDependency> safetyDependencies;
};

// Pure CPU planning, independent of queue selection or native allocation.
// Uses may be unordered or repeated; they are normalized. Each eligible
// candidate must have one producer use that precedes all its consumer uses.
// Only a strict happens-before relation permits reuse: CPU pass indices, queue
// hints and DAG depth never introduce ordering between independent branches.
Result<GraphResourceAliasPlan> buildGraphResourceAliasPlan(size_t passCount,
    std::span<const GraphResourceAliasCandidate> candidates,
    std::span<const GraphResourceAliasDependency> dependencies);

} // namespace metallic::render::detail
