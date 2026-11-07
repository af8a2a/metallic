#pragma once

#include "Runtime/Task/TaskGraph.h"

namespace metallic::scene {

// The owner keeps the primitive alive and exclusively owned until destruction.
// Add one stage, wait for its completion through graph dependencies, then call
// advance() on the coordinator. No stage blocks a worker waiting for its peers.
class DeferredMeshletBuild {
public:
    virtual ~DeferredMeshletBuild() = default;

    virtual bool complete() const = 0;
    virtual std::expected<task::TaskNodeHandle, task::TaskError> addStage(
        task::TaskGraph& graph, task::TaskDesc desc) = 0;
    virtual void advance() = 0;
};

} // namespace metallic::scene
