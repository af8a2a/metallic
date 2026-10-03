#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <filesystem>
#include <json.hpp>
#include <span>
#include <mutex>

namespace metallic::render::profiling {

bool workControlReplayRequested();
// Diagnostic-only serialization of all RHI submissions, including other queues.
std::recursive_mutex& workControlReplaySubmissionMutex();

struct WorkControlReplayBinding {
    std::string name;
    Buffer* buffer = nullptr;
    uint32_t shaderIndex = UINT32_MAX;
};

// Diagnostic, owner-thread-only capture. The graph owner must drain the target
// frame and keep it alive while run() uses the retained production executable.
// Only scratch allocations are bound by replay; production is never restored
// from replay outputs and no publication callbacks are recorded.
class WorkControlReplay final {
public:
    WorkControlReplay(Device& device, std::string phase, std::filesystem::path output);
    ~WorkControlReplay();
    void arm();
    static WorkControlReplay* selected(std::string_view phase);
    void before(CommandBuffer& commands, ComputePipeline& pipeline, BindlessHeap& heap,
        std::span<const WorkControlReplayBinding> bindings, Buffer& arguments,
        const void* push, uint32_t pushBytes, nlohmann::json identity);
    void after(CommandBuffer& commands);
    nlohmann::json run(Queue& queue, const nlohmann::json& frozenIdentity);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::profiling
