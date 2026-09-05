#pragma once

#include "Runtime/Debug/DebugTransport.h"
#include "Runtime/Render/RenderFrameContext.h"

#include <atomic>
#include <functional>
#include <unordered_map>

namespace metallic::render {
class RenderSubsystemHost;
class RenderGraphExecutionContext;
class GPUSceneSubsystem;
class MeshletStreamRuntime;
class ComputeProgram;
struct GPUSceneViewTag;
template <typename Tag> struct GPUSceneId;

// A borrow valid only during the boundary callback. Copies restore exactly this
// owner's state. No binding (or Vulkan pointer) enters DebugCore or the IPC queue.
struct DebugResourceBinding {
    std::string id;
    Buffer* buffer = nullptr;
    Texture* texture = nullptr;
    ResourceState state = ResourceState::Undefined;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::string layout = "u32";
    uint64_t allocation = 0;
    debug::DebugValue metadata = debug::DebugValue::object();
};

class IRenderDebugObserver {
public:
    virtual ~IRenderDebugObserver() = default;
    virtual void compiled(debug::DebugValue graph) = 0;
    virtual void beginExecution(Device& device, debug::DebugEvidenceStamp evidence,
        RenderSubsystemHost* subsystems) = 0;
    virtual void boundary(CommandBuffer& commands, std::string_view checkpoint,
        uint32_t passId, std::string_view pass, std::span<const DebugResourceBinding> resources,
        const debug::DebugValue& values) = 0;
    virtual void endExecution(bool success) = 0;
};

class RenderDebugRuntime final : public IRenderDebugObserver {
public:
    explicit RenderDebugRuntime(debug::DebugLimits limits = {});
    ~RenderDebugRuntime();
    debug::DebugCore& core() { return core_; }
    debug::DebugResult<void> start();
    void stop();
    ValidationSink validationSink();
    // Owner-thread polling. GPU memory is not mapped before completion.
    void poll();
    void drain();
    void compiled(debug::DebugValue graph) override;
    void beginExecution(Device& device, debug::DebugEvidenceStamp evidence,
        RenderSubsystemHost* subsystems) override;
    void boundary(CommandBuffer& commands, std::string_view checkpoint,
        uint32_t passId, std::string_view pass, std::span<const DebugResourceBinding> resources,
        const debug::DebugValue& values) override;
    void endExecution(bool success) override;
    const std::unordered_map<std::string, debug::DebugTypeDesc>& layouts() const { return layouts_; }

private:
    struct Execution;
    struct Readback;
    struct TaskSink;
    void capture(CommandBuffer& commands, const debug::DebugCaptureRequest& request,
        std::span<const DebugResourceBinding> resources, const debug::DebugSnapshot& snapshot);
    debug::DebugCore core_;
    debug::DebugServer server_;
    std::unordered_map<std::string, debug::DebugTypeDesc> layouts_;
    debug::DebugValue graph_;
    Device* device_ = nullptr;
    RenderSubsystemHost* subsystems_ = nullptr;
    std::shared_ptr<Execution> current_;
    std::vector<std::shared_ptr<Execution>> executions_;
    std::vector<std::shared_ptr<Readback>> readbacks_;
    std::shared_ptr<TaskSink> taskSink_;
    uint64_t taskSubscription_ = 0;
    uint64_t nextSample_ = 1;
    std::unique_ptr<ComputeProgram> probeProgram_;
};

std::unordered_map<std::string, debug::DebugTypeDesc> renderDebugLayouts();
void gpuDrivenDebugCheckpoint(RenderGraphExecutionContext& context, std::string_view checkpoint,
    GPUSceneSubsystem* gpuScene, GPUSceneId<GPUSceneViewTag> view, uint32_t frameSlot,
    MeshletStreamRuntime* streaming, uint32_t phase, uint64_t streamVisibleRecordBase = 0);

} // namespace metallic::render
