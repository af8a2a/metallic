#pragma once
#include "ShaderTraceRuntime.h"
#include "Runtime/Render/RenderFrameContext.h"
#include <filesystem>
#include <optional>

namespace metallic::render {
// Batch production adapter. One watch per process; seals after instance destruction.
// The shader lease is retained by the actual command recording through completion.
class WorkControlShaderTrace {
public:
    explicit WorkControlShaderTrace(debug::DebugCore& core, const std::filesystem::path& output);
    ~WorkControlShaderTrace();
    vulkan::ShaderPrintf& capture() { return capture_; }
    void qualify(Device& device, Queue& queue, const std::filesystem::path& output);
    void configure(debug::DebugValue graph);
    void prepare(Device& device, const debug::DebugValue& specification, const debug::DebugValue& input);
    void beginExecution(debug::DebugEvidenceStamp stamp);
    bool bind(CommandBuffer& commands, std::string_view pass, const debug::DebugValue& production);
    void targetDrained(Queue& queue);
    void restoration(debug::DebugValue evidence, bool readbackValid);
    void abort(std::string reason);
    void releaseGpu();
    void finishAfterDevice() noexcept;
    const debug::DebugValue& variant() const { return evidence_.at("variant"); }
    const debug::DebugValue& sites() const { return sites_; }
    const debug::DebugValue& plan() const { return runtime_.plan(); }
    const std::string& job() const { return job_; }
private:
    struct Lease;
    debug::DebugCore& core_;
    ShaderTraceRuntime runtime_;
    vulkan::ShaderPrintf capture_{{.settingsViaFile=true}};
    debug::DebugValue graph_, sites_, request_, evidence_;
    debug::DebugEvidenceStamp execution_;
    std::shared_ptr<Lease> lease_;
    GpuCompletionPoint completion_;
    std::filesystem::path output_;
    std::string job_, phase_;
    std::optional<std::string> previousSettings_;
    bool armed_ = false, bound_ = false, gpuComplete_ = false, readbackValid_ = false;
};
} // namespace metallic::render
