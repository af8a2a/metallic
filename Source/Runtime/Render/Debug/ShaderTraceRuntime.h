#pragma once
#include "Runtime/Debug/DebugCore.h"
#include "Runtime/Debug/ShaderTraceCore.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.h"

namespace metallic::render {
// Owner-thread bridge. One dedicated Printf session per observation in P1.
// The render adapter retains all GPU objects until completion AND backend closure.
class ShaderTraceRuntime {
public:
    explicit ShaderTraceRuntime(debug::DebugCore& core) : core_(core), trace_(core.session()) {}
    debug::DebugResult<void> begin(debug::DebugCaptureRequest request, debug::DebugValue site, debug::DebugValue identity);
    const debug::DebugValue& plan() const { return plan_; }
    bool maySubmit();
    void compiledVariant(debug::DebugValue variant) { trace_.compiledVariant(std::move(variant)); }
    void stop(std::string reason) { trace_.stop(std::move(reason)); }
    void submitted(debug::DebugValue queue, debug::DebugValue submit);
    void finish(vulkan::ShaderPrintf& printf, bool gpuComplete, bool backendClosed, bool readbackValid,
        debug::DebugValue evidence);
private:
    debug::DebugCore& core_;
    debug::ShaderTraceCore trace_;
    debug::DebugCaptureRequest request_;
    debug::DebugValue plan_;
};
} // namespace metallic::render
