#include "ShaderTraceRuntime.h"

namespace metallic::render {
debug::DebugResult<void> ShaderTraceRuntime::begin(debug::DebugCaptureRequest request,
    debug::DebugValue site, debug::DebugValue identity)
{
    if (!core_.reserve(request.id, debug::kShaderTraceArtifactBudget)) {
        return std::unexpected(debug::DebugError{"BudgetExceeded", "Shader job cancelled or trace reservation unavailable"});
    }
    auto plan = trace_.begin(request.specification, std::move(site), std::move(identity));
    if (!plan) { core_.fail(request.id, plan.error()); return std::unexpected(plan.error()); }
    request_ = std::move(request); plan_ = std::move(*plan);
    core_.transition(request_.id, "Compiling");
    return {};
}

bool ShaderTraceRuntime::maySubmit()
{
    core_.expire();
    if (core_.cancelled(request_.id)) { trace_.stop("Cancelled"); return false; }
    const auto graph = core_.dispatch({{"method", "rg.describe"}}).at("result");
    if (graph.value("id", "") != request_.graph || graph.value("generation", uint64_t(0)) != request_.generation) {
        trace_.stop("StaleHandle"); return false;
    }
    core_.transition(request_.id, "Recording");
    return true;
}

void ShaderTraceRuntime::submitted(debug::DebugValue queue, debug::DebugValue submit)
{
    trace_.submitted(std::move(queue), std::move(submit));
    core_.transition(request_.id, "Submitted");
}

void ShaderTraceRuntime::finish(vulkan::ShaderPrintf& printf, bool gpuComplete, bool backendClosed,
    bool readbackValid, debug::DebugValue evidence)
{
    core_.transition(request_.id, "Collecting");
    for (const auto& message : printf.drain()) {
        trace_.ingest({{"id",message.id}, {"idName",message.idName.data()}, {"severity",message.severity},
            {"text",message.text.data()}, {"truncated",message.truncated}});
    }
    trace_.accountLoss(printf.dropped(), printf.truncated());
    if (evidence.contains("error") && !core_.cancelled(request_.id)) { trace_.stop("ExecutionFailed"); }
    trace_.completion(gpuComplete, backendClosed, readbackValid);
    auto bundle = trace_.seal();
    if (!bundle) { throw std::logic_error(bundle.error().message); }
    (*bundle)["runtime"] = std::move(evidence);
    core_.completeShaderTrace(request_.id, std::move(*bundle));
}
} // namespace metallic::render
