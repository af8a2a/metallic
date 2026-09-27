#include "Requirements.h"

#include <algorithm>

namespace metallic::tests::bench {

const char* name(Capability value)
{
    switch (value) {
    case Capability::ShaderObject: return "shaderObject";
    case Capability::TimestampQueries: return "timestampQueries";
    case Capability::Bindless: return "bindlessDescriptorHeap";
    case Capability::IndependentCopy: return "independentCopyQueue";
    case Capability::IndependentCompute: return "independentComputeQueue";
    }
    return "unknown";
}

const char* name(Status value)
{
    switch (value) {
    case Status::Pass: return "Pass";
    case Status::Fail: return "Fail";
    case Status::SkipUnsupported: return "SkipUnsupported";
    case Status::SkipNotEnabled: return "SkipNotEnabled";
    case Status::EnvironmentFailure: return "EnvironmentFailure";
    case Status::InfrastructureFailure: return "InfrastructureFailure";
    case Status::Timeout: return "Timeout";
    case Status::Crash: return "Crash";
    case Status::DeviceLost: return "DeviceLost";
    }
    return "Unknown";
}

const char* name(Layer value)
{
    switch (value) {
    case Layer::Rhi: return "Rhi";
    case Layer::Core: return "Core";
    case Layer::RenderGraph: return "RenderGraph";
    case Layer::Harness: return "Harness";
    }
    return "Unknown";
}

bool failed(Status value)
{
    return value != Status::Pass && value != Status::SkipUnsupported && value != Status::SkipNotEnabled;
}

render::Result<Profile> profile(std::string id, Validation validation)
{
    if (id != "core" && id != "binding" && id != "async") {
        return render::makeError(render::Error::InvalidArgument);
    }
    Profile value;
    value.id = std::move(id);
    value.desc.applicationName = "Metallic Testbench";
    value.desc.enableValidation = validation == Validation::Core;
    value.desc.enableBindlessDescriptorHeap = value.id != "core";
    value.desc.enableAsyncCompute = value.id == "async";
    value.desc.enableOpacityMicromap = false;
    value.desc.enableRayTracingPositionFetch = false;
    value.desc.enableDeviceGeneratedCommands = false;
    value.desc.preferUnifiedImageLayouts = false;
    return value;
}

bool enabled(Capability capability, const render::DeviceCapabilities& caps)
{
    switch (capability) {
    case Capability::ShaderObject: return caps.shaderObject;
    case Capability::TimestampQueries: return caps.timestampQueries;
    case Capability::Bindless: return caps.bindlessDescriptorHeap;
    case Capability::IndependentCopy: return caps.independentCopyQueue;
    case Capability::IndependentCompute: return caps.independentComputeQueue;
    }
    return false;
}

bool requested(Capability capability, const Profile& value)
{
    switch (capability) {
    case Capability::Bindless: return value.desc.enableBindlessDescriptorHeap;
    case Capability::IndependentCompute: return value.desc.enableAsyncCompute;
    default: return true;
    }
}

Verdict evaluate(const Requirements& requirements, const Profile& value,
    const render::DeviceCapabilities& caps, const std::vector<render::QueueType>& queues,
    bool validationActive)
{
    if (requirements.validation == Validation::Core && !validationActive) {
        return {Status::EnvironmentFailure, "required validation layer/messenger is not active"};
    }
    for (const auto capability : requirements.capabilities) {
        if (!requested(capability, value)) {
            return {Status::SkipNotEnabled, name(capability)};
        }
        if (!enabled(capability, caps)) {
            return {Status::SkipUnsupported, name(capability)};
        }
    }
    for (const auto queue : requirements.queues) {
        if (std::find(queues.begin(), queues.end(), queue) == queues.end()) {
            return {Status::SkipUnsupported, "required queue is unavailable"};
        }
    }
    return {};
}

} // namespace metallic::tests::bench
