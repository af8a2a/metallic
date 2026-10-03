#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Requirements.h"

#include <algorithm>
#include <stdexcept>

namespace metallic::tests::bench {

const char* name(Capability value)
{
    switch (value) {
    case Capability::ShaderObject: return "shaderObject";
    case Capability::TimestampQueries: return "timestampQueries";
    case Capability::Bindless: return "bindlessDescriptorHeap";
    case Capability::IndependentCopy: return "independentCopyQueue";
    case Capability::IndependentCompute: return "independentComputeQueue";
    case Capability::RayQuery: return "rayQuery";
    case Capability::PositionFetch: return "positionFetch";
    case Capability::OpacityMicromap: return "opacityMicromap";
    case Capability::UnifiedLayouts: return "unifiedLayouts";
    case Capability::PartitionedAS: return "partitionedAS";
    case Capability::ClusterAS: return "clusterAS";
    case Capability::GeneratedCommands: return "dgc";
    case Capability::MemoryDecompression: return "memoryDecompression";

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
    case Layer::RHI: return "RHI";
    case Layer::Core: return "Core";
    case Layer::RenderGraph: return "RenderGraph";
    case Layer::Backend: return "Backend";
    case Layer::Harness: return "Harness";
    }
    return "Unknown";
}

const char* name(Validation value)
{
    switch (value) {
    case Validation::Off: return "off";
    case Validation::Core: return "core";
    case Validation::Synchronization: return "sync";
    }
    return "unknown";
}

Validation parseValidation(const std::string& value)
{
    if (value == "off") { return Validation::Off; }
    if (value == "core") { return Validation::Core; }
    if (value == "sync") { return Validation::Synchronization; }
    throw std::invalid_argument("validation must be off, core or sync");
}

bool failed(Status value)
{
    return value != Status::Pass && value != Status::SkipUnsupported && value != Status::SkipNotEnabled;
}

render::Result<Profile> profile(std::string id, Validation validation)
{
    if (id != "core" && id != "binding" && id != "async" && id != "core-unified" &&
        id != "ray-query" && id != "ray-query-position" && id != "ray-query-omm" &&
        id != "ray-query-ptlas" && id != "ray-query-clas" && id != "binding-dgc" && id != "decompression") {
        return render::makeError(render::Error::InvalidArgument);
    }
    Profile value;
    value.id = std::move(id);
    value.desc.applicationName = "Metallic Testbench";
    value.desc.enableValidation = validation != Validation::Off;
    value.desc.enableSynchronizationValidation = validation == Validation::Synchronization;
    value.desc.enableBindlessDescriptorHeap = value.id != "core" && value.id != "core-unified";
    value.desc.enableAsyncCompute = value.id == "async";
    value.desc.enableOpacityMicromap = false;
    value.desc.enableRayTracingPositionFetch = false;
    value.desc.enableDeviceGeneratedCommands = false;
    metallic::render::vulkan::deviceExtensions(value.desc).preferUnifiedImageLayouts = value.id == "core-unified";
    value.desc.enableRayTracingAccelerationStructure = value.id.starts_with("ray-query");
    value.desc.enableRayQuery = value.desc.enableRayTracingAccelerationStructure;
    value.desc.enableRayTracingPositionFetch = value.id == "ray-query-position";
    value.desc.enableOpacityMicromap = value.id == "ray-query-omm";
    value.desc.enablePartitionedAccelerationStructure = value.id == "ray-query-ptlas";
    value.desc.enableClusterAccelerationStructure = value.id == "ray-query-clas";
    value.desc.enableDeviceGeneratedCommands = value.id == "binding-dgc";
    return value;
}

bool enabled(Capability capability, const render::DeviceCapabilities& caps,
    const render::vulkan::VulkanDeviceCapabilities& backendCaps)
{
    switch (capability) {
    case Capability::ShaderObject: return caps.shaderObject;
    case Capability::TimestampQueries: return caps.timestampQueries;
    case Capability::Bindless: return caps.bindlessDescriptorHeap;
    case Capability::IndependentCopy: return caps.independentCopyQueue;
    case Capability::IndependentCompute: return caps.independentComputeQueue;
    case Capability::RayQuery: return caps.rayQuery;
    case Capability::PositionFetch: return caps.rayTracingPositionFetch;
    case Capability::OpacityMicromap: return caps.opacityMicromap;
    case Capability::UnifiedLayouts: return backendCaps.unifiedImageLayouts;
    case Capability::PartitionedAS: return caps.partitionedAccelerationStructure;
    case Capability::ClusterAS: return caps.clusterAccelerationStructure;
    case Capability::GeneratedCommands: return caps.deviceGeneratedCommands;
    case Capability::MemoryDecompression: return caps.memoryDecompression;

    }
    return false;
}

bool requested(Capability capability, const Profile& value)
{
    switch (capability) {
    case Capability::Bindless: return value.desc.enableBindlessDescriptorHeap;
    case Capability::IndependentCompute: return value.desc.enableAsyncCompute;
    case Capability::RayQuery: return value.desc.enableRayQuery;
    case Capability::PositionFetch: return value.desc.enableRayTracingPositionFetch;
    case Capability::OpacityMicromap: return value.desc.enableOpacityMicromap;
    case Capability::UnifiedLayouts: return metallic::render::vulkan::deviceExtensions(value.desc).preferUnifiedImageLayouts;
    case Capability::PartitionedAS: return value.desc.enablePartitionedAccelerationStructure;
    case Capability::ClusterAS: return value.desc.enableClusterAccelerationStructure;
    case Capability::GeneratedCommands: return value.desc.enableDeviceGeneratedCommands;

    default: return true;
    }
}

Verdict evaluate(const Requirements& requirements, const Profile& value,
    const render::DeviceCapabilities& caps, const std::vector<render::QueueType>& queues,
    Validation activeValidation, const render::vulkan::VulkanDeviceCapabilities& backendCaps)
{
    if (activeValidation < requirements.validation) {
        return {Status::EnvironmentFailure, "required validation mode/layer/messenger is not active"};
    }
    for (const auto capability : requirements.capabilities) {
        if (!requested(capability, value)) {
            return {Status::SkipNotEnabled, name(capability)};
        }
        if (!enabled(capability, caps, backendCaps)) {
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
