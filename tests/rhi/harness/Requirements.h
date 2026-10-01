#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace metallic::tests::bench {

enum class Capability { ShaderObject, TimestampQueries, Bindless, IndependentCopy, IndependentCompute, RayQuery, PositionFetch, OpacityMicromap,
    UnifiedLayouts, PartitionedAS, ClusterAS, GeneratedCommands, MemoryDecompression };
enum class Layer { RHI, Core, RenderGraph, Backend, Harness };
enum class Validation { Off, Core, Synchronization };

struct Requirements {
    bool requiresDevice = true;
    Validation validation = Validation::Core;
    std::vector<Capability> capabilities;
    std::vector<render::QueueType> queues{render::QueueType::Graphics};
    std::vector<render::QueueType> timestampQueues;
    bool nativeDescriptorPointers = false;
};

struct Comparison {
    std::string targetProfile;
    std::string toggle;
    Capability capability;
    double absoluteTolerance = 0.0;
    double relativeTolerance = 0.0;
    std::string reductionCounter;
};

struct Metadata {
    std::string suite = "core";
    std::string profile = "core";
    Layer layer = Layer::RHI;
    Requirements requirements;
    std::vector<std::string> coverage;
    std::chrono::milliseconds timeout{30000};
    std::vector<std::string> artifacts;
    std::optional<Comparison> comparison;
};

struct Profile {
    std::string id;
    render::DeviceDesc desc;
};

enum class Status { Pass, Fail, SkipUnsupported, SkipNotEnabled, EnvironmentFailure, InfrastructureFailure, Timeout, Crash, DeviceLost };
struct Verdict {
    Status status = Status::Pass;
    std::string message;
    bool executed = false;
};

const char* name(Capability value);
const char* name(Status value);
const char* name(Layer value);
const char* name(Validation value);
Validation parseValidation(const std::string& value);
bool failed(Status value);
render::Result<Profile> profile(std::string id, Validation validation);
bool enabled(Capability capability, const render::DeviceCapabilities& caps);
bool requested(Capability capability, const Profile& profile);
Verdict evaluate(const Requirements& requirements, const Profile& profile,
    const render::DeviceCapabilities& caps, const std::vector<render::QueueType>& queues,
    Validation activeValidation);

} // namespace metallic::tests::bench
