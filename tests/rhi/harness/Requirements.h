#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace metallic::tests::bench {

enum class Capability { ShaderObject, TimestampQueries, Bindless, IndependentCopy, IndependentCompute };
enum class Layer { Rhi, Core, RenderGraph, Harness };
enum class Validation { Off, Core };

struct Requirements {
    bool requiresDevice = true;
    Validation validation = Validation::Core;
    std::vector<Capability> capabilities;
    std::vector<render::QueueType> queues{render::QueueType::Graphics};
};

struct Metadata {
    std::string suite = "core";
    std::string profile = "core";
    Layer layer = Layer::Rhi;
    Requirements requirements;
    std::vector<std::string> coverage;
    std::chrono::milliseconds timeout{30000};
    std::vector<std::string> artifacts;
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
bool failed(Status value);
render::Result<Profile> profile(std::string id, Validation validation);
bool enabled(Capability capability, const render::DeviceCapabilities& caps);
bool requested(Capability capability, const Profile& profile);
Verdict evaluate(const Requirements& requirements, const Profile& profile,
    const render::DeviceCapabilities& caps, const std::vector<render::QueueType>& queues,
    bool validationActive);

} // namespace metallic::tests::bench
