#include "WorkControlReplay.h"
#include "NvPerf.h"
#include "Runtime/Render/Core/StreamRasterParameters.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace metallic::render::profiling {
namespace {
using Json = nlohmann::json;
thread_local WorkControlReplay* active = nullptr;
void require(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}
void barrier(CommandBuffer& commands)
{
    // Buffer-only copies; no image layouts or owner-side tracked states change.
    VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
    vkCmdPipelineBarrier2(vulkan::nativeCommandBuffer(commands), &dependency);
}
void copy(CommandBuffer& commands, Buffer& source, Buffer& destination)
{
    require(source.desc().size == destination.desc().size, "replay_copy_size_mismatch");
    {
        auto sourceSlice = (&source)->slice({0, source.desc().size});
        if (!sourceSlice) { throw std::runtime_error(std::string("source slice failed: ") + metallic::render::resultToString(sourceSlice)); }
        auto destinationSlice = (&destination)->slice({0, source.desc().size});
        if (!destinationSlice) { throw std::runtime_error(std::string("destination slice failed: ") + metallic::render::resultToString(destinationSlice)); }
        if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { throw std::runtime_error(std::string("copyBuffer failed: ") + metallic::render::resultToString(commandResult)); }
    }
}
void write(const std::filesystem::path& path, const std::vector<uint8_t>& bytes)
{
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}
std::vector<uint8_t> read(Buffer& buffer)
{
    buffer.invalidate();
    auto* data = buffer.map();
    require(data != nullptr, "replay_map_failed");
    std::vector<uint8_t> bytes(buffer.desc().size);
    std::memcpy(bytes.data(), data, bytes.size());
    buffer.unmap();
    return bytes;
}
} // namespace

bool workControlReplayRequested()
{
    const char* value = std::getenv("METALLIC_WORK_CONTROL_REPLAY");
    return value && std::strcmp(value, "1") == 0;
}

std::recursive_mutex& workControlReplaySubmissionMutex()
{
    static std::recursive_mutex mutex;
    return mutex;
}

struct WorkControlReplay::Impl {
    struct Resource {
        WorkControlReplayBinding binding;
        std::shared_ptr<void> sourceLease;
        std::unique_ptr<Buffer> initial, working, readback;
        std::vector<uint8_t> expected, production;
    };
    Device& device;
    std::string phase;
    std::filesystem::path output;
    std::vector<Resource> resources;
    std::unique_ptr<Buffer> control;
    ComputeKernel kernel;
    EncodedParameters replayParameters;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<Fence> fence;
    std::vector<uint8_t> push;
    size_t pixel = SIZE_MAX, arguments = SIZE_MAX;
    bool captured = false, completedControl = false;
    std::string fault;
    Json evidence{{"protocol", "metallic-work-control-replay-v2"}, {"status", "preparing"},
        {"measurementKind", "diagnostic"}, {"scope", "isolated-correctness-only"},
        {"counterEligible", false}, {"productionStatePublished", false},
        {"bindingPolicy", "canonical-handles-private-allocations"}};

    std::unique_ptr<Buffer> buffer(const BufferDesc& source, bool host)
    {
        auto desc = source;
        desc.usage = BufferUsageBits::Storage | BufferUsageBits::Indirect |
            BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination;
        desc.memoryLocation = host ? MemoryLocation::HostReadback : MemoryLocation::Device;
        std::unique_ptr<Buffer> result;
        require(bool(device.createBuffer(desc).transform([&](auto value) { result = std::move(value); })),
            "replay_allocation_failed");
        return result;
    }
    void save()
    {
        std::ofstream file(output / "Replay.json");
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << evidence.dump(2) << '\n';
    }
    void cancelled()
    {
        require(!std::filesystem::exists(output / "Cancel.request"), "replay_cancelled");
    }
    template<class Record> void submit(Queue& queue, const char* stage, Record&& record)
    {
        cancelled();
        require(bool(pool->reset()) && bool(fence->reset()) && bool(commands->begin()), "replay_record_begin_failed");
        record(*commands);
        require(bool(commands->end()), "replay_record_end_failed");
        if (std::strcmp(stage, "isolated-dispatch") == 0 && fault == "cancel-before-submit") {
            throw std::runtime_error("injected_" + fault);
        }
        CommandBuffer* buffers[] = {commands.get()};
        Result<> submitted;
        if (std::strcmp(stage, "isolated-dispatch") == 0 && (fault == "submit-failure" || fault == "device-error")) {
            submitted = makeError(fault == "device-error" ? Error::DeviceLost : Error::Failure);
        } else {
            submitted = queue.submit({.commandBuffers = {buffers, 1}, .signalFence = fence.get()});
        }
        if (!submitted && submitted.error() == Error::DeviceLost) {
            evidence["status"] = "failed"; evidence["error"] = "replay_submit_device_lost";
            evidence["processPoisoned"] = true; save(); std::_Exit(3);
        }
        require(bool(submitted), "replay_submit_failed");
        evidence["submissions"].push_back({{"stage", stage}, {"accepted", true}, {"completed", false}});
        save();
        // An unknown GPU completion cannot unwind leases into normal rendering.
        const auto waitStart = std::chrono::steady_clock::now();
        auto waited = fence->wait(10'000'000'000ull);
        if (waited && std::strcmp(stage, "isolated-dispatch") == 0 && fault == "timeout") {
            evidence["injectedAfterSafeRetirement"] = true;
            waited = makeError(Error::Failure);
        }
        if (!waited) {
            evidence["status"] = "failed";
            evidence["error"] = waited.error() == Error::DeviceLost ? "replay_device_lost" :
                (fault == "timeout" || std::chrono::steady_clock::now() - waitStart >= std::chrono::seconds(10)) ?
                    "replay_wait_timeout" : "replay_wait_failed";
            evidence["processPoisoned"] = true;
            save();
            std::_Exit(3);
        }
        evidence["submissions"].back()["completed"] = true;
        if (std::strcmp(stage, "isolated-dispatch") == 0 && fault == "cancel-after-submit") {
            // Simulated terminal after real retirement: exercise the failure gate
            // without inducing a TDR or freeing genuinely in-flight resources.
            evidence["injectedAfterSafeRetirement"] = true;
            throw std::runtime_error("injected_" + fault);
        }
        cancelled();
    }
};

WorkControlReplay::WorkControlReplay(Device& device, std::string phase, std::filesystem::path output)
    : impl_(std::make_unique<Impl>(device, std::move(phase), std::move(output)))
{
    require(workControlReplayRequested(), "replay_not_requested_at_device_creation");
    require(impl_->phase == "early" || impl_->phase == "late", "replay_invalid_phase");
    require(!std::filesystem::exists(impl_->output), "replay_output_not_fresh");
    std::filesystem::create_directories(impl_->output);
    if (const char* fault = std::getenv("METALLIC_WORK_CONTROL_REPLAY_FAULT")) {
        require(std::set<std::string>{"cancel-before-submit", "cancel-after-submit", "submit-failure", "device-error",
            "timeout", "restore-failure", "state-leak"}.contains(fault), "unknown_replay_fault");
        impl_->fault = fault;
        impl_->evidence["faultInjection"] = fault;
        impl_->evidence["faultInjectionVersion"] = 2;
    }
}
WorkControlReplay::~WorkControlReplay()
{
    if (active == this) { active = nullptr; }
}
void WorkControlReplay::arm()
{
    require(active == nullptr && !impl_->captured, "replay_already_armed");
    active = this;
}
WorkControlReplay* WorkControlReplay::selected(std::string_view phase)
{
    return active && active->impl_->phase == phase ? active : nullptr;
}

void WorkControlReplay::before(CommandBuffer& commands, const ComputeKernel& kernel, const EncodedParameters& parameters,
    std::span<const WorkControlReplayBinding> bindings, Buffer& arguments,
    std::span<const uint8_t> settings, Json identity)
{
    auto& s = *impl_;
    require(!s.captured && identity.at("mode") == 5 && identity.at("snapshotFrozen") == true &&
        identity.at("asyncRequested") == false && identity.at("forceHardware") == false,
        "replay_requires_frozen_serial_work_control");
    // First acceptance slice excludes the low-wave fallback's larger descriptor closure.
    require(s.device.capabilities().subgroupSize >= 32 &&
        s.device.capabilities().minSubgroupSize >= 32, "replay_low_subgroup_unsupported");
    require(kernel.valid() && parameters.abi() == parameterAbi<StreamRasterParameters>(kStreamRasterABI, ParameterTransport::InlinePush), "replay_push_abi_mismatch");
    s.kernel = kernel;
    auto& pipeline = *kernel.diagnosticPipeline();
    const auto push = parameters.inlineData();
    require(push.size() == sizeof(StreamRasterParameters), "replay_push_abi_mismatch");
    StreamRasterParameters source;
    std::memcpy(&source, push.data(), sizeof(source));
    require(source.settings.address && source.settings.count == 1 && source.settings.stride == settings.size() &&
        !settings.empty(), "replay_settings_abi_mismatch");
    StreamRasterParameters relocated = source;
    auto registry = s.device.resourceRegistry();
    require(bool(registry), "replay_registry_missing");
    ParameterWriter writer(s.device, **registry);
    relocated.settings = {writer.data(settings.data(), settings.size(), 16), 1, uint32_t(settings.size())};
    write(s.output / "Settings.bin", std::vector<uint8_t>(settings.begin(), settings.end()));
    write(s.output / "ReplaySettings.bin", std::vector<uint8_t>(settings.begin(), settings.end()));
    const std::map<std::string, ShaderBuffer StreamRasterParameters::*> fields{
        {"pages", &StreamRasterParameters::pages}, {"groups", &StreamRasterParameters::groups},
        {"header", &StreamRasterParameters::header}, {"pageTable", &StreamRasterParameters::pageTable},
        {"instances", &StreamRasterParameters::instances}, {"bins", &StreamRasterParameters::bins},
        {"pixels", &StreamRasterParameters::pixels}};
    std::set<std::string> names;

    const auto inputCode = vulkan::nativeComputeSpirv(pipeline, false);
    const auto deviceCode = vulkan::nativeComputeSpirv(pipeline, true);
    require(!inputCode.empty() && !deviceCode.empty(), "replay_missing_actual_spirv");
    write(s.output / "Input.spv", inputCode);
    write(s.output / "Device.spv", deviceCode);
    s.evidence["sameRetainedExecution"] = true;
    s.push.assign(push.begin(), push.end());
    write(s.output / "Push.bin", s.push);
    s.evidence["phase"] = s.phase;
    s.evidence["productionShader"] = std::move(identity);
    s.evidence["psoHash"] = std::to_string(pipeline.psoHash());
    s.evidence["subgroupSize"] = s.device.capabilities().subgroupSize;
    s.evidence["bindingGenerationEvidence"] = "allocation-id-and-frozen-graph-generation";
    s.evidence["heapAbi"] = {{"nativeDescriptorHeap", vulkan::nativeDevice(s.device).descriptorHeapEnabled},
        {"maxBuffers", (*registry)->heap()->desc().maxBuffers}};
    s.evidence["parameterABI"] = {{"id", std::to_string(kStreamRasterABI)}, {"size", sizeof(source)},
        {"settingsBytes", settings.size()}};
    std::vector<WorkControlReplayBinding> all(bindings.begin(), bindings.end());
    all.push_back({"arguments", &arguments});
    std::set<Buffer*> allocations;
    uint64_t totalBytes = 0;
    for (const auto& binding : all) {
        require(binding.buffer && allocations.insert(binding.buffer).second, "replay_resource_alias_unqualified");
        totalBytes += binding.buffer->desc().size;
    }
    require(totalBytes <= 1024ull * 1024 * 1024, "replay_snapshot_budget_exceeded");
    barrier(commands);
    for (const auto& binding : all) {
        Impl::Resource resource;
        resource.binding = binding;
        resource.sourceLease = binding.buffer->retainAllocation();
        require(bool(resource.sourceLease), "replay_missing_allocation_lease");
        resource.initial = s.buffer(binding.buffer->desc(), false);
        resource.working = s.buffer(binding.buffer->desc(), false);
        resource.readback = s.buffer(binding.buffer->desc(), true);
        copy(commands, *binding.buffer, *resource.initial);
        require(names.insert(binding.name).second, "replay_duplicate_resource_name");
        const auto field = fields.find(binding.name);
        uint64_t sourceHandle = UINT64_MAX, scratchHandle = UINT64_MAX;
        if (field != fields.end()) {
            sourceHandle = (source.*field->second).value;
            require(writer.buffer(binding.buffer).value == sourceHandle, "replay_source_handle_mismatch");
            relocated.*field->second = writer.buffer(resource.working.get());
            scratchHandle = (relocated.*field->second).value;
            require(sourceHandle != scratchHandle, "replay_handle_alias");
        }
        if (binding.name == "pixels") { s.pixel = s.resources.size(); }
        if (binding.name == "arguments") { s.arguments = s.resources.size(); }
        s.evidence["bindings"].push_back({{"name", binding.name}, {"sourceHandle", std::to_string(sourceHandle)}, {"scratchHandle", std::to_string(scratchHandle)},
            {"bytes", binding.buffer->desc().size}, {"stride", binding.buffer->desc().structureStride},
            {"sourceAddress", std::to_string(vulkan::nativeBuffer(*binding.buffer).address)},
            {"sourceAllocation", binding.buffer->memoryInfo().allocationId},
            {"scratchAllocation", resource.working->memoryInfo().allocationId},
            {"scratchAddress", std::to_string(vulkan::nativeBuffer(*resource.working).address)}});
        s.resources.push_back(std::move(resource));
    }
    require(s.pixel != SIZE_MAX && s.arguments != SIZE_MAX, "replay_incomplete_closure");
    for (const auto& [name, field] : fields) { require(names.contains(name), "replay_incomplete_closure"); }
    auto encoded = writer.encode(relocated, kStreamRasterABI, ParameterTransport::InlinePush);
    require(bool(encoded), "replay_parameter_encoding_failed");
    s.replayParameters = std::move(*encoded);
    const auto replayPush = s.replayParameters.inlineData();
    write(s.output / "ReplayPush.bin", std::vector<uint8_t>(replayPush.begin(), replayPush.end()));
    s.control = s.buffer(s.resources[s.pixel].binding.buffer->desc(), true);
    barrier(commands);
    s.captured = true;
    s.evidence["status"] = "captured";
    s.save();
}

void WorkControlReplay::after(CommandBuffer& commands)
{
    auto& s = *impl_;
    require(s.captured && !s.completedControl, "replay_control_boundary_mismatch");
    barrier(commands);
    copy(commands, *s.resources[s.pixel].binding.buffer, *s.control);
    barrier(commands);
    s.completedControl = true;
    active = nullptr;
}

Json WorkControlReplay::run(Queue& queue, const Json& frozenIdentity)
{
    auto& s = *impl_;
    std::unique_lock submissionLease(workControlReplaySubmissionMutex());
    try {
        require(bool(s.device.waitIdle()), "replay_initial_device_drain_failed");
        s.evidence["submissionIsolation"] = "all-RHI-queues-owner-lease";
        const auto nativeQueue = vulkan::nativeQueue(queue);
        s.evidence["queueFamily"] = nativeQueue.familyIndex;
        s.evidence["queueIdentity"] = std::to_string(reinterpret_cast<uintptr_t>(nativeQueue.queue));
        require(s.completedControl && active != this, "replay_control_not_captured");
        s.evidence["frozenIdentity"] = frozenIdentity;
        require(bool(s.device.createCommandPool(queue).transform([&](auto value) { s.pool = std::move(value); })) &&
            bool(s.pool->createCommandBuffer().transform([&](auto value) { s.commands = std::move(value); })) &&
            bool(s.device.createFence(false).transform([&](auto value) { s.fence = std::move(value); })), "replay_submit_setup_failed");
        const auto control = read(*s.control);
        write(s.output / "Control.bin", control);
        s.submit(queue, "archive-inputs", [&](auto& commands) {
            barrier(commands);
            for (auto& resource : s.resources) { copy(commands, *resource.initial, *resource.readback); }
            barrier(commands);
        });
        for (auto& resource : s.resources) {
            resource.expected = read(*resource.readback);
            write(s.output / (resource.binding.name + "-input.bin"), resource.expected);
        }
        const auto& args = s.resources[s.arguments].expected;
        require(args.size() >= 60, "replay_indirect_range_invalid");
        uint32_t dimensions[3];
        std::memcpy(dimensions, args.data() + 48, sizeof(dimensions));
        require(dimensions[0] && dimensions[1] && dimensions[2], "replay_empty_dispatch");
        s.evidence["indirect"] = {{"offset", 48}, {"dimensions", {dimensions[0], dimensions[1], dimensions[2]}}};
        s.submit(queue, "production-before", [&](auto& commands) {
            barrier(commands);
            for (auto& resource : s.resources) { copy(commands, *resource.binding.buffer, *resource.readback); }
            barrier(commands);
        });
        for (auto& resource : s.resources) {
            resource.production = read(*resource.readback);
            write(s.output / (resource.binding.name + "-production-before.bin"), resource.production);
        }
        // Two independently restored unprofiled passes expose accidental cumulative
        // atomics and establish the reset contract before any counter session.
        NvPerfSession counters;
        uint32_t counterPasses = 0;
        for (uint32_t pass = 0; pass < 18; ++pass) {
            if (pass == 2) {
                if (!nvPerfRequested()) { break; }
                // No profiler session exists until unprofiled output, binding,
                // reset and production-state gates have completed.
                s.submit(queue, "production-gate", [&](auto& commands) {
                    barrier(commands);
                    for (auto& resource : s.resources) { copy(commands, *resource.binding.buffer, *resource.readback); }
                    barrier(commands);
                });
                for (auto& resource : s.resources) {
                    const auto actual = read(*resource.readback);
                    write(s.output / (resource.binding.name + "-production-gate.bin"), actual);
                    require(actual == resource.production, "replay_pre_counter_state_leak");
                }
                std::string error;
                const bool started = counters.beginIsolated(s.device, queue, s.output / "nvperf", s.phase, error);
                require(started, error.c_str());
            }
            s.submit(queue, "restore-inputs", [&](auto& commands) {
                barrier(commands);
                for (auto& resource : s.resources) {
                    if (s.fault == "restore-failure" && pass == 1 && resource.binding.name == "pixels") { continue; }
                    copy(commands, *resource.initial, *resource.working);
                }
                barrier(commands);
                for (auto& resource : s.resources) { copy(commands, *resource.working, *resource.readback); }
                barrier(commands);
            });
            for (auto& resource : s.resources) {
                const auto restored = read(*resource.readback);
                write(s.output / (std::to_string(pass) + "-" + resource.binding.name + "-restored.bin"), restored);
                require(restored == resource.expected, "replay_restore_failed");
            }
            s.submit(queue, "isolated-dispatch", [&](auto& commands) {
                const std::string rangeName = "WorkControl/isolated/" + s.phase;
                NvPerfRange range(commands, rangeName.c_str());
                require(bool(s.kernel.dispatchIndirect(commands, s.replayParameters, *s.resources[s.arguments].working, 48)), "replay_dispatch_record_failed");
            });
            s.submit(queue, "compare-output", [&](auto& commands) {
                barrier(commands);
                for (auto& resource : s.resources) { copy(commands, *resource.working, *resource.readback); }
                barrier(commands);
            });
            for (size_t index = 0; index < s.resources.size(); ++index) {
                auto& resource = s.resources[index];
                const auto actual = read(*resource.readback);
                write(s.output / (std::to_string(pass) + "-" + resource.binding.name + "-output.bin"), actual);
                require(actual == (index == s.pixel ? control : resource.expected), "replay_output_or_readonly_state_mismatch");
            }
            if (pass >= 2) {
                ++counterPasses;
                std::string error;
                const bool finished = counters.finish(error);
                require(finished, error.c_str());
                if (counters.complete()) { break; }
            }
        }
        require(!nvPerfRequested() || counters.complete(), "replay_counter_pass_budget_exceeded");
        s.submit(queue, "production-after", [&](auto& commands) {
            barrier(commands);
            for (auto& resource : s.resources) { copy(commands, *resource.binding.buffer, *resource.readback); }
            barrier(commands);
        });
        for (auto& resource : s.resources) {
            auto actual = read(*resource.readback);
            if (s.fault == "state-leak" && resource.binding.name == "pageTable") { actual[4] ^= 1; }
            write(s.output / (resource.binding.name + "-production-after.bin"), actual);
            require(actual == resource.production, "replay_production_state_leak");
        }
        s.evidence["status"] = "complete";
        s.evidence["correctnessPasses"] = 2;
        s.evidence["counterPasses"] = counterPasses;
        if (counterPasses) {
            s.evidence["counterEligible"] = true;
            s.evidence["scope"] = "isolated-dispatch-RHI-exclusive";
        }
        s.save();
        return s.evidence;
    } catch (const std::exception& error) {
        s.evidence["status"] = "failed";
        s.evidence["error"] = error.what();
        s.save();
        throw;
    }
}

} // namespace metallic::render::profiling
