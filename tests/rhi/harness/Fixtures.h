#pragma once

#include "../RHITest.h"
#include "Evidence.h"

namespace metallic::tests::bench {

// Legacy tests may need their own feature configuration. In the testbench the
// runner owns that configuration and its recorder for the entire device lifetime.
class TestDevice {
public:
    TestDevice() = default;
    explicit TestDevice(render::Device& device) : borrowed_(&device) {}
    explicit TestDevice(std::unique_ptr<render::Device> device) : owned_(std::move(device)) {}
    render::Device* get() const { return owned_ ? owned_.get() : borrowed_; }
    render::Device* operator->() const { return get(); }
    render::Device& operator*() const { return *get(); }
    explicit operator bool() const { return get() != nullptr; }
private:
    std::unique_ptr<render::Device> owned_;
    render::Device* borrowed_ = nullptr;
};

inline render::Result<TestDevice> createTestDevice(RHITestContext& context,
    render::DeviceDesc legacyDesc, bool additionalDevice = false)
{
    if (context.evidence && !additionalDevice) { return TestDevice(context.device); }
    if (context.deviceDesc) { legacyDesc = *context.deviceDesc; }
    return render::createDevice(legacyDesc).transform([](auto device) { return TestDevice(std::move(device)); });
}

inline Metadata gpuMetadata(std::vector<std::string> coverage, Layer layer = Layer::RHI,
    std::string profile = "core", std::string suite = "core", std::vector<std::string> artifacts = {}, bool nativePointers = false)
{
    Metadata result{.suite = std::move(suite), .profile = std::move(profile), .layer = layer,
        .coverage = std::move(coverage), .artifacts = std::move(artifacts)};
    result.requirements.nativeDescriptorPointers = nativePointers;
    if (result.profile != "core") { result.requirements.capabilities.push_back(Capability::Bindless); }
    if (result.suite == "sync") { result.requirements.validation = Validation::Synchronization; }
    return result;
}

inline Metadata comparisonMetadata(std::vector<std::string> coverage, Layer layer,
    std::string referenceProfile, Comparison comparison, bool nativePointers = false)
{
    auto result = gpuMetadata(std::move(coverage), layer, std::move(referenceProfile), "extensions",
        {"fixture.json", "observations.json", "execution.json", "readback.bin"}, nativePointers);
    if (result.profile.starts_with("ray-query")) { result.requirements.capabilities.push_back(Capability::RayQuery); }
    if (!comparison.reductionCounter.empty()) { result.artifacts.push_back(comparison.reductionCounter); }
    result.comparison = std::move(comparison);
    result.timeout = std::chrono::seconds(120);
    return result;
}

inline void comparisonEvidence(RHITestContext& context, const Json& fixture, const Json& observations,
    bool targetUsed)
{
    if (!context.evidence) { return; }
    context.evidence->json("fixture.json", fixture);
    context.evidence->json("observations.json", observations);
    context.evidence->json("execution.json", {{"targetUsed", targetUsed}});
}

// Declare after resources so exceptional exits drain before resource destruction.
class GPUCommands {
public:
    explicit GPUCommands(render::Queue& queue) : queue_(queue) {}
    ~GPUCommands()
    {
        if (submitted_) { (void)queue_.waitIdle(); }
        commands.reset();
        pool_.reset();
    }
    render::Result<> initialize(render::Device& device)
    {
        auto pool = device.createCommandPool(queue_);
        if (!pool) { return std::unexpected(pool.error()); }
        pool_ = std::move(*pool);
        auto buffer = pool_->createCommandBuffer();
        if (!buffer) { return std::unexpected(buffer.error()); }
        commands = std::move(*buffer);
        auto fence = device.createFence(false);
        if (!fence) { return std::unexpected(fence.error()); }
        fence_ = std::move(*fence);
        return commands->begin();
    }
    render::Result<> submitAndWait()
    {
        auto end = commands->end();
        if (!end) { return end; }
        render::CommandBuffer* buffers[]{commands.get()};
        auto submitted = queue_.submit({.commandBuffers = buffers, .signalFence = fence_.get()});
        if (!submitted) { return submitted; }
        submitted_ = true;
        return fence_->wait(5'000'000'000ull);
    }
    std::unique_ptr<render::CommandBuffer> commands;
private:
    render::Queue& queue_;
    std::unique_ptr<render::CommandPool> pool_;
    std::unique_ptr<render::Fence> fence_;
    bool submitted_ = false;
};

template<typename T>
void readbackEvidence(RHITestContext& context, const std::string& filename, std::span<const T> actual)
{
    if (context.evidence) {
        std::string output = filename;
        for (size_t index = 1; std::filesystem::exists(context.evidence->root() / output); ++index) {
            output = filename + "." + std::to_string(index);
        }
        context.evidence->bytes(output, std::as_bytes(actual));
    }
}

} // namespace metallic::tests::bench
