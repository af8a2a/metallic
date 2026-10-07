#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

using SHReadback = std::array<float, 36>;

struct ProviderRecording {
    render::Device& device;
    render::RenderWorld world;
    render::RenderSubsystemHost host;
    render::RenderFrameContext frame;
    render::QueueSubmissionTracker tracker;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> commands;
    uint64_t frameIndex = 0;

    explicit ProviderRecording(render::Device& device) : device(device) {}
    ~ProviderRecording()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        host.endFrame();
        frame.cancel();
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
        host.shutdown();
    }
    render::Result<> initialize(std::string& log)
    {
        using namespace render;
        if (!host.registerSubsystem<EnvironmentLightingSubsystem>(log)) { return makeError(Error::Failure); }
        auto result = host.initialize(device, 1, log);
        if (!result) { return result; }
        world.setEnvironment({.enabled = true});
        host.setWorld(&world);
        if (!(result = host.activate(EnvironmentLightingSubsystem::kSubsystemId, log))) { return result; }
        auto& queue = *device.getQueue(QueueType::Graphics);
        if (!(result = tracker.initialize(device, queue))) { return result; }
        result = device.createCommandPool(queue).transform([&](auto value) { pool = std::move(value); });
        return result ? pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) : result;
    }
    render::Result<> begin(std::string& log)
    {
        auto result = frame.begin(frameIndex++);
        if (result) { result = pool->reset(); }
        if (result) { result = commands->begin(frame.submissionContext()); }
        if (result) { result = host.beginFrame(frame.frameIndex(), 0, nullptr, log, &frame); }
        const std::array required{render::EnvironmentLightingSubsystem::kSubsystemId};
        return result ? host.recordPreGraph(*commands, nullptr, required, log) : result;
    }
    render::Result<render::EnvironmentLightingSnapshot> resolve(const environment::WorldEnvironment& environment,
        std::string& log)
    {
        return host.get<render::EnvironmentLightingSubsystem>()->resolveRadiance(device, *commands, host,
            environment.snapshot(), {0.0, 2.0, 0.0}, log);
    }
    render::Result<> submit()
    {
        auto result = commands->end();
        host.endFrame();
        render::CommandBuffer* buffers[]{commands.get()};
        if (result) { result = tracker.submit({.commandBuffers = buffers}, frame); }
        return result ? frame.wait(5'000'000'000ull) : result;
    }
    render::Result<> cancel(bool resetPoolFirst)
    {
        auto result = commands->end();
        host.endFrame();
        if (result && resetPoolFirst) { result = pool->reset(); }
        frame.cancel();
        if (result && !resetPoolFirst) { result = pool->reset(); }
        return result;
    }
};

environment::WorldEnvironment providerPhysicalWorld()
{
    environment::WorldEnvironment world;
    world.source = environment::EnvironmentSource::PhysicalAtmosphere;
    world.sun.enabled = true;
    world.sun.direction = {0.0f, -1.0f, 0.0f};
    return world;
}

render::Result<> createSHReadback(render::Device& device, std::unique_ptr<render::Buffer>& output)
{
    using namespace render;
    return device.createBuffer({.size = sizeof(SHReadback), .structureStride = 16,
        .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
        .transform([&](auto value) { output = std::move(value); });
}

render::Result<> recordSHReadback(ProviderRecording& recording,
    const render::EnvironmentLightingSnapshot& environment, render::Buffer& output)
{
    using namespace render;
    if (!environment.sphericalHarmonicsBuffer) { return makeError(Error::InvalidArgument); }
    BufferBarrierDesc before{.buffer = environment.sphericalHarmonicsBuffer,
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {.offset = 0, .size = sizeof(SHReadback)}};
    auto result = recording.commands->synchronize({.buffers = {&before, 1}});
    auto source = environment.sphericalHarmonicsBuffer->slice({0, sizeof(SHReadback)});
    auto destination = output.slice({0, sizeof(SHReadback)});
    if (!source || !destination) { return makeError(Error::Failure); }
    if (result) { result = recording.commands->copyBuffer(*source, *destination); }
    BufferBarrierDesc after{.buffer = &output,
        .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = sizeof(SHReadback)}};
    return result ? recording.commands->synchronize({.buffers = {&after, 1}}) : result;
}

bool readSH(render::Buffer& output, SHReadback& values)
{
    output.invalidate();
    const void* mapped = output.map();
    if (!mapped) { return false; }
    std::memcpy(values.data(), mapped, sizeof(values));
    output.unmap();
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }) &&
        values[0] > 0.0f && values[1] > 0.0f && values[2] > 0.0f;
}

bool physicalFieldsValid(const render::EnvironmentLightingSnapshot& value)
{
    return value.valid() && value.source == environment::EnvironmentSource::PhysicalAtmosphere &&
        value.atmosphereParametersBuffer && value.transmittanceView && value.multiScatteringView &&
        value.skyView && value.aerialPerspectiveBuffer && value.retainedResources;
}

bool HDRIFieldsValid(const render::EnvironmentLightingSnapshot& value)
{
    return value.valid() && value.source == environment::EnvironmentSource::HDRI &&
        !value.atmosphereParametersBuffer && !value.transmittanceView && !value.multiScatteringView &&
        !value.skyView && !value.aerialPerspectiveBuffer;
}

class PhysicalEnvironmentPublicationTest final : public RHITest {
public:
    PhysicalEnvironmentPublicationTest() { name = "physical_environment_provider_content_eviction"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.physical.publication.cache", "environment.physical.retention.eviction",
            "environment.provider.hdri.switch"}, bench::Layer::Core, "binding", "binding", {"provider-lifetime.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Physical environment provider lifetime",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Provider bindless device: " + std::string(toString(result))); }
        std::unique_ptr<Buffer> readback;
        if (!createSHReadback(*device, readback)) { return RHITestResult::fail("Provider readback allocation failed"); }
        ProviderRecording recording(*device);
        std::string log;
        if (!recording.initialize(log) || !recording.begin(log)) { return RHITestResult::fail(log); }
        const auto base = providerPhysicalWorld();
        EnvironmentLightingSnapshot first, other, reused;
        auto resolve = [&](const environment::WorldEnvironment& world, EnvironmentLightingSnapshot& snapshot) {
            return recording.resolve(world, log).transform([&](auto value) { snapshot = std::move(value); });
        };
        if (!resolve(base, first) || !physicalFieldsValid(first) ||
            !recordSHReadback(recording, first, *readback) || !recording.submit()) {
            return RHITestResult::fail("Initial physical provider submission: " + log);
        }
        SHReadback initial{};
        if (!readSH(*readback, initial)) { return RHITestResult::fail("Submitted physical publication has no finite irradiance SH"); }
        auto changed = base;
        changed.atmosphere.mieExtinction = {0.00888f, 0.00888f, 0.00888f};
        if (!recording.begin(log) || !resolve(changed, other) || !recording.submit()) { return RHITestResult::fail(log); }
        if (!recording.begin(log) || !resolve(base, reused) || !recording.submit()) { return RHITestResult::fail(log); }
        if (reused.radianceView != first.radianceView || reused.retainedResources != first.retainedResources ||
            reused.resourceRevision != first.resourceRevision || other.resourceRevision == first.resourceRevision ||
            other.radianceView == first.radianceView) {
            return RHITestResult::fail("A/B/A did not preserve content revision and immutable resource reuse");
        }
        reused = {};
        other = {};
        // The public snapshot retains its owner after more than the eight-entry
        // cache capacity has been replaced by distinct physical publications.
        for (uint32_t index = 0; index < 10; ++index) {
            changed = base;
            changed.sun.topOfAtmosphereIrradiance.x += 0.1f * float(index + 1);
            EnvironmentLightingSnapshot temporary;
            if (!recording.begin(log) || !resolve(changed, temporary) || !recording.submit()) {
                return RHITestResult::fail("Physical cache eviction submission: " + log);
            }
        }
        if (!recording.begin(log) || !recordSHReadback(recording, first, *readback) || !recording.submit()) {
            return RHITestResult::fail("Retained evicted publication GPU read: " + log);
        }
        SHReadback retained{};
        if (!readSH(*readback, retained) || retained != initial) {
            return RHITestResult::fail("Eviction changed or destroyed the retained publication's GPU irradiance");
        }
        if (!recording.begin(log) || !resolve(base, reused) || !recording.submit() ||
            reused.resourceRevision != first.resourceRevision || reused.radianceView == first.radianceView) {
            return RHITestResult::fail("Evicted content did not rebuild with stable content identity");
        }
        environment::WorldEnvironment hdri;
        EnvironmentLightingSnapshot fallback;
        if (!recording.begin(log) || !resolve(hdri, fallback) || !HDRIFieldsValid(fallback) || !recording.submit()) {
            return RHITestResult::fail("HDRI source switch retained physical fields or lost the fallback provider");
        }
        const bench::Json evidence{{"firstRevision", first.resourceRevision}, {"rebuiltRevision", reused.resourceRevision},
            {"retainedSH", retained}, {"initialSH", initial}, {"evictionPublications", 10},
            {"physicalOwnerRetained", bool(first.retainedResources)}, {"HDRIPhysicalFieldsEmpty", HDRIFieldsValid(fallback)}};
        if (context.evidence) { context.evidence->json("provider-lifetime.json", evidence); }
        else { bench::writeJson(context.outputDirectory / "provider-lifetime.json", evidence); }
        return RHITestResult::pass("Submitted A/B/A reuse, stable post-eviction revisions, retained GPU SH and HDRI field isolation");
    }
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentPublicationTest);

class PhysicalEnvironmentCancellationTest final : public RHITest {
public:
    PhysicalEnvironmentCancellationTest() { name = "physical_environment_provider_cancel_recovery"; type = RHITestType::Command; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.physical.submission.cancel.recovery"}, bench::Layer::Core,
            "binding", "binding", {"provider-cancellation.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Physical provider cancellation recovery",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Cancellation bindless device setup failed"); }
        std::unique_ptr<Buffer> readback;
        if (!createSHReadback(*device, readback)) { return RHITestResult::fail("Cancellation readback allocation failed"); }
        ProviderRecording recording(*device);
        std::string log;
        if (!recording.initialize(log)) { return RHITestResult::fail(log); }
        // Commit the HDRI fallback first so cancellation isolates the physical
        // publication rather than cancelling the initial HDRI upload as well.
        if (!recording.begin(log) || !recording.submit()) { return RHITestResult::fail(log); }
        bench::Json evidence = bench::Json::array();
        for (bool resetPoolFirst : {false, true}) {
            auto world = providerPhysicalWorld();
            world.sun.topOfAtmosphereIrradiance.x += resetPoolFirst ? 0.5f : 0.25f;
            EnvironmentLightingSnapshot cancelled, recovered;
            if (!recording.begin(log) || !recording.resolve(world, log)
                    .transform([&](auto value) { cancelled = std::move(value); }) ||
                !physicalFieldsValid(cancelled) || !recording.cancel(resetPoolFirst)) {
                return RHITestResult::fail("Unsubmitted physical cancellation setup: " + log);
            }
            if (!recording.begin(log) || !recording.resolve(world, log)
                    .transform([&](auto value) { recovered = std::move(value); }) ||
                !physicalFieldsValid(recovered) || recovered.radianceView == cancelled.radianceView ||
                recovered.retainedResources == cancelled.retainedResources ||
                recovered.resourceRevision != cancelled.resourceRevision ||
                !recordSHReadback(recording, recovered, *readback) || !recording.submit()) {
                return RHITestResult::fail("Cancelled physical publication was reused or failed to rebuild: " + log);
            }
            SHReadback irradiance{};
            if (!readSH(*readback, irradiance)) { return RHITestResult::fail("Recovered provider has unexecuted or nonfinite GPU SH"); }
            evidence.push_back({{"poolResetFirst", resetPoolFirst}, {"contentRevision", recovered.resourceRevision},
                {"allocationRebuilt", recovered.radianceView != cancelled.radianceView}, {"submittedSH", irradiance}});
        }
        if (context.evidence) { context.evidence->json("provider-cancellation.json", evidence); }
        else { bench::writeJson(context.outputDirectory / "provider-cancellation.json", evidence); }
        return RHITestResult::pass("Frame cancellation and command-pool reset discard unexecuted physical resources; retry executes fresh immutable publications");
    }
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentCancellationTest);

} // namespace
} // namespace metallic::tests
