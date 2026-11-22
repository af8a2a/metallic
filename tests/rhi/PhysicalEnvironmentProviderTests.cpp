#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
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

struct ProviderConsumerControl {
    environment::WorldEnvironment world = providerPhysicalWorld();
    std::array<double, 3> observer{0.0, 2.0, 0.0};
    uint64_t viewId = 1;
    bool failInExecute = false;
    bool executed = false;
    render::EnvironmentResourceSet observedSet{};
    render::Buffer* readbackTarget = nullptr;
};

class ProviderConsumerTestPass final : public render::ComputePass {
public:
    static inline std::unordered_map<std::string, ProviderConsumerControl*> controls;

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("dummy").buffer(16, 16).storageWrite();
        return reflection;
    }

    render::Result<> declareResourceRequests(
        const render::RenderGraphResourceRequestContext& context,
        render::RenderGraphResourceRequestBuilder& builder,
        std::string& log) override
    {
        auto* env = context.subsystem<render::EnvironmentLightingSubsystem>();
        auto iter = controls.find(context.passName());
        if (env == nullptr || iter == controls.end() || iter->second == nullptr) {
            log = "Missing provider consumer control";
            return render::makeError(render::Error::InvalidArgument);
        }
        const auto& ctrl = *iter->second;
        auto declared = env->declareEnvironmentRequest(
            context,
            builder,
            render::EnvironmentRequest{
                .environment = ctrl.world.snapshot(),
                .observerWorldMetres = ctrl.observer,
                .viewId = ctrl.viewId,
                .needCelestial = true,
            },
            log);
        return declared ? render::Result<>{} : render::makeError(declared.error());
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto iter = controls.find(context.passName());
        if (iter == controls.end() || iter->second == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }
        auto& ctrl = *iter->second;
        const auto* resources = context.resources<render::EnvironmentResourceSet>();
        if (resources == nullptr || !resources->valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }
        ctrl.observedSet = *resources;
        ctrl.executed = true;
        if (ctrl.failInExecute) {
            return render::makeError(render::Error::Failure);
        }
        if (ctrl.readbackTarget != nullptr && resources->lighting.sphericalHarmonicsBuffer != nullptr) {
            using namespace render;
            BufferBarrierDesc before{.buffer = resources->lighting.sphericalHarmonicsBuffer,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.offset = 0, .size = sizeof(SHReadback)}};
            auto result = context.commandBuffer().synchronize({.buffers = {&before, 1}});
            auto source = resources->lighting.sphericalHarmonicsBuffer->slice({0, sizeof(SHReadback)});
            auto destination = ctrl.readbackTarget->slice({0, sizeof(SHReadback)});
            if (!result || !source || !destination) { return makeError(Error::Failure); }
            result = context.commandBuffer().copyBuffer(*source, *destination);
            if (!result) { return result; }
            BufferBarrierDesc after{.buffer = ctrl.readbackTarget,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
                .range = {.offset = 0, .size = sizeof(SHReadback)}};
            return context.commandBuffer().synchronize({.buffers = {&after, 1}});
        }
        return {};
    }
};

class RenderGraphEnvironmentProviderSchedulingTest final : public RHITest {
public:
    RenderGraphEnvironmentProviderSchedulingTest()
    {
        name = "physical_environment_render_graph_provider_scheduling";
        type = RHITestType::Rendering;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.provider.render_graph.scheduling",
            "environment.provider.render_graph.cross_view_medium_sharing",
            "environment.provider.render_graph.cancel_safety"},
            bench::Layer::RenderGraph, "provider", "sync", {"provider-rg-scheduling.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("ProviderConsumerTestPass",
            "Declarative EnvironmentResourceSet test consumer",
            [] { return std::make_unique<ProviderConsumerTestPass>(); });
        (void)registered;

        std::unique_ptr<Device> device;
        auto created = createDevice({.applicationName = "RenderGraph Environment Provider",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        Queue* graphics = device->getQueue(QueueType::Graphics);
        if (!graphics) { return RHITestResult::fail("Missing graphics queue"); }

        std::unique_ptr<Buffer> readbackA, readbackB;
        if (!createSHReadback(*device, readbackA) || !createSHReadback(*device, readbackB)) {
            return RHITestResult::fail("SH readback buffer allocation failed");
        }

        ProviderConsumerControl ctrlA{
            .world = providerPhysicalWorld(),
            .observer = {0.0, 2.0, 0.0},
            .viewId = 101,
            .readbackTarget = readbackA.get(),
        };
        ProviderConsumerControl ctrlB{
            .world = providerPhysicalWorld(),
            .observer = {0.0, 18000.0, 0.0},
            .viewId = 202,
            .readbackTarget = readbackB.get(),
        };
        struct ControlCleanup {
            ~ControlCleanup() { ProviderConsumerTestPass::controls.clear(); }
        } cleanup;
        ProviderConsumerTestPass::controls["ViewA"] = &ctrlA;
        ProviderConsumerTestPass::controls["ViewB"] = &ctrlB;

        RenderGraph graph;
        graph.setName("EnvironmentProviderTest");
        graph.addNode("ProviderConsumerTestPass", "ViewA");
        graph.addNode("ProviderConsumerTestPass", "ViewB");
        graph.markOutput("ViewA.dummy");
        graph.markOutput("ViewB.dummy");

        RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        if (!executor.compile(*device, graph, 64, 64, log)) {
            return RHITestResult::fail("compile failed: " + log);
        }
        auto* envSubsystem = executor.subsystemHost()->get<EnvironmentLightingSubsystem>();
        if (envSubsystem == nullptr) { return RHITestResult::fail("Missing EnvironmentLightingSubsystem"); }

        // Frame 0: Two views with different altitudes but identical static medium.
        // ViewA produces medium + radiance; ViewB shares ViewA's static medium in the same frame!
        if (!executor.execute({.graphicsQueue = graphics}) || !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Frame 0 execute failed");
        }
        const auto snap0 = executor.executionSnapshot();
        if (!snap0 || !snap0->success) { return RHITestResult::fail("Frame 0 snapshot missing"); }
        const size_t prodSegs0 = static_cast<size_t>(std::count_if(snap0->segments.begin(), snap0->segments.end(),
            [](const auto& s) { return s.role == RenderGraphSegmentRole::ProviderProducer; }));
        if (prodSegs0 != 2) {
            return RHITestResult::fail("Expected 2 ProviderProducer segments on initial miss, got " + std::to_string(prodSegs0));
        }
        if (ctrlA.observedSet.lighting.radianceView == ctrlB.observedSet.lighting.radianceView) {
            return RHITestResult::fail("Distinct views aliased their radianceView");
        }
        if (ctrlA.observedSet.lighting.transmittanceView != ctrlB.observedSet.lighting.transmittanceView ||
            ctrlA.observedSet.lighting.multiScatteringView != ctrlB.observedSet.lighting.multiScatteringView ||
            !ctrlB.observedSet.mediumShared) {
            return RHITestResult::fail("Distinct views with identical static medium did not share Transmittance/MultiScattering");
        }
        SHReadback shA0{}, shB0{};
        if (!readSH(*readbackA, shA0) || !readSH(*readbackB, shB0) || shA0 == shB0) {
            return RHITestResult::fail("Frame 0 multi-view GPU SH readback invalid or unexpectedly identical");
        }
        const auto frame0Transmittance = ctrlA.observedSet.lighting.transmittanceView;
        const auto frame0MultiScattering = ctrlA.observedSet.lighting.multiScatteringView;
        const uint64_t frame0ContentRevA = ctrlA.observedSet.version.contentRevision;
        const uint64_t frame0AllocGenA = ctrlA.observedSet.version.allocationGeneration;
        const uint64_t frame0GraphGen = snap0->graphGeneration;
        const uint64_t producersAfter0 = envSubsystem->physicalProducerCount();
        const uint64_t hitsAfter0 = envSubsystem->physicalCacheHitCount();

        // Frame 1: Identical parameters -> 0 GPU precompute, 0 graph recompilation, bit-identical SH.
        if (!executor.execute({.graphicsQueue = graphics}) || !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Frame 1 cache-hit execute failed");
        }
        const auto snap1 = executor.executionSnapshot();
        const size_t prodSegs1 = snap1 ? static_cast<size_t>(std::count_if(snap1->segments.begin(), snap1->segments.end(),
            [](const auto& s) { return s.role == RenderGraphSegmentRole::ProviderProducer; })) : 99;
        if (prodSegs1 != 0 || envSubsystem->physicalProducerCount() != producersAfter0 ||
            envSubsystem->physicalCacheHitCount() != hitsAfter0 + 2 || snap1->graphGeneration != frame0GraphGen) {
            return RHITestResult::fail("Frame 1 did not hit committed cache with zero producer segments and zero recompile");
        }
        SHReadback shA1{};
        if (!readSH(*readbackA, shA1) || shA1 != shA0) {
            return RHITestResult::fail("Frame 1 cache hit altered GPU SH readback");
        }

        // Evict ViewA's full PhysicalPublication via 9 distinct sun-irradiance edits without recompiling.
        // Static medium cache must survive full PhysicalPublication eviction!
        for (uint32_t i = 0; i < 9; ++i) {
            ctrlA.world = providerPhysicalWorld();
            ctrlA.world.sun.topOfAtmosphereIrradiance.x += 0.25f * float(i + 1);
            if (!executor.execute({.graphicsQueue = graphics}) || !executor.waitForSubmittedWork(5'000'000'000ull)) {
                return RHITestResult::fail("Eviction frame execute failed");
            }
            if (ctrlA.observedSet.lighting.transmittanceView != frame0Transmittance ||
                ctrlA.observedSet.lighting.multiScatteringView != frame0MultiScattering ||
                !ctrlA.observedSet.mediumShared) {
                return RHITestResult::fail("Sun edit did not reuse cached static medium LUTs");
            }
        }
        // Return ViewA to Frame 0 parameters: contentRevision stays identical, allocationGeneration increases,
        // and static medium LUTs are STILL shared!
        ctrlA.world = providerPhysicalWorld();
        if (!executor.execute({.graphicsQueue = graphics}) || !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Post-eviction rebuild execute failed");
        }
        if (ctrlA.observedSet.version.contentRevision != frame0ContentRevA ||
            ctrlA.observedSet.version.allocationGeneration <= frame0AllocGenA ||
            ctrlA.observedSet.lighting.transmittanceView != frame0Transmittance ||
            ctrlA.observedSet.lighting.multiScatteringView != frame0MultiScattering) {
            return RHITestResult::fail("Post-eviction rebuild did not preserve contentRevision / advance allocationGeneration / share medium");
        }
        SHReadback shARebuilt{};
        if (!readSH(*readbackA, shARebuilt) || shARebuilt != shA0) {
            return RHITestResult::fail("Post-eviction rebuilt GPU SH did not match Frame 0");
        }

        // Failure/cancellation safety: new cache-miss content whose frame fails in consumer execute
        // must cancel its unsubmitted producer and never expose it to the next frame's cache.
        ctrlA.world = providerPhysicalWorld();
        ctrlA.world.sun.topOfAtmosphereIrradiance.y += 1.75f;
        ctrlA.failInExecute = true;
        if (executor.execute({.graphicsQueue = graphics}).has_value()) {
            return RHITestResult::fail("Expected failing consumer frame to return error");
        }
        const auto cancelledRadiance = ctrlA.observedSet.lighting.radianceView;
        const uint64_t cancelledAllocGen = ctrlA.observedSet.version.allocationGeneration;
        ctrlA.failInExecute = false;
        // Recompile after failed recording (executor contract) and re-execute the same sun edit.
        if (!executor.compile(*device, graph, 64, 64, log) ||
            !executor.execute({.graphicsQueue = graphics}) ||
            !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Recovery after cancelled producer failed: " + log);
        }
        if (ctrlA.observedSet.cacheHit ||
            ctrlA.observedSet.lighting.radianceView == cancelledRadiance ||
            ctrlA.observedSet.version.allocationGeneration <= cancelledAllocGen) {
            return RHITestResult::fail("Cancelled unsubmitted producer polluted committed cache");
        }
        SHReadback shARecovered{};
        if (!readSH(*readbackA, shARecovered)) {
            return RHITestResult::fail("Recovered physical atmosphere produced invalid GPU SH");
        }

        const bench::Json evidence{
            {"frame0ProducerSegments", prodSegs0},
            {"frame1ProducerSegments", prodSegs1},
            {"frame0ContentRevision", frame0ContentRevA},
            {"frame0AllocGeneration", frame0AllocGenA},
            {"rebuiltAllocGeneration", ctrlA.observedSet.version.allocationGeneration},
            {"mediumShareCount", envSubsystem->mediumShareCount()},
            {"cacheHitCount", envSubsystem->physicalCacheHitCount()},
            {"producerCount", envSubsystem->physicalProducerCount()},
        };
        if (context.evidence) { context.evidence->json("provider-rg-scheduling.json", evidence); }
        else { bench::writeJson(context.outputDirectory / "provider-rg-scheduling.json", evidence); }
        return RHITestResult::pass("RenderGraph EnvironmentProvider scheduling, cross-view medium sharing, cache-hit zero-precompute, eviction separation, and cancellation safety verified");
    }
};
METALLIC_REGISTER_RHI_TEST(RenderGraphEnvironmentProviderSchedulingTest);

// ============================================================================
// Batch 2 External CommandBuffer Retention & Full Provider/Cloud Lifecycle Tests
// ============================================================================

class PhysicalEnvironmentExternalCmdBufRetentionTest final : public RHITest {
public:
    PhysicalEnvironmentExternalCmdBufRetentionTest()
    {
        name = "physical_environment_external_cmdbuf_retention_on_eviction";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("ProviderConsumerTestPass",
            "Declarative EnvironmentResourceSet test consumer",
            [] { return std::make_unique<ProviderConsumerTestPass>(); });
        (void)registered;

        std::unique_ptr<Device> device;
        DeviceDesc deviceDesc{
            .applicationName = "External CommandBuffer Environment Retention (optimal layouts)",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
        };
        vulkan::deviceExtensions(deviceDesc).preferUnifiedImageLayouts = false;
        auto created = createDevice(deviceDesc).transform([&](auto value) { device = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        Queue* graphics = device->getQueue(QueueType::Graphics);
        if (!graphics) { return RHITestResult::fail("Missing graphics queue"); }

        std::unique_ptr<Buffer> readbackRef, readbackEvicted;
        if (!createSHReadback(*device, readbackRef) || !createSHReadback(*device, readbackEvicted)) {
            return RHITestResult::fail("Readback buffer creation failed");
        }

        ProviderConsumerControl ctrl{
            .world = providerPhysicalWorld(),
            .observer = {0.0, 2.0, 0.0},
            .viewId = 701,
            .readbackTarget = readbackRef.get(),
        };
        struct ControlCleanup { ~ControlCleanup() { ProviderConsumerTestPass::controls.clear(); } } cleanup;
        ProviderConsumerTestPass::controls["ExtView"] = &ctrl;

        RenderGraph graph;
        graph.setName("ExternalCmdBufEnvRetention");
        graph.addNode("ProviderConsumerTestPass", "ExtView");
        graph.markOutput("ExtView.dummy");

        RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 64, 64, log)) {
            return RHITestResult::fail("compile failed: " + log);
        }

        // First compute a clean ground-truth reference SH for observer {0, 2, 0} on a separate standalone command buffer.
        std::unique_ptr<CommandPool> refPool;
        std::unique_ptr<CommandBuffer> refCmd;
        std::unique_ptr<Fence> refFence;
        if (!device->createCommandPool(*graphics).transform([&](auto p) { refPool = std::move(p); }) ||
            !refPool->createCommandBuffer().transform([&](auto c) { refCmd = std::move(c); }) ||
            !device->createFence(false).transform([&](auto f) { refFence = std::move(f); })) {
            return RHITestResult::fail("Reference command setup failed");
        }
        if (!refCmd->begin() || !executor.execute(*refCmd) || !refCmd->end()) {
            return RHITestResult::fail("Reference standalone execute failed");
        }
        CommandBuffer* refList[] = {refCmd.get()};
        if (!graphics->submit({.commandBuffers = refList, .signalFence = refFence.get()}) ||
            !refFence->wait(5'000'000'000ull)) {
            return RHITestResult::fail("Reference standalone submit failed");
        }
        SHReadback referenceSH{};
        if (!readSH(*readbackRef, referenceSH)) {
            return RHITestResult::fail("Reference SH readback invalid");
        }
        (void)refPool->reset();

        // Now record cmd0 WITHOUT RenderFrameContext for a brand-new observer {0, 50, 0},
        // gate its GPU execution on gateSem==1, and immediately evict it from the 8-entry cache
        // by submitting 10 MORE standalone CommandBuffers (no RenderFrameContext) with distinct observers!
        std::unique_ptr<Semaphore> gateSem;
        std::unique_ptr<CommandPool> pool0;
        std::unique_ptr<CommandBuffer> cmd0;
        std::unique_ptr<Fence> fenceAll;
        if (!device->createSemaphore().transform([&](auto s) { gateSem = std::move(s); }) ||
            !device->createCommandPool(*graphics).transform([&](auto p) { pool0 = std::move(p); }) ||
            !pool0->createCommandBuffer().transform([&](auto c) { cmd0 = std::move(c); }) ||
            !device->createFence(false).transform([&](auto f) { fenceAll = std::move(f); })) {
            return RHITestResult::fail("Gated standalone setup failed");
        }
        struct SignalGuard {
            Semaphore* sem = nullptr;
            Queue* q = nullptr;
            ~SignalGuard()
            {
                if (sem && sem->currentValue() < 1) { (void)sem->signal(1); }
                if (q) { (void)q->waitIdle(); }
            }
        } guard{gateSem.get(), graphics};

        // Record ground-truth for observer {0, 50, 0} by capturing its SH after eviction.
        // Note: observer altitude 50m vs 2m is close, so we also verify against a clean post-test run.
        ctrl.observer = {0.0, 50.0, 0.0};
        ctrl.readbackTarget = readbackEvicted.get();
        if (!cmd0->begin() || !executor.execute(*cmd0) || !cmd0->end()) {
            return RHITestResult::fail("Gated standalone cmd0 execute failed");
        }
        std::weak_ptr<void> weakPub0 = ctrl.observedSet.lighting.retainedResources;
        if (weakPub0.expired()) {
            return RHITestResult::fail("Missing retainedResources on cmd0 publication");
        }
        // Drop every test-side shared_ptr to pub0; ONLY cache + cmd0 hold it now.
        ctrl.observedSet = {};

        // Submit cmd0 blocked on gateSem == 1 (onAccepted commits pub0 into physicalPublications_ immediately on CPU!).
        CommandBuffer* list0[] = {cmd0.get()};
        const SemaphoreSubmitDesc gateWait{.semaphore = gateSem.get(), .value = 1, .stages = PipelineStageBits::AllCommands};
        if (!graphics->submit({.waitSemaphores = {&gateWait, 1}, .commandBuffers = list0})) {
            return RHITestResult::fail("Gated cmd0 submit failed");
        }

        // Record and submit 10 distinct observer positions WITHOUT RenderFrameContext while cmd0 is STILL blocked on GPU.
        // This overflows the 8-entry cache and evicts pub0 from EnvironmentLightingSubsystem::physicalPublications_!
        std::vector<std::unique_ptr<CommandPool>> evictPools;
        std::vector<std::unique_ptr<CommandBuffer>> evictCmds;
        for (uint32_t i = 0; i < 10; ++i) {
            std::unique_ptr<CommandPool> p;
            std::unique_ptr<CommandBuffer> c;
            if (!device->createCommandPool(*graphics).transform([&](auto v) { p = std::move(v); }) ||
                !p->createCommandBuffer().transform([&](auto v) { c = std::move(v); })) {
                return RHITestResult::fail("Eviction pool/cmd setup failed");
            }
            ctrl.observer = {0.0, 1000.0 * double(i + 1), 0.0};
            ctrl.readbackTarget = nullptr;
            if (!c->begin() || !executor.execute(*c) || !c->end()) {
                return RHITestResult::fail("Eviction standalone execute failed at i=" + std::to_string(i));
            }
            ctrl.observedSet = {};
            CommandBuffer* list[] = {c.get()};
            const bool last = (i == 9);
            if (!graphics->submit({.commandBuffers = list, .signalFence = last ? fenceAll.get() : nullptr})) {
                return RHITestResult::fail("Eviction standalone submit failed at i=" + std::to_string(i));
            }
            evictPools.push_back(std::move(p));
            evictCmds.push_back(std::move(c));
        }

        // pub0 is evicted from the subsystem cache, and NO RenderFrameContext exists:
        // cmd0's CommandBuffer::retainResource() MUST be keeping pub0 alive while GPU is still blocked!
        if (weakPub0.expired()) {
            return RHITestResult::fail("Evicted publication was destroyed while standalone CommandBuffer (no RenderFrameContext) was still in flight!");
        }

        // Unblock GPU, let cmd0 + all 10 eviction submissions execute and complete!
        if (!gateSem->signal(1) || !fenceAll->wait(5'000'000'000ull)) {
            return RHITestResult::fail("Failed waiting for unblocked standalone submissions");
        }
        SHReadback evictedSH{};
        if (!readSH(*readbackEvicted, evictedSH)) {
            return RHITestResult::fail("Readback from evicted standalone CommandBuffer produced invalid SH");
        }

        // Releasing cmd0's pool must drop the last reference to evicted pub0!
        cmd0.reset();
        (void)pool0->reset();
        pool0.reset();
        evictCmds.clear();
        evictPools.clear();
        if (!weakPub0.expired()) {
            return RHITestResult::fail("Evicted publication leaked after standalone CommandBuffer reset/destruction");
        }

        // Recompute observer {0, 50, 0} cleanly and verify evictedSH matches bit-for-bit!
        std::unique_ptr<CommandPool> verifyPool;
        std::unique_ptr<CommandBuffer> verifyCmd;
        std::unique_ptr<Fence> verifyFence;
        if (!device->createCommandPool(*graphics).transform([&](auto p) { verifyPool = std::move(p); }) ||
            !verifyPool->createCommandBuffer().transform([&](auto c) { verifyCmd = std::move(c); }) ||
            !device->createFence(false).transform([&](auto f) { verifyFence = std::move(f); })) {
            return RHITestResult::fail("Verify setup failed");
        }
        ctrl.observer = {0.0, 50.0, 0.0};
        ctrl.readbackTarget = readbackRef.get();
        if (!verifyCmd->begin() || !executor.execute(*verifyCmd) || !verifyCmd->end()) {
            return RHITestResult::fail("Verify execute failed");
        }
        CommandBuffer* vList[] = {verifyCmd.get()};
        if (!graphics->submit({.commandBuffers = vList, .signalFence = verifyFence.get()}) ||
            !verifyFence->wait(5'000'000'000ull)) {
            return RHITestResult::fail("Verify submit failed");
        }
        SHReadback verifySH{};
        if (!readSH(*readbackRef, verifySH) || verifySH != evictedSH) {
            return RHITestResult::fail("Evicted standalone CommandBuffer SH did not match fresh reference SH");
        }
        return RHITestResult::pass("Standalone CommandBuffer (no RenderFrameContext) retained evicted PhysicalPublication until GPU completion and read back exact SH");
    }
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentExternalCmdBufRetentionTest);

struct ProviderGroupReadbackBuffers {
    std::unique_ptr<render::Buffer> radianceTopMip; // 1x1 RGBA32Sfloat (16 bytes)
    std::unique_ptr<render::Buffer> pdfTopMip;      // 1x1 R32Sfloat (4 bytes)
    std::unique_ptr<render::Buffer> sh;             // 9x float4 (144 bytes)
    std::unique_ptr<render::Buffer> prefilterL7;    // 16x float4 from roughness-1.0 layer 7 (256 bytes)
    std::unique_ptr<render::Buffer> markerSource;   // 1x uint32 HostUpload (0x12345678)
    std::unique_ptr<render::Buffer> executedMarker; // 1x uint32 HostReadback (4 bytes)
};

struct ProviderGroupValues {
    std::array<float, 4> radianceTop{};
    float pdfTop = 0.0f;
    SHReadback sh{};
    std::array<float, 64> prefilterL7{};
    uint32_t marker = 0;
};

struct ProviderFullConsumerControl {
    environment::WorldEnvironment world = providerPhysicalWorld();
    std::array<double, 3> observer{0.0, 2.0, 0.0};
    uint64_t viewId = 101;
    bool injectFailingCustomProducer = false;
    bool failInExecute = false;
    bool executed = false;
    render::EnvironmentResourceSet observedSet{};
    ProviderGroupReadbackBuffers* readback = nullptr;
};

class ProviderGroupReadbackPass final : public render::ComputePass {
public:
    static inline std::unordered_map<std::string, ProviderFullConsumerControl*> controls;

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("dummy").buffer(16, 16).storageWrite();
        return reflection;
    }

    render::Result<> declareResourceRequests(
        const render::RenderGraphResourceRequestContext& context,
        render::RenderGraphResourceRequestBuilder& builder,
        std::string& log) override
    {
        auto* env = context.subsystem<render::EnvironmentLightingSubsystem>();
        auto iter = controls.find(context.passName());
        if (env == nullptr || iter == controls.end() || iter->second == nullptr) {
            log = "Missing full provider consumer control";
            return render::makeError(render::Error::InvalidArgument);
        }
        const auto& ctrl = *iter->second;
        auto declared = env->declareEnvironmentRequest(
            context,
            builder,
            render::EnvironmentRequest{
                .environment = ctrl.world.snapshot(),
                .observerWorldMetres = ctrl.observer,
                .viewId = ctrl.viewId,
                .needCelestial = true,
            },
            log);
        if (!declared) { return render::makeError(declared.error()); }
        if (ctrl.injectFailingCustomProducer) {
            render::RenderGraphProviderProducerWork failingWork;
            failingWork.debugName = "InjectedFailingProviderProducer";
            failingWork.providerId = render::EnvironmentLightingSubsystem::kSubsystemId;
            failingWork.dedupKey = 0xFA11DEADull ^ ctrl.viewId;
            failingWork.preferredQueue = render::QueueType::Graphics;
            failingWork.record = [](render::CommandBuffer&, std::string& recLog) -> render::Result<> {
                recLog = "Injected producer failure";
                return render::makeError(render::Error::Failure);
            };
            const uint64_t dedup = failingWork.dedupKey;
            if (auto r = builder.registerProducerWork(std::move(failingWork)); !r) { return r; }
            if (auto r = builder.requireProducer(dedup); !r) { return r; }
        }
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto iter = controls.find(context.passName());
        if (iter == controls.end() || iter->second == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        auto& ctrl = *iter->second;
        const auto* set = context.resources<EnvironmentResourceSet>();
        if (set == nullptr || !set->valid()) {
            return makeError(Error::InvalidArgument);
        }
        ctrl.observedSet = *set;
        ctrl.executed = true;
        if (ctrl.failInExecute) {
            return makeError(Error::Failure);
        }
        if (ctrl.readback == nullptr) { return {}; }
        auto& cmd = context.commandBuffer();

        // Write execution marker = 0x12345678u.
        if (ctrl.readback->executedMarker && ctrl.readback->markerSource) {
            auto src = ctrl.readback->markerSource->slice({0, 4});
            auto dst = ctrl.readback->executedMarker->slice({0, 4});
            if (!src || !dst || !cmd.copyBuffer(*src, *dst)) { return makeError(Error::Failure); }
            BufferBarrierDesc b{.buffer = ctrl.readback->executedMarker.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = 4}};
            if (!cmd.synchronize({.buffers = {&b, 1}})) { return makeError(Error::Failure); }
        }

        // Find radiance & pdf native Texture* from set->publications.
        Texture* radianceTex = nullptr;
        Texture* pdfTex = nullptr;
        for (const auto& pub : set->publications) {
            if (pub.logicalId.name == "atmosphere.radiance") { radianceTex = pub.texture; }
            else if (pub.logicalId.name == "atmosphere.pdf") { pdfTex = pub.texture; }
        }
        if (!radianceTex || !pdfTex || !set->lighting.sphericalHarmonicsBuffer || !set->lighting.prefilteredSpecularBuffer) {
            return makeError(Error::InvalidArgument);
        }

        const auto copyTopMip = [&](Texture* tex, Buffer& dstBuf, uint64_t byteSize) -> Result<> {
            const uint32_t topMip = tex->desc().mipCount - 1u;
            TextureBarrierDesc toSrc{.texture = tex,
                .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite | AccessBits::ShaderRead},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.baseMip = topMip, .mipCount = 1, .baseLayer = 0, .layerCount = 1}};
            if (auto r = cmd.synchronize({.textures = {&toSrc, 1}}); !r) { return r; }
            auto slice = dstBuf.slice({0, byteSize});
            if (!slice) { return makeError(slice.error()); }
            if (auto r = cmd.copyTextureToBuffer({.texture = tex, .buffer = *slice,
                    .width = 1, .height = 1, .depth = 1, .mipLevel = topMip}); !r) { return r; }
            TextureBarrierDesc backRead{.texture = tex,
                .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                .range = {.baseMip = topMip, .mipCount = 1, .baseLayer = 0, .layerCount = 1}};
            BufferBarrierDesc hostB{.buffer = &dstBuf,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = byteSize}};
            return cmd.synchronize({.textures = {&backRead, 1}, .buffers = {&hostB, 1}});
        };

        if (auto r = copyTopMip(radianceTex, *ctrl.readback->radianceTopMip, 16); !r) { return r; }
        if (auto r = copyTopMip(pdfTex, *ctrl.readback->pdfTopMip, 4); !r) { return r; }

        // Copy SH (144 bytes).
        {
            BufferBarrierDesc before{.buffer = set->lighting.sphericalHarmonicsBuffer,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite | AccessBits::ShaderRead},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.offset = 0, .size = sizeof(SHReadback)}};
            if (auto r = cmd.synchronize({.buffers = {&before, 1}}); !r) { return r; }
            auto src = set->lighting.sphericalHarmonicsBuffer->slice({0, sizeof(SHReadback)});
            auto dst = ctrl.readback->sh->slice({0, sizeof(SHReadback)});
            if (!src || !dst) { return makeError(Error::Failure); }
            if (auto r = cmd.copyBuffer(*src, *dst); !r) { return r; }
            BufferBarrierDesc hostB{.buffer = ctrl.readback->sh.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
                .range = {.offset = 0, .size = sizeof(SHReadback)}};
            if (auto r = cmd.synchronize({.buffers = {&hostB, 1}}); !r) { return r; }
        }

        // Copy PrefilteredSpecular roughness=1.0 layer 7 (first 16 texels = 256 bytes).
        {
            constexpr uint64_t kLayer7Offset = 7ull * 256ull * 128ull * 16ull;
            constexpr uint64_t kCopyBytes = 16ull * 16ull;
            BufferBarrierDesc before{.buffer = set->lighting.prefilteredSpecularBuffer,
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite | AccessBits::ShaderRead},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.offset = kLayer7Offset, .size = kCopyBytes}};
            if (auto r = cmd.synchronize({.buffers = {&before, 1}}); !r) { return r; }
            auto src = set->lighting.prefilteredSpecularBuffer->slice({kLayer7Offset, kCopyBytes});
            auto dst = ctrl.readback->prefilterL7->slice({0, kCopyBytes});
            if (!src || !dst) { return makeError(Error::Failure); }
            if (auto r = cmd.copyBuffer(*src, *dst); !r) { return r; }
            BufferBarrierDesc hostB{.buffer = ctrl.readback->prefilterL7.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
                .range = {.offset = 0, .size = kCopyBytes}};
            if (auto r = cmd.synchronize({.buffers = {&hostB, 1}}); !r) { return r; }
        }
        return {};
    }
};

class PhysicalEnvironmentFullLifecycleAndCloudsTest final : public RHITest {
public:
    PhysicalEnvironmentFullLifecycleAndCloudsTest()
    {
        name = "physical_environment_provider_full_lifecycle_and_clouds";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("ProviderGroupReadbackPass",
            "Full Radiance/PDF/SH/Prefilter provider group consumer",
            [] { return std::make_unique<ProviderGroupReadbackPass>(); });
        (void)registered;

        std::unique_ptr<Device> device;
        DeviceDesc deviceDesc{
            .applicationName = "Physical Environment Full Lifecycle & Clouds (optimal layouts)",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
        };
        vulkan::deviceExtensions(deviceDesc).preferUnifiedImageLayouts = false;
        auto created = createDevice(deviceDesc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        Queue* graphics = device->getQueue(QueueType::Graphics);
        if (!graphics) { return RHITestResult::fail("Missing graphics queue"); }

        const auto makeReadback = [&](ProviderGroupReadbackBuffers& rb) -> Result<> {
            auto alloc = [&](std::unique_ptr<Buffer>& buf, uint64_t size) {
                return device->createBuffer({.size = size, .structureStride = 4,
                    .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto b) { buf = std::move(b); });
            };
            if (auto r = alloc(rb.radianceTopMip, 16); !r) { return r; }
            if (auto r = alloc(rb.pdfTopMip, 4); !r) { return r; }
            if (auto r = alloc(rb.sh, sizeof(SHReadback)); !r) { return r; }
            if (auto r = alloc(rb.prefilterL7, 256); !r) { return r; }
            if (auto r = device->createBuffer({.size = 4, .structureStride = 4,
                    .usage = BufferUsageBits::TransferSource, .memoryLocation = MemoryLocation::HostUpload})
                    .transform([&](auto b) { rb.markerSource = std::move(b); }); !r) { return r; }
            if (void* m = rb.markerSource->map()) {
                const uint32_t magic = 0x12345678u;
                std::memcpy(m, &magic, sizeof(magic));
                rb.markerSource->flush({0, 4});
                rb.markerSource->unmap();
            }
            return alloc(rb.executedMarker, 4);
        };
        const auto resetMarker = [&](ProviderGroupReadbackBuffers& rb) {
            if (void* m = rb.executedMarker->map()) {
                uint32_t zero = 0;
                std::memcpy(m, &zero, sizeof(zero));
                rb.executedMarker->flush({0, 4});
                rb.executedMarker->unmap();
            }
        };
        const auto readValues = [&](ProviderGroupReadbackBuffers& rb, ProviderGroupValues& out) -> bool {
            auto copyOut = [](Buffer& b, void* dst, size_t n) -> bool {
                b.invalidate();
                const void* m = b.map();
                if (!m) { return false; }
                std::memcpy(dst, m, n);
                b.unmap();
                return true;
            };
            if (!copyOut(*rb.radianceTopMip, out.radianceTop.data(), 16) ||
                !copyOut(*rb.pdfTopMip, &out.pdfTop, 4) ||
                !copyOut(*rb.sh, out.sh.data(), sizeof(SHReadback)) ||
                !copyOut(*rb.prefilterL7, out.prefilterL7.data(), 256) ||
                !copyOut(*rb.executedMarker, &out.marker, 4)) {
                return false;
            }
            const bool finiteRad = std::all_of(out.radianceTop.begin(), out.radianceTop.end(), [](float v) { return std::isfinite(v); });
            const bool finiteSh = std::all_of(out.sh.begin(), out.sh.end(), [](float v) { return std::isfinite(v); });
            const bool finitePref = std::all_of(out.prefilterL7.begin(), out.prefilterL7.end(), [](float v) { return std::isfinite(v); });
            return finiteRad && std::isfinite(out.pdfTop) && finiteSh && finitePref &&
                out.radianceTop[0] > 0.0f && out.pdfTop > 0.0f && out.sh[0] > 0.0f && out.prefilterL7[0] > 0.0f;
        };
        const auto findPub = [](const EnvironmentResourceSet& set, std::string_view name) -> const ExternalResourcePublication* {
            for (const auto& p : set.publications) {
                if (p.logicalId.name == name) { return &p; }
            }
            return nullptr;
        };

        ProviderGroupReadbackBuffers rbA, rbB;
        if (!makeReadback(rbA) || !makeReadback(rbB)) { return RHITestResult::fail("Readback buffer setup failed"); }

        ProviderFullConsumerControl ctrlA{.world = providerPhysicalWorld(), .observer = {0.0, 2.0, 0.0}, .viewId = 101, .readback = &rbA};
        ProviderFullConsumerControl ctrlB{.world = providerPhysicalWorld(), .observer = {0.0, 12000.0, 0.0}, .viewId = 202, .readback = &rbB};
        struct Cleanup { ~Cleanup() { ProviderGroupReadbackPass::controls.clear(); } } cleanup;
        ProviderGroupReadbackPass::controls["ViewA"] = &ctrlA;
        ProviderGroupReadbackPass::controls["ViewB"] = &ctrlB;

        RenderGraph graph;
        graph.setName("FullEnvProviderLifecycle");
        graph.addNode("ProviderGroupReadbackPass", "ViewA");
        graph.addNode("ProviderGroupReadbackPass", "ViewB");
        graph.markOutput("ViewA.dummy");
        graph.markOutput("ViewB.dummy");

        RenderGraphExecutor exec;
        exec.setExecutionCaptureEnabled(true);
        std::string log;
        if (!exec.compile(*device, graph, 64, 64, log)) { return RHITestResult::fail("compile failed: " + log); }
        auto* envSub = exec.subsystemHost()->get<EnvironmentLightingSubsystem>();
        if (!envSub) { return RHITestResult::fail("Missing EnvironmentLightingSubsystem"); }

        // 1. Initial Frame 0: verify stable LogicalResourceId, complete sampled publication set (PDF, filteredSpecular, parameters),
        //    real Viewer capture resources with producerName & consumerNames, and cross-view static medium identity/generation.
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Frame 0 execute failed");
        }
        ProviderGroupValues vA0{}, vB0{};
        if (!readValues(rbA, vA0) || !readValues(rbB, vB0) || vA0.marker != 0x12345678u || vB0.marker != 0x12345678u) {
            return RHITestResult::fail("Frame 0 Radiance/PDF/SH/Prefilter readback invalid");
        }
        const auto snap0 = exec.executionSnapshot();
        if (!snap0) { return RHITestResult::fail("Frame 0 snapshot missing"); }
        // Verify GraphAccessPlan / Viewer snapshot contains ProviderOwned resources including pdf, parameters, filteredSpecular.
        bool sawPdf = false, sawParams = false, sawRadi = false, sawMedium = false;
        for (const auto& res : snap0->resources) {
            if (res.ownership != RenderGraphResourceOwnership::ProviderOwned) { continue; }
            if (res.producerName.empty() || res.consumerNames.empty()) {
                return RHITestResult::fail("Provider resource '" + res.name + "' missing producerName or consumerNames in snapshot");
            }
            if (res.name.find("atmosphere.pdf") != std::string::npos) { sawPdf = true; }
            if (res.name.find("atmosphere.parameters") != std::string::npos) { sawParams = true; }
            if (res.name.find("atmosphere.radiance") != std::string::npos) { sawRadi = true; }
            if (res.name.find("atmosphere.transmittance") != std::string::npos) { sawMedium = true; }
        }
        if (!sawPdf || !sawParams || !sawRadi || !sawMedium) {
            return RHITestResult::fail("Execution snapshot missing ProviderOwned PDF/parameters/radiance/transmittance resources");
        }
        const auto* transA0 = findPub(ctrlA.observedSet, "atmosphere.transmittance");
        const auto* transB0 = findPub(ctrlB.observedSet, "atmosphere.transmittance");
        if (!transA0 || !transB0 ||
            transA0->backingAllocationId != transB0->backingAllocationId ||
            transA0->version.allocationGeneration != transB0->version.allocationGeneration ||
            ctrlA.observedSet.logicalId == ctrlB.observedSet.logicalId) {
            return RHITestResult::fail("Cross-view isolation or shared medium allocationGeneration mismatch on Frame 0");
        }
        const LogicalResourceId stableLogicalA = ctrlA.observedSet.logicalId;
        const uint64_t mediumGen0 = transA0->version.allocationGeneration;
        const uint64_t mediumBacking0 = transA0->backingAllocationId;
        const uint64_t revA0 = ctrlA.observedSet.version.contentRevision;
        std::weak_ptr<void> weakPubA0 = ctrlA.observedSet.lighting.retainedResources;

        // 2. Same-content hit (Frame 1): 0 producers, bit-identical Radiance/PDF/SH/Prefilter, identical IDs & revisions.
        const uint64_t prodBefore1 = envSub->physicalProducerCount();
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Frame 1 cache-hit execute failed");
        }
        ProviderGroupValues vA1{};
        if (!readValues(rbA, vA1) || envSub->physicalProducerCount() != prodBefore1 || !ctrlA.observedSet.cacheHit ||
            vA1.radianceTop != vA0.radianceTop || vA1.pdfTop != vA0.pdfTop ||
            vA1.sh != vA0.sh || vA1.prefilterL7 != vA0.prefilterL7) {
            return RHITestResult::fail("Same-content hit triggered producer or altered Radiance/PDF/SH/Prefilter");
        }
        // Check ViewB's shared medium publication ALSO carries a valid submitted producerCompletion on cache hit!
        const auto* transB1 = findPub(ctrlB.observedSet, "atmosphere.transmittance");
        if (!transB1 || !transB1->producerCompletion.isSubmitted() || transB1->version.allocationGeneration != mediumGen0) {
            return RHITestResult::fail("Cross-view shared medium lost producerCompletion or allocationGeneration on cache hit");
        }

        // 3. Observer-only edit on ViewA: stable LogicalResourceId, advancing contentRevision, SAME static medium!
        ctrlA.observer = {0.0, 22000.0, 0.0};
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Observer-only edit execute failed");
        }
        ProviderGroupValues vAObs{};
        const auto* transAObs = findPub(ctrlA.observedSet, "atmosphere.transmittance");
        if (!readValues(rbA, vAObs) ||
            ctrlA.observedSet.logicalId != stableLogicalA ||
            ctrlA.observedSet.version.contentRevision <= revA0 ||
            !ctrlA.observedSet.mediumShared || !transAObs ||
            transAObs->backingAllocationId != mediumBacking0 ||
            transAObs->version.allocationGeneration != mediumGen0 ||
            vAObs.sh == vA0.sh) {
            return RHITestResult::fail("Observer-only edit broke stable LogicalResourceId, advancing contentRevision, or shared medium");
        }
        const uint64_t revAObs = ctrlA.observedSet.version.contentRevision;

        // 4. Sun-only edit on ViewA: stable LogicalResourceId, advancing contentRevision, SAME static medium, all 4 channels update!
        ctrlA.observer = {0.0, 2.0, 0.0};
        ctrlA.world.sun.topOfAtmosphereIrradiance = {3.5f, 3.2f, 2.9f};
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Sun-only edit execute failed");
        }
        ProviderGroupValues vASun{};
        const auto* transASun = findPub(ctrlA.observedSet, "atmosphere.transmittance");
        if (!readValues(rbA, vASun) ||
            ctrlA.observedSet.logicalId != stableLogicalA ||
            ctrlA.observedSet.version.contentRevision <= revAObs ||
            !ctrlA.observedSet.mediumShared || !transASun ||
            transASun->backingAllocationId != mediumBacking0 ||
            transASun->version.allocationGeneration != mediumGen0 ||
            vASun.radianceTop == vA0.radianceTop || vASun.pdfTop == vA0.pdfTop ||
            vASun.sh == vA0.sh || vASun.prefilterL7 == vA0.prefilterL7) {
            return RHITestResult::fail("Sun-only edit failed to advance contentRevision, preserve medium, or update all 4 channels");
        }
        // Brighter sun (~2x irradiance) must increase Radiance, PDF, SH, and Prefilter together!
        if (!(vASun.radianceTop[0] > vA0.radianceTop[0] && vASun.pdfTop > vA0.pdfTop &&
              vASun.sh[0] > vA0.sh[0] && vASun.prefilterL7[0] > vA0.prefilterL7[0])) {
            return RHITestResult::fail("Sun-only brightness increase did not consistently increase Radiance, PDF, SH, and Prefilter");
        }

        // 5. Evict Frame 0's PhysicalPublication (>8 distinct sun edits) and verify:
        //    (a) evicted PhysicalPublication IS freed (weakPubA0.expired() == true — MediumPublication does NOT pin it!)
        //    (b) static medium LUTs STILL survive with the EXACT SAME backingAllocationId and allocationGeneration!
        ctrlA.observedSet = {};
        ctrlB.observedSet = {};
        for (uint32_t i = 0; i < 10; ++i) {
            ctrlA.world = providerPhysicalWorld();
            ctrlA.world.sun.topOfAtmosphereIrradiance.x += 0.3f * float(i + 1);
            if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
                return RHITestResult::fail("Eviction loop execute failed");
            }
            ctrlA.observedSet = {};
            ctrlB.observedSet = {};
        }
        // Drain one extra idle frame so DeferredReleaseQueue collects retired evicted publications.
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Post-eviction drain frame failed");
        }
        const auto* transPostEvict = findPub(ctrlA.observedSet, "atmosphere.transmittance");
        if (!weakPubA0.expired()) {
            return RHITestResult::fail("Evicted PhysicalPublication was kept alive by MediumPublication!");
        }
        if (!transPostEvict || transPostEvict->backingAllocationId != mediumBacking0 ||
            transPostEvict->version.allocationGeneration != mediumGen0) {
            return RHITestResult::fail("Static medium LUTs did not survive full PhysicalPublication eviction");
        }

        // 6. Cloud environment consistent group publication (ViewA gets dense clouds; ViewB stays clear sky for isolation!).
        ctrlA.world = providerPhysicalWorld();
        ctrlA.world.weather.cloudEnabled = true;
        ctrlA.world.weather.cloudCoverage = 0.75f;
        ctrlA.world.weather.cloudDensity = 0.08f;
        ctrlA.world.weather.cloudExtinctionPerKm = 18.0f;
        ctrlA.world.weather.cloudBaseAltitudeKm = 1.5f;
        ctrlA.world.weather.cloudTopAltitudeKm = 4.0f;
        ctrlA.observer = {0.0, 2.0, 0.0};
        ctrlB.world = providerPhysicalWorld();
        ctrlB.observer = {0.0, 2.0, 0.0};
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Cloud environment frame failed");
        }
        ProviderGroupValues vCloud{}, vClear{};
        if (!readValues(rbA, vCloud) || !readValues(rbB, vClear)) {
            return RHITestResult::fail("Cloud/Clear readback invalid");
        }
        // Cross-view/scene isolation at the exact same observer altitude: ViewA has clouds, ViewB is clear sky!
        if (ctrlA.observedSet.lighting.radianceView == ctrlB.observedSet.lighting.radianceView ||
            vCloud.radianceTop == vClear.radianceTop ||
            vCloud.pdfTop == vClear.pdfTop ||
            vCloud.sh == vClear.sh ||
            vCloud.prefilterL7 == vClear.prefilterL7) {
            return RHITestResult::fail("Cloud publication did not produce a consistent distinct Radiance/PDF/SH/Prefilter group isolated from clear-sky ViewB");
        }
        // Verify numerical self-consistency: all 4 cloud channels are finite, positive, and share one publication revision/gen.
        const auto* cRad = findPub(ctrlA.observedSet, "atmosphere.radiance");
        const auto* cPdf = findPub(ctrlA.observedSet, "atmosphere.pdf");
        const auto* cSh = findPub(ctrlA.observedSet, "atmosphere.sphericalHarmonics");
        const auto* cPref = findPub(ctrlA.observedSet, "atmosphere.prefilteredSpecular");
        if (!cRad || !cPdf || !cSh || !cPref ||
            cRad->version.allocationGeneration != cPdf->version.allocationGeneration ||
            cRad->version.allocationGeneration != cSh->version.allocationGeneration ||
            cRad->version.allocationGeneration != cPref->version.allocationGeneration ||
            cRad->version.contentRevision != cSh->version.contentRevision) {
            return RHITestResult::fail("Cloud Radiance/PDF/SH/Prefilter publications did not share identical version/generation");
        }

        // 7. Inject unaccepted Producer failure -> consumer depending on unaccepted producer MUST NOT be submitted!
        //    Meanwhile, previously accepted cloud consumer data remains valid and recoverable after recompile.
        resetMarker(rbA);
        resetMarker(rbB);
        ctrlA.world.sun.topOfAtmosphereIrradiance.y += 2.25f; // New uncached content
        ctrlA.injectFailingCustomProducer = true;
        ctrlA.executed = false;
        ctrlB.executed = false;
        if (exec.execute({.graphicsQueue = graphics}).has_value()) {
            return RHITestResult::fail("Expected injected failing producer frame to return error");
        }
        (void)graphics->waitIdle();
        uint32_t unsubmittedMarkerA = 0xFFFFFFFFu;
        {
            rbA.executedMarker->invalidate();
            if (const void* m = rbA.executedMarker->map()) {
                std::memcpy(&unsubmittedMarkerA, m, sizeof(unsubmittedMarkerA));
                rbA.executedMarker->unmap();
            }
        }
        if (ctrlA.executed || unsubmittedMarkerA != 0u) {
            return RHITestResult::fail("Consumer depending on unaccepted producer was executed or submitted to GPU!");
        }

        // Recompile and verify:
        // (a) existing committed cloud content still hits cache cleanly,
        // (b) failed sun-edit content rebuilds cleanly from scratch (was never polluted into cache).
        ctrlA.injectFailingCustomProducer = false;
        if (!exec.compile(*device, graph, 64, 64, log)) {
            return RHITestResult::fail("Recompile after injected producer failure failed: " + log);
        }
        if (!exec.execute({.graphicsQueue = graphics}) || !exec.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("Recovery execute after producer failure failed");
        }
        ProviderGroupValues vRecovered{};
        if (ctrlA.observedSet.cacheHit || !readValues(rbA, vRecovered) || vRecovered.marker != 0x12345678u) {
            return RHITestResult::fail("Cancelled producer polluted cache or failed to recover cleanly after recompile");
        }

        // Also test committed previous version readback (clear-sky ViewB still cached & reads back bit-identically!).
        ProviderGroupValues vBAfterRecovery{};
        if (!ctrlB.observedSet.cacheHit || !readValues(rbB, vBAfterRecovery) ||
            vBAfterRecovery.sh != vClear.sh || vBAfterRecovery.radianceTop != vClear.radianceTop) {
            return RHITestResult::fail("Previously accepted ViewB did not read back cleanly after ViewA failure & recompile");
        }

        return RHITestResult::pass("Provider full lifecycle (hit, observer-only, sun-only, cross-view isolation, medium eviction independence, cloud group Radiance/PDF/SH/Prefilter consistency, unaccepted-producer non-submission & recompile recovery) verified under optimal layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentFullLifecycleAndCloudsTest);

} // namespace
} // namespace metallic::tests
