#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
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

} // namespace
} // namespace metallic::tests
