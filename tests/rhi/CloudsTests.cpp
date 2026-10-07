#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>

namespace metallic::tests {
namespace {

using CloudReadback = std::array<std::array<float, 4>, 47>;
struct alignas(16) CloudsProbeParams {
    render::AtmospherePrecomputeParams resources;
    render::GPUBufferSpan inputAerial;
    render::ShaderSampledImage radiance;
};
constexpr uint64_t kCloudsProbeABI = 0x434c4f5544500001ull;
static_assert(sizeof(CloudsProbeParams) == 80 && offsetof(CloudsProbeParams, radiance) == 76);

render::Result<> cloudProbe(render::Device& device, render::ComputeKernel& kernel,
    const environment::WorldEnvironment& world, double elapsed, CloudReadback& values, std::string& log,
    std::unique_ptr<render::AtmosphereResourcesGPU>& publication,
    const render::AtmosphereResourcesGPU* sharedMedium = nullptr,
    std::array<double, 3> observerWorldMetres = {0.0, 2.0, 0.0})
{
    using namespace render;
    publication = std::make_unique<AtmosphereResourcesGPU>();
    auto result = publication->initialize(device, log, sharedMedium);
    if (!result) { return result; }
    std::unique_ptr<Buffer> output;
    result = device.createBuffer(BufferDesc{.size = sizeof(values), .structureStride = 16,
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
        .transform([&](auto value) { output = std::move(value); });
    if (!result) { return result; }
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    if (!(result = gpu.initialize(device))) { return result; }
    if (!(result = publication->record(*gpu.commands, world.snapshot(1, 1, 1, 1, 1, elapsed), observerWorldMetres, log))) { return result; }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    CloudsProbeParams probe{};
    probe.resources.parameters = writer.bufferSpan(publication->parametersBuffer(), 16, 16);
    probe.resources.transmittance = writer.sampledImage(publication->transmittanceView());
    probe.resources.multiScattering = writer.sampledImage(publication->multiScatteringView());
    probe.resources.skyView = writer.sampledImage(publication->skyView());
    probe.resources.sourceMip = writer.sampledImage(publication->cloudShadowView());
    probe.resources.aerial = writer.bufferSpan(output.get(), 16, 16);
    probe.inputAerial = writer.bufferSpan(publication->aerialBuffer(), 16, 16);
    probe.radiance = writer.sampledImage(publication->radianceView());
    auto encoded = writer.encode(probe, kCloudsProbeABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (!(result = kernel.dispatch(*gpu.commands, *encoded, 1))) { return result; }
    BufferBarrierDesc barrier{.buffer = output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = sizeof(values)}};
    if (!(result = gpu.commands->synchronize(BarrierDesc{.buffers = {&barrier, 1}}))) { return result; }
    if (!(result = gpu.submitAndWait())) { return result; }
    output->invalidate();
    const void* mapped = output->map();
    if (mapped == nullptr) { return makeError(Error::Failure); }
    std::memcpy(values.data(), mapped, sizeof(values));
    output->unmap();
    for (const auto& value : values) {
        for (float channel : value) {
            if (!std::isfinite(channel)) { log = "Nonfinite cloud probe output"; return makeError(Error::Failure); }
        }
        if (value[3] != 1.0f) { log = "Cloud probe output ABI mismatch"; return makeError(Error::Failure); }
    }
    return {};
}

template<size_t Count>
bool saveCloudProbe(RHITestContext& context, const std::string& label, const std::array<std::array<float, 4>, Count>& values)
{
    std::ofstream output(context.outputDirectory / (label + ".tsv"));
    output << "probe\tx\ty\tz\tw\n" << std::setprecision(10);
    for (size_t index = 0; index < values.size(); ++index) {
        output << index;
        for (float value : values[index]) { output << '\t' << value; }
        output << '\n';
    }
    if (context.evidence) {
        bench::Json observations = bench::Json::array();
        for (const auto& value : values) { observations.push_back(value); }
        context.evidence->json(label + ".json", observations);
    }
    return bool(output);
}

bool cloudClose(const std::array<float, 4>& a, const std::array<float, 4>& b,
    float relative = 1e-5f, float absolute = 1e-6f)
{
    for (uint32_t channel = 0; channel < 3; ++channel) {
        if (std::abs(a[channel] - b[channel]) > absolute + relative * std::max(std::abs(a[channel]), std::abs(b[channel]))) { return false; }
    }
    return true;
}

bool cloudBoundedT(const std::array<float, 4>& value)
{
    return value[0] >= 0.0f && value[1] >= 0.0f && value[2] >= 0.0f &&
        value[0] <= 1.00001f && value[1] <= 1.00001f && value[2] <= 1.00001f;
}

class CloudTransportTest final : public RHITest {
public:
    CloudTransportTest() { name = "cloud_transport_weather_shadow_domains"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.clouds.zero", "environment.clouds.wind", "environment.clouds.sky.aerial.direct",
            "environment.clouds.shadow.map", "environment.clouds.static.medium.reuse"}, bench::Layer::Core,
            "binding", "binding", {"cloud-clear.tsv", "cloud-dense.tsv", "cloud-wind.tsv", "cloud-night.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Cloud transport numerical probes",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Cloud device setup: " + std::string(toString(result))); }
        ComputeKernel kernel;
        std::string log;
        result = ShaderRegistry::instance().getComputeKernel(*device,
            SlangShaderDesc{.moduleName = "CloudsProbe", .entryPointName = "cloudsProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            ComputeKernelDesc{.parameters = parameterAbi<CloudsProbeParams>(kCloudsProbeABI, ParameterTransport::InlinePush),
                .debugName = "Cloud medium numerical probe"}, kernel, log);
        if (!result) { return RHITestResult::fail("Cloud shader setup: " + log); }
        environment::WorldEnvironment world;
        world.source = environment::EnvironmentSource::PhysicalAtmosphere;
        world.sun.enabled = true;
        world.sun.direction = {0.0f, -1.0f, 0.0f};
        world.moon.enabled = true; // Below the secondary budget threshold: analytic fallback.
        std::unique_ptr<AtmosphereResourcesGPU> clear, dense, repeated, zero, wind, seed, night, terrain, orbit;
        CloudReadback clearValues, denseValues, repeatValues, zeroValues, windValues, seedValues, nightValues, terrainValues, orbitValues;
        auto probe = [&](const char* label, double elapsed, CloudReadback& values,
            std::unique_ptr<AtmosphereResourcesGPU>& publication, const AtmosphereResourcesGPU* medium,
            std::array<double, 3> observerWorldMetres = {0.0, 2.0, 0.0}) {
            auto probeResult = cloudProbe(*device, kernel, world, elapsed, values, log, publication, medium, observerWorldMetres);
            return probeResult && saveCloudProbe(context, label, values);
        };
        if (!probe("cloud-clear", 120.0, clearValues, clear, nullptr)) { return RHITestResult::fail(log); }
        const auto disabledLater = buildGPUAtmosphereParameters(world.snapshot(1, 1, 1, 1, 1, 10000.0), {0.0, 2.0, 0.0});
        if (clear->parameters().cloudAdvection != disabledLater.cloudAdvection ||
            clear->parameters().weatherComposition != disabledLater.weatherComposition) {
            return RHITestResult::fail("Disabled clouds created a time-dependent GPU publication");
        }
        world.weather.cloudEnabled = true;
        world.weather.cloudCoverage = 1.0f;
        world.weather.cloudDensity = 0.75f;
        world.weather.noiseSeed = 27;
        world.weather.windSpeed = 30.0f;
        if (!probe("cloud-dense", 0.0, denseValues, dense, clear.get())) { return RHITestResult::fail(log); }
        if (dense->transmittanceView() != clear->transmittanceView() || dense->multiScatteringView() != clear->multiScatteringView() ||
            !cloudClose(denseValues[27], clearValues[27], 0.0f, 0.0f) || !cloudClose(denseValues[28], clearValues[28], 0.0f, 0.0f)) {
            return RHITestResult::fail("Dynamic clouds rebuilt or altered the static medium LUTs");
        }
        clear.reset(); // Shared image owners must survive publication eviction.
        if (!probe("cloud-repeat", 0.0, repeatValues, repeated, dense.get())) { return RHITestResult::fail(log); }
        if (denseValues != repeatValues) { return RHITestResult::fail("Same seed/time cloud publication is not deterministic"); }
        if (denseValues[26][0] != 1.0f || denseValues[26][1] != 320.0f || denseValues[26][2] != 0.75f ||
            denseValues[30][0] != 48.0f || denseValues[30][1] != 0.0f || denseValues[30][2] != 1.0f ||
            denseValues[2][2] != 1.0f || denseValues[3][0] != 0.0f) {
            return RHITestResult::fail("320-byte atmosphere ABI or dominant Sun cloud-shadow budget mismatch");
        }
        if (!(denseValues[0][0] < 0.9f) || std::abs(denseValues[0][0] - denseValues[1][0]) > 0.035f ||
            !(denseValues[4][0] < 0.9f * denseValues[5][0]) || !cloudClose(denseValues[6], denseValues[7])) {
            return RHITestResult::fail("Cloud layer failed to attenuate ground sunlight, map disagrees, or above-cloud sunlight is attenuated");
        }
        for (uint32_t index : {9u, 11u, 13u, 23u, 25u, 33u, 35u}) {
            if (!cloudBoundedT(denseValues[index])) { return RHITestResult::fail("Cloud camera transmittance outside [0,1]"); }
        }
        if (!(denseValues[11][0] < denseValues[9][0]) || denseValues[10][0] <= 0.0f ||
            !cloudClose(denseValues[12], denseValues[10], 0.25f, 1.0f) ||
            !cloudClose(denseValues[13], denseValues[11], 0.0f, 0.05f) ||
            !cloudClose(denseValues[32], denseValues[34]) || !cloudClose(denseValues[33], denseValues[35])) {
            return RHITestResult::fail("Cloud aerial lookup, distance attenuation, or offset primary origin disagrees with segment integration");
        }
        if (std::abs(denseValues[17][0] - denseValues[17][1]) > 1e-6f ||
            std::abs(denseValues[18][0] - denseValues[18][1]) > 1e-6f || denseValues[18][0] != 1.0f ||
            denseValues[21][0] < 0.99f || denseValues[21][0] > 1.001f || denseValues[21][1] < 0.8f || denseValues[21][1] > 1.0f ||
            !cloudClose(denseValues[22], denseValues[24]) || !cloudClose(denseValues[23], denseValues[25]) ||
            denseValues[31][0] != 1.0f || denseValues[31][1] != 1.0f) {
            return RHITestResult::fail("Cloud fallback, bounded phase energy, or zero-density identity failed");
        }
        if (!(denseValues[36][0] < 0.9f * denseValues[37][0]) ||
            std::abs(denseValues[38][0] - denseValues[38][1]) > 1e-6f) {
            return RHITestResult::fail("Unmapped secondary daylight Moon failed analytic cloud receiver attenuation");
        }
        if (std::abs(denseValues[19][0] - denseValues[19][1]) > 0.003f ||
            !cloudClose(denseValues[16], denseValues[14], 0.1f, 10.0f) ||
            !cloudClose(denseValues[15], {denseValues[14][0] + denseValues[29][0],
                denseValues[14][1] + denseValues[29][1], denseValues[14][2] + denseValues[29][2], 1.0f}, 1e-5f, 0.01f)) {
            return RHITestResult::fail("Cloud advection covariance or disk-free environment capture policy failed");
        }
        world.weather.cloudDensity = 0.0f;
        if (!probe("cloud-zero", 120.0, zeroValues, zero, dense.get())) { return RHITestResult::fail(log); }
        for (uint32_t index : {0u, 4u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 27u, 28u, 29u}) {
            if (!cloudClose(zeroValues[index], clearValues[index], 0.0f, 0.0f)) {
                return RHITestResult::fail("Zero cloud density changed clear sky/camera/direct/capture domain at probe " + std::to_string(index));
            }
        }
        world.weather.cloudDensity = 0.75f;
        if (!probe("cloud-wind", 120.0, windValues, wind, dense.get())) { return RHITestResult::fail(log); }
        if (std::abs(windValues[20][0] - denseValues[20][0]) < 1e-5f ||
            std::abs(windValues[20][1] - denseValues[20][1]) <
                1e-3f * std::max(windValues[20][1], denseValues[20][1]) ||
            cloudClose(windValues[14], denseValues[14], 1e-5f, 0.001f) || windValues[39][0] <= 0.0f) {
            return RHITestResult::fail("Real-time wind did not change cloud density, receiver shadow and captured sky while astronomy is paused");
        }
        world.weather.noiseSeed = 28;
        if (!probe("cloud-seed", 120.0, seedValues, seed, dense.get())) { return RHITestResult::fail(log); }
        if (std::abs(seedValues[20][0] - windValues[20][0]) < 1e-5f) { return RHITestResult::fail("Cloud noise seed did not affect density field"); }
        world.sun.enabled = false;
        world.moon.enabled = true;
        world.moon.direction = {0.0f, -1.0f, 0.0f};
        world.weather.precipitation = 1.0f;
        if (!probe("cloud-night", 120.0, nightValues, night, dense.get())) { return RHITestResult::fail(log); }
        if (nightValues[30][0] != 0.0f || nightValues[30][1] != 48.0f || nightValues[30][2] != 2.0f ||
            nightValues[2][2] != 0.0f || nightValues[3][0] != 1.0f ||
            nightValues[26][2] != 1.125f || !(nightValues[20][1] < seedValues[20][1]) ||
            !(nightValues[36][0] < 0.9f * nightValues[37][0]) ||
            std::abs(nightValues[38][0] - nightValues[38][1]) > 0.035f) {
            return RHITestResult::fail("Dominant Moon map or Moon receiver attenuation failed");
        }
        world.sun.enabled = true;
        world.sun.direction = {-0.8660254f, -0.5f, 0.0f};
        world.moon.enabled = false;
        world.weather.cloudCoverage = 0.9f;
        world.weather.cloudDensity = 0.05f;
        world.weather.precipitation = 0.0f;
        world.weather.windSpeed = 0.0f;
        if (!probe("cloud-elevated-terrain", 120.0, terrainValues, terrain, dense.get())) { return RHITestResult::fail(log); }
        if (terrainValues[40][0] > 0.05f || terrainValues[40][1] < 0.005f ||
            terrainValues[40][2] <= 0.0f || terrainValues[40][2] >= 1.0f) {
            return RHITestResult::fail("300 m terrain cloud-shadow ray projection missed parallax or disagreed with analytic source columns");
        }
        // For this thin forward-lit layer, the exact source-facing cloud
        // radiance exceeds FP16. Check the 32F sky and capture preserve it.
        // The solar elevation is 30 degrees: the nearest sky nodes differ by
        // <2.8 degrees, versus the HG lobe width (1-g)/sqrt(g) ~=16.5 degrees.
        // A 10% bound includes that angular interpolation and finite LUT noise.
        if (terrainValues[46][0] <= 65504.0f || terrainValues[14][0] <= 65504.0f || terrainValues[16][0] <= 65504.0f ||
            !cloudClose(terrainValues[14], terrainValues[46], 0.1f, 1.0f) ||
            !cloudClose(terrainValues[16], terrainValues[14], 0.02f, 1.0f)) {
            return RHITestResult::fail("32F sky/capture clipped physically valid weak-cloud forward radiance or disagreed with exact direction integration");
        }
        if (!probe("cloud-orbital-downward", 120.0, orbitValues, orbit, dense.get(), {0.0, 150000.0, 0.0})) {
            return RHITestResult::fail(log);
        }
        const std::array<float, 4> expectedOrbitalT{orbitValues[44][0] * orbitValues[43][0],
            orbitValues[44][1] * orbitValues[43][0], orbitValues[44][2] * orbitValues[43][0], 1.0f};
        if (!cloudBoundedT(orbitValues[42]) || !cloudBoundedT(orbitValues[44]) || orbitValues[41][0] <= 0.0f ||
            orbitValues[43][0] <= 0.0f || orbitValues[43][0] >= 0.99f ||
            orbitValues[42][0] >= orbitValues[44][0] || !cloudClose(orbitValues[42], expectedOrbitalT, 0.05f, 0.001f)) {
            return RHITestResult::fail("150 km downward cloud ray missed thin shell or disagreed with independent cloud/gas extinction composition");
        }
        world.weather.humidity = 0.75f;
        AtmosphereResourcesGPU incompatible;
        if (!incompatible.initialize(*device, log, dense.get())) { return RHITestResult::fail(log); }
        bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
        if (!gpu.initialize(*device)) { return RHITestResult::fail("Medium mismatch command setup failed"); }
        if (incompatible.record(*gpu.commands, world.snapshot(), {0.0, 2.0, 0.0}, log)) {
            return RHITestResult::fail("Weather-modified effective aerosol reused incompatible static LUTs");
        }
        world.weather.humidity = 0.0f;
        world.weather.cloudTopAltitudeKm = 101.0f;
        if (incompatible.record(*gpu.commands, world.snapshot(), {0.0, 2.0, 0.0}, log)) {
            return RHITestResult::fail("Active cloud shell outside atmosphere was accepted with inconsistent transport domains");
        }
        return RHITestResult::pass("Nine submitted cloud publications: zero identity, seed/time reproducibility, wind/precipitation, bounded scattering, 32-cube aerial, Sun/Moon maps, elevated terrain projection, 150 km downward shell, fallback and retained static LUT ownership");
    }
};
METALLIC_REGISTER_RHI_TEST(CloudTransportTest);

using CloudIBLReadback = std::array<std::array<float, 4>, 14>;
struct alignas(16) CloudIBLProbeParams {
    render::ShaderSampledImage radiance;
    render::ShaderSampledImage pdf;
    render::GPUBufferSpan harmonics;
    render::GPUBufferSpan specular;
    render::GPUBufferSpan output;
    uint32_t reserved = 0;
};
constexpr uint64_t kCloudIBLProbeABI = 0x434c4f5544490001ull;
static_assert(sizeof(CloudIBLProbeParams) == 48 && offsetof(CloudIBLProbeParams, output) == 32);

struct CloudProviderRecording {
    render::Device& device;
    render::RenderWorld world;
    render::RenderSubsystemHost host;
    render::RenderFrameContext frame;
    render::QueueSubmissionTracker tracker;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> commands;
    uint64_t frameIndex = 0;

    explicit CloudProviderRecording(render::Device& device) : device(device) {}
    ~CloudProviderRecording()
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
    render::Result<> submit()
    {
        auto result = commands->end();
        host.endFrame();
        render::CommandBuffer* buffers[]{commands.get()};
        if (result) { result = tracker.submit({.commandBuffers = buffers}, frame); }
        return result ? frame.wait(30'000'000'000ull) : result;
    }
};

class CloudIBLPublicationTest final : public RHITest {
public:
    CloudIBLPublicationTest() { name = "cloud_provider_capture_sh_specular_pdf"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.clouds.capture", "environment.clouds.irradiance.sh",
            "environment.clouds.specular", "environment.clouds.importance.pdf", "environment.clouds.publication"},
            bench::Layer::Core, "binding", "binding", {"cloud-ibl-clear.tsv", "cloud-ibl-dense.tsv", "cloud-ibl-wind.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Cloud environment IBL publication",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Cloud IBL device setup: " + std::string(toString(result))); }
        ComputeKernel kernel;
        std::string log;
        result = ShaderRegistry::instance().getComputeKernel(*device,
            SlangShaderDesc{.moduleName = "CloudIBLProbe", .entryPointName = "cloudIBLProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            ComputeKernelDesc{.parameters = parameterAbi<CloudIBLProbeParams>(kCloudIBLProbeABI, ParameterTransport::InlinePush),
                .debugName = "Cloud IBL publication probe"}, kernel, log);
        if (!result) { return RHITestResult::fail(log); }
        std::unique_ptr<Buffer> output;
        result = device->createBuffer({.size = sizeof(CloudIBLReadback), .structureStride = 16,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
            .transform([&](auto value) { output = std::move(value); });
        if (!result) { return RHITestResult::fail("Cloud IBL readback allocation failed"); }
        CloudProviderRecording recording(*device);
        if (!recording.initialize(log)) { return RHITestResult::fail(log); }
        environment::WorldEnvironment world;
        world.source = environment::EnvironmentSource::PhysicalAtmosphere;
        world.sun.enabled = true;
        world.sun.direction = {0.0f, -1.0f, 0.0f};
        const auto authoredSun = world.sun;
        EnvironmentLightingSnapshot clear, dense, wind, repeated;
        CloudIBLReadback clearValues, denseValues, windValues, repeatedValues;
        auto readPublication = [&](const char* label, double elapsed, EnvironmentLightingSnapshot& publication, CloudIBLReadback& values) {
            if (!recording.begin(log)) { return false; }
            auto resolved = recording.host.get<EnvironmentLightingSubsystem>()->resolveRadiance(*device, *recording.commands,
                recording.host, world.snapshot(1, 1, 1, 1, 1, elapsed), {0.0, 2.0, 0.0}, log);
            if (!resolved) { return false; }
            publication = std::move(*resolved);
            if (!publication.valid() || !publication.prefilteredSpecularBuffer || !publication.cloudShadowView) { return false; }
            auto registry = ResourceRegistry::forDevice(*device);
            if (!registry) { return false; }
            ParameterWriter writer(*device, **registry, &recording.frame);
            CloudIBLProbeParams probe{};
            probe.radiance = writer.sampledImage(publication.radianceView);
            probe.pdf = writer.sampledImage(publication.pdfView);
            probe.harmonics = writer.bufferSpan(publication.sphericalHarmonicsBuffer, 16, 16);
            probe.specular = writer.bufferSpan(publication.prefilteredSpecularBuffer, 16, 16);
            probe.output = writer.bufferSpan(output.get(), 16, 16);
            auto encoded = writer.encode(probe, kCloudIBLProbeABI, ParameterTransport::InlinePush);
            if (!encoded || !kernel.dispatch(*recording.commands, *encoded, 1)) { return false; }
            BufferBarrierDesc barrier{.buffer = output.get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = sizeof(values)}};
            if (!recording.commands->synchronize({.buffers = {&barrier, 1}}) || !recording.submit()) { return false; }
            output->invalidate();
            const void* mapped = output->map();
            if (!mapped) { return false; }
            std::memcpy(values.data(), mapped, sizeof(values));
            output->unmap();
            for (const auto& value : values) {
                for (float channel : value) { if (!std::isfinite(channel)) { return false; } }
                if (value[3] != 1.0f) { return false; }
            }
            return saveCloudProbe(context, label, values);
        };
        if (!readPublication("cloud-ibl-clear", 0.0, clear, clearValues)) { return RHITestResult::fail("Clear IBL publication: " + log); }
        world.weather.cloudEnabled = true;
        world.weather.cloudCoverage = 1.0f;
        world.weather.cloudDensity = 0.75f;
        world.weather.windSpeed = 30.0f;
        world.weather.noiseSeed = 27;
        if (!readPublication("cloud-ibl-dense", 0.0, dense, denseValues) ||
            !readPublication("cloud-ibl-wind", 120.0, wind, windValues) ||
            !readPublication("cloud-ibl-repeat", 120.0, repeated, repeatedValues)) { return RHITestResult::fail("Cloud IBL publication: " + log); }
        auto changedResources = [](const EnvironmentLightingSnapshot& a, const EnvironmentLightingSnapshot& b) {
            return a.resourceRevision != b.resourceRevision && a.radianceView != b.radianceView && a.pdfView != b.pdfView &&
                a.sphericalHarmonicsBuffer != b.sphericalHarmonicsBuffer && a.prefilteredSpecularBuffer != b.prefilteredSpecularBuffer &&
                a.cloudShadowView != b.cloudShadowView && a.transmittanceView == b.transmittanceView && a.multiScatteringView == b.multiScatteringView;
        };
        if (!changedResources(clear, dense) || !changedResources(dense, wind) ||
            repeated.retainedResources != wind.retainedResources || repeated.resourceRevision != wind.resourceRevision ||
            repeatedValues != windValues || !(world.sun == authoredSun)) {
            return RHITestResult::fail("Cloud IBL invalidation, static LUT sharing, publication reuse, or authored Sun immutability failed");
        }
        for (uint32_t domain : {0u, 1u, 2u, 3u, 12u}) {
            if (cloudClose(clearValues[domain], denseValues[domain], 1e-5f, 1e-6f) ||
                cloudClose(denseValues[domain], windValues[domain], 1e-5f, 1e-6f)) {
                return RHITestResult::fail("Cloud coverage/wind left capture, sharp/rough IBL, irradiance SH or PDF stale at domain " + std::to_string(domain));
            }
        }
        for (const auto* values : {&clearValues, &denseValues, &windValues}) {
            if ((*values)[0][0] <= 0.0f || (*values)[1][0] <= 0.0f || (*values)[2][0] <= 0.0f ||
                (*values)[12][0] <= 0.0f || (*values)[13][0] <= 0.0f ||
                !cloudClose((*values)[1], (*values)[0], 0.15f, 1.0f)) {
                return RHITestResult::fail("Cloud capture/specular/PDF is empty or sharp convolution disagrees with captured sky");
            }
            const float shMeanFactor = 11.1366559937f; // 4*pi * Y00 * cosine convolution(pi).
            const std::array<float, 4> expectedSH{(*values)[0][0] * shMeanFactor, (*values)[0][1] * shMeanFactor,
                (*values)[0][2] * shMeanFactor, 1.0f};
            if (!cloudClose((*values)[3], expectedSH, 0.15f, 1.0f)) {
                return RHITestResult::fail("Cosine-convolved SH coefficient 0 disagrees with independent solid-angle capture quadrature");
            }
        }
        const bench::Json evidence{{"clearRevision", clear.resourceRevision}, {"denseRevision", dense.resourceRevision},
            {"windRevision", wind.resourceRevision}, {"staticMediumShared", true}, {"authoredSunUnchanged", true},
            {"domainsChanged", {"disk-free capture", "irradiance SH", "sharp/rough specular", "importance PDF"}}};
        if (context.evidence) { context.evidence->json("cloud-ibl-publication.json", evidence); }
        else { bench::writeJson(context.outputDirectory / "cloud-ibl-publication.json", evidence); }
        return RHITestResult::pass("Submitted provider publications rebuild disk-free capture, SH, sharp/rough GGX and PDF for coverage/wind; same medium shared, identical content reused, intrinsic Sun preserved");
    }
};
METALLIC_REGISTER_RHI_TEST(CloudIBLPublicationTest);

} // namespace
} // namespace metallic::tests
