#include "Runtime/Render/Core/RenderFrameContext.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>

namespace metallic::tests {
namespace {

using ProbeReadback = std::array<std::array<float, 4>, 32>;
struct alignas(16) AtmosphereProbeParams {
    render::AtmospherePrecomputeParams resources;
    render::GPUBufferSpan inputAerial;
    uint32_t reserved = 0;
};
constexpr uint64_t kAtmosphereProbeABI = 0x41544d50524f0001ull;
static_assert(sizeof(AtmosphereProbeParams) == 80 && offsetof(AtmosphereProbeParams, inputAerial) == 64);

render::Result<> atmosphereProbe(render::Device& device, render::ComputeKernel& kernel,
    const environment::WorldEnvironment& world, ProbeReadback& values, std::string& log,
    std::array<double, 3> observerWorldMetres = {0.0, 2.0, 0.0})
{
    using namespace render;
    AtmosphereResourcesGPU resources;
    auto result = resources.initialize(device, log);
    if (!result) { return result; }
    std::unique_ptr<Buffer> output;
    result = device.createBuffer(BufferDesc{.size = sizeof(values), .structureStride = 16,
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
        .transform([&](auto value) { output = std::move(value); });
    if (!result) { return result; }
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    if (!(result = gpu.initialize(device))) { return result; }
    if (!(result = resources.record(*gpu.commands, world.snapshot(), observerWorldMetres, log))) { return result; }
    if (resources.record(*gpu.commands, world.snapshot(), observerWorldMetres, log)) {
        log = "Immutable atmosphere publication accepted a second recording";
        return makeError(Error::Failure);
    }
    log.clear();
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    AtmospherePrecomputeParams params{};
    params.parameters = writer.bufferSpan(resources.parametersBuffer(), 16, 16);
    params.transmittance = writer.sampledImage(resources.transmittanceView());
    params.multiScattering = writer.sampledImage(resources.multiScatteringView());
    params.skyView = writer.sampledImage(resources.skyView());
    params.sourceMip = writer.sampledImage(resources.radianceView());
    params.aerial = writer.bufferSpan(output.get(), 16, 16);
    AtmosphereProbeParams probe{.resources = params, .inputAerial = writer.bufferSpan(resources.aerialBuffer(), 16, 16)};
    auto encoded = writer.encode(probe, kAtmosphereProbeABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (!(result = kernel.dispatch(*gpu.commands, *encoded, 1))) { return result; }
    BufferBarrierDesc readbackBarrier{.buffer = output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = sizeof(values)}};
    if (!(result = gpu.commands->synchronize(BarrierDesc{.buffers = {&readbackBarrier, 1}}))) { return result; }
    if (!(result = gpu.submitAndWait())) { return result; }
    output->invalidate();
    const void* mapped = output->map();
    if (mapped == nullptr) { return makeError(Error::Failure); }
    std::memcpy(values.data(), mapped, sizeof(values));
    output->unmap();
    for (const auto& value : values) {
        for (float channel : value) {
            if (!std::isfinite(channel)) { log = "Nonfinite atmosphere probe"; return makeError(Error::Failure); }
        }
        if (std::abs(value[3] - 1.0f) > 1e-6f) {
            log = "Atmosphere probe did not write the expected output ABI";
            return makeError(Error::Failure);
        }
    }
    return {};
}

render::Result<> createAtmosphereTestDevice(RHITestContext& context, std::unique_ptr<render::Device>& device,
    render::ComputeKernel& kernel, std::string& log, const char* entryPoint = "atmosphereProbeMain",
    const char* moduleName = "AtmosphereProbe")
{
    using namespace render;
    // Explicit production bindless configuration, independent of the runner's
    // default device. Unsupported production capability is a failed setup.
    auto result = createDevice({.applicationName = "Physical atmosphere numeric probes",
        .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
        .transform([&](auto value) { device = std::move(value); });
    if (!result) { return result; }
    return ShaderRegistry::instance().getComputeKernel(*device,
        SlangShaderDesc{.moduleName = moduleName, .entryPointName = entryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
        ComputeKernelDesc{.parameters = parameterAbi<AtmosphereProbeParams>(kAtmosphereProbeABI, ParameterTransport::InlinePush),
            .debugName = "Atmosphere numerical probe"}, kernel, log);
}

environment::WorldEnvironment physicalNoon()
{
    environment::WorldEnvironment world;
    world.source = environment::EnvironmentSource::PhysicalAtmosphere;
    world.sun.enabled = true;
    world.sun.direction = {0.0f, -1.0f, 0.0f};
    return world;
}

double photometricY(const std::array<float, 4>& value)
{
    const auto& m = render::sceneWorkingColorSpace() == render::SceneWorkingColorSpace::ACEScg ?
        render::color::kAP1ToXYZ : render::color::kRec709ToXYZ;
    return m[3] * value[0] + m[4] * value[1] + m[5] * value[2];
}

bool saveProbe(RHITestContext& context, const std::string& label, const ProbeReadback& values)
{
    std::ofstream output(context.outputDirectory / (label + ".tsv"));
    output << "probe\tworking0\tworking1\tworking2\talpha\n" << std::setprecision(10);
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

bool boundedTransmittance(const std::array<float, 4>& value)
{
    return value[0] >= 0.0f && value[0] <= 1.00001f && value[1] >= 0.0f && value[1] <= 1.00001f &&
        value[2] >= 0.0f && value[2] <= 1.00001f;
}

class AtmosphereNumericalTest final : public RHITest {
public:
    AtmosphereNumericalTest() { name = "atmosphere_spectral_transport_limits"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.transport", "environment.atmosphere.spectral.units",
            "environment.atmosphere.vacuum.orbit"}, bench::Layer::Core, "binding", "binding", {"noon.tsv", "sunset.tsv", "night.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (auto result = createAtmosphereTestDevice(context, device, kernel, log); !result) {
            return RHITestResult::fail("Atmosphere device/kernel setup: " + std::string(toString(result)) + " " + log);
        }
        const auto noonWorld = physicalNoon();
        ProbeReadback noon, sunset, night;
        if (!atmosphereProbe(*device, kernel, noonWorld, noon, log) || !saveProbe(context, "noon", noon)) {
            return RHITestResult::fail("Noon atmosphere probe: " + log);
        }
        auto world = noonWorld;
        world.sun.direction = {-0.9998477f, -0.0174524f, 0.0f};
        if (!atmosphereProbe(*device, kernel, world, sunset, log) || !saveProbe(context, "sunset", sunset)) {
            return RHITestResult::fail("Sunset atmosphere probe: " + log);
        }
        world.sun.direction = {0.0f, 1.0f, 0.0f};
        if (!atmosphereProbe(*device, kernel, world, night, log) || !saveProbe(context, "night", night)) {
            return RHITestResult::fail("Night atmosphere probe: " + log);
        }
        if (!(photometricY(noon[0]) > 50000.0 && photometricY(noon[1]) > photometricY(noon[0]) &&
            photometricY(sunset[0]) < photometricY(noon[0]) * 0.75 && photometricY(night[0]) < 1e-6 &&
            photometricY(noon[2]) < 1e-6)) {
            return RHITestResult::fail("Solar per-point altitude, sunset or planet-shadow attenuation failed");
        }
        // Independent physical bound: solar TOA ~130 klux, not an RGB/Rec.709
        // interpretation of wavelength samples. Unit-white basis is finite.
        if (photometricY(noon[13]) < 125000.0 || photometricY(noon[13]) > 135000.0 ||
            photometricY(noon[0]) >= photometricY(noon[13]) ||
            noon[13][0] / noon[13][1] < 0.8f || noon[13][0] / noon[13][1] > 1.2f ||
            noon[13][2] / noon[13][1] < 0.8f || noon[13][2] / noon[13][1] > 1.2f) {
            return RHITestResult::fail("Physical solar spectrum units or chromaticity are invalid");
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (!boundedTransmittance(noon[5]) || !boundedTransmittance(noon[7]) ||
                noon[7][channel] > noon[5][channel] + 1e-6f || noon[4][channel] < 0.0f || noon[6][channel] < noon[4][channel] - 1e-4f ||
                std::abs(noon[8][channel]) > 1e-6f || std::abs(noon[9][channel] - 1.0f) > 1e-6f ||
                !boundedTransmittance(noon[10]) || std::abs(noon[11][channel] - 1.0f) > 1e-6f ||
                noon[12][channel] <= 0.0f || std::abs(noon[14][channel] - noon[15][channel]) > 0.01f) {
                return RHITestResult::fail("Segment, vacuum, orbital, multiple-scattering or LUT/reference invariant failed");
            }
            // Vertical exponential columns and triangular ozone area have an
            // analytic optical-depth integral independent of the ray marcher.
            const auto& a = noonWorld.atmosphere;
            const std::array<float, 3> rayleigh{a.rayleighScattering.x, a.rayleighScattering.y, a.rayleighScattering.z};
            const std::array<float, 3> mie{a.mieExtinction.x, a.mieExtinction.y, a.mieExtinction.z};
            const std::array<float, 3> ozone{a.ozoneAbsorption.x, a.ozoneAbsorption.y, a.ozoneAbsorption.z};
            const double depth = rayleigh[channel] * a.rayleighScaleHeightKm * (std::exp(-0.002 / a.rayleighScaleHeightKm) -
                std::exp(-100.0 / a.rayleighScaleHeightKm)) + mie[channel] * a.mieScaleHeightKm *
                (std::exp(-0.002 / a.mieScaleHeightKm) - std::exp(-100.0 / a.mieScaleHeightKm)) + ozone[channel] * a.ozoneWidthKm;
            if (std::abs(noon[14][channel] - std::exp(-depth)) > 0.005) {
                return RHITestResult::fail("Vertical optical depth disagrees with analytic density integral");
            }
        }
        return RHITestResult::pass("Solar altitude/sunset/night, CIE photometry, analytic optical depth, segments, vacuum, orbit and multiple scattering");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereNumericalTest);

class AtmosphereDensityResponseTest final : public RHITest {
public:
    AtmosphereDensityResponseTest() { name = "atmosphere_density_ground_response"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.rayleigh.mie.ozone", "environment.atmosphere.multiple.ground"},
            bench::Layer::Core, "binding", "binding", {"rayleigh.tsv", "mie-heavy.tsv", "ozone-off.tsv", "ground-black.tsv", "ground-white.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (!createAtmosphereTestDevice(context, device, kernel, log)) { return RHITestResult::fail(log); }
        const auto base = physicalNoon();
        ProbeReadback baseline, rayleigh, mie, noOzone, blackGround, whiteGround;
        auto run = [&](const environment::WorldEnvironment& world, const char* label, ProbeReadback& values) {
            return bool(atmosphereProbe(*device, kernel, world, values, log)) && saveProbe(context, label, values);
        };
        if (!run(base, "baseline", baseline)) { return RHITestResult::fail(log); }
        auto world = base;
        world.atmosphere.mieScattering = {0.0f, 0.0f, 0.0f};
        world.atmosphere.mieExtinction = {0.0f, 0.0f, 0.0f};
        world.atmosphere.ozoneAbsorption = {0.0f, 0.0f, 0.0f};
        if (!run(world, "rayleigh", rayleigh)) { return RHITestResult::fail(log); }
        world = base;
        world.atmosphere.mieScattering = {0.03996f, 0.03996f, 0.03996f};
        world.atmosphere.mieExtinction = {0.0444f, 0.0444f, 0.0444f};
        if (!run(world, "mie-heavy", mie)) { return RHITestResult::fail(log); }
        world = base;
        world.atmosphere.ozoneAbsorption = {0.0f, 0.0f, 0.0f};
        if (!run(world, "ozone-off", noOzone)) { return RHITestResult::fail(log); }
        world = base;
        world.atmosphere.groundAlbedo = {0.0f, 0.0f, 0.0f};
        if (!run(world, "ground-black", blackGround)) { return RHITestResult::fail(log); }
        world.atmosphere.groundAlbedo = {1.0f, 1.0f, 1.0f};
        if (!run(world, "ground-white", whiteGround)) { return RHITestResult::fail(log); }
        if (photometricY(rayleigh[0]) <= photometricY(baseline[0]) || rayleigh[3][2] <= rayleigh[3][0] ||
            photometricY(mie[0]) >= photometricY(baseline[0]) ||
            photometricY(noOzone[0]) <= photometricY(baseline[0]) * 1.005 ||
            whiteGround[12][0] <= blackGround[12][0] * 1.01f ||
            photometricY(whiteGround[3]) <= photometricY(blackGround[3]) * 1.01) {
            return RHITestResult::fail("Rayleigh blue sky, aerosol extinction, ozone absorption or ground multiple-scattering response failed");
        }
        return RHITestResult::pass("Distinct physical parameter responses: Rayleigh, heavy aerosol, ozone, black/white ground");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereDensityResponseTest);

class AtmosphereDiskLimbTest final : public RHITest {
public:
    AtmosphereDiskLimbTest() { name = "atmosphere_finite_disk_planet_limb"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.disk.planet.limb"}, bench::Layer::Core,
            "binding", "binding", {"planet-limb.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (!createAtmosphereTestDevice(context, device, kernel, log, "atmosphereLimbProbeMain")) {
            return RHITestResult::fail(log);
        }
        auto world = physicalNoon();
        // At observer altitude 2 m the planet limb is near -0.0008 rad.
        // The disk centre is hidden, while its upper rays remain visible.
        world.sun.direction = {-1.0f, 0.002f, 0.0f};
        world.sun.angularRadius = 0.00465f;
        ProbeReadback values;
        if (!atmosphereProbe(*device, kernel, world, values, log) || !saveProbe(context, "planet-limb", values)) {
            return RHITestResult::fail(log);
        }
        if (photometricY(values[0]) <= 1.0 || photometricY(values[1]) > 1e-6 || photometricY(values[2]) > 1e-6 ||
            values[3][0] <= 0.0f || values[4][0] != 0.0f || values[4][1] != 0.0f || values[4][2] != 0.0f) {
            return RHITestResult::fail("Finite source disk did not clip or transmit each ray at the planet limb");
        }
        return RHITestResult::pass("Hidden disk centre, visible upper disk and occluded lower disk use per-ray planet/transmittance tests");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereDiskLimbTest);

class AtmosphereAerialLUTTest final : public RHITest {
public:
    AtmosphereAerialLUTTest() { name = "atmosphere_aerial_lut_disk_capture"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.aerial.lookup", "environment.atmosphere.capture.disk.exclusion",
            "environment.atmosphere.disk.dynamicRange"}, bench::Layer::Core, "binding", "binding",
            {"aerial-noon.tsv", "aerial-vacuum.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (!createAtmosphereTestDevice(context, device, kernel, log)) { return RHITestResult::fail(log); }
        auto world = physicalNoon();
        ProbeReadback noon, vacuum;
        if (!atmosphereProbe(*device, kernel, world, noon, log) || !saveProbe(context, "aerial-noon", noon)) {
            return RHITestResult::fail(log);
        }
        for (const auto comparison : {std::array<uint32_t, 4>{23, 24, 25, 26}, {4, 5, 16, 17}, {6, 7, 18, 19}}) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float expectedRadiance = noon[comparison[0]][channel];
                const float actualRadiance = noon[comparison[2]][channel];
                if (actualRadiance < 0.0f || std::abs(actualRadiance - expectedRadiance) >
                        0.15f * std::max(expectedRadiance, 1e-3f) ||
                    !boundedTransmittance(noon[comparison[3]]) ||
                    std::abs(noon[comparison[3]][channel] - noon[comparison[1]][channel]) > 0.015f) {
                    return RHITestResult::fail("Actual 32-cubed aerial lookup disagrees with integration reference at 10 m, 1 km or 50 km");
                }
            }
        }
        const double diskRadiance = photometricY(noon[20]);
        const double diskFreeSky = photometricY(noon[21]);
        const double capturedSky = photometricY(noon[22]);
        if (diskRadiance < 1e6 || diskFreeSky >= 1e5 || capturedSky >= 1e5 ||
            std::abs(capturedSky - diskFreeSky) > 0.02 * std::max(diskFreeSky, 1.0)) {
            return RHITestResult::fail("Visible finite disk dynamic range or disk-free lighting capture policy failed");
        }
        world.atmosphere.rayleighScattering = {0.0f, 0.0f, 0.0f};
        world.atmosphere.mieScattering = {0.0f, 0.0f, 0.0f};
        world.atmosphere.mieExtinction = {0.0f, 0.0f, 0.0f};
        world.atmosphere.ozoneAbsorption = {0.0f, 0.0f, 0.0f};
        if (!atmosphereProbe(*device, kernel, world, vacuum, log) || !saveProbe(context, "aerial-vacuum", vacuum)) {
            return RHITestResult::fail(log);
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            for (uint32_t index : {16u, 18u, 25u, 27u}) {
                if (std::abs(vacuum[index][channel]) > 1e-6f) { return RHITestResult::fail("Vacuum aerial lookup has nonzero scattering"); }
            }
            for (uint32_t index : {17u, 19u, 26u, 28u}) {
                if (std::abs(vacuum[index][channel] - 1.0f) > 1e-6f) { return RHITestResult::fail("Vacuum aerial lookup has nonunit transmission"); }
            }
            if (std::abs(noon[27][channel]) > 1e-6f || std::abs(noon[28][channel] - 1.0f) > 1e-6f) {
                return RHITestResult::fail("Zero-distance aerial lookup failed its identity limit");
            }
        }
        return RHITestResult::pass("Actual aerial LUT at 10 m/1 km/50 km matches reference, vacuum and zero limits; >1e6 visible disk stays outside the lighting capture");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereAerialLUTTest);

class AtmospherePrimaryRayOriginTest final : public RHITest {
public:
    AtmospherePrimaryRayOriginTest() { name = "atmosphere_actual_primary_ray_origin"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.orthographic.ray.origin", "environment.atmosphere.aerial.named.consumer"},
            bench::Layer::Core, "binding", "binding", {"parallel-ray-origins.tsv", "parallel-ray-vacuum.tsv", "offset-origin-limb.tsv"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (!createAtmosphereTestDevice(context, device, kernel, log, "atmosphereRayOriginProbeMain", "AtmosphereRayOriginProbe")) {
            return RHITestResult::fail(log);
        }
        auto world = physicalNoon();
        ProbeReadback values, vacuum;
        if (!atmosphereProbe(*device, kernel, world, values, log) || !saveProbe(context, "parallel-ray-origins", values)) {
            return RHITestResult::fail(log);
        }
        for (uint32_t ray = 0; ray < 2; ++ray) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float expectedRadiance = values[ray * 4 + 2][channel];
                const float actualRadiance = values[ray * 4][channel];
                const float expectedTransmission = values[ray * 4 + 3][channel];
                const float actualTransmission = values[ray * 4 + 1][channel];
                if (std::abs(actualRadiance - expectedRadiance) > 1e-5f * std::max(expectedRadiance, 1.0f) ||
                    std::abs(actualTransmission - expectedTransmission) > 1e-6f ||
                    !boundedTransmittance(values[ray * 4 + 1]) ||
                    actualTransmission <= values[9 + ray * 2][channel] + 0.01f) {
                    return RHITestResult::fail("Parallel ray at " + std::to_string((ray + 1) * 20) +
                        " km failed its actual-origin reference or retained camera-eye attenuation; see parallel-ray-origins.tsv");
                }
            }
            if (photometricY(values[ray * 4]) <= 0.0 ||
                photometricY(values[ray * 4]) >= photometricY(values[8 + ray * 2]) * 0.75) {
                return RHITestResult::fail("Offset parallel ray retained camera-eye scattering; see parallel-ray-origins.tsv");
            }
        }
        // The two rays have equal direction and distance, so the atmosphere's
        // altitude response must come from their distinct origins.
        if (photometricY(values[0]) <= photometricY(values[4]) * 1.5 || values[1][2] >= values[5][2] - 0.02f) {
            return RHITestResult::fail("Equal parallel segments did not respond to their actual 20/40 km starting altitudes");
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (std::abs(values[12][channel]) > 1e-6f || std::abs(values[13][channel] - 1.0f) > 1e-6f ||
                std::abs(values[14][channel] - values[16][channel]) > 1e-5f * std::max(values[16][channel], 1.0f) ||
                std::abs(values[15][channel] - values[17][channel]) > 1e-6f ||
                std::abs(values[20][channel] - values[21][channel]) > 1e-5f * std::max(values[21][channel], 1.0f) ||
                std::abs(values[23][channel] - values[24][channel]) > 1e-5f * std::max(values[24][channel], 1.0f)) {
                return RHITestResult::fail("Actual-origin zero segment, perspective LUT, offset sky or finite source disk reference failed");
            }
        }
        if (photometricY(values[20]) <= 0.0 ||
            std::abs(photometricY(values[20]) - photometricY(values[22])) < 0.25 * photometricY(values[20]) ||
            photometricY(values[23]) < 1e6 || photometricY(values[23]) <= photometricY(values[25]) * 1.02) {
            return RHITestResult::fail("Offset-origin primary sky or finite disk retained camera-eye transport; see parallel-ray-origins.tsv");
        }
        world.atmosphere.rayleighScattering = {0.0f, 0.0f, 0.0f};
        world.atmosphere.mieScattering = {0.0f, 0.0f, 0.0f};
        world.atmosphere.mieExtinction = {0.0f, 0.0f, 0.0f};
        world.atmosphere.ozoneAbsorption = {0.0f, 0.0f, 0.0f};
        if (!atmosphereProbe(*device, kernel, world, vacuum, log) || !saveProbe(context, "parallel-ray-vacuum", vacuum)) {
            return RHITestResult::fail(log);
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            for (uint32_t index : {0u, 4u, 12u, 14u}) {
                if (std::abs(vacuum[index][channel]) > 1e-6f) { return RHITestResult::fail("Offset-origin vacuum segment scattered light"); }
            }
            for (uint32_t index : {1u, 5u, 13u, 15u}) {
                if (std::abs(vacuum[index][channel] - 1.0f) > 1e-6f) { return RHITestResult::fail("Offset-origin vacuum segment attenuated light"); }
            }
        }
        render::ComputeKernel limbKernel;
        if (!render::ShaderRegistry::instance().getComputeKernel(*device,
                render::SlangShaderDesc{.moduleName = "AtmosphereRayOriginProbe", .entryPointName = "atmosphereOffsetLimbProbeMain",
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                render::ComputeKernelDesc{.parameters = render::parameterAbi<AtmosphereProbeParams>(kAtmosphereProbeABI,
                        render::ParameterTransport::InlinePush), .debugName = "Offset primary origin finite disk limb"}, limbKernel, log)) {
            return RHITestResult::fail(log);
        }
        world = physicalNoon();
        // Place the disk centre 1 mrad below the 20 km ray origin's limb.
        // Its upper ray is visible there but remains hidden at camera eye.
        const double sourceElevation = -std::acos(world.atmosphere.bottomRadiusKm /
            (world.atmosphere.bottomRadiusKm + 20.002)) - 0.001;
        world.sun.direction = {static_cast<float>(-std::cos(sourceElevation)),
            static_cast<float>(-std::sin(sourceElevation)), 0.0f};
        world.sun.angularRadius = 0.00465f;
        ProbeReadback limb;
        if (!atmosphereProbe(*device, limbKernel, world, limb, log) || !saveProbe(context, "offset-origin-limb", limb)) {
            return RHITestResult::fail(log);
        }
        if (photometricY(limb[0]) <= 1e6 || photometricY(limb[2]) <= 1e6) {
            return RHITestResult::fail("Offset-origin finite disk upper limb was hidden with the camera-eye planet mask");
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (std::abs(limb[0][channel] - limb[2][channel]) > 1e-5f * std::max(std::abs(limb[2][channel]), 1.0f) ||
                std::abs(limb[1][channel]) > 1e-3f || std::abs(limb[3][channel]) > 1e-6f ||
                limb[4][channel] != 0.0f || !boundedTransmittance(limb[5])) {
                return RHITestResult::fail("Actual-origin upper/lower disk rays or eye-origin planet mask disagree; see offset-origin-limb.tsv");
            }
        }
        return RHITestResult::pass("Production named-resource consumer: parallel 20/40 km origins match exact aerial/sky transport and differ from eye; finite source disk clips the actual-origin limb; zero, vacuum and perspective LUT paths verified");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmospherePrimaryRayOriginTest);

render::Result<> saveOrbitalSkyImages(RHITestContext& context, render::Device& device,
    const environment::WorldEnvironment& world, std::string& log)
{
    using namespace render;
    constexpr uint32_t width = 640, height = 360;
    AtmosphereResourcesGPU resources;
    auto result = resources.initialize(device, log);
    if (!result) { return result; }
    ComputeKernel imageKernel;
    result = ShaderRegistry::instance().getComputeKernel(device,
        SlangShaderDesc{.moduleName = "AtmosphereProbe", .entryPointName = "atmosphereHorizonImageMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
        ComputeKernelDesc{.parameters = parameterAbi<AtmosphereProbeParams>(kAtmosphereProbeABI, ParameterTransport::InlinePush),
            .debugName = "Orbital sky LUT and reference image"}, imageKernel, log);
    if (!result) { return result; }
    std::array<std::unique_ptr<Buffer>, 2> outputs;
    for (auto& output : outputs) {
        result = device.createBuffer({.size = uint64_t(width) * height * 16, .structureStride = 16,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { output = std::move(value); });
        if (!result) { return result; }
    }
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    if (!(result = gpu.initialize(device))) { return result; }
    if (!(result = resources.record(*gpu.commands, world.snapshot(), {0.0, 150000.0, 0.0}, log))) { return result; }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    for (uint32_t mode = 0; mode < 2; ++mode) {
        AtmosphereProbeParams params{};
        params.resources.parameters = writer.bufferSpan(resources.parametersBuffer(), 16, 16);
        params.resources.transmittance = writer.sampledImage(resources.transmittanceView());
        params.resources.multiScattering = writer.sampledImage(resources.multiScatteringView());
        params.resources.skyView = writer.sampledImage(resources.skyView());
        params.resources.aerial = writer.bufferSpan(outputs[mode].get(), 16, 16);
        params.resources.dimensions = {width, height, 1, mode};
        params.inputAerial = writer.bufferSpan(resources.aerialBuffer(), 16, 16);
        auto encoded = writer.encode(params, kAtmosphereProbeABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        if (!(result = imageKernel.dispatch(*gpu.commands, *encoded, width / 8, height / 8))) { return result; }
        BufferBarrierDesc barrier{.buffer = outputs[mode].get(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead}};
        if (!(result = gpu.commands->synchronize({.buffers = {&barrier, 1}}))) { return result; }
    }
    if (!(result = gpu.submitAndWait())) { return result; }
    constexpr std::array<const char*, 2> labels{"OrbitalSkyLookup", "OrbitalSkyReference"};
    for (uint32_t mode = 0; mode < 2; ++mode) {
        outputs[mode]->invalidate();
        const auto* mapped = static_cast<const float*>(outputs[mode]->map());
        if (!mapped) { return makeError(Error::Failure); }
        std::vector<uint8_t> pixels(size_t(width) * height * 4);
        for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
            const color::RGB working{mapped[pixel * 4], mapped[pixel * 4 + 1], mapped[pixel * 4 + 2]};
            const auto display = color::toLinearRec709(working);
            for (uint32_t channel = 0; channel < 3; ++channel) {
                if (!std::isfinite(working[channel])) {
                    outputs[mode]->unmap(); log = "Orbital image has nonfinite radiance"; return makeError(Error::Failure);
                }
                const float tone = 1.0f - std::exp(-std::max(display[channel], 0.0f) * 0.0001f);
                pixels[pixel * 4 + channel] = static_cast<uint8_t>(std::clamp(std::pow(tone, 1.0f / 2.2f) * 255.0f, 0.0f, 255.0f));
            }
            pixels[pixel * 4 + 3] = 255;
        }
        const std::string label = labels[mode];
        std::ofstream raw(context.outputDirectory / (label + ".rgba32f"), std::ios::binary);
        raw.write(reinterpret_cast<const char*>(mapped), std::streamsize(uint64_t(width) * height * 16));
        outputs[mode]->unmap();
        if (!raw || !saveRgba8Png(context.outputDirectory / (label + ".png"), pixels.data(), width, height, log)) {
            return makeError(Error::Failure);
        }
    }
    return {};
}

class AtmosphereOrbitalSkyTest final : public RHITest {
public:
    AtmosphereOrbitalSkyTest() { name = "atmosphere_orbital_limb_lookup_reference"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.orbital.limb.lookup", "environment.atmosphere.angular.parameterization"},
            bench::Layer::Core, "binding", "binding", {"orbital-limb.tsv", "OrbitalSkyLookup.png", "OrbitalSkyReference.png",
                "OrbitalSkyLookup.rgba32f", "OrbitalSkyReference.rgba32f"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::ComputeKernel kernel;
        std::string log;
        if (!createAtmosphereTestDevice(context, device, kernel, log, "atmosphereHorizonProbeMain")) {
            return RHITestResult::fail(log);
        }
        auto world = physicalNoon();
        world.sun.direction = {-1.0f, 0.0f, 0.0f};
        ProbeReadback values;
        if (!atmosphereProbe(*device, kernel, world, values, log, {0.0, 150000.0, 0.0}) ||
            !saveProbe(context, "orbital-limb", values)) {
            return RHITestResult::fail(log);
        }
        for (uint32_t index = 0; index < 5; ++index) {
            const double actual = photometricY(values[index]), reference = photometricY(values[index + 5]);
            if (reference <= 1.0 || std::abs(actual - reference) > 0.2 * reference ||
                values[index + 10][0] > 2e-5f || values[index + 10][1] > 2e-6f) {
                return RHITestResult::fail("Orbital limb lookup or radial angular mapping disagrees with ray-integration reference");
            }
        }
        if (!saveOrbitalSkyImages(context, *device, world, log)) { return RHITestResult::fail(log); }
        return RHITestResult::pass("150 km orbital observer: five directions across the planet limb match reference within 20%; paired 640x360 GPU images, exposure 1e-4");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereOrbitalSkyTest);

} // namespace
} // namespace metallic::tests
