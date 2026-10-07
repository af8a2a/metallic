#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <vector>

#include "Runtime/Environment/Astronomy.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "RHITest.h"
#include "harness/Fixtures.h"

namespace metallic::tests {
namespace {

using DiskProbeReadback = std::array<std::array<float, 4>, 72>;

render::Result<> runDiskProbe(render::Device& device, render::ComputeKernel& kernel,
    DiskProbeReadback& values, std::string& log)
{
    using namespace render;
    environment::WorldEnvironment world;
    world.source = environment::EnvironmentSource::PhysicalAtmosphere;
    world.sun.enabled = true;
    world.moon.enabled = true;
    world.sun.angularRadius = 0.03f;
    world.moon.angularRadius = 0.03f;
    world.atmosphere.rayleighScattering = float3(0.0f);
    world.atmosphere.mieScattering = float3(0.0f);
    world.atmosphere.mieExtinction = float3(0.0f);
    world.atmosphere.ozoneAbsorption = float3(0.0f);
    AtmosphereResourcesGPU atmosphere;
    auto result = atmosphere.initialize(device, log);
    if (!result) { return result; }
    std::unique_ptr<Buffer> output;
    result = device.createBuffer(BufferDesc{.size = sizeof(values), .structureStride = 16,
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
        .transform([&](auto value) { output = std::move(value); });
    if (!result) { return result; }
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    if (!(result = gpu.initialize(device))) { return result; }
    if (!(result = atmosphere.record(*gpu.commands, world.snapshot(), {0.0, 2.0, 0.0}, log))) { return result; }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    AtmospherePrecomputeParams params{};
    params.parameters = writer.bufferSpan(atmosphere.parametersBuffer(), 16, 16);
    params.transmittance = writer.sampledImage(atmosphere.transmittanceView());
    params.aerial = writer.bufferSpan(output.get(), 16, 16);
    auto encoded = writer.encode(params, kAtmospherePrecomputeABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (!(result = kernel.dispatch(*gpu.commands, *encoded, 1))) { return result; }
    BufferBarrierDesc readback{.buffer = output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}, .range = {.offset = 0, .size = sizeof(values)}};
    if (!(result = gpu.commands->synchronize(BarrierDesc{.buffers = {&readback, 1}}))) { return result; }
    if (!(result = gpu.submitAndWait())) { return result; }
    output->invalidate();
    const void* mapped = output->map();
    if (mapped == nullptr) { return makeError(Error::Failure); }
    std::memcpy(values.data(), mapped, sizeof(values));
    output->unmap();
    return {};
}

bool saveDiskProbe(RHITestContext& context, const DiskProbeReadback& values)
{
    std::ofstream output(context.outputDirectory / "finite-disk.tsv");
    output << "case\trecord\tx\ty\tz\tw\n" << std::setprecision(10);
    for (size_t index = 0; index < values.size(); ++index) {
        output << index / 8 << '\t' << index % 8;
        for (float value : values[index]) { output << '\t' << value; }
        output << '\n';
    }
    if (context.evidence) {
        bench::Json observations = bench::Json::array();
        for (const auto& value : values) { observations.push_back(value); }
        context.evidence->json("finite-disk.json", observations);
    }
    return bool(output);
}

render::Result<> saveMoonPhaseAtlas(RHITestContext& context, render::Device& device, std::string& log)
{
    using namespace render;
    constexpr uint32_t width = 256, height = 64;
    constexpr float displayExposure = 0.0002f;
    environment::WorldEnvironment world;
    world.source = environment::EnvironmentSource::PhysicalAtmosphere;
    world.moon.enabled = true;
    const environment::AstronomyEvaluation fullMoon;
    world.moon.angularRadius = fullMoon.moonAngularRadius;
    world.moon.topOfAtmosphereIrradiance = environment::reflectedMoonIrradiance(
        world.sun.topOfAtmosphereIrradiance, fullMoon, world.astronomy.moonAlbedo);
    world.atmosphere.rayleighScattering = float3(0.0f);
    world.atmosphere.mieScattering = float3(0.0f);
    world.atmosphere.mieExtinction = float3(0.0f);
    world.atmosphere.ozoneAbsorption = float3(0.0f);
    AtmosphereResourcesGPU atmosphere;
    auto result = atmosphere.initialize(device, log);
    if (!result) { return result; }
    ComputeKernel kernel;
    result = ShaderRegistry::instance().getComputeKernel(device,
        SlangShaderDesc{.moduleName = "CelestialDiskProbe", .entryPointName = "moonPhaseAtlasMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
        ComputeKernelDesc{.parameters = parameterAbi<AtmospherePrecomputeParams>(kAtmospherePrecomputeABI,
            ParameterTransport::InlinePush), .debugName = "Moon phase physical radiance atlas"}, kernel, log);
    if (!result) { return result; }
    std::unique_ptr<Buffer> output;
    result = device.createBuffer({.size = uint64_t(width) * height * 16, .structureStride = 16,
        .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
        .transform([&](auto value) { output = std::move(value); });
    if (!result) { return result; }
    bench::GPUCommands gpu(*device.getQueue(QueueType::Graphics));
    if (!(result = gpu.initialize(device))) { return result; }
    if (!(result = atmosphere.record(*gpu.commands, world.snapshot(), {0.0, 2.0, 0.0}, log))) { return result; }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    AtmospherePrecomputeParams params{};
    params.parameters = writer.bufferSpan(atmosphere.parametersBuffer(), 16, 16);
    params.transmittance = writer.sampledImage(atmosphere.transmittanceView());
    params.aerial = writer.bufferSpan(output.get(), 16, 16);
    auto encoded = writer.encode(params, kAtmospherePrecomputeABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (!(result = kernel.dispatch(*gpu.commands, *encoded, width / 8, height / 8))) { return result; }
    BufferBarrierDesc barrier{.buffer = output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    if (!(result = gpu.commands->synchronize({.buffers = {&barrier, 1}}))) { return result; }
    if (!(result = gpu.submitAndWait())) { return result; }
    output->invalidate();
    const auto* mapped = static_cast<const float*>(output->map());
    if (mapped == nullptr) { return makeError(Error::Failure); }
    std::vector<uint8_t> pixels(size_t(width) * height * 4);
    for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
        const color::RGB working{mapped[pixel * 4], mapped[pixel * 4 + 1], mapped[pixel * 4 + 2]};
        const auto display = color::toLinearRec709(working);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (!std::isfinite(working[channel]) || working[channel] < 0.0f) {
                output->unmap(); log = "Moon atlas contains invalid physical radiance"; return makeError(Error::Failure);
            }
            const float tone = 1.0f - std::exp(-std::max(display[channel], 0.0f) * displayExposure);
            pixels[pixel * 4 + channel] = uint8_t(std::clamp(std::pow(tone, 1.0f / 2.2f) * 255.0f, 0.0f, 255.0f));
        }
        pixels[pixel * 4 + 3] = 255;
    }
    std::ofstream raw(context.outputDirectory / "MoonPhaseAtlas.rgba32f", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(mapped), std::streamsize(uint64_t(width) * height * 16));
    output->unmap();
    if (!raw || !saveRgba8Png(context.outputDirectory / "MoonPhaseAtlas.png", pixels.data(), width, height, log)) {
        return makeError(Error::Failure);
    }
    if (context.evidence) {
        context.evidence->json("MoonPhaseAtlas.json", bench::Json{{"width", width}, {"height", height},
            {"panels", {"full", "quarter", "crescent", "new"}}, {"phaseRadians", {0.0, 0.5 * std::numbers::pi,
                0.8 * std::numbers::pi, std::numbers::pi}}, {"displayExposure", displayExposure},
            {"angularRadius", world.moon.angularRadius}, {"model", "solar spectrum reflected by a Lambert sphere, vacuum"}});
    }
    return {};
}

class FiniteCelestialDiskSamplingTest final : public RHITest {
public:
    FiniteCelestialDiskSamplingTest()
    {
        name = "celestial_finite_disk_energy_phase_bounds";
        type = RHITestType::Rendering;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.celestial.disk.sampling", "environment.moon.phase.radiometry",
            "environment.celestial.disk.planet_limb", "environment.celestial.disk.delta_miss"},
            bench::Layer::Core, "binding", "binding", {"finite-disk.tsv", "MoonPhaseAtlas.png", "MoonPhaseAtlas.rgba32f"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Finite celestial disk Monte Carlo probe",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Finite disk device: " + std::string(toString(result))); }
        ComputeKernel kernel;
        std::string log;
        result = ShaderRegistry::instance().getComputeKernel(*device,
            SlangShaderDesc{.moduleName = "CelestialDiskProbe", .entryPointName = "celestialDiskProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            ComputeKernelDesc{.parameters = parameterAbi<AtmospherePrecomputeParams>(kAtmospherePrecomputeABI,
                ParameterTransport::InlinePush), .debugName = "Finite celestial disk MC"}, kernel, log);
        if (!result) { return RHITestResult::fail("Finite disk kernel: " + log); }
        DiskProbeReadback values;
        if (!runDiskProbe(*device, kernel, values, log) || !saveDiskProbe(context, values)) {
            return RHITestResult::fail("Finite disk execution/evidence: " + log);
        }
        for (const auto& value : values) {
            for (const float channel : value) {
                if (!std::isfinite(channel)) { return RHITestResult::fail("Finite disk returned nonfinite values"); }
            }
        }
        for (uint32_t testCase = 0; testCase < 8; ++testCase) {
            const uint32_t base = testCase * 8;
            if (values[base + 7][0] != float(testCase) || values[base + 7][1] != 8192.0f) {
                return RHITestResult::fail("Finite disk probe ABI or dispatch did not write every case");
            }
            const float radius = values[base + 5][0];
            const double expectedSolidAngle = 4.0 * std::numbers::pi_v<double> *
                std::pow(std::sin(double(radius) * 0.5), 2.0);
            if ((radius > 0.0f && !(values[base + 5][1] > 0.0f)) ||
                std::abs(double(values[base + 5][1]) - expectedSolidAngle) > expectedSolidAngle * 0.0001) {
                return RHITestResult::fail("Finite disk solid angle lost its positive small-radius support");
            }
            if (values[base + 3][0] < std::cos(radius) - 2e-7f || values[base + 3][2] > 1.000001f ||
                std::abs(values[base + 3][1] - (1.0f + std::cos(radius)) * 0.5f) > 2e-5f) {
                return RHITestResult::fail("Finite disk direction escaped its cone or was not uniform in solid angle");
            }
            if (testCase <= 5) {
                const float tolerance = testCase == 3 ? 0.015f : 0.005f;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const float expected = values[base + 1][channel];
                    if (!(expected > 0.0f) || std::abs(values[base][channel] - expected) > tolerance * expected) {
                        return RHITestResult::fail("Cosine-projected disk MC does not preserve authored irradiance, case " +
                            std::to_string(testCase));
                    }
                }
            }
        }
        const uint32_t tinySun = 4 * 8;
        if (std::abs(values[tinySun + 6][0]) < 0.2f || values[tinySun + 6][1] < 0.8f ||
            std::abs(values[tinySun + 6][2]) < 0.15f || values[tinySun + 3][3] != 1.0f) {
            return RHITestResult::fail("Tiny non-axis Sun must retain every known-inside MC sample");
        }
        if (values[2 * 8 + 3][3] < 0.48f || values[2 * 8 + 3][3] > 0.52f ||
            values[2 * 8 + 4][0] < 0.2f || values[3 * 8 + 3][3] < 0.08f || values[3 * 8 + 3][3] > 0.12f ||
            values[3 * 8 + 4][0] < 0.5f) {
            return RHITestResult::fail("Automatic quarter/crescent Moon lost its solar-facing illuminated distribution");
        }
        const uint32_t limb = 6 * 8, newMoon = 7 * 8;
        if (!(values[limb][1] > 0.1f * values[limb + 1][1] && values[limb][1] < 0.9f * values[limb + 1][1]) ||
            values[limb + 3][3] <= 0.1f || values[limb + 3][3] >= 0.9f) {
            return RHITestResult::fail("Finite disk did not resolve partial planet-limb occlusion per sample");
        }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (values[newMoon][channel] != 0.0f || values[newMoon + 1][channel] != 0.0f) {
                return RHITestResult::fail("New Moon must have zero reflected flux");
            }
            if (std::abs(values[64][channel] - values[65][channel]) > 0.0001f * values[65][channel] ||
                values[66][channel] != 0.0f || values[67][channel] != 0.0f || values[71][0] != 8.0f) {
                return RHITestResult::fail("Ideal delta miss disk radiance or disabled-physical fallback is invalid");
            }
        }
        if (!saveMoonPhaseAtlas(context, *device, log)) { return RHITestResult::fail("Moon phase atlas: " + log); }
        return RHITestResult::pass("8192 samples per case: uniform finite cone, solar and full/quarter/crescent lunar energy, "
            "non-axis 1e-8-radian radius, delta fallback, per-ray planet limb, dark new Moon and secondary delta miss");
    }
};
METALLIC_REGISTER_RHI_TEST(FiniteCelestialDiskSamplingTest);

} // namespace
} // namespace metallic::tests
