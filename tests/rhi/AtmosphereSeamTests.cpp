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
#include <vector>

namespace metallic::tests {
namespace {

constexpr uint32_t kSeamBandCount = 8, kSeamPairCount = kSeamBandCount * 2;
constexpr uint32_t kSeamSweepWidth = 256, kSeamRecordCount = 15;
constexpr uint32_t kSeamSampleCount = kSeamPairCount + kSeamBandCount * kSeamSweepWidth;
constexpr uint32_t kSeamImageWidth = 512, kSeamImageHeight = 256;
constexpr uint64_t kAtmosphereSeamABI = 0x41544d5345410001ull;
using SeamValues = std::vector<std::array<float, 4>>;
struct alignas(16) AtmosphereSeamParams {
    render::AtmospherePrecomputeParams resources;
    render::GPUBufferSpan inputAerial;
    uint32_t reserved = 0;
};
static_assert(sizeof(AtmosphereSeamParams) == 80 && offsetof(AtmosphereSeamParams, inputAerial) == 64);

render::Result<> seamReadback(render::Device& device, render::ComputeKernel& kernel,
    render::AtmosphereResourcesGPU& resources, render::CommandBuffer& commands,
    uint32_t width, uint32_t height, uint32_t mode, uint32_t recordCount,
    uint32_t groupsX, uint32_t groupsY, std::unique_ptr<render::Buffer>& output)
{
    using namespace render;
    auto result = device.createBuffer({.size = uint64_t(width) * height * recordCount * 16,
        .structureStride = 16, .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
        .transform([&](auto value) { output = std::move(value); });
    if (!result) { return result; }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, nullptr);
    AtmosphereSeamParams params{};
    params.resources.parameters = writer.bufferSpan(resources.parametersBuffer(), 16, 16);
    params.resources.transmittance = writer.sampledImage(resources.transmittanceView());
    params.resources.multiScattering = writer.sampledImage(resources.multiScatteringView());
    params.resources.skyView = writer.sampledImage(resources.skyView());
    params.resources.sourceMip = writer.sampledImage(resources.radianceView());
    params.resources.aerial = writer.bufferSpan(output.get(), 16, 16);
    params.resources.dimensions = {width, height, 1, mode};
    params.inputAerial = writer.bufferSpan(resources.aerialBuffer(), 16, 16);
    auto encoded = writer.encode(params, kAtmosphereSeamABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (!(result = kernel.dispatch(commands, *encoded, groupsX, groupsY))) { return result; }
    BufferBarrierDesc barrier{.buffer = output.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    return commands.synchronize({.buffers = {&barrier, 1}});
}

render::Result<> copySeamValues(render::Buffer& output, uint32_t count, SeamValues& values)
{
    output.invalidate();
    const void* mapped = output.map();
    if (!mapped) { return render::makeError(render::Error::Failure); }
    values.resize(count);
    std::memcpy(values.data(), mapped, size_t(count) * 16);
    output.unmap();
    return {};
}

bool saveSeamValues(RHITestContext& context, const std::string& label, const SeamValues& values)
{
    std::ofstream output(context.outputDirectory / (label + ".tsv"));
    output << "sample\trecord\tx\ty\tz\tw\n" << std::setprecision(10);
    for (size_t index = 0; index < values.size(); ++index) {
        output << index / kSeamRecordCount << '\t' << index % kSeamRecordCount;
        for (float value : values[index]) { output << '\t' << value; }
        output << '\n';
    }
    return bool(output);
}

bool saveSeamJson(RHITestContext& context, const char* filename, const bench::Json& value)
{
    std::ofstream output(context.outputDirectory / filename);
    output << value.dump(2) << '\n';
    output.close();
    if (!output) { return false; }
    if (context.evidence) { context.evidence->json(filename, value); }
    return true;
}

bool saveSeamImage(RHITestContext& context, const std::string& label, const SeamValues& values,
    float exposure, std::string& log)
{
    std::ofstream raw(context.outputDirectory / (label + ".rgba32f"), std::ios::binary);
    raw.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * 16));
    std::vector<uint8_t> pixels(values.size() * 4);
    for (size_t index = 0; index < values.size(); ++index) {
        const auto& pixel = values[index];
        const auto display = render::color::toLinearRec709({pixel[0], pixel[1], pixel[2]});
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (!std::isfinite(pixel[channel])) { log = "Nonfinite seam image"; return false; }
            const float tone = 1.0f - std::exp(-std::max(display[channel], 0.0f) * exposure);
            pixels[index * 4 + channel] = static_cast<uint8_t>(std::clamp(
                std::pow(tone, 1.0f / 2.2f) * 255.0f, 0.0f, 255.0f));
        }
        pixels[index * 4 + 3] = 255;
    }
    return bool(raw) && saveRgba8Png(context.outputDirectory / (label + ".png"), pixels.data(),
        kSeamImageWidth, kSeamImageHeight, log);
}

double relativeDifference(const std::array<float, 4>& first, const std::array<float, 4>& second)
{
    double maximum = 0.0;
    for (uint32_t channel = 0; channel < 3; ++channel) {
        maximum = std::max(maximum, std::abs(double(first[channel]) - second[channel]) /
            std::max({std::abs(double(first[channel])), std::abs(double(second[channel])), 1e-8}));
    }
    return maximum;
}

class AtmosphereLongitudeSeamTest final : public RHITest {
public:
    AtmosphereLongitudeSeamTest() { name = "atmosphere_moon_longitude_seam"; type = RHITestType::Rendering; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"environment.atmosphere.moon.longitude.continuity",
            "environment.atmosphere.angular.roundtrip", "environment.atmosphere.named.consumer"},
            bench::Layer::Core, "binding", "binding", {"moon-seam.tsv", "sun-seam.tsv", "offset-moon-seam.tsv", "cloud-moon-seam.tsv",
                "MoonSeamLookup.png", "MoonSeamReference.png", "MoonSeamConsumer.png",
                "MoonSeamLookup.rgba32f", "MoonSeamReference.rgba32f", "MoonSeamConsumer.rgba32f",
                "AtmosphereSeamMetrics.json", "MoonSeamImage.json"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        std::string log;
        auto result = createDevice({.applicationName = "Moon atmosphere longitude seam regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!result) { return RHITestResult::fail("Seam device: " + std::string(toString(result))); }
        ComputeKernel probeKernel, imageKernel;
        for (uint32_t index = 0; index < 2; ++index) {
            result = ShaderRegistry::instance().getComputeKernel(*device,
                {.moduleName = "AtmosphereSeamProbe", .entryPointName = index == 0 ?
                    "atmosphereSeamProbeMain" : "atmosphereSeamImageMain",
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                {.parameters = parameterAbi<AtmosphereSeamParams>(kAtmosphereSeamABI, ParameterTransport::InlinePush),
                    .debugName = "Moon longitude seam probe"}, index == 0 ? probeKernel : imageKernel, log);
            if (!result) { return RHITestResult::fail("Seam shader: " + log); }
        }
        double maximumContinuityError = 0.0, maximumRoundtripError = 0.0, maximumReferenceSymmetryError = 0.0;
        double maximumProducerEdgeError = 0.0;
        std::string failure;
        constexpr std::array<const char*, 4> labels{"moon-seam", "sun-seam", "offset-moon-seam", "cloud-moon-seam"};
        constexpr std::array<const char*, 3> imageLabels{"MoonSeamLookup", "MoonSeamReference", "MoonSeamConsumer"};
        for (uint32_t caseIndex = 0; caseIndex < labels.size(); ++caseIndex) {
            environment::WorldEnvironment world;
            world.source = environment::EnvironmentSource::PhysicalAtmosphere;
            world.sun.enabled = false;
            world.moon.enabled = true;
            world.moon.direction = {0.84f, -1.0f, caseIndex == 2 ? 0.001f : 0.0f};
            world.moon.angularRadius = 0.004642576f; // User's manual 0.266-degree Moon.
            if (caseIndex == 1) {
                world.sun = world.moon; // Same source flux isolates Sun/Moon transport from exposure.
                world.sun.enabled = true;
                world.moon.enabled = false;
            }
            if (caseIndex == 3) {
                world.weather.cloudEnabled = true;
                world.weather.cloudCoverage = 0.45f;
                world.weather.humidity = 0.25f;
                world.weather.noiseSeed = 17;
            }
            AtmosphereResourcesGPU resources;
            if (!(result = resources.initialize(*device, log))) { return RHITestResult::fail(log); }
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            if (!(result = gpu.initialize(*device))) { return RHITestResult::fail(std::string(toString(result))); }
            if (!(result = resources.record(*gpu.commands, world.snapshot(), {0.0, 2.0, 0.0}, log))) {
                return RHITestResult::fail(log);
            }
            std::unique_ptr<Buffer> numericOutput;
            result = seamReadback(*device, probeKernel, resources, *gpu.commands,
                kSeamSampleCount, 1, 0, kSeamRecordCount, (kSeamSampleCount + 63) / 64, 1, numericOutput);
            if (!result) { return RHITestResult::fail("Seam dispatch: " + std::string(toString(result))); }
            std::array<std::unique_ptr<Buffer>, 3> imageOutputs;
            if (caseIndex == 0) {
                for (uint32_t image = 0; image < 3; ++image) {
                    result = seamReadback(*device, imageKernel, resources, *gpu.commands,
                        kSeamImageWidth, kSeamImageHeight, image, 1,
                        kSeamImageWidth / 8, kSeamImageHeight / 8, imageOutputs[image]);
                    if (!result) { return RHITestResult::fail("Seam image dispatch: " + std::string(toString(result))); }
                }
            }
            if (!(result = gpu.submitAndWait())) { return RHITestResult::fail(std::string(toString(result))); }
            SeamValues values;
            if (!copySeamValues(*numericOutput, kSeamSampleCount * kSeamRecordCount, values) ||
                !saveSeamValues(context, labels[caseIndex], values)) { return RHITestResult::fail("Seam readback/evidence"); }
            for (uint32_t sample = 0; sample < kSeamSampleCount; ++sample) {
                const uint32_t base = sample * kSeamRecordCount;
                for (uint32_t record = 0; record < kSeamRecordCount; ++record) {
                    for (float channel : values[base + record]) {
                        if (!std::isfinite(channel)) { failure = "Nonfinite seam mapping/radiance"; }
                    }
                }
                const auto& mapping = values[base + 1];
                maximumRoundtripError = std::max(maximumRoundtripError, double(mapping[2]));
                if (mapping[0] < -1e-6f || mapping[0] > 1.000001f || mapping[1] < 0.0f || mapping[1] > 1.0f ||
                    mapping[2] > 2e-5f) { failure = "Sky mapping failed periodic angular roundtrip"; }
                if (relativeDifference(values[base + 2], values[base + 6]) > 1e-4) {
                    failure = "Named physicalVisibleEnvironment consumer differs from disk-free sky lookup";
                }
            }
            for (uint32_t band = 0; band < kSeamBandCount; ++band) {
                const uint32_t first = band * 2 * kSeamRecordCount, second = first + kSeamRecordCount;
                // Capture and production consumer are checked along with the direct LUT;
                // none of these eight elevation bands crosses the small source disk.
                for (uint32_t record : {2u, 4u, 5u, 6u, 13u, 14u}) {
                    const double error = relativeDifference(values[first + record], values[second + record]);
                    maximumContinuityError = std::max(maximumContinuityError, error);
                    if (error > 0.001) { failure = "Longitude seam radiance jump exceeds 0.1%; see seam TSV"; }
                }
                if (caseIndex < 2) {
                    const double referenceError = relativeDifference(values[first + 3], values[second + 3]);
                    const double edgeError = relativeDifference(values[first + 7], values[first + 8]);
                    maximumReferenceSymmetryError = std::max(maximumReferenceSymmetryError, referenceError);
                    maximumProducerEdgeError = std::max(maximumProducerEdgeError, edgeError);
                    if (referenceError > 0.001 || edgeError > 0.001) {
                        failure = "Cloud-free source-plane integration/first-last producer texels lack symmetry";
                    }
                }
            }
            if (caseIndex == 0) {
                const auto& sky = values[10 * kSeamRecordCount + 2];
                const float exposure = 0.3f / std::max({sky[0], sky[1], sky[2], 1e-8f});
                for (uint32_t image = 0; image < 3; ++image) {
                    SeamValues pixels;
                    if (!copySeamValues(*imageOutputs[image], kSeamImageWidth * kSeamImageHeight, pixels) ||
                        !saveSeamImage(context, imageLabels[image], pixels, exposure, log)) {
                        return RHITestResult::fail("Seam image evidence: " + log);
                    }
                }
                if (!saveSeamJson(context, "MoonSeamImage.json", bench::Json{
                    {"width", kSeamImageWidth}, {"height", kSeamImageHeight}, {"exposure", exposure},
                    {"longitudeSpanRadians", 1.1}, {"elevationSpanRadians", 0.8}, {"observerMetres", {0.0, 2.0, 0.0}}})) {
                    return RHITestResult::fail("Cannot save Moon seam image metadata");
                }
            }
        }
        if (!saveSeamJson(context, "AtmosphereSeamMetrics.json", bench::Json{
            {"maximumContinuityRelativeError", maximumContinuityError}, {"maximumRoundtripLengthError", maximumRoundtripError},
            {"maximumReferenceSymmetryRelativeError", maximumReferenceSymmetryError},
            {"maximumProducerEdgeRelativeError", maximumProducerEdgeError}, {"directionsPerCase", kSeamSampleCount}})) {
            return RHITestResult::fail("Cannot save atmosphere seam metrics");
        }
        if (!failure.empty()) { return RHITestResult::fail(failure); }
        return RHITestResult::pass("Moon-only, equivalent Sun and dynamic clouds: eight elevations and 256-longitude sweeps, periodic sky/capture/production consumer/aerial <0.1%, roundtrip <2e-5; paired raw HDR and 512x256 images");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereLongitudeSeamTest);

} // namespace
} // namespace metallic::tests
