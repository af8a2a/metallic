#include "RhiTest.h"

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numbers>

namespace metallic::tests {
namespace {

#define REGIR_CHECK(condition) \
    do { \
        if (!(condition)) { \
            return RhiTestResult::fail(std::string("ReGIR virtual lights: ") + #condition + "; " + log_); \
        } \
    } while (false)

constexpr uint32_t kProbeSampleCount = 65'536;
constexpr uint32_t kProbeLightCapacity = 16;
constexpr uint32_t kReservoirSlots = 4'096;
using ProbeRecord = std::array<float, 4>;

struct ReGIRProbePush {
    float position[3] = {};
    uint32_t sampleCount = kProbeSampleCount;
    uint32_t seedOffset = 0;
    uint32_t padding[3] = {};
};
static_assert(sizeof(ReGIRProbePush) == 32);

struct ReGIRProbeResult {
    std::array<double, 3> mean{};
    std::array<float, 3> exact{};
    std::array<uint32_t, kProbeLightCapacity> selected{};
    std::array<float, kProbeLightCapacity> power{};
    std::array<float, kProbeLightCapacity> pdfWeight{};
    uint32_t nullSamples = 0;
    float rootWeight = 0;
};

float cpuLightPower(const render::GpuPunctualLight& light)
{
    const float luminance = light.colorIntensity[3] * (light.colorIntensity[0] * 0.2126f +
        light.colorIntensity[1] * 0.7152f + light.colorIntensity[2] * 0.0722f);
    if (light.directionType[3] < 0.5f) { return luminance; }
    const float solidAngle = light.directionType[3] < 1.5f ? 4.0f * std::numbers::pi_v<float>
        : 2.0f * std::numbers::pi_v<float> *
            ((1.0f - light.spot[0]) + (light.spot[0] - light.spot[1]) / 3.0f);
    return luminance * solidAngle;
}

class ReGIRProbeHarness {
public:
    ~ReGIRProbeHarness()
    {
        if (queue_ != nullptr) { (void)queue_->waitIdle(); }
    }

    RhiTestResult initialize(bool validation)
    {
        const auto result = render::createDevice({.applicationName = "ReGIR virtual-light tests",
            .enableValidation = validation, .enableBindlessDescriptorHeap = true}, device_);
        if (render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::skip("ReGIR tests require bindless compute support");
        }
        REGIR_CHECK(result);
        queue_ = device_->getQueue(render::QueueType::Graphics);
        REGIR_CHECK(queue_ != nullptr);
        REGIR_CHECK(device_->createCommandPool(*queue_, pool_));
        REGIR_CHECK(pool_->createCommandBuffer(commands_));
        REGIR_CHECK(pdfCompute_.initialize(*device_, log_));
        REGIR_CHECK(pdf_.initialize(*device_, 4, 4, "ReGIR test power PDF", log_));
        REGIR_CHECK(selector_.initialize(*device_, log_));
        REGIR_CHECK(selector_.ensureGrid(*device_, 1, kReservoirSlots, log_));
        REGIR_CHECK(device_->createTexture({.usage = render::TextureUsageBits::Sampled,
            .format = render::Format::R32Sfloat, .width = 1, .height = 1}, dummyEnvironment_));
        REGIR_CHECK(device_->createTextureView(*dummyEnvironment_, {}, dummyEnvironmentView_));
        render::ShaderCompileResult shader;
        const auto compiled = render::compileSlangShaderToSpirv({.moduleName = "ReGIRVirtualLightProbe",
            .entryPointName = "reGIRVirtualLightProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!compiled) { return RhiTestResult::fail(shader.diagnostics); }
        const std::array<render::ComputeProgramBindingDesc, 4> bindings{{
            {.binding = 0}, {.binding = 50}, {.binding = 52},
            {.binding = 53, .kind = render::ComputeResourceBindingKind::SampledImage},
        }};
        REGIR_CHECK(program_.initialize(*device_, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t),
            .pushConstantSize = sizeof(ReGIRProbePush), .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()), .requiresRayQuery = false}, log_));
        return RhiTestResult::pass();
    }

    // Every call replaces the physical-light upload and rebuilds both GPU
    // distributions, matching the edit/deletion path used by the render passes.
    RhiTestResult run(const std::vector<render::GpuPunctualLight>& lights,
        const std::array<float, 3>& position, ReGIRProbeResult& output,
        bool halfEmptyReservoir = false, bool invalidGridHeader = false,
        const scene::LightingSettings* cancelledWrapperSettings = nullptr)
    {
        REGIR_CHECK(!lights.empty() && lights.size() <= kProbeLightCapacity + 1);
        render::RenderSubsystemHost samplingHost;
        render::RenderFrameContext samplingFrame;
        render::QueueSubmissionTracker samplingSubmissions;
        render::SceneLightResources wrappedLights;
        struct HostScope {
            render::RenderSubsystemHost& host;
            ~HostScope() { host.shutdown(); }
        } hostScope{samplingHost};
        REGIR_CHECK(pool_->reset());
        if (cancelledWrapperSettings != nullptr) {
            REGIR_CHECK(samplingHost.initialize(*device_, 1, log_));
            REGIR_CHECK(samplingSubmissions.initialize(*device_, *queue_));
            REGIR_CHECK(samplingFrame.begin(frameIndex_));
            REGIR_CHECK(commands_->begin(&samplingFrame));
            REGIR_CHECK(samplingHost.beginFrame(frameIndex_, 0, nullptr, log_, &samplingFrame));
        } else {
            REGIR_CHECK(commands_->begin());
        }
        std::unique_ptr<render::Buffer> lightBuffer;
        REGIR_CHECK(device_->createBuffer({.size = lights.size() * sizeof(render::GpuPunctualLight),
            .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostUpload}, lightBuffer));
        void* mapped = lightBuffer->map();
        REGIR_CHECK(mapped != nullptr);
        std::memcpy(mapped, lights.data(), lights.size() * sizeof(render::GpuPunctualLight));
        lightBuffer->flush();
        lightBuffer->unmap();
        commands_->hostWriteBarrier();
        if (frameIndex_ == 0) {
            const render::TextureBarrierDesc barrier{.texture = dummyEnvironment_.get(),
                .before = render::ResourceState::Undefined, .after = render::ResourceState::ShaderRead};
            commands_->barrier({.textures = &barrier, .textureCount = 1});
        }
        const auto lightCount = static_cast<uint32_t>(lights.size() - 1);
        render::ReGIRBuildParameters parameters;
        parameters.lightCount = lightCount;
        parameters.buildSamples = 32;
        parameters.frameIndex = frameIndex_++;
        parameters.sceneRadius = 1.0f;
        parameters.samplingJitter = 0.0f;
        if (cancelledWrapperSettings != nullptr) {
            REGIR_CHECK(wrappedLights.update(*device_, *commands_, samplingHost,
                nullptr, *cancelledWrapperSettings));
            REGIR_CHECK(wrappedLights.buildSampling(*device_, *commands_, samplingHost,
                *dummyEnvironmentView_, parameters, 1, kReservoirSlots, true, log_));
            auto* abandonedPdf = wrappedLights.lightPdfView();
            auto* abandonedGrid = wrappedLights.reGIRBuffer();
            REGIR_CHECK(abandonedPdf != nullptr && abandonedGrid != nullptr);
            // No queue submission occurs here: texture transitions and ReGIR
            // writes exist only in the abandoned command recording.
            REGIR_CHECK(commands_->end());
            samplingHost.endFrame();
            samplingFrame.cancel();
            REGIR_CHECK(samplingFrame.completion().isCancelled());
            REGIR_CHECK(pool_->reset());
            REGIR_CHECK(samplingFrame.begin(frameIndex_));
            REGIR_CHECK(commands_->begin(&samplingFrame));
            REGIR_CHECK(samplingHost.beginFrame(frameIndex_, 0, nullptr, log_, &samplingFrame));
            if (parameters.frameIndex == 0) {
                const render::TextureBarrierDesc retryBarrier{.texture = dummyEnvironment_.get(),
                    .before = render::ResourceState::Undefined, .after = render::ResourceState::ShaderRead};
                commands_->barrier({.textures = &retryBarrier, .textureCount = 1});
            }
            REGIR_CHECK(wrappedLights.update(*device_, *commands_, samplingHost,
                nullptr, *cancelledWrapperSettings));
            REGIR_CHECK(wrappedLights.buildSampling(*device_, *commands_, samplingHost,
                *dummyEnvironmentView_, parameters, 1, kReservoirSlots, true, log_));
            REGIR_CHECK(wrappedLights.lightPdfView() != abandonedPdf);
            REGIR_CHECK(wrappedLights.reGIRBuffer() != abandonedGrid);
        } else {
            REGIR_CHECK(pdfCompute_.buildLocalLights(*commands_, *dummyEnvironmentView_, pdf_,
                *lightBuffer, lightCount));
            REGIR_CHECK(selector_.build(*commands_, *pdf_.view(), *lightBuffer, parameters));
        }

        std::unique_ptr<render::Buffer> syntheticGrid;
        if (halfEmptyReservoir || invalidGridHeader) {
            std::vector<std::array<uint32_t, 4>> slots(kReservoirSlots + render::kReGIRHeaderRecordCount);
            slots[0] = {1, 1, 1, kReservoirSlots};
            slots[1] = {0, 0, 0, std::bit_cast<uint32_t>(2.0f)};
            slots[2] = {invalidGridHeader ? 0u : 0x52454749u, kReservoirSlots, 0, frameIndex_};
            for (uint32_t i = 0; i < kReservoirSlots; ++i) {
                // A zero-weight proposal is retained as null mass. Retrying
                // through the global PDF doubles the estimator's expectation.
                slots[i + render::kReGIRHeaderRecordCount] = (i & 1u) == 0
                    ? std::array<uint32_t, 4>{0, std::bit_cast<uint32_t>(2.0f), 0, 0}
                    : std::array<uint32_t, 4>{UINT32_MAX, 0, 0, 0};
            }
            REGIR_CHECK(device_->createBuffer({.size = slots.size() * sizeof(slots[0]),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostUpload}, syntheticGrid));
            mapped = syntheticGrid->map();
            REGIR_CHECK(mapped != nullptr);
            std::memcpy(mapped, slots.data(), slots.size() * sizeof(slots[0]));
            syntheticGrid->flush();
            syntheticGrid->unmap();
            commands_->hostWriteBarrier();
        }

        constexpr uint64_t outputBytes = (kProbeSampleCount + 2 + kProbeLightCapacity) * sizeof(ProbeRecord);
        std::unique_ptr<render::Buffer> probe;
        std::unique_ptr<render::Buffer> readback;
        REGIR_CHECK(device_->createBuffer({.size = outputBytes,
            .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::Device}, probe));
        REGIR_CHECK(device_->createBuffer({.size = outputBytes,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}, readback));
        const render::BufferBarrierDesc outputBarrier{.buffer = probe.get(),
            .before = render::ResourceState::Undefined, .after = render::ResourceState::General};
        commands_->barrier({.buffers = &outputBarrier, .bufferCount = 1});
        render::TextureView* pdfViews[] = {
            cancelledWrapperSettings != nullptr ? wrappedLights.lightPdfView() : pdf_.view()};
        const std::array<render::ComputeDispatchBinding, 4> bindings{{
            {.binding = 0, .buffer = probe.get()},
            {.binding = 50, .buffer = cancelledWrapperSettings != nullptr ? wrappedLights.buffer() : lightBuffer.get()},
            {.binding = 52, .buffer = cancelledWrapperSettings != nullptr ? wrappedLights.reGIRBuffer()
                : (syntheticGrid ? syntheticGrid.get() : selector_.buffer())},
            {.binding = 53, .textureViews = pdfViews, .textureViewCount = 1},
        }};
        ReGIRProbePush push;
        std::copy(position.begin(), position.end(), push.position);
        push.seedOffset = frameIndex_;
        REGIR_CHECK(program_.dispatch({.commandBuffer = commands_.get(), .bindings = bindings.data(),
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pushData = &push,
            .pushDataSize = sizeof(push), .groupCountX = kProbeSampleCount / 256}));
        const std::array transferBarriers{
            render::BufferBarrierDesc{.buffer = probe.get(), .before = render::ResourceState::General,
                .after = render::ResourceState::TransferSource},
            render::BufferBarrierDesc{.buffer = readback.get(), .before = render::ResourceState::Undefined,
                .after = render::ResourceState::TransferDestination},
        };
        commands_->barrier({.buffers = transferBarriers.data(),
            .bufferCount = static_cast<uint32_t>(transferBarriers.size())});
        commands_->copyBuffer({.source = probe.get(), .destination = readback.get(), .size = outputBytes});
        REGIR_CHECK(commands_->end());
        render::CommandBuffer* submissions[] = {commands_.get()};
        if (cancelledWrapperSettings != nullptr) {
            samplingHost.endFrame();
            REGIR_CHECK(samplingSubmissions.submit({.commandBuffers = submissions, .commandBufferCount = 1}, samplingFrame));
            REGIR_CHECK(samplingFrame.wait(10'000'000'000ull));
        } else {
            REGIR_CHECK(queue_->submit({.commandBuffers = submissions, .commandBufferCount = 1}));
        }
        REGIR_CHECK(queue_->waitIdle());
        readback->invalidate();
        const auto* data = static_cast<const ProbeRecord*>(readback->map());
        REGIR_CHECK(data != nullptr);
        output = {};
        bool finite = true;
        bool indicesValid = true;
        for (uint32_t i = 0; i < kProbeSampleCount; ++i) {
            for (size_t channel = 0; channel < 3; ++channel) {
                finite &= std::isfinite(data[i][channel]) && data[i][channel] >= 0.0f;
                output.mean[channel] += double(data[i][channel]) / kProbeSampleCount;
            }
            const uint32_t selected = std::bit_cast<uint32_t>(data[i][3]);
            if (selected == UINT32_MAX) { ++output.nullSamples; }
            else if (selected < lightCount) { ++output.selected[selected]; }
            else { indicesValid = false; }
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            output.exact[channel] = data[kProbeSampleCount][channel];
        }
        output.rootWeight = data[kProbeSampleCount + 1][0];
        for (uint32_t i = 0; i < kProbeLightCapacity; ++i) {
            output.power[i] = data[kProbeSampleCount + 2 + i][0];
            output.pdfWeight[i] = data[kProbeSampleCount + 2 + i][1];
        }
        readback->unmap();
        REGIR_CHECK(finite && indicesValid);
        for (uint32_t i = 0; i < kProbeLightCapacity; ++i) {
            const float expected = i < lightCount ? cpuLightPower(lights[i + 1]) : 0.0f;
            const float tolerance = std::max(0.00001f, std::abs(expected) * 0.0001f);
            REGIR_CHECK(std::abs(output.power[i] - expected) <= tolerance);
            REGIR_CHECK(std::abs(output.pdfWeight[i] - expected) <= tolerance);
        }
        return RhiTestResult::pass();
    }

private:
    // Device must outlive every program, resource and command object.
    std::unique_ptr<render::Device> device_;
    render::Queue* queue_ = nullptr;
    std::unique_ptr<render::CommandPool> pool_;
    std::unique_ptr<render::CommandBuffer> commands_;
    render::ImportancePdfCompute pdfCompute_;
    render::ImportancePdfTexture pdf_;
    render::ReGIRLightSelector selector_;
    render::ComputeProgram program_;
    std::unique_ptr<render::Texture> dummyEnvironment_;
    std::unique_ptr<render::TextureView> dummyEnvironmentView_;
    std::string log_;
    uint32_t frameIndex_ = 0;
};

scene::LightingSettings mixedVirtualLights()
{
    scene::LightingSettings settings;
    for (const char* type : {"point", "spot", "directional"}) {
        scene::PunctualLight light;
        light.properties.type = type;
        light.position = float3(0, 0, 3);
        light.direction = float3(0, 0, -1);
        const size_t index = settings.lights.size();
        light.properties.color = index == 0 ? float3(1, 0, 0)
            : (index == 1 ? float3(0, 1, 0) : float3(0, 0, 1));
        light.properties.intensity = index == 2 ? 1.0 : 3.0;
        light.properties.intensityUnit = index == 2 ? scene::LightUnit::Lux : scene::LightUnit::Candela;
        settings.lights.push_back(light);
    }
    return settings;
}

std::vector<render::GpuPunctualLight> withInactiveSlot(const scene::LightingSettings& settings)
{
    auto records = render::buildPunctualLightRecords(nullptr, settings);
    records.insert(records.begin() + 1, render::GpuPunctualLight{});
    records[0].positionRange[0] = static_cast<float>(records.size() - 1);
    return records;
}

RhiTestResult expectEstimator(const ReGIRProbeResult& result, const char* phase, double tolerance = 0.12)
{
    for (size_t channel = 0; channel < 3; ++channel) {
        if (std::abs(result.mean[channel] - result.exact[channel]) >
            std::max(0.00001, std::abs(double(result.exact[channel])) * tolerance)) {
            return RhiTestResult::fail(std::string(phase) + " channel " + std::to_string(channel) +
                ": estimate=" + std::to_string(result.mean[channel]) +
                ", exact=" + std::to_string(result.exact[channel]));
        }
    }
    return RhiTestResult::pass();
}

class ReGIRVirtualLightGpuTest final : public RhiTest {
public:
    ReGIRVirtualLightGpuTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_gpu_power_edit_delete";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        ReGIRProbeHarness harness;
        auto status = harness.initialize(context.enableValidation);
        if (!status.passed) { return status; }
        auto settings = mixedVirtualLights();
        ReGIRProbeResult inside;
        status = harness.run(withInactiveSlot(settings), {0, 0, 0}, inside);
        if (!status.passed) { return status; }
        status = expectEstimator(inside, "inside grid");
        if (!status.passed) { return status; }
        if (inside.selected[0] != 0 || inside.selected[1] == 0 || inside.selected[2] == 0 ||
            inside.selected[3] == 0 || std::abs(inside.exact[0] - 1.0f / 3.0f) > 0.0001f ||
            std::abs(inside.exact[1] - 1.0f / 3.0f) > 0.0001f || inside.exact[2] != 1.0f) {
            return RhiTestResult::fail("ReGIR omitted a physical light type or sampled an inactive source slot");
        }
        ReGIRProbeResult outside;
        status = harness.run(withInactiveSlot(settings), {0, 0, 1.25f}, outside);
        if (!status.passed) { return status; }
        status = expectEstimator(outside, "outside-grid power-PDF fallback");
        if (!status.passed) { return status; }
        settings.lights[0].properties.intensity *= 2.0;
        settings.lights[1].direction = float3(0, 0, 1);
        ReGIRProbeResult edited;
        status = harness.run(withInactiveSlot(settings), {0, 0, 0}, edited);
        if (!status.passed) { return status; }
        status = expectEstimator(edited, "light edit");
        if (!status.passed) { return status; }
        if (std::abs(edited.exact[0] - 2.0f * inside.exact[0]) > 0.0001f || edited.exact[1] != 0.0f ||
            std::abs(edited.power[1] - 2.0f * inside.power[1]) > 0.0001f) {
            return RhiTestResult::fail("point intensity or spot orientation did not refresh GPU sampling");
        }
        settings.lights.erase(settings.lights.begin());
        settings.lights[0].enabled = false;
        ReGIRProbeResult removed;
        status = harness.run(withInactiveSlot(settings), {0, 0, 0}, removed);
        if (!status.passed) { return status; }
        status = expectEstimator(removed, "delete and disable");
        if (!status.passed) { return status; }
        if (removed.exact[0] != 0.0f || removed.exact[1] != 0.0f || removed.exact[2] != 1.0f ||
            removed.selected[1] != kProbeSampleCount) {
            return RhiTestResult::fail("removed light remained in a reused ReGIR/PDF resource");
        }
        settings.lights.clear();
        ReGIRProbeResult empty;
        status = harness.run(render::buildPunctualLightRecords(nullptr, settings), {0, 0, 0}, empty);
        if (!status.passed) { return status; }
        if (empty.nullSamples != kProbeSampleCount || empty.rootWeight != 0.0f ||
            empty.mean != std::array<double, 3>{}) {
            return RhiTestResult::fail("zero-light build retained stale ReGIR candidates or PDF mass");
        }
        // A non-empty source table can also carry no emitted power.
        status = harness.run(withInactiveSlot(settings), {0, 0, 0}, empty);
        if (!status.passed) { return status; }
        if (empty.mean != std::array<double, 3>{} || empty.rootWeight != 0.0f) {
            return RhiTestResult::fail("all-zero source table produced illumination");
        }
        return RhiTestResult::pass("GPU point/spot/directional power, ReGIR and global-PDF estimates, inactive slots and edits/deletion");
    }
};

class ReGIRNullReservoirGpuTest final : public RhiTest {
public:
    ReGIRNullReservoirGpuTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_null_reservoir_mass";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        ReGIRProbeHarness harness;
        auto status = harness.initialize(context.enableValidation);
        if (!status.passed) { return status; }
        auto settings = mixedVirtualLights();
        settings.lights.erase(settings.lights.begin(), settings.lights.begin() + 2);
        const auto records = render::buildPunctualLightRecords(nullptr, settings);
        ReGIRProbeResult probe;
        status = harness.run(records, {0, 0, 0}, probe, true);
        if (!status.passed) { return status; }
        status = expectEstimator(probe, "half-empty reservoirs", 0.03);
        if (!status.passed) { return status; }
        if (probe.nullSamples < kProbeSampleCount * 45 / 100 ||
            probe.nullSamples > kProbeSampleCount * 55 / 100) {
            return RhiTestResult::fail("empty ReGIR slot must remain a null sample, not retry the global PDF");
        }
        for (const bool invalidHeader : {false, true}) {
            status = harness.run(records, invalidHeader ? std::array<float, 3>{0, 0, 0}
                : std::array<float, 3>{0, 0, 2}, probe, true, invalidHeader);
            if (!status.passed) { return status; }
            status = expectEstimator(probe, "global fallback", 0.00001);
            if (!status.passed) { return status; }
            if (probe.nullSamples != 0 || probe.selected[0] != kProbeSampleCount) {
                return RhiTestResult::fail("outside/invalid grid did not fall back to the live physical-light PDF");
            }
        }
        return RhiTestResult::pass("null RIS probability mass is retained; outside/invalid grid uses global PDF");
    }
};

class ReGIRCancelledSamplingRetryTest final : public RhiTest {
public:
    ReGIRCancelledSamplingRetryTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_cancelled_sampling_retry";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        ReGIRProbeHarness harness;
        auto status = harness.initialize(context.enableValidation);
        if (!status.passed) { return status; }
        const auto settings = mixedVirtualLights();
        ReGIRProbeResult probe;
        status = harness.run(render::buildPunctualLightRecords(nullptr, settings),
            {0, 0, 0}, probe, false, false, &settings);
        if (!status.passed) { return status; }
        status = expectEstimator(probe, "cancelled wrapper sampling retry");
        if (!status.passed) { return status; }
        if (probe.selected[0] == 0 || probe.selected[1] == 0 || probe.selected[2] == 0 ||
            std::abs(probe.exact[0] - 1.0f / 3.0f) > 0.0001f ||
            std::abs(probe.exact[1] - 1.0f / 3.0f) > 0.0001f || probe.exact[2] != 1.0f) {
            return RhiTestResult::fail("cancelled sampling retry lost live physical-light power");
        }
        return RhiTestResult::pass("unsubmitted first build cancelled; replacement PDF/ReGIR retry read back correctly");
    }
};

std::array<uint64_t, 3> rgbEnergy(const std::vector<uint32_t>& pixels)
{
    std::array<uint64_t, 3> energy{};
    const auto* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
    for (size_t pixel = 0; pixel < pixels.size(); ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) { energy[channel] += bytes[pixel * 4 + channel]; }
    }
    return energy;
}

RhiTestResult renderVirtualLightLifecycle(RhiTestContext& context, const char* renderer)
{
    render::RenderGraphPreviewRenderer preview;
    auto result = preview.initialize(context.enableValidation, true);
    if (render::hasError(result, render::Error::Unsupported)) {
        return RhiTestResult::skip("virtual-light transport tests require ray query");
    }
    if (!result) { return RhiTestResult::fail("virtual-light preview initialization failed"); }
    preview.setEnvironment({.enabled = false, .visible = false});
    const bool rtxdi = std::string_view(renderer) == "rtxdi";
    render::RenderGraphProperties properties{
        {"path", "Asset/meet_mat.glb"}, {"maxDepth", 1}, {"samples", 4}, {"accumulate", true},
        {"bsdf", renderer}, {"lightSource", "scene"},
        {"temporalReuse", true}, {"spatialReuse", true}, {"environmentSamples", 0},
        {"regirEnabled", true}, {"regirGridSize", 4}, {"regirLightsPerCell", 8},
        {"regirBuildSamples", 8}, {"initialSamples", 8},
        {"camera", {{"eye", {0.0, 0.25, 3.0}}, {"center", {0.0, 0.15, 0.0}}}},
    };
    render::RenderGraph graph;
    graph.addNode(rtxdi ? "SceneRtxdiPass" : "ScenePathTracePass", "Lighting", properties);
    graph.markOutput("Lighting.color");
    auto renderFrames = [&]() -> RhiTestResult {
        // The second frame consumes temporal reservoirs / accumulated history.
        for (uint32_t frame = 0; frame < 2; ++frame) {
            const auto rendered = preview.render(graph, 64, 64);
            if (!rendered) {
                if (render::hasError(rendered, render::Error::Unsupported)) {
                    return RhiTestResult::skip(preview.lastLog());
                }
                return RhiTestResult::fail(std::string(renderer) + ": " + preview.lastLog());
            }
        }
        return RhiTestResult::pass();
    };
    auto status = renderFrames();
    if (!status.passed) { return status; }
    const auto dark = preview.pixels();
    const auto darkEnergy = rgbEnergy(dark);
    scene::LightingSettings settings;
    settings.exposureEV100 = 8;
    auto& light = settings.lights.emplace_back();
    light.properties.type = "directional";
    light.properties.intensityUnit = scene::LightUnit::Lux;
    light.properties.intensity = 1000;
    light.properties.color = float3(1, 0, 0);
    light.direction = float3(0.0f, -0.2f, -1.0f);
    if (!preview.setLighting(settings)) { return RhiTestResult::fail("invalid directional fixture"); }
    status = renderFrames();
    if (!status.passed) { return status; }
    auto litEnergy = rgbEnergy(preview.pixels());
    if (litEnergy[0] < darkEnergy[0] + 1024 || litEnergy[1] > darkEnergy[1] + 1024 ||
        litEnergy[2] > darkEnergy[2] + 1024) {
        return RhiTestResult::fail(std::string(renderer) + " did not transport the red scene directional light");
    }
    light.properties.color = float3(0, 0, 1);
    if (!preview.setLighting(settings)) { return RhiTestResult::fail("invalid color edit fixture"); }
    status = renderFrames();
    if (!status.passed) { return status; }
    litEnergy = rgbEnergy(preview.pixels());
    if (litEnergy[2] < darkEnergy[2] + 1024 || litEnergy[0] > darkEnergy[0] + 1024) {
        return RhiTestResult::fail(std::string(renderer) + " retained stale red-light history after a blue edit");
    }
    for (const char* localType : {"point", "spot"}) {
        light.properties.type = localType;
        light.properties.intensityUnit = scene::LightUnit::Candela;
        light.properties.intensity = 4000;
        light.properties.range = 10;
        light.properties.color = float3(0, 1, 0);
        light.position = float3(0.0f, 0.25f, 3.0f);
        light.direction = float3(0, 0, -1);
        if (!preview.setLighting(settings)) { return RhiTestResult::fail("invalid local-light fixture"); }
        status = renderFrames();
        if (!status.passed) { return status; }
        litEnergy = rgbEnergy(preview.pixels());
        if (litEnergy[1] < darkEnergy[1] + 1024 || litEnergy[0] > darkEnergy[0] + 1024 ||
            litEnergy[2] > darkEnergy[2] + 1024) {
            return RhiTestResult::fail(std::string(renderer) + " did not transport the green scene " + localType);
        }
    }
    std::string outputMessage;
    if (!saveRgba8Png(context.outputDirectory / (std::string("regir-virtual-") + renderer + ".png"),
            reinterpret_cast<const uint8_t*>(preview.pixels().data()), 64, 64, outputMessage)) {
        return RhiTestResult::fail(outputMessage);
    }
    settings.lights.clear();
    settings.exposureEV100 = 0;
    if (!preview.setLighting(settings)) { return RhiTestResult::fail("could not delete virtual lights"); }
    status = renderFrames();
    if (!status.passed) { return status; }
    if (preview.pixels() != dark) {
        return RhiTestResult::fail(std::string(renderer) + " retained illumination after deleting all virtual lights");
    }
    return RhiTestResult::pass(std::string(renderer) + ": directional/point/spot RGB transport and edit/delete history invalidation");
}

class ReGIRStandardPathTraceLightsTest final : public RhiTest {
public:
    ReGIRStandardPathTraceLightsTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_standard_path_trace_render";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        return renderVirtualLightLifecycle(context, "standard");
    }
};

class ReGIROpenPBRPathTraceLightsTest final : public RhiTest {
public:
    ReGIROpenPBRPathTraceLightsTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_openpbr_path_trace_render";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        return renderVirtualLightLifecycle(context, "openpbr");
    }
};

class ReGIRRtxdiVirtualLightsTest final : public RhiTest {
public:
    ReGIRRtxdiVirtualLightsTest()
    {
        type = RhiTestType::Rendering;
        name = "regir_virtual_lights_rtxdi_temporal_render";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        return renderVirtualLightLifecycle(context, "rtxdi");
    }
};

METALLIC_REGISTER_RHI_TEST(ReGIRVirtualLightGpuTest);
METALLIC_REGISTER_RHI_TEST(ReGIRNullReservoirGpuTest);
METALLIC_REGISTER_RHI_TEST(ReGIRCancelledSamplingRetryTest);
METALLIC_REGISTER_RHI_TEST(ReGIRStandardPathTraceLightsTest);
METALLIC_REGISTER_RHI_TEST(ReGIROpenPBRPathTraceLightsTest);
METALLIC_REGISTER_RHI_TEST(ReGIRRtxdiVirtualLightsTest);

#undef REGIR_CHECK

} // namespace
} // namespace metallic::tests
