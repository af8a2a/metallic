#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/Core/LightingKernelParameters.h"
#include "Runtime/Render/Core/RTXDIPostProcessParameters.h"
#include "Runtime/Render/Core/PathTraceStageParameters.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <array>
#include <cstring>
#include <thread>
#include <map>

namespace metallic::tests {
namespace {

#define REG_REQUIRE(expression) do { \
    const render::Result<> result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + toString(result)); } \
} while (false)
#define REG_CHECK(expression) do { \
    if (!(expression)) { return RHITestResult::fail(#expression); } \
} while (false)

constexpr uint64_t kABI = 0x5245475000000001ull;
struct ProbeParams {
    render::ShaderBuffer source, output;
    uint32_t add, index;
};
static_assert(sizeof(ProbeParams) == 16 && offsetof(ProbeParams, add) == 8);

render::Result<> makeBuffer(render::Device& device, std::unique_ptr<render::Buffer>& buffer, uint32_t value = 0)
{
    auto result = device.createBuffer({.size = 64, .structureStride = 4,
        .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Indirect,
        .memoryLocation = render::MemoryLocation::HostReadback,
        .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
    if (!result) { return result; }
    void* mapped = buffer->map();
    if (!mapped) { return render::makeError(render::Error::Failure); }
    std::memset(mapped, 0, 64);
    std::memcpy(mapped, &value, 4);
    buffer->flush();
    buffer->unmap();
    return {};
}

struct Commands {
    render::RenderFrameContext frame;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> commands;
    ~Commands()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
    }
    render::Result<> initialize(render::Device& device, render::Queue& queue)
    {
        auto result = device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        return result ? pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) : result;
    }
    render::Result<> begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    render::Result<> submit(render::QueueSubmissionTracker& tracker, render::Semaphore& gate)
    {
        auto result = commands->end();
        if (!result) { return result; }
        render::CommandBuffer* buffers[] = {commands.get()};
        render::SemaphoreSubmitDesc wait{.semaphore = &gate, .value = 1};
        return tracker.submit({
            .waitSemaphores = {&wait, 1},
            .commandBuffers = {buffers, 1},
        }, frame);
    }
};

struct Drain {
    render::Queue& queue;
    render::Semaphore& gate;
    ~Drain()
    {
        if (gate.currentValue() < 1) { (void)gate.signal(1); }
        (void)queue.waitIdle();
    }
};

render::Result<> makeKernel(render::Device& device, render::ComputeKernel& kernel, std::string& log,
    render::SlangDescriptorHeapMode mode = render::SlangDescriptorHeapMode::Default,
    render::ParameterTransport transport = render::ParameterTransport::DescriptorBuffer)
{
    const render::SlangMacroDefine defines[] = {{"INLINE_PARAMETERS", transport == render::ParameterTransport::InlinePush ? "1" : "0"}};
    render::ShaderCompileResult shader;
    auto result = render::compileSlangShaderToSpirv({.moduleName = "RegistryProbe",
        .entryPointName = "registryProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .macroDefines = defines, .descriptorHeapMode = mode}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
    if (!result) { log = shader.diagnostics; return result; }
    for (size_t word = 5; word < shader.spirv.size();) {
        const uint32_t count = shader.spirv[word] >> 16, opcode = shader.spirv[word] & 0xffff;
        if (!count || count > shader.spirv.size() - word ||
            (opcode == 17 && count == 2 && shader.spirv[word + 1] == 5347)) {
            log = "Ordinary resource/parameter access must not require PhysicalStorageBufferAddresses";
            return render::makeError(render::Error::Failure);
        }
        word += count;
    }
    return kernel.initialize(device, {.spirv = shader.spirv, .parameters = render::parameterAbi<ProbeParams>(kABI, transport)}, log);
}

// Inspect emitted layout, including fields unused by a particular entry point.
// Sharing declarations alone cannot detect a CPU/Slang packing disagreement.
class PostProcessParameterLayoutTest : public RHITest {
public:
    PostProcessParameterLayoutTest() { type = RHITestType::Resource; name = "post_process_parameter_spirv_layout"; }
    uint32_t category = 0;
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
#define FIELD(Type, Member) uint32_t(offsetof(Type, Member))
        struct Layout {
            const char* name;
            std::vector<uint32_t> offsets;
        };
        const Layout layouts[] = {
            {"Metallic.FinalBlitParams", {FIELD(FinalBlitParams, output), FIELD(FinalBlitParams, source),
                FIELD(FinalBlitParams, lut), FIELD(FinalBlitParams, lutSampler), FIELD(FinalBlitParams, display), FIELD(FinalBlitParams, padding)}},
            {"Metallic.SliderDebugParams", {FIELD(SliderDebugParams, sourceA), FIELD(SliderDebugParams, sourceB),
                FIELD(SliderDebugParams, output), FIELD(SliderDebugParams, display), FIELD(SliderDebugParams, padding)}},
            {"Metallic.AutoExposureParams", {FIELD(AutoExposureParams, source), FIELD(AutoExposureParams, output),
                FIELD(AutoExposureParams, histogram), FIELD(AutoExposureParams, history), FIELD(AutoExposureParams, exposure),
                FIELD(AutoExposureParams, display), FIELD(AutoExposureParams, padding)}},
            {"Metallic.ColorGradingLUTParams", {FIELD(ColorGradingLUTParams, output), FIELD(ColorGradingLUTParams, custom0),
                FIELD(ColorGradingLUTParams, custom1), FIELD(ColorGradingLUTParams, custom2), FIELD(ColorGradingLUTParams, custom3),
                FIELD(ColorGradingLUTParams, reach), FIELD(ColorGradingLUTParams, gamut), FIELD(ColorGradingLUTParams, gammaTable),
                FIELD(ColorGradingLUTParams, sampler), FIELD(ColorGradingLUTParams, padding0), FIELD(ColorGradingLUTParams, padding1),
                FIELD(ColorGradingLUTParams, padding2),
                FIELD(ColorGradingLUTParams, display)}},
            {"Metallic.ClusterLightGridBuildParams", {FIELD(ClusterLightGridBuildParams, grid), FIELD(ClusterLightGridBuildParams, lights),
                FIELD(ClusterLightGridBuildParams, candidates), FIELD(ClusterLightGridBuildParams, cells), FIELD(ClusterLightGridBuildParams, indices)}},
            {"Metallic.LightGridDebugParams", {FIELD(LightGridDebugParams, grid), FIELD(LightGridDebugParams, cells),
                FIELD(LightGridDebugParams, output), FIELD(LightGridDebugParams, settings)}},
            {"Metallic.PrepareLightsPdfParams", {FIELD(PrepareLightsPdfParams, environment), FIELD(PrepareLightsPdfParams, sourceMip),
                FIELD(PrepareLightsPdfParams, destinationMip), FIELD(PrepareLightsPdfParams, lights), FIELD(PrepareLightsPdfParams, settings)}},
            {"Metallic.BuildReGIRParams", {FIELD(BuildReGIRParams, localLightPdf), FIELD(BuildReGIRParams, output),
                FIELD(BuildReGIRParams, lights), FIELD(BuildReGIRParams, padding0), FIELD(BuildReGIRParams, settings)}},
            {"Metallic.EnvironmentLightingPrecomputeParams", {FIELD(EnvironmentLightingPrecomputeParams, radiance),
                FIELD(EnvironmentLightingPrecomputeParams, partials), FIELD(EnvironmentLightingPrecomputeParams, coefficients),
                FIELD(EnvironmentLightingPrecomputeParams, specular), FIELD(EnvironmentLightingPrecomputeParams, settings)}},
            {"Metallic.RTXDIConfidenceParams", {FIELD(RTXDIConfidenceParams, noisyDiffuse), FIELD(RTXDIConfidenceParams, noisySpecular),
                FIELD(RTXDIConfidenceParams, baseColorMetalness), FIELD(RTXDIConfidenceParams, motionVectors),
                FIELD(RTXDIConfidenceParams, previousLuminance), FIELD(RTXDIConfidenceParams, currentLuminance),
                FIELD(RTXDIConfidenceParams, gradientA), FIELD(RTXDIConfidenceParams, gradientB),
                FIELD(RTXDIConfidenceParams, previousDiffuseConfidence), FIELD(RTXDIConfidenceParams, previousSpecularConfidence),
                FIELD(RTXDIConfidenceParams, diffuseConfidence), FIELD(RTXDIConfidenceParams, specularConfidence),
                FIELD(RTXDIConfidenceParams, currentDiffuseConfidence), FIELD(RTXDIConfidenceParams, currentSpecularConfidence),
                FIELD(RTXDIConfidenceParams, settings)}},
            {"Metallic.RTXDICompositeParams", {FIELD(RTXDICompositeParams, denoisedDiffuse), FIELD(RTXDICompositeParams, denoisedSpecular),
                FIELD(RTXDICompositeParams, baseColorMetalness), FIELD(RTXDICompositeParams, emissive),
                FIELD(RTXDICompositeParams, output), FIELD(RTXDICompositeParams, settings)}},
            {"Metallic.SharcMaintenanceParams", {FIELD(SharcMaintenanceParams, hashEntries), FIELD(SharcMaintenanceParams, accumulation),
                FIELD(SharcMaintenanceParams, resolved), FIELD(SharcMaintenanceParams, padding0),
                FIELD(SharcMaintenanceParams, settings)}},
            {"Metallic.PathTraceTonemapParams", {FIELD(PathTraceTonemapParams, source), FIELD(PathTraceTonemapParams, output),
                FIELD(PathTraceTonemapParams, historyPrevious), FIELD(PathTraceTonemapParams, settings)}},
        };
#undef FIELD
        struct Program { const char* module; const char* entry; uint32_t layout; };
        const Program programs[] = {
            {"Features/PostProcess/FinalBlit", "finalBlitMain", 0},
            {"Features/PostProcess/FinalBlit", "finalBlitUvMain", 0},
            {"Features/Debug/SliderDebug", "sliderDebugMain", 1},
            {"Features/Debug/SliderDebug", "sliderDebugOverlayMain", 1},
            {"Features/PostProcess/AutoExposure", "autoExposureHistogramMain", 2},
            {"Features/PostProcess/AutoExposure", "autoExposureReduceMain", 2},
            {"Features/PostProcess/AutoExposure", "autoExposureApplyMain", 2},
            {"Features/PostProcess/ColorGradingLUT", "composeColorGradingLUT", 3},
            {"Features/Lighting/ClusterLightGrid", "clusterLightGridMain", 4},
            {"Features/Debug/LightGridDebug", "lightGridDebugMain", 5},
            {"Features/Lighting/PrepareLightsPdf", "prepareLightsPdfMain", 6},
            {"Features/Lighting/BuildReGIR", "buildReGIRMain", 7},
            {"Features/Environment/EnvironmentLightingPrecompute", "environmentLightingPrecomputeMain", 8},
            {"Features/ReSTIR/RTXDIConfidence", "rtxdiConfidenceMain", 9},
            {"Features/ReSTIR/RTXDIComposite", "rtxdiCompositeMain", 10},
            {"Features/PathTracing/SceneSharcMaintenance", "sharcClearMain", 11},
            {"Features/PathTracing/SceneSharcMaintenance", "sharcResolveMain", 11},
            {"Features/PostProcess/ScenePathTraceTonemap", "scenePathTraceTonemapMain", 12},
        };
        for (auto mode : {SlangDescriptorHeapMode::Mapped, SlangDescriptorHeapMode::Native}) {
            for (const auto& program : programs) {
                if ((program.layout >= 11 ? 3u : program.layout >= 9 ? 2u : program.layout >= 4 ? 1u : 0u) != category) { continue; }
                const SlangMacroDefine defines[] = {{"FINAL_USE_LUT", "1"}};
                std::string log;
                auto shader = compileSlangShaderToSpirv({.moduleName = program.module, .entryPointName = program.entry,
                    .searchPath = PROJECT_SOURCE_DIR "/Shaders", .macroDefines = defines, .descriptorHeapMode = mode}, log);
                if (!shader) { return RHITestResult::fail(log); }
                // Raw Load<T> removes storage decorations from the value type.
                // Keep compiling the real LUT entry above; use the exact shared
                // type in a storage-layout probe, plus the LUT pixel GPU test.
                if (program.layout == 3) {
                    shader = compileSlangShaderToSpirv({.moduleName = "ColorGradingParameterLayout",
                        .entryPointName = "main", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                        .descriptorHeapMode = mode}, log);
                    if (!shader) { return RHITestResult::fail(log); }
                }
                const auto& layout = layouts[program.layout];
                std::vector<uint32_t> ids;
                std::map<uint32_t, std::map<uint32_t, uint32_t>> offsets;
                const auto& words = shader->spirv;
                for (size_t i = 5; i < words.size();) {
                    const uint32_t count = words[i] >> 16, opcode = words[i] & 0xffff;
                    REG_CHECK(count && count <= words.size() - i);
                    if (opcode == 5 && count >= 3) { // OpName
                        const char* bytes = reinterpret_cast<const char*>(&words[i + 2]);
                        const size_t capacity = (count - 2) * sizeof(uint32_t);
                        const void* terminator = std::memchr(bytes, 0, capacity);
                        REG_CHECK(terminator);
                        const std::string_view name(bytes, static_cast<const char*>(terminator) - bytes);
                        if (name == layout.name || name.starts_with(std::string(layout.name) + "_")) {
                            ids.push_back(words[i + 1]);
                        }
                    } else if (opcode == 72 && count == 5 && words[i + 3] == 35) { // OpMemberDecorate Offset
                        offsets[words[i + 1]][words[i + 2]] = words[i + 4];
                    }
                    i += count;
                }
                bool matched = false;
                for (const auto id : ids) {
                    const auto& members = offsets[id];
                    if (members.size() != layout.offsets.size()) { continue; }
                    bool correct = true;
                    for (uint32_t i = 0; i < layout.offsets.size(); ++i) {
                        const auto member = members.find(i);
                        correct &= member != members.end() && member->second == layout.offsets[i];
                    }
                    matched |= correct;
                }
                if (!matched) { return RHITestResult::fail(std::string(program.entry) + ": C++/SPIR-V parameter offsets disagree"); }
                bool sharedHeader = false;
                for (const auto& dependency : shader->dependencies) {
                    sharedHeader |= dependency.ends_with(category == 3 ? "PathTraceStageParameters.h" : category == 2 ? "RTXDIPostProcessParameters.h" : category == 1 ? "LightingKernelParameters.h" : "PostProcessParameters.h");
                }
                REG_CHECK(sharedHeader); // Layout edits must invalidate the shader cache.
            }
        }
        return RHITestResult::pass("Mapped/native offsets and shared header cache dependency");
    }
};
METALLIC_REGISTER_RHI_TEST(PostProcessParameterLayoutTest);

class LightingParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    LightingParameterLayoutTest() { category = 1; name = "lighting_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(LightingParameterLayoutTest);

class RTXDIParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    RTXDIParameterLayoutTest() { category = 2; name = "rtxdi_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(RTXDIParameterLayoutTest);

class PathTraceStageParameterLayoutTest final : public PostProcessParameterLayoutTest {
public:
    PathTraceStageParameterLayoutTest() { category = 3; name = "path_trace_stage_parameter_spirv_layout"; }
};
METALLIC_REGISTER_RHI_TEST(PathTraceStageParameterLayoutTest);

class SharcTypedMaintenanceTest final : public RHITest {
public:
    SharcTypedMaintenanceTest() { type = RHITestType::Resource; name = "sharc_typed_maintenance_bounds_and_eviction"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "SHaRC typed stages",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(QueueType::Graphics);
        auto registry = device->resourceRegistry();
        REG_CHECK(registry);
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        // No legacy cacheParams buffer is provided. The inline entry count and
        // stale threshold must control both kernels, including the tail lanes.
        for (uint32_t testCase = 0; testCase < 3; ++testCase) {
            const bool clear = testCase == 0;
            ComputeKernel kernel;
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "Features/PathTracing/SceneSharcMaintenance",
                .entryPointName = clear ? "sharcClearMain" : "sharcResolveMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
            if (!shader) { return RHITestResult::fail(log); }
            REG_REQUIRE(kernel.initialize(*device, {.spirv = shader->spirv,
                .parameters = parameterAbi<SharcMaintenanceParams>(kSharcMaintenanceABI, ParameterTransport::InlinePush)}, log));
            std::array<std::unique_ptr<Buffer>, 3> buffers;
            std::array<std::vector<uint32_t>, 3> initial;
            for (uint32_t b = 0; b < 3; ++b) {
                const uint32_t words = b == 0 ? 2 : 4;
                initial[b].resize(32 * words, clear ? 0xffffffffu : 0u);
                if (!clear) {
                    for (uint32_t entry = 0; entry < 32; ++entry) {
                        if (b == 0) { initial[b][entry * words] = 1; }
                        // SharcPackedData.sampleData: seven stale frames.
                        if (b == 2) { initial[b][entry * words + 2] = 7u << 16; }
                    }
                }
                REG_REQUIRE(device->createBuffer({.size = initial[b].size() * sizeof(uint32_t),
                    .structureStride = words * sizeof(uint32_t), .usage = BufferUsageBits::Storage,
                    .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto value) { buffers[b] = std::move(value); }));
                void* mapped = buffers[b]->map();
                REG_CHECK(mapped);
                std::memcpy(mapped, initial[b].data(), initial[b].size() * sizeof(uint32_t));
                buffers[b]->flush();
                buffers[b]->unmap();
            }
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            REG_REQUIRE(recording.begin(testCase));
            {
                ParameterWriter writer(*device, **registry, &recording.frame);
                SharcMaintenanceParams params{};
                params.hashEntries = writer.buffer(buffers[0].get());
                params.accumulation = writer.buffer(buffers[1].get());
                params.resolved = writer.buffer(buffers[2].get());
                params.settings.entriesNum = 17;
                params.settings.sceneScale = 1.0f;
                params.settings.accumulationFrameNum = 20;
                params.settings.staleFrameNumMax = testCase == 1 ? 8 : 9;
                auto encoded = writer.encode(params, kSharcMaintenanceABI, ParameterTransport::InlinePush);
                REG_CHECK(encoded);
                REG_REQUIRE(kernel.dispatch(*recording.commands, *encoded, 1));
            }
            const MemoryBarrierDesc hostRead{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                {PipelineStageBits::Host, AccessBits::HostRead}};
            REG_REQUIRE(recording.commands->synchronize({.memory = {&hostRead, 1}}));
            std::unique_ptr<Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(recording.submit(tracker, *gate));
            kernel.clear(); // Submission owns the pipeline and immutable parameters.
            REG_REQUIRE(gate->signal(1));
            REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
            for (uint32_t b = 0; b < 3; ++b) {
                const uint32_t words = b == 0 ? 2 : 4;
                auto expected = initial[b];
                for (uint32_t entry = 0; entry < 17; ++entry) {
                    if (clear || testCase == 1) {
                        for (uint32_t word = 0; word < words; ++word) { expected[entry * words + word] = 0; }
                    } else if (b == 2) { expected[entry * words + 2] = (8u << 16) | 1u; }
                }
                buffers[b]->invalidate();
                void* mapped = buffers[b]->map();
                REG_CHECK(mapped);
                const bool equal = std::memcmp(mapped, expected.data(), expected.size() * sizeof(uint32_t)) == 0;
                buffers[b]->unmap();
                REG_CHECK(equal);
            }
        }
        REG_CHECK((*registry)->stats().parameterBytes == 0);
        return RHITestResult::pass("Clear/resolve respect inline bounds; thresholds 8/9 evict/retain; no parameter buffer upload");
    }
};
METALLIC_REGISTER_RHI_TEST(SharcTypedMaintenanceTest);

class RegistryIdentityTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"registry.identity.capacity.contract", "registry.descriptor.recycle.contract"}, bench::Layer::Core, "binding", "binding");
    }

    RegistryIdentityTest() { type = RHITestType::Resource; name = "registry_identity_capacity_and_views"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry identity", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        render::ResourceRegistry registry;
        REG_REQUIRE(registry.initialize(*device, {.maxSamplers = 2, .maxSampledImages = 2,
            .maxStorageImages = 1, .maxBuffers = 2}));
        std::unique_ptr<render::Buffer> a, b, c;
        REG_REQUIRE(makeBuffer(*device, a));
        REG_REQUIRE(makeBuffer(*device, b));
        REG_REQUIRE(makeBuffer(*device, c));
        render::ResourceLease aLease, bLease, duplicate, cLease;
        REG_REQUIRE(registry.storageBuffer(*a).transform([&](auto value) { aLease = std::move(value); }));
        REG_REQUIRE(registry.storageBuffer(*b).transform([&](auto value) { bLease = std::move(value); }));
        REG_REQUIRE(registry.storageBuffer(*b).transform([&](auto value) { duplicate = std::move(value); }));
        REG_CHECK(bLease.shaderValue() == duplicate.shaderValue());
        REG_CHECK(registry.stats().descriptorWrites == 2 && registry.stats().cacheHits == 1);
        std::weak_ptr<void> oldAllocation = a->retainAllocation();
        a.reset();
        REG_CHECK(!oldAllocation.expired());
        const auto exhausted = registry.storageBuffer(*c);
        REG_CHECK(render::hasError(exhausted, render::Error::OutOfMemory));
        REG_CHECK(!exhausted.has_value());
        const auto releasedIndex = aLease.shaderValue();
        aLease = {};
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(registry.storageBuffer(*c).transform([&](auto value) { cLease = std::move(value); }));
        REG_CHECK(cLease.shaderValue() == releasedIndex);
        REG_CHECK(registry.stats().liveDescriptors == 2);

        std::unique_ptr<render::Texture> texture;
        std::unique_ptr<render::TextureView> first, second;
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::RGBA8Unorm}).transform([&](auto rhiValue) { texture = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto rhiValue) { first = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*texture, {.format = render::Format::RGBA8Unorm}).transform([&](auto rhiValue) { second = std::move(rhiValue); }));
        render::ResourceLease imageA, imageB, generalImage, storageImage;
        bool written = false;
        REG_REQUIRE(registry.sampledImage(*first, render::ResourceState::ShaderRead, &written).transform([&](auto value) { imageA = std::move(value); }));
        REG_CHECK(written);
        REG_REQUIRE(registry.sampledImage(*second, render::ResourceState::ShaderRead, &written).transform([&](auto value) { imageB = std::move(value); }));
        REG_CHECK(!written);
        REG_CHECK(imageA.shaderValue() == imageB.shaderValue());
        REG_REQUIRE(registry.sampledImage(*second, render::ResourceState::General).transform([&](auto value) { generalImage = std::move(value); }));
        REG_CHECK(generalImage.shaderValue() != imageA.shaderValue());
        REG_REQUIRE(registry.storageImage(*second).transform([&](auto value) { storageImage = std::move(value); }));
        REG_CHECK(storageImage.kind() == render::ShaderResourceKind::StorageImage);
        REG_CHECK(!first->hasNativeView() && !second->hasNativeView());
        std::weak_ptr<void> textureAllocation = first->retainTexture();
        texture.reset(); first.reset(); second.reset();
        REG_CHECK(!textureAllocation.expired());
        imageA = {}; imageB = {}; generalImage = {}; storageImage = {};
        REG_CHECK(textureAllocation.expired());
        registry.collect();
        REG_CHECK(registry.stats().liveDescriptors == 2);
        render::ResourceLease samplerA, samplerB;
        REG_REQUIRE(registry.sampler({}).transform([&](auto value) { samplerA = std::move(value); }));
        REG_REQUIRE(registry.sampler({}).transform([&](auto value) { samplerB = std::move(value); }));
        REG_CHECK(samplerA.shaderValue() == samplerB.shaderValue());

        render::ResourceRegistry other;
        REG_REQUIRE(other.initialize(*device, {.maxSamplers = 1, .maxSampledImages = 1,
            .maxStorageImages = 1, .maxBuffers = 1}));
        REG_CHECK(registry.owns(bLease) && !other.owns(bLease) && !registry.owns({}));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::ParameterWriter writer(*device, frame, other);
        REG_CHECK(render::hasError(writer.use(bLease), render::Error::InvalidArgument));
        const auto invalid = writer.encode(ProbeParams{}, kABI);
        REG_CHECK(render::hasError(invalid, render::Error::InvalidArgument));
        render::ParameterWriter exhaustedRoot(*device, frame, registry);
        REG_CHECK(render::hasError(exhaustedRoot.encode(ProbeParams{}, kABI), render::Error::OutOfMemory));
        render::ParameterWriter staleWriter(*device, frame, registry);
        const auto encoded = staleWriter.encode(ProbeParams{}, kABI, render::ParameterTransport::InlinePush);
        REG_CHECK(encoded && encoded->valid());
        frame.cancel();
        REG_REQUIRE(frame.begin(1));
        const auto stale = staleWriter.encode(ProbeParams{}, kABI, render::ParameterTransport::InlinePush);
        REG_CHECK(render::hasError(stale, render::Error::InvalidArgument));
        REG_CHECK(encoded->valid()); // A failed encode cannot overwrite an earlier packet.
        frame.cancel();
        return RHITestResult::pass();
    }
};

class RegistrySubmissionTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.submission.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    explicit RegistrySubmissionTest(render::ParameterTransport transport = render::ParameterTransport::DescriptorBuffer)
        : transport_(transport)
    {
        type = RHITestType::Command;
        name = transport == render::ParameterTransport::InlinePush
            ? "registry_inline_submission_lifetime" : "registry_typed_submission_lifetime";
    }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry lifetime", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry, sameRegistry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { sameRegistry = std::move(rhiValue); }));
        REG_CHECK(registry == sameRegistry);
        render::ComputeKernel firstKernel, secondKernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, firstKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        REG_REQUIRE(makeKernel(*device, secondKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 11));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> oldAllocation = source->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands first, second;
        REG_REQUIRE(first.initialize(*device, queue));
        REG_REQUIRE(second.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(first.begin(0));
        render::EncodedParameters stale;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 100, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            if (transport_ == render::ParameterTransport::InlinePush) {
                REG_CHECK(encoded.root().count == 0 && encoded.inlineData().size() == sizeof(params));
                REG_CHECK(registry->stats().parameterBytes == 0 && registry->stats().parameterCapacity == 0);
            }
            stale = encoded;
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            params.add = 200; params.index = 1;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            render::BufferBarrierDesc barrier{
                .buffer = output.get(),
                .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = first.commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(secondKernel.dispatch(*first.commands, encoded, 1));
            // Force the parameter arena to grow without moving already encoded roots.
            std::array<uint32_t, 17000> burst{};
            render::EncodedParameters oversized;
            REG_REQUIRE(writer.encode(burst, kABI + 2).transform([&](auto value) { oversized = std::move(value); }));
            REG_CHECK(oversized.root() != stale.root());
            params.add = 999; // Encoded packets must not reference this mutable CPU struct.
            render::EncodedParameters wrong;
            REG_REQUIRE(writer.encode(params, kABI + 1, transport_).transform([&](auto value) { wrong = std::move(value); }));
            REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, wrong, 1), render::Error::InvalidArgument));
        }
        source.reset();
        REG_REQUIRE(makeBuffer(*device, source, 22));
        REG_REQUIRE(first.submit(tracker, *gate));
        REG_CHECK(!first.frame.completion().isComplete());
        REG_CHECK(!oldAllocation.expired());
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, stale, 1), render::Error::InvalidArgument));
        REG_REQUIRE(second.begin(1));
        REG_CHECK(render::hasError(secondKernel.dispatch(*second.commands, stale, 1), render::Error::InvalidArgument));
        {
            render::ParameterWriter writer(*device, second.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 300, 2};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            REG_CHECK(transport_ == render::ParameterTransport::InlinePush || encoded.root() != stale.root());
            REG_REQUIRE(secondKernel.dispatch(*second.commands, encoded, 1));
        }
        REG_REQUIRE(second.submit(tracker, *gate));
        firstKernel.clear(); secondKernel.clear(); stale = {};
        REG_CHECK(!oldAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(second.frame.wait(5'000'000'000ull));
        output->invalidate();
        std::array<uint32_t, 3> values{};
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::memcpy(values.data(), mapped, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{111, 211, 322}));
        REG_REQUIRE(first.pool->reset());
        REG_REQUIRE(first.frame.reset());
        REG_CHECK(oldAllocation.expired());
        REG_REQUIRE(second.pool->reset());
        REG_REQUIRE(second.frame.reset());
        registry->collect();
        // Completed frame arenas retain one descriptor per backing chunk.
        const uint64_t arenaDescriptors = transport_ == render::ParameterTransport::InlinePush ? 1 : 3;
        REG_CHECK(registry->stats().liveDescriptors == 2 + arenaDescriptors);
        const uint64_t capacity = registry->stats().parameterCapacity;

        REG_REQUIRE(makeKernel(*device, firstKernel, log, render::SlangDescriptorHeapMode::Default, transport_));
        REG_REQUIRE(first.begin(2));
        std::weak_ptr<void> cancelledAllocation = source->retainAllocation();
        render::EncodedParameters cancelled;
        {
            render::ParameterWriter writer(*device, first.frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI, transport_).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(firstKernel.dispatch(*first.commands, encoded, 1));
            cancelled = encoded;
        }
        source.reset();
        REG_CHECK(!cancelledAllocation.expired());
        REG_REQUIRE(first.commands->end());
        // A frame can still be recording while this command buffer has ended.
        REG_CHECK(render::hasError(firstKernel.dispatch(*first.commands, cancelled, 1), render::Error::InvalidArgument));
        cancelled = {};
        REG_REQUIRE(first.pool->reset());
        first.frame.cancel();
        REG_CHECK(cancelledAllocation.expired());
        REG_CHECK(registry->stats().parameterCapacity == capacity);
        registry->collect();
        REG_CHECK(registry->stats().liveDescriptors == 1 + arenaDescriptors);
        return RHITestResult::pass();
    }
private:
    render::ParameterTransport transport_;
};

class RegistryInlineSubmissionTest final : public RegistrySubmissionTest {
public:
    RegistryInlineSubmissionTest() : RegistrySubmissionTest(render::ParameterTransport::InlinePush) {}
};

METALLIC_REGISTER_RHI_TEST(RegistryIdentityTest);
METALLIC_REGISTER_RHI_TEST(RegistrySubmissionTest);
METALLIC_REGISTER_RHI_TEST(RegistryInlineSubmissionTest);

class ResourceABIProbeTest final : public RHITest {
public:
    ResourceABIProbeTest() { type = RHITestType::Resource; name = "resource_abi_spans_nonuniform_and_lifetime"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"resources.abi32.spans.nonuniform.lifetime"}, bench::Layer::Core,
            "binding", "binding", {"readback.bin"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        struct Record { uint32_t first, values[3]; };
        struct Params { GPUBufferSpan sources[2], records, output; };
        static_assert(sizeof(Params) == 48 && offsetof(Params, output) == 36);
        constexpr uint64_t abi = 0x44525350414e0001ull;
        std::atomic_uint validationErrors{0};
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "DR resource ABI",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                // Loader manifest failures are GENERAL environment messages;
                // count API validation errors generated by this workload.
                if ((message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                    (message.type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                    ++*static_cast<std::atomic_uint*>(target);
                }
            }, &validationErrors}})
            .transform([&](auto value) { device = std::move(value); }));
        ResourceRegistry registry;
        REG_REQUIRE(registry.initialize(*device));
        std::unique_ptr<Buffer> sources[2], output;
        for (uint32_t i = 0; i < 2; ++i) {
            REG_REQUIRE(makeBuffer(*device, sources[i]));
            auto* mapped = static_cast<uint32_t*>(sources[i]->map());
            REG_CHECK(mapped != nullptr);
            for (uint32_t j = 0; j < 16; ++j) { mapped[j] = 10 + i * 90 + j; }
            sources[i]->flush(); sources[i]->unmap();
        }
        REG_REQUIRE(makeBuffer(*device, output));
        auto* mapped = static_cast<uint32_t*>(output->map());
        REG_CHECK(mapped != nullptr);
        for (uint32_t i = 0; i < 16; ++i) { mapped[i] = 0xdeadbeefu; }
        output->flush(); output->unmap();
        // Bad ranges fail before allocating descriptors, and cannot publish a packet.
        for (const auto range : {BufferRange{0, 0}, BufferRange{1, 4}, BufferRange{60, 8}, BufferRange{0, 6}}) {
            ParameterWriter invalid(*device, registry);
            (void)invalid.bufferSpan<uint32_t>(sources[0].get(), range);
            REG_CHECK(hasError(invalid.status(), Error::InvalidArgument));
            REG_CHECK(hasError(invalid.encode(Params{}, abi), Error::InvalidArgument));
        }
        REG_CHECK(registry.stats().descriptorWrites == 0);
        ShaderCompileResult shader;
        REG_REQUIRE(compileSlangShaderToSpirv({.moduleName = "ResourceABIProbe",
            .entryPointName = "resourceABIProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
            shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
        ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(kernel.initialize(*device, {.spirv = shader.spirv,
            .parameters = parameterAbi<Params>(abi, ParameterTransport::InlinePush)}, log));
        std::weak_ptr<void> allocation = sources[0]->retainAllocation();
        {
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            REG_REQUIRE(gpu.initialize(*device));
            {
                ParameterWriter writer(*device, registry);
                Params params{{writer.bufferSpan<uint32_t>(sources[0].get(), {4, 16}),
                    writer.bufferSpan<uint32_t>(sources[1].get(), {20, 16})},
                    writer.bufferSpan<Record>(sources[0].get(), {16, 16}),
                    writer.bufferSpan<uint32_t>(output.get(), {4, 40})};
                REG_CHECK(params.sources[0].resource.index == params.records.resource.index);
                REG_CHECK(params.sources[0].resource.index != params.sources[1].resource.index);
                REG_CHECK(registry.stats().descriptorWrites == 3);
                auto encoded = writer.encode(params, abi, ParameterTransport::InlinePush);
                REG_CHECK(encoded.has_value());
                REG_REQUIRE(kernel.dispatch(*gpu.commands, *encoded, 1));
            }
            sources[0].reset(); sources[1].reset(); kernel.clear();
            REG_CHECK(!allocation.expired());
            REG_REQUIRE(gpu.submitAndWait());
            output->invalidate();
            auto* values = static_cast<uint32_t*>(output->map());
            REG_CHECK(values != nullptr);
            std::array<uint32_t, 16> actual{};
            std::memcpy(actual.data(), values, sizeof(actual)); output->unmap();
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
            const std::array<uint32_t, 10> expected{11, 105, 12, 106, 13, 107, 14, 108, 0, 31};
            REG_CHECK(std::equal(expected.begin(), expected.end(), actual.begin() + 1));
            REG_CHECK(actual[0] == 0xdeadbeefu);
            for (uint32_t i = 11; i < 16; ++i) { REG_CHECK(actual[i] == 0xdeadbeefu); }
        }
        REG_CHECK(allocation.expired());
        registry.collect();
        REG_CHECK(registry.stats().liveDescriptors == 1);
        REG_CHECK(validationErrors == 0);
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ResourceABIProbeTest);

class RegistryPipelinedParametersTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.pipelined.append.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    RegistryPipelinedParametersTest() { type = RHITestType::Command; name = "registry_pipelined_parameter_append"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Pipelined parameters", .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            std::shared_ptr<render::ResourceRegistry> registry;
            REG_REQUIRE(device->resourceRegistry().transform([&](auto value) { registry = std::move(value); }));
            render::ComputeKernel kernel;
            std::string log;
            REG_REQUIRE(makeKernel(*device, kernel, log, mode));
            std::unique_ptr<render::Buffer> source, output;
            REG_REQUIRE(makeBuffer(*device, source, 11));
            REG_REQUIRE(makeBuffer(*device, output));
            render::RenderFrameContext frame;
            std::array<render::CommandRecordingContext, 3> recordings;
            render::QueueSubmissionTracker tracker;
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(tracker.initialize(*device, queue));
            REG_REQUIRE(frame.begin(0, UINT64_MAX, render::FrameSubmissionMode::Pipelined));
            render::ParameterWriter writer(*device, frame, *registry);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 0, 0};
            std::array<render::EncodedParameters, 3> packets;
            for (uint32_t i = 0; i < 3; ++i) {
                // Iteration 1 appends while batch 0 is pending behind a gate;
                // iteration 2 appends after prior batches complete, frame open.
                params.add = (i + 1) * 100;
                params.index = i;
                REG_REQUIRE(writer.encode(params, kABI).transform([&](auto value) { packets[i] = std::move(value); }));
                if (i) { REG_CHECK(packets[i].root().resource == packets[i - 1].root().resource &&
                    packets[i].root().byteOffset > packets[i - 1].root().byteOffset); }
                REG_REQUIRE(recordings[i].initialize(*device, queue));
                render::CommandBuffer* commands = nullptr;
                REG_REQUIRE(recordings[i].prepare(frame).transform([&](auto value) { commands = value; }));
                REG_REQUIRE(recordings[i].record([&]() -> render::Result<> {
                    render::BufferBarrierDesc barrier{
                        .buffer = output.get(),
                        .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                        .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                    };
                    if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
                    auto result = kernel.dispatch(*commands, packets[i], 1);
                    // Re-read the very first packet after additional uploads.
                    if (result && i == 2) {
                        if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
                        result = kernel.dispatch(*commands, packets[0], 1);
                    }
                    return result ? commands->end() : result;
                }));
                render::RecordedBatch batch;
                REG_REQUIRE(batch.seal(frame, {&commands, 1}));
                render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
                render::SubmissionReceipt receipt;
                REG_REQUIRE(tracker.submitBatch(batch, {
                    .waitSemaphores = {i == 0 ? &wait : nullptr, i == 0 ? 1u : 0u},
                }, frame).transform([&](auto value) { receipt = std::move(value); }));
                REG_CHECK(receipt.accepted() && frame.recording() && !frame.completion().isSubmitted());
                if (i == 0) { REG_CHECK(!receipt.completion().isComplete()); }
                if (i == 1) {
                    REG_REQUIRE(gate->signal(1));
                    REG_REQUIRE(receipt.completion().wait(5'000'000'000ull));
                    REG_CHECK(!frame.completion().isComplete());
                }
            }
            REG_REQUIRE(frame.sealRecording());
            render::EncodedParameters rejected;
            REG_CHECK(!writer.encode(params, kABI).transform([&](auto value) { rejected = std::move(value); }));
            REG_REQUIRE(frame.finishSubmission());
            REG_REQUIRE(frame.wait(5'000'000'000ull));
            output->invalidate();
            auto* mapped = output->map();
            REG_CHECK(mapped);
            std::array<uint32_t, 3> values;
            std::memcpy(values.data(), mapped, sizeof(values));
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
            output->unmap();
            REG_CHECK((values == std::array<uint32_t, 3>{111, 211, 311}));
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryPipelinedParametersTest);

class RegistryPartialSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        auto metadata = bench::gpuMetadata({"parameters.partialSubmission.retention.contract"}, bench::Layer::Core, "async", "sync");
        metadata.requirements.queues.push_back(render::QueueType::Copy);
        return metadata;
    }

    RegistryPartialSubmissionTest() { type = RHITestType::Command; name = "registry_partial_multi_queue_retention"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry partial submission",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& graphics = *device->getQueue(render::QueueType::Graphics);
        auto* copy = device->getQueue(render::QueueType::Copy);
        if (!copy) { return RHITestResult::skip("Requires a copy queue"); }
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 17));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        auto owner = std::make_shared<uint32_t>(19);
        std::weak_ptr<void> transitiveOwner = owner;
        render::QueueSubmissionTracker graphicsTracker, copyTracker;
        REG_REQUIRE(graphicsTracker.initialize(*device, graphics));
        REG_REQUIRE(copyTracker.initialize(*device, *copy));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, graphics));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{*copy, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            writer.retain(owner);
            ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), 1, 0};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(kernel.dispatch(*recording.commands, encoded, 1));
        }
        REG_REQUIRE(recording.commands->end());
        source.reset(); owner.reset(); kernel.clear();
        render::CommandBuffer* buffers[] = {recording.commands.get()};
        render::GPUCompletionPoint graphicsDone, copyDone, rejected;
        REG_REQUIRE(graphicsTracker.submitSegment({.commandBuffers = {buffers, 1}}, recording.frame).transform([&](auto value) { graphicsDone = std::move(value); }));
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        REG_REQUIRE(copyTracker.submitSegment({.waitSemaphores = {&wait, 1}}, recording.frame).transform([&](auto value) { copyDone = std::move(value); }));
        REG_CHECK(!copyTracker.submitSegment({.commandBuffers = std::array<render::CommandBuffer*, 1>{nullptr}}, recording.frame).transform([&](auto value) { rejected = std::move(value); }));
        recording.frame.cancel(); // Must seal accepted segments, not release their packets.
        REG_REQUIRE(graphicsDone.wait(5'000'000'000ull));
        registry->collect();
        REG_CHECK(!recording.frame.completion().isComplete() && !copyDone.isComplete());
        REG_CHECK(!allocation.expired() && !transitiveOwner.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired() && transitiveOwner.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryPartialSubmissionTest);

class RegistryTextureSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"binding.texture.array.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    RegistryTextureSubmissionTest() { type = RHITestType::Rendering; name = "registry_texture_array_submission_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Registry texture array",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        struct Params { render::ShaderStorageImage image; render::GPUBufferSpan samples; render::ShaderBuffer output; };
        static_assert(sizeof(Params) == 20);
        const char* entries[] = {"registryTextureWriteMain", "registryTextureReadMain"};
        std::array<render::ComputeKernel, 2> kernels;
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "RegistryTextureProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kABI + 3)}, log));
        }
        std::unique_ptr<render::Texture> image;
        std::unique_ptr<render::TextureView> view;
        std::unique_ptr<render::Buffer> output;
        REG_REQUIRE(makeBuffer(*device, output));
        REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
            .format = render::Format::R32Uint}).transform([&](auto rhiValue) { image = std::move(rhiValue); }));
        REG_REQUIRE(device->createTextureView(*image, {}).transform([&](auto rhiValue) { view = std::move(rhiValue); }));
        std::weak_ptr<void> allocation = view->retainTexture();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const std::array<render::TextureView*, 3> views{view.get(), view.get(), view.get()};
            Params params{writer.storageImage(view.get()), writer.sampledImages(views), writer.buffer(output.get())};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI + 3).transform([&](auto value) { encoded = std::move(value); }));
            REG_CHECK(registry->stats().descriptorWrites == 4); // storage image, sampled image, output, array upload
            render::TextureBarrierDesc barrier{
                .texture = image.get(),
                .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::General,
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            barrier.oldLayout = render::TextureLayout::General; barrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite}; barrier.newLayout = render::TextureLayout::ShaderRead; barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
            if (auto commandResult = recording.commands->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[1].dispatch(*recording.commands, encoded, 1));
        }
        image.reset(); view.reset();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {};
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        void* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 3> values;
        std::memcpy(values.data(), mapped, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        REG_CHECK((values == std::array<uint32_t, 3>{1001, 1001, 1001}));
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(allocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(RegistryTextureSubmissionTest);
// Native provenance and narrowing are checked without creating descriptors.
class BufferSliceValidationTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"bufferSlice.range.provenance.contract"}, bench::Layer::Core, "binding", "binding");
    }

    BufferSliceValidationTest() { type = RHITestType::Resource; name = "buffer_slice_range_and_provenance"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device, other;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice ranges",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice foreign source",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, true).transform([&](auto rhiValue) { other = std::move(rhiValue); }));
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(*device, buffer));
        render::BufferSlice parent, child, invalid, empty;
        REG_REQUIRE(buffer->slice({16, 32}).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_REQUIRE(parent.subslice({8, 8}).transform([&](auto rhiValue) { child = std::move(rhiValue); }));
        REG_CHECK(child.offset() == 24 && child.size() == 8);
        REG_CHECK(child.deviceAddress() == buffer->deviceAddress() + 24);
        REG_CHECK(child.deviceIdentity() == device->identity());
        REG_CHECK(child.allocationIdentity() == buffer->retainAllocation().get());
        REG_REQUIRE(child.validateData(device->identity(), 4, 4));
        REG_CHECK(render::hasError(child.validateData(other->identity(), 4, 4), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice({32}).transform([&](auto value) { empty = std::move(value); }));
        REG_CHECK(empty.valid() && empty.size() == 0);
        REG_CHECK(!empty.validateData(device->identity(), 4, 4));
        for (uint64_t offset : {uint64_t(33), UINT64_MAX}) {
            const auto rejected = parent.subslice({offset});
            REG_CHECK(render::hasError(rejected, render::Error::InvalidArgument));
            REG_CHECK(parent.offset() == 16 && parent.size() == 32);
        }
        REG_CHECK(render::hasError(parent.subslice({0, 33}), render::Error::InvalidArgument));
        REG_CHECK(render::hasError(parent.subslice({31, UINT64_MAX - 1}), render::Error::InvalidArgument));
        REG_REQUIRE(parent.subslice({1, 8}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 4, 4));
        REG_REQUIRE(parent.subslice({0, 12}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
        REG_CHECK(!invalid.validateData(device->identity(), 8, 4));
        REG_CHECK(!child.validateData(device->identity(), 0, 4));
        REG_CHECK(!child.validateData(device->identity(), 4, 0));
        REG_CHECK(!child.validateData(device->identity(), 4, 3));
        REG_CHECK(!child.validateData(device->identity(), 3, 4));
        REG_CHECK(!child.validate(device->identity(), render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource));
        REG_REQUIRE(parent.subslice({8, 8}).transform([&](auto rhiValue) { parent = std::move(rhiValue); }));
        REG_CHECK(parent.offset() == child.offset() && parent.size() == child.size());

        std::weak_ptr<void> allocation = buffer->retainAllocation();
        const auto address = child.deviceAddress();
        render::Buffer moved = std::move(*buffer);
        REG_CHECK(!buffer->retainAllocation());
        REG_REQUIRE(makeBuffer(*device, buffer, 99));
        moved = {};
        REG_CHECK(!allocation.expired() && child.deviceAddress() == address);
        parent = {}; invalid = {}; empty = {};
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        render::RenderFrameContext frame;
        REG_REQUIRE(frame.begin(0));
        render::EncodedParameters packet;
        {
            render::ParameterWriter invalidWriter(*other, frame, *registry);
            invalidWriter.bufferSpan<uint32_t>(child);
            REG_CHECK(!invalidWriter.status());
            REG_CHECK(!invalidWriter.encode(render::GPUBufferSpan{}, kABI + 4).transform([&](auto value) { packet = std::move(value); }) && !packet.valid());
            render::ParameterWriter writer(*device, frame, *registry);
            const auto data = writer.bufferSpan<uint32_t>(child);
            REG_CHECK(data.resource.index != UINT32_MAX && data.count == 2 && data.byteOffset == 24);
            REG_REQUIRE(writer.encode(data, kABI + 4).transform([&](auto value) { packet = std::move(value); }));
        }
        child = {};
        REG_CHECK(!allocation.expired());
        packet = {};
        frame.cancel();
        REG_REQUIRE(frame.reset());
        REG_CHECK(allocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 2);
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceValidationTest);

class BufferSliceSubmissionTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"bufferSlice.dr.copy.indirect.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    BufferSliceSubmissionTest() { type = RHITestType::Rendering; name = "buffer_slice_dr_copy_compute_indirect_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Buffer slice data chain",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); }));
        struct Params { render::GPUBufferSpan source, output, arguments; uint32_t add; };
        static_assert(sizeof(Params) == 40 && offsetof(Params, add) == 36);
        std::array<render::ComputeKernel, 2> kernels;
        const char* entries[] = {"dataProduceMain", "dataIndirectMain"};
        std::string log;
        for (size_t i = 0; i < kernels.size(); ++i) {
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result) { return RHITestResult::fail(shader.diagnostics); }
            REG_REQUIRE(kernels[i].initialize(*device, {.spirv = shader.spirv,
                .parameters = render::parameterAbi<Params>(kABI + 5)}, log));
        }
        render::ShaderCompileResult shader;
        REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "DataSliceProbe",
            .entryPointName = "dataAdapterMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
        render::ComputeProgram adapter;
        const render::ComputeProgramBindingDesc layout{.binding = 0,
            .kind = render::ComputeResourceBindingKind::DataBuffer, .dataStride = 4, .dataAlignment = 4};
        REG_REQUIRE(adapter.initialize(*device, {
            .spirv = shader.spirv,
            .bindings = {&layout, 1},
            .requiresRayQuery = false,
        }, log));
        std::unique_ptr<render::Buffer> source, work, output;
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::TransferSource,
            .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { source = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::Storage |
            render::BufferUsageBits::TransferDestination | render::BufferUsageBits::Indirect}).transform([&](auto rhiValue) { work = std::move(rhiValue); }));
        REG_REQUIRE(device->createBuffer({.size = 64, .usage = render::BufferUsageBits::Storage |
            render::BufferUsageBits::TransferSource | render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        auto* sourceWords = static_cast<uint32_t*>(source->map());
        REG_CHECK(sourceWords);
        for (uint32_t i = 0; i < 16; ++i) { sourceWords[i] = 100 + i; }
        source->flush(); source->unmap();
        auto* outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        for (uint32_t i = 0; i < 16; ++i) { outputWords[i] = 0xdeadbeef; }
        output->flush(); output->unmap();
        std::weak_ptr<void> sourceAllocation = source->retainAllocation(), workAllocation = work->retainAllocation();
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        Commands recording;
        REG_REQUIRE(recording.initialize(*device, queue));
        std::unique_ptr<render::Semaphore> gate;
        REG_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        Drain drain{queue, *gate};
        REG_REQUIRE(recording.begin(0));
        {
            render::BufferSlice from, data, to, arguments, invalid;
            REG_REQUIRE(source->slice({8, 16}).transform([&](auto rhiValue) { from = std::move(rhiValue); }));
            REG_REQUIRE(work->slice({16, 16}).transform([&](auto rhiValue) { data = std::move(rhiValue); }));
            REG_REQUIRE(work->slice({48, 12}).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
            REG_REQUIRE(output->slice({20, 16}).transform([&](auto rhiValue) { to = std::move(rhiValue); }));
            // Transfer-only memory is not a shader data buffer.
            REG_CHECK(!from.validateData(device->identity(), 4, 4));
            REG_REQUIRE(to.subslice({0, 12}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->copyBuffer(from, invalid));
            REG_CHECK(!recording.commands->copyBuffer(to, to));
            REG_CHECK(!recording.commands->dispatchIndirect(to));
            REG_REQUIRE(arguments.subslice({1, 8}).transform([&](auto rhiValue) { invalid = std::move(rhiValue); }));
            REG_CHECK(!recording.commands->dispatchIndirect(invalid));
            render::BufferBarrierDesc workBarrier{
                .buffer = work.get(),
                .before = {},
                .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.buffers = {&workBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(recording.commands->copyBuffer(from, data));
            workBarrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite}; workBarrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            if (auto commandResult = recording.commands->synchronize({.buffers = {&workBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::BufferBarrierDesc outputBarrier{
                .buffer = output.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = recording.commands->synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::ParameterWriter writer(*device, recording.frame, *registry);
            const Params params{writer.bufferSpan<uint32_t>(data), writer.bufferSpan<uint32_t>(to),
                writer.bufferSpan<uint32_t>(arguments), 7};
            render::EncodedParameters encoded;
            REG_REQUIRE(writer.encode(params, kABI + 5).transform([&](auto value) { encoded = std::move(value); }));
            REG_REQUIRE(kernels[0].dispatch(*recording.commands, encoded, 1));
            outputBarrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite};
            workBarrier.before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite}; workBarrier.after = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead};
            const render::BufferBarrierDesc barriers[] = {outputBarrier, workBarrier};
            if (auto commandResult = recording.commands->synchronize({.buffers = {barriers, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            REG_REQUIRE(kernels[1].dispatchIndirect(*recording.commands, encoded, arguments));
            if (auto commandResult = recording.commands->synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            render::ComputeDispatchBinding binding{.binding = 0, .data = to};
            render::ComputeDispatchDesc dispatch{.commandBuffer = recording.commands.get(), .bindings = {&binding, 1}};
            binding.range.offset = 4;
            REG_CHECK(render::hasError(adapter.dispatch(dispatch), render::Error::InvalidArgument));
            binding.range.offset = 0;
            REG_REQUIRE(adapter.dispatch(dispatch));
        }
        source.reset(); work.reset();
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_CHECK(registry->stats().descriptorWrites == 3 && registry->stats().liveDescriptors == 3);
        REG_REQUIRE(recording.submit(tracker, *gate));
        kernels = {}; adapter.clear();
        REG_CHECK(!recording.frame.completion().isComplete());
        REG_CHECK(!sourceAllocation.expired() && !workAllocation.expired());
        REG_REQUIRE(gate->signal(1));
        REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
        output->invalidate();
        outputWords = static_cast<uint32_t*>(output->map());
        REG_CHECK(outputWords);
        std::array<uint32_t, 16> values;
        std::memcpy(values.data(), outputWords, sizeof(values));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values));
        output->unmap();
        for (uint32_t i = 0; i < values.size(); ++i) {
            const auto expected = i >= 5 && i < 9 ? 221 + (i - 5) * 2 : 0xdeadbeef;
            REG_CHECK(values[i] == expected);
        }
        REG_REQUIRE(recording.pool->reset());
        REG_REQUIRE(recording.frame.reset());
        REG_CHECK(sourceAllocation.expired() && workAllocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(BufferSliceSubmissionTest);

class ResourceRangeContractTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"resource.range.shaderInput.contract"}, bench::Layer::RHI, "binding", "binding");
    }

    ResourceRangeContractTest() { type = RHITestType::Resource; name = "resource_range_and_shader_input_contract"; }
    RHITestResult run(RHITestContext& context) override
    {
        using render::BufferRange;
        using render::Error;
        const auto tail = BufferRange{16}.resolve(64);
        REG_CHECK(tail && tail->offset == 16 && tail->size == 48);
        const auto end = BufferRange{64}.resolve(64);
        REG_CHECK(end && end->size == 0);
        REG_CHECK(render::hasError(BufferRange{65, 0}.resolve(64), Error::InvalidArgument));
        REG_CHECK(render::hasError(BufferRange{16, UINT64_MAX - 1}.resolve(64), Error::InvalidArgument));
        REG_CHECK(render::hasError(BufferRange{UINT64_MAX - 2, 4}.resolve(UINT64_MAX), Error::InvalidArgument));

        auto& device = context.device;
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(device, buffer));
        auto slice = buffer->slice({16, 32});
        REG_CHECK(slice && slice->offset() == 16 && slice->size() == 32);
        auto sub = slice->subslice({8});
        REG_CHECK(sub && sub->offset() == 24 && sub->size() == 24);
        const auto empty = slice->subslice({32});
        REG_CHECK(empty && empty->size() == 0);
        REG_CHECK(render::hasError(slice->subslice({31, 2}), Error::InvalidArgument));
        REG_CHECK(render::hasError(buffer->slice({UINT64_MAX, 1}), Error::InvalidArgument));
        if (device.capabilities().bindlessDescriptorHeap) {
            auto view = device.createBufferView(*buffer, {.range = {16}});
            REG_CHECK(view && (*view)->desc().range == *tail);
            REG_CHECK(render::hasError(device.createBufferView(*buffer, {.range = {64}}), Error::InvalidArgument));
            REG_CHECK(render::hasError(device.createBufferView(*buffer, {.range = {16, UINT64_MAX - 1}}), Error::InvalidArgument));
        }

        auto texture = device.createTexture({.usage = render::TextureUsageBits::Sampled,
            .format = render::Format::RGBA8Unorm, .width = 8, .height = 8, .mipCount = 3, .layerCount = 2});
        REG_CHECK(texture);
        const render::TextureSubresourceRange range{1, 2, 1, 1};
        REG_CHECK(range.valid(3, 2));
        auto view = device.createTextureView(**texture, {.range = range});
        REG_CHECK(view && (*view)->desc().range.baseMip == 1 && (*view)->desc().range.layerCount == 1);
        for (const auto invalid : std::array<render::TextureSubresourceRange, 5>{{
                 {3, 1, 0, 1}, {0, 0, 0, 1}, {1, UINT32_MAX, 0, 1}, {0, 1, 2, 1}, {0, 1, 1, UINT32_MAX}}}) {
            REG_CHECK(!invalid.valid(3, 2));
            REG_CHECK(render::hasError(device.createTextureView(**texture, {.range = invalid}), Error::InvalidArgument));
        }

        // Word spans cannot represent misaligned byte lengths. Malformed word streams
        // must still be rejected before reaching Vulkan or the OMM transformer.
        const std::array<uint32_t, 4> truncated{0x07230203u};
        const std::array<uint32_t, 5> badMagic{};
        const std::array<uint32_t, 6> badInstruction{0x07230203u, 0x00010600u, 0, 1, 0, 0};
        REG_CHECK(render::hasError(device.createShaderModule({}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = truncated}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = badMagic}), Error::InvalidArgument));
        REG_CHECK(render::hasError(device.createShaderModule({.spirv = badInstruction}), Error::InvalidArgument));
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ResourceRangeContractTest);


class SynchronizationScopesTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"synchronization.atomic.validation.contract"}, bench::Layer::RHI, "core", "sync");
    }

    SynchronizationScopesTest() { type = RHITestType::Command; name = "synchronization_scopes_batch_and_validation"; }
    RHITestResult run(RHITestContext& context) override
    {
        using S = render::PipelineStageBits;
        using A = render::AccessBits;
        auto& device = context.device;
        Commands recording;
        REG_REQUIRE(recording.initialize(device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        auto& command = *recording.commands;
        std::array<std::unique_ptr<render::Buffer>, 3> buffers;
        std::array<render::BufferBarrierDesc, 3> barriers;
        for (uint32_t i = 0; i < buffers.size(); ++i) {
            REG_REQUIRE(makeBuffer(device, buffers[i]));
            barriers[i] = {
                .buffer = buffers[i].get(),
                .before = {S::ComputeShader, A::ShaderWrite},
                .after = {S::ComputeShader, A::ShaderRead},
            };
        }
        REG_REQUIRE(command.synchronize({.buffers = {barriers.data(), 3}}));
        auto stats = command.synchronizationStats();
        REG_CHECK(stats.calls == 1 && stats.memoryBarriers == 1 && stats.coalescedResources == 3 && stats.imageTransitions == 0);
        for (auto& barrier : barriers) { barrier.before.access = A::ShaderRead; }
        REG_REQUIRE(command.synchronize({.buffers = {barriers.data(), 3}}));
        REG_CHECK(command.synchronizationStats().calls == 2); // Explicit read/read scopes still order execution.
        std::array<render::MemoryBarrierDesc, 2> memory{{
            {{S::ComputeShader, A::ShaderWrite}, {S::DrawIndirect, A::IndirectRead}},
            {{S::Transfer, A::TransferWrite}, {S::ComputeShader, A::ShaderRead}},
        }};
        REG_REQUIRE(command.synchronize({.memory = {memory.data(), 2}}));
        REG_CHECK(command.synchronizationStats().memoryBarriers == 4); // Keep distinct stage pairs.
        const std::array<render::SyncScope, 5> invalid{{
            {S::Transfer, A::ShaderWrite}, {S::ComputeShader, A::IndirectRead},
            {S::None, A::MemoryRead}, {static_cast<S>(1ull << 63), A::None}, {S::Transfer, static_cast<A>(1ull << 63)},
        }};
        for (const auto scope : invalid) {
            memory[1].after = scope;
            REG_CHECK(render::hasError(command.synchronize({.memory = {memory.data(), 2}}), render::Error::InvalidArgument));
            REG_CHECK(command.synchronizationStats().calls == 3); // Validation is atomic.
        }
        barriers[0].range.offset = 64;
        REG_CHECK(render::hasError(command.synchronize({.buffers = {barriers.data(), 3}}), render::Error::InvalidArgument));
        REG_REQUIRE(command.end());
        REG_CHECK(render::hasError(command.synchronize({}), render::Error::InvalidArgument));
        recording.frame.cancel();
        REG_REQUIRE(recording.begin(1));
        REG_CHECK(command.synchronizationStats().calls == 0);
        recording.frame.cancel();
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationScopesTest);

// Capture encoding without submitting work. Restore this device's entry point even on
// an assertion failure; these tests execute serially in the RHI test process.
struct BarrierEncodingCapture {
    inline static BarrierEncodingCapture* active = nullptr;
    PFN_vkCmdPipelineBarrier2& entry;
    PFN_vkCmdPipelineBarrier2 original;
    uint32_t calls = 0;
    std::vector<VkMemoryBarrier2> memory;
    std::vector<VkImageMemoryBarrier2> images;
    explicit BarrierEncodingCapture(render::Device& device)
        : entry(const_cast<VolkDeviceTable*>(render::vulkan::nativeDevice(device).functions)->vkCmdPipelineBarrier2),
          original(entry)
    {
        active = this;
        entry = capture;
    }
    ~BarrierEncodingCapture()
    {
        entry = original;
        active = nullptr;
    }
    static VKAPI_ATTR void VKAPI_CALL capture(VkCommandBuffer, const VkDependencyInfo* dependency)
    {
        ++active->calls;
        active->memory.clear();
        active->images.clear();
        for (uint32_t i = 0; i < dependency->memoryBarrierCount; ++i) {
            active->memory.push_back(dependency->pMemoryBarriers[i]);
        }
        for (uint32_t i = 0; i < dependency->imageMemoryBarrierCount; ++i) {
            active->images.push_back(dependency->pImageMemoryBarriers[i]);
        }
    }
};

class SynchronizationEncodingTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"synchronization.explicitScopes.encoding"}, bench::Layer::RHI, "core", "sync");
    }

    SynchronizationEncodingTest() { type = RHITestType::Command; name = "synchronization_explicit_scopes_and_layouts"; }
    RHITestResult run(RHITestContext& context) override
    {
        using S = render::PipelineStageBits;
        using A = render::AccessBits;
        using L = render::TextureLayout;
        Commands recording;
        REG_REQUIRE(recording.initialize(context.device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        auto& command = *recording.commands;
        auto texture = context.device.createTexture({.usage = render::TextureUsageBits::Storage | render::TextureUsageBits::Sampled,
            .format = render::Format::RGBA8Unorm, .width = 4, .height = 4});
        REG_CHECK(texture);
        std::unique_ptr<render::Buffer> buffer;
        REG_REQUIRE(makeBuffer(context.device, buffer));
        BarrierEncodingCapture capture(context.device);

        render::TextureBarrierDesc image{.texture = texture->get(), .oldLayout = L::General, .newLayout = L::General};
        render::BufferBarrierDesc bytes{.buffer = buffer.get()};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}, .buffers = {&bytes, 1}}));
        REG_CHECK(capture.calls == 0); // General layout must not invent accesses.

        image.oldLayout = L::Undefined;
        image.after = {S::ComputeShader, A::ShaderWrite};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}}));
        REG_CHECK(capture.calls == 1 && capture.images.size() == 1);
        REG_CHECK(capture.images[0].srcStageMask == VK_PIPELINE_STAGE_2_NONE && capture.images[0].srcAccessMask == 0);
        REG_CHECK(capture.images[0].dstStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
            capture.images[0].dstAccessMask == VK_ACCESS_2_SHADER_WRITE_BIT);

        // A semaphore-covered producer needs an empty source scope even when
        // its old layout is defined. Unified layouts must preserve that scope.
        image.oldLayout = L::General;
        image.newLayout = L::ShaderRead;
        image.after = {S::ComputeShader, A::ShaderRead};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}}));
        if (context.device.capabilities().unifiedImageLayouts) {
            REG_CHECK(capture.memory.size() == 1 && capture.images.empty());
            REG_CHECK(capture.memory[0].srcStageMask == 0 && capture.memory[0].srcAccessMask == 0);
        } else {
            REG_CHECK(capture.images.size() == 1 && capture.memory.empty());
            REG_CHECK(capture.images[0].srcStageMask == 0 && capture.images[0].srcAccessMask == 0);
            REG_CHECK(capture.images[0].newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        image.oldLayout = image.newLayout;
        image.before = bytes.before = {S::ComputeShader, A::None};
        image.after = bytes.after = {S::FragmentShader, A::None};
        REG_REQUIRE(command.synchronize({.textures = {&image, 1}, .buffers = {&bytes, 1}}));
        REG_CHECK(capture.memory.size() == 1 && capture.images.empty());
        REG_CHECK(capture.memory[0].srcStageMask == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
            capture.memory[0].dstStageMask == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        REG_CHECK(capture.memory[0].srcAccessMask == 0 && capture.memory[0].dstAccessMask == 0);
        const auto beforeInvalid = capture.calls;
        bytes.after = {S::None, A::ShaderRead};
        REG_CHECK(render::hasError(command.synchronize({
            .textures = {&image, 1},
            .buffers = {&bytes, 1},
        }), render::Error::InvalidArgument));
        image.newLayout = static_cast<L>(255);
        REG_CHECK(render::hasError(command.synchronize({.textures = {&image, 1}}), render::Error::InvalidArgument));
        image.newLayout = L::Undefined;
        REG_CHECK(render::hasError(command.synchronize({.textures = {&image, 1}}), render::Error::InvalidArgument));
        REG_CHECK(capture.calls == beforeInvalid);
        REG_REQUIRE(command.end());
        recording.frame.cancel();
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(SynchronizationEncodingTest);

class PreparedExecutionViewsTest final : public RHITest {
public:
    PreparedExecutionViewsTest() { type = RHITestType::Rendering; name = "prepared_execution_lazy_views_layout_policy"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::comparisonMetadata({"layouts.optimal.unified.preparedViews.draw.copy"}, bench::Layer::RHI,
            "core", {"core-unified", "unifiedLayouts", bench::Capability::UnifiedLayouts});
    }
    RHITestResult run(RHITestContext& context) override
    {
        constexpr uint32_t extent = 32, bytes = extent * extent * 4;
        std::array<uint8_t, bytes> reference{};
        bool unifiedTested = false;
        std::vector<uint8_t> observations;
        const auto variants = context.deviceDesc ? std::vector<bool>{context.deviceDesc->preferUnifiedImageLayouts} : std::vector<bool>{false, true};
        for (bool preferUnified : variants) {
            std::atomic_uint errors{0};
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Prepared execution lifetime", .enableValidation = context.enableValidation,
                .validationSink = {[](void* target, const render::ValidationMessage& message) noexcept {
                    if ((message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
                        (message.type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &errors}, .preferUnifiedImageLayouts = preferUnified}).transform([&](auto value) { device = std::move(value); }));
            const bool unified = device->capabilities().unifiedImageLayouts;
            REG_CHECK(preferUnified || !unified);
            unifiedTested |= unified;
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            std::array<render::ShaderCompileResult, 2> compiled;
            std::array<std::unique_ptr<render::ShaderModule>, 2> modules;
            const char* entries[] = {"triangleVertexMain", "triangleFragmentMain"};
            for (uint32_t i = 0; i < modules.size(); ++i) {
                REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "Features/Samples/Triangle",
                    .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled[i].diagnostics).transform([&](auto value) { compiled[i] = std::move(value); }));
                REG_REQUIRE(device->createShaderModule({
                    .spirv = compiled[i].spirv,
                }).transform([&](auto value) { modules[i] = std::move(value); }));
            }
            auto foreignDevice = bench::createTestDevice(context, {.enableValidation = context.enableValidation}, true);
            REG_CHECK(foreignDevice);
            auto foreign = foreignDevice->get()->createShaderModule({.spirv = compiled[0].spirv});
            REG_CHECK(foreign);
            const render::ShaderStageDesc fragment{modules[1].get()};
            REG_CHECK(render::hasError(device->createGraphicsPipeline({.vertexShader = {foreign->get()},
                .fragmentShader = fragment}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(device->createGraphicsShaderObjectProgram({.vertexShader = {foreign->get()},
                .fragmentShader = fragment}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(device->createComputePipeline({.computeShader = {foreign->get()}}), render::Error::InvalidArgument));
            for (const char* entry : std::array<const char*, 2>{nullptr, ""}) {
                REG_CHECK(render::hasError(device->createGraphicsPipeline({.vertexShader = {modules[0].get(), entry},
                    .fragmentShader = fragment}), render::Error::InvalidArgument));
                REG_CHECK(render::hasError(device->createGraphicsShaderObjectProgram({.vertexShader = {modules[0].get(), entry},
                    .fragmentShader = fragment}), render::Error::InvalidArgument));
                REG_CHECK(render::hasError(device->createComputePipeline({.computeShader = {modules[0].get(), entry}}), render::Error::InvalidArgument));
            }
            compiled = {}; // Both executable forms must use the module's owned words.
            std::unique_ptr<render::GraphicsPipeline> pipeline;
            REG_REQUIRE(device->createGraphicsPipeline({
                .vertexShader = {modules[0].get()},
                .fragmentShader = {modules[1].get()},
                .colorFormat = render::Format::RGBA8Unorm,
            }).transform([&](auto value) { pipeline = std::move(value); }));
            std::unique_ptr<render::GraphicsShaderObjectProgram> program;
            REG_REQUIRE(device->createGraphicsShaderObjectProgram({.vertexShader = {modules[0].get()}, .fragmentShader = {modules[1].get()}}).transform([&](auto value) { program = std::move(value); }));
            std::array<render::PreparedExecution, 3> executions{pipeline->execution(), program->execution(), pipeline->execution()};
            auto invalidState = program->execution({.colorAttachmentCount = 9});
            // Snapshots survive hot replacement of all source objects before recording.
            pipeline.reset(); program.reset(); modules = {};
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(recording.begin(0));
            auto& command = *recording.commands;
            REG_CHECK(render::hasError(command.bindExecution({}), render::Error::InvalidArgument));
            REG_CHECK(render::hasError(command.bindExecution(invalidState), render::Error::InvalidArgument));
            invalidState = {};
            std::array<std::unique_ptr<render::Buffer>, 3> readbacks;
            std::array<std::weak_ptr<void>, 3> allocations;
            for (uint32_t i = 0; i < executions.size(); ++i) {
                std::unique_ptr<render::Texture> texture;
                REG_REQUIRE(device->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                    .format = render::Format::RGBA8Unorm, .width = extent, .height = extent}).transform([&](auto value) { texture = std::move(value); }));
                std::unique_ptr<render::TextureView> view;
                REG_REQUIRE(device->createTextureView(*texture, {}).transform([&](auto value) { view = std::move(value); }));
                REG_CHECK(!view->hasNativeView());
                REG_CHECK(render::hasError(device->createTextureView(*texture, {.range = {.baseMip = 1}}).transform([](auto) {}), render::Error::InvalidArgument));
                REG_CHECK(render::vulkan::nativeImageLayout(*view, render::ResourceState::ColorAttachment) ==
                    (unified ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
                REG_CHECK(!view->hasNativeView());
                allocations[i] = view->retainTexture();
                REG_REQUIRE(device->createBuffer({.size = bytes, .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto value) { readbacks[i] = std::move(value); }));
                render::TextureBarrierDesc barrier{
                    .texture = texture.get(),
                    .oldLayout = render::TextureLayout::Undefined,
                    .newLayout = render::TextureLayout::ColorAttachment,
                    .before = {},
                    .after = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
                };
                REG_REQUIRE(command.synchronize({.textures = {&barrier, 1}}));
                render::RenderingAttachmentDesc attachment{.view = view.get(), .state = render::ResourceState::ColorAttachment,
                    .loadOp = render::LoadOp::Clear, .clearColor = {0, 0, 0, 1}};
                REG_REQUIRE(command.beginRendering({.renderArea = {0, 0, extent, extent}, .colorAttachments = {&attachment, 1}}));
                REG_CHECK(view->hasNativeView());
                const auto native = render::vulkan::nativeImageView(*view);
                REG_CHECK(native != VK_NULL_HANDLE && native == render::vulkan::nativeImageView(*view));
                // Simulate the SDK/DGC boundary, then explicitly establish the new state.
                if (i == 1) { render::vulkan::notifyExternalDescriptorSetBinding(command); }
                REG_REQUIRE(command.bindExecution(executions[i]));
                command.setViewport({0, 0, float(extent), float(extent), 0, 1});
                command.setScissor({0, 0, extent, extent});
                command.draw(3);
                command.endRendering();
                barrier.oldLayout = render::TextureLayout::ColorAttachment; barrier.before = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite};
                barrier.newLayout = render::TextureLayout::TransferSource; barrier.after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead};
                REG_REQUIRE(command.synchronize({.textures = {&barrier, 1}}));
                command.copyTextureToBuffer({.texture = texture.get(), .buffer = readbacks[i].get(), .width = extent, .height = extent});
                view.reset(); texture.reset();
                REG_CHECK(!allocations[i].expired());
            }
            const auto stats = command.synchronizationStats();
            REG_CHECK(stats.imageTransitions == (unified ? 3 : 6));
            REG_CHECK(stats.memoryBarriers == (unified ? 3 : 0));
            executions = {}; // Only recorded commands now own the native programs.
            REG_REQUIRE(recording.submit(tracker, *gate));
            REG_CHECK(!recording.frame.completion().isComplete());
            REG_REQUIRE(gate->signal(1));
            REG_REQUIRE(recording.frame.wait(5'000'000'000ull));
            for (uint32_t i = 0; i < readbacks.size(); ++i) {
                readbacks[i]->invalidate();
                const auto* pixels = static_cast<const uint8_t*>(readbacks[i]->map());
                REG_CHECK(pixels && pixels[(extent / 2 * extent + extent / 2) * 4] > 0);
                if ((!preferUnified || context.evidence) && i == 0) { std::memcpy(reference.data(), pixels, bytes); }
                bench::readbackEvidence(context, "readback.bin", std::span<const uint8_t>(pixels, bytes));
                observations.insert(observations.end(), pixels, pixels + bytes);
                const bool same = std::memcmp(reference.data(), pixels, bytes) == 0;
                readbacks[i]->unmap();
                REG_CHECK(same);
            }
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            for (const auto& allocation : allocations) { REG_CHECK(allocation.expired()); }
            // Cancellation also releases an exported native view without submitting it.
            REG_REQUIRE(recording.begin(1));
            std::weak_ptr<void> cancelled;
            {
                auto texture = device->createTexture({.usage = render::TextureUsageBits::ColorAttachment, .format = render::Format::RGBA8Unorm});
                REG_CHECK(texture);
                auto view = device->createTextureView(**texture, {});
                REG_CHECK(view);
                cancelled = (*view)->retainTexture();
                REG_REQUIRE(command.useNativeTextureView(**view));
            }
            REG_CHECK(!cancelled.expired());
            recording.frame.cancel();
            REG_REQUIRE(recording.pool->reset());
            REG_REQUIRE(recording.frame.reset());
            REG_CHECK(cancelled.expired());
            REG_CHECK(errors.load() == 0);
        }
        bench::comparisonEvidence(context, {{"extent", extent}, {"draws", 3}}, observations,
            context.deviceDesc && context.deviceDesc->preferUnifiedImageLayouts);
        if (context.evidence) { return RHITestResult::pass("prepared views and three readbacks passed; parent compares layout policies"); }
        return RHITestResult::pass(unifiedTested ? "GENERAL and optimal layouts produced identical PSO/shader-object readback" :
            "Optimal-layout fallback passed; unified image layouts unavailable on this device");
    }
};
METALLIC_REGISTER_RHI_TEST(PreparedExecutionViewsTest);

class ParallelRegistryTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"parameters.parallel.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"});
    }

    ParallelRegistryTest() { type = RHITestType::Command; name = "parallel_registry_packets"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            auto result = runMode(context, mode);
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass("Four concurrent writers sharing a registry and kernel in mapped/native modes");
    }
private:
    static RHITestResult runMode(RHITestContext& context, render::SlangDescriptorHeapMode mode)
    {
        bench::TestDevice device;
        REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Parallel registry", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::shared_ptr<render::ResourceRegistry> registry;
        REG_REQUIRE(device->resourceRegistry().transform([&](auto value) { registry = std::move(value); }));
        render::ComputeKernel kernel;
        std::string log;
        REG_REQUIRE(makeKernel(*device, kernel, log, mode));
        std::unique_ptr<render::Buffer> source, output;
        REG_REQUIRE(makeBuffer(*device, source, 73));
        REG_REQUIRE(makeBuffer(*device, output));
        std::weak_ptr<void> allocation = source->retainAllocation();
        render::RenderFrameContext frame;
        std::array<render::CommandRecordingContext, 4> contexts;
        std::array<render::CommandBuffer*, 4> commands{};
        std::array<render::Result<>, 4> results;
        render::QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(*device, queue));
        REG_REQUIRE(frame.begin(0));
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            REG_REQUIRE(contexts[i].initialize(*device, queue));
            REG_REQUIRE(contexts[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
        }
        std::vector<std::jthread> workers;
        for (uint32_t i = 0; i < contexts.size(); ++i) {
            workers.emplace_back([&, i] {
                results[i] = contexts[i].record([&]() -> render::Result<> {
                    render::ParameterWriter writer(*device, frame, *registry);
                    ProbeParams params{writer.buffer(source.get()), writer.buffer(output.get()), i, i};
                    render::EncodedParameters encoded;
                    auto result = writer.encode(params, kABI).transform([&](auto value) { encoded = std::move(value); });
                    if (result) { result = kernel.dispatch(*commands[i], encoded, 1); }
                    return result ? commands[i]->end() : result;
                });
            });
        }
        workers.clear(); // jthread joins every local resource/parameter writer.
        for (const auto& recorded : results) { REG_REQUIRE(recorded); }
        source.reset();
        kernel.clear();
        REG_CHECK(!allocation.expired());
        REG_REQUIRE(tracker.submit({.commandBuffers = commands}, frame));
        REG_REQUIRE(frame.wait(5'000'000'000ull));
        output->invalidate();
        auto* mapped = output->map();
        REG_CHECK(mapped != nullptr);
        std::array<uint32_t, 4> actual{};
        std::memcpy(actual.data(), mapped, sizeof(actual));
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
        output->unmap();
        REG_CHECK((actual == std::array<uint32_t, 4>{73, 74, 75, 76}));
        for (auto& recording : contexts) { REG_REQUIRE(recording.reset()); }
        REG_REQUIRE(frame.reset());
        registry->collect();
        REG_CHECK(allocation.expired());
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(ParallelRegistryTest);

class PreparedDispatchParallelTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.prepared.parallel.lifetime.readback"}, bench::Layer::Core, "binding", "binding", {"readback.bin"}, true);
    }

    PreparedDispatchParallelTest() { type = RHITestType::Rendering; name = "prepared_dispatch_parallel_snapshot_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (const auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Prepared dispatch snapshot",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "FrameResourceProbe", .entryPointName = "copyValue",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = mode}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            render::ComputeProgram programs[2];
            render::ComputeProgramBindingDesc layout[] = {
                {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer},
                {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
            std::string log;
            for (auto& program : programs) {
                REG_REQUIRE(program.initialize(*device, {
                    .spirv = shader.spirv,
                    .pushConstantSize = 4,
                    .bindings = {layout, 2},
                    .requiresRayQuery = false,
                }, log));
                std::swap(layout[0], layout[1]);
            }
            std::unique_ptr<render::Buffer> input, output, arguments;
            REG_REQUIRE(makeBuffer(*device, input, 137));
            REG_REQUIRE(makeBuffer(*device, output));
            REG_REQUIRE(makeBuffer(*device, arguments, 1));
            auto* counts = static_cast<uint32_t*>(arguments->map());
            REG_CHECK(counts);
            for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
            arguments->flush(); arguments->unmap();
            std::weak_ptr<void> inputLife = input->retainAllocation(), argumentLife = arguments->retainAllocation();
            render::RenderFrameContext frame;
            render::CommandRecordingContext contexts[2];
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            REG_REQUIRE(frame.begin(0));
            render::PreparedComputeDispatch packets[2];
            render::Result<> outcomes[2];
            uint32_t indices[] = {0, 1, 2};
            const render::ComputeDispatchBinding bindings[] = {{.binding = 0, .buffer = input.get()}, {.binding = 1, .buffer = output.get()}};
            std::jthread first([&] {
                outcomes[0] = programs[0].prepareDispatch(frame, {
                    .bindings = {bindings, 2},
                    .pushData = &indices[0],
                    .pushDataSize = 4,
                }).transform([&](auto value) { packets[0] = std::move(value); });
            });
            std::jthread second([&] {
                const render::ComputeIndirectDispatch items[] = {
                    {.pushData = &indices[1]}, {.pushData = &indices[2], .argumentOffset = 12, .program = &programs[1]}};
                outcomes[1] = programs[0].prepareIndirectBatch(frame, {
                    .bindings = {bindings, 2},
                    .pushDataSize = 4,
                    .indirectArguments = arguments.get(),
                }, items).transform([&](auto value) { packets[1] = std::move(value); });
            });
            first.join(); second.join();
            for (const auto& outcome : outcomes) { REG_REQUIRE(outcome); }
            const auto failed = programs[0].prepareDispatch(frame, {
                .bindings = {bindings, 2},
                .pushData = &indices[0],
                .pushDataSize = 3,
            });
            REG_CHECK(render::hasError(failed, render::Error::InvalidArgument) && packets[0].valid());
            // Preparation owns constant bytes, permutations, descriptors and argument ranges.
            indices[0] = indices[1] = indices[2] = 15;
            input.reset(); arguments.reset(); programs[0].clear(); programs[1].clear();
            REG_CHECK(!inputLife.expired() && !argumentLife.expired());
            render::CommandBuffer* commands[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                REG_REQUIRE(contexts[i].initialize(*device, queue));
                REG_REQUIRE(contexts[i].prepare(frame).transform([&](auto value) { commands[i] = value; }));
            }
            const render::BufferBarrierDesc barrier{
                .buffer = output.get(),
                .before = {},
                .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            };
            if (auto commandResult = commands[0]->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            std::jthread recordA([&] { outcomes[0] = contexts[0].record([&]() -> render::Result<> {
                auto recorded = packets[0].record(*commands[0]); return recorded ? commands[0]->end() : recorded; }); });
            std::jthread recordB([&] { outcomes[1] = contexts[1].record([&]() -> render::Result<> {
                auto recorded = packets[1].record(*commands[1]); return recorded ? commands[1]->end() : recorded; }); });
            recordA.join(); recordB.join();
            for (const auto& outcome : outcomes) { REG_REQUIRE(outcome); }
            auto stale = packets[0];
            packets[0] = {}; packets[1] = {};
            const render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
            REG_REQUIRE(tracker.submit({
                .waitSemaphores = {&wait, 1},
                .commandBuffers = {commands, 2},
            }, frame));
            REG_CHECK(!inputLife.expired() && !argumentLife.expired());
            REG_REQUIRE(gate->signal(1)); REG_REQUIRE(frame.wait());
            output->invalidate();
            auto* values = static_cast<uint32_t*>(output->map());
            REG_CHECK(values);
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values, 16));
            const bool correct = values[0] == 137 && values[1] == 137 && values[2] == 137 && values[15] == 0;
            output->unmap(); REG_CHECK(correct);
            for (auto& recording : contexts) { REG_REQUIRE(recording.reset()); }
            REG_REQUIRE(frame.reset()); REG_REQUIRE(frame.begin(1));
            REG_REQUIRE(contexts[0].prepare(frame).transform([&](auto value) { commands[0] = value; }));
            REG_CHECK(!stale.record(*commands[0]));
            stale = {};
            REG_CHECK(inputLife.expired() && argumentLife.expired());
            frame.cancel();
            REG_REQUIRE(contexts[0].reset()); REG_REQUIRE(frame.reset());
        }
        return RHITestResult::pass("Mapped/native: concurrent preparation and recording, frozen constants, indirect permutations, lifetime and stale generation");
    }
};
METALLIC_REGISTER_RHI_TEST(PreparedDispatchParallelTest);

// Standalone packets own their parameter storage; a batch can be prepared and
// recorded after the writer, source wrappers and kernel wrappers are destroyed.
class KernelPreparedDispatchTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.prepared.direct.indirectBatch.readback", "compute.abi.staleTail.contract"}, bench::Layer::Core, "binding", "binding", {"readback.bin"}, true);
    }

    KernelPreparedDispatchTest() { type = RHITestType::Rendering; name = "compute_kernel_prepared_standalone_batch"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (const auto transport : {render::ParameterTransport::DescriptorBuffer, render::ParameterTransport::InlinePush}) {
            for (const auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
                bench::TestDevice device;
                REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Kernel prepared dispatch",
                    .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                    .transform([&](auto value) { device = std::move(value); }));
                auto registry = device->resourceRegistry();
                REG_CHECK(registry);
                auto& queue = *device->getQueue(render::QueueType::Graphics);
                std::unique_ptr<render::Buffer> input, output, arguments;
                REG_REQUIRE(makeBuffer(*device, input, 9));
                REG_REQUIRE(makeBuffer(*device, output));
                REG_REQUIRE(makeBuffer(*device, arguments));
                auto* counts = static_cast<uint32_t*>(arguments->map());
                REG_CHECK(counts);
                for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
                arguments->flush(); arguments->unmap();
                std::weak_ptr<void> inputLife = input->retainAllocation(), argumentLife = arguments->retainAllocation();
                Commands recording;
                REG_REQUIRE(recording.initialize(*device, queue));
                std::unique_ptr<render::Semaphore> gate;
                REG_REQUIRE(device->createSemaphore().transform([&](auto value) { gate = std::move(value); }));
                Drain drain{queue, *gate};
                REG_REQUIRE(recording.commands->begin()); // Deliberately no frame.
                render::PreparedComputeDispatch direct, batch, rejected;
                {
                    render::ComputeKernel kernels[2];
                    std::string log;
                    for (auto& kernel : kernels) { REG_REQUIRE(makeKernel(*device, kernel, log, mode, transport)); }
                    render::ParameterWriter writer(*device, **registry);
                    ProbeParams params{writer.buffer(input.get()), writer.buffer(output.get()), 1, 0};
                    auto first = writer.encode(params, kABI, transport);
                    REG_CHECK(first);
                    REG_CHECK(first->inlineData().empty() == (transport == render::ParameterTransport::DescriptorBuffer));
                    const auto before = (*registry)->stats().parameterBytes;
                    const auto wrongTransport = transport == render::ParameterTransport::InlinePush
                        ? render::ParameterTransport::DescriptorBuffer : render::ParameterTransport::InlinePush;
                    auto mismatch = writer.encode(params, kABI, wrongTransport);
                    REG_CHECK(mismatch && !kernels[0].prepareDispatch(*mismatch, 1));
                    if (transport == render::ParameterTransport::DescriptorBuffer) {
                        REG_CHECK((*registry)->stats().parameterBytes == before);
                    }
                    REG_REQUIRE(kernels[0].prepareDispatch(*first, 1).transform([&](auto value) { direct = std::move(value); }));
                    REG_CHECK(!kernels[0].prepareDispatch(*first, 0));
                    auto wrongAbi = writer.encode(params, kABI + 1, transport);
                    REG_CHECK(wrongAbi && !kernels[0].prepareDispatch(*wrongAbi, 1));
                    REG_CHECK(!kernels[0].prepareIndirectBatch({}));
                    render::ComputeIndirectParameters items[2];
                    for (uint32_t i = 0; i < 2; ++i) {
                        params.add = i + 2; params.index = i + 1;
                        REG_REQUIRE(writer.encode(params, kABI, transport).transform([&](auto value) { items[i].parameters = std::move(value); }));
                        REG_REQUIRE(arguments->slice({12 * i, 12}).transform([&](auto value) { items[i].arguments = std::move(value); }));
                        items[i].kernel = &kernels[i];
                    }
                    REG_REQUIRE(kernels[0].prepareIndirectBatch(items).transform([&](auto value) { batch = std::move(value); }));
                    auto saved = items[1].arguments;
                    items[1].arguments = {};
                    REG_CHECK(!kernels[0].prepareIndirectBatch(items));
                    items[1].arguments = saved;
                    params.add = 999; params.index = 15;
                    REG_REQUIRE(writer.encode(params, kABI, transport).transform([&](auto value) { items[0].parameters = std::move(value); }));
                    render::RenderFrameContext frame;
                    REG_REQUIRE(frame.begin(0));
                    render::ParameterWriter scopedWriter(*device, frame, **registry);
                    const ProbeParams scoped{scopedWriter.buffer(input.get()), scopedWriter.buffer(output.get()), 999, 15};
                    REG_REQUIRE(scopedWriter.encode(scoped, kABI, transport).transform([&](auto value) { items[1].parameters = std::move(value); }));
                    REG_REQUIRE(kernels[0].prepareIndirectBatch(items).transform([&](auto value) { rejected = std::move(value); }));
                    frame.cancel();
                }
                input.reset(); arguments.reset();
                REG_CHECK(!inputLife.expired() && !argumentLife.expired());
                // No prefix of a batch may execute when a later parameter packet is stale.
                REG_CHECK(render::hasError(rejected.record(*recording.commands), render::Error::InvalidArgument));
                rejected = {};
                REG_REQUIRE(direct.record(*recording.commands));
                REG_REQUIRE(batch.record(*recording.commands));
                direct = {}; batch = {};
                REG_REQUIRE(recording.commands->end());
                render::CommandBuffer* submitted[] = {recording.commands.get()};
                REG_REQUIRE(queue.submit({.commandBuffers = submitted}));
                REG_REQUIRE(queue.waitIdle());
                output->invalidate();
                const auto* actual = static_cast<const uint32_t*>(output->map());
                REG_CHECK(actual);
                bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual, 16));
                const bool correct = actual[0] == 10 && actual[1] == 11 && actual[2] == 12 && actual[15] == 0;
                output->unmap();
                REG_CHECK(correct);
                REG_REQUIRE(recording.pool->reset());
                REG_REQUIRE(recording.commands->begin()); // Reuse releases the submitted command snapshot.
                REG_REQUIRE(recording.commands->end());
                REG_CHECK(inputLife.expired() && argumentLife.expired());
            }
        }
        return RHITestResult::pass("BDA/inline, mapped/native: standalone storage, direct/batch execution, ABI checks, stale-tail rejection and retained allocations");
    }
};
METALLIC_REGISTER_RHI_TEST(KernelPreparedDispatchTest);

// Exercise the common prepared resource-table path in mapped and native modes.
// Two writes to the same word require a memory-only dependency between dispatches.
class BatchMemoryBarrierTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"compute.batch.memoryBarrier.readback", "compute.batch.error.contract"}, bench::Layer::Core, "binding", "sync", {"readback.bin"}, true);
    }

    BatchMemoryBarrierTest() { type = RHITestType::Rendering; name = "compute_batch_memory_barrier_and_error_propagation"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (uint32_t path = 0; path < 2; ++path) {
            bench::TestDevice device;
            REG_REQUIRE(bench::createTestDevice(context, {.applicationName = "Batch barriers",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }));
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            render::ShaderCompileResult shader;
            REG_REQUIRE(render::compileSlangShaderToSpirv({
                .moduleName = "BatchBarrierProbe",
                .entryPointName = "batchBarrierMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .descriptorHeapMode = path == 1 ? render::SlangDescriptorHeapMode::Native : render::SlangDescriptorHeapMode::Mapped,
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            const render::ComputeProgramBindingDesc layout{.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer};
            render::ComputeProgram program;
            std::string log;
            REG_REQUIRE(program.initialize(*device, {
                .spirv = shader.spirv,
                .bindings = {&layout, 1},
                .requiresRayQuery = false,
            }, log));
            std::unique_ptr<render::Buffer> output, arguments;
            REG_REQUIRE(makeBuffer(*device, output));
            REG_REQUIRE(makeBuffer(*device, arguments));
            auto* counts = static_cast<uint32_t*>(arguments->map());
            REG_CHECK(counts);
            for (uint32_t i = 0; i < 6; ++i) { counts[i] = 1; }
            arguments->flush(); arguments->unmap();
            Commands recording;
            REG_REQUIRE(recording.initialize(*device, queue));
            render::QueueSubmissionTracker tracker;
            REG_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::Semaphore> gate;
            REG_REQUIRE(device->createSemaphore({.initialValue = 1}).transform([&](auto value) { gate = std::move(value); }));
            Drain drain{queue, *gate};
            const render::ComputeDispatchBinding binding{.binding = 0, .buffer = output.get()};
            const render::ComputeIndirectDispatch items[] = {{.argumentOffset = 0}, {.argumentOffset = 12}};
            render::MemoryBarrierDesc memory{
                .before = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderWrite},
                .after = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderRead | render::AccessBits::ShaderWrite}};
            const render::BarrierDesc barrier{.memory = {&memory, 1}};
            REG_REQUIRE(recording.begin(0));
            render::ComputeDispatchDesc dispatch{
                .commandBuffer = recording.commands.get(),
                .bindings = {&binding, 1},
                .indirectArguments = arguments.get(),
            };
            REG_REQUIRE(program.dispatchIndirectBatch(dispatch, items, barrier));
            REG_CHECK(recording.commands->synchronizationStats().memoryBarriers == 1);
            REG_REQUIRE(recording.submit(tracker, *gate));
            REG_REQUIRE(recording.frame.wait());
            output->invalidate();
            const auto* value = static_cast<const uint32_t*>(output->map());
            REG_CHECK(value);
            const auto actual = *value;
            bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(value, 16));
            output->unmap();
            REG_CHECK(actual == 2);

            REG_REQUIRE(recording.begin(1));
            memory.after = {render::PipelineStageBits::Transfer, render::AccessBits::ShaderRead};
            REG_CHECK(render::hasError(program.dispatchIndirectBatch(dispatch, items, barrier), render::Error::InvalidArgument));
            REG_CHECK(recording.commands->synchronizationStats().calls == 0);
            recording.frame.cancel(); // Discard the first dispatch of the rejected batch.
        }
        return RHITestResult::pass("Prepared mapped/native: memory-only ordering and barrier errors");
    }
};
METALLIC_REGISTER_RHI_TEST(BatchMemoryBarrierTest);

// Direct fields must be independent of CPU binding IDs and retain slice/root storage.
class NamedResourceParametersTest : public RHITest {
public:
    NamedResourceParametersTest() { type = RHITestType::Resource; name = "named_resource_parameters_layout_and_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto& device = context.device;
        ShaderCompileResult shader;
        REG_REQUIRE(compileSlangShaderToSpirv({.moduleName = "NamedResourceProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics)
            .transform([&](auto value) { shader = std::move(value); }));
        const ComputeProgramBindingDesc bindings[] = {
            {.binding = 213, .kind = ComputeResourceBindingKind::DataBuffer, .dataStride = 4, .dataAlignment = 4},
            {.binding = 7, .kind = ComputeResourceBindingKind::StorageBuffer}};
        ComputeResourceField fields[] = {
            {213, ComputeResourceBindingKind::DataBuffer, 4, ComputeResourceFieldFormat::DataSpan},
            {7, ComputeResourceBindingKind::StorageBuffer, 0}};
        ComputeProgram program;
        std::string log;
        ComputeProgramDesc description{.spirv = shader.spirv, .pushConstantSize = 4,
            .bindings = bindings, .requiresRayQuery = false, .resourceParameters = {16, fields}};
        fields[1].offset = 4;
        REG_CHECK(hasError(program.initialize(device, description, log), Error::InvalidArgument));
        fields[1].offset = 16;
        REG_CHECK(hasError(program.initialize(device, description, log), Error::InvalidArgument));
        fields[1].offset = 0;
        fields[1].kind = ComputeResourceBindingKind::SampledImage;
        REG_CHECK(hasError(program.initialize(device, description, log), Error::InvalidArgument));
        fields[1].kind = ComputeResourceBindingKind::StorageBuffer;
        fields[0].format = ComputeResourceFieldFormat::Handle;
        REG_CHECK(hasError(program.initialize(device, description, log), Error::InvalidArgument));
        fields[0].format = ComputeResourceFieldFormat::DataSpan;
        REG_REQUIRE(program.initialize(device, description, log));
        // Layout metadata is borrowed only at initialize; mutation cannot change a live program.
        fields[0].offset = 0;
        std::unique_ptr<Buffer> input, output;
        REG_REQUIRE(makeBuffer(device, input));
        REG_REQUIRE(makeBuffer(device, output));
        auto* source = static_cast<uint32_t*>(input->map());
        REG_CHECK(source);
        source[1] = 40;
        input->flush(); input->unmap();
        auto slice = input->slice({4, 4});
        REG_CHECK(slice.has_value());
        Commands recording;
        REG_REQUIRE(recording.initialize(device, context.graphicsQueue));
        REG_REQUIRE(recording.begin(0));
        ComputeDispatchBinding resources[] = {
            {.binding = 7, .buffer = output.get()}, {.binding = 213, .data = *slice}};
        const uint32_t add = 2;
        auto prepared = program.prepareDispatch(recording.frame, {.bindings = resources,
            .pushData = &add, .pushDataSize = sizeof(add)});
        REG_CHECK(prepared.has_value());
        input.reset(); slice = BufferSlice{};
        resources[1].data = {}; // The prepared packet is now the only source allocation owner.
        REG_REQUIRE(prepared->record(*recording.commands));
        QueueSubmissionTracker tracker;
        REG_REQUIRE(tracker.initialize(device, context.graphicsQueue));
        std::unique_ptr<Semaphore> gate;
        REG_REQUIRE(device.createSemaphore({.initialValue = 1}).transform([&](auto value) { gate = std::move(value); }));
        Drain drain{context.graphicsQueue, *gate};
        REG_REQUIRE(recording.submit(tracker, *gate));
        REG_REQUIRE(recording.frame.wait());
        output->invalidate();
        const auto* values = static_cast<const uint32_t*>(output->map());
        REG_CHECK(values);
        const uint32_t actual = values[0], guard = values[1];
        bench::readbackEvidence(context, "named-resources.bin", std::span<const uint32_t>(values, 16));
        output->unmap();
        REG_CHECK(actual == 42 && guard == 0);
        return RHITestResult::pass("Direct named fields, sparse IDs, bounded slice and retained packet");
    }
};
METALLIC_REGISTER_RHI_TEST(NamedResourceParametersTest);

#undef REG_REQUIRE
#undef REG_CHECK
} // namespace
} // namespace metallic::tests
