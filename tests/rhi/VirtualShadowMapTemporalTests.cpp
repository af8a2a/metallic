#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Shadow/VirtualShadowMapTemporal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
constexpr uint32_t kLayers = 8;

void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct TemporalCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t submissionIndex = 0;

    explicit TemporalCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        check(tracker.initialize(device, queue), "Temporal submission tracker");
        check(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "Temporal pool");
        check(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "Temporal commands");
    }
    ~TemporalCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        // Submission generations are distinct even when the logical render
        // frame is repeated. The temporal API receives that frame explicitly.
        check(frame.begin(submissionIndex++), "Begin temporal submission");
        check(pool->reset(), "Reset temporal pool");
        check(commands->begin(frame.submissionContext()), "Begin temporal commands");
    }
    void finish(bool cancel = false)
    {
        check(commands->end(), "End temporal commands");
        if (cancel) { frame.cancel(); return; }
        CommandBuffer* buffers[]{commands.get()};
        check(tracker.submit({.commandBuffers = buffers}, frame), "Submit temporal commands");
        check(frame.wait(), "Wait temporal commands");
    }
};

struct PlaneImages
{
    uint32_t width, height, pixels;
    uint64_t layerStride;
    float planeZ = 0;
    bool initialized = false;
    std::unique_ptr<Texture> depth, shadows;
    std::unique_ptr<TextureView> depthView, shadowView;
    std::unique_ptr<Buffer> depthUpload, shadowUpload, readback;

    PlaneImages(Device& device, uint32_t w, uint32_t h) : width(w), height(h), pixels(w * h),
        layerStride((uint64_t(pixels) + 3u) & ~uint64_t(3))
    {
        check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = Format::D32Sfloat, .width = w, .height = h})
            .transform([&](auto t) { depth = std::move(t); }), "Temporal depth image");
        check(device.createTextureView(*depth, {.format = Format::D32Sfloat})
            .transform([&](auto v) { depthView = std::move(v); }), "Temporal depth view");
        check(device.createTexture({.usage = TextureUsageBits::Storage | TextureUsageBits::Sampled |
            TextureUsageBits::TransferDestination | TextureUsageBits::TransferSource,
            .format = Format::R8Unorm, .width = w, .height = h, .layerCount = kLayers})
            .transform([&](auto t) { shadows = std::move(t); }), "Temporal shadow array");
        check(device.createTextureView(*shadows, {.format = Format::R8Unorm, .range = {.layerCount = kLayers}})
            .transform([&](auto v) { shadowView = std::move(v); }), "Temporal shadow view");
        check(device.createBuffer({.size = uint64_t(pixels) * sizeof(float), .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload})
            .transform([&](auto b) { depthUpload = std::move(b); }), "Temporal depth upload");
        check(device.createBuffer({.size = layerStride * kLayers, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload})
            .transform([&](auto b) { shadowUpload = std::move(b); }), "Temporal shadow upload");
        check(device.createBuffer({.size = layerStride * kLayers, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto b) { readback = std::move(b); }), "Temporal readback");
    }

    void upload(CommandBuffer& commands, const std::array<uint8_t, kLayers>& raw)
    {
        auto* z = static_cast<float*>(depthUpload->map()); check(z != nullptr, "Map temporal depth");
        // The orthographic camera stays at z=4, with near=.01 and far=10.
        std::fill_n(z, pixels, (10.f - (4.f - planeZ)) / (10.f - .01f));
        depthUpload->flush(); depthUpload->unmap();
        auto* values = static_cast<uint8_t*>(shadowUpload->map()); check(values != nullptr, "Map temporal input");
        for (uint32_t layer = 0; layer < kLayers; ++layer) { std::fill_n(values + layer * layerStride, pixels, raw[layer]); }
        shadowUpload->flush(); shadowUpload->unmap();
        const TextureBarrierDesc before[]{
            {.texture = depth.get(), .oldLayout = initialized ? TextureLayout::ShaderRead : TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}},
            {.texture = shadows.get(), .oldLayout = initialized ? TextureLayout::General : TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}, .range = {.layerCount = kLayers}}};
        check(commands.synchronize({.textures = before}), "Temporal upload transitions");
        check(depthUpload->slice().and_then([&](const auto& slice) {
            return commands.copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = width, .height = height});
        }), "Upload temporal receiver");
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            check(shadowUpload->slice({layer * layerStride, pixels}).and_then([&](const auto& slice) {
                return commands.copyBufferToTexture({.texture = shadows.get(), .buffer = slice,
                    .width = width, .height = height, .baseLayer = layer});
            }), "Upload temporal raw shadow");
        }
        const TextureBarrierDesc ready[]{
            {.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
            {.texture = shadows.get(), .oldLayout = TextureLayout::TransferDestination, .newLayout = TextureLayout::General,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                .range = {.layerCount = kLayers}}};
        check(commands.synchronize({.textures = ready}), "Temporal inputs ready");
        initialized = true;
    }

    void copyResult(CommandBuffer& commands)
    {
        const TextureBarrierDesc before{.texture = shadows.get(), .oldLayout = TextureLayout::General,
            .newLayout = TextureLayout::TransferSource,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {.layerCount = kLayers}};
        check(commands.synchronize({.textures = {&before, 1}}), "Temporal readback transition");
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            check(readback->slice({layer * layerStride, pixels}).and_then([&](const auto& slice) {
                return commands.copyTextureToBuffer({.texture = shadows.get(), .buffer = slice,
                    .width = width, .height = height, .baseLayer = layer});
            }), "Read temporal visibility");
        }
        const TextureBarrierDesc after{.texture = shadows.get(), .oldLayout = TextureLayout::TransferSource,
            .newLayout = TextureLayout::General, .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
            .range = {.layerCount = kLayers}};
        check(commands.synchronize({.textures = {&after, 1}}), "Restore temporal output");
    }

    std::vector<uint8_t> read()
    {
        readback->invalidate();
        const auto* bytes = static_cast<const uint8_t*>(readback->map()); check(bytes != nullptr, "Map temporal result");
        std::vector<uint8_t> result(uint64_t(pixels) * kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            std::memcpy(result.data() + uint64_t(layer) * pixels, bytes + layer * layerStride, pixels);
        }
        readback->unmap();
        return result;
    }
};

std::array<uint8_t, kLayers> rawLayers(uint32_t layer, uint8_t value)
{
    std::array<uint8_t, kLayers> result{29, 42, 55, 68, 81, 94, 107, 120};
    result[layer] = value;
    return result;
}

class VirtualShadowMapTemporalLifecycleTest final : public RHITest
{
public:
    VirtualShadowMapTemporalLifecycleTest() { name = "virtual_shadow_map_temporal_lifecycle"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapTemporalLifecycle");
        std::filesystem::create_directories(directory);
        Json report{{"passed", false}, {"frames", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapTemporalLifecycle.json") << report.dump(2) << '\n'; };
        try {
            auto created = createDevice({.applicationName = "VSM temporal lifecycle",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless and mesh shaders"); }
            check(created, "Temporal lifecycle device");
            auto& device = **created;
            TemporalCommands gpu(device);
            auto images = std::make_unique<PlaneImages>(device, 33, 25);
            RenderView camera;
            check(camera.setCamera({.eye = {0, 0, 4}, .center = {0, 0, 0}, .nearPlane = .01f,
                .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Temporal camera");
            VirtualShadowMapSettings settings;
            settings.filter = VirtualShadowMapFilter::SMRT;
            settings.smrtTemporalSampling = true;
            VirtualShadowMapLight light;
            light.angularRadius = .04f;
            VirtualShadowMapTemporal history;
            uint64_t invalidation = 1, content = 7;
            uint32_t maxFrames = 16, baseLayer = 1, faceCount = 1;
            bool compact = true;
            std::string log;
            auto record = [&](VirtualShadowMapTemporal& target, uint64_t index) {
                const auto view = camera.constants(index, images->width, images->height, images->width, images->height);
                return target.record(device, *gpu.commands, *images->depthView, *images->shadowView, view,
                    index, invalidation, content, light, settings, maxFrames, baseLayer, faceCount, compact, log);
            };
            auto render = [&](VirtualShadowMapTemporal& target, const std::string& name, uint64_t index,
                const std::array<uint8_t, kLayers>& raw) {
                validation.phase(name);
                gpu.begin(); images->upload(*gpu.commands, raw);
                check(record(target, index), name + ": temporal record: " + log);
                images->copyResult(*gpu.commands); gpu.finish();
                auto result = images->read();
                report["frames"].push_back({{"name", name}, {"frame", index}, {"width", images->width},
                    {"height", images->height}, {"historyBytes", target.memoryBytes()},
                    {"shadowSha256", debug::debugSha256(result)}});
                save();
                return result;
            };
            auto uniform = [&](const std::string& name, uint64_t index, uint8_t raw, double expected, uint32_t border = 2u) {
                const auto input = rawLayers(baseLayer, raw);
                auto result = render(history, name, index, input);
                uint32_t wrong = 0, guards = 0, checked = 0;
                const int reference = int(std::lround(expected * 255));
                for (uint32_t layer = 0; layer < kLayers; ++layer) {
                    for (uint32_t y = 0; y < images->height; ++y) { for (uint32_t x = 0; x < images->width; ++x) {
                        const auto value = result[uint64_t(layer) * images->pixels + y * images->width + x];
                        if (layer != baseLayer) { guards += value != input[layer]; }
                        else if (x >= border && y >= border && x + border < images->width && y + border < images->height) {
                            ++checked; wrong += std::abs(int(value) - reference) > 1;
                        }
                    }}
                }
                report["frames"].back().update({{"expectedVisibility", expected}, {"checkedPixels", checked},
                    {"wrongPixels", wrong}, {"changedGuardPixels", guards}});
                save();
                check(checked >= 500 && wrong == 0 && guards == 0, name + ": unexpected mean or unrelated layer write");
                return result;
            };
            (void)uniform("First", 0, 0, 0, 0);
            (void)uniform("Second", 1, 255, .5);
            const auto third = uniform("Third", 2, 0, 1.0 / 3);
            check(uniform("RepeatFrame", 2, 0, 1.0 / 3) == third, "Repeated frame accumulated twice");
            check(uniform("RepeatFrameAgain", 2, 0, 1.0 / 3) == third, "Third recording changed the same logical frame");

            validation.phase("CancelUnsubmitted");
            gpu.begin(); images->upload(*gpu.commands, rawLayers(baseLayer, 255));
            check(record(history, 3), "Record cancelled frame");
            const auto duplicate = record(history, 3);
            check(hasError(duplicate, Error::InvalidArgument), "Unsubmitted history accepted another recording");
            gpu.finish(true);
            report["unsubmittedRejected"] = true;
            // Frame 3 never reached the queue. Frame 4 is a gap, so it must not
            // blend with the cancelled frame's white input or its sample count.
            (void)uniform("CancelledThenGap", 4, 255, 1, 0);
            (void)uniform("AfterGap", 5, 0, .5);
            camera.cameraCut();
            (void)uniform("CameraCut", 6, 255, 1, 0);
            (void)uniform("AfterCut", 7, 0, .5);
            ++invalidation;
            (void)uniform("InvalidationRevision", 8, 255, 1, 0);
            (void)uniform("AfterInvalidation", 9, 0, .5);
            ++content;
            (void)uniform("CasterContent", 10, 255, 1, 0);
            (void)uniform("AfterContent", 11, 0, .5);
            light.direction = {.2f, -.9f, .1f};
            (void)uniform("LightChange", 12, 255, 1, 0);
            (void)uniform("AfterLight", 13, 0, .5);
            settings.smrtMaxRayLength = 9;
            (void)uniform("SearchChange", 14, 255, 1, 0);
            settings.forceRedraw = true; settings.cacheEnabled = true;
            (void)uniform("PagePolicyKeepsHistory", 15, 0, .5);
            baseLayer = 3;
            (void)uniform("OutputSlotMove", 16, 0, 1.0 / 3);
            maxFrames = 1;
            (void)uniform("OneFrameBlack", 17, 0, 0, 0);
            (void)uniform("OneFrameWhite", 18, 255, 1, 0);
            maxFrames = 16;
            (void)uniform("HistoryLengthChange", 19, 0, 0, 0);
            (void)uniform("AfterHistoryLength", 20, 255, .5);
            images->planeZ = 1;
            (void)uniform("DepthDisocclusion", 21, 0, 0, 0);
            (void)uniform("AfterDisocclusion", 22, 255, .5);
            auto movedCamera = camera.camera(); movedCamera.eye[0] += .08f; movedCamera.center[0] += .08f;
            check(camera.setCamera(movedCamera), "Move temporal camera");
            (void)uniform("CameraTranslation", 23, 0, 1.0 / 3);
            images = std::make_unique<PlaneImages>(device, 35, 27);
            (void)uniform("Resize", 24, 255, 1, 0);

            uint64_t steadyBytes = 0;
            for (uint64_t frame = 25; frame < 49; ++frame) {
                (void)render(history, "PoolReuse-" + std::to_string(frame), frame, rawLayers(baseLayer, uint8_t(frame % 2 * 255)));
                if (frame == 31) { steadyBytes = history.memoryBytes(); }
                if (frame > 31) { check(history.memoryBytes() == steadyBytes, "Steady temporal history retained an unbounded frame chain"); }
            }
            check(steadyBytes != 0, "Temporal pool did not own history memory");
            report["steadyHistoryBytes"] = steadyBytes;

            settings.shadowLOD = true;
            (void)uniform("ShadowLODChange", 49, 255, 1, 0);
            (void)uniform("AfterShadowLOD", 50, 0, .5);
            settings.shadowLODSmartBias = true;
            (void)uniform("SmartBiasChange", 51, 255, 1, 0);
            (void)uniform("AfterSmartBias", 52, 0, .5);
            settings.shadowLODScreenRayLength = .01f;
            (void)uniform("ScreenRayLengthChange", 53, 255, 1, 0);
            (void)uniform("AfterScreenRayLength", 54, 0, .5);

            // Point visibility is one signal per light, even when six public
            // layers are used. Distinct face inputs expose accidental wrapping.
            check(camera.setCamera({.eye = {0, 0, 4}, .center = {0, 0, 0}, .nearPlane = .01f,
                .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Reset point camera");
            light.type = VirtualShadowMapLightType::PointFace; light.position = {-2, -2, 2}; light.sourceRadius = .3f;
            faceCount = 6; baseLayer = 1; maxFrames = 16;
            for (const bool compactMode : {true, false}) {
                compact = compactMode;
                VirtualShadowMapTemporal pointHistory;
                auto first = rawLayers(baseLayer, 0), second = rawLayers(baseLayer, 255);
                if (!compact) {
                    for (uint32_t face = 0; face < 6; ++face) {
                        first[baseLayer + face] = uint8_t(face * 32);
                        second[baseLayer + face] = uint8_t(255 - face * 32);
                    }
                }
                const std::string label = compact ? "PointCompact" : "PointLegacy";
                (void)render(pointHistory, label + "-First", 0, first);
                const auto result = render(pointHistory, label + "-Second", 1, second);
                std::array<uint32_t, 6> coverage{};
                uint32_t wrong = 0, guards = 0;
                for (uint32_t y = 2; y + 2 < images->height; ++y) { for (uint32_t x = 2; x + 2 < images->width; ++x) {
                    const float dx = ((float(x) + .5f) / images->width * 2 - 1) * 2 * float(images->width) / images->height + 2;
                    const float dy = (1 - (float(y) + .5f) / images->height * 2) * 2 + 2;
                    const float dz = -2;
                    const std::array magnitude{std::abs(dx), std::abs(dy), std::abs(dz)};
                    auto sorted = magnitude; std::sort(sorted.begin(), sorted.end());
                    if (sorted[2] - sorted[1] < .1f) { continue; }
                    const uint32_t face = magnitude[0] > magnitude[1] && magnitude[0] > magnitude[2] ? (dx >= 0 ? 0u : 1u) :
                        magnitude[1] > magnitude[2] ? (dy >= 0 ? 2u : 3u) : (dz >= 0 ? 4u : 5u);
                    ++coverage[face];
                    const uint32_t selected = baseLayer + (compact ? 0 : face);
                    for (uint32_t layer = 0; layer < kLayers; ++layer) {
                        const auto value = result[uint64_t(layer) * images->pixels + y * images->width + x];
                        if (layer == selected) { wrong += std::abs(int(value) - 128) > 1; }
                        else { guards += value != second[layer]; }
                    }
                }}
                report["frames"].back().update({{"faceCoverage", coverage}, {"wrongPixels", wrong}, {"changedGuardPixels", guards}});
                save();
                check(std::count_if(coverage.begin(), coverage.end(), [](uint32_t count) { return count > 50; }) >= 3,
                    label + ": fixture missed canonical point faces");
                check(wrong == 0 && guards == 0, label + ": point history accumulated per face or wrote unrelated layers");
            }
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Temporal lifecycle Vulkan validation error");
            report["passed"] = true; save();
            return RHITestResult::pass("Temporal history preserves submission, repeat-frame, invalidation, point-layer and bounded-memory contracts");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapTemporalLifecycleTest);
} // namespace
} // namespace metallic::tests
