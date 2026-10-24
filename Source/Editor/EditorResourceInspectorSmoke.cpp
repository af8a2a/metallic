#include "Editor/EditorApplication.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <numeric>
#include <numbers>
#include <cstdio>

namespace metallic {
namespace {
class InspectorResourceSubsystem final : public render::IRenderSubsystem {
public:
    static constexpr render::RenderSubsystemId kSubsystemId = "test.inspector";
    bool replace = false, visible = true;
    render::Result<> initialize(const render::RenderSubsystemInitContext& context, std::string&) override
    {
        device_ = &context.device;
        replace = true;
        return {};
    }
    render::Result<> recordPreGraph(const render::RenderSubsystemFrameContext& context, std::string&) override
    {
        if (replace) {
            auto buffer = device_->createBuffer({.size = 16, .structureStride = 4,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload});
            if (!buffer) { return render::makeError(buffer.error()); }
            auto* words = static_cast<uint32_t*>((*buffer)->map());
            if (!words) { return render::makeError(render::Error::Failure); }
            ++revision_;
            for (uint32_t i = 0; i < 4; ++i) { words[i] = uint32_t(revision_) * 100 + i; }
            (*buffer)->flush(); (*buffer)->unmap();
            buffer_ = std::move(*buffer);
            replace = false;
        }
        context.commandBuffer->hostWriteBarrier();
        return {};
    }
    void appendDebugBindings(const render::RenderSubsystemFrameContext&, std::vector<render::DebugResourceBinding>& bindings) override
    {
        if (visible) {
            bindings.push_back({.id = "values", .buffer = buffer_.get(), .state = render::ResourceState::ShaderRead,
                .layout = "u32", .allocation = revision_});
        }
    }
private:
    render::Device* device_ = nullptr;
    std::unique_ptr<render::Buffer> buffer_;
    uint64_t revision_ = 0;
};

class InspectorHDRPass final : public render::RasterPass {
public:
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& color = reflection.addTextureOutput("color");
        color.format = render::Format::RGBA16Sfloat;
        color.colorEncoding = render::DisplayColorEncoding::SceneLinear;
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        environment::EnvironmentSnapshot atmosphere;
        atmosphere.source = environment::EnvironmentSource::PhysicalAtmosphere;
        atmosphere.celestial[0].enabled = true;
        std::string log;
        auto result = context.subsystem<render::EnvironmentLightingSubsystem>()->resolveRadiance(
            *context.subsystems()->device(), context.commandBuffer(), *context.subsystems(), atmosphere, {0.0, 2.0, 0.0}, log);
        if (!result) { return render::makeError(result.error()); }
        return clear_->execute(context);
    }
private:
    std::unique_ptr<render::RenderGraphPass> clear_ = render::builtin_pass::createClearColorPass();
};

class InspectorVolumePass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& volume = reflection.addTextureOutput("volume");
        volume.format = render::Format::RGBA8Unorm;
        volume.texture3D(8, 6, 4).transferWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        using namespace render;
        auto result = context.device->createBuffer({.size = 8 * 6 * 4 * 4, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto buffer) { upload_ = std::move(buffer); });
        if (!result) { return result; }
        auto* bytes = static_cast<uint8_t*>(upload_->map());
        if (!bytes) { return makeError(Error::Failure); }
        for (uint32_t z = 0; z < 4; ++z) {
            for (uint32_t y = 0; y < 6; ++y) {
                for (uint32_t x = 0; x < 8; ++x) {
                    const auto i = ((z * 6 + y) * 8 + x) * 4;
                    bytes[i] = uint8_t(x * 25); bytes[i + 1] = uint8_t(y * 35);
                    bytes[i + 2] = uint8_t(z * 70); bytes[i + 3] = 255;
                }
            }
        }
        upload_->flush(); upload_->unmap();
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return upload_->slice().and_then([&](const auto& source) {
            return context.commandBuffer().copyBufferToTexture({.texture = context.output("volume")->texture, .buffer = source,
                .bufferRowPitch = 8 * 4, .bufferSlicePitch = 8 * 6 * 4, .width = 8, .height = 6, .depth = 4});
        });
    }
private:
    std::unique_ptr<render::Buffer> upload_;
};

class InspectorBufferPass final : public render::ComputePass {
public:
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        if (!layout_.empty()) { return {}; }
        static constexpr std::array ids{InspectorResourceSubsystem::kSubsystemId,
            render::EnvironmentLightingSubsystem::kSubsystemId, render::GPUSceneSubsystem::kSubsystemId};
        return ids;
    }
    explicit InspectorBufferPass(std::string layout = {}, uint32_t stride = 16)
        : layout_(std::move(layout)), stride_(stride) {}
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("values").buffer(1024, stride_).bufferLayout(layout_).transferWrite();
        reflection.addBufferInput("exposure").buffer(16, 16).storageRead().setOptional();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        using namespace render;
        auto result = context.device->createBuffer({.size = 1024, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto buffer) { upload_ = std::move(buffer); });
        if (!result) { return result; }
        auto* words = static_cast<uint32_t*>(upload_->map());
        if (!words) { return makeError(Error::Failure); }
        for (uint32_t i = 0; i < 256; ++i) { words[i] = i * 17 + 3; }
        upload_->flush(); upload_->unmap();
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto source = upload_->slice();
        auto destination = context.output("values")->buffer->slice();
        if (!source || !destination) { return render::makeError(render::Error::Failure); }
        return context.commandBuffer().copyBuffer(*source, *destination);
    }
private:
    std::string layout_;
    uint32_t stride_;
    std::unique_ptr<render::Buffer> upload_;
};

bool readInspectorImage(render::Device& device, render::Queue& queue, render::vulkan::VulkanImGuiBackend& backend,
    render::Texture& texture, std::vector<uint8_t>& bytes, ImDrawData* drawData = nullptr,
    EditorResourceInspector* pendingPreview = nullptr)
{
    using namespace render;
    const uint32_t stride = texture.desc().format == Format::RGBA16Sfloat ? 8 : 4;
    const uint32_t width = texture.desc().width, height = texture.desc().height;
    bytes.resize(size_t(width) * height * stride);
    auto readback = device.createBuffer({.size = bytes.size(), .usage = BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::HostReadback});
    auto pool = device.createCommandPool(queue);
    auto fence = device.createFence(false);
    if (!readback || !pool || !fence) { return false; }
    auto commands = (*pool)->createCommandBuffer();
    if (!commands || !(*commands)->begin()) { return false; }
    if (pendingPreview && !pendingPreview->upload(**commands)) { return false; }
    TextureBarrierDesc barrier{.texture = &texture, .oldLayout = TextureLayout::ShaderRead,
        .newLayout = TextureLayout::TransferSource, .before = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead},
        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
    std::unique_ptr<TextureView> view;
    if (drawData) {
        if (!device.createTextureView(texture, {}).transform([&](auto value) { view = std::move(value); })) { return false; }
        TextureBarrierDesc target{.texture = &texture, .oldLayout = TextureLayout::Undefined,
            .newLayout = TextureLayout::ColorAttachment, .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite}};
        if (!(*commands)->synchronize({.textures = {&target, 1}})) { return false; }
        RenderingAttachmentDesc attachment{.view = view.get(), .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store};
        if (!(*commands)->beginRendering({.renderArea = {0, 0, width, height}, .colorAttachments = {&attachment, 1}}) ||
            !backend.render(**commands, drawData)) { return false; }
        (*commands)->endRendering();
        barrier.oldLayout = TextureLayout::ColorAttachment;
        barrier.before = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite};
    }
    if (!(*commands)->synchronize({.textures = {&barrier, 1}})) { return false; }
    if (!(*readback)->slice().and_then([&](const auto& slice) {
        return (*commands)->copyTextureToBuffer({.texture = &texture, .buffer = slice,
            .bufferRowPitch = width * stride, .bufferSlicePitch = width * height * stride, .width = width, .height = height});
    })) { return false; }
    if (!drawData) {
        barrier.oldLayout = TextureLayout::TransferSource; barrier.newLayout = TextureLayout::ShaderRead;
        barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferRead};
        barrier.after = {PipelineStageBits::FragmentShader, AccessBits::ShaderRead};
        if (!(*commands)->synchronize({.textures = {&barrier, 1}})) { return false; }
    }
    if (!(*commands)->end()) { return false; }
    CommandBuffer* raw = commands->get();
    if (!queue.submit({.commandBuffers = {&raw, 1}, .signalFence = fence->get()}) || !(*fence)->wait()) { return false; }
    (*readback)->invalidate();
    const void* mapped = (*readback)->map();
    if (!mapped) { return false; }
    std::memcpy(bytes.data(), mapped, bytes.size());
    (*readback)->unmap();
    return true;
}

bool writePPM(const std::filesystem::path& path, std::span<const uint8_t> bytes, uint32_t width, uint32_t height,
    render::Format format, float paperWhite)
{
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << ' ' << height << "\n255\n";
    for (size_t i = 0; i < size_t(width) * height; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            uint8_t value;
            if (format == render::Format::RGBA16Sfloat) {
                uint16_t h;
                std::memcpy(&h, bytes.data() + i * 8 + c * 2, 2);
                const int exponent = (h >> 10) & 31, mantissa = h & 1023;
                float linear = std::ldexp(exponent ? 1.f + mantissa / 1024.f : mantissa / 1024.f, exponent ? exponent - 15 : -14);
                if (h & 0x8000) { linear = -linear; }
                linear = std::clamp(linear * 80.f / paperWhite, 0.f, 1.f);
                const float srgb = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.f / 2.4f) - 0.055f;
                value = uint8_t(srgb * 255.f + 0.5f);
            } else {
                const bool bgra = format == render::Format::BGRA8Unorm || format == render::Format::BGRA8sRGB;
                value = bytes[i * 4 + (bgra ? 2 - c : c)];
            }
            file.put(static_cast<char>(value));
        }
    }
    return file.good();
}
} // namespace

bool EditorApplication::runResourceInspectorBenchmark(const std::filesystem::path& output, const char* sample)
{
    using Clock = std::chrono::steady_clock;
    // This measures synchronized editor frame latency, not uncapped throughput.
    // Keep validation/cache settings identical between before/after runs.
    if (debugRuntime_) { spdlog::error("Inspector benchmark requires the local observer; omit --debug-control"); return false; }
    debug::DebugValue report{{"sample", sample}, {"passed", false}, {"runs", debug::DebugValue::array()},
        {"scope", "CPU renderFrame plus submitted editor GPU completion; fixed camera, warmed shaders/streaming, no desktop automation"}};
    const auto save = [&] { std::ofstream file(output / "benchmark.json"); file << report.dump(2); };
    const auto frame = [&] { return renderFrame() && !viewportCompileFailed_ && frameSubmissions_.wait(); };
    scene_.setDirty(false);
    renderGraphEditorOpen_ = false; resourceInspectorVisible_ = false;
    loadBuiltInSample(sample);
    if (!waitForPendingSceneLoad(60000)) { return false; }
    const auto start = Clock::now();
    auto ready = Clock::time_point{};
    while (ready == Clock::time_point{} || Clock::now() - ready < std::chrono::seconds(5)) {
        if (!frame()) { return false; }
        const auto* streamer = subsystemHost_.get<render::StreamerSubsystem>();
        if (streamer && streamer->sceneReadiness().ready && streamer->sceneReadiness().requiredPages && ready == Clock::time_point{}) {
            ready = Clock::now();
        }
        if (Clock::now() - start > std::chrono::seconds(180)) { spdlog::error("Inspector benchmark readiness timeout"); return false; }
    }
    report["warmupSeconds"] = std::chrono::duration<double>(Clock::now() - start).count();
    const auto generation = graphExecutor_->executionStats().graphGeneration;
    const auto* previewTexture = graphExecutor_->outputResource("FinalBlit.color")->texture;
    report["initialGraphGeneration"] = generation;
    auto& inspector = resourceInspector_;
    const auto summarize = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return debug::DebugValue{{"mean", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
            {"p50", values[values.size() / 2]}, {"p95", values[(values.size() - 1) * 95 / 100]}, {"max", values.back()}};
    };
    // Reverse the phase order on the second round to expose warmup/order bias.
    for (const char* mode : {"closed", "frozen", "live", "live", "frozen", "closed"}) {
        spdlog::info("[Inspector benchmark transition] mode={}", mode);
        const bool open = std::string_view(mode) != "closed";
        renderGraphEditorOpen_ = open; resourceInspectorVisible_ = open;
        resourceInspectorSelectTab_ = open;
        inspector.live_ = std::string_view(mode) == "live";
        if (open) { inspector.select("Deferred.motionVectors"); inspector.refresh_ = true; }
        const auto transition = Clock::now();
        for (int i = 0; i < 40; ++i) { if (!frame()) { return false; } }
        // Retain a settling interval for like-for-like performance measurements.
        const auto settled = Clock::now();
        while (Clock::now() - settled < std::chrono::seconds(3) ||
            !subsystemHost_.get<render::StreamerSubsystem>()->sceneReadiness().ready) {
            if (!frame()) { return false; }
            if (Clock::now() - transition > std::chrono::seconds(180)) { return false; }
        }
        if (open && (!inspector.capture_ || !inspector.descriptor_)) {
            spdlog::error("Inspector benchmark capture failed: {}", inspector.status_); return false;
        }
        if (graphExecutor_->executionStats().graphGeneration != generation ||
            graphExecutor_->outputResource("FinalBlit.color")->texture != previewTexture) {
            spdlog::error("Inspector visibility rebuilt the warmed graph or replaced its output"); return false;
        }
        const double transitionSeconds = std::chrono::duration<double>(Clock::now() - transition).count();
        std::vector<double> frameMs, captureFrameMs;
        uint64_t captureChanges = 0;
        const auto measurementStart = Clock::now();
        for (int i = 0; i < 120 || (inspector.live_ && Clock::now() - measurementStart < std::chrono::seconds(5)); ++i) {
            const auto capture = inspector.capture_;
            const auto before = Clock::now();
            if (!frame()) { return false; }
            const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
            frameMs.push_back(elapsed);
            if (capture != inspector.capture_) { ++captureChanges; captureFrameMs.push_back(elapsed); }
        }
        debug::DebugValue result{{"mode", mode}, {"frames", frameMs.size()}, {"frameMs", summarize(frameMs)},
            {"graphGeneration", graphExecutor_->executionStats().graphGeneration}, {"graphResourcesPreserved", true},
            {"captureChanges", captureChanges}, {"transitionAndSettleSeconds", transitionSeconds},
            {"renderResolution", viewportView_.renderResolutionProperties()}, {"camera", viewportView_.cameraProperties()}};
        if (!captureFrameMs.empty()) { result["captureFrameMs"] = summarize(captureFrameMs); }
        if (open) {
            result["inspector"] = debug::encodeLossless(inspector.diagnostics());
            const auto snapshot = inspector.runtime().core().latestSnapshot();
            result["snapshotBytes"] = snapshot->values.dump().size();
            result["debugDetail"] = snapshot->values.at("debugControl").at("detail");
            const auto& artifact = inspector.capture_->artifacts[0];
            result["resourceBytes"] = artifact.bytes.size();
            std::vector<uint8_t> pixels;
            if (!readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, pixels)) { return false; }
            const auto name = std::string(mode) + std::to_string(report["runs"].size());
            if (!writePPM(output / (name + ".ppm"), pixels, inspector.image_->desc().width, inspector.image_->desc().height,
                render::Format::RGBA8Unorm, displayOutput_.paperWhiteNits)) { return false; }
            std::ofstream data(output / (name + ".bin"), std::ios::binary);
            data.write(reinterpret_cast<const char*>(artifact.bytes.data()), artifact.bytes.size());
        }
        report["runs"].push_back(result); save();
        spdlog::info("[Inspector benchmark] {}: {}", mode, result.at("frameMs").dump());
    }
    for (int cycle = 0; cycle < 3; ++cycle) {
        for (bool open : {true, false}) {
            renderGraphEditorOpen_ = open; resourceInspectorVisible_ = open; resourceInspectorSelectTab_ = open;
            inspector.live_ = false;
            if (open) { inspector.refresh_ = true; }
            spdlog::info("[Inspector toggle regression] cycle={} open={}", cycle, open);
            for (int i = 0; i < 8; ++i) { if (!frame()) { return false; } }
            if (graphExecutor_->executionStats().graphGeneration != generation ||
                graphExecutor_->outputResource("FinalBlit.color")->texture != previewTexture ||
                resourceInspectorAttached_ != open || (open && !inspector.capture_)) { return false; }
        }
    }
    report["reopenCyclesWithoutRecompile"] = 3;
    report["passed"] = true; save();
    return true;
}

bool EditorApplication::runResourceInspectorSmokeTest(const char* outputDirectory)
{
    const auto output = outputDirectory ? std::filesystem::path(outputDirectory) : std::filesystem::path(".cache/InspectorSmoke");
    std::error_code error;
    std::filesystem::create_directories(output, error);
    if (error) { spdlog::error("Cannot create inspector evidence directory: {}", error.message()); return false; }
    if (const char* sample = std::getenv("METALLIC_INSPECTOR_BENCHMARK_SAMPLE"); sample && *sample) {
        return runResourceInspectorBenchmark(output, sample);
    }
    struct Report {
        std::filesystem::path path;
        debug::DebugValue value{{"schema", "metallic-resource-inspector-smoke-v1"}, {"passed", false},
            {"checks", debug::DebugValue::array()}, {"artifacts", debug::DebugValue::array()}};
        ~Report() { std::ofstream file(path); file << value.dump(2); }
    } report{output / "report.json"};
    const auto expect = [&](bool condition, const char* message) {
        report.value["checks"].push_back({{"check", message}, {"passed", condition}});
        spdlog::info("[Smoke Resource Inspector] {}: {}", condition ? "PASS" : "FAIL", message);
        if (!condition) { spdlog::error("[Smoke Resource Inspector] {}", message); }
        return condition;
    };
    render::registerRenderGraphPassType("InspectorBufferFixture", "Inspector smoke fixture", [] { return std::make_unique<InspectorBufferPass>(); });
    std::string subsystemLog;
    if (!expect(subsystemHost_.registerSubsystem<InspectorResourceSubsystem>(subsystemLog), "Register independent resource inspection subsystem")) { return false; }
    render::registerRenderGraphPassType("InspectorHDRFixture", "Inspector HDR fixture", [] { return std::make_unique<InspectorHDRPass>(); });
    render::registerRenderGraphPassType("InspectorVolumeFixture", "Inspector volume fixture", [] { return std::make_unique<InspectorVolumePass>(); });
    render::registerRenderGraphPassType("InspectorBadStride", "Invalid schema fixture", [] { return std::make_unique<InspectorBufferPass>("float4", 4); });
    render::registerRenderGraphPassType("InspectorBadType", "Invalid schema fixture", [] { return std::make_unique<InspectorBufferPass>("UnregisteredType"); });
    for (const char* type : {"InspectorBadStride", "InspectorBadType"}) {
        render::RenderGraph invalid;
        invalid.addNode(type, "Invalid"); invalid.markOutput("Invalid.values");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!expect(!executor.compile(*device_, invalid, 16, 16, log) && log.find("layout/stride mismatch") != std::string::npos,
            "Reject an unknown schema or a schema/stride mismatch")) { return false; }
    }
    destroyViewportTexture();
    scene::LightingSettings lighting;
    lighting.autoExposure.lowPercent = 0;
    lighting.autoExposure.highPercent = 100;
    scene_.setLighting(lighting);
    renderWorld_.setLighting(lighting);
    renderGraph_ = render::RenderGraph{};
    renderGraph_.addNode("ClearColorPass", "Clear", {{"color", {0.25f, 0.5f, 0.75f, 1.f}}});
    renderGraph_.addNode("InspectorBufferFixture", "Data");
    renderGraph_.addNode("InspectorVolumeFixture", "Volume");
    renderGraph_.addNode("InspectorHDRFixture", "HDR", {{"color", {0.5f, 0.5f, 0.5f, 1.f}}});
    renderGraph_.addNode("AutoExposurePass", "AutoExposure", {{"adaptationDeltaSeconds", 0.1f}});
    renderGraph_.addEdge("HDR.color", "AutoExposure.source");
    renderGraph_.addEdge("AutoExposure.exposure", "Data.exposure");
    renderGraph_.markOutput("Clear.color");
    renderGraph_.markOutput("Data.values");
    renderGraph_.markOutput("Volume.volume");
    setActivePreviewOutput("Clear.color");
    renderGraphEditorOpen_ = true;
    resourceInspectorVisible_ = true;
    resourceInspectorSelectTab_ = true;
    auto& inspector = resourceInspector_;
    if (!expect(inspector.autoRange_, "Texture automatic range is enabled by default")) { return false; }
    // Keep the existing exact-color/manual-exposure regression independent of auto contrast.
    inspector.autoRange_ = false;
    {
        EditorResourceInspector probe;
        struct Cleanup {
            EditorResourceInspector& inspector;
            render::vulkan::VulkanImGuiBackend& backend;
            ~Cleanup() { inspector.shutdown(backend); }
        } cleanup{probe, imguiBackend_};
        std::vector<uint8_t> pixels;
        const auto previewValues = [&](std::initializer_list<float> values, const char* layout = "f32") {
            auto capture = std::make_shared<debug::DebugCapture>();
            debug::DebugArtifact artifact;
            artifact.layout = probe.runtime().layouts().at(layout);
            artifact.bytes.resize(values.size() * sizeof(float));
            std::memcpy(artifact.bytes.data(), values.begin(), artifact.bytes.size());
            artifact.metadata = {{"kind", "texture"}, {"roi", {{"width", artifact.bytes.size() / artifact.layout.stride}, {"height", 1}}}};
            capture->artifacts.push_back(std::move(artifact));
            probe.capture_ = std::move(capture);
            return probe.rebuildImage(*device_, imguiBackend_) &&
                readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *probe.image_, pixels, nullptr, &probe);
        };
        const auto gray = [&](size_t i, uint8_t value) {
            return pixels[i * 4] == value && pixels[i * 4 + 1] == value && pixels[i * 4 + 2] == value;
        };
        const auto magenta = [&](size_t i) { return pixels[i * 4] == 255 && pixels[i * 4 + 1] == 0 && pixels[i * 4 + 2] == 255; };
        if (!expect(previewValues({0.999f, 0.9992f, 0.9994f, 1.f, 1.f, 1.f}) && probe.rangeTrimmed_ &&
            gray(0, 0) && pixels[4] >= 127 && pixels[4] <= 128 && gray(2, 255), "Hardware depth expands a narrow finite interval on the GPU")) { return false; }
        if (!expect(previewValues({2, 3, 4, 100, 100, 100}) && probe.rangeMin_ == 2 && probe.rangeMax_ == 4 &&
            gray(0, 0) && gray(1, 128) && gray(2, 255), "Linear depth trims a repeated far plane without losing foreground contrast")) { return false; }
        probe.trimRange_ = false;
        if (!expect(previewValues({2, 3, 4, 100, 100, 100}) && probe.rangeMax_ == 100 && gray(3, 255) && pixels[8] < 10,
            "Full finite range includes the far plane when trimming is disabled")) { return false; }
        probe.trimRange_ = true;
        if (!expect(previewValues({0, 0, 1, 1}) && probe.rangeMin_ == 0 && probe.rangeMax_ == 1 && gray(0, 0) && gray(2, 255),
            "Two-valued masks keep both endpoints")) { return false; }
        if (!expect(previewValues({7, 7, 7}) && probe.rangeMin_ < 7 && probe.rangeMax_ > 7 && gray(0, 128),
            "Constant textures have a finite centered range")) { return false; }
        const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
        if (!expect(previewValues({-2, 0, 2, nan, inf, -inf}) && probe.rangeFinite_ == 3 && probe.rangeMin_ == -2 && probe.rangeMax_ == 2 &&
            gray(1, 128) && magenta(3) && magenta(4) && magenta(5), "NaN and infinity are excluded from bounds and remain magenta")) { return false; }
        if (!expect(previewValues({nan, inf, -inf}) && probe.rangeFinite_ == 0 && probe.rangeMin_ == 0 && probe.rangeMax_ == 1 && magenta(0),
            "All non-finite values reset the range instead of reusing stale bounds")) { return false; }
        const float largest = std::numeric_limits<float>::max(), tiny = std::numeric_limits<float>::denorm_min();
        if (!expect(previewValues({-largest, 0, largest}) && gray(0, 0) && gray(1, 128) && gray(2, 255),
            "Extreme finite floats normalize without overflow") ||
            !expect(previewValues({tiny, tiny * 2, tiny * 3}) && gray(0, 0) && gray(1, 128) && gray(2, 255),
            "Subnormal intervals retain their contrast")) { return false; }
        probe.trimRange_ = false;
        if (!expect(previewValues({2, 100, 0, 1000, 4, 200, 0, 2000}, "RGBA32F") && probe.rangeMax_ == 200,
            "RGB auto range excludes alpha")) { return false; }
        probe.channel_ = 1;
        if (!expect(previewValues({2, 100, 0, 1000, 4, 200, 0, 2000}, "RGBA32F") && probe.rangeMin_ == 2 && probe.rangeMax_ == 4,
            "Channel changes compute their own range")) { return false; }
        probe.channel_ = 4;
        if (!expect(previewValues({2, 100, 0, 1000, 4, 200, 0, 2000}, "RGBA32F") && probe.rangeMin_ == 1000 && probe.rangeMax_ == 2000,
            "Alpha can be ranged independently")) { return false; }
        probe.channel_ = 0; probe.autoRange_ = false; probe.rangeMin_ = -5; probe.rangeMax_ = 5;
        if (!expect(previewValues({-2, 0, 2}) && probe.rangeMin_ == -5 && probe.rangeMax_ == 5 && gray(1, 128),
            "Manual bounds survive replacement capture data")) { return false; }
        // Include every half encoding: motion vectors frequently use subnormals.
        // Compare actual GPU preview bytes against the original arithmetic decode.
        auto halfCapture = std::make_shared<debug::DebugCapture>();
        debug::DebugArtifact halfArtifact;
        halfArtifact.layout = probe.runtime().layouts().at("RG16F");
        halfArtifact.metadata = {{"kind", "texture"}, {"roi", {{"width", 256}, {"height", 256}}}};
        halfArtifact.bytes.resize(65536 * 4);
        for (uint32_t i = 0; i < 65536; ++i) {
            const uint16_t pair[] = {uint16_t(i), uint16_t(i)};
            std::memcpy(halfArtifact.bytes.data() + i * 4, pair, sizeof(pair));
        }
        halfCapture->artifacts.push_back(std::move(halfArtifact));
        probe.capture_ = std::move(halfCapture);
        probe.rangeMin_ = -0.00006; probe.rangeMax_ = 0.00006;
        if (!probe.rebuildImage(*device_, imguiBackend_) ||
            !readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *probe.image_, pixels, nullptr, &probe)) { return false; }
        bool halfMatches = true;
        for (uint32_t i = 0; i < 65536; ++i) {
            const int exponent = (i >> 10) & 31, mantissa = i & 1023;
            if (exponent == 31) { halfMatches &= magenta(i); continue; }
            const float magnitude = std::ldexp(exponent ? 1.0f + mantissa / 1024.0f : mantissa / 1024.0f, exponent ? exponent - 15 : -14);
            const double value = i & 0x8000 ? -magnitude : magnitude;
            const auto expected = uint8_t(std::clamp((value - probe.rangeMin_) / (probe.rangeMax_ - probe.rangeMin_), 0.0, 1.0) * 255.0 + 0.5);
            halfMatches &= pixels[i * 4] == expected && pixels[i * 4 + 1] == expected && pixels[i * 4 + 2] == 128;
        }
        if (!expect(halfMatches, "All 65536 RG16F encodings preserve subnormals, finite values and non-finite preview colors")) { return false; }
    }
    const auto exportCapture = [&](const char* name) {
        const auto& capture = *inspector.capture_;
        const auto& artifact = capture.artifacts[0];
        const auto directory = output / name;
        std::filesystem::create_directories(directory, error);
        if (error) { return false; }
        std::ofstream manifest(directory / "manifest.json");
        manifest << debug::encodeLossless(capture.manifest()).dump(2);
        std::ofstream data(directory / "0.bin", std::ios::binary);
        data.write(reinterpret_cast<const char*>(artifact.bytes.data()), artifact.bytes.size());
        report.value["artifacts"].push_back({{"name", name}, {"resource", artifact.metadata.at("id")},
            {"bytes", artifact.bytes.size()}, {"execution", capture.snapshot.evidence.execution}, {"generation", capture.snapshot.evidence.generation}});
        return manifest.good() && data.good();
    };
    const auto exportUI = [&](const char* name) {
        auto* window = ImGui::FindWindowByName("Render Graph Editor");
        if (!window || !window->Viewport->DrawData) { return false; }
        ImDrawData draw = *window->Viewport->DrawData;
        draw.DisplayPos = window->Pos; draw.DisplaySize = window->Size;
        const uint32_t width = uint32_t(draw.DisplaySize.x * draw.FramebufferScale.x);
        const uint32_t height = uint32_t(draw.DisplaySize.y * draw.FramebufferScale.y);
        const auto format = displayOutput_.mode == render::DisplayOutputMode::HDR10_PQ ? render::Format::RGBA16Sfloat : swapchain_->format();
        if (format != render::Format::RGBA16Sfloat && format != render::Format::BGRA8Unorm && format != render::Format::BGRA8sRGB &&
            format != render::Format::RGBA8Unorm && format != render::Format::RGBA8sRGB) { return false; }
        auto image = device_->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
            .format = format, .width = width, .height = height});
        std::vector<uint8_t> pixels;
        if (!image || !readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, **image, pixels, &draw)) { return false; }
        report.value["artifacts"].push_back({{"name", name}, {"kind", "native-imgui-render"}, {"width", width}, {"height", height}});
        return writePPM(output / (std::string(name) + ".ppm"), pixels, width, height, format, displayOutput_.paperWhiteNits);
    };
    inspector.select("Clear.color");
    const auto frames = [&](int count) {
        for (int i = 0; i < count; ++i) {
            if (!renderFrame() || !frameSubmissions_.wait()) { expect(false, "Editor frame or GPU completion failed"); return false; }
        }
        return true;
    };
    if (!frames(8) || !expect(inspector.capture_ && inspector.descriptor_, "Texture snapshot and ImGui preview were created")) { return false; }
    const auto& texture = inspector.capture_->artifacts[0];
    spdlog::info("[Smoke Resource Inspector] Captured texture layout={} bytes={} first={}", texture.layout.name,
        texture.bytes.size(), debug::hexEncode(std::span(texture.bytes).first(std::min(size_t(16), texture.bytes.size()))));
    if (!expect(texture.layout.name == "RGBA8" && texture.bytes.size() >= 4, "RGBA8 capture layout") ||
        !expect(texture.bytes[0] == 64 && (texture.bytes[1] == 127 || texture.bytes[1] == 128) && texture.bytes[2] == 191 && texture.bytes[3] == 255,
            "Texture readback matches the known GPU clear color")) { return false; }
    // Inspect the actual display conversion and RHI upload, after its submission completed.
    if (!expect(!inspector.upload_, "Preview upload was recorded") || !expect(inspector.image_->desc().width > 0, "Preview image has dimensions")) { return false; }
    std::vector<uint8_t> preview;
    if (!expect(readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview), "Read back the actual GPU preview texture") ||
        !expect(preview == texture.bytes, "Default display conversion preserves RGBA8 pixels") ||
        !expect(exportCapture("texture") && exportUI("texture-ui"), "Export texture evidence and native UI rendering")) { return false; }
    const auto frozen = inspector.capture_;
    if (!frames(3) || !expect(inspector.capture_ == frozen, "Frozen snapshot remains unchanged across frames")) { return false; }
    {
        // Drive ImGui events directly; never move the desktop cursor. The
        // canvas geometry comes from the real widget, including DPI/layout.
        auto& io = ImGui::GetIO();
        const auto oldMouse = io.MousePos;
        const bool oldTrickle = io.ConfigInputTrickleEventQueue, oldFocus = !io.AppFocusLost;
        io.ConfigInputTrickleEventQueue = false;
        struct RestoreInput {
            ImGuiIO& io; ImVec2 mouse; bool trickle, focused;
            ~RestoreInput() {
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
                io.AddMouseButtonEvent(ImGuiMouseButton_Middle, false);
                io.AddMousePosEvent(mouse.x, mouse.y); io.AddFocusEvent(focused);
                io.ConfigInputTrickleEventQueue = trickle;
            }
        } restore{io, oldMouse, oldTrickle, oldFocus};
        ImRect canvas;
        ImGuiWindow* testWindow = nullptr;
        bool tooltip = false;
        const auto interact = [&](ImVec2 mouse, float wheel = 0, bool down = false) {
            io.AddFocusEvent(true); io.AddMousePosEvent(mouse.x, mouse.y);
            io.AddMouseButtonEvent(ImGuiMouseButton_Left, down);
            if (wheel) { io.AddMouseWheelEvent(0, wheel); }
            io.DeltaTime = 1.f / 60;
            ImGui::NewFrame();
            ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
            ImGui::SetNextWindowPos(ImVec2(20, 20)); ImGui::SetNextWindowSize(ImVec2(1000, 750));
            ImGui::Begin("Texture navigation regression", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking);
            testWindow = ImGui::GetCurrentWindow();
            inspector.drawTexture(*device_, imguiBackend_);
            for (auto* window : GImGui->Windows) {
                if (window->ParentWindow == testWindow && std::strstr(window->Name, "TexturePixels")) {
                    canvas = ImRect(window->DC.CursorStartPos, window->WorkRect.Max);
                }
            }
            tooltip = GImGui->TooltipPreviousWindow && GImGui->TooltipPreviousWindow->LastFrameActive == ImGui::GetFrameCount();
            ImGui::End(); ImGui::Render();
        };
        interact(ImVec2(-1000, -1000));
        interact(ImVec2(-1000, -1000));
        const auto middle = canvas.GetCenter();
        const ImVec2 anchor(middle.x + canvas.GetWidth() * 0.1f, middle.y);
        interact(anchor);
        const float fitZoom = inspector.zoom_;
        const float anchorU = inspector.imageCenter_[0] + (anchor.x - middle.x) / (texture.metadata.at("roi").at("width").get<float>() * inspector.zoom_);
        const auto image = inspector.image_.get();
        const auto descriptor = inspector.descriptor_;
        interact(anchor, 3);
        const float afterU = inspector.imageCenter_[0] + (anchor.x - middle.x) / (texture.metadata.at("roi").at("width").get<float>() * inspector.zoom_);
        if (!expect(!inspector.fit_ && std::abs(inspector.zoom_ - fitZoom * std::pow(1.2f, 3.f)) < 0.001f &&
            std::abs(anchorU - afterU) < 0.0001f, "Wheel zoom leaves the cursor's source texel anchored")) { return false; }
        interact(anchor, -3);
        if (!expect(std::abs(inspector.zoom_ - fitZoom) < 0.001f, "Wheel zoom is reversible")) { return false; }
        interact(anchor, 4);
        interact(anchor, 0, true);
        const float panStart = inspector.imageCenter_[0];
        interact(ImVec2(anchor.x + 30, anchor.y), 0, true);
        if (!expect(inspector.imageCenter_[0] < panStart, "Dragging pans the magnified image")) { return false; }
        interact(ImVec2(anchor.x + 30, anchor.y));
        interact(ImVec2(anchor.x + 30, anchor.y));
        if (!expect(tooltip, "Magnified pixels retain the raw-value hover tooltip")) { return false; }
        const float beforeOutside = inspector.zoom_;
        interact(ImVec2(10, 10)); interact(ImVec2(10, 10), 2);
        if (!expect(inspector.zoom_ == beforeOutside, "Wheel outside the preview does not zoom")) { return false; }
        interact(middle, 1);
        if (!expect(std::abs(inspector.zoom_ - beforeOutside * 1.2f) < 0.001f,
            "Mouse entry and wheel in the same frame zoom the preview")) { return false; }
        interact(middle); interact(middle, 100);
        if (!expect(inspector.zoom_ == std::max(64.f, fitZoom), "Wheel zoom has a bounded maximum")) { return false; }
        interact(middle, -200);
        if (!expect(std::abs(inspector.zoom_ - std::min(0.01f, fitZoom)) < 1e-6f, "Wheel zoom has a positive minimum")) { return false; }
        // Separate the reset gesture from the preceding drag's click history.
        for (int i = 0; i < 30; ++i) { interact(middle); }
        interact(middle, 0, true); interact(middle); interact(middle, 0, true); interact(middle);
        if (!expect(inspector.fit_ && std::abs(inspector.zoom_ - fitZoom) < 0.001f &&
            inspector.imageCenter_ == std::array<float, 2>{0.5f, 0.5f}, "Double-click restores centered Fit view") ||
            !expect(inspector.capture_ == frozen && inspector.image_.get() == image && inspector.descriptor_ == descriptor &&
                !inspector.upload_ && !inspector.imageDirty_, "Navigation reuses the capture and GPU preview without conversion/upload")) { return false; }
    }
    inspector.channel_ = 1; inspector.imageDirty_ = true;
    if (!frames(2) || !readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview) ||
        !expect(preview[0] == 64 && preview[1] == 64 && preview[2] == 64, "R channel renders as grayscale on the GPU")) { return false; }
    inspector.channel_ = 0; inspector.exposure_ = 1; inspector.imageDirty_ = true;
    if (!frames(2) || !readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview) ||
        !expect(preview[0] == 128 && preview[2] == 255, "Exposure and clamping reach the GPU preview")) { return false; }
    inspector.exposure_ = 0; inspector.imageDirty_ = true;
    inspector.live_ = true; inspector.nextRefresh_ = 0;
    if (!frames(5) || !expect(inspector.capture_->snapshot.evidence.execution > frozen->snapshot.evidence.execution,
        "Live refresh obtains a newer completed execution")) { return false; }
    inspector.live_ = false;
    inspector.roi_ = {2, 3, 8, 4}; inspector.refresh_ = true;
    if (!frames(5) || !expect(inspector.capture_->artifacts[0].bytes.size() == 8 * 4 * 4, "ROI copies the requested pixel rectangle")) { return false; }
    inspector.select("Volume.volume");
    inspector.cube_ = false;
    for (const int slice : {0, 2, 3}) {
        inspector.slice_ = slice; inspector.refresh_ = true;
        if (!frames(5) || !expect(inspector.capture_ != nullptr, "Volume slice capture ready")) { return false; }
        const auto& volume = inspector.capture_->artifacts[0];
        bool matches = volume.bytes.size() == 8 * 6 * 4 && volume.metadata.at("slice") == slice &&
            !volume.metadata.at("completeCoverage").get<bool>();
        for (size_t y = 0; matches && y < 6; ++y) {
            for (size_t x = 0; x < 8; ++x) {
                const size_t i = (y * 8 + x) * 4;
                matches &= volume.bytes[i] == x * 25 && volume.bytes[i + 1] == y * 35 &&
                    volume.bytes[i + 2] == slice * 70 && volume.bytes[i + 3] == 255;
            }
        }
        if (!expect(matches, "Volume first/middle/last slice preserves XYZ voxel values and partial coverage") ||
            !expect(readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview) &&
                preview == volume.bytes, "Volume slice reaches the actual GPU preview")) { return false; }
    }
    if (!expect(exportCapture("volume") && exportUI("volume-ui"), "Export volume slice and native UI evidence")) { return false; }
    inspector.roi_ = {2, 3, 3, 2}; inspector.refresh_ = true;
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].bytes.size() == 24 &&
        inspector.capture_->artifacts[0].bytes[0] == 50 && inspector.capture_->artifacts[0].bytes[1] == 105 &&
        inspector.capture_->artifacts[0].bytes[2] == 210, "Volume ROI retains the selected Z slice")) { return false; }
    {
        // Exercise sampling and grid drawing through the native Vulkan backend.
        // The ROI ramp has distinct neighboring texels and nonzero XYZ offsets.
        const auto capture = inspector.capture_;
        const auto image = inspector.image_.get();
        const auto descriptor = inspector.descriptor_;
        auto& io = ImGui::GetIO();
        struct RestoreInput {
            ImGuiIO& io; ImVec2 mouse; bool focused;
            ~RestoreInput() { io.AddMousePosEvent(mouse.x, mouse.y); io.AddFocusEvent(focused); }
        } restore{io, io.MousePos, !io.AppFocusLost};
        inspector.fit_ = false; inspector.zoom_ = 32; inspector.pixelGrid_ = false;
        ImGuiWindow* window = nullptr;
        ImVec2 origin;
        std::string logged;
        const auto drawPreview = [&](ImVec2 mouse) {
            io.AddFocusEvent(true); io.AddMousePosEvent(mouse.x, mouse.y);
            ImGui::NewFrame();
            ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
            ImGui::SetNextWindowPos(ImVec2(20, 20)); ImGui::SetNextWindowSize(ImVec2(1000, 750));
            ImGui::Begin("Pixel inspection regression", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking);
            window = ImGui::GetCurrentWindow();
            ImGui::LogToBuffer();
            inspector.drawTexture(*device_, imguiBackend_);
            logged = GImGui->LogBuffer.c_str();
            ImGui::LogFinish();
            for (auto* child : GImGui->Windows) {
                if (child->ParentWindow == window && std::strstr(child->Name, "TexturePixels")) {
                    const auto center = ImRect(child->DC.CursorStartPos, child->WorkRect.Max).GetCenter();
                    origin = ImVec2(center.x - 1.5f * inspector.zoom_, center.y - inspector.zoom_);
                }
            }
            ImGui::End(); ImGui::Render();
        };
        const auto format = displayOutput_.mode == render::DisplayOutputMode::HDR10_PQ ? render::Format::RGBA16Sfloat : swapchain_->format();
        const size_t pixelBytes = format == render::Format::RGBA16Sfloat ? 8 : 4;
        uint32_t width = 0, height = 0;
        ImDrawData draw;
        const auto readPreview = [&](std::vector<uint8_t>& pixels) {
            draw = *window->Viewport->DrawData;
            draw.DisplayPos = window->Pos; draw.DisplaySize = window->Size;
            width = uint32_t(draw.DisplaySize.x * draw.FramebufferScale.x);
            height = uint32_t(draw.DisplaySize.y * draw.FramebufferScale.y);
            auto target = device_->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = format, .width = width, .height = height});
            return target && readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, **target, pixels, &draw);
        };
        const auto pixel = [&](const std::vector<uint8_t>& pixels, float u, float v) {
            const auto x = uint32_t((origin.x + u * inspector.zoom_ - draw.DisplayPos.x) * draw.FramebufferScale.x);
            const auto y = uint32_t((origin.y + v * inspector.zoom_ - draw.DisplayPos.y) * draw.FramebufferScale.y);
            return debug::hexEncode(std::span(pixels).subspan((size_t(y) * width + x) * pixelBytes, pixelBytes));
        };
        drawPreview(ImVec2(-1000, -1000)); drawPreview(ImVec2(-1000, -1000));
        std::vector<uint8_t> nearest, linear, grid;
        if (!expect(readPreview(nearest), "Render nearest-sampled pixel fixture through Vulkan") ||
            !expect(pixel(nearest, 1.15f, 0.5f) == pixel(nearest, 1.85f, 0.5f) &&
                pixel(nearest, 0.5f, 0.5f) != pixel(nearest, 1.5f, 0.5f),
                "Nearest sampling preserves flat texel interiors and distinct neighbors")) { return false; }
        inspector.nearest_ = false; drawPreview(ImVec2(-1000, -1000));
        if (!expect(readPreview(linear) && pixel(linear, 1.15f, 0.5f) != pixel(linear, 1.85f, 0.5f),
            "Disabling nearest restores linear interpolation on the same GPU preview")) { return false; }
        inspector.nearest_ = true; inspector.pixelGrid_ = true; drawPreview(ImVec2(-1000, -1000));
        if (!expect(readPreview(grid) && pixel(grid, 1.5f, 0.5f) == pixel(nearest, 1.5f, 0.5f) &&
            pixel(grid, 1.f, 0.5f) != pixel(nearest, 1.f, 0.5f) &&
            pixel(grid, 1.5f, 1.f) != pixel(nearest, 1.5f, 1.f),
            "Pixel grid marks horizontal and vertical texel boundaries without changing interiors")) { return false; }
        if (!expect(writePPM(output / "pixel-grid-ui.ppm", grid, width, height, format, displayOutput_.paperWhiteNits),
            "Export native nearest-sampled pixel grid")) { return false; }
        const ImVec2 hover(origin.x + 1.5f * inspector.zoom_, origin.y + 0.5f * inspector.zoom_);
        drawPreview(hover); drawPreview(hover);
        const auto bytes = debug::hexEncode(std::span(capture->artifacts[0].bytes).subspan(4, 4));
        if (!expect(logged.find("Voxel (3, 3, 3)") != std::string::npos && logged.find("Bytes: " + bytes) != std::string::npos,
            "Magnified hover reports the exact voxel coordinates and raw bytes including ROI offsets") ||
            !expect(readPreview(grid) && writePPM(output / "pixel-hover-ui.ppm", grid, width, height, format, displayOutput_.paperWhiteNits),
                "Export native highlighted pixel and raw-value tooltip")) { return false; }
        // A fractional zoom keeps grid lines registered to the image boundaries.
        inspector.zoom_ = 19.3f; drawPreview(ImVec2(-1000, -1000));
        if (!readPreview(grid)) { return false; }
        inspector.pixelGrid_ = false; drawPreview(ImVec2(-1000, -1000));
        if (!expect(readPreview(nearest) && pixel(grid, 1.f, 0.5f) != pixel(nearest, 1.f, 0.5f) &&
            pixel(grid, 1.5f, 0.5f) == pixel(nearest, 1.5f, 0.5f), "Fractional zoom keeps the pixel grid aligned")) { return false; }
        inspector.zoom_ = 4; drawPreview(ImVec2(-1000, -1000));
        if (!readPreview(nearest)) { return false; }
        inspector.pixelGrid_ = true; drawPreview(ImVec2(-1000, -1000));
        if (!expect(readPreview(grid) && pixel(grid, 1.f, 0.5f) == pixel(nearest, 1.f, 0.5f),
            "Pixel grid stays hidden below 8x") ||
            !expect(inspector.capture_ == capture && inspector.image_.get() == image && inspector.descriptor_ == descriptor &&
                !inspector.upload_ && !inspector.imageDirty_, "Sampling and grid changes reuse the capture without conversion or upload")) { return false; }
        inspector.fit_ = true;
    }
    inspector.cube_ = true; inspector.roi_ = {0, 0, 0, 0}; inspector.capture_.reset(); inspector.refresh_ = true;
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].bytes.size() == 8 * 6 * 4 * 4 &&
        inspector.capture_->artifacts[0].metadata.at("completeCoverage") == true, "Cube captures the full volume")) { return false; }
    if (!expect(readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview), "Read back cube face atlas")) { return false; }
    bool atlasMatches = preview.size() == 20 * 12 * 4;
    for (uint32_t axis = 0; atlasMatches && axis < 3; ++axis) {
        const uint32_t width = axis == 0 ? 4 : 8, height = axis == 1 ? 4 : 6;
        const uint32_t left = axis == 0 ? 0 : 4 + (axis - 1) * 8;
        for (uint32_t side = 0; side < 2; ++side) {
            for (uint32_t v = 0; v < height; ++v) {
                for (uint32_t u = 0; u < width; ++u) {
                    const auto i = ((side * 6 + v) * 20 + left + u) * 4;
                    atlasMatches &= preview[i] == (axis == 0 ? side * 7 : u) * 25 &&
                        preview[i + 1] == (axis == 1 ? side * 5 : v) * 35 &&
                        preview[i + 2] == (axis == 2 ? side * 3 : axis == 0 ? u : v) * 70 && preview[i + 3] == 255;
                }
            }
        }
    }
    if (!expect(atlasMatches, "All six GPU cube faces match their XYZ boundary slices") ||
        !expect(exportCapture("cube") && exportUI("cube-ui"), "Export cube and native UI evidence")) { return false; }
    const auto cubeCapture = inspector.capture_;
    inspector.cubeYaw_ += 3.14159265f; inspector.cubePitch_ = -0.45f;
    if (!frames(2) || !expect(inspector.capture_ == cubeCapture && exportUI("cube-rotated-ui"),
        "Rotate to opposite cube faces without recapturing")) { return false; }
    auto& captureRuntime = debugRuntime_ ? *debugRuntime_ : inspector.runtime();
    for (const int invalidCount : {0, 5}) {
        const auto queued = captureRuntime.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", "Volume"}, {"resources", debug::DebugValue::array({{{"id", "Volume.volume"}, {"sliceCount", invalidCount}}})}}}});
        if (!expect(queued.at("status") == "ok", "Queue invalid volume depth for validation") || !frames(3)) { return false; }
        const auto job = captureRuntime.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", queued.at("result").at("job")}}}});
        if (!expect(job.at("result").at("state") == "Failed" && job.at("result").at("error").at("code") == "OutOfRange",
            "Empty or oversized volume depth rejected before GPU copy")) { return false; }
    }
    for (const int invalidSlice : {-1, 4}) {
        const auto queued = captureRuntime.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", "Volume"}, {"resources", debug::DebugValue::array({{{"id", "Volume.volume"}, {"slice", invalidSlice}}})}}}});
        if (!expect(queued.at("status") == "ok", "Queue invalid volume slice for preflight validation") || !frames(3)) { return false; }
        const auto job = captureRuntime.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", queued.at("result").at("job")}}}});
        if (!expect(job.at("result").at("state") == "Failed" &&
            job.at("result").at("error").at("code") == (invalidSlice < 0 ? "InvalidArgument" : "OutOfRange"),
            "Invalid volume slice rejected before GPU copy")) { return false; }
    }
    inspector.select("subsystem.test.inspector.values");
    inspector.sourceFilter_ = 2;
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].metadata.at("subsystem") == "test.inspector" &&
        inspector.capture_->artifacts[0].metadata.at("checkpoint") == "AfterGraph", "Subsystem buffer is captured at the joined AfterGraph boundary")) { return false; }
    auto subsystemValues = debug::decodeBuffer(inspector.capture_->artifacts[0].bytes, inspector.capture_->artifacts[0].layout);
    if (!expect(subsystemValues && subsystemValues->size() == 4 && subsystemValues->at(0) == 100 &&
        subsystemValues->at(3) == 103, "Subsystem typed GPU values match the independent allocation") ||
        !expect(exportCapture("subsystem-buffer") && exportUI("subsystem-buffer-ui"), "Export grouped subsystem buffer evidence")) { return false; }
    auto* fixture = subsystemHost_.get<InspectorResourceSubsystem>();
    const auto subsystemCapture = inspector.capture_;
    fixture->replace = true;
    if (!frames(7) || !expect(inspector.capture_ && inspector.capture_ != subsystemCapture &&
        inspector.capture_->artifacts[0].metadata.at("allocation") == 2, "Subsystem allocation replacement refreshes frozen evidence")) { return false; }
    subsystemValues = debug::decodeBuffer(inspector.capture_->artifacts[0].bytes, inspector.capture_->artifacts[0].layout);
    if (!expect(subsystemValues && subsystemValues->at(0) == 200, "Replacement buffer captures new contents")) { return false; }
    fixture->visible = false;
    if (!frames(4) || !expect(!inspector.capture_ && inspector.job_.empty(), "Disappearing subsystem resources discard old evidence")) { return false; }
    fixture->visible = true;
    if (!frames(6) || !expect(inspector.capture_ != nullptr, "Returning subsystem resources can be captured again")) { return false; }
    inspector.select("subsystem.render.environment.radiance");
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].metadata.at("subsystem") == "render.environment" &&
        inspector.capture_->artifacts[0].layout.name == "RGBA32F" && inspector.descriptor_, "Real environment radiance supports texture inspection") ||
        !expect(exportCapture("subsystem-environment") && exportUI("subsystem-environment-ui"), "Export environment texture and subsystem UI")) { return false; }
    inspector.select("subsystem.render.environment.sphericalHarmonics");
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "float4" &&
        inspector.capture_->artifacts[0].bytes.size() == 9 * 16, "Real environment SH supports typed buffer inspection")) { return false; }
    {
        using Coefficients = EditorResourceInspector::SHCoefficients;
        constexpr double pi = std::numbers::pi;
        const double dc = std::sqrt(1.0 / (4.0 * pi)), linear = std::sqrt(3.0 / (4.0 * pi));
        Coefficients coefficients{};
        if (!expect(inspector.readSH(coefficients) && inspector.shSphere_,
            "Environment producer metadata enables the full irradiance SH sphere") ||
            !expect(exportCapture("environment-sh") && exportUI("environment-sh-sphere-ui"),
                "Export real environment SH coefficients and native sphere")) { return false; }
        const auto frozenSH = inspector.capture_;
        inspector.shYaw_ = 1.2f;
        if (!frames(2) || !expect(inspector.capture_ == frozenSH && exportUI("environment-sh-rotated-ui"),
            "SH sphere rotation reuses frozen environment coefficients")) { return false; }
        inspector.shSphere_ = false;
        if (!frames(2) || !expect(exportUI("environment-sh-table-ui"), "SH coefficients remain available in the typed table")) { return false; }
        inspector.shSphere_ = true; inspector.offset_ = 1; inspector.count_ = 8; inspector.refresh_ = true;
        if (!frames(5) || !expect(inspector.capture_ && !inspector.readSH(coefficients),
            "Partial SH captures cannot be interpreted as a complete sphere")) { return false; }
        inspector.offset_ = 0; inspector.count_ = 9; inspector.refresh_ = true;
        if (!frames(5) || !expect(inspector.readSH(coefficients), "Refreshing all nine coefficients restores the SH sphere")) { return false; }

        // Independent analytic basis values exercise signs, ordering and packing.
        const std::array<std::array<double, 3>, 7> directions{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
            {0, 0, 1}, {0, 0, -1}, {2.0 / 3, -1.0 / 3, 2.0 / 3}}};
        bool basisMatches = true;
        for (const auto& n : directions) {
            const auto [x, y, z] = n;
            const std::array<double, 9> basis{dc, linear * y, linear * z, linear * x,
                std::sqrt(15.0 / (4 * pi)) * x * y, std::sqrt(15.0 / (4 * pi)) * y * z,
                std::sqrt(5.0 / (16 * pi)) * (3 * z * z - 1), std::sqrt(15.0 / (4 * pi)) * x * z,
                std::sqrt(15.0 / (16 * pi)) * (x * x - y * y)};
            for (size_t i = 0; i < 9; ++i) {
                coefficients = {};
                coefficients[i] = {1, -2, 3, std::numeric_limits<float>::quiet_NaN()};
                const auto value = EditorResourceInspector::evaluateSH(coefficients, n);
                basisMatches &= std::abs(value[0] - basis[i]) < 1e-8 && std::abs(value[1] + 2 * basis[i]) < 1e-8 &&
                    std::abs(value[2] - 3 * basis[i]) < 1e-8;
            }
        }
        if (!expect(basisMatches, "SH evaluation matches independent L2 basis values, signs, RGB channels and ignores reserved W")) { return false; }
        coefficients = {};
        coefficients[0] = {float(pi * 0.25 / dc), float(pi * 0.5 / dc), float(pi * 0.75 / dc), 0};
        bool constantMatches = true;
        for (const auto& n : directions) {
            const auto value = EditorResourceInspector::evaluateSH(coefficients, n);
            for (size_t c = 0; c < 3; ++c) { constantMatches &= std::abs(value[c] / pi - (c + 1) * 0.25) < 1e-7; }
        }
        if (!expect(constantMatches, "Constant irradiance gives a direction-independent white Lambert response without double convolution")) { return false; }

        EditorResourceInspector probe;
        auto capture = std::make_shared<debug::DebugCapture>();
        debug::DebugArtifact artifact;
        artifact.layout = probe.runtime().layouts().at("float4");
        artifact.metadata = {{"visualization", "sh-irradiance-l2-rgb"}, {"elementOffset", 0}};
        artifact.bytes.resize(sizeof(coefficients));
        std::memcpy(artifact.bytes.data(), coefficients.data(), sizeof(coefficients));
        capture->artifacts.push_back(std::move(artifact)); probe.capture_ = capture;
        Coefficients decoded;
        if (!expect(probe.readSH(decoded) && decoded == coefficients, "SH sphere decodes the explicit nine-float4 capture ABI")) { return false; }
        capture->artifacts[0].metadata.erase("visualization");
        if (!expect(!probe.readSH(decoded), "Ordinary float4 buffers are not guessed to be spherical harmonics")) { return false; }
        capture->artifacts[0].metadata["visualization"] = "sh-irradiance-l2-rgb";
        probe.shNormalize_ = false;
        auto& io = ImGui::GetIO();
        struct RestoreInput {
            ImGuiIO& io; ImVec2 mouse; bool focused, trickle;
            ~RestoreInput() {
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); io.AddMousePosEvent(mouse.x, mouse.y);
                io.AddFocusEvent(focused); io.ConfigInputTrickleEventQueue = trickle;
            }
        } restore{io, io.MousePos, !io.AppFocusLost, io.ConfigInputTrickleEventQueue};
        io.ConfigInputTrickleEventQueue = false;
        ImGuiWindow* window = nullptr;
        ImRect canvas;
        std::string logged;
        const auto drawSphere = [&](ImVec2 mouse, bool down = false, float wheel = 0) {
            io.AddFocusEvent(true); io.AddMousePosEvent(mouse.x, mouse.y); io.AddMouseButtonEvent(ImGuiMouseButton_Left, down);
            if (wheel) { io.AddMouseWheelEvent(0, wheel); }
            ImGui::NewFrame();
            ImGui::SetNextWindowViewport(ImGui::GetMainViewport()->ID);
            ImGui::SetNextWindowPos(ImVec2(20, 20)); ImGui::SetNextWindowSize(ImVec2(1000, 750));
            ImGui::Begin("SH sphere regression", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking);
            window = ImGui::GetCurrentWindow(); ImGui::LogToBuffer(); probe.drawBuffer();
            logged = GImGui->LogBuffer.c_str(); ImGui::LogFinish();
            for (auto* child : GImGui->Windows) {
                if (child->ParentWindow == window && std::strstr(child->Name, "SHSphere")) { canvas = ImRect(child->DC.CursorStartPos, child->WorkRect.Max); }
            }
            ImGui::End(); ImGui::Render();
        };
        const auto format = displayOutput_.mode == render::DisplayOutputMode::HDR10_PQ ? render::Format::RGBA16Sfloat : swapchain_->format();
        const size_t pixelBytes = format == render::Format::RGBA16Sfloat ? 8 : 4;
        uint32_t width = 0, height = 0;
        ImDrawData draw;
        const auto readSphere = [&](std::vector<uint8_t>& pixels) {
            draw = *window->Viewport->DrawData; draw.DisplayPos = window->Pos; draw.DisplaySize = window->Size;
            width = uint32_t(draw.DisplaySize.x * draw.FramebufferScale.x); height = uint32_t(draw.DisplaySize.y * draw.FramebufferScale.y);
            auto target = device_->createTexture({.usage = render::TextureUsageBits::ColorAttachment | render::TextureUsageBits::TransferSource,
                .format = format, .width = width, .height = height});
            return target && readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, **target, pixels, &draw);
        };
        const auto pixel = [&](const std::vector<uint8_t>& pixels, ImVec2 p) {
            const auto x = uint32_t((p.x - draw.DisplayPos.x) * draw.FramebufferScale.x);
            const auto y = uint32_t((p.y - draw.DisplayPos.y) * draw.FramebufferScale.y);
            return debug::hexEncode(std::span(pixels).subspan((size_t(y) * width + x) * pixelBytes, pixelBytes));
        };
        drawSphere(ImVec2(-1000, -1000)); drawSphere(ImVec2(-1000, -1000));
        const auto center = canvas.GetCenter();
        const float radius = std::min(canvas.GetWidth(), canvas.GetHeight()) * 0.45f;
        std::vector<uint8_t> pixels, rotated;
        if (!expect(readSphere(pixels) && pixel(pixels, center) == pixel(pixels, ImVec2(center.x + radius * 0.5f, center.y)) &&
            pixel(pixels, center) != pixel(pixels, ImVec2(canvas.Min.x + 4, canvas.Min.y + 4)),
            "Native Vulkan sphere renders constant SH uniformly with a distinct background")) { return false; }
        // Add a +Z lobe to red. Rotating to -Z must change its Lambert response.
        coefficients[2][0] = float(pi * 0.5 / linear);
        std::memcpy(capture->artifacts[0].bytes.data(), coefficients.data(), sizeof(coefficients));
        drawSphere(ImVec2(-1000, -1000));
        if (!readSphere(pixels)) { return false; }
        probe.shYaw_ = float(pi); drawSphere(ImVec2(-1000, -1000));
        if (!expect(readSphere(rotated) && pixel(pixels, center) != pixel(rotated, center),
            "Native sphere rotation evaluates the opposite SH hemisphere")) { return false; }
        probe.shYaw_ = 0; drawSphere(center); drawSphere(center);
        const auto raw = logged.find("Irradiance E:");
        std::array<double, 3> value{};
        const bool parsed = raw != std::string::npos && std::sscanf(logged.c_str() + raw, "Irradiance E: (%lf, %lf, %lf)", &value[0], &value[1], &value[2]) == 3;
        if (!expect(parsed && logged.find("N: (0.000000, 0.000000, 1.000000)") != std::string::npos &&
            std::abs(value[0] - pi * 0.75) < 1e-6 && std::abs(value[1] - pi * 0.5) < 1e-6 && std::abs(value[2] - pi * 0.75) < 1e-6,
            "SH sphere hover reports the exact normal and unmodified irradiance") ||
            !expect(readSphere(pixels) && writePPM(output / "sh-sphere-hover-ui.ppm", pixels, width, height, format, displayOutput_.paperWhiteNits),
                "Export native analytic SH sphere and raw irradiance tooltip")) { return false; }
        drawSphere(center, true); drawSphere(ImVec2(center.x + 30, center.y + 15), true);
        if (!expect(probe.shYaw_ != 0 && probe.shPitch_ != 0, "Native mouse drag rotates the SH sampling directions")) { return false; }
        drawSphere(center); drawSphere(center, false, 1);
        if (!expect(std::abs(probe.shZoom_ - 1.2f) < 1e-6f && probe.capture_ == capture && !probe.image_ && !probe.upload_,
            "SH wheel zoom and rotation reuse coefficient evidence without a preview texture upload")) { return false; }
    }
    inspector.select("subsystem.render.environment.pdf");
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "f32" && inspector.descriptor_,
        "Environment PDF supports single-channel texture inspection")) { return false; }
    inspector.select("subsystem.render.gpu-scene.instances");
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "GPUSceneGPUInstanceRecord" &&
        inspector.capture_->artifacts[0].bytes.size() == sizeof(render::GPUSceneGPUInstanceRecord),
        "GPUScene publishes its typed empty-scene sentinel independently of raster passes")) { return false; }
    const auto subsystemSnapshot = captureRuntime.core().latestSnapshot();
    std::string transmittance;
    if (subsystemSnapshot) {
        for (auto it = subsystemSnapshot->values.at("resources").begin(); it != subsystemSnapshot->values.at("resources").end(); ++it) {
            if (it.key().starts_with("subsystem.render.environment.atmosphere.") && it.key().ends_with(".transmittance")) { transmittance = it.key(); break; }
        }
    }
    if (!expect(!transmittance.empty(), "Current physical atmosphere publishes its LUT resources")) { return false; }
    inspector.select(transmittance);
    if (!frames(6) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].bytes.size() == 256 * 64 * 8 &&
        inspector.capture_->artifacts[0].metadata.at("environmentSource") == "PhysicalAtmosphere",
        "Physical atmosphere transmittance LUT supports GPU readback") ||
        !expect(exportCapture("subsystem-atmosphere") && exportUI("subsystem-atmosphere-ui"), "Export physical atmosphere LUT and grouped UI")) { return false; }
    inspector.select(transmittance.substr(0, transmittance.size() - std::string("transmittance").size()) + "sphericalHarmonics");
    EditorResourceInspector::SHCoefficients atmosphereSH;
    if (!frames(6) || !expect(inspector.capture_ && inspector.readSH(atmosphereSH) &&
        inspector.capture_->artifacts[0].metadata.at("environmentSource") == "PhysicalAtmosphere",
        "Physical atmosphere publishes the same explicit irradiance SH visualization contract") ||
        !expect(exportCapture("atmosphere-sh") && exportUI("atmosphere-sh-sphere-ui"), "Export native physical-atmosphere SH sphere")) { return false; }
    inspector.sourceFilter_ = 0;
    inspector.select("Data.values");
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].metadata.at("id") == "Data.values", "Buffer snapshot ready")) { return false; }
    const auto& buffer = inspector.capture_->artifacts[0];
    if (!expect(buffer.bytes.size() == 1024, "Buffer readback range")) { return false; }
    bool bufferMatches = true;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t word;
        std::memcpy(&word, buffer.bytes.data() + i * 4, 4);
        bufferMatches &= word == i * 17 + 3;
    }
    if (!expect(bufferMatches, "All buffer contents match the GPU copy") ||
        !expect(exportCapture("buffer") && exportUI("buffer-ui"), "Export buffer evidence and native UI rendering")) { return false; }
    inspector.offset_ = 17; inspector.count_ = 9; inspector.refresh_ = true;
    if (!frames(5) || !expect(inspector.capture_->artifacts[0].bytes.size() == 36 &&
        inspector.capture_->artifacts[0].metadata.at("elementOffset") == 17, "Buffer range offset applied")) { return false; }
    inspector.offset_ = UINT64_MAX; inspector.refresh_ = true;
    if (!frames(2) || !expect(inspector.job_.empty() && inspector.status_.find("outside") != std::string::npos, "Out of bounds rejected before GPU work")) { return false; }
    inspector.bufferLayout_ = "uint4"; inspector.offset_ = 2; inspector.count_ = 3;
    inspector.capture_.reset(); inspector.refresh_ = true;
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "uint4" &&
        inspector.capture_->artifacts[0].bytes.size() == 48, "Manual uint4 capture uses typed element ranges")) { return false; }
    const auto& vector = inspector.capture_->artifacts[0];
    const auto vectors = debug::decodeBuffer(vector.bytes, vector.layout);
    if (!expect(vectors && vectors->size() == 3 && vectors->at(0).at("x") == 139 && vectors->at(2).at("w") == 326,
        "Typed vector components decode at the correct stride and offset") ||
        !expect(exportCapture("typed-buffer") && exportUI("typed-buffer-ui"), "Export typed vector capture and UI")) { return false; }
    inspector.rawBuffer_ = true; inspector.hex_ = true;
    if (!frames(1) || !expect(exportUI("typed-buffer-raw-ui"), "Typed capture can show original raw bytes")) { return false; }
    inspector.hex_ = false;
    inspector.select("AutoExposure.exposure");
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "AutoExposureState",
        "Real AutoExposure defaults to its producer-declared type")) { return false; }
    const auto exposureCapture = inspector.capture_;
    const auto& exposure = exposureCapture->artifacts[0];
    const auto decodedExposure = debug::decodeBuffer(exposure.bytes, exposure.layout);
    if (decodedExposure) {
        report.value["exposureValues"] = *decodedExposure;
        spdlog::info("[Smoke Resource Inspector] AutoExposure typed values: {}", decodedExposure->dump());
    }
    if (!expect(decodedExposure && decodedExposure->size() == 1 &&
        decodedExposure->at(0).at("multiplier").get<double>() > 0 &&
        std::isfinite(decodedExposure->at(0).at("adaptedEV100").get<double>()) &&
        std::isfinite(decodedExposure->at(0).at("targetEV100").get<double>()) &&
        std::abs(decodedExposure->at(0).at("luminance").get<double>() - 0.5) < 0.01 &&
        std::abs(decodedExposure->at(0).at("multiplier").get<double>() - 0.36) < 0.01 &&
        std::abs(decodedExposure->at(0).at("targetEV100").get<double>() - std::log2(0.5 / 0.18)) < 0.02,
        "GPU HDR exposure fields decode as floats with expected metered luminance") ||
        !expect(exportCapture("exposure") && exportUI("exposure-ui"), "Export real exposure capture and typed UI")) { return false; }
    inspector.select("Data.exposure");
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "AutoExposureState" &&
        inspector.capture_->artifacts[0].bytes == exposure.bytes, "Input aliases retain the producer schema and values")) { return false; }
    inspector.select("AutoExposure.histogram"); inspector.count_ = 64;
    if (!frames(5) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].layout.name == "u32",
        "Histogram declares uint32 elements")) { return false; }
    const auto& histogram = inspector.capture_->artifacts[0];
    uint64_t weight = 0;
    for (size_t i = 0; i < histogram.bytes.size(); i += 4) {
        uint32_t bin; std::memcpy(&bin, histogram.bytes.data() + i, 4); weight += bin;
    }
    if (!expect(weight == 16 * 16 * 256, "Typed histogram readback preserves the GPU tile weight")) { return false; }
    inspector.select("Data.values");
    inspector.offset_ = 0; inspector.count_ = 256; inspector.refresh_ = true;
    if (!frames(4)) { return false; }
    const auto generation = inspector.generation_;
    viewportPreviewValid_ = false;
    if (!frames(5) || !expect(inspector.generation_ != generation && inspector.capture_ &&
        inspector.capture_->snapshot.evidence.generation == inspector.generation_, "Recompile replaces stale resource evidence")) { return false; }
    renderGraphEditorOpen_ = false;
    if (!frames(3) || !expect(!resourceInspectorAttached_ || debugRuntime_, "Closing the inspector detaches local capture")) { return false; }
    renderGraphEditorOpen_ = true; resourceInspectorSelectTab_ = true;
    if (!frames(5)) { return false; }
    // Cover the external-command recording path as well as the editor's usual
    // self-submitted, multi-queue graph path.
    {
        auto& runtime = debugRuntime_ ? *debugRuntime_ : inspector.runtime();
        const auto queued = runtime.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", "subsystem.test.inspector"}, {"checkpoint", "AfterGraph"},
            {"resources", debug::DebugValue::array({{{"id", "subsystem.test.inspector.values"}, {"count", 4}}})}}}});
        if (!expect(queued.at("status") == "ok", "Queue external-command subsystem capture") || !graphExecutor_->waitForSubmittedWork()) { return false; }
        render::RenderFrameContext externalFrame;
        render::QueueSubmissionTracker tracker;
        auto pool = device_->createCommandPool(*graphicsQueue_);
        if (!pool || !tracker.initialize(*device_, *graphicsQueue_) || !externalFrame.begin(10000)) { return false; }
        auto commands = (*pool)->createCommandBuffer();
        if (!commands || !(*commands)->begin(externalFrame.submissionContext()) || !graphExecutor_->execute(**commands) || !(*commands)->end()) { return false; }
        render::CommandBuffer* raw = commands->get();
        if (!tracker.submit({.commandBuffers = {&raw, 1}}, externalFrame) || !externalFrame.wait()) { return false; }
        runtime.poll();
        const auto capture = runtime.core().completedCapture(queued.at("result").at("job").get<std::string>());
        if (!expect(capture && capture->artifacts.size() == 1 && capture->artifacts[0].bytes.size() == 16,
            "External command recording captures subsystem resources after graph completion")) { return false; }
        const auto values = debug::decodeBuffer(capture->artifacts[0].bytes, capture->artifacts[0].layout);
        if (!expect(values && values->at(0) == 200, "External capture preserves the subsystem's current allocation")) { return false; }
        const auto unsupported = runtime.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", "subsystem.test.inspector"}, {"checkpoint", "AfterPass"},
            {"resources", debug::DebugValue::array({{{"id", "subsystem.test.inspector.values"}}})}}}});
        if (!expect(unsupported.at("status") == "error" && unsupported.at("error").at("code") == "Unsupported",
            "Subsystem capture rejects unregistered checkpoints")) { return false; }
        (void)(*pool)->reset();
        (void)externalFrame.reset();
    }
    inspector.autoRange_ = true; inspector.trimRange_ = false;
    inspector.channel_ = 0; inspector.exposure_ = 0;
    inspector.select("Clear.color");
    if (!frames(5) || !expect(inspector.capture_ && std::abs(inspector.rangeMin_ - 64.0 / 255) < 1e-6 &&
        std::abs(inspector.rangeMax_ - 191.0 / 255) < 1e-6, "Auto range follows captured RGB values")) { return false; }
    const auto autoFrozen = inspector.capture_;
    const double frozenMin = inspector.rangeMin_, frozenMax = inspector.rangeMax_;
    if (!renderGraph_.setNodeRuntimeProperty(renderGraph_.findNode("Clear")->id, "color", {0.125f, 0.25f, 0.5f, 1.f}) ||
        !frames(3) || !expect(inspector.capture_ == autoFrozen && inspector.rangeMin_ == frozenMin && inspector.rangeMax_ == frozenMax,
            "Frozen auto range stays with its snapshot while the source changes")) { return false; }
    inspector.live_ = true; inspector.nextRefresh_ = 0;
    if (!frames(5) || !expect(inspector.capture_ != autoFrozen && std::abs(inspector.rangeMin_ - 32.0 / 255) < 1e-6 &&
        std::abs(inspector.rangeMax_ - 128.0 / 255) < 0.005, "Live capture recomputes automatic bounds") ||
        !expect(readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview) &&
            preview[0] == 0 && preview[2] == 255, "Live auto contrast reaches the GPU preview")) { return false; }
    inspector.autoRange_ = false; inspector.rangeMin_ = 0; inspector.rangeMax_ = 1;
    inspector.imageDirty_ = true; inspector.nextRefresh_ = 0;
    if (!frames(5) || !expect(inspector.rangeMin_ == 0 && inspector.rangeMax_ == 1,
        "Live refresh preserves manual display bounds")) { return false; }
    inspector.live_ = false;

    // Optional production scene path, using native capture and offscreen ImGui
    // rendering only. The default smoke remains independent of sample assets.
    if (const char* sample = std::getenv("METALLIC_INSPECTOR_SMOKE_SAMPLE"); sample && *sample) {
        scene_.setDirty(false);
        loadBuiltInSample(sample);
        if (!expect(waitForPendingSceneLoad(60000), "Inspector sample scene load completes")) { return false; }
        auto* pathTrace = renderGraph_.findNode("PathTrace");
        if (!expect(pathTrace != nullptr, "Inspector sample contains PathTrace")) { return false; }
        auto properties = pathTrace->properties;
        properties["exportDenoiserGuides"] = true;
        renderGraph_.setNodeProperties(pathTrace->id, std::move(properties));
        viewportPreviewValid_ = false;
        inspector.autoRange_ = true; inspector.trimRange_ = true;
        resourceInspectorVisible_ = true; resourceInspectorSelectTab_ = true;
        for (const char* name : {"depth", "linearDepth"}) {
            const std::string resource = std::string("PathTrace.") + name;
            inspector.select(resource);
            if (!frames(8) || !expect(inspector.capture_ && inspector.capture_->artifacts[0].metadata.at("id") == resource &&
                inspector.capture_->artifacts[0].layout.name == "f32", "Real PathTrace depth capture is ready") ||
                !expect(readInspectorImage(*device_, *graphicsQueue_, imguiBackend_, *inspector.image_, preview),
                    "Read real PathTrace depth GPU preview")) { return false; }
            std::array<bool, 256> levels{};
            for (size_t i = 0; i < preview.size(); i += 4) { levels[preview[i]] = true; }
            const auto distinct = std::count(levels.begin(), levels.end(), true);
            report.value["depthRanges"][name] = inspector.diagnostics().at("displayRange");
            report.value["depthRanges"][name]["previewGrayLevels"] = distinct;
            if (!expect(inspector.rangeMin_ < inspector.rangeMax_ && inspector.rangeTrimmed_ && levels[0] && levels[255] && distinct > 8,
                "Real PathTrace depth gets foreground contrast without manual Min/Max") ||
                !expect(exportCapture(name) && exportUI((std::string(name) + "-ui").c_str()) &&
                    writePPM(output / (std::string(name) + "-preview.ppm"), preview, inspector.image_->desc().width,
                        inspector.image_->desc().height, render::Format::RGBA8Unorm, displayOutput_.paperWhiteNits),
                    "Export real depth capture and native automatic-range preview")) { return false; }
            const auto depthCapture = inspector.capture_;
            const auto depthImage = inspector.image_.get();
            inspector.fit_ = false; inspector.zoom_ = 2;
            if (!frames(2) || !expect(inspector.capture_ == depthCapture && inspector.image_.get() == depthImage &&
                exportUI((std::string(name) + "-zoom-ui").c_str()), "Magnified real depth reuses captured data and renders in the native UI")) { return false; }
        }
    }
    report.value["inspector"] = debug::encodeLossless(inspector.diagnostics());
    if (debugRuntime_) {
        const auto events = debugRuntime_->core().events("validation");
        report.value["validation"] = debug::encodeLossless(events);
        bool valid = true;
        for (const auto& event : events.at("events")) {
            valid &= event.at("severity").get<render::ValidationSeverity>() != render::ValidationSeverity::Error ||
                !render::hasFlag(event.at("type").get<render::ValidationCategory>(), render::ValidationCategory::Validation);
        }
        if (!expect(valid, "No Vulkan API validation errors during inspector smoke")) { return false; }
    }
    report.value["passed"] = true;
    std::ofstream reportFile(report.path);
    reportFile << report.value.dump(2);
    if (!reportFile.good()) { report.value["passed"] = false; return false; }
    spdlog::info("[Smoke Resource Inspector] PASS: GPU texture and buffer values, preview upload, freeze, ROI, range, invalid range, recompile and close/reopen");
    return true;
}

} // namespace metallic
