#include "Editor/EditorApplication.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cmath>

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
    render::Texture& texture, std::vector<uint8_t>& bytes, ImDrawData* drawData = nullptr)
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

bool EditorApplication::runResourceInspectorSmokeTest(const char* outputDirectory)
{
    const auto output = outputDirectory ? std::filesystem::path(outputDirectory) : std::filesystem::path(".cache/InspectorSmoke");
    std::error_code error;
    std::filesystem::create_directories(output, error);
    if (error) { spdlog::error("Cannot create inspector evidence directory: {}", error.message()); return false; }
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
