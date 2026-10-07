#include "Editor/EditorApplication.h"

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <cmath>

namespace metallic {
namespace {
class InspectorBufferPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("values").buffer(1024, 16).transferWrite();
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
        if (!condition) { spdlog::error("[Smoke Resource Inspector] {}", message); }
        return condition;
    };
    render::registerRenderGraphPassType("InspectorBufferFixture", "Inspector smoke fixture", [] { return std::make_unique<InspectorBufferPass>(); });
    destroyViewportTexture();
    renderGraph_ = render::RenderGraph{};
    renderGraph_.addNode("ClearColorPass", "Clear", {{"color", {0.25f, 0.5f, 0.75f, 1.f}}});
    renderGraph_.addNode("InspectorBufferFixture", "Data");
    renderGraph_.markOutput("Clear.color");
    renderGraph_.markOutput("Data.values");
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
