#include "RHITest.h"
#include "Runtime/Render/Core/ImGuiDisplayShaders.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <SDL3/SDL.h>
#include <thread>
#include <atomic>
#include <cmath>

namespace metallic::tests {
namespace {

class EditorHDRCompositeTest final : public RHITest {
public:
    EditorHDRCompositeTest() { type = RHITestType::Rendering; name = "hdr_editor_imgui_composite"; }
    RHITestResult run(RHITestContext& context) override
    {
        auto& device = context.device;
        ImGui::CreateContext();
        struct UIScope {
            render::Device& device;
            render::vulkan::VulkanImGuiBackend display;
            ~UIScope()
            {
                (void)device.waitIdle();
                display.shutdown();
                ImGui::DestroyContext();
            }
        } ui{device};
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = ImVec2(32, 32);
        io.DeltaTime = 1.0f / 60.0f;
        const render::vulkan::ImGuiDisplayDesc displayDesc{
            .colorFormat = render::Format::RGBA16Sfloat,
            .pqOutputFormat = render::Format::A2B10G10R10UnormPack32};
        if (!ui.display.init(device, context.graphicsQueue, displayDesc, render::imGuiDisplayShaders())) {
            return RHITestResult::fail("HDR ImGui initialization failed");
        }
        ui.display.shutdown();
        if (!ui.display.init(device, context.graphicsQueue, displayDesc, render::imGuiDisplayShaders())) {
            return RHITestResult::fail("HDR ImGui cached initialization failed");
        }
        if (!render::ShaderRegistry::instance().flushPipelineCaches(device)) {
            return RHITestResult::fail("HDR ImGui cache flush failed");
        }
        auto cacheStats = render::ShaderRegistry::instance().pipelineCacheStats(device);
        bool editorCacheHit = false;
        if (cacheStats) {
            for (const auto& group : *cacheStats) {
                if (group.group.starts_with("ShaderRegistry-EditorDisplay-") && group.cache.hitCount >= 3 &&
                    group.cache.backendDataSize > 0) { editorCacheHit = true; }
            }
        }
        if (!editorCacheHit) { return RHITestResult::fail("HDR ImGui bypassed the Registry's persistent native cache"); }
        std::unique_ptr<render::Texture> output, source;
        std::unique_ptr<render::TextureView> outputView, sourceView;
        std::unique_ptr<render::Buffer> readback;
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        std::unique_ptr<render::Fence> fence;
        const render::TextureDesc texture{.usage = render::TextureUsageBits::ColorAttachment |
                render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferSource,
            .format = render::Format::RGBA16Sfloat, .width = 32, .height = 32};
        if (!device.createTexture(texture).transform([&](auto rhiValue) { output = std::move(rhiValue); }) || !device.createTexture(texture).transform([&](auto rhiValue) { source = std::move(rhiValue); }) ||
            !device.createTextureView(*output, {}).transform([&](auto rhiValue) { outputView = std::move(rhiValue); }) || !device.createTextureView(*source, {}).transform([&](auto rhiValue) { sourceView = std::move(rhiValue); }) ||
            !device.createBuffer({.size = 32 * 32 * 8, .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }) ||
            !device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }) || !pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) ||
            !device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }) || !commands->begin()) {
            return RHITestResult::fail("HDR ImGui fixture allocation failed");
        }
        render::TextureBarrierDesc barriers[] = {
            {
                .texture = source.get(),
                .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::ColorAttachment,
                .before = {},
                .after = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
            },
            {
                .texture = output.get(),
                .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::ColorAttachment,
                .before = {},
                .after = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
            },
        };
        if (auto commandResult = commands->synchronize({.textures = {barriers, 2}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        render::RenderingAttachmentDesc attachment{.view = sourceView.get(), .layout = render::TextureLayout::ColorAttachment,
            .loadOp = render::LoadOp::Clear, .storeOp = render::StoreOp::Store, .clearColor = {12.5f, 5.0f, 1.0f, 1.0f}};
        if (auto commandResult = commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = {&attachment, 1}}); !commandResult) { return RHITestResult::fail(std::string("beginRendering failed: ") + render::resultToString(commandResult)); }
        commands->endRendering();
        render::TextureBarrierDesc readable{
            .texture = source.get(),
            .oldLayout = render::TextureLayout::ColorAttachment,
            .newLayout = render::TextureLayout::ShaderRead,
            .before = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
        };
        if (auto commandResult = commands->synchronize({.textures = {&readable, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        auto registered = ui.display.addTexture(*sourceView);
        if (!registered) { return RHITestResult::fail("ImGui texture registration failed"); }
        const auto descriptor = *registered;
        // Registration owns the original native view and image, independent of Editor ownership.
        sourceView.reset();
        source.reset();
        if (!ui.display.newFrame()) { return RHITestResult::fail("ImGui NewFrame failed"); }
        ImGui::NewFrame();
        auto* list = ImGui::GetBackgroundDrawList();
        list->AddRectFilled(ImVec2(0, 0), ImVec2(32, 16), IM_COL32_WHITE);
        ui.display.beginScRgbImage(*list, ImGui::GetMainViewport());
        list->AddImage(static_cast<ImTextureID>(descriptor), ImVec2(0, 16), ImVec2(32, 32));
        list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        // Verify reset restores UI transfer/brightness after the HDR viewport.
        list->AddRectFilled(ImVec2(24, 24), ImVec2(32, 32), IM_COL32(128, 128, 128, 255));
        list->AddRectFilled(ImVec2(16, 24), ImVec2(24, 32), IM_COL32(255, 255, 255, 128));
        list->AddRectFilled(ImVec2(0, 0), ImVec2(4, 4), IM_COL32_BLACK);
        ImGui::Render();
        attachment.view = outputView.get();
        attachment.clearColor = {0, 0, 0, 1};
        if (auto commandResult = commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = {&attachment, 1}}); !commandResult) { return RHITestResult::fail(std::string("beginRendering failed: ") + render::resultToString(commandResult)); }
        if (!ui.display.render(*commands)) { return RHITestResult::fail("ImGui rendering failed"); }
        // Removal cannot free a descriptor/view still referenced by the recording.
        ui.display.removeTexture(descriptor);
        commands->endRendering();
        render::TextureBarrierDesc toReadback{
            .texture = output.get(),
            .oldLayout = render::TextureLayout::ColorAttachment,
            .newLayout = render::TextureLayout::TransferSource,
            .before = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorRead | render::AccessBits::ColorWrite},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
        };
        if (auto commandResult = commands->synchronize({.textures = {&toReadback, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        if (auto commandResult = (readback.get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = output.get(), .buffer = bufferSlice, .width = 32, .height = 32}); }); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
        render::CommandBuffer* command = commands.get();
        if (!commands->end() || !context.graphicsQueue.submit({
            .commandBuffers = {&command, 1},
            .signalFence = fence.get(),
        }) || !fence->wait()) { return RHITestResult::fail("HDR ImGui submit failed"); }
        readback->invalidate();
        const auto* pixels = static_cast<const uint16_t*>(readback->map());
        if (!pixels) { return RHITestResult::fail("HDR ImGui readback failed"); }
        auto channel = [&](uint32_t x, uint32_t y, uint32_t c) {
            const uint16_t v = pixels[(y * 32 + x) * 4 + c];
            return std::ldexp(float((v & 1023) + 1024), int((v >> 10) & 31) - 25);
        };
        auto near = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 0.015f; };
        const bool correct = near(channel(8, 8, 0), 203.0f / 80.0f) &&
            near(channel(8, 24, 0), 12.5f) && near(channel(8, 24, 1), 5.0f) &&
            near(channel(8, 24, 2), 1.0f) && near(channel(8, 24, 3), 1.0f) &&
            near(channel(28, 28, 0), std::pow((128.0f / 255.0f + 0.055f) / 1.055f, 2.4f) * 203.0f / 80.0f);
        readback->unmap();
        if (!correct) { return RHITestResult::fail("ImGui clipped HDR, applied gamma twice, or failed to restore UI brightness"); }

        std::unique_ptr<render::Texture> pq;
        std::unique_ptr<render::TextureView> pqView;
        auto pqDesc = texture;
        pqDesc.format = render::Format::A2B10G10R10UnormPack32;
        if (!device.createTexture(pqDesc).transform([&](auto value) { pq = std::move(value); }) ||
            !device.createTextureView(*pq, {}).transform([&](auto value) { pqView = std::move(value); }) ||
            !pool->reset() || !fence->reset() || !commands->begin()) {
            return RHITestResult::fail("PQ fixture allocation failed");
        }
        auto outputDescriptor = ui.display.addTexture(*outputView);
        if (!outputDescriptor) { return RHITestResult::fail("PQ texture registration failed"); }
        const render::TextureBarrierDesc encodeBarriers[] = {
            {.texture = output.get(), .oldLayout = render::TextureLayout::TransferSource,
                .newLayout = render::TextureLayout::ShaderRead,
                .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                .after = {render::PipelineStageBits::FragmentShader, render::AccessBits::ShaderRead}},
            {.texture = pq.get(), .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::ColorAttachment,
                .after = {render::PipelineStageBits::ColorAttachment, render::AccessBits::ColorWrite}},
        };
        if (!commands->synchronize({.textures = encodeBarriers})) { return RHITestResult::fail("PQ barriers failed"); }
        attachment.view = pqView.get();
        if (!commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = {&attachment, 1}})) {
            return RHITestResult::fail("PQ rendering failed");
        }
        if (!ui.display.encodeHDR10(*commands, *outputDescriptor, 32, 32)) {
            return RHITestResult::fail("PQ encoding failed");
        }
        commands->endRendering();
        toReadback.texture = pq.get();
        if (!commands->synchronize({.textures = {&toReadback, 1}})) { return RHITestResult::fail("PQ readback barrier failed"); }
        if (auto commandResult = (readback.get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = pq.get(), .buffer = bufferSlice, .width = 32, .height = 32}); }); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
        if (!commands->end() || !context.graphicsQueue.submit({.commandBuffers = {&command, 1},
                .signalFence = fence.get()}) || !fence->wait()) { return RHITestResult::fail("PQ submit failed"); }
        readback->invalidate();
        const auto* packed = static_cast<const uint32_t*>(readback->map());
        if (!packed) { return RHITestResult::fail("PQ readback failed"); }
        // Independent CPU reference in absolute nits, including 10-bit quantization.
        auto pqCode = [](double nits) {
            const double p = std::pow(nits / 10000.0, 2610.0 / 16384.0);
            return int(std::lround(1023.0 * std::pow((3424.0 / 4096.0 + 2413.0 / 128.0 * p) /
                (1.0 + 2392.0 / 128.0 * p), 2523.0 / 32.0)));
        };
        auto check = [&](uint32_t x, uint32_t y, double r, double g, double b) {
            const uint32_t pixel = packed[y * 32 + x];
            const double nits[] = {0.627403896 * r + 0.329283038 * g + 0.043313066 * b,
                0.069097289 * r + 0.919540395 * g + 0.011362316 * b,
                0.016391439 * r + 0.088013308 * g + 0.895595253 * b};
            for (uint32_t c = 0; c < 3; ++c) {
                if (std::abs(int((pixel >> (10 * c)) & 1023) - pqCode(nits[c])) > 1) { return false; }
            }
            return (pixel >> 30) == 3;
        };
        const double alpha = 128.0 / 255.0;
        const bool pqCorrect = check(1, 1, 0, 0, 0) && check(8, 8, 203, 203, 203) &&
            check(8, 24, 1000, 400, 80) &&
            check(20, 28, 203 * alpha + 1000 * (1 - alpha), 203 * alpha + 400 * (1 - alpha),
                203 * alpha + 80 * (1 - alpha));
        readback->unmap();
        ui.display.removeTexture(*outputDescriptor);
        return pqCorrect ? RHITestResult::pass("FP16 UI composition, linear alpha blending, BT.2020/PQ and RGB10A2 pixels verified") :
            RHITestResult::fail("HDR10 conversion, absolute luminance or linear UI blending mismatch");
    }
};

METALLIC_REGISTER_RHI_TEST(EditorHDRCompositeTest);

class ImGuiPlatformQueueTest final : public RHITest {
public:
    ImGuiPlatformQueueTest() { type = RHITestType::Rendering; name = "imgui_platform_windows_concurrent_submit"; }
    RHITestResult run(RHITestContext& context) override
    {
        auto* window = SDL_CreateWindow("ImGui queue regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
        if (!window) { return RHITestResult::fail(SDL_GetError()); }
        ImGui::CreateContext();
        struct Cleanup {
            SDL_Window* window;
            render::vulkan::VulkanImGuiBackend backend;
            bool platform = false;
            ~Cleanup()
            {
                backend.shutdown();
                if (platform) { ImGui_ImplSDL3_Shutdown(); }
                ImGui::DestroyContext();
                SDL_DestroyWindow(window);
            }
        } cleanup{window};
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.ConfigViewportsNoAutoMerge = true;
        cleanup.platform = ImGui_ImplSDL3_InitForVulkan(window);
        if (!cleanup.platform || !cleanup.backend.init(context.device, context.graphicsQueue,
                {.colorFormat = render::Format::BGRA8Unorm, .hdr = false}, render::imGuiDisplayShaders())) {
            return RHITestResult::fail("ImGui platform fixture initialization failed");
        }
        std::atomic_uint submits{0};
        std::atomic_bool failed{false};
        std::jthread worker([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                if (!context.graphicsQueue.submit({})) { failed = true; break; }
                ++submits;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        bool detached = false;
        for (uint32_t frame = 0; frame < 3; ++frame) {
            SDL_PumpEvents();
            if (!cleanup.backend.newFrame()) { return RHITestResult::fail("ImGui platform NewFrame failed"); }
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(100, 100));
            ImGui::SetNextWindowSize(ImVec2(128, 64));
            ImGui::Begin("Detached queue regression", nullptr, ImGuiWindowFlags_NoSavedSettings);
            ImGui::TextUnformatted("Concurrent RHI submission");
            ImGui::End();
            ImGui::Render();
            if (!cleanup.backend.renderPlatformWindows()) { return RHITestResult::fail("ImGui platform submit/present failed"); }
            detached |= ImGui::GetPlatformIO().Viewports.Size > 1;
        }
        worker.request_stop();
        worker.join();
        if (!context.graphicsQueue.waitIdle() || failed || !submits || !detached) {
            return RHITestResult::fail("Detached viewport or concurrent RHI submissions were not exercised");
        }
        return RHITestResult::pass("Detached viewport submit/present/font upload with concurrent RHI queue submission");
    }
};
METALLIC_REGISTER_RHI_TEST(ImGuiPlatformQueueTest);

} // namespace
} // namespace metallic::tests
