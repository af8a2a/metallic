#include "RHITest.h"
#include "Editor/EditorDisplayRenderer.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <cmath>

namespace metallic::tests {
namespace {

class EditorHDRCompositeTest final : public RHITest {
public:
    EditorHDRCompositeTest() { type = RHITestType::Rendering; name = "hdr_editor_imgui_composite"; }
    RHITestResult run(RHITestContext& context) override
    {
        auto& device = context.device;
        const auto native = render::vulkan::nativeDevice(device);
        const auto queue = render::vulkan::nativeQueue(context.graphicsQueue);
        ImGui::CreateContext();
        struct UIScope {
            render::Device& device;
            EditorDisplayRenderer display;
            bool initialized = false;
            ~UIScope()
            {
                (void)device.waitIdle();
                display.shutdown();
                if (initialized) { ImGui_ImplVulkan_Shutdown(); }
                ImGui::DestroyContext();
            }
        } ui{device};
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = ImVec2(32, 32);
        io.DeltaTime = 1.0f / 60.0f;
        const VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ImGui_ImplVulkan_InitInfo init{};
        init.ApiVersion = native.apiVersion;
        init.Instance = native.instance;
        init.PhysicalDevice = native.physicalDevice;
        init.Device = native.device;
        init.QueueFamily = queue.familyIndex;
        init.Queue = queue.queue;
        init.DescriptorPoolSize = 32;
        init.MinImageCount = init.ImageCount = 2;
        init.UseDynamicRendering = true;
        init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init.PipelineInfoMain.PipelineRenderingCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
        ui.initialized = EditorDisplayRenderer::loadBackendFunctions(native) && ImGui_ImplVulkan_Init(&init);
        if (!ui.initialized || !ui.display.initialize(device, format, true, 203.0f,
                VK_FORMAT_A2B10G10R10_UNORM_PACK32)) {
            return RHITestResult::fail("HDR ImGui initialization failed");
        }
        ui.display.shutdown();
        if (!ui.display.initialize(device, format, true, 203.0f, VK_FORMAT_A2B10G10R10_UNORM_PACK32)) {
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
        const auto descriptor = ImGui_ImplVulkan_AddTexture(render::vulkan::nativeImageView(*sourceView), render::vulkan::nativeImageLayout(*sourceView, render::TextureLayout::ShaderRead));
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
        auto* list = ImGui::GetBackgroundDrawList();
        list->AddRectFilled(ImVec2(0, 0), ImVec2(32, 16), IM_COL32_WHITE);
        ui.display.beginScRgbImage(*list, ImGui::GetMainViewport());
        list->AddImage(reinterpret_cast<ImTextureID>(descriptor), ImVec2(0, 16), ImVec2(32, 32));
        list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        // Verify reset restores UI transfer/brightness after the HDR viewport.
        list->AddRectFilled(ImVec2(24, 24), ImVec2(32, 32), IM_COL32(128, 128, 128, 255));
        list->AddRectFilled(ImVec2(16, 24), ImVec2(24, 32), IM_COL32(255, 255, 255, 128));
        list->AddRectFilled(ImVec2(0, 0), ImVec2(4, 4), IM_COL32_BLACK);
        ImGui::Render();
        attachment.view = outputView.get();
        attachment.clearColor = {0, 0, 0, 1};
        if (auto commandResult = commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = {&attachment, 1}}); !commandResult) { return RHITestResult::fail(std::string("beginRendering failed: ") + render::resultToString(commandResult)); }
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), render::vulkan::nativeCommandBuffer(*commands), ui.display.mainPipeline());
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
        ImGui_ImplVulkan_RemoveTexture(descriptor);
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
        const auto outputDescriptor = ImGui_ImplVulkan_AddTexture(render::vulkan::nativeImageView(*outputView),
            render::vulkan::nativeImageLayout(*outputView, render::TextureLayout::ShaderRead));
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
        ui.display.encodeHDR10(render::vulkan::nativeCommandBuffer(*commands), outputDescriptor, 32, 32);
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
        ImGui_ImplVulkan_RemoveTexture(outputDescriptor);
        return pqCorrect ? RHITestResult::pass("FP16 UI composition, linear alpha blending, BT.2020/PQ and RGB10A2 pixels verified") :
            RHITestResult::fail("HDR10 conversion, absolute luminance or linear UI blending mismatch");
    }
};

METALLIC_REGISTER_RHI_TEST(EditorHDRCompositeTest);

} // namespace
} // namespace metallic::tests
