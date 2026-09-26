#include "RhiTest.h"
#include "Editor/EditorDisplayRenderer.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <cmath>

namespace metallic::tests {
namespace {

class EditorHdrCompositeTest final : public RhiTest {
public:
    EditorHdrCompositeTest() { type = RhiTestType::Rendering; name = "hdr_editor_imgui_composite"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto& device = context.device;
        const auto native = render::vulkan::nativeDevice(device);
        const auto queue = render::vulkan::nativeQueue(context.graphicsQueue);
        ImGui::CreateContext();
        struct UiScope {
            render::Device& device;
            EditorDisplayRenderer display;
            bool initialized = false;
            ~UiScope()
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
        ui.initialized = ImGui_ImplVulkan_Init(&init);
        if (!ui.initialized || !ui.display.initialize(native.device, format, true, 203.0f)) {
            return RhiTestResult::fail("HDR ImGui initialization failed");
        }
        std::unique_ptr<render::Texture> output, source;
        std::unique_ptr<render::TextureView> outputView, sourceView;
        std::unique_ptr<render::Buffer> readback;
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        std::unique_ptr<render::Fence> fence;
        const render::TextureDesc texture{.usage = render::TextureUsageBits::ColorAttachment |
                render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferSource,
            .format = render::Format::Rgba16Sfloat, .width = 32, .height = 32};
        if (!device.createTexture(texture).transform([&](auto rhiValue) { output = std::move(rhiValue); }) || !device.createTexture(texture).transform([&](auto rhiValue) { source = std::move(rhiValue); }) ||
            !device.createTextureView(*output, {}).transform([&](auto rhiValue) { outputView = std::move(rhiValue); }) || !device.createTextureView(*source, {}).transform([&](auto rhiValue) { sourceView = std::move(rhiValue); }) ||
            !device.createBuffer({.size = 32 * 32 * 8, .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }) ||
            !device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }) || !pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) ||
            !device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }) || !commands->begin()) {
            return RhiTestResult::fail("HDR ImGui fixture allocation failed");
        }
        render::TextureBarrierDesc barriers[] = {
            {.texture = source.get(), .after = render::ResourceState::ColorAttachment},
            {.texture = output.get(), .after = render::ResourceState::ColorAttachment},
        };
        commands->barrier({.textures = barriers, .textureCount = 2});
        render::RenderingAttachmentDesc attachment{.view = sourceView.get(), .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear, .storeOp = render::StoreOp::Store, .clearColor = {12.5f, 5.0f, 1.0f, 1.0f}};
        commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = &attachment, .colorAttachmentCount = 1});
        commands->endRendering();
        render::TextureBarrierDesc readable{.texture = source.get(), .before = render::ResourceState::ColorAttachment,
            .after = render::ResourceState::ShaderRead};
        commands->barrier({.textures = &readable, .textureCount = 1});
        const auto descriptor = ImGui_ImplVulkan_AddTexture(render::vulkan::nativeImageView(*sourceView), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
        auto* list = ImGui::GetBackgroundDrawList();
        list->AddRectFilled(ImVec2(0, 0), ImVec2(32, 16), IM_COL32_WHITE);
        ui.display.beginScRgbImage(*list, ImGui::GetMainViewport());
        list->AddImage(reinterpret_cast<ImTextureID>(descriptor), ImVec2(0, 16), ImVec2(32, 32));
        list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
        // Verify reset restores UI transfer/brightness after the HDR viewport.
        list->AddRectFilled(ImVec2(24, 24), ImVec2(32, 32), IM_COL32(128, 128, 128, 255));
        ImGui::Render();
        attachment.view = outputView.get();
        attachment.clearColor = {0, 0, 0, 1};
        commands->beginRendering({.renderArea = {0, 0, 32, 32}, .colorAttachments = &attachment, .colorAttachmentCount = 1});
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), render::vulkan::nativeCommandBuffer(*commands), ui.display.mainPipeline());
        commands->endRendering();
        render::TextureBarrierDesc toReadback{.texture = output.get(), .before = render::ResourceState::ColorAttachment,
            .after = render::ResourceState::TransferSource};
        commands->barrier({.textures = &toReadback, .textureCount = 1});
        commands->copyTextureToBuffer({.texture = output.get(), .buffer = readback.get(), .width = 32, .height = 32});
        render::CommandBuffer* command = commands.get();
        if (!commands->end() || !context.graphicsQueue.submit({.commandBuffers = &command, .commandBufferCount = 1,
                .signalFence = fence.get()}) || !fence->wait()) { return RhiTestResult::fail("HDR ImGui submit failed"); }
        readback->invalidate();
        const auto* pixels = static_cast<const uint16_t*>(readback->map());
        if (!pixels) { return RhiTestResult::fail("HDR ImGui readback failed"); }
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
        return correct ? RhiTestResult::pass("FP16 UI paper white, scRGB image and callback reset verified") :
            RhiTestResult::fail("ImGui clipped HDR, applied gamma twice, or failed to restore UI brightness");
    }
};

METALLIC_REGISTER_RHI_TEST(EditorHdrCompositeTest);

} // namespace
} // namespace metallic::tests
