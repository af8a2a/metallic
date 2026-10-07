#include "VulkanImGuiBackend.h"
#include "VulkanInterop.h"
#include "VulkanResult.h"
#include "Runtime/Render/GAPI/Hash.h"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>
#include <map>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace metallic::render::vulkan {
namespace {
class DisplayPipelines {
public:
    ImGuiShaderServices shaders;
    bool failed = false;
    ~DisplayPipelines();
    bool initialize(render::Device& device, VkFormat mainFormat, bool hdr, float paperWhiteNits,
        VkFormat pqOutputFormat = VK_FORMAT_UNDEFINED);
    void shutdown();
    VkPipeline mainPipeline() const { return mainPipeline_; }
    bool canEncodeHDR10() const { return pqPipeline_ != VK_NULL_HANDLE; }
    void beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport);
    void encodeHDR10(VkCommandBuffer commands, VkDescriptorSet source, uint32_t width, uint32_t height);

private:
    static void bindImagePipeline(const ImDrawList*, const ImDrawCmd* command);
    VkPipeline createPipeline(VkFormat format, bool hdr, bool scRgbImage, float paperWhiteNits, bool encodePQ = false);

    VkDevice device_ = VK_NULL_HANDLE;
    render::Device* rhiDevice_ = nullptr;
    const VolkDeviceTable* functions_ = nullptr;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayouts_[2]{};
    VkPipeline mainPipeline_ = VK_NULL_HANDLE;
    VkPipeline mainImagePipeline_ = VK_NULL_HANDLE;
    VkPipeline pqPipeline_ = VK_NULL_HANDLE;
    VkFormat pqOutputFormat_ = VK_FORMAT_UNDEFINED;
    std::map<VkFormat, VkPipeline> secondaryImagePipelines_;
    VkFormat mainFormat_ = VK_FORMAT_UNDEFINED;
    bool hdr_ = false;
    float paperWhiteNits_ = 203.0f;
    struct ImageCallbackData {
        DisplayPipelines* renderer;
        ImGuiViewport* viewport;
    };
};

DisplayPipelines::~DisplayPipelines()
{
    shutdown();
}

void DisplayPipelines::shutdown()
{
    if (device_ == VK_NULL_HANDLE) { return; }
    for (VkPipeline pipeline : {mainPipeline_, mainImagePipeline_, pqPipeline_}) {
        if (pipeline) { functions_->vkDestroyPipeline(device_, pipeline, nullptr); }
    }
    for (const auto& [format, pipeline] : secondaryImagePipelines_) {
        if (pipeline) { functions_->vkDestroyPipeline(device_, pipeline, nullptr); }
    }
    secondaryImagePipelines_.clear();
    if (layout_) { functions_->vkDestroyPipelineLayout(device_, layout_, nullptr); }
    for (VkDescriptorSetLayout layout : setLayouts_) {
        if (layout) { functions_->vkDestroyDescriptorSetLayout(device_, layout, nullptr); }
    }
    mainPipeline_ = mainImagePipeline_ = pqPipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    setLayouts_[0] = setLayouts_[1] = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    rhiDevice_ = nullptr;
}

bool DisplayPipelines::initialize(render::Device& rhiDevice, VkFormat mainFormat, bool hdr, float paperWhiteNits,
    VkFormat pqOutputFormat)
{
    const auto native = render::vulkan::nativeDevice(rhiDevice);
    const VkDevice device = native.device;
    if (device_ == device && mainPipeline_ && mainImagePipeline_ && mainFormat_ == mainFormat &&
        hdr_ == hdr && paperWhiteNits_ == paperWhiteNits && pqOutputFormat_ == pqOutputFormat &&
        (pqOutputFormat == VK_FORMAT_UNDEFINED || pqPipeline_)) { return true; }
    shutdown();
    failed = false;
    device_ = device;
    rhiDevice_ = &rhiDevice;
    functions_ = native.functions;
    mainFormat_ = mainFormat;
    hdr_ = hdr;
    paperWhiteNits_ = paperWhiteNits;
    pqOutputFormat_ = pqOutputFormat;
    for (uint32_t index = 0; index < 2; ++index) {
        VkDescriptorSetLayoutBinding binding{.binding = 0,
            .descriptorType = index == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &binding};
        if (functions_->vkCreateDescriptorSetLayout(device_, &info, nullptr, &setLayouts_[index]) != VK_SUCCESS) { return false; }
    }
    VkPushConstantRange push{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 16};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = setLayouts_, .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
    if (functions_->vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_) != VK_SUCCESS) { return false; }
    mainPipeline_ = createPipeline(mainFormat, hdr, false, paperWhiteNits);
    mainImagePipeline_ = createPipeline(mainFormat, hdr, true, paperWhiteNits);
    if (pqOutputFormat != VK_FORMAT_UNDEFINED) {
        pqPipeline_ = createPipeline(pqOutputFormat, true, false, paperWhiteNits, true);
        if (!pqPipeline_) { return false; }
    }
    return mainPipeline_ && mainImagePipeline_;
}

VkPipeline DisplayPipelines::createPipeline(VkFormat format, bool hdr, bool scRgbImage, float paperWhiteNits, bool encodePQ)
{
    std::unique_ptr<render::ShaderModule> modules[2];
    const ImGuiShaderRequest request{.hdr = hdr, .scRgbImage = scRgbImage,
        .targetSrgb = resourceFormat(format) == Format::BGRA8sRGB || resourceFormat(format) == Format::RGBA8sRGB,
        .encodePQ = encodePQ, .paperWhiteNits = paperWhiteNits};
    for (uint32_t index = 0; index < 2; ++index) {
        auto module = shaders.load(*rhiDevice_, request, index == 0);
        if (!module) { return VK_NULL_HANDLE; }
        modules[index] = std::move(*module);
    }
    VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = render::vulkan::nativeShaderModule(*modules[0]), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = render::vulkan::nativeShaderModule(*modules[1]), .pName = "main"},
    };
    VkVertexInputBindingDescription binding{0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, pos)},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, uv)},
        {2, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ImDrawVert, col)},
    };
    VkPipelineVertexInputStateCreateInfo vertex{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes};
    VkPipelineInputAssemblyStateCreateInfo assembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    if (encodePQ) {
        vertex.vertexBindingDescriptionCount = 0;
        vertex.vertexAttributeDescriptionCount = 0;
    }
    VkPipelineViewportStateCreateInfo viewport{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo raster{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo multisample{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState attachment{.blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD, .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment};
    if (encodePQ) { attachment.blendEnable = VK_FALSE; }
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamicStates};
    VkPipelineRenderingCreateInfo rendering{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
    VkGraphicsPipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering, .stageCount = 2, .pStages = stages, .pVertexInputState = &vertex,
        .pInputAssemblyState = &assembly, .pViewportState = &viewport, .pRasterizationState = &raster,
        .pMultisampleState = &multisample, .pColorBlendState = &blend, .pDynamicState = &dynamic, .layout = layout_};
    VkPipeline pipeline = VK_NULL_HANDLE;
    // The native ImGui descriptor/vertex ABI has fixed raster/blend/layout
    // state above. Increment this version if that fixed contract changes.
    constexpr uint64_t kPipelineABIVersion = 1;
    uint64_t stateHash = render::detail::kFnvOffset;
    const uint64_t identity[] = {0x494d47554950534full, kPipelineABIVersion,
        modules[0]->contentHash(), modules[1]->contentHash(), uint64_t(format), uint64_t(encodePQ),
        sizeof(ImDrawVert), offsetof(ImDrawVert, pos), offsetof(ImDrawVert, uv), offsetof(ImDrawVert, col)};
    for (uint64_t value : identity) {
        stateHash = render::detail::hashValue(stateHash, value);
    }
    const auto result = shaders.cache(*rhiDevice_,
        modules[0]->contentHash(), [&](render::PipelineCache& cache) {
            return render::vulkan::createCachedGraphicsPipeline(cache, info, stateHash, pipeline);
        });
    if (!result) { spdlog::error("Editor display pipeline creation failed: {}", render::resultToString(result)); }
    return pipeline;
}

void DisplayPipelines::bindImagePipeline(const ImDrawList*, const ImDrawCmd* command)
{
    auto* state = static_cast<ImGui_ImplVulkan_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);
    const auto& data = *static_cast<const ImageCallbackData*>(command->UserCallbackData);
    auto& renderer = *data.renderer;
    VkPipeline pipeline = renderer.mainImagePipeline_;
    if (data.viewport != ImGui::GetMainViewport()) {
        const auto* window = ImGui_ImplVulkanH_GetWindowDataFromViewport(data.viewport);
        if (!window) { return; }
        auto& cached = renderer.secondaryImagePipelines_[window->SurfaceFormat.format];
        if (!cached) {
            cached = renderer.createPipeline(window->SurfaceFormat.format, false, true, renderer.paperWhiteNits_);
        }
        pipeline = cached;
    }
    if (pipeline) { renderer.functions_->vkCmdBindPipeline(state->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline); }
    else { renderer.failed = true; }
}

void DisplayPipelines::beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport)
{
    ImageCallbackData data{this, viewport};
    list.AddCallback(bindImagePipeline, &data, sizeof(data));
}

void DisplayPipelines::encodeHDR10(VkCommandBuffer commands, VkDescriptorSet source, uint32_t width, uint32_t height)
{
    VkViewport viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};
    functions_->vkCmdSetViewport(commands, 0, 1, &viewport);
    functions_->vkCmdSetScissor(commands, 0, 1, &scissor);
    functions_->vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pqPipeline_);
    functions_->vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &source, 0, nullptr);
    functions_->vkCmdDraw(commands, 3, 1, 0, 0);
}


} // namespace

struct VulkanImGuiBackend::Impl {
    Device* device = nullptr;
    Queue* queue = nullptr;
    DisplayPipelines display;
    bool initialized = false;
    VkResult error = VK_SUCCESS;
    struct Texture {
        std::unique_ptr<TextureView> view;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
    };
    std::map<ImGuiTexture, std::shared_ptr<Texture>> textures;
    std::vector<std::shared_ptr<Texture>> retired;
    inline static thread_local Impl* active = nullptr;
    struct Call {
        Impl* previous = active;
        explicit Call(Impl& impl) { active = &impl; impl.error = VK_SUCCESS; }
        ~Call() { active = previous; }
    };
    static void check(VkResult result)
    {
        if (active && result < 0 && active->error == VK_SUCCESS) { active->error = result; }
    }
    static VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence)
    {
        if (!active || nativeQueue(*active->queue).queue != queue) { return VK_ERROR_UNKNOWN; }
        struct Batch {
            std::vector<VkSemaphoreSubmitInfo> waits, signals;
            std::vector<VkCommandBufferSubmitInfo> commands;
        };
        std::vector<Batch> batches(count);
        std::vector<VkSubmitInfo2> converted(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto& input = submits[i];
            // ImGui submits binary semaphores and primary command buffers only.
            if (input.pNext) { check(VK_ERROR_FEATURE_NOT_PRESENT); return VK_ERROR_FEATURE_NOT_PRESENT; }
            auto& batch = batches[i];
            for (uint32_t j = 0; j < input.waitSemaphoreCount; ++j) {
                batch.waits.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = input.pWaitSemaphores[j], .stageMask = input.pWaitDstStageMask[j]});
            }
            for (uint32_t j = 0; j < input.signalSemaphoreCount; ++j) {
                batch.signals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = input.pSignalSemaphores[j], .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
            }
            for (uint32_t j = 0; j < input.commandBufferCount; ++j) {
                batch.commands.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                    .commandBuffer = input.pCommandBuffers[j]});
            }
            converted[i] = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .waitSemaphoreInfoCount = uint32_t(batch.waits.size()), .pWaitSemaphoreInfos = batch.waits.data(),
                .commandBufferInfoCount = uint32_t(batch.commands.size()), .pCommandBufferInfos = batch.commands.data(),
                .signalSemaphoreInfoCount = uint32_t(batch.signals.size()), .pSignalSemaphoreInfos = batch.signals.data()};
        }
        const auto result = submitInterop(*active->queue, converted, fence);
        check(result);
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL present(VkQueue queue, const VkPresentInfoKHR* info)
    {
        if (!active || nativeQueue(*active->queue).queue != queue) { return VK_ERROR_UNKNOWN; }
        const auto result = presentInterop(*active->queue, *info);
        // The ImGui platform backend handles resize/suboptimal itself.
        if (result != VK_ERROR_OUT_OF_DATE_KHR) { check(result); }
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL waitQueue(VkQueue queue)
    {
        if (!active || nativeQueue(*active->queue).queue != queue) { return VK_ERROR_UNKNOWN; }
        const auto result = waitInterop(*active->queue);
        check(result);
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL waitDevice(VkDevice device)
    {
        if (!active || nativeDevice(*active->device).device != device) { return VK_ERROR_UNKNOWN; }
        const auto result = waitInterop(*active->device);
        check(result);
        return result;
    }
    static PFN_vkVoidFunction load(const char* name, void* context)
    {
        if (std::strcmp(name, "vkQueueSubmit") == 0) { return reinterpret_cast<PFN_vkVoidFunction>(submit); }
        if (std::strcmp(name, "vkQueuePresentKHR") == 0) { return reinterpret_cast<PFN_vkVoidFunction>(present); }
        if (std::strcmp(name, "vkQueueWaitIdle") == 0) { return reinterpret_cast<PFN_vkVoidFunction>(waitQueue); }
        if (std::strcmp(name, "vkDeviceWaitIdle") == 0) { return reinterpret_cast<PFN_vkVoidFunction>(waitDevice); }
        const auto& native = *static_cast<NativeDevice*>(context);
        return native.getInstanceProcAddr(native.instance, name);
    }
};

namespace {
ImGuiDisplayDesc displayDesc(Swapchain& swapchain, float paperWhiteNits)
{
    const bool pq = swapchain.outputMode() == DisplayOutputMode::HDR10_PQ;
    return {.colorFormat = pq ? Format::RGBA16Sfloat : swapchain.format(),
        .pqOutputFormat = pq ? swapchain.format() : Format::Unknown,
        .hdr = isHDROutput(swapchain.outputMode()), .paperWhiteNits = paperWhiteNits,
        .imageCount = swapchain.imageCount()};
}
bool validDisplay(const ImGuiDisplayDesc& desc)
{
    return desc.imageCount >= 2 && nativeFormat(desc.colorFormat) != VK_FORMAT_UNDEFINED &&
        std::isfinite(desc.paperWhiteNits) && desc.paperWhiteNits > 0;
}
} // namespace

VulkanImGuiBackend::VulkanImGuiBackend() : impl_(std::make_unique<Impl>())
{
}

VulkanImGuiBackend::~VulkanImGuiBackend()
{
    shutdown();
}

Result<> VulkanImGuiBackend::init(Device& device, Queue& queue, Swapchain& swapchain,
    ImGuiShaderServices shaders, float paperWhiteNits)
{
    return init(device, queue, displayDesc(swapchain, paperWhiteNits), std::move(shaders));
}

Result<> VulkanImGuiBackend::init(Device& device, Queue& queue, const ImGuiDisplayDesc& desc, ImGuiShaderServices shaders)
{
    if (impl_->initialized || !validDisplay(desc) || !shaders.load || !shaders.cache ||
        !ImGui::GetCurrentContext() || ImGui::GetIO().BackendRendererUserData) { return makeError(Error::InvalidArgument); }
    auto native = nativeDevice(device);
    const auto nativeQ = nativeQueue(queue);
    if (!native.device || !nativeQ.queue || device.getQueue(QueueType::Graphics) == nullptr ||
        !queue.sameQueue(*device.getQueue(QueueType::Graphics))) { return makeError(Error::InvalidArgument); }
    impl_->device = &device;
    impl_->queue = &queue;
    impl_->display.shaders = std::move(shaders);
    Impl::Call call(*impl_);
    if (!ImGui_ImplVulkan_LoadFunctions(native.apiVersion, Impl::load, &native)) { return makeError(Error::Failure); }
    const VkFormat format = nativeFormat(desc.colorFormat);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = native.apiVersion;
    info.Instance = native.instance;
    info.PhysicalDevice = native.physicalDevice;
    info.Device = native.device;
    info.QueueFamily = nativeQ.familyIndex;
    info.Queue = nativeQ.queue;
    info.DescriptorPoolSize = 128;
    info.MinImageCount = 2;
    info.ImageCount = desc.imageCount;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
    info.CheckVkResultFn = Impl::check;
    impl_->initialized = ImGui_ImplVulkan_Init(&info);
    if (impl_->error != VK_SUCCESS) { return resultFromVk(impl_->error); }
    if (!impl_->initialized) { return makeError(Error::Failure); }
    return configure(desc);
}

Result<> VulkanImGuiBackend::configure(Swapchain& swapchain, float paperWhiteNits)
{
    return configure(displayDesc(swapchain, paperWhiteNits));
}

Result<> VulkanImGuiBackend::configure(const ImGuiDisplayDesc& desc)
{
    if (!impl_->initialized || !validDisplay(desc)) { return makeError(Error::InvalidArgument); }
    Impl::Call call(*impl_);
    // Configuration may replace pipelines referenced by any ImGui viewport.
    auto idle = impl_->device->waitIdle();
    if (!idle) { return idle; }
    ImGui_ImplVulkan_SetMinImageCount(2);
    const VkFormat format = nativeFormat(desc.colorFormat);
    ImGui_ImplVulkan_PipelineInfo info{};
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineRenderingCreateInfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1, .pColorAttachmentFormats = &format};
    ImGui_ImplVulkan_CreateMainPipeline(&info);
    if (impl_->error != VK_SUCCESS) { return resultFromVk(impl_->error); }
    if (!impl_->display.initialize(*impl_->device, format, desc.hdr, desc.paperWhiteNits, nativeFormat(desc.pqOutputFormat))) {
        return makeError(Error::Failure);
    }
    return {};
}

void VulkanImGuiBackend::shutdown()
{
    if (!impl_->initialized) { return; }
    Impl::Call call(*impl_);
    (void)impl_->device->waitIdle();
    impl_->textures.clear();
    impl_->retired.clear();
    // ImGui destroys its descriptor pool. Entries retained by RHI have no native deleter.
    impl_->display.shutdown();
    ImGui_ImplVulkan_Shutdown();
    impl_->initialized = false;
}

Result<> VulkanImGuiBackend::newFrame()
{
    if (!impl_->initialized) { return makeError(Error::InvalidArgument); }
    Impl::Call call(*impl_);
    // Release descriptors only on the ImGui context thread, after every recording
    // has dropped its lease. Cancellation/reset also releases those leases.
    std::erase_if(impl_->retired, [](const auto& texture) {
        if (texture.use_count() != 1) { return false; }
        ImGui_ImplVulkan_RemoveTexture(texture->descriptor);
        return true;
    });
    ImGui_ImplVulkan_NewFrame();
    return resultFromVk(impl_->error);
}

Result<ImGuiTexture> VulkanImGuiBackend::addTexture(TextureView& view)
{
    if (!impl_->initialized || view.deviceIdentity() != impl_->device->identity()) { return makeError(Error::InvalidArgument); }
    auto retained = retainInteropView(view);
    if (!retained) { return std::unexpected(retained.error()); }
    Impl::Call call(*impl_);
    auto texture = std::make_shared<Impl::Texture>();
    texture->view = std::move(*retained);
    texture->descriptor = ImGui_ImplVulkan_AddTexture(nativeImageView(*texture->view), nativeImageLayout(view, TextureLayout::ShaderRead));
    if (impl_->error != VK_SUCCESS) { return std::unexpected(resultFromVk(impl_->error).error()); }
    if (!texture->descriptor) { return makeError(Error::Failure); }
    const auto id = reinterpret_cast<ImGuiTexture>(texture->descriptor);
    impl_->textures.emplace(id, std::move(texture));
    return id;
}

void VulkanImGuiBackend::removeTexture(ImGuiTexture texture)
{
    const auto entry = impl_->textures.find(texture);
    if (entry == impl_->textures.end()) { return; }
    impl_->retired.push_back(std::move(entry->second));
    impl_->textures.erase(entry);
}

void VulkanImGuiBackend::beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport)
{
    impl_->display.beginScRgbImage(list, viewport);
}

Result<> VulkanImGuiBackend::retainTextures(CommandBuffer& commands)
{
    if (!impl_->initialized || commands.deviceIdentity() != impl_->device->identity()) { return makeError(Error::InvalidArgument); }
    ExternalCommandScope scope(commands);
    for (const auto& [id, texture] : impl_->textures) {
        auto view = scope.imageView(*texture->view);
        if (!view) { return std::unexpected(view.error()); }
        auto result = commands.retainResource(texture);
        if (!result) { return result; }
    }
    return scope ? Result<>{} : makeError(Error::InvalidArgument);
}

Result<> VulkanImGuiBackend::render(CommandBuffer& commands)
{
    auto retained = retainTextures(commands);
    if (!retained) { return retained; }
    Impl::Call call(*impl_);
    ExternalCommandScope scope(commands);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), scope.commandBuffer(), impl_->display.mainPipeline());
    if (impl_->display.failed) { return makeError(Error::Failure); }
    return resultFromVk(impl_->error);
}

Result<> VulkanImGuiBackend::encodeHDR10(CommandBuffer& commands, ImGuiTexture source, uint32_t width, uint32_t height)
{
    if (!impl_->initialized || !width || !height || !impl_->display.canEncodeHDR10() ||
        !impl_->textures.contains(source)) { return makeError(Error::InvalidArgument); }
    const auto& texture = impl_->textures.at(source);
    ExternalCommandScope scope(commands);
    auto view = scope.imageView(*texture->view);
    if (!view) { return std::unexpected(view.error()); }
    auto retained = commands.retainResource(texture);
    if (!retained) { return retained; }
    impl_->display.encodeHDR10(scope.commandBuffer(), texture->descriptor, width, height);
    return {};
}

Result<> VulkanImGuiBackend::renderPlatformWindows()
{
    if (!impl_->initialized) { return makeError(Error::InvalidArgument); }
    Impl::Call call(*impl_);
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    if (impl_->display.failed) { return makeError(Error::Failure); }
    return resultFromVk(impl_->error);
}

} // namespace metallic::render::vulkan
