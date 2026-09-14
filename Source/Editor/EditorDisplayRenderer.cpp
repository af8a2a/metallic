#include "Editor/EditorDisplayRenderer.h"
#include "Runtime/Render/SlangCompiler.h"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <string>

namespace metallic {

EditorDisplayRenderer::~EditorDisplayRenderer()
{
    shutdown();
}

void EditorDisplayRenderer::shutdown()
{
    if (device_ == VK_NULL_HANDLE) { return; }
    for (VkPipeline pipeline : {mainPipeline_, mainImagePipeline_}) {
        if (pipeline) { vkDestroyPipeline(device_, pipeline, nullptr); }
    }
    for (const auto& [format, pipeline] : secondaryImagePipelines_) {
        if (pipeline) { vkDestroyPipeline(device_, pipeline, nullptr); }
    }
    secondaryImagePipelines_.clear();
    if (layout_) { vkDestroyPipelineLayout(device_, layout_, nullptr); }
    for (VkDescriptorSetLayout layout : setLayouts_) {
        if (layout) { vkDestroyDescriptorSetLayout(device_, layout, nullptr); }
    }
    mainPipeline_ = mainImagePipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    setLayouts_[0] = setLayouts_[1] = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

bool EditorDisplayRenderer::initialize(VkDevice device, VkFormat mainFormat, bool hdr, float paperWhiteNits)
{
    if (device_ == device && mainPipeline_ && mainImagePipeline_ && mainFormat_ == mainFormat &&
        hdr_ == hdr && paperWhiteNits_ == paperWhiteNits) { return true; }
    shutdown();
    device_ = device;
    mainFormat_ = mainFormat;
    hdr_ = hdr;
    paperWhiteNits_ = paperWhiteNits;
    for (uint32_t index = 0; index < 2; ++index) {
        VkDescriptorSetLayoutBinding binding{.binding = 0,
            .descriptorType = index == 0 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &binding};
        if (vkCreateDescriptorSetLayout(device_, &info, nullptr, &setLayouts_[index]) != VK_SUCCESS) { return false; }
    }
    VkPushConstantRange push{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = 16};
    VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = setLayouts_, .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
    if (vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &layout_) != VK_SUCCESS) { return false; }
    mainPipeline_ = createPipeline(mainFormat, hdr, false, paperWhiteNits);
    mainImagePipeline_ = createPipeline(mainFormat, hdr, true, paperWhiteNits);
    return mainPipeline_ && mainImagePipeline_;
}

VkPipeline EditorDisplayRenderer::createPipeline(VkFormat format, bool hdr, bool scRgbImage, float paperWhiteNits)
{
    const std::string white = std::to_string(paperWhiteNits);
    const render::SlangMacroDefine defines[] = {
        {"DISPLAY_HDR", hdr ? "1" : "0"}, {"DISPLAY_SCRGB_IMAGE", scRgbImage ? "1" : "0"},
        {"DISPLAY_SRGB_ATTACHMENT", format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB ? "1" : "0"},
        {"DISPLAY_WHITE_NITS", white.c_str()},
    };
    VkShaderModule modules[2]{};
    auto destroyModules = [&] {
        for (auto module : modules) { if (module) { vkDestroyShaderModule(device_, module, nullptr); } }
    };
    const char* entries[] = {"editorDisplayVertex", "editorDisplayFragment"};
    for (uint32_t index = 0; index < 2; ++index) {
        render::ShaderCompileResult shader;
        if (!render::compileSlangShaderToSpirv({.moduleName = "Features/PostProcess/EditorDisplay",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .macroDefines = defines, .macroDefineCount = 4}, shader)) {
            spdlog::error("Editor display shader: {}", shader.diagnostics);
            destroyModules();
            return VK_NULL_HANDLE;
        }
        VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shader.spirv.size() * sizeof(uint32_t), .pCode = shader.spirv.data()};
        if (vkCreateShaderModule(device_, &info, nullptr, &modules[index]) != VK_SUCCESS) {
            destroyModules();
            return VK_NULL_HANDLE;
        }
    }
    VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = modules[0], .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = modules[1], .pName = "main"},
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
    const VkResult result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    destroyModules();
    if (result != VK_SUCCESS) { spdlog::error("Editor display pipeline creation failed: {}", int(result)); }
    return pipeline;
}

void EditorDisplayRenderer::bindImagePipeline(const ImDrawList*, const ImDrawCmd* command)
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
    if (pipeline) { vkCmdBindPipeline(state->CommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline); }
}

void EditorDisplayRenderer::beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport)
{
    ImageCallbackData data{this, viewport};
    list.AddCallback(bindImagePipeline, &data, sizeof(data));
}

} // namespace metallic
