#include "ImGuiDisplayShaders.h"
#include "ShaderRegistry.h"
#include "ShaderRequests.h"
#include "ResourceRegistry.h"
#include <spdlog/spdlog.h>

namespace metallic::render {
vulkan::ImGuiShaderServices imGuiDisplayShaders()
{
    return {
        .load = [](Device& device, const vulkan::ImGuiShaderRequest& options, bool vertex) -> Result<std::unique_ptr<ShaderModule>> {
            const char* entry = vertex ? (options.encodePQ ? "editorOutputVertex" : "editorDisplayVertex") :
                (options.encodePQ ? "editorOutputPQFragment" : "editorDisplayFragment");
            const auto request = makeEditorDisplayShaderRequest(entry, options.hdr, options.scRgbImage,
                options.targetSrgb, std::to_string(options.paperWhiteNits));
            const ShaderRequestView source(request);
            std::string diagnostics;
            auto shader = ShaderRegistry::instance().getShader(source.desc(), diagnostics);
            if (!shader) {
                spdlog::error("ImGui display shader: {}", diagnostics);
                return std::unexpected(shader.error());
            }
            return ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv});
        },
        .cache = [](Device& device, uint64_t hash, const std::function<Result<>(PipelineCache&)>& factory) {
            return ShaderRegistry::instance().getExternalGraphicsPipeline(device, hash, factory);
        },
    };
}
Result<> encodeEditorHDR10(vulkan::VulkanImGuiBackend& backend, Device& device,
    CommandBuffer& commands, TextureView& source, uint32_t width, uint32_t height)
{
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    auto lease = (*registry)->sampledImage(source);
    if (!lease) { return std::unexpected(lease.error()); }
    auto result = (*registry)->retain(commands, *lease);
    if (!result) { return result; }
    return backend.encodeHDR10(commands, *(*registry)->heap(), lease->descriptorHandle(), width, height);
}
} // namespace metallic::render
