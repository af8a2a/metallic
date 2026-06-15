#pragma once

#include "Runtime/Render/GAPI/rhi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifndef METALLIC_HAS_NRD
#define METALLIC_HAS_NRD 0
#endif

#if METALLIC_HAS_NRD
#include <NRD.h>
#include <volk.h>
#endif

namespace metallic::render::vulkan {

enum class NrdDenoiserMode : uint32_t {
    Reblur,
    Relax,
    Reference,
};

Format nrdNormalRoughnessFormat();

#if METALLIC_HAS_NRD

struct NrdTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
};

using NrdUserTexturePool = std::array<NrdTextureRef, static_cast<size_t>(nrd::ResourceType::MAX_NUM)>;

class NrdDenoiser {
public:
    NrdDenoiser();
    ~NrdDenoiser();

    NrdDenoiser(NrdDenoiser&&) noexcept;
    NrdDenoiser& operator=(NrdDenoiser&&) noexcept;

    NrdDenoiser(const NrdDenoiser&) = delete;
    NrdDenoiser& operator=(const NrdDenoiser&) = delete;

    Result initialize(
        Device& device,
        Queue& queue,
        uint16_t width,
        uint16_t height,
        const NrdUserTexturePool& userTexturePool,
        std::string& log);
    void clear();
    bool valid() const;

    uint16_t width() const;
    uint16_t height() const;

    void setUserPoolTexture(nrd::ResourceType resource, Texture& texture, TextureView& view);
    Result setCommonSettings(const nrd::CommonSettings& settings);
    Result setReblurSettings(const nrd::ReblurSettings& settings);
    Result setRelaxSettings(const nrd::RelaxSettings& settings);
    Result denoise(NrdDenoiserMode mode, CommandBuffer& commandBuffer);
    Result denoiseIdentifiers(
        const nrd::Identifier* denoisers,
        uint32_t denoiserCount,
        CommandBuffer& commandBuffer);

private:
    struct TextureResource {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
    };

    struct Pipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout resourceDescriptorLayout = VK_NULL_HANDLE;
        uint32_t bindingCount = 0;
    };

    Result createInternalTexture(
        Device& device,
        const nrd::TextureDesc& desc,
        uint16_t width,
        uint16_t height,
        TextureResource& outResource,
        std::string& log);
    Result initializeInternalTextureLayouts(Device& device, Queue& queue, std::string& log);
    Result createPipelines(std::string& log);
    Result setDenoiserSettings(nrd::Identifier identifier, const void* settings);
    Result dispatch(CommandBuffer& commandBuffer, const nrd::DispatchDesc& dispatchDesc);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif

} // namespace metallic::render::vulkan
