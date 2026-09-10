#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#ifndef METALLIC_HAS_NRD
#define METALLIC_HAS_NRD 0
#endif

#if METALLIC_HAS_NRD
#include "Runtime/Render/Denoising/NrdTypes.h"
#endif

namespace metallic::render {

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

using NrdUserTexturePool = std::array<NrdTextureRef, static_cast<size_t>(denoising::ResourceType::MAX_NUM)>;

// Vendored NRD kernels, scheduled and bound by Metallic. The caller must wait
// for the previous frame before reusing this temporal instance (the render graph
// enforces this through NrdDenoisePass::supportsFrameOverlap = false).
class NrdRuntime {
public:
    NrdRuntime();
    ~NrdRuntime();

    NrdRuntime(NrdRuntime&&) noexcept;
    NrdRuntime& operator=(NrdRuntime&&) noexcept;

    NrdRuntime(const NrdRuntime&) = delete;
    NrdRuntime& operator=(const NrdRuntime&) = delete;

    Result initialize(Device& device, uint16_t width, uint16_t height, const NrdUserTexturePool& userTexturePool,
                      std::string& log);
    void clear();
    bool valid() const;

    uint16_t width() const;
    uint16_t height() const;

    void setUserPoolTexture(denoising::ResourceType resource, Texture& texture, TextureView& view);
    Result setCommonSettings(const denoising::CommonSettings& settings);
    Result setReblurSettings(const denoising::ReblurSettings& settings);
    Result setRelaxSettings(const denoising::RelaxSettings& settings);
    Result denoise(NrdDenoiserMode mode, CommandBuffer& commandBuffer, Streamer& streamer);
    Result denoiseReference(bool specular, CommandBuffer& commandBuffer, Streamer& streamer);

private:
    Result record(uint32_t index, CommandBuffer& commandBuffer, Streamer& streamer);
    Result dispatch(CommandBuffer& commandBuffer, Streamer& streamer, const denoising::DispatchDesc& stage);

    struct Impl;
    std::shared_ptr<Impl> impl_;
};

#endif

} // namespace metallic::render
