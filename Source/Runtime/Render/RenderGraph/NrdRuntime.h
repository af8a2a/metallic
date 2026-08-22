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
#include <NRD.h>
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

using NrdUserTexturePool = std::array<NrdTextureRef, static_cast<size_t>(nrd::ResourceType::MAX_NUM)>;

// White-box NRD integration hosted above the backend. It consumes NRD's
// dispatch descriptions and records them exclusively through the public RHI.
class NrdRuntime {
public:
    NrdRuntime();
    ~NrdRuntime();

    NrdRuntime(NrdRuntime&&) noexcept;
    NrdRuntime& operator=(NrdRuntime&&) noexcept;

    NrdRuntime(const NrdRuntime&) = delete;
    NrdRuntime& operator=(const NrdRuntime&) = delete;

    Result initialize(
        Device& device,
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
    Result denoise(NrdDenoiserMode mode, CommandBuffer& commandBuffer, Streamer& streamer);
    Result denoiseIdentifiers(
        const nrd::Identifier* denoisers,
        uint32_t denoiserCount,
        CommandBuffer& commandBuffer,
        Streamer& streamer);

private:
    Result setDenoiserSettings(nrd::Identifier identifier, const void* settings);
    Result dispatch(
        CommandBuffer& commandBuffer,
        Streamer& streamer,
        const nrd::DispatchDesc& dispatchDesc,
        uint64_t& previousConstantAddress);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif

} // namespace metallic::render
