#pragma once

#include "Runtime/Render/GAPI/rhi.h"

#include <cstdint>
#include <memory>
#include <string_view>

namespace metallic::render {

enum class HistorySlot : uint8_t {
    Current,
    Previous,
};

struct HistoryTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
    const TextureDesc* desc = nullptr;
    bool valid = false;
};

struct HistoryBufferRef {
    Buffer* buffer = nullptr;
    BufferView* view = nullptr;
    const BufferDesc* desc = nullptr;
    const BufferViewDesc* viewDesc = nullptr;
    bool valid = false;
};

class HistoryResourceManager {
public:
    HistoryResourceManager();
    ~HistoryResourceManager();

    HistoryResourceManager(HistoryResourceManager&&) noexcept;
    HistoryResourceManager& operator=(HistoryResourceManager&&) noexcept;

    HistoryResourceManager(const HistoryResourceManager&) = delete;
    HistoryResourceManager& operator=(const HistoryResourceManager&) = delete;

    Result initialize(Device& device);
    void reset();

    void beginFrame(uint64_t frameIndex);
    void invalidate(std::string_view name);
    void invalidateAll();

    Result ensureTexture(
        std::string_view name,
        const TextureDesc& desc,
        TextureViewDesc viewDesc = {});

    Result ensureBuffer(
        std::string_view name,
        const BufferDesc& desc,
        const BufferViewDesc* viewDesc = nullptr);

    HistoryTextureRef texture(std::string_view name, HistorySlot slot) const;
    HistoryBufferRef buffer(std::string_view name, HistorySlot slot) const;
    bool hasPrevious(std::string_view name) const;

    void markWritten(std::string_view name);

    Result transitionTexture(
        CommandBuffer& commandBuffer,
        std::string_view name,
        HistorySlot slot,
        ResourceState after,
        bool forceBarrier = false);

    Result transitionBuffer(
        CommandBuffer& commandBuffer,
        std::string_view name,
        HistorySlot slot,
        ResourceState after,
        bool forceBarrier = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
