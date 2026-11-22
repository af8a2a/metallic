#pragma once

#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/GAPI/RHI.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace metallic::render {

enum class HistorySlot : uint8_t {
    Current,
    Previous,
};

enum class HistoryInvalidationReason : uint8_t {
    Discontinuity,
    CameraMotion,
    LightingChange,
};

struct HistoryTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
    const TextureDesc* desc = nullptr;
    bool valid = false;
    ResourceState state = ResourceState::Undefined;
    uint64_t allocationGeneration = 0;
    uint64_t backingAllocationId = 0;
    uint64_t version = 0;
    uint64_t writtenFrameIndex = 0;
    std::shared_ptr<void> owner;
};

struct HistoryBufferRef {
    Buffer* buffer = nullptr;
    BufferView* view = nullptr;
    const BufferDesc* desc = nullptr;
    const BufferViewDesc* viewDesc = nullptr;
    bool valid = false;
    ResourceState state = ResourceState::Undefined;
    uint64_t allocationGeneration = 0;
    uint64_t backingAllocationId = 0;
    uint64_t version = 0;
    uint64_t writtenFrameIndex = 0;
    std::shared_ptr<void> owner;
};

class HistoryResourceManager {
public:
    HistoryResourceManager();
    ~HistoryResourceManager();

    HistoryResourceManager(HistoryResourceManager&&) noexcept;
    HistoryResourceManager& operator=(HistoryResourceManager&&) noexcept;

    HistoryResourceManager(const HistoryResourceManager&) = delete;
    HistoryResourceManager& operator=(const HistoryResourceManager&) = delete;

    Result<> initialize(Device& device, uint32_t slotCount = 2);
    void reset();
    uint32_t slotCount() const;
    bool isValidFrameOffset(int32_t frameOffset) const;

    void beginFrame(uint64_t frameIndex);
    uint64_t frameIndex() const;
    void invalidate(std::string_view name);
    void invalidateAll(HistoryInvalidationReason reason = HistoryInvalidationReason::Discontinuity);
    uint64_t invalidationRevision() const;
    uint64_t reprojectionInvalidationRevision() const;
    // Reprojected visibility can validate light changes independently. Such
    // consumers still own their camera, light and geometry content signatures.
    uint64_t geometryReprojectionInvalidationRevision() const;

    Result<> ensureTexture(
        std::string_view name,
        const TextureDesc& desc,
        TextureViewDesc viewDesc = {});

    Result<> ensureBuffer(
        std::string_view name,
        const BufferDesc& desc,
        const BufferViewDesc* viewDesc = nullptr);

    HistoryTextureRef texture(std::string_view name, HistorySlot slot) const;
    HistoryBufferRef buffer(std::string_view name, HistorySlot slot) const;
    HistoryTextureRef textureAtOffset(std::string_view name, int32_t frameOffset) const;
    HistoryBufferRef bufferAtOffset(std::string_view name, int32_t frameOffset) const;
    bool hasPrevious(std::string_view name) const;

    void markWritten(std::string_view name);

    // Publish the state produced by an external access plan without recording a
    // barrier. Cancellation restores the prior layout and invalidates contents.
    Result<> publishTextureState(
        CommandBuffer& commandBuffer,
        std::string_view name,
        HistorySlot slot,
        ResourceState state,
        bool written = false);
    Result<> publishTextureStateAtOffset(
        CommandBuffer& commandBuffer,
        std::string_view name,
        int32_t frameOffset,
        ResourceState state,
        bool written = false);
    Result<> publishBufferState(
        CommandBuffer& commandBuffer,
        std::string_view name,
        HistorySlot slot,
        ResourceState state,
        bool written = false);
    Result<> publishBufferStateAtOffset(
        CommandBuffer& commandBuffer,
        std::string_view name,
        int32_t frameOffset,
        ResourceState state,
        bool written = false);

    // Coordinator-only state commit called in deterministic queue-acceptance
    // order after recording tasks have joined.
    Result<> commitAcceptedTextureStateAtOffset(
        std::string_view name,
        int32_t frameOffset,
        ResourceState state,
        bool written = false,
        uint64_t expectedGeneration = 0,
        std::optional<uint32_t> explicitSlotIndex = std::nullopt);
    Result<> commitAcceptedBufferStateAtOffset(
        std::string_view name,
        int32_t frameOffset,
        ResourceState state,
        bool written = false,
        uint64_t expectedGeneration = 0,
        std::optional<uint32_t> explicitSlotIndex = std::nullopt);
    void invalidateSlotAtOffset(
        std::string_view name,
        int32_t frameOffset,
        uint64_t expectedGeneration = 0,
        std::optional<uint32_t> explicitSlotIndex = std::nullopt);
    std::optional<uint32_t> slotIndexForOffset(int32_t frameOffset) const;
    bool hasRecord(std::string_view name) const;
    uint64_t peekNextVersion(std::string_view name) const;

    Result<> transitionTexture(
        CommandBuffer& commandBuffer,
        std::string_view name,
        HistorySlot slot,
        ResourceState after,
        bool forceBarrier = false);

    Result<> transitionBuffer(
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
