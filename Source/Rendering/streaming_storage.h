#pragma once

#include "rhi_backend.h"
#include "rhi_resource_utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

class StreamingStorage {
public:
    struct CopyRegion {
        uint64_t srcOffsetBytes = 0u;
        uint64_t dstOffsetBytes = 0u;
        uint64_t sizeBytes = 0u;
    };

    struct Range {
        uint32_t offset = 0u;
        uint32_t count = 0u;
    };

    bool ready() const {
        return m_buffer && m_capacityElements > 0u;
    }

    const RhiBuffer* buffer() const { return m_buffer.get(); }
    RhiBuffer* buffer() { return m_buffer.get(); }

    uint32_t capacityElements() const { return m_capacityElements; }
    uint64_t maxUploadBytesPerFrame() const { return m_maxUploadBytesPerFrame; }

    uint32_t usedElements() const {
        if (m_capacityElements == 0u) {
            return 0u;
        }

        uint32_t freeCount = 0u;
        for (const Range& range : m_freeRanges) {
            freeCount += range.count;
        }
        return m_capacityElements - std::min(m_capacityElements, freeCount);
    }

    void clear() {
        m_buffer.reset();
        m_capacityElements = 0u;
        m_freeRanges.clear();
        clearUploadState();
    }

    bool ensureBuffer(RhiFrameGraphBackend& resourceFactory,
                      uint32_t capacityElements,
                      const char* debugName) {
        capacityElements = std::max(1u, capacityElements);
        if (m_buffer && m_capacityElements == capacityElements) {
            return true;
        }

        RhiBufferDesc desc{};
        desc.size = size_t(capacityElements) * sizeof(uint32_t);
        desc.hostVisible = false;
        desc.debugName = debugName;

        std::unique_ptr<RhiBuffer> buffer = resourceFactory.createBuffer(desc);
        if (!buffer) {
            clear();
            return false;
        }

        m_buffer = std::move(buffer);
        m_capacityElements = capacityElements;
        resetAllocator();
        return true;
    }

    bool ensureUploadBuffers(RhiFrameGraphBackend& resourceFactory,
                             uint32_t framesInFlight,
                             uint64_t maxUploadBytesPerFrame,
                             const char* debugNamePrefix) {
        maxUploadBytesPerFrame = std::max<uint64_t>(maxUploadBytesPerFrame, sizeof(uint32_t));
        if (framesInFlight == 0u) {
            clearUploadState();
            return false;
        }

        const bool needsRecreate =
            m_uploadFrames.size() != framesInFlight ||
            m_maxUploadBytesPerFrame != maxUploadBytesPerFrame;
        if (!needsRecreate) {
            return uploadReady();
        }

        m_uploadFrames.clear();
        m_uploadFrames.resize(framesInFlight);
        m_maxUploadBytesPerFrame = maxUploadBytesPerFrame;

        for (uint32_t frameIndex = 0u; frameIndex < framesInFlight; ++frameIndex) {
            UploadFrame& uploadFrame = m_uploadFrames[frameIndex];

            RhiBufferDesc desc{};
            desc.size = static_cast<size_t>(maxUploadBytesPerFrame);
            desc.hostVisible = true;
            const std::string debugName =
                std::string(debugNamePrefix ? debugNamePrefix : "StreamingUpload") +
                "[" + std::to_string(frameIndex) + "]";
            desc.debugName = debugName.c_str();
            uploadFrame.stagingBuffer = resourceFactory.createBuffer(desc);
            uploadFrame.usedBytes = 0u;
            uploadFrame.copyRegions.clear();
            if (!uploadFrame.stagingBuffer || rhiBufferContents(*uploadFrame.stagingBuffer) == nullptr) {
                clearUploadState();
                return false;
            }
        }

        return true;
    }

    void resetAllocator() {
        m_freeRanges.clear();
        if (m_capacityElements > 0u) {
            m_freeRanges.push_back({0u, m_capacityElements});
        }
    }

    bool uploadReady() const {
        if (m_uploadFrames.empty()) {
            return false;
        }

        for (const UploadFrame& uploadFrame : m_uploadFrames) {
            if (!uploadFrame.stagingBuffer || rhiBufferContents(*uploadFrame.stagingBuffer) == nullptr) {
                return false;
            }
        }
        return true;
    }

    void resetUploadFrame(uint32_t frameSlot) {
        if (frameSlot >= m_uploadFrames.size()) {
            return;
        }

        UploadFrame& uploadFrame = m_uploadFrames[frameSlot];
        uploadFrame.usedBytes = 0u;
        uploadFrame.copyRegions.clear();
    }

    bool stageUpload(uint32_t frameSlot,
                     const void* data,
                     uint64_t sizeBytes,
                     uint64_t dstOffsetBytes,
                     uint64_t alignmentBytes = 16u) {
        if (frameSlot >= m_uploadFrames.size()) {
            return false;
        }
        if (sizeBytes == 0u) {
            return true;
        }
        if (!data) {
            return false;
        }

        UploadFrame& uploadFrame = m_uploadFrames[frameSlot];
        uint8_t* mappedBytes =
            uploadFrame.stagingBuffer
                ? static_cast<uint8_t*>(rhiBufferContents(*uploadFrame.stagingBuffer))
                : nullptr;
        if (!mappedBytes) {
            return false;
        }

        const uint64_t alignedOffset = alignUp(uploadFrame.usedBytes, alignmentBytes);
        if (alignedOffset + sizeBytes > m_maxUploadBytesPerFrame) {
            return false;
        }

        std::memcpy(mappedBytes + alignedOffset, data, static_cast<size_t>(sizeBytes));
        uploadFrame.copyRegions.push_back({alignedOffset, dstOffsetBytes, sizeBytes});
        uploadFrame.usedBytes = alignedOffset + sizeBytes;
        return true;
    }

    const RhiBuffer* uploadBuffer(uint32_t frameSlot) const {
        return frameSlot < m_uploadFrames.size() ? m_uploadFrames[frameSlot].stagingBuffer.get()
                                                 : nullptr;
    }

    const std::vector<CopyRegion>& copyRegions(uint32_t frameSlot) const {
        static const std::vector<CopyRegion> kEmpty;
        return frameSlot < m_uploadFrames.size() ? m_uploadFrames[frameSlot].copyRegions : kEmpty;
    }

    uint64_t uploadBytesUsed(uint32_t frameSlot) const {
        return frameSlot < m_uploadFrames.size() ? m_uploadFrames[frameSlot].usedBytes : 0u;
    }

    bool allocate(uint32_t elementCount, uint32_t& outOffset) {
        if (elementCount == 0u) {
            outOffset = 0u;
            return true;
        }

        for (size_t rangeIndex = 0; rangeIndex < m_freeRanges.size(); ++rangeIndex) {
            Range& range = m_freeRanges[rangeIndex];
            if (range.count < elementCount) {
                continue;
            }

            outOffset = range.offset;
            range.offset += elementCount;
            range.count -= elementCount;
            if (range.count == 0u) {
                m_freeRanges.erase(m_freeRanges.begin() +
                                   static_cast<std::vector<Range>::difference_type>(rangeIndex));
            }
            return true;
        }

        return false;
    }

    void release(uint32_t offset, uint32_t elementCount) {
        if (elementCount == 0u || offset == UINT32_MAX) {
            return;
        }

        Range releasedRange{offset, elementCount};
        auto insertIt = std::lower_bound(
            m_freeRanges.begin(),
            m_freeRanges.end(),
            releasedRange.offset,
            [](const Range& range, uint32_t value) { return range.offset < value; });
        m_freeRanges.insert(insertIt, releasedRange);

        if (m_freeRanges.empty()) {
            return;
        }

        std::vector<Range> mergedRanges;
        mergedRanges.reserve(m_freeRanges.size());
        for (const Range& range : m_freeRanges) {
            if (!mergedRanges.empty()) {
                Range& previous = mergedRanges.back();
                if (previous.offset + previous.count == range.offset) {
                    previous.count += range.count;
                    continue;
                }
            }
            mergedRanges.push_back(range);
        }

        m_freeRanges = std::move(mergedRanges);
    }

private:
    struct UploadFrame {
        std::unique_ptr<RhiBuffer> stagingBuffer;
        uint64_t usedBytes = 0u;
        std::vector<CopyRegion> copyRegions;
    };

    static uint64_t alignUp(uint64_t value, uint64_t alignment) {
        if (alignment == 0u) {
            return value;
        }
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

    void clearUploadState() {
        m_uploadFrames.clear();
        m_maxUploadBytesPerFrame = 0u;
    }

    std::unique_ptr<RhiBuffer> m_buffer;
    uint32_t m_capacityElements = 0u;
    std::vector<Range> m_freeRanges;
    std::vector<UploadFrame> m_uploadFrames;
    uint64_t m_maxUploadBytesPerFrame = 0u;
};
