#pragma once

#include "rhi_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class StreamingStorage {
public:
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

    void resetAllocator() {
        m_freeRanges.clear();
        if (m_capacityElements > 0u) {
            m_freeRanges.push_back({0u, m_capacityElements});
        }
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
    std::unique_ptr<RhiBuffer> m_buffer;
    uint32_t m_capacityElements = 0u;
    std::vector<Range> m_freeRanges;
};
