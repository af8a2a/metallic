#pragma once

#include "frame_graph.h"
#include "gpu_driven_constants.h"
#include "rhi_backend.h"

#include <cstddef>
#include <cstdint>

namespace GpuDriven {

template <typename T>
inline FGBufferDesc makeStructuredBufferDesc(size_t elementCapacity,
                                             const char* debugName,
                                             bool hostVisible = false) {
    FGBufferDesc desc;
    desc.size = elementCapacity * sizeof(T);
    desc.hostVisible = hostVisible;
    desc.debugName = debugName;
    return desc;
}

inline FGBufferDesc makeDispatchCounterBufferDesc(const char* debugName,
                                                  bool hostVisible = true) {
    FGBufferDesc desc;
    desc.size = DispatchCounterLayout::kBufferSize;
    desc.hostVisible = hostVisible;
    desc.debugName = debugName;
    return desc;
}

inline uint32_t* dispatchCounterWords(RhiBuffer* buffer) {
    if (!buffer || buffer->size() < DispatchCounterLayout::kBufferSize) {
        return nullptr;
    }
    return static_cast<uint32_t*>(buffer->mappedData());
}

inline const uint32_t* dispatchCounterWords(const RhiBuffer* buffer) {
    return dispatchCounterWords(const_cast<RhiBuffer*>(buffer));
}

inline void seedDispatchCounterBuffer(RhiBuffer* buffer) {
    uint32_t* words = dispatchCounterWords(buffer);
    if (!words) {
        return;
    }

    words[DispatchCounterLayout::kCountWord] = 0u;
    words[DispatchCounterLayout::kDispatchXWord] = 0u;
    words[DispatchCounterLayout::kDispatchYWord] = 1u;
    words[DispatchCounterLayout::kDispatchZWord] = 1u;
}

inline void ensureDispatchCounterBufferInitialized(RhiBuffer* buffer,
                                                   const RhiBuffer*& initializedBuffer) {
    if (!buffer) {
        initializedBuffer = nullptr;
        return;
    }

    if (buffer == initializedBuffer) {
        return;
    }

    seedDispatchCounterBuffer(buffer);
    if (dispatchCounterWords(buffer)) {
        initializedBuffer = buffer;
    }
}

inline uint32_t readBuiltDispatchCount(const RhiBuffer* buffer) {
    const uint32_t* words = dispatchCounterWords(buffer);
    if (!words) {
        return 0u;
    }

    return words[DispatchCounterLayout::kDispatchXWord];
}

} // namespace GpuDriven
