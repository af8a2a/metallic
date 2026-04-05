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

inline FGBufferDesc makeIndirectGridCommandBufferDesc(const char* debugName,
                                                      bool hostVisible = true) {
    FGBufferDesc desc;
    desc.size = IndirectGridCommandLayout::kBufferSize;
    desc.hostVisible = hostVisible;
    desc.debugName = debugName;
    return desc;
}

template <typename T>
struct IndirectWorklistResources {
    FGResource payload;
    FGResource state;
};

template <typename T>
inline IndirectWorklistResources<T> createIndirectWorklist(FGBuilder& builder,
                                                           const char* payloadName,
                                                           const char* payloadDebugName,
                                                           size_t payloadCapacity,
                                                           const char* stateName,
                                                           const char* stateDebugName,
                                                           bool hostVisibleState = true) {
    IndirectWorklistResources<T> resources;
    resources.payload = builder.create(payloadName,
                                       makeStructuredBufferDesc<T>(payloadCapacity,
                                                                   payloadDebugName,
                                                                   false));
    resources.state = builder.create(stateName,
                                     makeIndirectGridCommandBufferDesc(stateDebugName,
                                                                       hostVisibleState));
    return resources;
}

inline FGBufferDesc makeDispatchCounterBufferDesc(const char* debugName,
                                                  bool hostVisible = true) {
    return makeIndirectGridCommandBufferDesc(debugName, hostVisible);
}

inline uint32_t* indirectGridCommandWords(RhiBuffer* buffer) {
    if (!buffer || buffer->size() < IndirectGridCommandLayout::kBufferSize) {
        return nullptr;
    }
    return static_cast<uint32_t*>(buffer->mappedData());
}

inline const uint32_t* indirectGridCommandWords(const RhiBuffer* buffer) {
    return indirectGridCommandWords(const_cast<RhiBuffer*>(buffer));
}

inline uint32_t* dispatchCounterWords(RhiBuffer* buffer) {
    return indirectGridCommandWords(buffer);
}

inline const uint32_t* dispatchCounterWords(const RhiBuffer* buffer) {
    return indirectGridCommandWords(buffer);
}

inline void seedIndirectGridCommandBuffer(RhiBuffer* buffer) {
    uint32_t* words = indirectGridCommandWords(buffer);
    if (!words) {
        return;
    }

    words[IndirectGridCommandLayout::kCountWord] = 0u;
    words[IndirectGridCommandLayout::kDispatchXWord] = 0u;
    words[IndirectGridCommandLayout::kDispatchYWord] = 1u;
    words[IndirectGridCommandLayout::kDispatchZWord] = 1u;
}

inline void seedDispatchCounterBuffer(RhiBuffer* buffer) {
    seedIndirectGridCommandBuffer(buffer);
}

inline void ensureIndirectGridCommandBufferInitialized(RhiBuffer* buffer,
                                                       const RhiBuffer*& initializedBuffer) {
    if (!buffer) {
        initializedBuffer = nullptr;
        return;
    }

    if (buffer == initializedBuffer) {
        return;
    }

    seedIndirectGridCommandBuffer(buffer);
    if (indirectGridCommandWords(buffer)) {
        initializedBuffer = buffer;
    }
}

inline void ensureDispatchCounterBufferInitialized(RhiBuffer* buffer,
                                                   const RhiBuffer*& initializedBuffer) {
    ensureIndirectGridCommandBufferInitialized(buffer, initializedBuffer);
}

inline uint32_t readBuiltIndirectGridCount(const RhiBuffer* buffer) {
    const uint32_t* words = indirectGridCommandWords(buffer);
    if (!words) {
        return 0u;
    }

    return words[IndirectGridCommandLayout::kDispatchXWord];
}

inline uint32_t readBuiltDispatchCount(const RhiBuffer* buffer) {
    return readBuiltIndirectGridCount(buffer);
}

} // namespace GpuDriven
