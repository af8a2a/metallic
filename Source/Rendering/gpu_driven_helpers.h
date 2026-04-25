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

template <typename IndirectLayout>
inline FGBufferDesc makeWorklistStateBufferDesc(const char* debugName,
                                                bool hostVisible = true) {
    FGBufferDesc desc;
    desc.size = IndirectLayout::kBufferSize;
    desc.hostVisible = hostVisible;
    desc.debugName = debugName;
    return desc;
}

inline FGBufferDesc makeIndirectGridCommandBufferDesc(const char* debugName,
                                                      bool hostVisible = true) {
    return makeWorklistStateBufferDesc<ComputeDispatchCommandLayout>(debugName, hostVisible);
}

template <typename T, typename IndirectLayout>
struct TypedIndirectWorklistResources {
    FGResource payload;
    FGResource state;
};

template <typename T>
using IndirectWorklistResources = TypedIndirectWorklistResources<T, MeshDispatchCommandLayout>;

template <typename T, typename IndirectLayout>
inline TypedIndirectWorklistResources<T, IndirectLayout>
createTypedIndirectWorklist(FGBuilder& builder,
                            const char* payloadName,
                            const char* payloadDebugName,
                            size_t payloadCapacity,
                            const char* stateName,
                            const char* stateDebugName,
                            bool hostVisibleState = true) {
    TypedIndirectWorklistResources<T, IndirectLayout> resources;
    resources.payload = builder.create(payloadName,
                                       makeStructuredBufferDesc<T>(payloadCapacity,
                                                                   payloadDebugName,
                                                                   false));
    resources.state = builder.create(stateName,
                                     makeWorklistStateBufferDesc<IndirectLayout>(stateDebugName,
                                                                                 hostVisibleState));
    return resources;
}

template <typename T>
inline IndirectWorklistResources<T> createIndirectWorklist(FGBuilder& builder,
                                                           const char* payloadName,
                                                           const char* payloadDebugName,
                                                           size_t payloadCapacity,
                                                           const char* stateName,
                                                           const char* stateDebugName,
                                                           bool hostVisibleState = true) {
    return createTypedIndirectWorklist<T, MeshDispatchCommandLayout>(builder,
                                                                     payloadName,
                                                                     payloadDebugName,
                                                                     payloadCapacity,
                                                                     stateName,
                                                                     stateDebugName,
                                                                     hostVisibleState);
}

inline FGBufferDesc makeDispatchCounterBufferDesc(const char* debugName,
                                                  bool hostVisible = true) {
    return makeWorklistStateBufferDesc<ComputeDispatchCommandLayout>(debugName, hostVisible);
}

template <typename IndirectLayout>
inline uint32_t* worklistStateWords(RhiBuffer* buffer) {
    if (!buffer || buffer->size() < IndirectLayout::kBufferSize) {
        return nullptr;
    }
    return static_cast<uint32_t*>(buffer->mappedData());
}

template <typename IndirectLayout>
inline const uint32_t* worklistStateWords(const RhiBuffer* buffer) {
    return worklistStateWords<IndirectLayout>(const_cast<RhiBuffer*>(buffer));
}

inline uint32_t* indirectGridCommandWords(RhiBuffer* buffer) {
    return worklistStateWords<ComputeDispatchCommandLayout>(buffer);
}

inline const uint32_t* indirectGridCommandWords(const RhiBuffer* buffer) {
    return worklistStateWords<ComputeDispatchCommandLayout>(buffer);
}

inline uint32_t* dispatchCounterWords(RhiBuffer* buffer) {
    return worklistStateWords<ComputeDispatchCommandLayout>(buffer);
}

inline const uint32_t* dispatchCounterWords(const RhiBuffer* buffer) {
    return worklistStateWords<ComputeDispatchCommandLayout>(buffer);
}

template <typename IndirectLayout>
inline void seedWorklistStateBuffer(RhiBuffer* buffer) {
    uint32_t* words = worklistStateWords<IndirectLayout>(buffer);
    if (!words) {
        return;
    }

    words[IndirectLayout::kWriteCursorWord] = 0u;
    words[IndirectLayout::kProducedCountWord] = 0u;
    words[IndirectLayout::kConsumedCountWord] = 0u;
    words[IndirectLayout::kDispatchXWord] = 0u;
    words[IndirectLayout::kDispatchYWord] = 1u;
    words[IndirectLayout::kDispatchZWord] = 1u;
}

inline void seedIndirectGridCommandBuffer(RhiBuffer* buffer) {
    seedWorklistStateBuffer<ComputeDispatchCommandLayout>(buffer);
}

inline void seedDispatchCounterBuffer(RhiBuffer* buffer) {
    seedWorklistStateBuffer<ComputeDispatchCommandLayout>(buffer);
}

template <typename IndirectLayout>
inline void ensureWorklistStateBufferInitialized(RhiBuffer* buffer,
                                                 const RhiBuffer*& initializedBuffer) {
    if (!buffer) {
        initializedBuffer = nullptr;
        return;
    }

    if (buffer == initializedBuffer) {
        return;
    }

    seedWorklistStateBuffer<IndirectLayout>(buffer);
    if (worklistStateWords<IndirectLayout>(buffer)) {
        initializedBuffer = buffer;
    }
}

inline void ensureIndirectGridCommandBufferInitialized(RhiBuffer* buffer,
                                                       const RhiBuffer*& initializedBuffer) {
    ensureWorklistStateBufferInitialized<ComputeDispatchCommandLayout>(buffer, initializedBuffer);
}

inline void ensureDispatchCounterBufferInitialized(RhiBuffer* buffer,
                                                   const RhiBuffer*& initializedBuffer) {
    ensureWorklistStateBufferInitialized<ComputeDispatchCommandLayout>(buffer, initializedBuffer);
}

template <typename IndirectLayout>
inline uint32_t readWorklistWriteCursor(const RhiBuffer* buffer) {
    const uint32_t* words = worklistStateWords<IndirectLayout>(buffer);
    if (!words) {
        return 0u;
    }

    return words[IndirectLayout::kWriteCursorWord];
}

template <typename IndirectLayout>
inline uint32_t readPublishedWorkItemCount(const RhiBuffer* buffer) {
    const uint32_t* words = worklistStateWords<IndirectLayout>(buffer);
    if (!words) {
        return 0u;
    }

    return words[IndirectLayout::kProducedCountWord];
}

template <typename IndirectLayout>
inline uint32_t readConsumedWorkItemCount(const RhiBuffer* buffer) {
    const uint32_t* words = worklistStateWords<IndirectLayout>(buffer);
    if (!words) {
        return 0u;
    }

    return words[IndirectLayout::kConsumedCountWord];
}

template <typename IndirectLayout>
inline uint32_t readBuiltIndirectGroupCount(const RhiBuffer* buffer) {
    const uint32_t* words = worklistStateWords<IndirectLayout>(buffer);
    if (!words) {
        return 0u;
    }

    return words[IndirectLayout::kDispatchXWord];
}

inline uint32_t readBuiltIndirectGridCount(const RhiBuffer* buffer) {
    return readPublishedWorkItemCount<ComputeDispatchCommandLayout>(buffer);
}

inline uint32_t readBuiltDispatchCount(const RhiBuffer* buffer) {
    return readPublishedWorkItemCount<ComputeDispatchCommandLayout>(buffer);
}

} // namespace GpuDriven
