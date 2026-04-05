#pragma once

#include <cstdint>

#if defined(METALLIC_HAS_NVTX) && METALLIC_HAS_NVTX
#include <mutex>
#include <nvtx3/nvToolsExt.h>
#endif

namespace metallic {

inline constexpr bool kNsightMarkersAvailable =
#if defined(METALLIC_HAS_NVTX) && METALLIC_HAS_NVTX
    true;
#else
    false;
#endif

inline bool nsightMarkersAvailable() {
    return kNsightMarkersAvailable;
}

#if defined(METALLIC_HAS_NVTX) && METALLIC_HAS_NVTX

inline void initializeNsightMarkers() {
    static std::once_flag once;
    std::call_once(once, []() { nvtxInitialize(nullptr); });
}

class ScopedNsightRange {
public:
    ScopedNsightRange(const char* label, uint32_t colorArgb) {
        if (label == nullptr || label[0] == '\0') {
            return;
        }

        initializeNsightMarkers();

        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = colorArgb;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = label;
        nvtxRangePushEx(&attributes);
        m_active = true;
    }

    ~ScopedNsightRange() {
        if (m_active) {
            nvtxRangePop();
        }
    }

    ScopedNsightRange(const ScopedNsightRange&) = delete;
    ScopedNsightRange& operator=(const ScopedNsightRange&) = delete;

private:
    bool m_active = false;
};

#else

class ScopedNsightRange {
public:
    ScopedNsightRange(const char*, uint32_t) {}
};

#endif

} // namespace metallic
