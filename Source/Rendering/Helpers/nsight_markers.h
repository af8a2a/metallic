#pragma once

#include <cstdint>

#if defined(METALLIC_HAS_NVTX) && METALLIC_HAS_NVTX && __has_include(<nvtx3/nvToolsExt.h>)
#define METALLIC_ENABLE_NVTX 1
#else
#define METALLIC_ENABLE_NVTX 0
#endif

#if METALLIC_ENABLE_NVTX
#include <mutex>
#include <nvtx3/nvToolsExt.h>
#endif

namespace metallic {

inline constexpr bool kNsightMarkersAvailable =
#if METALLIC_ENABLE_NVTX
    true;
#else
    false;
#endif

inline bool nsightMarkersAvailable() {
    return kNsightMarkersAvailable;
}

#if METALLIC_ENABLE_NVTX

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

    ScopedNsightRange(const char* label, uint32_t colorArgb, int64_t payload) {
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
        attributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        attributes.payload.llValue = payload;
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
    ScopedNsightRange(const char*, uint32_t, int64_t) {}
};

#endif

} // namespace metallic

// --- Convenience macros (nvpro-style) ---

#if METALLIC_ENABLE_NVTX

#define METALLIC_NX_MARK(name) \
    do { ::metallic::initializeNsightMarkers(); nvtxMark(name); } while(0)

#define METALLIC_NX_RANGEPUSH(name) nvtxRangePush(name)

#define METALLIC_NX_RANGEPUSHCOL(name, c)                       \
    do {                                                        \
        nvtxEventAttributes_t _attr{};                          \
        _attr.version = NVTX_VERSION;                           \
        _attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;             \
        _attr.colorType = NVTX_COLOR_ARGB;                     \
        _attr.color = (c);                                      \
        _attr.messageType = NVTX_MESSAGE_TYPE_ASCII;            \
        _attr.message.ascii = (name);                           \
        nvtxRangePushEx(&_attr);                                \
    } while(0)

#define METALLIC_NX_RANGEPOP() nvtxRangePop()

#define METALLIC_CONCAT_IMPL(a, b) a##b
#define METALLIC_CONCAT(a, b) METALLIC_CONCAT_IMPL(a, b)

#define METALLIC_PROFILE_FUNC(name) \
    ::metallic::ScopedNsightRange METALLIC_CONCAT(_metallicNxProf_, __LINE__)(name, 0xFF0000FFu)

#define METALLIC_PROFILE_FUNC_COL(name, c) \
    ::metallic::ScopedNsightRange METALLIC_CONCAT(_metallicNxProf_, __LINE__)(name, c)

#define METALLIC_PROFILE_FUNC_COL2(name, c, payload) \
    ::metallic::ScopedNsightRange METALLIC_CONCAT(_metallicNxProf_, __LINE__)(name, c, static_cast<int64_t>(payload))

#else

#define METALLIC_NX_MARK(name) ((void)0)
#define METALLIC_NX_RANGEPUSH(name) ((void)0)
#define METALLIC_NX_RANGEPUSHCOL(name, c) ((void)0)
#define METALLIC_NX_RANGEPOP() ((void)0)
#define METALLIC_PROFILE_FUNC(name) ((void)0)
#define METALLIC_PROFILE_FUNC_COL(name, c) ((void)0)
#define METALLIC_PROFILE_FUNC_COL2(name, c, payload) ((void)0)

#endif
