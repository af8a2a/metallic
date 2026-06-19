#pragma once

#include <cstdint>
#include <string_view>

#if defined(METALLIC_NSIGHT_EVENTS_AVAILABLE) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define METALLIC_NSIGHT_EVENTS_ENABLED 1
#else
#define METALLIC_NSIGHT_EVENTS_ENABLED 0
#endif

namespace metallic::render::profiling {

inline uint32_t nsightColorFromName(std::string_view name)
{
    uint32_t hash = 2166136261u;
    for (const char value : name) {
        hash ^= static_cast<uint8_t>(value);
        hash *= 16777619u;
    }

    const uint8_t r = static_cast<uint8_t>(96u + (hash & 0x7fu));
    const uint8_t g = static_cast<uint8_t>(96u + ((hash >> 8u) & 0x7fu));
    const uint8_t b = static_cast<uint8_t>(96u + ((hash >> 16u) & 0x7fu));
    return 0xff000000u |
        (static_cast<uint32_t>(r) << 16u) |
        (static_cast<uint32_t>(g) << 8u) |
        static_cast<uint32_t>(b);
}

class NsightProfileRange {
public:
    NsightProfileRange(const char* name, uint32_t color, uint32_t payload = 0)
    {
#if METALLIC_NSIGHT_EVENTS_ENABLED
        nvtxEventAttributes_t eventAttributes{};
        eventAttributes.version = NVTX_VERSION;
        eventAttributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttributes.colorType = NVTX_COLOR_ARGB;
        eventAttributes.color = color;
        eventAttributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttributes.message.ascii = name;
        eventAttributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        eventAttributes.payload.llValue = static_cast<int64_t>(payload);
        eventAttributes.category = payload;
        nvtxRangePushEx(&eventAttributes);
#else
        (void)name;
        (void)color;
        (void)payload;
#endif
    }

    ~NsightProfileRange()
    {
#if METALLIC_NSIGHT_EVENTS_ENABLED
        nvtxRangePop();
#endif
    }

    NsightProfileRange(const NsightProfileRange&) = delete;
    NsightProfileRange& operator=(const NsightProfileRange&) = delete;
};

} // namespace metallic::render::profiling
