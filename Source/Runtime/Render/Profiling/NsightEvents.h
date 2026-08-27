#pragma once

#include <cstdint>
#include <string_view>

#if defined(METALLIC_NSIGHT_EVENTS_AVAILABLE) && __has_include(<nvtx3/nvToolsExt.h>)
// NVTX's implementation header includes <windows.h>. Keep that inclusion lean
// so it does not drag in winspool.h, whose ANSI/WIDE shims #define away common
// identifiers such as DeviceCapabilities (which collides with the RHI type).
#if defined(_WIN32) && !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#define METALLIC_NSIGHT_DEFINED_LEAN_AND_MEAN
#endif
#include <nvtx3/nvToolsExt.h>
#if defined(METALLIC_NSIGHT_DEFINED_LEAN_AND_MEAN)
#undef WIN32_LEAN_AND_MEAN
#undef METALLIC_NSIGHT_DEFINED_LEAN_AND_MEAN
#endif
// If <windows.h> was already included in full earlier in the translation unit,
// winspool.h already defined DeviceCapabilities as an ANSI/WIDE alias macro;
// remove it so the RHI type of the same name stays usable after this header.
#if defined(_WIN32)
#undef DeviceCapabilities
#endif
#define METALLIC_NSIGHT_EVENTS_ENABLED 1
#else
#define METALLIC_NSIGHT_EVENTS_ENABLED 0
#endif

namespace metallic::render::profiling {

// NVTX domains keep Metallic's markers separate from markers emitted by other
// libraries in the same process, and let Nsight tools filter the editor loop
// independently from the render backend.
enum class NsightDomain {
    Editor,
    Render,
};

// Fixed subsystem categories assigned to every NVTX range. The numeric values
// are stable identifiers: Nsight tools filter and group by category across runs,
// so they must not be renumbered.
enum class NsightCategory : uint32_t {
    Generic = 0,
    Frame = 1,
    EditorUi = 2,
    RenderGraph = 3,
    RenderPass = 4,
    QueueSubmit = 5,
    FenceWait = 6,
    ResourceUpload = 7,
};

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

inline uint32_t nsightCategoryColor(NsightCategory category)
{
    switch (category) {
    case NsightCategory::Frame:
        return 0xff0072b2u;
    case NsightCategory::EditorUi:
        return 0xffcc79a7u;
    case NsightCategory::RenderGraph:
        return 0xff56b4e9u;
    case NsightCategory::RenderPass:
        return 0xff8172b2u;
    case NsightCategory::QueueSubmit:
        return 0xffe69f00u;
    case NsightCategory::FenceWait:
        return 0xffd55e00u;
    case NsightCategory::ResourceUpload:
        return 0xff009e73u;
    case NsightCategory::Generic:
        break;
    }
    return 0xff808080u;
}

#if METALLIC_NSIGHT_EVENTS_ENABLED
inline nvtxDomainHandle_t nsightDomainHandle(NsightDomain domain)
{
    // NVTX recommends caching domain handles: every nvtxDomainCreate call
    // allocates a new handle even for an already-registered domain name. The
    // handles are intentionally never destroyed; domains live for the process
    // lifetime.
    switch (domain) {
    case NsightDomain::Editor:
    {
        static const nvtxDomainHandle_t handle = nvtxDomainCreateA("Metallic.Editor");
        return handle;
    }
    case NsightDomain::Render:
    {
        static const nvtxDomainHandle_t handle = nvtxDomainCreateA("Metallic.Render");
        return handle;
    }
    }
    return nullptr;
}
#endif

class NsightProfileRange {
public:
    NsightProfileRange(
        NsightDomain domain,
        const char* name,
        NsightCategory category,
        uint64_t payload = 0,
        uint32_t color = 0)
    {
#if METALLIC_NSIGHT_EVENTS_ENABLED
        domain_ = nsightDomainHandle(domain);
        nvtxEventAttributes_t eventAttributes{};
        eventAttributes.version = NVTX_VERSION;
        eventAttributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttributes.category = static_cast<uint32_t>(category);
        eventAttributes.colorType = NVTX_COLOR_ARGB;
        eventAttributes.color = color != 0 ? color : nsightCategoryColor(category);
        eventAttributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttributes.message.ascii = name;
        eventAttributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        eventAttributes.payload.ullValue = payload;
        nvtxDomainRangePushEx(domain_, &eventAttributes);
#else
        (void)domain;
        (void)name;
        (void)category;
        (void)payload;
        (void)color;
#endif
    }

    ~NsightProfileRange()
    {
#if METALLIC_NSIGHT_EVENTS_ENABLED
        nvtxDomainRangePop(domain_);
#endif
    }

    NsightProfileRange(const NsightProfileRange&) = delete;
    NsightProfileRange& operator=(const NsightProfileRange&) = delete;

private:
#if METALLIC_NSIGHT_EVENTS_ENABLED
    nvtxDomainHandle_t domain_ = nullptr;
#endif
};

} // namespace metallic::render::profiling
