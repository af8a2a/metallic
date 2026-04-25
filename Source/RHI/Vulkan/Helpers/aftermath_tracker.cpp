#include "aftermath_tracker.h"

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
#include <chrono>
#include <filesystem>
#include <thread>
#include <spdlog/spdlog.h>
#endif

AftermathTracker& AftermathTracker::getInstance() {
    static AftermathTracker instance;
    return instance;
}

AftermathTracker::AftermathTracker() {
#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    m_tracker = std::make_unique<AftermathGpuCrashTracker>(m_markers);
#endif
}

void AftermathTracker::initialize() {
#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    m_tracker->initialize();
    spdlog::info("Nsight Aftermath GPU crash tracking initialized");
#endif
}

void AftermathTracker::addShaderBinary(const std::span<const uint32_t>& data) {
#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    m_tracker->addShaderBinary(data);
#endif
}

void AftermathTracker::errorCallback(VkResult result) {
#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    if (result != VK_ERROR_DEVICE_LOST) return;

    auto timeout = std::chrono::seconds(5);
    auto tStart = std::chrono::steady_clock::now();

    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

    while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed
           && status != GFSDK_Aftermath_CrashDump_Status_Finished
           && std::chrono::steady_clock::now() - tStart < timeout) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));
    }

    if (status != GFSDK_Aftermath_CrashDump_Status_Finished) {
        spdlog::error("Aftermath: unexpected crash dump status: {}", static_cast<int>(status));
    }

    spdlog::info("Aftermath file dumped under: {}", std::filesystem::current_path().string());
#endif
}

bool AftermathTracker::isAvailable() const {
#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    return true;
#else
    return false;
#endif
}

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
AftermathGpuCrashTracker& AftermathTracker::gpuCrashTracker() {
    return *m_tracker;
}

const char* AftermathTracker::diagnosticsConfigExtensionName() const {
    return VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME;
}

VkPhysicalDeviceDiagnosticsConfigFeaturesNV* AftermathTracker::diagnosticsConfigFeatures() {
    return &m_diagnosticsConfigFeatures;
}

VkDeviceDiagnosticsConfigCreateInfoNV* AftermathTracker::diagnosticsConfigCreateInfo() {
    return &m_diagnosticsConfigInfo;
}
#endif
