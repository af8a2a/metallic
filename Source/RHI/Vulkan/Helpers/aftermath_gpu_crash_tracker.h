#pragma once

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH

#include <array>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "aftermath_helpers.h"

class AftermathGpuCrashTracker {
public:
    static constexpr unsigned int kMarkerFrameHistory = 4;
    using MarkerMap = std::array<std::map<uint64_t, std::string>, kMarkerFrameHistory>;

    AftermathGpuCrashTracker(const MarkerMap& markerMap);
    ~AftermathGpuCrashTracker();

    void initialize();
    void addShaderBinary(const std::span<const uint32_t>& data);

private:
    void onCrashDump(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize);
    void onShaderDebugInfo(const void* pShaderDebugInfo, uint32_t shaderDebugInfoSize);
    void onDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription);
    void onResolveMarker(const void* pMarkerData, uint32_t markerDataSize,
                         PFN_GFSDK_Aftermath_ResolveMarker resolveMarker);

    void writeGpuCrashDumpToFile(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize);
    void writeShaderDebugInfoToFile(GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
                                    const void* pData, uint32_t dataSize,
                                    const std::string& extension);

    void onShaderDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier,
                                 PFN_GFSDK_Aftermath_SetData setShaderDebugInfo) const;
    void onShaderLookup(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash,
                        PFN_GFSDK_Aftermath_SetData setShaderBinary) const;
    void onShaderSourceDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
                                       PFN_GFSDK_Aftermath_SetData setShaderBinary) const;

    bool findShaderBinary(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash,
                          std::vector<uint8_t>& shader) const;
    bool findShaderBinaryWithDebugData(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
                                       std::vector<uint8_t>& shader) const;

    static void gpuCrashDumpCallback(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize, void* pUserData);
    static void shaderDebugInfoCallback(const void* pShaderDebugInfo, uint32_t shaderDebugInfoSize, void* pUserData);
    static void crashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData);
    static void resolveMarkerCallback(const void* pMarkerData, uint32_t markerDataSize, void* pUserData,
                                      PFN_GFSDK_Aftermath_ResolveMarker resolveMarker);
    static void shaderDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
                                              PFN_GFSDK_Aftermath_SetData setShaderDebugInfo, void* pUserData);
    static void shaderLookupCallback(const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
                                     PFN_GFSDK_Aftermath_SetData setShaderBinary, void* pUserData);
    static void shaderSourceDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
                                                    PFN_GFSDK_Aftermath_SetData setShaderBinary, void* pUserData);

    bool m_initialized = false;
    mutable std::mutex m_mutex;
    std::map<GFSDK_Aftermath_ShaderDebugInfoIdentifier, std::vector<uint8_t>> m_shaderDebugInfo;
    std::map<GFSDK_Aftermath_ShaderBinaryHash, std::vector<uint8_t>> m_shaderBinaries;
    std::map<GFSDK_Aftermath_ShaderDebugName, std::vector<uint8_t>> m_shaderBinariesWithDebugInfo;
    const MarkerMap& m_markerMap;
};

#endif // METALLIC_HAS_AFTERMATH
