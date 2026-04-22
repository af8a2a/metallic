#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <array>

#include <spdlog/spdlog.h>

#include "aftermath_gpu_crash_tracker.h"

AftermathGpuCrashTracker::AftermathGpuCrashTracker(const MarkerMap& markerMap)
    : m_markerMap(markerMap) {}

AftermathGpuCrashTracker::~AftermathGpuCrashTracker() {
    if (m_initialized) {
        GFSDK_Aftermath_DisableGpuCrashDumps();
    }
}

void AftermathGpuCrashTracker::initialize() {
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_EnableGpuCrashDumps(
        GFSDK_Aftermath_Version_API,
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
        gpuCrashDumpCallback,
        shaderDebugInfoCallback,
        crashDumpDescriptionCallback,
        resolveMarkerCallback,
        this));
    m_initialized = true;
}

void AftermathGpuCrashTracker::onCrashDump(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize) {
    std::lock_guard<std::mutex> lock(m_mutex);
    writeGpuCrashDumpToFile(pGpuCrashDump, gpuCrashDumpSize);
}

void AftermathGpuCrashTracker::onShaderDebugInfo(const void* pShaderDebugInfo, uint32_t shaderDebugInfoSize) {
    std::lock_guard<std::mutex> lock(m_mutex);

    GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier = {};
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderDebugInfoIdentifier(
        GFSDK_Aftermath_Version_API, pShaderDebugInfo, shaderDebugInfoSize, &identifier));

    auto* bytes = static_cast<const uint8_t*>(pShaderDebugInfo);
    m_shaderDebugInfo[identifier] = std::vector<uint8_t>(bytes, bytes + shaderDebugInfoSize);

    writeShaderDebugInfoToFile(identifier, pShaderDebugInfo, shaderDebugInfoSize, ".nvdbg");

    std::vector<uint8_t> code;
    GFSDK_Aftermath_ShaderBinaryHash hash = {.hash = identifier.id[0]};
    if (findShaderBinary(hash, code)) {
        writeShaderDebugInfoToFile(identifier, code.data(), static_cast<uint32_t>(code.size()), ".spv");
    }
}

void AftermathGpuCrashTracker::onDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription) {
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "Metallic");
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "v1.0");
}

void AftermathGpuCrashTracker::onResolveMarker(const void* pMarkerData, uint32_t /*markerDataSize*/,
                                                PFN_GFSDK_Aftermath_ResolveMarker resolveMarker) {
    for (auto& map : m_markerMap) {
        auto it = map.find(reinterpret_cast<uint64_t>(pMarkerData));
        if (it != map.end()) {
            resolveMarker(it->second.data(), static_cast<uint32_t>(it->second.length()));
            return;
        }
    }
}

void AftermathGpuCrashTracker::writeGpuCrashDumpToFile(const void* pGpuCrashDump, uint32_t gpuCrashDumpSize) {
    spdlog::warn("--------------------------------------------------------------");

    GFSDK_Aftermath_GpuCrashDump_Decoder decoder = {};
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
        GFSDK_Aftermath_Version_API, pGpuCrashDump, gpuCrashDumpSize, &decoder));

    GFSDK_Aftermath_GpuCrashDump_BaseInfo baseInfo = {};
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetBaseInfo(decoder, &baseInfo));

    uint32_t applicationNameLength = 0;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescriptionSize(
        decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, &applicationNameLength));
    std::vector<char> applicationName(applicationNameLength, '\0');
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescription(
        decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
        static_cast<uint32_t>(applicationName.size()), applicationName.data()));

    static int count = 0;
    const std::string baseFileName =
        std::string(applicationName.data()) + "-" + std::to_string(baseInfo.pid) + "-" + std::to_string(++count);
    // PLACEHOLDER_WRITEDUMP_CONTINUE

    auto exePath = std::filesystem::current_path();
    const auto crashDumpFileName = exePath / (baseFileName + ".nv-gpudmp");
    std::ofstream dumpFile(crashDumpFileName, std::ios::out | std::ios::binary);
    if (dumpFile) {
        dumpFile.write(static_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
        dumpFile.close();
    }
    spdlog::warn("Writing Aftermath dump file to: {}", crashDumpFileName.string());

    uint32_t jsonSize = 0;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
        decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
        GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
        shaderDebugInfoLookupCallback, shaderLookupCallback,
        shaderSourceDebugInfoLookupCallback, this, &jsonSize));
    std::vector<char> json(jsonSize);
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetJSON(
        decoder, static_cast<uint32_t>(json.size()), json.data()));

    const auto jsonFileName = exePath / (baseFileName + ".json");
    std::ofstream jsonFile(jsonFileName, std::ios::out | std::ios::binary);
    if (jsonFile) {
        jsonFile.write(json.data(), json.size() - 1);
        jsonFile.close();
    }
    spdlog::warn("Writing JSON dump file to: {}", jsonFileName.string());

    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder));
    spdlog::warn("--------------------------------------------------------------");
}

void AftermathGpuCrashTracker::writeShaderDebugInfoToFile(
    GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
    const void* pData, uint32_t dataSize, const std::string& extension) {
    auto exePath = std::filesystem::current_path();
    const auto filePath = exePath / ("shader-" + std::to_string(identifier) + extension);
    std::ofstream f(filePath, std::ios::out | std::ios::binary);
    if (f) {
        f.write(static_cast<const char*>(pData), dataSize);
    }
    spdlog::warn("Saved {}", filePath.string());
}
    // PLACEHOLDER_LOOKUPS

void AftermathGpuCrashTracker::onShaderDebugInfoLookup(
    const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier,
    PFN_GFSDK_Aftermath_SetData setShaderDebugInfo) const {
    auto it = m_shaderDebugInfo.find(identifier);
    if (it != m_shaderDebugInfo.end()) {
        setShaderDebugInfo(it->second.data(), static_cast<uint32_t>(it->second.size()));
    }
}

void AftermathGpuCrashTracker::onShaderLookup(
    const GFSDK_Aftermath_ShaderBinaryHash& shaderHash,
    PFN_GFSDK_Aftermath_SetData setShaderBinary) const {
    std::vector<uint8_t> shader;
    if (findShaderBinary(shaderHash, shader)) {
        setShaderBinary(shader.data(), static_cast<uint32_t>(shader.size()));
    }
}

void AftermathGpuCrashTracker::onShaderSourceDebugInfoLookup(
    const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
    PFN_GFSDK_Aftermath_SetData setShaderBinary) const {
    std::vector<uint8_t> shader;
    if (findShaderBinaryWithDebugData(shaderDebugName, shader)) {
        setShaderBinary(shader.data(), static_cast<uint32_t>(shader.size()));
    }
}

void AftermathGpuCrashTracker::addShaderBinary(const std::span<const uint32_t>& data) {
    std::lock_guard<std::mutex> lock(m_mutex);

    GFSDK_Aftermath_ShaderBinaryHash shaderHash{};
    GFSDK_Aftermath_SpirvCode shader{};
    shader.pData = data.data();
    shader.size = data.size() * sizeof(uint32_t);
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderHashSpirv(
        GFSDK_Aftermath_Version_API, &shader, &shaderHash));

    const auto* bytes = reinterpret_cast<const uint8_t*>(data.data());
    m_shaderBinaries[shaderHash] = std::vector<uint8_t>(bytes, bytes + data.size() * sizeof(uint32_t));
}

bool AftermathGpuCrashTracker::findShaderBinary(
    const GFSDK_Aftermath_ShaderBinaryHash& shaderHash,
    std::vector<uint8_t>& shader) const {
    auto it = m_shaderBinaries.find(shaderHash);
    if (it == m_shaderBinaries.end()) return false;
    shader = it->second;
    return true;
}

bool AftermathGpuCrashTracker::findShaderBinaryWithDebugData(
    const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
    std::vector<uint8_t>& shader) const {
    auto it = m_shaderBinariesWithDebugInfo.find(shaderDebugName);
    if (it == m_shaderBinariesWithDebugInfo.end()) return false;
    shader = it->second;
    return true;
}

// Static callback wrappers
void AftermathGpuCrashTracker::gpuCrashDumpCallback(
    const void* pGpuCrashDump, uint32_t gpuCrashDumpSize, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onCrashDump(pGpuCrashDump, gpuCrashDumpSize);
}

void AftermathGpuCrashTracker::shaderDebugInfoCallback(
    const void* pShaderDebugInfo, uint32_t shaderDebugInfoSize, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onShaderDebugInfo(pShaderDebugInfo, shaderDebugInfoSize);
}

void AftermathGpuCrashTracker::crashDumpDescriptionCallback(
    PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onDescription(addDescription);
}

void AftermathGpuCrashTracker::resolveMarkerCallback(
    const void* pMarkerData, uint32_t markerDataSize, void* pUserData,
    PFN_GFSDK_Aftermath_ResolveMarker resolveMarker) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onResolveMarker(pMarkerData, markerDataSize, resolveMarker);
}

void AftermathGpuCrashTracker::shaderDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
    PFN_GFSDK_Aftermath_SetData setShaderDebugInfo, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onShaderDebugInfoLookup(*pIdentifier, setShaderDebugInfo);
}

void AftermathGpuCrashTracker::shaderLookupCallback(
    const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
    PFN_GFSDK_Aftermath_SetData setShaderBinary, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onShaderLookup(*pShaderHash, setShaderBinary);
}

void AftermathGpuCrashTracker::shaderSourceDebugInfoLookupCallback(
    const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
    PFN_GFSDK_Aftermath_SetData setShaderBinary, void* pUserData) {
    static_cast<AftermathGpuCrashTracker*>(pUserData)->onShaderSourceDebugInfoLookup(*pShaderDebugName, setShaderBinary);
}

#endif // METALLIC_HAS_AFTERMATH
