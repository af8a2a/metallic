#include "Runtime/Render/Profiling/NsightAftermath.h"

#include <spdlog/spdlog.h>
#include <volk.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef METALLIC_HAS_NSIGHT_AFTERMATH
#define METALLIC_HAS_NSIGHT_AFTERMATH 0
#endif

#ifndef METALLIC_NSIGHT_AFTERMATH_DUMP_DIR
#define METALLIC_NSIGHT_AFTERMATH_DUMP_DIR PROJECT_SOURCE_DIR "/.cache/aftermath"
#endif

#if METALLIC_HAS_NSIGHT_AFTERMATH
#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

#include <array>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#endif

namespace metallic::render::profiling {
namespace {

#if METALLIC_HAS_NSIGHT_AFTERMATH

std::string hexString(uint64_t value, uint32_t width)
{
    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(static_cast<int>(width)) << std::hex << value;
    return stream.str();
}

std::string identifierString(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier)
{
    return hexString(identifier.id[0], 16) + "-" + hexString(identifier.id[1], 16);
}

std::string aftermathResultMessage(GFSDK_Aftermath_Result result)
{
    switch (result) {
    case GFSDK_Aftermath_Result_FAIL_DriverVersionNotSupported:
        return "unsupported NVIDIA driver version for Nsight Aftermath";
    default:
        return "Nsight Aftermath error 0x" + hexString(static_cast<uint32_t>(result), 8);
    }
}

bool checkAftermath(GFSDK_Aftermath_Result result, const char* label)
{
    if (GFSDK_Aftermath_SUCCEED(result)) {
        return true;
    }

    spdlog::error("{} failed: {}", label, aftermathResultMessage(result));
    return false;
}

struct ShaderDebugInfoIdentifierLess {
    bool operator()(
        const GFSDK_Aftermath_ShaderDebugInfoIdentifier& lhs,
        const GFSDK_Aftermath_ShaderDebugInfoIdentifier& rhs) const
    {
        if (lhs.id[0] == rhs.id[0]) {
            return lhs.id[1] < rhs.id[1];
        }
        return lhs.id[0] < rhs.id[0];
    }
};

struct ShaderBinaryHashLess {
    bool operator()(const GFSDK_Aftermath_ShaderBinaryHash& lhs, const GFSDK_Aftermath_ShaderBinaryHash& rhs) const
    {
        return lhs.hash < rhs.hash;
    }
};

std::string sanitizeFileStem(std::string text)
{
    if (text.empty()) {
        return "Metallic";
    }

    for (char& ch : text) {
        const bool alphaNumeric =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        if (!alphaNumeric && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return text;
}

class AftermathCrashTracker {
public:
    ~AftermathCrashTracker()
    {
        if (initialized_) {
            GFSDK_Aftermath_DisableGpuCrashDumps();
        }
    }

    bool initialize(const char* applicationName)
    {
        std::lock_guard lock(mutex_);
        if (initialized_) {
            return true;
        }

        applicationName_ =
            applicationName != nullptr && applicationName[0] != '\0' ? applicationName : "Metallic";
        outputDirectory_ = std::filesystem::path(METALLIC_NSIGHT_AFTERMATH_DUMP_DIR);

        const GFSDK_Aftermath_Result result = GFSDK_Aftermath_EnableGpuCrashDumps(
            GFSDK_Aftermath_Version_API,
            GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
            GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
            gpuCrashDumpCallback,
            shaderDebugInfoCallback,
            crashDumpDescriptionCallback,
            nullptr,
            this);
        if (!checkAftermath(result, "GFSDK_Aftermath_EnableGpuCrashDumps")) {
            return false;
        }

        initialized_ = true;
        deviceLostHandled_ = false;
        return true;
    }

    bool initialized() const
    {
        std::lock_guard lock(mutex_);
        return initialized_;
    }

    void addShaderBinary(const uint32_t* code, uint64_t byteSize)
    {
        if (code == nullptr || byteSize == 0 || (byteSize % sizeof(uint32_t)) != 0 || byteSize > UINT32_MAX) {
            return;
        }

        std::lock_guard lock(mutex_);
        if (!initialized_) {
            return;
        }

        const GFSDK_Aftermath_SpirvCode shaderCode{
            code,
            static_cast<uint32_t>(byteSize),
        };
        GFSDK_Aftermath_ShaderBinaryHash shaderHash{};
        if (!checkAftermath(
                GFSDK_Aftermath_GetShaderHashSpirv(GFSDK_Aftermath_Version_API, &shaderCode, &shaderHash),
                "GFSDK_Aftermath_GetShaderHashSpirv")) {
            return;
        }

        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(code);
        shaderBinaries_[shaderHash] = std::vector<uint8_t>(bytes, bytes + byteSize);
    }

    void waitForCrashDump()
    {
        {
            std::lock_guard lock(mutex_);
            if (!initialized_ || deviceLostHandled_) {
                return;
            }
            deviceLostHandled_ = true;
        }

        spdlog::warn("VK_ERROR_DEVICE_LOST detected; waiting for Nsight Aftermath crash dump capture.");

        GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
        if (!checkAftermath(GFSDK_Aftermath_GetCrashDumpStatus(&status), "GFSDK_Aftermath_GetCrashDumpStatus")) {
            return;
        }

        const auto timeout = std::chrono::seconds(5);
        const auto start = std::chrono::steady_clock::now();
        while (status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed &&
            status != GFSDK_Aftermath_CrashDump_Status_Finished &&
            std::chrono::steady_clock::now() - start < timeout) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!checkAftermath(GFSDK_Aftermath_GetCrashDumpStatus(&status), "GFSDK_Aftermath_GetCrashDumpStatus")) {
                return;
            }
        }

        if (status == GFSDK_Aftermath_CrashDump_Status_Finished) {
            spdlog::info("Nsight Aftermath dump written under: {}", outputDirectory_.string());
        } else {
            spdlog::warn(
                "Nsight Aftermath crash dump capture did not finish; status {}",
                static_cast<int>(status));
        }
    }

private:
    void onCrashDump(const void* crashDump, uint32_t crashDumpSize)
    {
        std::lock_guard lock(mutex_);
        writeGpuCrashDumpToFile(crashDump, crashDumpSize);
    }

    void onShaderDebugInfo(const void* shaderDebugInfo, uint32_t shaderDebugInfoSize)
    {
        std::lock_guard lock(mutex_);

        GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier{};
        if (!checkAftermath(
                GFSDK_Aftermath_GetShaderDebugInfoIdentifier(
                    GFSDK_Aftermath_Version_API,
                    shaderDebugInfo,
                    shaderDebugInfoSize,
                    &identifier),
                "GFSDK_Aftermath_GetShaderDebugInfoIdentifier")) {
            return;
        }

        const uint8_t* bytes = static_cast<const uint8_t*>(shaderDebugInfo);
        shaderDebugInfo_[identifier] = std::vector<uint8_t>(bytes, bytes + shaderDebugInfoSize);
        writeBinaryFile(
            outputDirectory_ / ("shader-" + identifierString(identifier) + ".nvdbg"),
            shaderDebugInfo,
            shaderDebugInfoSize);
    }

    void onDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription) const
    {
        addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, applicationName_.c_str());
        addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "0.1.0");
    }

    void onShaderDebugInfoLookup(
        const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier,
        PFN_GFSDK_Aftermath_SetData setShaderDebugInfo) const
    {
        const auto it = shaderDebugInfo_.find(identifier);
        if (it != shaderDebugInfo_.end()) {
            setShaderDebugInfo(it->second.data(), static_cast<uint32_t>(it->second.size()));
        }
    }

    void onShaderLookup(
        const GFSDK_Aftermath_ShaderBinaryHash& shaderHash,
        PFN_GFSDK_Aftermath_SetData setShaderBinary) const
    {
        const auto it = shaderBinaries_.find(shaderHash);
        if (it != shaderBinaries_.end()) {
            setShaderBinary(it->second.data(), static_cast<uint32_t>(it->second.size()));
        }
    }

    static void gpuCrashDumpCallback(const void* crashDump, uint32_t crashDumpSize, void* userData)
    {
        static_cast<AftermathCrashTracker*>(userData)->onCrashDump(crashDump, crashDumpSize);
    }

    static void shaderDebugInfoCallback(const void* shaderDebugInfo, uint32_t shaderDebugInfoSize, void* userData)
    {
        static_cast<AftermathCrashTracker*>(userData)->onShaderDebugInfo(shaderDebugInfo, shaderDebugInfoSize);
    }

    static void crashDumpDescriptionCallback(
        PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription,
        void* userData)
    {
        static_cast<AftermathCrashTracker*>(userData)->onDescription(addDescription);
    }

    static void shaderDebugInfoLookupCallback(
        const GFSDK_Aftermath_ShaderDebugInfoIdentifier* identifier,
        PFN_GFSDK_Aftermath_SetData setShaderDebugInfo,
        void* userData)
    {
        static_cast<AftermathCrashTracker*>(userData)->onShaderDebugInfoLookup(*identifier, setShaderDebugInfo);
    }

    static void shaderLookupCallback(
        const GFSDK_Aftermath_ShaderBinaryHash* shaderHash,
        PFN_GFSDK_Aftermath_SetData setShaderBinary,
        void* userData)
    {
        static_cast<AftermathCrashTracker*>(userData)->onShaderLookup(*shaderHash, setShaderBinary);
    }

    static void shaderSourceDebugInfoLookupCallback(
        const GFSDK_Aftermath_ShaderDebugName*,
        PFN_GFSDK_Aftermath_SetData,
        void*)
    {
    }

    void writeGpuCrashDumpToFile(const void* crashDump, uint32_t crashDumpSize)
    {
        GFSDK_Aftermath_GpuCrashDump_Decoder decoder{};
        const bool decoderCreated = checkAftermath(
            GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
                GFSDK_Aftermath_Version_API,
                crashDump,
                crashDumpSize,
                &decoder),
            "GFSDK_Aftermath_GpuCrashDump_CreateDecoder");

        GFSDK_Aftermath_GpuCrashDump_BaseInfo baseInfo{};
        if (decoderCreated) {
            checkAftermath(
                GFSDK_Aftermath_GpuCrashDump_GetBaseInfo(decoder, &baseInfo),
                "GFSDK_Aftermath_GpuCrashDump_GetBaseInfo");
        }

        const std::string baseFileName =
            sanitizeFileStem(applicationName_) + "-" +
            std::to_string(baseInfo.pid) + "-" +
            std::to_string(++dumpCount_);
        const std::filesystem::path dumpPath = outputDirectory_ / (baseFileName + ".nv-gpudmp");
        writeBinaryFile(dumpPath, crashDump, crashDumpSize);
        spdlog::info("Writing Nsight Aftermath GPU crash dump: {}", dumpPath.string());

        if (decoderCreated) {
            writeJsonDump(decoder, outputDirectory_ / (baseFileName + ".json"));
            checkAftermath(
                GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder),
                "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder");
        }
    }

    void writeJsonDump(GFSDK_Aftermath_GpuCrashDump_Decoder decoder, const std::filesystem::path& path)
    {
        uint32_t jsonSize = 0;
        if (!checkAftermath(
                GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
                    decoder,
                    GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
                    GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
                    shaderDebugInfoLookupCallback,
                    shaderLookupCallback,
                    shaderSourceDebugInfoLookupCallback,
                    this,
                    &jsonSize),
                "GFSDK_Aftermath_GpuCrashDump_GenerateJSON") ||
            jsonSize == 0) {
            return;
        }

        std::vector<char> json(jsonSize);
        if (!checkAftermath(
                GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, jsonSize, json.data()),
                "GFSDK_Aftermath_GpuCrashDump_GetJSON")) {
            return;
        }

        writeBinaryFile(path, json.data(), json.size() > 0 ? json.size() - 1 : 0);
        spdlog::info("Writing Nsight Aftermath JSON dump: {}", path.string());
    }

    void writeBinaryFile(const std::filesystem::path& path, const void* data, size_t size) const
    {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            spdlog::error("Failed to open Nsight Aftermath output file: {}", path.string());
            return;
        }
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool deviceLostHandled_ = false;
    uint32_t dumpCount_ = 0;
    std::string applicationName_ = "Metallic";
    std::filesystem::path outputDirectory_;
    std::map<
        GFSDK_Aftermath_ShaderDebugInfoIdentifier,
        std::vector<uint8_t>,
        ShaderDebugInfoIdentifierLess> shaderDebugInfo_;
    std::map<GFSDK_Aftermath_ShaderBinaryHash, std::vector<uint8_t>, ShaderBinaryHashLess> shaderBinaries_;
};

AftermathCrashTracker& aftermathCrashTracker()
{
    static AftermathCrashTracker tracker;
    return tracker;
}

#endif

} // namespace

bool nsightAftermathSdkAvailable()
{
#if METALLIC_HAS_NSIGHT_AFTERMATH
    return true;
#else
    return false;
#endif
}

bool nsightAftermathInitialized()
{
#if METALLIC_HAS_NSIGHT_AFTERMATH
    return aftermathCrashTracker().initialized();
#else
    return false;
#endif
}

void initializeNsightAftermath(const char* applicationName)
{
#if METALLIC_HAS_NSIGHT_AFTERMATH
    aftermathCrashTracker().initialize(applicationName);
#else
    (void)applicationName;
#endif
}

void registerNsightAftermathShaderBinary(const uint32_t* code, uint64_t byteSize)
{
#if METALLIC_HAS_NSIGHT_AFTERMATH
    aftermathCrashTracker().addShaderBinary(code, byteSize);
#else
    (void)code;
    (void)byteSize;
#endif
}

void handleNsightAftermathDeviceLost()
{
#if METALLIC_HAS_NSIGHT_AFTERMATH
    aftermathCrashTracker().waitForCrashDump();
#endif
}

} // namespace metallic::render::profiling
