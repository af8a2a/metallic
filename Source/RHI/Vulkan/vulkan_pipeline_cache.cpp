#include "vulkan_pipeline_cache.h"

#ifdef _WIN32

#include <vulkan/vulkan.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <sstream>
#include <vector>

// --- Helpers ---

namespace {

// Convert the raw 16-byte pipelineCacheUUID to a hex string used as filename.
std::string uuidToHex(const uint8_t uuid[VK_UUID_SIZE]) {
    std::ostringstream oss;
    for (int i = 0; i < VK_UUID_SIZE; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned>(uuid[i]);
    }
    return oss.str();
}

} // namespace

// --- VulkanPipelineCacheManager ---

bool VulkanPipelineCacheManager::load(VkDevice device,
                                      VkPhysicalDevice physicalDevice,
                                      const std::string& cacheDir) {
    m_device = device;

    // Build a device-unique filename: <vendor>_<device>_<uuid>.bin
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);

    std::ostringstream name;
    name << std::hex << std::setfill('0')
         << std::setw(4) << props.vendorID << "_"
         << std::setw(4) << props.deviceID << "_"
         << uuidToHex(props.pipelineCacheUUID)
         << ".bin";

    const std::filesystem::path dir(cacheDir);
    const std::filesystem::path filePath = dir / name.str();
    m_cachePath = filePath.string();

    // Ensure the cache directory exists.
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        spdlog::warn("VulkanPipelineCache: could not create cache dir '{}': {}",
                     cacheDir, ec.message());
    }

    // Attempt to load existing cache data from disk.
    std::vector<uint8_t> cacheData;
    std::ifstream cacheFile(m_cachePath, std::ios::binary | std::ios::ate);
    if (cacheFile.is_open()) {
        const std::streamsize fileSize = cacheFile.tellg();
        if (fileSize > 0) {
            cacheFile.seekg(0, std::ios::beg);
            cacheData.resize(static_cast<size_t>(fileSize));
            if (!cacheFile.read(reinterpret_cast<char*>(cacheData.data()), fileSize)) {
                spdlog::warn("VulkanPipelineCache: could not read cache file '{}', starting fresh",
                             m_cachePath);
                cacheData.clear();
            } else {
                spdlog::info("VulkanPipelineCache: loaded {} bytes from '{}'",
                             cacheData.size(), m_cachePath);
            }
        }
    } else {
        spdlog::info("VulkanPipelineCache: no existing cache at '{}', starting fresh",
                     m_cachePath);
    }

    VkPipelineCacheCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cacheData.empty()) {
        createInfo.initialDataSize = cacheData.size();
        createInfo.pInitialData    = cacheData.data();
    }

    const VkResult result = vkCreatePipelineCache(m_device, &createInfo, nullptr, &m_cache);
    if (result != VK_SUCCESS) {
        spdlog::error("VulkanPipelineCache: vkCreatePipelineCache failed ({}), "
                      "pipelines will compile without cache", static_cast<int>(result));
        // Still set m_cache = VK_NULL_HANDLE — callers must handle this gracefully.
        m_cache = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

bool VulkanPipelineCacheManager::save() {
    if (m_cache == VK_NULL_HANDLE || m_device == VK_NULL_HANDLE) {
        return false;
    }

    // Retrieve serialised data size.
    size_t dataSize = 0;
    VkResult result = vkGetPipelineCacheData(m_device, m_cache, &dataSize, nullptr);
    if (result != VK_SUCCESS || dataSize == 0) {
        return false;
    }

    std::vector<uint8_t> data(dataSize);
    result = vkGetPipelineCacheData(m_device, m_cache, &dataSize, data.data());
    if (result != VK_SUCCESS) {
        spdlog::warn("VulkanPipelineCache: vkGetPipelineCacheData failed, cache not saved");
        return false;
    }

    // Write to a temp file first, then rename (atomic-ish update).
    const std::string tempPath = m_cachePath + ".tmp";
    std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        spdlog::warn("VulkanPipelineCache: could not open '{}' for writing", tempPath);
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    out.close();

    std::error_code ec;
    std::filesystem::rename(tempPath, m_cachePath, ec);
    if (ec) {
        // Fallback: copy then remove
        std::filesystem::copy_file(tempPath, m_cachePath,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tempPath, ec);
    }

    spdlog::info("VulkanPipelineCache: saved {} bytes to '{}' "
                 "(graphics: {}, compute: {}, total: {:.1f} ms)",
                 data.size(), m_cachePath,
                 m_graphicsCount, m_computeCount, m_totalCompileMs);
    return true;
}

void VulkanPipelineCacheManager::destroy() {
    if (m_cache != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(m_device, m_cache, nullptr);
        m_cache = VK_NULL_HANDLE;
    }
}

void VulkanPipelineCacheManager::recordCompile(double ms, bool isGraphics) {
    if (isGraphics) {
        ++m_graphicsCount;
    } else {
        ++m_computeCount;
    }
    m_totalCompileMs += ms;
}

void VulkanPipelineCacheManager::resetStats() {
    m_graphicsCount  = 0;
    m_computeCount   = 0;
    m_totalCompileMs = 0.0;
}

#endif // _WIN32
