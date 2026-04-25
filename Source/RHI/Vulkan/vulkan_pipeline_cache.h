#pragma once

#ifdef _WIN32

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

// Manages a VkPipelineCache whose binary data is persisted to disk between runs.
//
// The cache file is keyed by the physical device's pipelineCacheUUID so that
// incompatible data from a different device or driver is automatically ignored.
//
// Typical lifecycle:
//   1. After vkCreateDevice:  cache.load(device, physicalDevice, cacheDir)
//   2. Pass cache.handle() to every vkCreateGraphicsPipelines /
//      vkCreateComputePipelines call.
//   3. Before vkDestroyDevice:  cache.save();  cache.destroy();
class VulkanPipelineCacheManager {
public:
    VulkanPipelineCacheManager() = default;
    ~VulkanPipelineCacheManager() = default;

    // Non-copyable
    VulkanPipelineCacheManager(const VulkanPipelineCacheManager&) = delete;
    VulkanPipelineCacheManager& operator=(const VulkanPipelineCacheManager&) = delete;

    // Load (or create) the pipeline cache.
    // cacheDir: directory where .bin files are stored; created if it doesn't exist.
    // Returns true even when no on-disk data exists (cache is created empty).
    bool load(VkDevice device,
              VkPhysicalDevice physicalDevice,
              const std::string& cacheDir);

    // Write the current in-memory cache data back to disk.
    bool save();

    // Destroy the VkPipelineCache (must be called before vkDestroyDevice).
    void destroy();

    VkPipelineCache handle() const { return m_cache; }
    bool isValid()          const { return m_cache != VK_NULL_HANDLE; }

    // --- Compile telemetry ---

    // Called by pipeline-creation wrappers after each successful compile.
    // ms: wall-clock milliseconds for the compile call.
    void recordCompile(double ms, bool isGraphics);

    uint32_t graphicsPipelinesCompiled() const { return m_graphicsCount; }
    uint32_t computePipelinesCompiled()  const { return m_computeCount; }
    double   totalCompileMs()            const { return m_totalCompileMs; }

    // Reset counters (e.g. between reloads).
    void resetStats();

private:
    VkDevice        m_device          = VK_NULL_HANDLE;
    VkPipelineCache m_cache           = VK_NULL_HANDLE;
    std::string     m_cachePath;       // full path to the .bin file

    uint32_t m_graphicsCount  = 0;
    uint32_t m_computeCount   = 0;
    double   m_totalCompileMs = 0.0;
};

#endif // _WIN32
