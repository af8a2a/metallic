#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <volk.h>

namespace metallic::render {
class Device;
class Queue;
class CommandBuffer;
}

namespace metallic::render::profiling {

bool nvPerfRequested();
bool nvPerfInstanceExtensions(std::vector<const char*>& extensions, uint32_t apiVersion, std::string& error);
bool nvPerfDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
    std::vector<const char*>& extensions, std::string& error);
bool nvPerfPassActive();

// One process-owned session, one pass, one primed production target frame.
// No device clock changes. Destroy before Device; caller drains before boundaries.
class NvPerfSession final {
public:
    NvPerfSession();
    ~NvPerfSession();
    bool begin(Device& device, Queue& queue, const std::filesystem::path& output, std::string& error);
    bool finish(std::string& error);
    void cancel();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Explicit production dispatch annotation, independent of debug-label export.
class NvPerfRange final {
public:
    NvPerfRange(CommandBuffer& commands, const char* name);
    ~NvPerfRange();
    NvPerfRange(const NvPerfRange&) = delete;
    NvPerfRange& operator=(const NvPerfRange&) = delete;
private:
    VkCommandBuffer commands_ = VK_NULL_HANDLE;
};
} // namespace metallic::render::profiling
