#pragma once

#include <volk.h>
#include <array>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>

namespace metallic::render::vulkan {

struct ShaderPrintfOptions {
    uint32_t bufferBytes = 65536;
    // Fault injection for capability probes. Normal callers retain both defaults.
    bool subscribeInfo = true;
    bool toStdout = false;
};

struct ShaderPrintfMessage {
    uint32_t severity = 0;
    int32_t id = 0;
    bool truncated = false;
    std::array<char, 160> idName{};
    std::array<char, 4096> text{};
};

// One session per device. The caller owns it until after device destruction.
// No allocation, JSON formatting or Vulkan calls occur in capture().
class ShaderPrintf {
public:
    explicit ShaderPrintf(ShaderPrintfOptions options = {});
    ShaderPrintf(const ShaderPrintf&) = delete;
    ShaderPrintf& operator=(const ShaderPrintf&) = delete;
    const ShaderPrintfOptions& options() const { return options_; }
    const VkLayerSettingsCreateInfoEXT* settings() const { return &settingsInfo_; }
    bool valid() const { return options_.bufferBytes >= 128 && options_.bufferBytes <= 1048576; }
    void capture(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        const VkDebugUtilsMessengerCallbackDataEXT& data) noexcept;
    std::vector<ShaderPrintfMessage> snapshot() const;
    // Owner drain; lifetime counters remain monotonic. New callbacks never reuse borrowed storage.
    std::vector<ShaderPrintfMessage> drain();
    uint64_t dropped() const { return dropped_.load(); }
    uint64_t truncated() const { return truncated_.load(); }

    // Setup facts only; smoke verification belongs to the workload that checks echo/readback.
    bool layerDiscovered = false;
    uint32_t layerSpecVersion = 0;
    uint32_t layerImplementationVersion = 0;
    bool instanceConfigured = false;
    bool messengerConfigured = false;
    bool deviceConfigured = false;

private:
    ShaderPrintfOptions options_;
    VkBool32 enabled_ = VK_TRUE;
    VkBool32 disabled_ = VK_FALSE;
    VkBool32 stdout_ = VK_FALSE;
    uint32_t bufferBytes_ = 0;
    uint32_t duplicateLimit_ = 0;
    const char* reportFlags_[3] = {"info", "warn", "error"};
    const char* emptyFilter_ = "";
    std::array<VkLayerSettingEXT, 9> settings_{};
    VkLayerSettingsCreateInfoEXT settingsInfo_{};
    mutable std::mutex mutex_;
    std::unique_ptr<std::array<ShaderPrintfMessage, 256>> messages_ =
        std::make_unique<std::array<ShaderPrintfMessage, 256>>();
    size_t count_ = 0;
    std::atomic_uint64_t dropped_{0};
    std::atomic_uint64_t truncated_{0};
};

} // namespace metallic::render::vulkan
