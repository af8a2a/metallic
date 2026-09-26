#include "VulkanShaderPrintf.h"

namespace metallic::render::vulkan {
namespace {

template<size_t Size>
bool copyBounded(std::array<char, Size>& output, const char* input) noexcept
{
    if (input == nullptr) { return false; }
    size_t i = 0;
    for (; i + 1 < Size && input[i] != '\0'; ++i) { output[i] = input[i]; }
    output[i] = '\0';
    return input[i] != '\0';
}

} // namespace

ShaderPrintf::ShaderPrintf(ShaderPrintfOptions options) : options_(options),
    stdout_(options.toStdout ? VK_TRUE : VK_FALSE), bufferBytes_(options.bufferBytes)
{
    constexpr const char* layer = "VK_LAYER_KHRONOS_validation";
    settings_ = {{
        {layer, "printf_only_preset", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &disabled_},
        {layer, "printf_enable", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &enabled_},
        {layer, "printf_to_stdout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &stdout_},
        {layer, "printf_verbose", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &disabled_},
        {layer, "printf_buffer_size", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &bufferBytes_},
        {layer, "report_flags", VK_LAYER_SETTING_TYPE_STRING_EXT, 3, reportFlags_},
        {layer, "enable_message_limit", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &disabled_},
        {layer, "duplicate_message_limit", VK_LAYER_SETTING_TYPE_UINT32_EXT, 1, &duplicateLimit_},
        {layer, "message_id_filter", VK_LAYER_SETTING_TYPE_STRING_EXT, 1, &emptyFilter_},
    }};
    settingsInfo_ = {.sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        .settingCount = static_cast<uint32_t>(settings_.size()), .pSettings = settings_.data()};
}

void ShaderPrintf::capture(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    const VkDebugUtilsMessengerCallbackDataEXT& data) noexcept
{
    try {
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || count_ == messages_->size()) { ++dropped_; return; }
        auto& record = (*messages_)[count_++];
        record = {};
        record.severity = severity;
        record.id = data.messageIdNumber;
        record.truncated = copyBounded(record.idName, data.pMessageIdName);
        record.truncated |= copyBounded(record.text, data.pMessage);
        if (record.truncated) { ++truncated_; }
    } catch (...) {
        ++dropped_;
    }
}

std::vector<ShaderPrintfMessage> ShaderPrintf::snapshot() const
{
    std::lock_guard lock(mutex_);
    return {messages_->begin(), messages_->begin() + count_};
}

std::vector<ShaderPrintfMessage> ShaderPrintf::drain()
{
    std::lock_guard lock(mutex_);
    std::vector<ShaderPrintfMessage> result{messages_->begin(), messages_->begin() + count_};
    count_ = 0;
    return result;
}

} // namespace metallic::render::vulkan
