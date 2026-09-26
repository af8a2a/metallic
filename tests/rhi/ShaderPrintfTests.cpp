#include "Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.h"
#include <gtest/gtest.h>
#include <string>
#include <thread>

namespace {
using metallic::render::vulkan::ShaderPrintf;

TEST(ShaderPrintf, OwnsBorrowedCallbackData)
{
    ShaderPrintf capture;
    std::string text = "original";
    VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
        .pMessageIdName = "VVL-DEBUG-PRINTF", .messageIdNumber = 0x4fe1fef9, .pMessage = text.c_str()};
    capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, data);
    text.assign("modified");
    const auto messages = capture.snapshot();
    ASSERT_EQ(messages.size(), 1);
    EXPECT_STREQ(messages[0].text.data(), "original");
    EXPECT_EQ(messages[0].id, 0x4fe1fef9);
    EXPECT_EQ(capture.dropped(), 0);
}

TEST(ShaderPrintf, BoundedQueuePreservesFirstRecordsAndCountsLoss)
{
    ShaderPrintf capture;
    VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT, .pMessage = "record"};
    for (int i = 0; i < 260; ++i) {
        data.messageIdNumber = i;
        capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, data);
    }
    const auto messages = capture.snapshot();
    ASSERT_EQ(messages.size(), 256);
    EXPECT_EQ(messages.front().id, 0);
    EXPECT_EQ(messages.back().id, 255);
    EXPECT_EQ(capture.dropped(), 4);
}

TEST(ShaderPrintf, ExactCapacityAndTruncationAreDistinct)
{
    ShaderPrintf capture;
    std::string text(4095, 'x');
    VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT, .pMessage = text.c_str()};
    capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, data);
    text += 'x';
    data.pMessage = text.c_str();
    capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, data);
    const auto messages = capture.snapshot();
    ASSERT_EQ(messages.size(), 2);
    EXPECT_FALSE(messages[0].truncated);
    EXPECT_TRUE(messages[1].truncated);
    EXPECT_EQ(messages[1].text.back(), '\0');
    EXPECT_EQ(capture.truncated(), 1);
}

TEST(ShaderPrintf, ConcurrentCallbacksAccountForEveryRecord)
{
    ShaderPrintf capture;
    const auto produce = [&] {
        VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT, .pMessage = "concurrent"};
        for (int i = 0; i < 200; ++i) { capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT, data); }
    };
    std::thread first(produce), second(produce);
    first.join();
    second.join();
    EXPECT_EQ(capture.snapshot().size() + capture.dropped(), 400);
    EXPECT_EQ(capture.truncated(), 0);
}
TEST(ShaderPrintf, DrainReusesSlotsWithoutBorrowedOrStaleData)
{
    ShaderPrintf capture;
    VkDebugUtilsMessengerCallbackDataEXT data{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
        .pMessageIdName = "original-id", .pMessage = "original"};
    capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,data);
    const auto drained = capture.drain();
    ASSERT_EQ(drained.size(),1u); EXPECT_TRUE(capture.snapshot().empty());
    data.pMessage = nullptr; data.pMessageIdName = nullptr;
    capture.capture(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,data);
    const auto reused = capture.drain(); ASSERT_EQ(reused.size(),1u);
    EXPECT_STREQ(reused[0].text.data(),""); EXPECT_STREQ(reused[0].idName.data(),"");
    EXPECT_STREQ(drained[0].text.data(),"original"); EXPECT_EQ(capture.dropped(),0u);
}
} // namespace
