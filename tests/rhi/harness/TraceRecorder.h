#pragma once
#include "Evidence.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanTrace.h"
#include <atomic>
#include <map>
#include <mutex>

namespace metallic::tests::bench {
class TraceRecorder {
public:
    explicit TraceRecorder(size_t limit = 8192) : limit_(limit > 8192 ? 8192 : limit) {}
    ~TraceRecorder() { stop(); }
    bool start(render::Device& device);
    void stop();
    bool failed() const { return failed_.load(); }
    Json snapshot() const;
    uint64_t queueId(render::Queue& queue);
    uint64_t bufferId(render::Buffer& buffer);
    uint64_t textureId(render::Texture& texture);
    // Public for synthetic protocol tests; never submits native commands.
    void capture(const render::vulkan::TraceEvent& event) noexcept;
private:
    uint64_t identity(VkObjectType type, uint64_t handle);
    mutable std::mutex mutex_;
    std::map<std::pair<VkObjectType, uint64_t>, uint64_t> identities_;
    std::map<uint64_t, uint64_t> recordings_;
    uint64_t nextId_ = 1, nextRecording_ = 1;
    size_t limit_, bytes_ = 0;
    Json events_ = Json::array();
    std::atomic_bool failed_{false};
    render::vulkan::TraceSink sink_;
    bool started_ = false;
};
} // namespace metallic::tests::bench
