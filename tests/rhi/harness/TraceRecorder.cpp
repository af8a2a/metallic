#include "TraceRecorder.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

namespace metallic::tests::bench {
namespace rv = render::vulkan;
namespace {
Json scope(VkPipelineStageFlags2 stage, VkAccessFlags2 access) { return {{"stages", stage}, {"access", access}}; }
Json scopes(const auto& barrier)
{
    return {{"before", scope(barrier.srcStageMask, barrier.srcAccessMask)},
        {"after", scope(barrier.dstStageMask, barrier.dstAccessMask)}};
}
Json declared(render::SyncScope value) { return {{"stages", uint64_t(value.stages)}, {"access", uint64_t(value.access)}}; }
}

bool TraceRecorder::start(render::Device& device)
{
    if (started_) { return false; }
    sink_ = {rv::nativeDevice(device).device, [](void* context, const rv::TraceEvent& event) noexcept {
        static_cast<TraceRecorder*>(context)->capture(event);
    }, this};
    started_ = rv::installTraceSink(&sink_);
    return started_;
}
void TraceRecorder::stop()
{
    if (started_) { rv::removeTraceSink(&sink_); started_ = false; }
}
uint64_t TraceRecorder::identity(VkObjectType type, uint64_t handle)
{
    if (!handle) { return 0; }
    auto [it, inserted] = identities_.try_emplace({type, handle}, nextId_);
    if (inserted) { ++nextId_; }
    return it->second;
}
uint64_t TraceRecorder::queueId(render::Queue& queue)
{
    std::lock_guard lock(mutex_);
    return identity(VK_OBJECT_TYPE_QUEUE, uint64_t(rv::nativeQueue(queue).queue));
}
uint64_t TraceRecorder::bufferId(render::Buffer& buffer)
{
    std::lock_guard lock(mutex_);
    return identity(VK_OBJECT_TYPE_BUFFER, uint64_t(rv::nativeBuffer(buffer).buffer));
}
uint64_t TraceRecorder::textureId(render::Texture& texture)
{
    std::lock_guard lock(mutex_);
    return identity(VK_OBJECT_TYPE_IMAGE, uint64_t(rv::nativeTexture(texture).image));
}
void TraceRecorder::capture(const rv::TraceEvent& event) noexcept
{
    if (failed()) { return; }
    try {
        std::lock_guard lock(mutex_);
        if (events_.size() >= limit_) { failed_ = true; return; }
        Json value{{"sequence", events_.size()}};
        if (event.kind == rv::TraceKind::Retire) {
            const auto key = std::pair{event.objectType, event.object};
            const auto it = identities_.find(key);
            if (it == identities_.end()) { return; }
            value["kind"] = "retire"; value["object"] = it->second;
            if (event.objectType == VK_OBJECT_TYPE_COMMAND_BUFFER) { recordings_.erase(it->second); }
            identities_.erase(it);
        } else if (event.kind == rv::TraceKind::Submit) {
            const auto& submit = *event.submit;
            if (uint64_t(submit.waitSemaphoreInfoCount) + submit.signalSemaphoreInfoCount + submit.commandBufferInfoCount > 1024) {
                failed_ = true; return;
            }
            value["kind"] = "submit"; value["queue"] = identity(VK_OBJECT_TYPE_QUEUE, uint64_t(event.queue));
            value["queueFamily"] = event.queueFamily; value["result"] = int(event.result);
            value["commands"] = Json::array();
            for (uint32_t i = 0; i < submit.commandBufferInfoCount; ++i) {
                const auto id = identity(VK_OBJECT_TYPE_COMMAND_BUFFER, uint64_t(submit.pCommandBufferInfos[i].commandBuffer));
                value["commands"].push_back({{"id", id}, {"recording", recordings_[id]}});
            }
            const auto semaphores = [&](const VkSemaphoreSubmitInfo* data, uint32_t count) {
                Json array = Json::array();
                for (uint32_t i = 0; i < count; ++i) {
                    array.push_back({{"id", identity(VK_OBJECT_TYPE_SEMAPHORE, uint64_t(data[i].semaphore))},
                        {"value", data[i].value}, {"stages", data[i].stageMask}});
                }
                return array;
            };
            value["waits"] = semaphores(submit.pWaitSemaphoreInfos, submit.waitSemaphoreInfoCount);
            value["signals"] = semaphores(submit.pSignalSemaphoreInfos, submit.signalSemaphoreInfoCount);
        } else {
            const auto id = identity(VK_OBJECT_TYPE_COMMAND_BUFFER, uint64_t(event.command));
            value["command"] = id;
            if (event.kind == rv::TraceKind::CommandBegin) {
                value["kind"] = "begin"; recordings_[id] = nextRecording_++;
            } else {
                value["kind"] = "barrier";
                const auto& dependency = *event.dependency;
                if (uint64_t(dependency.memoryBarrierCount) + dependency.bufferMemoryBarrierCount + dependency.imageMemoryBarrierCount > 1024 ||
                    (event.requested && event.requested->textures.size() + event.requested->buffers.size() + event.requested->memory.size() > 1024)) {
                    failed_ = true; return;
                }
                value["flags"] = dependency.dependencyFlags;
                value["memory"] = Json::array(); value["buffers"] = Json::array(); value["images"] = Json::array();
                for (uint32_t i = 0; i < dependency.memoryBarrierCount; ++i) { value["memory"].push_back(scopes(dependency.pMemoryBarriers[i])); }
                for (uint32_t i = 0; i < dependency.bufferMemoryBarrierCount; ++i) {
                    const auto& b = dependency.pBufferMemoryBarriers[i]; auto entry = scopes(b);
                    entry["resource"] = identity(VK_OBJECT_TYPE_BUFFER, uint64_t(b.buffer));
                    entry["offset"] = b.offset; entry["size"] = b.size;
                    entry["sourceFamily"] = b.srcQueueFamilyIndex; entry["destinationFamily"] = b.dstQueueFamilyIndex;
                    value["buffers"].push_back(entry);
                }
                for (uint32_t i = 0; i < dependency.imageMemoryBarrierCount; ++i) {
                    const auto& b = dependency.pImageMemoryBarriers[i]; auto entry = scopes(b);
                    entry["resource"] = identity(VK_OBJECT_TYPE_IMAGE, uint64_t(b.image));
                    entry["oldLayout"] = int(b.oldLayout); entry["newLayout"] = int(b.newLayout);
                    entry["sourceFamily"] = b.srcQueueFamilyIndex; entry["destinationFamily"] = b.dstQueueFamilyIndex;
                    entry["range"] = {b.subresourceRange.aspectMask, b.subresourceRange.baseMipLevel, b.subresourceRange.levelCount,
                        b.subresourceRange.baseArrayLayer, b.subresourceRange.layerCount};
                    value["images"].push_back(entry);
                }
                value["requested"] = Json::array();
                if (event.requested) {
                    for (const auto& b : event.requested->memory) {
                        value["requested"].push_back({{"kind", "memory"}, {"before", declared(b.before)}, {"after", declared(b.after)}});
                    }
                    for (const auto& b : event.requested->buffers) {
                        const auto range = b.range.resolve(b.buffer->desc().size).value();
                        value["requested"].push_back({{"kind", "buffer"},
                            {"resource", identity(VK_OBJECT_TYPE_BUFFER, uint64_t(rv::nativeBuffer(*b.buffer).buffer))},
                            {"offset", range.offset}, {"size", range.size}, {"before", declared(b.before)}, {"after", declared(b.after)}});
                    }
                    for (const auto& b : event.requested->textures) {
                        value["requested"].push_back({{"kind", "image"},
                            {"resource", identity(VK_OBJECT_TYPE_IMAGE, uint64_t(rv::nativeTexture(*b.texture).image))},
                            {"range", {b.range.baseMip, b.range.mipCount, b.range.baseLayer, b.range.layerCount}},
                            {"oldLayout", int(b.oldLayout)}, {"newLayout", int(b.newLayout)},
                            {"before", declared(b.before)}, {"after", declared(b.after)}});
                    }
                }
            }
            value["recording"] = recordings_[id];
        }
        bytes_ += value.dump().size();
        if (bytes_ > 8 * 1024 * 1024) { failed_ = true; return; }
        events_.push_back(std::move(value));
    } catch (...) { failed_ = true; }
}
Json TraceRecorder::snapshot() const
{
    std::lock_guard lock(mutex_);
    return {{"schema", 1}, {"scope", "RHI Vulkan barrier/submit only; native SDK calls are outside capture"},
        {"compiled", rv::traceCompiled()}, {"captureFailed", failed()}, {"eventLimit", limit_}, {"events", events_}};
}
} // namespace metallic::tests::bench
