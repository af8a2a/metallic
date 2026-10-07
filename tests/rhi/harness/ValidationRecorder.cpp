#include "ValidationRecorder.h"
#include <chrono>
#include <sstream>
#include <thread>

namespace metallic::tests::bench {

void ValidationRecorder::capture(void* context, const render::ValidationMessage& message) noexcept
{
    auto& self = *static_cast<ValidationRecorder*>(context);
    ++self.messageCount;
    try {
        std::lock_guard lock(self.mutex_);
        if (self.messages_.size() >= self.limit_) { self.lost_ = true; return; }
        const std::string id = message.messageIdName ? message.messageIdName : "";
        const bool fatal = (message.severity == render::ValidationSeverity::Error || message.severity == render::ValidationSeverity::Warning) || id.find("SYNC-HAZARD") != std::string::npos;
        self.error_ |= fatal;
        Json objects = Json::array();
        for (const auto& object : message.objects) {
            objects.push_back({{"handle", object.handle}, {"type", object.type}, {"name", object.name ? object.name : ""}});
        }
        std::ostringstream thread;
        thread << std::this_thread::get_id();
        self.messages_.push_back({{"phase", self.phase_}, {"severity", message.severity}, {"type", message.type},
            {"id", id}, {"number", message.messageId}, {"text", message.message ? message.message : ""},
            {"thread", thread.str()}, {"monotonicNs", std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count()}, {"objects", objects}});
    } catch (...) { self.lost_ = true; }
}

void ValidationRecorder::phase(std::string phase)
{
    std::lock_guard lock(mutex_);
    phase_ = std::move(phase);
}

Json ValidationRecorder::snapshot() const
{
    std::lock_guard lock(mutex_);
    return {{"encoding", "metallic-validation-v1"}, {"messages", messages_}, {"captureFailed", lost_.load()}, {"count", messageCount.load()}};
}

bool ValidationRecorder::failed() const
{
    std::lock_guard lock(mutex_);
    return error_ || lost_.load();
}

} // namespace metallic::tests::bench
