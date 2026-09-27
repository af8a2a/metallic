#pragma once

#include "Evidence.h"
#include "Runtime/Render/GAPI/Rhi.h"
#include <atomic>
#include <mutex>

namespace metallic::tests::bench {

class ValidationRecorder {
public:
    explicit ValidationRecorder(size_t limit = 4096) : limit_(limit) {}
    render::ValidationSink sink() { return {capture, this}; }
    void phase(std::string phase);
    Json snapshot() const;
    bool failed() const;
    std::atomic_uint messageCount{0};
private:
    static void capture(void* context, const render::ValidationMessage& message) noexcept;
    mutable std::mutex mutex_;
    Json messages_ = Json::array();
    std::string phase_ = "createDevice";
    size_t limit_;
    std::atomic_bool lost_{false};
    bool error_ = false;
};

} // namespace metallic::tests::bench
