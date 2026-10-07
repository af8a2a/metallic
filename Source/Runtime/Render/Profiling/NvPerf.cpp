#include "NvPerf.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNvPerf.h"

namespace metallic::render::profiling {
struct NvPerfSession::Impl { vulkan::NvPerfSession backend; };
NvPerfSession::NvPerfSession() : impl_(std::make_unique<Impl>()) {}
NvPerfSession::~NvPerfSession() = default;
bool nvPerfRequested() { return vulkan::nvPerfRequested(); }
bool nvPerfPassActive() { return vulkan::nvPerfPassActive(); }
bool NvPerfSession::begin(Device& device, Queue& queue, const std::filesystem::path& output, std::string& error)
{
    return impl_->backend.begin(device, queue, output, error);
}
bool NvPerfSession::beginIsolated(Device& device, Queue& queue, const std::filesystem::path& output,
    std::string phase, std::string& error)
{
    return impl_->backend.beginIsolated(device, queue, output, std::move(phase), error);
}
bool NvPerfSession::complete() const { return impl_->backend.complete(); }
bool NvPerfSession::finish(std::string& error) { return impl_->backend.finish(error); }
void NvPerfSession::cancel() { impl_->backend.cancel(); }
NvPerfRange::NvPerfRange(CommandBuffer& commands, const char* name)
{
    if (vulkan::pushNvPerfRange(commands, name)) { commands_ = &commands; }
}
NvPerfRange::~NvPerfRange()
{
    if (commands_) { vulkan::popNvPerfRange(*commands_); }
}
} // namespace metallic::render::profiling
