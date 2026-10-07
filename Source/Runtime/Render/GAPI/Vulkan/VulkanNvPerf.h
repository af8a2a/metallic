#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace metallic::render {
class Device;
class Queue;
class CommandBuffer;
}

namespace metallic::render::vulkan {

bool nvPerfRequested();
bool nvPerfPassActive();

// One process-owned session. Public in-frame mode is single pass; the private
// isolated owner may advance SDK passes only after restoring its frozen inputs.
// No device clock changes. Destroy before Device; caller drains before boundaries.
class NvPerfSession final {
public:
    NvPerfSession();
    ~NvPerfSession();
    bool begin(Device& device, Queue& queue, const std::filesystem::path& output, std::string& error);
    bool finish(std::string& error);
    void cancel();
    // Backend entry points used by the neutral frontend.
    bool beginIsolated(Device& device, Queue& queue, const std::filesystem::path& output,
        std::string phase, std::string& error);
    bool complete() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool pushNvPerfRange(CommandBuffer& commands, const char* name);
void popNvPerfRange(CommandBuffer& commands);
} // namespace metallic::render::vulkan
