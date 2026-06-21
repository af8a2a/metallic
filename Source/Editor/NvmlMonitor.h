#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct SDL_SharedObject;

namespace metallic {

class NvmlMonitor {
public:
    struct Sample {
        float cpuLoad = 0.0f;
        float gpuLoad = 0.0f;
        float memoryLoad = 0.0f;
        uint64_t memoryUsed = 0;
        uint64_t memoryFree = 0;
        uint64_t memoryTotal = 0;
        uint32_t temperature = 0;
        uint32_t powerMilliwatts = 0;
        uint32_t powerLimitMilliwatts = 0;
        uint32_t fanSpeed = 0;
        uint32_t graphicsClockMHz = 0;
        uint32_t smClockMHz = 0;
        uint32_t memoryClockMHz = 0;
        uint32_t videoClockMHz = 0;
        uint64_t throttleReasons = 0;
        bool valid = false;
    };

    struct Device {
        void* handle = nullptr;
        std::string name;
        std::vector<Sample> samples;
    };

    struct NvmlApi;

    bool initialize(uint32_t sampleIntervalMilliseconds = 100, size_t historySize = 120);
    void shutdown();
    void refresh();
    void drawWindow(bool* open);

    bool valid() const { return valid_; }
    const std::string& status() const { return status_; }
    const Sample* currentSample(const Device& device) const;
    const Sample* orderedSample(const Device& device, size_t index) const;
    size_t orderedSampleCount() const;

private:
    void drawAllTab();
    void drawGpuTab(size_t deviceIndex);
    void drawOverview(size_t deviceIndex);
    void drawDetails(size_t deviceIndex);
    void drawPerformance(size_t deviceIndex);
    void drawPower(size_t deviceIndex);
    void drawUtilization(size_t deviceIndex);
    void drawMemory(size_t deviceIndex);
    void drawClocks(size_t deviceIndex);

    SDL_SharedObject* library_ = nullptr;
    NvmlApi* api_ = nullptr;
    std::vector<Device> devices_;
    std::string status_ = "NVML monitor not initialized.";
    std::string driverVersion_;
    std::chrono::steady_clock::time_point lastSampleTime_{};
    uint32_t sampleIntervalMilliseconds_ = 100;
    size_t historySize_ = 120;
    size_t offset_ = 0;
    size_t sampleCount_ = 0;
    bool valid_ = false;
};

} // namespace metallic
