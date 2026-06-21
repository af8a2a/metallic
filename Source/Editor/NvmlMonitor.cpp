#include "Editor/NvmlMonitor.h"

#include "imgui.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic {
namespace {

using NvmlReturn = int;
using NvmlDevice = void*;

constexpr NvmlReturn kNvmlSuccess = 0;
constexpr uint32_t kNvmlTemperatureGpu = 0;
constexpr uint32_t kNvmlClockGraphics = 0;
constexpr uint32_t kNvmlClockSm = 1;
constexpr uint32_t kNvmlClockMemory = 2;
constexpr uint32_t kNvmlClockVideo = 3;

struct NvmlMemory {
    uint64_t total;
    uint64_t free;
    uint64_t used;
};

struct NvmlUtilization {
    uint32_t gpu;
    uint32_t memory;
};

template <typename Func>
Func loadNvmlFunction(SDL_SharedObject* library, const char* primaryName, const char* fallbackName = nullptr)
{
    if (library == nullptr) {
        return nullptr;
    }
    if (SDL_FunctionPointer function = SDL_LoadFunction(library, primaryName)) {
        return reinterpret_cast<Func>(function);
    }
    if (fallbackName != nullptr) {
        if (SDL_FunctionPointer function = SDL_LoadFunction(library, fallbackName)) {
            return reinterpret_cast<Func>(function);
        }
    }
    return nullptr;
}

std::string nvmlErrorString(NvmlReturn result, const NvmlMonitor::NvmlApi* api);

float cpuLoad()
{
#if defined(_WIN32)
    static uint64_t previousTotalTicks = 0;
    static uint64_t previousIdleTicks = 0;

    FILETIME idleTime{};
    FILETIME kernelTime{};
    FILETIME userTime{};
    if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        return 0.0f;
    }

    auto toUInt64 = [](const FILETIME& time) {
        return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
    };

    const uint64_t idleTicks = toUInt64(idleTime);
    const uint64_t totalTicks = toUInt64(kernelTime) + toUInt64(userTime);
    const uint64_t totalDelta = totalTicks - previousTotalTicks;
    const uint64_t idleDelta = idleTicks - previousIdleTicks;
    previousTotalTicks = totalTicks;
    previousIdleTicks = idleTicks;

    if (totalDelta == 0) {
        return 0.0f;
    }
    return std::clamp((1.0f - static_cast<float>(idleDelta) / static_cast<float>(totalDelta)) * 100.0f, 0.0f, 100.0f);
#else
    return 0.0f;
#endif
}

std::string formatBytes(uint64_t bytes)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    char buffer[64] = {};
    if (bytes >= static_cast<uint64_t>(kGiB)) {
        std::snprintf(buffer, sizeof(buffer), "%.2f GiB", static_cast<double>(bytes) / kGiB);
    } else if (bytes >= static_cast<uint64_t>(kMiB)) {
        std::snprintf(buffer, sizeof(buffer), "%.1f MiB", static_cast<double>(bytes) / kMiB);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%llu bytes", static_cast<unsigned long long>(bytes));
    }
    return buffer;
}

std::string formatWatts(uint32_t milliwatts)
{
    if (milliwatts == 0) {
        return "--";
    }
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%.1f W", static_cast<double>(milliwatts) / 1000.0);
    return buffer;
}

void progressBar(float value, const char* label)
{
    const float clamped = std::clamp(value, 0.0f, 100.0f);
    char overlay[64] = {};
    std::snprintf(overlay, sizeof(overlay), "%s %.0f%%", label, clamped);
    ImGui::ProgressBar(clamped / 100.0f, ImVec2(-1.0f, 0.0f), overlay);
}

ImU32 colorU32(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
    return IM_COL32(r, g, b, a);
}

void drawField(const char* name, const std::string& value)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(name);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(value.c_str());
}

void drawField(const char* name, uint64_t value)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(name);
    ImGui::TableNextColumn();
    ImGui::Text("%llu", static_cast<unsigned long long>(value));
}

void drawField(const char* name, uint32_t value, const char* suffix = "")
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(name);
    ImGui::TableNextColumn();
    if (value == 0) {
        ImGui::TextDisabled("--");
    } else {
        ImGui::Text("%u%s", value, suffix);
    }
}

std::string throttleReasonString(uint64_t reasons)
{
    if (reasons == 0) {
        return "None";
    }

    struct Reason {
        uint64_t bit;
        const char* name;
    };
    constexpr Reason kReasons[] = {
        {0x0000000000000001ull, "GPU idle"},
        {0x0000000000000002ull, "Application clocks"},
        {0x0000000000000004ull, "SW power cap"},
        {0x0000000000000008ull, "HW slowdown"},
        {0x0000000000000010ull, "Sync boost"},
        {0x0000000000000020ull, "SW thermal"},
        {0x0000000000000040ull, "HW thermal"},
        {0x0000000000000080ull, "HW power brake"},
        {0x0000000000000100ull, "Display clocks"},
    };

    std::string text;
    for (const Reason& reason : kReasons) {
        if ((reasons & reason.bit) == 0) {
            continue;
        }
        if (!text.empty()) {
            text += ", ";
        }
        text += reason.name;
    }
    if (text.empty()) {
        text = "Unknown";
    }
    return text;
}

template <typename ValueFunc>
void drawHistoryGraph(
    const NvmlMonitor& monitor,
    const NvmlMonitor::Device& device,
    const char* label,
    float maxValue,
    ImU32 color,
    ValueFunc valueFunc,
    float height = 150.0f)
{
    const size_t sampleCount = monitor.orderedSampleCount();
    if (sampleCount == 0) {
        ImGui::TextDisabled("No samples yet.");
        return;
    }

    ImGui::TextUnformatted(label);
    const float width = ImGui::GetContentRegionAvail().x;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(width, height);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), colorU32(22, 22, 22));
    drawList->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), colorU32(96, 96, 96, 180));

    const float resolvedMax = std::max(maxValue, 0.001f);
    ImVec2 previous{};
    bool hasPrevious = false;
    for (size_t index = 0; index < sampleCount; ++index) {
        const NvmlMonitor::Sample* sample = monitor.orderedSample(device, index);
        if (sample == nullptr || !sample->valid) {
            continue;
        }
        const float value = std::clamp(valueFunc(*sample), 0.0f, resolvedMax);
        const float x = pos.x + (sampleCount == 1 ? 0.0f : static_cast<float>(index) / static_cast<float>(sampleCount - 1)) * size.x;
        const float y = pos.y + size.y - (value / resolvedMax) * size.y;
        const ImVec2 point(x, y);
        if (hasPrevious) {
            drawList->AddLine(previous, point, color, 2.0f);
        }
        previous = point;
        hasPrevious = true;
    }

    char maxLabel[64] = {};
    std::snprintf(maxLabel, sizeof(maxLabel), "%.0f", resolvedMax);
    drawList->AddText(ImVec2(pos.x + 6.0f, pos.y + 5.0f), colorU32(220, 220, 220), maxLabel);
    ImGui::Dummy(size);
}

void drawOverviewGraph(const NvmlMonitor& monitor, const NvmlMonitor::Device& device)
{
    const size_t sampleCount = monitor.orderedSampleCount();
    if (sampleCount == 0) {
        ImGui::TextDisabled("No samples yet.");
        return;
    }

    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 180.0f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(width, height);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), colorU32(22, 22, 22));
    drawList->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), colorU32(96, 96, 96, 180));

    auto drawSeries = [&](auto valueFunc, ImU32 color, bool fill) {
        std::vector<ImVec2> points;
        points.reserve(sampleCount + 2);
        for (size_t index = 0; index < sampleCount; ++index) {
            const NvmlMonitor::Sample* sample = monitor.orderedSample(device, index);
            if (sample == nullptr || !sample->valid) {
                continue;
            }
            const float value = std::clamp(valueFunc(*sample), 0.0f, 100.0f);
            const float x = pos.x + (sampleCount == 1 ? 0.0f : static_cast<float>(index) / static_cast<float>(sampleCount - 1)) * size.x;
            const float y = pos.y + size.y - (value / 100.0f) * size.y;
            points.emplace_back(x, y);
        }

        if (points.size() < 2) {
            return;
        }
        if (fill) {
            for (size_t index = 1; index < points.size(); ++index) {
                const ImVec2 fillPoints[] = {
                    points[index - 1],
                    points[index],
                    ImVec2(points[index].x, pos.y + size.y),
                    ImVec2(points[index - 1].x, pos.y + size.y),
                };
                drawList->AddConvexPolyFilled(fillPoints, 4, color);
            }
        }
        for (size_t index = 1; index < points.size(); ++index) {
            drawList->AddLine(points[index - 1], points[index], color | IM_COL32_A_MASK, 2.0f);
        }
    };

    drawSeries([](const NvmlMonitor::Sample& sample) { return sample.gpuLoad; }, colorU32(60, 220, 70, 92), true);
    drawSeries([](const NvmlMonitor::Sample& sample) { return sample.memoryLoad; }, colorU32(40, 150, 230, 92), true);
    drawSeries([](const NvmlMonitor::Sample& sample) { return sample.cpuLoad; }, colorU32(245, 220, 55, 220), false);

    drawList->AddText(ImVec2(pos.x + 6.0f, pos.y + 5.0f), colorU32(220, 220, 220), "100%");
    ImGui::Dummy(size);
}

} // namespace

struct NvmlMonitor::NvmlApi {
    using Init = NvmlReturn (*)();
    using Shutdown = NvmlReturn (*)();
    using ErrorString = const char* (*)(NvmlReturn);
    using SystemGetDriverVersion = NvmlReturn (*)(char*, uint32_t);
    using DeviceGetCount = NvmlReturn (*)(uint32_t*);
    using DeviceGetHandleByIndex = NvmlReturn (*)(uint32_t, NvmlDevice*);
    using DeviceGetName = NvmlReturn (*)(NvmlDevice, char*, uint32_t);
    using DeviceGetMemoryInfo = NvmlReturn (*)(NvmlDevice, NvmlMemory*);
    using DeviceGetUtilizationRates = NvmlReturn (*)(NvmlDevice, NvmlUtilization*);
    using DeviceGetTemperature = NvmlReturn (*)(NvmlDevice, uint32_t, uint32_t*);
    using DeviceGetPowerUsage = NvmlReturn (*)(NvmlDevice, uint32_t*);
    using DeviceGetPowerManagementLimit = NvmlReturn (*)(NvmlDevice, uint32_t*);
    using DeviceGetClockInfo = NvmlReturn (*)(NvmlDevice, uint32_t, uint32_t*);
    using DeviceGetFanSpeed = NvmlReturn (*)(NvmlDevice, uint32_t*);
    using DeviceGetCurrentClocksThrottleReasons = NvmlReturn (*)(NvmlDevice, uint64_t*);

    Init init = nullptr;
    Shutdown shutdown = nullptr;
    ErrorString errorString = nullptr;
    SystemGetDriverVersion systemGetDriverVersion = nullptr;
    DeviceGetCount deviceGetCount = nullptr;
    DeviceGetHandleByIndex deviceGetHandleByIndex = nullptr;
    DeviceGetName deviceGetName = nullptr;
    DeviceGetMemoryInfo deviceGetMemoryInfo = nullptr;
    DeviceGetUtilizationRates deviceGetUtilizationRates = nullptr;
    DeviceGetTemperature deviceGetTemperature = nullptr;
    DeviceGetPowerUsage deviceGetPowerUsage = nullptr;
    DeviceGetPowerManagementLimit deviceGetPowerManagementLimit = nullptr;
    DeviceGetClockInfo deviceGetClockInfo = nullptr;
    DeviceGetFanSpeed deviceGetFanSpeed = nullptr;
    DeviceGetCurrentClocksThrottleReasons deviceGetCurrentClocksThrottleReasons = nullptr;
};

namespace {

std::string nvmlErrorString(NvmlReturn result, const NvmlMonitor::NvmlApi* api)
{
    if (api != nullptr && api->errorString != nullptr) {
        if (const char* message = api->errorString(result)) {
            return message;
        }
    }
    return "NVML error " + std::to_string(result);
}

} // namespace

bool NvmlMonitor::initialize(uint32_t sampleIntervalMilliseconds, size_t historySize)
{
    shutdown();
    sampleIntervalMilliseconds_ = sampleIntervalMilliseconds;
    historySize_ = std::max<size_t>(historySize, 16);
    offset_ = historySize_ - 1;
    sampleCount_ = 0;
    lastSampleTime_ = {};

    constexpr const char* kLibraryNames[] = {
#if defined(_WIN32)
        "nvml.dll",
        "C:/Windows/System32/nvml.dll",
#else
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
#endif
    };

    for (const char* libraryName : kLibraryNames) {
        library_ = SDL_LoadObject(libraryName);
        if (library_ != nullptr) {
            break;
        }
    }
    if (library_ == nullptr) {
        status_ = "NVML library was not found.";
        return false;
    }

    auto api = std::make_unique<NvmlApi>();
    api->init = loadNvmlFunction<NvmlApi::Init>(library_, "nvmlInit_v2", "nvmlInit");
    api->shutdown = loadNvmlFunction<NvmlApi::Shutdown>(library_, "nvmlShutdown");
    api->errorString = loadNvmlFunction<NvmlApi::ErrorString>(library_, "nvmlErrorString");
    api->systemGetDriverVersion =
        loadNvmlFunction<NvmlApi::SystemGetDriverVersion>(library_, "nvmlSystemGetDriverVersion");
    api->deviceGetCount = loadNvmlFunction<NvmlApi::DeviceGetCount>(library_, "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
    api->deviceGetHandleByIndex =
        loadNvmlFunction<NvmlApi::DeviceGetHandleByIndex>(library_, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
    api->deviceGetName = loadNvmlFunction<NvmlApi::DeviceGetName>(library_, "nvmlDeviceGetName");
    api->deviceGetMemoryInfo = loadNvmlFunction<NvmlApi::DeviceGetMemoryInfo>(library_, "nvmlDeviceGetMemoryInfo");
    api->deviceGetUtilizationRates =
        loadNvmlFunction<NvmlApi::DeviceGetUtilizationRates>(library_, "nvmlDeviceGetUtilizationRates");
    api->deviceGetTemperature = loadNvmlFunction<NvmlApi::DeviceGetTemperature>(library_, "nvmlDeviceGetTemperature");
    api->deviceGetPowerUsage = loadNvmlFunction<NvmlApi::DeviceGetPowerUsage>(library_, "nvmlDeviceGetPowerUsage");
    api->deviceGetPowerManagementLimit =
        loadNvmlFunction<NvmlApi::DeviceGetPowerManagementLimit>(library_, "nvmlDeviceGetPowerManagementLimit");
    api->deviceGetClockInfo = loadNvmlFunction<NvmlApi::DeviceGetClockInfo>(library_, "nvmlDeviceGetClockInfo");
    api->deviceGetFanSpeed = loadNvmlFunction<NvmlApi::DeviceGetFanSpeed>(library_, "nvmlDeviceGetFanSpeed");
    api->deviceGetCurrentClocksThrottleReasons =
        loadNvmlFunction<NvmlApi::DeviceGetCurrentClocksThrottleReasons>(library_, "nvmlDeviceGetCurrentClocksThrottleReasons");

    if (api->init == nullptr ||
        api->shutdown == nullptr ||
        api->deviceGetCount == nullptr ||
        api->deviceGetHandleByIndex == nullptr) {
        status_ = "NVML library is missing required entry points.";
        SDL_UnloadObject(library_);
        library_ = nullptr;
        return false;
    }

    NvmlReturn result = api->init();
    if (result != kNvmlSuccess) {
        status_ = "NVML initialization failed: " + nvmlErrorString(result, api.get());
        SDL_UnloadObject(library_);
        library_ = nullptr;
        return false;
    }

    uint32_t deviceCount = 0;
    result = api->deviceGetCount(&deviceCount);
    if (result != kNvmlSuccess || deviceCount == 0) {
        status_ = "NVML found no NVIDIA GPUs.";
        api->shutdown();
        SDL_UnloadObject(library_);
        library_ = nullptr;
        return false;
    }

    if (api->systemGetDriverVersion != nullptr) {
        char driver[96] = {};
        if (api->systemGetDriverVersion(driver, static_cast<uint32_t>(sizeof(driver))) == kNvmlSuccess) {
            driverVersion_ = driver;
        }
    }

    devices_.clear();
    devices_.reserve(deviceCount);
    for (uint32_t index = 0; index < deviceCount; ++index) {
        NvmlDevice handle = nullptr;
        result = api->deviceGetHandleByIndex(index, &handle);
        if (result != kNvmlSuccess || handle == nullptr) {
            continue;
        }

        Device device;
        device.handle = handle;
        device.name = "GPU-" + std::to_string(index);
        if (api->deviceGetName != nullptr) {
            char name[128] = {};
            if (api->deviceGetName(handle, name, static_cast<uint32_t>(sizeof(name))) == kNvmlSuccess && name[0] != '\0') {
                device.name = name;
            }
        }
        device.samples.resize(historySize_);
        devices_.push_back(std::move(device));
    }

    api_ = api.release();
    valid_ = !devices_.empty();
    status_ = valid_ ? "NVML monitor initialized." : "NVML did not expose any usable GPU handles.";
    refresh();
    return valid_;
}

void NvmlMonitor::shutdown()
{
    if (api_ != nullptr) {
        if (api_->shutdown != nullptr) {
            api_->shutdown();
        }
        delete api_;
        api_ = nullptr;
    }

    if (library_ != nullptr) {
        SDL_UnloadObject(library_);
        library_ = nullptr;
    }

    devices_.clear();
    driverVersion_.clear();
    valid_ = false;
    status_ = "NVML monitor not initialized.";
}

void NvmlMonitor::refresh()
{
    if (!valid_ || api_ == nullptr) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (lastSampleTime_.time_since_epoch().count() != 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSampleTime_);
        if (elapsed.count() < sampleIntervalMilliseconds_) {
            return;
        }
    }
    lastSampleTime_ = now;

    offset_ = (offset_ + 1) % historySize_;
    sampleCount_ = std::min(sampleCount_ + 1, historySize_);
    const float cpu = cpuLoad();

    for (Device& device : devices_) {
        Sample& sample = device.samples[offset_];
        sample = {};
        sample.cpuLoad = cpu;
        sample.valid = true;

        if (api_->deviceGetMemoryInfo != nullptr) {
            NvmlMemory memory{};
            if (api_->deviceGetMemoryInfo(device.handle, &memory) == kNvmlSuccess) {
                sample.memoryTotal = memory.total;
                sample.memoryFree = memory.free;
                sample.memoryUsed = memory.used;
                if (memory.total > 0) {
                    sample.memoryLoad = static_cast<float>(static_cast<double>(memory.used) * 100.0 / static_cast<double>(memory.total));
                }
            }
        }

        if (api_->deviceGetUtilizationRates != nullptr) {
            NvmlUtilization utilization{};
            if (api_->deviceGetUtilizationRates(device.handle, &utilization) == kNvmlSuccess) {
                sample.gpuLoad = static_cast<float>(utilization.gpu);
                if (sample.memoryLoad == 0.0f) {
                    sample.memoryLoad = static_cast<float>(utilization.memory);
                }
            }
        }

        if (api_->deviceGetTemperature != nullptr) {
            api_->deviceGetTemperature(device.handle, kNvmlTemperatureGpu, &sample.temperature);
        }
        if (api_->deviceGetPowerUsage != nullptr) {
            api_->deviceGetPowerUsage(device.handle, &sample.powerMilliwatts);
        }
        if (api_->deviceGetPowerManagementLimit != nullptr) {
            api_->deviceGetPowerManagementLimit(device.handle, &sample.powerLimitMilliwatts);
        }
        if (api_->deviceGetFanSpeed != nullptr) {
            api_->deviceGetFanSpeed(device.handle, &sample.fanSpeed);
        }
        if (api_->deviceGetClockInfo != nullptr) {
            api_->deviceGetClockInfo(device.handle, kNvmlClockGraphics, &sample.graphicsClockMHz);
            api_->deviceGetClockInfo(device.handle, kNvmlClockSm, &sample.smClockMHz);
            api_->deviceGetClockInfo(device.handle, kNvmlClockMemory, &sample.memoryClockMHz);
            api_->deviceGetClockInfo(device.handle, kNvmlClockVideo, &sample.videoClockMHz);
        }
        if (api_->deviceGetCurrentClocksThrottleReasons != nullptr) {
            api_->deviceGetCurrentClocksThrottleReasons(device.handle, &sample.throttleReasons);
        }
    }
}

void NvmlMonitor::drawWindow(bool* open)
{
    refresh();
    if (open != nullptr && !*open) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(520.0f, 360.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("NVML Monitor", open)) {
        ImGui::End();
        return;
    }

    if (!valid_) {
        ImGui::TextWrapped("%s", status_.c_str());
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("MonitorTabs")) {
        if (ImGui::BeginTabItem("All")) {
            drawAllTab();
            ImGui::EndTabItem();
        }

        for (size_t deviceIndex = 0; deviceIndex < devices_.size(); ++deviceIndex) {
            const std::string tabName = "GPU-" + std::to_string(deviceIndex);
            if (ImGui::BeginTabItem(tabName.c_str())) {
                drawGpuTab(deviceIndex);
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

const NvmlMonitor::Sample* NvmlMonitor::currentSample(const Device& device) const
{
    if (device.samples.empty() || sampleCount_ == 0) {
        return nullptr;
    }
    return &device.samples[offset_];
}

const NvmlMonitor::Sample* NvmlMonitor::orderedSample(const Device& device, size_t index) const
{
    const size_t count = orderedSampleCount();
    if (device.samples.empty() || index >= count) {
        return nullptr;
    }
    const size_t start = sampleCount_ < historySize_ ? 0 : (offset_ + 1) % historySize_;
    return &device.samples[(start + index) % historySize_];
}

size_t NvmlMonitor::orderedSampleCount() const
{
    return std::min(sampleCount_, historySize_);
}

void NvmlMonitor::drawAllTab()
{
    if (!driverVersion_.empty()) {
        ImGui::Text("Driver: %s", driverVersion_.c_str());
        ImGui::Separator();
    }

    for (size_t deviceIndex = 0; deviceIndex < devices_.size(); ++deviceIndex) {
        const Device& device = devices_[deviceIndex];
        const Sample* sample = currentSample(device);
        if (sample == nullptr) {
            continue;
        }

        ImGui::PushID(static_cast<int>(deviceIndex));
        ImGui::Text("GPU-%llu: %s", static_cast<unsigned long long>(deviceIndex), device.name.c_str());
        progressBar(sample->gpuLoad, "Load");
        progressBar(sample->memoryLoad, "Memory");
        if (sample->memoryTotal > 0) {
            ImGui::Text(
                "Memory: %s / %s",
                formatBytes(sample->memoryUsed).c_str(),
                formatBytes(sample->memoryTotal).c_str());
        }
        const std::string temperature = sample->temperature == 0 ? "--" : std::to_string(sample->temperature) + " C";
        ImGui::Text(
            "CPU %.1f%%  Temp %s  Power %s",
            sample->cpuLoad,
            temperature.c_str(),
            formatWatts(sample->powerMilliwatts).c_str());
        ImGui::Separator();
        ImGui::PopID();
    }
}

void NvmlMonitor::drawGpuTab(size_t deviceIndex)
{
    if (deviceIndex >= devices_.size()) {
        return;
    }

    const std::string tabBarName = "GpuTabBar" + std::to_string(deviceIndex);
    if (!ImGui::BeginTabBar(tabBarName.c_str())) {
        return;
    }

    if (ImGui::BeginTabItem("Overview")) {
        drawOverview(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Details")) {
        drawDetails(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Performance")) {
        drawPerformance(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Power")) {
        drawPower(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Utilization")) {
        drawUtilization(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Memory")) {
        drawMemory(deviceIndex);
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Clocks")) {
        drawClocks(deviceIndex);
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

void NvmlMonitor::drawOverview(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    ImGui::TextUnformatted(device.name.c_str());
    drawOverviewGraph(*this, device);
    ImGui::TextColored(ImVec4(0.30f, 0.95f, 0.35f, 1.0f), "Load: %.0f%%", sample->gpuLoad);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.20f, 0.65f, 1.0f, 1.0f), "Memory: %.0f%%", sample->memoryLoad);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.90f, 0.25f, 1.0f), "CPU: %.1f%%", sample->cpuLoad);
}

void NvmlMonitor::drawDetails(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    if (ImGui::BeginTable("NvmlDetails", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 170.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        drawField("Device", device.name);
        drawField("Driver", driverVersion_.empty() ? "--" : driverVersion_);
        drawField("Memory total", formatBytes(sample->memoryTotal));
        drawField("Current throttle", throttleReasonString(sample->throttleReasons));
        drawField("History samples", static_cast<uint64_t>(orderedSampleCount()));
        ImGui::EndTable();
    }
}

void NvmlMonitor::drawPerformance(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    const bool throttled = (sample->throttleReasons & ~0x1ull) != 0;
    if (throttled) {
        ImGui::TextColored(
            ImVec4(1.0f, 0.18f, 0.12f, 1.0f),
            "Throttle: %s",
            throttleReasonString(sample->throttleReasons).c_str());
    } else {
        ImGui::Text("Throttle: %s", throttleReasonString(sample->throttleReasons).c_str());
    }

    drawHistoryGraph(
        *this,
        device,
        "Graphics clock (MHz)",
        static_cast<float>(std::max(sample->graphicsClockMHz, 1u)) * 1.25f,
        colorU32(80, 190, 255),
        [](const Sample& item) { return static_cast<float>(item.graphicsClockMHz); });
}

void NvmlMonitor::drawPower(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    if (ImGui::BeginTable("NvmlPower", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 170.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        drawField("Power", formatWatts(sample->powerMilliwatts));
        drawField("Power limit", formatWatts(sample->powerLimitMilliwatts));
        drawField("Temperature", sample->temperature, " C");
        drawField("Fan", sample->fanSpeed, "%");
        ImGui::EndTable();
    }

    drawHistoryGraph(
        *this,
        device,
        "Power (W)",
        static_cast<float>(std::max(sample->powerLimitMilliwatts, sample->powerMilliwatts)) / 1000.0f,
        colorU32(255, 180, 60),
        [](const Sample& item) { return static_cast<float>(item.powerMilliwatts) / 1000.0f; });
}

void NvmlMonitor::drawUtilization(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    progressBar(sample->gpuLoad, "GPU");
    progressBar(sample->memoryLoad, "Memory");
    progressBar(sample->cpuLoad, "CPU");

    drawHistoryGraph(
        *this,
        device,
        "GPU utilization (%)",
        100.0f,
        colorU32(70, 230, 80),
        [](const Sample& item) { return item.gpuLoad; });
}

void NvmlMonitor::drawMemory(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    if (ImGui::BeginTable("NvmlMemory", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 170.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        drawField("Used", formatBytes(sample->memoryUsed));
        drawField("Free", formatBytes(sample->memoryFree));
        drawField("Total", formatBytes(sample->memoryTotal));
        drawField("Load", std::to_string(static_cast<uint32_t>(sample->memoryLoad)) + "%");
        ImGui::EndTable();
    }

    constexpr float kGiB = 1024.0f * 1024.0f * 1024.0f;
    drawHistoryGraph(
        *this,
        device,
        "Memory used (GiB)",
        std::max(static_cast<float>(sample->memoryTotal) / kGiB, 1.0f),
        colorU32(65, 160, 255),
        [](const Sample& item) { return static_cast<float>(item.memoryUsed) / kGiB; });
}

void NvmlMonitor::drawClocks(size_t deviceIndex)
{
    const Device& device = devices_[deviceIndex];
    const Sample* sample = currentSample(device);
    if (sample == nullptr) {
        return;
    }

    if (ImGui::BeginTable("NvmlClocks", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 170.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        drawField("Graphics", sample->graphicsClockMHz, " MHz");
        drawField("SM", sample->smClockMHz, " MHz");
        drawField("Memory", sample->memoryClockMHz, " MHz");
        drawField("Video", sample->videoClockMHz, " MHz");
        ImGui::EndTable();
    }
}

} // namespace metallic
