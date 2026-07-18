#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

inline constexpr uint32_t kReGIRHeaderRecordCount = 3;
inline constexpr uint32_t kReGIRRecordByteSize = 16;

struct ReGIRGridLayout {
    uint32_t gridSize = 0;
    uint32_t lightsPerCell = 0;
    uint32_t cellCount = 0;
    uint32_t lightSlotCount = 0;
    uint64_t bufferByteSize = 0;

    bool valid() const;
};

ReGIRGridLayout computeReGIRGridLayout(uint32_t gridSize, uint32_t lightsPerCell);

struct ReGIRBuildParameters {
    uint32_t lightCount = 0;
    uint32_t buildSamples = 8;
    uint32_t frameIndex = 0;
    bool animateLights = true;
    float sceneCenter[3] = {};
    float sceneRadius = 1.0f;
    float lightIntensity = 1.0f;
    float samplingJitter = 1.0f;
};

class ReGIRLightSelector final {
public:
    ReGIRLightSelector();
    ~ReGIRLightSelector();

    ReGIRLightSelector(ReGIRLightSelector&&) noexcept;
    ReGIRLightSelector& operator=(ReGIRLightSelector&&) noexcept;

    ReGIRLightSelector(const ReGIRLightSelector&) = delete;
    ReGIRLightSelector& operator=(const ReGIRLightSelector&) = delete;

    Result initialize(Device& device, std::string& log);
    Result ensureGrid(
        Device& device,
        uint32_t gridSize,
        uint32_t lightsPerCell,
        std::string& log);
    Result build(
        CommandBuffer& commandBuffer,
        TextureView& localLightPdf,
        const ReGIRBuildParameters& parameters);
    void clear();

    bool valid() const;
    Buffer* buffer() const;
    const ReGIRGridLayout& layout() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
