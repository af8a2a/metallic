#pragma once

#include "Runtime/Environment/WorldEnvironment.h"
#include "Runtime/Render/Environment/AtmosphereKernelParameters.h"

#include <array>
#include <memory>
#include <string>

namespace metallic::render {

GPUAtmosphereParameters buildGPUAtmosphereParameters(const environment::EnvironmentSnapshot& environment,
    const std::array<double, 3>& observerWorldMetres);
// Piecewise-linear spectral reconstruction, CIE XYZ integration, then AP1.
// The 683 lm/W factor preserves the renderer's photometric irradiance units.
float3 atmosphereSpectrumToWorking(const float3& spectrum);
std::array<float, 3> atmosphereSpectrumToWorkingColor(std::array<float, 3> spectrum);

// One immutable publication. Record exactly once, then retain the instance for
// every frame which reads its resources; cancellation discards this instance.
class AtmosphereResourcesGPU {
public:
    static constexpr uint32_t kTransmittanceWidth = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;
    static constexpr uint32_t kMultiScatteringSize = 32;
    static constexpr uint32_t kSkyViewWidth = 192;
    static constexpr uint32_t kSkyViewHeight = 108;
    static constexpr uint32_t kRadianceWidth = 512;
    static constexpr uint32_t kRadianceHeight = 256;
    static constexpr uint32_t kAerialSize = 32;
    static constexpr uint32_t kRadianceMipCount = 10;
    static constexpr uint32_t kCloudShadowSize = 256;

    AtmosphereResourcesGPU();
    ~AtmosphereResourcesGPU();
    AtmosphereResourcesGPU(const AtmosphereResourcesGPU&) = delete;
    AtmosphereResourcesGPU& operator=(const AtmosphereResourcesGPU&) = delete;
    // A submitted publication with identical static medium can supply the two
    // medium LUTs. Their image owners remain retained independently of it.
    Result<> initialize(Device& device, std::string& log, const AtmosphereResourcesGPU* sharedMedium = nullptr);
    Result<> record(CommandBuffer& commands, const environment::EnvironmentSnapshot& environment,
        const std::array<double, 3>& observerWorldMetres, std::string& log);
    TextureView* radianceView() const;
    TextureView* transmittanceView() const;
    TextureView* multiScatteringView() const;
    TextureView* skyView() const;
    TextureView* cloudShadowView() const;
    Buffer* aerialBuffer() const;
    Buffer* parametersBuffer() const;
    const GPUAtmosphereParameters& parameters() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
