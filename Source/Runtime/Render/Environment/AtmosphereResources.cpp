#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Environment/CelestialLighting.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {

float3 atmosphereSpectrumToWorking(const float3& s)
{
    // 1 nm trapezoidal integration over 360..830 nm of piecewise-linear
    // spectral basis functions at 680/550/440 nm (constant endpoint tails).
    // CMFs: Wyman, Sloan and Shirley, JCGT 2(2), 2013, equation 4/table 1.
    // https://jcgt.org/published/0002/02/01/paper.pdf
    const float3 xyz{
        683.0f * (32.2813132251f * s.x + 57.6281726163f * s.y + 16.8049788713f * s.z),
        683.0f * (18.3286150698f * s.x + 76.9932864076f * s.y + 11.6242660516f * s.z),
        683.0f * (0.00697408432f * s.x + 17.1338590149f * s.y + 89.7146153395f * s.z),
    };
    const auto result = color::transform(sceneWorkingColorSpace() == SceneWorkingColorSpace::ACEScg ?
        color::kXYZToAP1 : color::kXYZToRec709, {xyz.x, xyz.y, xyz.z});
    return {result[0], result[1], result[2]};
}

GPUAtmosphereParameters buildGPUAtmosphereParameters(const environment::EnvironmentSnapshot& environment,
    const std::array<double, 3>& observerWorldMetres)
{
    const auto& a = environment.atmosphere;
    GPUAtmosphereParameters p{};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        // Subtract before narrowing: an observer can share a very large world
        // origin with the planet without losing its local altitude.
        p.observerPlanetBottom[axis] = static_cast<float>((observerWorldMetres[axis] - a.planetCenter[axis]) * 0.001);
        p.observerWorldTop[axis] = static_cast<float>(observerWorldMetres[axis]);
    }
    p.observerPlanetBottom[3] = a.bottomRadiusKm;
    p.observerWorldTop[3] = a.topRadiusKm;
    p.rayleighScaleHeight = {a.rayleighScattering.x, a.rayleighScattering.y, a.rayleighScattering.z, a.rayleighScaleHeightKm};
    p.mieScatteringScaleHeight = {a.mieScattering.x, a.mieScattering.y, a.mieScattering.z, a.mieScaleHeightKm};
    p.mieExtinctionAnisotropy = {a.mieExtinction.x, a.mieExtinction.y, a.mieExtinction.z, a.mieAnisotropy};
    p.ozoneCenter = {a.ozoneAbsorption.x, a.ozoneAbsorption.y, a.ozoneAbsorption.z, a.ozoneCenterAltitudeKm};
    p.groundOzoneWidth = {a.groundAlbedo.x, a.groundAlbedo.y, a.groundAlbedo.z, a.ozoneWidthKm};
    auto source = [](const environment::CelestialLight& light, auto& direction, auto& spectrum) {
        const double length = std::sqrt(double(light.direction.x) * light.direction.x +
            double(light.direction.y) * light.direction.y + double(light.direction.z) * light.direction.z);
        direction = {static_cast<float>(-light.direction.x / length), static_cast<float>(-light.direction.y / length),
            static_cast<float>(-light.direction.z / length), light.angularRadius};
        spectrum = {light.topOfAtmosphereIrradiance.x, light.topOfAtmosphereIrradiance.y,
            light.topOfAtmosphereIrradiance.z, light.enabled ? 1.0f : 0.0f};
    };
    source(environment.celestial[0], p.sunDirectionRadius, p.sunIrradianceEnabled);
    source(environment.celestial[1], p.moonDirectionRadius, p.moonIrradianceEnabled);
    p.settings = {a.maxAerialDistanceKm, environment.source == environment::EnvironmentSource::PhysicalAtmosphere ? 1.0f : 0.0f, 0.0f, 0.0f};
    const auto& weather = environment.weather;
    p.cloudLayer = {weather.cloudBaseAltitudeKm, weather.cloudTopAltitudeKm,
        weather.cloudCoverage, weather.cloudDensity * (1.0f + 0.5f * weather.precipitation)};
    p.cloudOptics = {weather.cloudExtinctionPerKm, 0.98f, 0.75f, weather.cloudEnabled ? 1.0f : 0.0f};
    const auto wind = environment::normalizedWeatherWindDirection(weather);
    // Every noise octave repeats at a divisor of 2048 km. Reduce in double
    // before narrowing so continuous advection also works after long runs.
    const bool movingClouds = weather.cloudEnabled && weather.cloudCoverage > 0.0f && weather.cloudDensity > 0.0f &&
        weather.cloudExtinctionPerKm > 0.0f && weather.windSpeed > 0.0f;
    const double cloudElapsed = movingClouds ? environment.elapsedSeconds : 0.0;
    const double windKm = cloudElapsed * double(weather.windSpeed) * 0.001;
    p.cloudAdvection = {static_cast<float>(std::remainder(windKm * wind.x, 2048.0)), 0.0f,
        static_cast<float>(std::remainder(windKm * wind.y, 2048.0)), std::bit_cast<float>(weather.noiseSeed)};
    const double observerRadius = std::sqrt(double(p.observerPlanetBottom[0]) * p.observerPlanetBottom[0] +
        double(p.observerPlanetBottom[1]) * p.observerPlanetBottom[1] + double(p.observerPlanetBottom[2]) * p.observerPlanetBottom[2]);
    for (uint32_t axis = 0; axis < 3; ++axis) {
        p.cloudShadowCentre[axis] = static_cast<float>(p.observerPlanetBottom[axis] * (a.bottomRadiusKm + 0.002) / std::max(observerRadius, 1e-6));
    }
    p.cloudShadowCentre[3] = 32.0f;
    const auto shadowPlan = buildCelestialShadowPlan(environment, observerWorldMetres);
    p.cloudShadowBudget = {float(shadowPlan.sampleCounts[0]), float(shadowPlan.sampleCounts[1]),
        float(shadowPlan.activeMask), shadowPlan.dominantIndex == 0xffffffffu ? -1.0f : float(shadowPlan.dominantIndex)};
    p.weatherComposition = {weather.aerosolDensity, weather.humidity, weather.precipitation,
        static_cast<float>(std::remainder(cloudElapsed, 1024.0))};
    const auto& astronomy = environment.evaluatedAstronomy;
    p.moonToSun = {astronomy.moonToSunDirection.x, astronomy.moonToSunDirection.y,
        astronomy.moonToSunDirection.z, astronomy.automatic ? 1.0f : 0.0f};
    p.moonPhase = {astronomy.moonPhaseAngleRadians, astronomy.moonIlluminatedFraction, astronomy.moonLambertPhase, 0.0f};
    return p;
}

std::array<float, 3> atmosphereSpectrumToWorkingColor(std::array<float, 3> spectrum)
{
    const auto result = atmosphereSpectrumToWorking({spectrum[0], spectrum[1], spectrum[2]});
    return {result.x, result.y, result.z};
}

struct AtmosphereResourcesGPU::Impl {
    struct Image {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        std::vector<std::unique_ptr<TextureView>> mips;
    };
    Device* device = nullptr;
    std::array<ComputeKernel, 6> kernels;
    std::shared_ptr<Image> transmittance = std::make_shared<Image>();
    std::shared_ptr<Image> multiScattering = std::make_shared<Image>();
    Image skyView, radiance, cloudShadow;
    std::unique_ptr<Buffer> aerial;
    std::unique_ptr<Buffer> parameterBuffer;
    GPUAtmosphereParameters parameters{};
    bool recorded = false;
    bool sharedMedium = false;
    GPUAtmosphereParameters mediumParameters{};

    Result<> createImage(Image& image, uint32_t width, uint32_t height, uint32_t mipCount,
        Format format = Format::RGBA16Sfloat)
    {
        auto result = device->createTexture(TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage | TextureUsageBits::TransferSource,
            .format = format, .width = width, .height = height, .depth = 1,
            .mipCount = mipCount, .layerCount = 1, .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
        }).transform([&](auto value) { image.texture = std::move(value); });
        if (!result) { return result; }
        result = device->createTextureView(*image.texture, TextureViewDesc{.format = format,
            .range = {.baseMip = 0, .mipCount = mipCount, .baseLayer = 0, .layerCount = 1}})
            .transform([&](auto value) { image.view = std::move(value); });
        if (!result) { return result; }
        image.mips.resize(mipCount);
        for (uint32_t mip = 0; mip < mipCount; ++mip) {
            result = device->createTextureView(*image.texture, TextureViewDesc{.format = format,
                .range = {.baseMip = mip, .mipCount = 1, .baseLayer = 0, .layerCount = 1}})
                .transform([&](auto value) { image.mips[mip] = std::move(value); });
            if (!result) { return result; }
        }
        return {};
    }

    Result<> imageBarrier(CommandBuffer& commands, Image& image, uint32_t mip, TextureLayout before, TextureLayout after)
    {
        TextureBarrierDesc barrier{.texture = image.texture.get(), .oldLayout = before, .newLayout = after,
            .before = {before == TextureLayout::Undefined ? PipelineStageBits::None : PipelineStageBits::AllCommands,
                before == TextureLayout::Undefined ? AccessBits::None : AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .range = {.baseMip = mip, .mipCount = 1, .baseLayer = 0, .layerCount = 1}};
        return commands.synchronize(BarrierDesc{.textures = {&barrier, 1}});
    }
};

AtmosphereResourcesGPU::AtmosphereResourcesGPU() : impl_(std::make_unique<Impl>()) {}
AtmosphereResourcesGPU::~AtmosphereResourcesGPU() = default;

Result<> AtmosphereResourcesGPU::initialize(Device& device, std::string& log, const AtmosphereResourcesGPU* sharedMedium)
{
    if (impl_->device != nullptr) { return makeError(Error::InvalidArgument); }
    impl_->device = &device;
    if (sharedMedium != nullptr) {
        if (sharedMedium->impl_->device != &device || !sharedMedium->impl_->recorded) {
            log = "Shared atmosphere medium must be a recorded publication on the same device";
            return makeError(Error::InvalidArgument);
        }
        impl_->transmittance = sharedMedium->impl_->transmittance;
        impl_->multiScattering = sharedMedium->impl_->multiScattering;
        impl_->mediumParameters = sharedMedium->impl_->parameters;
        impl_->sharedMedium = true;
    }
    constexpr std::array<const char*, 6> modules{"Transmittance", "MultiScattering", "SkyView", "EnvironmentCapture", "AerialPerspective", "CloudShadow"};
    constexpr std::array<const char*, 6> entries{"transmittanceMain", "multiScatteringMain", "skyViewMain", "environmentCaptureMain", "aerialPerspectiveMain", "cloudShadowMain"};
    for (uint32_t index = 0; index < modules.size(); ++index) {
        const std::string module = std::string("Features/Environment/") + modules[index];
        auto result = ShaderRegistry::instance().getComputeKernel(device,
            SlangShaderDesc{.moduleName = module.c_str(), .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            ComputeKernelDesc{.parameters = parameterAbi<AtmospherePrecomputeParams>(kAtmospherePrecomputeABI, ParameterTransport::InlinePush),
                .debugName = modules[index]}, impl_->kernels[index], log);
        if (!result) { return result; }
    }
    for (auto request : {std::array<uint32_t, 4>{0, kTransmittanceWidth, kTransmittanceHeight, 1},
        {1, kMultiScatteringSize, kMultiScatteringSize, 1}, {2, kSkyViewWidth, kSkyViewHeight, 1},
        {3, kRadianceWidth, kRadianceHeight, kRadianceMipCount}, {4, kCloudShadowSize, kCloudShadowSize, 1}}) {
        if (impl_->sharedMedium && request[0] < 2) { continue; }
        auto* image = request[0] == 0 ? impl_->transmittance.get() : request[0] == 1 ? impl_->multiScattering.get() :
            request[0] == 2 ? &impl_->skyView : request[0] == 3 ? &impl_->radiance : &impl_->cloudShadow;
        const auto format = request[0] == 2 || request[0] == 3 ? Format::RGBA32Sfloat : Format::RGBA16Sfloat;
        if (auto result = impl_->createImage(*image, request[1], request[2], request[3], format); !result) { return result; }
    }
    auto result = device.createBuffer(BufferDesc{.size = uint64_t(kAerialSize) * kAerialSize * kAerialSize * 2 * 16,
        .structureStride = 16, .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::Device,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy})
        .transform([&](auto value) { impl_->aerial = std::move(value); });
    if (!result) { return result; }
    return device.createBuffer(BufferDesc{.size = sizeof(GPUAtmosphereParameters), .structureStride = 16,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource, .memoryLocation = MemoryLocation::HostUpload,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
        .transform([&](auto value) { impl_->parameterBuffer = std::move(value); });
}

Result<> AtmosphereResourcesGPU::record(CommandBuffer& commands, const environment::EnvironmentSnapshot& environment,
    const std::array<double, 3>& observerWorldMetres, std::string& log)
{
    if (impl_->device == nullptr || impl_->recorded || !environment::validAtmosphereState(environment.atmosphere) ||
        !environment::validCelestialLight(environment.celestial[0]) || !environment::validCelestialLight(environment.celestial[1]) ||
        !environment::validWeatherState(environment.weather) || !std::isfinite(environment.elapsedSeconds) ||
        (environment.source == environment::EnvironmentSource::PhysicalAtmosphere && environment.weather.cloudEnabled &&
            environment.weather.cloudCoverage > 0.0f && environment.weather.cloudDensity > 0.0f &&
            environment.weather.cloudExtinctionPerKm > 0.0f && environment.weather.cloudTopAltitudeKm >
                double(environment.atmosphere.topRadiusKm) - environment.atmosphere.bottomRadiusKm) ||
        !std::all_of(observerWorldMetres.begin(), observerWorldMetres.end(), [](double value) { return std::isfinite(value); })) {
        log = "Invalid or already recorded immutable atmosphere publication";
        return makeError(Error::InvalidArgument);
    }
    impl_->parameters = buildGPUAtmosphereParameters(environment, observerWorldMetres);
    if (impl_->sharedMedium) {
        const auto& p = impl_->parameters;
        const auto& cached = impl_->mediumParameters;
        if (p.observerPlanetBottom[3] != cached.observerPlanetBottom[3] || p.observerWorldTop[3] != cached.observerWorldTop[3] ||
            p.rayleighScaleHeight != cached.rayleighScaleHeight || p.mieScatteringScaleHeight != cached.mieScatteringScaleHeight ||
            p.mieExtinctionAnisotropy != cached.mieExtinctionAnisotropy || p.ozoneCenter != cached.ozoneCenter ||
            p.groundOzoneWidth != cached.groundOzoneWidth) {
            log = "Shared atmosphere medium parameters do not match this publication";
            return makeError(Error::InvalidArgument);
        }
    }
    void* mapped = impl_->parameterBuffer->map();
    if (mapped == nullptr) { return makeError(Error::Failure); }
    std::memcpy(mapped, &impl_->parameters, sizeof(impl_->parameters));
    impl_->parameterBuffer->flush({0, sizeof(impl_->parameters)});
    impl_->parameterBuffer->unmap();
    commands.hostWriteBarrier();
    auto registry = ResourceRegistry::forDevice(*impl_->device);
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*impl_->device, **registry, RenderFrameContext::from(commands));
    AtmospherePrecomputeParams params{};
    params.parameters = writer.bufferSpan(impl_->parameterBuffer.get(), 16, 16);
    auto dispatch = [&](uint32_t kernel, uint32_t width, uint32_t height, uint32_t depth = 1) -> Result<> {
        auto encoded = writer.encode(params, kAtmospherePrecomputeABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return impl_->kernels[kernel].dispatch(commands, *encoded, (width + 7) / 8, (height + 7) / 8, depth);
    };
    auto imageDispatch = [&](uint32_t kernel, Impl::Image& image, uint32_t width, uint32_t height, uint32_t mip = 0) -> Result<> {
        if (auto result = impl_->imageBarrier(commands, image, mip, TextureLayout::Undefined, TextureLayout::General); !result) { return result; }
        params.output = writer.storageImage(image.mips[mip].get());
        params.dimensions = {width, height, 1, mip == 0 ? 0u : 1u};
        if (auto result = dispatch(kernel, width, height); !result) { return result; }
        return impl_->imageBarrier(commands, image, mip, TextureLayout::General, TextureLayout::ShaderRead);
    };
    if (!impl_->sharedMedium) {
        if (auto result = imageDispatch(0, *impl_->transmittance, kTransmittanceWidth, kTransmittanceHeight); !result) { return result; }
    }
    params.transmittance = writer.sampledImage(impl_->transmittance->view.get());
    if (!impl_->sharedMedium) {
        if (auto result = imageDispatch(1, *impl_->multiScattering, kMultiScatteringSize, kMultiScatteringSize); !result) { return result; }
    }
    params.multiScattering = writer.sampledImage(impl_->multiScattering->view.get());
    if (auto result = imageDispatch(5, impl_->cloudShadow, kCloudShadowSize, kCloudShadowSize); !result) { return result; }
    if (auto result = imageDispatch(2, impl_->skyView, kSkyViewWidth, kSkyViewHeight); !result) { return result; }
    params.skyView = writer.sampledImage(impl_->skyView.view.get());
    if (auto result = imageDispatch(3, impl_->radiance, kRadianceWidth, kRadianceHeight); !result) { return result; }
    for (uint32_t mip = 1; mip < kRadianceMipCount; ++mip) {
        params.sourceMip = writer.sampledImage(impl_->radiance.mips[mip - 1].get());
        if (auto result = imageDispatch(3, impl_->radiance, std::max(kRadianceWidth >> mip, 1u),
            std::max(kRadianceHeight >> mip, 1u), mip); !result) { return result; }
    }
    params.aerial = writer.bufferSpan(impl_->aerial.get(), 16, 16);
    params.dimensions = {kAerialSize, kAerialSize, kAerialSize, 0};
    if (auto result = dispatch(4, kAerialSize, kAerialSize, kAerialSize); !result) { return result; }
    BufferBarrierDesc barrier{.buffer = impl_->aerial.get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead},
        .range = {.offset = 0, .size = impl_->aerial->desc().size}};
    if (auto result = commands.synchronize(BarrierDesc{.buffers = {&barrier, 1}}); !result) { return result; }
    impl_->recorded = true;
    return {};
}

void AtmosphereResourcesGPU::appendDebugBindings(std::vector<DebugResourceBinding>& bindings, const std::string& prefix) const
{
    if (!impl_->recorded) { return; }
    const auto add = [&](const char* name, Texture* texture) {
        bindings.push_back({.id = prefix + name, .texture = texture, .state = ResourceState::ShaderRead});
    };
    add("radiance", impl_->radiance.texture.get());
    add("transmittance", impl_->transmittance->texture.get());
    add("multiScattering", impl_->multiScattering->texture.get());
    add("skyView", impl_->skyView.texture.get());
    add("cloudShadow", impl_->cloudShadow.texture.get());
    bindings.push_back({.id = prefix + "aerialPerspective", .buffer = impl_->aerial.get(), .state = ResourceState::ShaderRead, .layout = "float4"});
    bindings.push_back({.id = prefix + "parameters", .buffer = impl_->parameterBuffer.get(), .state = ResourceState::ShaderRead, .layout = "raw"});
}

TextureView* AtmosphereResourcesGPU::radianceView() const { return impl_->radiance.view.get(); }
TextureView* AtmosphereResourcesGPU::transmittanceView() const { return impl_->transmittance->view.get(); }
TextureView* AtmosphereResourcesGPU::multiScatteringView() const { return impl_->multiScattering->view.get(); }
TextureView* AtmosphereResourcesGPU::skyView() const { return impl_->skyView.view.get(); }
TextureView* AtmosphereResourcesGPU::cloudShadowView() const { return impl_->cloudShadow.view.get(); }
Buffer* AtmosphereResourcesGPU::aerialBuffer() const { return impl_->aerial.get(); }
Buffer* AtmosphereResourcesGPU::parametersBuffer() const { return impl_->parameterBuffer.get(); }
const GPUAtmosphereParameters& AtmosphereResourcesGPU::parameters() const { return impl_->parameters; }

} // namespace metallic::render
