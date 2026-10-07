#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Render/ImportanceSampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace metallic::render {

struct SceneLightResources::SamplingState {
    ImportancePdfCompute compute;
    ImportancePdfTexture pdf;
    ReGIRLightSelector grid;
    bool cancelled = false;
};

Result<> SceneLightResources::buildSampling(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
    const ReGIRBuildParameters& parameters,
    uint32_t gridSize, uint32_t lightsPerCell, bool buildGrid, std::string& log)
{
    if (buffer_ == nullptr || parameters.lightCount != lightCount()) {
        return makeError(Error::InvalidArgument);
    }
    const auto size = computeImportancePdfTextureSize(std::max(lightCount(), 1u));
    if (sampling_ == nullptr || sampling_->cancelled || sampling_->pdf.textureWidth() != size.width ||
        sampling_->pdf.textureHeight() != size.height || sampling_->grid.layout().gridSize != gridSize ||
        sampling_->grid.layout().lightsPerCell != lightsPerCell) {
        auto next = std::make_shared<SamplingState>();
        Result<> result = next->compute.initialize(device, log);
        if (!result) { return result; }
        result = next->grid.initialize(device, log);
        if (!result) { return result; }
        result = next->pdf.initialize(device, size.width, size.height, "Physical light importance PDF", log);
        if (!result) { return result; }
        result = next->grid.ensureGrid(device, gridSize, lightsPerCell, log);
        if (!result) { return result; }
        host.retire(sampling_);
        sampling_ = std::move(next);
    }
    if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(sampling_); }
    // PDF layout state is advanced while recording. If any later pass cancels
    // this recording, recreate it instead of assuming those GPU transitions ran.
    Result<> transaction = host.deferSubmission(commands, []() {},
        [state = sampling_]() { state->cancelled = true; }).transform([](auto) {});
    if (!transaction) { return transaction; }
    Result<> result = sampling_->compute.buildLocalLights(commands, sampling_->pdf, *buffer_, lightCount());
    if (!result || !buildGrid) { return result; }
    return sampling_->grid.build(commands, *sampling_->pdf.view(), *buffer_, parameters);
}

Buffer* SceneLightResources::reGIRBuffer() const
{
    return sampling_ != nullptr ? sampling_->grid.buffer() : nullptr;
}

TextureView* SceneLightResources::lightPdfView() const
{
    return sampling_ != nullptr ? sampling_->pdf.view() : nullptr;
}

scene::LightingSettings resolveSceneLighting(const scene::Scene* actualScene, const RenderWorld* world)
{
    if (world != nullptr && (actualScene == nullptr || actualScene == world->scene())) {
        return world->lighting();
    }
    scene::LightingSettings settings = actualScene != nullptr
        ? actualScene->authoredLighting() : scene::LightingSettings{};
    if (world != nullptr) {
        settings.exposureEV100 = world->lighting().exposureEV100;
        settings.autoExposure = world->lighting().autoExposure;
        for (const auto& light : world->lighting().lights) {
            if (!light.imported) { settings.lights.push_back(light); }
        }
    }
    return settings;
}

environment::EnvironmentSnapshot resolveWorldEnvironment(const scene::Scene* actualScene, const RenderWorld* world)
{
    if (world != nullptr && (actualScene == nullptr || actualScene == world->scene() ||
        (world->scene() == nullptr && world->hasWorldEnvironmentOverride()))) {
        return world->environmentSnapshot();
    }
    return actualScene != nullptr ? actualScene->environmentSnapshot() : environment::EnvironmentSnapshot{};
}

GPUCelestialLightRecords buildCelestialLightRecords(const environment::EnvironmentSnapshot& snapshot)
{
    GPUCelestialLightRecords records{};
    for (size_t index = 0; index < records.size(); ++index) {
        const auto& source = snapshot.celestial[index];
        const bool physical = snapshot.source == environment::EnvironmentSource::PhysicalAtmosphere;
        if (!source.enabled || !environment::validCelestialLight(source) || (!physical && source.illuminance <= 0.0f)) { continue; }
        const auto converted = color::fromLinearRec709({source.color.x, source.color.y, source.color.z});
        const double magnitude = std::sqrt(double(source.direction.x) * source.direction.x +
            double(source.direction.y) * source.direction.y + double(source.direction.z) * source.direction.z);
        GPUCelestialLight light;
        const auto topOfAtmosphere = physical ? atmosphereSpectrumToWorkingColor(
            {source.topOfAtmosphereIrradiance.x, source.topOfAtmosphereIrradiance.y, source.topOfAtmosphereIrradiance.z})
            : std::array<float, 3>{};
        for (size_t channel = 0; channel < 3; ++channel) {
            // Physical direct estimators apply position-dependent spectral
            // transmittance in shaders. The unattenuated value here supplies
            // selection/importance and keeps the fixed celestial wire ABI.
            light.irradiance[channel] = physical ? std::max(topOfAtmosphere[channel], 0.0f)
                : converted[channel] * source.illuminance;
        }
        if (!std::isfinite(light.irradiance[0]) || !std::isfinite(light.irradiance[1]) ||
            !std::isfinite(light.irradiance[2]) ||
            (light.irradiance[0] <= 0.0f && light.irradiance[1] <= 0.0f && light.irradiance[2] <= 0.0f)) { continue; }
        light.direction[0] = static_cast<float>(source.direction.x / magnitude);
        light.direction[1] = static_cast<float>(source.direction.y / magnitude);
        light.direction[2] = static_cast<float>(source.direction.z / magnitude);
        light.angularRadius = source.angularRadius;
        light.flags = kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow;
        // Integral of a uniform disk against the perpendicular receiver cosine.
        // A zero-radius source remains a delta light and has no finite disk radiance.
        if (source.angularRadius > 0.0f) {
            const double sine = std::sin(double(source.angularRadius));
            const double projectedSolidAngle = 3.14159265358979323846 * sine * sine;
            for (size_t channel = 0; channel < 3; ++channel) {
                light.diskRadiance[channel] = static_cast<float>(std::min(
                    double(light.irradiance[channel]) / projectedSolidAngle,
                    double(std::numeric_limits<float>::max())));
            }
        }
        light.shadowImportance = color::luminance({light.irradiance[0], light.irradiance[1], light.irradiance[2]});
        if (!std::isfinite(light.shadowImportance)) { light.shadowImportance = std::numeric_limits<float>::max(); }
        records[index] = light;
    }
    return records;
}

std::vector<SceneLightRecord> buildSceneLightRecords(
    std::span<const scene::RenderLight> renderLights,
    std::span<const scene::PunctualLight> virtualLights)
{
    std::vector<SceneLightRecord> records;
    records.reserve(renderLights.size() + virtualLights.size());
    // Native imported lights keep their source slots for stable GPUScene IDs,
    // but resolve placement against the current scene snapshot at collection
    // time. A RenderWorld copy of the authored light can outlive a scene update.
    std::unordered_multimap<scene::SceneEntity, const scene::RenderLight*> importedSources;
    importedSources.reserve(renderLights.size());
    auto populate = [](SceneLightRecord& record, const scene::LightProperties& properties,
                        const float3& position, const float3& direction) {
        if (!scene::validLightProperties(properties)) { return; }
        const double intensity = scene::lightIntensitySI(properties.type, properties.intensityUnit,
            properties.intensity, properties.innerConeAngle, properties.outerConeAngle);
        if (intensity <= 0.0 || static_cast<float>(intensity) == 0.0f ||
            (properties.color.x == 0.0f && properties.color.y == 0.0f && properties.color.z == 0.0f) ||
            !std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
            std::abs(properties.range) > std::numeric_limits<float>::max()) {
            return;
        }
        const float range = static_cast<float>(properties.range);
        // Do not turn a positive finite influence radius into an unbounded light
        // through float underflow (range zero means infinite influence).
        if (properties.range > 0.0 && range == 0.0f) { return; }
        GPUPunctualLight light;
        light.positionRange[0] = position.x;
        light.positionRange[1] = position.y;
        light.positionRange[2] = position.z;
        light.positionRange[3] = range;
        const double magnitude = std::sqrt(double(direction.x) * direction.x +
            double(direction.y) * direction.y + double(direction.z) * direction.z);
        if (!std::isfinite(magnitude) || magnitude < 1e-6f) { return; }
        const float3 normalized(static_cast<float>(direction.x / magnitude),
            static_cast<float>(direction.y / magnitude), static_cast<float>(direction.z / magnitude));
        light.directionType[0] = normalized.x;
        light.directionType[1] = normalized.y;
        light.directionType[2] = normalized.z;
        light.directionType[3] = properties.type == "point" ? 1.0f : 2.0f;
        const auto color = color::fromLinearRec709({properties.color.x, properties.color.y, properties.color.z});
        light.colorIntensity[0] = color[0];
        light.colorIntensity[1] = color[1];
        light.colorIntensity[2] = color[2];
        light.colorIntensity[3] = static_cast<float>(intensity);
        light.spot[0] = static_cast<float>(std::cos(properties.innerConeAngle));
        light.spot[1] = static_cast<float>(std::cos(properties.outerConeAngle));
        record.gpu = light;
        record.enabled = true;
    };
    for (size_t index = 0; index < renderLights.size(); ++index) {
        const scene::RenderLight& light = renderLights[index];
        if (light.type != "point" && light.type != "spot") { continue; }
        SceneLightRecord& record = records.emplace_back();
        record.sourceRenderLightIndex = static_cast<int32_t>(index);
        record.sourceObject = light.object;
        if (light.virtualLightSceneIdentity != 0) {
            importedSources.emplace(light.object, &light);
        } else if (light.visible) {
            const auto& m = light.worldMatrix;
            populate(record, scene::LightProperties{
                .type = light.type, .color = light.color, .intensity = light.intensity,
                .range = light.range, .innerConeAngle = light.innerConeAngle,
                .outerConeAngle = light.outerConeAngle, .intensityUnit = light.intensityUnit,
            }, float3(m.a03, m.a13, m.a23), light.type == "point"
                ? float3(0.0f, -1.0f, 0.0f) : float3(-m.a02, -m.a12, -m.a22));
        }
    }
    for (size_t index = 0; index < virtualLights.size(); ++index) {
        const scene::PunctualLight& light = virtualLights[index];
        if (light.properties.type != "point" && light.properties.type != "spot") { continue; }
        SceneLightRecord& record = records.emplace_back();
        record.sourceVirtualLightIndex = static_cast<int32_t>(index);
        if (light.imported) {
            const scene::ImportedLightBinding& binding = *light.imported;
            if (binding.sceneIdentity == 0 || binding.object == scene::kNullSceneEntity) { continue; }
            const scene::RenderLight* source = nullptr;
            const auto [first, last] = importedSources.equal_range(binding.object);
            for (auto candidate = first; candidate != last; ++candidate) {
                if (candidate->second->virtualLightSceneIdentity == binding.sceneIdentity) {
                    source = candidate->second;
                    break;
                }
            }
            // Never reinterpret an unresolved or cross-scene import as a free
            // light, nor fall back to the suppressed legacy source emission.
            if (source == nullptr) { continue; }
            record.sourceObject = source->object;
            if (!light.enabled || !source->visible) { continue; }
            const auto& m = source->worldMatrix;
            const float3& p = binding.localPosition;
            const float3& d = binding.localDirection;
            populate(record, light.properties,
                float3(m.a00 * p.x + m.a01 * p.y + m.a02 * p.z + m.a03,
                    m.a10 * p.x + m.a11 * p.y + m.a12 * p.z + m.a13,
                    m.a20 * p.x + m.a21 * p.y + m.a22 * p.z + m.a23),
                light.properties.type == "point" ? float3(0.0f, -1.0f, 0.0f)
                    : float3(m.a00 * d.x + m.a01 * d.y + m.a02 * d.z,
                    m.a10 * d.x + m.a11 * d.y + m.a12 * d.z,
                    m.a20 * d.x + m.a21 * d.y + m.a22 * d.z));
        } else if (light.enabled) {
            populate(record, light.properties, light.position, light.properties.type == "point"
                ? float3(0.0f, -1.0f, 0.0f) : light.direction);
        }
    }
    return records;
}

std::vector<GPUPunctualLight> buildPunctualLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& settings)
{
    const auto sceneRecords = buildSceneLightRecords(scene != nullptr
        ? std::span<const scene::RenderLight>(scene->lights())
        : std::span<const scene::RenderLight>(), settings.lights);
    std::vector<GPUPunctualLight> records(1);
    records.reserve(sceneRecords.size() + 1);
    records[0].positionRange[1] = std::exp2(-settings.exposureEV100);
    for (const SceneLightRecord& record : sceneRecords) {
        if (record.enabled) { records.push_back(record.gpu); }
    }
    records[0].positionRange[0] = static_cast<float>(records.size() - 1);
    return records;
}

Result<> SceneLightResources::update(Device& device, CommandBuffer& commands,
    RenderSubsystemHost& host, const scene::Scene* scene, const scene::LightingSettings& settings)
{
    if (!scene::validLightingSettings(settings)) { return makeError(Error::InvalidArgument); }
    auto records = buildPunctualLightRecords(scene, settings);
    const uint64_t bytes = records.size() * sizeof(GPUPunctualLight);
    if (records.size() != records_.size() ||
        std::memcmp(records.data(), records_.data(), static_cast<size_t>(bytes)) != 0) {
        std::unique_ptr<Buffer> next;
        Result<> result = device.createBuffer(BufferDesc{
            .size = bytes, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
        }).transform([&](auto rhiValue) { next = std::move(rhiValue); });
        if (!result) { return result; }
        if (next == nullptr) { return makeError(Error::Failure); }
        void* mapped = next->map();
        if (mapped == nullptr) { return makeError(Error::Failure); }
        std::memcpy(mapped, records.data(), static_cast<size_t>(bytes));
        next->flush({0, bytes});
        next->unmap();
        host.retire(buffer_);
        buffer_ = std::move(next);
        records_ = std::move(records);
        ++revision_;
    }
    if (auto* frame = metallic::render::RenderFrameContext::from(commands)) { frame->retain(buffer_); }
    commands.hostWriteBarrier();
    return {};
}

} // namespace metallic::render
