#include "Runtime/Render/SceneLightResources.h"
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

Result SceneLightResources::buildSampling(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
    TextureView& environment, const ReGIRBuildParameters& parameters,
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
        Result result = next->compute.initialize(device, log);
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
    if (auto* frame = commands.frameContext()) { frame->retain(sampling_); }
    // PDF layout state is advanced while recording. If any later pass cancels
    // this recording, recreate it instead of assuming those GPU transitions ran.
    Result transaction = host.deferSubmission(commands, []() {},
        [state = sampling_]() { state->cancelled = true; });
    if (!transaction) { return transaction; }
    Result result = sampling_->compute.buildLocalLights(commands, environment, sampling_->pdf, *buffer_, lightCount());
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
        for (const auto& light : world->lighting().lights) {
            if (!light.imported) { settings.lights.push_back(light); }
        }
    }
    return settings;
}

std::vector<SceneLightRecord> buildSceneLightRecords(
    std::span<const scene::RenderLight> renderLights,
    std::span<const scene::PunctualLight> virtualLights)
{
    std::vector<SceneLightRecord> records(renderLights.size() + virtualLights.size());
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
        GpuPunctualLight light;
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
        light.directionType[3] = properties.type == "directional" ? 0.0f :
            (properties.type == "point" ? 1.0f : 2.0f);
        light.colorIntensity[0] = properties.color.x;
        light.colorIntensity[1] = properties.color.y;
        light.colorIntensity[2] = properties.color.z;
        light.colorIntensity[3] = static_cast<float>(intensity);
        light.spot[0] = static_cast<float>(std::cos(properties.innerConeAngle));
        light.spot[1] = static_cast<float>(std::cos(properties.outerConeAngle));
        record.gpu = light;
        record.enabled = true;
    };
    for (size_t index = 0; index < renderLights.size(); ++index) {
        const scene::RenderLight& light = renderLights[index];
        SceneLightRecord& record = records[index];
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
        SceneLightRecord& record = records[renderLights.size() + index];
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

std::vector<GpuPunctualLight> buildPunctualLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& settings)
{
    const auto sceneRecords = buildSceneLightRecords(scene != nullptr
        ? std::span<const scene::RenderLight>(scene->lights())
        : std::span<const scene::RenderLight>(), settings.lights);
    std::vector<GpuPunctualLight> records(1);
    records.reserve(sceneRecords.size() + 1);
    records[0].positionRange[1] = std::exp2(-settings.exposureEV100);
    for (const SceneLightRecord& record : sceneRecords) {
        if (record.enabled) { records.push_back(record.gpu); }
    }
    records[0].positionRange[0] = static_cast<float>(records.size() - 1);
    return records;
}

Result SceneLightResources::update(Device& device, CommandBuffer& commands,
    RenderSubsystemHost& host, const scene::Scene* scene, const scene::LightingSettings& settings)
{
    if (!scene::validLightingSettings(settings)) { return makeError(Error::InvalidArgument); }
    auto records = buildPunctualLightRecords(scene, settings);
    const uint64_t bytes = records.size() * sizeof(GpuPunctualLight);
    if (records.size() != records_.size() ||
        std::memcmp(records.data(), records_.data(), static_cast<size_t>(bytes)) != 0) {
        std::unique_ptr<Buffer> next;
        Result result = device.createBuffer(BufferDesc{
            .size = bytes, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
        }, next);
        if (!result) { return result; }
        if (next == nullptr) { return makeError(Error::Failure); }
        void* mapped = next->map();
        if (mapped == nullptr) { return makeError(Error::Failure); }
        std::memcpy(mapped, records.data(), static_cast<size_t>(bytes));
        next->flush(0, bytes);
        next->unmap();
        host.retire(buffer_);
        buffer_ = std::move(next);
        records_ = std::move(records);
        ++revision_;
    }
    if (auto* frame = commands.frameContext()) { frame->retain(buffer_); }
    commands.hostWriteBarrier();
    return {};
}

} // namespace metallic::render
