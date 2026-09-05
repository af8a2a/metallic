#include "Runtime/Render/SceneLightResources.h"

#include <cstring>

namespace metallic::render {

std::vector<GpuPunctualLight> buildPunctualLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& settings)
{
    std::vector<GpuPunctualLight> records(1);
    records[0].positionRange[1] = std::exp2(-settings.exposureEV100);
    auto append = [&](const scene::LightProperties& properties,
                      const float3& position, const float3& direction) {
        if (!scene::validLightProperties(properties)) { return; }
        const double intensity = scene::lightIntensitySI(properties.type, properties.intensityUnit,
            properties.intensity, properties.innerConeAngle, properties.outerConeAngle);
        if (intensity == 0.0) { return; }
        GpuPunctualLight light;
        light.positionRange[0] = position.x;
        light.positionRange[1] = position.y;
        light.positionRange[2] = position.z;
        light.positionRange[3] = static_cast<float>(properties.range);
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
        records.push_back(light);
    };
    if (scene != nullptr) {
        for (const scene::RenderLight& light : scene->lights()) {
            if (!light.visible) { continue; }
            const auto& m = light.worldMatrix;
            append(scene::LightProperties{
                .type = light.type, .color = light.color, .intensity = light.intensity,
                .range = light.range, .innerConeAngle = light.innerConeAngle,
                .outerConeAngle = light.outerConeAngle, .intensityUnit = light.intensityUnit,
            }, float3(m.a03, m.a13, m.a23), light.type == "point"
                ? float3(0.0f, -1.0f, 0.0f) : float3(-m.a02, -m.a12, -m.a22));
        }
    }
    for (const scene::PunctualLight& light : settings.lights) {
        if (light.enabled) {
            append(light.properties, light.position, light.properties.type == "point"
                ? float3(0.0f, -1.0f, 0.0f) : light.direction);
        }
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
