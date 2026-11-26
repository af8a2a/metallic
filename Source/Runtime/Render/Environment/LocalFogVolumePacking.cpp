#include "Runtime/Render/Environment/LocalFogVolumePacking.h"
#include "Runtime/Render/Core/ColorSpace.h"

#include <bit>
#include <numbers>
#include <tuple>
#include "Runtime/Environment/VolumetricFog.h"

namespace metallic::render {
namespace {

uint32_t packHalf2(float x, float y)
{
    return f32tof16(x) | (f32tof16(y) << 16);
}

uint32_t packFloat111110(float x, float y, float z)
{
    return ((f32tof16(x) << 17) & 0xffe00000) |
        ((f32tof16(y) << 6) & 0x001ffc00) | ((f32tof16(z) >> 5) & 0x000003ff);
}

PackedLocalFogVolume pack(const environment::LocalFogVolume& v, const float3& translatedCm)
{
    constexpr float toRadians = std::numbers::pi_v<float> / 180.0f;
    float4x4 x, y, z;
    x.SetupByRotationX(v.rotationDegrees[0] * toRadians);
    y.SetupByRotationY(v.rotationDegrees[1] * toRadians);
    z.SetupByRotationZ(v.rotationDegrees[2] * toRadians);
    const auto rotation = z * y * x;
    // Convert the two inverse-transform rows from Metallic (x,y,z) to the
    // source's Z-up packing basis (x,-z,y). Its third row is reconstructed on GPU.
    const auto rowX = normalize(float3(rotation[0].x, -rotation[2].x, rotation[1].x));
    const auto rowY = normalize(float3(-rotation[0].z, rotation[2].z, -rotation[1].z));
    const auto emissive = color::fromLinearRec709(v.emissive);
    const auto albedo = color::fromLinearRec709(v.albedo);
    const auto unorm = [](float x) { return uint32_t(std::clamp(x * 255.0f, 0.0f, 255.0f)); };
    return {{
        {std::bit_cast<uint32_t>(translatedCm.x), std::bit_cast<uint32_t>(translatedCm.y),
            std::bit_cast<uint32_t>(translatedCm.z), std::bit_cast<uint32_t>(v.radiusMeters * 100.0f)},
        {packHalf2(rowX.x, rowX.y), packHalf2(rowX.z, rowY.x), packHalf2(rowY.y, rowY.z), 0},
        {packFloat111110(v.radialExtinction, v.heightExtinction, std::max(v.heightFalloff, 1.0f) * 0.01f),
            packFloat111110(emissive[0], emissive[1], emissive[2]),
            unorm(albedo[0]) | (unorm(albedo[1]) << 8) | (unorm(albedo[2]) << 16) | (unorm(0.2f) << 24),
            std::bit_cast<uint32_t>(v.heightOffset)}
    }};
}

} // namespace

std::vector<PackedLocalFogVolume> packLocalFogVolumes(std::span<const environment::LocalFogVolume> volumes,
    const ViewCameraConstants& camera)
{
    // UE GetLocalFogVolumeViewSortingData / CreateViewLocalFogVolumeBufferSRV.
    struct Candidate {
        int priority;
        float distance;
        size_t index;
        float3 position;
    };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < volumes.size(); ++i) {
        const auto& v = volumes[i];
        if (!v.enabled || (v.radialExtinction <= 0 && v.heightExtinction <= 0)) { continue; }
        const double3 offset = (double3(v.centerMeters[0], v.centerMeters[1], v.centerMeters[2]) -
            double3(camera.eye[0], camera.eye[1], camera.eye[2])) * 100.0;
        candidates.push_back({127 - v.sortPriority, float(length(offset)), i,
            float3(float(offset.x), float(offset.y), float(offset.z))});
    }
    std::ranges::sort(candidates, [](const Candidate& a, const Candidate& b) {
        return std::tie(a.priority, a.distance, a.index) > std::tie(b.priority, b.distance, b.index);
    });
    const auto count = std::min<size_t>(candidates.size(), kLocalFogMaxInstances);
    std::vector<PackedLocalFogVolume> packed;
    packed.reserve(count);
    for (const auto& c : std::span(candidates).last(count)) { packed.push_back(pack(volumes[c.index], c.position)); }
    return packed;
}

} // namespace metallic::render
