#pragma once

#include "ml.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::render {

// Matches the camera basis and geometric clip bounds in GPUDrivenCullingCommon
// and GPUDrivenStreamAsset. Planes face inward: dot(plane.xyz, point) + plane.w >= 0.
// A positive orthoHeight selects orthographic projection. Invalid inputs disable
// coarse light culling, preserving conservative visibility.
inline std::array<float4, 6> gpuSceneLightFrustumPlanes(
    const float3& eye,
    const float3& center,
    const float3& cameraUp,
    float aspect,
    float fovRadians,
    float zNear,
    float zFar,
    float orthoHeight = 0.0f)
{
    const auto finiteVector = [](const float3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    };
    if (!finiteVector(eye) || !finiteVector(center) || !finiteVector(cameraUp) ||
        !std::isfinite(aspect) || !std::isfinite(fovRadians) ||
        !std::isfinite(zNear) || !std::isfinite(zFar) ||
        !std::isfinite(orthoHeight) || orthoHeight < 0.0f) {
        return {};
    }
    const auto safeNormalize = [](const float3& value, const float3& fallback) {
        const float lengthSquared = dot(value, value);
        return lengthSquared > 0.00000001f
            ? value * (1.0f / std::sqrt(lengthSquared)) : fallback;
    };
    const float3 forward = safeNormalize(center - eye, float3(0.0f, 0.0f, -1.0f));
    const float3 sourceUp = safeNormalize(cameraUp, float3(0.0f, 1.0f, 0.0f));
    const float3 right = safeNormalize(cross(forward, sourceUp), float3(1.0f, 0.0f, 0.0f));
    const float3 up = safeNormalize(cross(right, forward), float3(0.0f, 1.0f, 0.0f));
    if (!finiteVector(forward) || !finiteVector(right) || !finiteVector(up) ||
        dot(forward, forward) == 0.0f || dot(right, right) == 0.0f || dot(up, up) == 0.0f) {
        return {};
    }
    aspect = std::max(aspect, 0.001f);
    zNear = std::max(zNear, 0.0001f);
    zFar = std::max(zFar, zNear + 0.0001f);
    const auto plane = [&](const float3& normal, float offset) {
        const double distance = double(normal.x) * eye.x +
            double(normal.y) * eye.y + double(normal.z) * eye.z;
        return float4(normal.x, normal.y, normal.z, static_cast<float>(double(offset) - distance));
    };
    std::array<float4, 6> planes;
    if (orthoHeight > 0.0f) {
        const float halfHeight = std::max(orthoHeight * 0.5f, 0.0001f);
        const float halfWidth = halfHeight * aspect;
        planes[0] = plane(right, halfWidth);
        planes[1] = plane(-right, halfWidth);
        planes[2] = plane(up, halfHeight);
        planes[3] = plane(-up, halfHeight);
    } else {
        const float tanY = std::tan(std::clamp(fovRadians, 0.017453292f, 3.12413936f) * 0.5f);
        const float tanX = tanY * aspect;
        planes[0] = plane(right + forward * tanX, 0.0f);
        planes[1] = plane(-right + forward * tanX, 0.0f);
        planes[2] = plane(up + forward * tanY, 0.0f);
        planes[3] = plane(-up + forward * tanY, 0.0f);
    }
    // These are the geometric near/far planes of Vulkan's [0,w] clip interval.
    // Reversed Z swaps depth values, not these world-space visibility bounds.
    planes[4] = plane(forward, -zNear);
    planes[5] = plane(-forward, zFar);
    for (const float4& value : planes) {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
            !std::isfinite(value.z) || !std::isfinite(value.w)) {
            return {};
        }
    }
    return planes;
}

} // namespace metallic::render
