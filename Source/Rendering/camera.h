#pragma once

#include <ml.h>
#include <cmath>

enum class CameraMode { Orbit, FPS };

struct OrbitCamera {
    static constexpr float kPi = 3.14159265358979323846f;

    // Shared
    CameraMode mode = CameraMode::Orbit;
    float fovY = 45.0f * (kPi / 180.0f);
    float nearZ = 0.001f;
    float farZ  = 100.0f;

    // Orbit state
    float3 target = float3(0.f);
    float distance = 1.0f;
    float azimuth  = 0.0f;   // radians
    float elevation = 0.2f;  // radians

    // FPS state
    float3 fpsPosition = float3(0.f);
    float fpsYaw   = 0.0f;   // radians
    float fpsPitch  = 0.0f;  // radians
    float moveSpeed = 5.0f;
    float mouseSensitivity = 0.003f;

    void initFromBounds(const float bboxMin[3], const float bboxMax[3]) {
        target = float3(
            (bboxMin[0] + bboxMax[0]) * 0.5f,
            (bboxMin[1] + bboxMax[1]) * 0.5f,
            (bboxMin[2] + bboxMax[2]) * 0.5f
        );
        float dx = bboxMax[0] - bboxMin[0];
        float dy = bboxMax[1] - bboxMin[1];
        float dz = bboxMax[2] - bboxMin[2];
        float maxExtent = std::fmax(dx, std::fmax(dy, dz));
        distance = maxExtent * 2.5f;
        nearZ = maxExtent * 0.001f;
        farZ  = maxExtent * 10.0f;
        azimuth = 0.0f;
        elevation = 0.2f;
        moveSpeed = maxExtent * 0.5f;
    }

    void rotate(float dx, float dy) {
        azimuth += dx;
        elevation += dy;
        float limit = kPi * 0.5f - 0.01f;
        if (elevation > limit) elevation = limit;
        if (elevation < -limit) elevation = -limit;
    }

    void zoom(float delta) {
        distance *= (1.0f - delta * 0.1f);
        if (distance < 0.001f) distance = 0.001f;
    }

    // FPS mouse look
    void lookFPS(float dx, float dy) {
        fpsYaw += dx * mouseSensitivity;
        fpsPitch -= dy * mouseSensitivity;
        float limit = kPi * 0.5f - 0.01f;
        if (fpsPitch > limit) fpsPitch = limit;
        if (fpsPitch < -limit) fpsPitch = -limit;
    }

    // FPS translation in world space
    void moveFPS(const float3& delta) {
        fpsPosition = fpsPosition + delta;
    }

    // Switch from Orbit to FPS: place FPS camera at current orbit eye position
    void switchToFPS() {
        fpsPosition = worldPosition3();
        // Derive yaw/pitch from orbit angles
        // In orbit: eye is at (target + distance*cosE*sinA, target + distance*sinE, target + distance*cosE*cosA)
        // Forward = normalize(target - eye), which points from eye toward target
        float3 fwd = forwardDirection();
        fpsPitch = std::asin(fwd.y);
        fpsYaw = std::atan2(fwd.x, fwd.z);
        mode = CameraMode::FPS;
    }

    // Switch from FPS to Orbit: set orbit target to a point in front of the FPS camera
    void switchToOrbit() {
        float3 fwd = forwardDirection();
        target = fpsPosition + fwd * distance;
        // Recompute azimuth/elevation from fpsPosition relative to target
        float3 offset = fpsPosition - target; // eye - target
        float horizDist = std::sqrt(offset.x * offset.x + offset.z * offset.z);
        azimuth = std::atan2(offset.x, offset.z);
        elevation = std::atan2(offset.y, horizDist);
        // distance stays the same
        mode = CameraMode::Orbit;
    }

    // Eye position in world space for either mode
    float3 worldPosition3() const {
        if (mode == CameraMode::FPS) {
            return fpsPosition;
        }
        float cosA = std::cos(azimuth), sinA = std::sin(azimuth);
        float cosE = std::cos(elevation), sinE = std::sin(elevation);
        return float3(
            target.x + distance * cosE * sinA,
            target.y + distance * sinE,
            target.z + distance * cosE * cosA
        );
    }

    float4 worldPosition() const {
        float3 p = worldPosition3();
        return float4(p.x, p.y, p.z, 1.0f);
    }

    // Normalized forward direction for either mode
    float3 forwardDirection() const {
        if (mode == CameraMode::FPS) {
            float cosP = std::cos(fpsPitch), sinP = std::sin(fpsPitch);
            float cosY = std::cos(fpsYaw), sinY = std::sin(fpsYaw);
            return float3(cosP * sinY, sinP, cosP * cosY);
        }
        float3 eye = worldPosition3();
        return normalize(target - eye);
    }

    float4x4 viewMatrix() const {
        float3 eye = worldPosition3();
        float3 f = forwardDirection();
        float3 worldUp(0.f, 1.f, 0.f);
        float3 r = normalize(cross(f, worldUp));
        float3 u = cross(r, f);

        float4x4 m(
            float4( r.x,  u.x, -f.x, 0.f),
            float4( r.y,  u.y, -f.y, 0.f),
            float4( r.z,  u.z, -f.z, 0.f),
            float4(-dot(r, eye), -dot(u, eye), dot(f, eye), 1.f)
        );
        return m;
    }

    static float4x4 perspectiveMatrix(float fovY, float aspect, float nearZ, float farZ) {
        float4x4 m;
        uint32_t projFlags = 0;
#if ML_DEPTH_REVERSED
        projFlags |= PROJ_REVERSED_Z;
#endif
        m.SetupByHalfFovy(fovY * 0.5f, aspect, nearZ, farZ, projFlags);
        return m;
    }

    float4x4 projectionMatrix(float aspect) const {
        return perspectiveMatrix(fovY, aspect, nearZ, farZ);
    }

    // Halton low-discrepancy sequence for TAA jitter
    static float halton(int index, int base) {
        float f = 1.0f, result = 0.0f;
        for (int i = index; i > 0; i /= base) {
            f /= static_cast<float>(base);
            result += f * static_cast<float>(i % base);
        }
        return result;
    }

    // Returns jitter in [-0.5, 0.5] pixel range using Halton bases 2,3
    static float2 haltonJitter(uint32_t frameIndex) {
        // Use 1-based index (Halton(0) = 0)
        int idx = static_cast<int>((frameIndex % 16) + 1);
        return float2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
    }

    // Perspective projection with sub-pixel jitter applied in clip space
    static float4x4 jitteredProjectionMatrix(float fovY, float aspect, float nearZ, float farZ,
                                              float jitterX, float jitterY,
                                              uint32_t screenWidth, uint32_t screenHeight) {
        float4x4 proj = perspectiveMatrix(fovY, aspect, nearZ, farZ);
        // Apply sub-pixel offset to projection matrix column 2 (clip-space translation)
        proj[2].x += (2.0f * jitterX) / static_cast<float>(screenWidth);
        proj[2].y += (2.0f * jitterY) / static_cast<float>(screenHeight);
        return proj;
    }
};
