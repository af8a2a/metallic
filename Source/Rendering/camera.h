#pragma once

#include <ml.h>
#include <cmath>

struct OrbitCamera {
    float3 target = float3(0.f);
    float distance = 1.0f;
    float azimuth  = 0.0f;   // radians
    float elevation = 0.2f;  // radians
    float fovY = 45.0f * (M_PI / 180.0f);
    float nearZ = 0.001f;
    float farZ  = 100.0f;

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
    }

    void rotate(float dx, float dy) {
        azimuth += dx;
        elevation += dy;
        // Clamp elevation to avoid gimbal lock
        float limit = (float)(M_PI / 2.0 - 0.01);
        if (elevation > limit) elevation = limit;
        if (elevation < -limit) elevation = -limit;
    }

    void zoom(float delta) {
        distance *= (1.0f - delta * 0.1f);
        if (distance < 0.001f) distance = 0.001f;
    }

    float4x4 viewMatrix() const {
        float cosA = std::cos(azimuth), sinA = std::sin(azimuth);
        float cosE = std::cos(elevation), sinE = std::sin(elevation);

        float3 eye(
            target.x + distance * cosE * sinA,
            target.y + distance * sinE,
            target.z + distance * cosE * cosA
        );

        // Right-handed look-at
        float3 f = normalize(target - eye);
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
};
