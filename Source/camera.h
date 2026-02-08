#pragma once

#include <simd/simd.h>
#include <cmath>

struct OrbitCamera {
    simd_float3 target = {0.f, 0.f, 0.f};
    float distance = 1.0f;
    float azimuth  = 0.0f;   // radians
    float elevation = 0.2f;  // radians
    float fovY = 45.0f * (M_PI / 180.0f);
    float nearZ = 0.001f;
    float farZ  = 100.0f;

    void initForBunny(const float bboxMin[3], const float bboxMax[3]) {
        target = {
            (bboxMin[0] + bboxMax[0]) * 0.5f,
            (bboxMin[1] + bboxMax[1]) * 0.5f,
            (bboxMin[2] + bboxMax[2]) * 0.5f
        };
        float dx = bboxMax[0] - bboxMin[0];
        float dy = bboxMax[1] - bboxMin[1];
        float dz = bboxMax[2] - bboxMin[2];
        float maxExtent = std::fmax(dx, std::fmax(dy, dz));
        distance = maxExtent * 2.5f;
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

    simd_float4x4 viewMatrix() const {
        float cosA = std::cos(azimuth), sinA = std::sin(azimuth);
        float cosE = std::cos(elevation), sinE = std::sin(elevation);

        simd_float3 eye = {
            target.x + distance * cosE * sinA,
            target.y + distance * sinE,
            target.z + distance * cosE * cosA
        };

        // Right-handed look-at
        simd_float3 f = simd_normalize(target - eye);
        simd_float3 worldUp = {0.f, 1.f, 0.f};
        simd_float3 r = simd_normalize(simd_cross(f, worldUp));
        simd_float3 u = simd_cross(r, f);

        simd_float4x4 m = {{
            {  r.x,  u.x, -f.x, 0.f },
            {  r.y,  u.y, -f.y, 0.f },
            {  r.z,  u.z, -f.z, 0.f },
            { -simd_dot(r, eye), -simd_dot(u, eye), simd_dot(f, eye), 1.f }
        }};
        return m;
    }

    static simd_float4x4 perspectiveMatrix(float fovY, float aspect, float nearZ, float farZ) {
        // Metal NDC: x,y in [-1,1], z in [0,1]
        float ys = 1.0f / std::tan(fovY * 0.5f);
        float xs = ys / aspect;
        float zs = farZ / (nearZ - farZ);

        simd_float4x4 m = {{
            { xs,  0.f, 0.f,          0.f },
            { 0.f, ys,  0.f,          0.f },
            { 0.f, 0.f, zs,          -1.f },
            { 0.f, 0.f, zs * nearZ,   0.f }
        }};
        return m;
    }

    simd_float4x4 projectionMatrix(float aspect) const {
        return perspectiveMatrix(fovY, aspect, nearZ, farZ);
    }
};
