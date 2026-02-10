#pragma once

#include <ml.h>
#include <cstdint>

struct Uniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4   lightColorIntensity; // xyz=color, w=intensity
    float4   frustumPlanes[6];
    float4   cameraPos; // object-space camera position
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t meshletBaseOffset;
    uint32_t instanceID;
};

struct ShadowUniforms {
    float4x4 invViewProj;
    float4   lightDir;       // world-space direction TO light
    uint32_t screenWidth;
    uint32_t screenHeight;
    float    normalBias;
    float    maxRayDistance;
    uint32_t reversedZ;
};

struct LightingUniforms {
    float4x4 mvp;
    float4x4 modelView;
    float4   lightDir;
    float4   lightColorIntensity;
    float4x4 invProj;
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t meshletCount;
    uint32_t materialCount;
    uint32_t textureCount;
    uint32_t instanceCount;
    uint32_t shadowEnabled;
    uint32_t pad2;
};

struct AtmosphereUniforms {
    float4x4 invViewProj;
    float4   cameraWorldPos;
    float4   sunDirection;
    float4   params; // x = exposure
    uint32_t screenWidth;
    uint32_t screenHeight;
    uint32_t pad0;
    uint32_t pad1;
};

struct SceneInstanceTransform {
    float4x4 mvp;
    float4x4 modelView;
};

inline void extractFrustumPlanes(const float4x4& mvp, float4* planes) {
    MvpToPlanes(ML_OGL ? STYLE_OGL : STYLE_D3D, mvp, planes);
}
