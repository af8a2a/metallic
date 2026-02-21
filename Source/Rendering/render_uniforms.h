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
    float4x4 prevViewProj;
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

struct TonemapUniforms {
    uint32_t isActive;
    uint32_t method;
    float exposure;
    float contrast;
    float brightness;
    float saturation;
    float vignette;
    uint32_t dither;
    float2 invResolution;
    uint32_t autoExposure;
    float pad;
};

struct AutoExposureUniforms {
    float evMinValue;
    float evMaxValue;
    float adaptationSpeed;
    float deltaTime;
    uint32_t screenWidth;
    uint32_t screenHeight;
    float lowPercentile;
    float highPercentile;
};

struct TAAUniforms {
    float2 jitterOffset;
    float2 invResolution;
    uint32_t screenWidth;
    uint32_t screenHeight;
    float blendMin;             // 0.05 — favor history
    float blendMax;             // 1.0 — full current on disocclusion
    float varianceClipGamma;    // 1.0
    uint32_t frameIndex;
    float motionWeightScale;    // motion rejection sensitivity
    float pad;
};

struct SceneInstanceTransform {
    float4x4 mvp;
    float4x4 modelView;
};

inline void extractFrustumPlanes(const float4x4& mvp, float4* planes) {
    MvpToPlanes(ML_OGL ? STYLE_OGL : STYLE_D3D, mvp, planes);
}
