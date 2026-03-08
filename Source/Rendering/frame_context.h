#pragma once

#include <ml.h>
#include "rhi_backend.h"

#include <string>
#include <vector>
#include <unordered_map>

class RhiTexture;
class RhiBuffer;
class RhiFrameGraphBackend;

// Per-frame runtime context for data-driven pipeline execution
// This holds all the dynamic data that changes each frame
struct FrameContext {
    // Screen dimensions
    int width = 0;
    int height = 0;

    // Camera matrices
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;

    // Previous frame camera matrices (for TAA motion vectors)
    float4x4 prevView;
    float4x4 prevProj;

    // TAA jitter
    float2 jitterOffset = float2(0.f, 0.f); // pixel-space jitter [-0.5, 0.5]
    uint32_t frameIndex = 0;
    bool enableTAA = true;

    // Light data
    float4 worldLightDir;
    float4 viewLightDir;
    float4 lightColorIntensity;

    // Scene data for deferred lighting
    uint32_t meshletCount = 0;
    uint32_t materialCount = 0;
    uint32_t textureCount = 0;

    // Visible nodes for rendering
    std::vector<uint32_t> visibleMeshletNodes;
    std::vector<uint32_t> visibleIndexNodes;
    uint32_t visibilityInstanceCount = 0;

    // Instance transform buffer (for visibility buffer mode)
    RhiBuffer* instanceTransformBufferRhi = nullptr;

    // Command buffer (for ImGui pass)
    void* commandBufferHandle = nullptr;

    // Depth clear value
    double depthClearValue = 1.0;

    // Camera far plane (for shadow rays)
    float cameraFarZ = 1000.0f;

    // Frame timing
    float deltaTime = 0.016f;

    // Feature flags
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableAtmosphereSky = true;
    bool gpuDrivenCulling = false;
    int renderMode = 2; // 0=Vertex, 1=Mesh, 2=Visibility, 3=Meshlet Debug

    // GPU-driven cull results (set by MeshletCullPass, consumed by VisibilityPass)
    RhiBuffer* gpuVisibleMeshletBufferRhi = nullptr;
    RhiBuffer* gpuCounterBufferRhi = nullptr;
    RhiBuffer* gpuInstanceDataBufferRhi = nullptr;
};

// Runtime context for pipeline building (pipelines, textures, samplers)
struct PipelineRuntimeContext {
    // Pipeline states (keyed by pass type)
    std::unordered_map<std::string, RhiGraphicsPipelineHandle> renderPipelinesRhi;
    std::unordered_map<std::string, RhiComputePipelineHandle> computePipelinesRhi;

    // Samplers
    std::unordered_map<std::string, RhiSamplerHandle> samplersRhi;

    // Imported textures (atmosphere, fallbacks, etc.)
    std::unordered_map<std::string, RhiTextureHandle> importedTexturesRhi;

    // Current frame's drawable
    RhiTexture* backbufferRhi = nullptr;

    // Resource creation for pass-owned persistent resources
    RhiFrameGraphBackend* resourceFactory = nullptr;
};
