#pragma once

#include <ml.h>
#include "rhi_backend.h"

#include <string>
#include <vector>
#include <unordered_map>

class RhiTexture;
class RhiBuffer;
class RhiFrameGraphBackend;
class StreamlineContext;
#ifdef _WIN32
class VulkanImageLayoutTracker;
#endif

// Per-frame runtime context for data-driven pipeline execution
// This holds all the dynamic data that changes each frame
struct FrameContext {
    // Screen dimensions
    int width = 0;
    int height = 0;

    // Camera matrices
    float4x4 view;
    float4x4 proj;
    float4x4 unjitteredProj;
    float4 cameraWorldPos;
    float4 cameraRight;
    float4 cameraUp;
    float4 cameraForward;
    float cameraNearZ = 0.001f;
    float cameraFovY = 45.0f * (3.14159265358979323846f / 180.0f);

    // Previous frame camera matrices (for TAA motion vectors)
    float4x4 prevView;
    float4x4 prevProj;
    float4x4 prevCullView;
    float4x4 prevCullProj;
    float4 prevCameraWorldPos;

    // TAA jitter
    float2 jitterOffset = float2(0.f, 0.f); // pixel-space jitter [-0.5, 0.5]
    uint32_t frameIndex = 0;
    bool enableTAA = true;

    // DLSS / upscaling
    int displayWidth = 0;   // output (display) resolution
    int displayHeight = 0;
    int renderWidth = 0;    // internal render resolution (== width/height when DLSS off)
    int renderHeight = 0;
    bool historyReset = false; // set on resize, camera cut, pipeline reload, preset change

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

    // Active native command buffer for backend integrations that are not fully abstracted yet.
    const RhiNativeCommandBuffer* commandBuffer = nullptr;

#ifdef _WIN32
    // Vulkan image layout tracker used by backend integrations such as Streamline.
    VulkanImageLayoutTracker* imageLayoutTracker = nullptr;
#endif

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

    // Phase 2 occlusion culling: occlusion-failed meshlets from phase 1
    RhiBuffer* gpuOcclusionFailedBufferRhi = nullptr;
    uint32_t gpuOcclusionFailedCount = 0;

    // Mid-frame HZB textures (written by HZBBuildPass_Mid, consumed by phase 2 cull)
    static constexpr uint32_t kMaxHzbLevels = 10;
    const RhiTexture* hzbMipTextures[kMaxHzbLevels] = {};
    uint32_t hzbMipCount = 0;
    float2 hzbTextureSize = float2(0.f, 0.f);
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

    // Output/render sizing for passes that bridge different resolution domains.
    int displayWidth = 0;
    int displayHeight = 0;
    int renderWidth = 0;
    int renderHeight = 0;

    // Optional Streamline / DLSS state used by authored upscaler passes.
    StreamlineContext* streamlineContext = nullptr;
    bool dlssAvailable = false;
    bool dlssEnabled = false;

    // Resource creation for pass-owned persistent resources
    RhiFrameGraphBackend* resourceFactory = nullptr;
};
