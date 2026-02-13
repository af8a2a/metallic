#pragma once

#include "render_uniforms.h"
#include <ml.h>
#include <Metal/Metal.hpp>
#include <vector>
#include <unordered_map>

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

    // Light data
    float4 worldLightDir;
    float4 viewLightDir;
    float4 lightColorIntensity;

    // Base uniforms (shared across passes)
    Uniforms baseUniforms;
    AtmosphereUniforms skyUniforms;
    LightingUniforms lightingUniforms;
    TonemapUniforms tonemapUniforms;

    // Visible nodes for rendering
    std::vector<uint32_t> visibleMeshletNodes;
    std::vector<uint32_t> visibleIndexNodes;
    uint32_t visibilityInstanceCount = 0;

    // Instance transform buffer (for visibility buffer mode)
    MTL::Buffer* instanceTransformBuffer = nullptr;

    // Command buffer (for ImGui pass)
    MTL::CommandBuffer* commandBuffer = nullptr;

    // Depth clear value
    double depthClearValue = 1.0;

    // Camera far plane (for shadow rays)
    float cameraFarZ = 1000.0f;

    // Feature flags
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableAtmosphereSky = true;
    int renderMode = 2; // 0=Vertex, 1=Mesh, 2=Visibility
};

// Runtime context for pipeline building (pipelines, textures, samplers)
struct PipelineRuntimeContext {
    MTL::Device* device = nullptr;

    // Pipeline states (keyed by pass type)
    std::unordered_map<std::string, MTL::RenderPipelineState*> renderPipelines;
    std::unordered_map<std::string, MTL::ComputePipelineState*> computePipelines;

    // Samplers
    std::unordered_map<std::string, MTL::SamplerState*> samplers;

    // Imported textures (atmosphere, fallbacks, etc.)
    std::unordered_map<std::string, MTL::Texture*> importedTextures;

    // Current frame's drawable
    MTL::Texture* backbuffer = nullptr;
};
