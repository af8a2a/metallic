#pragma once

#include "render_pass.h"
#include "render_uniforms.h"

class ShadowRayPass : public RenderPass {
public:
    ShadowRayPass(const RenderContext& ctx,
                  FGResource depthInput,
                  const float4x4& view, const float4x4& proj,
                  const float4& worldLightDir,
                  float cameraFarZ, bool shadowEnabled,
                  int w, int h)
        : m_ctx(ctx), m_depthInput(depthInput)
        , m_view(view), m_proj(proj)
        , m_worldLightDir(worldLightDir)
        , m_cameraFarZ(cameraFarZ), m_shadowEnabled(shadowEnabled)
        , m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return "Shadow Ray Pass"; }

    FGResource shadowMap;

    void setup(FGBuilder& builder) override {
        m_depthRead = builder.read(m_depthInput);
        shadowMap = builder.create("shadowMap",
            FGTextureDesc::storageTexture(m_width, m_height, MTL::PixelFormatR8Unorm));
    }

    void executeCompute(MTL::ComputeCommandEncoder* enc) override {
        enc->setComputePipelineState(m_ctx.shadowResources.pipeline);
        ShadowUniforms shadowUni;
        float4x4 viewProj = m_proj * m_view;
        float4x4 invViewProj = viewProj;
        invViewProj.Invert();
        shadowUni.invViewProj = invViewProj;
        shadowUni.lightDir = m_worldLightDir;
        shadowUni.screenWidth = (uint32_t)m_width;
        shadowUni.screenHeight = (uint32_t)m_height;
        shadowUni.normalBias = m_shadowEnabled ? 0.05f : 0.0f;
        shadowUni.maxRayDistance = m_shadowEnabled ? m_cameraFarZ : 0.0f;
        shadowUni.reversedZ = ML_DEPTH_REVERSED ? 1 : 0;
        enc->setBytes(&shadowUni, sizeof(shadowUni), 0);
        enc->setAccelerationStructure(m_ctx.shadowResources.tlas, 1);
        enc->setTexture(m_frameGraph->getTexture(m_depthRead), 0);
        enc->setTexture(m_frameGraph->getTexture(shadowMap), 1);
        enc->useResource(m_ctx.shadowResources.tlas, MTL::ResourceUsageRead);
        for (auto* blas : m_ctx.shadowResources.blasArray) {
            if (blas) enc->useResource(blas, MTL::ResourceUsageRead);
        }
        enc->useResource(m_ctx.sceneMesh.positionBuffer, MTL::ResourceUsageRead);
        enc->useResource(m_ctx.sceneMesh.indexBuffer, MTL::ResourceUsageRead);
        MTL::Size tgSize(8, 8, 1);
        MTL::Size grid((m_width + 7) / 8, (m_height + 7) / 8, 1);
        enc->dispatchThreadgroups(grid, tgSize);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_depthInput;
    FGResource m_depthRead;
    float4x4 m_view, m_proj;
    float4 m_worldLightDir;
    float m_cameraFarZ;
    bool m_shadowEnabled;
    int m_width, m_height;
};
