#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class ShadowRayPass : public RenderPass {
public:
    ShadowRayPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("normalBias")) {
            m_normalBias = config.config["normalBias"].get<float>();
        }
        if (config.config.contains("maxRayDistance")) {
            m_maxRayDistance = config.config["maxRayDistance"].get<float>();
        }
    }

    FGResource shadowMap;

    FGResource getOutput(const std::string& name) const override {
        if (name == "shadowMap") return shadowMap;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Read depth from input
        FGResource depthInput = getInput("depth");
        if (depthInput.isValid()) {
            m_depthRead = builder.read(depthInput);
        }
        shadowMap = builder.create("shadowMap",
            FGTextureDesc::storageTexture(m_width, m_height, MTL::PixelFormatR8Unorm));
    }

    void executeCompute(MTL::ComputeCommandEncoder* enc) override {
        ZoneScopedN("ShadowRayPass");
        if (!m_frameContext) return;
        if (!m_frameContext->enableRTShadows) return;

        enc->setComputePipelineState(m_ctx.shadowResources.pipeline);
        ShadowUniforms shadowUni;
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        float4x4 invViewProj = viewProj;
        invViewProj.Invert();
        shadowUni.invViewProj = invViewProj;
        shadowUni.lightDir = m_frameContext->worldLightDir;
        shadowUni.screenWidth = (uint32_t)m_width;
        shadowUni.screenHeight = (uint32_t)m_height;
        shadowUni.normalBias = m_normalBias;
        shadowUni.maxRayDistance = m_maxRayDistance > 0 ? m_maxRayDistance : m_frameContext->cameraFarZ;
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

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        if (m_frameContext) {
            ImGui::Text("Enabled: %s", m_frameContext->enableRTShadows ? "Yes" : "No");
        }
        ImGui::SliderFloat("Normal Bias", &m_normalBias, 0.0f, 0.5f, "%.3f");
        ImGui::SliderFloat("Max Ray Distance", &m_maxRayDistance, 0.0f, 2000.0f, "%.1f");
    }

private:
    const RenderContext& m_ctx;
    FGResource m_depthRead;
    int m_width, m_height;
    std::string m_name = "Shadow Ray Pass";
    float m_normalBias = 0.05f;
    float m_maxRayDistance = 1000.0f;
};
