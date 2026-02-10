#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "imgui.h"

class DeferredLightingPass : public RenderPass {
public:
    DeferredLightingPass(const RenderContext& ctx,
                         MTL::ComputePipelineState* pipeline,
                         FGResource visibilityInput,
                         FGResource depthInput,
                         FGResource shadowMapInput,
                         FGResource skyInput,
                         MTL::Texture* skyFallback,
                         MTL::Buffer* instanceTransformBuffer,
                         const LightingUniforms& lightUniforms,
                         int w, int h)
        : m_ctx(ctx), m_pipeline(pipeline)
        , m_visibilityInput(visibilityInput), m_depthInput(depthInput)
        , m_shadowMapInput(shadowMapInput)
        , m_skyInput(skyInput)
        , m_skyFallback(skyFallback)
        , m_instanceTransformBuffer(instanceTransformBuffer)
        , m_lightUniforms(lightUniforms)
        , m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return "Deferred Lighting"; }

    FGResource output;

    void setup(FGBuilder& builder) override {
        m_visRead = builder.read(m_visibilityInput);
        m_depthRead = builder.read(m_depthInput);
        if (m_shadowMapInput.isValid()) {
            m_shadowRead = builder.read(m_shadowMapInput);
        }
        if (m_skyInput.isValid()) {
            m_skyRead = builder.read(m_skyInput);
        }
        output = builder.create("output",
            FGTextureDesc::storageTexture(m_width, m_height, MTL::PixelFormatRGBA16Float));
    }

    void executeCompute(MTL::ComputeCommandEncoder* enc) override {
        ZoneScopedN("DeferredLightingPass");
        enc->setComputePipelineState(m_pipeline);
        enc->setBytes(&m_lightUniforms, sizeof(m_lightUniforms), 0);
        enc->setBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
        enc->setBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
        enc->setBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
        enc->setBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
        enc->setBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
        enc->setBuffer(m_ctx.sceneMesh.uvBuffer, 0, 6);
        enc->setBuffer(m_ctx.meshletData.materialIDs, 0, 7);
        enc->setBuffer(m_ctx.materials.materialBuffer, 0, 8);
        enc->setBuffer(m_instanceTransformBuffer, 0, 9);
        enc->setTexture(m_frameGraph->getTexture(m_visRead), 0);
        enc->setTexture(m_frameGraph->getTexture(m_depthRead), 1);
        enc->setTexture(m_frameGraph->getTexture(output), 2);
        if (!m_ctx.materials.textures.empty()) {
            enc->setTextures(
                const_cast<MTL::Texture* const*>(m_ctx.materials.textures.data()),
                NS::Range(3, m_ctx.materials.textures.size()));
        }
        MTL::Texture* shadowTex = m_shadowRead.isValid()
            ? m_frameGraph->getTexture(m_shadowRead)
            : m_ctx.shadowDummyTex;
        enc->setTexture(shadowTex, 99);
        MTL::Texture* skyTex = m_skyRead.isValid()
            ? m_frameGraph->getTexture(m_skyRead)
            : m_skyFallback;
        enc->setTexture(skyTex, 100);
        enc->setSamplerState(m_ctx.materials.sampler, 0);
        MTL::Size tgSize(8, 8, 1);
        MTL::Size grid((m_width + 7) / 8, (m_height + 7) / 8, 1);
        enc->dispatchThreadgroups(grid, tgSize);
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Instances: %u", m_lightUniforms.instanceCount);
        ImGui::Text("Meshlets: %u", m_lightUniforms.meshletCount);
        ImGui::Text("Materials: %u", m_lightUniforms.materialCount);
        ImGui::Text("Textures: %u", m_lightUniforms.textureCount);
        ImGui::Text("Shadows: %s", m_lightUniforms.shadowEnabled ? "Enabled" : "Disabled");
        ImGui::Text("Sky: %s", m_skyInput.isValid() ? "Atmosphere" : "Fallback");
    }

private:
    const RenderContext& m_ctx;
    MTL::ComputePipelineState* m_pipeline;
    FGResource m_visibilityInput, m_depthInput, m_shadowMapInput, m_skyInput;
    FGResource m_visRead, m_depthRead, m_shadowRead, m_skyRead;
    MTL::Texture* m_skyFallback;
    MTL::Buffer* m_instanceTransformBuffer;
    LightingUniforms m_lightUniforms;
    int m_width, m_height;
};
