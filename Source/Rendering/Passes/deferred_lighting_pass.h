#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class DeferredLightingPass : public RenderPass {
public:
    DeferredLightingPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource output;

    FGResource getOutput(const std::string& name) const override {
        if (name == "lightingOutput") return output;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Read inputs
        FGResource visInput = getInput("visibility");
        FGResource depthInput = getInput("depth");
        FGResource shadowInput = getInput("shadowMap");
        FGResource skyInput = getInput("skyOutput");

        if (visInput.isValid()) m_visRead = builder.read(visInput);
        if (depthInput.isValid()) m_depthRead = builder.read(depthInput);
        if (shadowInput.isValid()) m_shadowRead = builder.read(shadowInput);
        if (skyInput.isValid()) m_skyRead = builder.read(skyInput);

        output = builder.create("output",
            FGTextureDesc::storageTexture(m_width, m_height, MTL::PixelFormatRGBA16Float));
    }

    void executeCompute(MTL::ComputeCommandEncoder* enc) override {
        ZoneScopedN("DeferredLightingPass");
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->computePipelines.find("DeferredLightingPass");
        if (pipeIt == m_runtimeContext->computePipelines.end()) return;

        LightingUniforms lightUniforms = m_frameContext->lightingUniforms;

        enc->setComputePipelineState(pipeIt->second);
        enc->setBytes(&lightUniforms, sizeof(lightUniforms), 0);
        enc->setBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
        enc->setBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
        enc->setBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
        enc->setBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
        enc->setBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
        enc->setBuffer(m_ctx.sceneMesh.uvBuffer, 0, 6);
        enc->setBuffer(m_ctx.meshletData.materialIDs, 0, 7);
        enc->setBuffer(m_ctx.materials.materialBuffer, 0, 8);
        if (m_frameContext->instanceTransformBuffer) {
            enc->setBuffer(m_frameContext->instanceTransformBuffer, 0, 9);
        }
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
            : m_ctx.skyFallbackTex;
        enc->setTexture(skyTex, 100);
        enc->setSamplerState(m_ctx.materials.sampler, 0);
        MTL::Size tgSize(8, 8, 1);
        MTL::Size grid((m_width + 7) / 8, (m_height + 7) / 8, 1);
        enc->dispatchThreadgroups(grid, tgSize);
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        if (m_frameContext) {
            ImGui::Text("Instances: %u", m_frameContext->lightingUniforms.instanceCount);
            ImGui::Text("Meshlets: %u", m_frameContext->lightingUniforms.meshletCount);
            ImGui::Text("Materials: %u", m_frameContext->lightingUniforms.materialCount);
            ImGui::Text("Shadows: %s", m_frameContext->lightingUniforms.shadowEnabled ? "Enabled" : "Disabled");
        }
    }

private:
    const RenderContext& m_ctx;
    FGResource m_visRead, m_depthRead, m_shadowRead, m_skyRead;
    int m_width, m_height;
    std::string m_name = "Deferred Lighting";
};
