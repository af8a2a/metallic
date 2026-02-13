#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class SkyPass : public RenderPass {
public:
    SkyPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        m_sideEffect = config.sideEffect;
        if (config.config.contains("exposure")) {
            m_exposure = config.config["exposure"].get<float>();
        }
    }

    FGResource output;

    FGResource getOutput(const std::string& name) const override {
        if (name == "skyOutput") return output;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        output = builder.create("skyColor",
            FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatRGBA16Float));
        builder.setColorAttachment(0, output,
            MTL::LoadActionClear, MTL::StoreActionStore,
            MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
        if (m_sideEffect) {
            builder.setSideEffect();
        }
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("SkyPass");

        if (!m_frameContext || !m_runtimeContext) return;
        if (!m_frameContext->enableAtmosphereSky) return;

        auto pipeIt = m_runtimeContext->renderPipelines.find("SkyPass");
        if (pipeIt == m_runtimeContext->renderPipelines.end()) return;

        auto transmittanceIt = m_runtimeContext->importedTextures.find("transmittance");
        auto scatteringIt = m_runtimeContext->importedTextures.find("scattering");
        auto irradianceIt = m_runtimeContext->importedTextures.find("irradiance");
        auto samplerIt = m_runtimeContext->samplers.find("atmosphere");

        if (transmittanceIt == m_runtimeContext->importedTextures.end() ||
            scatteringIt == m_runtimeContext->importedTextures.end() ||
            irradianceIt == m_runtimeContext->importedTextures.end() ||
            samplerIt == m_runtimeContext->samplers.end()) {
            return;
        }

        // Build AtmosphereUniforms from FrameContext raw data
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        float4x4 invViewProj = viewProj;
        invViewProj.Invert();

        AtmosphereUniforms uniforms;
        uniforms.invViewProj = transpose(invViewProj);
        uniforms.cameraWorldPos = m_frameContext->cameraWorldPos;
        uniforms.sunDirection = m_frameContext->worldLightDir;
        uniforms.params = float4(m_exposure, 0.0f, 0.0f, 0.0f);
        uniforms.screenWidth = static_cast<uint32_t>(m_frameContext->width);
        uniforms.screenHeight = static_cast<uint32_t>(m_frameContext->height);
        uniforms.pad0 = 0;
        uniforms.pad1 = 0;

        enc->setRenderPipelineState(pipeIt->second);
        enc->setVertexBytes(&uniforms, sizeof(uniforms), 0);
        enc->setFragmentBytes(&uniforms, sizeof(uniforms), 0);
        enc->setFragmentTexture(transmittanceIt->second, 0);
        enc->setFragmentTexture(scatteringIt->second, 1);
        enc->setFragmentTexture(irradianceIt->second, 2);
        enc->setFragmentSamplerState(samplerIt->second, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::SliderFloat("Exposure", &m_exposure, 0.1f, 20.0f, "%.2f");
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Atmosphere Sky";
    bool m_sideEffect = false;
    float m_exposure = 10.0f;
};
