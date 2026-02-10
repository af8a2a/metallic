#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "imgui.h"

class SkyPass : public RenderPass {
public:
    SkyPass(FGResource target,
            MTL::RenderPipelineState* pipeline,
            MTL::Texture* transmittance,
            MTL::Texture* scattering,
            MTL::Texture* irradiance,
            MTL::SamplerState* sampler,
            const AtmosphereUniforms& uniforms,
            bool sideEffect,
            int w, int h)
        : m_target(target)
        , m_pipeline(pipeline)
        , m_transmittance(transmittance)
        , m_scattering(scattering)
        , m_irradiance(irradiance)
        , m_sampler(sampler)
        , m_uniforms(uniforms)
        , m_sideEffect(sideEffect)
        , m_width(w)
        , m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return "Atmosphere Sky"; }

    FGResource output;

    void setup(FGBuilder& builder) override {
        if (m_target.isValid()) {
            builder.setColorAttachment(0, m_target,
                MTL::LoadActionClear, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
            if (m_sideEffect) {
                builder.setSideEffect();
            }
            output = m_target;
        } else {
            output = builder.create("skyColor",
                FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatBGRA8Unorm));
            builder.setColorAttachment(0, output,
                MTL::LoadActionClear, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
        }
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("SkyPass");
        enc->setRenderPipelineState(m_pipeline);
        enc->setVertexBytes(&m_uniforms, sizeof(m_uniforms), 0);
        enc->setFragmentBytes(&m_uniforms, sizeof(m_uniforms), 0);
        enc->setFragmentTexture(m_transmittance, 0);
        enc->setFragmentTexture(m_scattering, 1);
        enc->setFragmentTexture(m_irradiance, 2);
        enc->setFragmentSamplerState(m_sampler, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Exposure: %.2f", m_uniforms.params.x);
    }

private:
    FGResource m_target;
    MTL::RenderPipelineState* m_pipeline;
    MTL::Texture* m_transmittance;
    MTL::Texture* m_scattering;
    MTL::Texture* m_irradiance;
    MTL::SamplerState* m_sampler;
    AtmosphereUniforms m_uniforms;
    bool m_sideEffect;
    int m_width;
    int m_height;
};
