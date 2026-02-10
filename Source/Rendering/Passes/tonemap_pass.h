#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "imgui.h"

class TonemapPass : public RenderPass {
public:
    TonemapPass(FGResource source,
                FGResource dest,
                MTL::RenderPipelineState* pipeline,
                MTL::SamplerState* sampler,
                const TonemapUniforms& uniforms,
                int w, int h)
        : m_source(source)
        , m_dest(dest)
        , m_pipeline(pipeline)
        , m_sampler(sampler)
        , m_uniforms(uniforms)
        , m_width(w)
        , m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return "Tonemap"; }

    void setup(FGBuilder& builder) override {
        m_sourceRead = builder.read(m_source);
        builder.setColorAttachment(0, m_dest,
            MTL::LoadActionDontCare, MTL::StoreActionStore,
            MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
        builder.setSideEffect();
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("TonemapPass");
        enc->setRenderPipelineState(m_pipeline);
        enc->setFragmentTexture(m_frameGraph->getTexture(m_sourceRead), 0);
        enc->setFragmentSamplerState(m_sampler, 0);
        enc->setFragmentBytes(&m_uniforms, sizeof(m_uniforms), 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Exposure: %.2f", m_uniforms.exposure);
        ImGui::Text("Contrast: %.2f", m_uniforms.contrast);
    }

private:
    FGResource m_source;
    FGResource m_dest;
    FGResource m_sourceRead;
    MTL::RenderPipelineState* m_pipeline;
    MTL::SamplerState* m_sampler;
    TonemapUniforms m_uniforms;
    int m_width;
    int m_height;
};
