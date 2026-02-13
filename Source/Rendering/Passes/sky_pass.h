#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class SkyPass : public RenderPass {
public:
    // Data-driven constructor
    SkyPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h), m_legacyMode(false) {}

    // Legacy constructor for forward rendering mode
    SkyPass(FGResource target,
            MTL::RenderPipelineState* pipeline,
            MTL::Texture* transmittance,
            MTL::Texture* scattering,
            MTL::Texture* irradiance,
            MTL::SamplerState* sampler,
            const AtmosphereUniforms& uniforms,
            bool sideEffect,
            int w, int h)
        : m_ctx(*(RenderContext*)nullptr)  // Not used in legacy mode
        , m_target(target)
        , m_pipeline(pipeline)
        , m_transmittance(transmittance)
        , m_scattering(scattering)
        , m_irradiance(irradiance)
        , m_sampler(sampler)
        , m_legacyUniforms(uniforms)
        , m_sideEffect(sideEffect)
        , m_width(w)
        , m_height(h)
        , m_legacyMode(true) {}

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
        if (m_legacyMode && m_target.isValid()) {
            builder.setColorAttachment(0, m_target,
                MTL::LoadActionClear, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
            if (m_sideEffect) {
                builder.setSideEffect();
            }
            output = m_target;
        } else {
            output = builder.create("skyColor",
                FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatRGBA16Float));
            builder.setColorAttachment(0, output,
                MTL::LoadActionClear, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
            if (m_sideEffect) {
                builder.setSideEffect();
            }
        }
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("SkyPass");

        if (m_legacyMode) {
            // Legacy execution path
            enc->setRenderPipelineState(m_pipeline);
            enc->setVertexBytes(&m_legacyUniforms, sizeof(m_legacyUniforms), 0);
            enc->setFragmentBytes(&m_legacyUniforms, sizeof(m_legacyUniforms), 0);
            enc->setFragmentTexture(m_transmittance, 0);
            enc->setFragmentTexture(m_scattering, 1);
            enc->setFragmentTexture(m_irradiance, 2);
            enc->setFragmentSamplerState(m_sampler, 0);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            return;
        }

        // Data-driven execution path
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

        AtmosphereUniforms uniforms = m_frameContext->skyUniforms;
        uniforms.params.x = m_exposure;

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
    FGResource m_target;
    MTL::RenderPipelineState* m_pipeline = nullptr;
    MTL::Texture* m_transmittance = nullptr;
    MTL::Texture* m_scattering = nullptr;
    MTL::Texture* m_irradiance = nullptr;
    MTL::SamplerState* m_sampler = nullptr;
    AtmosphereUniforms m_legacyUniforms;
    int m_width, m_height;
    std::string m_name = "Atmosphere Sky";
    bool m_sideEffect = false;
    float m_exposure = 10.0f;
    bool m_legacyMode = false;
};
