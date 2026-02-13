#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class TonemapPass : public RenderPass {
public:
    // Data-driven constructor
    TonemapPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h), m_legacyMode(false) {}

    // Legacy constructor
    TonemapPass(FGResource source,
                FGResource dest,
                MTL::RenderPipelineState* pipeline,
                MTL::SamplerState* sampler,
                const TonemapUniforms& uniforms,
                int w, int h)
        : m_ctx(*(RenderContext*)nullptr)  // Not used in legacy mode
        , m_legacySource(source)
        , m_legacyDest(dest)
        , m_pipeline(pipeline)
        , m_sampler(sampler)
        , m_legacyUniforms(uniforms)
        , m_width(w)
        , m_height(h)
        , m_legacyMode(true) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("method")) {
            std::string method = config.config["method"].get<std::string>();
            if (method == "Filmic") m_method = 0;
            else if (method == "Uncharted2") m_method = 1;
            else if (method == "Clip") m_method = 2;
            else if (method == "ACES") m_method = 3;
            else if (method == "AgX") m_method = 4;
            else if (method == "KhronosPBR") m_method = 5;
        }
        if (config.config.contains("exposure")) m_exposure = config.config["exposure"].get<float>();
        if (config.config.contains("contrast")) m_contrast = config.config["contrast"].get<float>();
        if (config.config.contains("brightness")) m_brightness = config.config["brightness"].get<float>();
        if (config.config.contains("saturation")) m_saturation = config.config["saturation"].get<float>();
        if (config.config.contains("vignette")) m_vignette = config.config["vignette"].get<float>();
        if (config.config.contains("dither")) m_dither = config.config["dither"].get<bool>();
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "$backbuffer") return m_dest;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        if (m_legacyMode) {
            m_sourceRead = builder.read(m_legacySource);
            builder.setColorAttachment(0, m_legacyDest,
                MTL::LoadActionDontCare, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
            builder.setSideEffect();
            m_dest = m_legacyDest;
        } else {
            // Read source from input
            FGResource sourceInput = getInput("lightingOutput");
            if (sourceInput.isValid()) {
                m_sourceRead = builder.read(sourceInput);
            }

            // Get backbuffer from runtime context (set via setInput)
            m_dest = getInput("$backbuffer");
            if (m_dest.isValid()) {
                builder.setColorAttachment(0, m_dest,
                    MTL::LoadActionDontCare, MTL::StoreActionStore,
                    MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
            }
            builder.setSideEffect();
        }
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("TonemapPass");

        if (m_legacyMode) {
            enc->setRenderPipelineState(m_pipeline);
            enc->setFragmentTexture(m_frameGraph->getTexture(m_sourceRead), 0);
            enc->setFragmentSamplerState(m_sampler, 0);
            enc->setFragmentBytes(&m_legacyUniforms, sizeof(m_legacyUniforms), 0);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            return;
        }

        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->renderPipelines.find("TonemapPass");
        if (pipeIt == m_runtimeContext->renderPipelines.end()) return;

        auto samplerIt = m_runtimeContext->samplers.find("tonemap");
        if (samplerIt == m_runtimeContext->samplers.end()) return;

        TonemapUniforms uniforms{};
        uniforms.isActive = 1u;
        uniforms.method = static_cast<uint32_t>(m_method);
        uniforms.exposure = m_exposure;
        uniforms.contrast = m_contrast;
        uniforms.brightness = m_brightness;
        uniforms.saturation = m_saturation;
        uniforms.vignette = m_vignette;
        uniforms.dither = m_dither ? 1u : 0u;
        uniforms.invResolution = float2(1.0f / float(m_width), 1.0f / float(m_height));

        enc->setRenderPipelineState(pipeIt->second);
        enc->setFragmentTexture(m_frameGraph->getTexture(m_sourceRead), 0);
        enc->setFragmentSamplerState(samplerIt->second, 0);
        enc->setFragmentBytes(&uniforms, sizeof(uniforms), 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        const char* methods[] = {"Filmic", "Uncharted2", "Clip", "ACES", "AgX", "Khronos PBR"};
        ImGui::Combo("Method", &m_method, methods, IM_ARRAYSIZE(methods));
        ImGui::SliderFloat("Exposure", &m_exposure, 0.1f, 4.0f, "%.2f");
        ImGui::SliderFloat("Contrast", &m_contrast, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Brightness", &m_brightness, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Saturation", &m_saturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vignette", &m_vignette, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Dither", &m_dither);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_legacySource;
    FGResource m_legacyDest;
    FGResource m_sourceRead;
    FGResource m_dest;
    MTL::RenderPipelineState* m_pipeline = nullptr;
    MTL::SamplerState* m_sampler = nullptr;
    TonemapUniforms m_legacyUniforms;
    int m_width, m_height;
    std::string m_name = "Tonemap";
    int m_method = 3; // ACES
    float m_exposure = 1.0f;
    float m_contrast = 1.0f;
    float m_brightness = 1.0f;
    float m_saturation = 1.0f;
    float m_vignette = 0.0f;
    bool m_dither = true;
    bool m_legacyMode = false;
};
