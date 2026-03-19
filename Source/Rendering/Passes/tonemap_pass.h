#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class TonemapPass : public RenderPass {
public:
    TonemapPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        m_sourceInputName.clear();
        m_hasExposureLutInput = false;
        for (const auto& inputName : config.inputs) {
            if (!inputName.empty() && inputName[0] != '$') {
                if (inputName == "exposureLut") {
                    m_hasExposureLutInput = true;
                } else if (m_sourceInputName.empty()) {
                    m_sourceInputName = inputName;
                }
            }
        }
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
        if (config.config.contains("autoExposure")) m_autoExposure = config.config["autoExposure"].get<bool>();
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "tonemapOutput") return m_dest;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_sourceRead = FGResource{};
        m_exposureLutRead = FGResource{};
        m_dest = FGResource{};

        FGResource sourceInput = getSourceInput();
        if (sourceInput.isValid()) {
            m_sourceRead = builder.read(sourceInput);
        }

        if (m_hasExposureLutInput) {
            FGResource lutInput = getInput("exposureLut");
            if (lutInput.isValid()) {
                m_exposureLutRead = builder.read(lutInput);
            }
        }

        m_dest = builder.create("tonemapOutput",
            FGTextureDesc::renderTarget(currentOutputWidth(),
                                        currentOutputHeight(),
                                        RhiFormat::RGBA8Srgb));
        m_dest = builder.setColorAttachment(0,
                                            m_dest,
                                            RhiLoadAction::DontCare,
                                            RhiStoreAction::Store,
                                            RhiClearColor(0.0, 0.0, 0.0, 1.0));
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("TonemapPass");
        MICROPROFILE_SCOPEI("RenderPass", "TonemapPass", 0xffff8800);

        if (!m_runtimeContext || !m_sourceRead.isValid()) return;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("TonemapPass");
        if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle()) return;

        auto samplerIt = m_runtimeContext->samplersRhi.find("tonemap");
        if (samplerIt == m_runtimeContext->samplersRhi.end() || !samplerIt->second.nativeHandle()) return;

        uint32_t outputWidth = static_cast<uint32_t>(m_width);
        uint32_t outputHeight = static_cast<uint32_t>(m_height);
        if (RhiTexture* destTex = m_dest.isValid() ? m_frameGraph->getTexture(m_dest) : nullptr) {
            outputWidth = destTex->width();
            outputHeight = destTex->height();
        } else if (m_frameContext && m_frameContext->displayWidth > 0 && m_frameContext->displayHeight > 0) {
            outputWidth = static_cast<uint32_t>(m_frameContext->displayWidth);
            outputHeight = static_cast<uint32_t>(m_frameContext->displayHeight);
        }

        TonemapUniforms uniforms{};
        uniforms.isActive = m_enabled ? 1u : 0u;
        uniforms.method = static_cast<uint32_t>(m_method);
        uniforms.exposure = m_exposure;
        uniforms.contrast = m_contrast;
        uniforms.brightness = m_brightness;
        uniforms.saturation = m_saturation;
        uniforms.vignette = m_vignette;
        uniforms.dither = m_dither ? 1u : 0u;
        const float safeOutputWidth = outputWidth > 0 ? static_cast<float>(outputWidth) : 1.0f;
        const float safeOutputHeight = outputHeight > 0 ? static_cast<float>(outputHeight) : 1.0f;
        uniforms.invResolution = float2(1.0f / safeOutputWidth,
                                        1.0f / safeOutputHeight);
        uniforms.pad = 0.0f;
        uniforms.autoExposure = (m_autoExposure && m_exposureLutRead.isValid()) ? 1u : 0u;

        encoder.setRenderPipeline(pipeIt->second);
        encoder.setViewport(static_cast<float>(outputWidth), static_cast<float>(outputHeight), false);
        encoder.setCullMode(RhiCullMode::None);
        encoder.setFragmentTexture(m_frameGraph->getTexture(m_sourceRead), 0);
        encoder.setFragmentSampler(&samplerIt->second, 0);
        if (m_exposureLutRead.isValid()) {
            encoder.setFragmentTexture(m_frameGraph->getTexture(m_exposureLutRead), 1);
        }
        encoder.setFragmentBytes(&uniforms, sizeof(uniforms), 0);
        encoder.drawPrimitives(RhiPrimitiveType::Triangle, 0, 3);
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Checkbox("Enable", &m_enabled);
        const char* methods[] = {"Filmic", "Uncharted2", "Clip", "ACES", "AgX", "Khronos PBR"};
        ImGui::Combo("Method", &m_method, methods, IM_ARRAYSIZE(methods));
        if (m_hasExposureLutInput) {
            ImGui::Checkbox("Auto Exposure", &m_autoExposure);
        }
        if (!m_autoExposure) {
            ImGui::SliderFloat("Exposure", &m_exposure, 0.1f, 4.0f, "%.2f");
        } else {
            ImGui::BeginDisabled();
            ImGui::SliderFloat("Exposure", &m_exposure, 0.1f, 4.0f, "%.2f");
            ImGui::EndDisabled();
        }
        ImGui::SliderFloat("Contrast", &m_contrast, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Brightness", &m_brightness, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Saturation", &m_saturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vignette", &m_vignette, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Dither", &m_dither);
    }

private:
    FGResource getSourceInput() const {
        if (!m_sourceInputName.empty()) {
            FGResource source = getInput(m_sourceInputName);
            if (source.isValid()) {
                return source;
            }
        }
        for (const auto& [inputName, resource] : m_inputResources) {
            if (!inputName.empty() && inputName[0] == '$') continue;
            if (resource.isValid()) return resource;
        }
        return FGResource{};
    }

    const RenderContext& m_ctx;
    FGResource m_sourceRead;
    FGResource m_exposureLutRead;
    FGResource m_dest;
    int m_width, m_height;
    std::string m_name = "Tonemap";
    bool m_enabled = true;
    int m_method = 3; // ACES
    float m_exposure = 1.0f;
    float m_contrast = 1.0f;
    float m_brightness = 1.0f;
    float m_saturation = 1.0f;
    float m_vignette = 0.0f;
    bool m_dither = true;
    bool m_autoExposure = false;
    bool m_hasExposureLutInput = false;
    std::string m_sourceInputName;

    int currentRenderWidth() const {
        if (m_frameContext && m_frameContext->renderWidth > 0) {
            return m_frameContext->renderWidth;
        }
        if (m_runtimeContext && m_runtimeContext->renderWidth > 0) {
            return m_runtimeContext->renderWidth;
        }
        return m_width;
    }

    int currentRenderHeight() const {
        if (m_frameContext && m_frameContext->renderHeight > 0) {
            return m_frameContext->renderHeight;
        }
        if (m_runtimeContext && m_runtimeContext->renderHeight > 0) {
            return m_runtimeContext->renderHeight;
        }
        return m_height;
    }

    int currentDisplayWidth() const {
        if (m_frameContext && m_frameContext->displayWidth > 0) {
            return m_frameContext->displayWidth;
        }
        if (m_runtimeContext && m_runtimeContext->displayWidth > 0) {
            return m_runtimeContext->displayWidth;
        }
        return m_width;
    }

    int currentDisplayHeight() const {
        if (m_frameContext && m_frameContext->displayHeight > 0) {
            return m_frameContext->displayHeight;
        }
        if (m_runtimeContext && m_runtimeContext->displayHeight > 0) {
            return m_runtimeContext->displayHeight;
        }
        return m_height;
    }

    int currentOutputWidth() const {
        return m_sourceInputName == "dlssOutput" ? currentDisplayWidth() : currentRenderWidth();
    }

    int currentOutputHeight() const {
        return m_sourceInputName == "dlssOutput" ? currentDisplayHeight() : currentRenderHeight();
    }
};



