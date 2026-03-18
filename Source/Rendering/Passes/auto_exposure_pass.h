#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class AutoExposurePass : public RenderPass {
public:
    AutoExposurePass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~AutoExposurePass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("evMin")) m_evMin = config.config["evMin"].get<float>();
        if (config.config.contains("evMax")) m_evMax = config.config["evMax"].get<float>();
        if (config.config.contains("adaptationSpeed")) m_adaptationSpeed = config.config["adaptationSpeed"].get<float>();
        if (config.config.contains("lowPercentile")) m_lowPercentile = config.config["lowPercentile"].get<float>();
        if (config.config.contains("highPercentile")) m_highPercentile = config.config["highPercentile"].get<float>();
        m_sourceInputName.clear();
        for (const auto& inputName : config.inputs) {
            if (!inputName.empty() && inputName[0] != '$') {
                m_sourceInputName = inputName;
                break;
            }
        }
    }

    FGResource getOutput(const std::string& outputName) const override {
        if (outputName == "exposureLut") return m_exposureLut;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_sourceRead = FGResource{};
        m_exposureLut = FGResource{};

        FGResource sourceInput = getSourceInput();
        if (sourceInput.isValid()) {
            m_sourceRead = builder.read(sourceInput);
        }

        m_exposureLut = builder.create("exposureLut",
            FGTextureDesc::storageTexture(1, 1, RhiFormat::R32Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("AutoExposurePass");
        MICROPROFILE_SCOPEI("RenderPass", "AutoExposurePass", 0xffff8800);
        if (!m_runtimeContext || !m_sourceRead.isValid()) return;

        // Lazy-create histogram buffer
        if (!m_histogramBuffer && m_runtimeContext->resourceFactory) {
            RhiBufferDesc desc;
            desc.size = 256 * sizeof(uint32_t);
            desc.hostVisible = false;
            desc.debugName = "AutoExposureHistogram";
            m_histogramBuffer = m_runtimeContext->resourceFactory->createBuffer(desc);
        }
        if (!m_histogramBuffer) return;

        auto histIt = m_runtimeContext->computePipelinesRhi.find("HistogramPass");
        auto expIt = m_runtimeContext->computePipelinesRhi.find("AutoExposurePass");
        if (histIt == m_runtimeContext->computePipelinesRhi.end() ||
            expIt == m_runtimeContext->computePipelinesRhi.end() ||
            !histIt->second.nativeHandle() ||
            !expIt->second.nativeHandle()) return;

        AutoExposureUniforms uniforms{};
        uniforms.evMinValue = m_evMin;
        uniforms.evMaxValue = m_evMax;
        uniforms.adaptationSpeed = m_adaptationSpeed;
        uniforms.deltaTime = m_frameContext ? m_frameContext->deltaTime : 0.016f;
        uniforms.lowPercentile = m_lowPercentile;
        uniforms.highPercentile = m_highPercentile;

        auto* hdrTex = m_frameGraph->getTexture(m_sourceRead);
        auto* lutTex = m_frameGraph->getTexture(m_exposureLut);
        if (!hdrTex || !lutTex) return;

        const uint32_t sourceWidth = hdrTex->width();
        const uint32_t sourceHeight = hdrTex->height();
        uniforms.screenWidth = sourceWidth > 0 ? sourceWidth : static_cast<uint32_t>(m_width);
        uniforms.screenHeight = sourceHeight > 0 ? sourceHeight : static_cast<uint32_t>(m_height);

        // Slang wraps all globals into KernelContext 鈥?both kernels expect all bindings:
        // buffer(0) = uniforms, texture(0) = hdrInput, buffer(1) = histogramBuffer, texture(1) = exposureLut

        // Dispatch 1: Histogram
        encoder.setComputePipeline(histIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), 0);
        encoder.setTexture(hdrTex, 0);
        encoder.setStorageTexture(lutTex, 1);
        encoder.setBuffer(m_histogramBuffer.get(), 0, 1);
        encoder.dispatchThreadgroups({static_cast<uint32_t>((uniforms.screenWidth + 15) / 16),
                                      static_cast<uint32_t>((uniforms.screenHeight + 15) / 16),
                                      1},
                                     {16, 16, 1});

        // Memory barrier between dispatches
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Dispatch 2: Exposure
        encoder.setComputePipeline(expIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), 0);
        encoder.setTexture(hdrTex, 0);
        encoder.setStorageTexture(lutTex, 1);
        encoder.setBuffer(m_histogramBuffer.get(), 0, 1);
        encoder.dispatchThreadgroups({1, 1, 1}, {256, 1, 1});
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::SliderFloat("EV Min", &m_evMin, -10.0f, 0.0f, "%.1f");
        ImGui::SliderFloat("EV Max", &m_evMax, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Adaptation Speed", &m_adaptationSpeed, 0.1f, 10.0f, "%.1f");
        ImGui::SliderFloat("Low Percentile", &m_lowPercentile, 0.0f, 0.5f, "%.2f");
        ImGui::SliderFloat("High Percentile", &m_highPercentile, 0.5f, 1.0f, "%.2f");
    }

private:
    FGResource getSourceInput() const {
        if (!m_sourceInputName.empty()) {
            FGResource source = getInput(m_sourceInputName);
            if (source.isValid()) return source;
        }
        for (const auto& [inputName, resource] : m_inputResources) {
            if (!inputName.empty() && inputName[0] == '$') continue;
            if (resource.isValid()) return resource;
        }
        return FGResource{};
    }

    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Auto Exposure";
    std::string m_sourceInputName;

    FGResource m_sourceRead;
    FGResource m_exposureLut;
    std::unique_ptr<RhiBuffer> m_histogramBuffer;

    float m_evMin = -5.0f;
    float m_evMax = 10.0f;
    float m_adaptationSpeed = 1.1f;
    float m_lowPercentile = 0.1f;
    float m_highPercentile = 0.9f;
};



