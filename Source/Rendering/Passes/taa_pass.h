#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "metal_frame_graph.h"
#include "imgui.h"

class TAAPass : public RenderPass {
public:
    TAAPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~TAAPass() override {
        releaseHistory();
    }

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        m_sourceInputName.clear();
        for (const auto& inputName : config.inputs) {
            if (!inputName.empty() && inputName[0] != '$') {
                if (inputName == "depth" || inputName == "motionVectors")
                    continue;
                if (m_sourceInputName.empty())
                    m_sourceInputName = inputName;
            }
        }
    }

    FGResource getOutput(const std::string& outputName) const override {
        if (outputName == "taaOutput") return m_taaOutput;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_sourceRead = FGResource{};
        m_depthRead = FGResource{};
        m_motionRead = FGResource{};
        m_taaOutput = FGResource{};
        m_passthroughNoPipeline = false;

        FGResource sourceInput = getSourceInput();
        if (!hasTAAPipeline() && sourceInput.isValid()) {
            // If TAA shader is unavailable, forward the current color so taaOutput stays valid.
            m_sourceRead = sourceInput;
            m_taaOutput = sourceInput;
            m_passthroughNoPipeline = true;
            return;
        }
        if (sourceInput.isValid())
            m_sourceRead = builder.read(sourceInput);

        FGResource depthInput = getInput("depth");
        if (depthInput.isValid())
            m_depthRead = builder.read(depthInput);

        FGResource motionInput = getInput("motionVectors");
        if (motionInput.isValid())
            m_motionRead = builder.read(motionInput);

        m_taaOutput = builder.create("taaOutput",
            FGTextureDesc::storageTexture(m_width, m_height, RhiFormat::RGBA16Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        auto* enc = metalEncoder(encoder);
        ZoneScopedN("TAAPass");
        if (!m_frameContext || !m_runtimeContext) return;
        if (m_passthroughNoPipeline) return;
        if (!m_frameContext->enableTAA) {
            copyCurrentToOutput(enc);
            return;
        }

        auto pipeIt = m_runtimeContext->computePipelines.find("TAAPass");
        if (pipeIt == m_runtimeContext->computePipelines.end() || !pipeIt->second) {
            copyCurrentToOutput(enc);
            return;
        }

        ensureHistory();

        MTL::Texture* currentTex = m_sourceRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_sourceRead)) : nullptr;
        MTL::Texture* depthTex = m_depthRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_depthRead)) : nullptr;
        MTL::Texture* motionTex = m_motionRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_motionRead)) : nullptr;
        MTL::Texture* outputTex = metalTexture(m_frameGraph->getTexture(m_taaOutput));
        if (!currentTex || !outputTex) return;
        if (!depthTex) depthTex = currentTex;
        if (!motionTex) motionTex = currentTex;

        TAAUniforms uniforms{};
        uniforms.jitterOffset = m_frameContext->jitterOffset;
        uniforms.invResolution = float2(1.0f / m_width, 1.0f / m_height);
        uniforms.screenWidth = static_cast<uint32_t>(m_width);
        uniforms.screenHeight = static_cast<uint32_t>(m_height);
        uniforms.blendMin = m_blendMin;
        uniforms.blendMax = m_blendMax;
        uniforms.varianceClipGamma = m_varianceClipGamma;
        uniforms.frameIndex = m_historyValid ? m_frameContext->frameIndex : 0;
        uniforms.motionWeightScale = m_motionWeightScale;
        uniforms.copyOnly = 0;

        // PLACEHOLDER_DISPATCH

        MTL::Texture* historyReadTex = m_historyTextures[1 - m_historyIndex];
        MTL::Texture* historyWriteTex = m_historyTextures[m_historyIndex];

        enc->setComputePipelineState(pipeIt->second);
        enc->setBytes(&uniforms, sizeof(uniforms), 0);
        enc->setTexture(currentTex, 0);
        enc->setTexture(depthTex, 1);
        enc->setTexture(motionTex, 2);
        enc->setTexture(historyReadTex, 3);
        enc->setTexture(outputTex, 4);
        enc->setTexture(historyWriteTex, 5);

        bindTonemapSampler(enc);

        MTL::Size tgSize(8, 8, 1);
        MTL::Size grid((m_width + 7) / 8, (m_height + 7) / 8, 1);
        enc->dispatchThreadgroups(grid, tgSize);

        m_historyIndex = 1 - m_historyIndex;
        m_historyValid = true;
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::SliderFloat("Blend Min", &m_blendMin, 0.01f, 0.5f, "%.3f");
        ImGui::SliderFloat("Blend Max", &m_blendMax, 0.5f, 1.0f, "%.2f");
        ImGui::SliderFloat("Variance Clip Gamma", &m_varianceClipGamma, 0.5f, 2.0f, "%.2f");
        ImGui::SliderFloat("Motion Weight Scale", &m_motionWeightScale, 0.1f, 100.0f, "%.1f");
    }

private:
    FGResource getSourceInput() const {
        if (!m_sourceInputName.empty()) {
            FGResource source = getInput(m_sourceInputName);
            if (source.isValid()) return source;
        }
        for (const auto& [inputName, resource] : m_inputResources) {
            if (!inputName.empty() && inputName[0] == '$') continue;
            if (inputName == "depth" || inputName == "motionVectors") continue;
            if (resource.isValid()) return resource;
        }
        return FGResource{};
    }

    bool hasTAAPipeline() const {
        if (!m_runtimeContext) return false;
        auto it = m_runtimeContext->computePipelines.find("TAAPass");
        return it != m_runtimeContext->computePipelines.end() && it->second;
    }

    void bindTonemapSampler(MTL::ComputeCommandEncoder* enc) const {
        if (!m_runtimeContext) return;
        auto samplerIt = m_runtimeContext->samplers.find("tonemap");
        if (samplerIt != m_runtimeContext->samplers.end() && samplerIt->second) {
            enc->setSamplerState(samplerIt->second, 0);
        }
    }

    // PLACEHOLDER_PRIVATE_METHODS

    void ensureHistory() {
        if (!m_runtimeContext || !m_runtimeContext->device) return;
        uint32_t w = static_cast<uint32_t>(m_width);
        uint32_t h = static_cast<uint32_t>(m_height);

        if (m_historyTextures[0] && m_historyTextures[0]->width() == w && m_historyTextures[0]->height() == h)
            return;

        releaseHistory();
        auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
            RhiFormat::RGBA16Float, w, h, false);
        desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        desc->setStorageMode(MTL::StorageModePrivate);
        m_historyTextures[0] = m_runtimeContext->device->newTexture(desc);
        m_historyTextures[1] = m_runtimeContext->device->newTexture(desc);
        m_historyValid = false;
        m_historyIndex = 0;
    }

    void releaseHistory() {
        for (auto& tex : m_historyTextures) {
            if (tex) { tex->release(); tex = nullptr; }
        }
    }

    void copyCurrentToOutput(MTL::ComputeCommandEncoder* enc) {
        // When TAA is disabled, run a copy-only dispatch that bypasses history sampling.
        auto pipeIt = m_runtimeContext->computePipelines.find("TAAPass");
        if (pipeIt == m_runtimeContext->computePipelines.end() || !pipeIt->second) return;

        ensureHistory();

        MTL::Texture* currentTex = m_sourceRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_sourceRead)) : nullptr;
        MTL::Texture* depthTex = m_depthRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_depthRead)) : nullptr;
        MTL::Texture* motionTex = m_motionRead.isValid() ? metalTexture(m_frameGraph->getTexture(m_motionRead)) : nullptr;
        MTL::Texture* outputTex = metalTexture(m_frameGraph->getTexture(m_taaOutput));
        if (!currentTex || !outputTex) return;
        if (!depthTex) depthTex = currentTex;
        if (!motionTex) motionTex = currentTex;

        MTL::Texture* historyReadTex = m_historyTextures[1 - m_historyIndex];
        MTL::Texture* historyWriteTex = m_historyTextures[m_historyIndex];

        TAAUniforms uniforms{};
        uniforms.invResolution = float2(1.0f / m_width, 1.0f / m_height);
        uniforms.screenWidth = static_cast<uint32_t>(m_width);
        uniforms.screenHeight = static_cast<uint32_t>(m_height);
        uniforms.blendMin = m_blendMin;
        uniforms.blendMax = m_blendMax;
        uniforms.varianceClipGamma = m_varianceClipGamma;
        uniforms.frameIndex = 0;
        uniforms.motionWeightScale = m_motionWeightScale;
        uniforms.copyOnly = 1;

        enc->setComputePipelineState(pipeIt->second);
        enc->setBytes(&uniforms, sizeof(uniforms), 0);
        enc->setTexture(currentTex, 0);
        enc->setTexture(depthTex, 1);
        enc->setTexture(motionTex, 2);
        enc->setTexture(historyReadTex, 3);
        enc->setTexture(outputTex, 4);
        enc->setTexture(historyWriteTex, 5);
        bindTonemapSampler(enc);

        MTL::Size tgSize(8, 8, 1);
        MTL::Size grid((m_width + 7) / 8, (m_height + 7) / 8, 1);
        enc->dispatchThreadgroups(grid, tgSize);

        m_historyIndex = 0;
        m_historyValid = false;
    }

    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "TAA";
    std::string m_sourceInputName;

    FGResource m_sourceRead, m_depthRead, m_motionRead, m_taaOutput;
    bool m_passthroughNoPipeline = false;

    MTL::Texture* m_historyTextures[2] = {nullptr, nullptr};
    int m_historyIndex = 0;
    bool m_historyValid = false;

    float m_blendMin = 0.05f;
    float m_blendMax = 1.0f;
    float m_varianceClipGamma = 1.0f;
    float m_motionWeightScale = 20.0f;
};


