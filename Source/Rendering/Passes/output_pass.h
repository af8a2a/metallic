#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

// Fullscreen passthrough: samples any input texture and writes to $backbuffer.
// Safely handles RGBA16Float → BGRA8Unorm conversion (clamp to [0,1]).
// Drop-in replacement for TonemapPass when debugging.
class OutputPass : public RenderPass {
public:
    OutputPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "$backbuffer") return m_dest;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Read the first non-special input as source texture
        for (const auto& [inputName, resource] : m_inputResources) {
            if (!inputName.empty() && inputName[0] != '$' && resource.isValid()) {
                m_sourceRead = builder.read(resource);
                break;
            }
        }

        m_dest = getInput("$backbuffer");
        if (m_dest.isValid()) {
            builder.setColorAttachment(0, m_dest,
                MTL::LoadActionDontCare, MTL::StoreActionStore,
                MTL::ClearColor(0.0, 0.0, 0.0, 1.0));
        }
        builder.setSideEffect();
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("OutputPass");
        if (!m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->renderPipelines.find("OutputPass");
        if (pipeIt == m_runtimeContext->renderPipelines.end()) return;

        auto samplerIt = m_runtimeContext->samplers.find("tonemap");
        if (samplerIt == m_runtimeContext->samplers.end()) return;

        enc->setRenderPipelineState(pipeIt->second);
        enc->setFragmentTexture(m_frameGraph->getTexture(m_sourceRead), 0);
        enc->setFragmentSamplerState(samplerIt->second, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }

    void renderUI() override {
        ImGui::Text("Passthrough %d x %d", m_width, m_height);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_sourceRead;
    FGResource m_dest;
    int m_width, m_height;
    std::string m_name = "Output";
};
