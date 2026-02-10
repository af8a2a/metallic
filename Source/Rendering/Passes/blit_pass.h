#pragma once

#include "render_pass.h"
#include "imgui.h"

class BlitPass : public RenderPass {
public:
    BlitPass(FGResource source, FGResource dest, int w, int h)
        : m_source(source), m_dest(dest), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Blit; }
    const char* name() const override { return "Blit to Drawable"; }

    void setup(FGBuilder& builder) override {
        m_sourceRead = builder.read(m_source);
        m_destWrite = builder.write(m_dest);
        builder.setSideEffect();
    }

    void executeBlit(MTL::BlitCommandEncoder* enc) override {
        ZoneScopedN("BlitPass");
        enc->copyFromTexture(
            m_frameGraph->getTexture(m_sourceRead), 0, 0,
            MTL::Origin(0, 0, 0), MTL::Size(m_width, m_height, 1),
            m_frameGraph->getTexture(m_destWrite), 0, 0,
            MTL::Origin(0, 0, 0));
    }

    void renderUI() override {
        ImGui::Text("Copy: %d x %d", m_width, m_height);
    }

private:
    FGResource m_source, m_dest;
    FGResource m_sourceRead, m_destWrite;
    int m_width, m_height;
};
