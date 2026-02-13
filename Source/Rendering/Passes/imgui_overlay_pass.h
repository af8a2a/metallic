#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui_metal_bridge.h"

class ImGuiOverlayPass : public RenderPass {
public:
    ImGuiOverlayPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "$backbuffer") return m_drawable;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_drawable = getInput("$backbuffer");
        FGResource depthInput = getInput("depth");

        if (m_drawable.isValid()) {
            builder.setColorAttachment(0, m_drawable,
                MTL::LoadActionLoad, MTL::StoreActionStore);
        }
        if (depthInput.isValid()) {
            m_depthRead = builder.read(depthInput);
            builder.setDepthAttachment(m_depthRead,
                MTL::LoadActionLoad, MTL::StoreActionDontCare, m_ctx.depthClearValue);
        }
        builder.setSideEffect();
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("ImGuiOverlayPass");
        if (!m_frameContext) return;
        imguiRenderDrawData(m_frameContext->commandBuffer, enc);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_drawable;
    FGResource m_depthRead;
    int m_width, m_height;
    std::string m_name = "ImGui Overlay";
};
