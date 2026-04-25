#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"

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
        if (name == "target") return m_drawable;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_drawable = getOutputTarget("target");

        if (m_drawable.isValid()) {
            m_drawable = builder.setColorAttachment(0,
                                                    m_drawable,
                                                    RhiLoadAction::Load,
                                                    RhiStoreAction::Store);
        }
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("ImGuiOverlayPass");
        MICROPROFILE_SCOPEI("RenderPass", "ImGuiOverlayPass", 0xffff8800);
        encoder.renderImGuiDrawData();
    }

private:
    const RenderContext& m_ctx;
    FGResource m_drawable;
    int m_width, m_height;
    std::string m_name = "ImGui Overlay";
};



