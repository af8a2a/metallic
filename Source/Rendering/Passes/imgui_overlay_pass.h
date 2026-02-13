#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui_metal_bridge.h"

class ImGuiOverlayPass : public RenderPass {
public:
    // Data-driven constructor
    ImGuiOverlayPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h), m_legacyMode(false) {}

    // Legacy constructor
    ImGuiOverlayPass(FGResource drawable, FGResource depth, double depthClearValue, MTL::CommandBuffer* cmdBuf)
        : m_ctx(*(RenderContext*)nullptr)  // Not used in legacy mode
        , m_drawable(drawable)
        , m_legacyDepth(depth)
        , m_legacyDepthClearValue(depthClearValue)
        , m_legacyCmdBuf(cmdBuf)
        , m_width(0)
        , m_height(0)
        , m_legacyMode(true) {}

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
        if (m_legacyMode) {
            builder.setColorAttachment(0, m_drawable,
                MTL::LoadActionLoad, MTL::StoreActionStore);
            if (m_legacyDepth.isValid()) {
                m_depthRead = builder.read(m_legacyDepth);
                builder.setDepthAttachment(m_depthRead,
                    MTL::LoadActionLoad, MTL::StoreActionDontCare, m_legacyDepthClearValue);
            }
            builder.setSideEffect();
            return;
        }

        // Data-driven mode: get backbuffer and depth from inputs
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
        if (m_legacyMode) {
            imguiRenderDrawData(m_legacyCmdBuf, enc);
            return;
        }
        if (!m_frameContext) return;
        imguiRenderDrawData(m_frameContext->commandBuffer, enc);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_drawable;
    FGResource m_legacyDepth;
    FGResource m_depthRead;
    double m_legacyDepthClearValue = 0.0;
    MTL::CommandBuffer* m_legacyCmdBuf = nullptr;
    int m_width, m_height;
    std::string m_name = "ImGui Overlay";
    bool m_legacyMode = false;
};
