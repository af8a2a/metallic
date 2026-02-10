#pragma once

#include "render_pass.h"
#include "imgui_metal_bridge.h"

class ImGuiOverlayPass : public RenderPass {
public:
    ImGuiOverlayPass(FGResource drawable, FGResource depth,
                     double depthClear, MTL::CommandBuffer* cmdBuf)
        : m_drawable(drawable), m_depth(depth)
        , m_depthClear(depthClear), m_commandBuffer(cmdBuf) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return "ImGui Overlay"; }

    void setup(FGBuilder& builder) override {
        builder.setColorAttachment(0, m_drawable,
            MTL::LoadActionLoad, MTL::StoreActionStore);
        m_depthRead = builder.read(m_depth);
        builder.setDepthAttachment(m_depthRead,
            MTL::LoadActionLoad, MTL::StoreActionDontCare, m_depthClear);
        builder.setSideEffect();
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("ImGuiOverlayPass");
        imguiRenderDrawData(m_commandBuffer, enc);
    }

private:
    FGResource m_drawable, m_depth;
    FGResource m_depthRead;
    double m_depthClear;
    MTL::CommandBuffer* m_commandBuffer;
};
