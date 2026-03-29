#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class MeshletVisualizePass : public RenderPass {
public:
    MeshletVisualizePass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("colorMode")) {
            std::string mode = config.config["colorMode"].get<std::string>();
            if (mode == "Instance") m_colorMode = 1;
            else if (mode == "Triangle") m_colorMode = 2;
            else m_colorMode = 0;
        }
    }

    FGResource output;

    FGResource getOutput(const std::string& name) const override {
        if (name == "lightingOutput") return output;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        FGResource visInput = getInput("visibility");
        if (visInput.isValid()) m_visRead = builder.read(visInput);

        output = builder.create("meshletVisOutput",
            FGTextureDesc::storageTexture(m_width, m_height, RhiFormat::RGBA16Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("MeshletVisualizePass");
        MICROPROFILE_SCOPEI("RenderPass", "MeshletVisualizePass", 0xffff8800);
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->computePipelinesRhi.find("MeshletVisualizePass");
        if (pipeIt == m_runtimeContext->computePipelinesRhi.end() || !pipeIt->second.nativeHandle()) return;

        struct MeshletVisUniforms {
            uint32_t screenWidth;
            uint32_t screenHeight;
            uint32_t colorMode;
            uint32_t pad;
        } uniforms;
        uniforms.screenWidth = static_cast<uint32_t>(m_frameContext->width);
        uniforms.screenHeight = static_cast<uint32_t>(m_frameContext->height);
        uniforms.colorMode = static_cast<uint32_t>(m_colorMode);
        uniforms.pad = 0;

        encoder.setComputePipeline(pipeIt->second);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.setTexture(m_frameGraph->getTexture(m_visRead), 0);
        encoder.setStorageTexture(m_frameGraph->getTexture(output), 1);
        encoder.dispatchThreadgroups({static_cast<uint32_t>((m_width + 7) / 8), static_cast<uint32_t>((m_height + 7) / 8), 1},
                                     {8, 8, 1});
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        const char* modes[] = {"Meshlet", "Instance", "Triangle"};
        ImGui::Combo("Color Mode", &m_colorMode, modes, IM_ARRAYSIZE(modes));
    }

private:
    const RenderContext& m_ctx;
    FGResource m_visRead;
    int m_width, m_height;
    int m_colorMode = 0;
    std::string m_name = "Meshlet Visualize";
};



