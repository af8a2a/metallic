#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class OutputPass : public RenderPass {
public:
    OutputPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(OutputPass, "Output", "Utility",
        (std::vector<PassSlotInfo>{
            makeInputSlot("source", "Source"),
            makeInputSlot("compareSource", "Compare Source", true),
            makeHiddenInputSlot("presentReady", "Present Ready", true)
        }),
        (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}),
        PassTypeInfo::PassType::Render);

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.is_object()) {
            if (config.config.contains("compareMode")) {
                const auto& mode = config.config["compareMode"];
                if (mode.is_boolean()) {
                    m_compareMode = mode.get<bool>();
                } else if (mode.is_string()) {
                    const std::string modeName = mode.get<std::string>();
                    m_compareMode = modeName == "split" || modeName == "Split";
                }
            }
        }
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "target") return m_dest;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_sourceRead = FGResource{};
        m_compareSourceRead = FGResource{};
        m_presentReady = FGResource{};
        m_dest = FGResource{};

        FGResource sourceInput = getInput("source");
        if (sourceInput.isValid()) {
            m_sourceRead = builder.read(sourceInput);
        }
        FGResource compareSourceInput = getInput("compareSource");
        if (compareSourceInput.isValid()) {
            m_compareSourceRead = builder.read(compareSourceInput);
        }
        FGResource presentReadyInput = getInput("presentReady");
        if (presentReadyInput.isValid()) {
            m_presentReady = builder.read(presentReadyInput);
        }

        m_dest = getOutputTarget("target");
        if (m_dest.isValid()) {
            m_dest = builder.setColorAttachment(0,
                                                m_dest,
                                                RhiLoadAction::DontCare,
                                                RhiStoreAction::Store,
                                                RhiClearColor(0.0, 0.0, 0.0, 1.0));
        }
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("OutputPass");
        MICROPROFILE_SCOPEI("RenderPass", "OutputPass", 0xffff8800);
        if (!m_runtimeContext || !m_sourceRead.isValid()) return;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("OutputPass");
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

        encoder.setRenderPipeline(pipeIt->second);
        encoder.setViewport(static_cast<float>(outputWidth), static_cast<float>(outputHeight), false);
        encoder.setCullMode(RhiCullMode::None);
        RhiTexture* sourceTex = m_frameGraph->getTexture(m_sourceRead);
        if (!sourceTex) return;
        RhiTexture* compareTex = m_compareSourceRead.isValid()
            ? m_frameGraph->getTexture(m_compareSourceRead)
            : sourceTex;
        if (!compareTex) {
            compareTex = sourceTex;
        }
        struct PushData {
            uint32_t compareMode;
            uint32_t pad[3];
        } pushData{};
        pushData.compareMode = (m_compareMode && m_compareSourceRead.isValid()) ? 1u : 0u;
        encoder.setPushConstants(&pushData, sizeof(pushData));
        encoder.setFragmentTexture(sourceTex, 0);
        encoder.setFragmentTexture(compareTex, 1);
        encoder.setFragmentSampler(&samplerIt->second, 0);
        encoder.drawPrimitives(RhiPrimitiveType::Triangle, 0, 3);
    }

    void renderUI() override {
        ImGui::Text("Passthrough %d x %d", m_width, m_height);
    }

private:
    const RenderContext& m_ctx;
    FGResource m_sourceRead;
    FGResource m_compareSourceRead;
    FGResource m_presentReady;
    FGResource m_dest;
    int m_width, m_height;
    std::string m_name = "Output";
    bool m_compareMode = false;
};

METALLIC_REGISTER_PASS(OutputPass);

