#pragma once

#ifdef _WIN32

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

// Vulkan-only DLSS Super Resolution pass.
// This pass behaves like a first-class upscaler node in the authored render graph.
// When DLSS is active, it reads render-resolution inputs and writes a display-resolution
// "dlssOutput". When DLSS is unavailable or disabled, it aliases its source input so the
// graph can keep running without a topology rewrite.
//
class StreamlineDlssPass : public RenderPass {
public:
    StreamlineDlssPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_renderWidth(w), m_renderHeight(h),
          m_displayWidth(w), m_displayHeight(h) {}

    ~StreamlineDlssPass() override = default;

    METALLIC_PASS_TYPE_INFO(StreamlineDlssPass, "DLSS", "Post-Process",
        (std::vector<PassSlotInfo>{
            makeInputSlot("source", "Source"),
            makeInputSlot("depth", "Depth", true),
            makeInputSlot("motionVectors", "Motion Vectors", true)
        }),
        (std::vector<PassSlotInfo>{makeOutputSlot("dlssOutput", "DLSS Output")}),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& outputName) const override {
        if (outputName == "dlssOutput") return m_dlssOutput;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_sourceRead = FGResource{};
        m_depthRead = FGResource{};
        m_motionRead = FGResource{};
        m_dlssOutput = FGResource{};
        m_passthrough = false;

        FGResource sourceInput = getSourceInput();
        if (!shouldEvaluateDlss() && sourceInput.isValid()) {
            m_sourceRead = sourceInput;
            m_dlssOutput = sourceInput;
            m_passthrough = true;
            return;
        }
        if (sourceInput.isValid()) {
            m_sourceRead = builder.read(sourceInput);
        }

        FGResource depthInput = getInput("depth");
        if (depthInput.isValid()) {
            m_depthRead = builder.read(depthInput);
        }

        FGResource motionInput = getInput("motionVectors");
        if (motionInput.isValid()) {
            m_motionRead = builder.read(motionInput);
        }

        m_dlssOutput = builder.create("dlssOutput",
            FGTextureDesc::storageTexture(currentDisplayWidth(),
                                          currentDisplayHeight(),
                                          RhiFormat::RGBA16Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("StreamlineDlssPass");
        MICROPROFILE_SCOPEI("RenderPass", "StreamlineDlssPass", 0xff00ff00);
        if (!m_frameContext || m_passthrough) return;

#ifdef METALLIC_HAS_STREAMLINE
        if (shouldEvaluateDlss()) {
            executeDlss(encoder);
            return;
        }
#endif
    }

    void renderUI() override {
        ImGui::Text("DLSS Pass %d x %d -> %d x %d",
                    currentRenderWidth(), currentRenderHeight(),
                    currentDisplayWidth(), currentDisplayHeight());

        const PipelineUiControls* uiControls = m_runtimeContext ? m_runtimeContext->uiControls : nullptr;
        if (auto* upscaler = currentUpscaler()) {
            ImGui::Text("Status: %s", upscaler->statusString().c_str());
        }
        if (uiControls && uiControls->hasDlssPass) {
            ImGui::TextUnformatted(uiControls->dlssAvailable
                ? "Streamline: Available"
                : "Streamline: Unavailable (pass-through)");
            ImGui::TextUnformatted(shouldEvaluateDlss() ? "Mode: DLSS" : "Mode: Pass-through");
            const char* presetNames[] = {
                "Off", "Max Performance", "Balanced", "Max Quality",
                "Ultra Performance", "Ultra Quality", "DLAA"
            };
            int presetIdx = static_cast<int>(uiControls->currentPreset);
            if (ImGui::Combo("DLSS Preset", &presetIdx, presetNames, IM_ARRAYSIZE(presetNames)) &&
                uiControls->onDlssPresetChanged) {
                uiControls->onDlssPresetChanged(static_cast<DlssPreset>(presetIdx));
            }
            ImGui::Text("Display: %d x %d", uiControls->displayWidth, uiControls->displayHeight);
            if (uiControls->dlssIsActiveUpscaler) {
                ImGui::TextUnformatted("DLSS is the active upscaler in this graph.");
                ImGui::Text("Render: %u x %u", uiControls->dlssRenderWidth, uiControls->dlssRenderHeight);
            } else {
                ImGui::TextUnformatted("DLSS pass is present but not driving the active post path.");
                if (!uiControls->dlssDiagnostic.empty()) {
                    ImGui::TextWrapped("%s", uiControls->dlssDiagnostic.c_str());
                }
            }
            if (!uiControls->dlssIsActiveUpscaler) {
                ImGui::BeginDisabled();
            }
            if (ImGui::Button("Reset DLSS History") && uiControls->onResetDlssHistory) {
                uiControls->onResetDlssHistory();
            }
            if (!uiControls->dlssIsActiveUpscaler) {
                ImGui::EndDisabled();
            }
        }
    }

private:
    FGResource getSourceInput() const {
        return getInput("source");
    }

    IUpscalerIntegration* currentUpscaler() const {
        return m_runtimeContext ? m_runtimeContext->upscaler : nullptr;
    }

    bool shouldEvaluateDlss() const {
        IUpscalerIntegration* upscaler = currentUpscaler();
        return upscaler && upscaler->isAvailable() && upscaler->isEnabled();
    }

    int currentRenderWidth() const {
        if (m_frameContext && m_frameContext->renderWidth > 0) {
            return m_frameContext->renderWidth;
        }
        if (m_runtimeContext && m_runtimeContext->renderWidth > 0) {
            return m_runtimeContext->renderWidth;
        }
        return m_renderWidth;
    }

    int currentRenderHeight() const {
        if (m_frameContext && m_frameContext->renderHeight > 0) {
            return m_frameContext->renderHeight;
        }
        if (m_runtimeContext && m_runtimeContext->renderHeight > 0) {
            return m_runtimeContext->renderHeight;
        }
        return m_renderHeight;
    }

    int currentDisplayWidth() const {
        if (m_frameContext && m_frameContext->displayWidth > 0) {
            return m_frameContext->displayWidth;
        }
        if (m_runtimeContext && m_runtimeContext->displayWidth > 0) {
            return m_runtimeContext->displayWidth;
        }
        return m_displayWidth;
    }

    int currentDisplayHeight() const {
        if (m_frameContext && m_frameContext->displayHeight > 0) {
            return m_frameContext->displayHeight;
        }
        if (m_runtimeContext && m_runtimeContext->displayHeight > 0) {
            return m_runtimeContext->displayHeight;
        }
        return m_displayHeight;
    }

#ifdef METALLIC_HAS_STREAMLINE
    void executeDlss(RhiComputeCommandEncoder& encoder);
#endif

    const RenderContext& m_ctx;
    int m_renderWidth, m_renderHeight;
    int m_displayWidth, m_displayHeight;
    std::string m_name = "DLSS";
    bool m_passthrough = false;

    FGResource m_sourceRead, m_depthRead, m_motionRead, m_dlssOutput;
};

METALLIC_REGISTER_PASS(StreamlineDlssPass);

#endif // _WIN32
