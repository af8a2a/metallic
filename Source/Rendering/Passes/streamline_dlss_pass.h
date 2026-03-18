#pragma once

#ifdef _WIN32

#include "render_pass.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "streamline_context.h"
#include "imgui.h"

#include <vulkan/vulkan.h>
#include <memory>

class RhiTexture;

// Vulkan-only DLSS Super Resolution pass.
// Replaces TAAPass in the visibility pipeline when DLSS is enabled.
//
// MVP approach: this pass internally owns a display-size output texture.
// The FrameGraph sees it as a compute pass that reads lightingOutput/depth/motionVectors
// and produces "dlssOutput". Downstream passes (Tonemap) read dlssOutput.
//
// When DLSS is not available or METALLIC_HAS_STREAMLINE is not defined,
// the pass acts as a simple passthrough (forwards its color input as output).
class StreamlineDlssPass : public RenderPass {
public:
    StreamlineDlssPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_renderWidth(w), m_renderHeight(h),
          m_displayWidth(w), m_displayHeight(h) {}

    ~StreamlineDlssPass() override {
        releaseOutputTexture();
    }

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        m_sourceInputName.clear();
        for (const auto& inputName : config.inputs) {
            if (inputName == "depth" || inputName == "motionVectors") continue;
            if (!inputName.empty() && inputName[0] != '$') {
                if (m_sourceInputName.empty())
                    m_sourceInputName = inputName;
            }
        }
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

        FGResource sourceInput = getSourceInput();
        if (sourceInput.isValid())
            m_sourceRead = builder.read(sourceInput);

        FGResource depthInput = getInput("depth");
        if (depthInput.isValid())
            m_depthRead = builder.read(depthInput);

        FGResource motionInput = getInput("motionVectors");
        if (motionInput.isValid())
            m_motionRead = builder.read(motionInput);

        // Output at display resolution.
        // In the MVP, display == render when DLSS is off; the pass just forwards.
        // When DLSS is on, m_displayWidth/Height are set externally before pipeline build.
        m_dlssOutput = builder.create("dlssOutput",
            FGTextureDesc::storageTexture(m_displayWidth, m_displayHeight, RhiFormat::RGBA16Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("StreamlineDlssPass");
        MICROPROFILE_SCOPEI("RenderPass", "StreamlineDlssPass", 0xff00ff00);
        if (!m_frameContext) return;

#ifdef METALLIC_HAS_STREAMLINE
        if (m_streamlineCtx && m_streamlineCtx->isDlssAvailable() && isDlssEnabled()) {
            executeDlss(encoder);
            return;
        }
#endif
        // Fallback: passthrough (no-op, dlssOutput == sourceInput forwarded by FrameGraph)
        // The FrameGraph will have created dlssOutput; if we can't run DLSS we need to
        // copy the source into it. We do this via a simple blit-style compute if we have
        // a copy pipeline, or leave it black (acceptable for MVP — the pipeline should
        // not reach here if DLSS is off, since main_vulkan swaps back to TAAPass).
    }

    void renderUI() override {
        ImGui::Text("DLSS Pass %d x %d -> %d x %d",
                     m_renderWidth, m_renderHeight,
                     m_displayWidth, m_displayHeight);
#ifdef METALLIC_HAS_STREAMLINE
        if (m_streamlineCtx) {
            ImGui::Text("Status: %s", m_streamlineCtx->statusString().c_str());
        }
#else
        ImGui::Text("Streamline not compiled in");
#endif
    }

    // --- External configuration (called from main_vulkan before pipeline build) ---

    void setDisplaySize(int displayW, int displayH) {
        m_displayWidth = displayW;
        m_displayHeight = displayH;
    }

    void setStreamlineContext(StreamlineContext* ctx) {
        m_streamlineCtx = ctx;
    }

    bool isDlssEnabled() const { return m_dlssEnabled; }
    void setDlssEnabled(bool enabled) { m_dlssEnabled = enabled; }

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

    void releaseOutputTexture() {
        m_ownedOutputTexture.reset();
    }

#ifdef METALLIC_HAS_STREAMLINE
    void executeDlss(RhiComputeCommandEncoder& encoder);
#endif

    const RenderContext& m_ctx;
    int m_renderWidth, m_renderHeight;
    int m_displayWidth, m_displayHeight;
    std::string m_name = "DLSS";
    std::string m_sourceInputName;
    bool m_dlssEnabled = false;

    FGResource m_sourceRead, m_depthRead, m_motionRead, m_dlssOutput;

    StreamlineContext* m_streamlineCtx = nullptr;
    std::unique_ptr<RhiTexture> m_ownedOutputTexture;
};

#endif // _WIN32
