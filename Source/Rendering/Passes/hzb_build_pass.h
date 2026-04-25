#pragma once

#include "frame_context.h"
#include "hzb_constants.h"
#include "pass_registry.h"
#include "render_pass.h"

#include "imgui.h"

#include <vector>

class HZBBuildPass : public RenderPass {
public:
    HZBBuildPass(const RenderContext&, int w, int h)
        : m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    void setup(FGBuilder& builder) override {
        m_depthRead = FGResource{};
        m_historyWrites.clear();

        FGResource depthInput = getInput("depth");
        if (depthInput.isValid()) {
            m_depthRead = builder.read(depthInput);
        } else {
            m_levelCount = 0;
            return;
        }

        m_levelCount = computeHzbLevelCount(static_cast<uint32_t>(m_width),
                                            static_cast<uint32_t>(m_height));
        m_historyWrites.reserve(m_levelCount);
        for (uint32_t level = 0; level < m_levelCount; ++level) {
            const std::string resourceName = hzbHistoryResourceName(level);
            m_historyWrites.push_back(
                builder.writeHistory(resourceName.c_str(),
                                     makeHzbTextureDesc(static_cast<uint32_t>(m_width),
                                                        static_cast<uint32_t>(m_height),
                                                        level)));
        }
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("HZBBuildPass");
        MICROPROFILE_SCOPEI("RenderPass", "HZBBuildPass", 0xff44aa44);
        if (!m_frameContext || !m_runtimeContext || !m_frameGraph) {
            return;
        }

        auto pipelineIt = m_runtimeContext->computePipelinesRhi.find("HZBBuildPass");
        if (pipelineIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipelineIt->second.nativeHandle()) {
            return;
        }

        RhiTexture* sourceTexture =
            m_depthRead.isValid() ? m_frameGraph->getTexture(m_depthRead) : nullptr;
        if (!sourceTexture) {
            return;
        }

        struct HZBBuildUniforms {
            uint32_t srcWidth = 0;
            uint32_t srcHeight = 0;
            uint32_t dstWidth = 0;
            uint32_t dstHeight = 0;
            uint32_t sourceScale = 1;
            uint32_t _pad[3] = {};
        };

        encoder.setComputePipeline(pipelineIt->second);

        for (uint32_t level = 0; level < m_levelCount; ++level) {
            RhiTexture* destinationTexture = m_frameGraph->getTexture(m_historyWrites[level]);
            if (!destinationTexture) {
                return;
            }

            HZBBuildUniforms uniforms{};
            uniforms.srcWidth = sourceTexture->width();
            uniforms.srcHeight = sourceTexture->height();
            uniforms.dstWidth = destinationTexture->width();
            uniforms.dstHeight = destinationTexture->height();
            uniforms.sourceScale = (level == 0u) ? 1u : 2u;

            encoder.setPushConstants(&uniforms, sizeof(uniforms));
            encoder.setTexture(sourceTexture, 0);
            encoder.setStorageTexture(destinationTexture, 1);
            encoder.dispatchThreadgroups({(uniforms.dstWidth + 7u) / 8u,
                                          (uniforms.dstHeight + 7u) / 8u,
                                          1u},
                                         {8u, 8u, 1u});
            m_frameGraph->commitHistory(m_historyWrites[level]);

            sourceTexture = destinationTexture;
            if (level + 1u < m_levelCount) {
                encoder.memoryBarrier(RhiBarrierScope::Textures);
            }
        }
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Levels: %u", m_levelCount);
        const bool historyValid =
            m_frameGraph && !m_historyWrites.empty() && m_frameGraph->isHistoryValid(m_historyWrites[0]);
        ImGui::Text("History Ready: %s", historyValid ? "Yes" : "No");
    }

private:
    int m_width = 0;
    int m_height = 0;
    uint32_t m_levelCount = 0;
    std::string m_name = "HZB Build";
    FGResource m_depthRead;
    std::vector<FGResource> m_historyWrites;
};
