#pragma once

#include "frame_context.h"
#include "hzb_constants.h"
#include "hzb_spd_constants.h"
#include "pass_registry.h"
#include "render_pass.h"

#include "imgui.h"

#include <vector>

class HZBBuildPass : public RenderPass {
public:
    HZBBuildPass(const RenderContext&, int w, int h)
        : m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(HZBBuildPass, "HZB Build", "Geometry",
        (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("hzb0", "HZB 0", true),
            makeOutputSlot("hzb1", "HZB 1", true),
            makeOutputSlot("hzb2", "HZB 2", true),
            makeOutputSlot("hzb3", "HZB 3", true),
            makeOutputSlot("hzb4", "HZB 4", true),
            makeOutputSlot("hzb5", "HZB 5", true),
            makeOutputSlot("hzb6", "HZB 6", true),
            makeOutputSlot("hzb7", "HZB 7", true),
            makeOutputSlot("hzb8", "HZB 8", true),
            makeOutputSlot("hzb9", "HZB 9", true),
            makeOutputSlot("hzb10", "HZB 10", true),
            makeOutputSlot("hzb11", "HZB 11", true),
            makeOutputSlot("hzb12", "HZB 12", true)
        }),
        PassTypeInfo::PassType::Compute);

    METALLIC_PASS_EDITOR_TYPE_INFO(HZBBuildPass, "HZB Build", "Geometry",
        (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
        (std::vector<PassSlotInfo>{
            makeHiddenOutputSlot("hzb0", "HZB 0", true),
            makeHiddenOutputSlot("hzb1", "HZB 1", true),
            makeHiddenOutputSlot("hzb2", "HZB 2", true),
            makeHiddenOutputSlot("hzb3", "HZB 3", true),
            makeHiddenOutputSlot("hzb4", "HZB 4", true),
            makeHiddenOutputSlot("hzb5", "HZB 5", true),
            makeHiddenOutputSlot("hzb6", "HZB 6", true),
            makeHiddenOutputSlot("hzb7", "HZB 7", true),
            makeHiddenOutputSlot("hzb8", "HZB 8", true),
            makeHiddenOutputSlot("hzb9", "HZB 9", true),
            makeHiddenOutputSlot("hzb10", "HZB 10", true),
            makeHiddenOutputSlot("hzb11", "HZB 11", true),
            makeHiddenOutputSlot("hzb12", "HZB 12", true)
        }),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.is_object()) {
            if (config.config.contains("writeHistory")) {
                m_writeHistory = config.config["writeHistory"].get<bool>();
            }
            if (config.config.contains("publishOutputs")) {
                m_publishOutputs = config.config["publishOutputs"].get<bool>();
            }
        }
    }

    FGResource getOutput(const std::string& name) const override {
        for (uint32_t level = 0; level < m_outputWrites.size(); ++level) {
            if (name == hzbOutputSlotName(level)) {
                return m_outputWrites[level];
            }
        }
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_depthRead = FGResource{};
        m_historyWrites.clear();
        m_outputWrites.clear();

        FGResource depthInput = getInput("depth");
        if (depthInput.isValid()) {
            m_depthRead = builder.read(depthInput);
        } else {
            m_levelCount = 0;
            return;
        }

        SpdSetupInfo spdInfo = spdSetup(static_cast<uint32_t>(m_width),
                                         static_cast<uint32_t>(m_height));
        m_levelCount = std::min(spdInfo.mipCount + 1u, kHzbMaxLevels);

        static const uint32_t kZeroCounter = 0;
        FGBufferDesc counterDesc;
        counterDesc.size = sizeof(uint32_t);
        counterDesc.initialData = &kZeroCounter;
        counterDesc.hostVisible = false;
        counterDesc.debugName = "HZB SPD Atomic Counter";
        m_atomicCounter = builder.create("HZBSpdAtomicCounter", counterDesc);

        if (m_publishOutputs) {
            m_outputWrites.reserve(m_levelCount);
            for (uint32_t level = 0; level < m_levelCount; ++level) {
                m_outputWrites.push_back(
                    builder.create(hzbOutputResourceName(level).c_str(),
                                   makeHzbTextureDesc(static_cast<uint32_t>(m_width),
                                                      static_cast<uint32_t>(m_height),
                                                      level)));
            }
        }

        if (!m_writeHistory) {
            return;
        }

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

        RhiBuffer* counterBuffer = m_frameGraph->getBuffer(m_atomicCounter);
        if (!counterBuffer) {
            return;
        }

        const uint32_t srcWidth = sourceTexture->width();
        const uint32_t srcHeight = sourceTexture->height();
        SpdSetupInfo spdInfo = spdSetup(srcWidth, srcHeight);
        spdInfo.mipCount = std::min(spdInfo.mipCount, m_levelCount > 0u ? m_levelCount - 1u : 0u);

        struct HZBSpdUniforms {
            uint32_t mipCount = 0;
            uint32_t numWorkGroups = 0;
            uint32_t workGroupOffsetX = 0;
            uint32_t workGroupOffsetY = 0;
            uint32_t sourceWidth = 0;
            uint32_t sourceHeight = 0;
            uint32_t _pad0 = 0;
            uint32_t _pad1 = 0;
        };

        HZBSpdUniforms uniforms{};
        uniforms.mipCount = spdInfo.mipCount;
        uniforms.numWorkGroups = spdInfo.numWorkGroups;
        uniforms.sourceWidth = srcWidth;
        uniforms.sourceHeight = srcHeight;

        encoder.setComputePipeline(pipelineIt->second);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.setTexture(sourceTexture, kHzbSpdSourceTextureBinding);

        for (uint32_t level = 0; level < m_levelCount; ++level) {
            FGResource destResource =
                m_publishOutputs ? m_outputWrites[level] : m_historyWrites[level];
            if (!destResource.isValid()) return;
            RhiTexture* destTexture = m_frameGraph->getTexture(destResource);
            if (!destTexture) return;
            encoder.setStorageTexture(destTexture, kHzbSpdMipTextureBindingBase + level);
            if (level == kHzbSpdMip6Level) {
                encoder.setStorageTexture(destTexture, kHzbSpdMip6TextureBinding);
            }
        }

        encoder.setBuffer(counterBuffer, 0, kHzbSpdAtomicCounterBinding);

        encoder.dispatchThreadgroups(
            {spdInfo.dispatchX, spdInfo.dispatchY, 1u},
            {kSpdThreadgroupSize, 1u, 1u});

        if (m_writeHistory && !m_publishOutputs) {
            for (uint32_t level = 0; level < m_levelCount; ++level) {
                m_frameGraph->commitHistory(m_historyWrites[level]);
            }
        }
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Levels: %u", m_levelCount);
        ImGui::Text("Publish Outputs: %s", m_publishOutputs ? "Yes" : "No");
        ImGui::Text("Write History: %s", m_writeHistory ? "Yes" : "No");
        const bool historyValid =
            m_frameGraph && !m_historyWrites.empty() && m_frameGraph->isHistoryValid(m_historyWrites[0]);
        ImGui::Text("History Ready: %s", historyValid ? "Yes" : "No");
    }

private:
    static std::string hzbOutputSlotName(uint32_t level) {
        return "hzb" + std::to_string(level);
    }

    static std::string hzbOutputResourceName(uint32_t level) {
        return "Current HZB Mip " + std::to_string(level);
    }

    int m_width = 0;
    int m_height = 0;
    uint32_t m_levelCount = 0;
    bool m_publishOutputs = false;
    bool m_writeHistory = true;
    std::string m_name = "HZB Build";
    FGResource m_depthRead;
    FGResource m_atomicCounter;
    std::vector<FGResource> m_historyWrites;
    std::vector<FGResource> m_outputWrites;
};

METALLIC_REGISTER_PASS(HZBBuildPass);
