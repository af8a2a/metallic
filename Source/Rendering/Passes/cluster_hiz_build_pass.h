#pragma once

#include "cluster_occlusion_state.h"
#include "frame_context.h"
#include "hzb_spd_constants.h"
#include "imgui.h"
#include "pass_registry.h"
#include "render_pass.h"

#include <algorithm>
#include <string>
#include <spdlog/spdlog.h>

class ClusterHizBuildPass : public RenderPass {
public:
    ClusterHizBuildPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(ClusterHizBuildPass, "Cluster HZB Build", "Geometry",
        (std::vector<PassSlotInfo>{
            makeInputSlot("depth", "Depth"),
            makeHiddenInputSlot("phaseReady", "Phase Ready", true)
        }),
        (std::vector<PassSlotInfo>{makeHiddenOutputSlot("hizReady", "HZB Ready", false)}),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (!config.config.is_object()) {
            return;
        }
        if (config.config.contains("targetPyramid")) {
            m_targetPyramid = std::clamp(config.config["targetPyramid"].get<uint32_t>(), 0u, 1u);
        }
    }

    FGResource getOutput(const std::string& outputName) const override {
        if (outputName == "hizReady") return m_hizReady;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_depthRead = FGResource{};
        if (FGResource depth = getInput("depth"); depth.isValid()) {
            m_depthRead = builder.read(depth, FGResourceUsage::Sampled);
        }
        if (FGResource ready = getInput("phaseReady"); ready.isValid()) {
            m_phaseReady = builder.read(ready);
        }
        m_hizReady = builder.createToken(m_name.c_str());
        builder.setQueueHint(RhiQueueHint::Graphics);
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("ClusterHizBuildPass");
        if (!m_frameContext || !m_runtimeContext || !m_runtimeContext->resourceFactory ||
            !m_depthRead.isValid()) {
            return;
        }

        ClusterOcclusionState* state = m_runtimeContext->clusterOcclusionState;
        if (!state) {
            return;
        }

        const uint32_t maxClusters = m_ctx.gpuScene.clusterVisWorklistCount;
        if (!state->ensure(*m_runtimeContext->resourceFactory,
                           static_cast<uint32_t>(std::max(m_width, 1)),
                           static_cast<uint32_t>(std::max(m_height, 1)),
                           maxClusters)) {
            return;
        }

        auto pipeIt = m_runtimeContext->computePipelinesRhi.find("ClusterHizBuild");
        if (pipeIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("ClusterHizBuildPass: compute pipeline not found");
                m_warnedMissingPipeline = true;
            }
            state->hizValid[m_targetPyramid] = false;
            return;
        }

        RhiTexture* depthTexture = m_frameGraph->getTexture(m_depthRead);
        RhiBuffer* atomicCounter = state->spdCounter();
        if (!depthTexture || !atomicCounter || state->mipCount == 0u ||
            !state->hizTexturesReady(m_targetPyramid)) {
            state->hizValid[m_targetPyramid] = false;
            return;
        }

        const SpdSetupInfo spdInfo = spdSetup(state->width, state->height);
        if (spdInfo.dispatchX == 0u || spdInfo.dispatchY == 0u || spdInfo.numWorkGroups == 0u) {
            state->hizValid[m_targetPyramid] = false;
            return;
        }

        ClusterHizSpdUniforms uniforms{};
        uniforms.mipCount = std::min(spdInfo.mipCount, state->mipCount - 1u);
        uniforms.numWorkGroups = spdInfo.numWorkGroups;
        uniforms.workGroupOffsetX = 0u;
        uniforms.workGroupOffsetY = 0u;
        uniforms.sourceWidth = state->width;
        uniforms.sourceHeight = state->height;

        encoder.setComputePipeline(pipeIt->second);
        encoder.setTexture(depthTexture, kHzbSpdSourceTextureBinding);
        for (uint32_t level = 0; level < state->mipCount; ++level) {
            RhiTexture* mipTexture = state->hizTexture(m_targetPyramid, level);
            if (mipTexture) {
                encoder.setStorageTexture(mipTexture, kHzbSpdMipTextureBindingBase + level);
            }
        }
        RhiTexture* mip6Texture =
            state->hizTexture(m_targetPyramid, std::min(kHzbSpdMip6Level, state->mipCount - 1u));
        if (mip6Texture) {
            encoder.setStorageTexture(mip6Texture, kHzbSpdMip6TextureBinding);
        }
        encoder.setBuffer(atomicCounter, 0, kHzbSpdAtomicCounterBinding);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({spdInfo.dispatchX, spdInfo.dispatchY, 1},
                                     {kSpdThreadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Textures | RhiBarrierScope::Buffers);

        state->hizValid[m_targetPyramid] = true;
    }

    void renderUI() override {
        ImGui::Text("Target Pyramid: %u", m_targetPyramid);
    }

private:
    struct ClusterHizSpdUniforms {
        uint32_t mipCount = 0;
        uint32_t numWorkGroups = 0;
        uint32_t workGroupOffsetX = 0;
        uint32_t workGroupOffsetY = 0;
        uint32_t sourceWidth = 0;
        uint32_t sourceHeight = 0;
        uint32_t _pad0 = 0;
        uint32_t _pad1 = 0;
    };
    static_assert(sizeof(ClusterHizSpdUniforms) <= 128,
                  "ClusterHizSpdUniforms must fit Vulkan push constants");

    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster HZB Build";
    uint32_t m_targetPyramid = 0;
    bool m_warnedMissingPipeline = false;
    FGResource m_depthRead;
    FGResource m_phaseReady;
    FGResource m_hizReady;
};

METALLIC_REGISTER_PASS(ClusterHizBuildPass);
