#pragma once

#include "cluster_lod_builder.h"
#include "cluster_occlusion_state.h"
#include "frame_context.h"
#include "imgui.h"
#include "pass_registry.h"
#include "render_pass.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <spdlog/spdlog.h>

namespace ClusterCullBindings {
static constexpr uint32_t kInputWorklist = 0u;
static constexpr uint32_t kPhase0Visible = 1u;
static constexpr uint32_t kPhase0Recheck = 2u;
static constexpr uint32_t kPhase1Visible = 3u;
static constexpr uint32_t kBounds = 4u;
static constexpr uint32_t kInstances = 5u;
static constexpr uint32_t kCounters = 6u;
static constexpr uint32_t kIndirectArgs = 7u;
static constexpr uint32_t kInstanceVisibility = 8u;
#if METALLIC_RHI_METAL
static constexpr uint32_t kHizMipBase = 8u;
static constexpr uint32_t kHzbViewProj = 9u;
#else
static constexpr uint32_t kHizMipBase = 9u;
static constexpr uint32_t kHzbViewProj = 10u;
#endif

inline constexpr uint32_t bufferBinding(uint32_t binding) {
#if METALLIC_RHI_METAL
    return binding + 1u;
#else
    return binding;
#endif
}
} // namespace ClusterCullBindings

class ClusterCullPass : public RenderPass {
public:
    ClusterCullPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(ClusterCullPass, "Cluster Cull", "Geometry",
        (std::vector<PassSlotInfo>{makeHiddenInputSlot("phaseReady", "Phase Ready", true)}),
        (std::vector<PassSlotInfo>{makeHiddenOutputSlot("cullDone", "Cull Done", false)}),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (!config.config.is_object()) {
            return;
        }
        if (config.config.contains("phase")) {
            m_phase = std::clamp(config.config["phase"].get<uint32_t>(), 0u, 1u);
        }
        if (config.config.contains("enableOcclusion")) {
            m_enableOcclusion = config.config["enableOcclusion"].get<bool>();
        }
        if (config.config.contains("enableInstanceFilter")) {
            m_enableInstanceFilter = config.config["enableInstanceFilter"].get<bool>();
        }
    }

    FGResource getOutput(const std::string& outputName) const override {
        if (outputName == "cullDone") return m_cullDone;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        if (FGResource ready = getInput("phaseReady"); ready.isValid()) {
            m_phaseReady = builder.read(ready);
        }
        m_cullDone = builder.createToken(m_name.c_str());
        builder.setQueueHint(RhiQueueHint::Graphics);
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("ClusterCullPass");
        if (!m_frameContext || !m_runtimeContext || !m_runtimeContext->resourceFactory) {
            return;
        }

        ClusterOcclusionState* state = m_runtimeContext->clusterOcclusionState;
        if (!state) {
            return;
        }

        if (m_phase == 0u && m_frameContext->historyReset) {
            state->resetHistory();
        }
        if (m_phase == 0u) {
            state->resetWorklists();
            // Phase0: HZB uses history pyramid. Supply its VP for correct temporal projection.
            if (m_enableOcclusion) {
                state->updateHzbViewProj(state->hizViewProj[0]);
            } else {
                float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
                float4x4 vpT = transpose(vp);
                state->updateHzbViewProj(vpT.a);
            }
        } else {
            // Phase1: HZB uses current-frame pyramid. Supply current VP.
            float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
            float4x4 vpT = transpose(vp);
            state->updateHzbViewProj(vpT.a);
        }

        const uint32_t inputCount = m_ctx.gpuScene.clusterVisWorklistCount;
        if (!state->ensure(*m_runtimeContext->resourceFactory,
                           static_cast<uint32_t>(std::max(m_width, 1)),
                           static_cast<uint32_t>(std::max(m_height, 1)),
                           inputCount)) {
            return;
        }

        auto resetIt = m_runtimeContext->computePipelinesRhi.find("ClusterCullReset");
        auto cullIt = m_runtimeContext->computePipelinesRhi.find("ClusterCullMain");
        auto finalizeIt = m_runtimeContext->computePipelinesRhi.find("ClusterCullFinalize");
        if (resetIt == m_runtimeContext->computePipelinesRhi.end() ||
            cullIt == m_runtimeContext->computePipelinesRhi.end() ||
            finalizeIt == m_runtimeContext->computePipelinesRhi.end() ||
            !resetIt->second.nativeHandle() ||
            !cullIt->second.nativeHandle() ||
            !finalizeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("ClusterCullPass: compute pipelines not found");
                m_warnedMissingPipeline = true;
            }
            state->worklistValid[m_phase] = false;
            return;
        }

        if (!m_ctx.gpuScene.clusterVisWorklistBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.boundsBuffer.nativeHandle() ||
            !m_ctx.gpuScene.instanceBuffer.nativeHandle() ||
            !state->phase0VisibleWorklist ||
            !state->phase0RecheckWorklist ||
            !state->phase1VisibleWorklist ||
            !state->counters ||
            !state->indirectArgs ||
            !state->instanceVisibilityBuffer ||
            !state->hzbViewProjBuffer) {
            state->worklistValid[m_phase] = false;
            return;
        }

        ClusterCullUniforms uniforms = makeUniforms(*state, inputCount);

        encoder.setComputePipeline(resetIt->second);
        bindBuffers(encoder, *state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        if (uniforms.dispatchCount > 0u) {
            encoder.setComputePipeline(cullIt->second);
            bindBuffers(encoder, *state);
            bindHizTextures(encoder, *state);
            encoder.setPushConstants(&uniforms, sizeof(uniforms));
            encoder.dispatchThreadgroups({(uniforms.dispatchCount + 63u) / 64u, 1, 1}, {64, 1, 1});
            encoder.memoryBarrier(RhiBarrierScope::Buffers);
        }

        encoder.setComputePipeline(finalizeIt->second);
        bindBuffers(encoder, *state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        state->worklistValid[m_phase] = true;
    }

    void renderUI() override {
        ImGui::Text("Phase: %u", m_phase);
        ImGui::Text("Instance Filter: %s", m_enableInstanceFilter ? "Enabled" : "Disabled");
        ImGui::Text("Occlusion: %s", m_enableOcclusion ? "Enabled" : "Disabled");
    }

private:
    struct ClusterCullUniforms {
        float viewProj[16];
        uint32_t inputCount = 0;
        uint32_t dispatchCount = 0;
        uint32_t maxClusters = 0;
        uint32_t phase = 0;
        uint32_t enableOcclusion = 0;
        uint32_t hzbValid = 0;
        uint32_t hzbWidth = 0;
        uint32_t hzbHeight = 0;
        uint32_t hzbMipCount = 0;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t instanceCount = 0;
        uint32_t useInstanceVisibility = 0;
    };
    static_assert(sizeof(ClusterCullUniforms) <= 128,
                  "ClusterCullUniforms must fit Vulkan push constants");

    ClusterCullUniforms makeUniforms(const ClusterOcclusionState& state,
                                     uint32_t inputCount) const {
        ClusterCullUniforms uniforms{};
        float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
        float4x4 vpT = transpose(vp);
        std::memcpy(uniforms.viewProj, &vpT, sizeof(uniforms.viewProj));
        uniforms.inputCount = inputCount;
        uniforms.dispatchCount = m_phase == 0u ? inputCount : state.maxClusters;
        uniforms.maxClusters = state.maxClusters;
        uniforms.phase = m_phase;
        uniforms.enableOcclusion = m_enableOcclusion ? 1u : 0u;
        const uint32_t hzbPyramid = m_phase == 0u ? 0u : 1u;
        uniforms.hzbValid =
            (m_enableOcclusion &&
             state.hizValid[hzbPyramid] &&
             state.hizTexturesReady(hzbPyramid)) ? 1u : 0u;
        uniforms.hzbWidth = state.width;
        uniforms.hzbHeight = state.height;
        uniforms.hzbMipCount = state.mipCount;
        uniforms.screenWidth = static_cast<uint32_t>(std::max(m_width, 1));
        uniforms.screenHeight = static_cast<uint32_t>(std::max(m_height, 1));
        uniforms.instanceCount = m_ctx.gpuScene.instanceCount;
        uniforms.useInstanceVisibility =
            (m_enableInstanceFilter &&
             state.instanceVisibilityValid &&
             state.instanceVisibilityFrameIndex == m_frameContext->frameIndex &&
             state.instanceVisibilityBuffer) ? 1u : 0u;
        return uniforms;
    }

    void bindBuffers(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        const RhiBuffer* inputWorklist =
            m_phase == 0u
                ? static_cast<const RhiBuffer*>(&m_ctx.gpuScene.clusterVisWorklistBuffer)
                : state.recheckWorklist();

        encoder.setBuffer(inputWorklist, 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kInputWorklist));
        encoder.setBuffer(state.phase0VisibleWorklist.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kPhase0Visible));
        encoder.setBuffer(state.phase0RecheckWorklist.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kPhase0Recheck));
        encoder.setBuffer(state.phase1VisibleWorklist.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kPhase1Visible));
        encoder.setBuffer(&m_ctx.clusterLodData.boundsBuffer, 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kBounds));
        encoder.setBuffer(&m_ctx.gpuScene.instanceBuffer, 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kInstances));
        encoder.setBuffer(state.counters.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kCounters));
        encoder.setBuffer(state.indirectArgs.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kIndirectArgs));
        encoder.setBuffer(state.instanceVisibilityBuffer.get(), 0, ClusterCullBindings::bufferBinding(ClusterCullBindings::kInstanceVisibility));
    }

    void bindHizTextures(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        const uint32_t pyramid = m_phase == 0u ? 0u : 1u;
        for (uint32_t level = 0; level < state.mipCount; ++level) {
            RhiTexture* texture = state.hizTexture(pyramid, level);
            if (texture) {
                encoder.setTexture(texture, ClusterCullBindings::kHizMipBase + level);
            }
        }
        encoder.setBuffer(state.hzbViewProjBuffer.get(), 0,
                          ClusterCullBindings::bufferBinding(ClusterCullBindings::kHzbViewProj));
    }

    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Cull";
    uint32_t m_phase = 0;
    bool m_enableOcclusion = true;
    bool m_enableInstanceFilter = false;
    bool m_warnedMissingPipeline = false;
    FGResource m_phaseReady;
    FGResource m_cullDone;
};

METALLIC_REGISTER_PASS(ClusterCullPass);
