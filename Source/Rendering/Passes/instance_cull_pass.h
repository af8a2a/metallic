#pragma once

#include "cluster_occlusion_state.h"
#include "frame_context.h"
#include "imgui.h"
#include "pass_registry.h"
#include "render_pass.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <spdlog/spdlog.h>

namespace InstanceCullBindings {
static constexpr uint32_t kInstances = 0u;
static constexpr uint32_t kGeometries = 1u;
static constexpr uint32_t kVisibleInstances = 2u;
static constexpr uint32_t kCounters = 3u;
static constexpr uint32_t kIndirectArgs = 4u;
static constexpr uint32_t kVisibilityFlags = 5u;
#if METALLIC_RHI_METAL
static constexpr uint32_t kHizMipBase = 8u;
#else
static constexpr uint32_t kHizMipBase = 6u;
#endif

inline constexpr uint32_t bufferBinding(uint32_t binding) {
#if METALLIC_RHI_METAL
    return binding + 1u;
#else
    return binding;
#endif
}
} // namespace InstanceCullBindings

class InstanceCullPass : public RenderPass {
public:
    InstanceCullPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(InstanceCullPass, "Instance Cull", "Geometry",
        (std::vector<PassSlotInfo>{makeHiddenInputSlot("phaseReady", "Phase Ready", true)}),
        (std::vector<PassSlotInfo>{makeHiddenOutputSlot("cullDone", "Cull Done", false)}),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (!config.config.is_object()) return;
        if (config.config.contains("phase")) {
            m_phase = std::clamp(config.config["phase"].get<uint32_t>(), 0u, 1u);
        }
        if (config.config.contains("enableOcclusion")) {
            m_enableOcclusion = config.config["enableOcclusion"].get<bool>();
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
        ZoneScopedN("InstanceCullPass");
        if (!m_frameContext || !m_runtimeContext || !m_runtimeContext->resourceFactory) return;

        ClusterOcclusionState* state = m_runtimeContext->clusterOcclusionState;
        if (!state) return;

        const uint32_t instanceCount = m_ctx.gpuScene.instanceCount;
        if (instanceCount == 0u) return;

        if (!state->ensure(*m_runtimeContext->resourceFactory,
                           static_cast<uint32_t>(std::max(m_width, 1)),
                           static_cast<uint32_t>(std::max(m_height, 1)),
                           m_ctx.gpuScene.clusterVisWorklistCount,
                           instanceCount)) {
            return;
        }
        state->instanceVisibilityValid = false;
        state->instanceVisibilityFrameIndex = m_frameContext->frameIndex;

        auto resetIt = m_runtimeContext->computePipelinesRhi.find("InstanceCullReset");
        auto cullIt = m_runtimeContext->computePipelinesRhi.find("InstanceCullMain");
        auto finalizeIt = m_runtimeContext->computePipelinesRhi.find("InstanceCullFinalize");
        if (resetIt == m_runtimeContext->computePipelinesRhi.end() ||
            cullIt == m_runtimeContext->computePipelinesRhi.end() ||
            finalizeIt == m_runtimeContext->computePipelinesRhi.end() ||
            !resetIt->second.nativeHandle() ||
            !cullIt->second.nativeHandle() ||
            !finalizeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("InstanceCullPass: compute pipelines not found");
                m_warnedMissingPipeline = true;
            }
            return;
        }

        if (!m_ctx.gpuScene.instanceBuffer.nativeHandle() ||
            !m_ctx.gpuScene.geometryBuffer.nativeHandle() ||
            !state->visibleInstanceBuffer ||
            !state->instanceVisibilityBuffer ||
            !state->instanceCounters ||
            !state->instanceIndirectArgs) {
            return;
        }

        InstanceCullUniforms uniforms = makeUniforms(*state, instanceCount);

        // Reset counters
        encoder.setComputePipeline(resetIt->second);
        bindBuffers(encoder, *state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Cull instances
        encoder.setComputePipeline(cullIt->second);
        bindBuffers(encoder, *state);
        bindHizTextures(encoder, *state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        uint32_t groups = (instanceCount + 63u) / 64u;
        encoder.dispatchThreadgroups({groups, 1, 1}, {64, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Finalize: write indirect args
        encoder.setComputePipeline(finalizeIt->second);
        bindBuffers(encoder, *state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        state->instanceVisibilityValid = true;
    }

    void renderUI() override {
        ImGui::Text("Phase: %u", m_phase);
        ImGui::Text("Hard Filter: Conservative frustum");
        ImGui::Text("HZB: %s", m_enableOcclusion ? "Diagnostic only" : "Disabled");

        const uint32_t instanceCount = m_ctx.gpuScene.instanceCount;
        const uint32_t sceneVisible = m_ctx.gpuScene.visibleInstanceCount;
        ImGui::Separator();
        ImGui::Text("Instances: %u total, %u scene-visible", instanceCount, sceneVisible);

        ClusterOcclusionState* state =
            m_runtimeContext ? m_runtimeContext->clusterOcclusionState : nullptr;
        if (!state) {
            ImGui::TextDisabled("Stats unavailable: no occlusion state");
            return;
        }

        ImGui::Text("State: %s (buffer frame %u, current frame %u)",
                    state->instanceVisibilityValid ? "valid" : "invalid",
                    state->instanceVisibilityFrameIndex,
                    m_frameContext ? m_frameContext->frameIndex : 0u);
        ImGui::Text("Capacity: %u instances", state->maxInstances);

        ClusterOcclusionState::InstanceCullStats stats = state->readInstanceCullStats();
        if (!stats.countersReadable) {
            ImGui::TextDisabled("GPU counters unavailable");
            return;
        }

        const uint32_t phase0Culled = sceneVisible > stats.phase0Visible
            ? sceneVisible - stats.phase0Visible
            : 0u;
        const float visiblePct = sceneVisible > 0u
            ? 100.0f * static_cast<float>(stats.phase0Visible) / static_cast<float>(sceneVisible)
            : 0.0f;
        const float culledPct = sceneVisible > 0u
            ? 100.0f * static_cast<float>(phase0Culled) / static_cast<float>(sceneVisible)
            : 0.0f;

        ImGui::SeparatorText("Last GPU Counters");
        ImGui::Text("Phase0 visible: %u (%.1f%%)", stats.phase0Visible, visiblePct);
        ImGui::Text("Phase0 rejected: %u", stats.phase0Rejected);
        ImGui::Text("Phase0 culled vs scene-visible: %u (%.1f%%)", phase0Culled, culledPct);
        ImGui::Text("Phase1 visible: %u", stats.phase1Visible);
        if (stats.indirectReadable) {
            ImGui::Text("Indirect dispatch groups: %u", stats.dispatchGroups);
        } else {
            ImGui::TextDisabled("Indirect dispatch groups unavailable");
        }
    }

private:
    struct InstanceCullUniforms {
        float viewProj[16];
        uint32_t instanceCount = 0;
        uint32_t phase = 0;
        uint32_t enableOcclusion = 0;
        uint32_t hzbValid = 0;
        uint32_t hzbWidth = 0;
        uint32_t hzbHeight = 0;
        uint32_t hzbMipCount = 0;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t maxInstances = 0;
        uint32_t _pad0 = 0;
        uint32_t _pad1 = 0;
    };
    static_assert(sizeof(InstanceCullUniforms) <= 128,
                  "InstanceCullUniforms must fit push constants");

    InstanceCullUniforms makeUniforms(const ClusterOcclusionState& state,
                                      uint32_t instanceCount) const {
        InstanceCullUniforms u{};
        float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
        float4x4 vpT = transpose(vp);
        std::memcpy(u.viewProj, &vpT, sizeof(u.viewProj));
        u.instanceCount = instanceCount;
        u.phase = m_phase;
        u.enableOcclusion = m_enableOcclusion ? 1u : 0u;
        const uint32_t hzbPyramid = m_phase == 0u ? 0u : 1u;
        u.hzbValid = (m_enableOcclusion &&
                      state.hizValid[hzbPyramid] &&
                      state.hizTexturesReady(hzbPyramid)) ? 1u : 0u;
        u.hzbWidth = state.width;
        u.hzbHeight = state.height;
        u.hzbMipCount = state.mipCount;
        u.screenWidth = static_cast<uint32_t>(std::max(m_width, 1));
        u.screenHeight = static_cast<uint32_t>(std::max(m_height, 1));
        u.maxInstances = state.maxInstances;
        return u;
    }

    void bindBuffers(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        encoder.setBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kInstances));
        encoder.setBuffer(&m_ctx.gpuScene.geometryBuffer, 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kGeometries));
        encoder.setBuffer(state.visibleInstanceBuffer.get(), 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kVisibleInstances));
        encoder.setBuffer(state.instanceCounters.get(), 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kCounters));
        encoder.setBuffer(state.instanceIndirectArgs.get(), 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kIndirectArgs));
        encoder.setBuffer(state.instanceVisibilityBuffer.get(), 0,
                          InstanceCullBindings::bufferBinding(InstanceCullBindings::kVisibilityFlags));
    }

    void bindHizTextures(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        const uint32_t pyramid = m_phase == 0u ? 0u : 1u;
        for (uint32_t level = 0; level < state.mipCount; ++level) {
            RhiTexture* texture = state.hizTexture(pyramid, level);
            if (texture) {
                encoder.setTexture(texture, InstanceCullBindings::kHizMipBase + level);
            }
        }
    }

    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Instance Cull";
    uint32_t m_phase = 0;
    bool m_enableOcclusion = true;
    bool m_warnedMissingPipeline = false;
    FGResource m_phaseReady;
    FGResource m_cullDone;
};

METALLIC_REGISTER_PASS(InstanceCullPass);
