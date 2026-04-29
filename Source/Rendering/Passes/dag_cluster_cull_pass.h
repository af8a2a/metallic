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

namespace DagClusterCullBindings {
static constexpr uint32_t kInstances = 0u;
static constexpr uint32_t kGeometries = 1u;
static constexpr uint32_t kBounds = 2u;
static constexpr uint32_t kNodes = 3u;
static constexpr uint32_t kGroups = 4u;
static constexpr uint32_t kGroupMeshletIndices = 5u;
static constexpr uint32_t kPhase0Visible = 6u;
static constexpr uint32_t kPhase0Recheck = 7u;
static constexpr uint32_t kPhase1Visible = 8u;
static constexpr uint32_t kCounters = 9u;
static constexpr uint32_t kIndirectArgs = 10u;
static constexpr uint32_t kInstanceVisibility = 11u;
static constexpr uint32_t kNodeQueue0 = 12u;
static constexpr uint32_t kNodeQueue1 = 13u;
static constexpr uint32_t kHizMipBase = 14u;
static constexpr uint32_t kHzbViewProj = 15u;

inline constexpr uint32_t bufferBinding(uint32_t binding) {
#if METALLIC_RHI_METAL
    return binding + 1u;
#else
    return binding;
#endif
}
} // namespace DagClusterCullBindings

class DagClusterCullPass : public RenderPass {
public:
    DagClusterCullPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(DagClusterCullPass, "DAG Cluster Cull", "Geometry",
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
        if (config.config.contains("enableFrustum")) {
            m_enableFrustumOverride =
                config.config["enableFrustum"].get<bool>() ? 1 : 0;
        }
        if (config.config.contains("maxIterations")) {
            m_maxIterations = std::clamp(config.config["maxIterations"].get<uint32_t>(), 1u, 64u);
        }
        // Reset UI override so the JSON value takes effect on pipeline reload.
        m_enableOcclusionOverride = -1;
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
        ZoneScopedN("DagClusterCullPass");
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
            // Phase0: HZB uses history pyramid (pyramid 0). Supply its VP so the shader
            // can project bounds against the pyramid that was actually built.
            if (enableOcclusion()) {
                state->updateHzbViewProj(state->hizViewProj[0]);
            } else {
                float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
                float4x4 vpT = transpose(vp);
                state->updateHzbViewProj(vpT.a);
            }
        } else {
            // Phase1: HZB uses current-frame pyramid (pyramid 1). Supply current VP.
            float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
            float4x4 vpT = transpose(vp);
            state->updateHzbViewProj(vpT.a);
        }

        const uint32_t maxClusters = m_ctx.gpuScene.clusterVisWorklistCount;
        const uint32_t instanceCount = m_ctx.gpuScene.instanceCount;
        if (!state->ensure(*m_runtimeContext->resourceFactory,
                           static_cast<uint32_t>(std::max(m_width, 1)),
                           static_cast<uint32_t>(std::max(m_height, 1)),
                           maxClusters,
                           instanceCount)) {
            return;
        }

        auto resetIt = m_runtimeContext->computePipelinesRhi.find("DagClusterCullReset");
        auto mainIt = m_runtimeContext->computePipelinesRhi.find("DagClusterCullMain");
        auto finalizeIt = m_runtimeContext->computePipelinesRhi.find("DagClusterCullFinalize");
        if (resetIt == m_runtimeContext->computePipelinesRhi.end() ||
            mainIt == m_runtimeContext->computePipelinesRhi.end() ||
            finalizeIt == m_runtimeContext->computePipelinesRhi.end() ||
            !resetIt->second.nativeHandle() ||
            !mainIt->second.nativeHandle() ||
            !finalizeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("DagClusterCullPass: compute pipelines not found");
                m_warnedMissingPipeline = true;
            }
            state->worklistValid[m_phase] = false;
            return;
        }

        if (!m_ctx.gpuScene.instanceBuffer.nativeHandle() ||
            !m_ctx.gpuScene.geometryBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.boundsBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.nodeBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.groupBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.groupMeshletIndicesBuffer.nativeHandle() ||
            !state->phase0VisibleWorklist ||
            !state->phase0RecheckWorklist ||
            !state->phase1VisibleWorklist ||
            !state->counters ||
            !state->indirectArgs ||
            !state->instanceVisibilityBuffer ||
            !state->dagNodeQueue(0u) ||
            !state->dagNodeQueue(1u) ||
            !state->hzbViewProjBuffer) {
            state->worklistValid[m_phase] = false;
            return;
        }

        DagCullUniforms uniforms = makeUniforms(*state, instanceCount);

        encoder.setComputePipeline(resetIt->second);
        bindBuffers(encoder, *state);
        uniforms.mode = kModeReset;
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        encoder.setComputePipeline(mainIt->second);
        if (m_phase == 0u) {
            dispatchMain(encoder,
                         *state,
                         uniforms,
                         kModeSeedNodes,
                         0u,
                         {(instanceCount + 63u) / 64u, 1, 1},
                         {64, 1, 1});

            const uint32_t maxNodeTasks = std::max(state->maxClusters, 1u);
            for (uint32_t iteration = 0; iteration < m_maxIterations; ++iteration) {
                dispatchMain(encoder,
                             *state,
                             uniforms,
                             kModeClearNextQueue,
                             iteration,
                             {1, 1, 1},
                             {64, 1, 1});
                dispatchMain(encoder,
                             *state,
                             uniforms,
                             kModeProcessNodes,
                             iteration,
                             {(maxNodeTasks + 63u) / 64u, 1, 1},
                             {64, 1, 1});
            }
        } else {
            dispatchMain(encoder,
                         *state,
                         uniforms,
                         kModeProcessRecheck,
                         0u,
                         {(state->maxClusters + 63u) / 64u, 1, 1},
                         {64, 1, 1});
        }

        encoder.setComputePipeline(finalizeIt->second);
        bindBuffers(encoder, *state);
        uniforms.mode = kModeFinalize;
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        state->worklistValid[m_phase] = true;
    }

    void renderUI() override {
        ImGui::Text("Phase: %u", m_phase);
        ImGui::Text("Instance Filter: %s", m_enableInstanceFilter ? "Enabled" : "Disabled");
        ImGui::Text("Frustum: %s", enableFrustumCull() ? "Enabled" : "Disabled");
        ImGui::Text("Max Iterations: %u", m_maxIterations);

        // HZB toggle — overrides the JSON config value for live testing.
        {
            bool occOn = enableOcclusion();
            if (ImGui::Checkbox("HZB Occlusion##dag", &occOn)) {
                m_enableOcclusionOverride = occOn ? 1 : 0;
            }
            if (m_enableOcclusionOverride >= 0) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Reset##dagOcc")) {
                    m_enableOcclusionOverride = -1;
                }
            }
        }

        ClusterOcclusionState* state =
            m_runtimeContext ? m_runtimeContext->clusterOcclusionState : nullptr;
        if (!state) {
            ImGui::TextDisabled("Stats unavailable: no occlusion state");
            return;
        }

        ClusterOcclusionState::DagCullStats stats = state->readDagCullStats();
        if (!stats.readable) {
            ImGui::TextDisabled("GPU counters unavailable");
            return;
        }

        // HZB pyramid validity
        const uint32_t hzbPyramid = m_phase == 0u ? 0u : 1u;
        const bool hzbReady = state->hizValid[hzbPyramid] && state->hizTexturesReady(hzbPyramid);
        if (enableOcclusion()) {
            if (hzbReady) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                                   "HZB pyramid %u: valid", hzbPyramid);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.1f, 1.0f),
                                   "HZB pyramid %u: stale/invalid", hzbPyramid);
            }
        }

        ImGui::SeparatorText("Last GPU Counters");
        ImGui::Text("Seeded instances: %u", stats.seededInstances);
        if (stats.invalidRoot > 0u) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.1f, 1.0f),
                               "Invalid root: %u", stats.invalidRoot);
        } else {
            ImGui::Text("Invalid root: 0");
        }
        ImGui::Text("Processed nodes: %u", stats.nodeProcessed);
        ImGui::Text("Queue counts: %u / %u", stats.queueCount0, stats.queueCount1);
        ImGui::Text("Phase0 visible/recheck: %u / %u",
                    stats.phase0Visible,
                    stats.phase0Recheck);
        ImGui::Text("Phase1 visible: %u", stats.phase1Visible);

        ImGui::SeparatorText("Cluster Cull");
        ImGui::Text("Frustum rejected: %u", stats.frustumRejected);
        ImGui::Text("HZB recheck (phase0): %u", stats.hzbRecheck);
        ImGui::Text("HZB rejected (phase1): %u", stats.hzbRejected);

        ImGui::SeparatorText("Traversal Health");
        if (stats.maxIterRemaining > 0u) {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                               "Iter remaining: %u (early exit)", stats.maxIterRemaining);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.1f, 1.0f),
                               "Iter remaining: 0 (hit limit %u)", m_maxIterations);
        }
        const uint32_t overflow = stats.nodeOverflow + stats.clusterOverflow;
        if (overflow > 0u) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.1f, 1.0f),
                               "Overflow: nodes %u, clusters %u",
                               stats.nodeOverflow,
                               stats.clusterOverflow);
        } else {
            ImGui::Text("Overflow: 0");
        }
    }

private:
    static constexpr uint32_t kModeReset = 0u;
    static constexpr uint32_t kModeSeedNodes = 1u;
    static constexpr uint32_t kModeClearNextQueue = 2u;
    static constexpr uint32_t kModeProcessNodes = 3u;
    static constexpr uint32_t kModeProcessRecheck = 4u;
    static constexpr uint32_t kModeFinalize = 5u;

    struct DagCullUniforms {
        float viewProj[16];
        uint32_t instanceCount = 0;
        uint32_t maxClusters = 0;
        uint32_t maxNodeTasks = 0;
        uint32_t phase = 0;
        uint32_t mode = 0;
        uint32_t iteration = 0;
        uint32_t enableOcclusion = 0;
        uint32_t hzbValid = 0;
        uint32_t hzbWidth = 0;
        uint32_t hzbHeight = 0;
        uint32_t hzbMipCount = 0;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t useInstanceVisibility = 0;
        uint32_t enableFrustum = 0;
        uint32_t maxIterations = 0;
    };
    static_assert(sizeof(DagCullUniforms) <= 128,
                  "DagCullUniforms must fit Vulkan push constants");

    bool enableFrustumCull() const {
        if (m_enableFrustumOverride >= 0) {
            return m_enableFrustumOverride != 0;
        }
        return m_frameContext ? m_frameContext->enableFrustumCull : true;
    }

    bool enableOcclusion() const {
        if (m_enableOcclusionOverride >= 0) {
            return m_enableOcclusionOverride != 0;
        }
        return m_enableOcclusion;
    }

    DagCullUniforms makeUniforms(const ClusterOcclusionState& state,
                                 uint32_t instanceCount) const {
        DagCullUniforms uniforms{};
        float4x4 vp = m_frameContext->unjitteredProj * m_frameContext->view;
        float4x4 vpT = transpose(vp);
        std::memcpy(uniforms.viewProj, &vpT, sizeof(uniforms.viewProj));
        uniforms.instanceCount = instanceCount;
        uniforms.maxClusters = state.maxClusters;
        uniforms.maxNodeTasks = state.maxClusters;
        uniforms.phase = m_phase;
        uniforms.enableOcclusion = enableOcclusion() ? 1u : 0u;
        const uint32_t hzbPyramid = m_phase == 0u ? 0u : 1u;
        uniforms.hzbValid =
            (enableOcclusion() &&
             state.hizValid[hzbPyramid] &&
             state.hizTexturesReady(hzbPyramid)) ? 1u : 0u;
        uniforms.hzbWidth = state.width;
        uniforms.hzbHeight = state.height;
        uniforms.hzbMipCount = state.mipCount;
        uniforms.screenWidth = static_cast<uint32_t>(std::max(m_width, 1));
        uniforms.screenHeight = static_cast<uint32_t>(std::max(m_height, 1));
        uniforms.useInstanceVisibility =
            (m_enableInstanceFilter &&
             state.instanceVisibilityValid &&
             state.instanceVisibilityFrameIndex == m_frameContext->frameIndex &&
             state.instanceVisibilityBuffer) ? 1u : 0u;
        uniforms.enableFrustum = enableFrustumCull() ? 1u : 0u;
        uniforms.maxIterations = m_maxIterations;
        return uniforms;
    }

    void dispatchMain(RhiComputeCommandEncoder& encoder,
                      ClusterOcclusionState& state,
                      DagCullUniforms uniforms,
                      uint32_t mode,
                      uint32_t iteration,
                      RhiSize3D threadgroups,
                      RhiSize3D threadsPerThreadgroup) const {
        if (threadgroups.width == 0u || threadgroups.height == 0u || threadgroups.depth == 0u) {
            return;
        }
        uniforms.mode = mode;
        uniforms.iteration = iteration;
        bindBuffers(encoder, state);
        bindHizTextures(encoder, state);
        encoder.setPushConstants(&uniforms, sizeof(uniforms));
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup);
        encoder.memoryBarrier(RhiBarrierScope::Buffers);
    }

    void bindBuffers(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        encoder.setBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kInstances));
        encoder.setBuffer(&m_ctx.gpuScene.geometryBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kGeometries));
        encoder.setBuffer(&m_ctx.clusterLodData.boundsBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kBounds));
        encoder.setBuffer(&m_ctx.clusterLodData.nodeBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kNodes));
        encoder.setBuffer(&m_ctx.clusterLodData.groupBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kGroups));
        encoder.setBuffer(&m_ctx.clusterLodData.groupMeshletIndicesBuffer, 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kGroupMeshletIndices));
        encoder.setBuffer(state.phase0VisibleWorklist.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kPhase0Visible));
        encoder.setBuffer(state.phase0RecheckWorklist.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kPhase0Recheck));
        encoder.setBuffer(state.phase1VisibleWorklist.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kPhase1Visible));
        encoder.setBuffer(state.counters.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kCounters));
        encoder.setBuffer(state.indirectArgs.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kIndirectArgs));
        encoder.setBuffer(state.instanceVisibilityBuffer.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kInstanceVisibility));
        encoder.setBuffer(state.dagNodeQueue(0u), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kNodeQueue0));
        encoder.setBuffer(state.dagNodeQueue(1u), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kNodeQueue1));
        encoder.setBuffer(state.hzbViewProjBuffer.get(), 0,
                          DagClusterCullBindings::bufferBinding(DagClusterCullBindings::kHzbViewProj));
    }

    void bindHizTextures(RhiComputeCommandEncoder& encoder, ClusterOcclusionState& state) const {
        const uint32_t pyramid = m_phase == 0u ? 0u : 1u;
        for (uint32_t level = 0; level < state.mipCount; ++level) {
            RhiTexture* texture = state.hizTexture(pyramid, level);
            if (texture) {
                encoder.setTexture(texture, DagClusterCullBindings::kHizMipBase + level);
            }
        }
    }

    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "DAG Cluster Cull";
    uint32_t m_phase = 0;
    uint32_t m_maxIterations = 16;
    bool m_enableOcclusion = true;
    bool m_enableInstanceFilter = true;
    int m_enableFrustumOverride = -1;
    int m_enableOcclusionOverride = -1; // -1=use JSON config, 0=force off, 1=force on
    bool m_warnedMissingPipeline = false;
    FGResource m_phaseReady;
    FGResource m_cullDone;
};

METALLIC_REGISTER_PASS(DagClusterCullPass);
