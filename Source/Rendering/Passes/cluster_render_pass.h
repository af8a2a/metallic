#pragma once

#include "cluster_lod_builder.h"
#include "cluster_occlusion_state.h"
#include "cluster_types.h"
#include "frame_context.h"
#include "imgui.h"
#include "pass_registry.h"
#include "render_pass.h"

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

namespace ClusterRenderBindings {
static constexpr uint32_t kClusterInfos = 1u;
static constexpr uint32_t kClusters = 2u;
static constexpr uint32_t kVertexData = 3u;
static constexpr uint32_t kIndexData = 4u;
static constexpr uint32_t kInstances = 5u;
} // namespace ClusterRenderBindings

class ClusterRenderPass : public RenderPass {
public:
    ClusterRenderPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(ClusterRenderPass, "Cluster Render", "Geometry",
        (std::vector<PassSlotInfo>{
            makeHiddenInputSlot("worklistReady", "Worklist Ready", true),
            makeInputSlot("colorIn", "Color In", true),
            makeInputSlot("depthIn", "Depth In", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("color", "Color"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);

    METALLIC_PASS_EDITOR_TYPE_INFO(ClusterRenderPass, "Cluster Render", "Geometry",
        (std::vector<PassSlotInfo>{
            makeHiddenInputSlot("worklistReady", "Worklist Ready", true),
            makeInputSlot("colorIn", "Color In", true),
            makeInputSlot("depthIn", "Depth In", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("color", "Color"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (!config.config.is_object()) {
            return;
        }

        if (config.config.contains("colorMode")) {
            const auto& modeConfig = config.config["colorMode"];
            if (modeConfig.is_number_unsigned()) {
                m_colorMode = modeConfig.get<uint32_t>();
            } else if (modeConfig.is_string()) {
                const std::string mode = modeConfig.get<std::string>();
                if (mode == "Instance") {
                    m_colorMode = 1u;
                } else if (mode == "LOD Level" || mode == "LOD") {
                    m_colorMode = 2u;
                } else if (mode == "Triangle") {
                    m_colorMode = 3u;
                } else {
                    m_colorMode = 0u;
                }
            }
        }
        if (config.config.contains("debugLabel")) {
            m_debugLabel = config.config["debugLabel"].get<std::string>();
        }
        if (config.config.contains("debugTint")) {
            const auto& tint = config.config["debugTint"];
            if (tint.is_array() && tint.size() >= 3) {
                m_debugTint[0] = tint[0].get<float>();
                m_debugTint[1] = tint[1].get<float>();
                m_debugTint[2] = tint[2].get<float>();
            }
        }
        if (config.config.contains("debugTintStrength")) {
            m_debugTintStrength =
                std::clamp(config.config["debugTintStrength"].get<float>(), 0.0f, 1.0f);
        }
        if (config.config.contains("worklist")) {
            const std::string mode = config.config["worklist"].get<std::string>();
            if (mode == "CullPhase0") {
                m_worklistMode = WorklistMode::CullPhase0;
            } else if (mode == "CullPhase1") {
                m_worklistMode = WorklistMode::CullPhase1;
            } else {
                m_worklistMode = WorklistMode::CpuFull;
            }
        }
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "color") return color;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        if (FGResource ready = getInput("worklistReady"); ready.isValid()) {
            m_worklistReady = builder.read(ready);
        }

        FGResource colorInput = getInput("colorIn");
        if (colorInput.isValid()) {
            color = builder.setColorAttachment(0, colorInput,
                RhiLoadAction::Load, RhiStoreAction::Store);
        } else {
            color = builder.create("clusterVisColor",
                FGTextureDesc::renderTarget(m_width, m_height, RhiFormat::RGBA8Unorm));
            color = builder.setColorAttachment(0, color,
                RhiLoadAction::Clear, RhiStoreAction::Store,
                RhiClearColor(0.05, 0.05, 0.08, 1.0));
        }

        FGResource depthInput = getInput("depthIn");
        if (depthInput.isValid()) {
            depth = builder.setDepthAttachment(depthInput,
                RhiLoadAction::Load, RhiStoreAction::Store);
        } else {
            depth = builder.create("clusterVisDepth",
                FGTextureDesc::depthTarget(m_width, m_height));
            depth = builder.setDepthAttachment(depth,
                RhiLoadAction::Clear, RhiStoreAction::Store,
                m_ctx.depthClearValue);
        }
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("ClusterRenderPass");
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("ClusterRenderPass");
        if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() ||
            !pipeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("ClusterRenderPass: pipeline not found");
                m_warnedMissingPipeline = true;
            }
            return;
        }

        const WorklistSelection worklist = selectWorklist();
        if (!worklist.buffer) {
            return;
        }

        if (!m_ctx.clusterLodData.packedClusterBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.clusterVertexDataBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.clusterIndexDataBuffer.nativeHandle() ||
            !m_ctx.gpuScene.instanceBuffer.nativeHandle()) {
            return;
        }

        if (!worklist.indirect && worklist.directCount == 0u) {
            return;
        }

        encoder.setDepthStencilState(&m_ctx.depthState);
        encoder.setFrontFacingWinding(RhiWinding::CounterClockwise);
        encoder.setCullMode(RhiCullMode::Back);
        encoder.setRenderPipeline(pipeIt->second);

        struct PushData {
            float viewProj[16];
            float lightDir[3];
            uint32_t colorMode;
            float debugTint[3];
            float debugTintStrength;
        } pushData;

        std::memset(&pushData, 0, sizeof(pushData));
        float4x4 vp = m_frameContext->proj * m_frameContext->view;
        float4x4 vpT = transpose(vp);
        std::memcpy(pushData.viewProj, &vpT, sizeof(pushData.viewProj));
        pushData.lightDir[0] = m_frameContext->viewLightDir.x;
        pushData.lightDir[1] = m_frameContext->viewLightDir.y;
        pushData.lightDir[2] = m_frameContext->viewLightDir.z;
        pushData.colorMode = m_colorMode;
        pushData.debugTint[0] = m_debugTint[0];
        pushData.debugTint[1] = m_debugTint[1];
        pushData.debugTint[2] = m_debugTint[2];
        pushData.debugTintStrength = m_debugTintStrength;

        encoder.setPushConstants(&pushData, sizeof(pushData));
        encoder.setMeshBuffer(worklist.buffer, 0, ClusterRenderBindings::kClusterInfos);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                              ClusterRenderBindings::kClusters);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterVertexDataBuffer, 0,
                              ClusterRenderBindings::kVertexData);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterIndexDataBuffer, 0,
                              ClusterRenderBindings::kIndexData);
        encoder.setMeshBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                              ClusterRenderBindings::kInstances);

        encoder.setFragmentBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                                  ClusterRenderBindings::kClusters);
        encoder.setFragmentBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                                  ClusterRenderBindings::kInstances);

        if (worklist.indirect && worklist.indirectArgs) {
            encoder.drawMeshThreadgroupsIndirect(*worklist.indirectArgs,
                                                 worklist.indirectOffset,
                                                 {1, 1, 1},
                                                 {32, 1, 1});
        } else {
            encoder.drawMeshThreadgroups({worklist.directCount, 1, 1}, {1, 1, 1}, {32, 1, 1});
        }
    }

    void renderUI() override {
        if (ImGui::CollapsingHeader("Cluster Visualization")) {
            const char* modes[] = {"Cluster", "Instance", "LOD Level", "Triangle"};
            int mode = static_cast<int>(m_colorMode);
            if (ImGui::Combo("Color Mode", &mode, modes, 4)) {
                m_colorMode = static_cast<uint32_t>(mode);
            }
            if (!m_debugLabel.empty()) {
                ImGui::Text("Debug View: %s", m_debugLabel.c_str());
            }
            ImGui::Text("Worklist: %s", worklistModeName());
            ImGui::Text("Clusters: %u", m_ctx.gpuScene.clusterVisWorklistCount);
            ImGui::ColorEdit3("Debug Tint", m_debugTint);
            ImGui::SliderFloat("Debug Tint Strength", &m_debugTintStrength, 0.0f, 1.0f, "%.2f");
            ImGui::Text("Packed data: %.1f KB vtx, %.1f KB idx",
                        m_ctx.clusterLodData.clusterVertexData.size() / 1024.0,
                        m_ctx.clusterLodData.clusterIndexData.size() / 1024.0);
        }
    }

private:
    enum class WorklistMode {
        CpuFull,
        CullPhase0,
        CullPhase1,
    };

    struct WorklistSelection {
        const RhiBuffer* buffer = nullptr;
        const RhiBuffer* indirectArgs = nullptr;
        uint64_t indirectOffset = 0;
        uint32_t directCount = 0;
        bool indirect = false;
    };

    WorklistSelection selectWorklist() const {
        WorklistSelection selection{};

        if (m_worklistMode == WorklistMode::CpuFull) {
            if (m_ctx.gpuScene.clusterVisWorklistBuffer.nativeHandle()) {
                selection.buffer = &m_ctx.gpuScene.clusterVisWorklistBuffer;
                selection.directCount = m_ctx.gpuScene.clusterVisWorklistCount;
            }
            return selection;
        }

        ClusterOcclusionState* state =
            m_runtimeContext ? m_runtimeContext->clusterOcclusionState : nullptr;
        const uint32_t phase =
            m_worklistMode == WorklistMode::CullPhase1
                ? ClusterOcclusionState::kPhase1
                : ClusterOcclusionState::kPhase0;

        if (state && state->worklistValid[phase] &&
            state->visibleWorklist(phase) &&
            state->indirectArgs) {
            selection.buffer = state->visibleWorklist(phase);
            selection.indirectArgs = state->indirectArgs.get();
            selection.indirectOffset = state->indirectOffset(phase);
            selection.indirect = true;
            return selection;
        }

        if (phase == ClusterOcclusionState::kPhase0 &&
            m_ctx.gpuScene.clusterVisWorklistBuffer.nativeHandle()) {
            selection.buffer = &m_ctx.gpuScene.clusterVisWorklistBuffer;
            selection.directCount = m_ctx.gpuScene.clusterVisWorklistCount;
        }
        return selection;
    }

    const char* worklistModeName() const {
        switch (m_worklistMode) {
            case WorklistMode::CullPhase0: return "CullPhase0";
            case WorklistMode::CullPhase1: return "CullPhase1";
            case WorklistMode::CpuFull:
            default: return "CpuFull";
        }
    }

    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "ClusterRenderPass";
    uint32_t m_colorMode = 0;
    WorklistMode m_worklistMode = WorklistMode::CpuFull;
    std::string m_debugLabel;
    float m_debugTint[3] = {1.0f, 1.0f, 1.0f};
    float m_debugTintStrength = 0.0f;
    bool m_warnedMissingPipeline = false;
    FGResource m_worklistReady;
    FGResource color;
    FGResource depth;
};

#ifdef _WIN32
METALLIC_REGISTER_PASS(ClusterRenderPass);
#endif
