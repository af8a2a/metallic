#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "cluster_lod_builder.h"
#include "cluster_types.h"
#include "gpu_driven_constants.h"
#include "gpu_driven_helpers.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

class ClusterRenderPass : public RenderPass {
    enum class WorklistMode {
        Auto,
        Full,
        GpuCulled,
    };

public:
    ClusterRenderPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(ClusterRenderPass, "Cluster Render", "Geometry",
        (std::vector<PassSlotInfo>{
            makeInputSlot("cullResult", "Cull Result", true),
            makeInputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeInputSlot("cullCounter", "Cull Counter", true),
            makeInputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeInputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("color", "Color"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);

    METALLIC_PASS_EDITOR_TYPE_INFO(ClusterRenderPass, "Cluster Render", "Geometry",
        (std::vector<PassSlotInfo>{
            makeHiddenInputSlot("cullResult", "Cull Result", true),
            makeHiddenInputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeHiddenInputSlot("cullCounter", "Cull Counter", true),
            makeHiddenInputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeHiddenInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeHiddenInputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true)
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
        if (config.config.contains("worklistMode")) {
            const auto& modeConfig = config.config["worklistMode"];
            if (modeConfig.is_string()) {
                const std::string mode = modeConfig.get<std::string>();
                if (mode == "full" || mode == "cpuFull" || mode == "noCull") {
                    m_worklistMode = WorklistMode::Full;
                } else if (mode == "gpu" || mode == "gpuCulled" || mode == "culled") {
                    m_worklistMode = WorklistMode::GpuCulled;
                } else {
                    m_worklistMode = WorklistMode::Auto;
                }
            }
        }
    }

    FGResource color;
    FGResource depth;

    FGResource getOutput(const std::string& name) const override {
        if (name == "color") return color;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_cullResultRead = FGResource{};
        m_visibleMeshletsRead = FGResource{};
        m_cullCounterRead = FGResource{};

        if (m_worklistMode != WorklistMode::Full) {
            FGResource cullInput = getInput("cullResult");
            if (cullInput.isValid()) {
                m_cullResultRead = builder.read(cullInput);
            }
            FGResource visibleMeshletsInput = getInput("visibleMeshlets");
            if (!visibleMeshletsInput.isValid()) {
                visibleMeshletsInput = getInput("visibilityWorklist");
            }
            if (visibleMeshletsInput.isValid()) {
                m_visibleMeshletsRead = builder.read(visibleMeshletsInput,
                                                     FGResourceUsage::StorageRead);
            }
            FGResource cullCounterInput = getInput("cullCounter");
            if (!cullCounterInput.isValid()) {
                cullCounterInput = getInput("visibilityWorklistState");
            }
            if (!cullCounterInput.isValid()) {
                cullCounterInput = getInput("visibilityIndirectArgs");
            }
            if (cullCounterInput.isValid()) {
                m_cullCounterRead = builder.read(cullCounterInput, FGResourceUsage::Indirect);
            }
        }

        color = builder.create("clusterVisColor",
            FGTextureDesc::renderTarget(m_width, m_height, RhiFormat::RGBA8Unorm));
        depth = builder.create("clusterVisDepth",
            FGTextureDesc::depthTarget(m_width, m_height));
        color = builder.setColorAttachment(0, color,
            RhiLoadAction::Clear, RhiStoreAction::Store,
            RhiClearColor(0.05, 0.05, 0.08, 1.0));
        depth = builder.setDepthAttachment(depth,
            RhiLoadAction::Clear, RhiStoreAction::Store,
            m_ctx.depthClearValue);
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("ClusterRenderPass");
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("ClusterRenderPass");
        if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle()) {
            if (!m_warnedMissingPipeline) {
                spdlog::warn("ClusterRenderPass: pipeline not found");
                m_warnedMissingPipeline = true;
            }
            return;
        }

        if (!m_ctx.clusterLodData.packedClusterBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.clusterVertexDataBuffer.nativeHandle() ||
            !m_ctx.clusterLodData.clusterIndexDataBuffer.nativeHandle() ||
            !m_ctx.gpuScene.instanceBuffer.nativeHandle() ||
            !m_ctx.gpuScene.geometryBuffer.nativeHandle()) {
            return;
        }

        const RhiBuffer* visibleMeshletBuffer =
            (m_frameGraph && m_visibleMeshletsRead.isValid())
                ? m_frameGraph->getBuffer(m_visibleMeshletsRead)
                : nullptr;
        const RhiBuffer* worklistStateBuffer =
            (m_frameGraph && m_cullCounterRead.isValid())
                ? m_frameGraph->getBuffer(m_cullCounterRead)
                : nullptr;
        const bool gpuWorklistAvailable =
            m_frameContext->gpuDrivenCulling &&
            visibleMeshletBuffer &&
            worklistStateBuffer;
        const bool useGpuCulledWorklist =
            m_worklistMode == WorklistMode::GpuCulled
                ? gpuWorklistAvailable
                : (m_worklistMode == WorklistMode::Auto && gpuWorklistAvailable);

        m_lastUsedGpuCulledWorklist = useGpuCulledWorklist;
        m_lastGpuVisibleCount = useGpuCulledWorklist
            ? GpuDriven::readPublishedWorkItemCount<GpuDriven::MeshDispatchCommandLayout>(
                  worklistStateBuffer)
            : 0u;
        m_lastGpuIndirectGroupCount = useGpuCulledWorklist
            ? GpuDriven::readBuiltIndirectGroupCount<GpuDriven::MeshDispatchCommandLayout>(
                  worklistStateBuffer)
            : 0u;

        if (!useGpuCulledWorklist &&
            (!m_ctx.gpuScene.clusterVisWorklistBuffer.nativeHandle() ||
             m_ctx.gpuScene.clusterVisWorklistCount == 0)) {
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
            uint32_t worklistMode;
            uint32_t pad[3];
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
        pushData.worklistMode = useGpuCulledWorklist ? 1u : 0u;

        encoder.setPushConstants(&pushData, sizeof(pushData));

        const RhiBuffer* worklistBuffer = useGpuCulledWorklist
            ? visibleMeshletBuffer
            : static_cast<const RhiBuffer*>(&m_ctx.gpuScene.clusterVisWorklistBuffer);
        encoder.setMeshBuffer(worklistBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kClusterInfos);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kClusters);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterVertexDataBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kVertexData);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterIndexDataBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kIndexData);
        encoder.setMeshBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kInstances);
        encoder.setMeshBuffer(&m_ctx.gpuScene.geometryBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kGeometries);

        encoder.setFragmentBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                                  GpuDriven::ClusterRenderBindings::kClusters);
        encoder.setFragmentBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                                  GpuDriven::ClusterRenderBindings::kInstances);

        if (useGpuCulledWorklist) {
            encoder.drawMeshThreadgroupsIndirect(
                *worklistStateBuffer,
                GpuDriven::MeshDispatchCommandLayout::kIndirectArgsOffset,
                {1, 1, 1},
                {32, 1, 1});
        } else {
            uint32_t clusterCount = m_ctx.gpuScene.clusterVisWorklistCount;
            encoder.drawMeshThreadgroups({clusterCount, 1, 1}, {1, 1, 1}, {32, 1, 1});
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
            ImGui::Text("Clusters: %u", m_ctx.gpuScene.clusterVisWorklistCount);
            const char* configuredMode = "Auto";
            if (m_worklistMode == WorklistMode::Full) {
                configuredMode = "Full";
            } else if (m_worklistMode == WorklistMode::GpuCulled) {
                configuredMode = "GPU Culled";
            }
            ImGui::Text("Configured Worklist: %s", configuredMode);
            ImGui::Text("Active Worklist: %s",
                        m_lastUsedGpuCulledWorklist ? "GPU Culled" : "CPU Full");
            if (m_lastUsedGpuCulledWorklist) {
                ImGui::Text("Visible Work Items: %u", m_lastGpuVisibleCount);
                ImGui::Text("Indirect Groups: %u", m_lastGpuIndirectGroupCount);
            }
            ImGui::ColorEdit3("Debug Tint", m_debugTint);
            ImGui::SliderFloat("Debug Tint Strength", &m_debugTintStrength, 0.0f, 1.0f, "%.2f");
            ImGui::Text("Packed data: %.1f KB vtx, %.1f KB idx",
                        m_ctx.clusterLodData.clusterVertexData.size() / 1024.0,
                        m_ctx.clusterLodData.clusterIndexData.size() / 1024.0);
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "ClusterRenderPass";
    uint32_t m_colorMode = 0;
    std::string m_debugLabel;
    float m_debugTint[3] = {1.0f, 1.0f, 1.0f};
    float m_debugTintStrength = 0.0f;
    WorklistMode m_worklistMode = WorklistMode::Auto;
    bool m_warnedMissingPipeline = false;
    bool m_lastUsedGpuCulledWorklist = false;
    uint32_t m_lastGpuVisibleCount = 0;
    uint32_t m_lastGpuIndirectGroupCount = 0;
    FGResource m_cullResultRead;
    FGResource m_visibleMeshletsRead;
    FGResource m_cullCounterRead;
};

#ifdef _WIN32
METALLIC_REGISTER_PASS(ClusterRenderPass);
#endif
