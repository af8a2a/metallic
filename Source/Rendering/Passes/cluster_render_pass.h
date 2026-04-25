#pragma once

#include "render_pass.h"
#include "frame_context.h"
#include "cluster_lod_builder.h"
#include "cluster_types.h"
#include "gpu_driven_constants.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>

class ClusterRenderPass : public RenderPass {
public:
    ClusterRenderPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(ClusterRenderPass, "Cluster Render", "Geometry",
        (std::vector<PassSlotInfo>{}),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("color", "Color"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource color;
    FGResource depth;

    FGResource getOutput(const std::string& name) const override {
        if (name == "color") return color;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
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
            !m_ctx.gpuScene.clusterVisWorklistBuffer.nativeHandle() ||
            m_ctx.gpuScene.clusterVisWorklistCount == 0) {
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
        } pushData;

        float4x4 vp = m_frameContext->proj * m_frameContext->view;
        float4x4 vpT = transpose(vp);
        std::memcpy(pushData.viewProj, &vpT, sizeof(pushData.viewProj));
        pushData.lightDir[0] = m_frameContext->viewLightDir.x;
        pushData.lightDir[1] = m_frameContext->viewLightDir.y;
        pushData.lightDir[2] = m_frameContext->viewLightDir.z;
        pushData.colorMode = m_colorMode;

        encoder.setPushConstants(&pushData, sizeof(pushData));

        encoder.setMeshBuffer(&m_ctx.gpuScene.clusterVisWorklistBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kClusterInfos);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kClusters);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterVertexDataBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kVertexData);
        encoder.setMeshBuffer(&m_ctx.clusterLodData.clusterIndexDataBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kIndexData);
        encoder.setMeshBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                              GpuDriven::ClusterRenderBindings::kInstances);

        encoder.setFragmentBuffer(&m_ctx.clusterLodData.packedClusterBuffer, 0,
                                  GpuDriven::ClusterRenderBindings::kClusters);
        encoder.setFragmentBuffer(&m_ctx.gpuScene.instanceBuffer, 0,
                                  GpuDriven::ClusterRenderBindings::kInstances);

        uint32_t clusterCount = m_ctx.gpuScene.clusterVisWorklistCount;
        encoder.drawMeshThreadgroups({clusterCount, 1, 1}, {1, 1, 1}, {32, 1, 1});
    }

    void renderUI() override {
        if (ImGui::CollapsingHeader("Cluster Visualization")) {
            const char* modes[] = {"Cluster", "Instance", "LOD Level", "Triangle"};
            int mode = static_cast<int>(m_colorMode);
            if (ImGui::Combo("Color Mode", &mode, modes, 4)) {
                m_colorMode = static_cast<uint32_t>(mode);
            }
            ImGui::Text("Clusters: %u", m_ctx.gpuScene.clusterVisWorklistCount);
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
    bool m_warnedMissingPipeline = false;
};

#ifdef _WIN32
METALLIC_REGISTER_PASS(ClusterRenderPass);
#endif
