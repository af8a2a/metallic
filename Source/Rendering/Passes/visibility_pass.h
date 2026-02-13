#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"
#include <vector>

class VisibilityPass : public RenderPass {
public:
    // Data-driven constructor
    VisibilityPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("clearColor")) {
            auto& cc = config.config["clearColor"];
            if (cc.is_array() && cc.size() >= 4) {
                m_clearColor = MTL::ClearColor(
                    cc[0].get<double>(), cc[1].get<double>(),
                    cc[2].get<double>(), cc[3].get<double>());
            }
        }
    }

    // Output resources
    FGResource visibility;
    FGResource depth;

    FGResource getOutput(const std::string& name) const override {
        if (name == "visibility") return visibility;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        visibility = builder.create("visibility",
            FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatR32Uint));
        depth = builder.create("depth",
            FGTextureDesc::depthTarget(m_width, m_height));
        builder.setColorAttachment(0, visibility,
            MTL::LoadActionClear, MTL::StoreActionStore, m_clearColor);
        builder.setDepthAttachment(depth,
            MTL::LoadActionClear, MTL::StoreActionStore, m_ctx.depthClearValue);
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("VisibilityPass");
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->renderPipelines.find("VisibilityPass");
        if (pipeIt == m_runtimeContext->renderPipelines.end()) return;
        MTL::RenderPipelineState* pipeline = pipeIt->second;

        enc->setDepthStencilState(m_ctx.depthState);
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        enc->setCullMode(MTL::CullModeBack);
        enc->setRenderPipelineState(pipeline);

        // Bind shared buffers once
        enc->setMeshBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
        enc->setMeshBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
        enc->setMeshBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
        enc->setMeshBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
        enc->setMeshBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
        enc->setMeshBuffer(m_ctx.meshletData.boundsBuffer, 0, 6);
        enc->setMeshBuffer(m_ctx.sceneMesh.uvBuffer, 0, 7);
        enc->setMeshBuffer(m_ctx.meshletData.materialIDs, 0, 8);
        enc->setMeshBuffer(m_ctx.materials.materialBuffer, 0, 9);
        enc->setFragmentBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
        enc->setFragmentBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
        enc->setFragmentBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
        enc->setFragmentBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
        enc->setFragmentBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
        enc->setFragmentBuffer(m_ctx.meshletData.boundsBuffer, 0, 6);
        enc->setFragmentBuffer(m_ctx.sceneMesh.uvBuffer, 0, 7);
        enc->setFragmentBuffer(m_ctx.meshletData.materialIDs, 0, 8);
        enc->setFragmentBuffer(m_ctx.materials.materialBuffer, 0, 9);
        if (!m_ctx.materials.textures.empty()) {
            enc->setFragmentTextures(
                const_cast<MTL::Texture* const*>(m_ctx.materials.textures.data()),
                NS::Range(0, m_ctx.materials.textures.size()));
            enc->setMeshTextures(
                const_cast<MTL::Texture* const*>(m_ctx.materials.textures.data()),
                NS::Range(0, m_ctx.materials.textures.size()));
        }
        enc->setFragmentSamplerState(m_ctx.materials.sampler, 0);
        enc->setMeshSamplerState(m_ctx.materials.sampler, 0);

        // Per-node dispatch
        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = m_frameContext->visibilityInstanceCount;
        for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++) {
            const auto& node = m_ctx.sceneGraph.nodes[visibleNodes[instanceID]];
            float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
            float4x4 nodeMVP = m_frameContext->proj * nodeModelView;
            Uniforms nodeUniforms = m_frameContext->baseUniforms;
            nodeUniforms.mvp = transpose(nodeMVP);
            nodeUniforms.modelView = transpose(nodeModelView);
            extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
            float4x4 invModel = node.transform.worldMatrix;
            invModel.Invert();
            nodeUniforms.cameraPos = invModel * m_frameContext->cameraWorldPos;
            nodeUniforms.meshletBaseOffset = node.meshletStart;
            nodeUniforms.instanceID = instanceID;
            enc->setMeshBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
            enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
            enc->drawMeshThreadgroups(
                MTL::Size(node.meshletCount, 1, 1),
                MTL::Size(1, 1, 1),
                MTL::Size(128, 1, 1));
        }
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        if (m_frameContext) {
            ImGui::Text("Visible Nodes: %u", m_frameContext->visibilityInstanceCount);
            ImGui::Text("Frustum Cull: %s", m_frameContext->enableFrustumCull ? "On" : "Off");
            ImGui::Text("Cone Cull: %s", m_frameContext->enableConeCull ? "On" : "Off");
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Visibility Pass";
    MTL::ClearColor m_clearColor = MTL::ClearColor(0xFFFFFFFF, 0, 0, 0);
};
