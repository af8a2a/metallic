#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "imgui.h"
#include <vector>

class VisibilityPass : public RenderPass {
public:
    VisibilityPass(const RenderContext& ctx,
                   MTL::RenderPipelineState* pipeline,
                   const Uniforms& baseUniforms,
                   const std::vector<uint32_t>& visibleNodes,
                   uint32_t instanceCount,
                   const float4x4& view, const float4x4& proj,
                   const float4& cameraWorldPos,
                   int w, int h)
        : m_ctx(ctx), m_pipeline(pipeline), m_baseUniforms(baseUniforms)
        , m_visibleNodes(visibleNodes), m_instanceCount(instanceCount)
        , m_view(view), m_proj(proj), m_cameraWorldPos(cameraWorldPos)
        , m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return "Visibility Pass"; }

    // Populated by setup(), read by later passes
    FGResource visibility;
    FGResource depth;

    void setup(FGBuilder& builder) override {
        visibility = builder.create("visibility",
            FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatR32Uint));
        depth = builder.create("depth",
            FGTextureDesc::depthTarget(m_width, m_height));
        builder.setColorAttachment(0, visibility,
            MTL::LoadActionClear, MTL::StoreActionStore,
            MTL::ClearColor(0xFFFFFFFF, 0, 0, 0));
        builder.setDepthAttachment(depth,
            MTL::LoadActionClear, MTL::StoreActionStore, m_ctx.depthClearValue);
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("VisibilityPass");
        enc->setDepthStencilState(m_ctx.depthState);
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        enc->setCullMode(MTL::CullModeBack);
        enc->setRenderPipelineState(m_pipeline);

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
        for (uint32_t instanceID = 0; instanceID < m_instanceCount; instanceID++) {
            const auto& node = m_ctx.sceneGraph.nodes[m_visibleNodes[instanceID]];
            float4x4 nodeModelView = m_view * node.transform.worldMatrix;
            float4x4 nodeMVP = m_proj * nodeModelView;
            Uniforms nodeUniforms = m_baseUniforms;
            nodeUniforms.mvp = transpose(nodeMVP);
            nodeUniforms.modelView = transpose(nodeModelView);
            extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
            float4x4 invModel = node.transform.worldMatrix;
            invModel.Invert();
            nodeUniforms.cameraPos = invModel * m_cameraWorldPos;
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
        ImGui::Text("Visible Nodes: %u", m_instanceCount);
        ImGui::Text("Frustum Cull: %s", m_baseUniforms.enableFrustumCull ? "On" : "Off");
        ImGui::Text("Cone Cull: %s", m_baseUniforms.enableConeCull ? "On" : "Off");
    }

private:
    const RenderContext& m_ctx;
    MTL::RenderPipelineState* m_pipeline;
    Uniforms m_baseUniforms;
    const std::vector<uint32_t>& m_visibleNodes;
    uint32_t m_instanceCount;
    float4x4 m_view, m_proj;
    float4 m_cameraWorldPos;
    int m_width, m_height;
};
