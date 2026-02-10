#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "imgui.h"
#include <vector>

class ForwardPass : public RenderPass {
public:
    ForwardPass(const RenderContext& ctx,
                int renderMode,
                MTL::RenderPipelineState* vertexPipeline,
                MTL::RenderPipelineState* meshPipeline,
                const Uniforms& baseUniforms,
                const std::vector<uint32_t>& visibleMeshletNodes,
                const std::vector<uint32_t>& visibleIndexNodes,
                const float4x4& view, const float4x4& proj,
                const float4& cameraWorldPos,
                int w, int h,
                bool useSky,
                FGResource target)
        : m_ctx(ctx), m_target(target), m_renderMode(renderMode)
        , m_vertexPipeline(vertexPipeline), m_meshPipeline(meshPipeline)
        , m_baseUniforms(baseUniforms)
        , m_visibleMeshletNodes(visibleMeshletNodes)
        , m_visibleIndexNodes(visibleIndexNodes)
        , m_view(view), m_proj(proj), m_cameraWorldPos(cameraWorldPos)
        , m_width(w), m_height(h)
        , m_useSky(useSky) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return "Forward Pass"; }

    FGResource output;
    FGResource depth;

    void setup(FGBuilder& builder) override {
        if (m_target.isValid()) {
            output = m_target;
        } else {
            output = builder.create("forwardColor",
                FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatRGBA16Float));
        }
        depth = builder.create("depth",
            FGTextureDesc::depthTarget(m_width, m_height));
        MTL::LoadAction colorLoad =
            (m_useSky && m_target.isValid()) ? MTL::LoadActionLoad : MTL::LoadActionClear;
        builder.setColorAttachment(0, output,
            colorLoad, MTL::StoreActionStore,
            MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
        builder.setDepthAttachment(depth,
            MTL::LoadActionClear, MTL::StoreActionDontCare, m_ctx.depthClearValue);
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("ForwardPass");
        enc->setDepthStencilState(m_ctx.depthState);
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        enc->setCullMode(MTL::CullModeBack);

        if (m_renderMode == 1) {
            enc->setRenderPipelineState(m_meshPipeline);
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

            for (uint32_t nodeID : m_visibleMeshletNodes) {
                const auto& node = m_ctx.sceneGraph.nodes[nodeID];
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
                nodeUniforms.instanceID = 0;
                enc->setMeshBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                enc->drawMeshThreadgroups(
                    MTL::Size(node.meshletCount, 1, 1),
                    MTL::Size(1, 1, 1),
                    MTL::Size(128, 1, 1));
            }
        } else {
            enc->setRenderPipelineState(m_vertexPipeline);
            enc->setVertexBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
            enc->setVertexBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);

            for (uint32_t nodeID : m_visibleIndexNodes) {
                const auto& node = m_ctx.sceneGraph.nodes[nodeID];
                float4x4 nodeModelView = m_view * node.transform.worldMatrix;
                float4x4 nodeMVP = m_proj * nodeModelView;
                Uniforms nodeUniforms = m_baseUniforms;
                nodeUniforms.mvp = transpose(nodeMVP);
                nodeUniforms.modelView = transpose(nodeModelView);
                float4x4 invModel = node.transform.worldMatrix;
                invModel.Invert();
                nodeUniforms.cameraPos = invModel * m_cameraWorldPos;
                enc->setVertexBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                enc->setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
                enc->drawIndexedPrimitives(
                    MTL::PrimitiveTypeTriangle,
                    node.indexCount, MTL::IndexTypeUInt32,
                    m_ctx.sceneMesh.indexBuffer, node.indexStart * sizeof(uint32_t));
            }
        }

    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::Text("Mode: %s", m_renderMode == 1 ? "Mesh Shader" : "Vertex Shader");
        if (m_renderMode == 1) {
            ImGui::Text("Visible Meshlet Nodes: %zu", m_visibleMeshletNodes.size());
            ImGui::Text("Frustum Cull: %s", m_baseUniforms.enableFrustumCull ? "On" : "Off");
            ImGui::Text("Cone Cull: %s", m_baseUniforms.enableConeCull ? "On" : "Off");
        } else {
            ImGui::Text("Visible Index Nodes: %zu", m_visibleIndexNodes.size());
        }
    }

private:
    const RenderContext& m_ctx;
    FGResource m_target;
    int m_renderMode;
    MTL::RenderPipelineState* m_vertexPipeline;
    MTL::RenderPipelineState* m_meshPipeline;
    Uniforms m_baseUniforms;
    const std::vector<uint32_t>& m_visibleMeshletNodes;
    const std::vector<uint32_t>& m_visibleIndexNodes;
    float4x4 m_view, m_proj;
    float4 m_cameraWorldPos;
    int m_width, m_height;
    bool m_useSky;
};
