#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "pass_registry.h"
#include "imgui.h"

class ForwardPass : public RenderPass {
public:
    ForwardPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    FGResource output;
    FGResource depth;

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "forwardColor") return output;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        FGResource skyInput = getInput("skyOutput");
        if (skyInput.isValid()) {
            builder.read(skyInput);  // establish dependency on SkyPass
            output = skyInput;       // render into sky's texture
        } else {
            output = builder.create("forwardColor",
                FGTextureDesc::renderTarget(m_width, m_height, MTL::PixelFormatRGBA16Float));
        }
        depth = builder.create("depth",
            FGTextureDesc::depthTarget(m_width, m_height));
        MTL::LoadAction colorLoad = skyInput.isValid() ? MTL::LoadActionLoad : MTL::LoadActionClear;
        builder.setColorAttachment(0, output,
            colorLoad, MTL::StoreActionStore,
            MTL::ClearColor(0.1, 0.2, 0.3, 1.0));
        builder.setDepthAttachment(depth,
            MTL::LoadActionClear, MTL::StoreActionStore, m_ctx.depthClearValue);
    }

    void executeRender(MTL::RenderCommandEncoder* enc) override {
        ZoneScopedN("ForwardPass");
        enc->setDepthStencilState(m_ctx.depthState);
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        enc->setCullMode(MTL::CullModeBack);

        if (!m_frameContext || !m_runtimeContext) return;

        // Build base uniforms from FrameContext raw data
        Uniforms baseUni{};
        baseUni.lightDir = m_frameContext->viewLightDir;
        baseUni.lightColorIntensity = m_frameContext->lightColorIntensity;
        baseUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        baseUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;

        int mode = m_frameContext->renderMode;

        if (mode == 1) {
            // Mesh shader path
            auto pipeIt = m_runtimeContext->renderPipelines.find("ForwardMeshPass");
            if (pipeIt == m_runtimeContext->renderPipelines.end()) return;
            enc->setRenderPipelineState(pipeIt->second);

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
            // PLACEHOLDER_FORWARD_PASS_REST
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

            for (uint32_t nodeID : m_frameContext->visibleMeshletNodes) {
                const auto& node = m_ctx.sceneGraph.nodes[nodeID];
                float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
                float4x4 nodeMVP = m_frameContext->proj * nodeModelView;
                Uniforms nodeUniforms = baseUni;
                nodeUniforms.mvp = transpose(nodeMVP);
                nodeUniforms.modelView = transpose(nodeModelView);
                extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
                float4x4 invModel = node.transform.worldMatrix;
                invModel.Invert();
                nodeUniforms.cameraPos = invModel * m_frameContext->cameraWorldPos;
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
            // Vertex shader path
            auto pipeIt = m_runtimeContext->renderPipelines.find("ForwardPass");
            if (pipeIt == m_runtimeContext->renderPipelines.end()) return;
            enc->setRenderPipelineState(pipeIt->second);
            enc->setVertexBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
            enc->setVertexBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);

            for (uint32_t nodeID : m_frameContext->visibleIndexNodes) {
                const auto& node = m_ctx.sceneGraph.nodes[nodeID];
                float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
                float4x4 nodeMVP = m_frameContext->proj * nodeModelView;
                Uniforms nodeUniforms = baseUni;
                nodeUniforms.mvp = transpose(nodeMVP);
                nodeUniforms.modelView = transpose(nodeModelView);
                float4x4 invModel = node.transform.worldMatrix;
                invModel.Invert();
                nodeUniforms.cameraPos = invModel * m_frameContext->cameraWorldPos;
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
        if (m_frameContext) {
            int mode = m_frameContext->renderMode;
            ImGui::Text("Mode: %s", mode == 1 ? "Mesh Shader" : "Vertex Shader");
            if (mode == 1) {
                ImGui::Text("Visible Meshlet Nodes: %zu", m_frameContext->visibleMeshletNodes.size());
                ImGui::Text("Frustum Cull: %s", m_frameContext->enableFrustumCull ? "On" : "Off");
                ImGui::Text("Cone Cull: %s", m_frameContext->enableConeCull ? "On" : "Off");
            } else {
                ImGui::Text("Visible Index Nodes: %zu", m_frameContext->visibleIndexNodes.size());
            }
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Forward Pass";
};
