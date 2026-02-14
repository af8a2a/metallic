#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
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
        // Read cullResult input if present (dependency edge from MeshletCullPass)
        FGResource cullInput = getInput("cullResult");
        if (cullInput.isValid()) {
            builder.read(cullInput);
        }
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

        enc->setDepthStencilState(m_ctx.depthState);
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        enc->setCullMode(MTL::CullModeBack);

        // GPU-driven indirect path
        bool useGPUPath = m_frameContext->gpuDrivenCulling &&
            m_frameContext->gpuVisibleMeshletBuffer &&
            m_frameContext->gpuCounterBuffer &&
            m_frameContext->gpuInstanceDataBuffer;

        if (useGPUPath) {
            auto pipeIt = m_runtimeContext->renderPipelines.find("VisibilityIndirectPass");
            if (pipeIt == m_runtimeContext->renderPipelines.end() || !pipeIt->second)
                useGPUPath = false;
            else
                enc->setRenderPipelineState(pipeIt->second);
        }

        if (useGPUPath) {

            // Bind shared geometry buffers (same indices as visibility.slang)
            enc->setMeshBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
            enc->setMeshBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
            enc->setMeshBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
            enc->setMeshBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
            enc->setMeshBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
            enc->setMeshBuffer(m_ctx.meshletData.boundsBuffer, 0, 6);
            enc->setMeshBuffer(m_ctx.sceneMesh.uvBuffer, 0, 7);
            enc->setMeshBuffer(m_ctx.meshletData.materialIDs, 0, 8);
            enc->setMeshBuffer(m_ctx.materials.materialBuffer, 0, 9);
            enc->setMeshBuffer(m_frameContext->gpuVisibleMeshletBuffer, 0, 10);
            enc->setMeshBuffer(m_frameContext->gpuInstanceDataBuffer, 0, 11);

            // Fragment stage needs all buffers too (Slang KernelContext)
            enc->setFragmentBuffer(m_ctx.sceneMesh.positionBuffer, 0, 1);
            enc->setFragmentBuffer(m_ctx.sceneMesh.normalBuffer, 0, 2);
            enc->setFragmentBuffer(m_ctx.meshletData.meshletBuffer, 0, 3);
            enc->setFragmentBuffer(m_ctx.meshletData.meshletVertices, 0, 4);
            enc->setFragmentBuffer(m_ctx.meshletData.meshletTriangles, 0, 5);
            enc->setFragmentBuffer(m_ctx.meshletData.boundsBuffer, 0, 6);
            enc->setFragmentBuffer(m_ctx.sceneMesh.uvBuffer, 0, 7);
            enc->setFragmentBuffer(m_ctx.meshletData.materialIDs, 0, 8);
            enc->setFragmentBuffer(m_ctx.materials.materialBuffer, 0, 9);
            enc->setFragmentBuffer(m_frameContext->gpuVisibleMeshletBuffer, 0, 10);
            enc->setFragmentBuffer(m_frameContext->gpuInstanceDataBuffer, 0, 11);

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

            // GlobalUniforms at buffer(0) via setMeshBytes
            struct { float4 lightDir; float4 lightColorIntensity; } globalUni;
            globalUni.lightDir = m_frameContext->viewLightDir;
            globalUni.lightColorIntensity = m_frameContext->lightColorIntensity;
            enc->setMeshBytes(&globalUni, sizeof(globalUni), 0);
            enc->setFragmentBytes(&globalUni, sizeof(globalUni), 0);

            // Single indirect draw call
            enc->drawMeshThreadgroups(
                m_frameContext->gpuCounterBuffer,
                kIndirectArgsOffset,
                MTL::Size(1, 1, 1),
                MTL::Size(128, 1, 1));

            m_gpuDrivenLastFrame = true;
            return;
        }

        // CPU per-node dispatch fallback
        m_gpuDrivenLastFrame = false;

        auto pipeIt = m_runtimeContext->renderPipelines.find("VisibilityPass");
        if (pipeIt == m_runtimeContext->renderPipelines.end()) return;
        MTL::RenderPipelineState* pipeline = pipeIt->second;
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

        // Per-node dispatch (CPU fallback)
        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = m_frameContext->visibilityInstanceCount;

        Uniforms baseUni{};
        baseUni.lightDir = m_frameContext->viewLightDir;
        baseUni.lightColorIntensity = m_frameContext->lightColorIntensity;
        baseUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        baseUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;

        for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++) {
            const auto& node = m_ctx.sceneGraph.nodes[visibleNodes[instanceID]];
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
            ImGui::Text("Dispatch: %s", m_gpuDrivenLastFrame ? "GPU Indirect" : "CPU Per-Node");
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Visibility Pass";
    MTL::ClearColor m_clearColor = MTL::ClearColor(0xFFFFFFFF, 0, 0, 0);
    bool m_gpuDrivenLastFrame = false;
};
