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
                m_clearColor = RhiClearColor(
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
            FGTextureDesc::renderTarget(m_width, m_height, RhiFormat::R32Uint));
        depth = builder.create("depth",
            FGTextureDesc::depthTarget(m_width, m_height));
        builder.setColorAttachment(0, visibility,
            RhiLoadAction::Clear, RhiStoreAction::Store, m_clearColor);
        builder.setDepthAttachment(depth,
            RhiLoadAction::Clear, RhiStoreAction::Store, m_ctx.depthClearValue);
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("VisibilityPass");
        if (!m_frameContext || !m_runtimeContext) return;

        encoder.setDepthStencilState(&m_ctx.depthStateRhi);
        encoder.setFrontFacingWinding(RhiWinding::CounterClockwise);
        encoder.setCullMode(RhiCullMode::Back);

        // GPU-driven indirect path
        bool useGPUPath = m_frameContext->gpuDrivenCulling &&
            m_frameContext->gpuVisibleMeshletBufferRhi &&
            m_frameContext->gpuCounterBufferRhi &&
            m_frameContext->gpuInstanceDataBufferRhi;

        if (useGPUPath) {
            auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityIndirectPass");
            if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle())
                useGPUPath = false;
            else
                encoder.setRenderPipeline(pipeIt->second);
        }

        std::vector<const RhiTexture*> materialTextures;
        materialTextures.reserve(m_ctx.materials.textureViews.size());
        for (auto* texture : m_ctx.materials.textureViews) {
            materialTextures.push_back(texture);
        }

        if (useGPUPath) {

            // Bind shared geometry buffers (same indices as visibility.slang)
            encoder.setMeshBuffer(&m_ctx.sceneMesh.positionBufferRhi, 0, 1);
            encoder.setMeshBuffer(&m_ctx.sceneMesh.normalBufferRhi, 0, 2);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletBufferRhi, 0, 3);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletVerticesRhi, 0, 4);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletTrianglesRhi, 0, 5);
            encoder.setMeshBuffer(&m_ctx.meshletData.boundsBufferRhi, 0, 6);
            encoder.setMeshBuffer(&m_ctx.sceneMesh.uvBufferRhi, 0, 7);
            encoder.setMeshBuffer(&m_ctx.meshletData.materialIDsRhi, 0, 8);
            encoder.setMeshBuffer(&m_ctx.materials.materialBufferRhi, 0, 9);
            encoder.setMeshBuffer(m_frameContext->gpuVisibleMeshletBufferRhi, 0, 10);
            encoder.setMeshBuffer(m_frameContext->gpuInstanceDataBufferRhi, 0, 11);

            // Fragment stage needs all buffers too (Slang KernelContext)
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.positionBufferRhi, 0, 1);
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.normalBufferRhi, 0, 2);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletBufferRhi, 0, 3);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletVerticesRhi, 0, 4);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletTrianglesRhi, 0, 5);
            encoder.setFragmentBuffer(&m_ctx.meshletData.boundsBufferRhi, 0, 6);
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.uvBufferRhi, 0, 7);
            encoder.setFragmentBuffer(&m_ctx.meshletData.materialIDsRhi, 0, 8);
            encoder.setFragmentBuffer(&m_ctx.materials.materialBufferRhi, 0, 9);
            encoder.setFragmentBuffer(m_frameContext->gpuVisibleMeshletBufferRhi, 0, 10);
            encoder.setFragmentBuffer(m_frameContext->gpuInstanceDataBufferRhi, 0, 11);

            if (!materialTextures.empty()) {
                encoder.setFragmentTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
                encoder.setMeshTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
            }
            encoder.setFragmentSampler(&m_ctx.materials.samplerRhi, 0);
            encoder.setMeshSampler(&m_ctx.materials.samplerRhi, 0);

            // GlobalUniforms at buffer(0) via setMeshBytes
            struct { float4 lightDir; float4 lightColorIntensity; } globalUni;
            globalUni.lightDir = m_frameContext->viewLightDir;
            globalUni.lightColorIntensity = m_frameContext->lightColorIntensity;
            encoder.setMeshBytes(&globalUni, sizeof(globalUni), 0);
            encoder.setFragmentBytes(&globalUni, sizeof(globalUni), 0);

            // Single indirect draw call
            encoder.drawMeshThreadgroupsIndirect(*m_frameContext->gpuCounterBufferRhi,
                                                 kIndirectArgsOffset,
                                                 {1, 1, 1},
                                                 {128, 1, 1});

            m_gpuDrivenLastFrame = true;
            return;
        }

        // CPU per-node dispatch fallback
        m_gpuDrivenLastFrame = false;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityPass");
        if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle()) return;
        encoder.setRenderPipeline(pipeIt->second);

        // Bind shared buffers once
        encoder.setMeshBuffer(&m_ctx.sceneMesh.positionBufferRhi, 0, 1);
        encoder.setMeshBuffer(&m_ctx.sceneMesh.normalBufferRhi, 0, 2);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletBufferRhi, 0, 3);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletVerticesRhi, 0, 4);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletTrianglesRhi, 0, 5);
        encoder.setMeshBuffer(&m_ctx.meshletData.boundsBufferRhi, 0, 6);
        encoder.setMeshBuffer(&m_ctx.sceneMesh.uvBufferRhi, 0, 7);
        encoder.setMeshBuffer(&m_ctx.meshletData.materialIDsRhi, 0, 8);
        encoder.setMeshBuffer(&m_ctx.materials.materialBufferRhi, 0, 9);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.positionBufferRhi, 0, 1);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.normalBufferRhi, 0, 2);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletBufferRhi, 0, 3);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletVerticesRhi, 0, 4);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletTrianglesRhi, 0, 5);
        encoder.setFragmentBuffer(&m_ctx.meshletData.boundsBufferRhi, 0, 6);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.uvBufferRhi, 0, 7);
        encoder.setFragmentBuffer(&m_ctx.meshletData.materialIDsRhi, 0, 8);
        encoder.setFragmentBuffer(&m_ctx.materials.materialBufferRhi, 0, 9);
        if (!materialTextures.empty()) {
            encoder.setFragmentTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
            encoder.setMeshTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
        }
        encoder.setFragmentSampler(&m_ctx.materials.samplerRhi, 0);
        encoder.setMeshSampler(&m_ctx.materials.samplerRhi, 0);

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
            encoder.setMeshBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
            encoder.setFragmentBytes(&nodeUniforms, sizeof(nodeUniforms), 0);
            encoder.drawMeshThreadgroups({node.meshletCount, 1, 1},
                                         {1, 1, 1},
                                         {128, 1, 1});
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
    RhiClearColor m_clearColor = RhiClearColor(0xFFFFFFFF, 0, 0, 0);
    bool m_gpuDrivenLastFrame = false;
};



