#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_driven_constants.h"
#include "bindless_scene_constants.h"
#include "cluster_lod_builder.h"
#include "pass_registry.h"
#include "imgui.h"
#include <vector>

class DeferredLightingPass : public RenderPass {
public:
    DeferredLightingPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("motionVectorIntensity")) {
            m_motionVectorIntensity = config.config["motionVectorIntensity"].get<float>();
        }
    }

    FGResource output;
    FGResource motionVectorsOutput;

    FGResource getOutput(const std::string& name) const override {
        if (name == "lightingOutput") return output;
        if (name == "motionVectors") return motionVectorsOutput;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Read inputs
        FGResource visInput = getInput("visibility");
        FGResource depthInput = getInput("depth");
        FGResource shadowInput = getInput("shadowMap");
        FGResource skyInput = getInput("skyOutput");
        FGResource visibleMeshletsInput = getInput("visibleMeshlets");
        if (!visibleMeshletsInput.isValid()) {
            visibleMeshletsInput = getInput("visibilityWorklist");
        }
        FGResource visibleMeshletStateInput = getInput("cullCounter");
        if (!visibleMeshletStateInput.isValid()) {
            visibleMeshletStateInput = getInput("visibilityWorklistState");
        }

        if (visInput.isValid()) m_visRead = builder.read(visInput);
        if (depthInput.isValid()) m_depthRead = builder.read(depthInput);
        if (shadowInput.isValid()) m_shadowRead = builder.read(shadowInput);
        if (skyInput.isValid()) m_skyRead = builder.read(skyInput);
        if (visibleMeshletsInput.isValid()) {
            m_visibleMeshletsRead = builder.read(visibleMeshletsInput, FGResourceUsage::StorageRead);
        }
        if (visibleMeshletStateInput.isValid()) {
            m_visibleMeshletStateRead = builder.read(visibleMeshletStateInput, FGResourceUsage::StorageRead);
        }

        output = builder.create("output",
            FGTextureDesc::storageTexture(m_width, m_height, RhiFormat::RGBA16Float));
        motionVectorsOutput = builder.create("motionVectors",
            FGTextureDesc::storageTexture(m_width, m_height, RhiFormat::RG16Float));
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("DeferredLightingPass");
        MICROPROFILE_SCOPEI("RenderPass", "DeferredLightingPass", 0xffff8800);
        if (!m_frameContext || !m_runtimeContext) return;

        auto pipeIt = m_runtimeContext->computePipelinesRhi.find("DeferredLightingPass");
        if (pipeIt == m_runtimeContext->computePipelinesRhi.end() || !pipeIt->second.nativeHandle()) return;
        if (!m_ctx.gpuScene.instanceBuffer.nativeHandle()) return;
        const bool useBindlessSceneTextures = m_runtimeContext->useBindlessSceneTextures;

        // Build LightingUniforms from shared frame and scene state.
        float4x4 invProj = m_frameContext->proj;
        invProj.Invert();

        LightingUniforms lightUniforms;
        lightUniforms.viewProj = transpose(m_frameContext->proj * m_frameContext->view);
        lightUniforms.prevViewProj =
            transpose(m_frameContext->prevProj * m_frameContext->prevView);
        lightUniforms.viewMatrix = transpose(m_frameContext->view);
        lightUniforms.lightDir = m_frameContext->viewLightDir;
        lightUniforms.lightColorIntensity = m_frameContext->lightColorIntensity;
        lightUniforms.invProj = transpose(invProj);
        lightUniforms.screenWidth = static_cast<uint32_t>(m_frameContext->width);
        lightUniforms.screenHeight = static_cast<uint32_t>(m_frameContext->height);
        lightUniforms.meshletCount = m_frameContext->meshletCount;
        lightUniforms.materialCount = m_frameContext->materialCount;
        lightUniforms.textureCount = m_frameContext->textureCount;
        lightUniforms.instanceCount = m_ctx.gpuScene.instanceCount;
        lightUniforms.shadowEnabled = m_frameContext->enableRTShadows ? 1 : 0;
        lightUniforms.visibilityUsesWorklistIds =
            (m_frameContext->gpuDrivenCulling &&
             m_visibleMeshletsRead.isValid() &&
             m_visibleMeshletStateRead.isValid())
                ? 1u
                : 0u;
        lightUniforms.motionVectorIntensity = m_motionVectorIntensity;
        lightUniforms.pad2 = 0;

        const RhiBuffer* visibleMeshletsBuffer =
            (m_frameGraph && m_visibleMeshletsRead.isValid())
                ? m_frameGraph->getBuffer(m_visibleMeshletsRead)
                : nullptr;
        const RhiBuffer* visibleMeshletStateBuffer =
            (m_frameGraph && m_visibleMeshletStateRead.isValid())
                ? m_frameGraph->getBuffer(m_visibleMeshletStateRead)
                : nullptr;
        const RhiBuffer* lodMeshletBuffer =
            m_ctx.clusterLodData.meshletBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.meshletBuffer
                : &m_ctx.meshletData.meshletBuffer;
        const RhiBuffer* lodMeshletVerticesBuffer =
            m_ctx.clusterLodData.meshletVerticesBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.meshletVerticesBuffer
                : &m_ctx.meshletData.meshletVertices;
        const RhiBuffer* lodMeshletTrianglesBuffer =
            m_ctx.clusterLodData.meshletTrianglesBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.meshletTrianglesBuffer
                : &m_ctx.meshletData.meshletTriangles;
        const RhiBuffer* lodMaterialIdsBuffer =
            m_ctx.clusterLodData.materialIDsBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.materialIDsBuffer
                : &m_ctx.meshletData.materialIDs;

        encoder.setComputePipeline(pipeIt->second);
        encoder.setBytes(&lightUniforms, sizeof(lightUniforms), 0);
        encoder.setBuffer(&m_ctx.sceneMesh.positionBuffer, 0, GpuDriven::DeferredLightingBindings::kPositions);
        encoder.setBuffer(&m_ctx.sceneMesh.normalBuffer, 0, GpuDriven::DeferredLightingBindings::kNormals);
        encoder.setBuffer(&m_ctx.meshletData.meshletBuffer, 0, GpuDriven::DeferredLightingBindings::kMeshlets);
        encoder.setBuffer(&m_ctx.meshletData.meshletVertices, 0, GpuDriven::DeferredLightingBindings::kMeshletVertices);
        encoder.setBuffer(&m_ctx.meshletData.meshletTriangles, 0, GpuDriven::DeferredLightingBindings::kMeshletTriangles);
        encoder.setBuffer(&m_ctx.sceneMesh.uvBuffer, 0, GpuDriven::DeferredLightingBindings::kUvs);
        encoder.setBuffer(&m_ctx.meshletData.materialIDs, 0, GpuDriven::DeferredLightingBindings::kMaterialIds);
        encoder.setBuffer(&m_ctx.materials.materialBuffer, 0, GpuDriven::DeferredLightingBindings::kMaterials);
        if (visibleMeshletsBuffer) {
            encoder.setBuffer(visibleMeshletsBuffer, 0, GpuDriven::DeferredLightingBindings::kVisibleMeshlets);
        }
        if (visibleMeshletStateBuffer) {
            encoder.setBuffer(visibleMeshletStateBuffer, 0, GpuDriven::DeferredLightingBindings::kVisibleMeshletState);
        }
        encoder.setBuffer(lodMeshletBuffer, 0, GpuDriven::DeferredLightingBindings::kLodMeshlets);
        encoder.setBuffer(lodMeshletVerticesBuffer, 0, GpuDriven::DeferredLightingBindings::kLodMeshletVertices);
        encoder.setBuffer(lodMeshletTrianglesBuffer, 0, GpuDriven::DeferredLightingBindings::kLodMeshletTriangles);
        encoder.setBuffer(lodMaterialIdsBuffer, 0, GpuDriven::DeferredLightingBindings::kLodMaterialIds);
        if (m_ctx.gpuScene.instanceBuffer.nativeHandle()) {
            encoder.setBuffer(&m_ctx.gpuScene.instanceBuffer,
                              0,
                              GpuDriven::DeferredLightingBindings::kInstanceData);
        }
        encoder.setTexture(m_frameGraph->getTexture(m_visRead), 0);
        encoder.setTexture(m_frameGraph->getTexture(m_depthRead), 1);
        encoder.setStorageTexture(m_frameGraph->getTexture(output), 2);
        // Vulkan reflection compacts resource bindings into dense logical slots.
        // Keep CPU-side binding indices aligned with shader declaration order.
        constexpr uint32_t kShadowTextureBinding = 3;
        constexpr uint32_t kSkyTextureBinding = 4;
        constexpr uint32_t kMotionVectorsBinding = 5;
        const RhiTexture* shadowTex = m_shadowRead.isValid()
            ? m_frameGraph->getTexture(m_shadowRead)
            : &m_ctx.shadowDummyTex;
        encoder.setTexture(shadowTex, kShadowTextureBinding);
        const RhiTexture* skyTex = m_skyRead.isValid()
            ? m_frameGraph->getTexture(m_skyRead)
            : &m_ctx.skyFallbackTex;
        encoder.setTexture(skyTex, kSkyTextureBinding);
        encoder.setStorageTexture(m_frameGraph->getTexture(motionVectorsOutput),
                                  kMotionVectorsBinding);
        if (!useBindlessSceneTextures && !m_ctx.materials.textureViews.empty()) {
            encoder.setTextures(m_ctx.materials.textureViews.data(),
                                METALLIC_METAL_DIRECT_BINDLESS_TEXTURE_BASE,
                                static_cast<uint32_t>(m_ctx.materials.textureViews.size()));
            encoder.setSampler(&m_ctx.materials.sampler,
                               METALLIC_METAL_DIRECT_BINDLESS_SAMPLER_BASE);
        }
        encoder.dispatchThreadgroups({static_cast<uint32_t>((m_width + 7) / 8), static_cast<uint32_t>((m_height + 7) / 8), 1},
                                     {8, 8, 1});
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        ImGui::SliderFloat("Motion Vector Intensity", &m_motionVectorIntensity, 0.0f, 2.0f, "%.2f");
        if (m_frameContext) {
            ImGui::Text("Scene Instances: %u", m_ctx.gpuScene.instanceCount);
            ImGui::Text("Meshlets: %u", m_frameContext->meshletCount);
            ImGui::Text("Materials: %u", m_frameContext->materialCount);
            ImGui::Text("Shadows: %s", m_frameContext->enableRTShadows ? "Enabled" : "Disabled");
        }
    }

private:
    const RenderContext& m_ctx;
    FGResource m_visRead, m_depthRead, m_shadowRead, m_skyRead;
    FGResource m_visibleMeshletsRead;
    FGResource m_visibleMeshletStateRead;
    int m_width, m_height;
    std::string m_name = "Deferred Lighting";
    float m_motionVectorIntensity = 1.0f;
};


