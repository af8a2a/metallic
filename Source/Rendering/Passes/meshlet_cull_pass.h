#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>

class MeshletCullPass : public RenderPass {
public:
    MeshletCullPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~MeshletCullPass() override {
        if (m_visibleMeshletBuffer) m_visibleMeshletBuffer->release();
        if (m_counterBuffer) m_counterBuffer->release();
        if (m_instanceDataBuffer) m_instanceDataBuffer->release();
    }

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource cullResult;

    FGResource getOutput(const std::string& name) const override {
        if (name == "cullResult") return cullResult;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Dummy 1x1 texture just for frame graph dependency ordering
        cullResult = builder.create("cullResult",
            FGTextureDesc::storageTexture(1, 1, MTL::PixelFormatR8Unorm));
        builder.setSideEffect();
    }

    void executeCompute(MTL::ComputeCommandEncoder* enc) override {
        ZoneScopedN("MeshletCullPass");
        if (!m_frameContext || !m_runtimeContext) return;
        if (!m_frameContext->gpuDrivenCulling) return;

        auto cullIt = m_runtimeContext->computePipelines.find("MeshletCullPass");
        auto buildIt = m_runtimeContext->computePipelines.find("BuildIndirectPass");
        if (cullIt == m_runtimeContext->computePipelines.end() || !cullIt->second) return;
        if (buildIt == m_runtimeContext->computePipelines.end() || !buildIt->second) return;

        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = m_frameContext->visibilityInstanceCount;
        if (instanceCount == 0) return;

        // Compute total meshlet count across all visible instances
        uint32_t totalMeshlets = 0;
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[visibleNodes[i]];
            totalMeshlets += node.meshletCount;
        }
        if (totalMeshlets == 0) return;

        // Ensure GPU buffers are large enough
        ensureBuffers(totalMeshlets, instanceCount);

        // Upload instance data (CPU → GPU, StorageModeShared)
        auto* instPtr = static_cast<GPUInstanceData*>(m_instanceDataBuffer->contents());
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[visibleNodes[i]];
            float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
            float4x4 nodeMVP = m_frameContext->proj * nodeModelView;

            GPUInstanceData& inst = instPtr[i];
            inst.mvp = transpose(nodeMVP);
            inst.modelView = transpose(nodeModelView);
            inst.worldMatrix = transpose(node.transform.worldMatrix);
            inst.meshletStart = node.meshletStart;
            inst.meshletCount = node.meshletCount;
            inst.instanceID = i;
            inst.pad = 0;
        }

        // Build CullUniforms
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        CullUniforms cullUni{};
        cullUni.viewProj = transpose(viewProj);
        cullUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        cullUni.totalDispatchCount = totalMeshlets;
        cullUni.instanceCount = instanceCount;
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;

        // --- Dispatch 1: Meshlet cull ---
        enc->setComputePipelineState(cullIt->second);
        enc->setBytes(&cullUni, sizeof(cullUni), 0);
        enc->setBuffer(m_instanceDataBuffer, 0, 1);
        enc->setBuffer(m_ctx.meshletData.boundsBuffer, 0, 2);
        enc->setBuffer(m_visibleMeshletBuffer, 0, 3);
        enc->setBuffer(m_counterBuffer, 0, 4);

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (totalMeshlets + threadgroupSize - 1) / threadgroupSize;
        enc->dispatchThreadgroups(MTL::Size(threadgroups, 1, 1), MTL::Size(threadgroupSize, 1, 1));

        // --- Dispatch 2: Build indirect args ---
        enc->setComputePipelineState(buildIt->second);
        enc->setBuffer(m_counterBuffer, 0, 0);
        enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

        // Publish results to FrameContext for VisibilityPass
        // (const_cast is safe here — we own the data and VisibilityPass reads it later in the same frame)
        auto* mutableCtx = const_cast<FrameContext*>(m_frameContext);
        mutableCtx->gpuVisibleMeshletBuffer = m_visibleMeshletBuffer;
        mutableCtx->gpuCounterBuffer = m_counterBuffer;
        mutableCtx->gpuInstanceDataBuffer = m_instanceDataBuffer;

        m_lastVisibleCount = totalMeshlets;
    }

    void renderUI() override {
        ImGui::Text("Total Meshlets: %u", m_lastVisibleCount);
        if (m_frameContext) {
            ImGui::Text("Instances: %u", m_frameContext->visibilityInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Meshlet Cull";

    MTL::Buffer* m_visibleMeshletBuffer = nullptr;
    MTL::Buffer* m_counterBuffer = nullptr;
    MTL::Buffer* m_instanceDataBuffer = nullptr;

    uint32_t m_maxMeshlets = 0;
    uint32_t m_maxInstances = 0;
    uint32_t m_lastVisibleCount = 0;

    void ensureBuffers(uint32_t totalMeshlets, uint32_t instanceCount) {
        auto* device = m_runtimeContext->device;

        if (!m_counterBuffer) {
            m_counterBuffer = device->newBuffer(kCounterBufferSize, MTL::ResourceStorageModeShared);
            m_counterBuffer->setLabel(NS::String::string("CullCounterBuffer", NS::UTF8StringEncoding));
            // Zero-initialize: atomic counter = 0, indirect args = {0, 1, 1}
            auto* ptr = static_cast<uint32_t*>(m_counterBuffer->contents());
            ptr[0] = 0; // atomic counter
            ptr[1] = 0; // indirect args x (will be filled by build_indirect)
            ptr[2] = 1; // indirect args y
            ptr[3] = 1; // indirect args z
        }

        if (totalMeshlets > m_maxMeshlets) {
            if (m_visibleMeshletBuffer) m_visibleMeshletBuffer->release();
            m_maxMeshlets = totalMeshlets;
            m_visibleMeshletBuffer = device->newBuffer(
                m_maxMeshlets * sizeof(MeshletDrawInfo), MTL::ResourceStorageModePrivate);
            m_visibleMeshletBuffer->setLabel(NS::String::string("VisibleMeshletBuffer", NS::UTF8StringEncoding));
        }

        if (instanceCount > m_maxInstances) {
            if (m_instanceDataBuffer) m_instanceDataBuffer->release();
            m_maxInstances = instanceCount;
            m_instanceDataBuffer = device->newBuffer(
                m_maxInstances * sizeof(GPUInstanceData), MTL::ResourceStorageModeShared);
            m_instanceDataBuffer->setLabel(NS::String::string("InstanceDataBuffer", NS::UTF8StringEncoding));
        }
    }
};
