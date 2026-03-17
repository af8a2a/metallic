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

    ~MeshletCullPass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void setFrameContext(const FrameContext* ctx) override {
        RenderPass::setFrameContext(ctx);
        syncFrameContextFlags();
    }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource cullResult;

    FGResource getOutput(const std::string& name) const override {
        if (name == "cullResult") return cullResult;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        cullResult = builder.createToken("cullResult");
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("MeshletCullPass");
        MICROPROFILE_SCOPEI("RenderPass", "MeshletCullPass", 0xffff8800);
        if (!m_frameContext || !m_runtimeContext) return;
        if (!m_frameContext->gpuDrivenCulling) return;

        auto cullIt = m_runtimeContext->computePipelinesRhi.find("MeshletCullPass");
        auto buildIt = m_runtimeContext->computePipelinesRhi.find("BuildIndirectPass");
        if (cullIt == m_runtimeContext->computePipelinesRhi.end() || !cullIt->second.nativeHandle()) return;
        if (buildIt == m_runtimeContext->computePipelinesRhi.end() || !buildIt->second.nativeHandle()) return;

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

        m_totalMeshlets = totalMeshlets;

        // Ensure GPU buffers are large enough
        ensureBuffers(totalMeshlets, instanceCount);

        // Read back previous frame's visible count (1-frame delayed).
        // build_indirect writes count to offset 4 (indirect args x) then resets offset 0.
        // By this point the previous frame's GPU work is complete.
        auto* counterPtr = static_cast<uint32_t*>(m_counterBuffer->mappedData());
        m_lastVisibleCount = counterPtr[1]; // indirect args x from previous frame

        // Upload instance data (CPU PU, StorageModeShared)
        auto* instPtr = static_cast<GPUInstanceData*>(m_instanceDataBuffer->mappedData());
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
        encoder.setComputePipeline(cullIt->second);
        encoder.setBytes(&cullUni, sizeof(cullUni), 0);
        encoder.setBuffer(m_instanceDataBuffer.get(), 0, 1);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, 2);
        encoder.setBuffer(m_visibleMeshletBuffer.get(), 0, 3);
        encoder.setBuffer(m_counterBuffer.get(), 0, 4);

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (totalMeshlets + threadgroupSize - 1) / threadgroupSize;
        encoder.dispatchThreadgroups({threadgroups, 1, 1}, {threadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // --- Dispatch 2: Build indirect args ---
        encoder.setComputePipeline(buildIt->second);
        encoder.setBuffer(m_counterBuffer.get(), 0, 0);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Publish results to FrameContext for VisibilityPass
        // (const_cast is safe here 鈥?we own the data and VisibilityPass reads it later in the same frame)
        auto* mutableCtx = const_cast<FrameContext*>(m_frameContext);
        mutableCtx->gpuVisibleMeshletBufferRhi = m_visibleMeshletBuffer.get();
        mutableCtx->gpuCounterBufferRhi = m_counterBuffer.get();
        mutableCtx->gpuInstanceDataBufferRhi = m_instanceDataBuffer.get();

        // m_lastVisibleCount is set at the top of this function via 1-frame-delayed readback
    }

    void renderUI() override {
        ImGui::Text("Total Meshlets: %u", m_totalMeshlets);
        ImGui::Text("Visible Meshlets: %u", m_lastVisibleCount);
        if (m_totalMeshlets > 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(m_totalMeshlets);
            ImGui::Text("Cull Rate: %.1f%%", cullRate * 100.0f);
        }
        if (ImGui::Checkbox("Frustum Cull", &m_enableFrustumCull)) {
            syncFrameContextFlags();
        }
        if (ImGui::Checkbox("Cone Cull", &m_enableConeCull)) {
            syncFrameContextFlags();
        }
        if (m_frameContext) {
            ImGui::Text("Instances: %u", m_frameContext->visibilityInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
    }

private:
    void syncFrameContextFlags() {
        if (!m_frameContext) {
            return;
        }
        auto* frameContext = const_cast<FrameContext*>(m_frameContext);
        frameContext->enableFrustumCull = m_enableFrustumCull;
        frameContext->enableConeCull = m_enableConeCull;
    }

    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Meshlet Cull";

    std::unique_ptr<RhiBuffer> m_visibleMeshletBuffer;
    std::unique_ptr<RhiBuffer> m_counterBuffer;
    std::unique_ptr<RhiBuffer> m_instanceDataBuffer;

    uint32_t m_maxMeshlets = 0;
    uint32_t m_maxInstances = 0;
    uint32_t m_totalMeshlets = 0;
    uint32_t m_lastVisibleCount = 0;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;

    void ensureBuffers(uint32_t totalMeshlets, uint32_t instanceCount) {
        auto* factory = m_runtimeContext->resourceFactory;
        if (!factory) return;

        if (!m_counterBuffer) {
            RhiBufferDesc desc;
            desc.size = kCounterBufferSize;
            desc.hostVisible = true;
            desc.debugName = "CullCounterBuffer";
            m_counterBuffer = factory->createBuffer(desc);
            // Zero-initialize: atomic counter = 0, indirect args = {0, 1, 1}
            auto* ptr = static_cast<uint32_t*>(m_counterBuffer->mappedData());
            ptr[0] = 0; // atomic counter
            ptr[1] = 0; // indirect args x (will be filled by build_indirect)
            ptr[2] = 1; // indirect args y
            ptr[3] = 1; // indirect args z
        }

        if (totalMeshlets > m_maxMeshlets) {
            m_maxMeshlets = totalMeshlets;
            RhiBufferDesc desc;
            desc.size = m_maxMeshlets * sizeof(MeshletDrawInfo);
            desc.hostVisible = false;
            desc.debugName = "VisibleMeshletBuffer";
            m_visibleMeshletBuffer = factory->createBuffer(desc);
        }

        if (instanceCount > m_maxInstances) {
            m_maxInstances = instanceCount;
            RhiBufferDesc desc;
            desc.size = m_maxInstances * sizeof(GPUInstanceData);
            desc.hostVisible = true;
            desc.debugName = "InstanceDataBuffer";
            m_instanceDataBuffer = factory->createBuffer(desc);
        }
    }
};



