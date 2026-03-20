#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "hzb_constants.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

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
        if (config.config.contains("phase")) {
            m_phase = config.config["phase"].get<int>();
        }
    }

    FGResource cullResult;

    FGResource getOutput(const std::string& name) const override {
        if (name == "cullResult") return cullResult;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        cullResult = builder.createToken("cullResult");

        if (m_phase == 0) {
            // Phase 1: read HZB history (previous frame's HZB)
            m_hzbHistoryRead.clear();
            m_hzbLevelCount = computeHzbLevelCount(static_cast<uint32_t>(m_width),
                                                   static_cast<uint32_t>(m_height));
            m_hzbHistoryRead.reserve(m_hzbLevelCount);
            for (uint32_t level = 0; level < m_hzbLevelCount; ++level) {
                const std::string resourceName = hzbHistoryResourceName(level);
                m_hzbHistoryRead.push_back(
                    builder.readHistory(resourceName.c_str(),
                                        makeHzbTextureDesc(static_cast<uint32_t>(m_width),
                                                           static_cast<uint32_t>(m_height),
                                                           level)));
            }
        } else {
            // Phase 2: read depth input to establish dependency on HZBBuildPass_Mid.
            // Actual HZB textures come from FrameContext at execute time.
            FGResource depthInput = getInput("depth");
            if (depthInput.isValid()) {
                builder.read(depthInput);
            }
            m_hzbLevelCount = computeHzbLevelCount(static_cast<uint32_t>(m_width),
                                                   static_cast<uint32_t>(m_height));
        }
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

        if (m_phase == 0) {
            executePhase1(encoder, cullIt->second, buildIt->second);
        } else {
            executePhase2(encoder, cullIt->second, buildIt->second);
        }
    }

    void renderUI() override {
        ImGui::Text("Phase: %d", m_phase);
        ImGui::Text("Total Meshlets: %u", m_totalMeshlets);
        ImGui::Text("Visible Meshlets: %u", m_lastVisibleCount);
        if (m_phase == 1) {
            ImGui::Text("Phase2 Input (occ-failed): %u", m_lastPhase2InputCount);
        }
        if (m_totalMeshlets > 0 && m_phase == 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(m_totalMeshlets);
            ImGui::Text("Cull Rate: %.1f%%", cullRate * 100.0f);
        }
        if (m_phase == 0) {
            if (ImGui::Checkbox("Frustum Cull", &m_enableFrustumCull)) {
                syncFrameContextFlags();
            }
            if (ImGui::Checkbox("Cone Cull", &m_enableConeCull)) {
                syncFrameContextFlags();
            }
            ImGui::Checkbox("HZB Occlusion Cull", &m_enableOcclusionCull);
            ImGui::SliderFloat("HZB Depth Bias", &m_occlusionDepthBias, 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("HZB Bounds Scale", &m_occlusionBoundsScale, 1.0f, 1.5f, "%.2f");
        }
        if (m_frameContext) {
            ImGui::Text("Instances: %u", m_frameContext->visibilityInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
        if (m_phase == 0) {
            const bool historyValid =
                m_frameGraph && !m_hzbHistoryRead.empty() && m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
            ImGui::Text("HZB History: %s (%u levels)", historyValid ? "Ready" : "Warming Up", m_hzbLevelCount);
        } else {
            ImGui::Text("HZB from FrameContext: %u levels", m_frameContext ? m_frameContext->hzbMipCount : 0u);
        }
    }

private:
    void executePhase1(RhiComputeCommandEncoder& encoder,
                       RhiComputePipelineHandle cullPipeline,
                       RhiComputePipelineHandle buildPipeline) {
        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = std::min<uint32_t>(
            m_frameContext->visibilityInstanceCount,
            static_cast<uint32_t>(visibleNodes.size()));
        if (instanceCount == 0) return;

        std::vector<uint32_t> validVisibleNodes;
        validVisibleNodes.reserve(instanceCount);
        for (uint32_t i = 0; i < instanceCount; ++i) {
            const uint32_t nodeID = visibleNodes[i];
            if (nodeID < m_ctx.sceneGraph.nodes.size()) {
                validVisibleNodes.push_back(nodeID);
            }
        }
        if (validVisibleNodes.empty()) return;
        instanceCount = static_cast<uint32_t>(validVisibleNodes.size());

        // Compute total meshlet count across all visible instances
        uint32_t totalMeshlets = 0;
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[validVisibleNodes[i]];
            totalMeshlets += node.meshletCount;
        }
        if (totalMeshlets == 0) return;

        m_totalMeshlets = totalMeshlets;

        // Ensure GPU buffers are large enough
        ensureBuffers(totalMeshlets, instanceCount);

        // Read back previous frame's visible count (1-frame delayed)
        auto* counterPtr = static_cast<uint32_t*>(m_counterBuffer->mappedData());
        m_lastVisibleCount = counterPtr[1]; // indirect args x from previous frame

        // Upload instance data
        auto* instPtr = static_cast<GPUInstanceData*>(m_instanceDataBuffer->mappedData());
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[validVisibleNodes[i]];
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
        CullUniforms cullUni = buildCullUniforms(instanceCount, totalMeshlets);
        cullUni.cullingPhase = 0;

        // Gather HZB textures from history (previous frame)
        std::array<const RhiTexture*, kHzbMaxLevels> hzbTextures{};
        uint32_t hzbTextureCount = 0;
        const bool historyValid =
            m_enableOcclusionCull &&
            !m_hzbHistoryRead.empty() &&
            m_frameGraph &&
            m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
        if (historyValid) {
            const uint32_t maxLevels =
                std::min<uint32_t>(static_cast<uint32_t>(m_hzbHistoryRead.size()), kHzbMaxLevels);
            for (uint32_t level = 0; level < maxLevels; ++level) {
                const RhiTexture* historyTexture = m_frameGraph->getTexture(m_hzbHistoryRead[level]);
                if (!historyTexture) break;
                hzbTextures[hzbTextureCount++] = historyTexture;
            }
        }

        cullUni.enableOcclusionCull = hzbTextureCount > 0 ? 1u : 0u;
        cullUni.hzbLevelCount = hzbTextureCount;
        if (hzbTextureCount > 0) {
            cullUni.hzbTextureSize =
                float2(static_cast<float>(hzbTextures[0]->width()),
                       static_cast<float>(hzbTextures[0]->height()));
        }
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;

        // --- Dispatch 1: Meshlet cull (phase 1) ---
        encoder.setComputePipeline(cullPipeline);
        encoder.setBytes(&cullUni, sizeof(cullUni), 0);
        encoder.setBuffer(m_instanceDataBuffer.get(), 0, 1);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, 2);
        encoder.setBuffer(m_visibleMeshletBuffer.get(), 0, 3);
        encoder.setBuffer(m_counterBuffer.get(), 0, 4);
        encoder.setBuffer(m_occlusionFailedBuffer.get(), 0, 5);
        // buffer(6) = phase2InputMeshlets — not used in phase 1, bind dummy (same buffer)
        encoder.setBuffer(m_occlusionFailedBuffer.get(), 0, 6);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(), 7, hzbTextureCount);
        }

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (totalMeshlets + threadgroupSize - 1) / threadgroupSize;
        encoder.dispatchThreadgroups({threadgroups, 1, 1}, {threadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // --- Dispatch 2: Build indirect args (phase 0) ---
        struct { uint32_t phase; } buildUni{0};
        encoder.setComputePipeline(buildPipeline);
        encoder.setBytes(&buildUni, sizeof(buildUni), 0);
        encoder.setBuffer(m_counterBuffer.get(), 0, 1);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Publish results to FrameContext
        auto* mutableCtx = const_cast<FrameContext*>(m_frameContext);
        mutableCtx->gpuVisibleMeshletBufferRhi = m_visibleMeshletBuffer.get();
        mutableCtx->gpuCounterBufferRhi = m_counterBuffer.get();
        mutableCtx->gpuInstanceDataBufferRhi = m_instanceDataBuffer.get();
        mutableCtx->gpuOcclusionFailedBufferRhi = m_occlusionFailedBuffer.get();

        static bool sLoggedGpuPublish = false;
        if (!sLoggedGpuPublish) {
            spdlog::info(
                "MeshletCullPass Phase1 published GPU cull buffers: instances={} meshlets={} visibleBuf={} counterBuf={} instanceBuf={} occFailedBuf={}",
                instanceCount,
                totalMeshlets,
                fmt::ptr(mutableCtx->gpuVisibleMeshletBufferRhi),
                fmt::ptr(mutableCtx->gpuCounterBufferRhi),
                fmt::ptr(mutableCtx->gpuInstanceDataBufferRhi),
                fmt::ptr(mutableCtx->gpuOcclusionFailedBufferRhi));
            sLoggedGpuPublish = true;
        }
    }

    void executePhase2(RhiComputeCommandEncoder& encoder,
                       RhiComputePipelineHandle cullPipeline,
                       RhiComputePipelineHandle buildPipeline) {
        auto* mutableCtx = const_cast<FrameContext*>(m_frameContext);

        // Read occlusion-failed buffer from FrameContext (written by phase 1)
        RhiBuffer* occFailedBuffer = mutableCtx->gpuOcclusionFailedBufferRhi;
        RhiBuffer* counterBuffer = mutableCtx->gpuCounterBufferRhi;
        RhiBuffer* instanceDataBuffer = mutableCtx->gpuInstanceDataBufferRhi;
        if (!occFailedBuffer || !counterBuffer || !instanceDataBuffer) return;

        // Read the occlusion-failed count from counter buffer offset 20
        // (build_indirect phase 0 wrote it there as indirect args x)
        auto* counterPtr = static_cast<uint32_t*>(counterBuffer->mappedData());
        uint32_t phase2DispatchCount = counterPtr[5]; // offset 20 / 4 = index 5
        m_lastPhase2InputCount = phase2DispatchCount;
        if (phase2DispatchCount == 0) return;

        // Read HZB textures from FrameContext (written by HZBBuildPass_Mid)
        std::array<const RhiTexture*, kHzbMaxLevels> hzbTextures{};
        uint32_t hzbTextureCount = 0;
        if (m_frameContext->hzbMipCount > 0) {
            for (uint32_t level = 0; level < m_frameContext->hzbMipCount; ++level) {
                if (!m_frameContext->hzbMipTextures[level]) break;
                hzbTextures[hzbTextureCount++] = m_frameContext->hzbMipTextures[level];
            }
        }

        // Build CullUniforms for phase 2
        uint32_t instanceCount = std::min<uint32_t>(
            m_frameContext->visibilityInstanceCount,
            static_cast<uint32_t>(m_frameContext->visibleMeshletNodes.size()));
        CullUniforms cullUni = buildCullUniforms(instanceCount, phase2DispatchCount);
        cullUni.cullingPhase = 1;
        cullUni.enableFrustumCull = 0; // already passed in phase 1
        cullUni.enableConeCull = 0;    // already passed in phase 1
        cullUni.enableOcclusionCull = hzbTextureCount > 0 ? 1u : 0u;
        cullUni.hzbLevelCount = hzbTextureCount;
        if (hzbTextureCount > 0) {
            cullUni.hzbTextureSize = m_frameContext->hzbTextureSize;
        }
        // For phase 2, use current frame's viewProj for occlusion test
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        cullUni.prevViewProj = transpose(viewProj);
        cullUni.prevView = transpose(m_frameContext->view);
        cullUni.prevCameraWorldPos = m_frameContext->cameraWorldPos;
        cullUni.prevProjScale = float2(std::abs(m_frameContext->proj[0].x),
                                       std::abs(m_frameContext->proj[1].y));
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;

        // Reset the occlusion-failed counter before phase 2 cull writes visible results
        // (visible counter at offset 0 was already reset by build_indirect phase 0)
        // Reset occ-failed counter at offset 16 so it's clean for next frame's phase 1
        counterPtr[4] = 0; // offset 16 / 4 = index 4

        // --- Dispatch 1: Meshlet cull (phase 2) ---
        encoder.setComputePipeline(cullPipeline);
        encoder.setBytes(&cullUni, sizeof(cullUni), 0);
        encoder.setBuffer(instanceDataBuffer, 0, 1);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, 2);
        // Phase 2 appends to the SAME visible meshlet buffer (after phase 1's results)
        encoder.setBuffer(mutableCtx->gpuVisibleMeshletBufferRhi, 0, 3);
        encoder.setBuffer(counterBuffer, 0, 4);
        encoder.setBuffer(occFailedBuffer, 0, 5); // RW but not written in phase 2
        encoder.setBuffer(occFailedBuffer, 0, 6); // phase2InputMeshlets (read)
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(), 7, hzbTextureCount);
        }

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (phase2DispatchCount + threadgroupSize - 1) / threadgroupSize;
        encoder.dispatchThreadgroups({threadgroups, 1, 1}, {threadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // --- Dispatch 2: Build indirect args (phase 1) ---
        struct { uint32_t phase; } buildUni{1};
        encoder.setComputePipeline(buildPipeline);
        encoder.setBytes(&buildUni, sizeof(buildUni), 0);
        encoder.setBuffer(counterBuffer, 0, 1);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Overwrite FrameContext pointers so VisibilityPass_Phase2 picks up phase 2 results
        // (visible buffer and counter buffer are the same objects, just updated)
        mutableCtx->gpuVisibleMeshletBufferRhi = mutableCtx->gpuVisibleMeshletBufferRhi;
        mutableCtx->gpuCounterBufferRhi = counterBuffer;
    }

    CullUniforms buildCullUniforms(uint32_t instanceCount, uint32_t totalDispatchCount) {
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        float4x4 prevViewProj = m_frameContext->prevCullProj * m_frameContext->prevCullView;
        CullUniforms cullUni{};
        cullUni.viewProj = transpose(viewProj);
        cullUni.prevViewProj = transpose(prevViewProj);
        cullUni.prevView = transpose(m_frameContext->prevCullView);
        cullUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        cullUni.prevCameraWorldPos = m_frameContext->prevCameraWorldPos;
        cullUni.prevProjScale = float2(std::abs(m_frameContext->prevCullProj[0].x),
                                       std::abs(m_frameContext->prevCullProj[1].y));
        cullUni.totalDispatchCount = totalDispatchCount;
        cullUni.instanceCount = instanceCount;
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;
        cullUni.cullingPhase = 0;
        cullUni._pad0 = 0;
        return cullUni;
    }

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
    int m_phase = 0; // 0 = phase1, 1 = phase2
    std::string m_name = "Meshlet Cull";

    std::unique_ptr<RhiBuffer> m_visibleMeshletBuffer;
    std::unique_ptr<RhiBuffer> m_counterBuffer;
    std::unique_ptr<RhiBuffer> m_instanceDataBuffer;
    std::unique_ptr<RhiBuffer> m_occlusionFailedBuffer;

    uint32_t m_maxMeshlets = 0;
    uint32_t m_maxInstances = 0;
    uint32_t m_totalMeshlets = 0;
    uint32_t m_lastVisibleCount = 0;
    uint32_t m_lastPhase2InputCount = 0;
    uint32_t m_hzbLevelCount = 0;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;
    bool m_enableOcclusionCull = true;
    float m_occlusionDepthBias = 0.0015f;
    float m_occlusionBoundsScale = 1.1f;
    std::vector<FGResource> m_hzbHistoryRead;

    void ensureBuffers(uint32_t totalMeshlets, uint32_t instanceCount) {
        auto* factory = m_runtimeContext->resourceFactory;
        if (!factory) return;

        if (!m_counterBuffer) {
            RhiBufferDesc desc;
            desc.size = kCounterBufferSize;
            desc.hostVisible = true;
            desc.debugName = "CullCounterBuffer";
            m_counterBuffer = factory->createBuffer(desc);
            // Zero-initialize all 32 bytes
            auto* ptr = static_cast<uint32_t*>(m_counterBuffer->mappedData());
            ptr[0] = 0; // atomic counter
            ptr[1] = 0; // indirect args x (will be filled by build_indirect)
            ptr[2] = 1; // indirect args y
            ptr[3] = 1; // indirect args z
            ptr[4] = 0; // occlusion-failed counter
            ptr[5] = 0; // occ-failed indirect args x
            ptr[6] = 1; // occ-failed indirect args y
            ptr[7] = 1; // occ-failed indirect args z
        }

        if (totalMeshlets > m_maxMeshlets) {
            m_maxMeshlets = totalMeshlets;
            {
                RhiBufferDesc desc;
                desc.size = m_maxMeshlets * sizeof(MeshletDrawInfo);
                desc.hostVisible = false;
                desc.debugName = "VisibleMeshletBuffer";
                m_visibleMeshletBuffer = factory->createBuffer(desc);
            }
            {
                RhiBufferDesc desc;
                desc.size = m_maxMeshlets * sizeof(MeshletDrawInfo);
                desc.hostVisible = false;
                desc.debugName = "OcclusionFailedMeshletBuffer";
                m_occlusionFailedBuffer = factory->createBuffer(desc);
            }
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
