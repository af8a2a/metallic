#pragma once

#include "cluster_streaming_service.h"
#include "gpu_cull_resources.h"
#include "pass_registry.h"
#include "render_pass.h"

class ClusterStreamingUpdatePass : public RenderPass {
public:
    ClusterStreamingUpdatePass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~ClusterStreamingUpdatePass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "streamingSync") {
            return m_streamingSync;
        }
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_streamingSync = builder.createToken("ClusterStreamingSync");
    }

    void prepareResources(RhiCommandBuffer&) override {
        if (!m_runtimeContext || !m_runtimeContext->clusterStreamingService) {
            return;
        }

        m_runtimeContext->clusterStreamingService->runUpdateStage(m_ctx.clusterLodData,
                                                                  *m_runtimeContext,
                                                                  m_frameContext);
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        if (!m_runtimeContext || !m_runtimeContext->clusterStreamingService) {
            return;
        }

        auto pipelineIt = m_runtimeContext->computePipelinesRhi.find("ClusterStreamingUpdatePass");
        if (pipelineIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipelineIt->second.nativeHandle()) {
            return;
        }

        ClusterStreamingService* streamingService = m_runtimeContext->clusterStreamingService;
        if (!streamingService->streamingEnabled() || !streamingService->ready()) {
            return;
        }

        const uint32_t patchCount = streamingService->streamingPatchCount();
        if (patchCount == 0u) {
            return;
        }

        const RhiBuffer* sourceGroupMeshletIndicesBuffer =
            m_ctx.clusterLodData.groupMeshletIndicesBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.groupMeshletIndicesBuffer
                : nullptr;
        const RhiBuffer* residentGroupMeshletIndicesBuffer =
            streamingService->residentGroupMeshletIndicesBuffer();
        const RhiBuffer* lodGroupPageTableBuffer = streamingService->lodGroupPageTableBuffer();
        const RhiBuffer* patchBuffer = streamingService->streamingPatchBuffer();
        if (!sourceGroupMeshletIndicesBuffer ||
            !residentGroupMeshletIndicesBuffer ||
            !lodGroupPageTableBuffer ||
            !patchBuffer) {
            return;
        }

        StreamingUpdateUniforms uniforms{};
        uniforms.patchCount = patchCount;

        encoder.setComputePipeline(pipelineIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), GpuDriven::StreamingUpdateBindings::kUniforms);
        encoder.setBuffer(sourceGroupMeshletIndicesBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kSourceGroupMeshletIndices);
        encoder.setBuffer(residentGroupMeshletIndicesBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kResidentGroupMeshletIndices);
        encoder.setBuffer(lodGroupPageTableBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kGroupPageTable);
        encoder.setBuffer(patchBuffer, 0, GpuDriven::StreamingUpdateBindings::kPatches);

        constexpr uint32_t kThreadCount = 64u;
        const uint32_t dispatchX = (patchCount + kThreadCount - 1u) / kThreadCount;
        encoder.dispatchThreadgroups({dispatchX, 1, 1}, {kThreadCount, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);
    }

private:
    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Streaming Update";
    FGResource m_streamingSync;
};
