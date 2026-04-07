#pragma once

#include "cluster_streaming_service.h"
#include "gpu_cull_resources.h"
#include "pass_registry.h"
#include "render_pass.h"

class ClusterStreamingAgeFilterPass : public RenderPass {
public:
    ClusterStreamingAgeFilterPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~ClusterStreamingAgeFilterPass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    void setup(FGBuilder& builder) override {
        FGResource cullInput = getInput("cullResult");
        if (cullInput.isValid()) {
            m_cullResultRead = builder.read(cullInput);
        }
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        if (!m_runtimeContext || !m_frameContext) {
            return;
        }

        auto pipelineIt =
            m_runtimeContext->computePipelinesRhi.find("ClusterStreamingAgeFilterPass");
        if (pipelineIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipelineIt->second.nativeHandle()) {
            return;
        }

        ClusterStreamingService* streamingService = m_runtimeContext->clusterStreamingService;
        if (!streamingService ||
            !streamingService->ready() ||
            !streamingService->streamingEnabled() ||
            m_ctx.clusterLodData.totalGroupCount == 0u) {
            return;
        }

        const RhiBuffer* groupResidencyBuffer = streamingService->groupResidencyBuffer();
        const RhiBuffer* groupAgeBuffer = streamingService->groupAgeBuffer();
        const RhiBuffer* unloadRequestBuffer = streamingService->unloadRequestBuffer();
        const RhiBuffer* unloadRequestStateBuffer = streamingService->unloadRequestStateBuffer();
        if (!groupResidencyBuffer || !groupAgeBuffer || !unloadRequestBuffer || !unloadRequestStateBuffer) {
            return;
        }

        StreamingAgeFilterUniforms uniforms{};
        uniforms.groupCount = m_ctx.clusterLodData.totalGroupCount;
        uniforms.ageThreshold = streamingService->ageThreshold();

        encoder.setComputePipeline(pipelineIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), GpuDriven::StreamingAgeFilterBindings::kUniforms);
        encoder.setBuffer(groupResidencyBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kGroupResidency);
        encoder.setBuffer(groupAgeBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kGroupAge);
        encoder.setBuffer(unloadRequestBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kUnloadRequests);
        encoder.setBuffer(unloadRequestStateBuffer,
                          0,
                          GpuDriven::StreamingAgeFilterBindings::kUnloadRequestState);

        constexpr uint32_t kThreadCount = 64u;
        const uint32_t dispatchX = (uniforms.groupCount + kThreadCount - 1u) / kThreadCount;
        encoder.dispatchThreadgroups({dispatchX, 1, 1}, {kThreadCount, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);
    }

private:
    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Streaming Age Filter";
    FGResource m_cullResultRead;
};
