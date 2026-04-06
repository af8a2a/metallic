#pragma once

#include "cluster_streaming_service.h"
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

private:
    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Streaming Update";
    FGResource m_streamingSync;
};
