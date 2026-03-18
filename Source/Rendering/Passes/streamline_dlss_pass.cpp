#ifdef _WIN32
#ifdef METALLIC_HAS_STREAMLINE

#include "streamline_dlss_pass.h"
#include "vulkan_resource_handles.h"

#include <spdlog/spdlog.h>

void StreamlineDlssPass::executeDlss(RhiComputeCommandEncoder& encoder) {
    StreamlineContext* streamlineCtx = currentStreamlineContext();
    if (!streamlineCtx || !m_frameContext) return;

    RhiTexture* colorTex = m_sourceRead.isValid() ? m_frameGraph->getTexture(m_sourceRead) : nullptr;
    RhiTexture* depthTex = m_depthRead.isValid() ? m_frameGraph->getTexture(m_depthRead) : nullptr;
    RhiTexture* motionTex = m_motionRead.isValid() ? m_frameGraph->getTexture(m_motionRead) : nullptr;
    RhiTexture* outputTex = m_frameGraph->getTexture(m_dlssOutput);

    if (!colorTex || !depthTex || !motionTex || !outputTex) {
        spdlog::warn("StreamlineDlssPass: missing input textures");
        return;
    }

    StreamlineDlssFrameData data{};
    data.colorInput = getVulkanImage(colorTex);
    data.colorInputView = getVulkanImageView(colorTex);
    data.colorOutput = getVulkanImage(outputTex);
    data.colorOutputView = getVulkanImageView(outputTex);
    data.depth = getVulkanImage(depthTex);
    data.depthView = getVulkanImageView(depthTex);
    data.motionVectors = getVulkanImage(motionTex);
    data.motionVectorsView = getVulkanImageView(motionTex);

    data.renderWidth = static_cast<uint32_t>(currentRenderWidth());
    data.renderHeight = static_cast<uint32_t>(currentRenderHeight());
    data.displayWidth = static_cast<uint32_t>(currentDisplayWidth());
    data.displayHeight = static_cast<uint32_t>(currentDisplayHeight());

    data.jitterOffsetX = m_frameContext->jitterOffset.x;
    data.jitterOffsetY = m_frameContext->jitterOffset.y;

    // Motion vectors are in UV space (currentUV - prevUV), not jittered.
    // Streamline expects pixel-space motion vectors, so scale by render resolution.
    data.mvecScaleX = static_cast<float>(currentRenderWidth());
    data.mvecScaleY = static_cast<float>(currentRenderHeight());
    data.motionVectorsJittered = false;
    data.depthInverted = (m_frameContext->depthClearValue == 0.0);
    data.reset = m_frameContext->historyReset;

    data.frameIndex = m_frameContext->frameIndex;
    data.commandBuffer = encoder.nativeHandle(); // VkCommandBuffer

    // Build clipToPrevClip = prevViewProj * inv(currentViewProj)
    // FrameContext provides view/proj as column-major float4x4.
    {
        float4x4 currentVP = m_frameContext->proj * m_frameContext->view;
        float4x4 invCurrentVP = currentVP;
        invCurrentVP.Invert();
        float4x4 prevVP = m_frameContext->prevProj * m_frameContext->prevView;
        float4x4 clipToPrevClip = prevVP * invCurrentVP;
        memcpy(data.clipToPrevClip, &clipToPrevClip, sizeof(float) * 16);
    }

    if (!streamlineCtx->evaluate(data)) {
        spdlog::warn("StreamlineDlssPass: evaluate failed");
    }
}

#endif // METALLIC_HAS_STREAMLINE
#endif // _WIN32
