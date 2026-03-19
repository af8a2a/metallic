#ifdef _WIN32
#ifdef METALLIC_HAS_STREAMLINE

#include "streamline_dlss_pass.h"
#include "vulkan_resource_handles.h"
#include "vulkan_image_tracker.h"

#include <spdlog/spdlog.h>
#include <algorithm>

namespace {

bool isDepthFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

VkImageUsageFlags toTrackedUsage(RhiTextureUsage usage, VkFormat format) {
    VkImageUsageFlags flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((usage & RhiTextureUsage::RenderTarget) != RhiTextureUsage::None) {
        flags |= isDepthFormat(format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                       : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderRead) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderWrite) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    return flags;
}

VkImageLayout inferLayout(const RhiTexture* texture, VkImageLayout fallback) {
    const VulkanTextureResource* resource = getVulkanTextureResource(texture);
    if (!resource) {
        return fallback;
    }
    if ((resource->usage & RhiTextureUsage::ShaderWrite) != RhiTextureUsage::None) {
        return VK_IMAGE_LAYOUT_GENERAL;
    }
    if (isDepthFormat(resource->format)) {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if ((resource->usage & RhiTextureUsage::RenderTarget) != RhiTextureUsage::None) {
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if ((resource->usage & RhiTextureUsage::ShaderRead) != RhiTextureUsage::None) {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    return fallback;
}

VkImageLayout getTrackedLayout(const FrameContext* frameContext,
                               const RhiTexture* texture,
                               VkImageLayout fallback) {
    if (frameContext && frameContext->imageLayoutTracker) {
        const VkImage image = getVulkanImage(texture);
        if (image != VK_NULL_HANDLE) {
            const VkImageLayout tracked = frameContext->imageLayoutTracker->getLayout(image);
            if (tracked != VK_IMAGE_LAYOUT_UNDEFINED || fallback == VK_IMAGE_LAYOUT_UNDEFINED) {
                return tracked;
            }
        }
    }
    return inferLayout(texture, fallback);
}

StreamlineDlssFrameData::VulkanTextureInfo makeTextureInfo(const RhiTexture* texture,
                                                           VkImageLayout layout) {
    StreamlineDlssFrameData::VulkanTextureInfo info{};
    const VulkanTextureResource* resource = getVulkanTextureResource(texture);
    if (!resource) {
        info.state = static_cast<uint32_t>(layout);
        return info;
    }

    info.state = static_cast<uint32_t>(layout);
    info.width = resource->width;
    info.height = resource->height;
    info.nativeFormat = static_cast<uint32_t>(resource->format);
    info.mipLevels = resource->mipLevels > 0 ? resource->mipLevels : 1;
    info.arrayLayers = 1;
    info.flags = 0;
    info.usage = static_cast<uint32_t>(toTrackedUsage(resource->usage, resource->format));
    return info;
}

} // namespace

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

    if (m_frameContext->imageLayoutTracker && data.colorOutput != VK_NULL_HANDLE) {
        VkCommandBuffer commandBuffer = static_cast<VkCommandBuffer>(encoder.nativeHandle());
        if (commandBuffer != VK_NULL_HANDLE) {
            m_frameContext->imageLayoutTracker->transition(commandBuffer,
                                                           data.colorOutput,
                                                           VK_IMAGE_LAYOUT_GENERAL,
                                                           VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    data.colorInputInfo = makeTextureInfo(colorTex,
                                          getTrackedLayout(m_frameContext, colorTex, VK_IMAGE_LAYOUT_GENERAL));
    data.colorOutputInfo = makeTextureInfo(outputTex,
                                           getTrackedLayout(m_frameContext, outputTex, VK_IMAGE_LAYOUT_GENERAL));
    data.depthInfo = makeTextureInfo(depthTex,
                                     getTrackedLayout(m_frameContext, depthTex, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    data.motionVectorsInfo = makeTextureInfo(motionTex,
                                             getTrackedLayout(m_frameContext, motionTex, VK_IMAGE_LAYOUT_GENERAL));

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
    data.motionVectors3D = false;
    data.depthInverted = (m_frameContext->depthClearValue == 0.0);
    data.reset = m_frameContext->historyReset;
    data.cameraPos[0] = m_frameContext->cameraWorldPos.x;
    data.cameraPos[1] = m_frameContext->cameraWorldPos.y;
    data.cameraPos[2] = m_frameContext->cameraWorldPos.z;
    data.cameraUp[0] = m_frameContext->cameraUp.x;
    data.cameraUp[1] = m_frameContext->cameraUp.y;
    data.cameraUp[2] = m_frameContext->cameraUp.z;
    data.cameraRight[0] = m_frameContext->cameraRight.x;
    data.cameraRight[1] = m_frameContext->cameraRight.y;
    data.cameraRight[2] = m_frameContext->cameraRight.z;
    data.cameraForward[0] = m_frameContext->cameraForward.x;
    data.cameraForward[1] = m_frameContext->cameraForward.y;
    data.cameraForward[2] = m_frameContext->cameraForward.z;
    data.cameraNear = m_frameContext->cameraNearZ;
    data.cameraFar = m_frameContext->cameraFarZ;
    data.cameraFov = m_frameContext->cameraFovY;
    data.cameraAspectRatio =
        static_cast<float>(currentDisplayWidth()) / static_cast<float>(std::max(currentDisplayHeight(), 1));

    data.frameIndex = m_frameContext->frameIndex;
    data.commandBuffer = encoder.nativeHandle(); // VkCommandBuffer

    memcpy(data.cameraViewToClip, &m_frameContext->unjitteredProj, sizeof(float) * 16);
    {
        float4x4 clipToCameraView = m_frameContext->unjitteredProj;
        clipToCameraView.Invert();
        memcpy(data.clipToCameraView, &clipToCameraView, sizeof(float) * 16);

        float4x4 currentVP = m_frameContext->unjitteredProj * m_frameContext->view;
        float4x4 invCurrentVP = currentVP;
        invCurrentVP.Invert();
        float4x4 prevVP = m_frameContext->prevProj * m_frameContext->prevView;
        float4x4 clipToPrevClip = prevVP * invCurrentVP;
        memcpy(data.clipToPrevClip, &clipToPrevClip, sizeof(float) * 16);

        float4x4 prevClipToClip = clipToPrevClip;
        prevClipToClip.Invert();
        memcpy(data.prevClipToClip, &prevClipToClip, sizeof(float) * 16);
    }

    if (!streamlineCtx->evaluate(data)) {
        spdlog::warn("StreamlineDlssPass: evaluate failed");
    }
}

#endif // METALLIC_HAS_STREAMLINE
#endif // _WIN32
