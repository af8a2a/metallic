#ifdef _WIN32
#ifdef METALLIC_HAS_STREAMLINE

#include "streamline_dlss_pass.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

void StreamlineDlssPass::executeDlss(RhiComputeCommandEncoder& encoder) {
    IUpscalerIntegration* upscaler = currentUpscaler();
    if (!upscaler || !m_frameContext) return;

    RhiTexture* colorTex = m_sourceRead.isValid() ? m_frameGraph->getTexture(m_sourceRead) : nullptr;
    RhiTexture* depthTex = m_depthRead.isValid() ? m_frameGraph->getTexture(m_depthRead) : nullptr;
    RhiTexture* motionTex = m_motionRead.isValid() ? m_frameGraph->getTexture(m_motionRead) : nullptr;
    RhiTexture* outputTex = m_frameGraph->getTexture(m_dlssOutput);

    if (!colorTex || !depthTex || !motionTex || !outputTex) {
        spdlog::warn("StreamlineDlssPass: missing input textures");
        return;
    }

    UpscalerEvaluateInputs inputs{};
    inputs.colorInput = colorTex;
    inputs.depth = depthTex;
    inputs.motionVectors = motionTex;
    inputs.colorOutput = outputTex;
    inputs.renderWidth = static_cast<uint32_t>(currentRenderWidth());
    inputs.renderHeight = static_cast<uint32_t>(currentRenderHeight());
    inputs.displayWidth = static_cast<uint32_t>(currentDisplayWidth());
    inputs.displayHeight = static_cast<uint32_t>(currentDisplayHeight());
    inputs.jitterOffsetX = m_frameContext->jitterOffset.x;
    inputs.jitterOffsetY = m_frameContext->jitterOffset.y;
    inputs.mvecScaleX = static_cast<float>(currentRenderWidth());
    inputs.mvecScaleY = static_cast<float>(currentRenderHeight());
    inputs.motionVectorsJittered = false;
    inputs.motionVectors3D = false;
    inputs.depthInverted = (m_frameContext->depthClearValue == 0.0);
    inputs.reset = m_frameContext->historyReset;
    inputs.cameraPos[0] = m_frameContext->cameraWorldPos.x;
    inputs.cameraPos[1] = m_frameContext->cameraWorldPos.y;
    inputs.cameraPos[2] = m_frameContext->cameraWorldPos.z;
    inputs.cameraUp[0] = m_frameContext->cameraUp.x;
    inputs.cameraUp[1] = m_frameContext->cameraUp.y;
    inputs.cameraUp[2] = m_frameContext->cameraUp.z;
    inputs.cameraRight[0] = m_frameContext->cameraRight.x;
    inputs.cameraRight[1] = m_frameContext->cameraRight.y;
    inputs.cameraRight[2] = m_frameContext->cameraRight.z;
    inputs.cameraForward[0] = m_frameContext->cameraForward.x;
    inputs.cameraForward[1] = m_frameContext->cameraForward.y;
    inputs.cameraForward[2] = m_frameContext->cameraForward.z;
    inputs.cameraNear = m_frameContext->cameraNearZ;
    inputs.cameraFar = m_frameContext->cameraFarZ;
    inputs.cameraFov = m_frameContext->cameraFovY;
    inputs.cameraAspectRatio =
        static_cast<float>(currentDisplayWidth()) / static_cast<float>(std::max(currentDisplayHeight(), 1));
    inputs.frameIndex = m_frameContext->frameIndex;

    memcpy(inputs.cameraViewToClip, &m_frameContext->unjitteredProj, sizeof(float) * 16);
    {
        float4x4 clipToCameraView = m_frameContext->unjitteredProj;
        clipToCameraView.Invert();
        memcpy(inputs.clipToCameraView, &clipToCameraView, sizeof(float) * 16);

        float4x4 currentVP = m_frameContext->unjitteredProj * m_frameContext->view;
        float4x4 invCurrentVP = currentVP;
        invCurrentVP.Invert();
        float4x4 prevVP = m_frameContext->prevProj * m_frameContext->prevView;
        float4x4 clipToPrevClip = prevVP * invCurrentVP;
        memcpy(inputs.clipToPrevClip, &clipToPrevClip, sizeof(float) * 16);

        float4x4 prevClipToClip = clipToPrevClip;
        prevClipToClip.Invert();
        memcpy(inputs.prevClipToClip, &prevClipToClip, sizeof(float) * 16);
    }

    if (!upscaler->evaluate(inputs, encoder)) {
        spdlog::warn("StreamlineDlssPass: evaluate failed");
    }
}

#endif // METALLIC_HAS_STREAMLINE
#endif // _WIN32
