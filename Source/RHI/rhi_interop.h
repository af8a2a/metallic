#pragma once

#include <cstdint>
#include <string>

class RhiTexture;
class RhiComputeCommandEncoder;

class IRhiInteropService {
public:
    virtual ~IRhiInteropService() = default;
};

enum class RhiInteropServiceType : uint32_t {
    FrameGraphExecution = 0,
    UpscalerIntegration = 1,
};

class IRhiInteropProvider {
public:
    virtual ~IRhiInteropProvider() = default;
    virtual IRhiInteropService* getService(RhiInteropServiceType type) const = 0;
};

struct UpscalerEvaluateInputs {
    const RhiTexture* colorInput = nullptr;
    const RhiTexture* depth = nullptr;
    const RhiTexture* motionVectors = nullptr;
    RhiTexture* colorOutput = nullptr;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth = 0;
    uint32_t displayHeight = 0;

    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;
    float mvecScaleX = 1.0f;
    float mvecScaleY = 1.0f;
    bool motionVectorsJittered = false;
    bool motionVectors3D = false;
    bool depthInverted = false;
    bool reset = false;

    float cameraViewToClip[16] = {};
    float clipToCameraView[16] = {};
    float clipToPrevClip[16] = {};
    float prevClipToClip[16] = {};

    float cameraPos[3] = {};
    float cameraUp[3] = {};
    float cameraRight[3] = {};
    float cameraForward[3] = {};
    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    float cameraFov = 0.0f;
    float cameraAspectRatio = 1.0f;

    uint32_t frameIndex = 0;
};

class IFrameGraphExecutionService : public IRhiInteropService {
public:
    virtual ~IFrameGraphExecutionService() = default;
};

class IUpscalerIntegration : public IRhiInteropService {
public:
    virtual ~IUpscalerIntegration() = default;

    virtual bool isAvailable() const = 0;
    virtual bool isEnabled() const = 0;
    virtual std::string statusString() const = 0;
    virtual bool getOptimalRenderSize(uint32_t displayWidth,
                                      uint32_t displayHeight,
                                      uint32_t& outRenderWidth,
                                      uint32_t& outRenderHeight) const = 0;
    virtual bool evaluate(const UpscalerEvaluateInputs& inputs,
                          RhiComputeCommandEncoder& encoder) = 0;
    virtual void resetHistory() = 0;
};
