#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <memory>
#include <string>

namespace metallic::render::vulkan {

struct DLSSNRSettings {
    uint32_t preset = 0;
    uint32_t style = 0;
    float intensity = 1.0f;
    float localToneStrength = 1.0f;
    float localStructureStrength = 1.0f;
    float skinStructureStrength = -1.0f;
    // Scale the supplied current-to-previous motion into input pixels.
    float motionVectorScaleX = 1.0f;
    float motionVectorScaleY = 1.0f;
    bool useAutoMask = false;
    bool uiCorrection = false;
    bool upscaling = false;
    bool reset = false;
};

struct DLSSNRTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
};

struct DLSSNRDesc {
    DLSSNRTextureRef inputColor;
    DLSSNRTextureRef outputColor;
    DLSSNRTextureRef motionVectors;
    DLSSNRTextureRef depth;
    DLSSNRSettings settings;
};

// Validates the recovered feature-18 contract without loading any runtime.
Result<> validateDlssNrDesc(const DLSSNRDesc& desc, std::string& log);
bool dlssNrSdkAvailable();

// One context per temporal view. Destroy before its Device. The graph must
// serialize this unsafe pass; feature replacement/destruction waits for GPU use.
class DLSSNRContext {
public:
    DLSSNRContext();
    ~DLSSNRContext();
    DLSSNRContext(const DLSSNRContext&) = delete;
    DLSSNRContext& operator=(const DLSSNRContext&) = delete;

    Result<> initialize(Device& device, std::string& log);
    // Resources enter and leave in General. Output is distinct from every input.
    Result<> evaluate(CommandBuffer& commandBuffer, const DLSSNRDesc& desc, std::string& log);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
