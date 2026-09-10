#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <memory>
#include <string>

namespace metallic::render::vulkan {

struct DlssNrSettings {
    uint32_t preset = 0;
    uint32_t style = 0;
    float intensity = 1.0f;
    float localToneStrength = 1.0f;
    float localStructureStrength = 1.0f;
    float skinStructureStrength = -1.0f;
    // Scale the supplied current-to-previous motion into input pixels.
    float motionVectorScaleX = 1.0f;
    float motionVectorScaleY = 1.0f;
    bool depthInverted = true;
    bool useAutoMask = false;
    bool uiCorrection = false;
    bool upscaling = false;
    bool reset = false;
};

struct DlssNrTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
};

struct DlssNrDesc {
    DlssNrTextureRef inputColor;
    DlssNrTextureRef outputColor;
    DlssNrTextureRef motionVectors;
    DlssNrTextureRef depth;
    DlssNrSettings settings;
};

// Validates the recovered feature-18 contract without loading any runtime.
Result validateDlssNrDesc(const DlssNrDesc& desc, std::string& log);
bool dlssNrSdkAvailable();

// One context per temporal view. Destroy before its Device. The graph must
// serialize this unsafe pass; feature replacement/destruction waits for GPU use.
class DlssNrContext {
public:
    DlssNrContext();
    ~DlssNrContext();
    DlssNrContext(const DlssNrContext&) = delete;
    DlssNrContext& operator=(const DlssNrContext&) = delete;

    Result initialize(Device& device, std::string& log);
    // Resources enter and leave in General. Output is distinct from every input.
    Result evaluate(CommandBuffer& commandBuffer, const DlssNrDesc& desc, std::string& log);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
