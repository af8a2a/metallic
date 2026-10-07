#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <array>

namespace metallic::render {

// Scalar-layout ABI shared with ColorGrading/Parameters.slang; vectors remain float32.
struct ColorGradingParameters {
    std::array<float, 4> saturation{1, 1, 1, 1};
    std::array<float, 4> contrast{1, 1, 1, 1};
    std::array<float, 4> gamma{1, 1, 1, 1};
    std::array<float, 4> gain{1, 1, 1, 1};
    std::array<float, 4> offset{0, 0, 0, 0};
    std::array<float, 4> film{0.88f, 0.55f, 0.26f, 0.0f};
    std::array<float, 4> filmExtra{0.04f, 0.6f, 1.0f, 1.0f};
    std::array<float, 4> lutWeights{};
};
static_assert(sizeof(ColorGradingParameters) == 128);

ColorGradingParameters colorGradingParameters(const RenderGraphProperties& properties);
std::vector<RenderGraphRuntimeSetting> colorGradingSettings();

// Immutable resources are prepared at graph compilation, never in the frame loop.
class ColorGradingResources {
public:
    Result<> initialize(Device& device, const RenderGraphProperties& properties, float peakNits, bool aces2,
                        std::string& log);
    std::array<TextureView*, 7> views() const;

private:
    std::array<std::unique_ptr<Texture>, 7> textures_;
    std::array<std::unique_ptr<TextureView>, 7> views_;
};

} // namespace metallic::render
