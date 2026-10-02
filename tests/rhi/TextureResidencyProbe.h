#pragma once
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/TextureResidencyProbeParameters.h"

namespace metallic::tests {
inline render::Result<> dispatchTextureResidencyProbe(render::Device& device, render::ComputeKernel& kernel,
    render::CommandBuffer& command, render::Buffer& feedback, uint32_t slot, uint32_t mip, uint32_t samples)
{
    auto registry = device.resourceRegistry();
    if (!registry) { return render::makeError(registry.error()); }
    render::ParameterWriter writer(device, **registry, command.frameContext());
    const render::TextureResidencyProbeParameters parameters{
        writer.dataBuffer(&feedback, 4, 4), slot, mip, samples, 0};
    auto encoded = writer.encode(parameters, render::kTextureResidencyProbeABI, render::ParameterTransport::InlinePush);
    if (!encoded) { return render::makeError(encoded.error()); }
    return kernel.dispatch(command, *encoded, 1);
}
} // namespace metallic::tests
