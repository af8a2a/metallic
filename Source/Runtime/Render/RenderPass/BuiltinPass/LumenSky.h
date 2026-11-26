#pragma once

#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <numbers>

namespace metallic::render::builtin_pass {

struct LumenSkyParameters {
    ShaderSampledImage radiance;
    ShaderSampler sampler;
    float rotationRadians;
    float intensity;
};
static_assert(sizeof(LumenSkyParameters) == 16);

inline Result<> declareLumenSky(const RenderGraphResourceRequestContext& context,
    RenderGraphResourceRequestBuilder& builder, std::string& log)
{
    std::array<double, 3> observer{};
    if (const auto* view = context.viewConstants()) {
        observer = {view->current.eye[0], view->current.eye[1], view->current.eye[2]};
    }
    const auto result = context.subsystem<EnvironmentLightingSubsystem>()->declareEnvironmentRequest(context, builder,
        {.environment = resolveWorldEnvironment(context.runtimeScene(), context.world()), .observerWorldMetres = observer,
            .viewId = context.viewKey(), .needCelestial = false}, log);
    return result ? Result<>{} : makeError(result.error());
}

inline LumenSkyParameters bindLumenSky(const RenderGraphExecutionContext& context, ParameterWriter& writer)
{
    const auto& lighting = context.resources<EnvironmentResourceSet>()->lighting;
    LumenSkyParameters sky{};
    if (lighting.settings.enabled) {
        sky.intensity = lighting.settings.intensity;
        sky.rotationRadians = lighting.settings.rotationDegrees * (std::numbers::pi_v<float> / 180.0f);
        if (lighting.mapAvailable) {
            sky.radiance = writer.sampledImageHandle(lighting.radianceView);
            sky.sampler = writer.samplerHandle({.addressU = SamplerAddressMode::Repeat});
        }
    }
    return sky;
}

} // namespace metallic::render::builtin_pass
