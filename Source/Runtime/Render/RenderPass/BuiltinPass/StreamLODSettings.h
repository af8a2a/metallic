#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

namespace metallic::render::builtin_pass {

inline void applyStreamRayLODSettings(const RenderGraphProperties& properties,
    MeshletStreamFrameDesc& view, bool preserveRasterLod = false)
{
    view.rtOffscreenErrorScale = properties.value("rtOffscreenErrorScale", 2.0f);
    view.rtLodHysteresis = properties.value("rtLodHysteresis", 0.2f);
    view.preserveRasterLod = preserveRasterLod;
    view.terminalMaskDiagnostics = properties.value("benchmarkTerminalMaskStats", false);
    view.benchmarkDemandTraversal = properties.value("benchmarkDemandTraversal", false);
}

inline void appendStreamRayLODRuntimeSettings(std::vector<RenderGraphRuntimeSetting>& settings)
{
    settings.push_back({.key = "rtOffscreenErrorScale", .label = "RT Offscreen LOD Error Scale",
        .type = RenderGraphRuntimeSettingType::Float, .defaultValue = 2.0f, .minValue = 1.0f, .maxValue = 16.0f});
    settings.push_back({.key = "rtLodHysteresis", .label = "RT LOD Hysteresis",
        .type = RenderGraphRuntimeSettingType::Float, .defaultValue = 0.2f, .minValue = 0.0f, .maxValue = 0.9f});
}

} // namespace metallic::render::builtin_pass
