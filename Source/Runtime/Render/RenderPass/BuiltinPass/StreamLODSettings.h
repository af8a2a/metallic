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
    view.rtTessellation.enabled = properties.value("rtTessellation", false);
    view.rtTessellation.edgePixels = properties.value("rtTessellationEdgePixels", 8.0f);
    view.rtTessellation.maxFactor = properties.value("rtTessellationMaxFactor", 4u);
    view.rtTessellation.maxSplitDepth = properties.value("rtTessellationMaxSplitDepth", 2u);
    view.rtTessellation.maxPatches = properties.value("rtTessellationMaxPatches", 16384u);
    view.rtTessellation.buildMode = parseStreamCLASTessellationBuildMode(
        properties.value("rtTessellationBuildMode", std::string("triangles")));
    view.rtTessellation.diagnosticBuildSizes = properties.value("benchmarkCLASBuildSizes", false);
    view.rtTessellation.templateShape = parseStreamCLASTessellationTemplateShape(
        properties.value("rtTessellationTemplateShape", std::string("right")));
    view.rtTessellation.instantiationFlags = parseStreamCLASTessellationInstantiationFlags(
        properties.value("rtTessellationInstantiationFlags", std::string("fastTrace")));
    view.rtTessellation.templateBoundsMode = parseStreamCLASTessellationTemplateBoundsMode(
        properties.value("rtTessellationTemplateBounds", std::string("none")));
}

inline void appendStreamRayLODRuntimeSettings(std::vector<RenderGraphRuntimeSetting>& settings)
{
    settings.push_back({.key = "rtOffscreenErrorScale", .label = "RT Offscreen LOD Error Scale",
        .type = RenderGraphRuntimeSettingType::Float, .defaultValue = 2.0f, .minValue = 1.0f, .maxValue = 16.0f});
    settings.push_back({.key = "rtLodHysteresis", .label = "RT LOD Hysteresis",
        .type = RenderGraphRuntimeSettingType::Float, .defaultValue = 0.2f, .minValue = 0.0f, .maxValue = 0.9f});
    settings.push_back({.key = "rtTessellation", .label = "CLAS Displacement Tessellation",
        .type = RenderGraphRuntimeSettingType::Bool, .defaultValue = false, .rebuildGraph = true});
    settings.push_back({.key = "rtTessellationBuildMode", .label = "CLAS Tessellation Build",
        .type = RenderGraphRuntimeSettingType::Enum, .defaultValue = "triangles",
        .options = {{"Triangle build", "triangles"}, {"Template instantiation", "templates"}}});
    settings.push_back({.key = "rtTessellationTemplateShape", .label = "CLAS Template Shape (advanced)",
        .type = RenderGraphRuntimeSettingType::Enum, .defaultValue = "right",
        .options = {{"Right triangle (default)", "right"}, {"Equilateral (spatial quality hint)", "equilateral"},
            {"Right triangle (dyadic guide)", "rightDyadic"},
            {"Right triangle (longest-edge corner)", "rightCorner"}}});
    settings.push_back({.key = "rtTessellationInstantiationFlags", .label = "CLAS Instantiation Policy (advanced)",
        .type = RenderGraphRuntimeSettingType::Enum, .defaultValue = "fastTrace",
        .options = {{"Prefer tracing speed", "fastTrace"}, {"Driver build/trace policy", "none"}}});
    settings.push_back({.key = "rtTessellationTemplateBounds", .label = "CLAS Template Bounds",
        .type = RenderGraphRuntimeSettingType::Enum, .defaultValue = "none",
        .options = {{"Unbounded", "none"}, {"Geometry and displacement", "geometry"}}});
    settings.push_back({.key = "rtTessellationEdgePixels", .label = "RT Tessellation Edge (render px)",
        .type = RenderGraphRuntimeSettingType::Float, .defaultValue = 8.0f, .minValue = 1.0f, .maxValue = 256.0f});
    settings.push_back({.key = "rtTessellationMaxFactor", .label = "RT Tessellation Leaf Edge Factor",
        .type = RenderGraphRuntimeSettingType::Int, .defaultValue = 4, .minValue = 1, .maxValue = 8});
    settings.push_back({.key = "rtTessellationMaxSplitDepth", .label = "RT Tessellation Split Depth",
        .type = RenderGraphRuntimeSettingType::Int, .defaultValue = 2, .minValue = 0, .maxValue = 3});
    settings.push_back({.key = "rtTessellationMaxPatches", .label = "RT Tessellation Patch Capacity",
        .type = RenderGraphRuntimeSettingType::Int, .defaultValue = 16384, .minValue = 1, .maxValue = 1048576,
        .rebuildGraph = true});
}

} // namespace metallic::render::builtin_pass
