// Metadata-only pass registrations for the standalone Pipeline Editor tool.
// Mirrors the platform-specific renderer pass list, but uses REGISTER_PASS_INFO
// (no factory, no renderer backend dependencies) so the editor can display the
// same Add menu entries without linking the runtime.

#include "pass_registry.h"

// Geometry
REGISTER_PASS_INFO(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"cullResult"}),
    PassTypeInfo::PassType::Compute);

#ifdef _WIN32
REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{"cullResult"}),
    (std::vector<std::string>{"visibility", "depth"}),
    PassTypeInfo::PassType::Render);
#else
REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"visibility", "depth"}),
    PassTypeInfo::PassType::Render);
#endif

REGISTER_PASS_INFO(HZBBuildPass, "HZB Build", "Geometry",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<std::string>{"skyOutput"}),
    (std::vector<std::string>{"forwardColor", "depth"}),
    PassTypeInfo::PassType::Render);

// Lighting
REGISTER_PASS_INFO(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"shadowMap"}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<std::string>{"visibility", "depth", "shadowMap", "skyOutput"}),
    (std::vector<std::string>{"lightingOutput", "motionVectors"}),
    PassTypeInfo::PassType::Compute);

#ifndef _WIN32
REGISTER_PASS_INFO(MeshletVisualizePass, "Meshlet Visualize", "Geometry",
    (std::vector<std::string>{"visibility"}),
    (std::vector<std::string>{"lightingOutput"}),
    PassTypeInfo::PassType::Compute);
#endif

// Environment
REGISTER_PASS_INFO(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}),
    PassTypeInfo::PassType::Render);

// Post-process
REGISTER_PASS_INFO(AutoExposurePass, "Auto Exposure", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"exposureLut"}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(TAAPass, "TAA", "Post-Process",
    (std::vector<std::string>{"lightingOutput", "depth", "motionVectors"}),
    (std::vector<std::string>{"taaOutput"}),
    PassTypeInfo::PassType::Compute);

#ifdef _WIN32
REGISTER_PASS_INFO(StreamlineDlssPass, "DLSS", "Post-Process",
    (std::vector<std::string>{"lightingOutput", "depth", "motionVectors"}),
    (std::vector<std::string>{"dlssOutput"}),
    PassTypeInfo::PassType::Compute);
#endif

REGISTER_PASS_INFO(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput", "exposureLut"}),
    (std::vector<std::string>{"$backbuffer"}),
    PassTypeInfo::PassType::Render);

// Utility
REGISTER_PASS_INFO(OutputPass, "Output", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"$backbuffer"}),
    PassTypeInfo::PassType::Render);

REGISTER_PASS_INFO(BlitPass, "Blit", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"destination"}),
    PassTypeInfo::PassType::Blit);

// UI
REGISTER_PASS_INFO(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"$backbuffer"}),
    PassTypeInfo::PassType::Render);

void registerEditorPassTypes() {}
