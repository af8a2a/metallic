// Metadata-only pass registrations for the standalone Pipeline Editor tool.
// Mirrors Source/Rendering/pass_registrations.cpp but uses REGISTER_PASS_INFO
// (no factory, no Metal dependencies) so the editor can display all pass types
// in its Add menu without linking the renderer.

#include "pass_registry.h"

// Geometry
REGISTER_PASS_INFO(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"cullResult"}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"visibility", "depth"}),
    PassTypeInfo::PassType::Render);

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
    (std::vector<std::string>{"lightingOutput"}),
    PassTypeInfo::PassType::Compute);

// Environment
REGISTER_PASS_INFO(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}),
    PassTypeInfo::PassType::Render);

// Post-process
REGISTER_PASS_INFO(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
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
