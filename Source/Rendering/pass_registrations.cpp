// Pass registrations for the Pipeline Editor
// This file registers all render passes with metadata for the visual editor

#include "pass_registry.h"

// Register all pass types with their metadata
// These are metadata-only registrations since the passes use legacy constructors

REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"visibility", "depth"}),
    PassTypeInfo::Type::Render);

REGISTER_PASS_INFO(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<std::string>{"skyOutput"}),
    (std::vector<std::string>{"forwardColor", "depth"}),
    PassTypeInfo::Type::Render);

REGISTER_PASS_INFO(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"shadowMap"}),
    PassTypeInfo::Type::Compute);

REGISTER_PASS_INFO(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}),
    PassTypeInfo::Type::Render);

REGISTER_PASS_INFO(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<std::string>{"visibility", "depth", "shadowMap", "skyOutput"}),
    (std::vector<std::string>{"lightingOutput"}),
    PassTypeInfo::Type::Compute);

REGISTER_PASS_INFO(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"$backbuffer"}),
    PassTypeInfo::Type::Render);

REGISTER_PASS_INFO(BlitPass, "Blit", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"destination"}),
    PassTypeInfo::Type::Blit);

REGISTER_PASS_INFO(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"$backbuffer"}),
    PassTypeInfo::Type::Render);
