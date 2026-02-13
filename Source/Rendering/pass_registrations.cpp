// Pass registrations for the Pipeline Editor
// This file registers all render passes with factories for data-driven pipeline building

#include "pass_registry.h"
#include "render_pass.h"

// Include all pass headers
#include "visibility_pass.h"
#include "shadow_ray_pass.h"
#include "sky_pass.h"
#include "deferred_lighting_pass.h"
#include "tonemap_pass.h"
#include "imgui_overlay_pass.h"
#include "forward_pass.h"
#include "blit_pass.h"

// Register all passes with factories for data-driven pipeline building

// Geometry passes
REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"visibility", "depth"}));

// Lighting passes
REGISTER_COMPUTE_PASS(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"shadowMap"}));

REGISTER_COMPUTE_PASS(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<std::string>{"visibility", "depth", "shadowMap", "skyOutput"}),
    (std::vector<std::string>{"lightingOutput"}));

// Environment passes
REGISTER_RENDER_PASS(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}));

// Post-process passes
REGISTER_RENDER_PASS(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"$backbuffer"}));

// UI passes
REGISTER_RENDER_PASS(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"$backbuffer"}));

// Metadata-only registrations for passes with complex constructors
REGISTER_PASS_INFO(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<std::string>{"skyOutput"}),
    (std::vector<std::string>{"forwardColor", "depth"}),
    PassTypeInfo::PassType::Render);

REGISTER_PASS_INFO(BlitPass, "Blit", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"destination"}),
    PassTypeInfo::PassType::Blit);
