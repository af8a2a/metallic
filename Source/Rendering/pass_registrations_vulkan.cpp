#ifdef _WIN32

#include "auto_exposure_pass.h"
#include "deferred_lighting_pass.h"
#include "forward_pass.h"
#include "imgui_overlay_pass.h"
#include "hzb_build_pass.h"
#include "meshlet_cull_pass.h"
#include "output_pass.h"
#include "pass_registry.h"
#include "shadow_ray_pass.h"
#include "sky_pass.h"
#include "streamline_dlss_pass.h"
#include "taa_pass.h"
#include "tonemap_pass.h"
#include "visibility_pass.h"

REGISTER_RENDER_PASS(OutputPass, "Output", "Utility",
    (std::vector<std::string>{"source"}),
    (std::vector<std::string>{"$backbuffer"}));

REGISTER_RENDER_PASS(SkyPass, "Sky Pass", "Environment",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"skyOutput"}));

REGISTER_RENDER_PASS(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<std::string>{"skyOutput"}),
    (std::vector<std::string>{"forwardColor", "depth"}));

REGISTER_COMPUTE_PASS(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<std::string>{}),
    (std::vector<std::string>{"cullResult"}));

REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{"cullResult"}),
    (std::vector<std::string>{"visibility", "depth"}));

REGISTER_COMPUTE_PASS(HZBBuildPass, "HZB Build", "Geometry",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{}));

REGISTER_COMPUTE_PASS(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"shadowMap"}));

REGISTER_COMPUTE_PASS(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<std::string>{"visibility", "depth", "shadowMap", "skyOutput"}),
    (std::vector<std::string>{"lightingOutput", "motionVectors"}));

REGISTER_COMPUTE_PASS(TAAPass, "TAA", "Post-Process",
    (std::vector<std::string>{"lightingOutput", "depth", "motionVectors"}),
    (std::vector<std::string>{"taaOutput"}));

REGISTER_COMPUTE_PASS(StreamlineDlssPass, "DLSS", "Post-Process",
    (std::vector<std::string>{"lightingOutput", "depth", "motionVectors"}),
    (std::vector<std::string>{"dlssOutput"}));

REGISTER_RENDER_PASS(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"$backbuffer"}));

REGISTER_COMPUTE_PASS(AutoExposurePass, "Auto Exposure", "Post-Process",
    (std::vector<std::string>{"lightingOutput"}),
    (std::vector<std::string>{"exposureLut"}));

REGISTER_RENDER_PASS(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"$backbuffer"}));

#endif
