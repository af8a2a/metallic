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
#include "hzb_build_pass.h"
#include "blit_pass.h"
#include "output_pass.h"
#include "meshlet_cull_pass.h"
#include "meshlet_visualize_pass.h"
#include "auto_exposure_pass.h"
#include "taa_pass.h"

// Register all passes with factories for data-driven pipeline building

// Geometry passes
REGISTER_COMPUTE_PASS(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{makeOutputSlot("cullResult", "Cull Result")}));

REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("visibility", "Visibility"),
        makeOutputSlot("depth", "Depth")
    }));

REGISTER_COMPUTE_PASS(HZBBuildPass, "HZB Build", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{}));

// Lighting passes
REGISTER_COMPUTE_PASS(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("shadowMap", "Shadow Map")}));

REGISTER_COMPUTE_PASS(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<PassSlotInfo>{
        makeInputSlot("visibility", "Visibility"),
        makeInputSlot("depth", "Depth"),
        makeInputSlot("shadowMap", "Shadow Map", true),
        makeInputSlot("skyOutput", "Sky", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("lightingOutput", "Lighting"),
        makeOutputSlot("motionVectors", "Motion Vectors")
    }));

REGISTER_COMPUTE_PASS(MeshletVisualizePass, "Meshlet Visualize", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("visibility", "Visibility")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("lightingOutput", "Lighting")}));

// Environment passes
REGISTER_RENDER_PASS(SkyPass, "Sky Pass", "Environment",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{makeOutputSlot("skyOutput", "Sky Output")}));

// Post-process passes
REGISTER_COMPUTE_PASS(AutoExposurePass, "Auto Exposure", "Post-Process",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("exposureLut", "Exposure LUT")}));

REGISTER_COMPUTE_PASS(TAAPass, "TAA", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("depth", "Depth", true),
        makeInputSlot("motionVectors", "Motion Vectors", true)
    }),
    (std::vector<PassSlotInfo>{makeOutputSlot("taaOutput", "TAA Output")}));

REGISTER_RENDER_PASS(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("exposureLut", "Exposure LUT", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("output", "Output", false, {"transient", "imported", "backbuffer"})
    }));

// UI passes
REGISTER_RENDER_PASS(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth", true)}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}));

// Geometry passes (forward)
REGISTER_RENDER_PASS(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("skyOutput", "Sky", true)}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("forwardColor", "Forward Color"),
        makeOutputSlot("depth", "Depth")
    }));

REGISTER_PASS_INFO(BlitPass, "Blit", "Utility",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeTargetSlot("destination", "Destination", false, {"transient", "imported", "backbuffer"})}),
    PassTypeInfo::PassType::Blit);

REGISTER_RENDER_PASS(OutputPass, "Output", "Utility",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}));
