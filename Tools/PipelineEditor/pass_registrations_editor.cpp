// Metadata-only pass registrations for the standalone Pipeline Editor tool.
// Mirrors the platform-specific renderer pass list, but uses REGISTER_PASS_INFO
// (no factory, no renderer backend dependencies) so the editor can display the
// same Add menu entries without linking the runtime.

#include "pass_registry.h"

// Geometry
REGISTER_PASS_INFO(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{makeOutputSlot("cullResult", "Cull Result")}),
    PassTypeInfo::PassType::Compute);

#ifdef _WIN32
REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("cullResult", "Cull Result")}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("visibility", "Visibility"),
        makeOutputSlot("depth", "Depth")
    }),
    PassTypeInfo::PassType::Render);
#else
REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("visibility", "Visibility"),
        makeOutputSlot("depth", "Depth")
    }),
    PassTypeInfo::PassType::Render);
#endif

REGISTER_PASS_INFO(HZBBuildPass, "HZB Build", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("skyOutput", "Sky", true)}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("forwardColor", "Forward Color"),
        makeOutputSlot("depth", "Depth")
    }),
    PassTypeInfo::PassType::Render);

// Lighting
REGISTER_PASS_INFO(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("shadowMap", "Shadow Map")}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<PassSlotInfo>{
        makeInputSlot("visibility", "Visibility"),
        makeInputSlot("depth", "Depth"),
        makeInputSlot("shadowMap", "Shadow Map", true),
        makeInputSlot("skyOutput", "Sky", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("lightingOutput", "Lighting"),
        makeOutputSlot("motionVectors", "Motion Vectors")
    }),
    PassTypeInfo::PassType::Compute);

#ifndef _WIN32
REGISTER_PASS_INFO(MeshletVisualizePass, "Meshlet Visualize", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("visibility", "Visibility")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("lightingOutput", "Lighting")}),
    PassTypeInfo::PassType::Compute);
#endif

// Environment
REGISTER_PASS_INFO(SkyPass, "Sky Pass", "Environment",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{makeOutputSlot("skyOutput", "Sky Output")}),
    PassTypeInfo::PassType::Render);

// Post-process
REGISTER_PASS_INFO(AutoExposurePass, "Auto Exposure", "Post-Process",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("exposureLut", "Exposure LUT")}),
    PassTypeInfo::PassType::Compute);

REGISTER_PASS_INFO(TAAPass, "TAA", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("depth", "Depth", true),
        makeInputSlot("motionVectors", "Motion Vectors", true)
    }),
    (std::vector<PassSlotInfo>{makeOutputSlot("taaOutput", "TAA Output")}),
    PassTypeInfo::PassType::Compute);

#ifdef _WIN32
REGISTER_PASS_INFO(StreamlineDlssPass, "DLSS", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("depth", "Depth", true),
        makeInputSlot("motionVectors", "Motion Vectors", true)
    }),
    (std::vector<PassSlotInfo>{makeOutputSlot("dlssOutput", "DLSS Output")}),
    PassTypeInfo::PassType::Compute);
#endif

REGISTER_PASS_INFO(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("exposureLut", "Exposure LUT", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("output", "Output", false, {"transient", "imported", "backbuffer"})
    }),
    PassTypeInfo::PassType::Render);

// Utility
REGISTER_PASS_INFO(OutputPass, "Output", "Utility",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}),
    PassTypeInfo::PassType::Render);

REGISTER_PASS_INFO(BlitPass, "Blit", "Utility",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeTargetSlot("destination", "Destination", false, {"transient", "imported", "backbuffer"})}),
    PassTypeInfo::PassType::Blit);

// UI
REGISTER_PASS_INFO(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth", true)}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}),
    PassTypeInfo::PassType::Render);

void registerEditorPassTypes() {}
