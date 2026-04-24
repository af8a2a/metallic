#ifdef _WIN32

#include "cluster_streaming_service.h"
#include "auto_exposure_pass.h"
#include "cluster_render_pass.h"
#include "cluster_streaming_age_filter_pass.h"
#include "deferred_lighting_pass.h"
#include "forward_pass.h"
#include "imgui_overlay_pass.h"
#include "hzb_build_pass.h"
#include "cluster_streaming_update_pass.h"
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
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}));

REGISTER_RENDER_PASS(SkyPass, "Sky Pass", "Environment",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{makeOutputSlot("skyOutput", "Sky Output")}));

REGISTER_RENDER_PASS(ForwardPass, "Forward Pass", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("skyOutput", "Sky", true)}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("forwardColor", "Forward Color"),
        makeOutputSlot("depth", "Depth")
    }));

REGISTER_COMPUTE_PASS(ClusterStreamingUpdatePass, "Cluster Streaming Update", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("streamingSync", "Streaming Sync", true)
    }));

REGISTER_COMPUTE_PASS(MeshletCullPass, "Meshlet Cull", "Geometry",
    (std::vector<PassSlotInfo>{
        makeInputSlot("streamingSync", "Streaming Sync", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("cullResult", "Cull Result", true),
        makeOutputSlot("visibleMeshlets", "Visible Meshlets", true),
        makeOutputSlot("cullCounter", "Cull Counter", true),
        makeOutputSlot("instanceData", "Instance Data", true),
        makeOutputSlot("visibilityWorklist", "Visibility Worklist", true),
        makeOutputSlot("visibilityWorklistState", "Visibility Worklist State", true),
        makeOutputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
        makeOutputSlot("visibilityInstances", "Visibility Instances", true)
    }));

REGISTER_COMPUTE_PASS(ClusterStreamingAgeFilterPass, "Cluster Streaming Age Filter", "Geometry",
    (std::vector<PassSlotInfo>{
        makeInputSlot("cullResult", "Cull Result", true)
    }),
    (std::vector<PassSlotInfo>{}));

REGISTER_RENDER_PASS(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<PassSlotInfo>{
        makeInputSlot("cullResult", "Cull Result", true),
        makeInputSlot("visibleMeshlets", "Visible Meshlets", true),
        makeInputSlot("cullCounter", "Cull Counter", true),
        makeInputSlot("instanceData", "Instance Data", true),
        makeInputSlot("visibilityWorklist", "Visibility Worklist", true),
        makeInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
        makeInputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
        makeInputSlot("visibilityInstances", "Visibility Instances", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("visibility", "Visibility"),
        makeOutputSlot("depth", "Depth")
    }));

REGISTER_COMPUTE_PASS(HZBBuildPass, "HZB Build", "Geometry",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{}));

REGISTER_RENDER_PASS(ClusterRenderPass, "Cluster Render", "Geometry",
    (std::vector<PassSlotInfo>{}),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("color", "Color"),
        makeOutputSlot("depth", "Depth")
    }));

REGISTER_COMPUTE_PASS(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("shadowMap", "Shadow Map")}));

REGISTER_COMPUTE_PASS(DeferredLightingPass, "Deferred Lighting", "Lighting",
    (std::vector<PassSlotInfo>{
        makeInputSlot("visibility", "Visibility"),
        makeInputSlot("depth", "Depth"),
        makeInputSlot("visibleMeshlets", "Visible Meshlets", true),
        makeInputSlot("cullCounter", "Cull Counter", true),
        makeInputSlot("visibilityWorklist", "Visibility Worklist", true),
        makeInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
        makeInputSlot("shadowMap", "Shadow Map", true),
        makeInputSlot("skyOutput", "Sky", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("lightingOutput", "Lighting"),
        makeOutputSlot("motionVectors", "Motion Vectors")
    }));

REGISTER_COMPUTE_PASS(TAAPass, "TAA", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("depth", "Depth", true),
        makeInputSlot("motionVectors", "Motion Vectors", true)
    }),
    (std::vector<PassSlotInfo>{makeOutputSlot("taaOutput", "TAA Output")}));

REGISTER_COMPUTE_PASS(StreamlineDlssPass, "DLSS", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("depth", "Depth", true),
        makeInputSlot("motionVectors", "Motion Vectors", true)
    }),
    (std::vector<PassSlotInfo>{makeOutputSlot("dlssOutput", "DLSS Output")}));

REGISTER_RENDER_PASS(TonemapPass, "Tonemap", "Post-Process",
    (std::vector<PassSlotInfo>{
        makeInputSlot("source", "Source"),
        makeInputSlot("exposureLut", "Exposure LUT", true)
    }),
    (std::vector<PassSlotInfo>{
        makeOutputSlot("output", "Output", false, {"transient", "imported", "backbuffer"})
    }));

REGISTER_COMPUTE_PASS(AutoExposurePass, "Auto Exposure", "Post-Process",
    (std::vector<PassSlotInfo>{makeInputSlot("source", "Source")}),
    (std::vector<PassSlotInfo>{makeOutputSlot("exposureLut", "Exposure LUT")}));

REGISTER_RENDER_PASS(ImGuiOverlayPass, "ImGui Overlay", "UI",
    (std::vector<PassSlotInfo>{makeInputSlot("depth", "Depth", true)}),
    (std::vector<PassSlotInfo>{makeTargetSlot("target", "Target")}));

#endif
