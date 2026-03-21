#include <ml.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "camera.h"
#include "input.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "scene_graph_ui.h"
#include "scene_context.h"

#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>
#include <cmath>
#include <type_traits>

#include <tracy/Tracy.hpp>
#include "frame_graph.h"
#include "visibility_constants.h"
#include "render_uniforms.h"
#include "render_pass.h"
#include "shader_manager.h"
#include "blit_pass.h"
#include "tonemap_pass.h"
#include "imgui_overlay_pass.h"
#include "visibility_pass.h"
#include "shadow_ray_pass.h"
#include "deferred_lighting_pass.h"
#include "forward_pass.h"
#include "sky_pass.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "frame_context.h"
#include "rhi_window_runtime.h"
#include "rhi_resource_utils.h"
#include "cluster_lod_builder.h"
#include <json.hpp>

namespace {

// --- AppState for input + resize ---

struct AppState {
    InputState input;
    bool framebufferResized = false;
};

static_assert(std::is_standard_layout_v<AppState>);
static_assert(offsetof(AppState, input) == 0);

static constexpr int kRenderResolutionBaseWidth = 1920;
static constexpr int kRenderResolutionBaseHeight = 1080;
static constexpr float kMinRenderScale = 0.25f;
static constexpr float kMaxRenderScale = 2.0f;

void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
    if (state && width > 0 && height > 0) {
        state->framebufferResized = true;
    }
}

// --- Pipeline asset manipulation helpers ---

bool loadPipelineAssetChecked(const std::string& path,
                              const char* label,
                              PipelineAsset& outAsset) {
    PipelineAsset loaded = PipelineAsset::load(path);
    if (loaded.name.empty()) {
        spdlog::error("Failed to load {} pipeline from '{}'", label, path);
        return false;
    }

    std::string validationError;
    if (!loaded.validate(validationError)) {
        spdlog::error("Invalid {} pipeline '{}': {}", label, path, validationError);
        return false;
    }

    outAsset = std::move(loaded);
    spdlog::info("Loaded {} pipeline: {} ({} passes, {} resources)",
                 label, outAsset.name, outAsset.passes.size(), outAsset.resources.size());
    return true;
}

void eraseResourceByName(PipelineAsset& asset, const char* resourceName) {
    asset.resources.erase(
        std::remove_if(asset.resources.begin(), asset.resources.end(),
                       [resourceName](const ResourceDecl& r) { return r.name == resourceName; }),
        asset.resources.end());
}

void erasePassByType(PipelineAsset& asset, const char* passType) {
    asset.passes.erase(
        std::remove_if(asset.passes.begin(), asset.passes.end(),
                       [passType](const PassDecl& p) { return p.type == passType; }),
        asset.passes.end());
}

void replacePassInput(PassDecl& pass, const char* oldName, const char* newName) {
    for (auto& input : pass.inputs) {
        if (input == oldName) input = newName;
    }
}

bool passProducesResource(const PassDecl& pass, const char* resourceName) {
    return std::find(pass.outputs.begin(), pass.outputs.end(), resourceName) != pass.outputs.end();
}

PassDecl* findFirstEnabledPassByType(PipelineAsset& asset, const char* passType) {
    for (auto& pass : asset.passes) {
        if (pass.enabled && pass.type == passType) return &pass;
    }
    return nullptr;
}

void normalizePipelineFinalDisplayOutput(PipelineAsset& asset) {
    PassDecl* tonemapPass = findFirstEnabledPassByType(asset, "TonemapPass");
    if (!tonemapPass) return;

    bool tonemapHasOutput = false;
    for (auto& output : tonemapPass->outputs) {
        if (output == "$backbuffer") output = "tonemapOutput";
        tonemapHasOutput = tonemapHasOutput || output == "tonemapOutput";
    }
    if (!tonemapHasOutput) tonemapPass->outputs.push_back("tonemapOutput");

    PassDecl finalOutputPass;
    bool hasExistingOutputPass = false;
    size_t existingOutputPassIndex = 0;
    for (size_t i = 0; i < asset.passes.size(); ++i) {
        if (asset.passes[i].enabled && asset.passes[i].type == "OutputPass") {
            finalOutputPass = asset.passes[i];
            existingOutputPassIndex = i;
            hasExistingOutputPass = true;
            break;
        }
    }

    if (hasExistingOutputPass) {
        asset.passes.erase(asset.passes.begin() + static_cast<std::ptrdiff_t>(existingOutputPassIndex));
    } else {
        finalOutputPass.name = "OutputPass_Final";
        finalOutputPass.type = "OutputPass";
        finalOutputPass.enabled = true;
        finalOutputPass.sideEffect = false;
        finalOutputPass.config = nlohmann::json::object();
    }

    finalOutputPass.inputs = {"tonemapOutput"};
    finalOutputPass.outputs = {"$backbuffer"};

    size_t insertPos = asset.passes.size();
    for (size_t i = 0; i < asset.passes.size(); ++i) {
        if (asset.passes[i].type == "ImGuiOverlayPass") { insertPos = i; break; }
    }
    asset.passes.insert(asset.passes.begin() + static_cast<std::ptrdiff_t>(insertPos), finalOutputPass);
}

void disablePipelineAutoExposure(PipelineAsset& asset) {
    erasePassByType(asset, "HistogramPass");
    erasePassByType(asset, "AutoExposurePass");
    eraseResourceByName(asset, "exposureLut");
    eraseResourceByName(asset, "histogram");
    for (auto& pass : asset.passes) {
        auto it = std::remove(pass.inputs.begin(), pass.inputs.end(), std::string("exposureLut"));
        pass.inputs.erase(it, pass.inputs.end());
    }
}

void disableVisibilityRayTracing(PipelineAsset& asset) {
    erasePassByType(asset, "ShadowRayPass");
    eraseResourceByName(asset, "shadowMap");
    for (auto& pass : asset.passes) {
        auto it = std::remove(pass.inputs.begin(), pass.inputs.end(), std::string("shadowMap"));
        pass.inputs.erase(it, pass.inputs.end());
    }
}

void disableVisibilityTAA(PipelineAsset& asset) {
    erasePassByType(asset, "TAAPass");
    eraseResourceByName(asset, "taaOutput");
    for (auto& pass : asset.passes) {
        replacePassInput(pass, "taaOutput", "lightingOutput");
    }
}

PipelineAsset makeSceneColorPostPipelineAsset() {
    PipelineAsset asset;
    asset.name = "MetalTriangleFallback";
    asset.passes.push_back({"Tonemap", "TonemapPass", {"sceneColor"}, {"tonemapOutput"}, true, false, {}});
    asset.passes.push_back({"Output", "OutputPass", {"tonemapOutput"}, {"$backbuffer"}, true, false, {}});
    asset.passes.push_back({"ImGui Overlay", "ImGuiOverlayPass", {}, {"$backbuffer"}, true, false, {}});
    return asset;
}

float4 orbitCameraWorldPosition(const OrbitCamera& camera) {
    const float cosAz = std::cos(camera.azimuth);
    const float sinAz = std::sin(camera.azimuth);
    const float cosEl = std::cos(camera.elevation);
    const float sinEl = std::sin(camera.elevation);
    return float4(camera.target.x + camera.distance * cosEl * sinAz,
                  camera.target.y + camera.distance * sinEl,
                  camera.target.z + camera.distance * cosEl * cosAz,
                  1.0f);
}

ImGuiID beginDockspace(bool& showSceneGraphWindow, bool& showGraphDebug,
                       bool& showRenderPassUI, bool& showImGuiDemo) {
    const ImGuiID dockspaceId =
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene Graph", nullptr, &showSceneGraphWindow);
            ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
            ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
            ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    return dockspaceId;
}

} // namespace

int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Metal Sponza", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    std::string runtimeError;
    auto runtime = createRhiWindowRuntime(RhiBackendType::Metal, window, runtimeError);
    if (!runtime) {
        spdlog::error("{}", runtimeError);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    spdlog::info("RHI device: {}", runtime->deviceName());
    RhiDeviceHandle device(runtime->device().nativeHandle());
    RhiCommandQueueHandle commandQueue(runtime->commandQueue().nativeHandle());

    const char* projectRoot = PROJECT_SOURCE_DIR;

    // --- Input + resize ---
    OrbitCamera camera;
    AppState appState{};
    appState.input.camera = &camera;
    glfwSetWindowUserPointer(window, &appState);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    setupInputCallbacks(window, &appState.input);

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    // --- Scene loading ---
    SceneContext scene(device, commandQueue, projectRoot);
    if (!scene.loadAll("Asset/Sponza/glTF/Sponza.gltf")) return 1;

    // Camera init + tuning
    camera.initFromBounds(scene.mesh().bboxMin, scene.mesh().bboxMax);
    camera.distance *= 0.8f;
    camera.azimuth = 0.55f;
    camera.elevation = 0.35f;

    // --- Cluster LOD ---
    ClusterLODData clusterLOD{};
    if (!buildClusterLOD(device, scene.mesh(), scene.meshlets(), clusterLOD)) {
        spdlog::warn("Cluster LOD build failed; continuing without LOD hierarchy");
    }

    // --- ImGui init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui_ImplGlfw_InitForOther(window, true);
    runtime->initImGui();

    // --- Shader compilation ---
    ShaderManager shaderManager(device, projectRoot);
    if (!shaderManager.buildAll()) return 1;

    PipelineRuntimeContext& runtimeContext = shaderManager.runtimeContext();

    if (scene.atmosphereLoaded()) {
        auto& atm = scene.atmosphereTextures();
        shaderManager.importTexture("transmittance", atm.transmittance);
        shaderManager.importTexture("scattering", atm.scattering);
        shaderManager.importTexture("irradiance", atm.irradiance);
        shaderManager.importSampler("atmosphere", atm.sampler);
    }

    // --- Pipeline availability checks ---
    auto hasComputePipeline = [&](const char* name) {
        return runtimeContext.computePipelinesRhi.count(name) > 0;
    };
    auto hasRenderPipeline = [&](const char* name) {
        return runtimeContext.renderPipelinesRhi.count(name) > 0;
    };

    bool visibilityAutoExposureAvailable = false;
    bool visibilityTaaAvailable = false;
    bool visibilityGpuCullingAvailable = false;
    bool useVisibilityRenderGraph = false;
    bool atmosphereSkyAvailable = false;
    bool rtShadowsAvailable = scene.rtShadowsAvailable();
    bool enableRTShadows = true;
    float visibilityRenderScale = 1.0f;
    bool visibilityHistoryResetRequested = false;

    // --- Load pipeline asset ---
    const std::string visibilityPipelinePath =
        std::string(projectRoot) + "/Pipelines/visibilitybuffer.json";
    PipelineAsset visibilityPipelineBaseAsset;
    PipelineAsset visibilityPipelineAsset;
    bool visibilityPipelineBaseLoaded =
        loadPipelineAssetChecked(visibilityPipelinePath, "Metal visibility", visibilityPipelineBaseAsset);
    bool visibilityPipelineAssetLoaded = false;

    auto refreshVisibilityPipelineState = [&]() {
        visibilityPipelineAssetLoaded = visibilityPipelineBaseLoaded;
        visibilityPipelineAsset =
            visibilityPipelineBaseLoaded ? visibilityPipelineBaseAsset : PipelineAsset{};
        visibilityAutoExposureAvailable =
            hasComputePipeline("HistogramPass") && hasComputePipeline("AutoExposurePass");
        visibilityTaaAvailable = hasComputePipeline("TAAPass");
        visibilityGpuCullingAvailable =
            hasComputePipeline("MeshletCullPass") &&
            hasComputePipeline("BuildIndirectPass") &&
            hasRenderPipeline("VisibilityIndirectPass");
        atmosphereSkyAvailable = scene.atmosphereLoaded() && shaderManager.hasSkyPipeline();

        auto validateVisibilityAsset = [&](const char* label) {
            std::string validationError;
            if (!visibilityPipelineAsset.validate(validationError)) {
                spdlog::error("{}: {}", label, validationError);
                visibilityPipelineAssetLoaded = false;
                return false;
            }
            return true;
        };

        if (visibilityPipelineAssetLoaded) {
            normalizePipelineFinalDisplayOutput(visibilityPipelineAsset);
        }
        if (visibilityPipelineAssetLoaded && !validateVisibilityAsset("Invalid Metal visibility pipeline")) {
            return;
        }
        if (visibilityPipelineAssetLoaded && !visibilityAutoExposureAvailable) {
            disablePipelineAutoExposure(visibilityPipelineAsset);
            if (!validateVisibilityAsset("Invalid pipeline after disabling auto-exposure")) return;
        }
        if (visibilityPipelineAssetLoaded && !rtShadowsAvailable) {
            disableVisibilityRayTracing(visibilityPipelineAsset);
            if (!validateVisibilityAsset("Invalid pipeline after disabling ray tracing")) return;
        }
        if (visibilityPipelineAssetLoaded && !visibilityTaaAvailable) {
            disableVisibilityTAA(visibilityPipelineAsset);
            if (!validateVisibilityAsset("Invalid pipeline after disabling TAA")) return;
        }

        useVisibilityRenderGraph = visibilityPipelineAssetLoaded;
    };

    refreshVisibilityPipelineState();

    auto logVisibilityMode = [&]() {
        if (useVisibilityRenderGraph) {
            spdlog::info("Metal RenderGraph mode: Visibility Buffer pipeline");
            spdlog::info("Metal visibility dispatch: {}",
                         visibilityGpuCullingAvailable ? "GPU-driven indirect" : "CPU");
            spdlog::info("Metal render resolution: {} x {} (base {} x {}, scale {:.2f}) -> {} x {}",
                         runtimeContext.renderWidth, runtimeContext.renderHeight,
                         kRenderResolutionBaseWidth, kRenderResolutionBaseHeight,
                         visibilityRenderScale,
                         runtimeContext.displayWidth, runtimeContext.displayHeight);
        } else {
            spdlog::info("Metal RenderGraph mode: Triangle fallback");
        }
    };

    auto syncVisibilityUpscalerState = [&](int displayWidth, int displayHeight) {
        visibilityRenderScale = std::clamp(visibilityRenderScale, kMinRenderScale, kMaxRenderScale);
        int renderW = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseWidth) * visibilityRenderScale)));
        int renderH = std::max(1, static_cast<int>(std::lround(
            static_cast<float>(kRenderResolutionBaseHeight) * visibilityRenderScale)));
        runtimeContext.displayWidth = displayWidth;
        runtimeContext.displayHeight = displayHeight;
        runtimeContext.renderWidth = renderW;
        runtimeContext.renderHeight = renderH;
        logVisibilityMode();
    };

    // --- Persistent render context and pipeline builder ---
    RenderContext ctx = scene.renderContext();
    PipelineBuilder pipelineBuilder(ctx);
    auto frameGraphBackend = runtime->createFrameGraphBackend();
    runtimeContext.resourceFactory = frameGraphBackend.get();

    bool postBuilderNeedsRebuild = true;
    bool showSceneGraphWindow = true;
    bool showGraphDebug = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;
    bool shaderReloadRequested = false;
    bool pipelineReloadRequested = false;
    double lastFrameTime = glfwGetTime();
    float4x4 prevView = float4x4::Identity();
    float4x4 prevProj = float4x4::Identity();
    float4x4 prevCullView = float4x4::Identity();
    float4x4 prevCullProj = float4x4::Identity();
    float4 prevCameraWorldPos = float4(0.0f, 0.0f, 0.0f, 1.0f);
    bool hasPrevMatrices = false;
    uint32_t frameIndex = 0;

    // Ring buffer to keep instance transform buffers alive for 2 frames
    static constexpr uint32_t kBufferRingSize = 2;
    std::unique_ptr<RhiBuffer> instanceTransformRing[kBufferRingSize];

    DirectionalLight sunLight = scene.sceneGraph().getSunDirectionalLight();
    std::vector<uint32_t> visibleMeshletNodes;
    std::vector<uint32_t> visibleIndexNodes;
    visibleMeshletNodes.reserve(scene.sceneGraph().nodes.size());
    visibleIndexNodes.reserve(scene.sceneGraph().nodes.size());

    auto refreshSceneState = [&]() {
        scene.sceneGraph().updateTransforms();
        sunLight = scene.sceneGraph().getSunDirectionalLight();
        visibleMeshletNodes.clear();
        visibleIndexNodes.clear();
        for (const auto& node : scene.sceneGraph().nodes) {
            if (!scene.sceneGraph().isNodeVisible(node.id)) continue;
            if (node.meshletCount > 0) visibleMeshletNodes.push_back(node.id);
            if (node.indexCount > 0) visibleIndexNodes.push_back(node.id);
        }
    };
    refreshSceneState();

    PipelineAsset fallbackAsset = makeSceneColorPostPipelineAsset();

    auto rebuildActivePipeline = [&](int targetWidth, int targetHeight) {
        syncVisibilityUpscalerState(targetWidth, targetHeight);
        const int buildWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : targetWidth;
        const int buildHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : targetHeight;
        PipelineAsset& activeAsset = useVisibilityRenderGraph ? visibilityPipelineAsset : fallbackAsset;
        if (!pipelineBuilder.build(activeAsset, runtimeContext, buildWidth, buildHeight)) {
            spdlog::error("Failed to build Metal pipeline");
            return false;
        }
        return true;
    };

    if (!rebuildActivePipeline(width, height)) {
        runtime->shutdownImGui();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // ==================== Main render loop ====================
    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("Frame");
        glfwPollEvents();

        // F5 shader reload
        const bool f5Down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5Down && !reloadKeyDown) shaderReloadRequested = true;
        reloadKeyDown = f5Down;

        // F6 pipeline reload
        const bool f6Down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6Down && !pipelineReloadKeyDown) pipelineReloadRequested = true;
        pipelineReloadKeyDown = f6Down;

        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            glfwWaitEvents();
            continue;
        }

        // Handle resize
        if (appState.framebufferResized) {
            postBuilderNeedsRebuild = true;
            appState.framebufferResized = false;
            visibilityHistoryResetRequested = true;
        }

        // Handle shader reload
        if (shaderReloadRequested) {
            shaderReloadRequested = false;
            spdlog::info("Reloading Metal shaders...");
            const bool prevUseVis = useVisibilityRenderGraph;
            const bool prevAutoExp = visibilityAutoExposureAvailable;
            const bool prevTaa = visibilityTaaAvailable;
            const bool prevAtmo = atmosphereSkyAvailable;
            auto [reloaded, failed] = shaderManager.reloadAll();
            refreshVisibilityPipelineState();

            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(device, scene.shadowResources(), projectRoot)) {
                    reloaded++;
                    spdlog::info("Reloaded Metal RT shadow shader");
                } else {
                    failed++;
                    spdlog::warn("Failed to reload Metal RT shadow shader; keeping previous pipeline");
                }
            }

            if (failed == 0) {
                spdlog::info("All {} Metal shaders reloaded successfully", reloaded);
            } else {
                spdlog::warn("{} Metal shaders reloaded, {} failed", reloaded, failed);
            }

            if (prevUseVis != useVisibilityRenderGraph || prevAutoExp != visibilityAutoExposureAvailable ||
                prevTaa != visibilityTaaAvailable || prevAtmo != atmosphereSkyAvailable) {
                logVisibilityMode();
            }
            postBuilderNeedsRebuild = true;
            visibilityHistoryResetRequested = true;
        }

        // Handle pipeline reload
        if (pipelineReloadRequested) {
            pipelineReloadRequested = false;
            spdlog::info("Reloading Metal pipeline asset...");
            PipelineAsset reloadedAsset;
            if (loadPipelineAssetChecked(visibilityPipelinePath, "Metal visibility", reloadedAsset)) {
                const bool prevUseVis = useVisibilityRenderGraph;
                visibilityPipelineBaseAsset = std::move(reloadedAsset);
                visibilityPipelineBaseLoaded = true;
                refreshVisibilityPipelineState();
                if (prevUseVis != useVisibilityRenderGraph) logVisibilityMode();
                postBuilderNeedsRebuild = true;
                visibilityHistoryResetRequested = true;
                hasPrevMatrices = false;
            }
        }

        // Rebuild pipeline if needed
        if (postBuilderNeedsRebuild) {
            if (!rebuildActivePipeline(width, height)) {
                spdlog::error("Failed to rebuild Metal pipeline; skipping frame");
                postBuilderNeedsRebuild = false;
                continue;
            }
            postBuilderNeedsRebuild = false;
        }

        if (!runtime->beginFrame(static_cast<uint32_t>(width), static_cast<uint32_t>(height))) {
            continue;
        }

        runtime->collectGpuTimestamps();

        // --- ImGui frame ---
        runtime->beginImGuiFrame(nullptr);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const ImGuiID dockspaceId =
            beginDockspace(showSceneGraphWindow, showGraphDebug, showRenderPassUI, showImGuiDemo);

        // --- Renderer UI panel ---
        ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
        ImGui::Begin("Metal Sponza");
        const bool autoExposureEnabled =
            useVisibilityRenderGraph &&
            findFirstEnabledPassByType(visibilityPipelineAsset, "AutoExposurePass") != nullptr;
        ImGui::Text("Resolution: %d x %d", width, height);
        ImGui::TextUnformatted(useVisibilityRenderGraph ? "Pipeline: Visibility Buffer" : "Pipeline: Triangle Fallback");
        ImGui::TextUnformatted(autoExposureEnabled ? "Exposure: Auto" : "Exposure: Manual");
        ImGui::TextUnformatted(visibilityGpuCullingAvailable ? "Visibility Dispatch: GPU" : "Visibility Dispatch: CPU");
        if (useVisibilityRenderGraph) {
            ImGui::Text("Render Base: %d x %d", kRenderResolutionBaseWidth, kRenderResolutionBaseHeight);
            float requestedRenderScale = visibilityRenderScale;
            if (ImGui::SliderFloat("Render Scale",
                                   &requestedRenderScale,
                                   kMinRenderScale,
                                   kMaxRenderScale,
                                   "%.2f")) {
                requestedRenderScale = std::clamp(requestedRenderScale, kMinRenderScale, kMaxRenderScale);
                if (std::abs(requestedRenderScale - visibilityRenderScale) > 0.0001f) {
                    visibilityRenderScale = requestedRenderScale;
                    postBuilderNeedsRebuild = true;
                    visibilityHistoryResetRequested = true;
                    hasPrevMatrices = false;
                }
            }
            ImGui::Text("Render Resolution: %d x %d", runtimeContext.renderWidth, runtimeContext.renderHeight);
        }
        if (rtShadowsAvailable && useVisibilityRenderGraph) {
            ImGui::Checkbox("RT Shadows", &enableRTShadows);
        }

        ImGui::Separator();
        if (ImGui::Button("Reload Shaders (F5)")) shaderReloadRequested = true;
        ImGui::SameLine();
        if (ImGui::Button("Reload Pipeline (F6)")) pipelineReloadRequested = true;
        ImGui::Checkbox("FrameGraph Debug", &showGraphDebug);
        ImGui::Checkbox("Render Pass UI", &showRenderPassUI);
        ImGui::Checkbox("Scene Graph", &showSceneGraphWindow);
        ImGui::Checkbox("ImGui Demo", &showImGuiDemo);

        // LOD stats panel
        if (clusterLOD.lodLevelCount > 0) {
            drawClusterLODStats(clusterLOD);
        }

        ImGui::End();

        // Scene graph UI
        if (showSceneGraphWindow) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            drawSceneGraphUI(scene.sceneGraph());
        }

        refreshSceneState();

        // --- Matrix computation ---
        const int renderWidth = useVisibilityRenderGraph ? runtimeContext.renderWidth : width;
        const int renderHeight = useVisibilityRenderGraph ? runtimeContext.renderHeight : height;
        const float aspect = static_cast<float>(width) / static_cast<float>(std::max(height, 1));
        const float3 sunDirection = normalize(sunLight.direction);
        const float4x4 view = camera.viewMatrix();
        const float4x4 unjitteredProj = camera.projectionMatrix(aspect);
        const float4 cameraWorldPos = orbitCameraWorldPosition(camera);
        const float3 cameraWorldPos3(cameraWorldPos.x, cameraWorldPos.y, cameraWorldPos.z);
        const float3 cameraForward = normalize(camera.target - cameraWorldPos3);
        const float3 worldUp(0.0f, 1.0f, 0.0f);
        const float3 cameraRight = normalize(cross(cameraForward, worldUp));
        const float3 cameraUp = cross(cameraRight, cameraForward);
        float4x4 proj = unjitteredProj;
        float2 jitterOffset = float2(0.0f, 0.0f);

        const bool enableVisibilityTAA = useVisibilityRenderGraph && visibilityTaaAvailable;
        if (enableVisibilityTAA) {
            jitterOffset = OrbitCamera::haltonJitter(frameIndex);
            proj = OrbitCamera::jitteredProjectionMatrix(
                camera.fovY, aspect, camera.nearZ, camera.farZ,
                jitterOffset.x, jitterOffset.y,
                static_cast<uint32_t>(renderWidth), static_cast<uint32_t>(renderHeight));
        }

        // --- Instance transform buffer (ring buffer) ---
        auto& instanceTransformBuffer = instanceTransformRing[frameIndex % kBufferRingSize];
        instanceTransformBuffer.reset();
        uint32_t visibilityInstanceCount = 0;
        if (useVisibilityRenderGraph && !visibleMeshletNodes.empty()) {
            visibilityInstanceCount = static_cast<uint32_t>(
                std::min<size_t>(visibleMeshletNodes.size(),
                                 static_cast<size_t>(kVisibilityInstanceMask + 1u)));
            std::vector<SceneInstanceTransform> instanceTransforms;
            instanceTransforms.reserve(visibilityInstanceCount);
            const float4x4 prevViewForMotion = hasPrevMatrices ? prevView : view;
            const float4x4 prevProjForMotion = hasPrevMatrices ? prevProj : unjitteredProj;
            for (uint32_t instanceId = 0; instanceId < visibilityInstanceCount; ++instanceId) {
                const SceneNode& node = scene.sceneGraph().nodes[visibleMeshletNodes[instanceId]];
                const float4x4 nodeModelView = view * node.transform.worldMatrix;
                const float4x4 nodeMvp = proj * nodeModelView;
                const float4x4 prevNodeModelView = prevViewForMotion * node.transform.worldMatrix;
                const float4x4 prevNodeMvp = prevProjForMotion * prevNodeModelView;

                SceneInstanceTransform transform{};
                transform.mvp = transpose(nodeMvp);
                transform.modelView = transpose(nodeModelView);
                transform.prevMvp = transpose(prevNodeMvp);
                instanceTransforms.push_back(transform);
            }

            RhiBufferDesc instanceBufferDesc;
            instanceBufferDesc.size = instanceTransforms.size() * sizeof(SceneInstanceTransform);
            instanceBufferDesc.initialData = instanceTransforms.data();
            instanceBufferDesc.hostVisible = true;
            instanceBufferDesc.debugName = "Metal Visibility Instance Transforms";
            instanceTransformBuffer = frameGraphBackend->createBuffer(instanceBufferDesc);
            if (!instanceTransformBuffer) {
                visibilityInstanceCount = 0;
            }
        }

        // --- FrameContext population ---
        const RhiNativeCommandBuffer& nativeCommandBuffer = runtime->currentCommandBuffer();
        FrameContext frameCtx{};
        frameCtx.width = renderWidth;
        frameCtx.height = renderHeight;
        frameCtx.view = view;
        frameCtx.proj = proj;
        frameCtx.unjitteredProj = unjitteredProj;
        frameCtx.cameraWorldPos = cameraWorldPos;
        frameCtx.cameraRight = float4(cameraRight.x, cameraRight.y, cameraRight.z, 0.0f);
        frameCtx.cameraUp = float4(cameraUp.x, cameraUp.y, cameraUp.z, 0.0f);
        frameCtx.cameraForward = float4(cameraForward.x, cameraForward.y, cameraForward.z, 0.0f);
        frameCtx.cameraNearZ = camera.nearZ;
        frameCtx.cameraFovY = camera.fovY;
        frameCtx.prevView = hasPrevMatrices ? prevView : view;
        frameCtx.prevProj = hasPrevMatrices ? prevProj : unjitteredProj;
        frameCtx.prevCullView = hasPrevMatrices ? prevCullView : view;
        frameCtx.prevCullProj = hasPrevMatrices ? prevCullProj : proj;
        frameCtx.prevCameraWorldPos = hasPrevMatrices ? prevCameraWorldPos : frameCtx.cameraWorldPos;
        frameCtx.jitterOffset = jitterOffset;
        frameCtx.frameIndex = frameIndex;
        frameCtx.enableTAA = enableVisibilityTAA;
        frameCtx.displayWidth = width;
        frameCtx.displayHeight = height;
        frameCtx.renderWidth = renderWidth;
        frameCtx.renderHeight = renderHeight;
        frameCtx.historyReset = visibilityHistoryResetRequested;
        frameCtx.worldLightDir = float4(sunDirection.x, sunDirection.y, sunDirection.z, 0.0f);
        frameCtx.viewLightDir = view * frameCtx.worldLightDir;
        frameCtx.lightColorIntensity =
            float4(sunLight.color.x, sunLight.color.y, sunLight.color.z, sunLight.intensity);
        frameCtx.meshletCount = scene.meshlets().meshletCount;
        frameCtx.materialCount = scene.materials().materialCount;
        frameCtx.textureCount = static_cast<uint32_t>(scene.materials().textures.size());
        frameCtx.visibleMeshletNodes = visibleMeshletNodes;
        if (useVisibilityRenderGraph &&
            frameCtx.visibleMeshletNodes.size() > static_cast<size_t>(visibilityInstanceCount)) {
            frameCtx.visibleMeshletNodes.resize(visibilityInstanceCount);
        }
        frameCtx.visibleIndexNodes = visibleIndexNodes;
        frameCtx.visibilityInstanceCount = visibilityInstanceCount;
        frameCtx.instanceTransformBufferRhi = instanceTransformBuffer.get();
        frameCtx.commandBuffer = &nativeCommandBuffer;
        frameCtx.depthClearValue = scene.depthClearValue();
        frameCtx.cameraFarZ = camera.farZ;
        {
            const double now = glfwGetTime();
            frameCtx.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameCtx.enableRTShadows = useVisibilityRenderGraph && rtShadowsAvailable && enableRTShadows;
        frameCtx.enableAtmosphereSky = atmosphereSkyAvailable;
        frameCtx.gpuDrivenCulling = useVisibilityRenderGraph && visibilityGpuCullingAvailable;
        frameCtx.renderMode = useVisibilityRenderGraph ? 2 : 0;

        pipelineBuilder.updateFrame(&runtime->currentBackbufferTexture(), &frameCtx);

        FrameGraph& activeFg = pipelineBuilder.frameGraph();
        if (showGraphDebug) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.debugImGui();
        }
        if (showRenderPassUI) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            activeFg.renderPassUI();
        }
        if (showImGuiDemo) {
            ImGui::SetNextWindowDockID(dockspaceId, ImGuiCond_FirstUseEver);
            ImGui::ShowDemoWindow(&showImGuiDemo);
        }
        ImGui::Render();

        // Update TLAS
        if (useVisibilityRenderGraph && rtShadowsAvailable) {
            updateTLAS(nativeCommandBuffer, scene.sceneGraph(), scene.shadowResources());
        }

        // Execute frame graph
        auto fgCommandBuffer = runtime->createCommandBuffer();
        pipelineBuilder.execute(*fgCommandBuffer, *frameGraphBackend);
        visibilityHistoryResetRequested = false;

        runtime->present();

        // Store matrices for next frame
        prevView = view;
        prevProj = unjitteredProj;
        prevCullView = view;
        prevCullProj = proj;
        prevCameraWorldPos = frameCtx.cameraWorldPos;
        hasPrevMatrices = true;
        frameIndex++;

        FrameMark;
    }

    // --- Cleanup ---
    pipelineBuilder.frameGraph().reset();

    // Release cluster LOD buffers
    rhiReleaseHandle(clusterLOD.meshletBuffer);
    rhiReleaseHandle(clusterLOD.meshletVerticesBuffer);
    rhiReleaseHandle(clusterLOD.meshletTrianglesBuffer);
    rhiReleaseHandle(clusterLOD.boundsBuffer);
    rhiReleaseHandle(clusterLOD.materialIDsBuffer);
    rhiReleaseHandle(clusterLOD.groupBuffer);
    rhiReleaseHandle(clusterLOD.nodeBuffer);

    // Release instance ring buffers
    for (auto& buf : instanceTransformRing) {
        buf.reset();
    }

    runtime->shutdownImGui();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Destroy runtime before GLFW to avoid accessing destroyed window/layer
    runtime.reset();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
