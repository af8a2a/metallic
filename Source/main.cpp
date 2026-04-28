#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>

#include <tracy/Tracy.hpp>
#include <ml.h>

#include "camera.h"
#include "input.h"

#include "scene_graph_ui.h"
#include "scene_context.h"

#include "cluster_occlusion_state.h"
#include "frame_graph.h"
#include "visibility_constants.h"
#include "render_uniforms.h"
#include "render_pass.h"
#include "shader_manager.h"
#include "blit_pass.h"
#include "tonemap_pass.h"
#include "imgui_overlay_pass.h"
#include "shadow_ray_pass.h"
#include "sky_pass.h"
#include "pipeline_asset.h"
#include "pipeline_builder.h"
#include "frame_context.h"
#include "rhi_window_runtime.h"

bool s_viewportHovered = true;

static bool loadPipelineAssetChecked(const std::string& path,
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

static const char* backendLabel(RhiBackendType backend) {
    switch (backend) {
    case RhiBackendType::Metal: return "Metal";
    case RhiBackendType::Vulkan: return "Vulkan";
    }
    return "Unknown";
}

int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Metallic - Sponza", nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create GLFW window");
        glfwTerminate();
        return 1;
    }

    std::string runtimeError;
    auto runtime = createRhiWindowRuntime(defaultRhiWindowRuntimeBackend(), window, runtimeError);
    if (!runtime) {
        spdlog::error("{}", runtimeError);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    spdlog::info("RHI device: {}", runtime->deviceName());
    RhiDeviceHandle device(runtime->device().nativeHandle(), runtime->device().ownerContext());
    RhiCommandQueueHandle commandQueue(runtime->commandQueue().nativeHandle());
    RhiContext* rhiContext = device.ownerContext();

    const char* projectRoot = PROJECT_SOURCE_DIR;

    // Load all scene data
    SceneContext scene(device, commandQueue, projectRoot);
    if (!scene.loadAll("Asset/Sponza/glTF/Sponza.gltf")) return 1;

    // Init orbit camera
    OrbitCamera camera;
    camera.initFromBounds(scene.mesh().bboxMin, scene.mesh().bboxMax);

    // Setup input
    InputState inputState;
    inputState.camera = &camera;
    setupInputCallbacks(window, &inputState);

    // Init Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui_ImplGlfw_InitForOther(window, true);
    runtime->initImGui();

    // Build all shader pipelines
    const bool supportsMeshShaders = !rhiContext || rhiContext->features().meshShaders;
    ShaderManagerProfile shaderProfile = runtime->backendType() == RhiBackendType::Vulkan
        ? ShaderManagerProfile::vulkanVisibility()
        : ShaderManagerProfile::full();
    ShaderManager shaderManager(device,
                                projectRoot,
                                supportsMeshShaders,
                                supportsMeshShaders,
                                shaderProfile);
    if (runtime->backendType() == RhiBackendType::Metal) {
        shaderManager.setGlobalDefines({
            {"METALLIC_BINDLESS_MAX_SAMPLED_IMAGES", "96"},
            {"METALLIC_BINDLESS_MAX_SAMPLERS", "1"},
            {"METALLIC_METAL_DIRECT_BINDING", "1"},
        });
    }
    if (!shaderManager.buildAll()) return 1;

    PipelineRuntimeContext& rtCtx = shaderManager.runtimeContext();
    rtCtx.rhi = rhiContext;

    auto importRuntimeTexture = [&](const std::string& name, const RhiTexture& texture) {
        shaderManager.importTexture(name, texture);
    };

    if (scene.atmosphereLoaded()) {
        auto& atm = scene.atmosphereTextures();
        importRuntimeTexture("transmittance", atm.transmittance);
        importRuntimeTexture("scattering", atm.scattering);
        importRuntimeTexture("irradiance", atm.irradiance);
        shaderManager.importSampler("atmosphere", atm.sampler);
    }
    bool skyAvailable = scene.atmosphereLoaded() && shaderManager.hasSkyPipeline();

    int renderMode = 3; // 0=Vertex, 1=Mesh, 2=Visibility Buffer, 3=Cluster Visualization
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableGPUCulling = false;
    bool enableAtmosphereSky = skyAvailable;
    bool enableTAA = true;
    uint32_t frameIndex = 0;
    float skyExposure = 10.0f;
    bool showGraphDebug = false;
    bool showSceneGraphWindow = true;
    bool showRenderPassUI = true;
    bool showTextureViewer = false;
    bool showImGuiDemo = false;
    bool exportGraphKeyDown = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;

    // Load pipeline assets
    PipelineAsset baselinePipelineAsset;
    PipelineAsset clusterVisPipelineAsset;
    std::string baselinePipelinePath = std::string(projectRoot) + "/Pipelines/visibilitybuffer.json";
    std::string clusterVisPipelinePath = std::string(projectRoot) + "/Pipelines/cluster_vis.json";

    bool baselinePipelineLoaded =
        loadPipelineAssetChecked(baselinePipelinePath, backendLabel(runtime->backendType()), baselinePipelineAsset);
    bool clusterVisPipelineLoaded =
        loadPipelineAssetChecked(clusterVisPipelinePath, "Cluster visualization", clusterVisPipelineAsset);
    if (!baselinePipelineLoaded && !clusterVisPipelineLoaded) {
        return 1;
    }
    bool useClusterVisMode = clusterVisPipelineLoaded;

    // Persistent render context and pipeline builder
    RenderContext ctx = scene.renderContext();
    PipelineBuilder pipelineBuilder(ctx);
    auto frameGraphBackend = runtime->createFrameGraphBackend();
    rtCtx.resourceFactory = frameGraphBackend.get();
    ClusterOcclusionState clusterOcclusionState;
    rtCtx.clusterOcclusionState = &clusterOcclusionState;
    bool pipelineNeedsRebuild = true;
    bool historyResetRequested = true;
    int lastRenderMode = -1;
    double lastFrameTime = glfwGetTime();
    float4x4 prevView, prevProj;
    float4x4 prevCullView, prevCullProj;
    float4 prevCameraWorldPos = float4(0.f, 0.f, 0.f, 1.f);
    bool hasPrevMatrices = false;

    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("Frame");
        glfwPollEvents();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            continue;
        }

        if (!runtime->beginFrame(static_cast<uint32_t>(width), static_cast<uint32_t>(height))) {
            continue;
        }

        // Compute matrices
        float aspect;
        float4x4 view, proj;
        float4 worldLightDir, viewLightDir, lightColorIntensity;
        float4 cameraWorldPos;
        float2 jitterOffset = float2(0.f, 0.f);
        {
            ZoneScopedN("Matrix Computation");
            aspect = (float)width / (float)height;
            view = camera.viewMatrix();
            proj = camera.projectionMatrix(aspect);

            // Apply jitter to projection for TAA
            if (enableTAA) {
                jitterOffset = OrbitCamera::haltonJitter(frameIndex);
                proj = OrbitCamera::jitteredProjectionMatrix(
                    camera.fovY, aspect, camera.nearZ, camera.farZ,
                    jitterOffset.x, jitterOffset.y,
                    static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            }

            // Light data from scene graph sun source.
            DirectionalLight sunLight = scene.sceneGraph().getSunDirectionalLight();
            worldLightDir = float4(sunLight.direction, 0.0f);
            viewLightDir = view * worldLightDir;
            lightColorIntensity = float4(
                sunLight.color.x,
                sunLight.color.y,
                sunLight.color.z,
                sunLight.intensity);

            // Camera position in world space
            float cosA = std::cos(camera.azimuth), sinA = std::sin(camera.azimuth);
            float cosE = std::cos(camera.elevation), sinE = std::sin(camera.elevation);
            cameraWorldPos = float4(
                camera.target.x + camera.distance * cosE * sinA,
                camera.target.y + camera.distance * sinE,
                camera.target.z + camera.distance * cosE * cosA,
                1.0f);

            scene.sceneGraph().updateTransforms();
            scene.updateGpuScene();
        }

        // Collect GPU timestamps from previous frames
        runtime->collectGpuTimestamps();

        // ImGui new frame
        {
            ZoneScopedN("ImGui Frame");
            // Always provide a depth attachment so the ImGui pipeline matches even when
            // switching render modes within the same frame.
            runtime->beginImGuiFrame(&scene.imguiDepthDummy());
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("View")) {
                    ImGui::MenuItem("Scene Browser", nullptr, &showSceneGraphWindow);
                    ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
                    ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
                    ImGui::MenuItem("Texture Viewer", nullptr, &showTextureViewer);
                    ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
                    ImGui::EndMenu();
                }
                ImGui::EndMainMenuBar();
            }

            ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("Renderer");
            ImGui::Text("%.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::Separator();
            ImGui::TextUnformatted(useClusterVisMode ? "Pipeline: Cluster Visualization" :
                                                       "Pipeline: Baseline");
            if (clusterVisPipelineLoaded && baselinePipelineLoaded) {
                if (ImGui::Checkbox("Cluster Vis", &useClusterVisMode)) {
                    spdlog::info("Cluster visualization mode: {}", useClusterVisMode ? "ON" : "OFF");
                    pipelineNeedsRebuild = true;
                    historyResetRequested = true;
                }
            }
            ImGui::Text("Meshlets: %u", scene.meshlets().meshletCount);
            ImGui::Text("Clusters: %u", scene.gpuScene().clusterVisWorklistCount);
            ImGui::Checkbox("Frustum Culling", &enableFrustumCull);
            ImGui::Checkbox("Backface Culling", &enableConeCull);
            if (!useClusterVisMode && scene.rtShadowsAvailable()) {
                ImGui::Checkbox("RT Shadows", &enableRTShadows);
            }
            if (!useClusterVisMode) {
                ImGui::Checkbox("TAA", &enableTAA);
            }
            if (skyAvailable) {
                ImGui::Checkbox("Atmosphere Sky", &enableAtmosphereSky);
                ImGui::SliderFloat("Sky Exposure", &skyExposure, 0.1f, 20.0f, "%.2f");
            } else {
                ImGui::TextDisabled("Atmosphere Sky (missing textures)");
            }
            ImGui::Checkbox("Show Graph", &showGraphDebug);
            ImGui::Checkbox("Render Pass UI", &showRenderPassUI);
            ImGui::Checkbox("Texture Viewer", &showTextureViewer);
            ImGui::Separator();
            bool f5Down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
            bool triggerReload = ImGui::Button("Reload Shaders (F5)") || (f5Down && !reloadKeyDown);
            reloadKeyDown = f5Down;
            if (triggerReload) {
                spdlog::info("Reloading shaders...");
                auto [reloaded, failed] = shaderManager.reloadAll();

                // Shadow ray shader is still rebuilt through the backend utility path.
                if (scene.rtShadowsAvailable()) {
                    if (reloadShadowPipeline(device, scene.shadowResources(), projectRoot)) {
                        reloaded++;
                    } else { failed++; }
                }

                skyAvailable = scene.atmosphereLoaded() && shaderManager.hasSkyPipeline();

                if (failed == 0)
                    spdlog::info("All {} shaders reloaded successfully", reloaded);
                else
                    spdlog::warn("{} shaders reloaded, {} failed (keeping old pipelines)", reloaded, failed);

                pipelineNeedsRebuild = true;
                historyResetRequested = true;
            }
            ImGui::End();
        } // end ImGui Frame zone

        if (showSceneGraphWindow)
            drawSceneGraphUI(scene.sceneGraph());

        if (showImGuiDemo)
            ImGui::ShowDemoWindow(&showImGuiDemo);

        // Pipeline hot-reload (F6)
        bool f6Down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6Down && !pipelineReloadKeyDown) {
            bool reloadedAnyPipeline = false;

            PipelineAsset reloadedBaseline;
            if (loadPipelineAssetChecked(baselinePipelinePath, backendLabel(runtime->backendType()), reloadedBaseline)) {
                baselinePipelineAsset = std::move(reloadedBaseline);
                baselinePipelineLoaded = true;
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous baseline pipeline: {}", baselinePipelineAsset.name);
            }

            PipelineAsset reloadedClusterVis;
            if (loadPipelineAssetChecked(clusterVisPipelinePath, "Cluster visualization", reloadedClusterVis)) {
                clusterVisPipelineAsset = std::move(reloadedClusterVis);
                clusterVisPipelineLoaded = true;
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous cluster visualization pipeline: {}", clusterVisPipelineAsset.name);
            }

            pipelineNeedsRebuild = reloadedAnyPipeline;
            if (reloadedAnyPipeline) {
                historyResetRequested = true;
            }
        }
        pipelineReloadKeyDown = f6Down;

        renderMode = (useClusterVisMode && clusterVisPipelineLoaded) ? 3 : 2;
        const bool gpuDrivenVisibilityPath =
            enableGPUCulling && (renderMode == 2 || renderMode == 3);
        const bool needsCpuMeshletVisibility =
            renderMode == 1 || ((renderMode == 2 || renderMode == 3) && !gpuDrivenVisibilityPath);
        const bool needsCpuIndexVisibility = renderMode == 0;

        std::vector<uint32_t> visibleMeshletNodes;
        std::vector<uint32_t> visibleIndexNodes;
        if (needsCpuMeshletVisibility) {
            visibleMeshletNodes.reserve(scene.sceneGraph().nodes.size());
        }
        if (needsCpuIndexVisibility) {
            visibleIndexNodes.reserve(scene.sceneGraph().nodes.size());
        }
        if (needsCpuMeshletVisibility || needsCpuIndexVisibility) {
            for (const auto& node : scene.sceneGraph().nodes) {
                if (!scene.sceneGraph().isNodeVisible(node.id))
                    continue;
                if (needsCpuMeshletVisibility && node.meshletCount > 0)
                    visibleMeshletNodes.push_back(node.id);
                if (needsCpuIndexVisibility && node.indexCount > 0)
                    visibleIndexNodes.push_back(node.id);
            }
        }

        FrameContext frameCtx;

        uint32_t visibilityInstanceCount = 0;
        if (renderMode == 2 || renderMode == 3) {
            ZoneScopedN("Visibility Instance Setup");

            static bool warnedInstanceOverflow = false;
            if (!warnedInstanceOverflow &&
                scene.gpuScene().instanceCount > (kVisibilityInstanceMask + 1u)) {
                spdlog::warn("GPU scene instance limit exceeded for visibility encoding ({} > {}), overflowing instances will be dropped in GPU visibility mode",
                             scene.gpuScene().instanceCount, kVisibilityInstanceMask + 1);
                warnedInstanceOverflow = true;
            }

            static bool warnedMeshletOverflow = false;
            if (!warnedMeshletOverflow &&
                scene.meshlets().meshletCount > (kVisibilityMeshletMask + 1u)) {
                spdlog::warn("Visibility meshlet id limit exceeded ({} > {}), overflowing meshlets will be culled",
                             scene.meshlets().meshletCount, kVisibilityMeshletMask + 1);
                warnedMeshletOverflow = true;
            }

            if (!gpuDrivenVisibilityPath) {
                visibilityInstanceCount =
                    static_cast<uint32_t>(std::min<size_t>(visibleMeshletNodes.size(),
                                                           kVisibilityInstanceMask + 1));
            }
        }

        // Populate frame context (shared across all modes)
        frameCtx.width = width;
        frameCtx.height = height;
        frameCtx.view = view;
        frameCtx.proj = proj;
        frameCtx.unjitteredProj = camera.projectionMatrix(aspect);
        frameCtx.cameraWorldPos = cameraWorldPos;
        {
            float3 eye(cameraWorldPos.x, cameraWorldPos.y, cameraWorldPos.z);
            float3 cameraForward = normalize(camera.target - eye);
            float3 cameraRight = normalize(cross(cameraForward, float3(0.f, 1.f, 0.f)));
            float3 cameraUp = cross(cameraRight, cameraForward);
            frameCtx.cameraRight = float4(cameraRight.x, cameraRight.y, cameraRight.z, 0.f);
            frameCtx.cameraUp = float4(cameraUp.x, cameraUp.y, cameraUp.z, 0.f);
            frameCtx.cameraForward = float4(cameraForward.x, cameraForward.y, cameraForward.z, 0.f);
        }
        frameCtx.cameraNearZ = camera.nearZ;
        frameCtx.cameraFovY = camera.fovY;
        frameCtx.worldLightDir = worldLightDir;
        frameCtx.viewLightDir = viewLightDir;
        frameCtx.lightColorIntensity = lightColorIntensity;
        frameCtx.meshletCount = scene.meshlets().meshletCount;
        frameCtx.materialCount = scene.materials().materialCount;
        frameCtx.textureCount = static_cast<uint32_t>(scene.materials().textures.size());
        frameCtx.visibleMeshletNodes = visibleMeshletNodes;
        frameCtx.visibleIndexNodes = visibleIndexNodes;
        frameCtx.visibilityInstanceCount = visibilityInstanceCount;
        frameCtx.depthClearValue = scene.depthClearValue();
        frameCtx.cameraFarZ = camera.farZ;
        frameCtx.displayWidth = width;
        frameCtx.displayHeight = height;
        frameCtx.renderWidth = width;
        frameCtx.renderHeight = height;
        frameCtx.historyReset = historyResetRequested;
        {
            double now = glfwGetTime();
            frameCtx.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameCtx.enableFrustumCull = enableFrustumCull;
        frameCtx.enableConeCull = enableConeCull;
        frameCtx.enableRTShadows = scene.rtShadowsAvailable() && enableRTShadows;
        frameCtx.enableAtmosphereSky = skyAvailable && enableAtmosphereSky;
        frameCtx.gpuDrivenCulling = gpuDrivenVisibilityPath;
        frameCtx.renderMode = renderMode;
        frameCtx.prevView = hasPrevMatrices ? prevView : view;
        frameCtx.prevProj = hasPrevMatrices ? prevProj : camera.projectionMatrix(aspect);
        frameCtx.prevCullView = hasPrevMatrices ? prevCullView : view;
        frameCtx.prevCullProj = hasPrevMatrices ? prevCullProj : proj;
        frameCtx.prevCameraWorldPos = hasPrevMatrices ? prevCameraWorldPos : cameraWorldPos;
        frameCtx.jitterOffset = jitterOffset;
        frameCtx.frameIndex = frameIndex;
        frameCtx.enableTAA = enableTAA;

        // Select active pipeline asset based on render mode
        const PipelineAsset& activePipelineAsset =
            (useClusterVisMode && clusterVisPipelineLoaded) || !baselinePipelineLoaded
                ? clusterVisPipelineAsset
                : baselinePipelineAsset;

        // Detect mode switch
        if (renderMode != lastRenderMode) {
            pipelineNeedsRebuild = true;
            lastRenderMode = renderMode;
        }

        // Rebuild pipeline only when needed (first frame, F6 reload, resolution change, mode switch)
        if (pipelineNeedsRebuild || pipelineBuilder.needsRebuild(width, height)) {
            rtCtx.backbufferRhi = &runtime->currentBackbufferTexture();
            bool buildSucceeded = pipelineBuilder.build(activePipelineAsset, rtCtx, width, height);
            if (!buildSucceeded) {
                spdlog::error("Failed to build pipeline: {}", pipelineBuilder.lastError());
            } else {
                pipelineBuilder.compile();
            }
            pipelineNeedsRebuild = !buildSucceeded;
        }

        // Per-frame: swap backbuffer, update frame context, reset transients
        rtCtx.backbufferRhi = &runtime->currentBackbufferTexture();
        pipelineBuilder.updateFrame(rtCtx.backbufferRhi, &frameCtx);

        FrameGraph& activeFg = pipelineBuilder.frameGraph();

        if (showGraphDebug)
            activeFg.debugImGui();

        if (showRenderPassUI)
            activeFg.renderPassUI();

        if (showTextureViewer)
            activeFg.textureViewerImGui();

        ImGui::Render();

        bool gKeyDown = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS;
        if (gKeyDown && !exportGraphKeyDown) {
            std::ofstream dot("framegraph.dot");
            if (dot.is_open()) {
                activeFg.exportGraphviz(dot);
                spdlog::info("Exported framegraph.dot");
            }
        }
        exportGraphKeyDown = gKeyDown;

        // Update TLAS with current scene transforms before executing the frame graph
        if (scene.rtShadowsAvailable() && renderMode == 2) {
            ZoneScopedN("Update TLAS");
            updateTLAS(runtime->currentCommandBuffer(), scene.sceneGraph(), scene.shadowResources());
        }

        auto fgCommandBuffer = runtime->createCommandBuffer();
        pipelineBuilder.execute(*fgCommandBuffer, *frameGraphBackend);

        if (showTextureViewer)
            activeFg.captureTextureViewerState();

        runtime->present();

        // Store unjittered matrices for next frame's motion vectors
        prevView = view;
        prevProj = camera.projectionMatrix(aspect);
        prevCullView = view;
        prevCullProj = proj;
        prevCameraWorldPos = cameraWorldPos;
        hasPrevMatrices = true;
        frameIndex++;
        historyResetRequested = false;

        FrameMark;
    }

    // Cleanup ImGui
    runtime->shutdownImGui();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
