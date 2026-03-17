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



int main() {
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Metallic - Sponza", nullptr, nullptr);
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
    const RhiDevice& device = runtime->device();
    const RhiCommandQueue& commandQueue = runtime->commandQueue();

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
    ShaderManager shaderManager(device, projectRoot);
    if (!shaderManager.buildAll()) return 1;

    PipelineRuntimeContext& rtCtx = shaderManager.runtimeContext();

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

    int renderMode = 0; // 0=Vertex, 1=Mesh, 2=Visibility Buffer, 3=Meshlet Debug
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableGPUCulling = true;
    bool enableAtmosphereSky = skyAvailable;
    bool enableTAA = true;
    uint32_t frameIndex = 0;
    float skyExposure = 10.0f;
    bool showGraphDebug = false;
    bool showSceneGraphWindow = true;
    bool showRenderPassUI = true;
    bool showImGuiDemo = false;
    bool exportGraphKeyDown = false;
    bool reloadKeyDown = false;
    bool pipelineReloadKeyDown = false;

    // Load pipeline assets
    PipelineAsset visPipelineAsset;
    PipelineAsset fwdPipelineAsset;
    PipelineAsset meshletDbgPipelineAsset;
    std::string visPipelinePath = std::string(projectRoot) + "/Pipelines/visibility_buffer.json";
    std::string fwdPipelinePath = std::string(projectRoot) + "/Pipelines/forward.json";
    std::string meshletDbgPipelinePath = std::string(projectRoot) + "/Pipelines/meshlet_debug.json";

    if (!loadPipelineAssetChecked(visPipelinePath, "visibility buffer", visPipelineAsset) ||
        !loadPipelineAssetChecked(fwdPipelinePath, "forward", fwdPipelineAsset) ||
        !loadPipelineAssetChecked(meshletDbgPipelinePath, "meshlet debug", meshletDbgPipelineAsset)) {
        return 1;
    }

    // Persistent render context and pipeline builder
    RenderContext ctx = scene.renderContext();
    PipelineBuilder pipelineBuilder(ctx);
    auto frameGraphBackend = runtime->createFrameGraphBackend();
    rtCtx.resourceFactory = frameGraphBackend.get();
    bool pipelineNeedsRebuild = true;
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
                    ImGui::MenuItem("Scene Graph", nullptr, &showSceneGraphWindow);
                    ImGui::MenuItem("Render Passes", nullptr, &showRenderPassUI);
                    ImGui::MenuItem("FrameGraph", nullptr, &showGraphDebug);
                    ImGui::MenuItem("ImGui Demo", nullptr, &showImGuiDemo);
                    ImGui::EndMenu();
                }
                ImGui::EndMainMenuBar();
            }

            ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("Renderer");
            ImGui::Text("%.1f FPS (%.3f ms)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
            ImGui::Separator();
            ImGui::RadioButton("Vertex Shader", &renderMode, 0);
            ImGui::RadioButton("Mesh Shader", &renderMode, 1);
            ImGui::RadioButton("Visibility Buffer", &renderMode, 2);
            ImGui::RadioButton("Meshlet Debug", &renderMode, 3);
            if (renderMode >= 1) {
                ImGui::Text("Meshlets: %u", scene.meshlets().meshletCount);
                ImGui::Checkbox("Frustum Culling", &enableFrustumCull);
                ImGui::Checkbox("Backface Culling", &enableConeCull);
            }
            if (renderMode == 2 && scene.rtShadowsAvailable()) {
                ImGui::Checkbox("RT Shadows", &enableRTShadows);
            }
            if (renderMode == 2) {
                ImGui::Checkbox("TAA", &enableTAA);
            }
            if (renderMode == 2 || renderMode == 3) {
                ImGui::Checkbox("GPU-Driven Culling", &enableGPUCulling);
            }
            if (skyAvailable) {
                ImGui::Checkbox("Atmosphere Sky", &enableAtmosphereSky);
                ImGui::SliderFloat("Sky Exposure", &skyExposure, 0.1f, 20.0f, "%.2f");
            } else {
                ImGui::TextDisabled("Atmosphere Sky (missing textures)");
            }
            ImGui::Checkbox("Show Graph", &showGraphDebug);
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

            PipelineAsset reloadedVis;
            if (loadPipelineAssetChecked(visPipelinePath, "visibility buffer", reloadedVis)) {
                visPipelineAsset = std::move(reloadedVis);
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous visibility buffer pipeline: {}", visPipelineAsset.name);
            }

            PipelineAsset reloadedFwd;
            if (loadPipelineAssetChecked(fwdPipelinePath, "forward", reloadedFwd)) {
                fwdPipelineAsset = std::move(reloadedFwd);
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous forward pipeline: {}", fwdPipelineAsset.name);
            }

            PipelineAsset reloadedDbg;
            if (loadPipelineAssetChecked(meshletDbgPipelinePath, "meshlet debug", reloadedDbg)) {
                meshletDbgPipelineAsset = std::move(reloadedDbg);
                reloadedAnyPipeline = true;
            } else {
                spdlog::warn("Keeping previous meshlet debug pipeline: {}", meshletDbgPipelineAsset.name);
            }

            pipelineNeedsRebuild = reloadedAnyPipeline;
        }
        pipelineReloadKeyDown = f6Down;

        std::vector<uint32_t> visibleMeshletNodes;
        std::vector<uint32_t> visibleIndexNodes;
        visibleMeshletNodes.reserve(scene.sceneGraph().nodes.size());
        visibleIndexNodes.reserve(scene.sceneGraph().nodes.size());
        for (const auto& node : scene.sceneGraph().nodes) {
            if (!scene.sceneGraph().isNodeVisible(node.id))
                continue;
            if (node.meshletCount > 0)
                visibleMeshletNodes.push_back(node.id);
            if (node.indexCount > 0)
                visibleIndexNodes.push_back(node.id);
        }

        // --- Build FrameGraph (unified data-driven path) ---
        std::unique_ptr<RhiBuffer> instanceTransformBuffer;

        FrameContext frameCtx;

        // Visibility buffer mode needs instance transform buffer
        uint32_t visibilityInstanceCount = 0;
        if (renderMode == 2 || renderMode == 3) {
            ZoneScopedN("Visibility Instance Setup");

            static bool warnedInstanceOverflow = false;
            if (!warnedInstanceOverflow &&
                visibleMeshletNodes.size() > static_cast<size_t>(kVisibilityInstanceMask + 1)) {
                spdlog::warn("Visibility buffer instance limit exceeded ({} > {}), extra nodes will be skipped in this mode",
                             visibleMeshletNodes.size(), kVisibilityInstanceMask + 1);
                warnedInstanceOverflow = true;
            }

            static bool warnedMeshletOverflow = false;
            if (!warnedMeshletOverflow &&
                scene.meshlets().meshletCount > (kVisibilityMeshletMask + 1u)) {
                spdlog::warn("Visibility meshlet id limit exceeded ({} > {}), overflowing meshlets will be culled",
                             scene.meshlets().meshletCount, kVisibilityMeshletMask + 1);
                warnedMeshletOverflow = true;
            }

            visibilityInstanceCount =
                static_cast<uint32_t>(std::min<size_t>(visibleMeshletNodes.size(),
                                                       kVisibilityInstanceMask + 1));

            std::vector<SceneInstanceTransform> visibilityInstanceTransforms;
            visibilityInstanceTransforms.reserve(std::max<size_t>(visibilityInstanceCount, 1));
            float4x4 prevViewForMotion = hasPrevMatrices ? prevView : view;
            float4x4 prevProjForMotion = hasPrevMatrices ? prevProj : camera.projectionMatrix(aspect);
            for (uint32_t instanceID = 0; instanceID < visibilityInstanceCount; instanceID++) {
                const auto& node = scene.sceneGraph().nodes[visibleMeshletNodes[instanceID]];
                float4x4 nodeModelView = view * node.transform.worldMatrix;
                float4x4 nodeMVP = proj * nodeModelView;
                float4x4 prevNodeModelView = prevViewForMotion * node.transform.worldMatrix;
                float4x4 prevNodeMVP = prevProjForMotion * prevNodeModelView;
                SceneInstanceTransform instanceTransform{};
                instanceTransform.mvp = transpose(nodeMVP);
                instanceTransform.modelView = transpose(nodeModelView);
                instanceTransform.prevMvp = transpose(prevNodeMVP);
                visibilityInstanceTransforms.push_back(instanceTransform);
            }

            if (visibilityInstanceTransforms.empty()) {
                float4x4 identity = float4x4::Identity();
                float4x4 fallbackMV = view * identity;
                float4x4 fallbackMVP = proj * fallbackMV;
                float4x4 prevFallbackMV = prevViewForMotion * identity;
                float4x4 prevFallbackMVP = prevProjForMotion * prevFallbackMV;
                SceneInstanceTransform fallbackTransform{};
                fallbackTransform.mvp = transpose(fallbackMVP);
                fallbackTransform.modelView = transpose(fallbackMV);
                fallbackTransform.prevMvp = transpose(prevFallbackMVP);
                visibilityInstanceTransforms.push_back(fallbackTransform);
            }

            RhiBufferDesc instanceBufferDesc;
            instanceBufferDesc.size =
                visibilityInstanceTransforms.size() * sizeof(SceneInstanceTransform);
            instanceBufferDesc.initialData = visibilityInstanceTransforms.data();
            instanceBufferDesc.hostVisible = true;
            instanceBufferDesc.debugName = "Visibility Instance Transforms";
            instanceTransformBuffer = frameGraphBackend->createBuffer(instanceBufferDesc);
        }

        // Populate frame context (shared across all modes)
        frameCtx.width = width;
        frameCtx.height = height;
        frameCtx.view = view;
        frameCtx.proj = proj;
        frameCtx.cameraWorldPos = cameraWorldPos;
        frameCtx.worldLightDir = worldLightDir;
        frameCtx.viewLightDir = viewLightDir;
        frameCtx.lightColorIntensity = lightColorIntensity;
        frameCtx.meshletCount = scene.meshlets().meshletCount;
        frameCtx.materialCount = scene.materials().materialCount;
        frameCtx.textureCount = static_cast<uint32_t>(scene.materials().textures.size());
        frameCtx.visibleMeshletNodes = visibleMeshletNodes;
        frameCtx.visibleIndexNodes = visibleIndexNodes;
        frameCtx.visibilityInstanceCount = visibilityInstanceCount;
        frameCtx.instanceTransformBufferRhi = instanceTransformBuffer.get();
        frameCtx.commandBuffer = &runtime->currentCommandBuffer();
        frameCtx.depthClearValue = scene.depthClearValue();
        frameCtx.cameraFarZ = camera.farZ;
        {
            double now = glfwGetTime();
            frameCtx.deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
        }
        frameCtx.enableFrustumCull = enableFrustumCull;
        frameCtx.enableConeCull = enableConeCull;
        frameCtx.enableRTShadows = scene.rtShadowsAvailable() && enableRTShadows;
        frameCtx.enableAtmosphereSky = skyAvailable && enableAtmosphereSky;
        frameCtx.gpuDrivenCulling = enableGPUCulling && (renderMode == 2 || renderMode == 3);
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
            (renderMode == 3) ? meshletDbgPipelineAsset :
            (renderMode == 2) ? visPipelineAsset : fwdPipelineAsset;

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
            updateTLAS(*frameCtx.commandBuffer, scene.sceneGraph(), scene.shadowResources());
        }

        auto fgCommandBuffer = runtime->createCommandBuffer();
        pipelineBuilder.execute(*fgCommandBuffer, *frameGraphBackend);

        runtime->present();

        // Store unjittered matrices for next frame's motion vectors
        prevView = view;
        prevProj = camera.projectionMatrix(aspect);
        prevCullView = view;
        prevCullProj = proj;
        prevCameraWorldPos = cameraWorldPos;
        hasPrevMatrices = true;
        frameIndex++;

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
