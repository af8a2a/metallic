#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <ml.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "glfw_metal_bridge.h"
#include "imgui_metal_bridge.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "material_loader.h"
#include "camera.h"
#include "input.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "scene_graph.h"
#include "scene_graph_ui.h"
#include "raytraced_shadows.h"

#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>

#include <tracy/Tracy.hpp>
#include "tracy_metal.h"
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


struct AtmosphereTextureSet {
    MTL::Texture* transmittance = nullptr;
    MTL::Texture* scattering = nullptr;
    MTL::Texture* irradiance = nullptr;
    MTL::SamplerState* sampler = nullptr;

    bool isValid() const {
        return transmittance && scattering && irradiance && sampler;
    }

    void release() {
        if (transmittance) { transmittance->release(); transmittance = nullptr; }
        if (scattering) { scattering->release(); scattering = nullptr; }
        if (irradiance) { irradiance->release(); irradiance = nullptr; }
        if (sampler) { sampler->release(); sampler = nullptr; }
    }
};

static std::vector<float> loadFloatData(const std::string& path, size_t expectedCount) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Atmosphere: missing texture data {}", path);
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if (size == 0 || size % sizeof(float) != 0) {
        spdlog::warn("Atmosphere: invalid data size {} ({} bytes)", path, size);
        return {};
    }

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file) {
        spdlog::warn("Atmosphere: failed to read {}", path);
        return {};
    }

    if (expectedCount > 0 && data.size() != expectedCount) {
        spdlog::warn("Atmosphere: unexpected element count in {} ({} vs {})",
                     path, data.size(), expectedCount);
    }
    return data;
}

static MTL::Texture* createTexture2D(MTL::Device* device, int width, int height,
                                     const float* data) {
    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA32Float, width, height, false);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    auto* tex = device->newTexture(desc);
    desc->release();
    if (!tex) {
        return nullptr;
    }
    size_t bytesPerRow = static_cast<size_t>(width) * 4 * sizeof(float);
    tex->replaceRegion(MTL::Region(0, 0, 0, width, height, 1), 0, data, bytesPerRow);
    return tex;
}

static MTL::Texture* createTexture3D(MTL::Device* device, int width, int height, int depth,
                                     const float* data) {
    auto* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType3D);
    desc->setPixelFormat(MTL::PixelFormatRGBA32Float);
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setDepth(depth);
    desc->setMipmapLevelCount(1);
    desc->setStorageMode(MTL::StorageModeShared);
    desc->setUsage(MTL::TextureUsageShaderRead);
    auto* tex = device->newTexture(desc);
    desc->release();
    if (!tex) {
        return nullptr;
    }
    size_t bytesPerRow = static_cast<size_t>(width) * 4 * sizeof(float);
    size_t bytesPerImage = bytesPerRow * static_cast<size_t>(height);
    tex->replaceRegion(MTL::Region(0, 0, 0, width, height, depth), 0, 0,
                       data, bytesPerRow, bytesPerImage);
    return tex;
}

static bool loadAtmosphereTextures(MTL::Device* device, const char* projectRoot,
                                   AtmosphereTextureSet& out) {
    constexpr int kTransmittanceWidth = 256;
    constexpr int kTransmittanceHeight = 64;
    constexpr int kScatteringWidth = 256; // NU_SIZE * MU_S_SIZE
    constexpr int kScatteringHeight = 128;
    constexpr int kScatteringDepth = 32;
    constexpr int kIrradianceWidth = 64;
    constexpr int kIrradianceHeight = 16;

    std::string basePath = std::string(projectRoot) + "/Asset/Atmosphere/";
    auto transmittance = loadFloatData(
        basePath + "transmittance.dat",
        static_cast<size_t>(kTransmittanceWidth) * kTransmittanceHeight * 4);
    auto scattering = loadFloatData(
        basePath + "scattering.dat",
        static_cast<size_t>(kScatteringWidth) * kScatteringHeight * kScatteringDepth * 4);
    auto irradiance = loadFloatData(
        basePath + "irradiance.dat",
        static_cast<size_t>(kIrradianceWidth) * kIrradianceHeight * 4);

    if (transmittance.empty() || scattering.empty() || irradiance.empty()) {
        return false;
    }

    out.transmittance = createTexture2D(
        device, kTransmittanceWidth, kTransmittanceHeight, transmittance.data());
    out.scattering = createTexture3D(
        device, kScatteringWidth, kScatteringHeight, kScatteringDepth, scattering.data());
    out.irradiance = createTexture2D(
        device, kIrradianceWidth, kIrradianceHeight, irradiance.data());

    if (!out.transmittance || !out.scattering || !out.irradiance) {
        out.release();
        return false;
    }

    auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    samplerDesc->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    out.sampler = device->newSamplerState(samplerDesc);
    samplerDesc->release();

    if (!out.sampler) {
        out.release();
        return false;
    }

    return true;
}


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

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        spdlog::error("Metal is not supported on this device");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    spdlog::info("Metal device: {}", device->name()->utf8String());

    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    // Tracy GPU profiling context
    TracyMetalCtxHandle tracyGpuCtx = tracyMetalCreate(device);

    CA::MetalLayer* metalLayer = static_cast<CA::MetalLayer*>(
        attachMetalLayerToGLFWWindow(window));
    metalLayer->setDevice(device);
    metalLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    metalLayer->setFramebufferOnly(false);

    // Load scene mesh
    LoadedMesh sceneMesh;
    if (!loadGLTFMesh(device, "Asset/Sponza/glTF/Sponza.gltf", sceneMesh)) {
        spdlog::error("Failed to load scene mesh");
        return 1;
    }

    // Build meshlets for mesh shader rendering
    MeshletData meshletData;
    if (!buildMeshlets(device, sceneMesh, meshletData)) {
        spdlog::error("Failed to build meshlets");
        return 1;
    }

    // Load materials and textures
    LoadedMaterials materials;
    if (!loadGLTFMaterials(device, commandQueue, "Asset/Sponza/glTF/Sponza.gltf", materials)) {
        spdlog::error("Failed to load materials");
        return 1;
    }

    // Build scene graph from glTF node hierarchy
    SceneGraph sceneGraph;
    if (!sceneGraph.buildFromGLTF("Asset/Sponza/glTF/Sponza.gltf", sceneMesh, meshletData)) {
        spdlog::error("Failed to build scene graph");
        return 1;
    }
    sceneGraph.updateTransforms();

    const char* projectRoot = PROJECT_SOURCE_DIR;

    // Build raytracing acceleration structures for shadows
    RaytracedShadowResources shadowResources;
    bool rtShadowsAvailable = false;
    if (device->supportsRaytracing()) {
        ZoneScopedN("Build Acceleration Structures");
        if (buildAccelerationStructures(device, commandQueue, sceneMesh, sceneGraph, shadowResources) &&
            createShadowPipeline(device, shadowResources, projectRoot)) {
            rtShadowsAvailable = true;
            spdlog::info("Raytraced shadows enabled");
        } else {
            spdlog::error("Failed to initialize raytraced shadows");
            shadowResources.release();
        }
    } else {
        spdlog::info("Raytracing not supported on this device");
    }

    // Init orbit camera
    OrbitCamera camera;
    camera.initFromBounds(sceneMesh.bboxMin, sceneMesh.bboxMax);

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
    imguiInit(device);

    // Build all shader pipelines
    ShaderManager shaderManager(device, projectRoot);
    if (!shaderManager.buildAll()) return 1;

    AtmosphereTextureSet atmosphereTextures;
    bool atmosphereLoaded = loadAtmosphereTextures(device, projectRoot, atmosphereTextures);
    if (!atmosphereLoaded) {
        spdlog::warn("Atmosphere textures not found or invalid; sky pass will use fallback");
    }

    if (atmosphereLoaded) {
        shaderManager.importTexture("transmittance", atmosphereTextures.transmittance);
        shaderManager.importTexture("scattering", atmosphereTextures.scattering);
        shaderManager.importTexture("irradiance", atmosphereTextures.irradiance);
        shaderManager.importSampler("atmosphere", atmosphereTextures.sampler);
    }
    bool skyAvailable = atmosphereLoaded && shaderManager.hasSkyPipeline();

    int renderMode = 0; // 0=Vertex, 1=Mesh, 2=Visibility Buffer
    bool enableFrustumCull = false;
    bool enableConeCull = false;
    bool enableRTShadows = true;
    bool enableAtmosphereSky = skyAvailable;
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
    std::string visPipelinePath = std::string(projectRoot) + "/Pipelines/visibility_buffer.json";
    std::string fwdPipelinePath = std::string(projectRoot) + "/Pipelines/forward.json";

    if (!loadPipelineAssetChecked(visPipelinePath, "visibility buffer", visPipelineAsset) ||
        !loadPipelineAssetChecked(fwdPipelinePath, "forward", fwdPipelineAsset)) {
        return 1;
    }

    const double depthClearValue = ML_DEPTH_REVERSED ? 0.0 : 1.0;

    // Create depth stencil state
    MTL::DepthStencilDescriptor* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(
        ML_DEPTH_REVERSED ? MTL::CompareFunctionGreater : MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(true);
    MTL::DepthStencilState* depthState = device->newDepthStencilState(depthDesc);
    depthDesc->release();

    // 1x1 depth texture used only to tell ImGui about the depth pixel format
    auto* imguiDepthTexDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, 1, 1, false);
    imguiDepthTexDesc->setStorageMode(MTL::StorageModePrivate);
    imguiDepthTexDesc->setUsage(MTL::TextureUsageRenderTarget);
    MTL::Texture* imguiDepthDummy = device->newTexture(imguiDepthTexDesc);

    // 1x1 shadow texture for non-RT paths (white = fully lit)
    auto* shadowDummyDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatR8Unorm, 1, 1, false);
    shadowDummyDesc->setStorageMode(MTL::StorageModeShared);
    shadowDummyDesc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* shadowDummyTex = device->newTexture(shadowDummyDesc);
    uint8_t shadowClear = 0xFF;
    shadowDummyTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, &shadowClear, 1);

    // 1x1 sky fallback texture (BGRA8)
    auto* skyFallbackDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatBGRA8Unorm, 1, 1, false);
    skyFallbackDesc->setStorageMode(MTL::StorageModeShared);
    skyFallbackDesc->setUsage(MTL::TextureUsageShaderRead);
    MTL::Texture* skyFallbackTex = device->newTexture(skyFallbackDesc);
    uint8_t skyFallbackColor[4] = {77, 51, 26, 255}; // B, G, R, A (~0.3, 0.2, 0.1)
    skyFallbackTex->replaceRegion(MTL::Region(0, 0, 0, 1, 1, 1), 0, skyFallbackColor, 4);

    // Create initial framebuffer size (used for metalLayer)
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

    // Persistent render context and pipeline builder (hoisted out of frame loop)
    RenderContext ctx{sceneMesh, meshletData, materials, sceneGraph,
                      shadowResources, depthState, shadowDummyTex, skyFallbackTex, depthClearValue};
    PipelineRuntimeContext& rtCtx = shaderManager.runtimeContext();

    PipelineBuilder pipelineBuilder(ctx);
    bool pipelineNeedsRebuild = true;
    int lastRenderMode = -1;

    while (!glfwWindowShouldClose(window)) {
        ZoneScopedN("Frame");
        glfwPollEvents();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) {
            pool->release();
            continue;
        }
        metalLayer->setDrawableSize(CGSizeMake(width, height));

        CA::MetalDrawable* drawable = metalLayer->nextDrawable();
        if (!drawable) {
            pool->release();
            continue;
        }

        // Compute matrices
        float aspect;
        float4x4 view, proj;
        float4 worldLightDir, viewLightDir, lightColorIntensity;
        float4 cameraWorldPos;
        {
            ZoneScopedN("Matrix Computation");
            aspect = (float)width / (float)height;
            view = camera.viewMatrix();
            proj = camera.projectionMatrix(aspect);

            // Light data from scene graph sun source.
            DirectionalLight sunLight = sceneGraph.getSunDirectionalLight();
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

            sceneGraph.updateTransforms();
        }

        // Render pass
        MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

        // Collect GPU timestamps from previous frames
        tracyMetalCollect(tracyGpuCtx);

        // ImGui new frame (needs a render pass descriptor for Metal backend)
        {
            ZoneScopedN("ImGui Frame");
        MTL::RenderPassDescriptor* imguiFramePass = MTL::RenderPassDescriptor::alloc()->init();
        imguiFramePass->colorAttachments()->object(0)->setTexture(drawable->texture());
        // Always provide a depth attachment so the ImGui pipeline matches even when
        // switching render modes within the same frame.
        imguiFramePass->depthAttachment()->setTexture(imguiDepthDummy);
        imguiNewFrame(imguiFramePass);
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
        if (renderMode >= 1) {
            ImGui::Text("Meshlets: %u", meshletData.meshletCount);
            ImGui::Checkbox("Frustum Culling", &enableFrustumCull);
            ImGui::Checkbox("Backface Culling", &enableConeCull);
        }
        if (renderMode == 2 && rtShadowsAvailable) {
            ImGui::Checkbox("RT Shadows", &enableRTShadows);
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

            // Shadow ray shader (native Metal, not managed by ShaderManager)
            if (rtShadowsAvailable) {
                if (reloadShadowPipeline(device, shadowResources, projectRoot)) {
                    reloaded++;
                } else { failed++; }
            }

            skyAvailable = atmosphereLoaded && shaderManager.hasSkyPipeline();

            if (failed == 0)
                spdlog::info("All {} shaders reloaded successfully", reloaded);
            else
                spdlog::warn("{} shaders reloaded, {} failed (keeping old pipelines)", reloaded, failed);

            pipelineNeedsRebuild = true;
        }
        ImGui::End();
        imguiFramePass->release();
        } // end ImGui Frame zone

        if (showSceneGraphWindow)
            drawSceneGraphUI(sceneGraph);

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

            pipelineNeedsRebuild = reloadedAnyPipeline;
        }
        pipelineReloadKeyDown = f6Down;

        std::vector<uint32_t> visibleMeshletNodes;
        std::vector<uint32_t> visibleIndexNodes;
        visibleMeshletNodes.reserve(sceneGraph.nodes.size());
        visibleIndexNodes.reserve(sceneGraph.nodes.size());
        for (const auto& node : sceneGraph.nodes) {
            if (!sceneGraph.isNodeVisible(node.id))
                continue;
            if (node.meshletCount > 0)
                visibleMeshletNodes.push_back(node.id);
            if (node.indexCount > 0)
                visibleIndexNodes.push_back(node.id);
        }

        // --- Build FrameGraph (unified data-driven path) ---
        MTL::Buffer* instanceTransformBuffer = nullptr;

        FrameContext frameCtx;

        // Visibility buffer mode needs instance transform buffer
        uint32_t visibilityInstanceCount = 0;
        if (renderMode == 2) {
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
                meshletData.meshletCount > (kVisibilityMeshletMask + 1u)) {
                spdlog::warn("Visibility meshlet id limit exceeded ({} > {}), overflowing meshlets will be culled",
                             meshletData.meshletCount, kVisibilityMeshletMask + 1);
                warnedMeshletOverflow = true;
            }

            visibilityInstanceCount =
                static_cast<uint32_t>(std::min<size_t>(visibleMeshletNodes.size(),
                                                       kVisibilityInstanceMask + 1));

            std::vector<SceneInstanceTransform> visibilityInstanceTransforms;
            visibilityInstanceTransforms.reserve(std::max<size_t>(visibilityInstanceCount, 1));
            for (uint32_t instanceID = 0; instanceID < visibilityInstanceCount; instanceID++) {
                const auto& node = sceneGraph.nodes[visibleMeshletNodes[instanceID]];
                float4x4 nodeModelView = view * node.transform.worldMatrix;
                float4x4 nodeMVP = proj * nodeModelView;
                visibilityInstanceTransforms.push_back({transpose(nodeMVP), transpose(nodeModelView)});
            }

            if (visibilityInstanceTransforms.empty()) {
                float4x4 identity = float4x4::Identity();
                float4x4 fallbackMV = view * identity;
                float4x4 fallbackMVP = proj * fallbackMV;
                visibilityInstanceTransforms.push_back({transpose(fallbackMVP), transpose(fallbackMV)});
            }

            instanceTransformBuffer = device->newBuffer(
                visibilityInstanceTransforms.data(),
                visibilityInstanceTransforms.size() * sizeof(SceneInstanceTransform),
                MTL::ResourceStorageModeShared);
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
        frameCtx.meshletCount = meshletData.meshletCount;
        frameCtx.materialCount = materials.materialCount;
        frameCtx.textureCount = static_cast<uint32_t>(materials.textures.size());
        frameCtx.visibleMeshletNodes = visibleMeshletNodes;
        frameCtx.visibleIndexNodes = visibleIndexNodes;
        frameCtx.visibilityInstanceCount = visibilityInstanceCount;
        frameCtx.instanceTransformBuffer = instanceTransformBuffer;
        frameCtx.commandBuffer = commandBuffer;
        frameCtx.depthClearValue = depthClearValue;
        frameCtx.cameraFarZ = camera.farZ;
        frameCtx.enableFrustumCull = enableFrustumCull;
        frameCtx.enableConeCull = enableConeCull;
        frameCtx.enableRTShadows = rtShadowsAvailable && enableRTShadows;
        frameCtx.enableAtmosphereSky = skyAvailable && enableAtmosphereSky;
        frameCtx.renderMode = renderMode;

        // Select active pipeline asset based on render mode
        const PipelineAsset& activePipelineAsset = (renderMode == 2) ? visPipelineAsset : fwdPipelineAsset;

        // Detect mode switch
        if (renderMode != lastRenderMode) {
            pipelineNeedsRebuild = true;
            lastRenderMode = renderMode;
        }

        // Rebuild pipeline only when needed (first frame, F6 reload, resolution change, mode switch)
        if (pipelineNeedsRebuild || pipelineBuilder.needsRebuild(width, height)) {
            rtCtx.backbuffer = drawable->texture();
            bool buildSucceeded = pipelineBuilder.build(activePipelineAsset, rtCtx, width, height);
            if (!buildSucceeded) {
                spdlog::error("Failed to build pipeline: {}", pipelineBuilder.lastError());
            } else {
                pipelineBuilder.compile();
            }
            pipelineNeedsRebuild = !buildSucceeded;
        }

        // Per-frame: swap backbuffer, update frame context, reset transients
        pipelineBuilder.updateFrame(drawable->texture(), &frameCtx);

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
        if (rtShadowsAvailable && renderMode == 2) {
            ZoneScopedN("Update TLAS");
            updateTLAS(commandBuffer, sceneGraph, shadowResources);
        }

        activeFg.execute(commandBuffer, device, tracyGpuCtx);

        commandBuffer->presentDrawable(drawable);
        commandBuffer->commit();

        if (instanceTransformBuffer)
            instanceTransformBuffer->release();

        FrameMark;
        pool->release();
    }

    // Cleanup ImGui
    imguiShutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup Tracy GPU context
    tracyMetalDestroy(tracyGpuCtx);

    // Cleanup
    shadowResources.release();
    imguiDepthDummy->release();
    shadowDummyTex->release();
    skyFallbackTex->release();
    atmosphereTextures.release();
    meshletData.meshletBuffer->release();
    meshletData.meshletVertices->release();
    meshletData.meshletTriangles->release();
    meshletData.boundsBuffer->release();
    meshletData.materialIDs->release();
    for (auto* tex : materials.textures) {
        if (tex) tex->release();
    }
    materials.materialBuffer->release();
    materials.sampler->release();
    sceneMesh.positionBuffer->release();
    sceneMesh.normalBuffer->release();
    sceneMesh.uvBuffer->release();
    sceneMesh.indexBuffer->release();
    depthState->release();
    commandQueue->release();
    device->release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
