#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_driven_helpers.h"
#include "gpu_cull_resources.h"
#include "cluster_lod_builder.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <vector>

class VisibilityPass : public RenderPass {
public:
    // Data-driven constructor
    VisibilityPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    METALLIC_PASS_TYPE_INFO(VisibilityPass, "Visibility Pass", "Geometry",
        (std::vector<PassSlotInfo>{
            makeInputSlot("cullResult", "Cull Result", true),
            makeInputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeInputSlot("cullCounter", "Cull Counter", true),
            makeInputSlot("instanceData", "Instance Data", true),
            makeInputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeInputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
            makeInputSlot("visibilityInstances", "Visibility Instances", true),
            makeInputSlot("visibilityInput", "Visibility Input", true),
            makeInputSlot("depthInput", "Depth Input", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("visibility", "Visibility"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);

#ifdef _WIN32
    METALLIC_PASS_EDITOR_TYPE_INFO(VisibilityPass, "Visibility Pass", "Geometry",
        (std::vector<PassSlotInfo>{
            makeInputSlot("cullResult", "Cull Result"),
            makeHiddenInputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeHiddenInputSlot("cullCounter", "Cull Counter", true),
            makeHiddenInputSlot("instanceData", "Instance Data", true),
            makeHiddenInputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeHiddenInputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeHiddenInputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
            makeHiddenInputSlot("visibilityInstances", "Visibility Instances", true),
            makeHiddenInputSlot("visibilityInput", "Visibility Input", true),
            makeHiddenInputSlot("depthInput", "Depth Input", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("visibility", "Visibility"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);
#else
    METALLIC_PASS_EDITOR_TYPE_INFO(VisibilityPass, "Visibility Pass", "Geometry",
        (std::vector<PassSlotInfo>{}),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("visibility", "Visibility"),
            makeOutputSlot("depth", "Depth")
        }),
        PassTypeInfo::PassType::Render);
#endif

    FGPassType passType() const override { return FGPassType::Render; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.contains("clearColor")) {
            auto& cc = config.config["clearColor"];
            if (cc.is_array() && cc.size() >= 4) {
                m_clearColor = RhiClearColor(
                    cc[0].get<double>(), cc[1].get<double>(),
                    cc[2].get<double>(), cc[3].get<double>());
            }
        }
        if (config.config.is_object() && config.config.contains("loadExisting")) {
            m_loadExisting = config.config["loadExisting"].get<bool>();
        }
    }

    // Output resources
    FGResource visibility;
    FGResource depth;

    FGResource getOutput(const std::string& name) const override {
        if (name == "visibility") return visibility;
        if (name == "depth") return depth;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        // Legacy dependency-only edge kept for older pipeline assets.
        FGResource cullInput = getInput("cullResult");
        if (cullInput.isValid()) {
            builder.read(cullInput);
        }
        FGResource visibleMeshletsInput = getInput("visibleMeshlets");
        if (!visibleMeshletsInput.isValid()) {
            visibleMeshletsInput = getInput("visibilityWorklist");
        }
        if (visibleMeshletsInput.isValid()) {
            m_visibleMeshletsRead = builder.read(visibleMeshletsInput, FGResourceUsage::StorageRead);
        }
        FGResource cullCounterInput = getInput("cullCounter");
        if (!cullCounterInput.isValid()) {
            cullCounterInput = getInput("visibilityWorklistState");
        }
        if (!cullCounterInput.isValid()) {
            cullCounterInput = getInput("visibilityIndirectArgs");
        }
        if (cullCounterInput.isValid()) {
            m_cullCounterRead = builder.read(cullCounterInput, FGResourceUsage::Indirect);
        }
        FGResource instanceDataInput = getInput("instanceData");
        if (!instanceDataInput.isValid()) {
            instanceDataInput = getInput("visibilityInstances");
        }
        if (instanceDataInput.isValid()) {
            m_instanceDataRead = builder.read(instanceDataInput, FGResourceUsage::StorageRead);
        }
        FGResource visibilityInput = getInput("visibilityInput");
        FGResource depthInput = getInput("depthInput");
        const bool loadVisibility = m_loadExisting && visibilityInput.isValid();
        const bool loadDepth = m_loadExisting && depthInput.isValid();

        visibility = loadVisibility
            ? visibilityInput
            : builder.create("visibility",
                             FGTextureDesc::renderTarget(m_width, m_height, RhiFormat::R32Uint));
        depth = loadDepth
            ? depthInput
            : builder.create("depth",
                             FGTextureDesc::depthTarget(m_width, m_height));
        visibility = builder.setColorAttachment(0,
                                                 visibility,
                                                 loadVisibility ? RhiLoadAction::Load
                                                                : RhiLoadAction::Clear,
                                                 RhiStoreAction::Store,
                                                 m_clearColor);
        depth = builder.setDepthAttachment(depth,
                                           loadDepth ? RhiLoadAction::Load
                                                     : RhiLoadAction::Clear,
                                           RhiStoreAction::Store,
                                           m_ctx.depthClearValue);
    }

    void executeRender(RhiRenderCommandEncoder& encoder) override {
        ZoneScopedN("VisibilityPass");
        MICROPROFILE_SCOPEI("RenderPass", "VisibilityPass", 0xffff8800);
        if (!m_frameContext || !m_runtimeContext) return;

        encoder.setDepthStencilState(&m_ctx.depthState);
        encoder.setFrontFacingWinding(RhiWinding::CounterClockwise);
        encoder.setCullMode(RhiCullMode::Back);

        // GPU-driven indirect path
        const RhiBuffer* visibleMeshletBuffer =
            (m_frameGraph && m_visibleMeshletsRead.isValid())
                ? m_frameGraph->getBuffer(m_visibleMeshletsRead)
                : nullptr;
        const RhiBuffer* worklistStateBuffer =
            (m_frameGraph && m_cullCounterRead.isValid())
                ? m_frameGraph->getBuffer(m_cullCounterRead)
                : nullptr;
        const RhiBuffer* sceneInstanceBuffer =
            m_ctx.gpuScene.instanceBuffer.nativeHandle() ? &m_ctx.gpuScene.instanceBuffer : nullptr;

        m_gpuPathRequiredLastFrame = m_frameContext->gpuDrivenCulling;
        m_gpuPathReadyLastFrame = false;
        m_lastLegacyVisibleNodeCount = 0;
        m_lastVisibleMeshletWorkItems =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);
        m_lastIndirectMeshGroupCount =
            GpuDriven::readBuiltIndirectGroupCount<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);

        bool useGPUPath = m_gpuPathRequiredLastFrame &&
            visibleMeshletBuffer &&
            worklistStateBuffer &&
            sceneInstanceBuffer;

        if (useGPUPath) {
            auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityIndirectPass");
            if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle())
                useGPUPath = false;
            else
                encoder.setRenderPipeline(pipeIt->second);
        }

        m_gpuPathReadyLastFrame = useGPUPath;

        static bool sLoggedPath = false;
        if (!sLoggedPath) {
            auto visIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityPass");
            auto visIndirectIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityIndirectPass");
            spdlog::info(
                "VisibilityPass path probe: ctx={} useGPUPath={} gpuDriven={} visibleBuf={} counterBuf={} instanceBuf={} visPipe={} visIndirectPipe={}",
                fmt::ptr(m_frameContext),
                useGPUPath,
                m_frameContext->gpuDrivenCulling,
                fmt::ptr(visibleMeshletBuffer),
                fmt::ptr(worklistStateBuffer),
                fmt::ptr(sceneInstanceBuffer),
                visIt != m_runtimeContext->renderPipelinesRhi.end() ? fmt::ptr(visIt->second.nativeHandle()) : fmt::ptr(static_cast<void*>(nullptr)),
                visIndirectIt != m_runtimeContext->renderPipelinesRhi.end() ? fmt::ptr(visIndirectIt->second.nativeHandle()) : fmt::ptr(static_cast<void*>(nullptr)));
            sLoggedPath = true;
        }

        const bool useBindlessSceneTextures = m_runtimeContext->useBindlessSceneTextures;
        std::vector<const RhiTexture*> materialTextures;
        if (!useBindlessSceneTextures) {
            materialTextures.reserve(m_ctx.materials.textureViews.size());
            for (auto* texture : m_ctx.materials.textureViews) {
                materialTextures.push_back(texture);
            }
        }

        if (useGPUPath) {
            const RhiBuffer* lodMeshletBuffer =
                m_ctx.clusterLodData.meshletBuffer.nativeHandle()
                    ? &m_ctx.clusterLodData.meshletBuffer
                    : &m_ctx.meshletData.meshletBuffer;
            const RhiBuffer* lodMeshletVerticesBuffer =
                m_ctx.clusterLodData.meshletVerticesBuffer.nativeHandle()
                    ? &m_ctx.clusterLodData.meshletVerticesBuffer
                    : &m_ctx.meshletData.meshletVertices;
            const RhiBuffer* lodMeshletTrianglesBuffer =
                m_ctx.clusterLodData.meshletTrianglesBuffer.nativeHandle()
                    ? &m_ctx.clusterLodData.meshletTrianglesBuffer
                    : &m_ctx.meshletData.meshletTriangles;
            const RhiBuffer* lodMaterialIdsBuffer =
                m_ctx.clusterLodData.materialIDsBuffer.nativeHandle()
                    ? &m_ctx.clusterLodData.materialIDsBuffer
                    : &m_ctx.meshletData.materialIDs;

            // Bind shared geometry buffers (same indices as visibility.slang)
            encoder.setMeshBuffer(&m_ctx.sceneMesh.positionBuffer, 0, GpuDriven::MeshletVisibilityBindings::kPositions);
            encoder.setMeshBuffer(&m_ctx.sceneMesh.normalBuffer, 0, GpuDriven::MeshletVisibilityBindings::kNormals);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMeshlets);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletVertices, 0, GpuDriven::MeshletVisibilityBindings::kMeshletVertices);
            encoder.setMeshBuffer(&m_ctx.meshletData.meshletTriangles, 0, GpuDriven::MeshletVisibilityBindings::kMeshletTriangles);
            encoder.setMeshBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kBounds);
            encoder.setMeshBuffer(&m_ctx.sceneMesh.uvBuffer, 0, GpuDriven::MeshletVisibilityBindings::kUvs);
            encoder.setMeshBuffer(&m_ctx.meshletData.materialIDs, 0, GpuDriven::MeshletVisibilityBindings::kMaterialIds);
            encoder.setMeshBuffer(&m_ctx.materials.materialBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMaterials);
            encoder.setMeshBuffer(visibleMeshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kVisibleMeshlets);
            encoder.setMeshBuffer(sceneInstanceBuffer, 0, GpuDriven::MeshletVisibilityBindings::kSceneInstances);
            encoder.setMeshBuffer(lodMeshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshlets);
            encoder.setMeshBuffer(lodMeshletVerticesBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshletVertices);
            encoder.setMeshBuffer(lodMeshletTrianglesBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshletTriangles);
            encoder.setMeshBuffer(lodMaterialIdsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMaterialIds);

            // Fragment stage needs all buffers too (Slang KernelContext)
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.positionBuffer, 0, GpuDriven::MeshletVisibilityBindings::kPositions);
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.normalBuffer, 0, GpuDriven::MeshletVisibilityBindings::kNormals);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMeshlets);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletVertices, 0, GpuDriven::MeshletVisibilityBindings::kMeshletVertices);
            encoder.setFragmentBuffer(&m_ctx.meshletData.meshletTriangles, 0, GpuDriven::MeshletVisibilityBindings::kMeshletTriangles);
            encoder.setFragmentBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kBounds);
            encoder.setFragmentBuffer(&m_ctx.sceneMesh.uvBuffer, 0, GpuDriven::MeshletVisibilityBindings::kUvs);
            encoder.setFragmentBuffer(&m_ctx.meshletData.materialIDs, 0, GpuDriven::MeshletVisibilityBindings::kMaterialIds);
            encoder.setFragmentBuffer(&m_ctx.materials.materialBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMaterials);
            encoder.setFragmentBuffer(visibleMeshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kVisibleMeshlets);
            encoder.setFragmentBuffer(sceneInstanceBuffer, 0, GpuDriven::MeshletVisibilityBindings::kSceneInstances);
            encoder.setFragmentBuffer(lodMeshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshlets);
            encoder.setFragmentBuffer(lodMeshletVerticesBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshletVertices);
            encoder.setFragmentBuffer(lodMeshletTrianglesBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMeshletTriangles);
            encoder.setFragmentBuffer(lodMaterialIdsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kLodMaterialIds);

            if (!useBindlessSceneTextures && !materialTextures.empty()) {
                encoder.setFragmentTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
                encoder.setMeshTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
                encoder.setFragmentSampler(&m_ctx.materials.sampler, 0);
                encoder.setMeshSampler(&m_ctx.materials.sampler, 0);
            }

            // GlobalUniforms at buffer(0) via setMeshBytes
            struct {
                float4x4 viewProj;
                float4 lightDir;
                float4 lightColorIntensity;
            } globalUni;
            globalUni.viewProj = transpose(m_frameContext->proj * m_frameContext->view);
            globalUni.lightDir = m_frameContext->viewLightDir;
            globalUni.lightColorIntensity = m_frameContext->lightColorIntensity;
            encoder.setMeshBytes(&globalUni,
                                 sizeof(globalUni),
                                 GpuDriven::MeshletVisibilityBindings::kGlobalUniforms);
            encoder.setFragmentBytes(&globalUni,
                                     sizeof(globalUni),
                                     GpuDriven::MeshletVisibilityBindings::kGlobalUniforms);

            // Single indirect draw call
            encoder.drawMeshThreadgroupsIndirect(*worklistStateBuffer,
                                                 GpuDriven::MeshDispatchCommandLayout::kIndirectArgsOffset,
                                                 {1, 1, 1},
                                                 {128, 1, 1});
            m_missingGpuPathWarningLogged = false;
            return;
        }

        if (m_gpuPathRequiredLastFrame) {
            if (!m_missingGpuPathWarningLogged) {
                spdlog::warn(
                    "VisibilityPass requires the GPU indirect raster path, but it is unavailable; skipping legacy CPU per-node fallback for this frame");
                m_missingGpuPathWarningLogged = true;
            }
            return;
        }

        // CPU per-node dispatch fallback for legacy non-GPU visibility modes.
        m_missingGpuPathWarningLogged = false;

        auto pipeIt = m_runtimeContext->renderPipelinesRhi.find("VisibilityPass");
        if (pipeIt == m_runtimeContext->renderPipelinesRhi.end() || !pipeIt->second.nativeHandle()) return;
        encoder.setRenderPipeline(pipeIt->second);

        // Bind shared buffers once
        encoder.setMeshBuffer(&m_ctx.sceneMesh.positionBuffer, 0, GpuDriven::MeshletVisibilityBindings::kPositions);
        encoder.setMeshBuffer(&m_ctx.sceneMesh.normalBuffer, 0, GpuDriven::MeshletVisibilityBindings::kNormals);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMeshlets);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletVertices, 0, GpuDriven::MeshletVisibilityBindings::kMeshletVertices);
        encoder.setMeshBuffer(&m_ctx.meshletData.meshletTriangles, 0, GpuDriven::MeshletVisibilityBindings::kMeshletTriangles);
        encoder.setMeshBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kBounds);
        encoder.setMeshBuffer(&m_ctx.sceneMesh.uvBuffer, 0, GpuDriven::MeshletVisibilityBindings::kUvs);
        encoder.setMeshBuffer(&m_ctx.meshletData.materialIDs, 0, GpuDriven::MeshletVisibilityBindings::kMaterialIds);
        encoder.setMeshBuffer(&m_ctx.materials.materialBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMaterials);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.positionBuffer, 0, GpuDriven::MeshletVisibilityBindings::kPositions);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.normalBuffer, 0, GpuDriven::MeshletVisibilityBindings::kNormals);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMeshlets);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletVertices, 0, GpuDriven::MeshletVisibilityBindings::kMeshletVertices);
        encoder.setFragmentBuffer(&m_ctx.meshletData.meshletTriangles, 0, GpuDriven::MeshletVisibilityBindings::kMeshletTriangles);
        encoder.setFragmentBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletVisibilityBindings::kBounds);
        encoder.setFragmentBuffer(&m_ctx.sceneMesh.uvBuffer, 0, GpuDriven::MeshletVisibilityBindings::kUvs);
        encoder.setFragmentBuffer(&m_ctx.meshletData.materialIDs, 0, GpuDriven::MeshletVisibilityBindings::kMaterialIds);
        encoder.setFragmentBuffer(&m_ctx.materials.materialBuffer, 0, GpuDriven::MeshletVisibilityBindings::kMaterials);
        if (!useBindlessSceneTextures && !materialTextures.empty()) {
            encoder.setFragmentTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
            encoder.setMeshTextures(materialTextures.data(), 0, static_cast<uint32_t>(materialTextures.size()));
            encoder.setFragmentSampler(&m_ctx.materials.sampler, 0);
            encoder.setMeshSampler(&m_ctx.materials.sampler, 0);
        }

        // Per-node dispatch (CPU fallback)
        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = std::min<uint32_t>(
            m_frameContext->visibilityInstanceCount,
            static_cast<uint32_t>(visibleNodes.size()));
        m_lastLegacyVisibleNodeCount = instanceCount;
        if (instanceCount == 0) return;

        Uniforms baseUni{};
        baseUni.lightDir = m_frameContext->viewLightDir;
        baseUni.lightColorIntensity = m_frameContext->lightColorIntensity;
        baseUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        baseUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;

        for (uint32_t instanceID = 0; instanceID < instanceCount; instanceID++) {
            const uint32_t nodeID = visibleNodes[instanceID];
            if (nodeID >= m_ctx.sceneGraph.nodes.size()) {
                continue;
            }
            const auto& node = m_ctx.sceneGraph.nodes[nodeID];
            float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
            float4x4 nodeMVP = m_frameContext->proj * nodeModelView;
            Uniforms nodeUniforms = baseUni;
            nodeUniforms.mvp = transpose(nodeMVP);
            nodeUniforms.modelView = transpose(nodeModelView);
            extractFrustumPlanes(nodeMVP, nodeUniforms.frustumPlanes);
            float4x4 invModel = node.transform.worldMatrix;
            invModel.Invert();
            nodeUniforms.cameraPos = invModel * m_frameContext->cameraWorldPos;
            nodeUniforms.meshletBaseOffset = node.meshletStart;
            nodeUniforms.instanceID = instanceID;
            encoder.setMeshBytes(&nodeUniforms,
                                 sizeof(nodeUniforms),
                                 GpuDriven::MeshletVisibilityBindings::kGlobalUniforms);
            encoder.setFragmentBytes(&nodeUniforms,
                                     sizeof(nodeUniforms),
                                     GpuDriven::MeshletVisibilityBindings::kGlobalUniforms);
            encoder.drawMeshThreadgroups({node.meshletCount, 1, 1},
                                         {1, 1, 1},
                                         {128, 1, 1});
        }
    }

    void renderUI() override {
        ImGui::Text("Resolution: %d x %d", m_width, m_height);
        if (m_frameContext) {
            ImGui::Text("Frustum Cull: %s", m_frameContext->enableFrustumCull ? "On" : "Off");
            ImGui::Text("Cone Cull: %s", m_frameContext->enableConeCull ? "On" : "Off");
            ImGui::Text("GPU Raster Requested: %s", m_frameContext->gpuDrivenCulling ? "Yes" : "No");
            ImGui::Text("Load Existing Targets: %s", m_loadExisting ? "Yes" : "No");
            ImGui::Text("Last Raster Path: %s",
                        m_gpuPathRequiredLastFrame
                            ? (m_gpuPathReadyLastFrame
                                   ? "GPU Indirect"
                                   : "GPU Indirect Unavailable")
                            : "CPU Per-Node (Legacy)");
            if (m_gpuPathRequiredLastFrame || m_frameContext->gpuDrivenCulling) {
                ImGui::Text("Last Meshlet Work Items: %u", m_lastVisibleMeshletWorkItems);
                ImGui::Text("Last Indirect Mesh Groups: %u", m_lastIndirectMeshGroupCount);
            } else {
                ImGui::Text("Visible Nodes (Legacy): %u", m_lastLegacyVisibleNodeCount);
            }
        }
    }

private:
    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Visibility Pass";
    RhiClearColor m_clearColor = RhiClearColor(0xFFFFFFFF, 0, 0, 0);
    bool m_loadExisting = false;
    bool m_gpuPathRequiredLastFrame = false;
    bool m_gpuPathReadyLastFrame = false;
    bool m_missingGpuPathWarningLogged = false;
    uint32_t m_lastVisibleMeshletWorkItems = 0;
    uint32_t m_lastIndirectMeshGroupCount = 0;
    uint32_t m_lastLegacyVisibleNodeCount = 0;
    FGResource m_visibleMeshletsRead;
    FGResource m_cullCounterRead;
    FGResource m_instanceDataRead;
};

METALLIC_REGISTER_PASS(VisibilityPass);
