#pragma once

#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <span>
#include <string>
#include <memory>
#include <unordered_map>
#include <utility>

namespace metallic::render {

struct GPUSceneSourceOverrideToken {
    uint64_t value = 0;

    bool valid() const { return value != 0; }
    explicit operator bool() const { return valid(); }
    auto operator<=>(const GPUSceneSourceOverrideToken&) const = default;
};

struct GPUSceneGpuUploadStats {
    uint64_t fullUploadCount = 0;
    uint64_t instanceUploadCount = 0;
    uint64_t uploadedByteCount = 0;
    uint32_t drawSetGeneration = 0;
    uint64_t drawSetRevision = 0;
};

struct GPUSceneCullRecordDesc {
    GPUSceneCullPhase phase = GPUSceneCullPhase::Early;
    BindlessHeap* bindlessHeap = nullptr;
    ComputePipeline* resetPipeline = nullptr;
    ComputePipeline* instanceCullPipeline = nullptr;
    ComputePipeline* compactPipeline = nullptr;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t instanceGroupCountX = 0;
    uint32_t meshletGroupCountX = 0;
};

struct GPUSceneInstanceCullRecordDesc {
    GPUSceneCullPhase phase = GPUSceneCullPhase::Early;
    BindlessHeap* bindlessHeap = nullptr;
    ComputePipeline* resetPipeline = nullptr;
    ComputePipeline* instanceCullPipeline = nullptr;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t instanceGroupCountX = 0;
};

struct GPUSceneComputeDispatchDesc {
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t groupCountX = 0;
    uint32_t groupCountY = 0;
    uint32_t groupCountZ = 1;
};

struct GPUSceneHzbRecordDesc {
    BindlessHeap* bindlessHeap = nullptr;
    ComputePipeline* pipeline = nullptr;
    std::span<const GPUSceneComputeDispatchDesc> dispatches;
};

// Non-owning snapshot of one {View, frame slot} GPU allocation. Buffer and
// BufferView addresses remain stable until ensureViewGpuResources replaces
// the allocation or destroyView retires it.
struct GPUSceneViewGpuResourcesView {
    GPUSceneViewId sourceView;
    GPUSceneViewDesc desc;
    uint32_t frameSlot = 0;
    GPUSceneBufferView instanceVisibilityStates;
    GPUSceneBufferView visibleInstanceIds;
    GPUSceneBufferView visibleInstanceCounter;
    std::array<GPUSceneCullPhaseGpuView, kGPUSceneCullPhaseCount> phases;
    std::array<GPUSceneBufferView, 2> hzbHistory;
    uint64_t allocationId = 0;
    bool frameSlotInitialized = false;
    bool hzbInitialized = false;
};

class GPUSceneSubsystem final : public IRenderSubsystem {
public:
    struct Desc {};

    static constexpr RenderSubsystemId kSubsystemId = "render.gpu-scene";

    Result initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    void onWorldChanged(RenderWorld* world) override;
    Result beginFrame(
        const RenderSubsystemFrameContext& context,
        RenderChangeBits& changes,
        std::string& log) override;
    Result recordPreGraph(
        const RenderSubsystemFrameContext& context,
        std::string& log) override;
    void shutdown() override;

    GPUScene& scene() { return scene_; }
    const GPUScene& scene() const { return scene_; }
    const GPUSceneDrawSet& drawSet() const { return scene_.drawSet(); }
    std::span<const GPUSceneGeometryRecord> geometries() const { return scene_.geometries(); }
    std::span<const GPUSceneMaterialRecord> materials() const { return scene_.materials(); }
    std::span<const GPUSceneInstanceRecord> instances() const { return scene_.instances(); }
    const GPUSceneStats& stats() const { return scene_.stats(); }

    const GPUSceneGeometryRecord* geometry(GPUSceneGeometryId id) const
    {
        return scene_.geometry(id);
    }
    const GPUSceneMaterialRecord* material(GPUSceneMaterialId id) const
    {
        return scene_.material(id);
    }
    const GPUSceneInstanceRecord* instance(GPUSceneInstanceId id) const
    {
        return scene_.instance(id);
    }
    GPUSceneInstanceId instanceForRenderNode(uint32_t renderNodeIndex) const
    {
        return scene_.instanceForRenderNode(renderNodeIndex);
    }

    uint64_t currentFrameIndex() const { return currentFrameIndex_; }
    uint32_t currentFrameSlot() const { return currentFrameSlot_; }
    uint32_t frameSlotCount() const { return frameSlotCount_; }

    Result acquireSourceOverride(
        const scene::Scene* scene,
        GPUSceneSourceOverrideToken& token,
        std::string& log);
    bool releaseSourceOverride(GPUSceneSourceOverrideToken token);
    void setSourceOverride(const scene::Scene* scene);
    bool clearSourceOverride(const scene::Scene* scene);
    const scene::Scene* sourceOverride() const;

    GPUSceneViewId createView(const GPUSceneViewDesc& desc = {});
    Result createView(
        const GPUSceneViewDesc& desc,
        GPUSceneViewId& view,
        std::string& log);
    bool destroyView(GPUSceneViewId view);
    Result ensureViewGpuResources(
        GPUSceneViewId view,
        const GPUSceneViewDesc& desc,
        std::string& log);
    bool viewGpuResources(
        GPUSceneViewId view,
        uint32_t frameSlot,
        GPUSceneViewGpuResourcesView& resources) const;
    Result recordInitialize(
        CommandBuffer& commandBuffer,
        GPUSceneViewId view,
        uint32_t frameSlot,
        std::string& log);
    Result publishViewGpuResources(
        GPUSceneViewId view,
        uint32_t frameSlot,
        uint32_t hzbWriteIndex,
        std::string& log);
    bool prepareView(
        GPUSceneViewId view,
        const GPUScene::VisibilityPredicate& predicate = {})
    {
        return scene_.prepareView(view, currentFrameSlot_, predicate);
    }
    bool prepareView(
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUScene::VisibilityPredicate& predicate = {})
    {
        return scene_.prepareView(view, frameSlot, predicate);
    }
    bool prepareView(
        GPUSceneViewId view,
        const GPUSceneViewPrepareInfo& info,
        const GPUScene::VisibilityPredicate& predicate = {})
    {
        return scene_.prepareView(view, currentFrameSlot_, info, predicate);
    }
    bool prepareView(
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUSceneViewPrepareInfo& info,
        const GPUScene::VisibilityPredicate& predicate = {})
    {
        return scene_.prepareView(view, frameSlot, info, predicate);
    }
    bool markViewHzbValid(
        GPUSceneViewId view,
        uint32_t frameSlot,
        bool valid = true)
    {
        return scene_.markViewHzbValid(view, frameSlot, valid);
    }
    const GPUSceneVisibleDrawSet* visibleDrawSet(GPUSceneViewId view) const
    {
        return scene_.visibleDrawSet(view, currentFrameSlot_);
    }
    const GPUSceneVisibleDrawSet* visibleDrawSet(
        GPUSceneViewId view,
        uint32_t frameSlot) const
    {
        return scene_.visibleDrawSet(view, frameSlot);
    }
    bool setVisibleGpuResources(
        GPUSceneViewId view,
        uint32_t frameSlot,
        GPUSceneVisibleGpuResources resources)
    {
        return scene_.setVisibleGpuResources(view, frameSlot, std::move(resources));
    }

    const GPUSceneGlobalBufferViews& globalBufferViews() const
    {
        return scene_.globalBufferViews();
    }
    bool setGlobalBufferViews(GPUSceneGlobalBufferViews views)
    {
        return scene_.setGlobalBufferViews(std::move(views));
    }

    const GPUSceneGpuUploadStats& gpuUploadStats() const { return gpuUploadStats_; }
    const GPUSceneRasterDrawLayout& rasterDrawLayout() const
    {
        return rasterDrawLayout_;
    }

    Result createBindings(
        BindlessHeap& heap,
        GPUSceneConsumerBindings& bindings,
        std::string& log) const;
    void releaseBindings(BindlessHeap& heap, GPUSceneConsumerBindings& bindings) const;

    Result recordCull(
        CommandBuffer& commandBuffer,
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUSceneCullRecordDesc& desc,
        std::string& log);
    Result recordInstanceCull(
        CommandBuffer& commandBuffer,
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUSceneInstanceCullRecordDesc& desc,
        std::string& log);
    Result recordBuildHzb(
        CommandBuffer& commandBuffer,
        GPUSceneViewId view,
        uint32_t frameSlot,
        const GPUSceneHzbRecordDesc& desc,
        std::string& log);

private:
    struct GpuBufferResource;
    struct GpuResources;
    struct UploadResources;
    struct ViewGpuResources;

    enum class PendingUpload : uint8_t {
        None,
        Instances,
        Full,
    };

    const scene::Scene* effectiveSourceOverride() const;
    void sourceOverrideChanged(const scene::Scene* previousOverride);
    void requestUpload(PendingUpload upload);
    Result uploadFullScene(
        const RenderSubsystemFrameContext& context,
        std::string& log);
    Result uploadInstances(
        const RenderSubsystemFrameContext& context,
        std::string& log);
    static uint64_t viewResourceKey(GPUSceneViewId view);
    void retireViewGpuResources(std::shared_ptr<ViewGpuResources> resources);

    Device* device_ = nullptr;
    RenderSubsystemHost* host_ = nullptr;
    RenderWorld* world_ = nullptr;
    const scene::Scene* sourceScene_ = nullptr;
    const scene::Scene* sourceOverride_ = nullptr;
    const scene::Scene* leasedSourceOverride_ = nullptr;
    std::unordered_map<uint64_t, const scene::Scene*> sourceOverrideLeases_;
    GPUScene scene_;
    uint64_t currentFrameIndex_ = 0;
    uint32_t currentFrameSlot_ = 0;
    uint32_t frameSlotCount_ = 1;
    uint64_t sourceOverrideRevision_ = 0;
    uint64_t nextSourceOverrideToken_ = 1;
    std::shared_ptr<GpuResources> gpuResources_;
    std::unordered_map<uint64_t, std::shared_ptr<ViewGpuResources>> viewGpuResources_;
    uint64_t nextViewGpuResourceAllocationId_ = 1;
    GPUSceneGpuUploadStats gpuUploadStats_;
    GPUSceneRasterDrawLayout rasterDrawLayout_;
    PendingUpload pendingUpload_ = PendingUpload::Full;
    bool sourceDirty_ = true;
};

} // namespace metallic::render
