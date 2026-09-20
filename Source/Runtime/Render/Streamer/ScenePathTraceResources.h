#pragma once

#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include "Runtime/Render/NeuralTextureResources.h"
#include "Runtime/Render/SceneShadingVertex.h"
#include "Runtime/Render/Streamer/Ktx2Texture.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/Scene.h"

#include "Runtime/Render/Profiling/CpuProfile.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

struct SceneTextureStats {
    TextureUploadProfile lastUpload;
    uint32_t logicalTextureCount = 0, ktxImageCount = 0, mimeMismatchCount = 0;
    uint32_t residentImageCount = 0, selectedMaxDimension = 0;
    uint32_t maskImageCount = 0, maskMaxDimension = 0;
    uint64_t budgetBytes = 0, plannedPayloadBytes = 0, plannedAllocationBytes = 0;
    uint64_t residentPayloadBytes = 0, residentAllocationBytes = 0, peakStagingBytes = 0;
    uint64_t configuredBudgetBytes = 0, sharedAvailableBytes = 0;
    uint64_t plannedHeapOverheadBytes = 0;
    bool streamingEnabled = false;
    uint32_t refinedImages = 0, requestedImages = 0, pendingImages = 0;
    uint64_t upgrades = 0, downgrades = 0, budgetDeferrals = 0, feedbackFrames = 0;
    uint64_t pendingAllocationBytes = 0, retiredAllocationBytes = 0, peakLiveAllocationBytes = 0;
    uint64_t streamingUploadBytes = 0, maxRequestLatencyFrames = 0;
};

struct SceneTextureLoadTiming {
    std::string path;
    double totalMs = 0, openMs = 0, readMs = 0, decodeMs = 0;
    double imageCreateMs = 0, stagingMs = 0, copyMs = 0, flushMs = 0;
};

struct SceneUploadStats {
    uint64_t submittedBatches = 0;
    uint64_t completedBatches = 0;
    uint64_t submittedBytes = 0;
    uint32_t inFlightBatches = 0;
    uint32_t peakInFlightBatches = 0;
    Ktx2ReadStats ktx;
    Ktx2PrefetchStats prefetch;
    double decodeWaitMs = 0;
    double textureHeaderMs = 0, texturePlanMs = 0, textureBuildMs = 0;
    double imageCreateMs = 0, stagingMs = 0, stagingCopyMs = 0, stagingFlushMs = 0;
    double commandSetupMs = 0, recordMs = 0, copySubmitMs = 0, acquireSubmitMs = 0;
    double backpressureMs = 0, finalWaitMs = 0, textureWallMs = 0;
    // Host-observed latency from successful submit to the first completion poll.
    // Includes scheduling/polling delay; these are NOT GPU execution timestamps.
    double copyCompletionObservedMs = 0, acquireCompletionObservedMs = 0;
    double maxCopyCompletionObservedMs = 0, maxAcquireCompletionObservedMs = 0;
    uint64_t copyCompletionSamples = 0, acquireCompletionSamples = 0;
    std::vector<SceneTextureLoadTiming> slowestTextures;
};

class ScenePathTraceResources final {
public:
    ScenePathTraceResources();
    ~ScenePathTraceResources();

    ScenePathTraceResources(ScenePathTraceResources&&) noexcept;
    ScenePathTraceResources& operator=(ScenePathTraceResources&&) noexcept;

    ScenePathTraceResources(const ScenePathTraceResources&) = default;
    ScenePathTraceResources& operator=(const ScenePathTraceResources&) = default;

    Result prepare(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        std::string& log);
    Result beginPrepareAsync(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene& runtimeScene,
        std::string& log,
        bool materialsOnly = false);
    Result pumpPrepareAsync(
        double budgetMilliseconds,
        bool& complete,
        scene::SceneLoadProgress& progress,
        std::string& log);
    bool preparing() const;
    Result syncRuntimeScene(const scene::Scene* runtimeScene, std::string& log);
    Result uploadMaterialTextures(CommandBuffer& commandBuffer);
    // Called once by the deferred consumer; never waits for feedback or decode.
    Result beginTextureStreaming(CommandBuffer& commands, uint64_t frameIndex, Buffer*& feedback, CpuProfileRecorder* profiler = nullptr);
    bool textureUploadsReady() const;
    bool gpuWorkComplete();
    SceneUploadStats uploadStats() const;
    SceneTextureStats textureStats() const;
    std::span<const uint32_t> logicalTextureIndices() const;

    void clear();
    bool valid() const;
    uint64_t revision() const;

    const scene::Bounds& bounds() const;
    SceneAccelerationStructureBuilder& accelerationStructure();
    const SceneAccelerationStructureBuilder& accelerationStructure() const;
    Buffer* shadingVertexBuffer() const;
    // Null when positions are supplied by ray-tracing position fetch.
    Buffer* fallbackPositionBuffer() const;
    Buffer* indexBuffer() const;
    Buffer* primitiveBuffer() const;
    Buffer* instanceBuffer() const;
    Buffer* materialBuffer() const;
    const std::vector<TextureView*>& materialTextureViews() const;
    uint32_t materialTextureCount() const;
    // Source-image indexed KTX tail selection; non-KTX entries are zero.
    const std::vector<uint32_t>& materialTextureFirstMips() const;
    const NeuralTextureResources& neuralTextures() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
