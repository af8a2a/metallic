#pragma once

#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include "Runtime/Render/NeuralTextureResources.h"
#include "Runtime/Render/SceneShadingVertex.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

struct SceneTextureStats {
    uint32_t logicalTextureCount = 0, ktxImageCount = 0, mimeMismatchCount = 0;
    uint32_t residentImageCount = 0, selectedMaxDimension = 0;
    uint64_t budgetBytes = 0, plannedPayloadBytes = 0, plannedAllocationBytes = 0;
    uint64_t residentPayloadBytes = 0, residentAllocationBytes = 0, peakStagingBytes = 0;
};

struct SceneUploadStats {
    uint64_t submittedBatches = 0;
    uint64_t completedBatches = 0;
    uint64_t submittedBytes = 0;
    uint32_t inFlightBatches = 0;
    uint32_t peakInFlightBatches = 0;
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
    const NeuralTextureResources& neuralTextures() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
