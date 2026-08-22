#pragma once

#include "Runtime/Render/RayTracing/SceneAccelerationStructure.h"
#include "Runtime/Render/NeuralTextureResources.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

inline constexpr uint32_t kScenePathTraceMaxMaterialTextures = 256;

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
        std::string& log);
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

    void clear();
    bool valid() const;
    uint64_t revision() const;

    const scene::Bounds& bounds() const;
    SceneAccelerationStructureBuilder& accelerationStructure();
    const SceneAccelerationStructureBuilder& accelerationStructure() const;
    Buffer* vertexBuffer() const;
    Buffer* indexBuffer() const;
    Buffer* primitiveBuffer() const;
    Buffer* instanceBuffer() const;
    Buffer* materialBuffer() const;
    const std::array<TextureView*, kScenePathTraceMaxMaterialTextures>& materialTextureViews() const;
    uint32_t materialTextureCount() const;
    const NeuralTextureResources& neuralTextures() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
