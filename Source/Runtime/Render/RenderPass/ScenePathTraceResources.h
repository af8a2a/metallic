#pragma once

#include "Runtime/Render/GAPI/SceneRtx.h"
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

    ScenePathTraceResources(const ScenePathTraceResources&) = delete;
    ScenePathTraceResources& operator=(const ScenePathTraceResources&) = delete;

    Result prepare(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        std::string& log);
    Result uploadMaterialTextures(CommandBuffer& commandBuffer);
    Result uploadEnvironmentTexture(CommandBuffer& commandBuffer);

    void clear();
    bool valid() const;
    uint64_t revision() const;

    const scene::Bounds& bounds() const;
    SceneRtxBuilder& accelerationStructure();
    const SceneRtxBuilder& accelerationStructure() const;
    Buffer* vertexBuffer() const;
    Buffer* indexBuffer() const;
    Buffer* primitiveBuffer() const;
    Buffer* instanceBuffer() const;
    Buffer* materialBuffer() const;
    const std::array<TextureView*, kScenePathTraceMaxMaterialTextures>& materialTextureViews() const;
    uint32_t materialTextureCount() const;
    TextureView* environmentTextureView() const;
    bool environmentMapAvailable() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
