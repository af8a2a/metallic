#pragma once

#include "Runtime/Render/GAPI/rhi.h"
#include "Runtime/Scene/scene.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render::vulkan {

struct SceneRtxStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
};

class SceneRtxBuilder {
public:
    SceneRtxBuilder();
    ~SceneRtxBuilder();

    SceneRtxBuilder(SceneRtxBuilder&&) noexcept;
    SceneRtxBuilder& operator=(SceneRtxBuilder&&) noexcept;

    SceneRtxBuilder(const SceneRtxBuilder&) = delete;
    SceneRtxBuilder& operator=(const SceneRtxBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    const SceneRtxStats& stats() const;

private:
    friend class SceneRayQueryProgram;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class SceneRayQueryBindingKind : uint8_t {
    AccelerationStructure,
    StorageImage,
    StorageBuffer,
    SampledImage,
};

struct SceneRayQueryBindingDesc {
    uint32_t binding = 0;
    SceneRayQueryBindingKind kind = SceneRayQueryBindingKind::StorageBuffer;
    uint32_t descriptorCount = 1;
};

struct SceneRayQueryProgramDesc {
    const uint32_t* spirv = nullptr;
    uint64_t byteSize = 0;
    uint32_t pushConstantSize = 0;
    const SceneRayQueryBindingDesc* bindings = nullptr;
    uint32_t bindingCount = 0;
    const char* debugName = nullptr;
};

struct SceneRayQueryDispatchBinding {
    uint32_t binding = 0;
    SceneRtxBuilder* accelerationStructure = nullptr;
    TextureView* textureView = nullptr;
    TextureView* const* textureViews = nullptr;
    uint32_t textureViewCount = 0;
    Buffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
};

struct SceneRayQueryDispatchDesc {
    CommandBuffer* commandBuffer = nullptr;
    const SceneRayQueryDispatchBinding* bindings = nullptr;
    uint32_t bindingCount = 0;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
};

class SceneRayQueryProgram {
public:
    SceneRayQueryProgram();
    ~SceneRayQueryProgram();

    SceneRayQueryProgram(SceneRayQueryProgram&&) noexcept;
    SceneRayQueryProgram& operator=(SceneRayQueryProgram&&) noexcept;

    SceneRayQueryProgram(const SceneRayQueryProgram&) = delete;
    SceneRayQueryProgram& operator=(const SceneRayQueryProgram&) = delete;

    Result initialize(Device& device, const SceneRayQueryProgramDesc& desc, std::string& log);
    void clear();
    bool valid() const;
    Result dispatch(const SceneRayQueryDispatchDesc& desc);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
