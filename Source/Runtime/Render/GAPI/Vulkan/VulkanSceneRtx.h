#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/Scene.h"

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

enum class SceneRtxBuildState : uint8_t {
    Idle,
    Building,
    Ready,
    Failed,
};

struct SceneClusterRtxStats {
    uint32_t clasCount = 0;
    uint32_t clusterBlasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t clusterTriangleCount = 0;
    uint64_t clusterVertexCount = 0;
    uint64_t clusterIndexBytes = 0;
    uint64_t selectedClusterReferenceCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t clasBytes = 0;
    uint64_t clusterBlasBytes = 0;
    uint64_t tlasBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
};

struct ScenePartitionedRtxStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint32_t partitionCount = 0;
    uint32_t maxInstancesPerPartition = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t blasBytes = 0;
    uint64_t ptlasBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
    uint64_t operationBytes = 0;
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
    Result beginBuild(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    bool pollBuild();
    SceneRtxBuildState buildState() const;
    Result updateInstanceTransforms(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        std::string& log);
    void clear();

    bool valid() const;
    const SceneRtxStats& stats() const;

private:
    friend class SceneRayQueryProgram;

    struct Impl;
    Result buildInternal(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        bool waitForCompletion,
        std::string& log);
    std::unique_ptr<Impl> impl_;
};

class SceneClusterRtxBuilder {
public:
    SceneClusterRtxBuilder();
    ~SceneClusterRtxBuilder();

    SceneClusterRtxBuilder(SceneClusterRtxBuilder&&) noexcept;
    SceneClusterRtxBuilder& operator=(SceneClusterRtxBuilder&&) noexcept;

    SceneClusterRtxBuilder(const SceneClusterRtxBuilder&) = delete;
    SceneClusterRtxBuilder& operator=(const SceneClusterRtxBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    const SceneClusterRtxStats& stats() const;

private:
    friend class SceneRayQueryProgram;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ScenePartitionedRtxBuilder {
public:
    ScenePartitionedRtxBuilder();
    ~ScenePartitionedRtxBuilder();

    ScenePartitionedRtxBuilder(ScenePartitionedRtxBuilder&&) noexcept;
    ScenePartitionedRtxBuilder& operator=(ScenePartitionedRtxBuilder&&) noexcept;

    ScenePartitionedRtxBuilder(const ScenePartitionedRtxBuilder&) = delete;
    ScenePartitionedRtxBuilder& operator=(const ScenePartitionedRtxBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    const ScenePartitionedRtxStats& stats() const;

private:
    friend class SceneRayQueryProgram;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class SceneRayQueryBindingKind : uint8_t {
    AccelerationStructure,
    PartitionedAccelerationStructure,
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
    // Multiple sets let a command buffer record several dispatches without
    // updating a descriptor set that is still referenced by an earlier dispatch.
    uint32_t descriptorSetCount = 1;
    // Most users execute ray-query shaders and therefore require both RTAS and
    // ray-query support. Pure compute users may opt out explicitly.
    bool requiresRayQuery = true;
};

struct SceneRayQueryDispatchBinding {
    uint32_t binding = 0;
    uint64_t accelerationStructureHandle = 0;
    SceneRtxBuilder* accelerationStructure = nullptr;
    SceneClusterRtxBuilder* clusterAccelerationStructure = nullptr;
    ScenePartitionedRtxBuilder* partitionedAccelerationStructure = nullptr;
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
    uint32_t descriptorSetIndex = 0;
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
