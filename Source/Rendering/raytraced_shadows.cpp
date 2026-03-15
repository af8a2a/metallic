#include "raytraced_shadows.h"
#include "rhi_resource_utils.h"

#ifdef __APPLE__

#include "mesh_loader.h"
#include "rhi_raytracing_utils.h"
#include "rhi_resource_utils.h"
#include "rhi_shader_utils.h"
#include "scene_graph.h"

#include <algorithm>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

namespace {

RhiRayTracingInstanceDesc makeInstanceDesc(const float4x4& transform, uint32_t blasIndex) {
    RhiRayTracingInstanceDesc instance{};
    instance.transform[0] = transform[0].x;
    instance.transform[1] = transform[0].y;
    instance.transform[2] = transform[0].z;
    instance.transform[3] = transform[1].x;
    instance.transform[4] = transform[1].y;
    instance.transform[5] = transform[1].z;
    instance.transform[6] = transform[2].x;
    instance.transform[7] = transform[2].y;
    instance.transform[8] = transform[2].z;
    instance.transform[9] = transform[3].x;
    instance.transform[10] = transform[3].y;
    instance.transform[11] = transform[3].z;
    instance.accelerationStructureIndex = blasIndex;
    instance.mask = 0xFF;
    instance.opaque = true;
    return instance;
}

std::string loadTextFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {};
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

} // namespace

void RaytracedShadowResources::release() {
    for (auto& blas : blasArray) {
        rhiReleaseHandle(blas);
    }
    blasArray.clear();
    referencedBlas.clear();

    rhiReleaseHandle(tlas);
    rhiReleaseHandle(instanceDescriptorBuffer);
    rhiReleaseHandle(scratchBuffer);
    rhiReleaseHandle(pipeline);
    rhiReleaseHandle(library);
    instanceCount = 0;
}

bool buildAccelerationStructures(const RhiDevice& device,
                                 const RhiCommandQueue& commandQueue,
                                 const LoadedMesh& mesh,
                                 const SceneGraph& sceneGraph,
                                 RaytracedShadowResources& out) {
    out.release();
    out.blasArray.resize(mesh.meshRanges.size());

    for (size_t meshIndex = 0; meshIndex < mesh.meshRanges.size(); ++meshIndex) {
        const auto& range = mesh.meshRanges[meshIndex];
        if (range.groupCount == 0) {
            continue;
        }

        std::vector<RhiRayTracingGeometryRange> geometryRanges;
        geometryRanges.reserve(range.groupCount);
        for (uint32_t groupIndex = 0; groupIndex < range.groupCount; ++groupIndex) {
            const auto& group = mesh.primitiveGroups[range.firstGroup + groupIndex];
            geometryRanges.push_back({group.indexOffset, group.indexCount});
        }

        std::string errorMessage;
        RhiAccelerationStructureHandle blas;
        if (!rhiBuildBottomLevelAccelerationStructure(device,
                                                      commandQueue,
                                                      mesh.positionBuffer,
                                                      static_cast<uint32_t>(sizeof(float) * 3),
                                                      mesh.indexBuffer,
                                                      geometryRanges.data(),
                                                      static_cast<uint32_t>(geometryRanges.size()),
                                                      blas,
                                                      errorMessage)) {
            spdlog::error("Failed to build BLAS for mesh {}: {}", meshIndex, errorMessage);
            out.release();
            return false;
        }

        out.blasArray[meshIndex] = blas;
    }

    std::vector<RhiRayTracingInstanceDesc> instances;
    out.referencedBlas.clear();

    for (const auto& node : sceneGraph.nodes) {
        if (node.meshIndex < 0 || !sceneGraph.isNodeVisible(node.id)) {
            continue;
        }

        const uint32_t meshIndex = static_cast<uint32_t>(node.meshIndex);
        if (meshIndex >= out.blasArray.size() || !out.blasArray[meshIndex].nativeHandle()) {
            continue;
        }

        uint32_t blasIndex = UINT32_MAX;
        for (uint32_t index = 0; index < out.referencedBlas.size(); ++index) {
            if (out.referencedBlas[index].nativeHandle() == out.blasArray[meshIndex].nativeHandle()) {
                blasIndex = index;
                break;
            }
        }

        if (blasIndex == UINT32_MAX) {
            blasIndex = static_cast<uint32_t>(out.referencedBlas.size());
            out.referencedBlas.push_back(out.blasArray[meshIndex]);
        }

        instances.push_back(makeInstanceDesc(node.transform.worldMatrix, blasIndex));
    }

    if (instances.empty()) {
        spdlog::error("No mesh instances found for TLAS");
        out.release();
        return false;
    }

    out.instanceCount = static_cast<uint32_t>(instances.size());
    std::vector<const RhiAccelerationStructure*> referencedBlasViews;
    referencedBlasViews.reserve(out.referencedBlas.size());
    for (const auto& blas : out.referencedBlas) {
        referencedBlasViews.push_back(&blas);
    }
    std::string errorMessage;
    if (!rhiBuildTopLevelAccelerationStructure(device,
                                               commandQueue,
                                               referencedBlasViews.data(),
                                               static_cast<uint32_t>(referencedBlasViews.size()),
                                               instances.data(),
                                               out.instanceCount,
                                               out.tlas,
                                               out.instanceDescriptorBuffer,
                                               out.scratchBuffer,
                                               errorMessage)) {
        spdlog::error("Failed to build TLAS: {}", errorMessage);
        out.release();
        return false;
    }

    spdlog::info("Built TLAS with {} instances, {} unique BLAS",
                 instances.size(),
                 out.referencedBlas.size());
    return true;
}

void updateTLAS(const RhiNativeCommandBuffer& commandBuffer,
                const SceneGraph& sceneGraph,
                RaytracedShadowResources& res) {
    if (!res.tlas.nativeHandle() ||
        !res.instanceDescriptorBuffer.nativeHandle() ||
        !res.scratchBuffer.nativeHandle() ||
        res.instanceCount == 0) {
        return;
    }

    std::vector<RhiRayTracingInstanceDesc> instances;
    instances.reserve(res.instanceCount);

    for (const auto& node : sceneGraph.nodes) {
        if (node.meshIndex < 0 || !sceneGraph.isNodeVisible(node.id)) {
            continue;
        }

        const uint32_t meshIndex = static_cast<uint32_t>(node.meshIndex);
        if (meshIndex >= res.blasArray.size() || !res.blasArray[meshIndex].nativeHandle()) {
            continue;
        }

        uint32_t blasIndex = UINT32_MAX;
        for (uint32_t index = 0; index < res.referencedBlas.size(); ++index) {
            if (res.referencedBlas[index].nativeHandle() == res.blasArray[meshIndex].nativeHandle()) {
                blasIndex = index;
                break;
            }
        }

        if (blasIndex == UINT32_MAX) {
            continue;
        }

        instances.push_back(makeInstanceDesc(node.transform.worldMatrix, blasIndex));
        if (instances.size() >= res.instanceCount) {
            break;
        }
    }

    if (instances.empty()) {
        return;
    }

    std::vector<const RhiAccelerationStructure*> referencedBlasViews;
    referencedBlasViews.reserve(res.referencedBlas.size());
    for (const auto& blas : res.referencedBlas) {
        referencedBlasViews.push_back(&blas);
    }
    std::string errorMessage;
    if (!rhiUpdateTopLevelAccelerationStructure(commandBuffer,
                                                referencedBlasViews.data(),
                                                static_cast<uint32_t>(referencedBlasViews.size()),
                                                instances.data(),
                                                static_cast<uint32_t>(instances.size()),
                                                res.tlas,
                                                res.instanceDescriptorBuffer,
                                                res.scratchBuffer,
                                                errorMessage)) {
        spdlog::error("Failed to update TLAS: {}", errorMessage);
    } else if (instances.size() < res.instanceCount) {
        spdlog::debug("TLAS updated with {} / {} instances", instances.size(), res.instanceCount);
    }
}

bool createShadowPipeline(const RhiDevice& device,
                          RaytracedShadowResources& out,
                          const char* shaderBasePath) {
    std::string shaderPath = "Shaders/Raytracing/raytraced_shadow.metal";
    if (shaderBasePath) {
        shaderPath = std::string(shaderBasePath) + "/" + shaderPath;
    }

    spdlog::info("Loading shader: {}", shaderPath);
    std::string metalSource = loadTextFile(shaderPath);
    if (metalSource.empty()) {
        spdlog::error("Failed to open {}", shaderPath);
        return false;
    }

    RhiShaderLibrarySourceDesc libraryDesc;
    libraryDesc.languageVersion = 31;

    std::string errorMessage;
    RhiShaderLibraryHandle newLibrary = rhiCreateShaderLibraryFromSource(device, metalSource, libraryDesc, errorMessage);
    if (!newLibrary.nativeHandle()) {
        spdlog::error("Failed to compile shadow ray shader: {}", errorMessage);
        return false;
    }

    RhiComputePipelineHandle newPipeline = rhiCreateComputePipelineFromLibrary(device,
                                                                               newLibrary,
                                                                               "shadowRayMain",
                                                                               errorMessage);
    if (!newPipeline.nativeHandle()) {
        spdlog::error("Failed to create shadow ray pipeline: {}", errorMessage);
        rhiReleaseHandle(newLibrary);
        return false;
    }

    rhiReleaseHandle(out.pipeline);
    rhiReleaseHandle(out.library);
    out.library = newLibrary;
    out.pipeline = newPipeline;

    spdlog::info("Shadow ray pipeline created");
    return true;
}

bool reloadShadowPipeline(const RhiDevice& device,
                          RaytracedShadowResources& res,
                          const char* shaderBasePath) {
    std::string shaderPath = "Shaders/Raytracing/raytraced_shadow.metal";
    if (shaderBasePath) {
        shaderPath = std::string(shaderBasePath) + "/" + shaderPath;
    }

    spdlog::info("Reloading shader: {}", shaderPath);
    std::string metalSource = loadTextFile(shaderPath);
    if (metalSource.empty()) {
        spdlog::error("Hot-reload: Failed to open {}", shaderPath);
        return false;
    }

    RhiShaderLibrarySourceDesc libraryDesc;
    libraryDesc.languageVersion = 31;

    std::string errorMessage;
    RhiShaderLibraryHandle newLibrary = rhiCreateShaderLibraryFromSource(device, metalSource, libraryDesc, errorMessage);
    if (!newLibrary.nativeHandle()) {
        spdlog::error("Hot-reload: Failed to compile shadow shader: {}", errorMessage);
        return false;
    }

    RhiComputePipelineHandle newPipeline = rhiCreateComputePipelineFromLibrary(device,
                                                                               newLibrary,
                                                                               "shadowRayMain",
                                                                               errorMessage);
    if (!newPipeline.nativeHandle()) {
        spdlog::error("Hot-reload: Failed to create shadow pipeline: {}", errorMessage);
        rhiReleaseHandle(newLibrary);
        return false;
    }

    rhiReleaseHandle(res.pipeline);
    rhiReleaseHandle(res.library);
    res.pipeline = newPipeline;
    res.library = newLibrary;

    spdlog::info("Hot-reload: Shadow ray pipeline reloaded");
    return true;
}

#else

void RaytracedShadowResources::release() {
    for (auto& blas : blasArray) {
        rhiReleaseHandle(blas);
    }
    blasArray.clear();
    referencedBlas.clear();
    rhiReleaseHandle(tlas);
    rhiReleaseHandle(instanceDescriptorBuffer);
    rhiReleaseHandle(scratchBuffer);
    rhiReleaseHandle(pipeline);
    rhiReleaseHandle(library);
    instanceCount = 0;
}

bool buildAccelerationStructures(const RhiDevice&,
                                 const RhiCommandQueue&,
                                 const LoadedMesh&,
                                 const SceneGraph&,
                                 RaytracedShadowResources& out) {
    out.release();
    return false;
}

void updateTLAS(const RhiNativeCommandBuffer&,
                const SceneGraph&,
                RaytracedShadowResources&) {
}

bool createShadowPipeline(const RhiDevice&,
                          RaytracedShadowResources& out,
                          const char*) {
    out.release();
    return false;
}

bool reloadShadowPipeline(const RhiDevice&,
                          RaytracedShadowResources&,
                          const char*) {
    return false;
}

#endif // __APPLE__
