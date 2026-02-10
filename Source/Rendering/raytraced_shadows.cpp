#include "raytraced_shadows.h"
#include "mesh_loader.h"
#include "scene_graph.h"

#include <Metal/Metal.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>

void RaytracedShadowResources::release() {
    for (auto* blas : blasArray) {
        if (blas) blas->release();
    }
    blasArray.clear();
    if (tlas) { tlas->release(); tlas = nullptr; }
    if (instanceDescriptorBuffer) { instanceDescriptorBuffer->release(); instanceDescriptorBuffer = nullptr; }
    if (scratchBuffer) { scratchBuffer->release(); scratchBuffer = nullptr; }
    if (pipeline) { pipeline->release(); pipeline = nullptr; }
    if (library) { library->release(); library = nullptr; }
}

bool buildAccelerationStructures(MTL::Device* device,
                                 MTL::CommandQueue* commandQueue,
                                 const LoadedMesh& mesh,
                                 const SceneGraph& sceneGraph,
                                 RaytracedShadowResources& out) {
    // --- Build one BLAS per glTF mesh ---
    out.blasArray.resize(mesh.meshRanges.size(), nullptr);

    for (size_t meshIdx = 0; meshIdx < mesh.meshRanges.size(); meshIdx++) {
        const auto& range = mesh.meshRanges[meshIdx];
        if (range.groupCount == 0) continue;

        // Create geometry descriptors for each primitive group
        std::vector<MTL::AccelerationStructureTriangleGeometryDescriptor*> geomDescs;
        geomDescs.reserve(range.groupCount);

        for (uint32_t g = 0; g < range.groupCount; g++) {
            const auto& group = mesh.primitiveGroups[range.firstGroup + g];
            auto* triGeom = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
            triGeom->setVertexBuffer(mesh.positionBuffer);
            triGeom->setVertexBufferOffset(0);
            triGeom->setVertexStride(sizeof(float) * 3);
            triGeom->setVertexFormat(MTL::AttributeFormatFloat3);
            triGeom->setIndexBuffer(mesh.indexBuffer);
            triGeom->setIndexBufferOffset(group.indexOffset * sizeof(uint32_t));
            triGeom->setIndexType(MTL::IndexTypeUInt32);
            triGeom->setTriangleCount(group.indexCount / 3);
            triGeom->setOpaque(true);
            geomDescs.push_back(triGeom);
        }

        auto* geomArray = NS::Array::array(
            reinterpret_cast<const NS::Object* const*>(geomDescs.data()),
            geomDescs.size());

        auto* primDesc = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
        primDesc->setGeometryDescriptors(geomArray);

        auto sizes = device->accelerationStructureSizes(primDesc);
        auto* blas = device->newAccelerationStructure(sizes.accelerationStructureSize);
        auto* scratch = device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);

        auto* cmdBuf = commandQueue->commandBuffer();
        auto* enc = cmdBuf->accelerationStructureCommandEncoder();
        enc->buildAccelerationStructure(blas, primDesc, scratch, 0);
        enc->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        scratch->release();
        primDesc->release();
        for (auto* gd : geomDescs) gd->release();

        out.blasArray[meshIdx] = blas;
    }

    // --- Build TLAS ---
    // Collect instances: one per scene node that has a mesh
    std::vector<MTL::AccelerationStructureInstanceDescriptor> instances;
    out.referencedBLAS.clear();

    for (const auto& node : sceneGraph.nodes) {
        if (node.meshIndex < 0) continue;
        if (!sceneGraph.isNodeVisible(node.id)) continue;
        uint32_t mi = static_cast<uint32_t>(node.meshIndex);
        if (mi >= out.blasArray.size() || !out.blasArray[mi]) continue;

        // Find or add this BLAS to the referenced list
        uint32_t blasIdx = UINT32_MAX;
        for (uint32_t i = 0; i < out.referencedBLAS.size(); i++) {
            if (out.referencedBLAS[i] == out.blasArray[mi]) {
                blasIdx = i;
                break;
            }
        }
        if (blasIdx == UINT32_MAX) {
            blasIdx = static_cast<uint32_t>(out.referencedBLAS.size());
            out.referencedBLAS.push_back(out.blasArray[mi]);
        }

        MTL::AccelerationStructureInstanceDescriptor inst = {};
        // Convert column-major float4x4 to PackedFloat4x3 (4 columns, 3 rows each)
        const float4x4& m = node.transform.worldMatrix;
        inst.transformationMatrix.columns[0] = {m[0].x, m[0].y, m[0].z};
        inst.transformationMatrix.columns[1] = {m[1].x, m[1].y, m[1].z};
        inst.transformationMatrix.columns[2] = {m[2].x, m[2].y, m[2].z};
        inst.transformationMatrix.columns[3] = {m[3].x, m[3].y, m[3].z};
        inst.options = MTL::AccelerationStructureInstanceOptionOpaque;
        inst.mask = 0xFF;
        inst.intersectionFunctionTableOffset = 0;
        inst.accelerationStructureIndex = blasIdx;
        instances.push_back(inst);
    }

    if (instances.empty()) {
        spdlog::error("No mesh instances found for TLAS");
        return false;
    }

    out.instanceCount = static_cast<uint32_t>(instances.size());
    out.instanceDescriptorBuffer = device->newBuffer(
        instances.data(),
        instances.size() * sizeof(MTL::AccelerationStructureInstanceDescriptor),
        MTL::ResourceStorageModeShared);

    auto* tlasDesc = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    tlasDesc->setInstanceCount(instances.size());
    tlasDesc->setInstanceDescriptorBuffer(out.instanceDescriptorBuffer);
    tlasDesc->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeDefault);

    auto* blasNSArray = NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(out.referencedBLAS.data()),
        out.referencedBLAS.size());
    tlasDesc->setInstancedAccelerationStructures(blasNSArray);

    auto tlasSizes = device->accelerationStructureSizes(tlasDesc);
    out.tlas = device->newAccelerationStructure(tlasSizes.accelerationStructureSize);
    out.scratchBuffer = device->newBuffer(tlasSizes.buildScratchBufferSize,
                                          MTL::ResourceStorageModePrivate);

    auto* cmdBuf = commandQueue->commandBuffer();
    auto* enc = cmdBuf->accelerationStructureCommandEncoder();
    enc->buildAccelerationStructure(out.tlas, tlasDesc, out.scratchBuffer, 0);
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    tlasDesc->release();

    spdlog::info("Built TLAS with {} instances, {} unique BLAS",
                 instances.size(), out.referencedBLAS.size());
    return true;
}

void updateTLAS(MTL::CommandBuffer* commandBuffer,
                const SceneGraph& sceneGraph,
                RaytracedShadowResources& res) {
    if (!res.tlas || !res.instanceDescriptorBuffer || res.instanceCount == 0)
        return;

    // Rewrite instance transforms from current scene graph world matrices
    auto* instDescs = static_cast<MTL::AccelerationStructureInstanceDescriptor*>(
        res.instanceDescriptorBuffer->contents());

    uint32_t idx = 0;
    for (const auto& node : sceneGraph.nodes) {
        if (node.meshIndex < 0) continue;
        if (!sceneGraph.isNodeVisible(node.id)) continue;
        uint32_t mi = static_cast<uint32_t>(node.meshIndex);
        if (mi >= res.blasArray.size() || !res.blasArray[mi]) continue;
        if (idx >= res.instanceCount) break;

        const float4x4& m = node.transform.worldMatrix;
        instDescs[idx].transformationMatrix.columns[0] = {m[0].x, m[0].y, m[0].z};
        instDescs[idx].transformationMatrix.columns[1] = {m[1].x, m[1].y, m[1].z};
        instDescs[idx].transformationMatrix.columns[2] = {m[2].x, m[2].y, m[2].z};
        instDescs[idx].transformationMatrix.columns[3] = {m[3].x, m[3].y, m[3].z};
        idx++;
    }

    // Rebuild TLAS with updated transforms
    auto* tlasDesc = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    tlasDesc->setInstanceCount(res.instanceCount);
    tlasDesc->setInstanceDescriptorBuffer(res.instanceDescriptorBuffer);
    tlasDesc->setInstanceDescriptorType(
        MTL::AccelerationStructureInstanceDescriptorTypeDefault);

    auto* blasNSArray = NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(res.referencedBLAS.data()),
        res.referencedBLAS.size());
    tlasDesc->setInstancedAccelerationStructures(blasNSArray);

    auto* enc = commandBuffer->accelerationStructureCommandEncoder();
    enc->buildAccelerationStructure(res.tlas, tlasDesc, res.scratchBuffer, 0);
    enc->endEncoding();

    tlasDesc->release();
}

bool createShadowPipeline(MTL::Device* device,
                          RaytracedShadowResources& out,
                          const char* shaderBasePath) {
    std::string shaderPath = "Shaders/Raytracing/raytraced_shadow.metal";
    if (shaderBasePath) {
        shaderPath = std::string(shaderBasePath) + "/" + shaderPath;
    }
    spdlog::info("Loading shader: {}", shaderPath);
    std::ifstream file(shaderPath);
    if (!file.is_open()) {
        spdlog::error("Failed to open {}", shaderPath);
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string metalSource = ss.str();

    NS::Error* error = nullptr;
    auto* sourceStr = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    compileOpts->setLanguageVersion(MTL::LanguageVersion3_1);

    out.library = device->newLibrary(sourceStr, compileOpts, &error);
    compileOpts->release();
    if (!out.library) {
        spdlog::error("Failed to compile shadow ray shader: {}",
                      error->localizedDescription()->utf8String());
        return false;
    }

    auto* fn = out.library->newFunction(
        NS::String::string("shadowRayMain", NS::UTF8StringEncoding));
    if (!fn) {
        spdlog::error("Failed to find shadowRayMain function");
        return false;
    }

    out.pipeline = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!out.pipeline) {
        spdlog::error("Failed to create shadow ray pipeline: {}",
                      error->localizedDescription()->utf8String());
        return false;
    }

    spdlog::info("Shadow ray pipeline created");
    return true;
}

bool reloadShadowPipeline(MTL::Device* device,
                           RaytracedShadowResources& res,
                           const char* shaderBasePath) {
    std::string shaderPath = "Shaders/Raytracing/raytraced_shadow.metal";
    if (shaderBasePath) {
        shaderPath = std::string(shaderBasePath) + "/" + shaderPath;
    }
    spdlog::info("Reloading shader: {}", shaderPath);
    std::ifstream file(shaderPath);
    if (!file.is_open()) {
        spdlog::error("Hot-reload: Failed to open {}", shaderPath);
        return false;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string metalSource = ss.str();

    NS::Error* error = nullptr;
    auto* sourceStr = NS::String::string(metalSource.c_str(), NS::UTF8StringEncoding);
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    compileOpts->setLanguageVersion(MTL::LanguageVersion3_1);

    auto* newLibrary = device->newLibrary(sourceStr, compileOpts, &error);
    compileOpts->release();
    if (!newLibrary) {
        spdlog::error("Hot-reload: Failed to compile shadow shader: {}",
                      error->localizedDescription()->utf8String());
        return false;
    }

    auto* fn = newLibrary->newFunction(
        NS::String::string("shadowRayMain", NS::UTF8StringEncoding));
    if (!fn) {
        spdlog::error("Hot-reload: Failed to find shadowRayMain function");
        newLibrary->release();
        return false;
    }

    auto* newPipeline = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!newPipeline) {
        spdlog::error("Hot-reload: Failed to create shadow pipeline: {}",
                      error->localizedDescription()->utf8String());
        newLibrary->release();
        return false;
    }

    // Success — swap old resources
    if (res.pipeline) res.pipeline->release();
    if (res.library) res.library->release();
    res.pipeline = newPipeline;
    res.library = newLibrary;
    spdlog::info("Hot-reload: Shadow ray pipeline reloaded");
    return true;
}
