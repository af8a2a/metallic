#include "meshlet_builder.h"
#include "mesh_loader.h"
#include "rhi_resource_utils.h"

#include <meshoptimizer.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <vector>

static constexpr size_t MAX_VERTICES  = 32;
static constexpr size_t MAX_TRIANGLES = 32;
static constexpr float  CONE_WEIGHT   = 0.5f;

namespace {

constexpr char kMeshletCacheMagic[8] = {'M', 'L', 'M', 'S', 'H', 'L', 'T', '1'};
constexpr uint32_t kMeshletCacheVersion = 2;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct MeshletCacheHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t maxVertices = 0;
    uint32_t maxTriangles = 0;
    float coneWeight = 0.0f;
    uint32_t reserved = 0;
    uint64_t meshSignature = 0;
    uint64_t meshletCount = 0;
    uint64_t meshletVertexCount = 0;
    uint64_t meshletTriangleByteCount = 0;
    uint64_t boundsCount = 0;
    uint64_t materialCount = 0;
    uint64_t meshletsPerGroupCount = 0;
};

static_assert(std::is_trivially_copyable_v<MeshletCacheHeader>);

void releaseMeshletHandles(MeshletData& meshlets) {
    rhiReleaseHandle(meshlets.meshletBuffer);
    rhiReleaseHandle(meshlets.meshletVertices);
    rhiReleaseHandle(meshlets.meshletTriangles);
    rhiReleaseHandle(meshlets.boundsBuffer);
    rhiReleaseHandle(meshlets.materialIDs);
    meshlets.meshletCount = 0;
}

void hashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

template <typename T>
void hashValue(uint64_t& hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    hashBytes(hash, &value, sizeof(T));
}

uint32_t expectedMeshletGroupCount(const LoadedMesh& mesh) {
    return mesh.primitiveGroups.empty()
        ? 1u
        : static_cast<uint32_t>(mesh.primitiveGroups.size());
}

uint64_t computeMeshSignature(const LoadedMesh& mesh) {
    uint64_t hash = kFnvOffsetBasis;
    hashValue(hash, mesh.vertexCount);
    hashValue(hash, mesh.indexCount);
    hashValue(hash, mesh.hasBakedRootScale);
    hashValue(hash, mesh.bakedRootScale);

    if (!mesh.cpuPositions.empty()) {
        hashBytes(hash, mesh.cpuPositions.data(), mesh.cpuPositions.size() * sizeof(float));
    }
    if (!mesh.cpuIndices.empty()) {
        hashBytes(hash, mesh.cpuIndices.data(), mesh.cpuIndices.size() * sizeof(uint32_t));
    }

    const uint32_t primitiveGroupCount = static_cast<uint32_t>(mesh.primitiveGroups.size());
    hashValue(hash, primitiveGroupCount);
    for (const auto& group : mesh.primitiveGroups) {
        hashValue(hash, group.indexOffset);
        hashValue(hash, group.indexCount);
        hashValue(hash, group.vertexOffset);
        hashValue(hash, group.vertexCount);
        hashValue(hash, group.materialIndex);
    }

    return hash;
}

std::string sanitizeCacheStem(std::string stem) {
    if (stem.empty()) {
        return "scene";
    }

    for (char& ch : stem) {
        const bool isAlphaNum =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        if (!isAlphaNum && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return stem;
}

std::filesystem::path makeCacheFilePath(const std::string& sourcePath,
                                        const std::string& cacheDirectory,
                                        uint64_t meshSignature) {
    std::ostringstream fileName;
    fileName << sanitizeCacheStem(std::filesystem::path(sourcePath).stem().string())
             << "_"
             << std::hex
             << std::setw(16)
             << std::setfill('0')
             << std::nouppercase
             << meshSignature
             << ".meshletcache";
    return std::filesystem::path(cacheDirectory) / fileName.str();
}

bool packRawTriangles(const std::vector<unsigned char>& rawTriangles,
                      std::vector<uint32_t>& outPackedTriangles) {
    if ((rawTriangles.size() % 3) != 0) {
        spdlog::error("Meshlet cache has invalid raw triangle byte count {}", rawTriangles.size());
        return false;
    }

    outPackedTriangles.clear();
    outPackedTriangles.reserve(rawTriangles.size() / 3);
    for (size_t i = 0; i + 2 < rawTriangles.size(); i += 3) {
        outPackedTriangles.push_back(uint32_t(rawTriangles[i + 0]) |
                                     (uint32_t(rawTriangles[i + 1]) << 8) |
                                     (uint32_t(rawTriangles[i + 2]) << 16));
    }
    return true;
}

bool validateMeshletPayload(const MeshletData& data, uint32_t expectedGroupCount) {
    if (data.cpuMeshlets.empty()) {
        spdlog::error("Meshlet payload is empty");
        return false;
    }
    if (data.cpuBounds.size() != data.cpuMeshlets.size()) {
        spdlog::error("Meshlet bounds count {} does not match meshlet count {}",
                      data.cpuBounds.size(),
                      data.cpuMeshlets.size());
        return false;
    }
    if (data.cpuMaterialIDs.size() != data.cpuMeshlets.size()) {
        spdlog::error("Meshlet material count {} does not match meshlet count {}",
                      data.cpuMaterialIDs.size(),
                      data.cpuMeshlets.size());
        return false;
    }
    if (data.meshletsPerGroup.size() != expectedGroupCount) {
        spdlog::error("Meshlet group count {} does not match expected {}",
                      data.meshletsPerGroup.size(),
                      expectedGroupCount);
        return false;
    }

    uint64_t meshletsFromGroups = 0;
    for (uint32_t groupMeshletCount : data.meshletsPerGroup) {
        meshletsFromGroups += groupMeshletCount;
    }
    if (meshletsFromGroups != data.cpuMeshlets.size()) {
        spdlog::error("Meshlet groups sum to {} meshlets but payload contains {}",
                      meshletsFromGroups,
                      data.cpuMeshlets.size());
        return false;
    }

    const size_t packedTriangleCount = data.cpuMeshletTriangles.size() / 3;
    for (const auto& meshlet : data.cpuMeshlets) {
        if (static_cast<size_t>(meshlet.vertex_offset) + meshlet.vertex_count > data.cpuMeshletVertices.size()) {
            spdlog::error("Meshlet vertex range [{}, {}) is out of bounds {}",
                          meshlet.vertex_offset,
                          static_cast<size_t>(meshlet.vertex_offset) + meshlet.vertex_count,
                          data.cpuMeshletVertices.size());
            return false;
        }
        if (static_cast<size_t>(meshlet.triangle_offset) + meshlet.triangle_count > packedTriangleCount) {
            spdlog::error("Meshlet triangle range [{}, {}) is out of bounds {}",
                          meshlet.triangle_offset,
                          static_cast<size_t>(meshlet.triangle_offset) + meshlet.triangle_count,
                          packedTriangleCount);
            return false;
        }
    }

    return true;
}

bool uploadMeshletBuffers(const RhiDevice& device, MeshletData& data) {
    std::vector<uint32_t> packedTriangles;
    if (!packRawTriangles(data.cpuMeshletTriangles, packedTriangles)) {
        return false;
    }

    const uint32_t expectedGroupCount = data.meshletsPerGroup.empty()
        ? 1u
        : static_cast<uint32_t>(data.meshletsPerGroup.size());
    if (!validateMeshletPayload(data, expectedGroupCount)) {
        return false;
    }

    data.meshletBuffer = rhiCreateSharedBuffer(
        device, data.cpuMeshlets.data(), data.cpuMeshlets.size() * sizeof(GPUMeshlet), "Meshlets");
    data.meshletVertices = rhiCreateSharedBuffer(
        device, data.cpuMeshletVertices.data(), data.cpuMeshletVertices.size() * sizeof(uint32_t), "Meshlet Vertices");
    data.meshletTriangles = rhiCreateSharedBuffer(
        device, packedTriangles.data(), packedTriangles.size() * sizeof(uint32_t), "Meshlet Triangles");
    data.boundsBuffer = rhiCreateSharedBuffer(
        device, data.cpuBounds.data(), data.cpuBounds.size() * sizeof(GPUMeshletBounds), "Meshlet Bounds");
    data.materialIDs = rhiCreateSharedBuffer(
        device, data.cpuMaterialIDs.data(), data.cpuMaterialIDs.size() * sizeof(uint32_t), "Meshlet Material IDs");

    if (!data.meshletBuffer.nativeHandle() ||
        !data.meshletVertices.nativeHandle() ||
        !data.meshletTriangles.nativeHandle() ||
        !data.boundsBuffer.nativeHandle() ||
        !data.materialIDs.nativeHandle()) {
        releaseMeshletHandles(data);
        spdlog::error("Failed to create GPU buffers for meshlet payload");
        return false;
    }

    data.meshletCount = static_cast<uint32_t>(data.cpuMeshlets.size());
    return true;
}

template <typename T>
bool writeVector(std::ofstream& file, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.empty()) {
        return true;
    }

    const size_t byteSize = values.size() * sizeof(T);
    if (byteSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(byteSize));
    return static_cast<bool>(file);
}

template <typename T>
bool readVector(std::ifstream& file, uint64_t count, std::vector<T>& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count == 0) {
        out.clear();
        return true;
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        return false;
    }

    out.resize(static_cast<size_t>(count));
    const size_t byteSize = out.size() * sizeof(T);
    if (byteSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(byteSize));
    return static_cast<bool>(file);
}

bool saveMeshletsToCache(const LoadedMesh& mesh,
                         const std::string& sourcePath,
                         const std::string& cacheDirectory,
                         const MeshletData& data) {
    const uint32_t expectedGroupCount = expectedMeshletGroupCount(mesh);
    if (!validateMeshletPayload(data, expectedGroupCount)) {
        spdlog::warn("Skipping meshlet cache write because payload validation failed");
        return false;
    }

    std::filesystem::path cacheDir(cacheDirectory);
    std::error_code createError;
    std::filesystem::create_directories(cacheDir, createError);
    if (createError) {
        spdlog::warn("Failed to create meshlet cache directory {}: {}",
                     cacheDir.string(),
                     createError.message());
        return false;
    }

    const uint64_t meshSignature = computeMeshSignature(mesh);
    const std::filesystem::path cachePath = makeCacheFilePath(sourcePath, cacheDirectory, meshSignature);

    MeshletCacheHeader header;
    std::memcpy(header.magic, kMeshletCacheMagic, sizeof(header.magic));
    header.version = kMeshletCacheVersion;
    header.maxVertices = static_cast<uint32_t>(MAX_VERTICES);
    header.maxTriangles = static_cast<uint32_t>(MAX_TRIANGLES);
    header.coneWeight = CONE_WEIGHT;
    header.meshSignature = meshSignature;
    header.meshletCount = data.cpuMeshlets.size();
    header.meshletVertexCount = data.cpuMeshletVertices.size();
    header.meshletTriangleByteCount = data.cpuMeshletTriangles.size();
    header.boundsCount = data.cpuBounds.size();
    header.materialCount = data.cpuMaterialIDs.size();
    header.meshletsPerGroupCount = data.meshletsPerGroup.size();

    std::ofstream file(cachePath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        spdlog::warn("Failed to open meshlet cache file for write: {}", cachePath.string());
        return false;
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    const bool ok = static_cast<bool>(file) &&
                    writeVector(file, data.meshletsPerGroup) &&
                    writeVector(file, data.cpuMeshlets) &&
                    writeVector(file, data.cpuMeshletVertices) &&
                    writeVector(file, data.cpuMeshletTriangles) &&
                    writeVector(file, data.cpuBounds) &&
                    writeVector(file, data.cpuMaterialIDs);
    if (!ok) {
        spdlog::warn("Failed to write meshlet cache file {}", cachePath.string());
        return false;
    }

    spdlog::info("Saved {} meshlets to cache {}", data.cpuMeshlets.size(), cachePath.string());
    return true;
}

bool loadMeshletsFromCache(const RhiDevice& device,
                           const LoadedMesh& mesh,
                           const std::string& sourcePath,
                           const std::string& cacheDirectory,
                           MeshletData& out) {
    const uint64_t meshSignature = computeMeshSignature(mesh);
    const std::filesystem::path cachePath = makeCacheFilePath(sourcePath, cacheDirectory, meshSignature);
    if (!std::filesystem::exists(cachePath)) {
        return false;
    }

    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Failed to open meshlet cache file {}", cachePath.string());
        return false;
    }

    MeshletCacheHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        spdlog::warn("Failed to read meshlet cache header {}", cachePath.string());
        return false;
    }

    if (std::memcmp(header.magic, kMeshletCacheMagic, sizeof(header.magic)) != 0 ||
        header.version != kMeshletCacheVersion ||
        header.maxVertices != MAX_VERTICES ||
        header.maxTriangles != MAX_TRIANGLES ||
        std::fabs(header.coneWeight - CONE_WEIGHT) > 1e-6f ||
        header.meshSignature != meshSignature ||
        header.meshletsPerGroupCount != expectedMeshletGroupCount(mesh)) {
        spdlog::warn("Meshlet cache {} is incompatible with the current mesh", cachePath.string());
        return false;
    }

    MeshletData cached;
    if (!readVector(file, header.meshletsPerGroupCount, cached.meshletsPerGroup) ||
        !readVector(file, header.meshletCount, cached.cpuMeshlets) ||
        !readVector(file, header.meshletVertexCount, cached.cpuMeshletVertices) ||
        !readVector(file, header.meshletTriangleByteCount, cached.cpuMeshletTriangles) ||
        !readVector(file, header.boundsCount, cached.cpuBounds) ||
        !readVector(file, header.materialCount, cached.cpuMaterialIDs)) {
        spdlog::warn("Failed to read meshlet cache payload {}", cachePath.string());
        return false;
    }

    const auto currentOffset = file.tellg();
    file.seekg(0, std::ios::end);
    const auto endOffset = file.tellg();
    if (currentOffset != endOffset) {
        spdlog::warn("Meshlet cache {} has unexpected trailing data", cachePath.string());
        return false;
    }

    if (!validateMeshletPayload(cached, expectedMeshletGroupCount(mesh))) {
        spdlog::warn("Meshlet cache {} failed payload validation", cachePath.string());
        return false;
    }

    if (!uploadMeshletBuffers(device, cached)) {
        spdlog::warn("Failed to create GPU meshlet buffers from cache {}", cachePath.string());
        return false;
    }

    out = std::move(cached);
    spdlog::info("Loaded {} meshlets from cache {}", out.meshletCount, cachePath.string());
    return true;
}

} // namespace

bool buildMeshlets(const RhiDevice& device, const LoadedMesh& mesh, MeshletData& out) {
    out.meshletsPerGroup.clear();

    const auto buildStart = std::chrono::steady_clock::now();

    const auto* allPositions = mesh.cpuPositions.empty()
        ? static_cast<const float*>(rhiBufferContents(mesh.positionBuffer))
        : mesh.cpuPositions.data();
    const auto* allIndices = mesh.cpuIndices.empty()
        ? static_cast<const uint32_t*>(rhiBufferContents(mesh.indexBuffer))
        : mesh.cpuIndices.data();
    constexpr size_t kPositionStride = sizeof(float) * 3;

    if (!allPositions || !allIndices) {
        spdlog::error("Meshlet builder requires CPU-readable position and index data");
        return false;
    }

    // Accumulated output across all primitive groups
    std::vector<GPUMeshlet> allGpuMeshlets;
    std::vector<unsigned int> allMeshletVertices;
    std::vector<uint32_t> allPackedTriangles;
    std::vector<unsigned char> allRawTriangles;
    std::vector<GPUMeshletBounds> allBounds;
    std::vector<uint32_t> allMaterialIDs;

    size_t meshletBound = 0;
    if (mesh.primitiveGroups.empty()) {
        meshletBound = meshopt_buildMeshletsBound(mesh.indexCount, MAX_VERTICES, MAX_TRIANGLES);
    } else {
        for (const auto& group : mesh.primitiveGroups) {
            meshletBound += meshopt_buildMeshletsBound(group.indexCount, MAX_VERTICES, MAX_TRIANGLES);
        }
    }
    allGpuMeshlets.reserve(meshletBound);
    allBounds.reserve(meshletBound);
    allMaterialIDs.reserve(meshletBound);

    std::vector<meshopt_Meshlet> meshlets;
    std::vector<unsigned int> meshletVertices;
    std::vector<unsigned char> meshletTriangles;
    std::vector<uint32_t> localIndices;

    auto buildGroupMeshlets = [&](const LoadedMesh::PrimitiveGroup& group) {
        const uint32_t* groupIndices = allIndices + group.indexOffset;
        size_t groupIndexCount = group.indexCount;
        const uint32_t groupVertexOffset = group.vertexOffset;
        const uint32_t groupVertexCount = group.vertexCount;
        const auto* groupPositions = allPositions + static_cast<size_t>(groupVertexOffset) * 3;

        if (groupIndexCount == 0 || groupVertexCount == 0) {
            out.meshletsPerGroup.push_back(0);
            return true;
        }

        // Compute worst-case buffer sizes for this group
        size_t maxMeshlets = meshopt_buildMeshletsBound(groupIndexCount, MAX_VERTICES, MAX_TRIANGLES);
        meshlets.resize(maxMeshlets);
        meshletVertices.resize(maxMeshlets * MAX_VERTICES);
        meshletTriangles.resize(maxMeshlets * MAX_TRIANGLES * 3);

        const uint32_t groupVertexEnd = groupVertexOffset + groupVertexCount;
        const uint32_t* meshletSourceIndices = groupIndices;
        if (groupVertexOffset != 0 || groupVertexCount != mesh.vertexCount) {
            localIndices.resize(groupIndexCount);
            for (size_t i = 0; i < groupIndexCount; ++i) {
                const uint32_t index = groupIndices[i];
                if (index < groupVertexOffset || index >= groupVertexEnd) {
                    spdlog::error("Primitive group index {} out of range [{}, {})",
                                  index,
                                  groupVertexOffset,
                                  groupVertexEnd);
                    return false;
                }
                localIndices[i] = index - groupVertexOffset;
            }
            meshletSourceIndices = localIndices.data();
        }

        // Build meshlets for this group
        size_t meshletCount = meshopt_buildMeshlets(
            meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
            meshletSourceIndices, groupIndexCount,
            groupPositions, groupVertexCount, kPositionStride,
            MAX_VERTICES, MAX_TRIANGLES, CONE_WEIGHT);

        meshlets.resize(meshletCount);

        // Optimize each meshlet
        for (size_t i = 0; i < meshletCount; i++) {
            meshopt_optimizeMeshlet(
                &meshletVertices[meshlets[i].vertex_offset],
                &meshletTriangles[meshlets[i].triangle_offset],
                meshlets[i].triangle_count,
                meshlets[i].vertex_count);
        }

        // Compute bounds
        for (size_t i = 0; i < meshletCount; i++) {
            meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                &meshletVertices[meshlets[i].vertex_offset],
                &meshletTriangles[meshlets[i].triangle_offset],
                meshlets[i].triangle_count,
                groupPositions, groupVertexCount, kPositionStride);

            GPUMeshletBounds gb;
            gb.center_radius[0] = bounds.center[0];
            gb.center_radius[1] = bounds.center[1];
            gb.center_radius[2] = bounds.center[2];
            gb.center_radius[3] = bounds.radius;
            gb.cone_apex_pad[0] = bounds.cone_apex[0];
            gb.cone_apex_pad[1] = bounds.cone_apex[1];
            gb.cone_apex_pad[2] = bounds.cone_apex[2];
            gb.cone_apex_pad[3] = 0.0f;
            gb.cone_axis_cutoff[0] = bounds.cone_axis[0];
            gb.cone_axis_cutoff[1] = bounds.cone_axis[1];
            gb.cone_axis_cutoff[2] = bounds.cone_axis[2];
            gb.cone_axis_cutoff[3] = bounds.cone_cutoff;
            allBounds.push_back(gb);
        }

        // Merge vertex indices (global offsets into allMeshletVertices)
        size_t vertexBaseOffset = allMeshletVertices.size();
        if (meshletCount > 0) {
            const meshopt_Meshlet& last = meshlets.back();
            size_t usedVertices = last.vertex_offset + last.vertex_count;
            allMeshletVertices.reserve(allMeshletVertices.size() + usedVertices);
            for (size_t i = 0; i < usedVertices; ++i) {
                allMeshletVertices.push_back(meshletVertices[i] + groupVertexOffset);
            }
        }

        // Pack triangles and merge
        size_t packedBaseOffset = allPackedTriangles.size();
        size_t rawTriBaseOffset = allRawTriangles.size();
        for (size_t i = 0; i < meshletCount; i++) {
            const meshopt_Meshlet& m = meshlets[i];
            for (size_t t = 0; t < m.triangle_count; t++) {
                size_t srcIdx = m.triangle_offset + t * 3;
                uint32_t v0 = meshletTriangles[srcIdx + 0];
                uint32_t v1 = meshletTriangles[srcIdx + 1];
                uint32_t v2 = meshletTriangles[srcIdx + 2];
                allPackedTriangles.push_back(v0 | (v1 << 8) | (v2 << 16));
                allRawTriangles.push_back(meshletTriangles[srcIdx + 0]);
                allRawTriangles.push_back(meshletTriangles[srcIdx + 1]);
                allRawTriangles.push_back(meshletTriangles[srcIdx + 2]);
            }
        }

        // Build GPU meshlet descriptors with global offsets
        size_t triOffset = packedBaseOffset;
        for (size_t i = 0; i < meshletCount; i++) {
            GPUMeshlet gm;
            gm.vertex_offset   = static_cast<uint32_t>(vertexBaseOffset + meshlets[i].vertex_offset);
            gm.triangle_offset = static_cast<uint32_t>(triOffset);
            gm.vertex_count    = meshlets[i].vertex_count;
            gm.triangle_count  = meshlets[i].triangle_count;
            allGpuMeshlets.push_back(gm);
            triOffset += meshlets[i].triangle_count;
        }

        // All meshlets in this group share the same material
        for (size_t i = 0; i < meshletCount; i++) {
            allMaterialIDs.push_back(group.materialIndex);
        }

        out.meshletsPerGroup.push_back(static_cast<uint32_t>(meshletCount));
        return true;
    };

    if (mesh.primitiveGroups.empty()) {
        LoadedMesh::PrimitiveGroup group;
        group.indexOffset = 0;
        group.indexCount = mesh.indexCount;
        group.vertexOffset = 0;
        group.vertexCount = mesh.vertexCount;
        group.materialIndex = 0;
        if (!buildGroupMeshlets(group)) {
            return false;
        }
    } else {
        for (const auto& group : mesh.primitiveGroups) {
            if (!buildGroupMeshlets(group)) {
                return false;
            }
        }
    }

    size_t totalMeshlets = allGpuMeshlets.size();
    if (totalMeshlets == 0) {
        spdlog::error("No meshlets built");
        return false;
    }

    // Retain CPU-side copies for LOD building
    out.cpuMeshlets = allGpuMeshlets;
    out.cpuMeshletVertices = allMeshletVertices;
    out.cpuMeshletTriangles = std::move(allRawTriangles);
    out.cpuBounds = allBounds;
    out.cpuMaterialIDs = allMaterialIDs;
    if (!uploadMeshletBuffers(device, out)) {
        return false;
    }

    // Print stats
    size_t totalTris = 0, totalVerts = 0;
    for (const auto& gm : allGpuMeshlets) {
        totalTris += gm.triangle_count;
        totalVerts += gm.vertex_count;
    }
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("Built {} meshlets from {} groups (avg {} verts, {} tris per meshlet)",
                 totalMeshlets,
                 out.meshletsPerGroup.size(),
                 totalVerts / totalMeshlets,
                 totalTris / totalMeshlets);
    spdlog::info("Meshlet build completed in {} ms", elapsedMs);

    return true;
}

bool loadOrBuildMeshlets(const RhiDevice& device,
                         const LoadedMesh& mesh,
                         const std::string& sourcePath,
                         const std::string& cacheDirectory,
                         MeshletData& out) {
    if (loadMeshletsFromCache(device, mesh, sourcePath, cacheDirectory, out)) {
        return true;
    }

    if (!buildMeshlets(device, mesh, out)) {
        return false;
    }

    if (!saveMeshletsToCache(mesh, sourcePath, cacheDirectory, out)) {
        spdlog::warn("Continuing with runtime-generated meshlets for {}", sourcePath);
    }

    return true;
}
