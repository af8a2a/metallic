#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "mesh_loader.h"
#include <spdlog/spdlog.h>
#include <cstring>
#include <vector>
#include <cfloat>
#include <algorithm>

bool loadGLTFMesh(MTL::Device* device, const std::string& gltfPath, LoadedMesh& out) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success) {
        spdlog::error("Failed to parse glTF: {}", gltfPath);
        return false;
    }

    result = cgltf_load_buffers(&options, data, gltfPath.c_str());
    if (result != cgltf_result_success) {
        spdlog::error("Failed to load glTF buffers");
        cgltf_free(data);
        return false;
    }

    if (data->meshes_count == 0) {
        spdlog::error("No meshes found");
        cgltf_free(data);
        return false;
    }

    // Merge all primitives from all meshes into single buffers
    std::vector<float> allPositions;
    std::vector<float> allNormals;
    std::vector<float> allUVs;
    std::vector<uint32_t> allIndices;

    float bboxMin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float bboxMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    size_t totalPrimitives = 0;

    for (cgltf_size mi = 0; mi < data->meshes_count; mi++) {
        const cgltf_mesh& mesh = data->meshes[mi];
        uint32_t groupStart = static_cast<uint32_t>(out.primitiveGroups.size());
        for (cgltf_size pi = 0; pi < mesh.primitives_count; pi++) {
            const cgltf_primitive& prim = mesh.primitives[pi];

            if (prim.type != cgltf_primitive_type_triangles)
                continue;

            const cgltf_accessor* posAccessor = nullptr;
            const cgltf_accessor* normAccessor = nullptr;
            const cgltf_accessor* uvAccessor = nullptr;

            for (cgltf_size i = 0; i < prim.attributes_count; i++) {
                if (prim.attributes[i].type == cgltf_attribute_type_position)
                    posAccessor = prim.attributes[i].data;
                else if (prim.attributes[i].type == cgltf_attribute_type_normal)
                    normAccessor = prim.attributes[i].data;
                else if (prim.attributes[i].type == cgltf_attribute_type_texcoord &&
                         prim.attributes[i].index == 0)
                    uvAccessor = prim.attributes[i].data;
            }

            if (!posAccessor || !normAccessor) {
                spdlog::warn("Primitive {} missing POSITION or NORMAL, skipping", pi);
                continue;
            }

            if (!prim.indices)
                continue;

            uint32_t vertexBase = static_cast<uint32_t>(allPositions.size() / 3);
            uint32_t indexBase = static_cast<uint32_t>(allIndices.size());

            // Unpack positions (handles any source component type)
            size_t posStart = allPositions.size();
            allPositions.resize(posStart + posAccessor->count * 3);
            cgltf_accessor_unpack_floats(posAccessor, &allPositions[posStart], posAccessor->count * 3);

            // Unpack normals
            size_t normStart = allNormals.size();
            allNormals.resize(normStart + normAccessor->count * 3);
            cgltf_accessor_unpack_floats(normAccessor, &allNormals[normStart], normAccessor->count * 3);

            // Unpack UVs (default to 0,0 if missing)
            size_t uvStart = allUVs.size();
            allUVs.resize(uvStart + posAccessor->count * 2);
            if (uvAccessor) {
                cgltf_accessor_unpack_floats(uvAccessor, &allUVs[uvStart], uvAccessor->count * 2);
            } else {
                std::fill(&allUVs[uvStart], &allUVs[uvStart] + posAccessor->count * 2, 0.0f);
            }

            // Update bounding box
            if (posAccessor->has_min && posAccessor->has_max) {
                for (int i = 0; i < 3; i++) {
                    bboxMin[i] = std::min(bboxMin[i], static_cast<float>(posAccessor->min[i]));
                    bboxMax[i] = std::max(bboxMax[i], static_cast<float>(posAccessor->max[i]));
                }
            }

            // Unpack indices (handles uint8, uint16, uint32 automatically)
            const cgltf_accessor* idxAccessor = prim.indices;
            size_t idxStart = allIndices.size();
            allIndices.resize(idxStart + idxAccessor->count);
            for (cgltf_size i = 0; i < idxAccessor->count; i++) {
                allIndices[idxStart + i] = vertexBase +
                    static_cast<uint32_t>(cgltf_accessor_read_index(idxAccessor, i));
            }

            // Determine material index
            uint32_t materialIndex = 0;
            if (prim.material) {
                materialIndex = static_cast<uint32_t>(prim.material - data->materials);
            }

            // Record primitive group
            LoadedMesh::PrimitiveGroup group;
            group.vertexOffset = vertexBase;
            group.vertexCount = static_cast<uint32_t>(posAccessor->count);
            group.indexOffset = indexBase;
            group.indexCount = static_cast<uint32_t>(idxAccessor->count);
            group.materialIndex = materialIndex;
            out.primitiveGroups.push_back(group);

            totalPrimitives++;
        }
        out.meshRanges.push_back({groupStart,
            static_cast<uint32_t>(out.primitiveGroups.size()) - groupStart});
    }

    if (allPositions.empty() || allIndices.empty()) {
        spdlog::error("No valid primitives found");
        cgltf_free(data);
        return false;
    }

    out.positionBuffer = device->newBuffer(
        allPositions.data(), allPositions.size() * sizeof(float),
        MTL::ResourceStorageModeShared);
    out.normalBuffer = device->newBuffer(
        allNormals.data(), allNormals.size() * sizeof(float),
        MTL::ResourceStorageModeShared);
    out.uvBuffer = device->newBuffer(
        allUVs.data(), allUVs.size() * sizeof(float),
        MTL::ResourceStorageModeShared);
    out.indexBuffer = device->newBuffer(
        allIndices.data(), allIndices.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);

    out.vertexCount = static_cast<uint32_t>(allPositions.size() / 3);
    out.indexCount  = static_cast<uint32_t>(allIndices.size());

    for (int i = 0; i < 3; i++) {
        out.bboxMin[i] = bboxMin[i];
        out.bboxMax[i] = bboxMax[i];
    }

    spdlog::info("Loaded {} primitives ({} groups): {} vertices, {} indices",
                 totalPrimitives, out.primitiveGroups.size(), out.vertexCount, out.indexCount);

    cgltf_free(data);
    return true;
}
