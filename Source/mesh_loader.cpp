#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "mesh_loader.h"
#include <iostream>
#include <cstring>

bool loadGLTFMesh(MTL::Device* device, const std::string& gltfPath, LoadedMesh& out) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success) {
        std::cerr << "Failed to parse glTF: " << gltfPath << std::endl;
        return false;
    }

    result = cgltf_load_buffers(&options, data, gltfPath.c_str());
    if (result != cgltf_result_success) {
        std::cerr << "Failed to load glTF buffers" << std::endl;
        cgltf_free(data);
        return false;
    }

    if (data->meshes_count == 0 || data->meshes[0].primitives_count == 0) {
        std::cerr << "No mesh primitives found" << std::endl;
        cgltf_free(data);
        return false;
    }

    const cgltf_primitive& prim = data->meshes[0].primitives[0];

    // Find position and normal accessors
    const cgltf_accessor* posAccessor = nullptr;
    const cgltf_accessor* normAccessor = nullptr;

    for (cgltf_size i = 0; i < prim.attributes_count; i++) {
        if (prim.attributes[i].type == cgltf_attribute_type_position)
            posAccessor = prim.attributes[i].data;
        else if (prim.attributes[i].type == cgltf_attribute_type_normal)
            normAccessor = prim.attributes[i].data;
    }

    if (!posAccessor || !normAccessor) {
        std::cerr << "Missing POSITION or NORMAL attribute" << std::endl;
        cgltf_free(data);
        return false;
    }

    const cgltf_accessor* idxAccessor = prim.indices;
    if (!idxAccessor) {
        std::cerr << "Missing index accessor" << std::endl;
        cgltf_free(data);
        return false;
    }

    // Extract raw data pointers via byte offsets
    auto bufferPtr = [](const cgltf_accessor* acc) -> const uint8_t* {
        const cgltf_buffer_view* bv = acc->buffer_view;
        return static_cast<const uint8_t*>(bv->buffer->data)
               + bv->offset + acc->offset;
    };

    const uint8_t* posData  = bufferPtr(posAccessor);
    const uint8_t* normData = bufferPtr(normAccessor);
    const uint8_t* idxData  = bufferPtr(idxAccessor);

    size_t posSize  = posAccessor->count * sizeof(float) * 3;
    size_t normSize = normAccessor->count * sizeof(float) * 3;
    size_t idxSize  = idxAccessor->count * sizeof(uint32_t);

    out.positionBuffer = device->newBuffer(posData, posSize, MTL::ResourceStorageModeShared);
    out.normalBuffer   = device->newBuffer(normData, normSize, MTL::ResourceStorageModeShared);
    out.indexBuffer    = device->newBuffer(idxData, idxSize, MTL::ResourceStorageModeShared);

    out.vertexCount = static_cast<uint32_t>(posAccessor->count);
    out.indexCount  = static_cast<uint32_t>(idxAccessor->count);

    // Copy bounding box from position accessor
    if (posAccessor->has_min && posAccessor->has_max) {
        for (int i = 0; i < 3; i++) {
            out.bboxMin[i] = static_cast<float>(posAccessor->min[i]);
            out.bboxMax[i] = static_cast<float>(posAccessor->max[i]);
        }
    }

    cgltf_free(data);
    return true;
}
