#include "scene.h"

#include <json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace {

bool nearlyZero(double v, double eps = 1e-5) { return std::fabs(v) <= eps; }
bool nearlyOne(double v, double eps = 1e-5) { return std::fabs(v - 1.0) <= eps; }

bool isIdentityRotation(const tinygltf::Node& node, double eps = 1e-5) {
    if (node.rotation.empty()) return true;
    return nearlyZero(node.rotation[0], eps) &&
           nearlyZero(node.rotation[1], eps) &&
           nearlyZero(node.rotation[2], eps) &&
           nearlyOne(std::fabs(node.rotation[3]), eps);
}

bool hasIdentityTransform(const tinygltf::Node& node, double eps = 1e-5) {
    if (!node.matrix.empty()) return false;
    bool identityT = node.translation.empty() ||
        (nearlyZero(node.translation[0], eps) &&
         nearlyZero(node.translation[1], eps) &&
         nearlyZero(node.translation[2], eps));
    bool identityS = node.scale.empty() ||
        (nearlyOne(node.scale[0], eps) &&
         nearlyOne(node.scale[1], eps) &&
         nearlyOne(node.scale[2], eps));
    return identityT && isIdentityRotation(node, eps) && identityS;
}

bool descendantsHaveIdentityTransforms(const tinygltf::Model& model, int nodeIndex) {
    const auto& node = model.nodes[nodeIndex];
    for (int child : node.children) {
        if (!hasIdentityTransform(model.nodes[child]) ||
            !descendantsHaveIdentityTransforms(model, child))
            return false;
    }
    return true;
}

bool tryGetSingleRootBakeScale(const tinygltf::Model& model, float& outScale) {
    outScale = 1.0f;
    if (model.scenes.empty()) return false;

    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    const auto& scene = model.scenes[sceneIndex];
    if (scene.nodes.size() != 1) return false;

    const auto& root = model.nodes[scene.nodes[0]];
    if (!root.matrix.empty()) return false;

    bool hasNonUniformScale = false;
    if (!root.scale.empty()) {
        double sx = root.scale[0], sy = root.scale[1], sz = root.scale[2];
        if (!nearlyOne(sx) || !nearlyOne(sy) || !nearlyOne(sz)) {
            if (nearlyZero(sx - sy) && nearlyZero(sy - sz)) {
                hasNonUniformScale = false;
            } else {
                return false;
            }
        }
    }

    bool hasTranslation = !root.translation.empty() &&
        !(nearlyZero(root.translation[0]) && nearlyZero(root.translation[1]) && nearlyZero(root.translation[2]));
    bool hasRotation = !isIdentityRotation(root);

    if (hasTranslation || hasRotation) return false;

    if (root.scale.empty() || (nearlyOne(root.scale[0]) && nearlyOne(root.scale[1]) && nearlyOne(root.scale[2])))
        return false;

    if (!descendantsHaveIdentityTransforms(model, scene.nodes[0]))
        return false;

    outScale = static_cast<float>(root.scale[0]);
    return true;
}

template <typename T>
const T* accessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& bv = model.bufferViews[accessor.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    return reinterpret_cast<const T*>(buf.data.data() + bv.byteOffset + accessor.byteOffset);
}

int accessorStride(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
    const auto& bv = model.bufferViews[accessor.bufferView];
    if (bv.byteStride > 0) return static_cast<int>(bv.byteStride);
    int componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    int numComponents = tinygltf::GetNumComponentsInType(accessor.type);
    return componentSize * numComponents;
}

void readFloatAccessor(const tinygltf::Model& model, int accessorIndex,
                       std::vector<float>& out, int expectedComponents) {
    if (accessorIndex < 0) return;
    const auto& acc = model.accessors[accessorIndex];
    const auto& bv = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;
    int stride = accessorStride(model, acc);
    int numComp = tinygltf::GetNumComponentsInType(acc.type);
    int readComp = std::min(numComp, expectedComponents);

    size_t startOut = out.size();
    out.resize(startOut + acc.count * expectedComponents);

    for (size_t i = 0; i < acc.count; ++i) {
        const float* src = reinterpret_cast<const float*>(base + i * stride);
        for (int c = 0; c < readComp; ++c)
            out[startOut + i * expectedComponents + c] = src[c];
        for (int c = readComp; c < expectedComponents; ++c)
            out[startOut + i * expectedComponents + c] = 0.0f;
    }
}

void readIndexAccessor(const tinygltf::Model& model, int accessorIndex,
                       std::vector<uint32_t>& out, uint32_t vertexOffset) {
    if (accessorIndex < 0) return;
    const auto& acc = model.accessors[accessorIndex];
    const auto& bv = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;
    int stride = accessorStride(model, acc);

    size_t startOut = out.size();
    out.resize(startOut + acc.count);

    for (size_t i = 0; i < acc.count; ++i) {
        const uint8_t* ptr = base + i * stride;
        uint32_t idx = 0;
        switch (acc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  idx = *ptr; break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: idx = *reinterpret_cast<const uint16_t*>(ptr); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   idx = *reinterpret_cast<const uint32_t*>(ptr); break;
            default: break;
        }
        out[startOut + i] = idx + vertexOffset;
    }
}

int findAccessorIndex(const tinygltf::Primitive& prim, const std::string& attr) {
    auto it = prim.attributes.find(attr);
    return (it != prim.attributes.end()) ? it->second : -1;
}

} // namespace

void Scene::clear() {
    positions.clear();
    normals.clear();
    uvs.clear();
    indices.clear();
    primitives.clear();
    meshInfos.clear();
    materials.clear();
    images.clear();
    nodes.clear();
    rootNodes.clear();
    cameras.clear();
    lights.clear();
    std::memset(bboxMin, 0, sizeof(bboxMin));
    std::memset(bboxMax, 0, sizeof(bboxMax));
    hasBakedRootScale = false;
    bakedRootScale = 1.0f;
    m_loaded = false;
    m_filePath.clear();
}

bool Scene::load(const std::string& gltfPath) {
    clear();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    loader.SetImageLoader(nullptr, nullptr);

    bool ok = false;
    if (gltfPath.size() >= 4 && gltfPath.substr(gltfPath.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, gltfPath);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    }

    if (!warn.empty()) spdlog::warn("glTF warning: {}", warn);
    if (!err.empty()) spdlog::error("glTF error: {}", err);
    if (!ok) {
        spdlog::error("Failed to load glTF: {}", gltfPath);
        return false;
    }

    m_filePath = gltfPath;
    std::filesystem::path baseDir = std::filesystem::path(gltfPath).parent_path();

    // --- Parse meshes (geometry) ---
    float bMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float bMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t totalPrimitives = 0;

    meshInfos.resize(model.meshes.size());
    for (int mi = 0; mi < static_cast<int>(model.meshes.size()); ++mi) {
        const auto& mesh = model.meshes[mi];
        meshInfos[mi].firstPrimitive = static_cast<uint32_t>(primitives.size());

        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1)
                continue;

            int posAccessor = findAccessorIndex(prim, "POSITION");
            if (posAccessor < 0) continue;

            uint32_t vertexOffset = static_cast<uint32_t>(positions.size() / 3);
            uint32_t vertexCountBefore = vertexOffset;

            readFloatAccessor(model, posAccessor, positions, 3);
            uint32_t vertexCount = static_cast<uint32_t>(positions.size() / 3) - vertexCountBefore;

            readFloatAccessor(model, findAccessorIndex(prim, "NORMAL"), normals, 3);
            if (normals.size() / 3 < positions.size() / 3) {
                normals.resize(positions.size(), 0.0f);
            }

            readFloatAccessor(model, findAccessorIndex(prim, "TEXCOORD_0"), uvs, 2);
            if (uvs.size() / 2 < positions.size() / 3) {
                uvs.resize((positions.size() / 3) * 2, 0.0f);
            }

            uint32_t indexOffset = static_cast<uint32_t>(indices.size());
            uint32_t indexCount = 0;
            if (prim.indices >= 0) {
                readIndexAccessor(model, prim.indices, indices, vertexOffset);
                indexCount = static_cast<uint32_t>(indices.size()) - indexOffset;
            } else {
                indexCount = vertexCount;
                indices.reserve(indices.size() + vertexCount);
                for (uint32_t vi = 0; vi < vertexCount; ++vi)
                    indices.push_back(vertexOffset + vi);
            }

            // Update bounds
            const auto& posAcc = model.accessors[posAccessor];
            if (posAcc.minValues.size() >= 3 && posAcc.maxValues.size() >= 3) {
                for (int a = 0; a < 3; ++a) {
                    bMin[a] = std::min(bMin[a], static_cast<float>(posAcc.minValues[a]));
                    bMax[a] = std::max(bMax[a], static_cast<float>(posAcc.maxValues[a]));
                }
            }

            ScenePrimitive sp;
            sp.vertexOffset = vertexOffset;
            sp.vertexCount = vertexCount;
            sp.indexOffset = indexOffset;
            sp.indexCount = indexCount;
            sp.materialIndex = (prim.material >= 0) ? static_cast<uint32_t>(prim.material) : 0;
            sp.meshID = mi;
            primitives.push_back(sp);
            ++totalPrimitives;
        }

        meshInfos[mi].primitiveCount =
            static_cast<uint32_t>(primitives.size()) - meshInfos[mi].firstPrimitive;
    }

    // Bake single-root scale
    float bakeScale = 1.0f;
    if (tryGetSingleRootBakeScale(model, bakeScale)) {
        for (float& p : positions) p *= bakeScale;
        for (int a = 0; a < 3; ++a) {
            bMin[a] *= bakeScale;
            bMax[a] *= bakeScale;
        }
        hasBakedRootScale = true;
        bakedRootScale = bakeScale;
        spdlog::info("Baked single-root scale {} into mesh data", bakeScale);
    }

    for (int a = 0; a < 3; ++a) {
        bboxMin[a] = bMin[a];
        bboxMax[a] = bMax[a];
    }

    // --- Parse materials ---
    materials.resize(std::max<size_t>(model.materials.size(), 1));
    for (int mi = 0; mi < static_cast<int>(model.materials.size()); ++mi) {
        const auto& mat = model.materials[mi];
        auto& sm = materials[mi];

        const auto& pbr = mat.pbrMetallicRoughness;
        for (int c = 0; c < 4; ++c)
            sm.baseColorFactor[c] = static_cast<float>(pbr.baseColorFactor[c]);
        sm.metallicFactor = static_cast<float>(pbr.metallicFactor);
        sm.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

        if (pbr.baseColorTexture.index >= 0)
            sm.baseColorTexture = model.textures[pbr.baseColorTexture.index].source;
        if (pbr.metallicRoughnessTexture.index >= 0)
            sm.metallicRoughnessTexture = model.textures[pbr.metallicRoughnessTexture.index].source;
        if (mat.normalTexture.index >= 0)
            sm.normalTexture = model.textures[mat.normalTexture.index].source;

        sm.alphaMode = (mat.alphaMode == "MASK") ? 1 : 0;
        sm.alphaCutoff = static_cast<float>(mat.alphaCutoff);
    }

    // --- Parse images ---
    images.resize(model.images.size());
    for (int ii = 0; ii < static_cast<int>(model.images.size()); ++ii) {
        const auto& gltfImage = model.images[ii];
        auto& si = images[ii];

        if (!gltfImage.image.empty()) {
            si.width = gltfImage.width;
            si.height = gltfImage.height;
            si.channels = gltfImage.component;
            si.pixels.assign(gltfImage.image.begin(), gltfImage.image.end());
        } else if (!gltfImage.uri.empty()) {
            si.uri = (baseDir / gltfImage.uri).string();
            int w, h, ch;
            uint8_t* px = stbi_load(si.uri.c_str(), &w, &h, &ch, 4);
            if (px) {
                si.width = w;
                si.height = h;
                si.channels = 4;
                si.pixels.assign(px, px + w * h * 4);
                stbi_image_free(px);
            } else {
                spdlog::warn("Failed to load image: {}", si.uri);
            }
        }
    }

    // --- Parse nodes ---
    nodes.resize(model.nodes.size());
    for (int ni = 0; ni < static_cast<int>(model.nodes.size()); ++ni) {
        const auto& gn = model.nodes[ni];
        auto& sn = nodes[ni];
        sn.name = gn.name;
        sn.mesh = gn.mesh;
        sn.camera = gn.camera;

        if (gn.extensions.count("KHR_lights_punctual")) {
            const auto& ext = gn.extensions.at("KHR_lights_punctual");
            if (ext.Has("light"))
                sn.light = ext.Get("light").GetNumberAsInt();
        }

        if (!gn.matrix.empty()) {
            sn.hasMatrix = true;
            for (int i = 0; i < 16; ++i)
                sn.localMatrix[i] = static_cast<float>(gn.matrix[i]);
        } else {
            if (!gn.translation.empty()) {
                sn.translation[0] = static_cast<float>(gn.translation[0]);
                sn.translation[1] = static_cast<float>(gn.translation[1]);
                sn.translation[2] = static_cast<float>(gn.translation[2]);
            }
            if (!gn.rotation.empty()) {
                sn.rotation[0] = static_cast<float>(gn.rotation[0]);
                sn.rotation[1] = static_cast<float>(gn.rotation[1]);
                sn.rotation[2] = static_cast<float>(gn.rotation[2]);
                sn.rotation[3] = static_cast<float>(gn.rotation[3]);
            }
            if (!gn.scale.empty()) {
                sn.scale[0] = static_cast<float>(gn.scale[0]);
                sn.scale[1] = static_cast<float>(gn.scale[1]);
                sn.scale[2] = static_cast<float>(gn.scale[2]);
            }
        }

        for (int child : gn.children) {
            sn.children.push_back(child);
            nodes[child].parent = ni;
        }
    }

    // Root nodes
    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIndex < static_cast<int>(model.scenes.size())) {
        for (int n : model.scenes[sceneIndex].nodes)
            rootNodes.push_back(n);
    }

    // --- Parse cameras ---
    cameras.resize(model.cameras.size());
    for (int ci = 0; ci < static_cast<int>(model.cameras.size()); ++ci) {
        const auto& gc = model.cameras[ci];
        auto& sc = cameras[ci];
        if (gc.type == "perspective") {
            sc.type = SceneCamera::Perspective;
            sc.yfov = static_cast<float>(gc.perspective.yfov);
            sc.znear = static_cast<float>(gc.perspective.znear);
            sc.zfar = static_cast<float>(gc.perspective.zfar);
            sc.aspectRatio = static_cast<float>(gc.perspective.aspectRatio);
        } else {
            sc.type = SceneCamera::Orthographic;
            sc.znear = static_cast<float>(gc.orthographic.znear);
            sc.zfar = static_cast<float>(gc.orthographic.zfar);
        }
    }

    // --- Parse lights (KHR_lights_punctual) ---
    if (model.extensions.count("KHR_lights_punctual")) {
        const auto& ext = model.extensions.at("KHR_lights_punctual");
        if (ext.Has("lights")) {
            const auto& lightsArr = ext.Get("lights");
            lights.resize(lightsArr.ArrayLen());
            for (int li = 0; li < static_cast<int>(lightsArr.ArrayLen()); ++li) {
                const auto& gl = lightsArr.Get(li);
                auto& sl = lights[li];
                if (gl.Has("type")) {
                    std::string t = gl.Get("type").Get<std::string>();
                    if (t == "directional") sl.type = SceneLight::Directional;
                    else if (t == "point") sl.type = SceneLight::Point;
                    else if (t == "spot") sl.type = SceneLight::Spot;
                }
                if (gl.Has("color")) {
                    const auto& c = gl.Get("color");
                    for (int ci2 = 0; ci2 < 3 && ci2 < static_cast<int>(c.ArrayLen()); ++ci2)
                        sl.color[ci2] = static_cast<float>(c.Get(ci2).GetNumberAsDouble());
                }
                if (gl.Has("intensity"))
                    sl.intensity = static_cast<float>(gl.Get("intensity").GetNumberAsDouble());
            }
        }
    }

    spdlog::info("Scene loaded: {} primitives, {} materials, {} images, {} nodes ({})",
                 totalPrimitives, materials.size(), images.size(), nodes.size(), gltfPath);

    m_loaded = true;
    return true;
}
