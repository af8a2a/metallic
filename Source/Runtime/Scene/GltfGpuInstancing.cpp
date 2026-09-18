#include "Runtime/Scene/GltfGpuInstancing.h"
#include "tiny_gltf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace metallic::scene::detail {
namespace {
using Json = nlohmann::json;
constexpr const char* kExtension = "EXT_mesh_gpu_instancing";

uint64_t unsignedValue(const Json& value)
{
    if (!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned() && value.get<int64_t>() < 0)) {
        throw std::runtime_error("Instance accessor contains a negative or non-integer index/size");
    }
    return value.get<uint64_t>();
}

std::vector<std::array<double, 4>> readInstances(const Json& root, uint64_t index,
    const std::string& semantic, const std::filesystem::path& directory, uint64_t& readBytes)
{
    const Json& accessor = root.at("accessors").at(index);
    const bool rotation = semantic == "ROTATION";
    const uint64_t components = rotation ? 4u : 3u;
    const uint64_t type = unsignedValue(accessor.at("componentType"));
    const bool normalized = accessor.value("normalized", false);
    if (accessor.value("type", "") != (rotation ? "VEC4" : "VEC3") ||
        (type != 5126 && !(rotation && normalized && (type == 5120 || type == 5122))) ||
        (type == 5126 && normalized) || accessor.contains("sparse")) {
        throw std::runtime_error("Unsupported instance accessor type/normalization/sparse storage for " + semantic);
    }
    const uint64_t count = unsignedValue(accessor.at("count"));
    if (count == 0 || count > INT32_MAX) { throw std::runtime_error("Invalid instance accessor count"); }
    const Json& view = root.at("bufferViews").at(unsignedValue(accessor.at("bufferView")));
    if (view.value("extensions", Json::object()).contains("EXT_meshopt_compression")) {
        throw std::runtime_error("Compressed instance accessors are not supported yet");
    }
    const Json& buffer = root.at("buffers").at(unsignedValue(view.at("buffer")));
    const std::string uri = buffer.value("uri", "");
    if (uri.empty() || uri.starts_with("data:") || uri.find("://") != std::string::npos) {
        throw std::runtime_error("Instance accessors require an external buffer URI");
    }
    std::string decodedUri;
    if (!tinygltf::URIDecode(uri, &decodedUri, nullptr)) { throw std::runtime_error("Invalid instance buffer URI"); }
    const auto path = directory / std::filesystem::u8path(decodedUri);
    const uint64_t componentSize = type == 5126 ? 4u : type == 5122 ? 2u : 1u;
    const uint64_t elementSize = componentSize * components;
    const uint64_t stride = unsignedValue(view.value("byteStride", Json(elementSize)));
    const uint64_t offset = unsignedValue(accessor.value("byteOffset", Json(0)));
    const uint64_t viewOffset = unsignedValue(view.value("byteOffset", Json(0)));
    const uint64_t viewSize = unsignedValue(view.at("byteLength"));
    const uint64_t bufferSize = unsignedValue(buffer.at("byteLength"));
    if (stride < elementSize || stride % componentSize != 0 || offset % componentSize != 0 ||
        viewOffset % componentSize != 0 || viewOffset > bufferSize || viewSize > bufferSize - viewOffset ||
        offset > viewSize || elementSize > viewSize - offset ||
        count - 1u > (viewSize - offset - elementSize) / stride) {
        throw std::runtime_error("Instance accessor range/alignment exceeds its buffer view");
    }
    const uint64_t bytes = (count - 1u) * stride + elementSize;
    const uint64_t fileOffset = viewOffset + offset;
    const uint64_t fileSize = std::filesystem::file_size(path);
    if (fileOffset > fileSize || bytes > fileSize - fileOffset || bytes > SIZE_MAX ||
        bytes > uint64_t(std::numeric_limits<std::streamsize>::max()) ||
        fileOffset > uint64_t(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("Instance accessor range exceeds external file");
    }
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(fileOffset));
    std::vector<uint8_t> data(static_cast<size_t>(bytes));
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Cannot read instance accessor range");
    }
    readBytes += bytes;
    std::vector<std::array<double, 4>> values(static_cast<size_t>(count));
    for (size_t i = 0; i < values.size(); ++i) {
        for (size_t c = 0; c < components; ++c) {
            const uint8_t* address = data.data() + i * stride + c * componentSize;
            double value = 0.0;
            if (type == 5126) { float v; std::memcpy(&v, address, 4); value = v; }
            else if (type == 5122) { int16_t v; std::memcpy(&v, address, 2); value = std::max(-1.0, double(v) / 32767.0); }
            else { int8_t v; std::memcpy(&v, address, 1); value = std::max(-1.0, double(v) / 127.0); }
            if (!std::isfinite(value)) { throw std::runtime_error("Non-finite instance transform"); }
            values[i][c] = value;
        }
        if (rotation) {
            auto& q = values[i];
            const double length = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (length < 1e-12) { throw std::runtime_error("Zero-length instance quaternion"); }
            for (auto& component : q) { component /= length; }
        }
    }
    return values;
}
} // namespace

bool expandGltfGpuInstances(Json& root, const std::filesystem::path& directory,
    GltfInstanceExpansion& expansion, std::string& reason)
{
    expansion = {};
    try {
        if (!root.contains("nodes")) { return true; }
        auto& nodes = root.at("nodes");
        if (!nodes.is_array() || nodes.size() > INT32_MAX) { throw std::runtime_error("Invalid node array"); }
        const size_t originalCount = nodes.size();
        expansion.sourceNodeCount = static_cast<uint32_t>(originalCount);
        std::map<std::pair<uint64_t, std::string>, std::vector<std::array<double, 4>>> cache;
        for (size_t n = 0; n < originalCount; ++n) {
            const Json extension = nodes[n].value("extensions", Json::object()).value(kExtension, Json());
            if (extension.is_null()) { continue; }
            const uint64_t mesh = unsignedValue(nodes[n].at("mesh"));
            if (mesh >= root.at("meshes").size() || nodes[n].contains("skin") || nodes[n].contains("weights")) {
                throw std::runtime_error("GPU instancing requires a valid static mesh node");
            }
            const Json attributes = extension.at("attributes");
            if (!attributes.is_object() || attributes.empty()) { throw std::runtime_error("Empty instance attributes"); }
            size_t count = 0;
            for (const auto& [semantic, index] : attributes.items()) {
                if (semantic != "TRANSLATION" && semantic != "ROTATION" && semantic != "SCALE") {
                    throw std::runtime_error("Unsupported instance attribute: " + semantic);
                }
                const auto key = std::make_pair(unsignedValue(index), semantic);
                auto [it, inserted] = cache.try_emplace(key);
                if (inserted) { it->second = readInstances(root, key.first, semantic, directory, expansion.rangeReadBytes); }
                if (count != 0 && count != it->second.size()) { throw std::runtime_error("Instance accessor count mismatch"); }
                count = it->second.size();
            }
            if (count > INT32_MAX - nodes.size()) { throw std::runtime_error("Expanded node count exceeds int32 capacity"); }
            Json children = Json::array();
            const Json originalChildren = nodes[n].value("children", Json::array());
            const std::string name = nodes[n].value("name", "Node " + std::to_string(n));
            for (size_t i = 0; i < count; ++i) {
                Json child{{"name", name + " [instance " + std::to_string(i) + "]"}, {"mesh", mesh}};
                for (const auto& [semantic, index] : attributes.items()) {
                    const auto& v = cache.at({unsignedValue(index), semantic})[i];
                    if (semantic == "ROTATION") { child["rotation"] = v; }
                    else { child[semantic == "TRANSLATION" ? "translation" : "scale"] = {v[0], v[1], v[2]}; }
                }
                const auto nodeIndex = static_cast<uint32_t>(nodes.size());
                expansion.instances.push_back({nodeIndex, static_cast<uint32_t>(n), static_cast<uint32_t>(i)});
                children.push_back(nodeIndex);
                nodes.push_back(std::move(child));
            }
            for (const auto& child : originalChildren) { children.push_back(child); }
            nodes[n]["children"] = std::move(children);
            nodes[n].erase("mesh"); // Do not draw the uninstanced node as an extra copy.
            nodes[n]["extensions"].erase(kExtension);
        }
        for (const char* field : {"extensionsRequired", "extensionsUsed"}) {
            if (root.contains(field)) {
                auto& extensions = root[field];
                extensions.erase(std::remove(extensions.begin(), extensions.end(), Json(kExtension)), extensions.end());
            }
        }
        return true;
    } catch (const std::exception& error) {
        reason = std::string("EXT_mesh_gpu_instancing: ") + error.what();
        expansion = {};
        return false;
    }
}
} // namespace metallic::scene::detail
