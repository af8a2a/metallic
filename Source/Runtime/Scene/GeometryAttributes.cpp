#include "Runtime/Scene/GeometryAttributes.h"
#include "meshoptimizer.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <array>
#include <unordered_map>

namespace metallic::scene {

uint32_t repairZeroGeometryNormals(RenderPrimitive& primitive)
{
    if (primitive.mode != 4 || primitive.normals.size() != primitive.positions.size()) { return 0; }
    struct Recovery {
        std::array<double, 3> sum{}, first{};
    };
    std::unordered_map<uint32_t, Recovery> invalid;
    for (uint32_t i = 0; i < primitive.normals.size(); ++i) {
        const float lengthSquared = dot(primitive.normals[i], primitive.normals[i]);
        if (std::isfinite(lengthSquared) && lengthSquared < 1e-12f) { invalid.emplace(i, Recovery{}); }
    }
    if (invalid.empty()) { return 0; }
    const size_t cornerCount = primitive.indices.empty() ? primitive.positions.size() : primitive.indices.size();
    for (size_t i = 0; i + 2 < cornerCount; i += 3) {
        uint32_t ids[3];
        for (uint32_t c = 0; c < 3; ++c) {
            ids[c] = primitive.indices.empty() ? uint32_t(i + c) : primitive.indices[i + c];
        }
        if (ids[0] >= primitive.positions.size() || ids[1] >= primitive.positions.size() ||
            ids[2] >= primitive.positions.size()) { continue; }
        if (!invalid.contains(ids[0]) && !invalid.contains(ids[1]) && !invalid.contains(ids[2])) { continue; }
        const auto& a = primitive.positions[ids[0]];
        const auto& b = primitive.positions[ids[1]];
        const auto& c = primitive.positions[ids[2]];
        const double ux = double(b.x)-a.x, uy = double(b.y)-a.y, uz = double(b.z)-a.z;
        const double vx = double(c.x)-a.x, vy = double(c.y)-a.y, vz = double(c.z)-a.z;
        const std::array<double, 3> n{uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx};
        const double length = n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
        if (!std::isfinite(length) || length == 0) { continue; }
        for (uint32_t id : ids) {
            if (auto entry = invalid.find(id); entry != invalid.end()) {
                for (uint32_t axis = 0; axis < 3; ++axis) { entry->second.sum[axis] += n[axis]; }
                if (entry->second.first == std::array<double, 3>{}) { entry->second.first = n; }
            }
        }
    }
    for (const auto& [id, recovery] : invalid) {
        auto n = recovery.sum;
        double length = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (length == 0) {
            n = recovery.first;
            length = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        }
        // Unreferenced vertices and exclusively degenerate triangles have no
        // surface orientation. Give them a deterministic finite tangent frame.
        primitive.normals[id] = length > 0 ? float3(float(n[0]/length), float(n[1]/length), float(n[2]/length))
            : float3(0, 1, 0);
    }
    return static_cast<uint32_t>(invalid.size());
}

bool validateGeometryAttributes(const RenderPrimitive& primitive, std::string& reason)
{
    const size_t count = primitive.positions.size();
    if (primitive.mode == 4 && primitive.indices.size() % 3 != 0) {
        reason = "Triangle index count is not divisible by three"; return false;
    }
    if ((!primitive.normals.empty() && primitive.normals.size() != count) ||
        (!primitive.texcoords0.empty() && primitive.texcoords0.size() != count) ||
        (!primitive.tangents.empty() && primitive.tangents.size() != count)) {
        reason = "Geometry attribute count differs from POSITION";
        return false;
    }
    for (const auto& p : primitive.positions) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            reason = "Non-finite POSITION"; return false;
        }
    }
    for (const auto& n : primitive.normals) {
        if (!std::isfinite(dot(n, n)) || dot(n, n) < 1e-12f) {
            reason = "Invalid NORMAL"; return false;
        }
    }
    for (const auto& uv : primitive.texcoords0) {
        if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            reason = "Non-finite TEXCOORD_0"; return false;
        }
    }
    for (const auto& t : primitive.tangents) {
        const float lengthSquared = t.x*t.x + t.y*t.y + t.z*t.z;
        if (!std::isfinite(lengthSquared) || lengthSquared < 1e-12f ||
            (t.w != -1.0f && t.w != 1.0f)) {
            reason = "Invalid TANGENT or handedness"; return false;
        }
    }
    for (uint32_t index : primitive.indices) {
        if (index >= count) { reason = "Geometry index exceeds POSITION count"; return false; }
    }
    return true;
}

void generateMissingTangents(RenderPrimitive& primitive)
{
    if (!primitive.tangents.empty() || primitive.mode != 4 || primitive.positions.empty() ||
        primitive.normals.size() != primitive.positions.size() ||
        primitive.texcoords0.size() != primitive.positions.size()) { return; }
    std::string reason;
    if (!validateGeometryAttributes(primitive, reason)) { return; }
    if (primitive.indices.empty()) {
        primitive.indices.resize(primitive.positions.size() / 3 * 3);
        std::iota(primitive.indices.begin(), primitive.indices.end(), 0u);
    }
    if (primitive.indices.size() % 3 != 0 || primitive.indices.empty()) { return; }

    // Normalize a temporary stream for the generator; authored normals remain
    // bit-identical in the payload, including meshopt quantization roundoff.
    std::vector<float3> normals;
    normals.reserve(primitive.normals.size());
    for (const auto& n : primitive.normals) { normals.push_back(n / std::sqrt(dot(n, n))); }
    std::vector<float4> corners(primitive.indices.size());
    meshopt_generateTangents(&corners.front().x, primitive.indices.data(), primitive.indices.size(),
        &primitive.positions.front().x, primitive.positions.size(), sizeof(float3),
        &normals.front().x, sizeof(float3), &primitive.texcoords0.front().x, sizeof(float2),
        meshopt_TangentCompatible);

    // Keep original vertex IDs whenever possible. A mirror seam can have two
    // different tangent frames even when position, normal and UV are identical.
    constexpr uint32_t invalid = UINT32_MAX;
    std::vector<uint32_t> heads(primitive.positions.size(), invalid);
    std::vector<uint32_t> next(primitive.positions.size(), invalid);
    primitive.tangents.resize(primitive.positions.size(), float4(1, 0, 0, 1));
    for (size_t corner = 0; corner < primitive.indices.size(); ++corner) {
        const uint32_t original = primitive.indices[corner];
        uint32_t vertex = heads[original];
        while (vertex != invalid && std::memcmp(&primitive.tangents[vertex], &corners[corner], sizeof(float4)) != 0) {
            vertex = next[vertex];
        }
        if (vertex == invalid) {
            if (heads[original] == invalid) { vertex = original; }
            else {
                if (primitive.positions.size() >= UINT32_MAX) { throw std::runtime_error("Tangent splits exceed uint32 vertices"); }
                vertex = static_cast<uint32_t>(primitive.positions.size());
                primitive.positions.push_back(primitive.positions[original]);
                primitive.normals.push_back(primitive.normals[original]);
                primitive.texcoords0.push_back(primitive.texcoords0[original]);
                primitive.tangents.emplace_back();
                next.push_back(invalid);
            }
            primitive.tangents[vertex] = corners[corner];
            next[vertex] = heads[original];
            heads[original] = vertex;
        }
        primitive.indices[corner] = vertex;
    }
    primitive.vertexCount = primitive.positions.size();
    primitive.indexCount = primitive.indices.size();
}

} // namespace metallic::scene
