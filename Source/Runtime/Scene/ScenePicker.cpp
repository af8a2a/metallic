#include "Runtime/Scene/ScenePicker.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <vector>

namespace metallic::scene {
namespace {

constexpr int32_t kTriangleListMode = 4;

struct BvhNode {
    Bounds bounds;
    uint32_t first = 0;
    uint32_t count = 0;
    int32_t left = kInvalidSceneIndex;
    int32_t right = kInvalidSceneIndex;
};

struct TriangleReference {
    Bounds bounds;
    float3 center{0.0f, 0.0f, 0.0f};
    uint32_t triangleIndex = 0;
};

struct InstanceReference {
    Bounds bounds;
    float3 center{0.0f, 0.0f, 0.0f};
    int32_t renderNodeIndex = kInvalidSceneIndex;
};

float axisValue(const float3& value, uint32_t axis)
{
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

uint32_t longestAxis(const Bounds& bounds)
{
    const float3 extent = bounds.max - bounds.min;
    if (extent.y > extent.x && extent.y >= extent.z) {
        return 1;
    }
    return extent.z > extent.x ? 2 : 0;
}

template <typename Reference>
int32_t buildBvhNode(
    std::vector<BvhNode>& nodes,
    std::vector<uint32_t>& order,
    const std::vector<Reference>& references,
    uint32_t first,
    uint32_t count)
{
    const int32_t nodeIndex = static_cast<int32_t>(nodes.size());
    nodes.push_back(BvhNode{});
    Bounds bounds;
    Bounds centerBounds;
    for (uint32_t index = first; index < first + count; ++index) {
        const Reference& reference = references[order[index]];
        bounds.include(reference.bounds.min);
        bounds.include(reference.bounds.max);
        centerBounds.include(reference.center);
    }
    nodes[static_cast<size_t>(nodeIndex)].bounds = bounds;
    if (count <= 8 || !centerBounds.valid) {
        nodes[static_cast<size_t>(nodeIndex)].first = first;
        nodes[static_cast<size_t>(nodeIndex)].count = count;
        return nodeIndex;
    }

    const uint32_t axis = longestAxis(centerBounds);
    const uint32_t middle = first + count / 2;
    std::nth_element(
        order.begin() + static_cast<ptrdiff_t>(first),
        order.begin() + static_cast<ptrdiff_t>(middle),
        order.begin() + static_cast<ptrdiff_t>(first + count),
        [&](uint32_t lhs, uint32_t rhs) {
            return axisValue(references[lhs].center, axis) <
                axisValue(references[rhs].center, axis);
        });
    nodes[static_cast<size_t>(nodeIndex)].left = buildBvhNode(
        nodes,
        order,
        references,
        first,
        middle - first);
    nodes[static_cast<size_t>(nodeIndex)].right = buildBvhNode(
        nodes,
        order,
        references,
        middle,
        first + count - middle);
    return nodeIndex;
}

bool rayIntersectsBounds(
    const ScenePickRay& ray,
    const Bounds& bounds,
    float maximumDistance)
{
    if (!bounds.valid) {
        return false;
    }
    float minimum = 0.0f;
    float maximum = maximumDistance;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        const float origin = axisValue(ray.origin, axis);
        const float direction = axisValue(ray.direction, axis);
        const float boundsMin = axisValue(bounds.min, axis);
        const float boundsMax = axisValue(bounds.max, axis);
        if (std::abs(direction) <= 0.0000001f) {
            if (origin < boundsMin || origin > boundsMax) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / direction;
        float nearDistance = (boundsMin - origin) * inverse;
        float farDistance = (boundsMax - origin) * inverse;
        if (nearDistance > farDistance) {
            std::swap(nearDistance, farDistance);
        }
        minimum = std::max(minimum, nearDistance);
        maximum = std::min(maximum, farDistance);
        if (minimum > maximum) {
            return false;
        }
    }
    return true;
}

bool rayIntersectsTriangle(
    const ScenePickRay& ray,
    const float3& vertex0,
    const float3& vertex1,
    const float3& vertex2,
    float& distance)
{
    const float3 edge1 = vertex1 - vertex0;
    const float3 edge2 = vertex2 - vertex0;
    const float3 p = cross(ray.direction, edge2);
    const float determinant = dot(edge1, p);
    if (std::abs(determinant) <= 0.0000001f) {
        return false;
    }
    const float inverseDeterminant = 1.0f / determinant;
    const float3 t = ray.origin - vertex0;
    const float u = dot(t, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const float3 q = cross(t, edge1);
    const float v = dot(ray.direction, q) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float hitDistance = dot(edge2, q) * inverseDeterminant;
    if (hitDistance <= 0.000001f || hitDistance >= distance) {
        return false;
    }
    distance = hitDistance;
    return true;
}

Bounds transformBounds(const Bounds& bounds, const float4x4& matrix)
{
    Bounds result;
    if (!bounds.valid) {
        return result;
    }
    for (uint32_t corner = 0; corner < 8; ++corner) {
        result.include(matrix * float3(
            (corner & 1) != 0 ? bounds.max.x : bounds.min.x,
            (corner & 2) != 0 ? bounds.max.y : bounds.min.y,
            (corner & 4) != 0 ? bounds.max.z : bounds.min.z));
    }
    return result;
}

bool invertible(const float4x4& matrix)
{
    const float3 column0(matrix.a00, matrix.a10, matrix.a20);
    const float3 column1(matrix.a01, matrix.a11, matrix.a21);
    const float3 column2(matrix.a02, matrix.a12, matrix.a22);
    const float determinant = dot(column0, cross(column1, column2));
    return std::isfinite(determinant) && std::abs(determinant) > 0.0000001f;
}

} // namespace

struct ScenePicker::Impl {
    struct PrimitiveBvh {
        std::vector<TriangleReference> triangles;
        std::vector<uint32_t> order;
        std::vector<BvhNode> nodes;
    };

    std::filesystem::path scenePath;
    uint64_t sceneLifetimeRevision = std::numeric_limits<uint64_t>::max();
    uint64_t transformRevision = std::numeric_limits<uint64_t>::max();
    uint64_t structuralRevision = std::numeric_limits<uint64_t>::max();
    std::vector<PrimitiveBvh> primitiveBvhs;
    std::vector<InstanceReference> instances;
    std::vector<uint32_t> instanceOrder;
    std::vector<BvhNode> instanceNodes;

    void buildPrimitives(const Scene& scene)
    {
        primitiveBvhs.clear();
        primitiveBvhs.resize(scene.renderPrimitives().size());
        for (size_t primitiveIndex = 0; primitiveIndex < scene.renderPrimitives().size(); ++primitiveIndex) {
            const RenderPrimitive& primitive = scene.renderPrimitives()[primitiveIndex];
            PrimitiveBvh& bvh = primitiveBvhs[primitiveIndex];
            if (primitive.mode != kTriangleListMode) {
                continue;
            }
            const size_t triangleCount = primitive.indices.empty()
                ? primitive.positions.size() / 3
                : primitive.indices.size() / 3;
            bvh.triangles.reserve(triangleCount);
            for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
                const uint32_t index0 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangle * 3)
                    : primitive.indices[triangle * 3];
                const uint32_t index1 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangle * 3 + 1)
                    : primitive.indices[triangle * 3 + 1];
                const uint32_t index2 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangle * 3 + 2)
                    : primitive.indices[triangle * 3 + 2];
                if (index0 >= primitive.positions.size() ||
                    index1 >= primitive.positions.size() ||
                    index2 >= primitive.positions.size()) {
                    continue;
                }
                TriangleReference reference;
                reference.triangleIndex = static_cast<uint32_t>(triangle);
                reference.bounds.include(primitive.positions[index0]);
                reference.bounds.include(primitive.positions[index1]);
                reference.bounds.include(primitive.positions[index2]);
                reference.center =
                    (primitive.positions[index0] + primitive.positions[index1] + primitive.positions[index2]) /
                    3.0f;
                bvh.triangles.push_back(reference);
            }
            bvh.order.resize(bvh.triangles.size());
            std::iota(bvh.order.begin(), bvh.order.end(), 0u);
            if (!bvh.order.empty()) {
                bvh.nodes.reserve(bvh.order.size() * 2);
                buildBvhNode(
                    bvh.nodes,
                    bvh.order,
                    bvh.triangles,
                    0,
                    static_cast<uint32_t>(bvh.order.size()));
            }
        }
        scenePath = scene.filename();
        sceneLifetimeRevision = scene.sceneGraph().lifetimeRevision();
    }

    void buildInstances(const Scene& scene)
    {
        instances.clear();
        for (size_t renderNodeIndex = 0; renderNodeIndex < scene.renderNodes().size(); ++renderNodeIndex) {
            const RenderNode& renderNode = scene.renderNodes()[renderNodeIndex];
            if (!renderNode.visible || renderNode.renderPrimitiveIndex < 0 ||
                static_cast<size_t>(renderNode.renderPrimitiveIndex) >= scene.renderPrimitives().size()) {
                continue;
            }
            const RenderPrimitive& primitive =
                scene.renderPrimitives()[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
            if (primitive.mode != kTriangleListMode) {
                continue;
            }
            const Bounds worldBounds = transformBounds(primitive.localBounds, renderNode.worldMatrix);
            if (!worldBounds.valid) {
                continue;
            }
            instances.push_back(InstanceReference{
                .bounds = worldBounds,
                .center = worldBounds.center(),
                .renderNodeIndex = static_cast<int32_t>(renderNodeIndex),
            });
        }
        instanceOrder.resize(instances.size());
        std::iota(instanceOrder.begin(), instanceOrder.end(), 0u);
        instanceNodes.clear();
        if (!instanceOrder.empty()) {
            instanceNodes.reserve(instanceOrder.size() * 2);
            buildBvhNode(
                instanceNodes,
                instanceOrder,
                instances,
                0,
                static_cast<uint32_t>(instanceOrder.size()));
        }
        transformRevision = scene.transformRevision();
        structuralRevision = scene.sceneGraph().structuralRevision();
    }

    bool hitPrimitive(
        const RenderPrimitive& primitive,
        const PrimitiveBvh& bvh,
        const ScenePickRay& localRay,
        float& nearestDistance,
        uint32_t& triangleIndex) const
    {
        if (bvh.nodes.empty()) {
            return false;
        }
        bool hit = false;
        std::vector<int32_t> stack{0};
        while (!stack.empty()) {
            const BvhNode& node = bvh.nodes[static_cast<size_t>(stack.back())];
            stack.pop_back();
            if (!rayIntersectsBounds(localRay, node.bounds, nearestDistance)) {
                continue;
            }
            if (node.count == 0) {
                stack.push_back(node.left);
                stack.push_back(node.right);
                continue;
            }
            for (uint32_t index = node.first; index < node.first + node.count; ++index) {
                const TriangleReference& triangle = bvh.triangles[bvh.order[index]];
                const size_t triangleOffset = static_cast<size_t>(triangle.triangleIndex) * 3;
                const uint32_t index0 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangleOffset)
                    : primitive.indices[triangleOffset];
                const uint32_t index1 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangleOffset + 1)
                    : primitive.indices[triangleOffset + 1];
                const uint32_t index2 = primitive.indices.empty()
                    ? static_cast<uint32_t>(triangleOffset + 2)
                    : primitive.indices[triangleOffset + 2];
                float distance = nearestDistance;
                if (rayIntersectsTriangle(
                        localRay,
                        primitive.positions[index0],
                        primitive.positions[index1],
                        primitive.positions[index2],
                        distance)) {
                    nearestDistance = distance;
                    triangleIndex = triangle.triangleIndex;
                    hit = true;
                }
            }
        }
        return hit;
    }
};

ScenePicker::ScenePicker()
    : impl_(std::make_unique<Impl>())
{
}

ScenePicker::~ScenePicker() = default;
ScenePicker::ScenePicker(ScenePicker&&) noexcept = default;
ScenePicker& ScenePicker::operator=(ScenePicker&&) noexcept = default;

ScenePickResult ScenePicker::pick(const Scene& scene, const ScenePickRay& ray)
{
    ScenePickResult result;
    if (!scene.valid() || length(ray.direction) <= 0.000001f) {
        return result;
    }
    if (impl_->scenePath != scene.filename() ||
        impl_->sceneLifetimeRevision != scene.sceneGraph().lifetimeRevision() ||
        impl_->primitiveBvhs.size() != scene.renderPrimitives().size()) {
        impl_->buildPrimitives(scene);
        impl_->transformRevision = std::numeric_limits<uint64_t>::max();
        impl_->structuralRevision = std::numeric_limits<uint64_t>::max();
    }
    if (impl_->transformRevision != scene.transformRevision() ||
        impl_->structuralRevision != scene.sceneGraph().structuralRevision()) {
        impl_->buildInstances(scene);
    }
    if (impl_->instanceNodes.empty()) {
        return result;
    }

    ScenePickRay worldRay = ray;
    worldRay.direction = worldRay.direction / length(worldRay.direction);
    float nearestDistance = std::numeric_limits<float>::max();
    std::vector<int32_t> stack{0};
    while (!stack.empty()) {
        const BvhNode& node = impl_->instanceNodes[static_cast<size_t>(stack.back())];
        stack.pop_back();
        if (!rayIntersectsBounds(worldRay, node.bounds, nearestDistance)) {
            continue;
        }
        if (node.count == 0) {
            stack.push_back(node.left);
            stack.push_back(node.right);
            continue;
        }
        for (uint32_t index = node.first; index < node.first + node.count; ++index) {
            const InstanceReference& reference = impl_->instances[impl_->instanceOrder[index]];
            const RenderNode& renderNode =
                scene.renderNodes()[static_cast<size_t>(reference.renderNodeIndex)];
            if (!renderNode.visible || !invertible(renderNode.worldMatrix)) {
                continue;
            }
            float4x4 inverseWorld = renderNode.worldMatrix;
            inverseWorld.Invert();
            const float4 localDirection4 = inverseWorld * float4(worldRay.direction, 0.0f);
            const ScenePickRay localRay{
                .origin = inverseWorld * worldRay.origin,
                .direction = float3(localDirection4.x, localDirection4.y, localDirection4.z),
            };
            const int32_t primitiveIndex = renderNode.renderPrimitiveIndex;
            if (primitiveIndex < 0 ||
                static_cast<size_t>(primitiveIndex) >= impl_->primitiveBvhs.size()) {
                continue;
            }
            uint32_t triangleIndex = 0;
            if (impl_->hitPrimitive(
                    scene.renderPrimitives()[static_cast<size_t>(primitiveIndex)],
                    impl_->primitiveBvhs[static_cast<size_t>(primitiveIndex)],
                    localRay,
                    nearestDistance,
                    triangleIndex)) {
                result = ScenePickResult{
                    .object = renderNode.object,
                    .nodeIndex = renderNode.nodeIndex,
                    .renderNodeIndex = reference.renderNodeIndex,
                    .renderPrimitiveIndex = primitiveIndex,
                    .triangleIndex = triangleIndex,
                    .distance = nearestDistance,
                };
            }
        }
    }
    return result;
}

void ScenePicker::clear()
{
    impl_ = std::make_unique<Impl>();
}

} // namespace metallic::scene
