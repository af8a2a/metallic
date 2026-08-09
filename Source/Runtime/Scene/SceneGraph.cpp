#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <utility>

#include "Runtime/Scene/SceneGraph.h"

namespace metallic::scene {
namespace {

bool matrixIsFinite(const float4x4& matrix)
{
    return std::all_of(std::begin(matrix.a), std::end(matrix.a), [](float value) {
        return std::isfinite(value);
    });
}

bool matricesNearlyEqual(const float4x4& lhs, const float4x4& rhs)
{
    constexpr float kEpsilon = 0.000001f;
    for (size_t index = 0; index < 16; ++index) {
        if (!std::isfinite(lhs.a[index]) || !std::isfinite(rhs.a[index]) ||
            std::abs(lhs.a[index] - rhs.a[index]) > kEpsilon) {
            return false;
        }
    }
    return true;
}

template <typename Type>
void eraseValue(std::vector<Type>& values, const Type& value)
{
    values.erase(std::remove(values.begin(), values.end(), value), values.end());
}

uint64_t nextSceneGraphLifetimeRevision()
{
    static std::atomic<uint64_t> revision{0};
    uint64_t next = revision.fetch_add(1, std::memory_order_relaxed) + 1;
    if (next == 0) {
        next = revision.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    return next;
}

} // namespace

SceneGraph::SceneGraph()
    : lifetimeRevision_(nextSceneGraphLifetimeRevision())
{
}

SceneGraph::SceneGraph(SceneGraph&& other) noexcept
    : registry_(std::move(other.registry_)),
      roots_(std::move(other.roots_)),
      sourceNodes_(std::move(other.sourceNodes_)),
      transformRevision_(other.transformRevision_),
      structuralRevision_(other.structuralRevision_),
      lifetimeRevision_(nextSceneGraphLifetimeRevision())
{
    incrementRevision(other.objectEpoch_);
    other.lifetimeRevision_ = nextSceneGraphLifetimeRevision();
    other.roots_.clear();
    other.sourceNodes_.clear();
    other.transformRevision_ = 0;
    other.structuralRevision_ = 0;
}

SceneGraph& SceneGraph::operator=(SceneGraph&& other) noexcept
{
    if (this == &other) {
        return *this;
    }

    incrementRevision(objectEpoch_);
    registry_ = std::move(other.registry_);
    roots_ = std::move(other.roots_);
    sourceNodes_ = std::move(other.sourceNodes_);
    transformRevision_ = other.transformRevision_;
    structuralRevision_ = other.structuralRevision_;
    lifetimeRevision_ = nextSceneGraphLifetimeRevision();

    other.registry_.clear();
    other.roots_.clear();
    other.sourceNodes_.clear();
    other.transformRevision_ = 0;
    other.structuralRevision_ = 0;
    incrementRevision(other.objectEpoch_);
    other.lifetimeRevision_ = nextSceneGraphLifetimeRevision();
    return *this;
}

SceneObject SceneGraph::createObject(std::string name, int32_t sourceNodeIndex)
{
    return createObject(
        std::move(name),
        sourceNodeIndex,
        float4x4::Identity(),
        true);
}

SceneObject SceneGraph::createObject(
    std::string name,
    int32_t sourceNodeIndex,
    const float4x4& authoredLocalMatrix,
    bool visible)
{
    if (!matrixIsFinite(authoredLocalMatrix)) {
        return {};
    }
    if (sourceNodeIndex >= 0) {
        const size_t index = static_cast<size_t>(sourceNodeIndex);
        if (index < sourceNodes_.size() && registry_.valid(sourceNodes_[index])) {
            return {};
        }
        if (index >= sourceNodes_.size()) {
            sourceNodes_.resize(index + 1u, kNullSceneEntity);
        }
    }

    const SceneEntity entity = registry_.create();
    registry_.emplace<TagComponent>(entity, std::move(name));
    registry_.emplace<TransformComponent>(entity, TransformComponent{
        .authoredLocalMatrix = authoredLocalMatrix,
        .localMatrix = authoredLocalMatrix,
        .worldMatrix = float4x4::Identity(),
    });
    registry_.emplace<RelationshipComponent>(entity);
    registry_.emplace<VisibilityComponent>(entity, visible, visible);
    registry_.emplace<RootComponent>(entity);
    registry_.emplace<ActiveSceneComponent>(entity);
    if (sourceNodeIndex >= 0) {
        registry_.emplace<SourceNodeComponent>(entity, sourceNodeIndex);
        sourceNodes_[static_cast<size_t>(sourceNodeIndex)] = entity;
    }
    roots_.push_back(entity);
    incrementRevision(structuralRevision_);
    return SceneObject(registry_, entity, objectEpoch_, &structuralRevision_);
}

bool SceneGraph::destroyObject(SceneEntity objectEntity)
{
    if (!registry_.valid(objectEntity)) {
        return false;
    }

    std::vector<SceneEntity> objects;
    collectSubtree(objectEntity, objects);
    if (objects.empty()) {
        return false;
    }

    const RelationshipComponent* relationship =
        registry_.try_get<RelationshipComponent>(objectEntity);
    if (relationship != nullptr && relationship->parent != kNullSceneEntity &&
        registry_.valid(relationship->parent)) {
        if (RelationshipComponent* parent =
                registry_.try_get<RelationshipComponent>(relationship->parent)) {
            eraseValue(parent->children, objectEntity);
        }
    }

    for (const SceneEntity entity : objects) {
        removeRoot(entity);
        if (const SourceNodeComponent* source = registry_.try_get<SourceNodeComponent>(entity)) {
            const size_t index = static_cast<size_t>(source->nodeIndex);
            if (index < sourceNodes_.size() && sourceNodes_[index] == entity) {
                sourceNodes_[index] = kNullSceneEntity;
            }
        }
    }
    for (auto iterator = objects.rbegin(); iterator != objects.rend(); ++iterator) {
        if (registry_.valid(*iterator)) {
            registry_.destroy(*iterator);
        }
    }
    incrementRevision(structuralRevision_);
    incrementRevision(transformRevision_);
    return true;
}

void SceneGraph::clear()
{
    incrementRevision(objectEpoch_);
    lifetimeRevision_ = nextSceneGraphLifetimeRevision();
    registry_.clear();
    roots_.clear();
    sourceNodes_.clear();
    transformRevision_ = 0;
    structuralRevision_ = 0;
}

SceneObject SceneGraph::object(SceneEntity entity)
{
    return registry_.valid(entity)
        ? SceneObject(registry_, entity, objectEpoch_, &structuralRevision_)
        : SceneObject{};
}

ConstSceneObject SceneGraph::object(SceneEntity entity) const
{
    return registry_.valid(entity)
        ? ConstSceneObject(registry_, entity, objectEpoch_)
        : ConstSceneObject{};
}

SceneObject SceneGraph::objectFromSourceNode(int32_t nodeIndex)
{
    if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= sourceNodes_.size()) {
        return {};
    }
    return object(sourceNodes_[static_cast<size_t>(nodeIndex)]);
}

ConstSceneObject SceneGraph::objectFromSourceNode(int32_t nodeIndex) const
{
    if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= sourceNodes_.size()) {
        return {};
    }
    return object(sourceNodes_[static_cast<size_t>(nodeIndex)]);
}

bool SceneGraph::setParent(SceneEntity childEntity, SceneEntity parentEntity)
{
    if (!registry_.valid(childEntity) || !registry_.valid(parentEntity) ||
        childEntity == parentEntity || isAncestor(childEntity, parentEntity)) {
        return false;
    }

    RelationshipComponent& child =
        registry_.get_or_emplace<RelationshipComponent>(childEntity);
    if (child.parent == parentEntity) {
        return false;
    }
    if (child.parent != kNullSceneEntity && registry_.valid(child.parent)) {
        if (RelationshipComponent* previousParent =
                registry_.try_get<RelationshipComponent>(child.parent)) {
            eraseValue(previousParent->children, childEntity);
        }
    }

    RelationshipComponent& parent =
        registry_.get_or_emplace<RelationshipComponent>(parentEntity);
    if (std::find(parent.children.begin(), parent.children.end(), childEntity) ==
        parent.children.end()) {
        parent.children.push_back(childEntity);
    }
    child.parent = parentEntity;
    removeRoot(childEntity);
    if (registry_.all_of<ActiveSceneComponent>(parentEntity)) {
        markActiveSubtree(childEntity);
    } else {
        clearActiveSubtree(childEntity);
    }
    markSubtreeDirty(childEntity);
    incrementRevision(structuralRevision_);
    incrementRevision(transformRevision_);
    return true;
}

bool SceneGraph::unsetParent(SceneEntity childEntity)
{
    if (!registry_.valid(childEntity)) {
        return false;
    }
    RelationshipComponent* child = registry_.try_get<RelationshipComponent>(childEntity);
    if (child == nullptr || child->parent == kNullSceneEntity) {
        return false;
    }
    if (registry_.valid(child->parent)) {
        if (RelationshipComponent* parent =
                registry_.try_get<RelationshipComponent>(child->parent)) {
            eraseValue(parent->children, childEntity);
        }
    }
    child->parent = kNullSceneEntity;
    addRoot(childEntity);
    markSubtreeDirty(childEntity);
    incrementRevision(structuralRevision_);
    incrementRevision(transformRevision_);
    return true;
}

bool SceneGraph::setRoots(std::span<const SceneEntity> roots)
{
    std::vector<SceneEntity> uniqueRoots;
    uniqueRoots.reserve(roots.size());
    for (const SceneEntity entity : roots) {
        if (!registry_.valid(entity) ||
            std::find(uniqueRoots.begin(), uniqueRoots.end(), entity) != uniqueRoots.end()) {
            return false;
        }
        for (const SceneEntity existingRoot : uniqueRoots) {
            if (isAncestor(existingRoot, entity) || isAncestor(entity, existingRoot)) {
                return false;
            }
        }
        uniqueRoots.push_back(entity);
    }
    if (uniqueRoots == roots_) {
        return false;
    }

    std::vector<SceneEntity> taggedRoots;
    for (const SceneEntity entity : registry_.view<RootComponent>()) {
        taggedRoots.push_back(entity);
    }
    for (const SceneEntity entity : taggedRoots) {
        registry_.remove<RootComponent>(entity);
    }
    std::vector<SceneEntity> activeObjects;
    for (const SceneEntity entity : registry_.view<ActiveSceneComponent>()) {
        activeObjects.push_back(entity);
    }
    for (const SceneEntity entity : activeObjects) {
        registry_.remove<ActiveSceneComponent>(entity);
    }

    for (const SceneEntity entity : registry_.view<TransformComponent>()) {
        TransformComponent& transform = registry_.get<TransformComponent>(entity);
        transform.worldMatrix = float4x4::Identity();
        transform.dirty = true;
    }
    for (const SceneEntity entity : registry_.view<VisibilityComponent>()) {
        VisibilityComponent& visibility = registry_.get<VisibilityComponent>(entity);
        visibility.worldVisible = visibility.localVisible;
    }

    roots_ = std::move(uniqueRoots);
    for (const SceneEntity entity : roots_) {
        registry_.emplace<RootComponent>(entity);
        markActiveSubtree(entity);
        markSubtreeDirty(entity);
    }
    incrementRevision(structuralRevision_);
    incrementRevision(transformRevision_);
    return true;
}

bool SceneGraph::setName(SceneEntity objectEntity, std::string name)
{
    if (!registry_.valid(objectEntity)) {
        return false;
    }
    TagComponent* tag = registry_.try_get<TagComponent>(objectEntity);
    if (tag == nullptr || tag->name == name) {
        return false;
    }
    tag->name = std::move(name);
    incrementRevision(structuralRevision_);
    return true;
}

bool SceneGraph::setLocalMatrix(SceneEntity objectEntity, const float4x4& localMatrix)
{
    if (!registry_.valid(objectEntity) || !matrixIsFinite(localMatrix)) {
        return false;
    }
    TransformComponent* transform = registry_.try_get<TransformComponent>(objectEntity);
    if (transform == nullptr || matricesNearlyEqual(transform->localMatrix, localMatrix)) {
        return false;
    }
    transform->localMatrix = localMatrix;
    incrementRevision(transformRevision_);
    markSubtreeDirty(objectEntity);
    return true;
}

bool SceneGraph::setVisible(SceneEntity objectEntity, bool visible)
{
    if (!registry_.valid(objectEntity)) {
        return false;
    }
    VisibilityComponent& visibility =
        registry_.get_or_emplace<VisibilityComponent>(objectEntity);
    if (visibility.localVisible == visible) {
        return false;
    }
    visibility.localVisible = visible;
    incrementRevision(structuralRevision_);
    return true;
}

bool SceneGraph::updateTransforms()
{
    bool changed = false;
    std::unordered_set<SceneEntity> visited;
    std::function<void(SceneEntity, const float4x4&, bool)> updateObject;
    updateObject = [&](SceneEntity entity, const float4x4& parentWorld, bool parentVisible) {
        if (!registry_.valid(entity) || !visited.insert(entity).second) {
            return;
        }

        float4x4 worldMatrix = parentWorld;
        if (TransformComponent* transform = registry_.try_get<TransformComponent>(entity)) {
            const float4x4 nextWorld = parentWorld * transform->localMatrix;
            if (!matricesNearlyEqual(transform->worldMatrix, nextWorld)) {
                transform->worldMatrix = nextWorld;
                transform->transformRevision = transformRevision_;
                changed = true;
            }
            transform->dirty = false;
            worldMatrix = transform->worldMatrix;
        }

        bool worldVisible = parentVisible;
        if (VisibilityComponent* visibility = registry_.try_get<VisibilityComponent>(entity)) {
            const bool nextVisible = parentVisible && visibility->localVisible;
            changed = changed || visibility->worldVisible != nextVisible;
            visibility->worldVisible = nextVisible;
            worldVisible = nextVisible;
        }

        if (const RelationshipComponent* relationship =
                registry_.try_get<RelationshipComponent>(entity)) {
            for (const SceneEntity child : relationship->children) {
                updateObject(child, worldMatrix, worldVisible);
            }
        }
    };

    for (const SceneEntity root : roots_) {
        updateObject(root, float4x4::Identity(), true);
    }
    return changed;
}

void SceneGraph::resetRevisions()
{
    transformRevision_ = 0;
    structuralRevision_ = 0;
    for (const SceneEntity entity : registry_.view<TransformComponent>()) {
        registry_.get<TransformComponent>(entity).transformRevision = 0;
    }
}

void SceneGraph::incrementRevision(uint64_t& revision)
{
    ++revision;
    if (revision == 0) {
        revision = 1;
    }
}

bool SceneGraph::isAncestor(SceneEntity candidate, SceneEntity objectEntity) const
{
    std::unordered_set<SceneEntity> visited;
    SceneEntity current = objectEntity;
    while (registry_.valid(current) && visited.insert(current).second) {
        if (current == candidate) {
            return true;
        }
        const RelationshipComponent* relationship =
            registry_.try_get<RelationshipComponent>(current);
        if (relationship == nullptr || relationship->parent == kNullSceneEntity) {
            break;
        }
        current = relationship->parent;
    }
    return false;
}

void SceneGraph::addRoot(SceneEntity objectEntity)
{
    if (!registry_.valid(objectEntity) ||
        std::find(roots_.begin(), roots_.end(), objectEntity) != roots_.end()) {
        return;
    }
    roots_.push_back(objectEntity);
    registry_.get_or_emplace<RootComponent>(objectEntity);
    markActiveSubtree(objectEntity);
}

void SceneGraph::removeRoot(SceneEntity objectEntity)
{
    eraseValue(roots_, objectEntity);
    if (registry_.valid(objectEntity)) {
        registry_.remove<RootComponent>(objectEntity);
    }
}

void SceneGraph::markSubtreeDirty(SceneEntity objectEntity)
{
    std::vector<SceneEntity> objects;
    collectSubtree(objectEntity, objects);
    for (const SceneEntity entity : objects) {
        if (TransformComponent* transform = registry_.try_get<TransformComponent>(entity)) {
            transform->dirty = true;
        }
    }
}

void SceneGraph::markActiveSubtree(SceneEntity objectEntity)
{
    std::vector<SceneEntity> objects;
    collectSubtree(objectEntity, objects);
    for (const SceneEntity entity : objects) {
        registry_.get_or_emplace<ActiveSceneComponent>(entity);
    }
}

void SceneGraph::clearActiveSubtree(SceneEntity objectEntity)
{
    std::vector<SceneEntity> objects;
    collectSubtree(objectEntity, objects);
    for (const SceneEntity entity : objects) {
        registry_.remove<ActiveSceneComponent>(entity);
    }
}

void SceneGraph::collectSubtree(
    SceneEntity objectEntity,
    std::vector<SceneEntity>& objects) const
{
    if (!registry_.valid(objectEntity) ||
        std::find(objects.begin(), objects.end(), objectEntity) != objects.end()) {
        return;
    }
    objects.push_back(objectEntity);
    if (const RelationshipComponent* relationship =
            registry_.try_get<RelationshipComponent>(objectEntity)) {
        for (const SceneEntity child : relationship->children) {
            collectSubtree(child, objects);
        }
    }
}

} // namespace metallic::scene
