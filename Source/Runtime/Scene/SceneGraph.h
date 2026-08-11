#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "Runtime/Scene/SceneObject.h"

namespace metallic::scene {

class SceneGraph {
public:
    SceneGraph();
    SceneGraph(const SceneGraph&) = delete;
    SceneGraph& operator=(const SceneGraph&) = delete;
    SceneGraph(SceneGraph&& other) noexcept;
    SceneGraph& operator=(SceneGraph&& other) noexcept;

    SceneObject createObject(std::string name = {}, int32_t sourceNodeIndex = -1);
    SceneObject createObject(
        std::string name,
        int32_t sourceNodeIndex,
        const float4x4& authoredLocalMatrix,
        bool visible);
    bool destroyObject(SceneEntity object);
    void clear();

    SceneObject object(SceneEntity entity);
    ConstSceneObject object(SceneEntity entity) const;
    SceneObject objectFromSourceNode(int32_t nodeIndex);
    ConstSceneObject objectFromSourceNode(int32_t nodeIndex) const;

    bool setParent(SceneEntity child, SceneEntity parent);
    bool unsetParent(SceneEntity child);
    bool setRoots(std::span<const SceneEntity> roots);
    bool setName(SceneEntity object, std::string name);
    bool setLocalMatrix(SceneEntity object, const float4x4& localMatrix);
    bool setWorldMatrix(SceneEntity object, const float4x4& worldMatrix);
    bool setCameraProperties(SceneEntity object, const CameraProperties& properties);
    bool setLightProperties(SceneEntity object, const LightProperties& properties);
    bool setVisible(SceneEntity object, bool visible);
    bool updateTransforms();
    void resetRevisions();

    size_t size() const { return registry_.storage<SceneEntity>()->free_list(); }
    size_t sourceNodeCount() const { return sourceNodes_.size(); }
    const std::vector<SceneEntity>& roots() const { return roots_; }
    uint64_t transformRevision() const { return transformRevision_; }
    uint64_t contentRevision() const { return contentRevision_; }
    uint64_t structuralRevision() const { return structuralRevision_; }
    uint64_t visibilityRevision() const { return visibilityRevision_; }
    uint64_t lifetimeRevision() const { return lifetimeRevision_; }

    const entt::registry& registry() const { return registry_; }

private:
    static void incrementRevision(uint64_t& revision);
    bool isAncestor(SceneEntity candidate, SceneEntity object) const;
    void addRoot(SceneEntity object);
    void removeRoot(SceneEntity object);
    void markSubtreeDirty(SceneEntity object);
    void markActiveSubtree(SceneEntity object);
    void clearActiveSubtree(SceneEntity object);
    void collectSubtree(SceneEntity object, std::vector<SceneEntity>& objects) const;

    entt::registry registry_;
    std::vector<SceneEntity> roots_;
    std::vector<SceneEntity> sourceNodes_;
    uint64_t transformRevision_ = 0;
    uint64_t contentRevision_ = 0;
    uint64_t structuralRevision_ = 0;
    uint64_t visibilityRevision_ = 0;
    uint64_t objectEpoch_ = 1;
    uint64_t lifetimeRevision_ = 0;
};

} // namespace metallic::scene
