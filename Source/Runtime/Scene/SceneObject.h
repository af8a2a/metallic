#pragma once

#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <entt/entity/registry.hpp>

#include "Runtime/Scene/SceneComponents.h"

namespace metallic::scene {

template <typename Component>
inline constexpr bool kSceneGraphCoreComponent =
    std::same_as<Component, TagComponent> ||
    std::same_as<Component, SourceNodeComponent> ||
    std::same_as<Component, TransformComponent> ||
    std::same_as<Component, RelationshipComponent> ||
    std::same_as<Component, VisibilityComponent> ||
    std::same_as<Component, RootComponent> ||
    std::same_as<Component, ActiveSceneComponent>;

template <typename Registry>
class BasicSceneObject {
public:
    using RegistryType = Registry;

    // Lightweight, non-owning handle. SceneGraph clear/move invalidates it.
    BasicSceneObject() = default;

    bool valid() const
    {
        return registry_ != nullptr && epoch_ != nullptr && *epoch_ == capturedEpoch_ &&
            entity_ != kNullSceneEntity && registry_->valid(entity_);
    }

    explicit operator bool() const { return valid(); }
    SceneEntity entity() const { return entity_; }

    template <typename OtherRegistry>
        requires(
            std::is_const_v<Registry> &&
            std::same_as<std::remove_const_t<Registry>, std::remove_const_t<OtherRegistry>>)
    BasicSceneObject(const BasicSceneObject<OtherRegistry>& other)
        : registry_(other.registry_),
          epoch_(other.epoch_),
          structuralRevision_(other.structuralRevision_),
          capturedEpoch_(other.capturedEpoch_),
          entity_(other.entity_)
    {
    }

    template <typename Component>
    bool hasComponent() const
    {
        return valid() && registry_->template all_of<Component>(entity_);
    }

    template <typename Component>
        requires(!std::is_empty_v<Component>)
    decltype(auto) getComponent() const
    {
        requireValid();
        if constexpr (std::is_const_v<Registry> || kSceneGraphCoreComponent<Component>) {
            return std::as_const(registry_->template get<Component>(entity_));
        } else {
            return registry_->template get<Component>(entity_);
        }
    }

    template <typename Component>
        requires(!std::is_empty_v<Component>)
    auto tryGetComponent() const
    {
        using Pointer = std::conditional_t<
            std::is_const_v<Registry> || kSceneGraphCoreComponent<Component>,
            const Component*,
            Component*>;
        if (!valid()) {
            return Pointer{};
        }
        return static_cast<Pointer>(registry_->template try_get<Component>(entity_));
    }

    template <typename Component, typename... Args>
        requires(!std::is_const_v<Registry> && !kSceneGraphCoreComponent<Component>)
    decltype(auto) addComponent(Args&&... args) const
    {
        requireValid();
        if constexpr (std::is_void_v<decltype(registry_->template emplace<Component>(
                          entity_,
                          std::forward<Args>(args)...))>) {
            registry_->template emplace<Component>(entity_, std::forward<Args>(args)...);
            notifyStructuralChange();
        } else {
            decltype(auto) result = registry_->template emplace<Component>(
                entity_,
                std::forward<Args>(args)...);
            notifyStructuralChange();
            return result;
        }
    }

    template <typename Component, typename... Args>
        requires(!std::is_const_v<Registry> && !kSceneGraphCoreComponent<Component>)
    decltype(auto) addOrReplaceComponent(Args&&... args) const
    {
        requireValid();
        if constexpr (std::is_void_v<decltype(registry_->template emplace_or_replace<Component>(
                          entity_,
                          std::forward<Args>(args)...))>) {
            registry_->template emplace_or_replace<Component>(
                entity_,
                std::forward<Args>(args)...);
            notifyStructuralChange();
        } else {
            decltype(auto) result = registry_->template emplace_or_replace<Component>(
                entity_,
                std::forward<Args>(args)...);
            notifyStructuralChange();
            return result;
        }
    }

    template <typename Component>
        requires(!std::is_const_v<Registry> && !kSceneGraphCoreComponent<Component>)
    bool removeComponent() const
    {
        if (!valid() || registry_->template remove<Component>(entity_) == 0u) {
            return false;
        }
        notifyStructuralChange();
        return true;
    }

    friend bool operator==(const BasicSceneObject&, const BasicSceneObject&) = default;

private:
    template <typename>
    friend class BasicSceneObject;
    friend class SceneGraph;

    BasicSceneObject(
        Registry& registry,
        SceneEntity entity,
        const uint64_t& epoch,
        uint64_t* structuralRevision = nullptr)
        : registry_(&registry),
          epoch_(&epoch),
          structuralRevision_(structuralRevision),
          capturedEpoch_(epoch),
          entity_(entity)
    {
    }

    void notifyStructuralChange() const
    {
        if (structuralRevision_ == nullptr) {
            return;
        }
        ++*structuralRevision_;
        if (*structuralRevision_ == 0) {
            *structuralRevision_ = 1;
        }
    }

    void requireValid() const
    {
        if (!valid()) {
            throw std::logic_error("SceneObject handle is no longer valid");
        }
    }

    Registry* registry_ = nullptr;
    const uint64_t* epoch_ = nullptr;
    uint64_t* structuralRevision_ = nullptr;
    uint64_t capturedEpoch_ = 0;
    SceneEntity entity_ = kNullSceneEntity;
};

using SceneObject = BasicSceneObject<entt::registry>;
using ConstSceneObject = BasicSceneObject<const entt::registry>;

} // namespace metallic::scene
