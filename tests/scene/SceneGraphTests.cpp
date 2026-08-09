#include "Runtime/Scene/SceneGraph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace metallic::scene {
namespace {

struct TestComponent {
    int value = 0;
    std::string label;
};

static_assert(std::same_as<
    decltype(std::declval<SceneObject>().getComponent<TransformComponent>()),
    const TransformComponent&>);
static_assert(std::same_as<
    decltype(std::declval<SceneObject>().tryGetComponent<RelationshipComponent>()),
    const RelationshipComponent*>);
static_assert(std::same_as<
    decltype(std::declval<SceneObject>().getComponent<TestComponent>()),
    TestComponent&>);

float4x4 translationMatrix(const float3& translation)
{
    float4x4 matrix;
    matrix.SetupByTranslation(translation);
    return matrix;
}

bool hasChild(const RelationshipComponent& relationship, SceneEntity child)
{
    return std::find(
               relationship.children.begin(),
               relationship.children.end(),
               child) != relationship.children.end();
}

TEST(SceneGraph, ComponentLifecycle)
{
    SceneGraph graph;
    SceneObject object = graph.createObject("Object", 0);

    ASSERT_TRUE(object);
    const uint64_t revisionBeforeRename = graph.structuralRevision();
    EXPECT_TRUE(graph.setName(object.entity(), "Renamed Object"));
    EXPECT_EQ(object.getComponent<TagComponent>().name, "Renamed Object");
    EXPECT_GT(graph.structuralRevision(), revisionBeforeRename);
    EXPECT_FALSE(graph.setName(object.entity(), "Renamed Object"));
    EXPECT_FALSE(object.hasComponent<TestComponent>());
    EXPECT_EQ(object.tryGetComponent<TestComponent>(), nullptr);
    const uint64_t revisionBeforeAdd = graph.structuralRevision();

    TestComponent& added = object.addComponent<TestComponent>(TestComponent{
        .value = 7,
        .label = "added",
    });
    EXPECT_EQ(added.value, 7);
    EXPECT_EQ(added.label, "added");
    EXPECT_GT(graph.structuralRevision(), revisionBeforeAdd);
    EXPECT_TRUE(object.hasComponent<TestComponent>());
    EXPECT_EQ(object.getComponent<TestComponent>().value, 7);
    ASSERT_NE(object.tryGetComponent<TestComponent>(), nullptr);
    EXPECT_EQ(object.tryGetComponent<TestComponent>()->label, "added");

    const uint64_t revisionBeforeReplace = graph.structuralRevision();
    TestComponent& replaced = object.addOrReplaceComponent<TestComponent>(TestComponent{
        .value = 19,
        .label = "replaced",
    });
    EXPECT_EQ(replaced.value, 19);
    EXPECT_EQ(object.getComponent<TestComponent>().label, "replaced");
    EXPECT_GT(graph.structuralRevision(), revisionBeforeReplace);

    const SceneGraph& constGraph = graph;
    const ConstSceneObject constObject = constGraph.object(object.entity());
    ASSERT_TRUE(constObject);
    ASSERT_NE(constObject.tryGetComponent<TestComponent>(), nullptr);
    EXPECT_EQ(constObject.getComponent<TestComponent>().value, 19);

    const uint64_t revisionBeforeRemove = graph.structuralRevision();
    EXPECT_TRUE(object.removeComponent<TestComponent>());
    EXPECT_GT(graph.structuralRevision(), revisionBeforeRemove);
    EXPECT_FALSE(object.hasComponent<TestComponent>());
    EXPECT_EQ(object.tryGetComponent<TestComponent>(), nullptr);
    EXPECT_FALSE(object.removeComponent<TestComponent>());
}

TEST(SceneGraph, HierarchyMutation)
{
    SceneGraph graph;
    const SceneObject firstParent = graph.createObject("First Parent", 0);
    const SceneObject secondParent = graph.createObject("Second Parent", 1);
    const SceneObject child = graph.createObject("Child", 2);

    ASSERT_TRUE(graph.setParent(child.entity(), firstParent.entity()));
    EXPECT_EQ(
        child.getComponent<RelationshipComponent>().parent,
        firstParent.entity());
    EXPECT_TRUE(hasChild(
        firstParent.getComponent<RelationshipComponent>(),
        child.entity()));

    ASSERT_TRUE(graph.setParent(child.entity(), secondParent.entity()));
    EXPECT_EQ(
        child.getComponent<RelationshipComponent>().parent,
        secondParent.entity());
    EXPECT_FALSE(hasChild(
        firstParent.getComponent<RelationshipComponent>(),
        child.entity()));
    EXPECT_TRUE(hasChild(
        secondParent.getComponent<RelationshipComponent>(),
        child.entity()));

    ASSERT_TRUE(graph.unsetParent(child.entity()));
    EXPECT_EQ(
        child.getComponent<RelationshipComponent>().parent,
        kNullSceneEntity);
    EXPECT_FALSE(hasChild(
        secondParent.getComponent<RelationshipComponent>(),
        child.entity()));

    EXPECT_FALSE(graph.setParent(firstParent.entity(), firstParent.entity()));

    ASSERT_TRUE(graph.setParent(child.entity(), firstParent.entity()));
    EXPECT_FALSE(graph.setParent(firstParent.entity(), child.entity()));
    EXPECT_EQ(
        firstParent.getComponent<RelationshipComponent>().parent,
        kNullSceneEntity);
    EXPECT_EQ(
        child.getComponent<RelationshipComponent>().parent,
        firstParent.entity());
}

TEST(SceneGraph, TransformAndVisibilityPropagation)
{
    SceneGraph graph;
    const SceneObject parent = graph.createObject("Parent", 0);
    const SceneObject child = graph.createObject("Child", 1);
    ASSERT_TRUE(graph.setParent(child.entity(), parent.entity()));

    ASSERT_TRUE(graph.setLocalMatrix(
        parent.entity(),
        translationMatrix(float3(2.0f, 3.0f, 4.0f))));
    ASSERT_TRUE(graph.setLocalMatrix(
        child.entity(),
        translationMatrix(float3(5.0f, 0.0f, 0.0f))));
    ASSERT_TRUE(graph.updateTransforms());

    const TransformComponent& parentTransform =
        parent.getComponent<TransformComponent>();
    const TransformComponent& childTransform =
        child.getComponent<TransformComponent>();
    EXPECT_FLOAT_EQ(parentTransform.worldMatrix.a03, 2.0f);
    EXPECT_FLOAT_EQ(parentTransform.worldMatrix.a13, 3.0f);
    EXPECT_FLOAT_EQ(parentTransform.worldMatrix.a23, 4.0f);
    EXPECT_FLOAT_EQ(childTransform.worldMatrix.a03, 7.0f);
    EXPECT_FLOAT_EQ(childTransform.worldMatrix.a13, 3.0f);
    EXPECT_FLOAT_EQ(childTransform.worldMatrix.a23, 4.0f);
    EXPECT_EQ(parentTransform.transformRevision, graph.transformRevision());
    EXPECT_EQ(childTransform.transformRevision, graph.transformRevision());

    ASSERT_TRUE(graph.setVisible(parent.entity(), false));
    ASSERT_TRUE(graph.updateTransforms());
    const VisibilityComponent& hiddenParent =
        parent.getComponent<VisibilityComponent>();
    const VisibilityComponent& hiddenChild =
        child.getComponent<VisibilityComponent>();
    EXPECT_FALSE(hiddenParent.localVisible);
    EXPECT_FALSE(hiddenParent.worldVisible);
    EXPECT_TRUE(hiddenChild.localVisible);
    EXPECT_FALSE(hiddenChild.worldVisible);

    ASSERT_TRUE(graph.setVisible(parent.entity(), true));
    ASSERT_TRUE(graph.setVisible(child.entity(), false));
    ASSERT_TRUE(graph.updateTransforms());
    EXPECT_TRUE(parent.getComponent<VisibilityComponent>().worldVisible);
    EXPECT_FALSE(child.getComponent<VisibilityComponent>().localVisible);
    EXPECT_FALSE(child.getComponent<VisibilityComponent>().worldVisible);
}

TEST(SceneGraph, ActiveRoots)
{
    SceneGraph graph;
    const SceneObject root = graph.createObject("Root", 0);
    const SceneObject child = graph.createObject("Child", 1);
    const SceneObject grandchild = graph.createObject("Grandchild", 2);
    const SceneObject inactive = graph.createObject("Inactive", 3);
    ASSERT_TRUE(graph.setParent(child.entity(), root.entity()));
    ASSERT_TRUE(graph.setParent(grandchild.entity(), child.entity()));

    const std::array overlappingRoots{grandchild.entity(), root.entity()};
    EXPECT_FALSE(graph.setRoots(std::span<const SceneEntity>(overlappingRoots)));

    const std::array firstRoots{root.entity()};
    ASSERT_TRUE(graph.setRoots(std::span<const SceneEntity>(firstRoots)));
    EXPECT_TRUE(root.hasComponent<RootComponent>());
    EXPECT_TRUE(root.hasComponent<ActiveSceneComponent>());
    EXPECT_TRUE(child.hasComponent<ActiveSceneComponent>());
    EXPECT_TRUE(grandchild.hasComponent<ActiveSceneComponent>());
    EXPECT_FALSE(inactive.hasComponent<ActiveSceneComponent>());

    const std::array secondRoots{inactive.entity()};
    ASSERT_TRUE(graph.setRoots(std::span<const SceneEntity>(secondRoots)));
    EXPECT_FALSE(root.hasComponent<RootComponent>());
    EXPECT_FALSE(root.hasComponent<ActiveSceneComponent>());
    EXPECT_FALSE(child.hasComponent<ActiveSceneComponent>());
    EXPECT_FALSE(grandchild.hasComponent<ActiveSceneComponent>());
    EXPECT_TRUE(inactive.hasComponent<RootComponent>());
    EXPECT_TRUE(inactive.hasComponent<ActiveSceneComponent>());
    ASSERT_EQ(graph.roots().size(), 1u);
    EXPECT_EQ(graph.roots().front(), inactive.entity());
}

TEST(SceneGraph, RecursiveDestroyAndStaleHandle)
{
    SceneGraph graph;
    const SceneObject root = graph.createObject("Root", 0);
    const SceneObject child = graph.createObject("Child", 1);
    const SceneObject grandchild = graph.createObject("Grandchild", 2);
    ASSERT_TRUE(graph.setParent(child.entity(), root.entity()));
    ASSERT_TRUE(graph.setParent(grandchild.entity(), child.entity()));

    const SceneEntity staleChildEntity = child.entity();
    const SceneEntity staleGrandchildEntity = grandchild.entity();
    ASSERT_TRUE(graph.destroyObject(child.entity()));
    EXPECT_TRUE(root);
    EXPECT_FALSE(child);
    EXPECT_FALSE(grandchild);
    EXPECT_FALSE(graph.object(staleChildEntity));
    EXPECT_FALSE(graph.object(staleGrandchildEntity));
    EXPECT_FALSE(graph.objectFromSourceNode(1));
    EXPECT_FALSE(graph.objectFromSourceNode(2));
    EXPECT_TRUE(root.getComponent<RelationshipComponent>().children.empty());

    const SceneObject replacement = graph.createObject("Replacement", 1);
    ASSERT_TRUE(replacement);
    EXPECT_NE(replacement.entity(), staleChildEntity);
    EXPECT_FALSE(child);

    ASSERT_TRUE(graph.destroyObject(root.entity()));
    EXPECT_FALSE(root);
    ASSERT_TRUE(graph.destroyObject(replacement.entity()));
    EXPECT_TRUE(graph.roots().empty());
}

TEST(SceneGraph, MovePreservesRegistryAndSourceMapping)
{
    SceneGraph source;
    const uint64_t sourceLifetime = source.lifetimeRevision();
    const SceneObject root = source.createObject("Root", 0);
    const SceneObject child = source.createObject("Child", 1);
    child.addComponent<TestComponent>(TestComponent{
        .value = 41,
        .label = "preserved",
    });
    ASSERT_TRUE(source.setParent(child.entity(), root.entity()));
    const SceneEntity rootEntity = root.entity();
    const SceneEntity childEntity = child.entity();

    SceneGraph moveConstructed(std::move(source));
    EXPECT_NE(moveConstructed.lifetimeRevision(), sourceLifetime);
    EXPECT_NE(source.lifetimeRevision(), sourceLifetime);
    EXPECT_FALSE(root);
    EXPECT_FALSE(child);
    const SceneObject movedRoot = moveConstructed.objectFromSourceNode(0);
    const SceneObject movedChild = moveConstructed.objectFromSourceNode(1);
    ASSERT_TRUE(movedRoot);
    ASSERT_TRUE(movedChild);
    EXPECT_EQ(movedRoot.entity(), rootEntity);
    EXPECT_EQ(movedChild.entity(), childEntity);
    EXPECT_EQ(
        movedChild.getComponent<RelationshipComponent>().parent,
        rootEntity);
    EXPECT_EQ(movedChild.getComponent<TestComponent>().value, 41);
    EXPECT_TRUE(movedRoot.hasComponent<RootComponent>());
    EXPECT_TRUE(movedChild.hasComponent<ActiveSceneComponent>());

    SceneGraph moveAssigned;
    const uint64_t assignmentTargetLifetime = moveAssigned.lifetimeRevision();
    const SceneObject replacedObject = moveAssigned.createObject("Replaced", 0);
    ASSERT_TRUE(replacedObject);
    moveAssigned = std::move(moveConstructed);
    EXPECT_NE(moveAssigned.lifetimeRevision(), assignmentTargetLifetime);
    EXPECT_FALSE(replacedObject);
    EXPECT_THROW(replacedObject.getComponent<TagComponent>(), std::logic_error);
    EXPECT_THROW(
        replacedObject.addOrReplaceComponent<TestComponent>(TestComponent{}),
        std::logic_error);
    EXPECT_FALSE(movedRoot);
    EXPECT_FALSE(movedChild);
    const SceneObject assignedRoot = moveAssigned.objectFromSourceNode(0);
    const SceneObject assignedChild = moveAssigned.objectFromSourceNode(1);
    ASSERT_TRUE(assignedRoot);
    ASSERT_TRUE(assignedChild);
    EXPECT_EQ(assignedRoot.entity(), rootEntity);
    EXPECT_EQ(assignedChild.entity(), childEntity);
    EXPECT_EQ(assignedChild.getComponent<TestComponent>().label, "preserved");
    ASSERT_EQ(moveAssigned.roots().size(), 1u);
    EXPECT_EQ(moveAssigned.roots().front(), rootEntity);
}

TEST(SceneGraph, Clear)
{
    SceneGraph graph;
    const SceneObject root = graph.createObject("Root", 0);
    const SceneObject child = graph.createObject("Child", 1);
    const SceneObject generated = graph.createObject("Generated");
    generated.addComponent<GeneratedComponent>();
    ASSERT_TRUE(graph.setParent(child.entity(), root.entity()));
    const uint64_t lifetimeBeforeClear = graph.lifetimeRevision();

    graph.clear();

    const SceneObject replacement = graph.createObject("Replacement", 0);
    ASSERT_TRUE(replacement);
    EXPECT_THROW(root.getComponent<TagComponent>(), std::logic_error);
    EXPECT_THROW(root.addComponent<TestComponent>(), std::logic_error);
    EXPECT_FALSE(replacement.hasComponent<TestComponent>());

    EXPECT_NE(graph.lifetimeRevision(), lifetimeBeforeClear);
    EXPECT_EQ(graph.size(), 1u);
    EXPECT_EQ(graph.sourceNodeCount(), 1u);
    ASSERT_EQ(graph.roots().size(), 1u);
    EXPECT_EQ(graph.roots().front(), replacement.entity());
    EXPECT_FALSE(root);
    EXPECT_FALSE(child);
    EXPECT_FALSE(generated);
    EXPECT_EQ(graph.objectFromSourceNode(0), replacement);
    EXPECT_FALSE(graph.objectFromSourceNode(1));
}

} // namespace
} // namespace metallic::scene
