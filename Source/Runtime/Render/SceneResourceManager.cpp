#include "Runtime/Render/SceneResourceManager.h"

#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

std::filesystem::path propertyPath(
    const RenderGraphProperties& properties,
    const char* name)
{
    if (properties.contains(name) && properties[name].is_string()) {
        return properties[name].get<std::string>();
    }
    return {};
}

std::string resourceKey(const std::filesystem::path& scenePath)
{
    return normalizedScenePath(scenePath).generic_string();
}

void stampSnapshot(
    SceneResourceSnapshot& snapshot,
    const scene::Scene& sourceScene)
{
    snapshot.sourceResourceIdentity = sourceScene.resourceIdentity();
    snapshot.sourceStructuralRevision = sourceScene.sceneGraph().structuralRevision();
    snapshot.sourceTransformRevision = sourceScene.transformRevision();
    snapshot.sourceVisibilityRevision = sourceScene.visibilityRevision();
}

bool snapshotMatchesScene(
    const SceneResourceSnapshot& snapshot,
    const scene::Scene& sourceScene)
{
    return snapshot.sourceResourceIdentity == sourceScene.resourceIdentity() &&
        snapshot.sourceStructuralRevision == sourceScene.sceneGraph().structuralRevision() &&
        snapshot.sourceVisibilityRevision == sourceScene.visibilityRevision();
}

} // namespace

struct SceneResourceManager::Impl {
    void collectRetired()
    {
        retiredSnapshots.erase(
            std::remove_if(
                retiredSnapshots.begin(),
                retiredSnapshots.end(),
                [](const std::shared_ptr<SceneResourceSnapshot>& snapshot) {
                    return snapshot == nullptr ||
                        snapshot->pathTraceResources == nullptr ||
                        snapshot->pathTraceResources->gpuWorkComplete();
                }),
            retiredSnapshots.end());
    }

    Device* device = nullptr;
    std::unordered_map<std::string, std::shared_ptr<SceneResourceSnapshot>> snapshots;
    std::unordered_map<std::string, std::shared_ptr<scene::SceneDocument>> scenes;
    std::vector<std::shared_ptr<SceneResourceSnapshot>> retiredSnapshots;
};

Result SceneResourceManager::resolveScene(
    const RenderGraphProperties& properties,
    const scene::Scene* runtimeScene,
    const scene::Scene*& outScene,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_shared<Impl>();
    }
    impl_->collectRetired();
    const std::filesystem::path scenePath = propertyPath(properties, "path");
    outScene = runtimeSceneForPath(runtimeScene, scenePath);
    if (outScene != nullptr) {
        return {};
    }

    const std::string key = normalizedScenePath(scenePath).generic_string();
    const auto found = impl_->scenes.find(key);
    if (found != impl_->scenes.end()) {
        outScene = found->second.get();
        return {};
    }

    auto loadedScene = std::make_shared<scene::SceneDocument>();
    const std::filesystem::path resolvedPath = normalizedScenePath(scenePath);
    if (!loadedScene->load(resolvedPath)) {
        const std::string& detail = loadedScene->documentWarning().empty()
            ? loadedScene->lastLoadResult().error
            : loadedScene->documentWarning();
        log = "SceneResourceManager failed to load scene: " +
            detail;
        return makeError(Error::Failure);
    }
    outScene = loadedScene.get();
    impl_->scenes.emplace(key, std::move(loadedScene));
    return {};
}

Result SceneResourceManager::acquire(
    Device& device,
    Queue& graphicsQueue,
    const RenderGraphProperties& properties,
    const scene::Scene* runtimeScene,
    SceneResourceFeatureBits features,
    std::shared_ptr<SceneResourceSnapshot>& outSnapshot,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_shared<Impl>();
    }
    impl_->collectRetired();
    if (impl_->device != nullptr && impl_->device != &device) {
        clear();
        impl_ = std::make_shared<Impl>();
    }
    impl_->device = &device;

    const scene::Scene* resolvedScene = nullptr;
    Result sceneResult = resolveScene(properties, runtimeScene, resolvedScene, log);
    if (!sceneResult) {
        return sceneResult;
    }

    const std::filesystem::path scenePath = propertyPath(properties, "path");
    const std::string key = resourceKey(scenePath);
    auto found = impl_->snapshots.find(key);
    if (found != impl_->snapshots.end()) {
        if (found->second != nullptr &&
            snapshotMatchesScene(*found->second, *resolvedScene)) {
            outSnapshot = found->second;
        } else {
            if (found->second != nullptr) {
                impl_->retiredSnapshots.push_back(found->second);
            }
            impl_->snapshots.erase(found);
            outSnapshot.reset();
        }
    }
    if (outSnapshot == nullptr) {
        outSnapshot = std::make_shared<SceneResourceSnapshot>();
        outSnapshot->scenePath = normalizedScenePath(scenePath);
        outSnapshot->features = features;
        stampSnapshot(*outSnapshot, *resolvedScene);
        outSnapshot->pathTraceResources = std::make_shared<ScenePathTraceResources>();
        impl_->snapshots[key] = outSnapshot;
    } else {
        outSnapshot->features = outSnapshot->features | features;
        if (outSnapshot->pathTraceResources != nullptr &&
            outSnapshot->pathTraceResources->valid()) {
            return {};
        }
    }

    Result result = outSnapshot->pathTraceResources->beginPrepareAsync(
        device,
        graphicsQueue,
        properties,
        *resolvedScene,
        log);
    bool complete = false;
    scene::SceneLoadProgress progress;
    while (result && !complete) {
        result = outSnapshot->pathTraceResources->pumpPrepareAsync(
            std::numeric_limits<double>::max(),
            complete,
            progress,
            log);
        if (result && !complete) {
            std::this_thread::yield();
        }
    }
    if (!result) {
        impl_->snapshots.erase(key);
        outSnapshot.reset();
    }
    return result;
}

Result SceneResourceManager::beginAcquireAsync(
    Device& device,
    Queue& graphicsQueue,
    const RenderGraphProperties& properties,
    const scene::Scene& runtimeScene,
    SceneResourceFeatureBits features,
    std::shared_ptr<SceneResourceSnapshot>& outSnapshot,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_shared<Impl>();
    }
    impl_->collectRetired();
    if (impl_->device != nullptr && impl_->device != &device) {
        clear();
    }
    impl_->device = &device;

    const std::filesystem::path scenePath = propertyPath(properties, "path");
    const std::string key = resourceKey(scenePath);
    auto found = impl_->snapshots.find(key);
    if (found != impl_->snapshots.end()) {
        if (found->second != nullptr &&
            snapshotMatchesScene(*found->second, runtimeScene)) {
            outSnapshot = found->second;
            outSnapshot->features = outSnapshot->features | features;
            if (outSnapshot->pathTraceResources->valid() ||
                outSnapshot->pathTraceResources->preparing()) {
                return {};
            }
        } else {
            if (found->second != nullptr) {
                impl_->retiredSnapshots.push_back(found->second);
            }
            impl_->snapshots.erase(found);
            outSnapshot.reset();
        }
    }
    if (outSnapshot == nullptr) {
        outSnapshot = std::make_shared<SceneResourceSnapshot>();
        outSnapshot->scenePath = normalizedScenePath(scenePath);
        outSnapshot->features = features;
        stampSnapshot(*outSnapshot, runtimeScene);
        outSnapshot->pathTraceResources = std::make_shared<ScenePathTraceResources>();
        impl_->snapshots.emplace(key, outSnapshot);
    }

    Result result = outSnapshot->pathTraceResources->beginPrepareAsync(
        device,
        graphicsQueue,
        properties,
        runtimeScene,
        log);
    if (!result) {
        impl_->snapshots.erase(key);
        outSnapshot.reset();
    }
    return result;
}

Result SceneResourceManager::pumpAsync(
    const std::shared_ptr<SceneResourceSnapshot>& snapshot,
    const scene::Scene& runtimeScene,
    double budgetMilliseconds,
    bool& complete,
    scene::SceneLoadProgress& progress,
    std::string& log)
{
    if (snapshot == nullptr || snapshot->pathTraceResources == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (snapshot->sourceResourceIdentity != runtimeScene.resourceIdentity() ||
        snapshot->sourceStructuralRevision !=
            runtimeScene.sceneGraph().structuralRevision() ||
        snapshot->sourceVisibilityRevision !=
            runtimeScene.visibilityRevision()) {
        log = "Scene changed while asynchronous GPU resources were being prepared.";
        return makeError(Error::Failure);
    }
    if (impl_ != nullptr) {
        impl_->collectRetired();
    }
    Result result = snapshot->pathTraceResources->pumpPrepareAsync(
        budgetMilliseconds,
        complete,
        progress,
        log);
    if (result && complete && snapshot->pathTraceResources->valid() &&
        snapshot->sourceTransformRevision !=
            runtimeScene.transformRevision()) {
        result = snapshot->pathTraceResources->syncRuntimeScene(
            &runtimeScene,
            log);
        if (result) {
            snapshot->sourceTransformRevision =
                runtimeScene.transformRevision();
        }
    }
    return result;
}

void SceneResourceManager::discard(const std::shared_ptr<SceneResourceSnapshot>& snapshot)
{
    if (impl_ == nullptr || snapshot == nullptr) {
        return;
    }
    for (auto iter = impl_->snapshots.begin(); iter != impl_->snapshots.end();) {
        if (iter->second == snapshot) {
            iter = impl_->snapshots.erase(iter);
        } else {
            ++iter;
        }
    }
    impl_->retiredSnapshots.push_back(snapshot);
    impl_->collectRetired();
}

void SceneResourceManager::clear()
{
    if (impl_ != nullptr) {
        impl_->snapshots.clear();
        impl_->scenes.clear();
        impl_->retiredSnapshots.clear();
        impl_->device = nullptr;
    }
}

} // namespace metallic::render
