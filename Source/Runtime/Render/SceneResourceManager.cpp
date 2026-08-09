#include "Runtime/Render/SceneResourceManager.h"

#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"

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

RenderGraphProperties sceneProperties(const RenderGraphProperties& properties)
{
    RenderGraphProperties result = properties;
    result.erase("environment");
    return result;
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
    std::unordered_map<std::string, std::shared_ptr<scene::Scene>> scenes;
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

    auto loadedScene = std::make_shared<scene::Scene>();
    const std::filesystem::path resolvedPath = normalizedScenePath(scenePath);
    if (!loadedScene->load(resolvedPath)) {
        log = "SceneResourceManager failed to load glTF: " + loadedScene->lastLoadResult().error;
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
    const auto found = impl_->snapshots.find(key);
    if (found != impl_->snapshots.end()) {
        outSnapshot = found->second;
    }
    if (outSnapshot == nullptr) {
        outSnapshot = std::make_shared<SceneResourceSnapshot>();
        outSnapshot->scenePath = normalizedScenePath(scenePath);
        outSnapshot->features = features;
        outSnapshot->pathTraceResources = std::make_shared<ScenePathTraceResources>();
        impl_->snapshots[key] = outSnapshot;
    } else {
        outSnapshot->features = outSnapshot->features | features;
        if (outSnapshot->pathTraceResources != nullptr &&
            outSnapshot->pathTraceResources->valid()) {
            return {};
        }
    }

    const RenderGraphProperties resourceProperties = sceneProperties(properties);
    Result result = outSnapshot->pathTraceResources->beginPrepareAsync(
        device,
        graphicsQueue,
        resourceProperties,
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
    const auto found = impl_->snapshots.find(key);
    if (found != impl_->snapshots.end()) {
        outSnapshot = found->second;
        outSnapshot->features = outSnapshot->features | features;
        if (outSnapshot->pathTraceResources->valid() ||
            outSnapshot->pathTraceResources->preparing()) {
            return {};
        }
    } else {
        outSnapshot = std::make_shared<SceneResourceSnapshot>();
        outSnapshot->scenePath = normalizedScenePath(scenePath);
        outSnapshot->features = features;
        outSnapshot->pathTraceResources = std::make_shared<ScenePathTraceResources>();
        impl_->snapshots.emplace(key, outSnapshot);
    }

    const RenderGraphProperties resourceProperties = sceneProperties(properties);
    Result result = outSnapshot->pathTraceResources->beginPrepareAsync(
        device,
        graphicsQueue,
        resourceProperties,
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
    double budgetMilliseconds,
    bool& complete,
    scene::SceneLoadProgress& progress,
    std::string& log)
{
    if (snapshot == nullptr || snapshot->pathTraceResources == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    if (impl_ != nullptr) {
        impl_->collectRetired();
    }
    return snapshot->pathTraceResources->pumpPrepareAsync(
        budgetMilliseconds,
        complete,
        progress,
        log);
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
