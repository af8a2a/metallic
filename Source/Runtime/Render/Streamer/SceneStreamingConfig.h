#pragma once
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"

namespace metallic::render::streaming_detail {
inline bool boolProperty(const RenderGraphProperties* props, const char* key, bool fallback)
{
    const auto it = props->find(key);
    return it != props->end() && it->is_boolean() ? it->get<bool>() : fallback;
}
inline uint32_t pageLoadConcurrencyFromProperties(
    const RenderGraphProperties& properties,
    uint32_t fallback = 2)
{
    const auto readProperty = [&properties](const char* key, uint32_t propertyFallback) {
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number_integer()) {
            return propertyFallback;
        }
        const int64_t value = iter->get<int64_t>();
        return value < 0 || value > std::numeric_limits<uint32_t>::max()
            ? propertyFallback
            : static_cast<uint32_t>(value);
    };

    const uint32_t legacyValue = readProperty("pageLoadWorkerCount", fallback);
    return std::min(
        readProperty("pageLoadConcurrency", legacyValue),
        kMeshletStreamMaxPageLoadConcurrency);
}

bool previewStreamEnabled(const RenderGraphProperties& properties)
{
    const auto enabled = properties.find("enableMeshletStreaming");
    if (enabled != properties.end() && enabled->is_boolean()) {
        return enabled->get<bool>();
    }
    const auto assetPath = properties.find("streamAssetPath");
    return assetPath != properties.end() && assetPath->is_string();
}

uint32_t previewStreamUintProperty(
    const RenderGraphProperties& properties,
    const char* key,
    uint32_t fallback)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 || value > std::numeric_limits<uint32_t>::max()
        ? fallback
        : static_cast<uint32_t>(value);
}

uint64_t previewStreamUint64Property(
    const RenderGraphProperties& properties,
    const char* key,
    uint64_t fallback)
{
    const auto iter = properties.find(key);
    if (iter == properties.end() || !iter->is_number_integer()) {
        return fallback;
    }
    const int64_t value = iter->get<int64_t>();
    return value < 0 ? fallback : static_cast<uint64_t>(value);
}

std::filesystem::path previewStreamAssetPath(
    const RenderGraphProperties& properties,
    const std::filesystem::path& sourcePath)
{
    const auto iter = properties.find("streamAssetPath");
    if (iter == properties.end() || !iter->is_string()) {
        return scene::meshletStreamAssetPathFor(sourcePath);
    }
    std::filesystem::path path = iter->get<std::string>();
    return path.is_relative()
        ? std::filesystem::path(PROJECT_SOURCE_DIR) / path
        : path;
}

struct PreviewStreamSourceSelection {
    std::string sourceId;
    std::filesystem::path sourcePath;
};

bool resolvePreviewStreamSource(
    const RenderGraphProperties& properties,
    const scene::Scene* runtimeScene,
    const std::filesystem::path& scenePath,
    PreviewStreamSourceSelection& outSelection,
    std::string& log)
{
    outSelection = {};
    if (runtimeScene == nullptr) {
        log = "VisibilityBufferPass stream integration requires a runtime scene";
        return false;
    }

    std::string requestedSourceId;
    const auto sourceIdProperty = properties.find("streamSourceId");
    if (sourceIdProperty != properties.end()) {
        if (!sourceIdProperty->is_string() ||
            sourceIdProperty->get_ref<const std::string&>().empty()) {
            log = "VisibilityBufferPass streamSourceId must be a non-empty string";
            return false;
        }
        requestedSourceId = sourceIdProperty->get<std::string>();
    }

    const std::vector<scene::SceneSourceDesc>& sources = runtimeScene->sources();
    if (sources.empty()) {
        if (!requestedSourceId.empty()) {
            log = "VisibilityBufferPass streamSourceId was provided for a non-composed scene";
            return false;
        }
        outSelection.sourcePath = scenePath;
        return true;
    }

    const auto selectSource = [&](const scene::SceneSourceDesc& source) {
        outSelection.sourceId = source.id;
        outSelection.sourcePath = source.path;
    };
    if (!requestedSourceId.empty()) {
        const auto source = std::find_if(
            sources.begin(),
            sources.end(),
            [&](const scene::SceneSourceDesc& candidate) {
                return candidate.id == requestedSourceId;
            });
        if (source == sources.end()) {
            log = "VisibilityBufferPass streamSourceId '" + requestedSourceId +
                "' does not name a composed scene source";
            return false;
        }
        selectSource(*source);
        return true;
    }

    if (sources.size() == 1) {
        selectSource(sources.front());
        return true;
    }

    std::vector<const scene::SceneSourceDesc*> matches;
    const std::filesystem::path normalizedRequestedPath = normalizedScenePath(scenePath);
    for (const scene::SceneSourceDesc& source : sources) {
        if (normalizedScenePath(source.path) == normalizedRequestedPath) {
            matches.push_back(&source);
        }
    }
    if (matches.size() == 1) {
        selectSource(*matches.front());
        return true;
    }

    matches.clear();
    const auto assetPathProperty = properties.find("streamAssetPath");
    if (assetPathProperty != properties.end() && assetPathProperty->is_string()) {
        scene::MeshletStreamAsset candidateAsset;
        std::string reason;
        const std::filesystem::path assetPath =
            previewStreamAssetPath(properties, scenePath);
        if (!candidateAsset.open(assetPath, reason)) {
            log = "VisibilityBufferPass cannot inspect streamAssetPath '" +
                assetPath.string() + "': " + reason;
            return false;
        }
        for (const scene::SceneSourceDesc& source : sources) {
            if (candidateAsset.isRuntimeCompatibleForSource(source.path, reason)) {
                matches.push_back(&source);
            }
        }
    }
    if (matches.size() == 1) {
        selectSource(*matches.front());
        return true;
    }

    log = matches.empty()
        ? "VisibilityBufferPass could not uniquely match the stream asset to a composed scene source; set streamSourceId"
        : "VisibilityBufferPass stream asset matches multiple composed scene sources; set streamSourceId to disambiguate the owner";
    return false;
}

MeshletStreamRuntimeDesc previewStreamRuntimeDesc(
    const RenderGraphProperties& properties,
    const std::filesystem::path& sourcePath, uint32_t textureCapacity)
{
    const uint32_t maxGpuPageRequests = std::max(
        previewStreamUintProperty(
            properties,
            "maxGpuPageRequests",
            kMeshletStreamDefaultMaxGpuPageRequests),
        1u);
    return MeshletStreamRuntimeDesc{
        .sourcePath = sourcePath,
        .streamAssetPath = previewStreamAssetPath(properties, sourcePath),
        .autoBuildStreamAsset = false,
        .maxResidentBytes = previewStreamUint64Property(
            properties,
            "maxResidentBytes",
            0),
        .maxResidentPages = previewStreamUintProperty(
            properties,
            "maxResidentPages",
            4096),
        .maxLockedFallbackPages = previewStreamUintProperty(
            properties,
            "maxLockedFallbackPages",
            1024),
        .maxPageUploadsPerFrame = previewStreamUintProperty(
            properties,
            "maxPageUploadsPerFrame",
            64),
        .maxUploadBytesPerFrame = previewStreamUint64Property(properties, "maxUploadBytesPerFrame", 8ull * 1024ull * 1024ull),
        .maxGpuPageRequests = maxGpuPageRequests,
        .maxGpuPageUnloadRequests = std::max(
            previewStreamUintProperty(
                properties,
                "maxGpuPageUnloadRequests",
                maxGpuPageRequests),
            1u),
        .maxActiveGroups = std::max(
            previewStreamUintProperty(
                properties,
                "maxActiveGroups",
                kMeshletStreamDefaultMaxActiveGroups),
            1u),
        .maxTraversalWorkers = std::max(
            previewStreamUintProperty(
                properties,
                "maxTraversalWorkers",
                kMeshletStreamDefaultTraversalWorkers),
            1u),
        .maxTraversalWorkItems = std::min(
            std::max(
                previewStreamUintProperty(
                    properties,
                    "maxTraversalWorkItems",
                    kMeshletStreamDefaultTraversalWorkItems),
                1u),
            kMeshletStreamMaxTraversalWorkItems),
        .pageLoadConcurrency = pageLoadConcurrencyFromProperties(properties),
        .maxPageLoadsInFlight = std::max(
            previewStreamUintProperty(
                properties,
                "maxPageLoadsInFlight",
                128),
            1u),
        .queuedFrameCount = 3,
        .enableClusterRtx = boolProperty(&properties, "enableClusterRtx", false),
        .enableClas = boolProperty(&properties, "enableClas", false),
        .compactClas = boolProperty(&properties, "compactClas", false),
        .coldPageRetentionFrames = previewStreamUintProperty(properties, "coldPageRetentionFrames", 0),
        .maxClasBytes = previewStreamUint64Property(properties, "maxClasBytes", 512ull * 1024ull * 1024ull),
        .maxClasBuildClusters = previewStreamUintProperty(properties, "maxClasBuildClusters", 0),
        .maxBlasClusterReferences = previewStreamUintProperty(properties, "maxBlasClusterReferences", 0),
        .maxBlasBytes = previewStreamUint64Property(properties, "maxBlasBytes", 512ull * 1024ull * 1024ull),
        .maxBlasBuilds = previewStreamUintProperty(properties, "maxBlasBuilds", kMeshletStreamDefaultMaxBlasBuilds),
        .maxFallbackBlasBytes = previewStreamUint64Property(properties, "maxFallbackBlasBytes", 512ull * 1024ull * 1024ull),
        .screenSpacePagePriority = boolProperty(&properties, "screenSpacePagePriority", true),
        .viewDrivenPageDemand = boolProperty(&properties, "viewDrivenPageDemand", true),
        .distributedPageDemand = boolProperty(&properties, "distributedPageDemand", true),
        .distributedDemandMinGroups = previewStreamUintProperty(properties, "distributedDemandMinGroups", 65536u),
        .measurePageLatency = boolProperty(&properties, "measurePageLatency", true),
        .lowLatencyRequests = boolProperty(&properties, "lowLatencyRequests", true),
        .completionDrivenUploads = boolProperty(&properties, "completionDrivenUploads", true),
        .enableGpuDecompression = boolProperty(&properties, "enableGpuDecompression", false),
        .gpuDecompressionMinBatchBytes = previewStreamUint64Property(properties, "gpuDecompressionMinBatchBytes", 1024 * 1024),
        .prefetchPages = boolProperty(&properties, "prefetchPages", true),
        .rasterMaterialTextureCapacity = textureCapacity,
        .compactShadingAttributes = boolProperty(&properties, "compactShadingAttributes", false),
    };
}
MeshletStreamRuntimeDesc assetStreamRuntimeDesc(const RenderGraphProperties& properties)
{
    const std::filesystem::path scenePath = normalizedScenePath(properties.value("path", std::string(PROJECT_SOURCE_DIR "/Asset/SuperSponza/NewSponza_Main_glTF_003.gltf")));
    const uint32_t maxGpuPageRequests = std::max<uint32_t>(
        previewStreamUintProperty(properties, "maxGpuPageRequests", kMeshletStreamDefaultMaxGpuPageRequests),
        1u);
    return MeshletStreamRuntimeDesc{
        .sourcePath = scenePath,
        .streamAssetPath = previewStreamAssetPath(properties, scenePath),
        .autoBuildStreamAsset = boolProperty(&properties, "autoBuildStreamAsset", false),
        .maxResidentBytes = previewStreamUint64Property(properties, "maxResidentBytes", 0),
        .maxResidentPages = previewStreamUintProperty(properties, "maxResidentPages", 4096),
        .maxLockedFallbackPages = previewStreamUintProperty(properties, "maxLockedFallbackPages", 1024),
        .maxPageUploadsPerFrame = previewStreamUintProperty(properties, "maxPageUploadsPerFrame", 64),
        .maxUploadBytesPerFrame = previewStreamUint64Property(properties, "maxUploadBytesPerFrame", 8ull * 1024ull * 1024ull),
        .maxGpuPageRequests = maxGpuPageRequests,
        .maxGpuPageUnloadRequests = std::max<uint32_t>(
            previewStreamUintProperty(properties, "maxGpuPageUnloadRequests", maxGpuPageRequests),
            1u),
        .maxActiveGroups = std::max<uint32_t>(
            previewStreamUintProperty(properties, "maxActiveGroups", kMeshletStreamDefaultMaxActiveGroups),
            1u),
        .maxTraversalWorkers = std::max<uint32_t>(
            previewStreamUintProperty(properties, "maxTraversalWorkers", kMeshletStreamDefaultTraversalWorkers),
            1u),
        .maxTraversalWorkItems = std::min(
            std::max<uint32_t>(
                previewStreamUintProperty(properties, "maxTraversalWorkItems", kMeshletStreamDefaultTraversalWorkItems),
                1u),
            kMeshletStreamMaxTraversalWorkItems),
        .pageLoadConcurrency = pageLoadConcurrencyFromProperties(properties),
        .maxPageLoadsInFlight = std::max<uint32_t>(
            previewStreamUintProperty(properties, "maxPageLoadsInFlight", 128),
            1u),
        .queuedFrameCount = 3,
        .enableClusterRtx = boolProperty(&properties, "enableClusterRtx", false),
        .enableClas = boolProperty(&properties, "enableClas", false),
        .compactClas = boolProperty(&properties, "compactClas", false),
        .coldPageRetentionFrames = previewStreamUintProperty(properties, "coldPageRetentionFrames", 0),
        .maxClasBytes = previewStreamUint64Property(properties, "maxClasBytes", 512ull * 1024ull * 1024ull),
        .maxClasBuildClusters = previewStreamUintProperty(properties, "maxClasBuildClusters", 0),
        .maxBlasClusterReferences = previewStreamUintProperty(properties, "maxBlasClusterReferences", 0),
        .maxBlasBytes = previewStreamUint64Property(properties, "maxBlasBytes", 512ull * 1024ull * 1024ull),
        .maxBlasBuilds = std::max<uint32_t>(
            previewStreamUintProperty(properties, "maxBlasBuilds", kMeshletStreamDefaultMaxBlasBuilds),
            1u),
        .maxFallbackBlasBytes = previewStreamUint64Property(
            properties,
            "maxFallbackBlasBytes",
            512ull * 1024ull * 1024ull),
        .screenSpacePagePriority = boolProperty(&properties, "screenSpacePagePriority", true),
        .viewDrivenPageDemand = boolProperty(&properties, "viewDrivenPageDemand", true),
        .distributedPageDemand = boolProperty(&properties, "distributedPageDemand", true),
        .distributedDemandMinGroups = previewStreamUintProperty(properties, "distributedDemandMinGroups", 65536u),
        .measurePageLatency = boolProperty(&properties, "measurePageLatency", true),
        .lowLatencyRequests = boolProperty(&properties, "lowLatencyRequests", true),
        .completionDrivenUploads = boolProperty(&properties, "completionDrivenUploads", true),
        .enableGpuDecompression = boolProperty(&properties, "enableGpuDecompression", false),
        .gpuDecompressionMinBatchBytes = previewStreamUint64Property(properties, "gpuDecompressionMinBatchBytes", 1024 * 1024),
        .prefetchPages = boolProperty(&properties, "prefetchPages", true),
        .compactShadingAttributes = boolProperty(&properties, "compactShadingAttributes", false),
    };
}
} // namespace metallic::render::streaming_detail
