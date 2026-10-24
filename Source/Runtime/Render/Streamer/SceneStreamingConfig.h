#pragma once
#include "Runtime/Render/Streamer/UploadStreamer.h"
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

uint32_t sceneStreamUintProperty(
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

uint64_t sceneStreamUint64Property(
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

std::filesystem::path sceneStreamAssetPath(
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

MeshletStreamRuntimeDesc sceneStreamRuntimeDesc(
    const RenderGraphProperties& properties,
    const std::filesystem::path& sourcePath, uint32_t textureCapacity)
{
    const uint32_t maxGpuPageRequests = std::max(
        sceneStreamUintProperty(
            properties,
            "maxGpuPageRequests",
            kMeshletStreamDefaultMaxGPUPageRequests),
        1u);
    return MeshletStreamRuntimeDesc{
        .sourcePath = sourcePath,
        .streamAssetPath = sceneStreamAssetPath(properties, sourcePath),
        .maxResidentBytes = sceneStreamUint64Property(
            properties,
            "maxResidentBytes",
            0),
        .maxResidentPages = sceneStreamUintProperty(
            properties,
            "maxResidentPages",
            4096),
        .maxLockedFallbackPages = sceneStreamUintProperty(
            properties,
            "maxLockedFallbackPages",
            1024),
        .maxPageUploadsPerFrame = sceneStreamUintProperty(
            properties,
            "maxPageUploadsPerFrame",
            64),
        .maxUploadBytesPerFrame = sceneStreamUint64Property(properties, "maxUploadBytesPerFrame", 8ull * 1024ull * 1024ull),
        .maxGpuPageRequests = maxGpuPageRequests,
        .maxGpuPageUnloadRequests = std::max(
            sceneStreamUintProperty(
                properties,
                "maxGpuPageUnloadRequests",
                maxGpuPageRequests),
            1u),
        .maxActiveGroups = std::max(
            sceneStreamUintProperty(
                properties,
                "maxActiveGroups",
                kMeshletStreamDefaultMaxActiveGroups),
            1u),
        .maxRasterCandidates = sceneStreamUintProperty(properties, "maxRasterCandidates", kMeshletStreamDefaultMaxRasterCandidates),
        .maxTraversalWorkers = std::max(
            sceneStreamUintProperty(
                properties,
                "maxTraversalWorkers",
                kMeshletStreamDefaultTraversalWorkers),
            1u),
        .maxTraversalWorkItems = std::min(
            std::max(
                sceneStreamUintProperty(
                    properties,
                    "maxTraversalWorkItems",
                    kMeshletStreamDefaultTraversalWorkItems),
                1u),
            kMeshletStreamMaxTraversalWorkItems),
        .pageLoadConcurrency = pageLoadConcurrencyFromProperties(properties),
        .maxPageLoadsInFlight = std::max(
            sceneStreamUintProperty(
                properties,
                "maxPageLoadsInFlight",
                128),
            1u),
        .queuedFrameCount = 3,
        .enableClusterRtx = boolProperty(&properties, "enableClusterRtx", false),
        .asyncComputePreferred = boolProperty(&properties, "AsyncComputePreferred", true),
        .enableClas = boolProperty(&properties, "enableClas", false),
        .compactClas = boolProperty(&properties, "compactClas", false),
        .coldPageRetentionFrames = sceneStreamUintProperty(properties, "coldPageRetentionFrames", 0),
        .adaptivePageRetention = boolProperty(&properties, "adaptivePageRetention", true),
        .maxClasBytes = sceneStreamUint64Property(properties, "maxClasBytes", 512ull * 1024ull * 1024ull),
        .startClasBytes = sceneStreamUint64Property(properties, "startClasBytes", 0),
        .growClasBytes = sceneStreamUint64Property(properties, "growClasBytes", 64ull * 1024ull * 1024ull),
        .clasEmptyChunkRetentionFrames = sceneStreamUintProperty(properties, "clasEmptyChunkRetentionFrames", 0),
        .persistentClasGrowBytes = sceneStreamUint64Property(properties, "persistentClasGrowBytes", 0),
        .maxClasBuildClusters = sceneStreamUintProperty(properties, "maxClasBuildClusters", 0),
        .maxBlasClusterReferences = sceneStreamUintProperty(properties, "maxBlasClusterReferences", 0),
        .maxBlasBytes = sceneStreamUint64Property(properties, "maxBlasBytes", 512ull * 1024ull * 1024ull),
        .maxBlasStagingBytes = sceneStreamUint64Property(properties, "maxBlasStagingBytes", 512ull * 1024ull * 1024ull),
        .maxBlasBuilds = sceneStreamUintProperty(properties, "maxBlasBuilds", kMeshletStreamDefaultMaxBLASBuilds),
        .maxFallbackBlasBytes = sceneStreamUint64Property(properties, "maxFallbackBlasBytes", 512ull * 1024ull * 1024ull),
        .screenSpacePagePriority = boolProperty(&properties, "screenSpacePagePriority", true),
        .viewDrivenPageDemand = boolProperty(&properties, "viewDrivenPageDemand", true),
        .distributedPageDemand = boolProperty(&properties, "distributedPageDemand", true),
        .distributedDemandMinGroups = sceneStreamUintProperty(properties, "distributedDemandMinGroups", 65536u),
        .measurePageLatency = boolProperty(&properties, "measurePageLatency", true),
        .lowLatencyRequests = boolProperty(&properties, "lowLatencyRequests", true),
        .completionDrivenUploads = boolProperty(&properties, "completionDrivenUploads", true),
        .enableGpuDecompression = boolProperty(&properties, "enableGpuDecompression", false),
        .gpuDecompressionMinBatchBytes = sceneStreamUint64Property(properties, "gpuDecompressionMinBatchBytes", 1024 * 1024),
        .prefetchPages = boolProperty(&properties, "prefetchPages", true),
        .predictivePrefetch = boolProperty(&properties, "predictivePrefetch", true),
        .enableLodTransitionTelemetry = boolProperty(&properties, "enableLodTransitionTelemetry", false),
        .rasterMaterialTextureCapacity = textureCapacity,
        .compactShadingAttributes = boolProperty(&properties, "compactShadingAttributes", false),
        .deviceImmutableMetadata = boolProperty(&properties, "deviceImmutableMetadata", true),
        .initialLoad = boolProperty(&properties, "initialLoad", true),
    };
}
} // namespace metallic::render::streaming_detail
