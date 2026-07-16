#include "Editor/EditorApplication.h"
#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <cstdlib>
#include <cstdio>
#include <charconv>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>

namespace {

bool isEnabledValue(std::string_view value)
{
    return !value.empty() && value != "0" && value != "false" && value != "FALSE";
}

bool waitForGraphicsDebuggerFromEnv()
{
#if defined(_WIN32)
    size_t requiredSize = 0;
    if (getenv_s(&requiredSize, nullptr, 0, "METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER") != 0 ||
        requiredSize == 0) {
        return false;
    }

    std::string value(requiredSize, '\0');
    if (getenv_s(&requiredSize, value.data(), value.size(), "METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER") != 0) {
        return false;
    }
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return isEnabledValue(value);
#else
    const char* value = std::getenv("METALLIC_WAIT_FOR_GRAPHICS_DEBUGGER");
    return value != nullptr && isEnabledValue(value);
#endif
}

void printUsage()
{
    std::puts(
        "Metallic options:\n"
        "  --smoke-test                                  Render one frame and exit\n"
        "  --wait-for-graphics-debugger                  Wait before Vulkan initialization\n"
        "  --rhi-smoke-test                              Run the RHI smoke test\n"
        "  --rhi-triangle-preview-test                   Run the RHI triangle preview test\n"
        "  --rhi-bindless-descriptor-heap-smoke-test     Run the bindless descriptor heap smoke test\n"
        "  --rhi-no-validation                           Disable RHI validation for smoke tests\n"
        "  --build-meshstream <source.gltf>              Build a meshlet StreamAsset and exit\n"
        "  --output <file.meshstream.bin>                Optional output path for --build-meshstream\n"
        "  --meshstream-compression <none|byte-rle>      Optional payload compression for --build-meshstream\n"
        "  --meshstream-max-geometries <count>           Pause after this many new geometries (0 = unlimited)\n"
        "  --meshstream-checkpoint-interval <count>      Partial checkpoint interval (0 = pause only)");
}

bool parseUint32(std::string_view value, uint32_t& outValue)
{
    uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    outValue = static_cast<uint32_t>(parsed);
    return true;
}

bool parseMeshletStreamCompression(
    std::string_view value,
    metallic::scene::MeshletStreamPayloadCompression& outCompressionMode)
{
    if (value == "none") {
        outCompressionMode = metallic::scene::MeshletStreamPayloadCompression::None;
        return true;
    }
    if (value == "byte-rle" || value == "byte_rle" || value == "byterle") {
        outCompressionMode = metallic::scene::MeshletStreamPayloadCompression::ByteRle;
        return true;
    }
    return false;
}

int buildMeshletStreamAssetOffline(
    const std::filesystem::path& sourcePath,
    const std::filesystem::path& outputPath,
    metallic::scene::MeshletStreamPayloadCompression compressionMode,
    uint32_t maxNewGeometriesPerInvocation,
    uint32_t partialCheckpointGeometryInterval)
{
    if (sourcePath.empty()) {
        std::fputs("Missing source path for --build-meshstream\n", stderr);
        return 1;
    }

    const std::filesystem::path resolvedOutputPath = outputPath.empty()
        ? metallic::scene::meshletStreamAssetPathFor(sourcePath)
        : outputPath;
    metallic::scene::MeshletStreamAssetOfflineBuildStats buildStats;
    std::string reason;
    if (!metallic::scene::buildMeshletStreamAssetOffline(
            metallic::scene::MeshletStreamAssetOfflineBuildDesc{
                .sourcePath = sourcePath,
                .outputPath = resolvedOutputPath,
                .compressionMode = compressionMode,
                .maxNewGeometriesPerInvocation = maxNewGeometriesPerInvocation,
                .partialCheckpointGeometryInterval = partialCheckpointGeometryInterval,
                .stats = &buildStats,
            },
            reason)) {
        const bool paused = reason.find("paused after geometry budget") != std::string::npos;
        std::fprintf(
            paused ? stdout : stderr,
            "%s: %s (checkpoints=%u)\n",
            paused ? "Paused StreamAsset build" : "StreamAsset build failed",
            reason.c_str(),
            buildStats.partialCheckpointCount);
        return paused ? 2 : 1;
    }

    metallic::scene::MeshletStreamAsset asset;
    if (!asset.open(resolvedOutputPath, reason)) {
        std::fprintf(stderr, "Built streamasset failed validation: %s\n", reason.c_str());
        return 1;
    }

    uint64_t fallbackPageCount = 0;
    for (const metallic::scene::MeshletStreamPrimitiveInfo& primitive : asset.primitives()) {
        fallbackPageCount += primitive.fallbackPageCount;
    }
    std::printf(
        "Built meshlet StreamAsset '%s': primitives=%u geometries=%u instances=%u lodLevels=%u groups=%u nodes=%u pages=%u fallbackPages=%llu maxPagePayloadBytes=%u\n",
        resolvedOutputPath.string().c_str(),
        asset.primitiveCount(),
        asset.geometryCount(),
        asset.instanceCount(),
        asset.lodLevelCount(),
        asset.groupCount(),
        asset.nodeCount(),
        asset.pageCount(),
        static_cast<unsigned long long>(fallbackPageCount),
        asset.maxPagePayloadBytes());
    if (buildStats.accessorRangeReadCount != 0) {
        std::printf(
            "External buffer range reads: buffersBytes=%llu readBytes=%llu maxRangeBytes=%llu ranges=%u\n",
            static_cast<unsigned long long>(buildStats.externalBufferDeclaredBytes),
            static_cast<unsigned long long>(buildStats.accessorRangeReadBytes),
            static_cast<unsigned long long>(buildStats.maxAccessorRangeReadBytes),
            buildStats.accessorRangeReadCount);
    }
    std::printf("Partial checkpoints written: %u\n", buildStats.partialCheckpointCount);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    bool smokeTest = false;
    bool rhiSmokeTest = false;
    bool rhiTrianglePreviewTest = false;
    bool rhiBindlessDescriptorHeapSmokeTest = false;
    bool rhiValidation = true;
    bool waitForGraphicsDebugger = waitForGraphicsDebuggerFromEnv();
    std::filesystem::path buildMeshstreamSourcePath;
    std::filesystem::path buildMeshstreamOutputPath;
    metallic::scene::MeshletStreamPayloadCompression buildMeshstreamCompressionMode =
        metallic::scene::MeshletStreamPayloadCompression::None;
    uint32_t buildMeshstreamMaxNewGeometries = 0;
    uint32_t buildMeshstreamCheckpointInterval = 64;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            printUsage();
            return 0;
        } else if (argument == "--smoke-test") {
            smokeTest = true;
        } else if (argument == "--rhi-smoke-test") {
            rhiSmokeTest = true;
        } else if (argument == "--rhi-triangle-preview-test") {
            rhiTrianglePreviewTest = true;
        } else if (argument == "--rhi-bindless-descriptor-heap-smoke-test") {
            rhiBindlessDescriptorHeapSmokeTest = true;
        } else if (argument == "--rhi-no-validation") {
            rhiValidation = false;
        } else if (argument == "--wait-for-graphics-debugger") {
            waitForGraphicsDebugger = true;
        } else if (argument == "--build-meshstream") {
            if (index + 1 >= argc) {
                std::fputs("--build-meshstream requires a source path\n", stderr);
                return 1;
            }
            buildMeshstreamSourcePath = argv[++index];
        } else if (argument == "--output") {
            if (index + 1 >= argc) {
                std::fputs("--output requires a file path\n", stderr);
                return 1;
            }
            buildMeshstreamOutputPath = argv[++index];
        } else if (argument == "--meshstream-compression") {
            if (index + 1 >= argc) {
                std::fputs("--meshstream-compression requires one of: none, byte-rle\n", stderr);
                return 1;
            }
            const std::string_view compression(argv[++index]);
            if (!parseMeshletStreamCompression(compression, buildMeshstreamCompressionMode)) {
                std::fprintf(
                    stderr,
                    "Unsupported meshstream compression '%.*s'; expected none or byte-rle\n",
                    static_cast<int>(compression.size()),
                    compression.data());
                return 1;
            }
        } else if (argument == "--meshstream-max-geometries" ||
                   argument == "--meshstream-checkpoint-interval") {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "%.*s requires a non-negative integer\n",
                    static_cast<int>(argument.size()), argument.data());
                return 1;
            }
            uint32_t& value = argument == "--meshstream-max-geometries"
                ? buildMeshstreamMaxNewGeometries
                : buildMeshstreamCheckpointInterval;
            const std::string_view text(argv[++index]);
            if (!parseUint32(text, value)) {
                std::fprintf(stderr, "Invalid value '%.*s' for %.*s\n",
                    static_cast<int>(text.size()), text.data(),
                    static_cast<int>(argument.size()), argument.data());
                return 1;
            }
        }
    }

    if (!buildMeshstreamSourcePath.empty()) {
        return buildMeshletStreamAssetOffline(
            buildMeshstreamSourcePath,
            buildMeshstreamOutputPath,
            buildMeshstreamCompressionMode,
            buildMeshstreamMaxNewGeometries,
            buildMeshstreamCheckpointInterval);
    }

    if (rhiTrianglePreviewTest) {
        return metallic::render::runRhiTrianglePreviewTest(rhiValidation);
    }

    if (rhiBindlessDescriptorHeapSmokeTest) {
        return metallic::render::runRhiBindlessDescriptorHeapSmokeTest(rhiValidation);
    }

    if (rhiSmokeTest) {
        return metallic::render::runRhiSmokeTest(rhiValidation);
    }

    metallic::EditorApplication app;
    return app.run(smokeTest, waitForGraphicsDebugger);
}
