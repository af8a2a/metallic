#include "GpuPageCodecChecks.h"
#include <volk.h>
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <json.hpp>

namespace {
using namespace metallic;
void checked(VkResult result) { tests::requireGpuPage(result == VK_SUCCESS, "Vulkan error " + std::to_string(result)); }

struct DecompressionProbe {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    std::atomic_uint errors = 0;
    bool validationEnabled = false;
    uint32_t timestampValidBits = 0;
    float timestampPeriod = 0;
    struct Buffer { VkBuffer buffer; VkDeviceMemory memory; VkDeviceAddress address; };
    std::vector<Buffer> buffers;

    ~DecompressionProbe()
    {
        if (device) {
            vkDeviceWaitIdle(device);
            for (auto buffer : buffers) { vkDestroyBuffer(device, buffer.buffer, nullptr); vkFreeMemory(device, buffer.memory, nullptr); }
            if (pool) { vkDestroyCommandPool(device, pool, nullptr); }
            vkDestroyDevice(device, nullptr);
        }
        if (messenger) { vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr); }
        if (instance) { vkDestroyInstance(instance, nullptr); }
    }

    bool initialize(bool requestValidation = true)
    {
        if (volkInitialize() != VK_SUCCESS) { return false; }
        uint32_t version = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion) { checked(vkEnumerateInstanceVersion(&version)); }
        if (version < VK_API_VERSION_1_4) { return false; }
        uint32_t count = 0;
        checked(vkEnumerateInstanceLayerProperties(&count, nullptr));
        std::vector<VkLayerProperties> layers(count);
        checked(vkEnumerateInstanceLayerProperties(&count, layers.data()));
        const char* validation = "VK_LAYER_KHRONOS_validation";
        const bool validate = requestValidation && std::ranges::any_of(layers, [&](const auto& l) { return std::strcmp(l.layerName, validation) == 0; });
        validationEnabled = validate;
        const char* debug = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        const VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "Metallic GDeflate probe", .apiVersion = VK_API_VERSION_1_4};
        const VkInstanceCreateInfo create{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &application,
            .enabledLayerCount = validate ? 1u : 0u, .ppEnabledLayerNames = &validation,
            .enabledExtensionCount = 1, .ppEnabledExtensionNames = &debug};
        checked(vkCreateInstance(&create, nullptr, &instance)); volkLoadInstance(instance);
        const VkDebugUtilsMessengerCreateInfoEXT debugInfo{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) -> VkBool32 {
                ++*static_cast<std::atomic_uint*>(user); std::cerr << data->pMessage << '\n'; return VK_FALSE;
            }, .pUserData = &errors};
        checked(vkCreateDebugUtilsMessengerEXT(instance, &debugInfo, nullptr, &messenger));
        checked(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        checked(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        for (auto candidate : devices) {
            checked(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr));
            std::vector<VkExtensionProperties> extensions(count);
            checked(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, extensions.data()));
            if (!std::ranges::any_of(extensions, [](const auto& e) { return std::strcmp(e.extensionName, VK_EXT_MEMORY_DECOMPRESSION_EXTENSION_NAME) == 0; })) { continue; }
            VkPhysicalDeviceMemoryDecompressionFeaturesEXT feature{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT};
            VkPhysicalDeviceVulkan12Features core{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &feature};
            VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &core};
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            VkPhysicalDeviceMemoryDecompressionPropertiesEXT methods{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &methods};
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (!feature.memoryDecompression || !core.bufferDeviceAddress || properties.properties.apiVersion < VK_API_VERSION_1_4 ||
                !(methods.decompressionMethods & VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT)) { continue; }
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
            uint32_t family = 0;
            while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) { ++family; }
            if (family == count) { continue; }
            timestampValidBits = families[family].timestampValidBits;
            timestampPeriod = properties.properties.limits.timestampPeriod;
            const float priority = 1;
            const VkDeviceQueueCreateInfo queueInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
            VkPhysicalDeviceVulkan13Features synchronization{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                .pNext = &feature, .synchronization2 = VK_TRUE};
            core = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &synchronization, .bufferDeviceAddress = VK_TRUE};
            const char* extension = VK_EXT_MEMORY_DECOMPRESSION_EXTENSION_NAME;
            const VkDeviceCreateInfo info{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &core,
                .queueCreateInfoCount = 1, .pQueueCreateInfos = &queueInfo, .enabledExtensionCount = 1, .ppEnabledExtensionNames = &extension};
            checked(vkCreateDevice(candidate, &info, nullptr, &device)); physical = candidate; volkLoadDevice(device);
            vkGetDeviceQueue(device, family, 0, &queue);
            const VkCommandPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
            checked(vkCreateCommandPool(device, &poolInfo, nullptr, &pool));
            std::cout << properties.properties.deviceName << "; validation=" << validate << '\n';
            return true;
        }
        return false;
    }

    Buffer buffer(uint64_t size, VkBufferUsageFlags2 usage, VkMemoryPropertyFlags properties)
    {
        Buffer result{};
        const VkBufferUsageFlags2CreateInfo usage2{.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO, .usage = usage | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT};
        const VkBufferCreateInfo create{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &usage2, .size = size};
        checked(vkCreateBuffer(device, &create, nullptr, &result.buffer));
        VkMemoryRequirements requirements; vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memory; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t type = 0;
        while (type < memory.memoryTypeCount && (!(requirements.memoryTypeBits & (1u << type)) ||
            (memory.memoryTypes[type].propertyFlags & properties) != properties)) { ++type; }
        tests::requireGpuPage(type < memory.memoryTypeCount, "Memory type unavailable");
        const VkMemoryAllocateFlagsInfo flags{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        const VkMemoryAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &flags, .allocationSize = requirements.size, .memoryTypeIndex = type};
        checked(vkAllocateMemory(device, &allocation, nullptr, &result.memory));
        checked(vkBindBufferMemory(device, result.buffer, result.memory, 0));
        const VkBufferDeviceAddressInfo address{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = result.buffer};
        result.address = vkGetBufferDeviceAddress(device, &address); buffers.push_back(result); return result;
    }
};

void verifyGpuPage(DecompressionProbe& probe, std::span<const uint8_t> stored,
    const scene::MeshletStreamGpuPage& metadata, std::span<const uint8_t> decoded, uint32_t repeats)
{
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto input = probe.buffer(stored.size(), VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, host);
    const auto compressed = probe.buffer(stored.size(), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto output = probe.buffer(decoded.size(), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto readback = probe.buffer(decoded.size(), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, host);
    void* mapped;
    checked(vkMapMemory(probe.device, input.memory, 0, stored.size(), 0, &mapped)); std::memcpy(mapped, stored.data(), stored.size()); vkUnmapMemory(probe.device, input.memory);
    VkCommandBuffer command;
    const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = probe.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    checked(vkAllocateCommandBuffers(probe.device, &allocate, &command));
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        checked(vkResetCommandBuffer(command, 0));
        const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; checked(vkBeginCommandBuffer(command, &begin));
        const auto barrier = [&](VkBuffer buffer, uint64_t offset, uint64_t size, VkPipelineStageFlags2 beforeStage,
            VkAccessFlags2 beforeAccess, VkPipelineStageFlags2 afterStage, VkAccessFlags2 afterAccess) {
            const VkBufferMemoryBarrier2 range{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask = beforeStage, .srcAccessMask = beforeAccess, .dstStageMask = afterStage, .dstAccessMask = afterAccess,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buffer, .offset = offset, .size = size};
            const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &range};
            vkCmdPipelineBarrier2(command, &dependency);
        };
        barrier(compressed.buffer, 0, stored.size(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        barrier(output.buffer, 0, decoded.size(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        std::vector<VkDecompressMemoryRegionEXT> regions;
        for (const auto& tile : metadata.tiles) {
            const VkBufferCopy copy{tile.sourceOffset, tile.codec ? tile.sourceOffset : tile.destinationOffset, tile.storedBytes};
            vkCmdCopyBuffer(command, input.buffer, tile.codec ? compressed.buffer : output.buffer, 1, &copy);
            if (tile.codec) { regions.push_back({compressed.address + tile.sourceOffset, output.address + tile.destinationOffset, tile.storedBytes, tile.decodedBytes}); }
        }
        for (const auto& tile : metadata.tiles) {
            if (!tile.codec) { continue; }
            barrier(compressed.buffer, tile.sourceOffset, tile.storedBytes, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT);
            barrier(output.buffer, tile.destinationOffset, tile.decodedBytes, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT);
        }
        const VkDecompressMemoryInfoEXT decompress{.sType = VK_STRUCTURE_TYPE_DECOMPRESS_MEMORY_INFO_EXT,
            .decompressionMethod = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT, .regionCount = uint32_t(regions.size()), .pRegions = regions.data()};
        if (!regions.empty()) { vkCmdDecompressMemoryEXT(command, &decompress); }
        barrier(output.buffer, 0, decoded.size(), VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferCopy copy{0, 0, decoded.size()}; vkCmdCopyBuffer(command, output.buffer, readback.buffer, 1, &copy);
        checked(vkEndCommandBuffer(command));
        const VkSubmitInfo submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
        checked(vkQueueSubmit(probe.queue, 1, &submit, VK_NULL_HANDLE)); checked(vkQueueWaitIdle(probe.queue));
        checked(vkMapMemory(probe.device, readback.memory, 0, decoded.size(), 0, &mapped));
        const bool equal = std::memcmp(mapped, decoded.data(), decoded.size()) == 0; vkUnmapMemory(probe.device, readback.memory);
        ASSERT_TRUE(equal) << "GPU output differs at repeat " << repeat;
    }
}

TEST(GpuPageCodec, ExtDecompressionByteOracle)
{
    DecompressionProbe probe;
    if (!probe.initialize()) { GTEST_SKIP() << "EXT GDeflate unavailable"; }
    const auto decoded = tests::makeMixedTileGpuPagePayload();
    std::vector<uint8_t> stored, cpu;
    std::string reason;
    ASSERT_TRUE(scene::encodeMeshletStreamGpuPage(decoded, true, stored, reason)) << reason;
    scene::MeshletStreamPageInfo page;
    page.payloadSize = stored.size(); page.uncompressedSize = decoded.size(); page.payloadFlags = scene::kMeshletStreamPayloadCompactPositions;
    page.clusterCount = 1; page.vertexCount = 3; page.triangleIndexCount = 3; page.attributeFlags = scene::kMeshletStreamPayloadAttributePosition;
    page.compressionMode = uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles);
    scene::MeshletStreamGpuPage metadata;
    ASSERT_TRUE(scene::inspectMeshletStreamGpuPage(page, stored, metadata, reason)) << reason;
    ASSERT_TRUE(scene::decodeMeshletStreamGpuPage(page, stored, cpu, reason)) << reason;
    ASSERT_EQ(cpu, decoded);
    ASSERT_EQ(metadata.tiles.size(), 3u); ASSERT_EQ(metadata.tiles[0].codec, 1u); ASSERT_EQ(metadata.tiles[1].codec, 0u);
    verifyGpuPage(probe, stored, metadata, decoded, 4);
    EXPECT_EQ(probe.errors.load(), 0u);
}
TEST(GpuPageCodec, MiniZorahSampleOracle)
{
    if (!std::getenv("METALLIC_TEST_MINIZORAH")) { GTEST_SKIP() << "Set METALLIC_TEST_MINIZORAH=1 to validate the local GPU-ready cache"; }
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
    const char* encodedOverride = std::getenv("METALLIC_MINIZORAH_STREAM_ASSET");
    const auto encodedPath = encodedOverride ? std::filesystem::path(encodedOverride) : root / ".cache/fast-streaming/MiniZorah.gdeflate.meshstream.bin";
    scene::MeshletStreamAsset original, encoded;
    std::string reason;
    ASSERT_TRUE(original.open(root / "Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin", reason)) << reason;
    ASSERT_TRUE(encoded.open(encodedPath, reason)) << reason;
    ASSERT_EQ(original.pageCount(), encoded.pageCount());
    ASSERT_EQ(original.groupCount(), encoded.groupCount());
    ASSERT_EQ(original.nodeCount(), encoded.nodeCount());
    ASSERT_TRUE(std::ranges::equal(original.refinedGroups(), encoded.refinedGroups()));
    ASSERT_EQ(std::memcmp(original.groups().data(), encoded.groups().data(), original.groups().size_bytes()), 0);
    ASSERT_EQ(std::memcmp(original.nodes().data(), encoded.nodes().data(), original.nodes().size_bytes()), 0);
    DecompressionProbe probe;
    if (!probe.initialize()) { GTEST_SKIP() << "EXT GDeflate unavailable"; }
    constexpr uint32_t kSamples = 1024;
    for (uint32_t sample = 0; sample < kSamples; ++sample) {
        const uint32_t index = uint32_t(uint64_t(sample) * (original.pageCount() - 1) / (kSamples - 1));
        SCOPED_TRACE(index);
        std::vector<uint8_t> referenceStorage, decoded;
        std::span<const uint8_t> reference, output;
        ASSERT_TRUE(scene::decodeMeshletStreamPayloadForDevice(original.pages()[index], original.pagePayload(index), referenceStorage, reference, reason)) << reason;
        ASSERT_TRUE(scene::decodeMeshletStreamPayloadForDevice(encoded.pages()[index], encoded.pagePayload(index), decoded, output, reason)) << reason;
        ASSERT_TRUE(std::ranges::equal(reference, output));
        scene::MeshletStreamGpuPage metadata;
        ASSERT_TRUE(scene::inspectMeshletStreamGpuPage(encoded.pages()[index], encoded.pagePayload(index), metadata, reason)) << reason;
        render::MeshletStreamClasPagePlan cpu, gpu;
        ASSERT_TRUE(render::buildMeshletStreamClasPagePlan(original.pages()[index], reference, index, index * original.maxPageClusters(), cpu, reason)) << reason;
        ASSERT_TRUE(render::buildMeshletStreamClasGpuPagePlan(metadata, index, index * original.maxPageClusters(), gpu, reason)) << reason;
        ASSERT_EQ(cpu.clusters.size(), gpu.clusters.size());
        ASSERT_EQ(std::memcmp(cpu.clusters.data(), gpu.clusters.data(), cpu.clusters.size() * sizeof(render::MeshletStreamClasClusterInput)), 0);
        if (sample % 16 == 0 || sample + 1 == kSamples) { verifyGpuPage(probe, encoded.pagePayload(index), metadata, reference, 1); }
    }
    EXPECT_EQ(probe.errors.load(), 0u);
    std::cout << "MiniZorah: 1024 CPU page oracles, 65 GPU page oracles (Raw copy/EXT decode); topology and CLAS plans agree\n";
}
// Opt-in, warm-memory microbenchmark. The same fixed pages and output bytes are
// installed in every case; no scene traversal, disk I/O, CLAS or rendering work
// is included. Keep it separate from the full-engine roaming performance gate.
TEST(GpuPageCodec, FixedPageUploadBenchmark)
{
    const char* reportPath = std::getenv("METALLIC_GPU_PAGE_BENCHMARK");
    if (!reportPath) { GTEST_SKIP() << "Set METALLIC_GPU_PAGE_BENCHMARK to an output JSON path"; }
    DecompressionProbe probe;
    if (!probe.initialize(!std::getenv("METALLIC_GPU_PAGE_NO_VALIDATION"))) { GTEST_SKIP() << "EXT GDeflate unavailable"; }
    if (!probe.timestampValidBits) { GTEST_SKIP() << "Queue timestamps unavailable"; }
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto duration) { return std::chrono::duration<double, std::milli>(duration).count(); };
    const auto distribution = [](const std::vector<double>& values) {
        auto ordered = values;
        std::ranges::sort(ordered);
        return nlohmann::json{{"p50", ordered[ordered.size() / 2]},
            {"p95", ordered[(ordered.size() * 95 + 99) / 100 - 1]}, {"samples", values}};
    };
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
    const char* assetOverride = std::getenv("METALLIC_MINIZORAH_STREAM_ASSET");
    const auto assetPath = assetOverride ? std::filesystem::path(assetOverride) : root / ".cache/fast-streaming/MiniZorah.gdeflate.meshstream.bin";
    scene::MeshletStreamAsset asset;
    std::string reason;
    ASSERT_TRUE(asset.open(assetPath, reason)) << reason;
    constexpr uint32_t kPageCount = 128, kWarmup = 4, kRepeats = 32;
    ASSERT_GE(asset.pageCount(), kPageCount);
    struct Page {
        scene::MeshletStreamPageInfo info, rawInfo;
        scene::MeshletStreamGpuPage metadata;
        std::vector<uint8_t> stored, raw;
        uint64_t inputOffset = 0, outputOffset = 0;
    };
    std::vector<Page> pages;
    std::vector<uint8_t> reference;
    uint64_t storedBytes = 0, payloadBytes = 0;
    nlohmann::json pageIds = nlohmann::json::array();
    for (uint32_t sample = 0; sample < kPageCount; ++sample) {
        const uint32_t index = uint32_t(uint64_t(sample) * (asset.pageCount() - 1) / (kPageCount - 1));
        Page page;
        page.info = asset.pages()[index];
        const auto stored = asset.pagePayload(index);
        page.stored.assign(stored.begin(), stored.end());
        ASSERT_TRUE(scene::inspectMeshletStreamGpuPage(page.info, page.stored, page.metadata, reason)) << reason;
        std::vector<uint8_t> decoded;
        ASSERT_TRUE(scene::decodeMeshletStreamGpuPage(page.info, page.stored, decoded, reason)) << reason;
        ASSERT_TRUE(scene::encodeMeshletStreamGpuPage(decoded, false, page.raw, reason)) << reason;
        page.rawInfo = page.info;
        page.rawInfo.payloadSize = page.raw.size();
        page.outputOffset = reference.size();
        page.inputOffset = storedBytes;
        reference.insert(reference.end(), decoded.begin(), decoded.end());
        storedBytes = (storedBytes + page.stored.size() + 15) & ~uint64_t(15);
        for (const auto& tile : page.metadata.tiles) { payloadBytes += tile.storedBytes; }
        pages.push_back(std::move(page));
        pageIds.push_back(index);
    }
    ASSERT_FALSE(reference.empty());
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto input = probe.buffer(std::max(storedBytes, uint64_t(reference.size())), VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, host);
    const auto compressed = probe.buffer(storedBytes, VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto output = probe.buffer(reference.size(), VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto readback = probe.buffer(reference.size(), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, host);
    struct Resources {
        VkDevice device; VkDeviceMemory input;
        VkQueryPool queries = VK_NULL_HANDLE;
        void* mapped = nullptr;
        ~Resources() { vkDeviceWaitIdle(device); if (mapped) { vkUnmapMemory(device, input); } if (queries) { vkDestroyQueryPool(device, queries, nullptr); } }
    } resources{probe.device, input.memory};
    checked(vkMapMemory(probe.device, input.memory, 0, VK_WHOLE_SIZE, 0, &resources.mapped));
    const VkQueryPoolCreateInfo queryInfo{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 3};
    checked(vkCreateQueryPool(probe.device, &queryInfo, nullptr, &resources.queries));
    VkCommandBuffer command;
    const VkCommandBufferAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = probe.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    checked(vkAllocateCommandBuffers(probe.device, &allocate, &command));
    const auto barrier = [&](VkPipelineStageFlags2 beforeStage, VkAccessFlags2 beforeAccess,
        VkPipelineStageFlags2 afterStage, VkAccessFlags2 afterAccess) {
        const VkMemoryBarrier2 memory{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = beforeStage, .srcAccessMask = beforeAccess, .dstStageMask = afterStage, .dstAccessMask = afterAccess};
        const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memory};
        vkCmdPipelineBarrier2(command, &dependency);
    };
    const auto submit = [&]() {
        const VkSubmitInfo submission{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
        checked(vkQueueSubmit(probe.queue, 1, &submission, VK_NULL_HANDLE));
        checked(vkQueueWaitIdle(probe.queue));
    };
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
    vkGetPhysicalDeviceProperties2(probe.physical, &properties);
    nlohmann::json report{{"asset", assetPath.generic_string()}, {"pageIds", pageIds}, {"warmup", kWarmup}, {"repeats", kRepeats},
        {"device", properties.properties.deviceName}, {"driver", driver.driverInfo}, {"validation", probe.validationEnabled},
        {"decodedBytes", reference.size()}, {"compressedPayloadBytes", payloadBytes}, {"storedEnvelopeSpanBytes", storedBytes},
        {"scope", "warm-memory, single CPU thread, same queue; excludes disk I/O, command recording, CLAS and rendering"},
        {"cases", nlohmann::json::array()}};
    for (uint32_t mode = 0; mode < 3; ++mode) {
        const bool gpu = mode == 2;
        const char* name = mode == 0 ? "raw_cpu_upload" : mode == 1 ? "gdeflate_cpu_upload" : "gdeflate_ext_upload";
        checked(vkResetCommandBuffer(command, 0));
        const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        checked(vkBeginCommandBuffer(command, &begin));
        vkCmdResetQueryPool(command, resources.queries, 0, 3);
        barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, resources.queries, 0);
        std::vector<VkDecompressMemoryRegionEXT> regions;
        for (const auto& page : pages) {
            if (!gpu) {
                const VkBufferCopy copy{page.outputOffset, page.outputOffset, page.info.uncompressedSize};
                vkCmdCopyBuffer(command, input.buffer, output.buffer, 1, &copy);
                continue;
            }
            for (const auto& tile : page.metadata.tiles) {
                const uint64_t source = page.inputOffset + tile.sourceOffset;
                const uint64_t destination = page.outputOffset + tile.destinationOffset;
                const VkBufferCopy copy{source, tile.codec ? source : destination, tile.storedBytes};
                vkCmdCopyBuffer(command, input.buffer, tile.codec ? compressed.buffer : output.buffer, 1, &copy);
                if (tile.codec) { regions.push_back({compressed.address + source, output.address + destination, tile.storedBytes, tile.decodedBytes}); }
            }
        }
        vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, resources.queries, 1);
        if (gpu && !regions.empty()) {
            barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT | VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT);
            const VkDecompressMemoryInfoEXT decompress{.sType = VK_STRUCTURE_TYPE_DECOMPRESS_MEMORY_INFO_EXT,
                .decompressionMethod = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
                .regionCount = uint32_t(regions.size()), .pRegions = regions.data()};
            vkCmdDecompressMemoryEXT(command, &decompress);
        }
        barrier(VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        vkCmdWriteTimestamp2(command, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, resources.queries, 2);
        checked(vkEndCommandBuffer(command));
        std::vector<double> preparation, upload, decode, total, wall;
        const uint64_t mask = probe.timestampValidBits == 64 ? UINT64_MAX : (uint64_t(1) << probe.timestampValidBits) - 1;
        std::vector<uint8_t> decoded;
        for (uint32_t repeat = 0; repeat < kWarmup + kRepeats; ++repeat) {
            const auto start = Clock::now();
            for (const auto& page : pages) {
                if (gpu) {
                    scene::MeshletStreamGpuPage metadata;
                    ASSERT_TRUE(scene::inspectMeshletStreamGpuPage(page.info, page.stored, metadata, reason)) << reason;
                    std::memcpy(static_cast<uint8_t*>(resources.mapped) + page.inputOffset, page.stored.data(), page.stored.size());
                } else {
                    std::span<const uint8_t> payload;
                    ASSERT_TRUE(scene::decodeMeshletStreamPayloadForDevice(mode == 0 ? page.rawInfo : page.info,
                        mode == 0 ? page.raw : page.stored, decoded, payload, reason)) << reason;
                    std::memcpy(static_cast<uint8_t*>(resources.mapped) + page.outputOffset, payload.data(), payload.size());
                }
            }
            const auto ready = Clock::now();
            submit();
            const auto complete = Clock::now();
            std::array<uint64_t, 3> timestamps;
            checked(vkGetQueryPoolResults(probe.device, resources.queries, 0, 3, sizeof(timestamps), timestamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            if (repeat < kWarmup) { continue; }
            preparation.push_back(milliseconds(ready - start));
            upload.push_back(double((timestamps[1] - timestamps[0]) & mask) * probe.timestampPeriod / 1e6);
            decode.push_back(double((timestamps[2] - timestamps[1]) & mask) * probe.timestampPeriod / 1e6);
            total.push_back(double((timestamps[2] - timestamps[0]) & mask) * probe.timestampPeriod / 1e6);
            wall.push_back(milliseconds(complete - start));
        }
        // The correctness readback is outside all timed intervals.
        checked(vkResetCommandBuffer(command, 0));
        checked(vkBeginCommandBuffer(command, &begin));
        barrier(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const VkBufferCopy copy{0, 0, reference.size()};
        vkCmdCopyBuffer(command, output.buffer, readback.buffer, 1, &copy);
        checked(vkEndCommandBuffer(command)); submit();
        void* mapped = nullptr;
        checked(vkMapMemory(probe.device, readback.memory, 0, reference.size(), 0, &mapped));
        const bool equal = std::memcmp(mapped, reference.data(), reference.size()) == 0;
        vkUnmapMemory(probe.device, readback.memory);
        ASSERT_TRUE(equal) << name;
        report["cases"].push_back({{"name", name}, {"gpuCopyPayloadBytes", gpu ? payloadBytes : reference.size()},
            {"decompressionRegions", regions.size()}, {"cpuPrepareMs", distribution(preparation)},
            {"gpuCopyMs", distribution(upload)}, {"gpuDecodeAndBarrierMs", distribution(decode)},
            {"gpuInstallMs", distribution(total)}, {"prepareSubmitWaitMs", distribution(wall)}, {"byteOracle", true}});
        std::cout << name << ": CPU=" << distribution(preparation)["p50"] << "ms GPU=" << distribution(total)["p50"] << "ms\n";
    }
    EXPECT_EQ(probe.errors.load(), 0u);
    report["validationErrors"] = probe.errors.load();
    std::ofstream file(reportPath);
    ASSERT_TRUE(file.is_open()) << reportPath;
    file << report.dump(2) << '\n';
    ASSERT_TRUE(file.good());
}
} // namespace
