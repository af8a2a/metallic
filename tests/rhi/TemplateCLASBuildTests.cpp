#include "RHITest.h"
#include "TestComputeProgram.h"
#include "TestResourceParameters.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/CommandSubmission.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/TessellationPatterns.h"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kPatternCount = 512;
constexpr uint32_t kVertexCapacity = 64;
constexpr uint32_t kTriangleCapacity = 96;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

std::unique_ptr<Buffer> buffer(Device& device, uint64_t bytes, MemoryLocation location = MemoryLocation::Device)
{
    return checked(device.createBuffer({.size = bytes,
        .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
            BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::AccelerationStructureStorage |
            BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
        .memoryLocation = location}));
}

BufferSlice slice(Buffer& value, BufferRange range = {})
{
    return checked(value.slice(range));
}

BufferSlice alignedSlice(Buffer& value, uint64_t alignment, uint64_t bytes)
{
    const uint64_t offset = (alignment - value.deviceAddress() % alignment) % alignment;
    return slice(value, {offset, bytes});
}

void upload(Buffer& value, const void* data, size_t bytes)
{
    void* mapped = value.map();
    require(mapped != nullptr, "Upload map failed");
    std::memcpy(mapped, data, bytes); value.flush(); value.unmap();
}

template<typename T>
std::vector<T> read(Buffer& value)
{
    value.invalidate();
    const void* mapped = value.map();
    require(mapped != nullptr, "Readback map failed");
    std::vector<T> result(value.desc().size / sizeof(T));
    std::memcpy(result.data(), mapped, result.size() * sizeof(T)); value.unmap();
    return result;
}

void hostReadBarrier(CommandBuffer& commands)
{
    const MemoryBarrierDesc barrier{
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    checked(commands.synchronize({.memory = {&barrier, 1}}));
}

struct TemplateVertex { float x, y, z, padding[5]; };
struct TemplateRay { float x, y, z, padding; };
struct TemplateRayObservation {
    uint32_t hit, cluster, geometry, primitive;
    float distance, u, v, front;
};
static_assert(sizeof(TemplateVertex) == 32 && sizeof(TemplateRayObservation) == 32);

std::vector<TemplateRayObservation> trace(RHITestContext& context, uint64_t blasAddress,
    Buffer& rays, uint32_t rayCount)
{
    auto& device = context.device;
    auto& queue = context.graphicsQueue;
    RayTracingGPUInstance instance;
    instance.customIndexAndMask = 37 | (1u << 24);
    instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
    instance.accelerationStructureReference = blasAddress;
    auto instances = buffer(device, sizeof(instance), MemoryLocation::HostUpload);
    upload(*instances, &instance, sizeof(instance));
    const auto sizes = checked(device.queryRayTracingAccelerationStructureBuildSizes({.instanceCount = 1}));
    auto tlas = checked(device.createRayTracingAccelerationStructure({.size = sizes.accelerationStructureSize}));
    const auto properties = checked(device.queryRayTracingAccelerationStructureProperties());
    auto scratch = buffer(device, sizes.buildScratchSize + properties.scratchAlignment);
    {
        bench::GPUCommands build(queue); checked(build.initialize(device));
        checked(build.commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
            .instanceBuffer = slice(*instances), .instanceCount = 1, .scratchBuffer = slice(*scratch)}));
        checked(build.submitAndWait());
    }
    const char* capabilities[]{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure",
        "spvRayTracingClusterAccelerationStructureNV"};
    std::string log;
    const auto shader = checked(compileSlangShaderToSpirv({.moduleName = "TemplateCLASProbe",
        .entryPointName = "templateCLASProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .capabilities = capabilities, .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log));
    const ComputeResourceBindingDesc layout[]{
        {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), ComputeResourceBindingKind::AccelerationStructure},
        {METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output)}};
    struct Constants { uint64_t rays; uint32_t count, reserved; };
    const Constants constants{rays.deviceAddress(), rayCount, 0};
    ComputeProgram program;
    checked(program.initialize(device, {.spirv = shader.spirv, .pushConstantSize = sizeof(constants),
        .bindings = layout, .resourceParameterSize = sizeof(UnifiedTopLevelProbeResources)}, log));
    auto output = buffer(device, sizeof(TemplateRayObservation) * rayCount, MemoryLocation::HostReadback);
    auto pool = checked(device.createCommandPool(queue));
    auto commands = checked(pool->createCommandBuffer());
    QueueSubmissionTracker tracker; checked(tracker.initialize(device, queue));
    RenderFrameContext frame;
    struct Drain {
        RenderFrameContext& frame; CommandPool& pool;
        ~Drain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
    } drain{frame, *pool};
    checked(frame.begin(0)); checked(commands->begin(frame.submissionContext()));
    checked(commands->retainResource(rays.retainAllocation()));
    commands->hostWriteBarrier();
    const ComputeDispatchBinding bindings[]{
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, scene), .accelerationStructure = tlas.get()},
        {.binding = METALLIC_RESOURCE_MEMBER(UnifiedTopLevelProbeResources, output), .buffer = output.get()}};
    checked(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings,
        .pushData = &constants, .pushDataSize = sizeof(constants), .groupCountX = (rayCount + 63) / 64}));
    hostReadBarrier(*commands);
    checked(commands->end());
    CommandBuffer* submitted[]{commands.get()};
    checked(tracker.submit({.commandBuffers = submitted}, frame)); checked(frame.wait(5'000'000'000ull));
    auto observations = read<TemplateRayObservation>(*output);
    bench::readbackEvidence(context, "readback.bin", std::span<const TemplateRayObservation>(observations));
    return observations;
}

class TemplateCLASBuildTest final : public RHITest {
public:
    TemplateCLASBuildTest() { type = RHITestType::Rendering; name = "clas_template_patterns"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"clas.template.patterns512.instantiate.ray.contracts"},
            bench::Layer::RHI, "ray-query", "core", {"readback.bin"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto created = bench::createTestDevice(context, {.applicationName = "CLAS template patterns",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("CLAS/ray query unavailable"); }
        if (!created) { return RHITestResult::fail("CLAS template device creation failed"); }
        auto& device = **created;
        if (!device.capabilities().clusterAccelerationStructure || !device.capabilities().rayQuery) {
            return RHITestResult::skip("CLAS/ray query unavailable");
        }
        RHITestContext active{device, *device.getQueue(QueueType::Graphics), context.outputDirectory,
            context.enableValidation, context.validationMessageCount, context.nsightCapture, context.evidence, context.deviceDesc};
        try { return runDevice(active); }
        catch (const std::exception& error) { (void)device.waitIdle(); return RHITestResult::fail(error.what()); }
    }
private:
    RHITestResult runDevice(RHITestContext& context)
    {
#ifndef VK_NV_cluster_acceleration_structure
        return RHITestResult::skip("CLAS unavailable in Vulkan headers");
#else
        auto& device = context.device;
        auto& queue = context.graphicsQueue;
        const auto properties = checked(device.queryClusterAccelerationStructureProperties());
        const ClusterAccelerationStructureTriangleBuildSizesDesc templateLimits{
            .maxClusterTriangleCount = kTriangleCapacity, .maxClusterVertexCount = kVertexCapacity,
            .maxGeometryIndexValue = 12, .maxTotalTriangleCount = kPatternCount * kTriangleCapacity,
            .maxTotalVertexCount = kPatternCount * kVertexCapacity, .maxAccelerationStructureCount = kPatternCount,
            .operation = ClusterAccelerationStructureTriangleOperation::BuildTemplate};
        auto instanceLimits = templateLimits;
        instanceLimits.operation = ClusterAccelerationStructureTriangleOperation::InstantiateTemplate;
        instanceLimits.flags = RayTracingAccelerationStructureBuildFlags::PreferFastBuild;
        for (const auto limits : {templateLimits, instanceLimits}) {
            auto invalid = limits;
            invalid.operation = static_cast<ClusterAccelerationStructureTriangleOperation>(255);
            require(hasError(device.queryClusterAccelerationStructureTriangleBuildSizes(invalid), Error::InvalidArgument),
                "Unknown triangle operation accepted by size query");
            invalid = limits; invalid.flags = RayTracingAccelerationStructureBuildFlags::AllowUpdate;
            require(hasError(device.queryClusterAccelerationStructureTriangleBuildSizes(invalid), Error::InvalidArgument),
                "Unsupported triangle flags accepted by size query");
        }
        const auto templateSizes = checked(device.queryClusterAccelerationStructureTriangleBuildSizes(templateLimits,
            ClusterAccelerationStructureDestinationMode::Implicit));
        const auto instanceSizes = checked(device.queryClusterAccelerationStructureTriangleBuildSizes(instanceLimits,
            ClusterAccelerationStructureDestinationMode::Implicit));
        auto templates = buffer(device, templateSizes.accelerationStructureSize + properties.templateStorageAlignment);
        auto instances = buffer(device, instanceSizes.accelerationStructureSize + properties.clusterStorageAlignment);
        auto templateStorage = alignedSlice(*templates, properties.templateStorageAlignment, templateSizes.accelerationStructureSize);
        const auto instanceStorage = alignedSlice(*instances, properties.clusterStorageAlignment, instanceSizes.accelerationStructureSize);
        auto scratch = buffer(device, std::max(templateSizes.buildScratchSize, instanceSizes.buildScratchSize) + properties.scratchAlignment);
        auto canonical = buffer(device, kPatternCount * kVertexCapacity * sizeof(TemplateVertex), MemoryLocation::HostUpload);
        auto vertices = buffer(device, canonical->desc().size, MemoryLocation::HostUpload);
        auto indices = buffer(device, kPatternCount * kTriangleCapacity * 3 * sizeof(uint32_t), MemoryLocation::HostUpload);
        auto bounds = buffer(device, 24 + properties.templateBoundsAlignment, MemoryLocation::HostUpload);
        auto boundsSlice = alignedSlice(*bounds, properties.templateBoundsAlignment, 24);
        const float limits[]{-1, -1, -1, 1600, 2, 2};
        std::vector<uint8_t> boundsData(bounds->desc().size);
        std::memcpy(boundsData.data() + boundsSlice.offset(), limits, sizeof(limits));
        upload(*bounds, boundsData.data(), boundsData.size());
        std::vector<TemplateVertex> canonicalData(kPatternCount * kVertexCapacity), vertexData(canonicalData.size());
        std::vector<uint32_t> indexData(kPatternCount * kTriangleCapacity * 3);
        std::vector<TemplateRay> rayData(kPatternCount + 1);
        std::vector<float> distances(kPatternCount);
        constexpr uint64_t templateStride = 80, instanceStride = 48;
        std::vector<uint8_t> templateRecords(16 + kPatternCount * templateStride);
        for (uint32_t index = 0; index < kPatternCount; ++index) {
            const auto pattern = makeTessellationPattern({1 + index % 8, 1 + (index / 8) % 8, 1 + index / 64});
            require(pattern.vertices.size() <= kVertexCapacity && pattern.triangles.size() <= kTriangleCapacity,
                "Tessellation pattern exceeds declared native bounds");
            const float height = .25f + float(index % 7) * .03f;
            for (uint32_t vertex = 0; vertex < pattern.vertices.size(); ++vertex) {
                const auto& bary = pattern.vertices[vertex];
                const uint32_t destination = index * kVertexCapacity + vertex;
                canonicalData[destination] = {float(bary[1]) / 65535.f, float(bary[2]) / 65535.f, 0, {}};
                vertexData[destination] = canonicalData[destination];
                vertexData[destination].x += float(index) * 3.f;
                vertexData[destination].z = height;
            }
            std::memcpy(indexData.data() + index * kTriangleCapacity * 3,
                pattern.triangles.data(), pattern.triangles.size() * 3 * sizeof(uint32_t));
            TemplateRay ray{0, 0, 2, 0};
            for (uint32_t vertex : pattern.triangles.front()) {
                const auto& position = vertexData[index * kVertexCapacity + vertex];
                ray.x += position.x / 3.f; ray.y += position.y / 3.f;
            }
            rayData[index] = ray; distances[index] = 2.f - height;
            ClusterAccelerationStructureTriangleTemplateIndirectBuildInfo record{};
            record.triangle.clusterId = 7;
            record.triangle.countsAndIndexType = METALLIC_CLAS_PACK_COUNTS(uint32_t(pattern.triangles.size()),
                uint32_t(pattern.vertices.size()), 0u, METALLIC_CLAS_INDEX_UINT32);
            record.triangle.geometryIndexAndFlags = 5u | METALLIC_CLAS_GEOMETRY_OPAQUE;
            record.triangle.indexAndVertexStrides = METALLIC_CLAS_PACK_STRIDES(4u, 32u);
            record.triangle.indexBufferAddress = indices->deviceAddress() + index * kTriangleCapacity * 12u;
            record.triangle.vertexBufferAddress = canonical->deviceAddress() + index * kVertexCapacity * sizeof(TemplateVertex);
            record.instantiationBoundingBoxLimitAddress = index % 2 ? boundsSlice.deviceAddress() : 0;
            VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV native{};
            native.clusterID = 7; native.triangleCount = uint32_t(pattern.triangles.size());
            native.vertexCount = uint32_t(pattern.vertices.size());
            native.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;
            native.baseGeometryIndexAndGeometryFlags.geometryIndex = 5;
            native.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
            native.indexBufferStride = 4; native.vertexBufferStride = 32;
            native.indexBuffer = record.triangle.indexBufferAddress; native.vertexBuffer = record.triangle.vertexBufferAddress;
            native.instantiationBoundingBoxLimit = record.instantiationBoundingBoxLimitAddress;
            require(sizeof(native) == sizeof(record) && std::memcmp(&native, &record, sizeof(record)) == 0,
                "Template build record differs from native ABI");
            std::memcpy(templateRecords.data() + 16 + index * templateStride, &record, sizeof(record));
        }
        rayData.back() = {-10, -10, 2, 0};
        upload(*canonical, canonicalData.data(), canonicalData.size() * sizeof(TemplateVertex));
        upload(*vertices, vertexData.data(), vertexData.size() * sizeof(TemplateVertex));
        upload(*indices, indexData.data(), indexData.size() * sizeof(uint32_t));
        auto rays = buffer(device, rayData.size() * sizeof(TemplateRay), MemoryLocation::HostUpload);
        upload(*rays, rayData.data(), rayData.size() * sizeof(TemplateRay));
        auto templateInfos = buffer(device, templateRecords.size(), MemoryLocation::HostUpload);
        upload(*templateInfos, templateRecords.data(), templateRecords.size());
        auto countUpload = buffer(device, 4, MemoryLocation::HostUpload);
        auto count = buffer(device, 8);
        auto addresses = buffer(device, 8 + kPatternCount * 16, MemoryLocation::HostReadback);
        auto sizes = buffer(device, 8 + kPatternCount * 8, MemoryLocation::HostReadback);
        std::array<BufferSlice, 3> templateInputs{slice(*indices), slice(*canonical), slice(*bounds)};
        ClusterAccelerationStructureTriangleIndirectBuildDesc build{
            .limits = templateLimits, .buildInfoBuffer = slice(*templateInfos, {16}), .buildInfoStride = templateStride,
            .buildInfoCountBuffer = slice(*count, {4, 4}), .destinationStorageBuffer = templateStorage,
            .destinationAddressBuffer = slice(*addresses, {8}), .destinationAddressStride = 16,
            .destinationSizeBuffer = slice(*sizes, {8}), .destinationSizeStride = 8,
            .scratchBuffer = slice(*scratch), .geometryBuffers = templateInputs};
        const auto execute = [&](const ClusterAccelerationStructureTriangleIndirectBuildDesc& command, uint32_t activeCount,
                                 uint64_t storageAlignment) {
            std::vector<uint64_t> addressWords(addresses->desc().size / 8, UINT64_MAX);
            std::vector<uint32_t> sizeWords(sizes->desc().size / 4, UINT32_MAX);
            upload(*addresses, addressWords.data(), addressWords.size() * 8);
            upload(*sizes, sizeWords.data(), sizeWords.size() * 4);
            upload(*countUpload, &activeCount, 4);
            bench::GPUCommands commands(queue); checked(commands.initialize(device));
            checked(commands.commands->copyBuffer(slice(*countUpload), command.buildInfoCountBuffer));
            checked(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(command));
            hostReadBarrier(*commands.commands); checked(commands.submitAndWait());
            addressWords = read<uint64_t>(*addresses); sizeWords = read<uint32_t>(*sizes);
            for (uint32_t index = 0; index < activeCount; ++index) {
                const uint64_t address = addressWords[1 + index * 2];
                const uint32_t size = sizeWords[2 + index * 2];
                require(address >= command.destinationStorageBuffer.deviceAddress() &&
                    address < command.destinationStorageBuffer.deviceAddress() + command.destinationStorageBuffer.size() &&
                    address % storageAlignment == 0 && size > 0 &&
                    size <= command.destinationStorageBuffer.deviceAddress() + command.destinationStorageBuffer.size() - address,
                    "Template operation output exceeds aligned implicit arena");
            }
            for (uint32_t index = 0; index < addressWords.size(); ++index) {
                const bool output = index >= 1 && (index - 1) % 2 == 0 && (index - 1) / 2 < activeCount;
                require(output || addressWords[index] == UINT64_MAX, "Template operation overwrote an address guard");
            }
            for (uint32_t index = 0; index < sizeWords.size(); ++index) {
                const bool output = index >= 2 && (index - 2) % 2 == 0 && (index - 2) / 2 < activeCount;
                require(output || sizeWords[index] == UINT32_MAX, "Template operation overwrote a size guard");
            }
            return addressWords;
        };
        {
            bench::GPUCommands commands(queue); checked(commands.initialize(device));
            auto invalid = build; invalid.buildInfoStride = 64;
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Triangle-sized template record accepted");
            invalid = build; invalid.destinationStorageBuffer = slice(*templates, {0, 1});
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Undersized template arena accepted");
            invalid = build; invalid.limits.maxClusterTriangleCount = 512;
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Template operation accepted excessive per-cluster bound");
            checked(commands.commands->end());
        }
        execute(build, 0, properties.templateStorageAlignment);
        const auto templateAddresses = execute(build, kPatternCount, properties.templateStorageAlignment);
        // An unsubmitted repeat build must retain every template source until
        // cancellation. The completed templates must then be self-contained.
        const std::weak_ptr<void> indexLifetime = indices->retainAllocation();
        {
            auto pool = checked(device.createCommandPool(queue));
            auto commands = checked(pool->createCommandBuffer()); checked(commands->begin());
            checked(commands->buildClusterAccelerationStructureTrianglesIndirect(build));
            templateInputs = {}; build.geometryBuffers = {}; boundsSlice = {};
            indices.reset(); canonical.reset(); bounds.reset();
            require(!indexLifetime.expired(), "Template build did not retain original indices");
            checked(commands->end()); checked(pool->reset());
            require(indexLifetime.expired(), "Cancelled template build leaked its original indices");
        }
        std::vector<uint8_t> instanceRecords(16 + kPatternCount * instanceStride);
        for (uint32_t index = 0; index < kPatternCount; ++index) {
            const uint32_t offset = index % 2 ? 7u : (uint32_t(-2) & 0xffffffu);
            const ClusterAccelerationStructureTriangleTemplateInstantiationInfo record{
                .clusterIdOffset = 1000 + index, .geometryIndexOffset = offset,
                .clusterTemplateAddress = templateAddresses[1 + index * 2],
                .vertexBufferAddress = vertices->deviceAddress() + index * kVertexCapacity * sizeof(TemplateVertex),
                .vertexBufferStride = sizeof(TemplateVertex)};
            VkClusterAccelerationStructureInstantiateClusterInfoNV native{};
            native.clusterIdOffset = record.clusterIdOffset; native.geometryIndexOffset = offset;
            native.clusterTemplateAddress = record.clusterTemplateAddress;
            native.vertexBuffer = {record.vertexBufferAddress, record.vertexBufferStride};
            require(sizeof(native) == sizeof(record) && std::memcmp(&native, &record, sizeof(record)) == 0,
                "Template instantiation record differs from native ABI");
            std::memcpy(instanceRecords.data() + 16 + index * instanceStride, &record, sizeof(record));
        }
        auto instanceInfos = buffer(device, instanceRecords.size(), MemoryLocation::HostUpload);
        upload(*instanceInfos, instanceRecords.data(), instanceRecords.size());
        std::array<BufferSlice, 1> instanceInputs{slice(*vertices)};
        build.limits = instanceLimits; build.buildInfoBuffer = slice(*instanceInfos, {16}); build.buildInfoStride = instanceStride;
        build.destinationStorageBuffer = instanceStorage; build.geometryBuffers = instanceInputs;
        build.templateStorageBuffer = templateStorage;
        {
            bench::GPUCommands commands(queue); checked(commands.initialize(device));
            auto invalid = build; invalid.buildInfoStride = 24;
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Undersized instantiation record accepted");
            invalid = build; invalid.templateStorageBuffer = {};
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Instantiation without retained template storage accepted");
            invalid = build; invalid.buildInfoCountBuffer = slice(*count, {4, 3});
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Instantiation accepted undersized count");
            invalid = build; invalid.scratchBuffer = slice(*scratch, {0, 1});
            require(hasError(commands.commands->buildClusterAccelerationStructureTrianglesIndirect(invalid), Error::InvalidArgument),
                "Instantiation accepted undersized scratch");
            checked(commands.commands->end());
        }
        execute(build, 0, properties.clusterStorageAlignment);
        execute(build, 1, properties.clusterStorageAlignment);
        const auto clusterAddresses = execute(build, kPatternCount, properties.clusterStorageAlignment);
        const auto blasSizes = checked(device.queryClusterAccelerationStructureBottomLevelBuildSizes({
            .maxClusterCountPerAccelerationStructure = kPatternCount, .maxTotalClusterCount = kPatternCount}));
        auto blas = buffer(device, blasSizes.accelerationStructureSize + properties.bottomLevelStorageAlignment);
        const auto blasStorage = alignedSlice(*blas, properties.bottomLevelStorageAlignment, blasSizes.accelerationStructureSize);
        uint64_t blasAddress = 0;
        {
            std::array<uint64_t, kPatternCount> refs{};
            for (uint32_t index = 0; index < kPatternCount; ++index) { refs[index] = clusterAddresses[1 + index * 2]; }
            auto references = buffer(device, sizeof(refs), MemoryLocation::HostUpload); upload(*references, refs.data(), sizeof(refs));
            const ClusterAccelerationStructureBottomLevelBuildInfo info{.clusterReferencesCount = kPatternCount,
                .clusterReferencesAddress = references->deviceAddress()};
            auto infos = buffer(device, sizeof(info), MemoryLocation::HostUpload); upload(*infos, &info, sizeof(info));
            auto output = buffer(device, 8, MemoryLocation::HostReadback);
            auto blasScratch = buffer(device, blasSizes.buildScratchSize + properties.scratchAlignment);
            bench::GPUCommands commands(queue); checked(commands.initialize(device));
            checked(commands.commands->buildClusterAccelerationStructureBottomLevels({
                .maxClusterCountPerAccelerationStructure = kPatternCount, .maxTotalClusterCount = kPatternCount,
                .buildInfoBuffer = slice(*infos), .destinationStorageBuffer = blasStorage,
                .destinationAddressBuffer = slice(*output), .scratchBuffer = slice(*blasScratch)}));
            hostReadBarrier(*commands.commands); checked(commands.submitAndWait());
            blasAddress = read<uint64_t>(*output)[0];
        }
        const auto observations = trace(context, blasAddress, *rays, uint32_t(rayData.size()));
        for (uint32_t index = 0; index < kPatternCount; ++index) {
            const auto& hit = observations[index];
            if (hit.hit != 1 || hit.cluster != 1007 + index || hit.geometry != (index % 2 ? 12u : 3u) ||
                hit.primitive != 0 || hit.front != 1.f || !std::isfinite(hit.distance) ||
                std::abs(hit.distance - distances[index]) > .0001f || std::abs(hit.u - 1.f / 3.f) > .01f ||
                std::abs(hit.v - 1.f / 3.f) > .01f) {
                throw std::runtime_error("Instantiated pattern ray mismatch at " + std::to_string(index) + ": " +
                    bench::Json::array({hit.hit, hit.cluster, hit.geometry, hit.primitive, hit.distance, hit.u, hit.v, hit.front}).dump());
            }
        }
        require(observations.back().hit == 0, "Instantiated pattern miss ray hit geometry");
        {
            auto pool = checked(device.createCommandPool(queue));
            auto commands = checked(pool->createCommandBuffer()); checked(commands->begin());
            checked(commands->buildClusterAccelerationStructureTrianglesIndirect(build));
            const std::weak_ptr<void> templateLifetime = templates->retainAllocation();
            const std::weak_ptr<void> vertexLifetime = vertices->retainAllocation();
            templateStorage = {}; build.templateStorageBuffer = {}; instanceInputs = {}; build.geometryBuffers = {};
            templates.reset(); vertices.reset();
            require(!templateLifetime.expired() && !vertexLifetime.expired(), "Instantiation did not retain GPU-addressed inputs");
            checked(commands->end()); checked(pool->reset());
            require(templateLifetime.expired() && vertexLifetime.expired(), "Cancelled instantiation leaked inputs");
        }
        return RHITestResult::pass("512 template patterns, native ABI, signed geometry/cluster offsets, retired source indices, GPU counts 0/1/512, strided bounds and cancellation");
#endif
    }
};

METALLIC_REGISTER_RHI_TEST(TemplateCLASBuildTest);

} // namespace
} // namespace metallic::tests
