#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"

#include <type_traits>
#include <limits>
#include <utility>

namespace metallic::tests {
namespace {

class ValidateDeviceTest : public RHITest {
public:
    ValidateDeviceTest()
    {
        type = RHITestType::Validation;
        name = "validate_device";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::Queue* graphicsQueue = context.device.getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RHITestResult::fail("graphics queue is unavailable");
        }
        if (graphicsQueue->type() != render::QueueType::Graphics) {
            return RHITestResult::fail("graphics queue reported the wrong type");
        }
        render::Queue* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (context.device.capabilities().independentCopyQueue != (copyQueue != nullptr)) {
            return RHITestResult::fail(
                "independentCopyQueue capability does not match QueueType::Copy availability");
        }
        if (copyQueue != nullptr &&
            (copyQueue == graphicsQueue || copyQueue->type() != render::QueueType::Copy)) {
            return RHITestResult::fail("copy queue did not expose an independent Copy wrapper");
        }

        render::Result<> result = context.device.waitIdle();
        if (!result) {
            return RHITestResult::fail(std::string("Device::waitIdle returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

class OptionalFeatureSoftRequestTest : public RHITest {
public:
    OptionalFeatureSoftRequestTest()
    {
        type = RHITestType::Validation;
        name = "optional_feature_soft_request";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Optional Feature Soft Request Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enableClusterAccelerationStructure = true,
                .enablePartitionedAccelerationStructure = true,
                .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{
                    .enablePushDescriptor = true,
                },
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RHITestResult::fail(
                std::string("createDevice(optional features) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RHITestResult::fail("createDevice(optional features) returned a null device");
        }

        const render::DeviceCapabilities& capabilities = device->capabilities();
        if (capabilities.rayQuery && !capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail("rayQuery capability was enabled without acceleration structure support");
        }
        if (capabilities.rayTracingPositionFetch && !capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail("position fetch was enabled without acceleration structure support");
        }
        if (capabilities.clusterAccelerationStructure && !capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail(
                "clusterAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (capabilities.partitionedAccelerationStructure && !capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail(
                "partitionedAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (capabilities.bindlessDescriptorHeap &&
            (capabilities.maxBindlessSamplers == 0 ||
                capabilities.maxBindlessSampledImages == 0 ||
                capabilities.maxBindlessBuffers == 0)) {
            return RHITestResult::fail("bindless descriptor heap capability reported zero capacity");
        }

        result = device->waitIdle();
        if (!result) {
            return RHITestResult::fail(
                std::string("Device::waitIdle(optional features) returned ") + toString(result));
        }

        return RHITestResult::pass();
    }
};

class ShaderObjectRequiredTest : public RHITest {
public:
    ShaderObjectRequiredTest()
    {
        type = RHITestType::Validation;
        name = "shader_object_required";
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!render::DeviceDesc{}.enableShaderObject) {
            return RHITestResult::fail("DeviceDesc must enable required shader objects by default");
        }
        if (!context.device.capabilities().shaderObject) {
            return RHITestResult::fail("A successfully created device must expose shader object support");
        }

        // Reject this invalid request before creating another Vulkan device.
        // Feature-off driver-cache reproductions belong to the standalone app.
        std::unique_ptr<render::Device> rejectedDevice;
        const render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Required Shader Object Test",
                .enableValidation = context.enableValidation,
                .enableShaderObject = false,
            }).transform([&](auto rhiValue) { rejectedDevice = std::move(rhiValue); });
        if (!render::hasError(result, render::Error::InvalidArgument) || rejectedDevice != nullptr) {
            return RHITestResult::fail(
                std::string("createDevice(enableShaderObject=false) must reject the request without a device, got ") +
                toString(result));
        }

        return RHITestResult::pass("Shader objects are enabled by default and cannot be disabled");
    }
};

class ScenePathNormalizationTest : public RHITest {
public:
    ScenePathNormalizationTest()
    {
        type = RHITestType::Validation;
        name = "scene_path_normalization";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::error_code error;
        const std::filesystem::path originalWorkingDirectory =
            std::filesystem::current_path(error);
        if (error) {
            return RHITestResult::fail("failed to query the current working directory");
        }

        const std::filesystem::path alternateWorkingDirectory =
            std::filesystem::temp_directory_path(error);
        if (error) {
            return RHITestResult::fail("failed to query the temporary directory");
        }
        std::filesystem::current_path(alternateWorkingDirectory, error);
        if (error) {
            return RHITestResult::fail("failed to switch to the RHI test output directory");
        }
        const std::filesystem::path normalizedRelative =
            render::normalizedScenePath("Asset/meet_mat.glb");
        std::filesystem::current_path(originalWorkingDirectory, error);
        if (error) {
            return RHITestResult::fail("failed to restore the current working directory");
        }

        const std::filesystem::path normalizedAbsolute = render::normalizedScenePath(
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb");
        if (normalizedRelative != normalizedAbsolute) {
            return RHITestResult::fail(
                "relative scene paths were resolved against the process working directory");
        }
        return RHITestResult::pass();
    }
};

class ClusterAccelerationStructureSupportTest : public RHITest {
public:
    ClusterAccelerationStructureSupportTest()
    {
        type = RHITestType::Validation;
        name = "cluster_acceleration_structure_support";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Cluster Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RHITestResult::fail(
                std::string("createDevice(cluster acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RHITestResult::fail("createDevice(cluster acceleration structure) returned a null device");
        }

        render::ClusterAccelerationStructureBuildSizes triangleSizes;
        result = device->queryClusterAccelerationStructureTriangleBuildSizes(render::ClusterAccelerationStructureTriangleBuildSizesDesc{
                .maxClusterTriangleCount = 1,
                .maxClusterVertexCount = 3,
                .maxTotalTriangleCount = 1,
                .maxTotalVertexCount = 3,
            }).transform([&](auto rhiValue) { triangleSizes = std::move(rhiValue); });

        const render::DeviceCapabilities& capabilities = device->capabilities();
        if (!capabilities.clusterAccelerationStructure) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::pass();
            }
            return RHITestResult::fail(
                std::string("CLAS size query without capability returned ") + toString(result));
        }
        if (!capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail(
                "clusterAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("queryClusterAccelerationStructureTriangleBuildSizes returned ") + toString(result));
        }
        if (triangleSizes.accelerationStructureSize == 0 || triangleSizes.buildScratchSize == 0) {
            return RHITestResult::fail("triangle CLAS size query returned zero build size");
        }

        render::ClusterAccelerationStructureBuildSizes bottomLevelSizes;
        result = device->queryClusterAccelerationStructureBottomLevelBuildSizes(render::ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                .maxClusterCountPerAccelerationStructure = 1,
                .maxTotalClusterCount = 1,
            }).transform([&](auto rhiValue) { bottomLevelSizes = std::move(rhiValue); });
        if (!result) {
            return RHITestResult::fail(
                std::string("queryClusterAccelerationStructureBottomLevelBuildSizes returned ") + toString(result));
        }
        if (bottomLevelSizes.accelerationStructureSize == 0 || bottomLevelSizes.buildScratchSize == 0) {
            return RHITestResult::fail("bottom-level CLAS size query returned zero build size");
        }

        return RHITestResult::pass();
    }
};

class PartitionedAccelerationStructureSupportTest : public RHITest {
public:
    PartitionedAccelerationStructureSupportTest()
    {
        type = RHITestType::Validation;
        name = "partitioned_acceleration_structure_support";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Partitioned Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enablePartitionedAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RHITestResult::fail(
                std::string("createDevice(partitioned acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RHITestResult::fail("createDevice(partitioned acceleration structure) returned a null device");
        }

        render::PartitionedAccelerationStructureBuildSizes sizes;
        result = device->queryPartitionedAccelerationStructureBuildSizes(render::PartitionedAccelerationStructureBuildInputs{
                .instanceCount = 1,
                .partitionCount = 1,
                .maxInstancePerPartitionCount = 1,
                .maxOperationCount = 1,
            }).transform([&](auto rhiValue) { sizes = std::move(rhiValue); });

        const render::DeviceCapabilities& capabilities = device->capabilities();
        if (!capabilities.partitionedAccelerationStructure) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RHITestResult::pass();
            }
            return RHITestResult::fail(
                std::string("PTLAS size query without capability returned ") + toString(result));
        }
        if (!capabilities.rayTracingAccelerationStructure) {
            return RHITestResult::fail(
                "partitionedAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (!result) {
            return RHITestResult::fail(
                std::string("queryPartitionedAccelerationStructureBuildSizes returned ") + toString(result));
        }
        if (sizes.accelerationStructureSize == 0 ||
            sizes.buildScratchSize == 0 ||
            sizes.operationInfoSize == 0 ||
            sizes.operationCountSize == 0 ||
            sizes.instanceWriteInfoSize == 0) {
            return RHITestResult::fail("PTLAS size query returned zero build size");
        }

        return RHITestResult::pass();
    }
};

class ExpectedResourceResultsTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"result.resource.error.contract"}, bench::Layer::RHI, "core", "core");
    }

    ExpectedResourceResultsTest()
    {
        type = RHITestType::Resource;
        name = "expected_resource_results";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static_assert(std::is_same_v<Result<BufferSlice>, std::expected<BufferSlice, Error>>);
        static_assert(!std::is_copy_constructible_v<Result<std::unique_ptr<Buffer>>>);
        static_assert(std::is_move_constructible_v<Result<std::unique_ptr<Buffer>>>);

        Device emptyDevice;
        if (!hasError(emptyDevice.createBuffer({.size = 64}), Error::InvalidArgument) ||
            !hasError(emptyDevice.createSemaphore(), Error::InvalidArgument) ||
            !hasError(metallic::render::ResourceRegistry::forDevice(emptyDevice), Error::InvalidArgument) ||
            !hasError(emptyDevice.reserveMemoryBudget(64), Error::InvalidArgument) ||
            !hasError(emptyDevice.textureAllocationSize({}), Error::InvalidArgument) ||
            !hasError(emptyDevice.queryRayTracingAccelerationStructureProperties(), Error::InvalidArgument) ||
            !hasError(BufferSlice{}.subslice(), Error::InvalidArgument)) {
            return RHITestResult::fail("invalid objects did not return the expected error");
        }

        bool visited = false;
        const auto rejected = emptyDevice.createBuffer({.size = 64}).transform(
            [&](auto) { visited = true; });
        if (visited || !hasError(rejected, Error::InvalidArgument) ||
            std::string_view(resultToString(rejected)) != "InvalidArgument") {
            return RHITestResult::fail("failed creation exposed a value or lost its error");
        }

        auto created = context.device.createBuffer({.size = 64, .usage = BufferUsageBits::Storage});
        if (!created) { return RHITestResult::fail(resultToString(created)); }
        auto buffer = std::move(*created);
        if (!buffer || *created) {
            return RHITestResult::fail("buffer ownership was not transferred from the result");
        }
        auto slice = buffer->slice({16, 32});
        if (!slice || slice->offset() != 16 || slice->size() != 32 ||
            !hasError(slice->subslice({33}), Error::InvalidArgument)) {
            return RHITestResult::fail("slice result lost its range or error");
        }
        auto remainder = slice->subslice({8});
        if (!remainder || remainder->offset() != 24 || remainder->size() != 24) {
            return RHITestResult::fail("subslice default size did not preserve the remaining range");
        }
        std::weak_ptr<void> allocation = buffer->retainAllocation();
        buffer.reset();
        if (allocation.expired()) {
            return RHITestResult::fail("returned slices did not retain the buffer allocation");
        }
        slice = makeError(Error::Failure);
        remainder = makeError(Error::Failure);
        if (!allocation.expired()) {
            return RHITestResult::fail("discarded result values leaked a buffer allocation");
        }
        return RHITestResult::pass();
    }
};

// Scoped interception of this test's private device table; forwards real driver calls.
class BufferAddressQueryCounter {
public:
    explicit BufferAddressQueryCounter(render::Device& device)
        : functions_(*const_cast<VolkDeviceTable*>(render::vulkan::nativeDevice(device).functions))
    {
        original_ = functions_.vkGetBufferDeviceAddress;
        count_ = 0;
        functions_.vkGetBufferDeviceAddress = query;
    }
    ~BufferAddressQueryCounter() { functions_.vkGetBufferDeviceAddress = original_; }
    BufferAddressQueryCounter(const BufferAddressQueryCounter&) = delete;
    BufferAddressQueryCounter& operator=(const BufferAddressQueryCounter&) = delete;
    uint32_t count() const { return count_; }
private:
    static VKAPI_ATTR VkDeviceAddress VKAPI_CALL query(VkDevice device, const VkBufferDeviceAddressInfo* info)
    {
        ++count_;
        return original_(device, info);
    }
    VolkDeviceTable& functions_;
    inline static thread_local PFN_vkGetBufferDeviceAddress original_ = nullptr;
    inline static thread_local uint32_t count_ = 0;
};

class ExpectedBindlessResultsTest final : public RHITest {
public:
    ExpectedBindlessResultsTest()
    {
        type = RHITestType::Resource;
        name = "expected_bindless_results";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto device = createDevice({.applicationName = "Expected bindless results",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        if (hasError(device, Error::Unsupported)) { return RHITestResult::skip(resultToString(device)); }
        if (!device) { return RHITestResult::fail(resultToString(device)); }
        auto created = (*device)->createBindlessHeap({
            .maxSamplers = 1, .maxSampledImages = 1, .maxStorageImages = 1, .maxBuffers = 1});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip(resultToString(created)); }
        if (!created) { return RHITestResult::fail(resultToString(created)); }
        auto& heap = **created;
        const BindlessHandleKind cases[] = {BindlessHandleKind::Sampler, BindlessHandleKind::SampledImage,
            BindlessHandleKind::StorageImage, BindlessHandleKind::Buffer, BindlessHandleKind::AccelerationStructure};
        for (const auto kind : cases) {
            const auto handle = heap.allocate(kind);
            if (!handle || !handle->valid() || handle->kind != kind) {
                return RHITestResult::fail("allocation did not return the requested handle kind");
            }
            // Sampled and storage images share one pool with the summed capacity.
            const bool image = kind == BindlessHandleKind::SampledImage || kind == BindlessHandleKind::StorageImage;
            auto second = image ? heap.allocate(kind) : Result<BindlessHandle>(makeError(Error::Unsupported));
            if (image && (!second || second->kind != kind || second->index == handle->index)) {
                return RHITestResult::fail("shared image pool did not expose both slots");
            }
            if (!hasError(heap.allocate(kind), Error::OutOfMemory)) {
                return RHITestResult::fail("exhausted allocation did not return OutOfMemory");
            }
            heap.release(*handle);
            const auto reused = heap.allocate(kind);
            if (!reused || reused->index != handle->index) {
                return RHITestResult::fail("released slot was not reusable");
            }
            heap.release(*reused);
            if (second) { heap.release(*second); }
        }
        for (const auto kind : {BindlessHandleKind::Invalid, static_cast<BindlessHandleKind>(255)}) {
            if (!hasError(heap.allocate(kind), Error::InvalidArgument)) {
                return RHITestResult::fail("invalid handle kind did not return InvalidArgument");
            }
        }
        const auto bufferSlot = heap.allocate(BindlessHandleKind::Buffer);
        if (!bufferSlot || !hasError(heap.allocate(BindlessHandleKind::AccelerationStructure), Error::OutOfMemory)) {
            return RHITestResult::fail("buffer and AS allocations stopped sharing capacity");
        }
        heap.release(*bufferSlot);
        const auto asSlot = heap.allocate(BindlessHandleKind::AccelerationStructure);
        if (!asSlot || asSlot->shaderIndex != bufferSlot->shaderIndex) {
            return RHITestResult::fail("shared buffer/AS shader index changed on reuse");
        }
        heap.release(*asSlot);
        auto constant = (*device)->createBuffer({.size = 256, .usage = BufferUsageBits::Constant});
        auto storage = (*device)->createBuffer({.size = 256, .usage = BufferUsageBits::Storage});
        auto foreign = createDevice({.applicationName = "Foreign bindless resource",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        if (!constant || !storage || !foreign) { return RHITestResult::fail("buffer validation setup failed"); }
        auto foreignBuffer = (*foreign)->createBuffer({.size = 256,
            .usage = BufferUsageBits::Constant | BufferUsageBits::Storage});
        if (!foreignBuffer) { return RHITestResult::fail("foreign buffer creation failed"); }
        auto foreignView = (*foreign)->createBufferView(**foreignBuffer, {.type = BufferViewType::Constant});
        if (!foreignView) { return RHITestResult::fail("foreign view creation failed"); }
        const auto handle = heap.allocate(BindlessHandleKind::Buffer);
        if (!handle) { return RHITestResult::fail("buffer handle allocation failed"); }
        BufferAddressQueryCounter addressQueries(**device);
        auto constantView = (*device)->createBufferView(**constant, {.type = BufferViewType::Constant});
        auto storageView = (*device)->createBufferView(**storage, {.type = BufferViewType::Raw});
        if (!constantView || !storageView) { return RHITestResult::fail("local view creation failed"); }
        if (!hasError(heap.writeConstantBuffer(*handle, **storage), Error::InvalidArgument) ||
            !hasError((**constant).slice().and_then([&](const auto& bufferSlice) { return heap.writeStorageBuffer(*handle, bufferSlice); }), Error::InvalidArgument) ||
            !hasError(heap.writeConstantBuffer(*handle, **foreignBuffer), Error::InvalidArgument) ||
            !hasError((**foreignBuffer).slice().and_then([&](const auto& bufferSlice) { return heap.writeStorageBuffer(*handle, bufferSlice); }), Error::InvalidArgument) ||
            !hasError(heap.writeBufferView(*handle, **foreignView), Error::InvalidArgument) ||
            !hasError((*device)->createBufferView(**storage, {.type = BufferViewType::Constant}), Error::InvalidArgument) ||
            !hasError((*device)->createBufferView(**constant, {.type = BufferViewType::Raw}), Error::InvalidArgument)) {
            return RHITestResult::fail("buffer descriptor accepted foreign ownership or incompatible usage");
        }
        for (uint32_t i = 0; i < 3; ++i) {
            if (!heap.writeConstantBuffer(*handle, **constant) || !(**storage).slice().and_then([&](const auto& bufferSlice) { return heap.writeStorageBuffer(*handle, bufferSlice); }) ||
                !heap.writeBufferView(*handle, **constantView) || !heap.writeBufferView(*handle, **storageView)) {
                return RHITestResult::fail("valid buffer descriptor write failed");
            }
        }
        // Views and slices keep allocations alive after the public Buffer wrapper is gone.
        auto slice = (*storage)->slice();
        constant->reset();
        storage->reset();
        if (!slice || !heap.writeStorageBuffer(*handle, *slice) ||
            !heap.writeBufferView(*handle, **constantView) || !heap.writeBufferView(*handle, **storageView)) {
            return RHITestResult::fail("retained buffer allocation lost its cached address");
        }
        if (addressQueries.count() != 0) {
            return RHITestResult::fail("descriptor writes or view creation re-queried buffer device address");
        }
        heap.release(*handle);
        BindlessHeap moved = std::move(heap);
        for (const auto kind : cases) {
            if (!hasError(heap.allocate(kind), Error::InvalidArgument)) {
                return RHITestResult::fail("moved-from heap did not return InvalidArgument");
            }
        }
        return RHITestResult::pass();
    }
};

class CommandRecordingResultsTest final : public RHITest {
public:
    CommandRecordingResultsTest() { type = RHITestType::Validation; name = "command_recording_results"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        CommandBuffer empty;
        Texture emptyTexture;
        Buffer emptyBuffer;
        const Viewport viewport{0, 0, 16, 16, 0, 1};
        if (!hasError(empty.copyTexture({}), Error::InvalidArgument) ||
            !hasError(empty.copyTextureToBuffer({}), Error::InvalidArgument) ||
            !hasError(empty.copyBufferToTexture({}), Error::InvalidArgument) ||
            !hasError(empty.clearColorTexture(emptyTexture, TextureLayout::TransferDestination), Error::InvalidArgument) ||
            !hasError(empty.setViewport(viewport), Error::InvalidArgument) ||
            !hasError(empty.draw(3), Error::InvalidArgument) ||
            !hasError(empty.drawMeshTasks(1), Error::InvalidArgument) ||
            !hasError(empty.drawMeshTasksIndirect({}), Error::InvalidArgument)) {
            return RHITestResult::fail("empty command objects did not report InvalidArgument");
        }
        auto device = createDevice({.applicationName = "Command recording results",
            .enableValidation = context.enableValidation, .enableMeshShader = false});
        if (!device) { return RHITestResult::fail(resultToString(device)); }
        auto pool = (*device)->createCommandPool(*(*device)->getQueue(QueueType::Graphics));
        if (!pool) { return RHITestResult::fail(resultToString(pool)); }
        auto commands = (*pool)->createCommandBuffer();
        auto texture = (*device)->createTexture({.usage = TextureUsageBits::TransferSource | TextureUsageBits::TransferDestination,
            .format = Format::RGBA8Unorm, .width = 16, .height = 16});
        auto buffer = (*device)->createBuffer({.size = 1024,
            .usage = BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination | BufferUsageBits::Indirect});
        if (!commands || !texture || !buffer) { return RHITestResult::fail("command test resource setup failed"); }
        auto& command = **commands;
        auto full = (*buffer)->slice();
        auto emptyRange = (*buffer)->slice({1024, 0});
        auto shortRange = (*buffer)->slice({0, 1020});
        auto unaligned = (*buffer)->slice({1, 12});
        auto shortIndirect = (*buffer)->slice({0, 8});
        if (!full || !emptyRange || !shortRange || !unaligned || !shortIndirect) {
            return RHITestResult::fail("slice setup failed");
        }
        const BufferTextureRegion valid{.texture = texture->get(), .buffer = *full, .width = 16, .height = 16};
        if (!hasError(command.copyBufferToTexture(valid), Error::InvalidArgument) ||
            !hasError(command.setViewport(viewport), Error::InvalidArgument) ||
            !hasError(command.draw(3), Error::InvalidArgument)) {
            return RHITestResult::fail("commands outside recording did not report InvalidArgument");
        }
        if (!command.begin()) { return RHITestResult::fail("begin failed"); }
        auto zeroExtent = valid; zeroExtent.width = 0;
        auto badOffset = valid; badOffset.buffer = *emptyRange;
        auto badPitch = valid; badPitch.bufferRowPitch = 1;
        auto tooSmall = valid; tooSmall.buffer = *shortRange;
        auto padded = valid; padded.bufferRowPitch = 80;
        auto overflow = valid; overflow.depth = overflow.layerCount = std::numeric_limits<uint32_t>::max();
        for (const auto& region : {zeroExtent, badOffset, badPitch, tooSmall, padded, overflow}) {
            if (!hasError(command.copyBufferToTexture(region), Error::InvalidArgument) ||
                !hasError(command.copyTextureToBuffer(region), Error::InvalidArgument)) {
                return RHITestResult::fail("invalid copy region did not report InvalidArgument");
            }
        }
        if (!hasError(command.copyTexture({.source = texture->get(), .destination = texture->get()}), Error::InvalidArgument) ||
            !hasError(command.clearColorTexture(**texture, TextureLayout::ShaderRead), Error::InvalidArgument)) {
            return RHITestResult::fail("invalid texture command did not report InvalidArgument");
        }
        for (const float width : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
            auto invalid = viewport; invalid.width = width;
            if (!hasError(command.setViewport(invalid), Error::InvalidArgument)) {
                return RHITestResult::fail("invalid viewport did not report InvalidArgument");
            }
        }
        if (!hasError(command.drawMeshTasks(0), Error::InvalidArgument) ||
            !hasError(command.drawMeshTasks(1), Error::Unsupported) ||
            !hasError(command.drawMeshTasksIndirect(*unaligned), Error::InvalidArgument) ||
            !hasError(command.drawMeshTasksIndirect(*shortIndirect), Error::InvalidArgument) ||
            !hasError(command.drawMeshTasksIndirect(*full), Error::Unsupported)) {
            return RHITestResult::fail("mesh draw argument and capability errors were not distinguished");
        }
        auto foreignTexture = context.device.createTexture({.usage = TextureUsageBits::TransferSource | TextureUsageBits::TransferDestination,
            .format = Format::RGBA8Unorm, .width = 16, .height = 16});
        if (!foreignTexture) { return RHITestResult::fail(resultToString(foreignTexture)); }
        auto foreignRegion = valid; foreignRegion.texture = foreignTexture->get();
        if (!hasError(command.copyTextureToBuffer(foreignRegion), Error::InvalidArgument) ||
            !hasError(command.copyBufferToTexture(foreignRegion), Error::InvalidArgument) ||
            !hasError(command.copyTexture({.source = foreignTexture->get(), .destination = texture->get(),
                .width = 16, .height = 16}), Error::InvalidArgument) ||
            !hasError(command.clearColorTexture(**foreignTexture, TextureLayout::TransferDestination), Error::InvalidArgument)) {
            return RHITestResult::fail("foreign textures were accepted");
        }
        // Vulkan permits negative viewport height and reversed depth; keep these legal cases.
        if (!command.setViewport({0, 16, 16, -16, 1, 0}) || !command.end() ||
            !hasError(command.setViewport(viewport), Error::InvalidArgument) ||
            !hasError(command.draw(3), Error::InvalidArgument)) {
            return RHITestResult::fail("valid viewport or completed recording contract failed");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(CommandRecordingResultsTest);

class VulkanDeviceExtensionsTest final : public RHITest {
public:
    VulkanDeviceExtensionsTest()
    {
        type = RHITestType::Validation;
        name = "vulkan_device_extensions_contract";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        DeviceDesc description{.applicationName = "Device extension contract",
            .enableValidation = context.enableValidation};
        const auto& defaults = vulkan::deviceExtensions(std::as_const(description));
        if (!defaults.preferUnifiedImageLayouts || defaults.enableStreamline ||
            defaults.enableAftermath || defaults.enablePushDescriptor || defaults.shaderPrintf ||
            description.backendExtensions.has_value()) {
            return RHITestResult::fail("Empty extensions must preserve defaults without mutating the descriptor");
        }

        description.backendExtensions = 42;
        auto rejected = createDevice(description);
        if (!hasError(rejected, Error::InvalidArgument)) {
            return RHITestResult::fail("An unrelated extension payload must be rejected before initialization");
        }
        {
            DeviceDesc original{.backendExtensions = vulkan::VulkanDeviceExtensions{
                .enablePushDescriptor = true, .enableStreamline = true, .enableAftermath = true}};
            description.backendExtensions = original.backendExtensions;
            auto& copied = vulkan::deviceExtensions(description);
            copied.enablePushDescriptor = false;
            copied.enableStreamline = false;
            copied.enableAftermath = false;
            copied.preferUnifiedImageLayouts = false;
            const auto& source = vulkan::deviceExtensions(std::as_const(original));
            if (!source.enablePushDescriptor || !source.enableStreamline || !source.enableAftermath ||
                !source.preferUnifiedImageLayouts) {
                return RHITestResult::fail("Editing a descriptor copy changed the source extension value");
            }
        }
        auto created = createDevice(description);
        if (!created) { return RHITestResult::fail("Create device from independently owned extension copy"); }
        // Initialization snapshots configuration; the caller can discard its payload.
        description.backendExtensions.reset();
        const auto capabilities = vulkan::deviceCapabilities(**created);
        if (capabilities.unifiedImageLayouts || capabilities.pushDescriptor || capabilities.streamline ||
            capabilities.streamlineDlssSr || capabilities.streamlineDlssRr || capabilities.aftermath ||
            !(**created).capabilities().shaderObject) {
            return RHITestResult::fail("Backend capabilities did not reflect the copied request");
        }
        if (!(**created).waitIdle()) { return RHITestResult::fail("Extension-configured device failed waitIdle"); }
        return RHITestResult::pass("Owned extension copies, invalid payload rejection and independent capability snapshot");
    }
};

METALLIC_REGISTER_RHI_TEST(VulkanDeviceExtensionsTest);
METALLIC_REGISTER_RHI_TEST(ExpectedResourceResultsTest);
METALLIC_REGISTER_RHI_TEST(ExpectedBindlessResultsTest);
METALLIC_REGISTER_RHI_TEST(ValidateDeviceTest);
METALLIC_REGISTER_RHI_TEST(ShaderObjectRequiredTest);
METALLIC_REGISTER_RHI_TEST(ScenePathNormalizationTest);
METALLIC_REGISTER_RHI_TEST(OptionalFeatureSoftRequestTest);
METALLIC_REGISTER_RHI_TEST(ClusterAccelerationStructureSupportTest);
METALLIC_REGISTER_RHI_TEST(PartitionedAccelerationStructureSupportTest);

} // namespace
} // namespace metallic::tests
