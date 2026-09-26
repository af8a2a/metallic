#include "RhiTest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"

#include <type_traits>

namespace metallic::tests {
namespace {

class ValidateDeviceTest : public RhiTest {
public:
    ValidateDeviceTest()
    {
        type = RhiTestType::Validation;
        name = "validate_device";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::Queue* graphicsQueue = context.device.getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("graphics queue is unavailable");
        }
        if (graphicsQueue->type() != render::QueueType::Graphics) {
            return RhiTestResult::fail("graphics queue reported the wrong type");
        }
        render::Queue* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (context.device.capabilities().independentCopyQueue != (copyQueue != nullptr)) {
            return RhiTestResult::fail(
                "independentCopyQueue capability does not match QueueType::Copy availability");
        }
        if (copyQueue != nullptr &&
            (copyQueue == graphicsQueue || copyQueue->type() != render::QueueType::Copy)) {
            return RhiTestResult::fail("copy queue did not expose an independent Copy wrapper");
        }

        render::Result<> result = context.device.waitIdle();
        if (!result) {
            return RhiTestResult::fail(std::string("Device::waitIdle returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

class OptionalFeatureSoftRequestTest : public RhiTest {
public:
    OptionalFeatureSoftRequestTest()
    {
        type = RhiTestType::Validation;
        name = "optional_feature_soft_request";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Optional Feature Soft Request Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enablePushDescriptor = true,
                .enableClusterAccelerationStructure = true,
                .enablePartitionedAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(optional features) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(optional features) returned a null device");
        }

        const render::DeviceCapabilities& capabilities = device->capabilities();
        if (capabilities.rayQuery && !capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail("rayQuery capability was enabled without acceleration structure support");
        }
        if (capabilities.rayTracingPositionFetch && !capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail("position fetch was enabled without acceleration structure support");
        }
        if (capabilities.clusterAccelerationStructure && !capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail(
                "clusterAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (capabilities.partitionedAccelerationStructure && !capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail(
                "partitionedAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (capabilities.bindlessDescriptorHeap &&
            (capabilities.maxBindlessSamplers == 0 ||
                capabilities.maxBindlessSampledImages == 0 ||
                capabilities.maxBindlessBuffers == 0)) {
            return RhiTestResult::fail("bindless descriptor heap capability reported zero capacity");
        }

        result = device->waitIdle();
        if (!result) {
            return RhiTestResult::fail(
                std::string("Device::waitIdle(optional features) returned ") + toString(result));
        }

        return RhiTestResult::pass();
    }
};

class ShaderObjectRequiredTest : public RhiTest {
public:
    ShaderObjectRequiredTest()
    {
        type = RhiTestType::Validation;
        name = "shader_object_required";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!render::DeviceDesc{}.enableShaderObject) {
            return RhiTestResult::fail("DeviceDesc must enable required shader objects by default");
        }
        if (!context.device.capabilities().shaderObject) {
            return RhiTestResult::fail("A successfully created device must expose shader object support");
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
            return RhiTestResult::fail(
                std::string("createDevice(enableShaderObject=false) must reject the request without a device, got ") +
                toString(result));
        }

        return RhiTestResult::pass("Shader objects are enabled by default and cannot be disabled");
    }
};

class ScenePathNormalizationTest : public RhiTest {
public:
    ScenePathNormalizationTest()
    {
        type = RhiTestType::Validation;
        name = "scene_path_normalization";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::error_code error;
        const std::filesystem::path originalWorkingDirectory =
            std::filesystem::current_path(error);
        if (error) {
            return RhiTestResult::fail("failed to query the current working directory");
        }

        const std::filesystem::path alternateWorkingDirectory =
            std::filesystem::temp_directory_path(error);
        if (error) {
            return RhiTestResult::fail("failed to query the temporary directory");
        }
        std::filesystem::current_path(alternateWorkingDirectory, error);
        if (error) {
            return RhiTestResult::fail("failed to switch to the RHI test output directory");
        }
        const std::filesystem::path normalizedRelative =
            render::normalizedScenePath("Asset/meet_mat.glb");
        std::filesystem::current_path(originalWorkingDirectory, error);
        if (error) {
            return RhiTestResult::fail("failed to restore the current working directory");
        }

        const std::filesystem::path normalizedAbsolute = render::normalizedScenePath(
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb");
        if (normalizedRelative != normalizedAbsolute) {
            return RhiTestResult::fail(
                "relative scene paths were resolved against the process working directory");
        }
        return RhiTestResult::pass();
    }
};

class ClusterAccelerationStructureSupportTest : public RhiTest {
public:
    ClusterAccelerationStructureSupportTest()
    {
        type = RhiTestType::Validation;
        name = "cluster_acceleration_structure_support";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Cluster Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(cluster acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(cluster acceleration structure) returned a null device");
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
                return RhiTestResult::pass();
            }
            return RhiTestResult::fail(
                std::string("CLAS size query without capability returned ") + toString(result));
        }
        if (!capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail(
                "clusterAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("queryClusterAccelerationStructureTriangleBuildSizes returned ") + toString(result));
        }
        if (triangleSizes.accelerationStructureSize == 0 || triangleSizes.buildScratchSize == 0) {
            return RhiTestResult::fail("triangle CLAS size query returned zero build size");
        }

        render::ClusterAccelerationStructureBuildSizes bottomLevelSizes;
        result = device->queryClusterAccelerationStructureBottomLevelBuildSizes(render::ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                .maxClusterCountPerAccelerationStructure = 1,
                .maxTotalClusterCount = 1,
            }).transform([&](auto rhiValue) { bottomLevelSizes = std::move(rhiValue); });
        if (!result) {
            return RhiTestResult::fail(
                std::string("queryClusterAccelerationStructureBottomLevelBuildSizes returned ") + toString(result));
        }
        if (bottomLevelSizes.accelerationStructureSize == 0 || bottomLevelSizes.buildScratchSize == 0) {
            return RhiTestResult::fail("bottom-level CLAS size query returned zero build size");
        }

        return RhiTestResult::pass();
    }
};

class PartitionedAccelerationStructureSupportTest : public RhiTest {
public:
    PartitionedAccelerationStructureSupportTest()
    {
        type = RhiTestType::Validation;
        name = "partitioned_acceleration_structure_support";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> result = render::createDevice(render::DeviceDesc{
                .applicationName = "Metallic RHI Partitioned Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enablePartitionedAccelerationStructure = true,
            }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(partitioned acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(partitioned acceleration structure) returned a null device");
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
                return RhiTestResult::pass();
            }
            return RhiTestResult::fail(
                std::string("PTLAS size query without capability returned ") + toString(result));
        }
        if (!capabilities.rayTracingAccelerationStructure) {
            return RhiTestResult::fail(
                "partitionedAccelerationStructure capability was enabled without acceleration structure support");
        }
        if (!result) {
            return RhiTestResult::fail(
                std::string("queryPartitionedAccelerationStructureBuildSizes returned ") + toString(result));
        }
        if (sizes.accelerationStructureSize == 0 ||
            sizes.buildScratchSize == 0 ||
            sizes.operationInfoSize == 0 ||
            sizes.operationCountSize == 0 ||
            sizes.instanceWriteInfoSize == 0) {
            return RhiTestResult::fail("PTLAS size query returned zero build size");
        }

        return RhiTestResult::pass();
    }
};

class ExpectedResourceResultsTest final : public RhiTest {
public:
    ExpectedResourceResultsTest()
    {
        type = RhiTestType::Resource;
        name = "expected_resource_results";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        static_assert(std::is_same_v<Result<BufferSlice>, std::expected<BufferSlice, Error>>);
        static_assert(!std::is_copy_constructible_v<Result<std::unique_ptr<Buffer>>>);
        static_assert(std::is_move_constructible_v<Result<std::unique_ptr<Buffer>>>);

        Device emptyDevice;
        if (!hasError(emptyDevice.createBuffer({.size = 64}), Error::InvalidArgument) ||
            !hasError(emptyDevice.createSemaphore(), Error::InvalidArgument) ||
            !hasError(emptyDevice.resourceRegistry(), Error::InvalidArgument) ||
            !hasError(emptyDevice.reserveMemoryBudget(64), Error::InvalidArgument) ||
            !hasError(emptyDevice.textureAllocationSize({}), Error::InvalidArgument) ||
            !hasError(emptyDevice.queryRayTracingAccelerationStructureProperties(), Error::InvalidArgument) ||
            !hasError(BufferSlice{}.subslice(), Error::InvalidArgument)) {
            return RhiTestResult::fail("invalid objects did not return the expected error");
        }

        bool visited = false;
        const auto rejected = emptyDevice.createBuffer({.size = 64}).transform(
            [&](auto) { visited = true; });
        if (visited || !hasError(rejected, Error::InvalidArgument) ||
            std::string_view(resultToString(rejected)) != "InvalidArgument") {
            return RhiTestResult::fail("failed creation exposed a value or lost its error");
        }

        auto created = context.device.createBuffer({.size = 64, .usage = BufferUsageBits::Storage});
        if (!created) { return RhiTestResult::fail(resultToString(created)); }
        auto buffer = std::move(*created);
        if (!buffer || *created) {
            return RhiTestResult::fail("buffer ownership was not transferred from the result");
        }
        auto slice = buffer->slice(16, 32);
        if (!slice || slice->offset() != 16 || slice->size() != 32 ||
            !hasError(slice->subslice(33), Error::InvalidArgument)) {
            return RhiTestResult::fail("slice result lost its range or error");
        }
        auto remainder = slice->subslice(8);
        if (!remainder || remainder->offset() != 24 || remainder->size() != 24) {
            return RhiTestResult::fail("subslice default size did not preserve the remaining range");
        }
        std::weak_ptr<void> allocation = buffer->retainAllocation();
        buffer.reset();
        if (allocation.expired()) {
            return RhiTestResult::fail("returned slices did not retain the buffer allocation");
        }
        slice = makeError(Error::Failure);
        remainder = makeError(Error::Failure);
        if (!allocation.expired()) {
            return RhiTestResult::fail("discarded result values leaked a buffer allocation");
        }
        return RhiTestResult::pass();
    }
};

class ExpectedBindlessResultsTest final : public RhiTest {
public:
    ExpectedBindlessResultsTest()
    {
        type = RhiTestType::Resource;
        name = "expected_bindless_results";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        auto device = createDevice({.applicationName = "Expected bindless results",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true});
        if (hasError(device, Error::Unsupported)) { return RhiTestResult::skip(resultToString(device)); }
        if (!device) { return RhiTestResult::fail(resultToString(device)); }
        auto created = (*device)->createBindlessHeap({
            .maxSamplers = 1, .maxSampledImages = 1, .maxStorageImages = 1, .maxBuffers = 1});
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip(resultToString(created)); }
        if (!created) { return RhiTestResult::fail(resultToString(created)); }
        auto& heap = **created;
        using Allocate = Result<BindlessHandle> (BindlessHeap::*)();
        const std::pair<Allocate, BindlessHandleKind> cases[] = {
            {&BindlessHeap::allocateSampler, BindlessHandleKind::Sampler},
            {&BindlessHeap::allocateSampledImage, BindlessHandleKind::SampledImage},
            {&BindlessHeap::allocateStorageImage, BindlessHandleKind::StorageImage},
            {&BindlessHeap::allocateBuffer, BindlessHandleKind::Buffer},
            {&BindlessHeap::allocateAccelerationStructure, BindlessHandleKind::AccelerationStructure},
            {&BindlessHeap::allocatePartitionedAccelerationStructure, BindlessHandleKind::PartitionedAccelerationStructure},
        };
        for (const auto& [allocate, kind] : cases) {
            const auto handle = (heap.*allocate)();
            if (!handle || !handle->valid() || handle->kind != kind) {
                return RhiTestResult::fail("allocation did not return the requested handle kind");
            }
            // Sampled and storage images share one pool with the summed capacity.
            const bool image = kind == BindlessHandleKind::SampledImage || kind == BindlessHandleKind::StorageImage;
            auto second = image ? (heap.*allocate)() : Result<BindlessHandle>(makeError(Error::Unsupported));
            if (image && (!second || second->kind != kind || second->index == handle->index)) {
                return RhiTestResult::fail("shared image pool did not expose both slots");
            }
            if (!hasError((heap.*allocate)(), Error::OutOfMemory)) {
                return RhiTestResult::fail("exhausted allocation did not return OutOfMemory");
            }
            heap.release(*handle);
            const auto reused = (heap.*allocate)();
            if (!reused || reused->index != handle->index) {
                return RhiTestResult::fail("released slot was not reusable");
            }
            heap.release(*reused);
            if (second) { heap.release(*second); }
        }
        BindlessHeap moved = std::move(heap);
        for (const auto& [allocate, kind] : cases) {
            if (!hasError((heap.*allocate)(), Error::InvalidArgument)) {
                return RhiTestResult::fail("moved-from heap did not return InvalidArgument");
            }
        }
        return RhiTestResult::pass();
    }
};

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
