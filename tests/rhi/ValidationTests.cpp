#include "RhiTest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"

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

        render::Result result = context.device.waitIdle();
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
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RHI Optional Feature Soft Request Test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableShaderObject = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enablePushDescriptor = true,
                .enableClusterAccelerationStructure = true,
                .enablePartitionedAccelerationStructure = true,
            },
            device);
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
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RHI Cluster Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enableClusterAccelerationStructure = true,
            },
            device);
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(cluster acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(cluster acceleration structure) returned a null device");
        }

        render::ClusterAccelerationStructureBuildSizes triangleSizes;
        result = device->queryClusterAccelerationStructureTriangleBuildSizes(
            render::ClusterAccelerationStructureTriangleBuildSizesDesc{
                .maxClusterTriangleCount = 1,
                .maxClusterVertexCount = 3,
                .maxTotalTriangleCount = 1,
                .maxTotalVertexCount = 3,
            },
            triangleSizes);

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
        result = device->queryClusterAccelerationStructureBottomLevelBuildSizes(
            render::ClusterAccelerationStructureBottomLevelBuildSizesDesc{
                .maxClusterCountPerAccelerationStructure = 1,
                .maxTotalClusterCount = 1,
            },
            bottomLevelSizes);
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
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic RHI Partitioned Acceleration Structure Test",
                .enableValidation = context.enableValidation,
                .enablePartitionedAccelerationStructure = true,
            },
            device);
        if (!result) {
            return RhiTestResult::fail(
                std::string("createDevice(partitioned acceleration structure) returned ") + toString(result));
        }
        if (device == nullptr) {
            return RhiTestResult::fail("createDevice(partitioned acceleration structure) returned a null device");
        }

        render::PartitionedAccelerationStructureBuildSizes sizes;
        result = device->queryPartitionedAccelerationStructureBuildSizes(
            render::PartitionedAccelerationStructureBuildInputs{
                .instanceCount = 1,
                .partitionCount = 1,
                .maxInstancePerPartitionCount = 1,
                .maxOperationCount = 1,
            },
            sizes);

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

METALLIC_REGISTER_RHI_TEST(ValidateDeviceTest);
METALLIC_REGISTER_RHI_TEST(ScenePathNormalizationTest);
METALLIC_REGISTER_RHI_TEST(OptionalFeatureSoftRequestTest);
METALLIC_REGISTER_RHI_TEST(ClusterAccelerationStructureSupportTest);
METALLIC_REGISTER_RHI_TEST(PartitionedAccelerationStructureSupportTest);

} // namespace
} // namespace metallic::tests
