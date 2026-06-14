#include "rhi_test.h"

#include "Runtime/Render/GAPI/Vulkan/vulkan_scene_rtx.h"
#include "Runtime/Scene/scene.h"

#include <filesystem>
#include <memory>
#include <string>

namespace metallic::tests {
namespace {

class SceneRtxAccelerationStructureBuildTest : public RhiTest {
public:
    SceneRtxAccelerationStructureBuildTest()
    {
        type = RhiTestType::Resource;
        name = "scene_rtx_acceleration_structure_build";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result result = render::createDevice(
            render::DeviceDesc{
                .applicationName = "Metallic Scene RTX Test",
                .enableValidation = context.enableValidation,
                .enableRayTracingAccelerationStructure = true,
            },
            device);
        if (!result) {
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip(std::string("createDevice returned ") + toString(result));
            }
            return RhiTestResult::fail(std::string("createDevice returned ") + toString(result));
        }
        if (!device->capabilities().rayTracingAccelerationStructure) {
            return RhiTestResult::skip("ray tracing acceleration structure capability is unavailable");
        }

        render::Queue* graphicsQueue = device->getQueue(render::QueueType::Graphics);
        if (graphicsQueue == nullptr) {
            return RhiTestResult::fail("scene RTX test device has no graphics queue");
        }

        scene::Scene loadedScene;
        const std::filesystem::path scenePath =
            std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        if (!loadedScene.load(scenePath)) {
            const scene::LoadResult& loadResult = loadedScene.lastLoadResult();
            return RhiTestResult::fail(
                loadResult.error.empty() ? "failed to load Stanford Bunny scene" : loadResult.error);
        }

        render::vulkan::SceneRtxBuilder builder;
        std::string log;
        result = builder.build(*device, *graphicsQueue, loadedScene, log);
        if (!result) {
            return RhiTestResult::fail(
                std::string("SceneRtxBuilder::build returned ") +
                toString(result) +
                ": " +
                log);
        }
        if (!builder.valid() || builder.tlas() == VK_NULL_HANDLE || builder.tlasDeviceAddress() == 0) {
            return RhiTestResult::fail("SceneRtxBuilder did not produce a valid TLAS");
        }

        const render::vulkan::SceneRtxStats& stats = builder.stats();
        if (stats.blasCount == 0 || stats.instanceCount == 0 || stats.triangleCount == 0) {
            return RhiTestResult::fail("SceneRtxBuilder produced empty RTX stats");
        }

        return RhiTestResult::pass(log);
    }
};

METALLIC_REGISTER_RHI_TEST(SceneRtxAccelerationStructureBuildTest);

} // namespace
} // namespace metallic::tests
