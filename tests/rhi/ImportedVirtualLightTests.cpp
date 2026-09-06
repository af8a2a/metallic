#include "RhiTest.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/SceneResourceManager.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

namespace metallic::tests {
namespace {

constexpr uint64_t kImportedSceneIdentity = 57;
const scene::SceneEntity kImportedLightObject = static_cast<scene::SceneEntity>(42);

scene::RenderLight makeImportedSource()
{
    scene::RenderLight light;
    light.object = kImportedLightObject;
    light.type = "point";
    light.intensity = 900.0;
    light.intensityUnit = scene::LightUnit::Candela;
    light.range = 2.0;
    light.virtualLightSceneIdentity = kImportedSceneIdentity;
    return light;
}

scene::PunctualLight makeNativeImportedLight()
{
    scene::PunctualLight light;
    light.properties.type = "spot";
    light.properties.intensity = 25.0;
    light.properties.intensityUnit = scene::LightUnit::Candela;
    light.properties.range = 2.0;
    light.position = float3(100.0f, 200.0f, 300.0f);
    light.direction = float3(0.0f, 1.0f, 0.0f);
    light.imported = scene::ImportedLightBinding{
        .sourceId = "fixture",
        .sourceNodeIndex = 3,
        .sceneIdentity = kImportedSceneIdentity,
        .object = kImportedLightObject,
    };
    return light;
}

size_t enabledLightCount(const std::vector<render::SceneLightRecord>& records)
{
    return static_cast<size_t>(std::count_if(records.begin(), records.end(),
        [](const auto& record) { return record.enabled; }));
}

bool near(float actual, float expected)
{
    return std::abs(actual - expected) < 1e-5f;
}

class ImportedVirtualLightOwnershipTest final : public RhiTest {
public:
    ImportedVirtualLightOwnershipTest()
    {
        type = RhiTestType::Validation;
        name = "imported_virtual_light_ownership";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        std::array sources{makeImportedSource(), makeImportedSource()};
        sources[1].object = static_cast<scene::SceneEntity>(43);
        sources[1].virtualLightSceneIdentity = 0;
        sources[1].intensity = 13.0;
        std::array lights{makeNativeImportedLight(), scene::PunctualLight{}};
        lights[1].position = float3(7.0f, 8.0f, 9.0f);
        lights[1].properties.intensity = 17.0;
        auto records = render::buildSceneLightRecords(sources, lights);
        if (records.size() != 4 || enabledLightCount(records) != 3 || records[0].enabled ||
            records[0].sourceRenderLightIndex != 0 ||
            records[0].sourceObject != kImportedLightObject ||
            records[1].gpu.colorIntensity[3] != 13.0f ||
            records[2].sourceVirtualLightIndex != 0 ||
            records[2].sourceRenderLightIndex != scene::kInvalidSceneIndex ||
            records[2].sourceObject != kImportedLightObject ||
            records[2].gpu.colorIntensity[3] != 25.0f || records[2].gpu.directionType[3] != 2.0f ||
            records[3].sourceObject != scene::kNullSceneEntity ||
            records[3].gpu.positionRange[0] != 7.0f || records[3].gpu.colorIntensity[3] != 17.0f) {
            return RhiTestResult::fail("native ownership must suppress only the adopted source and preserve slot provenance");
        }
        lights[0].enabled = false;
        records = render::buildSceneLightRecords(sources, lights);
        if (records[0].enabled || records[2].enabled || enabledLightCount(records) != 2 ||
            records[2].sourceObject != kImportedLightObject) {
            return RhiTestResult::fail("disabled native import re-enabled legacy emission or lost its matched provenance");
        }
        records = render::buildSceneLightRecords(sources, std::span(lights).subspan(1));
        if (records.size() != 3 || records[0].enabled || enabledLightCount(records) != 2) {
            return RhiTestResult::fail("deleting a native light must not fall back to its old imported source");
        }
        lights[0].enabled = true;
        lights[0].properties.type = "point";
        lights[0].imported->localDirection = float3(0.0f);
        records = render::buildSceneLightRecords(sources, lights);
        if (enabledLightCount(records) != 3 || !records[2].enabled || records[2].gpu.directionType[3] != 1.0f) {
            return RhiTestResult::fail("point lights must ignore an unused zero local direction, just like manual lights");
        }
        return RhiTestResult::pass("native, manual and unadopted imports retain independent stable emission slots");
    }
};

class ImportedVirtualLightBindingTest final : public RhiTest {
public:
    ImportedVirtualLightBindingTest()
    {
        type = RhiTestType::Validation;
        name = "imported_virtual_light_live_binding";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        std::array sources{makeImportedSource()};
        std::array lights{makeNativeImportedLight()};
        lights[0].imported->localPosition = float3(1.0f, 2.0f, 3.0f);
        lights[0].imported->localDirection = float3(1.0f, 2.0f, -1.0f);
        auto& m = sources[0].worldMatrix;
        // A rotated, nonuniformly scaled parent transform must affect both the
        // imported node-local offset and its emitted direction, not its range.
        m.a00 = 0.0f; m.a01 = 0.0f; m.a02 = 4.0f; m.a03 = 10.0f;
        m.a10 = 0.0f; m.a11 = 3.0f; m.a12 = 0.0f; m.a13 = 20.0f;
        m.a20 = -2.0f; m.a21 = 0.0f; m.a22 = 0.0f; m.a23 = 30.0f;
        auto records = render::buildSceneLightRecords(sources, lights);
        const float inverseLength = 1.0f / std::sqrt(56.0f);
        if (enabledLightCount(records) != 1 || !records[1].enabled ||
            !near(records[1].gpu.positionRange[0], 22.0f) ||
            !near(records[1].gpu.positionRange[1], 26.0f) ||
            !near(records[1].gpu.positionRange[2], 28.0f) ||
            !near(records[1].gpu.positionRange[3], 2.0f) ||
            !near(records[1].gpu.directionType[0], -4.0f * inverseLength) ||
            !near(records[1].gpu.directionType[1], 6.0f * inverseLength) ||
            !near(records[1].gpu.directionType[2], -2.0f * inverseLength)) {
            return RhiTestResult::fail("native import did not resolve its local transform and normalized direction");
        }
        // Deliberately leave the native RenderWorld snapshot untouched.
        m.a03 = -5.0f; m.a13 = 7.0f; m.a23 = -9.0f;
        records = render::buildSceneLightRecords(sources, lights);
        if (!near(records[1].gpu.positionRange[0], 7.0f) ||
            !near(records[1].gpu.positionRange[1], 13.0f) ||
            !near(records[1].gpu.positionRange[2], -11.0f)) {
            return RhiTestResult::fail("updated parent transform was replaced by stale virtual world coordinates");
        }
        sources[0].visible = false;
        if (enabledLightCount(render::buildSceneLightRecords(sources, lights)) != 0) {
            return RhiTestResult::fail("parent/source visibility must gate the native emission immediately");
        }
        sources[0].visible = true;
        ++lights[0].imported->sceneIdentity;
        if (enabledLightCount(render::buildSceneLightRecords(sources, lights)) != 0) {
            return RhiTestResult::fail("a reused object ID from a different scene must not activate the native light");
        }
        lights[0].imported->sceneIdentity = 0;
        if (enabledLightCount(render::buildSceneLightRecords(sources, lights)) != 0) {
            return RhiTestResult::fail("an unresolved imported binding must remain inactive");
        }
        lights[0].imported->sceneIdentity = kImportedSceneIdentity;
        lights[0].imported->object = scene::kNullSceneEntity;
        if (enabledLightCount(render::buildSceneLightRecords(sources, lights)) != 0) {
            return RhiTestResult::fail("an unresolved imported object must remain inactive");
        }
        lights[0].imported->object = kImportedLightObject;
        records = render::buildSceneLightRecords({}, lights);
        if (records.size() != 1 || records[0].enabled ||
            records[0].sourceObject != scene::kNullSceneEntity) {
            return RhiTestResult::fail("missing imported sources must not become free virtual lights");
        }
        return RhiTestResult::pass("current source transforms, visibility and scene identity override stale snapshots");
    }
};

class ImportedVirtualLightGPUSceneTest final : public RhiTest {
public:
    ImportedVirtualLightGPUSceneTest()
    {
        type = RhiTestType::Validation;
        name = "imported_virtual_light_gpu_scene_collection";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        std::array sources{makeImportedSource()};
        sources[0].worldMatrix.a23 = -5.0f;
        std::array lights{makeNativeImportedLight()};
        render::GPUScene gpuScene;
        gpuScene.setDefaultFrameSlotCount(1);
        if (!gpuScene.syncLights(sources, lights) || gpuScene.lights().size() != 2 ||
            gpuScene.drawSet().lights.size() != 1) {
            return RhiTestResult::fail("GPUScene must expose one native candidate while preserving the suppressed source slot");
        }
        const auto nativeId = gpuScene.lights()[1].id;
        const auto view = gpuScene.createView();
        render::GPUSceneViewPrepareInfo info{.width = 64, .height = 64};
        info.lightFrustumPlanes = {
            float4(1.0f, 0.0f, 0.0f, 3.0f), float4(-1.0f, 0.0f, 0.0f, 3.0f),
            float4(0.0f, 1.0f, 0.0f, 3.0f), float4(0.0f, -1.0f, 0.0f, 3.0f),
            float4(0.0f, 0.0f, -1.0f, -1.0f), float4(0.0f, 0.0f, 1.0f, 10.0f),
        };
        if (!gpuScene.prepareView(view, 0, info)) {
            return RhiTestResult::fail("cannot prepare the imported light test view");
        }
        const auto* visible = gpuScene.visibleLights(view, 0);
        const auto* native = gpuScene.light(nativeId);
        if (visible == nullptr || native == nullptr || visible->localLights != std::vector{nativeId} ||
            visible->sourceLightCount != 1 || native->source.sourceObject != kImportedLightObject ||
            native->source.sourceVirtualLightIndex != 0 || native->source.gpu.positionRange[2] != -5.0f) {
            return RhiTestResult::fail("GPUScene collection lost native source provenance or used stale world placement");
        }
        const auto generation = gpuScene.drawSet().lightGeneration;
        const auto revision = gpuScene.drawSet().lightRevision;
        sources[0].worldMatrix.a03 = 100.0f;
        if (!gpuScene.syncLights(sources, lights) || gpuScene.light(nativeId) == nullptr ||
            gpuScene.drawSet().lightGeneration != generation || gpuScene.drawSet().lightRevision == revision ||
            gpuScene.visibleLights(view, 0) != nullptr || !gpuScene.prepareView(view, 0, info)) {
            return RhiTestResult::fail("source transform edit did not invalidate light visibility while preserving source IDs");
        }
        visible = gpuScene.visibleLights(view, 0);
        if (visible == nullptr || !visible->localLights.empty()) {
            return RhiTestResult::fail("coarse culling ignored the current source transform");
        }
        sources[0].worldMatrix.a03 = 0.0f;
        sources[0].visible = false;
        if (!gpuScene.syncLights(sources, lights) || !gpuScene.drawSet().lights.empty() ||
            !gpuScene.prepareView(view, 0, info) || gpuScene.visibleLights(view, 0) == nullptr ||
            !gpuScene.visibleLights(view, 0)->localLights.empty()) {
            return RhiTestResult::fail("hidden parent/source leaked either native or legacy light into collection");
        }
        sources[0].visible = true;
        lights[0].enabled = false;
        gpuScene.syncLights(sources, lights);
        if (!gpuScene.drawSet().lights.empty()) {
            return RhiTestResult::fail("disabled native light reactivated imported emission in GPUScene");
        }
        return RhiTestResult::pass("native light provenance, stable IDs and live transform/visibility drive GPUScene culling");
    }
};

class ImportedVirtualLightResolvedSceneTest final : public RhiTest {
public:
    ImportedVirtualLightResolvedSceneTest()
    {
        type = RhiTestType::Validation;
        name = "imported_virtual_light_resolved_scene_override";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        // Exercise the real resource-manager fallback, which loads a document
        // even though its public result is a Scene pointer.
        const auto directory = context.outputDirectory / "imported-light-resolver";
        std::filesystem::create_directories(directory);
        const auto path = std::filesystem::absolute(directory / "fallback.gltf");
        {
            std::ofstream fixture(path);
            fixture << R"json({
                "asset": { "version": "2.0" }, "scene": 0,
                "extensionsUsed": ["KHR_lights_punctual"],
                "extensions": { "KHR_lights_punctual": { "lights": [
                    { "name": "Native Point", "type": "point", "intensity": 100, "range": 8 }
                ] } },
                "nodes": [{ "translation": [1, 2, -5],
                    "extensions": { "KHR_lights_punctual": { "light": 0 } } }],
                "scenes": [{ "nodes": [0] }]
            })json";
            if (!fixture) { return RhiTestResult::fail("cannot write resource-manager light fixture"); }
        }
        render::SceneResourceManager manager;
        const scene::Scene* actualScene = nullptr;
        std::string log;
        if (!manager.resolveScene({{"path", path.string()}}, nullptr, actualScene, log) ||
            actualScene == nullptr || actualScene->lights().size() != 1 ||
            actualScene->lights()[0].virtualLightSceneIdentity == 0) {
            return RhiTestResult::fail("resource manager failed to resolve a native-light document: " + log);
        }
        auto resolved = render::resolveSceneLighting(actualScene, nullptr);
        auto records = render::buildSceneLightRecords(actualScene->lights(), resolved.lights);
        auto packed = render::buildPunctualLightRecords(actualScene, resolved);
        if (resolved.lights.size() != 1 || !resolved.lights[0].imported ||
            records.size() != 2 || records[0].enabled || !records[1].enabled ||
            records[1].gpu.positionRange[0] != 1.0f || records[1].gpu.colorIntensity[3] != 100.0f ||
            packed.size() != 2 || packed[0].positionRange[0] != 1.0f) {
            return RhiTestResult::fail("no-world resolved document lost or duplicated its native light");
        }
        render::GPUSceneSubsystem subsystem;
        render::RenderSubsystemHost host;
        subsystem.setSourceOverride(actualScene);
        render::RenderSubsystemFrameContext frame{.device = context.device, .host = host};
        render::RenderChangeBits changes = render::RenderChangeBits::None;
        if (!subsystem.beginFrame(frame, changes, log) || subsystem.lights().size() != 2 ||
            subsystem.drawSet().lights.size() != 1 ||
            subsystem.lights()[1].source.sourceObject != actualScene->lights()[0].object) {
            return RhiTestResult::fail("GPUScene source override without a world lost authored lights: " + log);
        }

        scene::SceneDocument foreignScene;
        if (!foreignScene.load(path)) {
            return RhiTestResult::fail("cannot load independent world fixture");
        }
        render::RenderWorld world;
        world.setScene(&foreignScene);
        auto worldLighting = foreignScene.lighting();
        scene::PunctualLight manual;
        manual.properties.intensity = 17.0;
        manual.properties.range = 3.0;
        worldLighting.lights.push_back(manual);
        worldLighting.exposureEV100 = 6.0f;
        if (!world.setLighting(worldLighting)) { return RhiTestResult::fail("invalid independent world lights"); }
        resolved = render::resolveSceneLighting(actualScene, &world);
        records = render::buildSceneLightRecords(actualScene->lights(), resolved.lights);
        packed = render::buildPunctualLightRecords(actualScene, resolved);
        if (resolved.lights.size() != 2 || !resolved.lights[0].imported || resolved.lights[1].imported ||
            resolved.lights[0].imported->sceneIdentity != actualScene->resourceIdentity() ||
            resolved.lights[0].imported->sceneIdentity == foreignScene.resourceIdentity() ||
            resolved.exposureEV100 != 6.0f || records.size() != 3 || enabledLightCount(records) != 2 ||
            packed.size() != 3 || packed[0].positionRange[0] != 2.0f ||
            packed[0].positionRange[1] != 1.0f / 64.0f) {
            return RhiTestResult::fail("override must combine its own native lights with only independent world lights");
        }
        frame.world = &world;
        ++frame.frameIndex;
        subsystem.onWorldChanged(&world);
        changes = render::RenderChangeBits::None;
        if (!subsystem.beginFrame(frame, changes, log) || subsystem.lights().size() != 3 ||
            subsystem.drawSet().lights.size() != 2 ||
            subsystem.lights()[1].source.gpu.colorIntensity[3] != 100.0f ||
            subsystem.lights()[2].source.gpu.colorIntensity[3] != 17.0f ||
            subsystem.lights()[2].source.sourceObject != scene::kNullSceneEntity) {
            return RhiTestResult::fail("GPUScene override dropped manual lights or included foreign imported bindings");
        }

        // A same-scene world is authoritative, including explicit deletion of
        // every light. Falling back to authored storage would resurrect imports.
        world.setScene(actualScene);
        if (!world.setLighting(actualScene->authoredLighting())) {
            return RhiTestResult::fail("same-scene lighting fixture was rejected");
        }
        resolved = render::resolveSceneLighting(actualScene, &world);
        if (resolved.lights.size() != 1 ||
            enabledLightCount(render::buildSceneLightRecords(actualScene->lights(), resolved.lights)) != 1) {
            return RhiTestResult::fail("same-scene world duplicated its document-native lights");
        }
        world.setLighting({});
        resolved = render::resolveSceneLighting(actualScene, &world);
        if (!resolved.lights.empty() ||
            enabledLightCount(render::buildSceneLightRecords(actualScene->lights(), resolved.lights)) != 0) {
            return RhiTestResult::fail("an intentionally empty same-scene world resurrected authored native lights");
        }
        return RhiTestResult::pass("resolved documents and overrides keep their own native lights, without foreign imports or duplicates");
    }
};

METALLIC_REGISTER_RHI_TEST(ImportedVirtualLightOwnershipTest);
METALLIC_REGISTER_RHI_TEST(ImportedVirtualLightBindingTest);
METALLIC_REGISTER_RHI_TEST(ImportedVirtualLightGPUSceneTest);
METALLIC_REGISTER_RHI_TEST(ImportedVirtualLightResolvedSceneTest);

} // namespace
} // namespace metallic::tests
