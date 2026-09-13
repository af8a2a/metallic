#include "RhiTest.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

class MeshletRefinementBoundsTest final : public RhiTest {
  public:
    MeshletRefinementBoundsTest()
    {
        type = RhiTestType::Validation;
        name = "meshlet_lod_refinement_bounds";
    }
    RhiTestResult run(RhiTestContext&) override
    {
        // A displaced parent must preserve the visible child it replaces.
        std::vector<MeshletLodGroupRecord> groups{
            {.sphere = {0, 0, 0, 1}, .error = .1f},
            {.sphere = {100, 0, 0, 1}, .error = .2f},
            {.sphere = {-100, 0, 0, 1}, .error = std::numeric_limits<float>::max(), .flags = kMeshletLodTerminalGroup}};
        const std::vector<MeshletLodGroupRange> ranges{{0, 1}, {1, 1}, {2, 2}};
        std::vector<uint32_t> refined{UINT32_MAX, 0, 0, 1};
        std::vector<MeshletLodRefinementBounds> bounds;
        std::string reason;
        if (!buildMeshletLodRefinementBounds(groups, ranges, refined, bounds, reason)) {
            return RhiTestResult::fail(reason);
        }
        GPUSceneGpuInstanceRecord instance{};
        instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        MeshletLodView view;
        view.eye = {0, 0, 10, .1f};
        view.forward = {0, 0, -1, 0};
        view.projection = {1080, .57735f, 100, 1.5f};
        const auto visible = [&](const MeshletLodRefinementBounds& box) {
            return meshletLodBoundsVisible(box, instance, view, {0, 1, 0}, 1.f, 1000.f);
        };
        for (uint32_t group = 0; group < bounds.size(); ++group) {
            if (!visible(bounds[group]) || !std::isfinite(bounds[group].max[0]) || bounds[group].max[0] > 102.f ||
                bounds[group].min[0] > -1.1f || bounds[group].max[0] < 1.1f) {
                return RhiTestResult::fail(
                    "Visible shared child lost an ancestor or root sentinel inflated the bounds");
            }
        }
        const MeshletLodRefinementBounds parentOnly{{98, -2, -2}, {102, 2, 2}};
        if (visible(parentOnly) || !visible({})) {
            return RhiTestResult::fail("Offscreen/disabled bound policy mismatch");
        }
        const float edge = 10.f * view.projection[1] * .25f * (1.f + 1.f / (view.projection[0] * .25f));
        const MeshletLodRefinementBounds jitterEdge{{edge, 0, 0}, {edge, 0, 0}};
        if (!meshletLodBoundsVisible(jitterEdge, instance, view, {0, 1, 0}, .25f, 1000.f)) {
            return RhiTestResult::fail("Portrait viewport lost the horizontal pixel guard");
        }
        instance.worldMatrix[12] = 100;
        if (visible(bounds[0])) { return RhiTestResult::fail("Side-plane demand was not rejected"); }
        instance.worldMatrix[12] = 0;
        instance.worldMatrix[14] = 100;
        if (visible(bounds[0])) { return RhiTestResult::fail("Behind-camera demand was not rejected"); }
        instance.worldMatrix[14] = -2000;
        if (visible(bounds[0])) { return RhiTestResult::fail("Far-plane demand was not rejected"); }
        instance.worldMatrix[14] = 10;
        if (!visible(bounds[0])) { return RhiTestResult::fail("Near-plane intersection was lost"); }
        view.forward[3] = 1;
        instance.worldMatrix[14] = 0;
        instance.worldMatrix[12] = 50;
        if (!visible(bounds[0])) { return RhiTestResult::fail("Orthographic edge intersection was lost"); }
        instance.worldMatrix[12] = 100;
        if (visible(bounds[0])) { return RhiTestResult::fail("Orthographic offscreen demand was retained"); }
        instance.worldMatrix[4] = -100;
        if (!visible(bounds[0])) { return RhiTestResult::fail("Sheared visible extent was lost"); }
        refined[1] = 2;
        if (buildMeshletLodRefinementBounds(groups, ranges, refined, bounds, reason) || !bounds.empty()) {
            return RhiTestResult::fail("Invalid dependency order accepted");
        }
        return RhiTestResult::pass("Shared descendants, root sentinel, side/near/far planes, orthographic/sheared "
                                   "bounds and invalid topology");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletRefinementBoundsTest);

struct QualityView {
    MeshletLodView metric;
    float3 right, up;
    float farPlane = 30000.f;
    float aspect = 1920.f / 1080.f;
};

bool qualitySphereVisible(const MeshletLodGroupRecord& group, const GPUSceneGpuInstanceRecord& instance,
                          const QualityView& view)
{
    const auto& m = instance.worldMatrix;
    const float3 a(m[0], m[1], m[2]), b(m[4], m[5], m[6]), c(m[8], m[9], m[10]);
    const float ab = std::abs(dot(a, b)), ac = std::abs(dot(a, c)), bc = std::abs(dot(b, c));
    const float scale = std::sqrt(std::max({dot(a, a) + ab + ac, dot(b, b) + ab + bc, dot(c, c) + ac + bc}));
    const float radius = (group.sphere[3] + group.error) * scale;
    const float3 delta = a * group.sphere[0] + b * group.sphere[1] + c * group.sphere[2] + float3(m[12], m[13], m[14]) -
                         float3(view.metric.eye[0], view.metric.eye[1], view.metric.eye[2]);
    const float z = dot(delta, float3(view.metric.forward[0], view.metric.forward[1], view.metric.forward[2]));
    if (z + radius < view.metric.eye[3] || z - radius > view.farPlane) { return false; }
    const float ty = view.metric.projection[1], tx = ty * view.aspect;
    return std::abs(dot(delta, view.right)) <= z * tx + radius * std::sqrt(1 + tx * tx) &&
           std::abs(dot(delta, view.up)) <= z * ty + radius * std::sqrt(1 + ty * ty);
}

class MiniZorahQualityAuditTest final : public RhiTest {
  public:
    MiniZorahQualityAuditTest()
    {
        type = RhiTestType::Validation;
        name = "minizorah_quality_audit";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        const char* optIn = std::getenv("METALLIC_TEST_MINIZORAH");
        if (!optIn || std::string_view(optIn) != "1") {
            return RhiTestResult::skip("Opt in with METALLIC_TEST_MINIZORAH=1");
        }
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log)) { return RhiTestResult::fail(log); }
        const auto path = sample.graph.findNode("GPUDriven")->properties.at("streamAssetPath").get<std::string>();
        scene::MeshletStreamAsset asset;
        if (!asset.open(std::filesystem::path(PROJECT_SOURCE_DIR) / path, log)) { return RhiTestResult::fail(log); }
        std::vector<MeshletLodGroupRecord> metrics(asset.groupCount());
        std::vector<MeshletLodGroupRange> ranges(asset.groupCount());
        for (uint32_t i = 0; i < asset.groupCount(); ++i) {
            const auto& g = asset.groups()[i];
            std::copy_n(g.boundsCenterRadius, 4, metrics[i].sphere.begin());
            metrics[i].error = g.maxQuadricError;
            metrics[i].flags = g.flags;
            metrics[i].level = g.lodLevel;
            ranges[i] = {g.clusterRefinedOffset, g.clusterCount};
        }
        std::vector<MeshletLodRefinementBounds> bounds;
        if (!buildMeshletLodRefinementBounds(metrics, ranges, asset.refinedGroups(), bounds, log)) {
            return RhiTestResult::fail(log);
        }
        const auto original = sample.graph.viewProperties().at("camera");
        Json report{{"targetPixelError", 1.5}, {"resolution", {1920, 1080}}, {"views", Json::array()}};
        for (uint32_t pose = 0; pose < 3; ++pose) {
            float3 eye(original["eye"][0], original["eye"][1], original["eye"][2]);
            float3 center(original["center"][0], original["center"][1], original["center"][2]);
            if (pose == 1) {
                const auto move = normalize(center - eye) * 2.7f;
                eye += move;
                center += move;
            }
            if (pose == 2) {
                const auto d = center - eye;
                center = eye + float3(d.x * std::cos(.9f) - d.z * std::sin(.9f), d.y,
                                      d.x * std::sin(.9f) + d.z * std::cos(.9f));
            }
            const auto forward = normalize(center - eye);
            QualityView view;
            view.metric.eye = {eye.x, eye.y, eye.z, original["znear"].get<float>()};
            view.metric.forward = {forward.x, forward.y, forward.z, 0};
            view.metric.projection = {1080, std::tan(original["fovDegrees"].get<float>() * .00872664626f), 1, 1.5f};
            view.right = normalize(cross(forward, float3(0, 1, 0)));
            view.up = cross(view.right, forward);
            std::vector<uint8_t> activePages(asset.pageCount()), selectedPages(asset.pageCount()),
                closurePages(asset.pageCount());
            std::vector<uint8_t> culledPages(asset.pageCount());
            uint64_t activeGroups = 0, selectedGroups = 0, selectedClusters = 0, blocked = 0, visibleBlocked = 0,
                     closureGroups = 0;
            uint64_t culledGroups = 0, lostVisibleDemand = 0, sphereOnlyPruned = 0;
            for (const auto& source : asset.instances()) {
                if (!source.visible) { continue; }
                const auto& primitive = asset.primitives()[source.primitiveIndex];
                GPUSceneGpuInstanceRecord instance{};
                std::copy_n(source.worldMatrix, 16, instance.worldMatrix.begin());
                std::vector<uint8_t> active(primitive.groupCount), inactiveParent(primitive.groupCount),
                    wanted(primitive.groupCount), visible(primitive.groupCount);
                std::vector<uint8_t> culledActive(primitive.groupCount), culledInactiveParent(primitive.groupCount);
                std::vector<float> errors(primitive.groupCount);
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    const uint32_t id = primitive.groupOffset + local;
                    errors[local] = meshletLodPixelError(metrics[id], instance, view.metric);
                    visible[local] = qualitySphereVisible(metrics[id], instance, view);
                    wanted[local] = (metrics[id].flags & 1u) != 0 || (visible[local] && errors[local] > 1.5f);
                }
                // Residency-independent dependency closure for visible demand.
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    const auto& group = asset.groups()[primitive.groupOffset + local];
                    for (uint32_t c = 0; c < group.clusterCount; ++c) {
                        const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + c];
                        if (child != UINT32_MAX && wanted[child - primitive.groupOffset]) { wanted[local] = 1; }
                    }
                    if (wanted[local]) {
                        closurePages[group.pageIndex] = 1;
                        ++closureGroups;
                    }
                }
                for (uint32_t reverse = primitive.groupCount; reverse != 0; --reverse) {
                    const uint32_t local = reverse - 1, id = primitive.groupOffset + local;
                    const auto& group = asset.groups()[id];
                    active[local] = !inactiveParent[local] && ((group.flags & 1u) != 0 || errors[local] > 1.5f);
                    culledActive[local] =
                        !culledInactiveParent[local] &&
                        ((group.flags & 1u) != 0 ||
                         (errors[local] > 1.5f && meshletLodBoundsVisible(bounds[id], instance, view.metric, {0, 1, 0},
                                                                          view.aspect, view.farPlane)));
                    if (culledActive[local]) {
                        culledPages[group.pageIndex] = 1;
                        ++culledGroups;
                    } else {
                        MeshletLodRefinementBounds ownBounds;
                        for (uint32_t axis = 0; axis < 3; ++axis) {
                            const double radius = double(metrics[id].sphere[3]) + metrics[id].error;
                            ownBounds.min[axis] =
                                std::nextafter(float(double(metrics[id].sphere[axis]) - radius), -INFINITY);
                            ownBounds.max[axis] =
                                std::nextafter(float(double(metrics[id].sphere[axis]) + radius), INFINITY);
                        }
                        const bool ownVisible = meshletLodBoundsVisible(ownBounds, instance, view.metric, {0, 1, 0},
                                                                        view.aspect, view.farPlane);
                        lostVisibleDemand += active[local] && ownVisible;
                        // The world sphere inflates nonuniform transforms in
                        // every direction; its plane test can retain additional
                        // empty corners beyond the transformed local envelope.
                        sphereOnlyPruned += active[local] && visible[local] && !ownVisible;
                        for (uint32_t c = 0; c < group.clusterCount; ++c) {
                            const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + c];
                            if (child != UINT32_MAX) { culledInactiveParent[child - primitive.groupOffset] = 1; }
                        }
                    }
                    if (active[local]) {
                        activePages[group.pageIndex] = 1;
                        ++activeGroups;
                    } else {
                        if (errors[local] > 1.5f && inactiveParent[local]) {
                            ++blocked;
                            visibleBlocked += visible[local];
                        }
                        for (uint32_t c = 0; c < group.clusterCount; ++c) {
                            const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + c];
                            if (child != UINT32_MAX) { inactiveParent[child - primitive.groupOffset] = 1; }
                        }
                    }
                }
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    if (!active[local]) { continue; }
                    const auto& group = asset.groups()[primitive.groupOffset + local];
                    uint32_t count = 0;
                    for (uint32_t c = 0; c < group.clusterCount; ++c) {
                        const uint32_t child = asset.refinedGroups()[group.clusterRefinedOffset + c];
                        count += child == UINT32_MAX || !active[child - primitive.groupOffset];
                    }
                    if (count) {
                        ++selectedGroups;
                        selectedPages[group.pageIndex] = 1;
                        selectedClusters += count;
                    }
                }
            }
            const auto bytes = [&](const std::vector<uint8_t>& pages) {
                uint64_t sum = 0;
                for (uint32_t i = 0; i < pages.size(); ++i) {
                    if (pages[i]) { sum += (asset.pages()[i].uncompressedSize + 255u) & ~uint64_t(255u); }
                }
                return sum;
            };
            Json result{{"pose", pose},
                        {"activeGroups", activeGroups},
                        {"selectedGroups", selectedGroups},
                        {"selectedClusters", selectedClusters},
                        {"activePayloadBytes", bytes(activePages)},
                        {"selectedPayloadBytes", bytes(selectedPages)},
                        {"visibleDemandClosureBytes", bytes(closurePages)},
                        {"visibleDemandClosureGroups", closureGroups},
                        {"viewDrivenPayloadBytes", bytes(culledPages)},
                        {"viewDrivenGroups", culledGroups},
                        {"lostVisibleDemand", lostVisibleDemand},
                        {"sphereOnlyPruned", sphereOnlyPruned},
                        {"blockedDesiredGroups", blocked},
                        {"visibleBlockedDesiredGroups", visibleBlocked}};
            report["views"].push_back(result);
            std::printf("[MiniZorahQuality] %s\n", result.dump().c_str());
            std::fflush(stdout);
            if (lostVisibleDemand != 0 || visibleBlocked != 0 || bytes(culledPages) > (1ull << 30)) {
                return RhiTestResult::fail("Visible demand lost or the ideal cut exceeded the 1 GiB quality budget");
            }
        }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream(context.outputDirectory / "MiniZorahQualityAudit.json") << report.dump(2) << '\n';
        return RhiTestResult::pass("Residency-independent demand and ancestor footprint audited");
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahQualityAuditTest);
} // namespace
} // namespace metallic::tests
