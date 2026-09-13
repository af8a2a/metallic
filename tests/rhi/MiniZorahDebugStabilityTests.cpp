#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
void checkDebug(bool value, const std::string& message)
{
    if (!value) { throw std::runtime_error(message); }
}

class DebugIdentityObserver final : public IRenderDebugObserver {
public:
    bool capture = false;
    bool terminalReady = false;
    Device* device = nullptr;
    std::map<std::string, std::unique_ptr<Buffer>> copies;
    void compiled(Json) override {}
    void beginExecution(Device& d, debug::DebugEvidenceStamp, RenderSubsystemHost*) override { device = &d; }
    void endExecution(bool) override {}
    void boundary(CommandBuffer& commands, std::string_view checkpoint, uint32_t, std::string_view pass,
        std::span<const DebugResourceBinding> resources, const Json& values) override
    {
        if (pass != "GPUDriven" || checkpoint != "AfterPass") { return; }
        if (values.contains("streaming")) { terminalReady = values.at("streaming").at("instances").at(0).value("terminalReady", false); }
        if (!capture) { return; }
        for (const auto& resource : resources) {
            const bool color = resource.id == "GPUDriven.color";
            if (!color && resource.id != "streaming.GPUDriven.activeGroups" &&
                resource.id != "streaming.GPUDriven.visibleClusters" && resource.id != "streaming.GPUDriven.activeHeader") { continue; }
            const uint64_t bytes = color ? uint64_t(resource.texture->desc().width) * resource.texture->desc().height * 4 :
                resource.size != 0 ? resource.size : resource.buffer->desc().size;
            auto& copy = copies[resource.id];
            if (!copy || copy->desc().size != bytes) {
                checkDebug(bool(device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}, copy)), "Cannot allocate identity snapshot");
            }
            if (color) {
                checkDebug(resource.texture->desc().format == Format::Rgba8Unorm, "Unexpected debug color format");
                TextureBarrierDesc barrier{.texture = resource.texture, .before = resource.state, .after = ResourceState::TransferSource};
                commands.barrier({.textures = &barrier, .textureCount = 1});
                commands.copyTextureToBuffer({.texture = resource.texture, .buffer = copy.get(),
                    .width = resource.texture->desc().width, .height = resource.texture->desc().height});
                std::swap(barrier.before, barrier.after); commands.barrier({.textures = &barrier, .textureCount = 1});
            } else {
                BufferBarrierDesc barrier{.buffer = resource.buffer, .before = resource.state, .after = ResourceState::TransferSource,
                    .offset = resource.offset, .size = bytes};
                commands.barrier({.buffers = &barrier, .bufferCount = 1});
                commands.copyBuffer({.source = resource.buffer, .destination = copy.get(), .sourceOffset = resource.offset, .size = bytes});
                std::swap(barrier.before, barrier.after); commands.barrier({.buffers = &barrier, .bufferCount = 1});
            }
        }
    }
    template<typename T> std::vector<T> read(const std::string& name)
    {
        auto& buffer = copies.at(name);
        buffer->invalidate(); const void* data = buffer->map();
        checkDebug(data != nullptr, "Cannot map identity snapshot");
        std::vector<T> result(buffer->desc().size / sizeof(T));
        std::memcpy(result.data(), data, result.size() * sizeof(T)); buffer->unmap(); return result;
    }
};

class MiniZorahDebugStabilityTest final : public RhiTest {
public:
    MiniZorahDebugStabilityTest() { type = RhiTestType::Rendering; name = "minizorah_debug_identity_stability"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!std::getenv("METALLIC_TEST_MINIZORAH")) { return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1 for the full cooked scene"); }
        Json report{{"width", 1920}, {"height", 1080}, {"lodPixelError", 1.5}, {"poses", Json::array()}};
        const auto save = [&]() { std::ofstream(context.outputDirectory / "MiniZorahDebugStability.json") << report.dump(2) << '\n'; };
        try {
            std::string log;
            RenderSampleLoadResult sample;
            checkDebug(loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log), log);
            auto graph = std::move(sample.graph);
            graph.removeNode(graph.findNode("FinalBlit")->id);
            graph.removeNode(graph.findNode("MaterialResolve")->id);
            graph.markOutput("GPUDriven.visibility"); graph.markOutput("GPUDriven.color");
            const auto node = graph.findNode("GPUDriven")->id;
            graph.findNode(node)->properties["debugStreamingPages"] = false;
            const Json original = graph.viewProperties().at("camera");
            RenderView view;
            checkDebug(view.setCameraProperties(original), "Invalid initial camera");
            DebugIdentityObserver observer;
            RenderGraphPreviewRenderer preview;
            // Readbacks belong to the preview device and must die before it.
            struct ReadbackLifetime {
                RenderGraphPreviewRenderer& preview;
                DebugIdentityObserver& observer;
                ~ReadbackLifetime()
                {
                    preview.setDebugObserver(nullptr);
                    observer.copies.clear();
                }
            } readbackLifetime{preview, observer};
            const Result initialized = preview.initialize(context.enableValidation, false, false);
            if (hasError(initialized, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and bindless heap"); }
            checkDebug(bool(initialized), preview.lastLog());
            preview.bindRenderView(&view); preview.setDebugObserver(&observer);
            const auto renderFrame = [&](bool capture) {
                observer.capture = capture;
                checkDebug(bool(preview.render(graph, 1920, 1080, "GPUDriven.visibility", capture)), preview.lastLog());
            };
            graph.setNodeRuntimeProperty(node, "visualization", "meshlet");
            const auto start = std::chrono::steady_clock::now();
            do { renderFrame(false); }
            while ((!observer.terminalReady || std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) &&
                std::chrono::steady_clock::now() - start < std::chrono::seconds(45));
            checkDebug(observer.terminalReady, "Terminal pages did not load");
            uint64_t totalCompared = 0, totalRemapped = 0;
            for (const char* mode : {"meshlet", "triangle", "lod"}) {
                graph.setNodeRuntimeProperty(node, "visualization", mode);
                std::map<uint64_t, uint32_t> knownColors, previousColors;
                using InstanceKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
                std::map<InstanceKey, uint32_t> previousSlots;
                uint32_t pose = 0;
                for (float angle : {0.f, -3.f, 3.f, 0.f}) {
                    Json camera = original;
                    const float x = original["center"][0].get<float>() - original["eye"][0].get<float>();
                    const float z = original["center"][2].get<float>() - original["eye"][2].get<float>();
                    const float a = angle * 3.14159265359f / 180.f;
                    camera["center"][0] = original["eye"][0].get<float>() + std::cos(a)*x + std::sin(a)*z;
                    camera["center"][2] = original["eye"][2].get<float>() - std::sin(a)*x + std::cos(a)*z;
                    checkDebug(view.setCameraProperties(camera), "Invalid rotated camera");
                    for (uint32_t f = 0; f < 24; ++f) { renderFrame(false); }
                    renderFrame(true);
                    const auto groups = observer.read<MeshletStreamGpuActiveGroup>("streaming.GPUDriven.activeGroups");
                    const auto header = observer.read<MeshletStreamGpuActiveHeader>("streaming.GPUDriven.activeHeader").front();
                    const auto records = observer.read<VisibleClusterRecord>("streaming.GPUDriven.visibleClusters");
                    const auto colors = observer.read<uint32_t>("GPUDriven.color");
                    std::map<uint64_t, uint32_t> currentColors;
                    std::map<InstanceKey, uint32_t> currentSlots;
                    uint64_t compared = 0, remapped = 0;
                    const bool triangleMode = std::string_view(mode) == "triangle", lodMode = std::string_view(mode) == "lod";
                    // Sample the whole image; match identity rather than screen coordinates after rotation.
                    for (size_t pixel = 0; pixel < preview.pixels().size(); pixel += 7) {
                        const uint32_t id = preview.pixels()[pixel];
                        if (id == 0) { continue; }
                        const uint32_t slot = (id >> 7) - 1, triangle = triangleMode ? id & 127 : 0;
                        checkDebug(slot < records.size(), "Stream visibility outside record storage");
                        const auto& record = records[slot];
                        checkDebug(record.dataIndex < header.activeGroupCount, "Invalid active group");
                        const auto& group = groups[record.dataIndex];
                        checkDebug(record.clusterIndex < group.clusterCount, "Invalid local cluster");
                        const uint64_t key = lodMode ? group.lodLevel : (uint64_t(group.pageIndex) << 12) | (record.clusterIndex << 7) | triangle;
                        const uint32_t color = colors[pixel];
                        auto [known, inserted] = knownColors.emplace(key, color);
                        checkDebug(inserted || known->second == color, "Same geometry/LOD changed color across pixels or camera poses");
                        if (!inserted) { ++compared; }
                        currentColors.emplace(key, color);
                        currentSlots.emplace(InstanceKey{group.pageIndex, record.clusterIndex, record.instanceIndex, triangle}, slot);
                    }
                    uint32_t shared = 0;
                    for (const auto& [key, color] : currentColors) { shared += previousColors.contains(key); }
                    for (const auto& [key, slot] : currentSlots) {
                        const auto previous = previousSlots.find(key);
                        remapped += previous != previousSlots.end() && previous->second != slot;
                    }
                    checkDebug(currentColors.size() > (lodMode ? 1u : 100u), "Insufficient visible debug identities");
                    if (pose != 0) { checkDebug(shared > (lodMode ? 1u : 100u), "Insufficient cross-view identity overlap"); }
                    const auto image = context.outputDirectory / (std::string("MiniZorah-") + mode + "-" + std::to_string(pose) + ".png");
                    checkDebug(saveRgba8Png(image, reinterpret_cast<const uint8_t*>(colors.data()), 1920, 1080, log), log);
                    report["poses"].push_back({{"mode", mode}, {"yawDegrees", angle}, {"identities", currentColors.size()},
                        {"previousIdentities", previousColors.size()}, {"sharedIdentities", shared}, {"remappedInstanceRecords", remapped},
                        {"matchedColorSamples", compared}, {"image", image.generic_string()}});
                    totalCompared += compared; totalRemapped += remapped;
                    previousColors = std::move(currentColors); previousSlots = std::move(currentSlots); ++pose;
                }
            }
            checkDebug(totalRemapped > 1000, "Candidate relocation was not exercised");
            report["status"] = "passed"; report["matchedColorSamples"] = totalCompared;
            report["remappedInstanceRecords"] = totalRemapped; report["colorMismatches"] = 0; save();
            return RhiTestResult::pass("MiniZorah meshlet/triangle/LOD colors stable across yaw and temporary record relocation");
        } catch (const std::exception& error) {
            report["status"] = "failed"; report["error"] = error.what(); save();
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahDebugStabilityTest);
} // namespace
} // namespace metallic::tests
